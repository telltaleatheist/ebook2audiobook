from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.classes.tts_engines.common.audio import trim_audio
from lib.classes.tts_engines.common.orpheus_text import to_tts_form
# TEST Step 3c: Direct imports since headers no longer provides them
import torch
import torchaudio
import numpy as np
import platform
import sys
import os
import re
import math
import atexit
import weakref

# Track active Orpheus instances for cleanup on exit
_active_instances = weakref.WeakSet()


class TokenStreamMisaligned(ValueError):
    """The model emitted an audio-token stream whose codes don't fit their
    positional slots (see _redistribute_codes). Sampling is stochastic, so a
    single re-render usually fixes it — but the malformed codes must NEVER
    reach the GPU, where they trigger a device-side assert that poisons the
    CUDA context for the rest of the process."""


# Error-message markers that mean the CUDA context is dead for THIS PROCESS —
# every subsequent CUDA call will fail instantly, so continuing sentence-by-
# sentence just burns the rest of the book in a fast error loop (2026-07-05:
# 1034 sentences "failed" in seconds after one device-side assert). The only
# recovery is a fresh process; callers must re-raise, letting the worker die
# so BookForge's retry machinery respawns it and resumes from files on disk.
_FATAL_CUDA_MARKERS = (
    'device-side assert',
    'illegal memory access',
    'unspecified launch failure',
    'context is destroyed',
)


def is_fatal_cuda_error(err: BaseException) -> bool:
    """True if `err` indicates a poisoned CUDA context (unrecoverable in-process)."""
    msg = str(err)
    return any(marker in msg for marker in _FATAL_CUDA_MARKERS)

# Platform-specific vLLM configuration
if platform.system() == 'Windows':
    # Required for vLLM on Windows
    # See: https://github.com/SystemPanic/vllm-windows
    os.environ['USE_LIBUV'] = '0'
    # vLLM needs the path to cudart DLL - find it in torch installation
    if 'VLLM_CUDART_SO_PATH' not in os.environ:
        try:
            import torch
            torch_lib = os.path.dirname(torch.__file__)
            cudart_path = os.path.join(torch_lib, 'lib', 'cudart64_12.dll')
            if os.path.exists(cudart_path):
                os.environ['VLLM_CUDART_SO_PATH'] = cudart_path
        except Exception:
            pass  # Let vLLM handle the error if cudart not found
else:
    # On Linux/WSL, enable CUDA graphs for vLLM performance
    # Override conf.py settings that disable CUDA graphs globally
    os.environ['TORCH_CUDA_ENABLE_CUDA_GRAPH'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def _cleanup_on_exit():
    """Called on process exit to ensure all Orpheus instances are cleaned up"""
    import gc
    for instance in list(_active_instances):
        try:
            instance.cleanup()
        except Exception:
            pass
    # Final CUDA cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception:
        pass

atexit.register(_cleanup_on_exit)


# ── MLX LoRA application (stage B2) ──────────────────────────────────────────
#
# The two pieces of state that make a LoRA removable from a RESIDENT MLX model.
# They live at module scope rather than inside Orpheus because both are built on
# mlx.nn, which must not be imported on a machine that has no MLX at all — hence
# the lazy class factory below rather than a plain class statement.

class _MlxAdapterState:
    """What is applied to the resident MLX model right now, and how to take it off.

    `sites` is a list of (path, parent_module, attr_name, original_module): the
    ORIGINAL projection module for every site the current adapter wrapped. Clearing
    is therefore exact UNWRAPPING — put each original back where it was — never
    arithmetic un-fusing, which on bf16 weights would leave a slightly different
    model behind after every voice switch and drift over a session.

    Deliberately a plain object and not a dict/tuple: mlx's Module.__setattr__ files
    arrays, dicts, lists and tuples into the module's own parameter dict (a Module IS
    a dict), so a state object of any of those shapes would be walked by
    parameters()/eval() as if it were part of the model. Anything else lands as an
    ordinary Python attribute, which is what this needs to be.
    """

    __slots__ = ('adapter_dir', 'fingerprint', 'sites')

    def __init__(self, adapter_dir, fingerprint, sites):
        self.adapter_dir = adapter_dir
        self.fingerprint = fingerprint
        self.sites = sites


_MLX_LORA_LINEAR = None


def _mlx_lora_linear_cls():
    """The nn.Module that adds one LoRA delta to one projection, built on first use.

    y = base(x) + scale * (x @ A.T) @ B.T

    A PURE FUNCTION OF x, with no state and no shape assumptions beyond the last
    dimension: that is what keeps it compatible with mlx_lm's BatchGenerator, which
    calls the model with (batch, tokens) during prefill and (batch, 1) per decode
    step, at whatever width the batch happens to be. Anything that cached a shape,
    or that behaved differently on the first call, would produce a model that is
    correct solo and wrong in a batch — the hardest possible thing to hear.

    The two matmuls are kept SEPARATE (x@A.T then @B.T) rather than folded into a
    single (B@A) delta matrix: r=64 against 3072x8192 means the pair costs ~2% of
    the base projection, while materialising B@A would cost a full-size matmul per
    layer per step AND another 5 GB resident.
    """
    global _MLX_LORA_LINEAR
    if _MLX_LORA_LINEAR is None:
        import mlx.nn as nn

        class _OrpheusMlxLoRALinear(nn.Module):
            def __init__(self, base, lora_a, lora_b, scale):
                super().__init__()
                self.base = base          # the ORIGINAL nn.Linear, untouched
                self.lora_a = lora_a      # (r, in_features)
                self.lora_b = lora_b      # (out_features, r)
                self.scale = scale        # float: lora_alpha / r

            def __call__(self, x):
                return self.base(x) + (x @ self.lora_a.T) @ self.lora_b.T * self.scale

        _MLX_LORA_LINEAR = _OrpheusMlxLoRALinear
    return _MLX_LORA_LINEAR


class Orpheus(TTSUtils, TTSRegistry, name='orpheus'):
    """
    Orpheus TTS engine - SOTA open-source TTS built on Llama-3b backbone.
    Excellent prosody and naturalness, ideal for audiobooks.

    Supports three backends (auto-detected by platform):
    - MLX (preferred on Mac) - Fast, uses Apple Silicon efficiently (~1.4x realtime)
    - vLLM (preferred on Windows/Linux with CUDA) - Fast batched inference
    - Transformers (fallback) - Slow but works everywhere

    IMPORTANT: Orpheus does NOT benefit from multiple workers like XTTS.
    - MLX uses unified memory - multiple workers compete, no speedup
    - vLLM has built-in batching - use single instance
    Always run with workers=1 for Orpheus.

    Voices: tara, leah, jess, leo, dan, mia, zac, zoe
    Emotion tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>
    """

    # Valid Orpheus voices (leah has best quality, tara has echo artifacts).
    # Custom finetunes are NOT listed here — they arrive via a folder-discovered
    # model dir (session['orpheus_model_dir']) and bypass this allowlist.
    VALID_VOICES = {'tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe'}
    DEFAULT_VOICE = 'leah'

    # Model configuration
    # MLX model (for Mac): mlx-community/orpheus-3b-0.1-ft-bf16
    # Transformers/vLLM model: unsloth/orpheus-3b-0.1-ft
    MLX_MODEL = "mlx-community/orpheus-3b-0.1-ft-bf16"
    TRANSFORMERS_MODEL = "unsloth/orpheus-3b-0.1-ft"
    SAMPLE_RATE = 24000

    # ---- Adapter mode --------------------------------------------------------
    #
    # One shared base model + a per-voice LoRA adapter, instead of one 6.6 GB merged
    # model per voice. Selected by the session keys orpheus_base_dir (the base) +
    # orpheus_adapter_dir (this voice's adapter); --fine_tuned carries the voice token
    # the adapter was trained on.
    #
    # TWO backends apply it, and they are not the same mechanism:
    #   vLLM  PER REQUEST, as a LoRARequest over a multi-LoRA engine. Several voices
    #         are servable at once and one batch may mix them. The voice token is also
    #         its key in the process-global adapter registry (_register_lora), and the
    #         LORA_* constants below size that engine.
    #   MLX   APPLIED to the resident model by wrapping its projection modules
    #         (_apply_mlx_adapter, stage B2). Exactly ONE voice is servable at a time;
    #         a switch swaps the wrappers, which costs ~a second instead of a 6.2 GB
    #         reload. None of the LORA_* constants below apply — see
    #         _validate_adapter_config_mlx for what MLX refuses instead.
    #
    # max_lora_rank must be one of vLLM 0.7.3's legal values (8/16/32/64/128/256)
    # and >= the adapter's own r; every deployed Orpheus adapter is r=64.
    # max_loras is how many DISTINCT adapters one batch may mix — the audiobook
    # worker renders one voice per process, so 1. max_cpu_loras is the host-side
    # adapter cache.
    #
    # lora_extra_vocab_size is deliberately NOT passed, leaving vLLM's default 256:
    # 0 crashes 0.7.3's startup warmup, where create_dummy_lora_weights allocates a
    # hardcoded 10-row embeddings tensor that cannot be copied into a 0-row buffer.
    # The padding is harmless for a non-embedding adapter — LogitsProcessorWithLoRA
    # fills the padded vocab with -inf, so those ids can never be sampled (measured:
    # 0 of 29,119 generated tokens landed outside the base vocab).
    LORA_MAX_RANK = 64
    LORA_MAX_LORAS = 1
    LORA_MAX_CPU_LORAS = 4

    # Batched inference (vLLM): feed many prompts to ONE engine.generate() call.
    # This is how vLLM is meant to be driven — it's faster (real batching) AND
    # avoids the host-RAM creep from tens of thousands of single-prompt calls over
    # a long book. The worker (worker_core) honors SUPPORTS_BATCH/BATCH_SIZE.
    # Override the size with ORPHEUS_BATCH_SIZE.
    SUPPORTS_BATCH = True
    BATCH_SIZE = int(os.environ.get('ORPHEUS_BATCH_SIZE', '16'))

    # Max audio tokens per MLX batched generation. Rows that hit this cap without
    # emitting the end-of-audio token would ship CLIPPED audio; the batch paths
    # detect that and re-render the row split at sentence boundaries
    # (_generate_mlx_safe with force_split, the cap being already proven),
    # mirroring the vLLM ladder below.
    #
    # 3700 matches MAX_AUDIO_TOKENS (the vLLM cap): ~8 audio tokens/char means the
    # ~450-char packed chunks need ~2500-3400 tokens, so the old 2048 default made
    # nearly EVERY prose chunk overflow into the split ladder — and those split
    # boundaries were a measured junk source. Validated 2026-07-08 on the 13-chunk
    # real-book set (M1 Ultra): 2048 → 11 cap-hits, 12.1% WER, 1 catastrophic row,
    # ~310s; 3700 → 0 cap-hits, 3.5% WER (Whisper noise floor), 0 catastrophic,
    # ~105s (2.8x faster), peak memory ~13.7 GB (vs ~14.2 at 2048).
    MLX_MAX_TOKENS = int(os.environ.get('ORPHEUS_MLX_MAX_TOKENS', '3700'))

    # ---- MLX repetition-penalty window -------------------------------------
    #
    # How far back the repetition penalty looks. mlx_lm's own default is 20
    # tokens (~0.24s of audio at 84 tok/s), and that is what this ran on until
    # 2026-08-21. It is far too short for SILENCE: a pause is a run of repeated
    # silence tokens, so once a pause passes a quarter second it falls out of
    # the window and the penalty stops discouraging it — the pause is free to
    # run on. vLLM applies the same penalty over the FULL context, which is why
    # the identical voice paused normally there and dragged here.
    #
    # 4096 exceeds MLX_MAX_TOKENS, so the window is effectively the whole
    # generation and the two backends now shape pauses the same way. Measured
    # on a 58-chunk God's People slice (thirdreich, M1 Ultra, width 96):
    #   window 20   → 650s, 4 cap-hits, 38.5% silence, pause p90 2.34s, max 6.37s
    #   window 4096 → 377s, 0 cap-hits, 30.9% silence, pause p90 1.63s, max 3.65s
    # (vLLM reference for the same voice: 31.0%, p90 1.68s, max 2.92s.)
    # Speech seconds went UP 2.3% while silence fell 26.9% — it removes dead
    # air, it does not clip. Killing the cap-hits also kills the serial
    # re-render ladder those trigger, which is the larger win on a full book.
    MLX_REP_WINDOW = int(os.environ.get('ORPHEUS_MLX_REP_WINDOW', '4096'))

    # ---- MLX batch memory budget -------------------------------------------
    #
    # HISTORY (read before "optimizing" this): until mlx-lm 0.31.3 the MLX batch
    # paths bucketed prompts by near-uniform length and pooled 4x BATCH_SIZE to
    # refill those buckets. That whole apparatus existed for ONE reason —
    # BatchGenerator LEFT-padded every prompt in a prefill to the longest row, and
    # a heavily left-padded short row (a "Chapter one." heading next to a 450-char
    # prose row) stochastically collapsed into gibberish audio. mlx-lm 0.31.3
    # flipped batch prefill padding to the RIGHT (_right_pad_prompts +
    # BatchKVCache.prepare(right_padding=) + per-row dynamic_roll in finalize()),
    # which ROOT-FIXES the corruption: verified 2026-07 by whisper WER on a
    # deliberately mixed-length insertion — 3.1% WER, zero corrupted rows. So
    # bucketing and pooling are GONE, not kept as a fallback, and batches are now
    # simply the next BATCH_SIZE-capped slice in BOOK ORDER (which also makes the
    # cap-budget below accurate and progress reporting monotonic again).
    #
    # What replaces them is a memory bound. KV cache is the dominant term and it
    # scales with width x depth: MEASURED 0.1147 MB per token per row (28 layers,
    # 8 KV heads, head_dim 128, bf16 => 28*2*8*128*2 bytes). The old
    # "~0.153 GB per batch slot" figure in the docstrings was ~2x understated for
    # real books, because it was measured on short chunks that never approached
    # the token cap: a 75-row batch at cap 3700 actually peaked at 36.8 GB.
    # 96 rows x 3700 tokens would be ~40.7 GB of KV alone, plus ~6.9 GB weights
    # plus the 8 GB buffer cache = ~55 GB — far too close to a 64 GB ceiling.
    # So _mlx_batch_groups derives a per-batch WIDTH cap from the batch's own
    # token depth (see _mlx_width_for_depth) targeting this budget.
    MLX_MEM_BUDGET_GB = float(os.environ.get('ORPHEUS_MLX_MEM_BUDGET_GB', '45'))
    # Measured KV bytes per generated token per row, in MB (see above).
    MLX_KV_MB_PER_TOKEN_ROW = 0.1147
    # Resident bf16 weights for orpheus-3b, in GB (measured at load).
    MLX_WEIGHTS_GB = 6.9

    # Max audio tokens per generation. ~8 audio tokens/char, so ~3700 covers the
    # ~450-char multi-sentence chunks core.py now feeds Orpheus while staying under
    # the 4096 model context (prompt + audio). A chunk that would exceed this is
    # re-rendered split at sentence boundaries (see _generate_audio_vllm_safe) so the
    # audio is never clipped. Override with ORPHEUS_MAX_TOKENS.
    MAX_AUDIO_TOKENS = int(os.environ.get('ORPHEUS_MAX_TOKENS', '3700'))

    # Audio-token <-> wall-clock relationship. Orpheus emits 7 audio tokens per
    # SNAC frame (_redistribute_codes); the SNAC-24kHz coarse-frame rate is ~12 Hz,
    # so ~84 audio tokens == 1 second of audio. Cross-checks with this file's own
    # anchor: a 2048-token clip of a ~450-char chunk measured ~18.4 ch/s
    # (_generate_mlx docstring), i.e. 450/18.4 = 24.5 s -> 2048/24.5 = 83.6 tok/s.
    # Rounded UP to 84 because it only sizes a per-chunk max_tokens CEILING
    # (_mlx_token_budget), where a larger value is the safe direction.
    TOKENS_PER_AUDIO_SECOND = 84

    # Truncation-guard rate, in characters of TEXT per second of AUDIO: above this,
    # the audio is too short for the text and the chunk is re-rendered split
    # (_needs_resplit). 19.0 is e2a's documented default for an UNCATALOGUED voice;
    # a catalogued one is given its own value through ORPHEUS_MAX_CHARS_PER_SEC
    # (BookForge passes e.g. 23.5 for deathstalker), and 0 disables the guard.
    #
    # This is the DEFAULT only — never the live value. Read it through
    # _max_chars_per_sec(), which resolves per voice on EVERY call (registered cap
    # -> env -> this default; see VOICE_CAP_SOURCES): the resident streaming worker
    # switches voices in-process, so a value captured at import or construction
    # would pin the first voice's threshold onto every later one. The constant
    # exists so the 19.0 literal lives in exactly one place instead of being
    # repeated at each of the three call sites.
    DEFAULT_MAX_CHARS_PER_SEC = '19.0'

    # Sampling params at the Orpheus reference values. The spurious leading-syllable
    # artifact once chased here (cooling temperature to 0.5 only flattened prosody and
    # didn't remove it) was a prompt-framing bug — a stray BOS in the vLLM prompt,
    # fixed in _format_prompt_ids — NOT sampling or chunk size. With the framing correct
    # core.py packs 2-3 sentences per chunk (~450 chars); temp 0.6 is the reference.
    # All three override via env so they can be A/B-tuned live
    # (ORPHEUS_TEMPERATURE / ORPHEUS_TOP_P / ORPHEUS_REP_PENALTY).
    # Temperature: back at the 0.6 upstream reference (2026-07-13) as the clean
    # baseline for the retrained voices — the 0.85 ladder winner (2026-07-12,
    # deathstalker_v3, 350-char packing) was ear-picked around the OLD fine-tune
    # and the next full book came out rough. Re-run the ladder per retrain via
    # ORPHEUS_TEMPERATURE rather than baking a voice-specific value in here.
    TEMPERATURE = float(os.environ.get('ORPHEUS_TEMPERATURE', '0.6'))
    TOP_P = float(os.environ.get('ORPHEUS_TOP_P', '0.8'))
    # Rep penalty 1.1: the penalty is the PAUSE governor (audio silence tokens
    # repeat; ladder-proven 2026-07-12: 1.0 = pauses sprawl to 2x runtime +
    # token-cap runaways, 1.05/1.07 = audibly long, 1.1 = right) but it also
    # chokes legitimately repeating codes mid-vowel, causing occasional breathy
    # voicing "cracks". Owen chose pauses-right over cracks-fewer; the real fix
    # is retrain-side — see orpheus-finetune SAMPLING_AND_VOICE_QUALITY_FINDINGS.md
    # (pause-capped re-cut lets this drop to ~1.02-1.05 where cracks fade).
    REP_PENALTY = float(os.environ.get('ORPHEUS_REP_PENALTY', '1.1'))
    # min_p drops tokens below this fraction of the top token's probability —
    # cuts the rare-junk tail without flattening expressive variety the way
    # lowering top_p does (confidence-scaled cutoff vs fixed probability mass).
    # Applied on the vLLM paths and the MLX BATCH paths (mlx_lm make_sampler
    # takes min_p). The MLX single-sentence path can't honor it: mlx_audio's
    # llama Model.generate builds its own sampler and only forwards top_k /
    # repetition_penalty from kwargs, so min_p would be silently dropped there.
    # Default 0 = off, matching upstream (which never uses min_p); the Jul 12
    # tuning pass ran 0.05 — re-enable per-voice via ORPHEUS_MIN_P if the
    # retrain still shows a junk tail.
    MIN_P = float(os.environ.get('ORPHEUS_MIN_P', '0.0'))

    # EOS logit boost (2026-07-21, ghost-whisper campaign): voices trained on
    # bed-free corpora carry a thinner GREEDY end-of-speech margin (mb_2hd:
    # 15/20 greedy stops vs bed model's 20/20) — runaways are EOS losing
    # razor-thin ties to "one more audio frame". This adds a small bias to the
    # EOS logit, but ONLY once generation is past EOS_BOOST_START x the
    # chunk's expected token count (derived from text length), so it cannot
    # truncate speech that hasn't been spoken yet. Past that point the bias
    # ramps with the overrun. Default 0 = OFF; enable per-voice via
    # models.json backends.vllm.eosBoost, which reaches this either as the
    # registered 'eosBoost' cap (streaming) or as ORPHEUS_EOS_BOOST (audiobook
    # worker spawn env). vLLM paths only. Gate any tuning with eos_gate.py +
    # an endings ear-check.
    EOS_BOOST = float(os.environ.get('ORPHEUS_EOS_BOOST', '0.0'))
    EOS_BOOST_START = float(os.environ.get('ORPHEUS_EOS_BOOST_START', '1.2'))

    # Tail allowance for a chunk's expected token count, in AUDIO TOKENS
    # (2026-08-28, short-heading repeat). The expected-length estimate every
    # anti-runaway lever is sized from is `chars / 18.4 ch-per-s x 84 tokens-per-s`
    # — a pure SPEECH estimate that forgets the model's own trained tail. Orpheus
    # emits that tail as part of the generation (nothing trims it: the save-time
    # trim was removed 2026-07-11, NO-FALLBACK), and it does NOT scale with the
    # text: deathstalker's catalog measures modelSelfTailS 0.81 s, and the 52
    # short chunks of the witches render measured a mean trailing silence of
    # 0.824 s whether the chunk was 4 chars or 59.
    #
    # For a 450-char chunk that tail is ~3% of the clip and ignoring it is
    # harmless. For a 13-char heading it is most of the clip, so the estimate
    # lands 3-5x low — which is exactly why _eos_boost_processor carried a flat
    # `max(300.0, ...)` floor instead. That floor is what this replaces; see
    # _expected_audio_tokens for why a flat floor left short chunks unguarded.
    #
    # 96 tokens = 1.14 s at TOKENS_PER_AUDIO_SECOND, swept against those 52 real
    # chunks. Every value in [84, 110] engages the EOS boost before the known
    # doubled take ("Introduction.", 13 chars, 358 tokens) and engages on NO
    # healthy clip; below 84 healthy clips start being boosted mid-speech (at 71
    # "Turtles", at 60 "NEW AGE."), and above ~110 the doubled take escapes again.
    # 96 is the middle of that window because the two consumers want opposite
    # ends of it: the boost start wants a SMALL allowance (more ramp against a
    # runaway — 47 tokens of it here, vs 71 at the floor and 19 at the ceiling),
    # while _mlx_token_budget wants a LARGE one (its tightest healthy render,
    # "NEW AGE." at 194 tokens, clears the ceiling by 9.1% here vs 1.3% at 84).
    # Truncation is the worse failure of the two, so the tie goes to margin, and
    # the duration backstop below covers what a shorter ramp lets through.
    # A voice whose corpus keeps much longer tails wants a larger value.
    SHORT_CHUNK_TAIL_TOKENS = float(os.environ.get('ORPHEUS_SHORT_TAIL_TOKENS', '96'))

    # Short-chunk overrun REPORT (2026-08-28). A sub-floor chunk whose audio runs
    # far longer than its text can justify — the model saying a two-word heading
    # twice instead of stopping — is PRINTED and then left exactly as it is.
    #
    # THIS MEASURES A MODEL DEFECT. IT DOES NOT COMPENSATE FOR ONE. An earlier
    # version of this re-rendered the chunk and kept the shorter take; Owen removed
    # it (2026-08-28): "i dont want a guard. i want to fix it at the source." He is
    # right twice over. The text is provably single — the duplication is in the
    # generated AUDIO (385 tokens where a healthy render is ~130), i.e. the model
    # not committing to end-of-audio after a very short utterance, which is a
    # fine-tuning problem and is being fixed in the Orpheus fine-tune repo. And a
    # silent second render is exactly the kind of fallback this codebase refuses:
    # it would hide the very signal the retrain has to be judged by.
    #
    # So this exists to COUNT the defect across a whole book, before and after a
    # retrain, with one grep on the worker log:
    #     grep -c SHORT_CHUNK_OVERRUN <log>
    # Nothing is re-rendered, nothing is selected, nothing is thrown away.
    #
    # Scope: chars < SHORT_CHUNK_MAX_CHARS only — 25 is core.py's SENTENCE_MIN_CHARS
    # default, i.e. exactly the rows _apply_min_chars_floor used to merge away and
    # now exempts when they are headings. Nothing longer is judged at all. Set
    # ORPHEUS_SHORT_CHUNK_CHARS=0 to silence the report entirely (the same
    # "0 disables" convention as the sentence gap and the min-chars floor).
    #
    # The curve `max_seconds = 0.9 + 0.19 x chars` is measured, not guessed. Against
    # all 52 non-empty short chunks of the witches render it names the known doubled
    # take ("Introduction.", 13 chars, 4.267 s vs 3.37 s allowed) and NOTHING else —
    # including the genuinely slow one-word titles ("GOD." 1.280 s, "SATAN." 1.792 s,
    # "NEW AGE." 2.304 s), whose thinnest margin is 4.8%. It is a deliberately tight
    # fit meant to see a DOUBLING (~3x), not slow delivery.
    #
    # KNOWN IMPRECISION, stated so nobody reads the count as gospel: measured live
    # on 36 renders of "Introduction." (2026-08-28), 5 crossed this line while Owen
    # heard roughly 1-2 of 60 headings as doubled. Takes land either side of the
    # threshold at 3.157/3.243 s (under) and 3.755/3.840 s (over) and NOBODY HAS
    # LISTENED to those, so where "slow" ends and "doubled" begins is not yet known.
    # Treat the count as an upper bound and a trend line, not a defect tally.
    #
    # Judged on the generated waveform BEFORE _save_audio appends the inter-clip
    # gaps, so the threshold measures what the model emitted and is independent of
    # a voice's sentenceGap. The 0.9 s intercept carries the model's own tail
    # (deathstalker 0.81 s), so a voice with a much longer trained tail needs a
    # larger intercept — hence the env overrides.
    SHORT_CHUNK_OVERRUN_TAG = 'SHORT_CHUNK_OVERRUN'

    # One machine-readable line per guard fire, on stdout, so an orchestrator can
    # count runaways and truncations without reaching into the worker's filesystem
    # (on Windows the worker runs inside WSL, where its scratch is not a path the
    # host can conveniently read). The prose prints stay: they are what a person
    # watching the console reads. This is the same event, in a form a parser can
    # trust — tag, then one compact JSON object, on a single line.
    GUARD_EVENT_TAG = 'ORPHEUS_GUARD_EVENT'
    SHORT_CHUNK_MAX_CHARS = int(os.environ.get('ORPHEUS_SHORT_CHUNK_CHARS', '25'))
    SHORT_CHUNK_SECONDS_BASE = float(os.environ.get('ORPHEUS_SHORT_CHUNK_SECONDS_BASE', '0.9'))
    SHORT_CHUNK_SECONDS_PER_CHAR = float(os.environ.get('ORPHEUS_SHORT_CHUNK_SECONDS_PER_CHAR', '0.19'))

    # ── Per-voice tuning caps ────────────────────────────────────────────────
    # Every constant above is a DEFAULT, never the live value. One process can
    # serve more than one voice — the BookForge streaming worker switches voices
    # in-process against a resident engine, and per-character casting will mix
    # voices inside a single batch — while each fine-tune wants its own numbers
    # (deathstalker reads ~23.5 ch/s where the default guard is 19.0; the
    # bed-free voices need an EOS boost the stock ones must not get). Reading a
    # class attribute bound at IMPORT pins whichever voice happened to be loaded
    # first onto every later one, which is exactly the bug the streaming server
    # worked around by rewriting os.environ per voice (process-global state that
    # cannot describe a mixed-voice batch at all).
    #
    # So the live value is resolved per call by _voice_cap(), in this order:
    #   1. a cap registered for that voice (register_voice_caps)
    #   2. the process environment (ORPHEUS_*) — how the AUDIOBOOK worker is
    #      configured: parallel-tts-bridge spawns one process per voice and
    #      exports the catalog's values into it, so that path is unchanged
    #   3. the class default above
    #
    # Keys are the camelCase names BookForge's catalog already uses
    # (orpheusVoiceCapsForModel in electron/orpheus-models.ts), so a caps payload
    # crosses the process boundary verbatim. Value = (env var, class attr).
    VOICE_CAP_SOURCES = {
        'temperature':    ('ORPHEUS_TEMPERATURE',       'TEMPERATURE'),
        'topP':           ('ORPHEUS_TOP_P',             'TOP_P'),
        'minP':           ('ORPHEUS_MIN_P',             'MIN_P'),
        'repPenalty':     ('ORPHEUS_REP_PENALTY',       'REP_PENALTY'),
        'eosBoost':       ('ORPHEUS_EOS_BOOST',         'EOS_BOOST'),
        'eosBoostStart':  ('ORPHEUS_EOS_BOOST_START',   'EOS_BOOST_START'),
        'maxCharsPerSec': ('ORPHEUS_MAX_CHARS_PER_SEC', 'DEFAULT_MAX_CHARS_PER_SEC'),
    }

    # Catalog caps that are NOT generation tuning and are consumed elsewhere:
    # maxChars is a PREP concern (how core.py packs sentences into chunks) and
    # sentenceGap is an ASSEMBLY concern (_classify_gap / ORPHEUS_SENTENCE_GAP).
    # They ride the same catalog payload, so register_voice_caps accepts and
    # ignores them by NAME rather than swallowing anything it doesn't recognise —
    # an unknown key is a typo or a catalog/e2a version skew and must be loud.
    VOICE_CAP_IGNORED = ('maxChars', 'sentenceGap')

    # voice token -> {cap key: float}. CLASS-level, not an instance attribute and
    # deliberately NOT in loaded_tts: the streaming worker rebuilds its Orpheus
    # instance whenever the model changes, and loaded_tts is the MODEL cache —
    # lib/core.cleanup_models_cache() deletes every key in it that doesn't name an
    # active model, which would drop the caps and silently revert every voice to
    # the defaults. Caps hold no weights, so keeping them for the life of the
    # process costs nothing.
    #
    # The LoRA registry, by contrast, BELONGS in loaded_tts: its ids key a cache
    # that lives inside the vLLM engine object, so it must die exactly when the
    # engine does. _evict_global_cache drops both together.
    #
    # cleanup_models_cache (the single-process GUI path) also wipes loaded_tts, and
    # that is HARMLESS here for a reason that has nothing to do with object
    # lifetimes: the GUI path can never have an adapter registered in the first
    # place. orpheus_adapter_dir is produced by exactly two callers — worker.py /
    # bookforge_ext.parallel (the BookForge audiobook worker) and BookForge's
    # streaming server — and neither ever calls cleanup_models_cache. A session
    # built by app.py/convert_ebook carries no adapter and no base, so
    # _lora_request() returns None unconditionally on that path and there is no
    # registry for a wipe to desynchronise.
    _voice_caps = {}

    # Special token IDs
    END_OF_AUDIO_TOKEN = 128258

    # Device tracking
    _device = None

    def __init__(self, session: DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.resampler_cache = {}
            self.audio_segments = []
            self.backend = None  # 'mlx', 'vllm', or 'transformers'
            self.snac_model = None
            self.tokenizer = None
            self.mlx_model = None  # For MLX backend

            # Session-scoped self-calibrating ceiling for the truncation guard's
            # chars/sec threshold (see _guard_truncation). Starts empty (no
            # ratchet); the guard only ever raises it, and only from rates
            # measured on force-split re-renders (which cannot be truncated).
            #
            # Keyed BY VOICE, because the thing it calibrates is one voice's natural
            # speaking rate: a resident streaming engine serves several voices, and a
            # ceiling ratcheted up by a fast fine-tune (deathstalker ~23.5 ch/s) would
            # otherwise disarm the guard for a slow one still reading at ~15.
            self._rate_ceilings = {}

            # Try to load presets, but don't fail if they're missing
            try:
                self.models = load_engine_presets(self.session['tts_engine'])
            except Exception:
                self.models = {}

            self.params = {}
            self.params['samplerate'] = self.SAMPLE_RATE

            # Custom Orpheus model (BookForge folder-discovered voices).
            # orpheus_model_dir is an absolute path to a HF/MLX model folder whose
            # FOLDER NAME is the voice token it was fine-tuned on (e.g. .../owen →
            # token "owen"). When present, point every backend at the local dir and
            # use the token verbatim — these single-speaker finetunes are NOT in the
            # built-in allowlist, so the usual validation would wrongly drop them to
            # the default voice (the leah fallback bug).
            self.custom_model_dir = self.session.get('orpheus_model_dir')

            # Adapter mode (see the LORA_* block at the top of the class): the served
            # weights are orpheus_base_dir and the voice arrives per request as a
            # LoRARequest built from orpheus_adapter_dir. Validated NOW so a broken
            # install fails before the engine loads, never mid-book and never by
            # quietly rendering the base voice.
            #
            # orpheus_base_dir WITHOUT an adapter is the third legal shape:
            # "stock-from-local-base" (see _validate_adapter_mode).
            self.adapter_dir = self.session.get('orpheus_adapter_dir')
            self.base_dir = self.session.get('orpheus_base_dir')
            self._validate_adapter_mode()

            # Get voice from session or preset
            voice = self.session.get('fine_tuned', self.DEFAULT_VOICE)
            print(f"[ORPHEUS] Session fine_tuned value: '{voice}'")

            if self.custom_model_dir:
                self.MLX_MODEL = self.custom_model_dir
                self.TRANSFORMERS_MODEL = self.custom_model_dir
                self.voice = (voice or '').strip().lower() or self.DEFAULT_VOICE
                print(f"[ORPHEUS] Custom model dir: {self.custom_model_dir}")
                print(f"[ORPHEUS] Using custom voice token: '{self.voice}'")
            elif self.adapter_dir:
                # The base is the model that gets SERVED; the adapter carries the
                # voice. The token is taken verbatim for the same reason the custom
                # branch above takes it verbatim (a fine-tune's token is not in the
                # allowlist, so validation would drop it to leah), and it is REQUIRED
                # here: it is also the adapter's registry key, so there is nothing
                # sane to default to.
                #
                # BOTH model refs point at the base, because on BOTH backends that can
                # serve an adapter the base is the thing that gets LOADED and the
                # adapter is applied to it afterwards: vLLM per request (LoRARequest),
                # MLX by wrapping the projection modules of the resident model
                # (_apply_mlx_adapter — stage B2). The base dir is an HF-format
                # bf16 checkpoint, which is exactly what mlx_audio's load_model reads
                # for the legacy merged Mac voices, so nothing about the load changes.
                self.TRANSFORMERS_MODEL = self.base_dir
                self.MLX_MODEL = self.base_dir
                self.voice = (voice or '').strip().lower()
                # 'internal' is conf_models.default_fine_tuned — the sentinel both
                # session builders substitute when --fine_tuned was never passed, so
                # an empty check alone can never fire. Rendering with "internal: "
                # would apply the right LoRA under a prompt token the adapter was
                # never trained on: audio that sounds plausible and is wrong.
                if self.voice in ('', 'internal'):
                    raise ValueError(
                        f'Orpheus adapter mode ({self.adapter_dir}) needs --fine_tuned: '
                        'the voice token the adapter was trained on.'
                    )
                print(f"[ORPHEUS] Adapter base dir: {self.base_dir}")
                print(f"[ORPHEUS] Adapter dir: {self.adapter_dir}")
                print(f"[ORPHEUS] Using adapter voice token: '{self.voice}'")
            else:
                # STOCK-FROM-LOCAL-BASE: orpheus_base_dir with no adapter. The weights
                # served are the same unsloth/orpheus-3b-0.1-ft checkpoint the stock
                # path pulls from the HF cache — just read from the local `_base`
                # folder instead — so the voice is still an allowlisted prompt prefix
                # and validation below is unchanged. The point is the ENGINE: on vLLM
                # it is built with enable_lora (see _load_vllm_engine), so a stock
                # session and an adapter session over the same base are the SAME
                # engine, and switching between them costs a registration instead of a
                # 6 GB reload plus a possible mid-session HF download.
                if self.base_dir:
                    # MLX_MODEL moves with it: the engine cache key is the (merged,
                    # base) pair, so a stock session naming this base and an adapter
                    # session naming it are the SAME engine — and they had better be
                    # the same weights. Leaving MLX pointed at mlx-community's
                    # converted copy while the key claims the local dir is how a warm
                    # switch ends up serving a checkpoint nobody named. Same
                    # fine-tune either way (unsloth/orpheus-3b-0.1-ft), so the audio
                    # is unchanged; only the honesty of the key is.
                    self.TRANSFORMERS_MODEL = self.base_dir
                    self.MLX_MODEL = self.base_dir
                    print(f"[ORPHEUS] Stock voice served from local base dir: {self.base_dir}")

                # Handle preset lookups
                if voice in self.models:
                    voice = self.models[voice].get('voice', voice)

                # Normalize to lowercase for comparison
                voice_lower = voice.lower() if voice else self.DEFAULT_VOICE

                # Validate voice
                if voice_lower not in self.VALID_VOICES:
                    print(f"Warning: Unknown Orpheus voice '{voice}', defaulting to '{self.DEFAULT_VOICE}'")
                    voice_lower = self.DEFAULT_VOICE

                self.voice = voice_lower
                print(f"[ORPHEUS] Using voice: '{self.voice}'")

            self.engine = None
            self.engine = self.load_engine()

            # Register this instance for cleanup on exit
            _active_instances.add(self)

        except Exception as e:
            error = f'Orpheus.__init__() error: {e}'
            raise ValueError(error)

    def _validate_adapter_mode(self):
        """Reject a half-configured adapter mode before anything loads.

        Every case here would otherwise still PRODUCE audio — in the base voice, in
        a merged voice, or in leah — which is indistinguishable from success until
        someone listens to the finished book. There is no fallback to stock or to
        merged mode: a voice that cannot be served as asked is a hard error.

        THREE legal shapes, not two:

          (model_dir, None)          MERGED — the voice IS the weights.
          (None, base+adapter)       ADAPTER — a LoRA over the shared base.
          (None, base only)          STOCK-FROM-LOCAL-BASE — the built-in voices,
                                     served from the local `_base` copy of
                                     unsloth/orpheus-3b-0.1-ft instead of the HF
                                     cache, on an engine built LoRA-capable.

        base-without-adapter used to be fatal. It is now the deliberate KEY COLLAPSE
        that makes stock↔adapter switching free on the resident streaming server:
        both sessions name the same base, so they are the same engine, so no 6 GB
        reload and no chance of a mid-session HF download on a cold cache.

        The safety argument that makes it acceptable to have a "no adapter attached"
        render path on an adapter-capable engine: a CUSTOM voice can never reach it.
        Only tokens in VALID_VOICES are accepted by the stock branch of __init__ and
        by the streaming server's _row_voice, and every custom voice must have been
        registered against the LIVE engine before a request may name it. An unknown
        or unregistered custom voice fails loudly at _row_voice / resolveRequestVoice
        (BookForge's pool) — it can never fall through to a bare-base render. Only
        allowlisted stock tokens ever ride the lora_request=None path.

        ADAPTER-without-base is still fatal: an adapter has nothing to apply itself
        to, and the base must already be installed — a render job never downloads it
        mid-job.
        """
        if self.adapter_dir and self.custom_model_dir:
            raise ValueError(
                f'Orpheus got both orpheus_model_dir ({self.custom_model_dir}) and '
                f'orpheus_adapter_dir ({self.adapter_dir}). They select different '
                'weights and only one voice can be rendered — pass exactly one.'
            )
        if self.base_dir and self.custom_model_dir:
            raise ValueError(
                f'Orpheus got both orpheus_model_dir ({self.custom_model_dir}) and '
                f'orpheus_base_dir ({self.base_dir}). A merged fine-tune IS its own '
                'weights and is never served on top of a base — pass exactly one.'
            )
        if self.adapter_dir and not self.base_dir:
            raise ValueError(
                f'Orpheus adapter mode ({self.adapter_dir}) requires orpheus_base_dir. '
                'The base model must already be installed locally — a render job never '
                'downloads it mid-job.'
            )
        if not self.base_dir:
            return
        base_config = os.path.join(self.base_dir, 'config.json')
        if not os.path.isfile(base_config):
            raise ValueError(
                f'Orpheus base dir {self.base_dir} is not a model folder: {base_config} is missing'
            )
        if self.adapter_dir:
            # UNIVERSAL checks only: the backend has not been detected yet (that happens
            # in load_engine, which re-validates against it before loading any weights).
            # An adapter that is broken for everyone should still fail here, at
            # construction, rather than after a 6 GB load.
            self.validate_adapter_dir(self.adapter_dir)

    @classmethod
    def validate_adapter_dir(cls, adapter_dir: str, backend: str = None):
        """Every check an adapter folder must pass before it is asked to serve a voice.

        Extracted so ENGINE CONSTRUCTION (_validate_adapter_mode, first adapter) and
        LATE REGISTRATION (register_adapter / set_voice, the 2nd..Nth adapter added to
        a warm engine on the streaming server) run the IDENTICAL validation. They used
        not to: only __init__ validated, so a bad adapter registered onto a live engine
        detonated inside engine.generate() — an opaque vLLM error that fails the
        whole batch, at render time, with nothing naming the adapter. A bad adapter
        must fail at REGISTRATION, with the readable reason, every time.

        `backend` selects which BACKEND-SPECIFIC checks run on top of the universal
        ones. Two backends can now apply a LoRA and they refuse different things, so
        parroting one's limits at the other is worse than not checking: a Mac told
        that its adapter exceeds "max_lora_rank, which vLLM 0.7.3 cannot serve" is
        being read an error from software it does not have installed. None (the
        default) runs the universal checks alone — that is what __init__ does, before
        the backend has been detected; load_engine re-validates with the real backend
        before it loads any weights.
        """
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            path = os.path.join(adapter_dir, name)
            if not os.path.isfile(path):
                raise ValueError(f'Orpheus adapter dir {adapter_dir} is missing {name}')
        path = os.path.join(adapter_dir, 'adapter_config.json')
        config = cls._read_adapter_config(adapter_dir)
        cls._validate_adapter_config(path, config)
        if backend == 'vllm':
            cls._validate_adapter_config_vllm(path, config)
        elif backend == 'mlx':
            cls._validate_adapter_config_mlx(path, config)

    @staticmethod
    def _read_adapter_config(adapter_dir: str) -> dict:
        """The adapter's PEFT config as a dict. One reader, so validation and the MLX
        applier can never disagree about what the file said."""
        import json as _json
        with open(os.path.join(adapter_dir, 'adapter_config.json'), 'r', encoding='utf-8') as handle:
            return _json.load(handle)

    @classmethod
    def _validate_adapter_config(cls, path: str, config: dict):
        """The checks that hold for ANY applier of a LoRA, on any backend.

        Each one names something that would otherwise be silently DROPPED from the
        voice — the failure mode this whole file is built against, because dropping
        part of a fine-tune still produces confident, finished-sounding audio."""
        if config.get('peft_type', 'LORA') != 'LORA':
            raise ValueError(
                f'Orpheus adapter {path}: peft_type={config["peft_type"]!r}. Only plain '
                'LoRA adapters can be applied — anything else would be ignored.'
            )
        rank = config.get('r')
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f'Orpheus adapter {path}: r={rank!r} is not a positive integer rank')
        if config.get('modules_to_save') is not None:
            raise ValueError(
                f'Orpheus adapter {path}: modules_to_save={config["modules_to_save"]!r}. '
                'Only the LoRA A/B pairs are applied, so fully-saved modules would be '
                'silently dropped from the voice.'
            )
        if config.get('use_dora'):
            raise ValueError(
                f'Orpheus adapter {path}: use_dora=true. DoRA rescales each column of the '
                'base weight as well as adding the low-rank delta; applying only the '
                'delta would render a voice that is not the one that was trained.'
            )
        bias = config.get('bias', 'none')
        if bias != 'none':
            raise ValueError(
                f"Orpheus adapter {path}: bias={bias!r}. Only the A/B pairs are applied, "
                'so a trained bias term would be dropped.'
            )

    @classmethod
    def _validate_adapter_config_vllm(cls, path: str, config: dict):
        """What vLLM 0.7.3 in particular refuses, HERE, where the reason is readable.

        PEFTHelper.validate_legal checks the rank against the engine's max_lora_rank
        deep inside engine startup, where the failure surfaces as an opaque worker
        crash with no mention of the adapter."""
        rank = config.get('r')
        if rank > cls.LORA_MAX_RANK:
            raise ValueError(
                f'Orpheus adapter {path}: r={rank!r}, which the engine cannot serve at '
                f'max_lora_rank={cls.LORA_MAX_RANK}'
            )

    @classmethod
    def _validate_adapter_config_mlx(cls, path: str, config: dict):
        """What the MLX applier in particular refuses (see _apply_mlx_adapter).

        No rank ceiling: MLX has no pre-allocated LoRA slots to overflow — the
        wrappers are sized from the weights themselves, so any rank works.

        Per-module rank/alpha overrides do NOT work: _apply_mlx_adapter resolves ONE
        scale from r and lora_alpha and applies it to every wrapped projection.
        Honouring the patterns would be a small change; pretending they are honoured
        while scaling half the model wrongly is a voice that is subtly not the one
        that was trained, with nothing to see."""
        for key in ('rank_pattern', 'alpha_pattern'):
            if config.get(key):
                raise ValueError(
                    f'Orpheus adapter {path}: {key}={config[key]!r}. The MLX applier uses '
                    'one scale (lora_alpha / r) for every wrapped projection and cannot '
                    'honour per-module overrides.'
                )

    @staticmethod
    def _evict_global_cache():
        """Drop the process-global Orpheus model cache so the next load fetches a
        fresh model. Used when switching to a different orpheus_model_dir or
        orpheus_base_dir and on cleanup, so a custom voice's weights actually free
        instead of being reused.

        The LoRA registry (orpheus_lora_ids / orpheus_lora_paths / the fingerprints
        / the id counter) goes WITH the engine. lora_int_ids only ever need to be
        unique for the life of the ENGINE that caches weights under them — the cache
        is vLLM-internal and dies with the LLM object — so an engine reload starts
        from an empty adapter cache and ids may safely restart at 1.

        Keeping the registry across a reload was worse than useless: it made
        _register_lora refuse to re-point a voice at a different adapter dir
        (retraining a voice would have needed a process restart on the resident
        streaming server), while protecting an id space that no longer had anything
        cached under it.

        Note this is not what protects the single-process GUI path from a stale
        registry — see the _voice_caps block at the top of the class. That path never
        sets orpheus_adapter_dir at all, so it has no registry to go stale."""
        for k in ('orpheus', 'orpheus_mlx_model', 'orpheus_snac', 'orpheus_tokenizer',
                  'orpheus_backend', 'orpheus_device', 'orpheus_model_dir',
                  'orpheus_base_dir', 'orpheus_lora_ids', 'orpheus_lora_paths',
                  'orpheus_lora_fingerprints', 'orpheus_lora_next_id'):
            if k in loaded_tts:
                try:
                    del loaded_tts[k]
                except Exception:
                    loaded_tts[k] = None

    def cleanup(self):
        """Explicitly release all resources (CUDA, vLLM, etc.)"""
        try:
            # Delete vLLM engine first (releases GPU memory)
            if hasattr(self, 'engine') and self.engine is not None:
                del self.engine
                self.engine = None

            # Delete SNAC decoder
            if hasattr(self, 'snac_model') and self.snac_model is not None:
                del self.snac_model
                self.snac_model = None

            # Delete tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # If the process-global cache holds THIS instance's model, evict it too so
            # the weights actually free and the next load reloads. Guarded by BOTH
            # model keys (merged dir and adapter base dir) so a stale instance's
            # __del__ can't clobber a newer, different model.
            try:
                if (loaded_tts.get('orpheus_model_dir', '\0') == getattr(self, 'custom_model_dir', None)
                        and loaded_tts.get('orpheus_base_dir', '\0') == getattr(self, 'base_dir', None)):
                    self._evict_global_cache()
            except Exception:
                pass
            self.mlx_model = None

            # Clear CUDA cache
            self._cleanup_memory()
            print("[ORPHEUS] Cleanup complete - resources released")
        except Exception as e:
            print(f"[ORPHEUS] Cleanup warning: {e}")

    def __del__(self):
        """Destructor - ensure cleanup when object is garbage collected"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction

    @classmethod
    def detect_backend(cls) -> str:
        """PUBLIC: which backend Orpheus WOULD use in this process, without loading
        anything.

        BookForge's streaming server reports this on its 'ready' line so the pool
        knows, authoritatively and before any voice is loaded, whether the engine can
        serve a voice PER REQUEST. Only vLLM can (multi-LoRA is a vLLM feature, and
        MLX builds one sampler per batch bucket from the engine's own caps, so even a
        stock per-row prompt token would carry the wrong voice's tuning). Guessing
        that from the OS or from a torch.cuda probe is exactly the kind of parallel
        second implementation that drifts — so it delegates to the same function
        load_engine uses.
        """
        return cls._detect_backend()

    @staticmethod
    def _detect_backend() -> str:
        """Detect best available backend for this platform.

        Priority:
        1. MLX on Mac (fastest, ~1.4x realtime)
        2. vLLM on CUDA (fast, good for Windows/Linux)
        3. Transformers (slow fallback, ~27x realtime on Mac MPS)
        """
        is_mac = platform.system() == 'Darwin'
        has_cuda = torch.cuda.is_available()

        # Check for backend override via environment variable
        # ORPHEUS_BACKEND can be: mlx, vllm, transformers
        forced_backend = os.environ.get('ORPHEUS_BACKEND', '').lower()
        if forced_backend:
            print(f"Orpheus: Backend override via ORPHEUS_BACKEND={forced_backend}")
            if forced_backend in ('mlx', 'vllm', 'transformers'):
                return forced_backend
            else:
                print(f"Warning: Unknown backend '{forced_backend}', using auto-detect")

        # Try MLX first on Mac (19x faster than transformers!)
        if is_mac:
            try:
                from mlx_audio.tts.utils import load_model
                print("Orpheus: Using MLX backend (Apple Silicon optimized)")
                return 'mlx'
            except ImportError:
                print("Orpheus: MLX not available (install with: pip install mlx-audio)")

        # Try vLLM on CUDA (best for Windows/Linux)
        if has_cuda and not is_mac:
            try:
                from vllm import LLM
                print("Orpheus: Using vLLM backend (CUDA detected)")
                return 'vllm'
            except ImportError:
                print("Orpheus: vLLM not available, trying transformers...")

        # Fall back to transformers (works everywhere but slow on Mac)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            backend_device = "MPS" if is_mac else ("CUDA" if has_cuda else "CPU")
            print(f"Orpheus: Using transformers backend ({backend_device})")
            if is_mac:
                print("WARNING: Transformers on Mac MPS is ~27x slower than MLX!")
                print("         Install mlx-audio for much better performance: pip install mlx-audio")
            return 'transformers'
        except ImportError:
            raise ImportError(
                "No Orpheus backend available. Install one of:\n"
                "  Mac: pip install mlx-audio\n"
                "  Windows/Linux: pip install vllm (requires CUDA)\n"
                "  Fallback: pip install transformers"
            )

    def _load_mlx_engine(self):
        """Load model using MLX backend (Mac only, fastest)."""
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        # Bound the MLX buffer cache. Without a limit the allocator's cache of
        # freed buffers grows to ~46 GB over a single batched chunk (measured,
        # M1 Ultra: SNAC decode + per-step generation buffers come in many
        # distinct sizes, so freed buffers pile up per size-bucket) — that's the
        # memory-pressure spike, NOT the active weights+KV (~22 GB at batch 96).
        # A bounded cache also beats the old per-chunk mx.clear_cache() flush:
        # buffers still get reused (a full flush forces cold Metal re-allocation
        # every chunk) and the footprint stays flat for the whole run.
        # Measured at batch 96: 8 GB limit = 28.1 sent/min vs 27.2 flush-per-chunk.
        cache_gb = float(os.environ.get('ORPHEUS_MLX_CACHE_LIMIT_GB', '8'))
        mx.set_cache_limit(int(cache_gb * 1e9))
        print(f"Orpheus MLX buffer cache limited to {cache_gb:g} GB")
        # Fail fast on an impossible memory budget, before any audio is rendered.
        headroom = self._mlx_kv_headroom_gb()
        print(f"Orpheus MLX batch budget {self.MLX_MEM_BUDGET_GB:g} GB "
              f"({headroom:.1f} GB for KV → max {self._mlx_width_for_depth(self.MLX_MAX_TOKENS)} "
              f"rows at the {self.MLX_MAX_TOKENS}-token cap)")

        print(f"Loading Orpheus model with MLX: {self.MLX_MODEL}")
        model = load_model(self.MLX_MODEL)
        # Fine-tuned Orpheus voices were TRAINED with the prompt ending in
        # [128261, 128257] (START_OF_AI, START_OF_SPEECH) immediately before the audio
        # codes (orpheus_owen.py build_dataset; mirrored by _format_prompt_ids). But
        # mlx_audio's prepare_input_ids stops at [128009, 128260] (END_OF_TEXT,
        # END_OF_HUMAN) and relies on the BASE model to free-generate SOA/SOS — which is
        # out-of-distribution for our fine-tunes, so they vocalize the voice token /
        # stray syllables at chunk starts. The vLLM/transformers backends frame this
        # correctly via _format_prompt_ids; MLX bypasses that helper, so patch the
        # library's framing at load to restore the exact training frame.
        self._patch_mlx_prompt_framing(model)
        self._device = 'mlx'  # MLX manages its own device
        print("Orpheus MLX model loaded!")
        return model

    def _patch_mlx_prompt_framing(self, model):
        """Make mlx_audio frame fine-tuned voices exactly like training: append
        [START_OF_AI, START_OF_SPEECH] to the plain-voice prompt. Wrapping
        prepare_input_ids (which model.generate() also calls, llama.py:396) covers
        EVERY MLX path in one place — single (_generate_mlx), batch/streaming
        (_generate_mlx_batch_audio) and audiobook (_convert_mlx_batch). Voice-cloning
        calls (ref_audio/ref_text -> tuple return) are left untouched."""
        import mlx.core as mx
        orig = model.prepare_input_ids
        AI_SPEECH = mx.array([[128261, 128257]], dtype=mx.int64)  # START_OF_AI, START_OF_SPEECH

        def prepare_input_ids(prompt, voice=None, zeroprompt=None, ref_audio=None,
                              ref_text=None, *args, **kwargs):
            ids = orig(prompt, voice, zeroprompt, ref_audio, ref_text, *args, **kwargs)
            plain = (voice is not None and zeroprompt is None
                     and ref_audio is None and ref_text is None)
            # Only the plain fine-tuned-voice frame ([SOH]..[EOT EOH]) is missing the
            # SOA/SOS suffix; guard on the exact tail so nothing else is altered.
            if plain and not isinstance(ids, tuple) and ids.shape[0] == 1 \
                    and ids[0].tolist()[-2:] == [128009, 128260]:
                ids = mx.concatenate([ids, AI_SPEECH], axis=1)
            return ids

        model.prepare_input_ids = prepare_input_ids
        print("Orpheus MLX prompt framing patched for fine-tuned voices (SOA+SOS suffix)")

    # ── MLX adapter mode (stage B2) ──────────────────────────────────────────
    #
    # ONE base model stays resident and a voice is a LoRA WRAPPED ONTO its
    # projection modules; switching voices swaps the wrappers instead of reloading
    # 6.2 GB. This is what makes a mid-article voice switch on a Mac cost ~a second
    # rather than a minute, and it is why the darwin fuse-at-install preference in
    # BookForge's resolveOrpheusInstall is gone.
    #
    # It is NOT vLLM's per-request multi-LoRA. MLX has exactly ONE adapter attached
    # at a time, because the adapter is part of the weights the forward pass runs
    # through — there is no per-row selection to be had. Every caller that could mix
    # voices in one batch is gated on the vLLM backend (BookForge's
    # canServeVoicePerRequest, orpheus_stream's _reject_per_request_voice), and
    # _lora_request is never reached on this backend.
    #
    # PEFT stores a LoRA as, per targeted projection:
    #   base_model.model.model.layers.N.<self_attn|mlp>.<proj>.lora_A.weight  (r, in)
    #   base_model.model.model.layers.N.<self_attn|mlp>.<proj>.lora_B.weight  (out, r)
    # and mlx_audio's Orpheus model IS mlx_lm's Llama (mlx_audio.tts.models.llama
    # .Model extends mlx_lm.models.llama.Model), so stripping the `base_model.model.`
    # PEFT prefix leaves a path that walks the LIVE object verbatim:
    # model.model.layers[N].self_attn.q_proj. The remap is a prefix strip and an
    # attribute walk — no per-name table to fall out of date with either library.
    MLX_LORA_PREFIX = 'base_model.model.'

    @staticmethod
    def _mlx_lora_scale(config: dict) -> float:
        """The one multiplier applied to every LoRA delta: PEFT's own alpha / r
        (alpha / sqrt(r) under rsLoRA).

        READ FROM THE ADAPTER, never assumed. The three deployed Orpheus voices are
        r=64 / alpha=64, i.e. exactly 1.0, so a hardcoded scale would work today and
        silently mis-weight the first voice trained at any other setting — by a
        factor of 2 or 4, which is not a subtle artifact but it is a plausible-
        sounding one.
        """
        rank = config['r']
        alpha = config.get('lora_alpha', rank)
        if config.get('use_rslora'):
            return float(alpha) / math.sqrt(rank)
        return float(alpha) / float(rank)

    @classmethod
    def _mlx_walk(cls, model, path: str):
        """Resolve a dotted PEFT module path against the live MLX model."""
        obj = model
        for part in path.split('.'):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def _mlx_adapter_plan(self, model, adapter_dir: str, config: dict):
        """Build every wrapper this adapter needs, WITHOUT touching the model.

        Returns (sites, swap) where `sites` is the new _MlxAdapterState.sites and
        `swap` is the complete list of (parent, attr, module) assignments that make
        the model serve this adapter and nothing else — including restoring any site
        the PREVIOUS adapter wrapped that this one does not.

        Everything that can fail — a missing pair, an unexpected module path, a shape
        that does not match the base weight — fails HERE, with the model still
        rendering the voice it was rendering. That is the whole reason this is a
        separate pass: a half-wrapped model has no name, cannot be reported, and
        sounds like neither voice.
        """
        import mlx.core as mx
        import mlx.nn as nn

        weights = mx.load(os.path.join(adapter_dir, 'adapter_model.safetensors'))
        scale = self._mlx_lora_scale(config)

        # Group the flat key list into {module path: {lora_A: …, lora_B: …}}.
        pairs = {}
        for key, value in weights.items():
            if not key.startswith(self.MLX_LORA_PREFIX):
                raise ValueError(
                    f'Orpheus adapter {adapter_dir}: key {key!r} does not start with '
                    f'{self.MLX_LORA_PREFIX!r} — this is not a PEFT LoRA over a '
                    'transformers causal LM.'
                )
            body = key[len(self.MLX_LORA_PREFIX):]
            if not body.endswith('.weight'):
                raise ValueError(f'Orpheus adapter {adapter_dir}: unexpected key {key!r}')
            body = body[:-len('.weight')]
            module_path, _, which = body.rpartition('.')
            if which not in ('lora_A', 'lora_B'):
                raise ValueError(
                    f'Orpheus adapter {adapter_dir}: key {key!r} is neither a lora_A nor a '
                    'lora_B weight. Only the A/B pairs are applied, so it would be dropped.'
                )
            pairs.setdefault(module_path, {})[which] = value

        # The ORIGINAL module at each site, even when a previous adapter is currently
        # wrapped around it: the recorded state is the authority, the live object is
        # only consulted for sites nothing has wrapped.
        state = getattr(model, '_orpheus_mlx_lora', None)
        originals = {path: (parent, attr, original)
                     for path, parent, attr, original in (state.sites if state else [])}

        lora_linear = _mlx_lora_linear_cls()
        sites = []
        swap = []
        for module_path in sorted(pairs):
            entry = pairs[module_path]
            lora_a, lora_b = entry.get('lora_A'), entry.get('lora_B')
            if lora_a is None or lora_b is None:
                raise ValueError(
                    f'Orpheus adapter {adapter_dir}: {module_path} has only '
                    f'{"lora_A" if lora_a is not None else "lora_B"}. Half a LoRA pair '
                    'cannot be applied.'
                )
            if module_path in originals:
                parent, attr, original = originals[module_path]
            else:
                parent_path, _, attr = module_path.rpartition('.')
                try:
                    parent = self._mlx_walk(model, parent_path)
                    original = getattr(parent, attr)
                except (AttributeError, IndexError, KeyError, TypeError) as err:
                    raise ValueError(
                        f'Orpheus adapter {adapter_dir}: {module_path} does not name a module '
                        f'on the loaded MLX model ({err}). The adapter was trained against a '
                        'different architecture than the base model this engine loaded.'
                    )
            if not isinstance(original, nn.Linear):
                raise ValueError(
                    f'Orpheus adapter {adapter_dir}: {module_path} is a '
                    f'{type(original).__name__}, not an nn.Linear. A LoRA can only be applied '
                    'to a plain linear projection (a quantized base would need its own path).'
                )
            weight = original.weight
            out_features, in_features = weight.shape
            if (lora_a.shape[1] != in_features or lora_b.shape[0] != out_features
                    or lora_a.shape[0] != lora_b.shape[1]):
                raise ValueError(
                    f'Orpheus adapter {adapter_dir}: {module_path} shapes do not fit — '
                    f'A{tuple(lora_a.shape)} B{tuple(lora_b.shape)} against a '
                    f'{out_features}x{in_features} projection.'
                )
            # Cast to the BASE's dtype (bf16). The checkpoint stores fp32, and adding an
            # fp32 delta to a bf16 projection would promote every activation on the way
            # through — a different, slower model than the one that was measured.
            wrapper = lora_linear(original, lora_a.astype(weight.dtype),
                                  lora_b.astype(weight.dtype), scale)
            sites.append((module_path, parent, attr, original))
            swap.append((parent, attr, wrapper))

        # Sites the OLD adapter wrapped and this one does not: restore the original,
        # or they would keep serving the previous voice's delta on top of the new one.
        wrapped = {path for path, _, _, _ in sites}
        for path, parent, attr, original in (state.sites if state else []):
            if path not in wrapped:
                swap.append((parent, attr, original))
        return sites, swap

    def _apply_mlx_adapter(self, model, adapter_dir: str, fingerprint=None) -> None:
        """Make the resident MLX model render `adapter_dir`'s voice.

        Replaces whatever adapter is currently applied (see _mlx_adapter_plan), so
        adapter→adapter is one call, not a clear followed by an apply — there is no
        window in which the model is a bare base under a custom voice's name.

        ALL-OR-NOTHING in both phases. The plan is built first and every failure mode
        lives there, with the model untouched; the assignment loop that follows is
        pure setattr, and if one of those ever raised, every site already moved is put
        back exactly as it was before the exception propagates.
        """
        import mlx.core as mx

        config = self._read_adapter_config(adapter_dir)
        self._validate_adapter_config(os.path.join(adapter_dir, 'adapter_config.json'), config)
        self._validate_adapter_config_mlx(os.path.join(adapter_dir, 'adapter_config.json'), config)
        if fingerprint is None:
            fingerprint = self._adapter_fingerprint(adapter_dir)
        sites, swap = self._mlx_adapter_plan(model, adapter_dir, config)

        rollback = [(parent, attr, getattr(parent, attr)) for parent, attr, _ in swap]
        try:
            for parent, attr, module in swap:
                setattr(parent, attr, module)
        except Exception:
            for parent, attr, previous in rollback:
                setattr(parent, attr, previous)
            raise
        model._orpheus_mlx_lora = _MlxAdapterState(adapter_dir, fingerprint, sites)
        # Materialise the new weights NOW. mx.load is lazy, so without this the first
        # sentence of a switched voice pays a 0.4 GB read inside the generation loop —
        # which is exactly the latency the resident base was supposed to remove.
        mx.eval(model.parameters())
        print(f'[ORPHEUS] MLX adapter applied: {adapter_dir} '
              f'({len(sites)} projections, scale {self._mlx_lora_scale(config):g})')

    def _clear_mlx_adapter(self, model) -> None:
        """Put the resident MLX model back to the bare base.

        Exact unwrapping: every original module goes back where it was. Nothing is
        subtracted from any weight, so the base after a hundred voice switches is the
        same object it was after the load."""
        state = getattr(model, '_orpheus_mlx_lora', None)
        if state is None:
            return
        for _path, parent, attr, original in state.sites:
            setattr(parent, attr, original)
        model._orpheus_mlx_lora = None
        print(f'[ORPHEUS] MLX adapter cleared: {state.adapter_dir}')

    def _sync_mlx_adapter(self, model, adapter_dir: str) -> None:
        """Make the resident MLX model serve exactly `adapter_dir` (None = bare base).

        A no-op when the SAME adapter content is already applied — same dir AND same
        fingerprint. The fingerprint half is not paranoia: re-installing a retrained
        voice overwrites the same folder, and a dir-only comparison would keep the old
        training run applied for the life of the process (the identical trap
        _register_lora's fingerprint defends against on vLLM)."""
        if model is None:
            raise ValueError('Orpheus: no MLX model is loaded — cannot apply an adapter to it.')
        state = getattr(model, '_orpheus_mlx_lora', None)
        if not adapter_dir:
            self._clear_mlx_adapter(model)
            return
        fingerprint = self._adapter_fingerprint(adapter_dir)
        if state is not None and state.adapter_dir == adapter_dir and state.fingerprint == fingerprint:
            return
        if state is not None and state.adapter_dir == adapter_dir:
            print(f"[ORPHEUS] MLX adapter at {adapter_dir} changed on disk; re-applying "
                  'so the live model cannot keep serving the previous training run.')
        self._apply_mlx_adapter(model, adapter_dir, fingerprint)

    def _load_snac(self):
        """Load the SNAC audio decoder (not needed for MLX - it handles decoding internally)."""
        if self.backend == 'mlx':
            return None  # MLX handles SNAC internally

        if self.snac_model is not None:
            return self.snac_model

        try:
            from snac import SNAC
            print("Loading SNAC audio decoder...")

            # Determine device
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                self._device = 'cpu'

            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self._device)
            self.snac_model.eval()
            print(f"SNAC loaded on {self._device}")
            return self.snac_model
        except Exception as e:
            raise ValueError(f"Failed to load SNAC decoder: {e}")

    def _load_vllm_engine(self):
        """Load model using vLLM backend."""
        import os
        import random
        import gc

        is_windows = platform.system() == 'Windows'

        # Clear any leftover CUDA state from previous failed attempts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        # On Windows, use random port to avoid ZMQ port conflicts between workers
        if is_windows:
            # Use random port in high range to avoid conflicts
            random_port = random.randint(40000, 50000)
            os.environ['VLLM_RPC_PORT'] = str(random_port)

        from vllm import LLM
        from transformers import AutoTokenizer

        print(f"Loading Orpheus model with vLLM: {self.TRANSFORMERS_MODEL}")

        # Load tokenizer (needed for prompt formatting with special tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.TRANSFORMERS_MODEL)

        # On Windows, CUDA graph capture can fail. Check env var to override behavior.
        # Set ORPHEUS_DISABLE_EAGER=1 to try CUDA graphs (faster if it works)
        # Set ORPHEUS_FORCE_EAGER=1 to always use eager mode (slower but stable)
        force_eager = os.environ.get('ORPHEUS_FORCE_EAGER', '0') == '1'
        disable_eager = os.environ.get('ORPHEUS_DISABLE_EAGER', '0') == '1'

        if disable_eager:
            use_eager = False
            print("Orpheus: CUDA graphs ENABLED (ORPHEUS_DISABLE_EAGER=1)")
        elif force_eager or is_windows:
            use_eager = True
            print("Orpheus: Using eager mode (no CUDA graphs) for Windows compatibility")
        else:
            use_eager = False

        # Clean up CUDA state before vLLM initialization
        # This helps prevent CUDA graph capture failures caused by prior CUDA operations
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset CUDA context to clean state
            torch.cuda.reset_peak_memory_stats()
            print("Orpheus: CUDA state cleaned before vLLM init")

        # gpu_memory_utilization: fraction of TOTAL VRAM vLLM reserves for weights
        # + KV cache. Default 0.70 (not 0.85) because on a desktop the GPU is SHARED
        # with the Windows compositor / browser / Electron GPU process. At 0.85 vLLM
        # grabs ~20.4 GiB of 24, leaving too little for the desktop — and when VRAM
        # is oversubscribed the WDDM driver spills GPU memory into SYSTEM RAM, which
        # thrashes and maxes both (observed: hard OOM crash). 0.70 ≈ 16.8 GiB still
        # leaves ample KV cache for batched Orpheus (weights are only ~6.2 GiB).
        # Override with ORPHEUS_GPU_MEM_UTIL for headless / dedicated-GPU machines.
        gpu_mem_util = float(os.environ.get('ORPHEUS_GPU_MEM_UTIL', '0.70'))
        print(f"Orpheus: vLLM gpu_memory_utilization={gpu_mem_util}")
        # dtype: the fine-tune checkpoints are bfloat16; the old hardcoded
        # "float16" forced a lossy cast on load (vLLM warned "Casting
        # torch.bfloat16 to torch.float16" every run). A/B'd 2026-07-12: bf16 is
        # AUDIBLY clearer / less muffled (Owen: "significantly... a keeper")
        # even though the >8kHz RMS delta measured only 0.6dB — trust ears over
        # a single band metric. Same VRAM and speed; Ampere+ runs bf16 natively
        # (pre-Ampere would need the env override back to float16).
        # ORPHEUS_VLLM_DTYPE overrides ("bfloat16"/"float16"/"auto").
        vllm_dtype = os.environ.get('ORPHEUS_VLLM_DTYPE', 'bfloat16')
        print(f"Orpheus: vLLM dtype={vllm_dtype}")
        # Multi-LoRA is an ENGINE-CONSTRUCTION property — it cannot be turned on for a
        # later request — so it is set here, for this session, from the session's own
        # base key. A session with no base_dir (stock from the HF cache, or a merged
        # fine-tune) builds exactly the engine it always did.
        #
        # Keyed on base_dir, NOT adapter_dir: a stock-from-local-base session names the
        # base and no adapter, and it must still build LoRA-capable, because that is
        # what lets the resident streaming server add an adapter voice to a warm stock
        # engine (and vice-versa) without a 6 GB reload. base_dir is the engine's
        # identity in load_engine's cache key, so "built with LoRA" and "cache key says
        # base X" are the same fact — an adapter request can never reach a LoRA-less
        # engine.
        lora_kwargs = {}
        if self.base_dir:
            lora_kwargs = dict(
                enable_lora=True,
                max_lora_rank=self.LORA_MAX_RANK,
                max_loras=self.LORA_MAX_LORAS,
                max_cpu_loras=self.LORA_MAX_CPU_LORAS,
            )
            print(f"Orpheus: vLLM multi-LoRA enabled (max_lora_rank={self.LORA_MAX_RANK}, "
                  f"max_loras={self.LORA_MAX_LORAS}, max_cpu_loras={self.LORA_MAX_CPU_LORAS})")
        engine = LLM(
            model=self.TRANSFORMERS_MODEL,
            dtype=vllm_dtype,
            max_model_len=4096,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=use_eager,
            **lora_kwargs,
        )
        self._device = 'cuda'
        return engine

    @staticmethod
    def _adapter_fingerprint(adapter_dir: str):
        """A cheap CONTENT identity for an adapter folder: (st_mtime_ns, st_size) of
        adapter_model.safetensors.

        The path alone is not an identity. Re-installing a retrained voice writes the
        new weights to the SAME folder — that is what the installer does, by design —
        so a registry keyed on the path would keep serving the previous training run
        for the life of the engine, with nothing anywhere reporting a problem. Size
        and mtime together change on every real re-download or rsync, and reading them
        costs one stat, so registration keys on them instead.

        Deliberately NOT hashed: a 400 MB adapter would cost ~1s of I/O per voice
        load, and the failure this defends against is "the file was replaced", which
        stat sees.
        """
        st = os.stat(os.path.join(adapter_dir, 'adapter_model.safetensors'))
        return (st.st_mtime_ns, st.st_size)

    @staticmethod
    def _register_lora(voice: str, adapter_dir: str, fingerprint=None) -> int:
        """The int id `voice`'s adapter is served under, registering it if new.

        vLLM caches an adapter's weights under lora_int_id, so within one ENGINE the
        same voice must always present the same id (otherwise its weights are re-read
        from disk on every request) and two voices must NEVER share one (the second
        would be served the first's cached weights — a silent wrong-voice render).
        Ids start at 1 because 0 means "no adapter" to vLLM.

        Engine LIFETIME, not process lifetime: the cache the ids key into lives
        inside the LLM object, so _evict_global_cache drops this registry along with
        the engine and the next engine starts numbering from 1 against an empty
        cache. Within one engine the counter only ever moves FORWARD
        (orpheus_lora_next_id) — an id is never reused, even for a voice that has
        been re-pointed.

        RE-POINTING a voice mints a FRESH id rather than raising, and re-pointing is
        detected on CONTENT, not on the path: a different adapter_dir OR the same
        adapter_dir whose adapter_model.safetensors fingerprint changed (see
        _adapter_fingerprint). That second case is the common one — re-installing a
        retrained voice overwrites the same folder — and keying on the path alone
        meant the live engine kept serving the old training run until the process
        died. The old id may still have the old weights cached inside the engine, so
        reusing it would do exactly that; a new id is a cache miss by construction and
        loads the new adapter. (It also leaks the old id's slot in the host-side cache
        until the engine is torn down, which is what max_cpu_loras is sized for.)

        `fingerprint` is captured at REGISTRATION (register_adapter / engine
        construction) and passed in. It is None on the per-REQUEST path
        (_lora_request), which must not stat the adapter once per row of every batch —
        there a registered path simply keeps its id, and the registration that
        re-points it is what mints the new one.

        Process-global, not per-instance: the streaming worker switches voices
        in-process against ONE engine, and the ids must line up across those
        instances. The audiobook worker registers exactly one.
        """
        ids = loaded_tts.get('orpheus_lora_ids')
        paths = loaded_tts.get('orpheus_lora_paths')
        fingerprints = loaded_tts.get('orpheus_lora_fingerprints')
        if ids is None or paths is None or fingerprints is None:
            ids, paths, fingerprints = {}, {}, {}
            loaded_tts['orpheus_lora_ids'] = ids
            loaded_tts['orpheus_lora_paths'] = paths
            loaded_tts['orpheus_lora_fingerprints'] = fingerprints
        registered = paths.get(voice)
        registered_fp = fingerprints.get(voice)
        if voice in ids and registered == adapter_dir and (fingerprint is None
                                                           or registered_fp == fingerprint):
            return ids[voice]
        next_id = int(loaded_tts.get('orpheus_lora_next_id', 1))
        loaded_tts['orpheus_lora_next_id'] = next_id + 1
        if registered is not None:
            what = ('different weights at the same path' if registered == adapter_dir
                    else f'adapter {registered}')
            print(f"[ORPHEUS] Voice '{voice}' re-pointed ({what}) to {adapter_dir}; "
                  f'issuing a fresh lora id {next_id} (was {ids.get(voice)}) '
                  'so the live engine cannot serve the old cached weights.')
        ids[voice] = next_id
        paths[voice] = adapter_dir
        fingerprints[voice] = fingerprint
        return next_id

    @classmethod
    def register_adapter(cls, voice: str, adapter_dir: str, backend: str = None) -> int:
        """PUBLIC: make `voice` servable as a per-request LoRA on the live engine.

        The BookForge streaming server calls this when it loads an adapter voice onto
        an engine that is already up: a 'load' in adapter mode registers weights, it
        does not build anything, so switching between two adapters over the same base
        costs no vLLM reload and no CUDA-graph recapture. Returns the lora_int_id.

        Runs the SAME validation engine construction does (validate_adapter_dir), for
        the reason spelled out there: the 2nd..Nth adapter on a warm engine used to be
        registered unchecked, so a broken one surfaced as an opaque failure inside
        engine.generate() that sank the whole batch.

        Registration alone does NOT change which voice a bare generate() renders —
        that is `set_voice`. Both are needed for a default-voice switch; only this one
        is needed to make a voice available to a per-item mixed-voice batch.

        vLLM ONLY, and there is no MLX counterpart on purpose: an int id keying a
        weight cache inside the engine is meaningful only where several adapters can
        be resident at once. MLX attaches exactly one adapter to the model itself, so
        set_voice applies it directly (_sync_mlx_adapter) and there is nothing to
        register in advance.
        """
        if not voice:
            raise ValueError('Orpheus.register_adapter() requires a voice token')
        if not adapter_dir:
            raise ValueError(
                f"Orpheus.register_adapter({voice!r}) requires the voice's adapter dir"
            )
        cls.validate_adapter_dir(adapter_dir, backend)
        return cls._register_lora(voice, adapter_dir,
                                  cls._adapter_fingerprint(adapter_dir))

    def set_voice(self, voice: str, adapter_dir: str = None) -> None:
        """Re-point THIS resident engine's default voice, without reloading weights.

        The streaming server keeps one Orpheus instance alive across voice switches,
        so the instance's own idea of its voice has to move with it — and in adapter
        mode `self.adapter_dir` must move with `self.voice` in LOCKSTEP, because
        _lora_request() serves the instance's own voice straight from self.adapter_dir
        (skipping the registry). Setting `voice` alone would leave the previous
        voice's adapter attached to the new token: the right prompt prefix over the
        wrong LoRA, which renders as plausible audio in the wrong voice.

        What may NOT change here is the engine's LoRA capability or its weights: a
        merged voice IS its weights, and enable_lora is a construction-time property.
        Callers that need a different base (or to leave/enter merged mode) must tear
        the engine down and rebuild; this method refuses rather than pretending.

        Within ONE adapter-capable engine (built with a base_dir), adapter↔adapter AND
        adapter↔stock are both free: passing an adapter_dir attaches that adapter,
        passing none detaches it and serves an allowlisted stock token on the bare
        base. Only the streaming server and the audiobook worker ever reach this, and
        both validate the token before calling — a custom voice can never arrive here
        with adapter_dir=None (see _validate_adapter_mode's safety argument).

        The two backends attach an adapter differently and the difference is real, not
        cosmetic. On vLLM "attach" is a REGISTRATION: the adapter becomes servable per
        request and the previous one stays servable too. On MLX it is an
        APPLICATION: the wrappers on the resident model are swapped, so exactly one
        voice is servable at a time and the previous one stops being renderable the
        moment this returns. Callers that track which voices an engine can serve must
        therefore REPLACE their record on MLX rather than add to it — orpheus_stream's
        engine_voices does.
        """
        if not voice:
            raise ValueError('Orpheus.set_voice() requires a voice token')
        if self.custom_model_dir:
            raise ValueError(
                f'Orpheus is serving the merged model {self.custom_model_dir}, whose voice IS '
                f"its weights — it cannot be re-pointed at '{voice}' in place; reload the engine."
            )
        if adapter_dir and not self.adapter_capable:
            raise ValueError(
                f"Orpheus.set_voice({voice!r}): this engine cannot serve an adapter voice in "
                f'place (base_dir={self.base_dir!r}, backend={self.backend!r}); reload the '
                'engine against the shared base.'
            )
        if self.backend == 'mlx':
            # Apply (or detach) BEFORE moving self.voice/self.adapter_dir. _sync is
            # all-or-nothing, so a failure here leaves the model rendering the voice it
            # was rendering AND this instance still describing that same voice — the
            # two cannot disagree. Validation happens inside, before the model is
            # touched at all.
            self._sync_mlx_adapter(self.mlx_model, adapter_dir)
            self.adapter_dir = adapter_dir
            self.voice = voice
            return
        if adapter_dir:
            self.register_adapter(voice, adapter_dir, self.backend)
        # Lockstep: self.adapter_dir is what _lora_request serves the INSTANCE's own
        # voice from, so it has to move with self.voice in both directions. Leaving a
        # stale adapter attached while switching to a stock token is the right prompt
        # prefix over the wrong LoRA — plausible audio in the wrong voice.
        self.adapter_dir = adapter_dir
        self.voice = voice

    @property
    def adapter_capable(self) -> bool:
        """True when the LIVE engine can be re-pointed at an adapter voice WITHOUT
        being reloaded.

        Same condition both loaders build on: a base_dir, on a backend that can apply
        a LoRA. vLLM applies it per request (enable_lora, a construction-time
        property); MLX applies it to the resident model's projection modules
        (_apply_mlx_adapter). transformers has no PEFT wiring here and is excluded —
        load_engine refuses adapter mode there outright.

        Deliberately NOT `bool(self.adapter_dir)` — a stock-from-local-base engine has
        no adapter attached and is still fully capable, which is the entire point of
        the key collapse.

        NOT the same question as "can this engine serve a voice PER REQUEST", which is
        vLLM-only: see _lora_request and BookForge's canServeVoicePerRequest."""
        return bool(self.base_dir) and self.backend in ('vllm', 'mlx')

    def _lora_request(self, voice: str = None):
        """The LoRARequest that renders `voice` (default: this instance's voice), or
        None when this row must reach the BASE weights unmodified.

        None is returned for exactly two things, and nothing else:

          - a session with no adapter attached at all (merged weights, or stock from
            the HF cache) — the pre-adapter behaviour, and the ONLY thing the GUI /
            audiobook path ever hits, since neither produces orpheus_adapter_dir;
          - an allowlisted STOCK token on a LoRA-capable engine. Stock voices are a
            prompt prefix over the base checkpoint, so they are correctly rendered
            with no adapter even while other voices on the same engine have one.

        A voice that is neither raises. That is the guarantee the whole per-request
        voice design rests on: a CUSTOM voice with no registered adapter can never be
        quietly served by the base, which would sound finished and be the wrong
        narrator."""
        if voice is None:
            voice = self.voice
        if voice == self.voice:
            adapter_dir = self.adapter_dir
        else:
            adapter_dir = loaded_tts.get('orpheus_lora_paths', {}).get(voice)
        if not adapter_dir:
            if self.custom_model_dir:
                return None          # merged: the weights ARE the voice
            if voice == self.voice and not self.adapter_dir:
                return None          # this session simply has no adapter
            if voice in self.VALID_VOICES:
                return None          # allowlisted stock token over the base weights
            raise ValueError(f"Orpheus has no registered adapter for voice '{voice}'")
        # Imported only on the arm that needs it: a stock/merged session may be running
        # on MLX or transformers, where vllm is not installed at all.
        from vllm.lora.request import LoRARequest
        return LoRARequest(voice, self._register_lora(voice, adapter_dir), adapter_dir)

    @classmethod
    def register_voice_caps(cls, voice: str, caps: dict) -> dict:
        """Register `voice`'s per-voice generation tuning (see VOICE_CAP_SOURCES).

        This is the API the BookForge streaming worker calls on every voice load:
        it is RESIDENT and switches voices without respawning, so its caps cannot
        ride the spawn environment the way the audiobook worker's do. Registering
        them per voice — instead of rewriting os.environ, which is one global slot
        for the whole process — is also what makes a mixed-voice batch possible:
        every sampling value is resolved from the voice of the ITEM.

        Registration REPLACES whatever the voice had, so a cap the new payload
        omits reverts to env/class default rather than lingering from an earlier
        one. Values must be numeric; a key that is neither a cap nor one of the
        explicitly-ignored non-tuning keys (VOICE_CAP_IGNORED) raises — silently
        dropping it would mean a mis-tuned voice renders a whole book with no
        sign anything was wrong. Returns the normalised caps actually stored.
        """
        if not voice:
            raise ValueError('Orpheus.register_voice_caps() requires a voice token')
        stored = {}
        for key, value in (caps or {}).items():
            if key in cls.VOICE_CAP_IGNORED:
                continue
            if key not in cls.VOICE_CAP_SOURCES:
                raise ValueError(
                    f"Orpheus.register_voice_caps({voice!r}): unknown cap '{key}'. "
                    f'Known: {", ".join(sorted(cls.VOICE_CAP_SOURCES))} '
                    f'(ignored: {", ".join(cls.VOICE_CAP_IGNORED)}).'
                )
            if value is None:
                continue
            try:
                stored[key] = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Orpheus.register_voice_caps({voice!r}): cap '{key}' must be a "
                    f'number, got {value!r}'
                )
        cls._voice_caps[voice] = stored
        return stored

    def _voice_cap(self, key: str, voice: str = None) -> float:
        """The LIVE value of one tuning cap for `voice` (default: this instance's
        voice): registered per-voice cap -> ORPHEUS_* env var -> class default.
        See VOICE_CAP_SOURCES for the full rationale.

        The env var is re-read on every call by design — it is the audiobook
        worker's channel (one voice per process, exported at spawn) and reading it
        here rather than at import is what lets the same lookup serve both paths.
        """
        env_name, attr = self.VOICE_CAP_SOURCES[key]
        if voice is None:
            voice = self.voice
        registered = self._voice_caps.get(voice)
        if registered is not None and key in registered:
            return registered[key]
        raw = os.environ.get(env_name)
        if raw is not None:
            return float(raw)
        return float(getattr(self, attr))

    def _load_transformers_engine(self):
        """Load model using transformers backend."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Orpheus model with transformers: {self.TRANSFORMERS_MODEL}")

        # Determine device and dtype
        if torch.cuda.is_available():
            self._device = 'cuda'
            dtype = torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = 'mps'
            dtype = torch.float16
        else:
            self._device = 'cpu'
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.TRANSFORMERS_MODEL)

        # Load on CPU first (more reliable), then move to device
        print("Loading model weights on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            self.TRANSFORMERS_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

        if self._device != 'cpu':
            print(f"Moving model to {self._device}...")
            model = model.to(self._device)

        model.eval()
        print(f"Model ready on {self._device}")
        return model

    def load_engine(self) -> Any:
        try:
            msg = f"Loading Orpheus TTS with voice '{self.voice}'..."
            print(msg)
            self._cleanup_memory()

            # Check if already loaded — but ONLY reuse the process-global cache when
            # it holds the SAME model. The cache is keyed by engine name ('orpheus'),
            # so without a model-dir check a custom single-speaker finetune
            # (orpheus_model_dir) would be served from a cache populated by a DIFFERENT
            # model. The voice of a finetune is its WEIGHTS, not just the prompt token,
            # so reusing the wrong weights makes voice switching in the streaming worker
            # silently keep the first-loaded voice. Reload whenever the dir differs.
            #
            # ADAPTER MODE inverts that: the voice is NOT the weights, it is a
            # per-request LoRA over a shared base. So the cache is keyed on the pair
            # (merged dir, base dir), and two adapter voices sharing a base REUSE the
            # engine — never evict it, which is the whole point of the migration (no
            # vLLM reload, no CUDA-graph recapture to switch voice). Both keys are
            # None in stock mode and base_dir is None in merged mode, so a session
            # with no adapter keys compares exactly as it did before.
            engine_key = 'orpheus'
            engine = loaded_tts.get(engine_key, False)
            cached_dir = loaded_tts.get('orpheus_model_dir', None)
            cached_base = loaded_tts.get('orpheus_base_dir', None)
            if engine and cached_dir == self.custom_model_dir and cached_base == self.base_dir:
                self.backend = loaded_tts.get('orpheus_backend', 'transformers')
                self.snac_model = loaded_tts.get('orpheus_snac', None)
                self.tokenizer = loaded_tts.get('orpheus_tokenizer', None)
                self._device = loaded_tts.get('orpheus_device', 'cpu')
                self.mlx_model = loaded_tts.get('orpheus_mlx_model', None)
                print(f"Orpheus already loaded (backend: {self.backend}, model_dir: {cached_dir}, "
                      f"base_dir: {cached_base})")
                # A cached MLX model carries whatever adapter the PREVIOUS instance
                # applied to it — the weights are shared, so "same engine" does not
                # mean "same voice". Bring it to this session's adapter (or to the bare
                # base) before returning it, or a second instance over the same base
                # would render its own voice's prompt token through the first voice's
                # LoRA. No-op when they already agree.
                if self.backend == 'mlx' and (self.adapter_dir or
                                              getattr(self.mlx_model, '_orpheus_mlx_lora', None)):
                    self._sync_mlx_adapter(self.mlx_model, self.adapter_dir)
                return engine
            if engine:
                # A different model is cached (switching custom voices, custom<->stock,
                # or merged<->adapter). Evict it so its weights free before we load the new one.
                print(f"Orpheus model changed (cached={cached_dir!r}/{cached_base!r} -> "
                      f"want={self.custom_model_dir!r}/{self.base_dir!r}); reloading")
                self._evict_global_cache()

            # Detect and load appropriate backend
            self.backend = self._detect_backend()

            # Adapter mode needs a backend that can actually APPLY a LoRA. vLLM does it
            # per request; MLX does it by wrapping the resident model's projections
            # (_apply_mlx_adapter, stage B2). The transformers path has no PEFT wiring
            # at all, so it would serve the BARE BASE under the voice's name — a render
            # that sounds finished and is in the wrong voice.
            if self.adapter_dir and self.backend not in ('vllm', 'mlx'):
                raise ValueError(
                    f"Orpheus adapter mode is not supported on the '{self.backend}' backend: "
                    'there is no PEFT wiring there, so the adapter would be silently ignored '
                    'and the base rendered under this voice\'s name. Use vLLM (CUDA) or MLX '
                    '(Apple Silicon), or install this voice as a merged model.'
                )
            # Now that the backend is known, re-validate the adapter against ITS limits.
            # __init__ ran the universal checks before anything loaded; this is the pass
            # that knows whether the rank ceiling or the per-module-pattern rule applies,
            # and it still runs BEFORE a single byte of the base is read.
            if self.adapter_dir:
                self.validate_adapter_dir(self.adapter_dir, self.backend)

            if self.backend == 'mlx':
                engine = self._load_mlx_engine()
                self.mlx_model = engine
                # The base is loaded; the voice is the LoRA on top of it. Applied here
                # rather than lazily at first generate, so a broken adapter fails the
                # LOAD — the one moment a caller is prepared for a voice to be refused.
                if self.adapter_dir:
                    self._apply_mlx_adapter(engine, self.adapter_dir)
            elif self.backend == 'vllm':
                engine = self._load_vllm_engine()
                self._load_snac()
            else:
                engine = self._load_transformers_engine()
                self._load_snac()

            # Cache everything
            loaded_tts[engine_key] = engine
            loaded_tts['orpheus_backend'] = self.backend
            loaded_tts['orpheus_snac'] = self.snac_model
            loaded_tts['orpheus_tokenizer'] = self.tokenizer
            loaded_tts['orpheus_device'] = self._device
            loaded_tts['orpheus_mlx_model'] = self.mlx_model
            loaded_tts['orpheus_model_dir'] = self.custom_model_dir
            loaded_tts['orpheus_base_dir'] = self.base_dir

            # Register the session's own adapter against the FRESH engine, with its
            # content fingerprint. Doing it here rather than in __init__ matters: the
            # eviction above wipes the registry, so anything registered earlier would
            # be gone. Without this the first registration would happen lazily on the
            # first request, with no fingerprint — and a later re-install of the same
            # voice would then have nothing to compare against.
            #
            # vLLM only: the registry maps a voice to the int id vLLM caches its weights
            # under, and MLX has no such cache — its adapter is applied to the model and
            # its content identity is tracked by _MlxAdapterState.fingerprint instead.
            # A registry entry there would describe a request shape MLX never serves.
            if self.adapter_dir and self.backend == 'vllm':
                self._register_lora(self.voice, self.adapter_dir,
                                    self._adapter_fingerprint(self.adapter_dir))

            print('Orpheus TTS Loaded!')
            return engine

        except Exception as e:
            error = f'Orpheus.load_engine() error: {e}'
            import traceback
            traceback.print_exc()
            raise ValueError(error)

    def _max_chars_per_sec(self, voice: str = None) -> float:
        """The truncation guard's chars/sec threshold for `voice` (default: this
        instance's voice), resolved fresh on every call.

        Single source for both the env var name and its default (see
        DEFAULT_MAX_CHARS_PER_SEC). Callers keep the '<= 0 disables the guard'
        semantics themselves — this returns whatever was configured, including 0,
        rather than deciding for them.

        Was a @classmethod reading os.environ directly; it now goes through
        _voice_cap so a registered per-voice cap wins over the env (the resident
        streaming worker registers one per voice instead of rewriting the
        process-global env). The env branch is unchanged, so the audiobook
        worker — one voice per process, caps exported at spawn — resolves exactly
        the same number it always did.
        """
        return self._voice_cap('maxCharsPerSec', voice)

    def _mlx_token_budget(self, text: str) -> int:
        """Per-chunk MLX max_tokens CEILING sized from the text length, so a short
        repetition-primed chunk can't burn the flat MLX_MAX_TOKENS looping one
        sentence (the MLX runaway: minutes of the same line, because mlx-lm's
        ~20-token repetition penalty can't see a loop that is hundreds of audio
        tokens long).

        The divisor is the SLOWEST plausible reading rate, so the ceiling never
        clips HEALTHY audio (prose runs as slow as ~11 ch/s): the vLLM duration
        guard's per-voice rate (ORPHEUS_MAX_CHARS_PER_SEC — same source) when it is
        slower than the 12 ch/s slow-narrator floor, else the 12 floor. A smaller
        divisor = a MORE generous ceiling (safe); a larger one risks truncating
        slow narration, so we only ever go slower than 12, never faster. The 1.4
        factor is head-room over that worst case — only a runaway (many times the
        text's real token count) exceeds it.

        The tail allowance is added for the same reason _expected_audio_tokens
        exists (2026-08-28): chars/rate estimates SPEECH only, and the model's
        trained tail (~0.8-1.0 s) rides on every clip no matter how short the
        text. Without it this ceiling was catastrophically low for a heading:
        13 chars budgeted 128 tokens (1.52 s) against healthy renders of the same
        headings measured at 1.6-2.05 s, and 4 chars ("GOD.") budgeted 40 tokens
        against a measured 108 — a 37% ceiling, i.e. it would have truncated
        MOST of every one-word title the 2026-08-27 header-own-chunks change
        created. Adding a flat tail can only ever make
        the ceiling MORE generous, which this docstring already calls the safe
        direction; for prose it is noise (a 450-char chunk's budget stays far
        above MLX_MAX_TOKENS either way, so its effective cap does not move)."""
        env_rate = self._max_chars_per_sec()
        chars_per_sec_floor = min(env_rate, 12.0) if env_rate > 0 else 12.0
        speech = len(text) / chars_per_sec_floor * self.TOKENS_PER_AUDIO_SECOND
        return math.ceil((speech + self.SHORT_CHUNK_TAIL_TOKENS) * 1.4)

    def _generate_mlx(self, text: str, max_tokens: int = None, sentence_index: int = None) -> np.ndarray:
        """Generate audio using MLX backend.

        max_tokens defaults to MLX_MAX_TOKENS. The old bare default was a stale
        2048 literal, which silently clipped ~450-char packed chunks (need
        2500-3400 tokens) on every bare call — and a 2048-clipped chunk implies
        ~18.4 chars/sec, just UNDER the truncation guard's measured 19.0
        threshold, so the guard could not catch what this default caused.

        Returns an EMPTY array when the model produced no audio — never
        fabricated silence, which would sail past _save_audio's empty-rejection
        and ship a missing sentence as success.
        """
        if max_tokens is None:
            max_tokens = self.MLX_MAX_TOKENS
        # Per-chunk anti-runaway ceiling — only ever LOWERS the cap (min), never
        # raises it (see _mlx_token_budget). A short chunk that tries to loop is
        # cut at its text-proportional budget instead of the flat MLX_MAX_TOKENS.
        budget = self._mlx_token_budget(text)
        effective_max = min(max_tokens, budget)
        # Match vLLM/transformers sampling params - repetition_penalty prevents
        # repeated audio patterns that can sound like echo/reverb
        print(f"[ORPHEUS] Generating with voice='{self.voice}' for: {text[:50]}...")
        # mlx_audio's generate() splits its input on newlines and yields ONE
        # GenerationResult per segment — accumulate them ALL; keeping only the
        # last silently dropped every segment before the final one.
        segments = []
        for result in self.mlx_model.generate(
            text,
            voice=self.voice,
            temperature=self._voice_cap('temperature'),
            top_p=self._voice_cap('topP'),
            repetition_penalty=self._voice_cap('repPenalty'),
            max_tokens=effective_max
        ):
            audio_data = result.audio
            if audio_data is None:
                continue
            # MLX returns audio as numpy array or MLX array
            if hasattr(audio_data, 'tolist'):
                audio_data = np.array(audio_data, dtype=np.float32)
            if len(audio_data) > 0:
                segments.append(audio_data)

        if not segments:
            return np.zeros(0, dtype=np.float32)
        audio = segments[0] if len(segments) == 1 else np.concatenate(segments)

        # Visibility: log only when the budget was the binding cap AND generation
        # ran right up to it — an actual runaway-shaped truncation, not a chunk
        # that finished early. tokens-emitted is ESTIMATED from audio duration
        # (_mlx_est_tokens; mlx_audio's generate() doesn't surface a token count).
        if effective_max < max_tokens:
            est_tokens = self._mlx_est_tokens(audio)
            if est_tokens >= effective_max * 0.95:
                idx_str = '' if sentence_index is None else f'sentence {sentence_index}: '
                print(f"[ORPHEUS] {idx_str}MLX per-chunk token budget truncated generation "
                      f"(len={len(text)}, budget={effective_max}, ~{int(est_tokens)} tokens emitted)")

        return audio

    def _mlx_est_tokens(self, audio) -> float:
        """Audio tokens the model must have emitted to produce `audio`, ESTIMATED
        from its duration — mlx_audio's generate() surfaces no token count, so
        duration x TOKENS_PER_AUDIO_SECOND is the only signal available. One home
        for the arithmetic (_generate_mlx's budget log and _mlx_looks_capped)."""
        if audio is None or len(audio) == 0:
            return 0.0
        return len(audio) / self.SAMPLE_RATE * self.TOKENS_PER_AUDIO_SECOND

    def _mlx_looks_capped(self, clean: str, audio) -> bool:
        """True when a whole-text MLX render ran right up to its token cap, i.e. it
        was almost certainly CUT mid-sentence rather than finishing — the MLX
        stand-in for the vLLM ladder's `END_OF_AUDIO_TOKEN in tokens` check, which
        has no MLX equivalent. The cap is the one _generate_mlx would have applied:
        min(MLX_MAX_TOKENS, the per-chunk anti-runaway budget); 0.95 of it is
        'ran to the end' allowing for the estimate's slop.

        EMPTY audio is NOT capped (estimate 0): an empty render is a FAILED render
        (_guard_truncation's job), not a too-long one, and splitting it would only
        multiply the failure."""
        cap = min(self.MLX_MAX_TOKENS, self._mlx_token_budget(clean))
        return self._mlx_est_tokens(audio) >= cap * 0.95

    def _generate_mlx_safe(self, clean: str, depth: int = 0, force_split: bool = False) -> np.ndarray:
        """The safe general-purpose SOLO MLX render — MLX sibling of
        _generate_audio_vllm_safe. Renders `clean` WHOLE first; only if that render
        hit the token cap (_mlx_looks_capped — it would ship clipped) does it split
        at the nearest sentence/space boundary and render each half, concatenating
        the audio. Recurses up to a small depth so even an unusually dense chunk
        produces complete, un-clipped audio; a part that still can't finish is
        accepted as-is (same bottom rung as the vLLM ladder).

        It used to split EAGERLY — every text >= 80 chars was halved BEFORE any
        render — which was harmless for its original proven-cap callers but wrong
        for the streamed opener, which renders each play action's first sentence
        through here: every half was voiced as a standalone utterance, so an
        ordinary sentence came out with a mid-sentence pause and sentence-final
        prosody. Callers that have already proven the whole render fails ask for the
        old behaviour explicitly, with force_split.

        force_split=True skips the whole-chunk render and splits IMMEDIATELY, for
        callers whose whole render ALREADY failed — a proven token-cap hit, or
        _guard_truncation's clean early EOS. Re-rendering the whole text would very
        likely fail the same way, and (for a clean early EOS) the accept rung below
        would then return it unsplit: a re-roll, not a fix. Parts recurse WITHOUT
        force_split (the normal cap logic applies to them); text that can't be split
        (parts < 2) falls through to a normal whole render.
        """
        import numpy as np
        if force_split:
            parts = self._split_long_text(clean, max_length=max(60, len(clean) // 2))
            if len(parts) >= 2:
                return np.concatenate([self._generate_mlx_safe(p, depth + 1) for p in parts])
        audio = self._generate_mlx(clean, max_tokens=self.MLX_MAX_TOKENS)
        # Accept what we have once it fits, can't be split sensibly, or we've recursed enough.
        if not self._mlx_looks_capped(clean, audio) or depth >= 3 or len(clean) < 80:
            return audio
        parts = self._split_long_text(clean, max_length=max(60, len(clean) // 2))
        if len(parts) < 2:
            return audio
        return np.concatenate([self._generate_mlx_safe(p, depth + 1) for p in parts])

    @property
    def batch_pool_size(self) -> int:
        """How many sentences the worker should accumulate before calling
        convert_batch(). Plain BATCH_SIZE on every backend.

        (This used to return a 4x pool on MLX so length-bucketing had enough rows
        to rebuild full-width buckets. Both the bucketing and the pool that fed it
        are gone — mlx-lm 0.31.3 right-pads batch prefills, so mixed-length batches
        are safe; see the MLX memory-budget block at the top of the class. The
        property itself stays because the worker/manager plumbing is generic and
        other engines may want a pool.)"""
        return max(1, int(self.BATCH_SIZE or 1))

    def _mlx_width_for_depth(self, depth: int) -> int:
        """Widest batch that keeps peak MLX memory inside MLX_MEM_BUDGET_GB when
        every row may generate up to `depth` tokens.

            width x depth x MLX_KV_MB_PER_TOKEN_ROW  +  weights  +  buffer cache
                <=  MLX_MEM_BUDGET_GB

        The buffer cache is pinned at load by mx.set_cache_limit
        (ORPHEUS_MLX_CACHE_LIMIT_GB) so it is a known constant, not a variable.
        Never returns more than BATCH_SIZE, never less than 1 (a single row must
        always be attemptable — it is the same work the solo path would do).

        Worked example at the shipped defaults (budget 45, cache 8, cap 3700):
        headroom = 45 - 6.9 - 8 = 30.1 GB; per-row KV = 3700 x 0.1147 / 1024 =
        0.414 GB; width = 72. That is the measured-safe replacement for the old
        "0.153 GB per slot" arithmetic, which said 96 rows fit in 21.5 GB when a
        75-row batch really peaked at 36.8 GB."""
        width = max(1, int(self.BATCH_SIZE or 1))
        if depth <= 0:
            return width
        headroom = self._mlx_kv_headroom_gb()
        kv_gb_per_row = depth * self.MLX_KV_MB_PER_TOKEN_ROW / 1024.0
        return max(1, min(width, int(headroom / kv_gb_per_row)))

    def _mlx_kv_headroom_gb(self) -> float:
        """GB left for KV after the resident weights and the pinned buffer cache.

        Raises when the configured budget cannot even hold those two: there is no
        sane width to fall back to, and silently running at full width would defeat
        the whole point of the budget. Validated once at engine load
        (_load_mlx_engine) so a bad ORPHEUS_MLX_MEM_BUDGET_GB fails before any
        audio is rendered rather than mid-book."""
        cache_gb = float(os.environ.get('ORPHEUS_MLX_CACHE_LIMIT_GB', '8'))
        headroom = self.MLX_MEM_BUDGET_GB - self.MLX_WEIGHTS_GB - cache_gb
        if headroom <= 0:
            raise ValueError(
                f'ORPHEUS_MLX_MEM_BUDGET_GB={self.MLX_MEM_BUDGET_GB:g} leaves no room for the '
                f'KV cache: {self.MLX_WEIGHTS_GB:g} GB of weights + {cache_gb:g} GB of pinned '
                f'buffer cache (ORPHEUS_MLX_CACHE_LIMIT_GB) already meet or exceed it. '
                f'Raise the budget or lower the cache limit.'
            )
        return headroom

    def _mlx_batch_groups(self, entries: list) -> list:
        """entries: list of (key, prompt_token_list, payload, token_budget) in BOOK
        ORDER. Returns (batch_entries, depth) pairs to feed BatchGenerator.

        Batches are consecutive slices — no sorting, no bucketing. mlx-lm 0.31.3
        right-pads batch prefills, so a batch may mix a heading with packed prose
        without corrupting the short row (that hazard, and the bucketing that
        worked around it, are described in the class-level MLX memory block).
        Book order matters for two reasons: the depth bound below is computed from
        the rows actually in the batch, and sentences finish in roughly the order
        the listener/progress bar expects.

        `depth` is the batch's max_tokens: the largest per-row anti-runaway budget
        (_mlx_token_budget) in the slice, clamped to MLX_MAX_TOKENS. A batch of
        short rows therefore gets a shallow cap — which both tightens the runaway
        guard and, via _mlx_width_for_depth, buys back width.

        Width is capped so width x depth of KV stays inside MLX_MEM_BUDGET_GB. When
        a slice is too deep to run at full width it is split EVENLY rather than into
        [allowed, remainder]: MLX throughput is bought by width (12.4 sent/min at
        16 vs 27-28 at 96), so a 96-row slice capped at 79 runs 48+48 rather than
        79+17 — same batch COUNT, but no near-solo tail batch. Each sub-batch
        recomputes its own depth, which can only be <= the slice's, so every part
        stays inside the budget the width was derived from."""
        width = max(1, int(self.BATCH_SIZE or 1))

        def _depth(rows):
            return min(self.MLX_MAX_TOKENS, max(e[3] for e in rows))

        groups = []
        i, n = 0, len(entries)
        while i < n:
            take = min(width, n - i)
            window = entries[i:i + take]
            i += take
            depth = _depth(window)
            allowed = self._mlx_width_for_depth(depth)
            if allowed >= take:
                groups.append((window, depth))
                continue
            parts = -(-take // allowed)  # ceil: fewest equal parts that all fit
            base, extra = divmod(take, parts)
            print(f"[ORPHEUS] MLX batch narrowed {take} rows -> {parts} x ~{base} "
                  f"(depth {depth} tok, cap {allowed}, budget {self.MLX_MEM_BUDGET_GB:g} GB)",
                  flush=True)
            pos = 0
            for p in range(parts):
                size = base + (1 if p < extra else 0)
                sub = window[pos:pos + size]
                pos += size
                groups.append((sub, _depth(sub)))
        return groups

    def _resolve_mlx_row(self, i: int, ptoks: list, clean: str, tokens: list, depth: int):
        """Turn ONE retired BatchGenerator row's tokens into audio (or None).

        Factored out of _generate_mlx_batch_audio so a row can be resolved the
        moment it retires (per-row streaming emission) using the exact same ladder
        the end-of-batch loop used to run:
          - tokens >= depth  -> the row hit the cap and would ship clipped, so
            re-render it split (force_split: the cap hit is PROVEN, skip the whole
            re-render _generate_mlx_safe would otherwise try first);
          - otherwise decode prompt+generated through parse_output/SNAC;
          - no codes, or a decode error -> None (the caller's failure contract).
        """
        import numpy as np
        import mlx.core as mx
        from mlx_audio.tts.models.llama.llama import decode_audio_from_codes
        try:
            if len(tokens) >= depth:
                # Cap hit without finishing — re-render split so the played audio is
                # never clipped mid-sentence.
                print(f"Orpheus: stream sentence [{i}] hit the MLX audio-token cap; re-rendering split")
                return self._generate_mlx_safe(clean, force_split=True)
            ids = mx.array([ptoks + tokens])
            code_lists = self.mlx_model.parse_output(ids)
            if code_lists and len(code_lists[0]) > 0:
                return np.array(decode_audio_from_codes(code_lists[0])[0], dtype=np.float32)
            return None
        except Exception as decode_err:
            print(f"Orpheus _generate_mlx_batch_audio decode error [{i}]: {decode_err}")
            return None

    def _generate_mlx_batch_audio(self, texts: list, on_row=None, should_stop=None) -> list:
        """Batch-generate raw audio for many (already-cleaned) sentences in ONE
        MLX BatchGenerator pass — continuous batching, ~3.6x the per-sentence
        throughput. Returns float32 waveforms aligned to `texts` (None for an
        empty/failed item).

        In-memory sibling of _convert_mlx_batch's core (which writes WAVs): the
        streaming server uses this to generate a whole paragraph's sentences in
        parallel so MLX stays ahead of playback instead of dribbling one slow
        sentence at a time. MLX backend only.

        `on_row(i, audio)` — optional. Rows do NOT all finish together: mlx-lm
        retires each row as it hits its stop token, and the shortest row of a batch
        typically retires around 70% of the batch's depth. With a callback, each row
        is decoded (through the full cap/resplit ladder above) and handed over AT
        RETIREMENT instead of everything landing at the end, which is worth ~12s to
        the earliest sentences of a 30-43s batch. Called exactly ONCE per non-empty
        row, in RETIREMENT order (not `texts` order) — `i` is the index into `texts`,
        so an out-of-order-tolerant caller can reassemble. The full aligned list is
        still returned either way; callers that used on_row may ignore it.
        Default None -> byte-identical behaviour to before (audiobook, warmup).

        `should_stop()` — optional. Checked once per DECODE STEP (and once before
        each bucket), which is cheap next to a forward pass over the batch. When it
        goes true the generation is ABANDONED where it stands: the live
        BatchGenerator is closed, no further bucket is started, and every row that
        had not yet retired is left as None — deliberately NOT handed to on_row, so
        a caller streaming rows cannot mistake an abandoned row for a finished one.
        Rows that already retired through on_row stay delivered; they were real
        audio before the stop arrived.

        This exists because the streaming worker is SERIAL and a read-ahead batch is
        30-43s long: when the listener switches voice or presses play, everything in
        flight is stale the instant they act, and without a way out the new voice's
        load and the new block's batch queue behind ~40s of renders nobody will ever
        hear. Default None -> the audiobook path is untouched and uninterruptible,
        which is what it wants.

        The model itself is never touched: each bucket builds its own
        BatchGenerator, so an abandoned one leaves nothing behind for the next call.
        """
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        results = [None] * len(texts)
        gen = []  # (index, prompt_tokens, clean_text, token_budget) for non-empty sentences
        for i, t in enumerate(texts):
            clean = (t or '').strip()
            if clean:
                ptoks = self.mlx_model.prepare_input_ids(clean, self.voice)[0].tolist()
                gen.append((i, ptoks, clean, self._mlx_token_budget(clean)))

        if not gen:
            return results
        # Batches are consecutive BOOK-ORDER slices, width-capped against the MLX
        # memory budget (_mlx_batch_groups). No length bucketing: mlx-lm 0.31.3
        # right-pads batch prefills, so a heading may share a batch with prose.
        # Each group is its own BatchGenerator; the model stays loaded so extra
        # prefills are cheap. One bad group must not kill the rest, so each is
        # wrapped independently (matching the old whole-batch try/except
        # granularity).
        stopped = False
        for bucket, depth in self._mlx_batch_groups(gen):
            # A stop between buckets: start nothing new. The rows of every remaining
            # bucket stay None, which is the caller's "not rendered" contract.
            if should_stop is not None and should_stop():
                stopped = True
                break
            try:
                bg = BatchGenerator(
                    self.mlx_model,
                    max_tokens=depth,
                    # 0.31.3: stop SEQUENCES, not a set of ints. A set iterates
                    # without raising and silently fails to stop.
                    stop_tokens=[[self.END_OF_AUDIO_TOKEN]],
                    sampler=make_sampler(self._voice_cap('temperature'),
                                         top_p=self._voice_cap('topP'),
                                         min_p=self._voice_cap('minP')),
                    logits_processors=make_logits_processors(
                        None, self._voice_cap('repPenalty'), self.MLX_REP_WINDOW),
                    completion_batch_size=len(bucket),
                    prefill_batch_size=len(bucket),
                )
                boosts = [self._mlx_eos_boost_processor(len(c)) for _, _, c, _ in bucket]
                if any(boosts):
                    rep = make_logits_processors(
                        None, self._voice_cap('repPenalty'), self.MLX_REP_WINDOW)
                    uids = bg.insert(
                        [list(p) for _, p, _, _ in bucket],
                        logits_processors=[rep + [b] if b else list(rep) for b in boosts])
                else:
                    uids = bg.insert([list(p) for _, p, _, _ in bucket])
                out = {u: [] for u in uids}
                row_by_uid = dict(zip(uids, bucket))   # uid -> (i, ptoks, clean, budget)
                pending = set(uids)                    # rows not yet resolved
                # 0.31.3: next() returns (prompt_responses, generation_responses),
                # so `while responses := bg.next()` would NEVER terminate.
                # next_generated() yields just the generation list and returns
                # empty once every row has retired.
                while responses := bg.next_generated():
                    for r in responses:
                        if r.finish_reason != 'stop':  # stop token (128258) is dropped
                            out[r.uid].append(r.token)
                        # A row reports finish_reason ('stop' or the 'length' cap)
                        # exactly once, on its LAST response, before BatchGenerator
                        # drops it from the live set — so its token list is final
                        # here and it can be resolved (and handed to on_row) now
                        # rather than after the slowest row of the batch. `pending`
                        # keeps that a promise, not an assumption: exactly once.
                        if r.finish_reason is not None and r.uid in pending:
                            pending.discard(r.uid)
                            i, ptoks, clean, _budget = row_by_uid[r.uid]
                            results[i] = self._resolve_mlx_row(i, ptoks, clean, out[r.uid], depth)
                            if on_row is not None:
                                on_row(i, results[i])
                    # Once per decode step — a dict lookup against a forward pass over
                    # the whole batch. Rows that retired above are already delivered;
                    # the rest are abandoned unresolved.
                    if should_stop is not None and should_stop():
                        stopped = True
                        break
                bg.close()
                if stopped:
                    # stderr, not stdout: the streaming worker's stdout IS the
                    # JSON-lines protocol, and this line lands mid-batch.
                    print(f"[ORPHEUS] MLX batch abandoned on request "
                          f"({len(pending)} of {len(bucket)} rows unrendered)",
                          file=sys.stderr, flush=True)
                    break
                # Anything that never reported a finish_reason (shouldn't happen)
                # still gets resolved — and emitted — exactly once.
                for (i, ptoks, clean, _budget), uid in zip(bucket, uids):
                    if uid not in pending:
                        continue
                    pending.discard(uid)
                    results[i] = self._resolve_mlx_row(i, ptoks, clean, out[uid], depth)
                    if on_row is not None:
                        on_row(i, results[i])
            except Exception as e:
                print(f"Orpheus._generate_mlx_batch_audio() bucket error: {e}")
                import traceback
                traceback.print_exc()
        # No mx.clear_cache(): the cache is bounded at load (set_cache_limit),
        # and flushing per read-ahead batch just forces cold re-allocation.
        return results

    def _format_prompt_ids(self, text: str, voice: str = None) -> list:
        """Return the exact Orpheus input token IDs for `text` in `voice` (default:
        this instance's voice).

        Framing (MUST match training in orpheus_owen.py's build_dataset):
          [128259]                        START_OF_HUMAN
          + tokenizer("voice: text")      (leading BOS 128000 + text tokens)
          + [128009, 128260, 128261, 128257]   END_OF_TEXT, END_OF_HUMAN,
                                                START_OF_AI, START_OF_SPEECH

        These IDs are fed straight to vLLM via TokensPrompt. The OLD code decoded
        this sequence back to a STRING and let vLLM re-tokenize it, which prepended
        a STRAY second BOS (128000) — an out-of-distribution prompt that made the
        model vocalize the voice token at chunk starts (the "rohan"/"deathstalker"
        leak). Feeding IDs directly makes the runtime prompt byte-identical to the
        one the model was trained on, so the voice token is consumed silently.

        `voice` is per-item because a single batched generate() may eventually carry
        sentences in different voices (per-character casting), each with its own
        prompt token and its own LoRARequest.
        """
        if voice is None:
            voice = self.voice
        body = self.tokenizer(f"{voice}: {text}").input_ids   # includes leading BOS
        return [128259] + list(body) + [128009, 128260, 128261, 128257]

    def _expected_audio_tokens(self, n_chars: int) -> float:
        """How many audio tokens a HEALTHY render of `n_chars` should take — the
        one estimate every anti-runaway lever is scaled from (both EOS-boost
        processors). Speech at the file's 18.4 ch/s anchor, plus the model's
        flat tail (SHORT_CHUNK_TAIL_TOKENS), never above the old flat floor.

        THE SHORT-CHUNK HOLE THIS CLOSES (2026-08-28). The estimate used to be
        `max(300.0, speech)`. The 300-token floor binds for every chunk under
        ~66 chars, so EVERY short chunk was handed the SAME expectation — and at
        deathstalker's eosBoostStart 2.0 that put the boost's start at 600 tokens
        (7.14 s of audio) for a 4-char title just as much as for a 65-char one.
        A healthy 13-char heading renders in ~130 tokens, so the boost — the only
        pressure that ever ends a stalled generation — could not engage until the
        clip had run more than 4x its natural length. It never engaged at all on
        the take that started this: "Introduction." spoken twice, 358 tokens.
        Until the 2026-08-27 header-own-chunks change no chunk was ever that
        short (_apply_min_chars_floor merged them away), so the hole was never
        exposed; exempting headings from that floor exposed it.

        A flat floor was the wrong shape, not the wrong number: what a short
        chunk actually costs is its speech PLUS a tail that does not shrink with
        the text. So the floor becomes that sum, and the `min` keeps the change
        one-directional — this may only ever LOWER the expectation (engage the
        boost EARLIER), never raise it. Speech alone reaches 300 tokens at ~66
        chars and the tail-aware sum reaches it at ~45, so from there up the
        `min` returns the old value and NOTHING changes: ordinary prose, whose
        packing/gap/EOS behaviour is tuned, is untouched by construction. The
        whole change lives in chunks of 44 chars or fewer — tools/
        test_short_chunk_repeat.py asserts that boundary rather than trusting it.
        """
        speech = n_chars / 18.4 * self.TOKENS_PER_AUDIO_SECOND
        return min(max(300.0, speech), self.SHORT_CHUNK_TAIL_TOKENS + speech)

    def _eos_boost_processor(self, n_chars: int, voice: str = None):
        """Per-request vLLM logits processor implementing the EOS boost (see the
        EOS_BOOST comment at the top of the class). Returns None when disabled.
        Expected token count uses the file's own anchors: ~18.4 chars/sec of
        speech and TOKENS_PER_AUDIO_SECOND audio tokens/sec.

        `voice` is per-request because the boost is a property of the FINE-TUNE
        (only the bed-free voices carry the thin greedy EOS margin it corrects),
        and a batch may eventually mix voices."""
        base = self._voice_cap('eosBoost', voice)
        if base <= 0:
            return None
        expected = self._expected_audio_tokens(n_chars)
        start = self._voice_cap('eosBoostStart', voice) * expected
        eos = self.END_OF_AUDIO_TOKEN

        def _boost(token_ids, logits):
            n = len(token_ids)
            if n > start:
                # ramp with overrun, capped at 4x the base bias
                logits[eos] += base * min(4.0, 1.0 + (n - start) / expected)
            return logits
        return _boost

    def _mlx_eos_boost_processor(self, n_chars: int):
        """The EOS boost, ported to mlx-lm's logits-processor contract
        (`(tokens_context, logits(1, vocab)) -> logits`, applied PER ROW by
        BatchGenerator with that row's own token history — the context is seeded
        empty at insert, so `len(tokens)` counts generated tokens plus one, the
        same quantity vLLM's processor sees). Returns None when the voice does
        not carry a boost, so an unconfigured voice pays nothing.

        Same base/ramp/4x arithmetic as `_eos_boost_processor`, with ONE
        deliberate divergence: the start is clamped to fire no later than 90% of
        MLX_MAX_TOKENS. With eosBoostStart 2.0 a ~500-char chunk's start
        (~4,600 tokens) sits beyond the 3,700-token cap, so the vLLM-tuned start
        can NEVER fire for a full-size chunk here — and the rows this port
        exists for (measured 2026-08-21: 2 cap-hits in 623 chunks AFTER the
        rep-window fix, both ~500-char slow-delivery passages) are exactly those.
        The clamp only moves the start for chunks whose natural start would
        outrun the cap; short-chunk runaways keep the vLLM behaviour. The
        legitimate slow tail is safe margin: the catalog's measured p05 delivery
        (13.78 ch/s) puts a correct 500-char render at ~3,050 tokens, under the
        ~3,330 clamped start."""
        base = self._voice_cap('eosBoost')
        if base <= 0:
            return None
        expected = self._expected_audio_tokens(n_chars)
        start = min(self._voice_cap('eosBoostStart') * expected,
                    0.9 * self.MLX_MAX_TOKENS)
        eos = self.END_OF_AUDIO_TOKEN

        def _boost(tokens, logits):
            n = len(tokens)
            if n > start:
                bias = base * min(4.0, 1.0 + (n - start) / expected)
                return logits.at[:, eos].add(bias)
            return logits
        return _boost

    def _vllm_sampling_params(self, n_chars: int, max_tokens: int = None, voice: str = None):
        """The ONE place vLLM SamplingParams are built, so every path carries
        identical sampling config + the (optional) per-request EOS boost.

        Every value is resolved for `voice` (default: this instance's voice) —
        see VOICE_CAP_SOURCES. That is what lets a single batch carry per-item
        voices, and it is why the BookForge streaming server must build its batch
        params HERE instead of assembling its own SamplingParams (which silently
        dropped the EOS boost for every streamed voice)."""
        from vllm import SamplingParams
        proc = self._eos_boost_processor(n_chars, voice)
        return SamplingParams(
            temperature=self._voice_cap('temperature', voice),
            top_p=self._voice_cap('topP', voice),
            min_p=self._voice_cap('minP', voice),
            repetition_penalty=self._voice_cap('repPenalty', voice),
            max_tokens=max_tokens if max_tokens is not None else self.MAX_AUDIO_TOKENS,
            stop_token_ids=[self.END_OF_AUDIO_TOKEN],
            logits_processors=[proc] if proc else None,
        )

    def _generate_tokens_vllm(self, prompt: str, max_tokens: int = None) -> list:
        """Generate audio tokens using vLLM backend."""
        from vllm import SamplingParams, TokensPrompt
        if max_tokens is None:
            max_tokens = self.MAX_AUDIO_TOKENS

        # Feed token IDs directly (no decode->re-tokenize round-trip; see _format_prompt_ids)
        prompt_ids = self._format_prompt_ids(prompt)

        sampling_params = self._vllm_sampling_params(len(prompt), max_tokens)

        outputs = self.engine.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sampling_params,
                                       lora_request=self._lora_request())
        tokens = list(outputs[0].outputs[0].token_ids)

        # Truncate at end-of-audio token if present
        if self.END_OF_AUDIO_TOKEN in tokens:
            end_idx = tokens.index(self.END_OF_AUDIO_TOKEN)
            tokens = tokens[:end_idx]

        return tokens

    def _generate_audio_vllm_safe(self, clean: str, depth: int = 0, force_split: bool = False,
                                  voice: str = None):
        """Render audio for `clean`; if the model hits the token cap before emitting the
        end-of-audio token (the chunk is too long to finish), split it at the nearest
        sentence/space boundary and render each half, concatenating the audio. Recurses
        up to a small depth so even an unusually dense chunk produces complete,
        un-clipped audio. Returns a numpy waveform (same as _tokens_to_audio).

        force_split=True skips the whole-chunk render and splits IMMEDIATELY. The
        truncation _guard_truncation detects is a CLEAN early EOS — a whole-chunk
        re-render would very likely EOS cleanly (and truncated) again, and the
        `finished` accept below would return it without ever splitting: a re-roll,
        not a fix. Forcing the split renders half-length parts well inside the
        reliable zone. Parts recurse WITHOUT force_split (the normal cap logic
        applies to them); text that can't be split (parts < 2) falls through to a
        normal render.

        `voice` (default: this instance's voice) selects the prompt token, the
        sampling caps AND the adapter, so a mixed-voice batch's per-row retakes
        re-render in the row's OWN voice. Every recursion carries it: a retake that
        silently dropped back to the default voice would swap the narrator
        mid-sentence, on exactly the rows that already needed help.
        """
        from vllm import SamplingParams, TokensPrompt
        import numpy as np
        if voice is None:
            voice = self.voice
        if force_split:
            parts = self._split_long_text(clean, max_length=max(60, len(clean) // 2))
            if len(parts) >= 2:
                return np.concatenate(self._generate_parts_batched(parts, depth + 1, voice))
        sampling_params = self._vllm_sampling_params(len(clean), voice=voice)
        prompt = TokensPrompt(prompt_token_ids=self._format_prompt_ids(clean, voice))
        # lora_request goes on BOTH calls: the fallback exists only for a vLLM build
        # that doesn't take use_tqdm, and a retry without the adapter would render the
        # base voice while reporting success.
        lora_request = self._lora_request(voice)
        try:
            outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False,
                                           lora_request=lora_request)
        except TypeError:
            outputs = self.engine.generate([prompt], sampling_params, lora_request=lora_request)
        tokens = list(outputs[0].outputs[0].token_ids)
        finished = self.END_OF_AUDIO_TOKEN in tokens
        if finished:
            tokens = tokens[:tokens.index(self.END_OF_AUDIO_TOKEN)]
        # Accept what we have once it fits, can't be split sensibly, or we've recursed enough.
        if finished or depth >= 3 or len(clean) < 80:
            return self._tokens_to_audio(tokens)
        parts = self._split_long_text(clean, max_length=max(60, len(clean) // 2))
        if len(parts) < 2:
            return self._tokens_to_audio(tokens)
        return np.concatenate(self._generate_parts_batched(parts, depth + 1, voice))

    def _generate_parts_batched(self, parts: list, depth: int, voice: str = None) -> list:
        """Render many text parts in ONE vLLM generate() call. Returns waveforms
        aligned to `parts`.

        This used to be `[self._generate_audio_vllm_safe(p, depth+1) for p in parts]` —
        a Python loop of SINGLE-prompt generate() calls, each running at a concurrency
        of one while the 60-odd other chunks of the batch that triggered it sat waiting
        for their turn through the result loop. vLLM schedules a list of prompts
        natively, so every part now runs concurrently in one scheduling round.

        A part that STILL hits the token cap (or whose token stream misaligns) falls
        back to the recursive serial ladder at `depth`. That is rare, and preserving its
        exact recursion/accept behaviour matters more than batching it.
        """
        from vllm import TokensPrompt
        if voice is None:
            voice = self.voice
        sampling_params = [self._vllm_sampling_params(len(p), voice=voice) for p in parts]
        prompts = [TokensPrompt(prompt_token_ids=self._format_prompt_ids(p, voice)) for p in parts]
        # Every part is the same voice (they are pieces of ONE sentence), so one
        # LoRARequest covers the call — vLLM applies a scalar lora_request to all
        # prompts (_validate_and_add_requests).
        lora_request = self._lora_request(voice)
        try:
            outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False,
                                           lora_request=lora_request)
        except TypeError:
            outputs = self.engine.generate(prompts, sampling_params, lora_request=lora_request)
        waves = []
        for part, out in zip(parts, outputs):
            tokens = list(out.outputs[0].token_ids)
            if self.END_OF_AUDIO_TOKEN in tokens:
                tokens = tokens[:tokens.index(self.END_OF_AUDIO_TOKEN)]
                try:
                    waves.append(self._tokens_to_audio(tokens))
                    continue
                except TokenStreamMisaligned as align_err:
                    print(f"Orpheus: split part token stream misaligned ({align_err}); re-rendering once")
            waves.append(self._generate_audio_vllm_safe(part, depth, voice=voice))
        return waves

    def _generate_tokens_transformers(self, prompt: str, max_tokens: int = None) -> list:
        """Generate audio tokens using transformers backend."""
        if max_tokens is None:
            max_tokens = self.MAX_AUDIO_TOKENS
        # Apply the SAME framing as the vLLM path (_format_prompt_ids): SOH + BOS+text
        # + EOT,EOH,SOA,SOS. Without it the model gets an un-framed prompt and vocalizes
        # the voice token at chunk starts. `prompt` arrives pre-joined as "voice: text".
        body = self.tokenizer(prompt).input_ids
        framed = [128259] + list(body) + [128009, 128260, 128261, 128257]
        input_ids = torch.tensor([framed], dtype=torch.long).to(self.engine.device)

        # Generate with repetition penalty to avoid garbage loops
        with torch.no_grad():
            outputs = self.engine.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=self._voice_cap('temperature'),
                top_p=self._voice_cap('topP'),
                repetition_penalty=self._voice_cap('repPenalty'),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.END_OF_AUDIO_TOKEN
            )

        # Extract generated tokens (excluding prompt)
        generated = outputs[0][input_ids.shape[1]:].tolist()

        # Truncate at end-of-audio token if present
        if self.END_OF_AUDIO_TOKEN in generated:
            end_idx = generated.index(self.END_OF_AUDIO_TOKEN)
            generated = generated[:end_idx]

        return generated

    def _redistribute_codes(self, tokens: list) -> list:
        """Redistribute Orpheus tokens to SNAC's 3-layer format.

        Orpheus outputs 7 tokens per audio frame. SNAC expects these
        distributed across 3 layers:
        - Layer 1: 1 code per frame
        - Layer 2: 2 codes per frame
        - Layer 3: 4 codes per frame

        Raises TokenStreamMisaligned if any token lands outside its positional
        slot. Each of the 7 positions in a frame must fall in ITS OWN 4096-wide
        sub-range (position p in [p*4096, (p+1)*4096)); the global-range filter
        below silently drops any interleaved non-audio token, which SHIFTS every
        later token one position over and makes the offset subtraction produce
        negative/oversized codes. Those used to go straight into SNAC's codebook
        index_select on the GPU → `srcIndex < srcSelectDimSize` device-side
        assert → the CUDA context is poisoned and EVERY later sentence in the
        run fails instantly (2026-07-05: one bad sentence burned 1200). Failing
        Python-side keeps it a one-sentence problem the caller can re-render.
        """
        # Filter to valid audio tokens (128266 to 128266+4096*7)
        audio_tokens = [t for t in tokens if 128266 <= t < 128266 + 4096 * 7]

        if not audio_tokens:
            return None

        # Truncate to multiple of 7
        code_length = (len(audio_tokens) // 7) * 7
        audio_tokens = audio_tokens[:code_length]

        if code_length == 0:
            return None

        # Subtract base offset
        code_list = [t - 128266 for t in audio_tokens]

        # Every code must sit in its positional slot BEFORE the per-layer offset
        # subtraction — see the docstring for why an out-of-slot code is fatal.
        for i in range(len(code_list) // 7):
            for pos in range(7):
                code = code_list[7 * i + pos]
                if not (pos * 4096 <= code < (pos + 1) * 4096):
                    raise TokenStreamMisaligned(
                        f"frame {i} slot {pos}: code {code} outside [{pos * 4096}, {(pos + 1) * 4096})"
                    )

        # Distribute across 3 layers with offset subtraction
        layer_1, layer_2, layer_3 = [], [], []

        for i in range(len(code_list) // 7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i + 1] - 4096)
            layer_3.append(code_list[7*i + 2] - (2 * 4096))
            layer_3.append(code_list[7*i + 3] - (3 * 4096))
            layer_2.append(code_list[7*i + 4] - (4 * 4096))
            layer_3.append(code_list[7*i + 5] - (5 * 4096))
            layer_3.append(code_list[7*i + 6] - (6 * 4096))

        codes = [
            torch.tensor(layer_1).unsqueeze(0).to(self._device),
            torch.tensor(layer_2).unsqueeze(0).to(self._device),
            torch.tensor(layer_3).unsqueeze(0).to(self._device)
        ]
        return codes

    def _tokens_to_audio(self, tokens: list) -> np.ndarray:
        """Convert Orpheus tokens to audio using SNAC decoder.

        OOM-resilient: a CUDA OOM here used to fail the whole sentence, which the
        batch path then re-generated through vLLM — wasted GPU-minutes for an
        audio-decode hiccup. Seen in production as tiny (32–88 MiB) allocations
        failing with ~11 GiB reported free: allocator/VA fragmentation in the
        long-lived WSL worker, not real memory pressure. Recovery ladder:
          (1) torch.cuda.empty_cache() and retry on the GPU;
          (2) decode THIS sentence on CPU (SNAC is a small convnet — one CPU
              decode is cheap) and move the model back to the GPU after;
          (3) if it can't move back, leave SNAC on CPU permanently — decodes
              keep working (slightly slower) instead of failing the run.
        Every step logs loudly so a degraded run is visible, never silent.
        """
        if not tokens:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

        # Redistribute tokens to SNAC's 3-layer format
        codes = self._redistribute_codes(tokens)
        if codes is None:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

        # Decode on whatever device SNAC currently lives on (it may have been
        # stranded on the CPU by a previous OOM — see the ladder below).
        snac_device = next(self.snac_model.parameters()).device
        codes = [c.to(snac_device) for c in codes]

        try:
            with torch.no_grad():
                audio = self.snac_model.decode(codes)
        except torch.cuda.OutOfMemoryError:
            print("Orpheus SNAC decode hit CUDA OOM; freeing the allocator cache and retrying on GPU")
            torch.cuda.empty_cache()
            try:
                with torch.no_grad():
                    audio = self.snac_model.decode(codes)
            except torch.cuda.OutOfMemoryError:
                print("Orpheus SNAC decode OOM persists; decoding this sentence on the CPU")
                cpu_codes = [c.cpu() for c in codes]
                self.snac_model.to('cpu')
                with torch.no_grad():
                    audio = self.snac_model.decode(cpu_codes)
                try:
                    self.snac_model.to(self._device)
                except torch.cuda.OutOfMemoryError:
                    # Stay on CPU: future decodes follow snac_device above, so the
                    # run keeps producing audio instead of dying. Loud, not silent.
                    print("Orpheus SNAC could not return to the GPU; leaving the decoder on CPU for the rest of this run")

        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()
        return audio_np

    def _sentence_file(self, sentence_index: int) -> str:
        return os.path.join(self.session['sentences_dir'], f'{sentence_index}.{default_audio_proc_format}')

    def _clean_sentence_for_tts(self, sentence: str) -> str:
        """Strip whitespace + the e2a SML tags Orpheus doesn't understand
        (SML_UNSPOKEN_PATTERN: [break]/[pause]/[heading]/[music]/[sfx]/[silence]);
        Orpheus has its own emotion tags. 2026-08-27: the alternation used to be
        spelled out here, and identically in voxtral.py and f5.py — it moved to
        conf_models so the [heading] marker (and anything after it) cannot be
        taught to one engine and missed by the other two.

        Then apply the book-exact -> model transform (to_tts_form: scripture refs
        + bare-integer expansion, the exact transforms the fine-tunes trained
        with). Sentences arrive here BOOK-EXACT — that is what session['chapters']
        stores and what the m4b transcript is built from — so this boundary is
        the ONLY place the model text diverges from the display text. Everything
        downstream (prompt, token budgets, chars/sec + truncation guards, resplit
        ladders) measures the transformed text because they all consume this
        function's return. Idempotent: sentences from a pre-display-text session
        (already expanded) pass through unchanged."""
        sentence = (sentence or '').strip()
        sentence = SML_UNSPOKEN_PATTERN.sub('', sentence)
        return to_tts_form(sentence.strip()).strip()

    def _classify_gap(self, sentence: str):
        """Inter-clip silence for a chunk. Returns (lead_gap_sec, trail_gap_sec).

        2026-07-17 — AUTO PARAGRAPH/SECTION GAPS REMOVED FOR ORPHEUS (deliberately).
        Root cause of the long-standing "dialogue has huge pauses" complaint: e2a's
        SHARED text prep (lib/core.py ~2098) rewrites EVERY blank line to a valueless
        [pause] token, and prose puts EVERY dialogue turn in its own blank-line-
        separated paragraph. The old three-tier logic here then stamped a
        deterministic SECTION gap (1.0-1.6s lead) on top of each turn's sentence-gap
        tail. Measured with auto-editor on the mistborn 0.6s-cap model: dialogue-turn
        gaps were 2.0-2.5s (13 of them, one per paragraph break) vs the narrator's own
        ~0.6-1.0s; a STRAIGHT-NARRATION render from the same model measured median
        0.57s / max 1.17s with NO outliers — i.e. the Orpheus model, trained on clips
        that keep their natural inter-sentence pauses (only the clip EDGES trimmed),
        already reproduces the narrator's pausing itself. The e2a paragraph/section
        insertion was therefore PURELY additive dead air, and it was e2a — not the
        model — putting those pauses in.

        So for Orpheus we now keep ONLY:
          - the sentence-gap FLOOR on every chunk's tail (each chunk is a separate
            generation whose trailing silence is trimmed, so without a small floor the
            chunks butt together), and
          - an EXPLICIT [pause:X] (intentional, markup-specified beat) — still honored,
            because that's a deliberate pause, not the auto blank-line noise.
        The auto valueless [pause] (blank line) and [break]/[silence] (<p>/<br>) tiers
        are GONE. The tokens themselves still drive chunk boundaries in the packer
        (core.py) and are stripped by _clean_sentence_for_tts before TTS — only their
        deterministic GAP is removed here.

        If a real scene/section break ever under-pauses as a result, re-introduce a
        section tier HERE (or emit an explicit [pause:X] at that break) — do NOT
        restore the blanket blank-line gap that caused the dialogue problem. (The old
        ORPHEUS_PARAGRAPH_GAP / ORPHEUS_SECTION_GAP env knobs are retired with it.)

        Env override: ORPHEUS_SENTENCE_GAP (the floor; 0 disables).
        """
        raw = (sentence or '').strip()
        lowered = raw.lower()

        def _env(name):
            v = os.environ.get(name)
            return float(v) if v is not None else None

        # Sentence-gap floor: the minimum tail every chunk gets so a chunk-to-chunk
        # join is never bare. Override with ORPHEUS_SENTENCE_GAP.
        sentence_gap = _env('ORPHEUS_SENTENCE_GAP')
        if sentence_gap is None:
            sentence_gap = 0.6   # ear-approved on rohan (2026-07-12)

        # Honor ONLY an EXPLICIT [pause:X] — a deliberate, markup-specified beat.
        # (Auto valueless [pause] from a blank line is intentionally NOT matched: that
        # is the dialogue-pause noise removed 2026-07-17; see the docstring.)
        m = re.search(r'\[pause:([0-9.]+)\]', raw, flags=re.IGNORECASE)
        if m:
            token_gap = float(m.group(1))
            if lowered.startswith('[pause:'):
                return token_gap, sentence_gap               # explicit beat as lead + normal tail
            return 0.0, max(token_gap, sentence_gap)

        # Everything else — plain sentence end, auto [break]/[silence], auto valueless
        # [pause] — collapses to the sentence-gap floor. Orpheus supplies the real
        # inter-sentence pausing itself (learned from natural-pause training clips).
        return 0.0, sentence_gap

    def _write_silence(self, sentence_index: int) -> bool:
        """Write a tiny silent clip for an empty sentence."""
        silence = torch.zeros(1, int(self.params['samplerate'] * 0.1))
        torchaudio.save(self._sentence_file(sentence_index), silence,
                        self.params['samplerate'], format=default_audio_proc_format)
        return True

    def _speech_rate(self, clean: str, audio_np):
        """chars-per-second of the SML-stripped text over the audio's speech-only
        duration — trailing/leading silence trimmed EXACTLY as _save_audio does
        (trim_audio, silence_threshold=0.01, buffer_sec=0.20), and BEFORE any
        inter-clip gap pad is appended. Returns None when the metric can't apply
        (no audio). This is how a silent early-EOS truncation is caught: a
        fine-tuned voice trained on short clips tends to emit end-of-audio EARLY
        past ~300 chars, dropping trailing text. The clip then FINISHES cleanly
        (no token-cap hit — the existing cap re-split can't see it) but is far too
        short for the words, i.e. an impossibly high chars/sec (healthy prose is
        ~15 ch/s, p90 ~17; a truncated 400+ char chunk measured ~21)."""
        if audio_np is None or len(audio_np) == 0:
            return None
        audio_tensor = torch.from_numpy(np.asarray(audio_np)).float()
        if audio_tensor.dim() == 1:
            audio_tensor = trim_audio(audio_tensor, self.SAMPLE_RATE, silence_threshold=0.01, buffer_sec=0.20)
        seconds = len(audio_tensor) / self.SAMPLE_RATE
        if seconds <= 0:
            return None
        return len(clean) / seconds

    def _reject_dir(self):
        """Where a rejected render is kept for post-mortem.

        OUTSIDE the session dir on purpose: BookForge deletes the whole scratch
        session when a job finishes, which would take the evidence with it. Default
        is a sibling of the session: <tmp>/tts_rejects/<ebook-uuid>/. Override with
        ORPHEUS_REJECT_DIR to collect them somewhere permanent.
        """
        override = os.environ.get('ORPHEUS_REJECT_DIR', '').strip()
        if override:
            return override
        # A guard can fire from a context that never adopted a session (a probe,
        # a harness exercising the guard alone). No session means nowhere to file
        # a reject — the same answer as a session with no process_dir, not an
        # error worth breaking the guard's one-line report over.
        session = getattr(self, 'session', None)
        if not session:
            return None
        proc = session.get('process_dir')
        if not proc:
            return None
        ebook_dir = os.path.dirname(os.path.normpath(proc))   # <tmp>/ebook-<uuid>
        tmp_root = os.path.dirname(ebook_dir)                 # <tmp>
        if not tmp_root or not os.path.basename(ebook_dir):
            return None
        return os.path.join(tmp_root, 'tts_rejects', os.path.basename(ebook_dir))

    def _keep_reject(self, sentence_index: int, clean: str, audio_np, reason: str, detail: dict = None):
        """Preserve a render the guards threw away, plus WHY.

        Without this a truncation is only ever a log line: the bad audio is
        overwritten by the re-render, so there is no way to see WHERE it cut — mid
        word, at a clause boundary, right after a number. Keeping it turns every
        guard fire into a data point (Owen 2026-07-28, after chunk 1005 of Killing
        America truncated at 55% of its correct length and the evidence was gone).

        BEST-EFFORT BY DESIGN: this is diagnostics, not product. Any failure here is
        reported and swallowed — losing a post-mortem must never take down a book.
        """
        try:
            _rec_session = getattr(self, 'session', None) or {}
            directory = self._reject_dir()
            if not directory:
                return
            os.makedirs(directory, exist_ok=True)
            stem = os.path.join(directory, f'{sentence_index:06d}_{reason}')
            seconds = None
            if audio_np is not None and len(audio_np) > 0:
                wave = torch.from_numpy(np.asarray(audio_np)).float()
                if wave.dim() == 1:
                    wave = wave.unsqueeze(0)
                torchaudio.save(stem + '.wav', wave, self.SAMPLE_RATE)
                seconds = round(float(wave.shape[-1]) / self.SAMPLE_RATE, 3)
            record = {
                'sentence_index': sentence_index,
                # 'short' = truncated, re-rendered split | 'empty' = no audio |
                # 'cap' = never emitted EOS, hit the token ceiling |
                # 'overrun' = short chunk spoke too long; the take SHIPPED anyway
                'reason': reason,
                'chars': len(clean),
                'audio_seconds': seconds,
                'chars_per_second': round(len(clean) / seconds, 2) if seconds else None,
                # Null when the engine has no session (see _reject_dir): an
                # honest hole in the diagnostics beats an exception that eats
                # the whole record. Reachable via ORPHEUS_REJECT_DIR, which
                # skips _reject_dir's own no-session return.
                'voice': _rec_session.get('fine_tuned'),
                'model_dir': _rec_session.get('orpheus_model_dir'),
                # Adapter mode: model_dir is None and the voice lives in the adapter,
                # so a post-mortem needs both refs to know what actually rendered this.
                'adapter_dir': _rec_session.get('orpheus_adapter_dir'),
                'base_dir': _rec_session.get('orpheus_base_dir'),
                'max_audio_tokens': self.MAX_AUDIO_TOKENS,
                'text': clean,
            }
            if detail:
                record.update(detail)
            import json as _json
            with open(stem + '.json', 'w', encoding='utf-8') as handle:
                _json.dump(record, handle, indent=1, ensure_ascii=False)
            # One append-only file per run, so a post-mortem is a single read
            # rather than a directory walk, and so the record survives even if
            # the per-event wav could not be written.
            with open(os.path.join(directory, 'events.jsonl'), 'a', encoding='utf-8') as handle:
                handle.write(_json.dumps(record, ensure_ascii=False) + '\n')
            self._emit_guard_event(record)
        except Exception as err:
            print(f'Orpheus: could not keep the rejected render for sentence {sentence_index} ({err})')

    def _emit_guard_event(self, record: dict):
        """Print one parseable line for a guard fire. Best-effort like its caller.

        Kept separate from _keep_reject so a chunk whose audio is NOT thrown away
        (the short-chunk overrun — the take ships and is only counted) can report
        through the same channel without pretending to be a rejection.
        """
        try:
            import json as _json
            compact = dict(record)
            # The text can be 500 characters of book; the console line stays
            # readable and the full text is already in the per-event JSON.
            text = compact.get('text') or ''
            if len(text) > 120:
                compact['text'] = text[:120]
                compact['text_truncated'] = True
            print(f'[ORPHEUS][{self.GUARD_EVENT_TAG}] '
                  + _json.dumps(compact, ensure_ascii=False, separators=(',', ':')))
        except Exception:
            pass

    def _guard_truncation(self, sentence_index: int, clean: str, audio_np, resplit,
                          voice: str = None):
        """Backstop for silent early-EOS truncation (see _speech_rate). If the
        rendered clip is too short for the text, re-render it split at sentence
        boundaries via `resplit` (the backend's _generate_*_safe ladder). `resplit`
        takes the clean text and returns a numpy waveform — and it must ACTUALLY
        split, not merely re-render: the failure detected here is a CLEAN early
        EOS, so a whole-chunk re-render would most likely clean-EOS (truncated)
        again. Both backends' callers therefore pass their ladder with
        force_split=True — the plain form renders the whole text first and accepts
        it when it finishes cleanly, which is correct for its token-cap callers but
        useless here (it would just re-roll the same clean early EOS).

        Guard applies only when clean length > 150 chars (short chunks' rates are
        noisy) and ORPHEUS_MAX_CHARS_PER_SEC > 0 (DEFAULT_MAX_CHARS_PER_SEC when
        unset; set 0 to disable). Read via _max_chars_per_sec().

        SELF-CALIBRATING RATCHET (self._rate_ceiling(voice)): that default was
        calibrated on ~15-17 ch/s voices; a faster fine-tune (e.g. one that
        naturally reads ~20 ch/s) trips the guard on every long healthy chunk,
        each false positive costing a full wasted serialized re-render. But the
        force-split re-render hands us GROUND TRUTH for the voice's natural rate:
        the pieces are short, and early-EOS truncation only strikes past ~300
        chars, so a split render CANNOT itself be truncated. Therefore when the
        re-render's rate2 STILL exceeds the threshold, that is not "still short"
        — it PROVES the voice's true speaking rate is above the configured
        threshold, and staying at 19.0 would keep re-splitting every fast-but-
        healthy chunk for the rest of the session. So we ratchet the session
        ceiling up to rate2 + 0.5 and log it LOUDLY. The effective threshold used
        for every check is max(env-configured, ratchet); the ratchet only ever
        moves UP, only from split-render rates (never a first-pass render), and
        NEVER resurrects a disabled guard (env <= 0 short-circuits before the
        ratchet is ever consulted). This is not an infinite loop — a first-pass
        chunk measured above the raised ceiling still trips exactly once and
        re-splits; the _generate_*_safe ladders carry their own depth guard.

        EMPTY audio for non-empty text (immediate early-EOS / unparseable stream)
        is a definite failure, not a noisy metric, so it gets its one re-render
        regardless of the 150-char floor. If the retake is empty too, the empty
        result flows back to the caller — _save_audio rejects it (row fails
        loudly) and the streaming worker emits 'No audio generated'.

        `voice` (default: this instance's voice) is the row's own voice, so a
        mixed-voice batch judges each row against ITS fine-tune's rate threshold."""
        reason = self._needs_resplit(sentence_index, clean, audio_np, voice)
        if reason is None:
            return audio_np
        audio_np = resplit(clean)
        if reason == 'short':
            self._ratchet_after_resplit(clean, audio_np, voice)
        return audio_np

    def _rate_ceiling(self, voice: str = None) -> float:
        """This session's ratcheted chars/sec ceiling for `voice` (0.0 = never
        ratcheted). See the _rate_ceilings note in __init__ for why it is per-voice."""
        if voice is None:
            voice = self.voice
        return self._rate_ceilings.get(voice, 0.0)

    def _needs_resplit(self, sentence_index: int, clean: str, audio_np, voice: str = None):
        """DECISION half of _guard_truncation. Returns 'empty', 'short', or None.

        Split out from the re-render so the BATCH path can defer the expensive half:
        the verdict depends only on text and already-rendered audio, so every chunk in
        a batch can be judged immediately and their re-renders pooled into one call
        (see _render_deferred_resplits). The log lines still fire here, at detection
        time, so the reason a chunk gets re-rendered is visible where it happens.

        `voice` (default: this instance's voice) picks BOTH the configured threshold
        and the ratchet, because how many chars a second is "too fast to be real" is a
        property of the fine-tune, not of the process.
        """
        if (audio_np is None or len(audio_np) == 0) and clean and clean.strip():
            print(f"Orpheus: sentence {sentence_index} produced no audio — re-rendering split at sentence boundaries")
            self._keep_reject(sentence_index, clean, audio_np, 'empty')
            return 'empty'
        env_rate = self._max_chars_per_sec(voice)
        # A disabled guard (env <= 0) stays disabled — the ratchet must NEVER
        # resurrect it. Only past this gate does the session ceiling apply.
        if env_rate <= 0 or len(clean) <= 150:
            return None
        max_rate = max(env_rate, self._rate_ceiling(voice))
        rate = self._speech_rate(clean, audio_np)
        if rate is None or rate <= max_rate:
            return None
        print(f"Orpheus: sentence {sentence_index} audio too short for text "
              f"({rate:.1f} ch/s > {max_rate:.1f}) — re-rendering split at sentence boundaries")
        self._keep_reject(sentence_index, clean, audio_np, 'short',
                          {'measured_chars_per_second': round(rate, 2),
                           'threshold_chars_per_second': round(max_rate, 2)})
        return 'short'

    def _report_short_chunk_overrun(self, sentence_index: int, clean: str, audio_np) -> bool:
        """PRINT — and only print — when a sub-floor chunk's audio runs far longer
        than its text can justify. Returns whether it reported, for the tests; the
        audio is never touched and every caller ignores the return.

        THIS MEASURES A MODEL DEFECT, IT DOES NOT COMPENSATE FOR ONE. The
        duplication is in the generated audio, not the text, so the cure is a
        fine-tune that commits to end-of-audio after a very short utterance — see
        the SHORT_CHUNK_* constants for why the re-render this replaced was
        removed, and for what the count does and does not mean.

        Deliberately NOT folded into _needs_resplit. That guard answers the
        opposite question (audio too SHORT for the text, i.e. a truncation), it
        explicitly refuses chunks of 150 chars or fewer so it has never had an
        opinion about a heading, and it ACTS where this only observes. Same shape,
        different question — keeping them apart is what stops either from being
        read as the other's opposite.

        Measured on the GENERATED waveform: every caller reports BEFORE _save_audio
        appends the inter-clip gaps, so the number is what the model emitted.
        """
        if audio_np is None or len(audio_np) == 0:
            return False
        n_chars = len(clean)
        if self.SHORT_CHUNK_MAX_CHARS <= 0 or n_chars <= 0 or n_chars >= self.SHORT_CHUNK_MAX_CHARS:
            return False
        seconds = len(audio_np) / self.SAMPLE_RATE
        allowed = self.SHORT_CHUNK_SECONDS_BASE + self.SHORT_CHUNK_SECONDS_PER_CHAR * n_chars
        if seconds <= allowed:
            return False
        # ONE line, one tag, everything needed to count and to find the clip.
        # The text is last because it is the only unbounded field.
        print(f"[ORPHEUS][{self.SHORT_CHUNK_OVERRUN_TAG}] sentence={sentence_index} "
              f"chars={n_chars} seconds={seconds:.3f} allowed={allowed:.3f} "
              f"ratio={seconds / allowed:.2f} text={clean!r}")
        # Also report through the structured channel, so one collector sees every
        # guard fire. The audio is KEPT alongside it: this take ships, but it is
        # the only recording of the doubling, and a count nobody can listen to
        # cannot settle where "slow" ends and "doubled" begins (see the
        # SHORT_CHUNK_* note). Saved under its own reason so it is never mistaken
        # for a chunk that was thrown away and re-rendered.
        self._keep_reject(sentence_index, clean, audio_np, 'overrun',
                          {'allowed_seconds': round(allowed, 3),
                           'overrun_ratio': round(seconds / allowed, 3),
                           'audio_kept': True})
        return True

    def _ratchet_after_resplit(self, clean: str, audio_np, voice: str = None) -> None:
        """RATCHET half of _guard_truncation — see its docstring for the full rationale.

        The split render is un-truncatable, so if its rate STILL exceeds the threshold
        that is the voice's real speaking rate, not a truncation: raise the session
        ceiling (never lower it) so subsequent healthy fast chunks stop tripping.
        Only ever called for a 'short' verdict — an 'empty' one carries no rate signal.
        """
        if voice is None:
            voice = self.voice
        env_rate = self._max_chars_per_sec(voice)
        if env_rate <= 0:
            return
        max_rate = max(env_rate, self._rate_ceiling(voice))
        rate2 = self._speech_rate(clean, audio_np)
        if rate2 is not None and rate2 > max_rate:
            new_ceiling = rate2 + 0.5
            self._rate_ceilings[voice] = new_ceiling
            print(f"Orpheus: voice '{voice}' measured natural rate {rate2:.1f} ch/s exceeds guard "
                  f"threshold {max_rate:.1f} — recalibrating threshold to {new_ceiling:.1f} for this session")

    def _save_audio(self, sentence_index: int, audio_np, lead_gap: float = 0.0, trail_gap: float = 0.0) -> bool:
        """Trim trailing silence, normalize, add the inter-clip pauses, and write a
        decoded waveform to the sentence file. Shared by convert(), convert_batch()
        and _convert_mlx_batch(). (lead_gap, trail_gap) come from _classify_gap():
        the trailing gap is appended after the speech, the leading gap prepended
        before it. Both can be non-zero — a chunk opened by a boundary token gets a
        long lead AND still gets its sentence-gap tail, so a chunk's tail is never
        bare (the invariant _classify_gap now guarantees)."""
        if audio_np is None or len(audio_np) == 0:
            print(f"Orpheus returned no audio data for sentence {sentence_index}")
            return False
        final_sentence_file = self._sentence_file(sentence_index)
        audio_tensor = torch.from_numpy(audio_np).float()
        # NO-FALLBACK (2026-07-11): the trailing-silence trim was REMOVED here. It
        # silently erased the runaway silence a mis-behaving model emits (a stop-token
        # failure), masking the bug and destroying data — a forbidden fallback. The
        # end-of-sentence pause is added DETERMINISTICALLY below (trail_gap); the model
        # must be trained to stop cleanly (trimmed-tail clips), not have its dead air
        # papered over at save time. If a clip has abnormal trailing silence, that must
        # be VISIBLE, not hidden.
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        # Normalize to prevent clipping
        max_val = audio_tensor.abs().max()
        if max_val > 0:
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val * 0.95
        else:
            audio_tensor = torch.zeros(1, int(self.params['samplerate'] * 0.1))
        # Inter-clip silence (tiers + durations decided by _classify_gap, matching
        # e2a's standard SML semantics). The leading pad is prepended (a boundary
        # opening a new paragraph isn't placed one sentence too late) and the
        # trailing pad appended — BOTH can apply so the tail is never bare.
        if (lead_gap and lead_gap > 0) or (trail_gap and trail_gap > 0):
            audio_tensor = audio_tensor.cpu()
            if lead_gap and lead_gap > 0:
                lead_pad = torch.zeros(1, int(self.params['samplerate'] * lead_gap))
                audio_tensor = torch.cat([lead_pad, audio_tensor], dim=1)
            if trail_gap and trail_gap > 0:
                trail_pad = torch.zeros(1, int(self.params['samplerate'] * trail_gap))
                audio_tensor = torch.cat([audio_tensor, trail_pad], dim=1)
        torchaudio.save(final_sentence_file, audio_tensor.cpu(),
                        self.params['samplerate'], format=default_audio_proc_format)
        del audio_tensor
        if os.path.exists(final_sentence_file):
            return True
        print(f"Failed to create {final_sentence_file}")
        return False

    def convert(self, sentence_index: int, sentence: str) -> bool:
        try:
            if not self.engine:
                print("Orpheus TTS engine not loaded!")
                return False

            lead_gap, trail_gap = self._classify_gap(sentence)
            clean = self._clean_sentence_for_tts(sentence)
            if not clean:
                return self._write_silence(sentence_index)

            try:
                if self.backend == 'mlx':
                    audio_np = self._generate_mlx(clean, sentence_index=sentence_index)
                    # Backstop a silent early-EOS truncation (audio too short for text).
                    # force_split: a whole-chunk re-render would just clean-EOS
                    # (truncated) again — the resplit must actually split.
                    audio_np = self._guard_truncation(
                        sentence_index, clean, audio_np,
                        lambda c: self._generate_mlx_safe(c, force_split=True)
                    )
                    # …and REPORT the opposite failure on a sub-floor chunk: audio
                    # far LONGER than the text can justify (a heading spoken twice).
                    # Reported only — the take stands; see the SHORT_CHUNK_* block.
                    self._report_short_chunk_overrun(sentence_index, clean, audio_np)
                elif self.backend == 'vllm':
                    try:
                        audio_np = self._tokens_to_audio(self._generate_tokens_vllm(clean))
                    except TokenStreamMisaligned as align_err:
                        # Stochastic sampling glitch — one re-render (fresh tokens)
                        # almost always fixes it. If it misaligns again, let it fail.
                        print(f"Orpheus: sentence {sentence_index} token stream misaligned ({align_err}); re-rendering once")
                        audio_np = self._generate_audio_vllm_safe(clean)
                    # Backstop a silent early-EOS truncation (audio too short for text).
                    # force_split: a whole-chunk re-render would just clean-EOS
                    # (truncated) again — the resplit must actually split.
                    audio_np = self._guard_truncation(
                        sentence_index, clean, audio_np,
                        lambda c: self._generate_audio_vllm_safe(c, force_split=True)
                    )
                    # …and REPORT the opposite failure on a sub-floor chunk: audio
                    # far LONGER than the text can justify (a heading spoken twice).
                    # Reported only — the take stands; see the SHORT_CHUNK_* block.
                    self._report_short_chunk_overrun(sentence_index, clean, audio_np)
                else:
                    audio_np = self._tokens_to_audio(
                        self._generate_tokens_transformers(f"{self.voice}: {clean}")
                    )
                ok = self._save_audio(sentence_index, audio_np, lead_gap, trail_gap)
                self._cleanup_memory()
                return ok
            except Exception as gen_error:
                print(f"Orpheus generation error for sentence {sentence_index}: {gen_error}")
                import traceback
                traceback.print_exc()
                if is_fatal_cuda_error(gen_error):
                    # Poisoned CUDA context: nothing else in this process can
                    # succeed. Die loudly so the worker respawns fresh.
                    raise
                return False

        except Exception as e:
            if is_fatal_cuda_error(e):
                raise
            print(f'Orpheus.convert() error: {e}')
            import traceback
            traceback.print_exc()
            return False

    def _render_deferred_resplits(self, deferred: list, results: dict) -> None:
        """Re-render every truncated chunk of a batch in ONE pooled generate() call.

        deferred: list of (sentence_index, clean, gap, reason). Fills `results` in place.

        Each chunk is split exactly as the inline guard split it
        (_generate_audio_vllm_safe with force_split — half-length parts, well inside the
        reliable zone). The ONLY change is scheduling: the parts of every deferred chunk
        in the batch are submitted together, so a batch carrying 3 truncations costs one
        extra scheduling round rather than 3+ serial single-prompt rounds, each of which
        used to hold up the whole batch's remaining results.

        Audio is byte-for-byte the same decision path — same text, same split, same
        per-part sampling params. Only the wall-clock changes.
        """
        import numpy as np
        flat, owners = [], []     # part texts, and the deferred[] position each belongs to
        for pos, (_idx, clean, _gap, _reason) in enumerate(deferred):
            parts = self._split_long_text(clean, max_length=max(60, len(clean) // 2))
            if len(parts) < 2:
                # Unsplittable text — the ladder falls through to a plain render, and so
                # do we. _generate_parts_batched still applies the cap ladder to it.
                parts = [clean]
            for p in parts:
                flat.append(p)
                owners.append(pos)

        waves = self._generate_parts_batched(flat, 1)

        for pos, (idx, clean, gap, reason) in enumerate(deferred):
            try:
                mine = [w for w, o in zip(waves, owners) if o == pos]
                audio_np = np.concatenate(mine) if len(mine) > 1 else mine[0]
                if reason == 'short':
                    self._ratchet_after_resplit(clean, audio_np)
                results[idx] = self._save_audio(idx, audio_np, gap[0], gap[1])
            except Exception as resplit_err:
                print(f"Orpheus deferred re-render failed for sentence {idx}: {resplit_err}")
                if is_fatal_cuda_error(resplit_err):
                    # Poisoned context — every remaining sentence would fail instantly.
                    raise
                results[idx] = False

    def convert_batch(self, items: list) -> list:
        """Convert many sentences in ONE vLLM generate() call.

        items: list of (sentence_index, sentence). Returns list[bool] aligned to items.

        vLLM is built to run many prompts at once: batching is faster (real
        concurrency) AND collapses tens of thousands of single-prompt calls into
        ~len(book)/BATCH_SIZE calls, which avoids the steady host-RAM growth the
        per-call path caused over a long book. MLX has its own batched path
        (_convert_mlx_batch); transformers falls back to per-item convert().

        `items` is batch_pool_size long on both backends (== BATCH_SIZE).
        _convert_mlx_batch re-slices it against the MLX memory budget, so an MLX
        batch is never WIDER than BATCH_SIZE and may be narrower.
        """
        if self.backend == 'mlx' and self.mlx_model:
            return self._convert_mlx_batch(items)
        if self.backend != 'vllm' or not self.engine:
            return [self.convert(idx, s) for idx, s in items]
        try:
            from vllm import SamplingParams, TokensPrompt

            results = {}
            gen = []  # (idx, clean_text, prompt_ids, (gap_sec, gap_pos)) for non-empty sentences
            for idx, sentence in items:
                gap = self._classify_gap(sentence)
                clean = self._clean_sentence_for_tts(sentence)
                if not clean:
                    results[idx] = self._write_silence(idx)
                else:
                    gen.append((idx, clean, self._format_prompt_ids(clean), gap))

            if gen:
                # Per-item SamplingParams: the EOS-boost threshold depends on each
                # sentence's expected length. vLLM accepts a list aligned to prompts.
                sampling_params = [self._vllm_sampling_params(len(clean)) for _, clean, _, _ in gen]
                prompts = [TokensPrompt(prompt_token_ids=fp) for _, _, fp, _ in gen]
                # One voice per audiobook worker, so one LoRARequest covers the whole
                # batch (vLLM applies a scalar to every prompt). It rides on BOTH
                # calls — the use_tqdm fallback without it would render the base voice.
                lora_request = self._lora_request()
                # use_tqdm=False: a per-call progress bar adds overhead and noise.
                try:
                    outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False,
                                                   lora_request=lora_request)
                except TypeError:
                    outputs = self.engine.generate(prompts, sampling_params,
                                                   lora_request=lora_request)
                # Chunks whose audio came back truncated are NOT re-rendered here. The
                # verdict (_needs_resplit) is pure text-vs-audio, and the force-split
                # re-render is deterministic — the split follows from the text alone —
                # so every failure in this batch can be pooled into ONE extra generate()
                # after the loop instead of N serial single-prompt calls that stall the
                # remaining results behind them. See _render_deferred_resplits.
                deferred = []   # (sentence_index, clean, gap, reason)
                # vLLM returns outputs in the same order as prompts.
                for (idx, clean, _, gap), out in zip(gen, outputs):
                    try:
                        tokens = list(out.outputs[0].token_ids)
                        if self.END_OF_AUDIO_TOKEN in tokens:
                            # Finished cleanly: decode up to the end-of-audio token.
                            tokens = tokens[:tokens.index(self.END_OF_AUDIO_TOKEN)]
                            try:
                                audio_np = self._tokens_to_audio(tokens)
                            except TokenStreamMisaligned as align_err:
                                # Stochastic sampling glitch — one re-render (fresh
                                # tokens) almost always fixes it. If it misaligns
                                # again, the outer except fails just this sentence.
                                print(f"Orpheus: sentence {idx} token stream misaligned ({align_err}); re-rendering once")
                                audio_np = self._generate_audio_vllm_safe(clean)
                        else:
                            # Hit the token cap without finishing → the chunk was too long
                            # and the audio would be clipped. Re-render it split at sentence
                            # boundaries so nothing is cut off.
                            print(f"Orpheus: sentence {idx} hit the audio-token cap; re-rendering split at sentence boundaries")
                            # Keep the RUNAWAY itself before the re-render overwrites it:
                            # this is the one failure that never leaves evidence otherwise.
                            try:
                                capped_np = self._tokens_to_audio(tokens)
                            except Exception:
                                capped_np = None
                            self._keep_reject(idx, clean, capped_np, 'cap',
                                              {'tokens_emitted': len(tokens),
                                               'token_cap': self.MAX_AUDIO_TOKENS})
                            audio_np = self._generate_audio_vllm_safe(clean)
                        # Backstop a silent early-EOS truncation (clean EOS, audio too
                        # short for the text) that the cap check above can't catch.
                        # Only the VERDICT is taken here; the re-render is pooled.
                        reason = self._needs_resplit(idx, clean, audio_np)
                        if reason is not None:
                            deferred.append((idx, clean, gap, reason))
                            continue
                        # REPORT the opposite failure on a sub-floor chunk — audio
                        # far LONGER than the text can justify. Nothing is deferred
                        # and nothing re-rendered: the take stands and is counted.
                        self._report_short_chunk_overrun(idx, clean, audio_np)
                        results[idx] = self._save_audio(idx, audio_np, gap[0], gap[1])
                    except Exception as decode_err:
                        print(f"Orpheus batch decode error for sentence {idx}: {decode_err}")
                        if is_fatal_cuda_error(decode_err):
                            # Poisoned CUDA context: every remaining sentence would
                            # fail instantly too. Die loudly; the worker respawns and
                            # resumes from the sentence files already on disk.
                            raise
                        results[idx] = False

                if deferred:
                    self._render_deferred_resplits(deferred, results)

            # NO _cleanup_memory() here: it empty_cache()s the CUDA allocator, and at
            # one flush per batch (~113 per book) every subsequent batch re-allocates
            # from the raw driver — through WSL's paravirtual (dxg) path that is both
            # slow and a VA-fragmentation source (the very OOMs the decode ladder
            # above recovers from). The caching allocator is bounded by vLLM's
            # reservation regardless; host-side garbage is handled by refcounting and
            # the periodic cleanup the worker does between chunks.
            import gc
            gc.collect()
            return [results.get(idx, False) for idx, _ in items]
        except Exception as e:
            print(f'Orpheus.convert_batch() error: {e}')
            import traceback
            traceback.print_exc()
            if is_fatal_cuda_error(e):
                # Do NOT fall through to the per-item retry: with a poisoned CUDA
                # context each convert() would fail in microseconds and the run
                # would churn through the whole book marking sentences failed.
                raise
            # A batch-level failure shouldn't lose the whole chunk — retry per item.
            return [self.convert(idx, s) for idx, s in items]

    def _convert_mlx_batch(self, items: list) -> list:
        """Batched MLX decode via mlx_lm.BatchGenerator (Mac).

        items: list of (sentence_index, sentence). Returns list[bool] aligned to items.

        Mirrors the vLLM batch path — same per-item clean / _classify_gap /
        _write_silence handling and _save_audio finalize (so inter-sentence and
        paragraph pauses are preserved) — but drives ONE continuous-batching
        generate over the whole chunk instead of len(chunk) single-prompt calls.

        mlx_lm.BatchGenerator handles left-padding, a per-row BatchKVCache, and
        per-row stop tokens; insert() takes pre-tokenized prompts, which Orpheus
        needs (custom special-token prompts, not plain text). Audio is then
        reconstructed per row exactly as llama.py generate() does for the
        non-streaming path: parse_output(prompt+generated) -> decode_audio_from_codes.

        Memory (measured on M1 Ultra 64 GB, Jul 2026): weights ~6.9 GB, plus KV at
        a MEASURED 0.1147 MB per generated token per row — so peak scales with
        width x DEPTH, not with width alone. On top of that the allocator's buffer
        CACHE would grow unbounded (~46 GB over one chunk) if not limited —
        bounded once at load via mx.set_cache_limit (_load_mlx_engine), NOT by
        per-chunk flushes. _mlx_batch_groups derives each batch's width from its
        own token depth so the total stays inside MLX_MEM_BUDGET_GB (default 45);
        at the full 3700-token cap that lands at 72 rows.

        Throughput is bought by BATCH WIDTH: 12.4 sent/min at B=16 → 27-28 at
        B=96 (the knee; B=128 is slower). `items` is exactly one BATCH_SIZE chunk
        (batch_pool_size) — the old 4x pool existed only to refill length buckets,
        and both are gone now that 0.31.3 right-pads prefills.

        Sampling follows the per-voice caps
        (temperature/topP/minP/repPenalty, see VOICE_CAP_SOURCES) like the vLLM
        paths — except min_p, which the single-seq MLX path (_generate_mlx) can't apply
        (mlx_audio's generate doesn't plumb it; see the MIN_P comment).
        """
        try:
            import numpy as np
            import mlx.core as mx
            from mlx_lm.generate import BatchGenerator
            from mlx_lm.sample_utils import make_sampler, make_logits_processors
            from mlx_audio.tts.models.llama.llama import decode_audio_from_codes

            results = {}
            sentence_by_idx = dict(items)  # for per-item retry on a bucket failure
            gen = []  # (idx, prompt_tokens, (clean_text, gap), token_budget)
            for idx, sentence in items:
                gap = self._classify_gap(sentence)
                clean = self._clean_sentence_for_tts(sentence)
                if not clean:
                    results[idx] = self._write_silence(idx)
                else:
                    # prepare_input_ids prepends "voice: " itself; pass voice, not a
                    # pre-formatted string. Single-string call returns a [1, T] array.
                    ptoks = self.mlx_model.prepare_input_ids(clean, self.voice)[0].tolist()
                    gen.append((idx, ptoks, (clean, gap), self._mlx_token_budget(clean)))

            # Batches are consecutive BOOK-ORDER slices whose width is capped
            # against the MLX memory budget from their own token depth
            # (_mlx_batch_groups). No length bucketing: mlx-lm 0.31.3 right-pads
            # batch prefills, so a short heading batched next to packed prose is
            # safe. Each group is its own BatchGenerator; the model stays loaded so
            # the extra prefills are cheap relative to generation.
            groups = self._mlx_batch_groups(gen)
            for group_no, (bucket, depth) in enumerate(groups, 1):
                try:
                    bg = BatchGenerator(
                        self.mlx_model,
                        max_tokens=depth,
                        # 0.31.3: stop SEQUENCES, not a set of ints — a set iterates
                        # without raising and silently fails to stop.
                        stop_tokens=[[self.END_OF_AUDIO_TOKEN]],
                        sampler=make_sampler(self._voice_cap('temperature'),
                                             top_p=self._voice_cap('topP'),
                                             min_p=self._voice_cap('minP')),
                        logits_processors=make_logits_processors(
                            None, self._voice_cap('repPenalty'), self.MLX_REP_WINDOW),
                        completion_batch_size=len(bucket),
                        prefill_batch_size=len(bucket),
                    )
                    # Row slot 2 here is (clean_text, gap) — NOT the bare string the
                    # other call site carries. len(c) on the tuple is 2, which floors
                    # `expected` at 300 and fired the boost at ~600 tokens on EVERY
                    # row (measured 2026-08-21: 46/46 chunks truncated to ~7s at
                    # 40-60 ch/s before this line said c[0]).
                    boosts = [self._mlx_eos_boost_processor(len(c[0])) for _, _, c, _ in bucket]
                    if any(boosts):
                        rep = make_logits_processors(
                            None, self._voice_cap('repPenalty'), self.MLX_REP_WINDOW)
                        uids = bg.insert(
                            [list(p) for _, p, _, _ in bucket],
                            logits_processors=[rep + [b] if b else list(rep) for b in boosts])
                    else:
                        uids = bg.insert([list(p) for _, p, _, _ in bucket])
                    out = {u: [] for u in uids}
                    # Heartbeat: a long MLX batch is otherwise SILENT for minutes, which
                    # the BookForge worker watchdog reads as "stuck" and false-kills the
                    # worker (its GENERATION_ACTIVITY_RE only knew vLLM's tqdm). Emit a
                    # throttled liveness line carrying real progress — it keeps the
                    # watchdog fresh AND is the ONLY thing the queue UI can draw a
                    # within-batch progress bar from (the rendered files all land at
                    # once when the batch ends, so the chunk bar is frozen until then).
                    #
                    # Payload, in the one order that keeps BOTH directions compatible:
                    #   [ORPHEUS] MLX batch generating: 95 rows, ~1259 tokens
                    #             (step 1260/3400), 12/95 rows done, batch 1/2
                    # The leading "<N> rows, ~<T> tokens" is byte-identical to the old
                    # line, so an OLD BookForge (whose regex ends there) still parses a
                    # NEW fork; everything after is additive, so a NEW BookForge reads
                    # the extra fields when they're there and degrades to token-only
                    # progress when they aren't. `rows done` counts rows RETIRED
                    # (finish_reason set — 'stop' or the 'length' cap), which each row
                    # reports exactly once before BatchGenerator filters it out of the
                    # live set: monotone, and exact at completion. step/depth is the
                    # token-depth bound the width derivation used, so a fraction is
                    # computable before any row has retired.
                    #
                    # Interval is 10 s, down from 15: the line is now a UI progress
                    # source, not just a liveness ping, and 10 s is under the "is it
                    # frozen?" threshold while still costing ~36 log lines per 6-minute
                    # batch. The watchdog (12 min) has an enormous margin either way.
                    import time as _time
                    _step = 0
                    _retired = 0
                    _HB_SECONDS = 10.0
                    _last_hb = 0.0  # 0 → the first generated step prints immediately
                    # 0.31.3: next() returns (prompt_responses, generation_responses),
                    # so `while responses := bg.next()` would never terminate.
                    # next_generated() yields just the generation list and returns
                    # empty once every row has retired.
                    while responses := bg.next_generated():
                        _step += 1
                        for r in responses:
                            if r.finish_reason is not None:
                                # 'stop' (EOS matched) or 'length' (hit max_tokens) —
                                # either way this row is done and won't be seen again.
                                _retired += 1
                            if r.finish_reason != 'stop':  # stop token (128258) is dropped
                                out[r.uid].append(r.token)
                        _now = _time.time()
                        if _now - _last_hb >= _HB_SECONDS:
                            _maxtok = max((len(v) for v in out.values()), default=0)
                            print(f"[ORPHEUS] MLX batch generating: {len(bucket)} rows, "
                                  f"~{_maxtok} tokens (step {_step}/{depth}), "
                                  f"{_retired}/{len(bucket)} rows done, "
                                  f"batch {group_no}/{len(groups)}", flush=True)
                            _last_hb = _now
                    bg.close()
                    # uids come back in insert order == bucket order.
                    for (idx, ptoks, (clean, gap), _budget), uid in zip(bucket, uids):
                        try:
                            if len(out[uid]) >= depth:
                                # Hit the token cap without finishing → the audio would be
                                # clipped. Re-render split at sentence boundaries instead.
                                print(f"Orpheus: sentence {idx} hit the MLX audio-token cap; re-rendering split at sentence boundaries")
                                # Keep the runaway before the re-render replaces it (see
                                # _keep_reject); same evidence problem as the vLLM path.
                                try:
                                    ids = mx.array([ptoks + out[uid]])
                                    code_lists = self.mlx_model.parse_output(ids)
                                    capped = np.array(decode_audio_from_codes(code_lists[0])[0],
                                                      dtype=np.float32) if code_lists and len(code_lists[0]) > 0 else None
                                except Exception:
                                    capped = None
                                self._keep_reject(idx, clean, capped, 'cap',
                                                  {'tokens_emitted': len(out[uid]),
                                                   'token_cap': depth,
                                                   'backend': 'mlx'})
                                # force_split: the cap hit is PROVEN, so skip the whole
                                # re-render _generate_mlx_safe would otherwise try first.
                                audio = self._generate_mlx_safe(clean, force_split=True)
                                results[idx] = self._save_audio(idx, audio, gap[0], gap[1])
                                continue
                            ids = mx.array([ptoks + out[uid]])
                            code_lists = self.mlx_model.parse_output(ids)
                            if code_lists and len(code_lists[0]) > 0:
                                audio = np.array(
                                    decode_audio_from_codes(code_lists[0])[0], dtype=np.float32
                                )
                            else:
                                # No valid codes (immediate early-EOS / unparseable
                                # stream): EMPTY, not fabricated silence — fabricated
                                # zeros sailed past _save_audio's empty-rejection and
                                # shipped the sentence as success with no audio. The
                                # guard below re-renders empties once; if that fails
                                # too, _save_audio fails the row loudly.
                                audio = np.zeros(0, dtype=np.float32)
                            # Backstop a silent early-EOS truncation (clean stop, audio
                            # too short for the text) the token-cap check can't catch.
                            # force_split: a whole-chunk re-render would just clean-EOS
                            # (truncated) again — the resplit must actually split.
                            audio = self._guard_truncation(
                                idx, clean, audio,
                                lambda c: self._generate_mlx_safe(c, force_split=True)
                            )
                            results[idx] = self._save_audio(idx, audio, gap[0], gap[1])
                        except Exception as decode_err:
                            print(f"Orpheus MLX batch decode error for sentence {idx}: {decode_err}")
                            results[idx] = False
                except Exception as bucket_err:
                    # A generation-phase failure (BatchGenerator/insert/next) for
                    # ONE bucket must not kill the others. Mirror the outer batch-
                    # level recovery at bucket granularity: retry this bucket's rows
                    # per item via convert() rather than dropping them.
                    print(f"Orpheus._convert_mlx_batch() bucket error: {bucket_err}")
                    import traceback
                    traceback.print_exc()
                    for idx, _ptoks, _payload, _budget in bucket:
                        if idx not in results:
                            results[idx] = self.convert(idx, sentence_by_idx[idx])

            # NO mx.clear_cache() / _cleanup_memory() here. The buffer cache is
            # bounded once at load (_load_mlx_engine sets mx.set_cache_limit), so
            # the footprint stays flat without flushing; a per-chunk flush forces
            # every next chunk to re-allocate cold from Metal (measured ~3% slower
            # at batch 96, and it HID the real problem: the cache ballooned to
            # ~46 GB DURING each chunk, between the flushes).
            return [results.get(idx, False) for idx, _ in items]
        except Exception as e:
            print(f'Orpheus._convert_mlx_batch() error: {e}')
            import traceback
            traceback.print_exc()
            # A batch-level failure shouldn't lose the whole chunk — retry per item.
            return [self.convert(idx, s) for idx, s in items]

    def create_vtt(self, all_sentences: list) -> bool:
        """Generate VTT subtitle file from sentences."""
        audio_dir = self.session['sentences_dir']
        vtt_path = os.path.join(
            self.session['process_dir'],
            Path(self.session['final_name']).stem + '.vtt'
        )
        if self._build_vtt_file(all_sentences, audio_dir, vtt_path):
            return True
        return False

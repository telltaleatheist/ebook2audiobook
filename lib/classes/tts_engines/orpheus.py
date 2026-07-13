from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.classes.tts_engines.common.audio import trim_audio
# TEST Step 3c: Direct imports since headers no longer provides them
import torch
import torchaudio
import numpy as np
import platform
import sys
import os
import re
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
    # (_generate_mlx_safe), mirroring the vLLM ladder below.
    #
    # 3700 matches MAX_AUDIO_TOKENS (the vLLM cap): ~8 audio tokens/char means the
    # ~450-char packed chunks need ~2500-3400 tokens, so the old 2048 default made
    # nearly EVERY prose chunk overflow into the split ladder — and those split
    # boundaries were a measured junk source. Validated 2026-07-08 on the 13-chunk
    # real-book set (M1 Ultra): 2048 → 11 cap-hits, 12.1% WER, 1 catastrophic row,
    # ~310s; 3700 → 0 cap-hits, 3.5% WER (Whisper noise floor), 0 catastrophic,
    # ~105s (2.8x faster), peak memory ~13.7 GB (vs ~14.2 at 2048).
    MLX_MAX_TOKENS = int(os.environ.get('ORPHEUS_MLX_MAX_TOKENS', '3700'))

    # Max prompt-length spread (longest/shortest TOKEN count) allowed inside a
    # single MLX BatchGenerator prefill. Mixed-length prompts get heavily left-
    # padded to the longest row, which stochastically corrupts the short rows into
    # gibberish audio — so the MLX batch paths bucket prompts by near-uniform
    # length (see _mlx_length_buckets). Override with ORPHEUS_MLX_BUCKET_RATIO.
    MLX_BUCKET_LEN_RATIO = float(os.environ.get('ORPHEUS_MLX_BUCKET_RATIO', '1.5'))

    # Max audio tokens per generation. ~8 audio tokens/char, so ~3700 covers the
    # ~450-char multi-sentence chunks core.py now feeds Orpheus while staying under
    # the 4096 model context (prompt + audio). A chunk that would exceed this is
    # re-rendered split at sentence boundaries (see _generate_audio_vllm_safe) so the
    # audio is never clipped. Override with ORPHEUS_MAX_TOKENS.
    MAX_AUDIO_TOKENS = int(os.environ.get('ORPHEUS_MAX_TOKENS', '3700'))

    # Sampling params at the Orpheus reference values. The spurious leading-syllable
    # artifact once chased here (cooling temperature to 0.5 only flattened prosody and
    # didn't remove it) was a prompt-framing bug — a stray BOS in the vLLM prompt,
    # fixed in _format_prompt_ids — NOT sampling or chunk size. With the framing correct
    # core.py packs 2-3 sentences per chunk (~450 chars); temp 0.6 is the reference.
    # All three override via env so they can be A/B-tuned live
    # (ORPHEUS_TEMPERATURE / ORPHEUS_TOP_P / ORPHEUS_REP_PENALTY).
    # Temperature 0.85: ear-picked winner of a 0.6-1.0 ladder (2026-07-12,
    # deathstalker_v3, 350-char packing) — livelier prosody; EVERY arm rendered
    # clean (0 guard trips), so the old "cooler is safer" assumption didn't hold
    # on EOS-safe voices. Cooling to 0.5 flattened prosody (see note above).
    TEMPERATURE = float(os.environ.get('ORPHEUS_TEMPERATURE', '0.85'))
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
    # vLLM path only; the MLX sampler has no min_p. 0 disables.
    MIN_P = float(os.environ.get('ORPHEUS_MIN_P', '0.05'))

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

            # Get voice from session or preset
            voice = self.session.get('fine_tuned', self.DEFAULT_VOICE)
            print(f"[ORPHEUS] Session fine_tuned value: '{voice}'")

            if self.custom_model_dir:
                self.MLX_MODEL = self.custom_model_dir
                self.TRANSFORMERS_MODEL = self.custom_model_dir
                self.voice = (voice or '').strip().lower() or self.DEFAULT_VOICE
                print(f"[ORPHEUS] Custom model dir: {self.custom_model_dir}")
                print(f"[ORPHEUS] Using custom voice token: '{self.voice}'")
            else:
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

    @staticmethod
    def _evict_global_cache():
        """Drop the process-global Orpheus model cache so the next load fetches a
        fresh model. Used when switching to a different orpheus_model_dir and on
        cleanup, so a custom voice's weights actually free instead of being reused."""
        for k in ('orpheus', 'orpheus_mlx_model', 'orpheus_snac', 'orpheus_tokenizer',
                  'orpheus_backend', 'orpheus_device', 'orpheus_model_dir'):
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
            # the weights actually free and the next load reloads. Guarded by model_dir
            # so a stale instance's __del__ can't clobber a newer, different model.
            try:
                if loaded_tts.get('orpheus_model_dir', '\0') == getattr(self, 'custom_model_dir', None):
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

    def _detect_backend(self) -> str:
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
        engine = LLM(
            model=self.TRANSFORMERS_MODEL,
            dtype=vllm_dtype,
            max_model_len=4096,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=use_eager,
        )
        self._device = 'cuda'
        return engine

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
            engine_key = 'orpheus'
            engine = loaded_tts.get(engine_key, False)
            cached_dir = loaded_tts.get('orpheus_model_dir', None)
            if engine and cached_dir == self.custom_model_dir:
                self.backend = loaded_tts.get('orpheus_backend', 'transformers')
                self.snac_model = loaded_tts.get('orpheus_snac', None)
                self.tokenizer = loaded_tts.get('orpheus_tokenizer', None)
                self._device = loaded_tts.get('orpheus_device', 'cpu')
                self.mlx_model = loaded_tts.get('orpheus_mlx_model', None)
                print(f"Orpheus already loaded (backend: {self.backend}, model_dir: {cached_dir})")
                return engine
            if engine:
                # A different model is cached (switching custom voices, or custom<->stock).
                # Evict it so its weights free before we load the new one.
                print(f"Orpheus model changed (cached={cached_dir!r} -> want={self.custom_model_dir!r}); reloading")
                self._evict_global_cache()

            # Detect and load appropriate backend
            self.backend = self._detect_backend()

            if self.backend == 'mlx':
                engine = self._load_mlx_engine()
                self.mlx_model = engine
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

            print('Orpheus TTS Loaded!')
            return engine

        except Exception as e:
            error = f'Orpheus.load_engine() error: {e}'
            import traceback
            traceback.print_exc()
            raise ValueError(error)

    def _generate_mlx(self, text: str, max_tokens: int = None) -> np.ndarray:
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
            temperature=self.TEMPERATURE,
            top_p=self.TOP_P,
            repetition_penalty=self.REP_PENALTY,
            max_tokens=max_tokens
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
        return segments[0] if len(segments) == 1 else np.concatenate(segments)

    def _generate_mlx_safe(self, clean: str, depth: int = 0) -> np.ndarray:
        """Render a sentence whose batched MLX generation hit the token cap:
        split at sentence/space boundaries and render each part on its own,
        concatenating the audio so nothing is clipped — the MLX sibling of
        _generate_audio_vllm_safe. Recurses a couple of levels for unusually
        dense text; a part that still can't finish is accepted as-is (same
        bottom rung as the vLLM ladder)."""
        import numpy as np
        if depth >= 3 or len(clean) < 80:
            return self._generate_mlx(clean, max_tokens=self.MLX_MAX_TOKENS)
        parts = self._split_long_text(clean, max_length=max(60, len(clean) // 2))
        if len(parts) < 2:
            return self._generate_mlx(clean, max_tokens=self.MLX_MAX_TOKENS)
        return np.concatenate([self._generate_mlx_safe(p, depth + 1) for p in parts])

    def _mlx_length_buckets(self, entries: list) -> list:
        """entries: list of (key, prompt_token_list, payload). Returns a list of
        buckets (each a list of entries) whose prompt TOKEN lengths are near-
        uniform, so no row in a BatchGenerator prefill gets heavy left-padding.

        WHY (2026-07 investigation): a BatchGenerator pads every prompt in a
        prefill to the LONGEST row. When lengths are MIXED (a 12-char "Chapter
        one." heading batched next to a ~450-char packed-prose row), the heavily
        left-padded short rows stochastically (~1-in-3) collapse into gibberish
        audio. Uniform-length batches are clean at any width; a batch of 1 is
        clean. The padding hazard is set by the longest NEIGHBOR, not the batch
        width. So the fix is caller-side length homogeneity: sort by len(tokens),
        then greedy-group while max_len/min_len <= MLX_BUCKET_LEN_RATIO. A lone
        short outlier naturally becomes its own bucket of 1 == the proven-clean
        solo path.

        Do NOT instead "fix" this by lowering prefill_batch_size below the batch
        size: on mlx-lm 0.30.5 that activates the continuous-batching
        BatchKVCache.extend() path, which is buggy upstream until 0.31.3.
        """
        ratio = self.MLX_BUCKET_LEN_RATIO
        ordered = sorted(entries, key=lambda e: len(e[1]))
        buckets = []
        cur = []
        cur_min = None  # shortest prompt in the current bucket (== first, sorted asc)
        for e in ordered:
            n = len(e[1])
            if not cur:
                cur, cur_min = [e], n
            elif n <= cur_min * ratio:
                cur.append(e)
            else:
                buckets.append(cur)
                cur, cur_min = [e], n
        if cur:
            buckets.append(cur)
        return buckets

    def _generate_mlx_batch_audio(self, texts: list) -> list:
        """Batch-generate raw audio for many (already-cleaned) sentences in ONE
        MLX BatchGenerator pass — continuous batching, ~3.6x the per-sentence
        throughput. Returns float32 waveforms aligned to `texts` (None for an
        empty/failed item).

        In-memory sibling of _convert_mlx_batch's core (which writes WAVs): the
        streaming server uses this to generate a whole paragraph's sentences in
        parallel so MLX stays ahead of playback instead of dribbling one slow
        sentence at a time. MLX backend only.
        """
        import numpy as np
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        from mlx_audio.tts.models.llama.llama import decode_audio_from_codes

        results = [None] * len(texts)
        gen = []  # (index, prompt_tokens, clean_text) for non-empty sentences
        for i, t in enumerate(texts):
            clean = (t or '').strip()
            if clean:
                ptoks = self.mlx_model.prepare_input_ids(clean, self.voice)[0].tolist()
                gen.append((i, ptoks, clean))

        if not gen:
            return results
        # Group prompts into near-uniform-length buckets so no BatchGenerator
        # prefill row gets heavy left-padding (mixed lengths → stochastic short-
        # row gibberish; see _mlx_length_buckets). Each bucket is its own
        # BatchGenerator; the model stays loaded so extra prefills are cheap. One
        # bad bucket must not kill the rest, so each is wrapped independently
        # (matching the old whole-batch try/except granularity).
        for bucket in self._mlx_length_buckets(gen):
            try:
                bg = BatchGenerator(
                    self.mlx_model,
                    max_tokens=self.MLX_MAX_TOKENS,
                    stop_tokens={self.END_OF_AUDIO_TOKEN},
                    sampler=make_sampler(self.TEMPERATURE, top_p=self.TOP_P),
                    logits_processors=make_logits_processors(None, self.REP_PENALTY, 20),
                    completion_batch_size=len(bucket),
                    prefill_batch_size=len(bucket),
                )
                uids = bg.insert([list(p) for _, p, _ in bucket])
                out = {u: [] for u in uids}
                while responses := bg.next():
                    for r in responses:
                        if r.finish_reason != 'stop':  # stop token (128258) is dropped
                            out[r.uid].append(r.token)
                bg.close()
                for (i, ptoks, clean), uid in zip(bucket, uids):
                    try:
                        if len(out[uid]) >= self.MLX_MAX_TOKENS:
                            # Cap hit without finishing — re-render split so the
                            # played audio is never clipped mid-sentence.
                            print(f"Orpheus: stream sentence [{i}] hit the MLX audio-token cap; re-rendering split")
                            results[i] = self._generate_mlx_safe(clean)
                            continue
                        ids = mx.array([ptoks + out[uid]])
                        code_lists = self.mlx_model.parse_output(ids)
                        if code_lists and len(code_lists[0]) > 0:
                            results[i] = np.array(
                                decode_audio_from_codes(code_lists[0])[0], dtype=np.float32
                            )
                    except Exception as decode_err:
                        print(f"Orpheus _generate_mlx_batch_audio decode error [{i}]: {decode_err}")
            except Exception as e:
                print(f"Orpheus._generate_mlx_batch_audio() bucket error: {e}")
                import traceback
                traceback.print_exc()
        # No mx.clear_cache(): the cache is bounded at load (set_cache_limit),
        # and flushing per read-ahead batch just forces cold re-allocation.
        return results

    def _format_prompt_ids(self, text: str) -> list:
        """Return the exact Orpheus input token IDs for `text`.

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
        """
        body = self.tokenizer(f"{self.voice}: {text}").input_ids   # includes leading BOS
        return [128259] + list(body) + [128009, 128260, 128261, 128257]

    def _generate_tokens_vllm(self, prompt: str, max_tokens: int = None) -> list:
        """Generate audio tokens using vLLM backend."""
        from vllm import SamplingParams, TokensPrompt
        if max_tokens is None:
            max_tokens = self.MAX_AUDIO_TOKENS

        # Feed token IDs directly (no decode->re-tokenize round-trip; see _format_prompt_ids)
        prompt_ids = self._format_prompt_ids(prompt)

        sampling_params = SamplingParams(
            temperature=self.TEMPERATURE,
            top_p=self.TOP_P,
            min_p=self.MIN_P,
            repetition_penalty=self.REP_PENALTY,
            max_tokens=max_tokens,
            stop_token_ids=[self.END_OF_AUDIO_TOKEN]
        )

        outputs = self.engine.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sampling_params)
        tokens = list(outputs[0].outputs[0].token_ids)

        # Truncate at end-of-audio token if present
        if self.END_OF_AUDIO_TOKEN in tokens:
            end_idx = tokens.index(self.END_OF_AUDIO_TOKEN)
            tokens = tokens[:end_idx]

        return tokens

    def _generate_audio_vllm_safe(self, clean: str, depth: int = 0, force_split: bool = False):
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
        """
        from vllm import SamplingParams, TokensPrompt
        import numpy as np
        if force_split:
            parts = self._split_long_text(clean, max_length=max(60, len(clean) // 2))
            if len(parts) >= 2:
                return np.concatenate([self._generate_audio_vllm_safe(p, depth + 1) for p in parts])
        sampling_params = SamplingParams(
            temperature=self.TEMPERATURE, top_p=self.TOP_P, min_p=self.MIN_P,
            repetition_penalty=self.REP_PENALTY,
            max_tokens=self.MAX_AUDIO_TOKENS, stop_token_ids=[self.END_OF_AUDIO_TOKEN]
        )
        prompt = TokensPrompt(prompt_token_ids=self._format_prompt_ids(clean))
        try:
            outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
        except TypeError:
            outputs = self.engine.generate([prompt], sampling_params)
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
        return np.concatenate([self._generate_audio_vllm_safe(p, depth + 1) for p in parts])

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
                temperature=self.TEMPERATURE,
                top_p=self.TOP_P,
                repetition_penalty=self.REP_PENALTY,
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
        ([break]/[pause]/[music]/[sfx]/[silence]); Orpheus has its own emotion tags."""
        sentence = (sentence or '').strip()
        sentence = re.sub(r'\[(?:break|pause|music|sfx|silence)(?::[^\]]+)?\]', '', sentence, flags=re.IGNORECASE)
        return sentence.strip()

    def _classify_gap(self, sentence: str):
        """Decide the inter-clip silence for this sentence from the e2a SML tokens
        it carries, matching the cross-engine standard in common/utils.py
        (_convert_sml). Returns (lead_gap_sec, trail_gap_sec):

          [pause] / [pause:X]  -> SECTION gap: explicit X, else 1.0-1.6s. Marks
                                  blank-line / heading / <div> boundaries.
          [break] / [silence]  -> PARAGRAPH gap: 0.5-0.7s. Marks <p>/<br> ends.
          (no token)           -> SENTENCE gap: fixed 0.75s so Orpheus breathes
                                  instead of rushing.

        INVARIANT: a chunk's tail is NEVER bare. Every chunk gets at least the
        sentence gap on its trailing edge; a boundary token at the chunk START
        adds its (longer) paragraph/section gap as a LEADING pad ON TOP. This
        replaces the old one-gap-either-side design, where a chunk that STARTED
        with [break]/[pause] got the whole gap prepended and its tail got nothing
        but the 0.20s trim buffer — ~25% of chunks (the lead chunks) so joined the
        NEXT chunk almost instantly, verified in the rendered flac. Now:

          lead_gap  = the classified paragraph/section gap IF a token opens the
                      chunk, else 0.
          trail_gap = the sentence gap, EXCEPT a token that sits at the chunk end
                      raises the tail to max(token gap, sentence gap) — a [break]'s
                      0.5-0.7 must never yield a SHORTER tail than plain prose's
                      0.75. Checked on the RAW sentence, before
                      _clean_sentence_for_tts strips the tokens.

        Why three tiers and not the old binary [break]==[pause]: the standard
        treats [break] as a SHORT gap and [pause] as the LONG one. Lumping them
        meant every <p>/<br>/<span> [break] got a long gap — long pauses in odd
        places. Now [break] is a modest paragraph gap and only [pause] (rare, real
        section breaks) is long.

        Env overrides (a fixed value, 0 disables): ORPHEUS_SENTENCE_GAP,
        ORPHEUS_PARAGRAPH_GAP, ORPHEUS_SECTION_GAP.
        """
        raw = (sentence or '').strip()
        lowered = raw.lower()

        def _env(name):
            v = os.environ.get(name)
            return float(v) if v is not None else None

        # The sentence gap is the tail FLOOR: every chunk ends with at least this
        # so a chunk-to-chunk join is never bare. Was a random 0.55-0.75; the short
        # end left some transitions too tight while the long end sounded right, so
        # it's a FIXED 0.75. Override with ORPHEUS_SENTENCE_GAP.
        sentence_gap = _env('ORPHEUS_SENTENCE_GAP')
        if sentence_gap is None:
            sentence_gap = 0.6   # tightened from 0.75 (2026-07-12, ear-approved on rohan)

        # Section pause ([pause] / [pause:X]) — the strong, long boundary.
        m = re.search(r'\[pause(?::([0-9.]+))?\]', raw, flags=re.IGNORECASE)
        if m:
            override = _env('ORPHEUS_SECTION_GAP')
            if override is not None:
                token_gap = override
            elif m.group(1):
                token_gap = float(m.group(1))               # honor an explicit [pause:1.4]
            else:
                token_gap = int(np.random.uniform(1.0, 1.6) * 100) / 100
            if lowered.startswith('[pause'):
                return token_gap, sentence_gap               # long lead + normal sentence tail
            return 0.0, max(token_gap, sentence_gap)          # token at end → tail at least the floor

        # Paragraph break ([break] / [silence]) — modest gap.
        m = re.search(r'\[(?:break|silence)(?::[^\]]+)?\]', raw, flags=re.IGNORECASE)
        if m:
            override = _env('ORPHEUS_PARAGRAPH_GAP')
            token_gap = override if override is not None else int(np.random.uniform(0.5, 0.7) * 100) / 100
            if re.match(r'\[(?:break|silence)', lowered):
                return token_gap, sentence_gap
            return 0.0, max(token_gap, sentence_gap)

        # Plain sentence end — the gap BETWEEN chunks (each packed chunk of 2-3
        # sentences ends here). Pauses WITHIN a chunk are the model's own.
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

    def _guard_truncation(self, sentence_index: int, clean: str, audio_np, resplit):
        """Backstop for silent early-EOS truncation (see _speech_rate). If the
        rendered clip is too short for the text, re-render it split at sentence
        boundaries via `resplit` (the backend's _generate_*_safe ladder). `resplit`
        takes the clean text and returns a numpy waveform — and it must ACTUALLY
        split, not merely re-render: the failure detected here is a CLEAN early
        EOS, so a whole-chunk re-render would most likely clean-EOS (truncated)
        again. The vLLM callers therefore pass _generate_audio_vllm_safe with
        force_split=True (its plain form accepts any clean EOS unsplit — that's
        correct for its token-cap callers, useless here); _generate_mlx_safe
        splits eagerly before rendering, so the MLX callers pass it as-is.

        Guard applies only when clean length > 150 chars (short chunks' rates are
        noisy) and ORPHEUS_MAX_CHARS_PER_SEC > 0 (default 19.0; set 0 to disable).
        If the re-render STILL trips the guard we accept it and warn — no infinite
        loop; the _generate_*_safe ladders already carry their own depth guard.

        EMPTY audio for non-empty text (immediate early-EOS / unparseable stream)
        is a definite failure, not a noisy metric, so it gets its one re-render
        regardless of the 150-char floor. If the retake is empty too, the empty
        result flows back to the caller — _save_audio rejects it (row fails
        loudly) and the streaming worker emits 'No audio generated'."""
        if (audio_np is None or len(audio_np) == 0) and clean and clean.strip():
            print(f"Orpheus: sentence {sentence_index} produced no audio — re-rendering split at sentence boundaries")
            return resplit(clean)
        max_rate = float(os.environ.get('ORPHEUS_MAX_CHARS_PER_SEC', '19.0'))
        if max_rate <= 0 or len(clean) <= 150:
            return audio_np
        rate = self._speech_rate(clean, audio_np)
        if rate is None or rate <= max_rate:
            return audio_np
        print(f"Orpheus: sentence {sentence_index} audio too short for text "
              f"({rate:.1f} ch/s > {max_rate:.1f}) — re-rendering split at sentence boundaries")
        audio_np = resplit(clean)
        rate2 = self._speech_rate(clean, audio_np)
        if rate2 is not None and rate2 > max_rate:
            print(f"Orpheus: sentence {sentence_index} still short after re-render "
                  f"({rate2:.1f} ch/s > {max_rate:.1f}); accepting")
        return audio_np

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
                    audio_np = self._generate_mlx(clean)
                    audio_np = self._guard_truncation(sentence_index, clean, audio_np, self._generate_mlx_safe)
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

    def convert_batch(self, items: list) -> list:
        """Convert many sentences in ONE vLLM generate() call.

        items: list of (sentence_index, sentence). Returns list[bool] aligned to items.

        vLLM is built to run many prompts at once: batching is faster (real
        concurrency) AND collapses tens of thousands of single-prompt calls into
        ~len(book)/BATCH_SIZE calls, which avoids the steady host-RAM growth the
        per-call path caused over a long book. MLX has its own batched path
        (_convert_mlx_batch); transformers falls back to per-item convert().
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
                sampling_params = SamplingParams(
                    temperature=self.TEMPERATURE, top_p=self.TOP_P, min_p=self.MIN_P,
                    repetition_penalty=self.REP_PENALTY,
                    max_tokens=self.MAX_AUDIO_TOKENS, stop_token_ids=[self.END_OF_AUDIO_TOKEN]
                )
                prompts = [TokensPrompt(prompt_token_ids=fp) for _, _, fp, _ in gen]
                # use_tqdm=False: a per-call progress bar adds overhead and noise.
                try:
                    outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False)
                except TypeError:
                    outputs = self.engine.generate(prompts, sampling_params)
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
                            audio_np = self._generate_audio_vllm_safe(clean)
                        # Backstop a silent early-EOS truncation (clean EOS, audio too
                        # short for the text) that the cap check above can't catch.
                        # force_split: a whole-chunk re-render would just clean-EOS
                        # (truncated) again — the resplit must actually split.
                        audio_np = self._guard_truncation(
                            idx, clean, audio_np,
                            lambda c: self._generate_audio_vllm_safe(c, force_split=True)
                        )
                        results[idx] = self._save_audio(idx, audio_np, gap[0], gap[1])
                    except Exception as decode_err:
                        print(f"Orpheus batch decode error for sentence {idx}: {decode_err}")
                        if is_fatal_cuda_error(decode_err):
                            # Poisoned CUDA context: every remaining sentence would
                            # fail instantly too. Die loudly; the worker respawns and
                            # resumes from the sentence files already on disk.
                            raise
                        results[idx] = False

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

        Memory (measured on M1 Ultra 64 GB, Jul 2026): peak ACTIVE memory is
        weights ~6.9 GB + ~0.153 GB per batch slot (KV + activations) — 12.8 GB
        at B=16, 21.5 GB at B=96. On top of that the allocator's buffer CACHE
        would grow unbounded (~46 GB over one chunk) if not limited — bounded
        once at load via mx.set_cache_limit (_load_mlx_engine), NOT by per-chunk
        flushes. Throughput: 12.4 sent/min at B=16 → 27-28 at B=96 (the knee;
        B=128 is slower). Sampling matches the single-seq MLX path
        (_generate_mlx): temp 0.6, top_p 0.8, repetition_penalty 1.1.
        """
        try:
            import numpy as np
            import mlx.core as mx
            from mlx_lm.generate import BatchGenerator
            from mlx_lm.sample_utils import make_sampler, make_logits_processors
            from mlx_audio.tts.models.llama.llama import decode_audio_from_codes

            results = {}
            sentence_by_idx = dict(items)  # for per-item retry on a bucket failure
            gen = []  # (idx, prompt_tokens, (clean_text, gap)) for non-empty sentences
            for idx, sentence in items:
                gap = self._classify_gap(sentence)
                clean = self._clean_sentence_for_tts(sentence)
                if not clean:
                    results[idx] = self._write_silence(idx)
                else:
                    # prepare_input_ids prepends "voice: " itself; pass voice, not a
                    # pre-formatted string. Single-string call returns a [1, T] array.
                    ptoks = self.mlx_model.prepare_input_ids(clean, self.voice)[0].tolist()
                    gen.append((idx, ptoks, (clean, gap)))

            # Group prompts into near-uniform-length buckets so no BatchGenerator
            # prefill row gets heavy left-padding (mixed lengths → stochastic
            # short-row gibberish; see _mlx_length_buckets). Each bucket is its own
            # BatchGenerator; the model stays loaded so the extra prefills are cheap
            # relative to generation. A lone short heading falls into its own
            # bucket of 1 == the proven-clean solo path.
            for bucket in self._mlx_length_buckets(gen):
                try:
                    bg = BatchGenerator(
                        self.mlx_model,
                        max_tokens=self.MLX_MAX_TOKENS,
                        stop_tokens={self.END_OF_AUDIO_TOKEN},
                        sampler=make_sampler(self.TEMPERATURE, top_p=self.TOP_P),
                        logits_processors=make_logits_processors(None, self.REP_PENALTY, 20),
                        completion_batch_size=len(bucket),
                        prefill_batch_size=len(bucket),
                    )
                    uids = bg.insert([list(p) for _, p, _ in bucket])
                    out = {u: [] for u in uids}
                    while responses := bg.next():
                        for r in responses:
                            if r.finish_reason != 'stop':  # stop token (128258) is dropped
                                out[r.uid].append(r.token)
                    bg.close()
                    # uids come back in insert order == bucket order.
                    for (idx, ptoks, (clean, gap)), uid in zip(bucket, uids):
                        try:
                            if len(out[uid]) >= self.MLX_MAX_TOKENS:
                                # Hit the token cap without finishing → the audio would be
                                # clipped. Re-render split at sentence boundaries instead.
                                print(f"Orpheus: sentence {idx} hit the MLX audio-token cap; re-rendering split at sentence boundaries")
                                audio = self._generate_mlx_safe(clean)
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
                            audio = self._guard_truncation(idx, clean, audio, self._generate_mlx_safe)
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
                    for idx, _ptoks, _payload in bucket:
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

from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.classes.tts_engines.common.audio import trim_audio
import torch
import torchaudio
import numpy as np
import platform
import os
import re
import atexit
import weakref

# Track active instances for cleanup on exit (mirrors orpheus.py).
_active_instances = weakref.WeakSet()

# vLLM-omni runtime fixes discovered bringing Voxtral up on Windows/WSL CUDA:
#  - flashinfer's top-k/top-p sampler JIT-compiles a CUDA kernel and needs nvcc /
#    CUDA_HOME, which a pip-only env lacks → use vLLM's native torch sampler.
#  - expandable_segments cuts fragmentation OOMs on a shared card.
# Harmless on macOS (MLX path ignores them).
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def _cleanup_on_exit():
    import gc
    for instance in list(_active_instances):
        try:
            instance.cleanup()
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception:
        pass


atexit.register(_cleanup_on_exit)


class Voxtral(TTSUtils, TTSRegistry, name='voxtral'):
    """
    Mistral Voxtral TTS — open-weight, ElevenLabs-class quality. AR semantic stage
    + flow-matching acoustic stage + neural codec, 24 kHz output. 20 preset voices
    across 9 languages, plus zero-shot cloning from a reference clip.

    Backends (auto-detected by platform, like orpheus.py):
      - MLX (Mac, mlx-audio + the *-mlx-4bit weights) — light (~2.5 GB), native.
      - vLLM-omni (Windows-WSL / Linux + CUDA) — the offline Omni batch API.

    Single-worker like Orpheus: vLLM batches internally, so e2a must run it with
    workers=1 (enforced by the BookForge engine registry: maxWorkers=1).
    """

    MODEL = 'mistralai/Voxtral-4B-TTS-2603'
    # bf16 (full precision) — 4bit/6bit produced noticeably worse prosody on Mac
    # MLX; bf16 is the best of the three (~9 GB weights, pulled from HF on first use).
    MLX_MODEL = 'mlx-community/Voxtral-4B-TTS-2603-mlx-bf16'
    SAMPLE_RATE = 24000

    # English-relevant presets (the model also ships de/es/fr/it/nl/pt/hi/ar).
    VALID_VOICES = {
        'neutral_male', 'neutral_female', 'casual_male', 'casual_female',
        'cheerful_female', 'ar_male', 'de_female', 'de_male', 'es_female',
        'es_male', 'fr_female', 'fr_male', 'hi_female', 'hi_male', 'it_female',
        'it_male', 'nl_female', 'nl_male', 'pt_female', 'pt_male',
    }
    DEFAULT_VOICE = 'neutral_male'

    def __init__(self, session):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.resampler_cache = {}
            self.backend = None          # 'mlx' or 'vllm-omni'
            self.tokenizer = None        # mistral instruct tokenizer (vLLM path)
            self.mlx_model = None
            self.ref_audio_path = None   # set when cloning from a reference clip

            try:
                self.models = load_engine_presets(self.session['tts_engine'])
            except Exception:
                self.models = {}

            self.params = {}
            self.params['samplerate'] = self.SAMPLE_RATE

            # Voice resolution: a preset name in session['fine_tuned'] selects a preset;
            # otherwise a reference clip in session['voice'] drives zero-shot cloning.
            voice = self.session.get('fine_tuned', self.DEFAULT_VOICE)
            if voice in self.models:
                voice = self.models[voice].get('voice', voice)
            voice_lower = voice.lower() if voice else self.DEFAULT_VOICE

            if voice_lower in self.VALID_VOICES:
                self.voice = voice_lower
                self.clone = False
                print(f"[VOXTRAL] Using preset voice: '{self.voice}'")
            else:
                ref = self.session.get('voice')
                if ref and os.path.exists(ref):
                    self.voice = None
                    self.clone = True
                    self.ref_audio_path = ref
                    print(f"[VOXTRAL] Cloning from reference clip: {ref}")
                else:
                    self.voice = self.DEFAULT_VOICE
                    self.clone = False
                    print(f"[VOXTRAL] Unknown voice '{voice}' and no reference clip; "
                          f"defaulting to preset '{self.DEFAULT_VOICE}'")

            self.engine = None
            self.engine = self.load_engine()
            _active_instances.add(self)
        except Exception as e:
            raise ValueError(f'Voxtral.__init__() error: {e}')

    # ── backend / load ──────────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        is_mac = platform.system() == 'Darwin'
        forced = os.environ.get('VOXTRAL_BACKEND', '').lower()
        if forced in ('mlx', 'vllm-omni'):
            print(f"Voxtral: backend override VOXTRAL_BACKEND={forced}")
            return forced
        if is_mac:
            try:
                import mlx_audio  # noqa: F401
                print("Voxtral: using MLX backend (Apple Silicon)")
                return 'mlx'
            except ImportError:
                print("Voxtral: mlx-audio not available (pip install mlx-audio)")
        try:
            import vllm_omni  # noqa: F401
            print("Voxtral: using vLLM-omni backend (CUDA)")
            return 'vllm-omni'
        except ImportError:
            raise ImportError(
                "No Voxtral backend available. Install one of:\n"
                "  Mac:           pip install mlx-audio\n"
                "  Windows/Linux: pip install vllm==0.22.0 vllm-omni==0.22.0 (CUDA)"
            )

    def load_engine(self):
        try:
            print("Loading Voxtral TTS...")
            self._cleanup_memory()

            engine = loaded_tts.get('voxtral', False)
            if engine:
                self.backend = loaded_tts.get('voxtral_backend', 'vllm-omni')
                self.tokenizer = loaded_tts.get('voxtral_tokenizer', None)
                self.mlx_model = loaded_tts.get('voxtral_mlx_model', None)
                print(f"Voxtral already loaded (backend: {self.backend})")
                return engine

            self.backend = self._detect_backend()
            if self.backend == 'mlx':
                engine = self._load_mlx_engine()
                self.mlx_model = engine
            else:
                engine = self._load_vllm_omni_engine()

            loaded_tts['voxtral'] = engine
            loaded_tts['voxtral_backend'] = self.backend
            loaded_tts['voxtral_tokenizer'] = self.tokenizer
            loaded_tts['voxtral_mlx_model'] = self.mlx_model
            print('Voxtral TTS loaded!')
            return engine
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f'Voxtral.load_engine() error: {e}')

    def _load_mlx_engine(self):
        # Mac path — verify/finish on Apple Silicon (can't be tested on Windows).
        from mlx_audio.tts.utils import load
        print(f"Loading Voxtral (MLX): {self.MLX_MODEL}")
        model = load(self.MLX_MODEL)
        return model

    def _load_vllm_omni_engine(self):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from vllm_omni.entrypoints.omni import Omni
        # deploy_config: honor an override (BookForge writes a low-VRAM one for
        # shared cards); None lets vllm-omni auto-load the model's default.
        deploy = os.environ.get('VOXTRAL_DEPLOY_CONFIG') or None
        self.tokenizer = MistralTokenizer.from_hf_hub(self.MODEL).instruct_tokenizer
        print(f"Loading Voxtral (vLLM-omni): {self.MODEL} (deploy_config={deploy or 'default'})")
        return Omni(model=self.MODEL, deploy_config=deploy, log_stats=False)

    # ── generation ──────────────────────────────────────────────────────────

    def _compose_inputs(self, text: str) -> dict:
        """Build the vLLM-omni request dict for one sentence (preset or clone)."""
        from mistral_common.protocol.speech.request import SpeechRequest
        if self.clone:
            with open(self.ref_audio_path, 'rb') as f:
                ref_bytes = f.read()
            tokenized = self.tokenizer.encode_speech_request(
                SpeechRequest(input=text, ref_audio=ref_bytes)
            )
            audio = tokenized.audios[0]
            return {
                'multi_modal_data': {'audio': [(audio.audio_array, audio.sampling_rate)]},
                'prompt_token_ids': tokenized.tokens,
            }
        tokenized = self.tokenizer.encode_speech_request(
            SpeechRequest(input=text, voice=self.voice)
        )
        return {
            'additional_information': {'voice': [self.voice]},
            'prompt_token_ids': tokenized.tokens,
        }

    def _generate_vllm_omni(self, text: str) -> np.ndarray:
        from vllm import SamplingParams
        cfg_alpha = float(os.environ.get('VOXTRAL_CFG_ALPHA', '1.2'))
        sp = SamplingParams(max_tokens=2500, extra_args={'cfg_alpha': cfg_alpha})
        inputs = self._compose_inputs(text)
        # sampling_params is per-stage (LLM stage, acoustic stage).
        outputs = self.engine.generate(inputs, [sp, sp])
        audio_tensor = torch.cat((outputs[0].multimodal_output['audio'],))
        return audio_tensor.float().detach().cpu().numpy()

    def _generate_mlx(self, text: str) -> np.ndarray:
        # Verified on Apple Silicon (M1 Ultra, mlx-audio 0.4.4): voxtral_tts
        # Model.generate() is a GENERATOR yielding GenerationResult chunks whose
        # .audio is an mlx array — same pattern as orpheus.py. Concatenate the
        # chunks into one 24 kHz waveform.
        #
        # MLX exposes preset voices only (generate(voice=...) has no reference-audio
        # argument), so a clone request — self.voice is None — falls back to the
        # default preset on this backend. Reference-clip cloning stays on vLLM-omni.
        voice = self.voice or self.DEFAULT_VOICE
        # Sampling tuned on Apple Silicon bf16: temp 0.35 gave the most natural
        # prosody in listening tests (0.8 acceptable; 0.5/0.25 degraded badly).
        # Env-overridable so the BookForge wizard / quick experiments can retune
        # without code edits (mirrors VOXTRAL_CFG_ALPHA on the vLLM path).
        temperature = float(os.environ.get('VOXTRAL_TEMPERATURE', '0.35'))
        top_p = float(os.environ.get('VOXTRAL_TOP_P', '0.9'))
        top_k = int(os.environ.get('VOXTRAL_TOP_K', '50'))
        chunks = []
        for result in self.mlx_model.generate(
            text=text, voice=voice, temperature=temperature, top_p=top_p, top_k=top_k
        ):
            audio = getattr(result, 'audio', None)
            if audio is not None:
                chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            # No audio for non-empty text = generation FAILURE. Return empty so
            # convert() reports the sentence as failed — never substitute silence.
            print("[VOXTRAL] MLX generation yielded no audio chunks (generation failed)")
            return np.array([], dtype=np.float32)
        audio = np.concatenate(chunks)
        # mlx-audio's voxtral_tts emits consistently low-amplitude audio (~0.1 peak)
        # — fully voiced, just quiet. e2a's convert() only attenuates (max>1.0), never
        # boosts, so without this the assembled book sounds faint/whispery. Peak-
        # normalize to a normal speech level here (the vLLM-omni path is already hot).
        peak = float(np.abs(audio).max())
        if peak > 1e-5:
            audio = audio / peak * 0.95
        return audio

    def convert(self, sentence_index: int, sentence: str) -> bool:
        try:
            if not self.engine:
                print("Voxtral TTS engine not loaded!")
                return False

            final_sentence_file = os.path.join(
                self.session['sentences_dir'],
                f'{sentence_index}.{default_audio_proc_format}'
            )

            sentence = sentence.strip()
            # Strip e2a SML tags Voxtral doesn't understand — including the
            # [heading] marker, which is markup for the splitter only and must
            # never be spoken (2026-08-27; the alternation lives in conf_models
            # now, shared with orpheus.py and f5.py).
            sentence = SML_UNSPOKEN_PATTERN.sub('', sentence).strip()

            if not sentence:
                silence = torch.zeros(1, int(self.params['samplerate'] * 0.1))
                torchaudio.save(final_sentence_file, silence, self.params['samplerate'],
                                format=default_audio_proc_format)
                return True

            try:
                if self.backend == 'mlx':
                    audio_np = self._generate_mlx(sentence)
                else:
                    audio_np = self._generate_vllm_omni(sentence)

                if audio_np is None or len(audio_np) == 0:
                    print(f"Voxtral returned no audio for sentence {sentence_index}")
                    return False

                audio_tensor = torch.from_numpy(audio_np).float()
                if audio_tensor.dim() == 1:
                    # Low threshold: Voxtral's sentence-initial unvoiced consonants
                    # (th/s/f/h, "Those"/"By"/"An") sit well below 0.01 and were being
                    # trimmed as silence, clipping the first syllable. 0.002 keeps them
                    # while still trimming true lead/trail silence.
                    audio_tensor = trim_audio(audio_tensor, self.SAMPLE_RATE,
                                              silence_threshold=0.002, buffer_sec=0.10)
                    if len(audio_tensor) == 0:
                        # Non-empty text produced only silence — a failed render.
                        # Fail loudly instead of saving silence as a valid sentence.
                        print(f"Voxtral produced only silence for sentence {sentence_index} "
                              f"(all audio below trim threshold)")
                        return False
                    audio_tensor = audio_tensor.unsqueeze(0)

                max_val = audio_tensor.abs().max()
                if max_val > 1.0:
                    audio_tensor = audio_tensor / max_val * 0.95
                elif max_val == 0:
                    # All-zero audio for non-empty text = failed render, not silence.
                    print(f"Voxtral produced all-zero audio for sentence {sentence_index}")
                    return False

                torchaudio.save(final_sentence_file, audio_tensor.cpu(),
                                self.params['samplerate'], format=default_audio_proc_format)
                del audio_tensor
                self._cleanup_memory()
                return os.path.exists(final_sentence_file)
            except Exception as gen_error:
                import traceback
                traceback.print_exc()
                print(f"Voxtral generation error for sentence {sentence_index}: {gen_error}")
                return False
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'Voxtral.convert() error: {e}')
            return False

    def create_vtt(self, all_sentences: list) -> bool:
        audio_dir = self.session['sentences_dir']
        vtt_path = os.path.join(
            self.session['process_dir'],
            Path(self.session['final_name']).stem + '.vtt'
        )
        return bool(self._build_vtt_file(all_sentences, audio_dir, vtt_path))

    # ── cleanup ─────────────────────────────────────────────────────────────

    def cleanup(self):
        try:
            if getattr(self, 'engine', None) is not None:
                del self.engine
                self.engine = None
            if getattr(self, 'tokenizer', None) is not None:
                self.tokenizer = None
            self._cleanup_memory()
            print("[VOXTRAL] Cleanup complete")
        except Exception as e:
            print(f"[VOXTRAL] Cleanup warning: {e}")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

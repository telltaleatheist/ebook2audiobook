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

_active_instances = weakref.WeakSet()


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


class F5(TTSUtils, TTSRegistry, name='f5'):
    """
    F5-TTS — flow-matching (DiT) TTS with strong long-form prosody and zero-shot
    cloning from a reference clip + its transcript. 24 kHz output.

    Backends (auto-detected by platform, like orpheus.py / voxtral.py):
      - MLX (Mac, f5-tts-mlx) — native Apple Silicon, light.
      - CUDA (Windows/Linux, f5-tts) — the SWivid f5_tts package on a CUDA GPU.

    Unlike Voxtral's presets, F5 has NO built-in speakers: every voice is cloned
    from the selected reference clip (session['voice']) + its transcript
    (session voice's sidecar .txt, the language default.txt, or empty → F5
    auto-transcribes the reference with Whisper).
    """

    MLX_MODEL = 'lucasnewman/f5-tts-mlx'
    SAMPLE_RATE = 24000
    DEFAULT_STEPS = 8     # MLX flow-matching steps
    DEFAULT_NFE = 32      # CUDA f5_tts NFE steps

    def __init__(self, session):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.resampler_cache = {}
            self.backend = None          # 'mlx' or 'cuda'
            self.ref_audio_path = None
            self.ref_audio_text = ''

            try:
                self.models = load_engine_presets(self.session['tts_engine'])
            except Exception:
                self.models = {}

            self.params = {}
            self.params['samplerate'] = self.SAMPLE_RATE

            self._resolve_reference()
            self.engine = None
            self.engine = self.load_engine()
            _active_instances.add(self)
        except Exception as e:
            raise ValueError(f'F5.__init__() error: {e}')

    # ── reference clip + transcript ──────────────────────────────────────────

    def _resolve_reference(self):
        ref = self.session.get('voice')
        if not (ref and os.path.exists(ref)):
            raise ValueError("F5 needs a reference voice clip (session['voice']); none was found.")
        self.ref_audio_path = ref
        # Reference transcript: a sidecar .txt next to the clip, else the language's
        # default.txt, else empty (F5 auto-transcribes the ref with Whisper).
        sidecar = os.path.splitext(ref)[0] + '.txt'
        text = ''
        if os.path.exists(sidecar):
            text = Path(sidecar).read_text(encoding='utf-8').strip()
        else:
            lang = self.session.get('language', 'eng')
            lang_dir = 'con-' if lang == 'con' else lang
            default_txt = os.path.join(voices_dir, lang_dir, 'default.txt')
            if os.path.exists(default_txt):
                text = Path(default_txt).read_text(encoding='utf-8').strip()
        self.ref_audio_text = text
        print(f"[F5] Reference clip: {ref} (ref_text {'set' if text else 'empty → auto-transcribe'})")

    # ── backend / load ───────────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        is_mac = platform.system() == 'Darwin'
        forced = os.environ.get('F5_BACKEND', '').lower()
        if forced in ('mlx', 'cuda'):
            print(f"F5: backend override F5_BACKEND={forced}")
            return forced
        if is_mac:
            try:
                import f5_tts_mlx  # noqa: F401
                print("F5: using MLX backend (Apple Silicon)")
                return 'mlx'
            except ImportError:
                print("F5: f5-tts-mlx not available (pip install f5-tts-mlx)")
        try:
            import f5_tts  # noqa: F401
            print("F5: using CUDA backend (f5-tts)")
            return 'cuda'
        except ImportError:
            raise ImportError(
                "No F5 backend available. Install one of:\n"
                "  Mac:           pip install f5-tts-mlx\n"
                "  Windows/Linux: pip install f5-tts (CUDA)"
            )

    def load_engine(self):
        try:
            print("Loading F5-TTS...")
            self._cleanup_memory()

            engine = loaded_tts.get('f5', False)
            if engine:
                self.backend = loaded_tts.get('f5_backend', self.backend)
                print(f"F5 already loaded (backend: {self.backend})")
                return engine

            self.backend = self._detect_backend()
            engine = self._load_mlx_engine() if self.backend == 'mlx' else self._load_cuda_engine()

            loaded_tts['f5'] = engine
            loaded_tts['f5_backend'] = self.backend
            print('F5-TTS loaded!')
            return engine
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f'F5.load_engine() error: {e}')

    def _load_mlx_engine(self):
        # Mac path — the Mac verifies/fixes the exact API (as it did for
        # voxtral._generate_mlx). f5_tts_mlx exposes F5TTS.from_pretrained.
        from f5_tts_mlx.cfm import F5TTS
        print(f"Loading F5 (MLX): {self.MLX_MODEL}")
        return F5TTS.from_pretrained(self.MLX_MODEL)

    def _load_cuda_engine(self):
        # SWivid f5_tts package; F5TTS() loads the default base model + vocoder.
        self._patch_torchaudio_to_soundfile()
        from f5_tts.api import F5TTS
        print("Loading F5 (CUDA): f5_tts default model")
        return F5TTS()

    @staticmethod
    def _patch_torchaudio_to_soundfile():
        """torchaudio 2.x routes load() through torchcodec, which is brittle on
        Windows (a fragile torch/torchcodec/ffmpeg DLL version match). f5_tts loads
        the reference clip via torchaudio.load — redirect it to libsndfile
        (soundfile), which reads wav/flac directly with no ffmpeg/torchcodec.
        NOTE: with an empty ref_text, f5_tts auto-transcribes via the transformers
        Whisper pipeline, which ALSO pulls torchcodec — so prefer providing
        ref_audio_text (resolved in _resolve_reference)."""
        import torchaudio
        import soundfile as sf
        import torch as _torch
        if getattr(torchaudio.load, '_bf_soundfile', False):
            return
        def _sf_load(uri, *args, **kwargs):
            data, sr = sf.read(str(uri), dtype='float32', always_2d=True)
            return _torch.from_numpy(data.T.copy()), sr
        _sf_load._bf_soundfile = True
        torchaudio.load = _sf_load

    # ── generation ───────────────────────────────────────────────────────────

    def _generate_mlx(self, text: str) -> np.ndarray:
        # Mac verifies/fixes. f5_tts_mlx F5TTS.generate synthesizes from ref audio
        # + ref text; returns audio samples (mlx/np array) at 24 kHz.
        result = self.engine.generate(
            text=text,
            ref_audio=self.ref_audio_path,
            ref_audio_text=self.ref_audio_text or None,
            steps=self.DEFAULT_STEPS,
        )
        audio = getattr(result, 'wave', result)
        return np.asarray(audio, dtype=np.float32)

    def _generate_cuda(self, text: str) -> np.ndarray:
        # f5_tts.api.F5TTS.infer → (wav ndarray, sample_rate, spectrogram).
        # ref_text='' makes f5_tts auto-transcribe the reference with Whisper.
        wav, sr, _ = self.engine.infer(
            ref_file=self.ref_audio_path,
            ref_text=self.ref_audio_text,
            gen_text=text,
            nfe_step=self.DEFAULT_NFE,
            remove_silence=False,
        )
        audio = np.asarray(wav, dtype=np.float32)
        if sr and sr != self.SAMPLE_RATE:
            t = torch.from_numpy(audio).unsqueeze(0)
            t = self._get_resampler(sr, self.SAMPLE_RATE)(t)
            audio = t.squeeze(0).cpu().numpy()
        return audio

    def convert(self, sentence_index: int, sentence: str) -> bool:
        try:
            if not self.engine:
                print("F5 TTS engine not loaded!")
                return False

            final_sentence_file = os.path.join(
                self.session['sentences_dir'],
                f'{sentence_index}.{default_audio_proc_format}'
            )

            sentence = sentence.strip()
            sentence = re.sub(r'\[(?:break|pause|music|sfx|silence)(?::[^\]]+)?\]', '',
                              sentence, flags=re.IGNORECASE).strip()

            if not sentence:
                silence = torch.zeros(1, int(self.params['samplerate'] * 0.1))
                torchaudio.save(final_sentence_file, silence, self.params['samplerate'],
                                format=default_audio_proc_format)
                return True

            try:
                audio_np = self._generate_mlx(sentence) if self.backend == 'mlx' \
                    else self._generate_cuda(sentence)

                if audio_np is None or len(audio_np) == 0:
                    print(f"F5 returned no audio for sentence {sentence_index}")
                    return False

                audio_tensor = torch.from_numpy(audio_np).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = trim_audio(audio_tensor, self.SAMPLE_RATE,
                                              silence_threshold=0.01, buffer_sec=0.20)
                    if len(audio_tensor) == 0:
                        audio_tensor = torch.zeros(int(self.SAMPLE_RATE * 0.1))
                    audio_tensor = audio_tensor.unsqueeze(0)

                max_val = audio_tensor.abs().max()
                if max_val > 1.0:
                    audio_tensor = audio_tensor / max_val * 0.95
                elif max_val == 0:
                    audio_tensor = torch.zeros(1, int(self.params['samplerate'] * 0.1))

                torchaudio.save(final_sentence_file, audio_tensor.cpu(),
                                self.params['samplerate'], format=default_audio_proc_format)
                del audio_tensor
                self._cleanup_memory()
                return os.path.exists(final_sentence_file)
            except Exception as gen_error:
                import traceback
                traceback.print_exc()
                print(f"F5 generation error for sentence {sentence_index}: {gen_error}")
                return False
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'F5.convert() error: {e}')
            return False

    def create_vtt(self, all_sentences: list) -> bool:
        audio_dir = self.session['sentences_dir']
        vtt_path = os.path.join(
            self.session['process_dir'],
            Path(self.session['final_name']).stem + '.vtt'
        )
        return bool(self._build_vtt_file(all_sentences, audio_dir, vtt_path))

    # ── cleanup ──────────────────────────────────────────────────────────────

    def cleanup(self):
        try:
            if getattr(self, 'engine', None) is not None:
                del self.engine
                self.engine = None
            self._cleanup_memory()
            print("[F5] Cleanup complete")
        except Exception as e:
            print(f"[F5] Cleanup warning: {e}")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

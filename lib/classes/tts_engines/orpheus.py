from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
import platform
import sys

class Orpheus(TTSUtils, TTSRegistry, name='orpheus'):
    """
    Orpheus TTS engine - SOTA open-source TTS built on Llama-3b backbone.
    Excellent prosody and naturalness, ideal for audiobooks.

    Supports two backends:
    - vLLM (preferred, fast) - requires CUDA on Linux/Windows
    - Transformers (fallback) - works on all platforms including Mac MPS

    Voices: tara, leah, jess, leo, dan, mia, zac, zoe
    Emotion tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>
    """

    # Valid Orpheus voices (in order of conversational realism per docs)
    VALID_VOICES = {'tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe'}
    DEFAULT_VOICE = 'tara'

    # Model configuration
    MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"
    SAMPLE_RATE = 24000

    def __init__(self, session: DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.resampler_cache = {}
            self.audio_segments = []
            self.backend = None  # 'vllm' or 'transformers'
            self.snac_model = None
            self.tokenizer = None

            # Try to load presets, but don't fail if they're missing
            try:
                self.models = load_engine_presets(self.session['tts_engine'])
            except Exception:
                self.models = {}

            self.params = {}
            self.params['samplerate'] = self.SAMPLE_RATE

            # Get voice from session or preset
            voice = self.session.get('fine_tuned', self.DEFAULT_VOICE)

            # Handle preset lookups
            if voice in self.models:
                voice = self.models[voice].get('voice', voice)

            # Validate voice
            if voice not in self.VALID_VOICES:
                print(f"Warning: Unknown Orpheus voice '{voice}', defaulting to '{self.DEFAULT_VOICE}'")
                voice = self.DEFAULT_VOICE

            self.voice = voice
            self.engine = None
            self.engine = self.load_engine()

        except Exception as e:
            error = f'Orpheus.__init__() error: {e}'
            raise ValueError(error)

    def _detect_backend(self) -> str:
        """Detect best available backend for this platform."""
        is_mac = platform.system() == 'Darwin'
        has_cuda = torch.cuda.is_available()

        # Try vLLM first (best performance on CUDA)
        if has_cuda and not is_mac:
            try:
                from vllm import LLM
                print("Orpheus: Using vLLM backend (CUDA detected)")
                return 'vllm'
            except ImportError:
                print("Orpheus: vLLM not available, trying transformers...")

        # Fall back to transformers (works on all platforms)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            backend_device = "MPS" if is_mac else ("CUDA" if has_cuda else "CPU")
            print(f"Orpheus: Using transformers backend ({backend_device})")
            return 'transformers'
        except ImportError:
            raise ImportError(
                "Neither vLLM nor transformers available. "
                "Install with: pip install transformers"
            )

    def _load_snac(self):
        """Load the SNAC audio decoder."""
        if self.snac_model is not None:
            return self.snac_model

        try:
            from snac import SNAC
            print("Loading SNAC audio decoder...")

            # Determine device
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
            self.snac_model.eval()
            print(f"SNAC loaded on {device}")
            return self.snac_model
        except Exception as e:
            raise ValueError(f"Failed to load SNAC decoder: {e}")

    def _load_vllm_engine(self):
        """Load model using vLLM backend."""
        from vllm import LLM, SamplingParams

        print(f"Loading Orpheus model with vLLM: {self.MODEL_NAME}")
        engine = LLM(
            model=self.MODEL_NAME,
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.8
        )
        return engine

    def _load_transformers_engine(self):
        """Load model using transformers backend."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Orpheus model with transformers: {self.MODEL_NAME}")

        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
            dtype = torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            dtype = torch.float16
        else:
            device = 'cpu'
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=dtype,
            device_map=device
        )
        model.eval()

        return model

    def load_engine(self) -> Any:
        try:
            msg = f"Loading Orpheus TTS with voice '{self.voice}'..."
            print(msg)
            self._cleanup_memory()

            # Check if already loaded
            engine_key = 'orpheus'
            engine = loaded_tts.get(engine_key, False)
            if engine:
                self.backend = loaded_tts.get('orpheus_backend', 'transformers')
                self.snac_model = loaded_tts.get('orpheus_snac', None)
                self.tokenizer = loaded_tts.get('orpheus_tokenizer', None)
                print(f"Orpheus already loaded (backend: {self.backend})")
                return engine

            # Detect and load appropriate backend
            self.backend = self._detect_backend()

            if self.backend == 'vllm':
                engine = self._load_vllm_engine()
            else:
                engine = self._load_transformers_engine()

            # Load SNAC decoder
            self._load_snac()

            # Cache everything
            loaded_tts[engine_key] = engine
            loaded_tts['orpheus_backend'] = self.backend
            loaded_tts['orpheus_snac'] = self.snac_model
            loaded_tts['orpheus_tokenizer'] = self.tokenizer

            print('Orpheus TTS Loaded!')
            return engine

        except Exception as e:
            error = f'Orpheus.load_engine() error: {e}'
            import traceback
            traceback.print_exc()
            raise ValueError(error)

    def _generate_tokens_vllm(self, prompt: str, max_tokens: int = 2048) -> list:
        """Generate audio tokens using vLLM backend."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            stop_token_ids=[128258]  # End of audio token
        )

        outputs = self.engine.generate([prompt], sampling_params)
        tokens = outputs[0].outputs[0].token_ids
        return list(tokens)

    def _generate_tokens_transformers(self, prompt: str, max_tokens: int = 2048) -> list:
        """Generate audio tokens using transformers backend."""
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.engine.device)

        # Generate
        with torch.no_grad():
            outputs = self.engine.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=128258  # End of audio token
            )

        # Extract generated tokens (excluding prompt)
        generated = outputs[0][input_ids.shape[1]:].tolist()
        return generated

    def _tokens_to_audio(self, tokens: list) -> np.ndarray:
        """Convert Orpheus tokens to audio using SNAC decoder."""
        if not tokens:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

        # Filter to valid audio tokens (128266 to 128266+4096*7)
        audio_tokens = []
        for t in tokens:
            if 128266 <= t < 128266 + 4096 * 7:
                audio_tokens.append(t - 128266)

        if not audio_tokens:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

        # Reshape tokens for SNAC (7 codebooks)
        # SNAC expects tokens in groups of 7
        n_frames = len(audio_tokens) // 7
        if n_frames == 0:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

        audio_tokens = audio_tokens[:n_frames * 7]

        # Reshape to (7, n_frames) for SNAC
        codes = torch.tensor(audio_tokens).reshape(n_frames, 7).T.unsqueeze(0)
        codes = codes.to(self.snac_model.device)

        # Apply modulo to get actual codebook indices
        for i in range(7):
            codes[0, i] = codes[0, i] % 4096

        # Decode with SNAC
        with torch.no_grad():
            audio = self.snac_model.decode(codes)

        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()
        return audio_np

    def convert(self, sentence_index: int, sentence: str) -> bool:
        try:
            if not self.engine:
                error = "Orpheus TTS engine not loaded!"
                print(error)
                return False

            final_sentence_file = os.path.join(
                self.session['sentences_dir'],
                f'{sentence_index}.{default_audio_proc_format}'
            )

            # Clean up the sentence for TTS
            sentence = sentence.strip()
            if not sentence:
                # Create a tiny silent audio file for empty sentences
                silence = torch.zeros(1, int(self.params['samplerate'] * 0.1))
                torchaudio.save(
                    final_sentence_file,
                    silence,
                    self.params['samplerate'],
                    format=default_audio_proc_format
                )
                return True

            # Format prompt with voice name and special tokens
            # Orpheus uses a specific format: <|audio_start|>voice: text<|audio_end|>
            prompt = f"<|audio_start|>{self.voice}: {sentence}<|audio_end|>"

            try:
                # Generate audio tokens
                if self.backend == 'vllm':
                    tokens = self._generate_tokens_vllm(prompt)
                else:
                    tokens = self._generate_tokens_transformers(prompt)

                # Convert tokens to audio
                audio_np = self._tokens_to_audio(tokens)

                if audio_np is not None and len(audio_np) > 0:
                    # Convert to tensor format for saving
                    audio_tensor = torch.from_numpy(audio_np).float()

                    # Ensure proper shape (1, samples) for torchaudio
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)

                    # Normalize audio to prevent clipping
                    max_val = audio_tensor.abs().max()
                    if max_val > 0:
                        if max_val > 1.0:
                            audio_tensor = audio_tensor / max_val * 0.95
                    else:
                        # Audio is all zeros, create minimal silence
                        audio_tensor = torch.zeros(1, int(self.params['samplerate'] * 0.1))

                    # Save the audio file
                    torchaudio.save(
                        final_sentence_file,
                        audio_tensor.cpu(),
                        self.params['samplerate'],
                        format=default_audio_proc_format
                    )

                    # Cleanup
                    del audio_tensor
                    self._cleanup_memory()

                    if os.path.exists(final_sentence_file):
                        return True
                    else:
                        error = f"Failed to create {final_sentence_file}"
                        print(error)
                        return False
                else:
                    error = f"Orpheus returned no audio data for sentence {sentence_index}"
                    print(error)
                    return False

            except Exception as gen_error:
                error = f"Orpheus generation error for sentence {sentence_index}: {gen_error}"
                print(error)
                import traceback
                traceback.print_exc()
                return False

        except Exception as e:
            error = f'Orpheus.convert() error: {e}'
            print(error)
            import traceback
            traceback.print_exc()
            return False

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

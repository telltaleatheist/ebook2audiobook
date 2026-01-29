from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.classes.tts_engines.common.audio import trim_audio
import platform
import sys
import os
import re
import atexit
import weakref

# Track active Orpheus instances for cleanup on exit
_active_instances = weakref.WeakSet()

# Required for vLLM on Windows
# See: https://github.com/SystemPanic/vllm-windows
if platform.system() == 'Windows':
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

    # Valid Orpheus voices (leah has best quality, tara has echo artifacts)
    VALID_VOICES = {'tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe'}
    DEFAULT_VOICE = 'leah'

    # Model configuration
    # MLX model (for Mac): mlx-community/orpheus-3b-0.1-ft-bf16
    # Transformers/vLLM model: unsloth/orpheus-3b-0.1-ft
    MLX_MODEL = "mlx-community/orpheus-3b-0.1-ft-bf16"
    TRANSFORMERS_MODEL = "unsloth/orpheus-3b-0.1-ft"
    SAMPLE_RATE = 24000

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

            # Get voice from session or preset
            voice = self.session.get('fine_tuned', self.DEFAULT_VOICE)
            print(f"[ORPHEUS] Session fine_tuned value: '{voice}'")

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
        from mlx_audio.tts.utils import load_model

        print(f"Loading Orpheus model with MLX: {self.MLX_MODEL}")
        model = load_model(self.MLX_MODEL)
        self._device = 'mlx'  # MLX manages its own device
        print("Orpheus MLX model loaded!")
        return model

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

        engine = LLM(
            model=self.TRANSFORMERS_MODEL,
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
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

            # Check if already loaded
            engine_key = 'orpheus'
            engine = loaded_tts.get(engine_key, False)
            if engine:
                self.backend = loaded_tts.get('orpheus_backend', 'transformers')
                self.snac_model = loaded_tts.get('orpheus_snac', None)
                self.tokenizer = loaded_tts.get('orpheus_tokenizer', None)
                self._device = loaded_tts.get('orpheus_device', 'cpu')
                self.mlx_model = loaded_tts.get('orpheus_mlx_model', None)
                print(f"Orpheus already loaded (backend: {self.backend})")
                return engine

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

            print('Orpheus TTS Loaded!')
            return engine

        except Exception as e:
            error = f'Orpheus.load_engine() error: {e}'
            import traceback
            traceback.print_exc()
            raise ValueError(error)

    def _generate_mlx(self, text: str, max_tokens: int = 2048) -> np.ndarray:
        """Generate audio using MLX backend.

        Note: 125 tokens ≈ 10 seconds of audio, so 2048 tokens ≈ 163 seconds max.
        """
        audio_data = None
        # Match vLLM/transformers sampling params - repetition_penalty prevents
        # repeated audio patterns that can sound like echo/reverb
        print(f"[ORPHEUS] Generating with voice='{self.voice}' for: {text[:50]}...")
        for result in self.mlx_model.generate(
            text,
            voice=self.voice,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=max_tokens
        ):
            audio_data = result.audio

        if audio_data is None:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

        # MLX returns audio as numpy array or MLX array
        if hasattr(audio_data, 'tolist'):
            # Convert MLX array to numpy
            import numpy as np
            audio_np = np.array(audio_data, dtype=np.float32)
        else:
            audio_np = audio_data

        return audio_np

    def _format_prompt_with_special_tokens(self, text: str) -> str:
        """Format prompt with Orpheus special tokens for audio generation.

        Orpheus requires specific tokens to trigger audio generation:
        - Start token: 128259 (<custom_token_3>)
        - End tokens: [128009, 128260, 128261, 128257]
        """
        import torch

        # Format: "voice: text"
        adapted_prompt = f"{self.voice}: {text}"
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")

        # Add special tokens
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)

        # Decode back to string for vLLM
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string

    def _generate_tokens_vllm(self, prompt: str, max_tokens: int = 2048) -> list:
        """Generate audio tokens using vLLM backend."""
        from vllm import SamplingParams

        # Format prompt with special tokens
        formatted_prompt = self._format_prompt_with_special_tokens(prompt)

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.1,
            max_tokens=max_tokens,
            stop_token_ids=[self.END_OF_AUDIO_TOKEN]
        )

        outputs = self.engine.generate([formatted_prompt], sampling_params)
        tokens = list(outputs[0].outputs[0].token_ids)

        # Truncate at end-of-audio token if present
        if self.END_OF_AUDIO_TOKEN in tokens:
            end_idx = tokens.index(self.END_OF_AUDIO_TOKEN)
            tokens = tokens[:end_idx]

        return tokens

    def _generate_tokens_transformers(self, prompt: str, max_tokens: int = 2048) -> list:
        """Generate audio tokens using transformers backend."""
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.engine.device)

        # Generate with repetition penalty to avoid garbage loops
        with torch.no_grad():
            outputs = self.engine.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
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
        """Convert Orpheus tokens to audio using SNAC decoder."""
        if not tokens:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

        # Redistribute tokens to SNAC's 3-layer format
        codes = self._redistribute_codes(tokens)
        if codes is None:
            return np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)

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

            # Strip SML tags that Orpheus doesn't understand (it has its own emotion tags)
            # E2a uses [break], [pause], [music], [sfx], [silence] etc.
            sentence = re.sub(r'\[(?:break|pause|music|sfx|silence)(?::[^\]]+)?\]', '', sentence, flags=re.IGNORECASE)
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

            try:
                # Generate audio based on backend
                if self.backend == 'mlx':
                    # MLX handles everything internally
                    audio_np = self._generate_mlx(sentence)
                else:
                    # vLLM and transformers use token generation + SNAC decoding
                    if self.backend == 'vllm':
                        # vLLM needs special token formatting
                        tokens = self._generate_tokens_vllm(sentence)
                    else:
                        # Transformers uses simple prompt format
                        prompt = f"{self.voice}: {sentence}"
                        tokens = self._generate_tokens_transformers(prompt)

                    audio_np = self._tokens_to_audio(tokens)

                if audio_np is not None and len(audio_np) > 0:
                    # Convert to tensor format for saving
                    audio_tensor = torch.from_numpy(audio_np).float()

                    # Trim trailing silence (Orpheus tends to add long pauses at end)
                    # Use moderate threshold to catch obvious silence
                    # Keep 200ms buffer for natural inter-sentence pauses
                    if audio_tensor.dim() == 1:
                        audio_tensor = trim_audio(audio_tensor, self.SAMPLE_RATE, silence_threshold=0.01, buffer_sec=0.20)
                        if len(audio_tensor) == 0:
                            # If trimming removed everything, use minimal silence
                            audio_tensor = torch.zeros(int(self.SAMPLE_RATE * 0.1))

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

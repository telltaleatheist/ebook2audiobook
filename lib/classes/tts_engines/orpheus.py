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

    # Valid Orpheus voices (leah has best quality, tara has echo artifacts)
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
        engine = LLM(
            model=self.TRANSFORMERS_MODEL,
            dtype="float16",
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
        (_convert_sml). Returns (gap_seconds, position):

          [pause] / [pause:X]  -> SECTION gap: explicit X, else 1.0-1.6s. Marks
                                  blank-line / heading / <div> boundaries.
          [break] / [silence]  -> PARAGRAPH gap: 0.5-0.7s. Marks <p>/<br> ends.
          (no token)           -> SENTENCE gap: ~0.30s, so Orpheus breathes
                                  instead of rushing.

        Why three tiers and not the old binary [break]==[pause]: the standard
        treats [break] as a SHORT gap and [pause] as the LONG one. Lumping them
        meant every <p>/<br>/<span> [break] got a long 0.6-0.9s gap — long pauses
        in odd places. Now [break] is a modest paragraph gap and only [pause]
        (rare, real section breaks) is long.

        position is 'lead' when the boundary token sits at the START of the
        sentence (the gap belongs BEFORE this clip — e.g. the first sentence of a
        new paragraph) or 'trail' otherwise (gap AFTER this clip). Checked on the
        RAW sentence, before _clean_sentence_for_tts strips the tokens.
        Env overrides (a fixed value, 0 disables): ORPHEUS_SENTENCE_GAP,
        ORPHEUS_PARAGRAPH_GAP, ORPHEUS_SECTION_GAP.
        """
        raw = (sentence or '').strip()
        lowered = raw.lower()

        def _env(name):
            v = os.environ.get(name)
            return float(v) if v is not None else None

        # Section pause ([pause] / [pause:X]) — the strong, long boundary.
        m = re.search(r'\[pause(?::([0-9.]+))?\]', raw, flags=re.IGNORECASE)
        if m:
            override = _env('ORPHEUS_SECTION_GAP')
            if override is not None:
                gap = override
            elif m.group(1):
                gap = float(m.group(1))               # honor an explicit [pause:1.4]
            else:
                gap = int(np.random.uniform(1.0, 1.6) * 100) / 100
            pos = 'lead' if lowered.startswith('[pause') else 'trail'
            return gap, pos

        # Paragraph break ([break] / [silence]) — modest gap.
        m = re.search(r'\[(?:break|silence)(?::[^\]]+)?\]', raw, flags=re.IGNORECASE)
        if m:
            override = _env('ORPHEUS_PARAGRAPH_GAP')
            gap = override if override is not None else int(np.random.uniform(0.5, 0.7) * 100) / 100
            pos = 'lead' if re.match(r'\[(?:break|silence)', lowered) else 'trail'
            return gap, pos

        # Plain sentence end — short breathing gap.
        override = _env('ORPHEUS_SENTENCE_GAP')
        gap = override if override is not None else int(np.random.uniform(0.25, 0.35) * 100) / 100
        return gap, 'trail'

    def _write_silence(self, sentence_index: int) -> bool:
        """Write a tiny silent clip for an empty sentence."""
        silence = torch.zeros(1, int(self.params['samplerate'] * 0.1))
        torchaudio.save(self._sentence_file(sentence_index), silence,
                        self.params['samplerate'], format=default_audio_proc_format)
        return True

    def _save_audio(self, sentence_index: int, audio_np, gap_sec: float = 0.3, gap_pos: str = 'trail') -> bool:
        """Trim trailing silence, normalize, add the inter-clip pause, and write a
        decoded waveform to the sentence file. Shared by convert() and
        convert_batch(). gap_sec/gap_pos come from _classify_gap(): a 'trail' gap
        is appended after the speech, a 'lead' gap is prepended before it (so a
        boundary at the start of a new paragraph lands at the right place)."""
        if audio_np is None or len(audio_np) == 0:
            print(f"Orpheus returned no audio data for sentence {sentence_index}")
            return False
        final_sentence_file = self._sentence_file(sentence_index)
        audio_tensor = torch.from_numpy(audio_np).float()
        # Trim trailing silence (Orpheus tends to add long end pauses); keep 200ms buffer.
        if audio_tensor.dim() == 1:
            audio_tensor = trim_audio(audio_tensor, self.SAMPLE_RATE, silence_threshold=0.01, buffer_sec=0.20)
            if len(audio_tensor) == 0:
                audio_tensor = torch.zeros(int(self.SAMPLE_RATE * 0.1))
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        # Normalize to prevent clipping
        max_val = audio_tensor.abs().max()
        if max_val > 0:
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val * 0.95
        else:
            audio_tensor = torch.zeros(1, int(self.params['samplerate'] * 0.1))
        # Inter-clip silence (tier + duration decided by _classify_gap, matching
        # e2a's standard SML semantics). 'lead' prepends it before the speech so a
        # boundary opening a new paragraph isn't placed one sentence too late.
        if gap_sec and gap_sec > 0:
            audio_tensor = audio_tensor.cpu()
            pad = torch.zeros(1, int(self.params['samplerate'] * gap_sec))
            audio_tensor = torch.cat([pad, audio_tensor], dim=1) if gap_pos == 'lead' \
                else torch.cat([audio_tensor, pad], dim=1)
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

            gap_sec, gap_pos = self._classify_gap(sentence)
            clean = self._clean_sentence_for_tts(sentence)
            if not clean:
                return self._write_silence(sentence_index)

            try:
                if self.backend == 'mlx':
                    audio_np = self._generate_mlx(clean)
                elif self.backend == 'vllm':
                    audio_np = self._tokens_to_audio(self._generate_tokens_vllm(clean))
                else:
                    audio_np = self._tokens_to_audio(
                        self._generate_tokens_transformers(f"{self.voice}: {clean}")
                    )
                ok = self._save_audio(sentence_index, audio_np, gap_sec, gap_pos)
                self._cleanup_memory()
                return ok
            except Exception as gen_error:
                print(f"Orpheus generation error for sentence {sentence_index}: {gen_error}")
                import traceback
                traceback.print_exc()
                return False

        except Exception as e:
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
            from vllm import SamplingParams

            results = {}
            gen = []  # (idx, formatted_prompt, (gap_sec, gap_pos)) for non-empty sentences
            for idx, sentence in items:
                gap = self._classify_gap(sentence)
                clean = self._clean_sentence_for_tts(sentence)
                if not clean:
                    results[idx] = self._write_silence(idx)
                else:
                    gen.append((idx, self._format_prompt_with_special_tokens(clean), gap))

            if gen:
                sampling_params = SamplingParams(
                    temperature=0.6, top_p=0.8, repetition_penalty=1.1,
                    max_tokens=2048, stop_token_ids=[self.END_OF_AUDIO_TOKEN]
                )
                prompts = [fp for _, fp, _ in gen]
                # use_tqdm=False: a per-call progress bar adds overhead and noise.
                try:
                    outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False)
                except TypeError:
                    outputs = self.engine.generate(prompts, sampling_params)
                # vLLM returns outputs in the same order as prompts.
                for (idx, _, gap), out in zip(gen, outputs):
                    try:
                        tokens = list(out.outputs[0].token_ids)
                        if self.END_OF_AUDIO_TOKEN in tokens:
                            tokens = tokens[:tokens.index(self.END_OF_AUDIO_TOKEN)]
                        results[idx] = self._save_audio(idx, self._tokens_to_audio(tokens), gap[0], gap[1])
                    except Exception as decode_err:
                        print(f"Orpheus batch decode error for sentence {idx}: {decode_err}")
                        results[idx] = False

            self._cleanup_memory()
            return [results.get(idx, False) for idx, _ in items]
        except Exception as e:
            print(f'Orpheus.convert_batch() error: {e}')
            import traceback
            traceback.print_exc()
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

        Memory stays bounded: the bf16 weights (~6 GB) dominate, so the batched
        KV cache + activations add little — peak RSS is roughly flat across batch
        sizes (measured ~10 GB at B=16 on M1 Ultra, ~3.6x throughput vs per-item).
        Sampling matches the single-seq MLX path (_generate_mlx): temp 0.6,
        top_p 0.9, repetition_penalty 1.1.
        """
        try:
            import numpy as np
            import mlx.core as mx
            from mlx_lm.generate import BatchGenerator
            from mlx_lm.sample_utils import make_sampler, make_logits_processors
            from mlx_audio.tts.models.llama.llama import decode_audio_from_codes

            results = {}
            gen = []  # (idx, prompt_tokens, (gap_sec, gap_pos)) for non-empty sentences
            for idx, sentence in items:
                gap = self._classify_gap(sentence)
                clean = self._clean_sentence_for_tts(sentence)
                if not clean:
                    results[idx] = self._write_silence(idx)
                else:
                    # prepare_input_ids prepends "voice: " itself; pass voice, not a
                    # pre-formatted string. Single-string call returns a [1, T] array.
                    ptoks = self.mlx_model.prepare_input_ids(clean, self.voice)[0].tolist()
                    gen.append((idx, ptoks, gap))

            if gen:
                bg = BatchGenerator(
                    self.mlx_model,
                    max_tokens=2048,
                    stop_tokens={self.END_OF_AUDIO_TOKEN},
                    sampler=make_sampler(0.6, top_p=0.9),
                    logits_processors=make_logits_processors(None, 1.1, 20),
                    completion_batch_size=len(gen),
                    prefill_batch_size=len(gen),
                )
                uids = bg.insert([list(p) for _, p, _ in gen])
                out = {u: [] for u in uids}
                while responses := bg.next():
                    for r in responses:
                        if r.finish_reason != 'stop':  # stop token (128258) is dropped
                            out[r.uid].append(r.token)
                bg.close()
                # uids come back in insert order == gen order.
                for (idx, ptoks, gap), uid in zip(gen, uids):
                    try:
                        ids = mx.array([ptoks + out[uid]])
                        code_lists = self.mlx_model.parse_output(ids)
                        if code_lists and len(code_lists[0]) > 0:
                            audio = np.array(
                                decode_audio_from_codes(code_lists[0])[0], dtype=np.float32
                            )
                        else:
                            audio = np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32)
                        results[idx] = self._save_audio(idx, audio, gap[0], gap[1])
                    except Exception as decode_err:
                        print(f"Orpheus MLX batch decode error for sentence {idx}: {decode_err}")
                        results[idx] = False

            # Bound MLX's reclaimable scratch pool between chunks (mlx_audio's own
            # generate() does this per segment). Active memory is already flat —
            # the weights dominate — but this keeps the cache pool from sitting
            # large on a machine running other apps.
            mx.clear_cache()
            self._cleanup_memory()
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

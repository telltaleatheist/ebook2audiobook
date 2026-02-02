# ebook2audiobook - Claude Code Context

This is a fork of ebook2audiobook with custom modifications for BookForge integration.

## Fork Info

- **Original repo:** https://github.com/DrewThomasson/ebook2audiobook
- **Fork:** https://github.com/telltaleatheist/ebook2audiobook
- **Branch:** `bookforge`

## Orpheus TTS Engine

Orpheus is a SOTA open-source TTS engine added to this fork. It produces better prosody and naturalness than XTTS, ideal for audiobooks.

### Key Files

- `lib/classes/tts_engines/orpheus.py` - Main engine implementation
- `lib/classes/tts_engines/presets/orpheus_presets.py` - Voice presets
- `lib/conf_models.py` - Engine registration (includes `"ORPHEUS": "orpheus"`)

### Backend Priority

The Orpheus engine auto-detects the best backend:

1. **MLX** (Mac only) - Fastest on Apple Silicon (~1.4x realtime)
   - Model: `mlx-community/orpheus-3b-0.1-ft-bf16`
   - Install: `pip install mlx-audio`

2. **vLLM** (Windows/Linux with CUDA) - Fast batched inference
   - Model: `unsloth/orpheus-3b-0.1-ft`
   - Install: `pip install vllm`

3. **Transformers** (Fallback) - Works everywhere but SLOW on Mac (~27x realtime)
   - Model: `unsloth/orpheus-3b-0.1-ft`
   - Install: `pip install transformers snac`

### IMPORTANT: Single Worker Only

Unlike XTTS, Orpheus does NOT benefit from multiple workers:
- MLX uses unified memory - workers compete, no speedup
- vLLM has built-in batching - single instance is optimal

**Always run Orpheus with workers=1**

### Voices

8 voices: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`
- `tara` is most natural (default)
- All voices are American English

### Emotion Tags

Embed in text for expressiveness:
```
<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>
```

### Windows Setup (vLLM)

```bash
# Install vLLM (requires CUDA)
pip install vllm

# Install SNAC audio decoder
pip install snac

# Model downloads automatically on first run (~6GB)
```

### Mac Setup (MLX)

```bash
# Install mlx-audio (Apple Silicon only)
pip install mlx-audio

# Model downloads automatically on first run
```

### Usage via BookForge

1. In BookForge's Audiobook Producer, go to TTS Settings
2. Select "Orpheus 3B" as the TTS engine
3. Choose a voice (tara recommended)
4. Worker count is forced to 1 for Orpheus

### Direct CLI Usage

```bash
python app.py --tts_engine orpheus --fine_tuned tara input.epub
```

### Technical Notes

- Sample rate: 24000 Hz
- Audio format: Uses SNAC decoder to convert Orpheus tokens to audio
- Token format: Orpheus outputs 7 tokens per audio frame, redistributed to SNAC's 3-layer format
- End token: 128258 (must truncate output at this token)

### Prompt Format

Simple format: `{voice}: {text}`

Example: `tara: Hello, this is a test.`

Do NOT use special tokens like `<|audio_start|>` - the model will speak them literally.

## BookForge Extension

All BookForge-specific code is isolated in `bookforge_ext/` for easy upstream merging.

### Directory Structure

```
bookforge_ext/
├── __init__.py          # Extension entry point
├── hooks.py             # Hook registry for core.py integration
├── parallel/
│   ├── args.py          # CLI argument definitions
│   ├── handlers.py      # Mode dispatching (prep_only, worker_mode, assemble_only)
│   ├── session.py       # Session state management
│   └── worker_core.py   # Lightweight TTS worker
└── config/
    └── __init__.py      # Reserved for config overrides
```

### Worker Memory Management

Workers use `worker.py` (not `app.py`) to minimize memory:
- `worker.py` → `bookforge_ext/parallel/worker_core.py` → ~2.5GB per worker
- `app.py --headless --worker_mode` → ~25GB per worker (imports everything)

**Critical files for low-memory workers:**
1. `worker.py` must import from `bookforge_ext.parallel.worker_core`
2. `worker_core.py` must have `register_tts_engine()` to load only needed engine
3. `tts_manager.py` must have `import lib.classes.tts_engines` for TTSRegistry
4. `tts_engines/__init__.py` must import all engines for auto-registration

### Hook System

Extension integrates with core.py via hooks:

```python
# In lib/core.py (at end of file)
from bookforge_ext import hooks as bf_hooks
bf_hooks.register('get_context', lambda: context)

# In extension code
context = hooks.call('get_context')
```

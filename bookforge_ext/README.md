# BookForge Extension for ebook2audiobook

This extension provides parallel TTS processing capabilities for the BookForge audiobook production application.

## Overview

The BookForge extension isolates all BookForge-specific customizations from the upstream e2a codebase, making it easier to:

1. **Merge upstream changes** - Only ~25 lines touch upstream files
2. **Test independently** - Extension can be tested without full e2a stack
3. **Remove cleanly** - Extension can be deleted without breaking base e2a

## Architecture

```
ebook2audiobook-latest/
├── app.py                      # +15 lines: import + hook calls
├── worker.py                   # +2 lines: import from extension
├── lib/
│   └── core.py                 # +8 lines: context accessor hooks
│
└── bookforge_ext/              # All BookForge custom code
    ├── __init__.py             # Extension entry point
    ├── hooks.py                # Hook registry system
    ├── README.md               # This file
    ├── parallel/
    │   ├── __init__.py
    │   ├── args.py             # CLI argument definitions
    │   ├── handlers.py         # Mode dispatching
    │   ├── session.py          # Session state management (~600 lines)
    │   └── worker_core.py      # Lightweight TTS worker (~300 lines)
    └── config/
        └── __init__.py         # Reserved for config overrides
```

## Three-Phase Parallel TTS Architecture

### Phase 1: Preparation (`--prep_only`)

```bash
python app.py --prep_only --ebook book.epub --language eng
```

Parses the EPUB and creates session state for parallel workers:
- Creates unique session ID
- Extracts text into chapters and sentences
- Saves `session-state.json` with all data needed by workers
- Returns JSON with `session_id`, `total_sentences`, `total_chapters`

### Phase 2: Worker Processing (`--worker_mode` or `worker.py`)

```bash
# Via app.py (loads full dependencies, ~25GB memory)
python app.py --worker_mode --session SESSION_ID --sentence_start 0 --sentence_end 999

# Via worker.py (lightweight, ~8GB memory) - RECOMMENDED
python worker.py --session SESSION_ID --sentence_start 0 --sentence_end 999
```

Runs TTS for assigned sentence range:
- Loads session state from Phase 1
- Initializes TTS engine (XTTS, Orpheus, etc.)
- Converts sentences to audio files
- Skips already-completed sentences (resume capability)

### Phase 3: Assembly (`--assemble_only`)

```bash
python app.py --assemble_only --session SESSION_ID
```

Combines sentence audio into final audiobook:
- Merges sentences into chapter files
- Builds VTT subtitle file
- Creates final M4B/MP3 with chapter markers
- Adds metadata (title, author, cover)

## BookForge Integration

BookForge (Electron app) coordinates the three phases:

1. **Prep**: Call `--prep_only` to get session ID and sentence count
2. **Distribute**: Spawn multiple `worker.py` processes for different sentence ranges
3. **Monitor**: Poll `sentences_dir/` for completed files
4. **Assemble**: Call `--assemble_only` when all sentences complete

## Session State

Session data is stored in `tmp/ebook-{session_id}/{hash}/session-state.json`:

```json
{
  "version": 2,
  "session_id": "uuid-string",
  "total_sentences": 5000,
  "total_chapters": 42,
  "chapter_sentences": [["sentence 1", "sentence 2", ...], ...],
  "language": "eng",
  "tts_engine": "xtts",
  "voice": "/path/to/voice.wav",
  "metadata": {"title": "Book Title", "creator": "Author Name"}
}
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--prep_only` | Phase 1: Parse EPUB, create session state |
| `--worker_mode` | Phase 2: Run TTS for assigned range |
| `--assemble_only` | Phase 3: Combine audio into final audiobook |
| `--sentence_start` | First sentence index (0-indexed) |
| `--sentence_end` | Last sentence index (0-indexed) |
| `--chapter_start` | First chapter (1-indexed) |
| `--chapter_end` | Last chapter (1-indexed) |
| `--resume_session` | Resume a partially completed session |
| `--list_sessions` | List all resumable sessions |
| `--chapters` | Which chapters to assemble: "1-5", "1,3,5", or "auto" |
| `--no_split` | Disable splitting output into parts |
| `--skip_deps` | Skip dependency checks |

## Hook System

The extension uses a simple hook system to access core.py internals:

```python
# In lib/core.py
from bookforge_ext import hooks
hooks.register('get_context', lambda: context)

# In extension
context = hooks.call('get_context')
```

## Development

To test the extension standalone:

```python
from bookforge_ext import hooks, parallel

# Check hooks
print(hooks.is_registered('get_context'))

# Check parallel options
print(parallel.PARALLEL_OPTIONS)
```

## Removing the Extension

To remove BookForge extensions and revert to upstream:

1. Delete `bookforge_ext/` directory
2. Remove try/import blocks from `app.py` and `lib/core.py`
3. The base e2a will continue to work normally

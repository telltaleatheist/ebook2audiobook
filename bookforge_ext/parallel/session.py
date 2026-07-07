"""
Session Management for Parallel TTS Processing

This module contains all session state management functions for the
three-phase parallel TTS architecture:
  Phase 1: prep_ebook_info() - Parse EPUB, create session state
  Phase 2: worker_only() - Run TTS for assigned sentence range
  Phase 3: assemble_audiobook() - Combine audio into final M4B

Functions are extracted from lib/core.py to isolate BookForge customizations.
"""

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

# A "sentence" in the parallel pipeline is a GENERATION CHUNK, which for Orpheus/Voxtral
# packs 2-3 real sentences per chunk (see lib/core.py get_sentences PASS 5). To report a
# true sentences/min rate for analytics — while the pipeline keeps scheduling by chunk —
# count the real sentence boundaries inside each chunk: terminal .!?… (optionally wrapped
# by closing quotes/brackets) followed by whitespace or end-of-string. A chunk with no
# terminal punctuation (e.g. a heading) still counts as one sentence.
_SENTENCE_END_RE = re.compile(r'[.!?…]+["\'”’)\]]*(?:\s|$)')


def count_real_sentences(chunk: str) -> int:
    return max(1, len(_SENTENCE_END_RE.findall(chunk or '')))

from lib.conf import (
    tmp_dir,
    models_dir,
    voices_dir,
    audiobooks_cli_dir,
    default_device,
    default_output_format,
    default_output_split,
    default_output_split_hours,
    default_audio_proc_format,
    NATIVE,
)
from lib.conf_models import default_fine_tuned, TTS_ENGINES
from lib.conf_lang import default_language_code


# =============================================================================
# Session State I/O
# =============================================================================

def save_session_state(session_id: str, args: dict, prep_result: dict, core_module) -> bool:
    """
    Save session state to session-state.json for resume capability.
    """
    try:
        context = core_module.context
        session = context.get_session(session_id)
        state = {
            'version': 2,
            'session_id': prep_result['session_id'],
            'epub_path': session['ebook'],
            'source_epub_path': args.get('ebook'),
            'epub_content_hash': hashlib.md5(session['ebook'].encode()).hexdigest(),
            'total_sentences': prep_result['total_sentences'],
            'total_chapters': prep_result['total_chapters'],
            'chapters': prep_result['chapters'],
            'chapter_sentences': prep_result.get('chapter_sentences', []),
            'language': session.get('language', default_language_code),
            'language_iso1': session.get('language_iso1', 'en'),
            'voice': session.get('voice'),
            'fine_tuned': session.get('fine_tuned'),
            'custom_model': session.get('custom_model'),
            'custom_model_dir': session.get('custom_model_dir'),
            'orpheus_model_dir': session.get('orpheus_model_dir'),
            'tts_engine': session.get('tts_engine'),
            'device': args.get('device', default_device),
            'output_format': args.get('output_format', default_output_format),
            'audiobooks_dir': args.get('audiobooks_dir', audiobooks_cli_dir),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'status': 'prepared',
            'metadata': prep_result.get('metadata', {}),
            'session_dir': session['session_dir'],
            'process_dir': session['process_dir'],
            'chapters_dir': session['chapters_dir'],
            'chapters_dir_sentences': session['sentences_dir'],
            'epub_path_internal': session.get('epub_path'),
            'filename_noext': session.get('filename_noext'),
            'cover': session.get('cover'),
            'final_name': session.get('final_name'),
            'chapter_titles': session.get('chapter_titles', []),
            # BookForge metadata (title, author, year from BFP project)
            'bookforge_metadata': args.get('bookforge_metadata', {}),
        }

        state_path = os.path.join(session['process_dir'], 'session-state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

        print(f'Session state saved to {state_path}')
        return True
    except Exception as e:
        print(f'Warning: Failed to save session state: {e}')
        return False


def load_session_state(session_dir: str) -> dict | None:
    """
    Load session state from session-state.json.
    """
    try:
        process_dirs = [d for d in os.listdir(session_dir)
                       if os.path.isdir(os.path.join(session_dir, d))]

        for process_dir_name in process_dirs:
            state_path = os.path.join(session_dir, process_dir_name, 'session-state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                state['process_dir'] = os.path.join(session_dir, process_dir_name)
                state['session_dir'] = session_dir
                return state
        return None
    except Exception as e:
        print(f'Failed to load session state: {e}')
        return None


# =============================================================================
# Sentence Scanning & Progress Tracking
# =============================================================================

def scan_completed_sentences(sentences_dir: str, total_sentences: int, min_file_size: int = 1024) -> dict:
    """
    Scan the sentences directory to find completed audio files.
    """
    completed = []
    missing = []

    for i in range(total_sentences):
        found = False
        for ext in ['flac', 'wav', 'mp3']:
            file_path = os.path.join(sentences_dir, f'{i}.{ext}')
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size >= min_file_size:
                    completed.append(i)
                    found = True
                    break
        if not found:
            missing.append(i)

    return {
        'completed': completed,
        'missing': missing,
        'completed_count': len(completed),
        'missing_count': len(missing),
        'progress_percent': round(len(completed) / total_sentences * 100, 1) if total_sentences > 0 else 0
    }


def calculate_missing_ranges(missing_indices: list) -> list:
    """
    Convert list of missing sentence indices into contiguous ranges.
    """
    if not missing_indices:
        return []

    ranges = []
    sorted_indices = sorted(missing_indices)
    range_start = sorted_indices[0]
    range_end = sorted_indices[0]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == range_end + 1:
            range_end = sorted_indices[i]
        else:
            ranges.append({'start': range_start, 'end': range_end, 'count': range_end - range_start + 1})
            range_start = sorted_indices[i]
            range_end = sorted_indices[i]

    ranges.append({'start': range_start, 'end': range_end, 'count': range_end - range_start + 1})
    return ranges


# =============================================================================
# Session Listing & Resume
# =============================================================================

def list_resumable_sessions() -> list:
    """
    List all sessions that have session-state.json and incomplete sentences.
    """
    resumable = []

    if not os.path.exists(tmp_dir):
        return resumable

    for session_name in os.listdir(tmp_dir):
        if not session_name.startswith('ebook-'):
            continue

        session_dir = os.path.join(tmp_dir, session_name)
        if not os.path.isdir(session_dir):
            continue

        state = load_session_state(session_dir)
        if not state:
            continue

        sentences_dir = os.path.join(state['process_dir'], 'chapters', 'sentences')
        if not os.path.exists(sentences_dir):
            continue

        scan_result = scan_completed_sentences(sentences_dir, state['total_sentences'])

        if scan_result['missing_count'] > 0:
            resumable.append({
                'session_id': state['session_id'],
                'session_dir': session_dir,
                'title': state.get('metadata', {}).get('title', 'Unknown'),
                'total_sentences': state['total_sentences'],
                'completed_sentences': scan_result['completed_count'],
                'missing_sentences': scan_result['missing_count'],
                'progress_percent': scan_result['progress_percent'],
                'created_at': state.get('created_at'),
                'language': state.get('language'),
                'voice': state.get('voice')
            })

    return resumable


def check_resume_compatibility(state: dict, args: dict) -> dict:
    """
    Check if current settings are compatible with saved session state.
    """
    warnings = []

    saved_voice = state.get('voice')
    current_voice = args.get('voice')
    if saved_voice and current_voice and saved_voice != current_voice:
        warnings.append(f"Voice changed from '{saved_voice}' to '{current_voice}'")

    saved_engine = state.get('tts_engine')
    current_engine = args.get('tts_engine')
    if saved_engine and current_engine and saved_engine != current_engine:
        warnings.append(f"TTS engine changed from '{saved_engine}' to '{current_engine}'")

    return {'compatible': True, 'warnings': warnings}


def resume_session(args: dict) -> dict:
    """
    Resume a partially completed TTS session.
    """
    try:
        session_path = args.get('resume_session')
        if not session_path:
            return {'success': False, 'error': 'No session path provided'}

        if not os.path.isabs(session_path):
            if session_path.startswith('ebook-'):
                session_dir = os.path.join(tmp_dir, session_path)
            else:
                session_dir = os.path.join(tmp_dir, f'ebook-{session_path}')
        else:
            session_dir = session_path

        if not os.path.exists(session_dir):
            return {'success': False, 'error': f'Session directory not found: {session_dir}'}

        state = load_session_state(session_dir)
        if not state:
            return {'success': False, 'error': 'No session-state.json found'}

        compat = check_resume_compatibility(state, args)
        if compat['warnings']:
            for warning in compat['warnings']:
                print(f'Warning: {warning}')

        sentences_dir = os.path.join(state['process_dir'], 'chapters', 'sentences')
        if not os.path.exists(sentences_dir):
            return {'success': False, 'error': f'Sentences directory not found: {sentences_dir}'}

        scan_result = scan_completed_sentences(sentences_dir, state['total_sentences'])

        if scan_result['missing_count'] == 0:
            return {
                'success': True,
                'complete': True,
                'message': 'All sentences already complete - ready for assembly',
                'session_id': state['session_id'],
                'session_dir': session_dir,
                'process_dir': state['process_dir']
            }

        missing_ranges = calculate_missing_ranges(scan_result['missing'])

        return {
            'success': True,
            'complete': False,
            'session_id': state['session_id'],
            'session_dir': session_dir,
            'process_dir': state['process_dir'],
            'chapters_dir': os.path.join(state['process_dir'], 'chapters'),
            'chapters_dir_sentences': sentences_dir,
            'total_sentences': state['total_sentences'],
            'total_chapters': state['total_chapters'],
            'completed_sentences': scan_result['completed_count'],
            'missing_sentences': scan_result['missing_count'],
            'missing_indices': scan_result['missing'],
            'missing_ranges': missing_ranges,
            'progress_percent': scan_result['progress_percent'],
            'chapters': state.get('chapters', []),
            'metadata': state.get('metadata', {}),
            'warnings': compat['warnings']
        }

    except Exception as e:
        print(f'resume_session() Exception: {e}')
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# Phase 1: Preparation
# =============================================================================

def prep_ebook_info(args: dict, core_module) -> dict | None:
    """
    Prepare EPUB information for parallel worker coordination.
    Returns a dict with session_id, total_sentences, total_chapters, and session_dir.
    This is used by BookForgeApp to coordinate parallel TTS workers.
    """
    try:
        from iso639 import Lang
        from ebooklib import epub

        context = core_module.context
        context_tracker = core_module.context_tracker

        # Import helper functions from core
        from lib.core import (
            prepare_dirs,
            convert2epub,
            get_cover,
            get_chapters,
            get_sanitized,
            get_compatible_tts_engines,
            VRAMDetector,
        )

        # Create session
        session_id = args.get('session') or str(uuid.uuid4())
        session = context.set_session(session_id)
        context_tracker.start_session(session_id)

        # Basic setup
        session['script_mode'] = args.get('script_mode', NATIVE)
        session['is_gui_process'] = args.get('is_gui_process', False)
        session['ebook'] = args['ebook']
        session['device'] = args.get('device', default_device)
        session['language'] = args.get('language', default_language_code)

        # Handle language ISO conversion
        try:
            if len(session['language']) in (2, 3):
                lang_dict = Lang(session['language'])
                if lang_dict:
                    session['language'] = lang_dict.pt3
                    session['language_iso1'] = lang_dict.pt1
            if not session.get('language_iso1'):
                session['language_iso1'] = session['language'][:2] if len(session['language']) >= 2 else 'en'
        except Exception:
            session['language_iso1'] = 'en'

        session['tts_engine'] = args.get('tts_engine') or get_compatible_tts_engines(session['language'])[0]
        session['audiobooks_dir'] = args.get('audiobooks_dir', audiobooks_cli_dir)
        session['fine_tuned'] = args.get('fine_tuned', default_fine_tuned)
        session['voice'] = args.get('voice')
        # Pre-staged custom model (BookForge user-added voices): --custom_model is a
        # NAME and --custom_model_dir is the staging root. No zip extraction here —
        # the checkpoint files are loaded in place by the engine. fine_tuned stays a
        # valid preset key (used for samplerate); the custom checkpoint overrides it.
        session['custom_model'] = args.get('custom_model')
        # Folder-discovered custom Orpheus model (folder name = voice token).
        session['orpheus_model_dir'] = args.get('orpheus_model_dir')
        session['output_format'] = args.get('output_format', default_output_format)

        # XTTS settings - save to session-state.json so workers use them instead of defaults
        # CLI args use short names (temperature, top_p, etc.) but session uses xtts_ prefix
        xtts_settings_saved = []
        if args.get('temperature') is not None:
            session['xtts_temperature'] = float(args['temperature'])
            xtts_settings_saved.append(f"temperature={args['temperature']}")
        if args.get('length_penalty') is not None:
            session['xtts_length_penalty'] = float(args['length_penalty'])
            xtts_settings_saved.append(f"length_penalty={args['length_penalty']}")
        if args.get('num_beams') is not None:
            session['xtts_num_beams'] = int(args['num_beams'])
            xtts_settings_saved.append(f"num_beams={args['num_beams']}")
        if args.get('repetition_penalty') is not None:
            session['xtts_repetition_penalty'] = float(args['repetition_penalty'])
            xtts_settings_saved.append(f"repetition_penalty={args['repetition_penalty']}")
        if args.get('top_k') is not None:
            session['xtts_top_k'] = int(args['top_k'])
            xtts_settings_saved.append(f"top_k={args['top_k']}")
        if args.get('top_p') is not None:
            session['xtts_top_p'] = float(args['top_p'])
            xtts_settings_saved.append(f"top_p={args['top_p']}")
        if args.get('speed') is not None:
            session['xtts_speed'] = float(args['speed'])
            xtts_settings_saved.append(f"speed={args['speed']}")
        if args.get('enable_text_splitting'):
            session['xtts_enable_text_splitting'] = True
            xtts_settings_saved.append("enable_text_splitting=True")

        if xtts_settings_saved:
            print(f"[PREP] XTTS settings saved to session: {', '.join(xtts_settings_saved)}")
        else:
            print("[PREP] WARNING: No XTTS settings found in args - workers will use defaults")

        # Must set BEFORE get_chapters() - controls whether breaks between words are preserved
        session['sentence_per_paragraph'] = bool(args.get('sentence_per_paragraph', False))
        # Skip heading text in TTS (headings parsed for chapter detection but not spoken)
        session['skip_headings'] = bool(args.get('skip_headings', False))

        print(f"[PREP] fine_tuned={session['fine_tuned']}, voice={session['voice']}, tts_engine={session['tts_engine']}, sentence_per_paragraph={session['sentence_per_paragraph']}, skip_headings={session['skip_headings']}")

        # Setup directories using ebook- prefix for parallel sessions
        session['session_dir'] = os.path.join(tmp_dir, f"ebook-{session_id}")
        session['process_dir'] = os.path.join(session['session_dir'], f"{hashlib.md5(session['ebook'].encode()).hexdigest()}")
        session['chapters_dir'] = os.path.join(session['process_dir'], 'chapters')
        session['sentences_dir'] = os.path.join(session['chapters_dir'], 'sentences')
        # A pre-staged custom_model_dir (BookForge) is used in place; otherwise the
        # session-derived dir is used (legacy zip-extraction path in lib/core.py).
        session['custom_model_dir'] = args.get('custom_model_dir') or os.path.join(models_dir, '__sessions', f"model-{session_id}")
        session['voice_dir'] = os.path.join(voices_dir, '__sessions', f"voice-{session_id}", session['language'])

        # Detect VRAM
        vram_dict = VRAMDetector().detect_vram(session['device'], session['script_mode'])
        session['free_vram_gb'] = vram_dict.get('free_vram_gb', 4)

        if prepare_dirs(args['ebook'], session_id):
            session['filename_noext'] = os.path.splitext(os.path.basename(session['ebook']))[0]
            session['epub_path'] = os.path.join(session['process_dir'], f"__{session['filename_noext']}.epub")
            session['final_name'] = get_sanitized(session['filename_noext'] + '.' + session['output_format'])

            if convert2epub(session_id):
                epubBook = epub.read_epub(session['epub_path'], {'ignore_ncx': True})

                # Get metadata
                metadata = dict(session['metadata'])
                for key in metadata.keys():
                    data = epubBook.get_metadata('DC', key)
                    if data:
                        for value, attributes in data:
                            metadata[key] = value
                metadata['title'] = metadata['title'] if metadata['title'] else session['filename_noext'].replace('_', ' ')
                session['metadata'] = metadata

                # Get cover
                try:
                    session['cover'] = get_cover(epubBook, session_id)
                except Exception as cover_err:
                    print(f'Warning: Could not get cover: {cover_err}')
                    session['cover'] = None

                # Get chapters
                session['chapters'] = get_chapters(session_id, epubBook)
                if session['chapters']:
                    total_chapters = len(session['chapters'])
                    # total_sentences is the number of GENERATION CHUNKS (scheduling unit).
                    # total_raw_sentences is the REAL sentence count inside those chunks —
                    # for a true sentences/min analytics rate; the pipeline is unchanged.
                    total_sentences = sum(len(chapter) for chapter in session['chapters'])
                    total_raw_sentences = sum(
                        count_real_sentences(c) for chapter in session['chapters'] for c in chapter
                    )

                    # Build chapter info
                    chapter_info = []
                    sentence_offset = 0
                    for i, chapter in enumerate(session['chapters']):
                        chapter_info.append({
                            'chapter_num': i + 1,
                            'sentence_count': len(chapter),
                            'raw_sentence_count': sum(count_real_sentences(c) for c in chapter),
                            'sentence_start': sentence_offset,
                            'sentence_end': sentence_offset + len(chapter) - 1
                        })
                        sentence_offset += len(chapter)

                    result = {
                        'session_id': session_id,
                        'session_dir': session['session_dir'],
                        'process_dir': session['process_dir'],
                        'chapters_dir': session['chapters_dir'],
                        'chapters_dir_sentences': session['sentences_dir'],
                        'total_chapters': total_chapters,
                        'total_sentences': total_sentences,
                        'total_raw_sentences': total_raw_sentences,
                        'chapters': chapter_info,
                        'chapter_sentences': list(session['chapters']),
                        'metadata': {
                            'title': metadata.get('title'),
                            'creator': metadata.get('creator'),
                            'language': metadata.get('language')
                        }
                    }

                    # Save session state for workers
                    save_session_state(session_id, args, result, core_module)
                    return result

        return None
    except Exception as e:
        print(f'prep_ebook_info() Exception: {e}')
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Phase 2: Worker TTS
# =============================================================================

def worker_only(args: dict, core_module) -> dict:
    """
    Run TTS conversion for an assigned sentence range only.
    This is Phase 2 of the parallel processing architecture.

    Note: For optimal memory usage, use worker.py instead of this function.
    This function still loads all core.py dependencies.
    """
    try:
        from lib.core import (
            get_compatible_tts_engines,
            convert_chapters2audio,
            VRAMDetector,
        )

        context = core_module.context
        context_tracker = core_module.context_tracker

        session_id = args.get('session')
        if not session_id:
            return {'success': False, 'error': 'No session ID provided'}

        sentence_start = args.get('sentence_start')
        sentence_end = args.get('sentence_end')
        chapter_start = args.get('chapter_start')
        chapter_end = args.get('chapter_end')

        sentence_mode = sentence_start is not None and sentence_end is not None
        chapter_mode = chapter_start is not None and chapter_end is not None

        if not sentence_mode and not chapter_mode:
            return {'success': False, 'error': 'Must specify sentence or chapter range'}

        # Load session state — honor an explicit --session_dir (BookForge points this
        # at the durable project cache; the tmp session may be gone/relocated on WSL).
        session_dir = args.get('session_dir') or os.path.join(tmp_dir, f"ebook-{session_id}")
        if not os.path.exists(session_dir):
            return {'success': False, 'error': f"Session directory not found: {session_dir}"}

        state = load_session_state(session_dir)
        if not state:
            return {'success': False, 'error': 'No session-state.json found'}

        if 'chapter_sentences' not in state or not state['chapter_sentences']:
            return {'success': False, 'error': 'session-state.json missing chapter_sentences'}

        print(f"[WORKER] Loaded: {state['total_chapters']} chapters, {state['total_sentences']} sentences")

        # Get or create session
        session = context.get_session(session_id)
        if not session or not session.get('id'):
            session = context.set_session(session_id)

        # Populate session from state
        session['session_dir'] = state['session_dir']
        session['process_dir'] = state['process_dir']
        session['chapters_dir'] = state.get('chapters_dir') or os.path.join(state['process_dir'], 'chapters')
        # --sentences_dir (BookForge) is the single authoritative sentence store: new
        # sentences are written here and existing ones are skipped (resume). Explicit
        # override wins over the (possibly stale) session-state path.
        session['sentences_dir'] = args.get('sentences_dir') or state.get('chapters_dir_sentences') or os.path.join(state['process_dir'], 'chapters', 'sentences')
        session['epub_path'] = state.get('epub_path_internal')
        session['filename_noext'] = state.get('filename_noext')
        session['cover'] = state.get('cover')
        session['language'] = state.get('language', default_language_code)
        session['language_iso1'] = state.get('language_iso1', 'en')
        session['tts_engine'] = args.get('tts_engine') or state.get('tts_engine') or get_compatible_tts_engines(session['language'])[0]
        session['fine_tuned'] = args.get('fine_tuned') or state.get('fine_tuned') or default_fine_tuned
        session['voice'] = args.get('voice') or state.get('voice')
        session['output_format'] = args.get('output_format') or state.get('output_format', default_output_format)
        session['audiobooks_dir'] = args.get('audiobooks_dir') or state.get('audiobooks_dir', audiobooks_cli_dir)
        session['cancellation_requested'] = False
        session['final_name'] = state.get('final_name')
        session['chapters'] = state['chapter_sentences']
        session['script_mode'] = NATIVE
        session['is_gui_process'] = False
        session['model_cache'] = f"{session['tts_engine']}-{session['fine_tuned']}"
        session['custom_model_dir'] = state.get('custom_model_dir') or os.path.join(models_dir, '__sessions', f"model-{session_id}")
        session['custom_model'] = state.get('custom_model')

        # Device setup
        session['device'] = args.get('device') or state.get('device', default_device)
        if session['device'] == 'cuda':
            import torch
            session['device'] = session['device'] if torch.cuda.is_available() else 'cpu'
        elif session['device'] == 'mps':
            import torch
            session['device'] = session['device'] if torch.backends.mps.is_available() else 'cpu'

        # VRAM detection
        vram_dict = VRAMDetector().detect_vram(session['device'], session['script_mode'])
        session['free_vram_gb'] = vram_dict.get('free_vram_gb', 4)

        # Calculate work range for chapter mode
        if chapter_mode:
            sentence_offset = 0
            for i, chapter in enumerate(state['chapter_sentences']):
                chapter_num = i + 1
                if chapter_num == chapter_start:
                    sentence_start = sentence_offset
                if chapter_num == chapter_end:
                    sentence_end = sentence_offset + len(chapter) - 1
                    break
                sentence_offset += len(chapter)
            print(f"[WORKER] Chapters {chapter_start}-{chapter_end} = sentences {sentence_start}-{sentence_end}")

        # Set worker mode flags
        session['worker_mode'] = True
        session['sentence_start'] = sentence_start
        session['sentence_end'] = sentence_end

        print(f"[WORKER] Processing sentences {sentence_start}-{sentence_end} on {session['device'].upper()}")

        # Run TTS conversion
        if convert_chapters2audio(session_id):
            return {
                'success': True,
                'session_id': session_id,
                'sentence_start': sentence_start,
                'sentence_end': sentence_end
            }
        else:
            return {'success': False, 'error': 'convert_chapters2audio() failed'}

    except Exception as e:
        print(f'worker_only() Exception: {e}')
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# Phase 3: Assembly
# =============================================================================

def parse_chapters_arg(chapters_arg: str, total_chapters: int) -> list[int] | None:
    """
    Parse the --chapters argument into a list of chapter numbers (1-indexed).

    Supported formats:
    - "1-5" -> [1, 2, 3, 4, 5]
    - "1,3,5" -> [1, 3, 5]
    - "1-3,7,9-11" -> [1, 2, 3, 7, 9, 10, 11]
    - "auto" -> None (signals auto-detection)

    Returns None for "auto" mode, or a sorted list of chapter numbers.
    """
    if not chapters_arg:
        return list(range(1, total_chapters + 1))  # All chapters

    chapters_arg = chapters_arg.strip().lower()

    if chapters_arg == 'auto':
        return None  # Signal auto-detection

    result = set()
    parts = chapters_arg.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range: "1-5"
            try:
                start, end = part.split('-', 1)
                start = int(start.strip())
                end = int(end.strip())
                for ch in range(start, end + 1):
                    if 1 <= ch <= total_chapters:
                        result.add(ch)
            except ValueError:
                print(f"[ASSEMBLE] Warning: Invalid chapter range '{part}'")
        else:
            # Single chapter: "5"
            try:
                ch = int(part)
                if 1 <= ch <= total_chapters:
                    result.add(ch)
            except ValueError:
                print(f"[ASSEMBLE] Warning: Invalid chapter number '{part}'")

    return sorted(result) if result else list(range(1, total_chapters + 1))


def detect_completed_chapters(session: dict, state: dict) -> list[int]:
    """
    Auto-detect which chapters have all their sentence audio files completed.
    Returns a list of completed chapter numbers (1-indexed).
    """
    sentences_dir = session.get('sentences_dir') or os.path.join(session['process_dir'], 'chapters', 'sentences')
    chapter_sentences = state.get('chapter_sentences', [])

    completed = []
    sentence_offset = 0

    for i, chapter in enumerate(chapter_sentences):
        chapter_num = i + 1
        chapter_complete = True

        # Check if all sentence files for this chapter exist
        for j in range(len(chapter)):
            sentence_idx = sentence_offset + j
            sentence_file = os.path.join(sentences_dir, f'{sentence_idx}.{default_audio_proc_format}')
            if not os.path.exists(sentence_file):
                chapter_complete = False
                break

        if chapter_complete:
            completed.append(chapter_num)
        else:
            # Once we hit an incomplete chapter, stop (chapters are sequential)
            break

        sentence_offset += len(chapter)

    return completed


def build_vtt_file(session_id: str, all_sentences: list, core_module) -> bool:
    """
    Build a VTT (WebVTT) subtitle file from sentence audio files.
    Called during assembly phase to create subtitle track for the audiobook.
    """
    from pathlib import Path
    import re
    import subprocess

    def get_duration_ffprobe(filepath: str) -> float:
        """Get audio duration using ffprobe (more reliable than mediainfo)."""
        try:
            ffprobe = shutil.which('ffprobe')
            if not ffprobe:
                return 0.0
            cmd = [ffprobe, '-v', 'quiet', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', filepath]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    try:
        context = core_module.context
        session = context.get_session(session_id)
        if not session:
            print(f"[VTT] No session found for {session_id}")
            return False

        sentences_dir = session.get('sentences_dir')
        process_dir = session.get('process_dir')
        final_name = session.get('final_name', 'audiobook.m4b')

        if not sentences_dir or not process_dir:
            print("[VTT] Missing sentences_dir or process_dir in session")
            return False

        # Flatten all_sentences if it's a list of chapters
        flat_sentences = []
        for item in all_sentences:
            if isinstance(item, list):
                flat_sentences.extend(item)
            else:
                flat_sentences.append(item)

        vtt_path = os.path.join(process_dir, Path(final_name).stem + '.vtt')
        print(f"[VTT] Building VTT file: {vtt_path}")

        # Get sorted audio files
        audio_dir = Path(sentences_dir)
        audio_files = sorted(
            audio_dir.glob(f'*.{default_audio_proc_format}'),
            key=lambda p: int(p.stem)
        )

        if len(audio_files) != len(flat_sentences):
            print(f"[VTT] Warning: {len(audio_files)} audio files vs {len(flat_sentences)} sentences")
            # Use minimum to avoid index errors
            count = min(len(audio_files), len(flat_sentences))
            audio_files = audio_files[:count]
            flat_sentences = flat_sentences[:count]

        if not audio_files:
            print("[VTT] No audio files found")
            return False

        # Get durations using ffprobe (more reliable than mediainfo)
        print("[VTT] Getting audio durations...")
        durations = {str(p): get_duration_ffprobe(str(p)) for p in audio_files}

        # Build VTT content
        print("[VTT] Creating VTT blocks...")
        vtt_blocks = []
        current_time = 0.0

        # SML tag pattern for cleaning
        SML_PATTERN = re.compile(r'\[(?:break|music|sfx|silence)(?::[^\]]+)?\]', re.IGNORECASE)

        def format_timestamp(seconds: float) -> str:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            return f'{int(h):02}:{int(m):02}:{s:06.3f}'

        for idx, audio_file in enumerate(audio_files):
            start_time = current_time
            duration = durations.get(str(audio_file), 0.0)
            end_time = start_time + duration
            current_time = end_time

            # Clean sentence text
            text = re.sub(r'\s+', ' ', SML_PATTERN.sub('', str(flat_sentences[idx]))).strip()

            start_str = format_timestamp(start_time)
            end_str = format_timestamp(end_time)
            vtt_blocks.append(f'{start_str} --> {end_str}\n{text}\n')

        # Write VTT file
        print(f"[VTT] Writing {len(vtt_blocks)} blocks to {vtt_path}")
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write('WEBVTT\n\n')
            f.write('\n'.join(vtt_blocks))

        print("[VTT] VTT file created successfully")
        return True

    except Exception as e:
        print(f"[VTT] Error building VTT file: {e}")
        import traceback
        traceback.print_exc()
        return False


def assemble_audiobook(args: dict, core_module) -> dict:
    """
    Assemble sentence audio files into the final audiobook.
    """
    try:
        from iso639 import Lang
        from lib.core import (
            get_sanitized,
            combine_audio_sentences,
            combine_audio_chapters,
        )

        context = core_module.context
        context_tracker = core_module.context_tracker

        session_id = args.get('session')
        if not session_id:
            return {'success': False, 'error': 'No session ID provided'}

        # Setup session from saved state
        # Use explicit --session_dir if provided, otherwise fall back to tmp_dir
        session_dir = args.get('session_dir') or os.path.join(tmp_dir, f"ebook-{session_id}")
        if not os.path.exists(session_dir):
            return {'success': False, 'error': f"Session directory not found: {session_dir}"}

        state = load_session_state(session_dir)
        if not state:
            return {'success': False, 'error': 'No session-state.json found'}

        # Get or create session
        session = context.get_session(session_id)
        if not session or not session.get('id'):
            session = context.set_session(session_id)

        # Populate from state
        # Always derive directories from the corrected process_dir/session_dir
        # (session-state.json may contain stale paths from a different machine or WSL)
        session['session_dir'] = state['session_dir']
        session['process_dir'] = state['process_dir']
        session['chapters_dir'] = os.path.join(state['process_dir'], 'chapters')
        # BookForge: --sentences_dir overrides ONLY the sentence-audio source so we can
        # assemble a post-processed set (e.g. RVC voice-enhanced renders) while leaving
        # the original XTTS sentences cached in place. Chapter mapping / metadata / VTT
        # still come from the unchanged session state. Default = the session's own sentences.
        sentences_dir_override = args.get('sentences_dir')
        if sentences_dir_override:
            if not os.path.isdir(sentences_dir_override):
                return {'success': False, 'error': f"Sentences directory not found: {sentences_dir_override}"}
            session['sentences_dir'] = os.path.abspath(sentences_dir_override)
            print(f"[ASSEMBLE] Using overridden sentences_dir: {session['sentences_dir']}")
        else:
            session['sentences_dir'] = os.path.join(state['process_dir'], 'chapters', 'sentences')
        session['language'] = state.get('language', default_language_code)
        session['output_format'] = args.get('output_format') or state.get('output_format', default_output_format)
        session['audiobooks_dir'] = args.get('audiobooks_dir') or state.get('audiobooks_dir', audiobooks_cli_dir)
        session['cancellation_requested'] = False
        session['script_mode'] = NATIVE
        session['is_gui_process'] = False
        session['output_channel'] = args.get('output_channel', 'mono')
        session['output_split'] = args.get('output_split', default_output_split)
        session['output_split_hours'] = args.get('output_split_hours', default_output_split_hours)

        # Language ISO
        try:
            if len(session['language']) in (2, 3):
                lang_dict = Lang(session['language'])
                if lang_dict:
                    session['language'] = lang_dict.pt3
                    session['language_iso1'] = lang_dict.pt1
            if not session.get('language_iso1'):
                session['language_iso1'] = session['language'][:2] if len(session['language']) >= 2 else 'en'
        except Exception:
            session['language_iso1'] = 'en'

        # Load chapter data
        if state.get('chapter_sentences'):
            print('[ASSEMBLE] Loading chapter data from session-state.json')
            session['chapters'] = state['chapter_sentences']
            session['metadata'] = state.get('metadata', {})
            session['filename_noext'] = state.get('filename_noext')
            # Re-derive cover path from corrected process_dir
            # (session-state.json may contain stale WSL path)
            raw_cover = state.get('cover')
            if isinstance(raw_cover, str) and session.get('filename_noext'):
                session['cover'] = os.path.join(state['process_dir'], session['filename_noext'] + '.jpg')
            else:
                session['cover'] = raw_cover
            # Load TOC chapter titles for proper chapter markers
            session['chapter_titles'] = state.get('chapter_titles', [])
            if session['chapter_titles']:
                print(f"[ASSEMBLE] Loaded {len(session['chapter_titles'])} chapter titles from TOC")
        else:
            return {'success': False, 'error': 'No chapter_sentences in session state'}

        if not session['metadata'].get('title'):
            session['metadata']['title'] = session.get('filename_noext', 'audiobook').replace('_', ' ')

        # Apply any command-line metadata overrides
        if args.get('title'):
            session['metadata']['title'] = args['title']
        if args.get('author'):
            session['metadata']['creator'] = args['author']

        # Check for bookforge_metadata in session state (has author, year, etc.)
        if state.get('bookforge_metadata'):
            bf_meta = state['bookforge_metadata']
            if bf_meta.get('title') and not args.get('title'):
                session['metadata']['title'] = bf_meta['title']
            if bf_meta.get('author') and not args.get('author'):
                session['metadata']['creator'] = bf_meta['author']
            if bf_meta.get('year'):
                session['metadata']['year'] = bf_meta['year']

        # Extract year from 'published' field if not already set (format: "YYYY-MM-DD...")
        if not session['metadata'].get('year') and session['metadata'].get('published'):
            published = session['metadata']['published']
            if isinstance(published, str) and len(published) >= 4:
                session['metadata']['year'] = published[:4]

        # Set final_name: use --output_filename if provided, otherwise format with author/year
        if args.get('output_filename'):
            # Use explicit filename (without extension, we'll add it)
            base_name = args['output_filename']
            if base_name.lower().endswith('.' + session['output_format']):
                base_name = base_name[:-len(session['output_format']) - 1]
            session['final_name'] = get_sanitized(base_name + '.' + session['output_format'])
        else:
            # Build filename: "Title. Author. (Year).ext" or just "Title.ext"
            title = session['metadata'].get('title', 'Untitled')
            author = session['metadata'].get('creator', '')
            year = session['metadata'].get('year', '')

            if author and year:
                filename_base = f"{title}. {author}. ({year})"
            elif author:
                filename_base = f"{title}. {author}"
            else:
                filename_base = title

            session['final_name'] = get_sanitized(filename_base + '.' + session['output_format'])

        # Parse --chapters argument for partial assembly
        total_chapters = len(session['chapters'])
        chapters_arg = args.get('chapters')
        selected_chapters = parse_chapters_arg(chapters_arg, total_chapters)

        # Auto-detect completed chapters if requested
        if selected_chapters is None:
            print("[ASSEMBLE] Auto-detecting completed chapters...")
            selected_chapters = detect_completed_chapters(session, state)
            if not selected_chapters:
                return {'success': False, 'error': 'No completed chapters found. TTS may still be in progress.'}
            print(f"[ASSEMBLE] Found {len(selected_chapters)} completed chapters: {selected_chapters[0]}-{selected_chapters[-1]}")

        # Check if this is a partial assembly
        is_partial = len(selected_chapters) < total_chapters
        if is_partial:
            print(f"[ASSEMBLE] Partial assembly: chapters {selected_chapters[0]}-{selected_chapters[-1]} of {total_chapters}")
            # Update filename to indicate partial
            if not args.get('output_filename'):
                base_name = session['final_name'].rsplit('.', 1)[0]
                ext = session['output_format']
                partial_suffix = f" (Partial Ch {selected_chapters[0]}-{selected_chapters[-1]})"
                session['final_name'] = get_sanitized(base_name + partial_suffix + '.' + ext)
        else:
            print(f"[ASSEMBLE] Assembling all {total_chapters} chapters...")

        # Store selected chapters for combine_audio_chapters
        session['selected_chapters'] = selected_chapters

        # Check for bilingual mode
        if args.get('bilingual'):
            # Bilingual assembly: pair source/target sentences with pauses
            from .bilingual import combine_bilingual_audio, build_bilingual_vtt

            pause_duration = args.get('bilingual_pause', 0.3)
            gap_duration = args.get('bilingual_gap', 1.0)

            print(f"[ASSEMBLE] Bilingual mode: pause={pause_duration}s, gap={gap_duration}s")

            # Count total sentences
            all_sentences = []
            for chapter in session['chapters']:
                all_sentences.extend(chapter)
            num_sentences = len(all_sentences)

            print(f"[ASSEMBLE] Total sentences: {num_sentences}")

            # Combine all sentences as bilingual pairs
            chapter_file = os.path.join(session['chapters_dir'], f'1.{default_audio_proc_format}')
            if not combine_bilingual_audio(
                sentences_dir=session['sentences_dir'],
                output_file=chapter_file,
                num_sentences=num_sentences,
                pause_duration=pause_duration,
                gap_duration=gap_duration,
                audio_format=default_audio_proc_format
            ):
                return {'success': False, 'error': 'Bilingual audio combination failed'}

            # Build bilingual VTT
            print("[ASSEMBLE] Creating bilingual VTT subtitle file...")
            vtt_path = os.path.join(session['process_dir'], Path(session['final_name']).stem + '.vtt')
            if not build_bilingual_vtt(
                sentences_dir=session['sentences_dir'],
                sentence_texts=all_sentences,
                output_path=vtt_path,
                pause_duration=pause_duration,
                gap_duration=gap_duration,
                audio_format=default_audio_proc_format
            ):
                print("[ASSEMBLE] Warning: Bilingual VTT creation failed (continuing without subtitles)")

            # Update session to have single chapter for combine_audio_chapters
            session['chapters'] = [all_sentences]
            session['selected_chapters'] = [1]

            # Combine (single chapter) into final audiobook
            print("[ASSEMBLE] Combining into final audiobook...")
            exported_files = combine_audio_chapters(session_id)

        else:
            # Normal (non-bilingual) assembly
            # Combine sentences into chapters (only for selected chapters)
            sentence_offset = 0
            for i, chapter in enumerate(session['chapters']):
                chapter_num = i + 1

                # Skip chapters not in selection
                if chapter_num not in selected_chapters:
                    sentence_offset += len(chapter)
                    continue

                chapter_filename = f'{chapter_num}.{default_audio_proc_format}'
                start_sentence = sentence_offset
                end_sentence = sentence_offset + len(chapter) - 1

                print(f"[ASSEMBLE] Chapter {chapter_num}: sentences {start_sentence}-{end_sentence}")

                if not combine_audio_sentences(session_id, chapter_filename, start_sentence, end_sentence):
                    return {'success': False, 'error': f'Failed to combine sentences for chapter {chapter_num}'}

                sentence_offset += len(chapter)

            # Build VTT subtitle file (only for selected chapters)
            print("[ASSEMBLE] Creating VTT subtitle file...")
            selected_chapter_data = [session['chapters'][i-1] for i in selected_chapters]
            if not build_vtt_file(session_id, selected_chapter_data, core_module):
                print("[ASSEMBLE] Warning: VTT file creation failed (continuing without subtitles)")

            # Combine chapters into final audiobook
            print("[ASSEMBLE] Combining chapters into final audiobook...")
            exported_files = combine_audio_chapters(session_id)

        if exported_files and len(exported_files) > 0:
            return {
                'success': True,
                'session_id': session_id,
                'output_files': exported_files,
                'output_dir': session['audiobooks_dir']
            }
        else:
            return {'success': False, 'error': 'combine_audio_chapters() failed'}

    except Exception as e:
        print(f'assemble_audiobook() Exception: {e}')
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

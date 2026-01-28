"""
Lightweight Worker Core for Parallel TTS Processing

This module provides minimal-import functions for TTS workers.
Unlike core.py which imports gradio, stanza, pytesseract, etc.,
this module only imports what's needed for TTS conversion.

This reduces worker memory from ~25GB to ~8GB.

Usage:
    from lib.worker_core import load_session_state, run_worker_tts
"""

import gc
import json
import os
import sys
import time
from datetime import datetime

import torch

from lib.conf import (
    tmp_dir,
    default_audio_proc_format,
    devices,
    NATIVE,
)
from lib.conf_models import TTS_ENGINES, default_engine_settings, default_fine_tuned
from lib.conf_lang import default_language_code

# Import TTS Manager and VRAM detector (lightweight)
from lib.classes.tts_manager import TTSManager
from lib.classes.vram_detector import VRAMDetector


def load_session_state(session_dir: str) -> dict | None:
    """
    Load session state from session-state.json.
    Returns the state dict or None if not found/invalid.
    """
    try:
        # Find process_dir within session_dir
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


def scan_completed_sentences(sentences_dir: str, total_sentences: int, min_file_size: int = 1024) -> dict:
    """
    Scan the sentences directory to find completed audio files.
    Returns dict with 'completed' and 'missing' sentence indices.
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


def get_compatible_tts_engines(lang_code: str) -> list:
    """Get list of compatible TTS engines for a language."""
    # Default to xtts which supports most languages
    return ['xtts', 'vits', 'bark', 'tacotron', 'fairseq', 'yourtts']


def create_worker_session(state: dict, args: dict) -> dict:
    """
    Create a session dict from loaded state and CLI args.
    This mimics what core.py's SessionContext.get_session() returns,
    but without the multiprocessing Manager overhead.
    """
    # Detect device and VRAM
    device = args.get('device') or state.get('device', 'cpu')
    if device == 'cuda':
        device = device if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("[WORKER] CUDA not available, falling back to CPU")
    elif device == 'mps':
        device = device if torch.backends.mps.is_available() else 'cpu'
        if device == 'cpu':
            print("[WORKER] MPS not available, falling back to CPU")

    # Detect VRAM
    vram_detector = VRAMDetector()
    vram_info = vram_detector.detect_vram(device, NATIVE)
    free_vram_gb = vram_info.get('free_vram_gb', 4)

    # Get TTS engine
    tts_engine = args.get('tts_engine') or state.get('tts_engine') or get_compatible_tts_engines(state.get('language', 'eng'))[0]
    fine_tuned = args.get('fine_tuned') or state.get('fine_tuned') or default_fine_tuned

    # Build model cache key
    model_cache = f"{tts_engine}-{fine_tuned}"

    session = {
        'id': state['session_id'],
        'script_mode': NATIVE,
        'is_gui_process': False,
        'device': device,
        'free_vram_gb': free_vram_gb,
        'tts_engine': tts_engine,
        'fine_tuned': fine_tuned,
        'model_cache': model_cache,
        'model_zs_cache': None,
        'language': state.get('language', default_language_code),
        'language_iso1': state.get('language_iso1', 'en'),
        'voice': args.get('voice') or state.get('voice'),
        'voice_dir': os.path.join(state['process_dir'], 'voices'),
        'custom_model': None,
        'custom_model_dir': os.path.join(state['process_dir'], 'models'),
        'audiobooks_dir': args.get('output_dir') or state.get('audiobooks_dir'),
        'session_dir': state['session_dir'],
        'process_dir': state['process_dir'],
        'chapters_dir': state.get('chapters_dir') or os.path.join(state['process_dir'], 'chapters'),
        'sentences_dir': state.get('chapters_dir_sentences') or os.path.join(state['process_dir'], 'chapters', 'sentences'),
        'epub_path': state.get('epub_path_internal'),
        'filename_noext': state.get('filename_noext'),
        'final_name': state.get('final_name'),
        'cover': state.get('cover'),
        'output_format': args.get('output_format') or state.get('output_format', 'm4b'),
        'cancellation_requested': False,
        # TTS settings from state
        'xtts_temperature': state.get('xtts_temperature') or default_engine_settings[TTS_ENGINES['XTTSv2']]['temperature'],
        'xtts_length_penalty': state.get('xtts_length_penalty') or default_engine_settings[TTS_ENGINES['XTTSv2']]['length_penalty'],
        'xtts_num_beams': state.get('xtts_num_beams') or default_engine_settings[TTS_ENGINES['XTTSv2']]['num_beams'],
        'xtts_repetition_penalty': state.get('xtts_repetition_penalty') or default_engine_settings[TTS_ENGINES['XTTSv2']]['repetition_penalty'],
        'xtts_top_k': state.get('xtts_top_k') or default_engine_settings[TTS_ENGINES['XTTSv2']]['top_k'],
        'xtts_top_p': state.get('xtts_top_p') or default_engine_settings[TTS_ENGINES['XTTSv2']]['top_p'],
        'xtts_speed': state.get('xtts_speed') or default_engine_settings[TTS_ENGINES['XTTSv2']]['speed'],
        'xtts_enable_text_splitting': state.get('xtts_enable_text_splitting') or default_engine_settings[TTS_ENGINES['XTTSv2']]['enable_text_splitting'],
    }

    return session


def memory_cleanup(sentence_idx: int = 0, interval: int = 100):
    """
    Perform memory cleanup after processing sentences.
    Called periodically to prevent memory accumulation.
    """
    if sentence_idx % interval == 0:
        gc.collect()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_worker_tts(
    session_id: str,
    sentence_start: int,
    sentence_end: int,
    args: dict,
    chapter_start: int | None = None,
    chapter_end: int | None = None
) -> dict:
    """
    Run TTS conversion for an assigned sentence range.

    This is the main worker function that:
    1. Loads session state from session-state.json
    2. Creates a lightweight session dict
    3. Initializes TTS engine
    4. Converts assigned sentences to audio

    Args:
        session_id: Session ID from prep phase
        sentence_start: First sentence index (0-indexed, inclusive)
        sentence_end: Last sentence index (0-indexed, inclusive)
        args: CLI args dict
        chapter_start: Optional chapter start (1-indexed, for chapter mode)
        chapter_end: Optional chapter end (1-indexed, for chapter mode)

    Returns:
        dict with success status and details
    """
    try:
        # Load session state
        session_dir = os.path.join(tmp_dir, f"ebook-{session_id}")
        if not os.path.exists(session_dir):
            return {'success': False, 'error': f"Session directory not found: {session_dir}"}

        state = load_session_state(session_dir)
        if not state:
            return {'success': False, 'error': f"No session-state.json found in {session_dir}"}

        # Verify we have sentence data
        if 'chapter_sentences' not in state or not state['chapter_sentences']:
            return {'success': False, 'error': 'session-state.json missing chapter_sentences'}

        print(f"[WORKER] Loaded session: {state['total_chapters']} chapters, {state['total_sentences']} sentences")

        # Create worker session
        session = create_worker_session(state, args)

        # Ensure sentences directory exists
        os.makedirs(session['sentences_dir'], exist_ok=True)

        # Flatten chapter_sentences to get all sentences
        all_sentences = []
        for chapter in state['chapter_sentences']:
            all_sentences.extend(chapter)

        # Determine work range
        sentence_mode = sentence_start is not None and sentence_end is not None
        chapter_mode = chapter_start is not None and chapter_end is not None

        if chapter_mode:
            # Convert chapter range to sentence range
            sentence_offset = 0
            work_start = 0
            work_end = 0
            for i, chapter in enumerate(state['chapter_sentences']):
                chapter_num = i + 1
                if chapter_num == chapter_start:
                    work_start = sentence_offset
                if chapter_num == chapter_end:
                    work_end = sentence_offset + len(chapter) - 1
                    break
                sentence_offset += len(chapter)
            sentence_start = work_start
            sentence_end = work_end
            print(f"[WORKER] Chapter mode: chapters {chapter_start}-{chapter_end} = sentences {sentence_start}-{sentence_end}")

        # Validate range
        if sentence_start < 0 or sentence_end >= len(all_sentences):
            return {'success': False, 'error': f'Invalid sentence range: {sentence_start}-{sentence_end} (total: {len(all_sentences)})'}

        print(f"[WORKER] Processing sentences {sentence_start}-{sentence_end} on {session['device'].upper()}")
        print(f"[WORKER] TTS engine: {session['tts_engine']}, fine_tuned: {session['fine_tuned']}")

        # Initialize TTS engine
        tts_manager = TTSManager(session)

        # Process sentences
        total_to_process = sentence_end - sentence_start + 1
        processed = 0
        skipped = 0
        start_time = time.time()

        for i in range(sentence_start, sentence_end + 1):
            # Check if already exists (for resume)
            output_file = os.path.join(session['sentences_dir'], f'{i}.{default_audio_proc_format}')
            if os.path.exists(output_file) and os.path.getsize(output_file) > 1024:
                skipped += 1
                processed += 1
                continue

            sentence = all_sentences[i]
            if not sentence or not sentence.strip():
                # Empty sentence - create silence
                skipped += 1
                processed += 1
                continue

            # Show progress
            progress_pct = (processed / total_to_process) * 100
            print(f"**Recovering missing file sentence {i}** - Converting {progress_pct:.2f}%: : {processed}/{total_to_process}")

            # Convert sentence to audio
            success = tts_manager.convert_sentence2audio(i, sentence)
            if not success:
                print(f"[WORKER] Warning: Failed to convert sentence {i}")
                # Continue anyway - some sentences may fail

            processed += 1

            # Periodic memory cleanup
            memory_cleanup(processed, interval=100)

        elapsed = time.time() - start_time
        actual_converted = processed - skipped

        print(f"[WORKER] Completed: {actual_converted} converted, {skipped} skipped in {elapsed:.1f}s")

        return {
            'success': True,
            'session_id': session_id,
            'sentences_processed': processed,
            'sentences_converted': actual_converted,
            'sentences_skipped': skipped,
            'sentence_start': sentence_start,
            'sentence_end': sentence_end,
            'elapsed_seconds': elapsed
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

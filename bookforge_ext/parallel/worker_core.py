"""
Lightweight Worker Core for Parallel TTS Processing

This module provides minimal-import functions for TTS workers.
Unlike core.py which imports gradio, stanza, pytesseract, etc.,
this module only imports what's needed for TTS conversion.

This reduces worker memory from ~25GB to ~8GB.

Usage:
    from bookforge_ext.parallel.worker_core import load_session_state, run_worker_tts

Note: This is the same code as lib/worker_core.py, moved to the extension
for better organization. worker.py imports from here.
"""

import gc
import json
import os
import sys
import time
from datetime import datetime

import psutil
import torch


def log_memory(label: str):
    """Log current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)
    print(f"[MEMORY] {label}: {rss_gb:.2f} GB")

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

# Fix device detection: DeviceInstaller doesn't run in workers, so devices['MPS']['found']
# stays False and xtts.py falls back to CPU even when user selects MPS.
# Set the flag here so user's device choice is respected.
if torch.backends.mps.is_available():
    devices['MPS']['found'] = True
if torch.cuda.is_available():
    devices['CUDA']['found'] = True

log_memory("After imports")


def register_tts_engine(engine_name: str) -> None:
    """
    Dynamically import and register only the needed TTS engine.
    This avoids loading all 8 engines (and their heavy deps) at startup.

    Each engine imports torch, transformers, TTS library, etc. at module level.
    By only importing the engine we need, we save ~20GB+ of memory.
    """
    engine_modules = {
        'xtts': 'lib.classes.tts_engines.xtts',
        'bark': 'lib.classes.tts_engines.bark',
        'vits': 'lib.classes.tts_engines.vits',
        'tortoise': 'lib.classes.tts_engines.tortoise',
        'fairseq': 'lib.classes.tts_engines.fairseq',
        'tacotron': 'lib.classes.tts_engines.tacotron',
        'yourtts': 'lib.classes.tts_engines.yourtts',
        'orpheus': 'lib.classes.tts_engines.orpheus',
    }

    module_name = engine_modules.get(engine_name)
    if module_name:
        import importlib
        importlib.import_module(module_name)
        print(f"[WORKER] Registered TTS engine: {engine_name}")
    else:
        raise ValueError(f"Unknown TTS engine: {engine_name}")


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

    # Get TTS engine — the session state (or CLI args) must name it explicitly;
    # silently guessing an engine here would run the wrong model.
    tts_engine = args.get('tts_engine') or state.get('tts_engine')
    if not tts_engine:
        raise ValueError('No tts_engine in session state or args — the session state must name its engine')
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
        # Pre-staged custom voice (BookForge): name + staging root come from the
        # prep args or the persisted session state. None → built-in/fine-tuned model.
        'custom_model': args.get('custom_model') or state.get('custom_model'),
        'custom_model_dir': (args.get('custom_model_dir') or state.get('custom_model_dir')
                             or os.path.join(state['process_dir'], 'models')),
        # Folder-discovered custom Orpheus model: absolute path to the model dir,
        # folder name = voice token. None → built-in Orpheus model.
        'orpheus_model_dir': args.get('orpheus_model_dir') or state.get('orpheus_model_dir'),
        'audiobooks_dir': args.get('output_dir') or state.get('audiobooks_dir'),
        'session_dir': state['session_dir'],
        'process_dir': state['process_dir'],
        'chapters_dir': state.get('chapters_dir') or os.path.join(state['process_dir'], 'chapters'),
        # --sentences_dir (BookForge) is the single authoritative sentence store: the
        # worker writes new sentences here AND skips ones already present here (resume).
        # Explicit override wins over the session-state path (which may be a stale tmp
        # location from a different machine/WSL). Precedence, not a bug-hiding fallback.
        'sentences_dir': args.get('sentences_dir') or state.get('chapters_dir_sentences') or os.path.join(state['process_dir'], 'chapters', 'sentences'),
        'epub_path': state.get('epub_path_internal'),
        'filename_noext': state.get('filename_noext'),
        'final_name': state.get('final_name'),
        'cover': state.get('cover'),
        'output_format': args.get('output_format') or state.get('output_format', 'm4b'),
        'cancellation_requested': False,
        # TTS settings from state — default ONLY on absence (`.get(k, default)`),
        # never on falsy values: `or default` would clobber legitimate settings
        # like enable_text_splitting=False or temperature=0.
        'xtts_temperature': state.get('xtts_temperature', default_engine_settings[TTS_ENGINES['XTTSv2']]['temperature']),
        'xtts_length_penalty': state.get('xtts_length_penalty', default_engine_settings[TTS_ENGINES['XTTSv2']]['length_penalty']),
        'xtts_num_beams': state.get('xtts_num_beams', default_engine_settings[TTS_ENGINES['XTTSv2']]['num_beams']),
        'xtts_repetition_penalty': state.get('xtts_repetition_penalty', default_engine_settings[TTS_ENGINES['XTTSv2']]['repetition_penalty']),
        'xtts_top_k': state.get('xtts_top_k', default_engine_settings[TTS_ENGINES['XTTSv2']]['top_k']),
        'xtts_top_p': state.get('xtts_top_p', default_engine_settings[TTS_ENGINES['XTTSv2']]['top_p']),
        # CLI --speed overrides the session state (explicit-arg precedence, not a fallback)
        'xtts_speed': args.get('speed') if args.get('speed') is not None
                      else state.get('xtts_speed', default_engine_settings[TTS_ENGINES['XTTSv2']]['speed']),
        'xtts_enable_text_splitting': state.get('xtts_enable_text_splitting', default_engine_settings[TTS_ENGINES['XTTSv2']]['enable_text_splitting']),
    }

    return session


def memory_cleanup(sentence_idx: int = 0, interval: int = 100):
    """
    Perform memory cleanup after processing sentences.
    Called periodically to prevent memory accumulation.
    """
    if sentence_idx % interval == 0:
        gc.collect()
        # Check is_available() not just hasattr - MPS module exists on Linux but backend isn't available
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_worker_tts(
    session_id: str,
    sentence_start: int,
    sentence_end: int,
    args: dict,
    chapter_start: int | None = None,
    chapter_end: int | None = None,
    session_dir_override: str | None = None,
    sentence_indices: list[int] | None = None,
    num_takes: int = 1,
    take_temperatures: list[float] | None = None,
    sentence_overrides: dict | None = None
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
    # Sentence indices whose output file may be half-written right now. A cooperative
    # stop (SIGTERM → SystemExit, see worker.py) can land mid-conversion; the except
    # below deletes these so resume re-renders them instead of keeping a truncated
    # file that passes the >1KB resume check.
    in_flight: list[int] = []
    sentences_dir_for_cleanup: str | None = None
    num_takes = max(1, int(num_takes))
    try:
        # Load session state — use override path if provided (for cached sessions
        # that aren't in the default e2a tmp dir)
        session_dir = session_dir_override or os.path.join(tmp_dir, f"ebook-{session_id}")
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
        sentences_dir_for_cleanup = session['sentences_dir']

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

        # Determine the work list. Discrete-index mode (BookForge "Correct
        # Sentences" regeneration) processes an explicit, possibly-scattered set
        # of indices; otherwise we process the contiguous [start, end] range.
        # Default book renders never pass sentence_indices, so their behavior is
        # unchanged.
        if sentence_indices is not None:
            for idx in sentence_indices:
                if idx < 0 or idx >= len(all_sentences):
                    return {'success': False, 'error': f'Invalid sentence index {idx} (total: {len(all_sentences)})'}
            work_indices = list(sentence_indices)
        else:
            if sentence_start < 0 or sentence_end >= len(all_sentences):
                return {'success': False, 'error': f'Invalid sentence range: {sentence_start}-{sentence_end} (total: {len(all_sentences)})'}
            work_indices = list(range(sentence_start, sentence_end + 1))

        print(f"[WORKER] Processing {len(work_indices)} sentence(s) on {session['device'].upper()}")
        print(f"[WORKER] TTS engine: {session['tts_engine']}, fine_tuned: {session['fine_tuned']}")

        # DEBUG: Dump full session dict for memory debugging
        print(f"[WORKER DEBUG] Full session dict:")
        for k, v in sorted(session.items()):
            if k not in ('chapter_sentences',):  # Skip large data
                print(f"  {k}: {v}")

        # Dynamically register only the needed TTS engine (saves ~20GB memory)
        register_tts_engine(session['tts_engine'])
        log_memory("After register_tts_engine")

        # Initialize TTS engine
        log_memory("Before TTSManager init")
        tts_manager = TTSManager(session)
        log_memory("After TTSManager init (model loaded)")

        # Process sentences. For num_takes > 1 (BookForge "Correct Sentences"
        # re-roll), generate each index num_takes times in ONE model load, writing
        # each take into its own take{k}/ subdir of the target sentences_dir so the
        # {i}.flac files don't overwrite each other. Sampling is stochastic (no seed),
        # so every take is a genuinely different reading of the same sentence.
        base_sentences_dir = session['sentences_dir']
        # Write take{k}/ subdirs whenever there's >1 take OR explicit per-take temps
        # (so a single-temp "long override" take still lands in take0/, matching how the
        # BookForge bridge collects candidates).
        multi_take = num_takes > 1 or bool(take_temperatures)
        total_to_process = len(work_indices) * num_takes
        total_sentences = state['total_sentences']
        processed = 0
        skipped = 0
        failed = []  # sentence indices whose conversion failed (any take)
        start_time = time.time()

        def _write_empty_sentence_silence(output_file: str):
            # Empty sentence — WRITE a short silence clip at the expected path.
            # Sentence indices are positional: assembly (combine_audio_sentences)
            # and detect_completed_chapters require '{i}.<ext>' to EXIST for every
            # index, so skipping without writing anything left a permanently
            # un-assemblable hole. Writing silence matches the engines' own
            # convention for text that renders to nothing (orpheus._write_silence,
            # f5/voxtral empty-text handling): 0.1s of zeros at the engine's
            # sample rate, saved via torchaudio in default_audio_proc_format.
            # Note: a digital-silence FLAC is ~100 bytes, below the 1024-byte
            # min_file_size used by scan_completed_sentences, so resume scans
            # still list this index — the rewrite here is idempotent and cheap;
            # assembly correctness only needs the file to exist.
            import torchaudio
            samplerate = tts_manager.engine.params['samplerate']
            silence = torch.zeros(1, int(samplerate * 0.1))
            torchaudio.save(output_file, silence, samplerate, format=default_audio_proc_format)

        for take in range(num_takes):
            # Each take writes to its own dir when multi_take, so same-index files
            # don't collide (and don't skip each other via the >1KB resume check).
            pass_dir = os.path.join(base_sentences_dir, f'take{take}') if multi_take else base_sentences_dir
            os.makedirs(pass_dir, exist_ok=True)
            session['sentences_dir'] = pass_dir
            sentences_dir_for_cleanup = pass_dir

            # Per-take temperature (BookForge "Correct Sentences" entropy): give each take
            # a different sampling temperature IN THE SAME model load, so re-rolls are
            # genuinely varied rather than near-identical. make_sampler reads
            # engine.TEMPERATURE fresh per batch, so setting it here takes effect for this
            # take. Orpheus exposes TEMPERATURE; engines that don't just use their default.
            if take_temperatures and take < len(take_temperatures):
                _temp = float(take_temperatures[take])
                if hasattr(tts_manager.engine, 'TEMPERATURE'):
                    tts_manager.engine.TEMPERATURE = _temp
                    print(f"[WORKER] Take {take}: sampling temperature = {_temp}")
                else:
                    print(f"[WORKER] Take {take}: engine has no TEMPERATURE knob; using default")

            # Batched path: engines like Orpheus (vLLM) convert many sentences in one
            # call — faster, and it avoids the host-RAM creep of tens of thousands of
            # single-prompt calls. We still emit ONE "Converting sentence" line per
            # sentence so BookForge's stdout progress counter stays accurate.
            use_batch = tts_manager.supports_batch and tts_manager.batch_size > 1
            if use_batch:
                batch_size = tts_manager.batch_size
                # Flush threshold. Usually == batch_size, but an engine may request a
                # deeper POOL: Orpheus on MLX buckets a batch by prompt length before
                # generating (padding safety), and bucketing exactly batch_size rows
                # shatters them into part-width batches — which is where MLX throughput
                # lives (~12 vs ~28 sent/min). Pooling several batches' worth lets it
                # rebuild full-width buckets; it still caps each generated batch at
                # batch_size, so peak memory is unchanged. vLLM keeps flushing at
                # batch_size (it does its own continuous batching internally).
                pool_size = max(batch_size, getattr(tts_manager, 'batch_pool_size', batch_size))
                if take == 0:
                    if pool_size != batch_size:
                        print(f"[WORKER] Batched inference enabled (batch size {batch_size}, "
                              f"pooling {pool_size} sentences per call)")
                    else:
                        print(f"[WORKER] Batched inference enabled (batch size {batch_size})")
                first_logged = False
                pending = []  # (sentence_index, sentence)

                def _flush_batch():
                    nonlocal processed, first_logged
                    if not pending:
                        return
                    # in_flight covers the WHOLE pending flush: a cooperative stop
                    # deletes these outputs so resume re-renders them cleanly. With a
                    # pool the flush is larger, so a stop discards a bit more finished
                    # work — correctness is unaffected (they're just re-rendered), and
                    # the engine gives no per-row completion signal to narrow it.
                    in_flight[:] = [idx for idx, _ in pending]
                    results = tts_manager.convert_sentences_batch(pending)
                    in_flight.clear()
                    for (idx, _), ok in zip(pending, results):
                        if not ok:
                            print(f"[WORKER] Warning: Failed to convert sentence {idx}")
                            # Keep processing remaining batches, but record the failure —
                            # the final result must report success=False so the caller
                            # doesn't treat a run with holes as complete.
                            failed.append(idx)
                        processed += 1
                        progress_pct = (processed / total_to_process) * 100
                        print(f"Converting sentence {idx}/{total_sentences} ({progress_pct:.1f}%)")
                        if not first_logged:
                            log_memory("After first sentence TTS")
                            first_logged = True
                    pending.clear()

                for i in work_indices:
                    # Skip already-rendered (resume) and empty sentences — same as the
                    # per-sentence path; empties never reach the batch.
                    output_file = os.path.join(session['sentences_dir'], f'{i}.{default_audio_proc_format}')
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 1024:
                        skipped += 1
                        processed += 1
                        continue
                    sentence = sentence_overrides.get(i, all_sentences[i]) if sentence_overrides else all_sentences[i]
                    if not sentence or not sentence.strip():
                        # See _write_empty_sentence_silence: empties must still produce
                        # a file at their index or the book is un-assemblable.
                        _write_empty_sentence_silence(output_file)
                        skipped += 1
                        processed += 1
                        continue
                    pending.append((i, sentence))
                    if len(pending) >= pool_size:
                        _flush_batch()
                _flush_batch()
            else:
                for i in work_indices:
                    # Check if already exists (for resume)
                    output_file = os.path.join(session['sentences_dir'], f'{i}.{default_audio_proc_format}')
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 1024:
                        skipped += 1
                        processed += 1
                        continue

                    sentence = sentence_overrides.get(i, all_sentences[i]) if sentence_overrides else all_sentences[i]
                    if not sentence or not sentence.strip():
                        # See _write_empty_sentence_silence: empties must still produce
                        # a file at their index or the book is un-assemblable.
                        _write_empty_sentence_silence(output_file)
                        skipped += 1
                        processed += 1
                        continue

                    # Show progress (global sentence position)
                    progress_pct = (processed / total_to_process) * 100
                    print(f"Converting sentence {i}/{total_sentences} ({progress_pct:.1f}%)")

                    # Convert sentence to audio
                    in_flight[:] = [i]
                    success = tts_manager.convert_sentence2audio(i, sentence)
                    in_flight.clear()
                    if not success:
                        print(f"[WORKER] Warning: Failed to convert sentence {i}")
                        # Keep processing the remaining sentences, but record the
                        # failure — the final result must report success=False so the
                        # caller doesn't treat a run with holes as complete.
                        failed.append(i)

                    processed += 1

                    # Log memory after first sentence
                    if processed == 1:
                        log_memory("After first sentence TTS")

                    # Periodic memory cleanup (every 10 sentences for MPS memory pressure)
                    memory_cleanup(processed, interval=10)

        elapsed = time.time() - start_time
        actual_converted = processed - skipped
        log_memory("After all sentences processed")

        print(f"[WORKER] Completed: {actual_converted} converted, {skipped} skipped in {elapsed:.1f}s")
        if failed:
            print(f"[WORKER] ERROR: {len(failed)} sentence(s) failed to convert: {failed}")

        result = {
            'success': not failed,
            'session_id': session_id,
            'sentences_processed': processed,
            'sentences_converted': actual_converted,
            'sentences_skipped': skipped,
            'sentences_failed': len(failed),
            'failed_indices': failed,
            'sentence_start': sentence_start,
            'sentence_end': sentence_end,
            'elapsed_seconds': elapsed
        }
        if failed:
            result['error'] = f"{len(failed)} sentence(s) failed to convert: {failed}"
        return result

    except (KeyboardInterrupt, SystemExit):
        # Cooperative stop (SIGTERM → SystemExit from worker.py's handler): delete any
        # half-written in-flight outputs so resume re-renders them, then RE-RAISE so
        # the interpreter exits normally — finally/atexit blocks run (orpheus.py's
        # CUDA cleanup) and the GPU is released from inside the process.
        if sentences_dir_for_cleanup and in_flight:
            for idx in in_flight:
                p = os.path.join(sentences_dir_for_cleanup, f'{idx}.{default_audio_proc_format}')
                try:
                    os.remove(p)
                except OSError:
                    pass
            print(f"[WORKER] Stop requested — dropped {len(in_flight)} in-flight output(s); exiting cleanly")
        else:
            print("[WORKER] Stop requested — exiting cleanly")
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

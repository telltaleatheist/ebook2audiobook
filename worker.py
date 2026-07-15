#!/usr/bin/env python3
"""
Lightweight Worker Entry Point for Parallel TTS Processing

This script provides a minimal-memory entry point for TTS workers.
Unlike app.py which imports gradio and all dependencies (~25GB memory),
this script only imports what's needed for TTS (~8GB memory).

Usage:
    python worker.py --session SESSION_ID --sentence_start 0 --sentence_end 999
    python worker.py --session SESSION_ID --chapter_start 1 --chapter_end 5

The worker loads session state from session-state.json (created by --prep_only)
and processes only the assigned sentence/chapter range.
"""

import argparse
import json
import signal
import sys
import os

# Add lib to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration (lightweight, no heavy deps)
from lib.conf import devices, default_device


def _graceful_exit(signum, frame):
    """Cooperative shutdown: SIGTERM/SIGINT → SystemExit(143).

    Without this handler Python's default SIGTERM disposition kills the process
    WITHOUT running atexit hooks, so torch/vLLM never release the GPU — the zombie
    then collides with the next job, and force-killing a process stuck in a WSL dxg
    GPU wait is what kernel-wedges the whole WSL VM. Raising SystemExit instead
    unwinds through the sentence loop (worker_core drops any half-written in-flight
    outputs), runs finally/atexit blocks (orpheus.py's CUDA cleanup), and exits 143 —
    the GPU is released from INSIDE the process, no force-kill needed.
    """
    print(f"[WORKER] Signal {signum} received — shutting down cleanly (releasing GPU)...", flush=True)
    raise SystemExit(143)


# Installed at import time so even a TERM during the heavy imports/model load exits
# cleanly. (On native Windows taskkill bypasses signals — this is for POSIX/WSL.)
signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT, _graceful_exit)


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight TTS worker for parallel audiobook conversion',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required arguments
    parser.add_argument('--session', type=str, required=True,
                       help='Session ID from prep phase')
    parser.add_argument('--session_dir', type=str, default=None,
                       help='Session directory path (overrides default tmp location)')

    # Sentence mode arguments
    parser.add_argument('--sentence_start', type=int, default=None,
                       help='First sentence index to process (0-indexed)')
    parser.add_argument('--sentence_end', type=int, default=None,
                       help='Last sentence index to process (0-indexed)')
    parser.add_argument('--sentence_indices', type=str, default=None,
                       help='Comma-separated explicit sentence indices to (re)generate '
                            '(0-indexed). Overrides --sentence_start/--sentence_end. Used by '
                            'the BookForge "Correct Sentences" flow to regenerate a scattered '
                            'set of sentences in one warm model load.')
    parser.add_argument('--num_takes', type=int, default=1,
                       help='Generate each target sentence this many times in ONE model load '
                            '(BookForge "Correct Sentences" re-roll). When >1, each take is '
                            'written to a take{k}/ subdir of --sentences_dir. Sampling is '
                            'stochastic, so every take differs. Default 1 (normal render).')
    parser.add_argument('--take_temperatures', type=str, default=None,
                       help='Comma-separated per-take sampling temperatures (e.g. "0.4,0.8,1.0"). '
                            'Sets num_takes = the count; each take renders at its own temperature '
                            'in ONE model load (Orpheus), for genuinely varied "Correct Sentences" '
                            're-rolls.')
    parser.add_argument('--sentence_overrides', type=str, default=None,
                       help='Path to a JSON file mapping sentence index -> replacement text '
                            '(BookForge "Correct Sentences" text edits). Those indices render the '
                            'given text instead of the cached sentence. Overlong edits are split '
                            'and re-merged by the engine, so they still produce one {i} file.')

    # Chapter mode arguments
    parser.add_argument('--chapter_start', type=int, default=None,
                       help='First chapter to process (1-indexed)')
    parser.add_argument('--chapter_end', type=int, default=None,
                       help='Last chapter to process (1-indexed)')

    # Optional arguments
    parser.add_argument('--device', type=str, default=default_device,
                       choices=list(devices.keys()),
                       help=f'Processor device (default: {default_device})')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for audiobooks')
    parser.add_argument('--tts_engine', type=str, default=None,
                       help='TTS engine override (default: from session state)')
    parser.add_argument('--fine_tuned', type=str, default=None,
                       help='Fine-tuned model override (default: from session state)')
    parser.add_argument('--voice', type=str, default=None,
                       help='Voice file override (default: from session state)')
    parser.add_argument('--output_format', type=str, default=None,
                       help='Output format override (default: from session state)')
    parser.add_argument('--speed', type=float, default=None,
                       help='TTS speed override for XTTS (default: from session state)')
    parser.add_argument('--custom_model', type=str, default=None,
                       help='Custom model NAME for a pre-staged user voice (default: from session state)')
    parser.add_argument('--custom_model_dir', type=str, default=None,
                       help='Staging root for a pre-staged custom model (default: from session state)')
    parser.add_argument('--orpheus_model_dir', type=str, default=None,
                       help='Absolute path to a folder-discovered custom Orpheus model; '
                            'its folder name is the voice token (default: from session state)')
    parser.add_argument('--sentences_dir', type=str, default=None,
                       help='Override the folder where sentence audio is written and '
                            'read for skip/resume. When set, this is the single '
                            'authoritative sentence store (BookForge project cache); '
                            'existing {i}.<ext> files are skipped (default: from session state)')

    args = parser.parse_args()

    # Parse the optional explicit index list (BookForge "Correct Sentences").
    sentence_indices = None
    if args.sentence_indices is not None:
        try:
            sentence_indices = [int(x) for x in args.sentence_indices.split(',') if x.strip() != '']
        except ValueError:
            print("Error: --sentence_indices must be a comma-separated list of integers")
            sys.exit(1)
        if not sentence_indices:
            print("Error: --sentence_indices was provided but empty")
            sys.exit(1)

    # Parse optional per-take temperatures; their count sets num_takes.
    take_temperatures = None
    if args.take_temperatures is not None:
        try:
            take_temperatures = [float(x) for x in args.take_temperatures.split(',') if x.strip() != '']
        except ValueError:
            print("Error: --take_temperatures must be a comma-separated list of numbers")
            sys.exit(1)
        if not take_temperatures:
            print("Error: --take_temperatures was provided but empty")
            sys.exit(1)
    effective_num_takes = len(take_temperatures) if take_temperatures else args.num_takes

    # Parse optional per-index text overrides (edited sentences).
    sentence_overrides = None
    if args.sentence_overrides is not None:
        try:
            with open(args.sentence_overrides, 'r', encoding='utf-8') as f:
                _ov = json.load(f)
            sentence_overrides = {int(k): str(v) for k, v in _ov.items()}
        except Exception as e:
            print(f"Error: failed to read --sentence_overrides: {e}")
            sys.exit(1)

    # Validate mode. Discrete-index mode counts as sentence mode.
    sentence_mode = (args.sentence_start is not None and args.sentence_end is not None) or sentence_indices is not None
    chapter_mode = args.chapter_start is not None and args.chapter_end is not None

    if not sentence_mode and not chapter_mode:
        print("Error: Must specify --sentence_start/--sentence_end, --sentence_indices, or --chapter_start/--chapter_end")
        sys.exit(1)

    if sentence_mode and chapter_mode:
        print("Error: Cannot specify both sentence and chapter modes")
        sys.exit(1)

    # Now import the worker core (after arg parsing to fail fast on bad args)
    # This is where the heavy imports happen
    # TEST Step 2: Using bookforge_ext worker_core to test if extension causes memory growth
    print(f"[WORKER] Loading TTS worker core...")
    from bookforge_ext.parallel.worker_core import run_worker_tts

    # Build args dict for worker
    worker_args = {
        'device': args.device.lower(),
        'output_dir': args.output_dir,
        'tts_engine': args.tts_engine,
        'fine_tuned': args.fine_tuned,
        'voice': args.voice,
        'output_format': args.output_format,
        'speed': args.speed,
        'custom_model': args.custom_model,
        'custom_model_dir': args.custom_model_dir,
        'orpheus_model_dir': args.orpheus_model_dir,
        'sentences_dir': args.sentences_dir,
    }

    # Run TTS conversion
    result = run_worker_tts(
        session_id=args.session,
        sentence_start=args.sentence_start,
        sentence_end=args.sentence_end,
        args=worker_args,
        chapter_start=args.chapter_start,
        chapter_end=args.chapter_end,
        session_dir_override=args.session_dir,
        sentence_indices=sentence_indices,
        num_takes=effective_num_takes,
        take_temperatures=take_temperatures,
        sentence_overrides=sentence_overrides
    )

    # Output result as JSON (for parsing by caller)
    print(json.dumps(result))

    # Exit with appropriate code
    sys.exit(0 if result.get('success') else 1)


if __name__ == '__main__':
    main()

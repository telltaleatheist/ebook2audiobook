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
import sys
import os

# Add lib to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration (lightweight, no heavy deps)
from lib.conf import devices, default_device


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight TTS worker for parallel audiobook conversion',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required arguments
    parser.add_argument('--session', type=str, required=True,
                       help='Session ID from prep phase')

    # Sentence mode arguments
    parser.add_argument('--sentence_start', type=int, default=None,
                       help='First sentence index to process (0-indexed)')
    parser.add_argument('--sentence_end', type=int, default=None,
                       help='Last sentence index to process (0-indexed)')

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

    args = parser.parse_args()

    # Validate mode
    sentence_mode = args.sentence_start is not None and args.sentence_end is not None
    chapter_mode = args.chapter_start is not None and args.chapter_end is not None

    if not sentence_mode and not chapter_mode:
        print("Error: Must specify either --sentence_start/--sentence_end or --chapter_start/--chapter_end")
        sys.exit(1)

    if sentence_mode and chapter_mode:
        print("Error: Cannot specify both sentence and chapter modes")
        sys.exit(1)

    # Now import the worker core (after arg parsing to fail fast on bad args)
    # This is where the heavy imports happen
    print(f"[WORKER] Loading TTS worker core...")
    from lib.worker_core import run_worker_tts

    # Build args dict for worker
    worker_args = {
        'device': args.device.lower(),
        'output_dir': args.output_dir,
        'tts_engine': args.tts_engine,
        'fine_tuned': args.fine_tuned,
        'voice': args.voice,
        'output_format': args.output_format,
    }

    # Run TTS conversion
    result = run_worker_tts(
        session_id=args.session,
        sentence_start=args.sentence_start,
        sentence_end=args.sentence_end,
        args=worker_args,
        chapter_start=args.chapter_start,
        chapter_end=args.chapter_end
    )

    # Output result as JSON (for parsing by caller)
    print(json.dumps(result))

    # Exit with appropriate code
    sys.exit(0 if result.get('success') else 1)


if __name__ == '__main__':
    main()

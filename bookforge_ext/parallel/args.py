"""
Parallel Worker Argument Definitions

Defines command-line arguments for parallel TTS processing.
These are added to the main argparse parser when BookForge extension is loaded.
"""

# List of parallel option names (for validation in app.py)
PARALLEL_OPTIONS = [
    '--prep_only', '--worker_mode', '--assemble_only',
    '--sentence_start', '--sentence_end', '--chapter_start', '--chapter_end',
    '--resume_session', '--list_sessions', '--no_split', '--chapters', '--skip_deps'
]


def add_arguments(parser):
    """
    Add parallel worker arguments to the argparse parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parallel_group = parser.add_argument_group(
        '**** Parallel worker options (for external coordination)'
    )

    parallel_group.add_argument(
        '--prep_only', action='store_true',
        help='Phase 1: Parse EPUB and prepare session state for parallel workers. '
             'Returns JSON with session_id and sentence counts.'
    )

    parallel_group.add_argument(
        '--worker_mode', action='store_true',
        help='Phase 2: Run TTS for assigned sentence/chapter range only. '
             'Requires --session and range parameters.'
    )

    parallel_group.add_argument(
        '--assemble_only', action='store_true',
        help='Phase 3: Combine sentence audio files into final audiobook. '
             'Requires --session.'
    )

    parallel_group.add_argument(
        '--sentence_start', type=int, default=None,
        help='(worker_mode) First sentence index to process (0-indexed, inclusive)'
    )

    parallel_group.add_argument(
        '--sentence_end', type=int, default=None,
        help='(worker_mode) Last sentence index to process (0-indexed, inclusive)'
    )

    parallel_group.add_argument(
        '--chapter_start', type=int, default=None,
        help='(worker_mode) First chapter to process (1-indexed, inclusive)'
    )

    parallel_group.add_argument(
        '--chapter_end', type=int, default=None,
        help='(worker_mode) Last chapter to process (1-indexed, inclusive)'
    )

    parallel_group.add_argument(
        '--resume_session', type=str, default=None,
        help='Resume a partially completed session. Provide session ID or directory path.'
    )

    parallel_group.add_argument(
        '--list_sessions', action='store_true',
        help='List all resumable sessions with incomplete TTS conversion.'
    )

    parallel_group.add_argument(
        '--no_split', action='store_true',
        help='(assemble_only) Disable splitting output into parts.'
    )

    parallel_group.add_argument(
        '--chapters', type=str, default=None,
        help='(assemble_only) Specify which chapters to include. '
             'Formats: "1-5" (range), "1,3,5" (list), "auto" (detect completed chapters). '
             'Default: all chapters.'
    )

    parallel_group.add_argument(
        '--skip_deps', action='store_true',
        help='Skip dependency/device package checks. Use when deps are already installed.'
    )

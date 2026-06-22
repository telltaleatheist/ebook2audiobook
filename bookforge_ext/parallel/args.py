"""
Parallel Worker Argument Definitions

Defines command-line arguments for parallel TTS processing.
These are added to the main argparse parser when BookForge extension is loaded.
"""

# List of parallel option names (for validation in app.py)
PARALLEL_OPTIONS = [
    '--prep_only', '--worker_mode', '--assemble_only',
    '--sentence_start', '--sentence_end', '--chapter_start', '--chapter_end',
    '--resume_session', '--list_sessions', '--no_split', '--chapters', '--skip_deps',
    '--bilingual', '--bilingual_pause', '--bilingual_gap', '--skip_assembly',
    '--sentence_per_paragraph', '--skip_headings', '--session_dir', '--custom_model_dir',
    '--sentences_dir'
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

    # Bilingual audiobook options
    parallel_group.add_argument(
        '--bilingual', action='store_true',
        help='(assemble_only) Bilingual mode: pairs alternating source/target sentences '
             'with pauses between them. Sentence 0 is source, 1 is target, 2 is source, etc.'
    )

    parallel_group.add_argument(
        '--bilingual_pause', type=float, default=0.3,
        help='(bilingual) Pause duration in seconds between source and target sentence. '
             'Default: 0.3'
    )

    parallel_group.add_argument(
        '--bilingual_gap', type=float, default=1.0,
        help='(bilingual) Gap duration in seconds between sentence pairs. '
             'Default: 1.0'
    )

    parallel_group.add_argument(
        '--skip_assembly', action='store_true',
        help='Skip the assembly phase - only generate sentence audio files. '
             'Returns the sentences directory path for external assembly. '
             'Used for dual-voice bilingual workflows where assembly happens later.'
    )

    parallel_group.add_argument(
        '--sentence_per_paragraph', action='store_true',
        help='Treat each HTML paragraph (<p> tag) as a single sentence. '
             'Disables automatic sentence splitting. Use for pre-segmented EPUBs '
             'where paragraph boundaries should be preserved exactly.'
    )

    parallel_group.add_argument(
        '--skip_headings', action='store_true',
        help='Skip reading heading tags (h1-h4) as audio. Use for bilingual EPUBs '
             'where chapter/section headings should not be narrated.'
    )

    parallel_group.add_argument(
        '--session_dir', type=str, default=None,
        help='(assemble_only) Explicit path to session directory. '
             'Overrides default tmp_dir/ebook-{session} location. '
             'Used when session has been cached to an external location.'
    )

    parallel_group.add_argument(
        '--sentences_dir', type=str, default=None,
        help='(assemble_only) Override ONLY the sentence-audio source folder, while '
             'keeping chapter mapping / metadata / VTT from the session state. The '
             'folder must contain sentence files named {i}.<ext> matching the session. '
             'Used by BookForge to assemble a post-processed sentence set (e.g. RVC '
             'voice-enhanced renders) without disturbing the cached original sentences.'
    )

    parallel_group.add_argument(
        '--custom_model_dir', type=str, default=None,
        help='(BookForge) Root of a PRE-STAGED custom model layout '
             '(custom_model_dir/<engine>/<custom_model>/{config.json,model.pth,vocab.json}). '
             'When set, --custom_model is treated as the model NAME and no zip '
             'extraction is performed — the files are loaded in place. Used for '
             'user-added fine-tuned XTTS voices in full audiobook generation.'
    )

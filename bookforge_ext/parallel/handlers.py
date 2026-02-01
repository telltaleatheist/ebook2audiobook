"""
Parallel Mode Handlers

Dispatches parallel mode operations (prep_only, worker_mode, assemble_only).
These handlers are called from app.py after args are parsed.
"""

import json
import os
import sys


def dispatch(args: dict, core_module) -> dict | None:
    """
    Dispatch to the appropriate parallel mode handler.

    Args:
        args: Parsed command-line arguments
        core_module: The imported lib.core module

    Returns:
        dict with result if a parallel mode was handled, None otherwise
    """
    from lib.conf import audiobooks_cli_dir, devices
    from lib.conf_models import TTS_ENGINES

    # Import session functions from this extension
    from . import session

    # Handle --list_sessions
    if args.get('list_sessions'):
        sessions = session.list_resumable_sessions()
        print(json.dumps(sessions, indent=2))
        return {'success': True, 'sessions': sessions}

    # Handle --resume_session
    if args.get('resume_session'):
        args['audiobooks_dir'] = (
            os.path.abspath(args['output_dir'])
            if args.get('output_dir')
            else audiobooks_cli_dir
        )
        result = session.resume_session(args)
        print(json.dumps(result, indent=2))
        return result

    # Handle --prep_only
    if args.get('prep_only'):
        if not args.get('ebook'):
            print('Error: --prep_only requires --ebook')
            return {'success': False, 'error': '--prep_only requires --ebook'}

        args['ebook'] = os.path.abspath(args['ebook'])
        args['audiobooks_dir'] = (
            os.path.abspath(args['output_dir'])
            if args.get('output_dir')
            else audiobooks_cli_dir
        )
        args['device'] = (
            devices.get(args['device'].upper(), {}).get('proc')
            or devices['CPU']['proc']
        )
        args['tts_engine'] = (
            TTS_ENGINES.get(args['tts_engine'])
            if args.get('tts_engine') in TTS_ENGINES
            else args.get('tts_engine')
        )

        result = session.prep_ebook_info(args, core_module)
        if result:
            print(json.dumps(result, indent=2, default=str))
            return result
        else:
            error_result = {'success': False, 'error': 'prep_ebook_info failed'}
            print(json.dumps(error_result))
            return error_result

    # Handle --worker_mode
    if args.get('worker_mode'):
        if not args.get('session'):
            print('Error: --worker_mode requires --session')
            return {'success': False, 'error': '--worker_mode requires --session'}

        sentence_mode = (
            args.get('sentence_start') is not None and
            args.get('sentence_end') is not None
        )
        chapter_mode = (
            args.get('chapter_start') is not None and
            args.get('chapter_end') is not None
        )

        if not sentence_mode and not chapter_mode:
            print('Error: --worker_mode requires --sentence_start/--sentence_end '
                  'or --chapter_start/--chapter_end')
            return {
                'success': False,
                'error': '--worker_mode requires sentence or chapter range'
            }

        args['audiobooks_dir'] = (
            os.path.abspath(args['output_dir'])
            if args.get('output_dir')
            else audiobooks_cli_dir
        )
        args['device'] = (
            devices.get(args['device'].upper(), {}).get('proc')
            or devices['CPU']['proc']
        )
        args['tts_engine'] = (
            TTS_ENGINES.get(args['tts_engine'])
            if args.get('tts_engine') in TTS_ENGINES
            else args.get('tts_engine')
        )

        result = session.worker_only(args, core_module)
        print(json.dumps(result, indent=2))
        return result

    # Handle --assemble_only
    if args.get('assemble_only'):
        if not args.get('session'):
            print('Error: --assemble_only requires --session')
            return {'success': False, 'error': '--assemble_only requires --session'}

        from lib.conf import default_output_split_hours, default_output_channel

        args['audiobooks_dir'] = (
            os.path.abspath(args['output_dir'])
            if args.get('output_dir')
            else audiobooks_cli_dir
        )
        args['output_split'] = not args.get('no_split', False)
        args['output_split_hours'] = default_output_split_hours
        args['output_channel'] = args.get('output_channel', default_output_channel)

        result = session.assemble_audiobook(args, core_module)
        print(json.dumps(result, indent=2))
        return result

    # No parallel mode requested
    return None

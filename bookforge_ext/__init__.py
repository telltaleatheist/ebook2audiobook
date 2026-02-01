"""
BookForge Extension for ebook2audiobook

This module contains all BookForge-specific customizations to the e2a pipeline,
isolated from upstream code for easier maintenance and merging.

Features:
- Parallel TTS processing (prep_only, worker_mode, assemble_only)
- Session state management for multi-worker coordination
- Lightweight worker entry point

Usage:
    # In app.py, after imports:
    try:
        from bookforge_ext import hooks, parallel
        BOOKFORGE_EXT = True
    except ImportError:
        BOOKFORGE_EXT = False

    # After creating parser:
    if BOOKFORGE_EXT:
        parallel.add_arguments(parser)

    # After args parsed:
    if BOOKFORGE_EXT:
        result = hooks.handle_parallel_mode(args, core_module)
        if result is not None:
            sys.exit(0 if result.get('success', True) else 1)
"""

from . import hooks
from . import parallel

__version__ = '1.0.0'
__all__ = ['hooks', 'parallel']

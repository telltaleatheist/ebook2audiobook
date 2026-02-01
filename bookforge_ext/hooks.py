"""
Hook Registry for BookForge Extension

Provides a simple hook system for extending e2a without modifying core files.
Core files register accessors (like get_context), and the extension uses them.
"""

_hooks = {}


def register(name: str, func):
    """Register a hook function by name."""
    _hooks[name] = func


def call(name: str, *args, **kwargs):
    """Call a registered hook and return its result, or None if not registered."""
    if name in _hooks:
        return _hooks[name](*args, **kwargs)
    return None


def is_registered(name: str) -> bool:
    """Check if a hook is registered."""
    return name in _hooks


def handle_parallel_mode(args: dict, core_module) -> dict | None:
    """
    Main entry point for parallel worker modes.

    Called from app.py after args are parsed. Returns:
    - dict with 'success' key if a parallel mode was handled
    - None if no parallel mode requested (normal operation continues)

    Args:
        args: Parsed command-line arguments
        core_module: The imported lib.core module (for accessing context, etc.)

    Returns:
        dict with result if parallel mode handled, None otherwise
    """
    from .parallel import handlers
    return handlers.dispatch(args, core_module)


def should_skip_deps(args: dict) -> bool:
    """
    Check if dependency checks should be skipped.

    Called from app.py before dependency installation.
    Returns True for lightweight parallel operations.
    """
    return (
        args.get('skip_deps') or
        args.get('list_sessions') or
        args.get('prep_only') or
        args.get('worker_mode') or
        args.get('assemble_only') or
        args.get('resume_session')
    )

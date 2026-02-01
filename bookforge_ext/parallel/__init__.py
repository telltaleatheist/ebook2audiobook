"""
Parallel TTS Processing Module

This module provides the three-phase parallel TTS architecture:
  Phase 1: --prep_only    - Parse EPUB, save sentence data to session-state.json
  Phase 2: --worker_mode  - Load session state, run TTS for assigned range
  Phase 3: --assemble_only - Combine sentence audio files into final audiobook

For external coordination (e.g., BookForge), this enables:
1. Running prep to get sentence counts
2. Spawning multiple TTS workers for different sentence ranges
3. Assembling all audio files into the final M4B
"""

from .args import add_arguments, PARALLEL_OPTIONS
from .handlers import dispatch

__all__ = ['add_arguments', 'dispatch', 'PARALLEL_OPTIONS']

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
import subprocess
import sys
import os
import threading
import time

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


# ─────────────────────────────────────────────────────────────────────────────
# Parent-death watchdog
# ─────────────────────────────────────────────────────────────────────────────
#
# THE WORKER MUST NOT OUTLIVE WHOEVER STARTED IT.
#
# The handler above is only ever reached if somebody sends the signal. BookForge
# sends it from `before-quit` (killAllWorkers) — and `before-quit` does not run
# when Electron is killed rather than quit. On Sep 1 2026 the user Ctrl-C'd
# `npm run electron:dev`; the app died, nothing signalled anything, and this
# worker went on rendering as an orphan for 1h31m holding ~6 GB of weights plus
# its KV cache, writing into a stdout pipe with no reader. The next render then
# started on top of it. Three Orpheus engines on one 64 GB Mac is 55-60 GB wired,
# an OOM-killed desktop, and every throughput number from that night void.
#
# So the worker watches its own parent. Nothing outside it has to remember to.
#
# PLATFORMS, and why one poll covers all three:
#
#   macOS / Linux — an orphan is reparented (to launchd / init / a subreaper), so
#     `os.getppid()` CHANGES the moment the parent is gone. That change is the
#     whole signal; we never have to ask whether a pid we remember is alive, which
#     is the check that races against pid reuse.
#
#   WSL — the worker's parent is the shell that `wsl.exe` started for it. If
#     BookForge dies, that shell dies with it and the worker reparents inside the
#     guest exactly as on Linux. It matters that what fires here is COOPERATIVE:
#     the ladder in this file exists because SIGKILLing a guest process that is
#     parked in a dxg GPU wait wedges the whole WSL VM. A process raising
#     SystemExit inside itself is not that — it unwinds, releases the GPU, exits.
#     Nothing is force-killed from outside.
#
#   A DETACHED RUN — `nohup`, `setsid`, a CLI run whose launcher has already
#     exited: the worker's parent is pid 1 FROM THE START. There is nothing to
#     outlive and no change to detect, so the watchdog is switched off and says
#     so. This is why we compare against the ppid RECORDED AT STARTUP instead of
#     testing `getppid() == 1`: those two rules agree on every normal spawn and
#     disagree exactly here, where the second one would shoot a legitimate run in
#     the head two seconds after it began.

# WHY PPID ALONE IS NOT ENOUGH: THE WRAPPER.
#
# BookForge does not spawn this worker directly. For Orpheus it resolves a conda
# `prefix` env, so the command is `conda run --no-capture-output -p <env> python
# worker.py`. Measured on the Mac, from a spawn built exactly like the bridge's:
#
#     node (BookForge / the CLI)      pid 15840
#       └─ Miniforge3/bin/python      pid 15842   <- `conda run` itself
#            └─ /bin/bash             pid 15843   <- conda's activation shell
#                 └─ python worker.py pid 15870   <- ppid is 15843
#
# Our parent is that BASH. Kill BookForge and the wrapper pair is reparented to
# launchd and goes right on waiting for us: our ppid never changes, and a
# ppid-only watchdog sleeps through the whole thing. That is precisely the
# 1h31m zombie this file exists to prevent.
#
# So the app also NAMES ITSELF, in BOOKFORGE_OWNER_PID, and we watch that pid
# directly — through any number of wrappers. Two rules, either of which fires:
#
#   the owner is gone       `os.kill(owner, 0)` raises ProcessLookupError, or
#                           the owner's START TIME no longer matches the one we
#                           recorded. The start time is the pid-reuse guard: a
#                           pid is a small recycled integer, and "the pid still
#                           exists" is not the same claim as "it is still them".
#
#   the ppid changed        the rule above, unchanged — it is the one that still
#                           works when there is no owner pid to watch (an older
#                           app, a hand-run worker, or a WSL guest).
#
# WSL, again: BOOKFORGE_OWNER_PID is a WINDOWS pid, and the guest has its own pid
# namespace where that number means nothing (or worse, means some unrelated guest
# process). We therefore arm the owner rule only when the owner's platform and
# ours are in the same namespace, and otherwise say so and fall back to ppid —
# which is the rule that works there anyway, because the guest shell dies with
# the app.

ORPHAN_GRACE_ENV = 'ORPHEUS_WORKER_ORPHAN_GRACE_SECONDS'
ORPHAN_GRACE_DEFAULT_SECONDS = 60.0
PARENT_POLL_SECONDS = 2.0
OWNER_PID_ENV = 'BOOKFORGE_OWNER_PID'
OWNER_PLATFORM_ENV = 'BOOKFORGE_OWNER_PLATFORM'


def orphan_grace_seconds():
    """How long the cooperative stop gets before the GPU is taken back by force."""
    raw = os.environ.get(ORPHAN_GRACE_ENV, '').strip()
    if not raw:
        return ORPHAN_GRACE_DEFAULT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        print(f"[WORKER] {ORPHAN_GRACE_ENV}={raw!r} is not a number; using "
              f"{ORPHAN_GRACE_DEFAULT_SECONDS:g}s", flush=True)
        return ORPHAN_GRACE_DEFAULT_SECONDS
    return max(0.0, value)


def process_start_time(pid):
    """The owner's start time as `ps` reports it, or None if `ps` cannot say.

    THIS IS THE PID-REUSE GUARD. A pid is a small recycled integer; "pid 17311
    still exists" and "pid 17311 is still the app that spawned me" are different
    claims, and on a machine that has been up for weeks only the second one is
    worth killing a render over. `ps -o lstart=` is second-granular and stable
    for the life of the process on macOS, Linux and a WSL guest alike.

    None means "could not establish it" — the caller then falls back to the
    existence check alone rather than treating an unreadable clock as a change.
    """
    try:
        out = subprocess.run(
            ['ps', '-o', 'lstart=', '-p', str(pid)],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    stamp = out.stdout.strip()
    return stamp or None


def pid_is_alive(pid):
    """True if `pid` exists. A PermissionError means it exists and is not ours."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def resolve_owner():
    """The pid BookForge named for itself, if we can honestly watch it.

    Returns (pid, start_time) or None. Every refusal prints its reason: a
    watchdog that quietly loses half its coverage is how the zombie survived
    the first fix.
    """
    raw = os.environ.get(OWNER_PID_ENV, '').strip()
    if not raw:
        print(f"[WORKER] no {OWNER_PID_ENV} in the environment — the parent-pid rule "
              f"is the only orphan check", flush=True)
        return None

    try:
        owner = int(raw)
    except ValueError:
        print(f"[WORKER] {OWNER_PID_ENV}={raw!r} is not a pid; ignoring it", flush=True)
        return None
    if owner <= 1 or owner == os.getpid():
        print(f"[WORKER] {OWNER_PID_ENV}={owner} is not a usable owner; ignoring it", flush=True)
        return None

    # A Windows pid means nothing in a WSL guest's pid namespace — and might
    # collide with an unrelated guest process, which would be worse than having
    # no rule. Only arm it inside one namespace.
    owner_platform = os.environ.get(OWNER_PLATFORM_ENV, '').strip().lower()
    here_is_windows = sys.platform.startswith('win')
    owner_is_windows = owner_platform.startswith('win')
    if owner_platform and owner_is_windows != here_is_windows:
        print(f"[WORKER] owner pid {owner} belongs to a {owner_platform} host, not this "
              f"{sys.platform} process (WSL guest) — the parent-pid rule is the only "
              f"orphan check", flush=True)
        return None

    if not pid_is_alive(owner):
        # Either the app died between spawning us and now (rare but real), or the
        # pid is not visible from here. Both mean this rule cannot be armed; the
        # ppid rule still can, so say so and carry on rather than exiting a render
        # on an ambiguity.
        print(f"[WORKER] owner pid {owner} is not visible from here — the parent-pid "
              f"rule is the only orphan check", flush=True)
        return None

    return owner, process_start_time(owner)


def _orphan_reason(initial_ppid, owner):
    """Which rule fired, and about which pid — or None if neither has."""
    try:
        current = os.getppid()
    except OSError:
        current = initial_ppid
    if initial_ppid is not None and current != initial_ppid:
        return initial_ppid, f'reparented from {initial_ppid} to {current}'

    if owner is not None:
        owner_pid, owner_started = owner
        if not pid_is_alive(owner_pid):
            return owner_pid, f'{OWNER_PID_ENV} {owner_pid} exited'
        if owner_started is not None:
            now_started = process_start_time(owner_pid)
            if now_started is not None and now_started != owner_started:
                return owner_pid, (f'{OWNER_PID_ENV} {owner_pid} was replaced — a new '
                                   f'process reused the pid')
    return None


def _parent_watch_loop(initial_ppid, owner, poll_seconds, grace_seconds):
    """Poll both rules; on either, take the SAME path the app's stop takes."""
    while True:
        time.sleep(poll_seconds)
        fired = _orphan_reason(initial_ppid, owner)
        if fired is None:
            continue
        gone_pid, rule = fired

        print(f"[WORKER] parent process {gone_pid} is gone; shutting down "
              f"cooperatively ({rule})", flush=True)

        # Deliberately the signal, not a raise: the handler runs in the MAIN
        # thread, so the SystemExit unwinds the sentence loop, worker_core drops
        # the in-flight rows (resume re-renders them), and atexit/finally release
        # the GPU from inside the process. Raising here would only kill this
        # daemon thread and leave the render running.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as err:  # pragma: no cover - the ladder still applies
            print(f"[WORKER] could not signal self ({err}); going straight to the "
                  f"hard exit", flush=True)
            grace_seconds = 0.0

        # The main thread may be inside a long native call (an MLX batch decode,
        # a model load) and will not see the SystemExit until it returns to
        # bytecode. Give it the grace period — then take the GPU back anyway. A
        # partial row on disk is exactly what the resume path already handles;
        # a wedged 6 GB engine with no owner is not.
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.01, poll_seconds)))
        print(f"[WORKER] still running {grace_seconds:g}s after the orphan stop — "
              f"forcing exit 143 to release the GPU", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(143)


def start_parent_watch(poll_seconds=PARENT_POLL_SECONDS, grace_seconds=None):
    """Start the parent-death watchdog.

    Returns {'parent': ppid_or_None, 'owner': owner_pid_or_None} naming which
    rules are armed, or None if neither could be — in which case nothing is
    watching and the reason has been printed.

    Started from main(), not at import, so importing this module (the test does)
    never leaves a thread behind.
    """
    getppid = getattr(os, 'getppid', None)
    initial_ppid = None
    if getppid is None:  # native Windows has no reparenting to observe
        print("[WORKER] no getppid() on this platform — the parent-pid rule is off",
              flush=True)
    else:
        ppid = getppid()
        if ppid <= 1:
            print(f"[WORKER] started with no parent (ppid {ppid}) — a detached run has "
                  f"nothing to outlive, so the parent-pid rule is off", flush=True)
        else:
            initial_ppid = ppid

    owner = resolve_owner()

    if initial_ppid is None and owner is None:
        print("[WORKER] parent watchdog disabled: no parent to outlive and no owner "
              "pid to watch — nothing to outlive", flush=True)
        return None

    if grace_seconds is None:
        grace_seconds = orphan_grace_seconds()

    thread = threading.Thread(
        target=_parent_watch_loop,
        args=(initial_ppid, owner, poll_seconds, grace_seconds),
        name='parent-watch',
        daemon=True,
    )
    thread.start()
    armed = []
    if initial_ppid is not None:
        armed.append(f'parent pid {initial_ppid}')
    if owner is not None:
        started = 'start time recorded' if owner[1] else 'no start time available'
        armed.append(f'owner pid {owner[0]} ({started})')
    print(f"[WORKER] parent watchdog on: {', '.join(armed)}; polling every "
          f"{poll_seconds:g}s, {grace_seconds:g}s grace", flush=True)
    return {'parent': initial_ppid, 'owner': owner[0] if owner else None}


def main():
    # Before anything else, including argument parsing: if the app that started
    # this worker dies, the worker dies with it. See the block above.
    start_parent_watch()

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
    parser.add_argument('--orpheus_adapter_dir', type=str, default=None,
                       help='Absolute path to an Orpheus LoRA voice adapter (adapter mode). '
                            'Requires --orpheus_base_dir; --fine_tuned carries the voice token')
    parser.add_argument('--orpheus_base_dir', type=str, default=None,
                       help='Absolute path to the shared Orpheus base model that '
                            '--orpheus_adapter_dir is applied to (adapter mode)')
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
        'orpheus_adapter_dir': args.orpheus_adapter_dir,
        'orpheus_base_dir': args.orpheus_base_dir,
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

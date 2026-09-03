#!/usr/bin/env python
"""The worker must not outlive whoever started it.

    python tools/test_worker_parent_watch.py

WHY THIS EXISTS. worker.py already exits cleanly on SIGTERM — but only if
somebody sends one. BookForge sends it from Electron's `before-quit`, and
`before-quit` does not run when Electron is KILLED rather than quit. On
Sep 1 2026 the user Ctrl-C'd `npm run electron:dev`; the app vanished, nobody
signalled anything, and the Orpheus worker rendered on as an orphan for 1h31m
holding ~6 GB of weights and a KV cache while the next render started on top of
it. Three engines, 64 GB machine, OOM-killed desktop, every timing that night
worthless.

WHAT IS TESTED, and why each case is a real way to get this wrong:

  ORPHANED     the parent dies -> the child must notice, say so, and be GONE.
               The exit must carry 143 AND atexit must have run: that pair is
               what proves it went through worker.py's cooperative handler
               rather than being shot. It is the same path worker_core.py's
               `except (KeyboardInterrupt, SystemExit)` uses to drop in-flight
               rows so resume re-renders them, and the path that lets atexit
               release the GPU from INSIDE the process. Force-killing a GPU
               process from outside is what wedges a WSL VM.

  LIVE PARENT  the parent stays -> the child must still be rendering. A watchdog
               that fires on a healthy run is worse than no watchdog: it would
               kill every book mid-render.

  DETACHED     the child's parent is pid 1 FROM THE START (nohup / setsid / a
               CLI launcher that has already exited). There is no parent to
               outlive, so the watchdog must switch itself OFF and say so. This
               is the case that forces the rule to be "the ppid CHANGED", not
               "the ppid is 1" — the second rule would shoot a legitimate
               detached run two seconds after it started.

  HARD EXIT    the cooperative SIGTERM is swallowed (a main thread parked in a
               native call is the real version of this) -> after the grace
               period the process must exit 143 anyway. The GPU has to come
               back; a partial row on disk is what resume already handles.

  WRAPPER      THE CASE THAT ACTUALLY HAPPENED. BookForge never spawns this
               worker directly: for Orpheus it runs `conda run -p <env> python
               worker.py`, and conda run forks a python which forks an
               activation bash which forks us. Our parent is that bash. When
               BookForge dies the whole wrapper pair is reparented to launchd
               and goes on WAITING for us, so our ppid never changes and a
               ppid-only watchdog sleeps through the orphaning — 1h31m of it,
               on Sep 1 2026. So the app names itself in BOOKFORGE_OWNER_PID and
               we watch that pid THROUGH the wrapper. Here: kill the owner only;
               the wrapper must survive (proving ppid never moved) and the child
               must still be gone in seconds.

  OWNER ALIVE  the owner is fine -> the child keeps rendering, wrapper or no.

  NOT VISIBLE  BOOKFORGE_OWNER_PID names a pid this process cannot see — the
               real case is WSL, where it is a Windows pid in another namespace.
               The rule must refuse to arm, SAY so, and leave the ppid rule
               doing its job. Silently half-disabling the watchdog is how the
               first fix would have shipped with the same hole in it.

No engine, no model, no GPU: the children import worker.py (which pulls in only
lib.conf) and sit in a sleep loop standing in for the sentence loop.
"""
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='parent-watch-test-')

failures = []


def check(cond, label):
    if cond:
        print(f'  ok   {label}')
    else:
        print(f'  FAIL {label}')
        failures.append(label)


def write_script(name, body):
    path = os.path.join(TMP, name)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(textwrap.dedent(body))
    return path


# The child: the REAL worker.py watchdog over a stand-in for the sentence loop.
# stdout goes to a file, because the whole point of an orphan is that nobody is
# reading its pipe any more.
CHILD = write_script('child.py', f'''
    import atexit, os, sys, time
    sys.path.insert(0, {REPO!r})
    import worker

    log = open(sys.argv[1], 'w', buffering=1)
    sys.stdout = log
    sys.stderr = log

    # What the process exited WITH, written from an atexit hook — so a passing
    # assertion proves both the code and that atexit ran at all (which is how
    # the real worker's engine releases the GPU).
    status = {{'code': None}}
    atexit.register(lambda: (log.write('ATEXIT_RAN code=%r\\n' % status['code']), log.flush()))

    poll = float(os.environ.get('TEST_POLL', '0.25'))
    grace = float(os.environ.get('TEST_GRACE', '1.0'))

    if os.environ.get('TEST_SWALLOW_SIGTERM') == '1':
        # Stand in for a main thread parked in a native call: the cooperative
        # SystemExit never gets a chance to unwind.
        import signal as _signal
        _signal.signal(_signal.SIGTERM, lambda *a: None)

    watched = worker.start_parent_watch(poll_seconds=poll, grace_seconds=grace)
    print('WATCHDOG_RETURNED %r' % (watched,))
    print('CHILD_READY %d %d' % (os.getpid(), os.getppid()))

    try:
        deadline = time.time() + 60          # the sentence loop, minus the sentences
        while time.time() < deadline:
            time.sleep(0.05)
        print('CHILD_RAN_TO_COMPLETION')
    except SystemExit as exc:
        status['code'] = exc.code
        raise
''')

# A throwaway parent: starts the child, writes the child's pid where the test can
# see it, then waits to be killed.
PARENT = write_script('parent.py', '''
    import os, subprocess, sys, time
    child_log, pidfile = sys.argv[1], sys.argv[2]
    p = subprocess.Popen([sys.executable, os.environ['TEST_CHILD'], child_log])
    with open(pidfile, 'w') as fh:
        fh.write(str(p.pid))
    time.sleep(120)
''')

# The wrapper: stands in for `conda run` + its activation bash. It spawns the
# child and WAITS on it, exactly as conda run does — which is why the child's
# ppid is the wrapper's and stays the wrapper's when the owner dies.
WRAPPER = write_script('wrapper.py', '''
    import os, subprocess, sys, time
    child_log, pidfile = sys.argv[1], sys.argv[2]
    p = subprocess.Popen([sys.executable, os.environ['TEST_CHILD'], child_log])
    with open(pidfile, 'w') as fh:
        fh.write(str(p.pid))
    p.wait()
    time.sleep(30)   # stay up afterwards so the test can prove we survived
''')

# The owner: BookForge's stand-in. It names itself to the worker and then does
# nothing but exist, so killing it is the only event in the test.
OWNER = write_script('owner.py', '''
    import os, subprocess, sys, time
    child_log, pidfile, wrapper_pidfile = sys.argv[1], sys.argv[2], sys.argv[3]
    env = dict(os.environ)
    env['BOOKFORGE_OWNER_PID'] = str(os.getpid())
    env['BOOKFORGE_OWNER_PLATFORM'] = sys.platform
    p = subprocess.Popen([sys.executable, os.environ['TEST_WRAPPER'], child_log, pidfile],
                         env=env)
    with open(wrapper_pidfile, 'w') as fh:
        fh.write(str(p.pid))
    time.sleep(120)
''')

# A launcher that makes its child a REAL orphan before the child starts: fork,
# let the intermediate exit, and have the survivor wait until it has actually
# been reparented to pid 1 before it execs the child.
DETACHED = write_script('detached.py', '''
    import os, sys, time
    child_log = sys.argv[1]
    if os.fork() > 0:
        os._exit(0)
    for _ in range(400):
        if os.getppid() <= 1:
            break
        time.sleep(0.01)
    os.execv(sys.executable, [sys.executable, os.environ['TEST_CHILD'], child_log])
''')


def read_log(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            return fh.read()
    except FileNotFoundError:
        return ''


def wait_for(predicate, timeout, step=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_quietly(pid):
    try:
        if pid:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def start_case(name, env_extra=None, launcher=None):
    log = os.path.join(TMP, f'{name}.log')
    pidfile = os.path.join(TMP, f'{name}.pid')
    env = dict(os.environ)
    env['TEST_CHILD'] = CHILD
    env.setdefault('TEST_POLL', '0.25')
    env.setdefault('TEST_GRACE', '1.0')
    if env_extra:
        env.update(env_extra)
    parent = subprocess.Popen([sys.executable, launcher or PARENT, log, pidfile], env=env)
    return parent, log, pidfile


def child_pid_from(pidfile):
    try:
        return int(open(pidfile).read().strip())
    except Exception:
        return None


# ── 1. the parent dies ───────────────────────────────────────────────────────
print('an orphaned worker shuts itself down')
parent, log, pidfile = start_case('orphan')
child_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 20),
          'the child started and reported its pid')
    child_pid = child_pid_from(pidfile)
    check('parent watchdog on' in read_log(log),
          'the watchdog announced itself with the parent pid')

    parent.kill()
    parent.wait(timeout=10)

    check(wait_for(lambda: not pid_alive(child_pid), 15),
          'the child is gone within seconds of the parent dying')

    wait_for(lambda: 'ATEXIT_RAN' in read_log(log), 5)
    out = read_log(log)
    check(f'parent process {parent.pid} is gone; shutting down cooperatively' in out,
          'it printed ONE line naming the parent that vanished')
    check('Signal 15 received' in out,
          "it went through worker.py's own SIGTERM handler — the cooperative path, "
          'which is what makes worker_core drop its in-flight rows')
    check('ATEXIT_RAN code=143' in out,
          'it exited 143 AND atexit ran — the engine gets to release the GPU from '
          'inside the process')
    check('CHILD_RAN_TO_COMPLETION' not in out, 'it did not simply finish its loop')
    check('forcing exit 143' not in out,
          'the hard exit was NOT needed: SystemExit unwound inside the grace period')
finally:
    if parent.poll() is None:
        parent.kill()
    kill_quietly(child_pid)

# ── 2. the parent lives ──────────────────────────────────────────────────────
print('a worker whose parent is alive keeps rendering')
parent, log, pidfile = start_case('healthy')
child_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 20), 'the child started')
    child_pid = child_pid_from(pidfile)
    time.sleep(3.0)  # a dozen polls at 0.25s
    check(pid_alive(child_pid), 'still running after a dozen watchdog polls')
    check('is gone' not in read_log(log), 'the watchdog said nothing')
finally:
    if parent.poll() is None:
        parent.kill()
        parent.wait(timeout=10)
    kill_quietly(child_pid)

# ── 3. no parent to begin with ───────────────────────────────────────────────
print('a detached run (ppid 1 from the start) disables the watchdog')
log = os.path.join(TMP, 'detached.log')
env = dict(os.environ)
env.update({'TEST_CHILD': CHILD, 'TEST_POLL': '0.25', 'TEST_GRACE': '1.0'})
launcher = subprocess.Popen([sys.executable, DETACHED, log], env=env)
launcher.wait(timeout=10)
child_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 20), 'the detached child started')
    out = read_log(log)
    ready = ([l for l in out.splitlines() if l.startswith('CHILD_READY')] or [''])[0]
    check(ready.split()[2:3] == ['1'], f'its parent really is pid 1 ({ready!r})')
    check('parent watchdog disabled' in out and 'nothing to outlive' in out,
          'the watchdog switched itself off and said why')
    check('WATCHDOG_RETURNED None' in out, 'no thread was started')
    child_pid = int(ready.split()[1]) if ready else None
    time.sleep(3.0)
    check(pid_alive(child_pid),
          'and it is still running — a detached run is not an orphan to be killed')
finally:
    kill_quietly(child_pid)

# ── 4. the cooperative stop is swallowed ─────────────────────────────────────
print('a worker that cannot honour the SIGTERM is still made to release the GPU')
parent, log, pidfile = start_case('stuck', {'TEST_SWALLOW_SIGTERM': '1', 'TEST_GRACE': '1.0'})
child_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 20), 'the child started')
    child_pid = child_pid_from(pidfile)
    parent.kill()
    parent.wait(timeout=10)
    check(wait_for(lambda: not pid_alive(child_pid), 15),
          'it is gone anyway, after the grace period')
    out = read_log(log)
    check('is gone; shutting down cooperatively' in out,
          'the cooperative stop was tried first')
    check('forcing exit 143' in out,
          'and the hard exit named itself — 6 GB of weights with no owner is not a '
          'state to leave the machine in')
    check('ATEXIT_RAN' not in out,
          'os._exit skips atexit, as it must: the point is that this path is the '
          'last resort, not the normal one')
finally:
    if parent.poll() is None:
        parent.kill()
    kill_quietly(child_pid)

# ── 5. the wrapper: the owner dies, the wrapper does not ─────────────────────
print('an owner that dies behind a wrapper still takes the worker with it')
log = os.path.join(TMP, 'wrapper.log')
child_pidfile = os.path.join(TMP, 'wrapper-child.pid')
wrapper_pidfile = os.path.join(TMP, 'wrapper-wrapper.pid')
env = dict(os.environ)
env.update({'TEST_CHILD': CHILD, 'TEST_WRAPPER': WRAPPER,
            'TEST_POLL': '0.25', 'TEST_GRACE': '1.0'})
owner = subprocess.Popen([sys.executable, OWNER, log, child_pidfile, wrapper_pidfile], env=env)
child_pid = wrapper_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 25), 'the child started')
    child_pid = child_pid_from(child_pidfile)
    wrapper_pid = child_pid_from(wrapper_pidfile)
    out = read_log(log)
    ready = ([l for l in out.splitlines() if l.startswith('CHILD_READY')] or [''])[0]
    check(ready.split()[2:3] == [str(wrapper_pid)],
          f'the child\'s parent is the WRAPPER ({wrapper_pid}), not the owner '
          f'({owner.pid}) — this is the shape that defeated the ppid rule')
    check(f'owner pid {owner.pid}' in out, 'the watchdog armed the owner rule on the owner pid')

    owner.kill()
    owner.wait(timeout=10)

    check(wait_for(lambda: not pid_alive(child_pid), 15),
          'the child is gone within seconds of the OWNER dying')
    check(pid_alive(wrapper_pid),
          'and the wrapper is still alive — so the child\'s ppid never changed and '
          'the ppid rule could not have been what fired')

    wait_for(lambda: 'ATEXIT_RAN' in read_log(log), 5)
    out = read_log(log)
    check(f'parent process {owner.pid} is gone; shutting down cooperatively' in out,
          'the same line fired, naming the owner')
    check(f'BOOKFORGE_OWNER_PID {owner.pid} exited' in out,
          'and it named WHICH rule fired')
    check('reparented' not in out, 'the ppid rule stayed quiet, as it must here')
    check('ATEXIT_RAN code=143' in out,
          'same cooperative ladder: exit 143 with atexit run')
finally:
    if owner.poll() is None:
        owner.kill()
    kill_quietly(child_pid)
    kill_quietly(wrapper_pid)

# ── 6. the owner is fine ─────────────────────────────────────────────────────
print('a worker whose owner is alive keeps rendering behind the wrapper')
log = os.path.join(TMP, 'owner-alive.log')
child_pidfile = os.path.join(TMP, 'owner-alive-child.pid')
wrapper_pidfile = os.path.join(TMP, 'owner-alive-wrapper.pid')
owner = subprocess.Popen([sys.executable, OWNER, log, child_pidfile, wrapper_pidfile], env=env)
child_pid = wrapper_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 25), 'the child started')
    child_pid = child_pid_from(child_pidfile)
    wrapper_pid = child_pid_from(wrapper_pidfile)
    time.sleep(3.0)
    check(pid_alive(child_pid), 'still running after a dozen polls of both rules')
    check('is gone' not in read_log(log), 'neither rule said anything')
finally:
    if owner.poll() is None:
        owner.kill()
        owner.wait(timeout=10)
    kill_quietly(child_pid)
    kill_quietly(wrapper_pid)

# ── 7. an owner pid we cannot see (the WSL shape) ────────────────────────────
print('an owner pid that is not visible falls back instead of firing')
# 4000000 is above every pid_max we run on, so it can never exist.
parent, log, pidfile = start_case('unseen-owner', {
    'BOOKFORGE_OWNER_PID': '4000000',
    'BOOKFORGE_OWNER_PLATFORM': sys.platform,
})
child_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 20), 'the child started')
    child_pid = child_pid_from(pidfile)
    out = read_log(log)
    check('owner pid 4000000 is not visible from here' in out,
          'it said the owner pid is not visible')
    check('the parent-pid rule is the only orphan check' in out,
          'and named what it is relying on instead')
    check("'owner': None" in out, 'the owner rule was not armed')
    check("'parent':" in out and "'parent': None" not in out,
          'but the parent-pid rule still is')
    time.sleep(3.0)
    check(pid_alive(child_pid),
          'and it did NOT fire — an unseeable owner is not a dead one')
finally:
    if parent.poll() is None:
        parent.kill()
        parent.wait(timeout=10)
    kill_quietly(child_pid)

# ── 8. a Windows owner pid inside a posix guest (WSL, by name) ───────────────
print('a Windows owner pid is refused inside a posix worker')
parent, log, pidfile = start_case('cross-namespace', {
    'BOOKFORGE_OWNER_PID': str(os.getpid()),   # a pid that IS visible here
    'BOOKFORGE_OWNER_PLATFORM': 'win32',
})
child_pid = None
try:
    check(wait_for(lambda: 'CHILD_READY' in read_log(log), 20), 'the child started')
    child_pid = child_pid_from(pidfile)
    out = read_log(log)
    check('belongs to a win32 host' in out,
          'a visible-but-foreign pid is refused on the namespace, not on visibility — '
          'a Windows pid can collide with a live guest pid, and watching the wrong '
          'process is worse than watching none')
    check("'owner': None" in out, 'the owner rule was not armed')
finally:
    if parent.poll() is None:
        parent.kill()
        parent.wait(timeout=10)
    kill_quietly(child_pid)

# ── 9. the grace period ──────────────────────────────────────────────────────
print('the grace period')
sys.path.insert(0, REPO)
os.environ.pop('ORPHEUS_WORKER_ORPHAN_GRACE_SECONDS', None)
import worker  # noqa: E402  (light: lib.conf only, no torch)
check(worker.orphan_grace_seconds() == 60.0, 'defaults to 60s')
os.environ['ORPHEUS_WORKER_ORPHAN_GRACE_SECONDS'] = '5'
check(worker.orphan_grace_seconds() == 5.0, 'ORPHEUS_WORKER_ORPHAN_GRACE_SECONDS overrides it')
os.environ['ORPHEUS_WORKER_ORPHAN_GRACE_SECONDS'] = 'soon'
check(worker.orphan_grace_seconds() == 60.0, 'a junk value falls back to 60s and says so')
os.environ['ORPHEUS_WORKER_ORPHAN_GRACE_SECONDS'] = '-3'
check(worker.orphan_grace_seconds() == 0.0, 'a negative grace is 0, not a wait forever')
os.environ.pop('ORPHEUS_WORKER_ORPHAN_GRACE_SECONDS', None)

print()
if failures:
    print(f'{len(failures)} check(s) FAILED:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('All parent-watchdog checks passed.')

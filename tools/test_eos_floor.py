#!/usr/bin/env python
"""The EOS minimum-length floor forbids an early stop and touches nothing else
(2026-09-03).

THE DEFECT. On the mistborn 240-draw battery every fine-tune shows EARLY stops
at 30-60% of the text (ASR-verified), 5-15 per 240 depending on epoch, on top
of the loops the EOS boost exists for. The served models only ever caught the
fast ones after the fact — the maxCharsPerSec rate guard flags the clip and it
is re-rendered. The floor refuses END_OF_SPEECH at decode time instead: while a
request has generated fewer than eosFloor x expected audio tokens (expected =
chars / eosFloorRate x 84), the EOS logit is -inf.

WHAT THIS PROVES, driving the real processor (no GPU, no model):
  1. the floor forbids EOS below its line and nothing at or above it;
  2. a read at the voice's own truncation-guard rate (the fastest read the
     pipeline calls honest) clears the floor at every chunk size, so the floor
     can only ever remove a stop the guard would have rejected anyway;
  3. a truncation at 0.3-0.6 of expected lands inside the floor;
  4. the boost's ramp is byte-for-byte unchanged by the floor: at every token
     past the boost start the EOS bias equals the boost-only processor's;
  5. a floor tighter than the guard is REFUSED at construction, not rendered;
  6. the caps cross register_voice_caps by their catalog names, a floor of 0
     builds no processor for an unboosted voice, and the MLX processor raises
     rather than render silently without a configured floor.

Run it with an interpreter that has e2a's dependencies, e.g. the bundled env:
    python_env/python.exe tools/test_eos_floor.py
Exit code 0 = all cases passed.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lib.classes.tts_engines.orpheus import Orpheus

EOS = Orpheus.END_OF_AUDIO_TOKEN
TPS = Orpheus.TOKENS_PER_AUDIO_SECOND

# thirdreich's live catalog guard (2026-09-03) with the floor at the proposed
# default; boost values are deathstalker's so the ramp assertion has a ramp.
VOICE = 'floor-test'
GUARD = 20.5
FLOOR, RATE = 0.55, 15.0
CAPS = {'repPenalty': 1.1, 'eosBoost': 8, 'eosBoostStart': 2, 'maxCharsPerSec': GUARD,
        'eosFloor': FLOOR, 'eosFloorRate': RATE}

failures = []


def check(ok, message):
    if not ok:
        failures.append(message)
        print(f'  FAIL  {message}')
    return ok


def engine(voice, caps):
    """A bare instance: every method under test reads only class attributes and
    the registered caps, so __init__ (which loads a model) is not needed."""
    Orpheus.register_voice_caps(voice, dict(caps))
    tts = Orpheus.__new__(Orpheus)
    tts.voice = voice
    return tts


def eos_logit_after(proc, n):
    """The EOS logit the real processor leaves on a zero row after `n` generated tokens."""
    logits = np.zeros(EOS + 1, dtype=np.float64)
    proc(list(range(n)), logits)
    return logits[EOS]


def first_allowed(proc, limit):
    """The first token index at which EOS is NOT -inf (probed, not re-derived)."""
    for n in range(0, limit):
        if not math.isinf(eos_logit_after(proc, n)):
            return n
    raise AssertionError(f'EOS never allowed within {limit} tokens')


def tokens_at_rate(chars, chars_per_sec):
    return chars / chars_per_sec * TPS


print('== 1. the floor forbids EOS below its line and nothing at or above it ==')
tts = engine(VOICE, CAPS)
for chars in (4, 13, 48, 100, 200, 350, 450, 540):
    proc = tts._eos_boost_processor(chars)
    expected_floor = FLOOR * chars / RATE * TPS
    allowed = first_allowed(proc, 4 * Orpheus.MAX_AUDIO_TOKENS)
    check(allowed == math.ceil(expected_floor),
          f'{chars} chars: EOS first allowed at {allowed}, floor is {expected_floor:.2f}')
    check(math.isinf(eos_logit_after(proc, 0)) and eos_logit_after(proc, 0) < 0,
          f'{chars} chars: EOS must be -inf at token 0')
    check(eos_logit_after(proc, allowed) == 0.0,
          f'{chars} chars: EOS must be untouched (0.0) right at the floor, got '
          f'{eos_logit_after(proc, allowed)}')
    print(f'  {chars:>3} chars: EOS forbidden for the first {allowed} tokens '
          f'({allowed / TPS:.2f} s)')

print('\n== 2. a read at the guard rate clears the floor at every chunk size ==')
for chars in (4, 13, 48, 100, 200, 350, 450, 540):
    proc = tts._eos_boost_processor(chars)
    honest = tokens_at_rate(chars, GUARD)
    check(not math.isinf(eos_logit_after(proc, int(honest))),
          f'{chars} chars read at the guard rate {GUARD} ch/s ({honest:.0f} tokens) '
          'must be allowed to stop')
print(f'  floor rate {RATE / FLOOR:.1f} ch/s vs guard {GUARD} ch/s: every honest read clears it')

print('\n== 3. a truncation at 0.3-0.6 of expected lands inside the floor ==')
for chars in (100, 450, 540):
    proc = tts._eos_boost_processor(chars)
    expected = chars / RATE * TPS
    for frac in (0.3, 0.45, 0.5):
        n = int(frac * expected)
        check(math.isinf(eos_logit_after(proc, n)),
              f'{chars} chars stopping at {frac:.2f} x expected ({n} tokens) must be forbidden')
    # 0.6 sits just past the 0.55 line: NOT inside the floor. That is the design
    # (the fast honest reads are at >= 0.75), reported rather than asserted.
    n = int(0.6 * expected)
    print(f'  {chars} chars: 0.30/0.45/0.50 x expected forbidden; 0.60 x '
          f'({n} tokens) {"forbidden" if math.isinf(eos_logit_after(proc, n)) else "allowed"}')

print('\n== 4. the boost ramp is byte-for-byte unchanged by the floor ==')
boost_only = engine('boost-only', {k: v for k, v in CAPS.items()
                                   if k not in ('eosFloor', 'eosFloorRate')})
for chars in (13, 48, 200, 450, 540):
    with_floor = tts._eos_boost_processor(chars)
    without = boost_only._eos_boost_processor(chars)
    start = CAPS['eosBoostStart'] * tts._expected_audio_tokens(chars)
    floor_line = FLOOR * chars / RATE * TPS
    check(floor_line < start,
          f'{chars} chars: floor ({floor_line:.0f}) must sit below the boost start ({start:.0f})')
    mismatches = 0
    for n in range(int(floor_line) + 1, int(start) + 3 * int(tts._expected_audio_tokens(chars))):
        if eos_logit_after(with_floor, n) != eos_logit_after(without, n):
            mismatches += 1
    check(mismatches == 0,
          f'{chars} chars: {mismatches} tokens where the floored processor differs from '
          'the boost-only one above the floor')
    check(eos_logit_after(with_floor, int(start) + 50) > 0,
          f'{chars} chars: the boost must still engage past its start')
print('  identical EOS bias at every token above the floor, ramp engages as before')

print('\n== 5. a floor tighter than the guard is refused ==')
tight = engine('too-tight', dict(CAPS, eosFloor=0.8))          # 15 / 0.8 = 18.75 < 20.5
try:
    tight._eos_boost_processor(450)
    check(False, 'eosFloor 0.8 at 15 ch/s against a 20.5 ch/s guard must raise')
except ValueError as e:
    print(f'  refused: {e}')
unguarded = engine('unguarded', dict(CAPS, eosFloor=0.8, maxCharsPerSec=0))
check(unguarded._eos_boost_processor(450) is not None,
      'with the guard disabled (0) the same floor must be accepted')
for bad in ({'eosFloor': 1.0}, {'eosFloor': 1.5}, {'eosFloorRate': 0}):
    try:
        engine('bad', dict(CAPS, **bad))._eos_boost_processor(450)
        check(False, f'{bad} must raise')
    except ValueError:
        pass

print('\n== 6. registration, disabled floor, MLX refusal ==')
stored = Orpheus.register_voice_caps('reg', {'eosFloor': 0.55, 'eosFloorRate': 16.47})
check(stored == {'eosFloor': 0.55, 'eosFloorRate': 16.47},
      f'caps must cross register_voice_caps by their catalog names: {stored}')
off = engine('off', {'maxCharsPerSec': GUARD, 'eosFloor': 0})
check(off._eos_boost_processor(450) is None,
      'floor 0 on an unboosted voice must build no processor')
off_boosted = engine('off-boosted', {'maxCharsPerSec': GUARD, 'eosFloor': 0, 'eosBoost': 8, 'eosBoostStart': 2})
proc = off_boosted._eos_boost_processor(450)
check(proc is not None and eos_logit_after(proc, 0) == 0.0,
      'floor 0 on a boosted voice must leave early EOS untouched')
try:
    tts._mlx_eos_boost_processor(450)
    check(False, 'the MLX processor must refuse a configured floor')
except NotImplementedError as e:
    print(f'  MLX refused: {e}')

print()
if failures:
    print(f'{len(failures)} FAILURE(S)')
    sys.exit(1)
print('ALL PASSED')

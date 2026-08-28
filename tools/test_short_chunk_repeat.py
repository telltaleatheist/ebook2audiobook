#!/usr/bin/env python
"""An ultra-short chunk must not be left unguarded by the anti-runaway levers
(2026-08-28).

THE DEFECT. The 2026-08-27 header-own-chunks change exempted headings from
_apply_min_chars_floor, which existed precisely so an ultra-short row is never
handed to TTS as its own starved prompt. Orpheus then sometimes failed to stop
after saying a two-word title and said it again: measured in a real render,
"Introduction." (13 chars) took 4.267 s where same-length titles took 1.5-1.6 s.

WHY it could happen: every anti-runaway lever in orpheus.py sizes itself from
`chars / 18.4 ch-per-s`, a SPEECH estimate that ignores the model's own trained
tail (~0.8 s, flat, on every clip). The EOS boost papered over that with a flat
`max(300.0, ...)` floor, which made the boost's start identical — 600 tokens,
7.14 s of audio, at deathstalker's eosBoostStart 2.0 — for a 4-char title and a
65-char one alike. A healthy 13-char heading renders in ~130 tokens, so the boost
could not engage until a clip had run 4x its natural length; the doubled take
ended at 358 tokens having never received a single logit of help. The MLX
per-chunk budget had the mirror bug, capping "GOD." at 40 tokens against a
measured 108.

WHAT THIS PROVES, driving the real methods (no GPU, no model):
  1. the EOS boost now engages before the known doubled take, and on NO healthy
     clip in the reference set;
  2. it is a NO-OP at 48 chars and above — ordinary prose is untouched;
  3. the MLX per-chunk token budget covers every healthy short render;
  4. the duration backstop flags the doubled take and nothing else.

FIXTURE. The 52 non-empty chunks of <= 60 chars from the real render at
Z:\\bookforge\\projects\\witches_-_Unknown\\stages\\03-tts\\sessions\\en\\
ebook-38a708a4-cba0-486c-8af2-1bc7857c2092\\2055c81ef480ca96c96465726894841c
(deathstalker). Raw file duration IS the generated audio here: _save_audio does
not trim (NO-FALLBACK 2026-07-11) and deathstalker ships sentenceGap 0 — proven
by the set's minimum trailing silence, 0.490 s, sitting BELOW the 0.6 s gap
default, so no pad was appended. Hence tokens = seconds * TOKENS_PER_AUDIO_SECOND.

Run it with an interpreter that has e2a's dependencies, e.g. the bundled env:
    python_env/python.exe tools/test_short_chunk_repeat.py
Exit code 0 = all cases passed.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lib.classes.tts_engines.orpheus import Orpheus

# (chars, raw_seconds, label). BAD marks the one take Owen heard spoken twice.
BAD = 'Introduction.'
# Named in the report as a second suspect (6.3 ch/s where 30-40 char peers run
# 10-13): not asserted either way, only reported.
SUSPECT = 'Turtles Are "Zeroes," Not "Heroes".'
FIXTURE = [
    (11, 1.621, 'Dedication.'),
    (8, 1.365, 'Preface.'),
    (13, 4.267, BAD),
    (55, 3.669, 'CAN CHRISTIAN CHILDREN BE AFFECTED BY DEMONIC ACTIVITY?'),
    (33, 2.645, 'UnderstandingWhich WitchIs Which.'),
    (51, 3.243, 'HOW CAN THEY TELL THE FUTUREIF THEY ARE NOT OF GOD?'),
    (40, 2.304, 'WHAT IS THE TRUE TEST OF A REAL PROPHET?'),
    (46, 3.072, 'SHOULD WE BE AFRAID OF THESEWITCHES AND SEERS?'),
    (12, 1.536, 'WHITE MAGIC.'),
    (12, 2.048, 'BLACK MAGIC.'),
    (37, 3.157, 'Witches ActivelyProselytize Children.'),
    (7, 1.621, 'CASE 1.'),
    (7, 1.365, 'CASE 2.'),
    (16, 1.963, 'Major festivals.'),
    (16, 1.621, 'Minor festivals.'),
    (35, 3.072, 'PROBLEMS WHICH STEM FROM ASTROLOGY.'),
    (7, 1.621, 'CASE 3.'),
    (48, 3.499, 'Witches and Satanists HaveSome Things In Common.'),
    (42, 4.779, 'NINE ABOMINATIONS IN DEUTERONOMY 18:10-11.'),
    (48, 3.584, 'WHAT SUPERNATURAL PHENOMENA DO TEENS BELIEVE IN?'),
    (54, 3.584, 'Do people who die have a desire to talk to the living?'),
    (40, 3.072, 'THREE DANGERS INVOLVED IN REINCARNATION.'),
    (46, 4.096, 'Witches and Satanists Usethe Media to Recruit.'),
    (43, 2.816, 'THREE WAYS THAT THE MEDIA HELPS WITCHCRAFT.'),
    (34, 2.987, 'SYMBOLISM BEHIND THE WIZARD OF OZ.'),
    (30, 3.243, 'Satan Is "Trolling" for Souls.'),
    (59, 4.267, 'In the occult, they teach that there are two sexual demons.'),
    (27, 2.304, 'Lucifer Is Quite a Charmer.'),
    (27, 2.987, 'Witches Do "Knot"Play Fair.'),
    (26, 3.840, 'Smurfs Are Not"True Blue".'),
    (57, 6.144, 'Something Smells a "Little Fishy" WithThe Little Mermaid.'),
    (35, 5.547, SUSPECT),
    (34, 3.072, 'DANGERS OF THESE REPTILIAN HEROES.'),
    (25, 2.219, 'HERE ARE A FEW QUESTIONS.'),
    (49, 2.816, 'Are they teaching that mutants can turn out good?'),
    (39, 3.413, 'Bart Simpson Has His Own Values System.'),
    (26, 3.840, '"He\'s Man" and"She\'s God".'),
    (38, 3.755, 'Ecology or Theology,You Make the Call.'),
    (56, 3.413, 'HOW DO WE KNOW THAT GOD DOES NOT SHOW FAVORITISM TO MEN?'),
    (21, 2.048, 'Beauty and the Beast.'),
    (27, 2.048, 'THE METABOLISM OF THE BODY.'),
    (7, 1.109, 'Barney.'),
    (41, 3.413, 'Situation Ethics and Those Who Have None.'),
    (25, 2.731, 'Bereavement andBraindead.'),
    (23, 2.901, 'Yin and YangMade Plain.'),
    (6, 1.792, 'SATAN.'),
    (4, 1.280, 'GOD.'),
    (25, 1.963, 'Twenty-Seven Admonitions.'),
    (10, 1.280, 'THE FLESH.'),
    (19, 2.219, "SATAN'S BACKGROUND."),
    (8, 2.304, 'NEW AGE.'),
    (17, 1.621, 'About the Author.'),
]

VOICE = 'deathstalker'
# deathstalker's live catalog values (electron/data/orpheus-models.json).
CAPS = {'repPenalty': 1.1, 'eosBoost': 8, 'eosBoostStart': 2, 'maxCharsPerSec': 23.5}

failures = []


def check(ok, message):
    if not ok:
        failures.append(message)
        print(f'  FAIL  {message}')
    return ok


def engine():
    """A bare instance: every method under test reads only class attributes and
    the registered caps, so __init__ (which loads a model) is not needed."""
    Orpheus.register_voice_caps(VOICE, dict(CAPS))
    tts = Orpheus.__new__(Orpheus)
    tts.voice = VOICE
    return tts


def boost_start(tts, n_chars):
    """The token index where the real processor first adds EOS bias, found by
    probing it rather than by re-deriving its arithmetic."""
    proc = tts._eos_boost_processor(n_chars)
    if proc is None:
        raise AssertionError('the reference voice must have a boost configured')
    for n in range(1, 4 * Orpheus.MAX_AUDIO_TOKENS):
        logits = np.zeros(Orpheus.END_OF_AUDIO_TOKEN + 1, dtype=np.float64)
        proc(list(range(n)), logits)
        if logits[Orpheus.END_OF_AUDIO_TOKEN] > 0:
            return n
    raise AssertionError(f'boost never engaged for {n_chars} chars')


def old_start(n_chars):
    """The pre-fix start, for the no-op assertion: eosBoostStart x the flat floor."""
    return CAPS['eosBoostStart'] * max(300.0, n_chars / 18.4 * Orpheus.TOKENS_PER_AUDIO_SECOND)


def tokens(seconds):
    return seconds * Orpheus.TOKENS_PER_AUDIO_SECOND


print('== 1. the EOS boost engages on the doubled take, and on no healthy clip ==')
tts = engine()
starts = {}
tightest = (1e9, None)
for chars, seconds, label in FIXTURE:
    if chars not in starts:
        starts[chars] = boost_start(tts, chars)
    start, generated = starts[chars], tokens(seconds)
    engaged = generated > start
    if label == BAD:
        check(engaged,
              f'the doubled take must be boosted: {generated:.0f} tokens vs start {start:.0f}')
        print(f'  BOOSTED  {label!r}: ran to {generated:.0f} tokens, boost engaged at '
              f'{start:.0f} ({generated - start:.0f} tokens of ramp)')
    elif label == SUSPECT:
        print(f'  (suspect, not asserted) {label!r}: {generated:.0f} tokens vs start '
              f'{start:.0f} — {"boosted" if engaged else "not boosted"}')
    else:
        check(not engaged,
              f'healthy clip must NOT be boosted mid-speech: {label!r} '
              f'({chars} chars, {generated:.0f} tokens) vs start {start:.0f}')
        margin = (start - generated) / generated
        if margin < tightest[0]:
            tightest = (margin, label)
print(f'  tightest healthy margin: {tightest[0] * 100:.1f}% on {tightest[1]!r}')

print('\n== 2. no-op for ordinary prose (>= 48 chars) ==')
for chars in (48, 55, 66, 100, 200, 350, 450, 540):
    check(abs(boost_start(tts, chars) - old_start(chars)) <= 1.0,
          f'boost start must be unchanged at {chars} chars: '
          f'{boost_start(tts, chars)} vs {old_start(chars)}')
print('  boost start unchanged at 48, 55, 66, 100, 200, 350, 450, 540 chars')
# …and the whole change really is confined below 48.
changed = [c for c in range(1, 200) if abs(boost_start(tts, c) - old_start(c)) > 1.0]
check(changed and max(changed) < 48,
      f'the change must be confined to chunks under 48 chars; highest changed = '
      f'{max(changed) if changed else None}')
print(f'  behaviour differs only for {min(changed)}-{max(changed)} chars')

print('\n== 3. the MLX per-chunk budget covers every healthy short render ==')
tightest = (1e9, None)
for chars, seconds, label in FIXTURE:
    if label in (BAD, SUSPECT):
        continue
    budget = tts._mlx_token_budget('x' * chars)
    generated = tokens(seconds)
    check(budget >= generated,
          f'MLX budget must not truncate a healthy render: {label!r} '
          f'({chars} chars) needs {generated:.0f} tokens, budget {budget}')
    margin = (budget - generated) / budget
    if margin < tightest[0]:
        tightest = (margin, label)
print(f'  tightest budget margin: {tightest[0] * 100:.1f}% on {tightest[1]!r}')

print('\n== 4. the duration backstop flags the doubled take and nothing else ==')
flagged = []
for chars, seconds, label in FIXTURE:
    audio = np.zeros(int(seconds * Orpheus.SAMPLE_RATE), dtype=np.float32)
    if tts._needs_reroll(0, 'x' * chars, audio):
        flagged.append(label)
check(flagged == [BAD], f'backstop must flag exactly [{BAD!r}], flagged {flagged}')
# The re-roll is only ever allowed to shorten a clip, never to lose one.
short_take = np.zeros(1000, dtype=np.float32)
long_take = np.zeros(4000, dtype=np.float32)
check(tts._pick_shorter_take(0, long_take, short_take) is short_take,
      'a shorter re-roll must be kept')
check(tts._pick_shorter_take(0, short_take, long_take) is short_take,
      'a longer re-roll must be discarded')
check(tts._pick_shorter_take(0, short_take, np.zeros(0, dtype=np.float32)) is short_take,
      'an empty re-roll must never replace real audio')
check(tts._pick_shorter_take(0, short_take, None) is short_take,
      'a missing re-roll must never replace real audio')

print('\n==================== RESULT ====================')
if failures:
    print(f'{len(failures)} case(s) FAILED')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('all short-chunk repeat cases passed')
sys.exit(0)

#!/usr/bin/env python
"""The fast-start windowed decoder must reproduce the whole clip, exactly once.

WHAT IS BEING PROVED. WindowedFrameEmitter cuts a row's audio into ~0.34 s
payloads while the row is still generating. Nothing downstream can check its
work: by the time a wrong slice is noticeable it is already playing in someone's
browser. So the arithmetic is proved here, against a FAKE decoder, with no
model, no GPU and no Mac:

  * CONTIGUITY / NO OVERLAP / NO GAP — the concatenation of a row's payloads is
    exactly frames 0..n in order, sample for sample, where n = tokens // 7;
  * TOTAL LENGTH — that concatenation is exactly (tokens // 7) * 2048 samples,
    which is what the whole-clip decode this replaces would have produced;
  * CADENCE — every payload but the last is exactly 4 frames, and the first one
    is not emitted until 6 frames exist (4 payload + 2 right context);
  * CONTEXT — each window really is [max(0, a-1), b+2) during the row and
    [max(0, a-1), n) at the flush, so every payload frame is decoded with a real
    left neighbour and (except at the very end) two real right ones;
  * SEQ — seq numbers start at 0 and increase by one, per row;
  * the awkward counts: 0, 1, 5, 6, 27 (not a frame boundary), 28 (exactly 4
    frames), 63 and 1015 (9 and 145 frames - the cases whose flush tail is the
    maximum FIVE frames), 100 and 1001 tokens, each driven token-by-token as
    generation really arrives, and again in one shot to prove the answer does
    not depend on when push() was called.

HOW THE FAKE DECODER WORKS. It returns frame k as 2048 copies of the value k,
so the concatenated payloads can be compared against the ideal whole-clip
waveform element by element — a payload cut one frame off would show up as a
run of the wrong integer, not as a length that happens to match.

Run:  python_env/python.exe tools/test_stream_window_decode.py
"""
import os
import sys

# The WORKTREE's lib, not the main checkout's — this test exists to prove the
# code sitting next to it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lib.classes.tts_engines.orpheus_stream_decode import (
    PAYLOAD_FRAMES, RIGHT_CONTEXT_FRAMES, LEFT_CONTEXT_FRAMES,
    SAMPLES_PER_FRAME, TOKENS_PER_FRAME,
    StreamDecodeMisaligned, WindowedFrameEmitter,
)

failures = []


def check(ok, msg):
    if ok:
        print(f'  ok: {msg}')
    else:
        print(f'  FAIL: {msg}')
        failures.append(msg)


class FakeDecoder:
    """Frame k decodes to 2048 samples all equal to k. Records every window."""

    def __init__(self, total_frames):
        self.total_frames = total_frames
        self.calls = []

    def __call__(self, first, last):
        self.calls.append((first, last))
        if not (0 <= first <= last <= self.total_frames):
            raise AssertionError(
                f'decoder asked for frames [{first}, {last}) but the row only '
                f'has {self.total_frames}')
        return np.concatenate([
            np.full(SAMPLES_PER_FRAME, float(k), dtype=np.float32)
            for k in range(first, last)
        ]) if last > first else np.zeros(0, dtype=np.float32)


def ideal(n_frames):
    """What the whole-clip decode would have produced for n frames."""
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([
        np.full(SAMPLES_PER_FRAME, float(k), dtype=np.float32)
        for k in range(n_frames)
    ])


def run_row(n_tokens, incremental):
    """Drive one row and return (chunks, decoder). `incremental` feeds the
    emitter one token at a time, the way a decode step really does."""
    n_frames = n_tokens // TOKENS_PER_FRAME
    dec = FakeDecoder(n_frames)
    em = WindowedFrameEmitter(dec, label=f'row({n_tokens})')
    chunks = []
    if incremental:
        for t in range(1, n_tokens + 1):
            chunks.extend(em.push(t))
    chunks.extend(em.flush(n_tokens))
    return chunks, dec, em


def assert_row(n_tokens, incremental):
    label = f'{n_tokens} tokens ({"per-token" if incremental else "one shot"})'
    n_frames = n_tokens // TOKENS_PER_FRAME
    chunks, dec, em = run_row(n_tokens, incremental)

    # seq numbering
    check([seq for seq, _ in chunks] == list(range(len(chunks))),
          f'{label}: seq runs 0..{len(chunks) - 1}')

    # contiguity, no overlap, no gap, exact total
    joined = (np.concatenate([pcm for _, pcm in chunks]) if chunks
              else np.zeros(0, dtype=np.float32))
    want = ideal(n_frames)
    check(len(joined) == n_frames * SAMPLES_PER_FRAME,
          f'{label}: {len(joined)} samples == {n_frames} frames x {SAMPLES_PER_FRAME}')
    check(len(joined) == len(want) and bool(np.array_equal(joined, want)),
          f'{label}: payloads reassemble the whole clip frame for frame')
    check(em.emitted_samples == n_frames * SAMPLES_PER_FRAME,
          f'{label}: emitter agrees it emitted {em.emitted_samples} samples')

    # cadence: everything but the last chunk is exactly PAYLOAD_FRAMES
    sizes = [len(pcm) // SAMPLES_PER_FRAME for _, pcm in chunks]
    check(all(s == PAYLOAD_FRAMES for s in sizes[:-1]),
          f'{label}: cadence {sizes} - every chunk but the last is {PAYLOAD_FRAMES} frames')
    if sizes:
        # The tail can be up to PAYLOAD + RIGHT_CONTEXT - 1 = 5 frames, NOT 4:
        # the drain stops as soon as n < a + 4 + 2, so a row of 5, 9, 13 or 17
        # frames flushes 5 at once. An earlier version of this test asserted
        # <= 4 and never picked a count that produced a 5-tail, so it agreed
        # with itself; 63 and 1015 tokens (9 and 145 frames) are here to make
        # sure a 5-frame tail is actually exercised.
        max_tail = PAYLOAD_FRAMES + RIGHT_CONTEXT_FRAMES - 1
        check(1 <= sizes[-1] <= max_tail,
              f'{label}: the flush tail is {sizes[-1]} frame(s), within [1, {max_tail}]')
    check(bool(chunks) == (n_frames > 0),
          f'{label}: {"emits" if n_frames else "emits nothing"} for {n_frames} frame(s)')

    # windows: right context during the row, left context always
    expected_calls = []
    a = 0
    while n_frames >= a + PAYLOAD_FRAMES + RIGHT_CONTEXT_FRAMES:
        b = a + PAYLOAD_FRAMES
        expected_calls.append((max(0, a - LEFT_CONTEXT_FRAMES),
                               b + RIGHT_CONTEXT_FRAMES))
        a = b
    if n_frames > a:
        expected_calls.append((max(0, a - LEFT_CONTEXT_FRAMES), n_frames))
    check(dec.calls == expected_calls,
          f'{label}: decode windows {dec.calls} carry the specified context')
    check(len(dec.calls) == len(chunks),
          f'{label}: one window decoded per payload')
    # The first payload cannot exist before 6 frames do.
    if chunks:
        first_win = dec.calls[0]
        check(first_win[0] == 0,
              f'{label}: the first window starts at frame 0 (no left context to have)')
        if len(chunks) > 1:
            check(first_win[1] == PAYLOAD_FRAMES + RIGHT_CONTEXT_FRAMES,
                  f'{label}: the first full-cadence window is '
                  f'{PAYLOAD_FRAMES + RIGHT_CONTEXT_FRAMES} frames wide')
    return chunks


print('== 1. the awkward token counts, driven per token and in one shot ==')
per_token = {}
for n in (0, 1, 5, 6, 27, 28, 63, 100, 1001, 1015):
    per_token[n] = assert_row(n, incremental=True)
    assert_row(n, incremental=False)

print('\n== 2. per-token and one-shot must agree exactly ==')
for n in (0, 1, 5, 6, 27, 28, 63, 100, 1001, 1015):
    one_shot, _dec, _em = run_row(n, incremental=False)
    a = [(s, pcm.tolist()) for s, pcm in per_token[n]]
    b = [(s, pcm.tolist()) for s, pcm in one_shot]
    check(a == b, f'{n} tokens: WHEN push() is called cannot change WHAT is emitted')

print('\n== 3. rows ending on and off a frame boundary ==')
# 28 tokens = exactly 4 frames; 27 = 3 frames + 6 stray codes that are not audio
# yet. Both must emit whole frames only, and the stray codes must vanish.
on_boundary, _d, _e = run_row(28, incremental=True)
off_boundary, _d2, _e2 = run_row(27, incremental=True)
check(sum(len(p) for _s, p in on_boundary) == 4 * SAMPLES_PER_FRAME,
      '28 tokens (on a frame boundary) -> 4 whole frames')
check(sum(len(p) for _s, p in off_boundary) == 3 * SAMPLES_PER_FRAME,
      '27 tokens (mid-frame) -> 3 whole frames, the partial frame dropped')
# A row that gains its seventh token AFTER the flush window would have been
# closed is not a case that can happen — but the boundary between 6 and 7
# tokens is where the first frame appears, so pin it.
check(sum(len(p) for _s, p in run_row(6, incremental=True)[0]) == 0,
      '6 tokens are not a frame yet')
check(sum(len(p) for _s, p in run_row(7, incremental=True)[0]) == SAMPLES_PER_FRAME,
      '7 tokens are exactly one frame, emitted by the flush')

# The MAXIMUM tail, which the earlier version of this test never produced: the
# drain stops as soon as n < a + 4 + 2, so 5, 9, 13 and 17 frames all flush five
# frames in one payload.
for frames in (5, 9, 13, 17):
    tail_chunks, _d3, _e3 = run_row(frames * TOKENS_PER_FRAME, incremental=True)
    tail = len(tail_chunks[-1][1]) // SAMPLES_PER_FRAME
    check(tail == PAYLOAD_FRAMES + RIGHT_CONTEXT_FRAMES - 1,
          f'{frames} frames flush a {tail}-frame tail (the maximum, '
          f'{PAYLOAD_FRAMES + RIGHT_CONTEXT_FRAMES - 1})')
    check(sum(len(p) for _s, p in tail_chunks) == frames * SAMPLES_PER_FRAME,
          f'{frames} frames still total {frames} frames of audio')

print('\n== 4. the emitter refuses states it cannot honour ==')
dec = FakeDecoder(20)
em = WindowedFrameEmitter(dec, label='refusals')
em.push(7 * 20)
em.flush(7 * 20)
for name, call in (('push after flush', lambda: em.push(7 * 20)),
                   ('flush twice', lambda: em.flush(7 * 20))):
    try:
        call()
        check(False, f'{name} must raise')
    except StreamDecodeMisaligned as e:
        print(f'  refused ({name}): {e}')

shrink = WindowedFrameEmitter(FakeDecoder(20), label='shrink')
shrink.push(7 * 20)
try:
    shrink.push(7 * 2)
    check(False, 'a shrinking token count must raise')
except StreamDecodeMisaligned as e:
    print(f'  refused (token count went backwards): {e}')

try:
    WindowedFrameEmitter(FakeDecoder(20), label='negative').push(-1)
    check(False, 'a negative token count must raise')
except StreamDecodeMisaligned as e:
    print(f'  refused (negative tokens): {e}')


def short_decoder(first, last):
    # One sample short of a whole number of frames: the exact failure that
    # would silently shift every later payload.
    return np.zeros((last - first) * SAMPLES_PER_FRAME - 1, dtype=np.float32)


try:
    WindowedFrameEmitter(short_decoder, label='short').push(7 * 6)
    check(False, 'a decoder returning the wrong sample count must raise')
except StreamDecodeMisaligned as e:
    print(f'  refused (wrong sample count): {e}')

try:
    WindowedFrameEmitter(lambda a, b: None, label='none').push(7 * 6)
    check(False, 'a decoder returning None must raise')
except StreamDecodeMisaligned as e:
    print(f'  refused (None): {e}')

print('\n== 5. a long row: cadence holds all the way down ==')
chunks, dec, em = run_row(1001, incremental=True)
n_frames = 1001 // TOKENS_PER_FRAME
full = [c for c in chunks[:-1]]
check(len(full) == 35 and n_frames == 143,
      f'1001 tokens = {n_frames} frames -> {len(full)} full chunks + a tail')
check(all(w[1] - w[0] == PAYLOAD_FRAMES + LEFT_CONTEXT_FRAMES + RIGHT_CONTEXT_FRAMES
          for w in dec.calls[1:-1]),
      'every interior window is 1 + 4 + 2 = 7 frames wide')
check(em.decoded_windows == len(chunks),
      f'{em.decoded_windows} decodes for {len(chunks)} chunks (1.75x the frames '
      'of a whole-clip decode, as designed)')

print()
if failures:
    print(f'{len(failures)} FAILURE(S)')
    sys.exit(1)
print('ALL PASSED')

#!/usr/bin/env python
"""Continuous batching must finish every row exactly once, at its OWN token cap.

WHAT IS BEING PROVED. With ORPHEUS_MLX_CONTINUOUS=1 (OFF by default; the switch is set on the stub
tree) _convert_mlx_batch stops splitting a call into fresh groups and hands ONE
BatchGenerator every row up front; mlx-lm's scheduler then refills a retired slot
from its own queue instead of letting the tail of each group decode at dwindling
width. Three things change and each is asserted here:

  * ONE generator, `completion_batch_size` = the memory-derived width and
    `prefill_batch_size` = min(width, MLX_CONTINUOUS_PREFILL) — not one generator
    per group;
  * the anti-runaway ceiling is now PER ROW (`insert(max_tokens=[...])`), so the
    cap-hit test after retirement must compare a row's token count against THAT
    row's cap. A row with a small budget sitting next to a long one must be
    caught at its own cap, and a long row that stops cleanly past a SHORT row's
    cap must NOT be mistaken for a runaway;
  * every one of the 40 rows lands in `results` exactly once, mapped to its own
    audio, with the decode-overlap hand-off unchanged;
  * the heartbeat's new ` live N` field never exceeds the width (it is what tells
    the A/B apart in the log), and it does reach the width — i.e. refill really
    happened rather than the queue draining as one wide group.

And the kill switch: ORPHEUS_MLX_CONTINUOUS=0 reproduces today's fresh-group
path — five generators for 40 rows at width 8, each with a UNIFORM cap equal to
its group's depth, and no ` live ` field in the heartbeat.

HOW. mlx_lm.generate.BatchGenerator is replaced by a fake that models the real
scheduler: it admits at most `completion_batch_size` rows, generates one token
for each live row per step, and only then refills up to `prefill_batch_size` from
its queue — so a queued uid appears in a response only once it is GENERATING,
which is exactly what `live` is derived from. mlx_audio's llama module (which
loads SNAC weights at import) is stubbed; mlx itself is real but pinned to the
CPU device. No model, no GPU, a fraction of a second.
"""
import os
import re
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

# SNAC's weights load at IMPORT of mlx_audio...llama, and this test must not load
# a model. Stub the module BEFORE anything can import it for real.
_fake_llama = types.ModuleType('mlx_audio.tts.models.llama.llama')


def _fake_decode_audio_from_codes(code_list):
    return [[float(code_list[0])] * 4]


_fake_llama.decode_audio_from_codes = _fake_decode_audio_from_codes
for _name in ('mlx_audio', 'mlx_audio.tts', 'mlx_audio.tts.models',
              'mlx_audio.tts.models.llama'):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules['mlx_audio.tts.models.llama.llama'] = _fake_llama

import importlib

mlx_lm_generate = importlib.import_module('mlx_lm.generate')
from lib.classes.tts_engines.orpheus import Orpheus

failures = []


def check(cond, label):
    if cond:
        print(f'  ok   {label}')
    else:
        print(f'  FAIL {label}')
        failures.append(label)


# ---------------------------------------------------------------------------
# The book slice: 40 rows, width 8, prefill 3.
#
#   row 5   long budget, runs away            -> retires 'length' at cap 30
#   row 11  SHORT budget (6), runs away       -> retires 'length' at cap 6
#                                                (a depth-based check would
#                                                 miss it: 6 < 30)
#   row 17  long budget, stops cleanly at 20  -> 19 tokens, well past row 11's
#                                                cap, and must NOT be flagged
#   everything else                           -> stops cleanly in 3-9 tokens
# ---------------------------------------------------------------------------
N_ROWS = 40
WIDTH = 8
PREFILL = 3
MAX_TOKENS = 30           # MLX_MAX_TOKENS for this test
LONG_BUDGET = 30
SHORT_BUDGET = 6

CAP_ROW = 5               # hits its (long) cap
SHORT_CAP_ROW = 11        # hits its own SHORT cap
LONG_CLEAN_ROW = 17       # stops cleanly past the short row's cap

MARKER_BATCH = 1000
MARKER_RERENDER = 9000
RUNAWAY = 10 ** 6         # "would generate forever": always hits the cap

BUDGET = {i: LONG_BUDGET for i in range(N_ROWS)}
BUDGET[SHORT_CAP_ROW] = SHORT_BUDGET

PLAN = {i: 3 + (i % 7) for i in range(N_ROWS)}   # tokens before a clean stop
PLAN[CAP_ROW] = RUNAWAY
PLAN[SHORT_CAP_ROW] = RUNAWAY
PLAN[LONG_CLEAN_ROW] = 20


class FakeResponse:
    __slots__ = ('uid', 'token', 'finish_reason')

    def __init__(self, uid, token, finish_reason):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason


class FakeBatchGenerator:
    """The real scheduler's shape, minus the model.

    Mirrors BatchGenerator._next in the order that matters: GENERATE for every
    live row first, THEN refill retired slots from the queue (bounded by
    completion_batch_size and prefill_batch_size). A queued row therefore reports
    for the first time on the step AFTER it is admitted, which is the property
    `live` in the heartbeat is derived from.
    """

    instances = []

    def __init__(self, model, *, max_tokens=None, stop_tokens=None, sampler=None,
                 logits_processors=None, completion_batch_size=1,
                 prefill_batch_size=1, **kwargs):
        self.default_max_tokens = max_tokens
        self.completion_batch_size = completion_batch_size
        self.prefill_batch_size = prefill_batch_size
        self.queue = []
        self.live = {}          # uid -> tokens emitted so far
        self.caps = {}          # uid -> the max_tokens this row was inserted with
        self.plan = {}
        self.inserted_max_tokens = None
        self.max_live_seen = 0
        self.closed = False
        FakeBatchGenerator.instances.append(self)

    def insert(self, prompts, max_tokens=None, logits_processors=None):
        caps = (list(max_tokens) if max_tokens is not None
                else [self.default_max_tokens] * len(prompts))
        self.inserted_max_tokens = list(caps)
        uids = []
        for prompt, cap in zip(prompts, caps):
            idx = prompt[0] - MARKER_BATCH
            uid = f'u{idx}'
            self.caps[uid] = cap
            self.plan[uid] = PLAN[idx]
            self.queue.append(uid)
            uids.append(uid)
        return uids

    def _step_once(self):
        responses = []
        for uid in list(self.live):
            self.live[uid] += 1
            n = self.live[uid]
            if n >= self.caps[uid]:
                responses.append(FakeResponse(uid, 7, 'length'))
                del self.live[uid]
            elif n >= self.plan[uid]:
                # 'stop' drops its token, exactly as the real one does.
                responses.append(FakeResponse(uid, 7, 'stop'))
                del self.live[uid]
            else:
                responses.append(FakeResponse(uid, 7, None))
        room = self.completion_batch_size - len(self.live)
        for _ in range(min(room, self.prefill_batch_size, len(self.queue))):
            self.live[self.queue.pop(0)] = 0
        self.max_live_seen = max(self.max_live_seen, len(self.live))
        return responses

    def next_generated(self):
        # Real next_generated() spins until it has GENERATION responses (a step
        # that only prefilled returns nothing to the caller).
        while True:
            responses = self._step_once()
            if responses:
                return responses
            if not self.live and not self.queue:
                return []

    def close(self):
        self.closed = True


class FakeMlxModel:
    def prepare_input_ids(self, clean, voice):
        idx = int(clean.split()[0][1:])
        return mx.array([[MARKER_BATCH + idx, 1, 2]])

    def parse_output(self, ids):
        return [[ids.tolist()[0][0]]]


class Tee:
    """Keep printing to the terminal AND keep the lines, so the heartbeat can be
    asserted on without hiding the run."""

    def __init__(self, real):
        self.real = real
        self.lines = []
        self._buf = ''
        self._lock = threading.Lock()

    def write(self, s):
        self.real.write(s)
        with self._lock:
            self._buf += s
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                self.lines.append(line)
        return len(s)

    def flush(self):
        self.real.flush()


class Events:
    def __init__(self):
        self.lock = threading.Lock()
        self.saves = []        # (idx, value, thread)
        self.rerenders = []    # idx
        self.rejects = []      # (idx, token_cap)
        self.converts = []     # idx


def build_engine(continuous, events):
    eng = Orpheus.__new__(Orpheus)
    eng.voice = 'testvoice'
    eng.mlx_model = FakeMlxModel()
    eng.backend = 'mlx'
    eng.MLX_DECODE_OVERLAP = True
    eng.MLX_DECODE_JOIN_SECONDS = 30.0
    eng.MLX_CONTINUOUS = continuous
    eng.MLX_CONTINUOUS_PREFILL = PREFILL
    eng.MLX_MAX_TOKENS = MAX_TOKENS
    eng.BATCH_SIZE = WIDTH
    eng._rate_ceilings = {}

    eng._classify_gap = lambda sentence: (0.0, 0.0)
    eng._clean_sentence_for_tts = lambda sentence: sentence
    eng._mlx_token_budget = lambda clean: BUDGET[int(clean.split()[0][1:])]
    # Both paths get the SAME width rule; only the scheduling differs.
    eng._mlx_width_for_depth = lambda depth, steady=False: WIDTH
    eng._voice_cap = lambda key, voice=None: {
        'temperature': 0.6, 'topP': 0.8, 'minP': 0.0, 'repPenalty': 1.1,
    }.get(key, 0.0)
    eng._mlx_eos_boost_processor = lambda n_chars: None

    def fake_convert(idx, sentence):
        with events.lock:
            events.converts.append(idx)
        return False
    eng.convert = fake_convert

    def fake_save_audio(idx, audio, lead, trail):
        with events.lock:
            events.saves.append((idx,
                                 float(audio[0]) if audio is not None and len(audio) else None,
                                 threading.current_thread().name))
        return True
    eng._save_audio = fake_save_audio

    eng._needs_resplit = lambda idx, clean, audio, voice=None: None

    def fake_generate_mlx_safe(clean, depth=0, force_split=False):
        idx = int(clean.split()[0][1:])
        with events.lock:
            events.rerenders.append(idx)
        return np.array([float(MARKER_RERENDER + idx)] * 4, dtype=np.float32)
    eng._generate_mlx_safe = fake_generate_mlx_safe

    def fake_keep_reject(idx, clean, audio, reason, detail=None):
        with events.lock:
            events.rejects.append((idx, (detail or {}).get('token_cap')))
    eng._keep_reject = fake_keep_reject

    eng._ratchet_after_resplit = lambda clean, audio, voice=None: None
    return eng


ITEMS = [(i, f'S{i} ' + 'lorem ipsum ' * 20) for i in range(N_ROWS)]


def run(continuous):
    """One _convert_mlx_batch call against the fake, with the heartbeat's 10 s
    throttle defeated (fake clock jumps 11 s per reading) so EVERY step prints a
    line and `live` can be checked over the whole run."""
    events = Events()
    FakeBatchGenerator.instances = []
    stock_gen = mlx_lm_generate.BatchGenerator
    stock_time = time.time
    tee = Tee(sys.stdout)
    clock = [0.0]

    def fake_time():
        clock[0] += 11.0
        return clock[0]

    mlx_lm_generate.BatchGenerator = FakeBatchGenerator
    time.time = fake_time
    sys.stdout = tee
    try:
        out = build_engine(continuous, events)._convert_mlx_batch(ITEMS)
    finally:
        sys.stdout = tee.real
        time.time = stock_time
        mlx_lm_generate.BatchGenerator = stock_gen
    return out, events, tee.lines, list(FakeBatchGenerator.instances)


HB = re.compile(
    r'\[ORPHEUS\] MLX batch generating: (\d+) rows, ~(\d+) tokens '
    r'\(step (\d+)/(\d+)\), (\d+)/(\d+) rows done, batch (\d+)/(\d+)(?: live (\d+))?$')


def heartbeats(lines):
    return [HB.match(l) for l in lines if HB.match(l)]


# ═════════════════════════════════════════════════════════════════════════════
print('==================== continuous (default) ====================')
out_c, ev_c, lines_c, gens_c = run(continuous=True)

check(len(gens_c) == 1, f'exactly ONE BatchGenerator for the whole call ({len(gens_c)})')
bg = gens_c[0]
check(bg.completion_batch_size == WIDTH,
      f'completion_batch_size is the memory-derived width ({bg.completion_batch_size})')
check(bg.prefill_batch_size == PREFILL,
      f'prefill_batch_size is min(width, MLX_CONTINUOUS_PREFILL) ({bg.prefill_batch_size})')
check(bg.closed, 'the generator was closed')
check(len(bg.inserted_max_tokens or []) == N_ROWS,
      f'insert() got a PER-ROW max_tokens list ({len(bg.inserted_max_tokens or [])} entries)')
check(bg.inserted_max_tokens[SHORT_CAP_ROW] == SHORT_BUDGET
      and bg.inserted_max_tokens[CAP_ROW] == LONG_BUDGET,
      f'each row carries its own cap (row {SHORT_CAP_ROW} = '
      f'{bg.inserted_max_tokens[SHORT_CAP_ROW]}, row {CAP_ROW} = '
      f'{bg.inserted_max_tokens[CAP_ROW]})')
check(bg.max_live_seen <= WIDTH,
      f'the generator never held more than the width live ({bg.max_live_seen})')

check(out_c == [True] * N_ROWS, f'all {N_ROWS} rows succeeded')
saved = sorted(idx for idx, _, _ in ev_c.saves)
check(saved == list(range(N_ROWS)),
      f'every row saved exactly once ({len(saved)} saves, '
      f'{len(set(saved))} distinct)')
values = {idx: value for idx, value, _ in ev_c.saves}
want = {i: float(MARKER_BATCH + i) for i in range(N_ROWS)}
want[CAP_ROW] = float(MARKER_RERENDER + CAP_ROW)
want[SHORT_CAP_ROW] = float(MARKER_RERENDER + SHORT_CAP_ROW)
check(values == want, 'each save carries ITS OWN audio')
check(ev_c.converts == [], f'no row fell back to per-item convert() ({ev_c.converts})')

# The cap check is per row, and it is the ROW's cap that is recorded.
check(sorted(ev_c.rerenders) == sorted([CAP_ROW, SHORT_CAP_ROW]),
      f'exactly the two runaway rows were re-rendered ({sorted(ev_c.rerenders)})')
rejects = dict(ev_c.rejects)
check(rejects.get(SHORT_CAP_ROW) == SHORT_BUDGET,
      f'the short-budget row was caught at ITS OWN cap '
      f'({rejects.get(SHORT_CAP_ROW)}, not the batch depth {MAX_TOKENS})')
check(rejects.get(CAP_ROW) == LONG_BUDGET,
      f'the long runaway was caught at its own cap ({rejects.get(CAP_ROW)})')
check(LONG_CLEAN_ROW not in ev_c.rerenders
      and values[LONG_CLEAN_ROW] == float(MARKER_BATCH + LONG_CLEAN_ROW),
      f'a clean row {PLAN[LONG_CLEAN_ROW] - 1} tokens long — past row '
      f'{SHORT_CAP_ROW}\'s cap of {SHORT_BUDGET} — was NOT mistaken for a runaway')

hb_c = heartbeats(lines_c)
check(len(hb_c) > 5, f'the heartbeat fired every step ({len(hb_c)} lines)')
check(all(m.group(1) == str(N_ROWS) and m.group(7) == '1' and m.group(8) == '1'
          for m in hb_c),
      'the continuous heartbeat reports the whole call as batch 1/1')
check(all(m.group(9) is not None for m in hb_c),
      'every continuous heartbeat carries the additive ` live N` field')
lives = [int(m.group(9)) for m in hb_c]
check(max(lives) <= WIDTH, f'`live` never exceeds the width ({max(lives)} <= {WIDTH})')
check(max(lives) == WIDTH,
      f'`live` reaches the width — the scheduler really did refill ({max(lives)})')
check(all(int(m.group(5)) <= int(m.group(6)) for m in hb_c),
      'rows-done never exceeds the row count')
done = [int(m.group(5)) for m in hb_c]
check(done == sorted(done) and done[-1] == N_ROWS,
      f'rows-done is monotone and exact at completion ({done[-1]}/{N_ROWS})')

announce = [l for l in lines_c if 'MLX continuous batching' in l]
check(len(announce) == 1 and announce[0].startswith(
    f'[ORPHEUS] MLX continuous batching ON: width {WIDTH}, prefill {PREFILL}, '
    f'{N_ROWS} rows queued'),
      f'announced once, naming width/prefill/rows: {announce}')
final = [l for l in lines_c if 'MLX continuous batch done' in l]
check(len(final) == 1 and ' peak ' in final[0] and ' GB' in final[0],
      f'the call ends with the row/step/peak-memory line: {final}')


# ═════════════════════════════════════════════════════════════════════════════
print('\n==================== fresh groups (ORPHEUS_MLX_CONTINUOUS=0) ====================')
out_g, ev_g, lines_g, gens_g = run(continuous=False)

check(len(gens_g) == N_ROWS // WIDTH,
      f'one BatchGenerator per group ({len(gens_g)} for {N_ROWS} rows at width {WIDTH})')
check(all(g.completion_batch_size == WIDTH and g.prefill_batch_size == WIDTH
          for g in gens_g),
      'each group runs at its own full width, prefilled in one go')
check(all(set(g.inserted_max_tokens) == {MAX_TOKENS} for g in gens_g),
      'the group path keeps a UNIFORM cap per group (== the group depth), '
      'as it did before continuous batching existed')
check(all(len(g.inserted_max_tokens) == WIDTH for g in gens_g),
      'each group holds exactly its own rows')

check(out_g == [True] * N_ROWS, f'all {N_ROWS} rows succeeded')
check(sorted(idx for idx, _, _ in ev_g.saves) == list(range(N_ROWS)),
      'every row saved exactly once')
check(ev_g.converts == [], f'no row fell back to per-item convert() ({ev_g.converts})')

hb_g = heartbeats(lines_g)
check(hb_g and all(m.group(9) is None for m in hb_g),
      'the group heartbeat is byte-identical to today — no ` live ` field')
check({m.group(8) for m in hb_g} == {str(N_ROWS // WIDTH)},
      f'the group heartbeat counts the groups ({ {m.group(8) for m in hb_g} })')
check(sorted({int(m.group(7)) for m in hb_g}) == list(range(1, N_ROWS // WIDTH + 1)),
      'every group reported its own batch number')

announce_g = [l for l in lines_g if 'MLX continuous batching' in l]
check(announce_g == ['[ORPHEUS] MLX continuous batching OFF: fresh groups'],
      f'the kill switch announces itself: {announce_g}')
check(not [l for l in lines_g if 'MLX continuous batch done' in l],
      'no continuous-done line on the group path')

# The one place the two paths are DELIBERATELY different.
check(dict(ev_g.rejects).get(SHORT_CAP_ROW) == MAX_TOKENS,
      f'group path: the short-budget row is capped at its GROUP depth '
      f'({dict(ev_g.rejects).get(SHORT_CAP_ROW)}), which is exactly the '
      f'per-row ceiling continuous batching restores')


print('\n==================== RESULT ====================')
if failures:
    print(f'{len(failures)} case(s) FAILED')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('all MLX continuous-batching cases passed')
sys.exit(0)

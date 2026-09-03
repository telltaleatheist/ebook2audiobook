#!/usr/bin/env python
"""_convert_mlx_batch must finish every row exactly once whether the decode runs
on the decoder thread or serially after the batch.

WHAT IS BEING PROVED. A retired row used to wait for the slowest row of its
batch before anything was decoded; now it is handed to ONE decoder thread the
moment its finish_reason lands (ORPHEUS_MLX_DECODE_OVERLAP=1, the default), and
that thread does the model-free half only. The split is the whole risk surface:

  * every row lands in the returned list exactly once, mapped to ITS OWN audio,
    in BOTH modes and with the same values;
  * a cap-hit row and a guard-rerender row are finished on the MAIN thread AFTER
    the batch — _generate_mlx_safe is never called from the decoder thread, so
    the single-sentence path never runs next to a live BatchGenerator;
  * a decode exception fails ONE row (False) and leaves the others alone;
  * rows really are written while the batch is still generating (saves are
    recorded before close());
  * the decoder thread is joined — nothing outlives the call.

HOW. mlx_lm.generate.BatchGenerator is replaced with a fake that retires rows on
a fixed schedule, mlx_audio's llama module (which loads SNAC weights at import)
is stubbed in sys.modules, and the engine's parse_output / _save_audio /
_needs_resplit / _generate_mlx_safe / _keep_reject are fakes that record the
THREAD they ran on. mlx itself is real but pinned to the CPU device, so the
stream plumbing (new_thread_unsafe_stream + mx.stream in another thread) is
genuinely exercised: no model, no GPU, a fraction of a second.
"""
import os
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
    # One "waveform" per row, carrying the row's marker code so a save can be
    # traced back to the row it came from.
    return [[float(code_list[0])] * 4]


_fake_llama.decode_audio_from_codes = _fake_decode_audio_from_codes
for _name in ('mlx_audio', 'mlx_audio.tts', 'mlx_audio.tts.models',
              'mlx_audio.tts.models.llama'):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules['mlx_audio.tts.models.llama.llama'] = _fake_llama

import importlib

# `import mlx_lm.generate as x` binds the generate FUNCTION (mlx_lm re-exports it
# from __init__), not the submodule the engine imports BatchGenerator from.
mlx_lm_generate = importlib.import_module('mlx_lm.generate')
from lib.classes.tts_engines.orpheus import Orpheus

DEPTH = 5            # token cap for the fake batch — small so a cap hit is cheap
MARKER_BATCH = 1000  # audio value for a row decoded from its batch tokens
MARKER_RERENDER = 9000   # audio value for a row re-rendered by _generate_mlx_safe

failures = []


def check(cond, label):
    if cond:
        print(f'  ok   {label}')
    else:
        print(f'  FAIL {label}')
        failures.append(label)


# ---------------------------------------------------------------------------
# The fake batch: six rows, each exercising one branch.
#
#   0  plain row, retires early                      -> decoded, saved
#   1  plain row, retires later                      -> decoded, saved
#   2  hits the token cap (finish_reason 'length')   -> DEFERRED, re-rendered
#   3  clean stop but the guard says 'short'         -> DEFERRED, re-rendered
#   4  parse_output raises                           -> results False
#   5  NEVER reports a finish_reason                 -> swept up after close()
# ---------------------------------------------------------------------------
ROWS = [0, 1, 2, 3, 4, 5]
RETIRE_AT = {0: 2, 1: 4, 2: 6, 3: 3, 4: 5}   # step at which the row retires
# Row 5 never retires AND must not look capped, so it stops emitting after this
# step: its token list stays under DEPTH while the row remains live.
SILENT_AFTER = 3
CAP_ROW = 2
GUARD_ROW = 3
RAISE_ROW = 4
NEVER_RETIRES = 5
TOTAL_STEPS = 7          # the fake stops generating here, row 5 still unretired


class FakeResponse:
    __slots__ = ('uid', 'token', 'finish_reason')

    def __init__(self, uid, token, finish_reason):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason


class Recorder:
    """Every observable event, in order, with the thread that produced it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = []

    def add(self, kind, **fields):
        with self.lock:
            fields['kind'] = kind
            fields['thread'] = threading.current_thread().name
            fields['seq'] = len(self.events)
            self.events.append(fields)
            return fields

    def of(self, kind):
        with self.lock:
            return [e for e in self.events if e['kind'] == kind]


class FakeBatchGenerator:
    """Retires rows on RETIRE_AT. `before_close` lets the test wait for the
    decoder thread to have written something WHILE generation is still live."""

    before_close = None

    def __init__(self, *_args, **_kwargs):
        self.uids = []
        self.step = 0
        self.retired = set()

    def insert(self, prompts, max_tokens=None, logits_processors=None):
        # Per-row max_tokens is what the continuous path needs; the group path
        # passes [depth] * n, which is BatchGenerator's own default. Either way
        # the fake records it so the cap check can be read back.
        self.uids = [f'u{i}' for i in range(len(prompts))]
        self.max_tokens = list(max_tokens) if max_tokens else [DEPTH] * len(prompts)
        return list(self.uids)

    def next_generated(self):
        if self.step >= TOTAL_STEPS:
            return []
        self.step += 1
        responses = []
        for row, uid in enumerate(self.uids):
            if uid in self.retired:
                continue
            retire_at = RETIRE_AT.get(row)
            if retire_at is not None and self.step >= retire_at:
                self.retired.add(uid)
                # 'length' rows keep their last token (the cap), 'stop' rows
                # drop it — exactly what the real BatchGenerator reports.
                reason = 'length' if row == CAP_ROW else 'stop'
                responses.append(FakeResponse(uid, 7, reason))
            elif retire_at is not None or self.step <= SILENT_AFTER:
                responses.append(FakeResponse(uid, 7, None))
        return responses

    def close(self):
        if FakeBatchGenerator.before_close is not None:
            FakeBatchGenerator.before_close()
        RECORDER.add('close')


class FakeMlxModel:
    def prepare_input_ids(self, clean, voice):
        # A [1, T] array whose [0].tolist() is the prompt token list. The first
        # token is the row's marker, so parse_output can tell rows apart.
        idx = int(clean.split()[0][1:])
        return mx.array([[MARKER_BATCH + idx, 1, 2]])

    def parse_output(self, ids):
        marker = ids.tolist()[0][0]
        RECORDER.add('parse_output', marker=marker)
        if marker - MARKER_BATCH == RAISE_ROW:
            raise RuntimeError('synthetic parse_output failure')
        return [[marker]]


def build_engine(overlap):
    eng = Orpheus.__new__(Orpheus)
    eng.voice = 'testvoice'
    eng.mlx_model = FakeMlxModel()
    eng.MLX_DECODE_OVERLAP = overlap
    eng.MLX_DECODE_JOIN_SECONDS = 30.0
    # This file tests the FRESH-GROUP path (the _mlx_batch_groups stub below is
    # what it drives). Continuous batching has its own file,
    # tools/test_mlx_continuous.py.
    eng.MLX_CONTINUOUS = False
    eng.MLX_MAX_TOKENS = DEPTH
    eng._rate_ceilings = {}

    eng._classify_gap = lambda sentence: (0.0, 0.0)
    eng._clean_sentence_for_tts = lambda sentence: sentence
    eng._mlx_token_budget = lambda clean: DEPTH
    eng._mlx_batch_groups = lambda gen: [(list(gen), DEPTH)]
    eng._voice_cap = lambda key, voice=None: {
        'temperature': 0.6, 'topP': 0.8, 'minP': 0.0, 'repPenalty': 1.1,
    }.get(key, 0.0)
    eng._mlx_eos_boost_processor = lambda n_chars: None
    eng.convert = lambda idx, sentence: RECORDER.add('convert', idx=idx) and False

    def fake_save_audio(idx, audio, lead, trail):
        RECORDER.add('save', idx=idx,
                     value=(float(audio[0]) if audio is not None and len(audio) else None))
        return True
    eng._save_audio = fake_save_audio

    def fake_needs_resplit(idx, clean, audio_np, voice=None):
        RECORDER.add('verdict', idx=idx)
        return 'short' if idx == GUARD_ROW else None
    eng._needs_resplit = fake_needs_resplit

    def fake_generate_mlx_safe(clean, depth=0, force_split=False):
        idx = int(clean.split()[0][1:])
        RECORDER.add('rerender', idx=idx, force_split=force_split)
        return np.array([float(MARKER_RERENDER + idx)] * 4, dtype=np.float32)
    eng._generate_mlx_safe = fake_generate_mlx_safe

    def fake_keep_reject(idx, clean, audio_np, reason, detail=None):
        RECORDER.add('keep_reject', idx=idx, reason=reason)
    eng._keep_reject = fake_keep_reject

    def fake_ratchet(clean, audio_np, voice=None):
        RECORDER.add('ratchet')
    eng._ratchet_after_resplit = fake_ratchet
    return eng


ITEMS = [(i, f'S{i} ' + 'lorem ipsum ' * 30) for i in ROWS]


def run(overlap, before_close=None):
    global RECORDER
    RECORDER = Recorder()
    FakeBatchGenerator.before_close = before_close
    stock = mlx_lm_generate.BatchGenerator
    mlx_lm_generate.BatchGenerator = FakeBatchGenerator
    try:
        return build_engine(overlap)._convert_mlx_batch(ITEMS), RECORDER
    finally:
        mlx_lm_generate.BatchGenerator = stock
        FakeBatchGenerator.before_close = None


def expected_values():
    """idx -> the audio value its save must carry (None = the row must fail)."""
    want = {i: float(MARKER_BATCH + i) for i in ROWS}
    want[CAP_ROW] = float(MARKER_RERENDER + CAP_ROW)
    want[GUARD_ROW] = float(MARKER_RERENDER + GUARD_ROW)
    want[RAISE_ROW] = None
    return want


def assert_common(label, out, rec):
    want = expected_values()
    check(len(out) == len(ITEMS), f'{label}: one result per item ({len(out)})')
    check(out == [want[i] is not None for i in ROWS],
          f'{label}: results aligned to items = {out}')

    saves = rec.of('save')
    saved_idx = [e['idx'] for e in saves]
    expected_saved = sorted(i for i in ROWS if want[i] is not None)
    check(sorted(saved_idx) == expected_saved,
          f'{label}: every finishable row saved exactly once ({sorted(saved_idx)})')
    check(len(saved_idx) == len(set(saved_idx)), f'{label}: no row saved twice')
    mapping_ok = all(e['value'] == want[e['idx']] for e in saves)
    check(mapping_ok,
          f'{label}: each save carries ITS OWN audio '
          f'({ {e["idx"]: e["value"] for e in saves} })')

    rerendered = sorted(e['idx'] for e in rec.of('rerender'))
    check(rerendered == sorted([CAP_ROW, GUARD_ROW]),
          f'{label}: exactly the cap row and the guard row were re-rendered ({rerendered})')
    check(all(e['force_split'] for e in rec.of('rerender')),
          f'{label}: every re-render asked for force_split')
    check([e['reason'] for e in rec.of('keep_reject')] == ['cap'],
          f'{label}: the cap row kept its runaway as evidence')
    check(len(rec.of('ratchet')) == 1,
          f'{label}: the ratchet fired once, for the short row')
    check(rec.of('convert') == [],
          f'{label}: no row fell back to the per-item convert() recovery')


print('==================== serial (ORPHEUS_MLX_DECODE_OVERLAP=0) ====================')
out_serial, rec_serial = run(overlap=False)
assert_common('serial', out_serial, rec_serial)
main_name = threading.current_thread().name
off_main = [e for e in rec_serial.events if e['thread'] != main_name]
check(not off_main, f'serial: everything ran on the main thread ({off_main})')
check(all(e['seq'] > rec_serial.of('close')[0]['seq'] for e in rec_serial.of('save')),
      'serial: every save happened AFTER the batch closed')


print('\n==================== overlap (default) ====================')
# Hold close() until the decoder thread has written the rows that retired
# early — that is what "overlapped" means, and without the wait the assertion
# would be a race rather than a proof.
def wait_for_early_saves():
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if len([e for e in RECORDER.of('save')]) >= 2:
            return
        time.sleep(0.01)


out_overlap, rec_overlap = run(overlap=True, before_close=wait_for_early_saves)
assert_common('overlap', out_overlap, rec_overlap)
check(out_overlap == out_serial, 'overlap: identical result list to the serial path')

close_seq = rec_overlap.of('close')[0]['seq']
early = [e for e in rec_overlap.of('save') if e['seq'] < close_seq]
check(len(early) >= 2,
      f'overlap: rows were written WHILE the batch was still generating '
      f'({[e["idx"] for e in early]} before close)')

worker_saves = {e['idx'] for e in rec_overlap.of('save')
                if e['thread'].startswith('orpheus-mlx-decode')}
check(worker_saves == {0, 1, NEVER_RETIRES},
      f'overlap: the plain rows were saved by the decoder thread ({sorted(worker_saves)})')

main_saves = {e['idx'] for e in rec_overlap.of('save') if e['thread'] == main_name}
check(main_saves == {CAP_ROW, GUARD_ROW},
      f'overlap: the deferred rows were saved by the MAIN thread ({sorted(main_saves)})')

rerender_threads = {e['thread'] for e in rec_overlap.of('rerender')}
check(rerender_threads == {main_name},
      f'overlap: _generate_mlx_safe ran ONLY on the main thread ({rerender_threads})')
check({e['thread'] for e in rec_overlap.of('keep_reject')} == {main_name},
      'overlap: _keep_reject for the cap row ran on the main thread')
check(all(e['seq'] > close_seq for e in rec_overlap.of('rerender')),
      'overlap: every model re-render happened AFTER bg.close()')

verdict_threads = {e['thread'] for e in rec_overlap.of('verdict')}
check(verdict_threads and verdict_threads.issubset({'orpheus-mlx-decode-1'}),
      f'overlap: the truncation VERDICT was taken on the decoder thread ({verdict_threads})')

check(out_overlap[RAISE_ROW] is False and sum(1 for v in out_overlap if v) == 5,
      'overlap: the raising row failed alone; the other five rows survived')

live = [t.name for t in threading.enumerate() if t.name.startswith('orpheus-mlx-decode')]
check(not live, f'overlap: no decoder thread outlived the call ({live})')


print('\n==================== RESULT ====================')
if failures:
    print(f'{len(failures)} case(s) FAILED')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('all MLX decode-overlap cases passed')
sys.exit(0)

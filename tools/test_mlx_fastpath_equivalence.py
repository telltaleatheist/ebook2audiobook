#!/usr/bin/env python
"""The MLX fast path's batched logits math must equal mlx-lm's per-row loop.

The fast path (lib/classes/tts_engines/orpheus_mlx_fastpath.py) replaces
GenerationBatch._step with a version that (a) applies the repetition penalty and
the EOS boost to the whole batch in two array ops instead of a Python loop over
rows, and (b) projects the head onto only the 28,680 ids Orpheus can emit. (b)
is a restriction of the same matmul and is exact by construction; (a) is a
REWRITE, and this is what proves the rewrite.

WHAT IS COMPARED. For a batch of synthetic rows, the reference applies
`mlx_lm.sample_utils.make_repetition_penalty` and a literal copy of the EOS
boost closure as it stood before this change, per row, over the FULL 156,940-wide
logits — exactly what stock GenerationBatch._step does. The fast path's own
functions (_bf_sync / _bf_mark_inputs / _bf_row_scalars / _bf_apply) run over
the sliced logits. The two must agree on the slice.

THE ROWS exercise every branch that differs:
  0  rep penalty only, no boost                       (mixed-sign logits)
  1  rep + boost, well past its start                 (ramp, uncapped)
  2  rep + boost, n EXACTLY at start                  (must NOT fire: n > start)
  3  rep + boost, n below start                       (must not fire)
  4  rep only, every logit forced NEGATIVE            (the l*p arm)
  5  rep only, every logit forced POSITIVE            (the l/p arm)
  6  no processors at all                             (identity)
  Row 1's overrun is large enough to hit the 4x cap on the ramp, and every row's
  history mixes prompt-text ids (outside the emittable block, which the mask must
  ignore) with repeated in-slice ids (which the mask must collapse the same way
  stock's duplicate scatter does).

Runs on the CPU: no model, no GPU, a few seconds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx_lm.sample_utils import make_repetition_penalty
from lib.classes.tts_engines import orpheus_mlx_fastpath as fp

V = 156940
B = 7
TOL = 1e-6

failures = []


def check(cond, label):
    if cond:
        print(f'  ok   {label}')
    else:
        print(f'  FAIL {label}')
        failures.append(label)


def reference_boost(base, start, expected, eos):
    """The `_boost` closure of Orpheus._mlx_eos_boost_processor as it stood
    BEFORE this change — copied here so the reference cannot drift into being
    the thing under test."""
    def _boost(tokens, logits):
        n = len(tokens)
        if n > start:
            bias = base * min(4.0, 1.0 + (n - start) / expected)
            return logits.at[:, eos].add(bias)
        return logits
    return _boost


class FakeBatch:
    """Only the attributes _bf_sync / _bf_row_scalars read off a GenerationBatch."""

    def __init__(self, uids, tokens, logits_processors):
        self.uids = uids
        self.tokens = tokens
        self.logits_processors = logits_processors


# ─────────────────────────────────────────────────────────────────────────────
# 1. batched penalty + boost == stock per-row loop
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== batched penalty + boost vs the stock per-row loop ===')

PENALTY = 1.1
WINDOW = 8192
EOS = fp.END_OF_AUDIO_TOKEN

rng = np.random.default_rng(20260901)


def history(n_text, n_codes, repeats, seed):
    """A row's KV-cache tokens: some prompt text ids, then in-slice codes with
    deliberate repeats (the case where stock's gather writes the same column
    twice and the mask must not double-apply)."""
    r = np.random.default_rng(seed)
    text = r.integers(0, 128000, size=n_text).tolist()
    codes = r.integers(fp.AUDIO_LO, fp.SLICE_HI, size=n_codes).tolist()
    codes = codes + codes[:repeats]                 # exact duplicates
    codes.append(EOS)                               # EOS lives in the slice too
    return text + codes


# (n_generated_hint, base, start, expected) per row; start/expected are the
# arithmetic the real _mlx_eos_boost_processor produces.
rows = [
    dict(label='rep only, mixed signs',    boost=None,                 hist=(12, 40, 5)),
    dict(label='boost past start (4x cap)', boost=(1.75, 300.0, 40.0),  hist=(9, 900, 11)),
    dict(label='boost exactly AT start',    boost=(1.75, 251.0, 60.0),  hist=(10, 240, 0)),
    dict(label='boost below start',         boost=(2.5, 5000.0, 400.0), hist=(8, 300, 3)),
    dict(label='rep only, all negative',    boost=None,                 hist=(7, 55, 9)),
    dict(label='rep only, all positive',    boost=None,                 hist=(6, 61, 2)),
    dict(label='no processors',             boost=None,                 hist=(15, 33, 4), bare=True),
]

tokens = []
procs = []
for i, spec in enumerate(rows):
    tokens.append(history(*spec['hist'], seed=1000 + i))
    if spec.get('bare'):
        procs.append([])
        continue
    p = [fp.make_rep_penalty(PENALTY, WINDOW)]
    if spec['boost'] is not None:
        base, start, expected = spec['boost']
        p.append(fp.make_eos_boost(base, start, expected, EOS))
    procs.append(p)

# Row 2 must sit exactly at n == start. n is len(tokens[i]) + 1.
n2 = len(tokens[2]) + 1
rows[2]['boost'] = (1.75, float(n2), 60.0)
procs[2] = [fp.make_rep_penalty(PENALTY, WINDOW),
            fp.make_eos_boost(1.75, float(n2), 60.0, EOS)]
check(len(tokens[1]) + 1 - rows[1]['boost'][1] > 4.0 * rows[1]['boost'][2],
      'row 1 overruns far enough to saturate the 4x ramp cap')

# The current input token of each step. Row 6 is given an out-of-slice input
# (128257, START_OF_SPEECH — what the real first step sees) so the "mark
# nothing" branch is exercised.
inputs_list = [
    int(rng.integers(fp.AUDIO_LO, fp.SLICE_HI)),
    int(rng.integers(fp.AUDIO_LO, fp.SLICE_HI)),
    EOS,
    int(rng.integers(fp.AUDIO_LO, fp.SLICE_HI)),
    int(rng.integers(fp.AUDIO_LO, fp.SLICE_HI)),
    int(rng.integers(fp.AUDIO_LO, fp.SLICE_HI)),
    128257,
]

full = rng.standard_normal((B, V)).astype(np.float32) * 3.0
full[4] = -np.abs(full[4]) - 0.5      # every logit negative  -> l * p
full[5] = np.abs(full[5]) + 0.5       # every logit positive  -> l / p
full_mx = mx.array(full)

# ---- reference: stock, per row, over the full vocab -------------------------
ref_rows = []
for i in range(B):
    buf = mx.array(tokens[i] + [inputs_list[i]], dtype=mx.int32)
    row = full_mx[i:i + 1]
    if procs[i]:
        row = make_repetition_penalty(PENALTY, WINDOW)(buf, row)
        spec = rows[i]['boost']
        if spec is not None:
            base, start, expected = spec
            row = reference_boost(base, start, expected, EOS)(buf, row)
    ref_rows.append(row[0, fp.SLICE_LO:fp.SLICE_HI])
ref = mx.stack(ref_rows)
mx.eval(ref)

# ---- fast path: batched, over the slice ------------------------------------
batch = FakeBatch(uids=list(range(B)), tokens=[list(t) for t in tokens],
                  logits_processors=procs)
state = fp._bf_sync(batch)
fp._bf_mark_inputs(state, mx.array(inputs_list, dtype=mx.uint32), B)
bias, any_bias = fp._bf_row_scalars(batch, state)
fast = fp._bf_apply(full_mx[:, fp.SLICE_LO:fp.SLICE_HI], state, bias, any_bias)
mx.eval(fast)

diff = np.abs(np.array(fast) - np.array(ref))
for i, spec in enumerate(rows):
    check(diff[i].max() <= TOL,
          f'row {i} ({spec["label"]}) matches stock, max |diff| = {diff[i].max():.3e}')
check(diff.max() <= TOL, f'whole batch within {TOL}, max |diff| = {diff.max():.3e}')

# ---- the same, in bf16: production logits ARE bf16 -------------------------
# Stock multiplies a bf16 row by a Python float; the fast path multiplies by a
# bf16 penalty array. Those must agree exactly on the bf16 grid, or the "same
# 1.1 penalty" is quietly two different numbers.
full_bf = full_mx.astype(mx.bfloat16)
ref_bf_rows = []
for i in range(B):
    buf = mx.array(tokens[i] + [inputs_list[i]], dtype=mx.int32)
    row = full_bf[i:i + 1]
    if procs[i]:
        row = make_repetition_penalty(PENALTY, WINDOW)(buf, row)
        spec = rows[i]['boost']
        if spec is not None:
            base, start, expected = spec
            row = reference_boost(base, start, expected, EOS)(buf, row)
    ref_bf_rows.append(row[0, fp.SLICE_LO:fp.SLICE_HI])
ref_bf = mx.stack(ref_bf_rows)
batch_bf = FakeBatch(uids=list(range(B)), tokens=[list(t) for t in tokens],
                     logits_processors=procs)
state_bf = fp._bf_sync(batch_bf)
fp._bf_mark_inputs(state_bf, mx.array(inputs_list, dtype=mx.uint32), B)
bias_bf, any_bias_bf = fp._bf_row_scalars(batch_bf, state_bf)
fast_bf = fp._bf_apply(full_bf[:, fp.SLICE_LO:fp.SLICE_HI], state_bf, bias_bf, any_bias_bf)
mx.eval(ref_bf, fast_bf)
check(fast_bf.dtype == mx.bfloat16, f'fast path keeps bf16 logits bf16 ({fast_bf.dtype})')
diff_bf = np.abs(np.array(fast_bf.astype(mx.float32)) - np.array(ref_bf.astype(mx.float32)))
check(diff_bf.max() == 0.0,
      f'bf16 batch is BIT-identical to stock, max |diff| = {diff_bf.max():.3e}')

# The boost must have fired on exactly one row, and been clamped at 4x.
check(bias[0] == 0.0 and bias[2] == 0.0 and bias[3] == 0.0 and bias[1] > 0.0,
      f'boost fired on row 1 only (bias = {bias})')
check(abs(bias[1] - rows[1]['boost'][0] * 4.0) < 1e-12,
      f'row 1 boost is clamped at 4x base ({bias[1]} vs {rows[1]["boost"][0] * 4.0})')
check(any_bias, 'any_bias is set when a row fires')

# Untouched columns must be bit-identical, not merely close.
sliced_in = np.array(full_mx[:, fp.SLICE_LO:fp.SLICE_HI])
untouched = ~np.array(state.seen)
untouched[:, fp.EOS_INDEX] = False
check(np.array_equal(np.array(fast)[untouched], sliced_in[untouched]),
      'columns no row has seen are byte-identical to the input logits')

# The seen mask must ignore out-of-slice history and the out-of-slice input.
seen_np = np.array(state.seen)
expected6 = np.zeros(fp.SLICE_N, dtype=bool)
codes6 = np.array([t for t in tokens[6] if fp.SLICE_LO <= t < fp.SLICE_HI])
expected6[codes6 - fp.SLICE_LO] = True
check(np.array_equal(seen_np[6], expected6),
      'row 6: out-of-slice history and an out-of-slice input mark nothing')
check(seen_np[2][EOS - fp.SLICE_LO],
      'row 2: the current input (EOS) is marked before the penalty runs')

# ─────────────────────────────────────────────────────────────────────────────
# 2. the marked factories still behave exactly like the stock closures
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== marked factories are behaviourally identical ===')

probe_tokens = mx.array(tokens[1] + [inputs_list[1]], dtype=mx.int32)
probe = mx.array(full[1:2])
a = fp.make_rep_penalty(PENALTY, WINDOW)(probe_tokens, mx.array(full[1:2]))
b = make_repetition_penalty(PENALTY, WINDOW)(probe_tokens, mx.array(full[1:2]))
mx.eval(a, b)
check(np.array_equal(np.array(a), np.array(b)),
      'make_rep_penalty == mlx_lm.make_repetition_penalty, bit for bit')

base, start, expected = 1.75, 300.0, 40.0
a = fp.make_eos_boost(base, start, expected, EOS)(probe_tokens, mx.array(full[1:2]))
b = reference_boost(base, start, expected, EOS)(probe_tokens, mx.array(full[1:2]))
mx.eval(a, b)
check(np.array_equal(np.array(a), np.array(b)),
      'make_eos_boost == the pre-change _boost closure, bit for bit')

# ─────────────────────────────────────────────────────────────────────────────
# 3. state rebuild on a uid change
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== state follows the live row set ===')

keep = [0, 2, 5]
batch.uids = [batch.uids[i] for i in keep]
batch.tokens = [batch.tokens[i] for i in keep]
batch.logits_processors = [batch.logits_processors[i] for i in keep]
state2 = fp._bf_sync(batch)
mx.eval(state2.seen)
check(state2.seen.shape == (len(keep), fp.SLICE_N), 'seen is re-shaped to the kept rows')
check(np.array_equal(np.array(state2.seen), seen_np[np.array(keep)]),
      'a retirement permutes the mask rather than losing or shifting a row')

# A row the state has never seen forces a full rebuild from self.tokens.
batch.uids = [99]
batch.tokens = [list(tokens[3])]
batch.logits_processors = [procs[3]]
state3 = fp._bf_sync(batch)
mx.eval(state3.seen)
rebuilt = np.zeros(fp.SLICE_N, dtype=bool)
c3 = np.array([t for t in tokens[3] if fp.SLICE_LO <= t < fp.SLICE_HI])
rebuilt[c3 - fp.SLICE_LO] = True
check(np.array_equal(np.array(state3.seen)[0], rebuilt),
      'an unknown uid rebuilds the mask exactly from the row\'s KV-cache tokens')

# A uid set that GROWS is the continuous-batching case: BatchGenerator._next
# prefills queued prompts into their own GenerationBatch and extend()s it into
# the live one, so mid-generation self.uids gains ids the state has never seen.
# _bf_sync must take the FULL-REBUILD branch there (the permute branch is only
# valid for a subset) and rebuild every row exactly from self.tokens.
#
# WHY len(self.tokens[i]) IS STILL THE RIGHT LENGTH AFTER AN EXTEND, even though
# the row's KV cache is now PADDED to the oldest live row's _idx:
#   * BatchKVCache.extend right-justifies the new rows at max_idx and records the
#     gap as left_padding; create_causal_mask ANDs in `left_padding <= rinds`, so
#     those positions are masked out of attention and are not tokens.
#   * self.tokens is mlx-lm's own record of the tokens IN the cache
#     (PromptProcessingBatch.prompt appends the prompt, GenerationBatch._step
#     appends each generated token); padding is never appended to it.
#   * stock's own penalty window is measured over _token_context, a TokenBuffer
#     seeded from `tokens` at GenerationBatch construction and concatenated by
#     extend() — the same quantity, never the padded length.
# So the window bound below must fire on prompt length + 1, not on the padded
# length, and the mask must be built from the row's real tokens only.
print('\n=== a uid set that GROWS (BatchGenerator.extend) rebuilds exactly ===')

old_uid, old_tokens = 7, history(11, 120, 4, seed=7007)
new_prompts = [
    [128259, 128000, 3923, 128009, 128260, 128261, 128257],          # all text ids
    [128259, 128000, 5000, 128009, 128260, 128261,
     fp.AUDIO_LO + 33, fp.AUDIO_LO + 33, 128257],                    # + repeats
]
new_uids = [8, 9]

grow = FakeBatch(uids=[old_uid], tokens=[list(old_tokens)],
                 logits_processors=[[fp.make_rep_penalty(PENALTY, WINDOW)]])
before = fp._bf_sync(grow)
mx.eval(before.seen)

# ... one step later mlx-lm prefilled two queued rows and extended them in.
grow.uids = [old_uid] + new_uids
grow.tokens = [list(old_tokens)] + [list(p) for p in new_prompts]
grow.logits_processors = [[fp.make_rep_penalty(PENALTY, WINDOW)]] * 3
after = fp._bf_sync(grow)
mx.eval(after.seen)

check(after is not before, 'the grown uid set invalidates the cached state')
check(after.seen.shape == (3, fp.SLICE_N),
      f'seen grew to the new live row count ({after.seen.shape})')
after_np = np.array(after.seen)
check(np.array_equal(after_np[0], np.array(before.seen)[0]),
      'the row that was already generating keeps its accumulated history')
for k, prompt in enumerate(new_prompts, start=1):
    want = np.zeros(fp.SLICE_N, dtype=bool)
    codes = np.array([t for t in prompt if fp.SLICE_LO <= t < fp.SLICE_HI],
                     dtype=np.int64)
    if codes.size:
        want[codes - fp.SLICE_LO] = True
    check(np.array_equal(after_np[k], want),
          f'extended row {k} is rebuilt EXACTLY from its own prompt tokens '
          f'({int(after_np[k].sum())} marked, {int(want.sum())} expected)')

# The window bound is the row's own token count, not the padded cache length.
# Row 0 carries ~135 tokens; the extended rows carry 7 and 9. A window of 10
# must therefore accept both new rows and refuse only the old one.
narrow = FakeBatch(uids=new_uids, tokens=[list(p) for p in new_prompts],
                   logits_processors=[[fp.make_rep_penalty(PENALTY, 10)]] * 2)
bias_n, _ = fp._bf_row_scalars(narrow, fp._bf_sync(narrow))
check(bias_n == [0.0, 0.0],
      'a freshly extended row is bounded by len(self.tokens[i]) + 1 — its PROMPT '
      'length — not by the padded KV length it now shares with the straggler')
too_long = FakeBatch(uids=[old_uid] + new_uids,
                     tokens=[list(old_tokens)] + [list(p) for p in new_prompts],
                     logits_processors=[[fp.make_rep_penalty(PENALTY, 10)]] * 3)
try:
    fp._bf_row_scalars(too_long, fp._bf_sync(too_long))
    check(False, 'a straggler past the window still raises after an extend')
except fp.FastPathUnsupported as err:
    check('repetition window is 10' in str(err),
          f'a straggler past the window still raises after an extend ({err})')


# ─────────────────────────────────────────────────────────────────────────────
# 4. processors the batched form cannot reproduce are REFUSED, not ignored
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== unbatchable processors are refused by name ===')


def refuses(fn, label, needle):
    try:
        fn()
    except fp.FastPathUnsupported as err:
        ok = needle in str(err)
        check(ok, f'{label} -> {"names the reason" if ok else "WRONG message: " + str(err)}')
        return
    check(False, f'{label} -> did NOT raise FastPathUnsupported')


refuses(lambda: fp._row_params([make_repetition_penalty(1.1, 20)], 0),
        'an unmarked mlx-lm processor', 'unmarked logits processor')
refuses(lambda: fp._row_params([fp.make_eos_boost(1.0, 1.0, 1.0, EOS),
                                fp.make_rep_penalty(1.1, WINDOW)], 3),
        'boost ordered before the penalty', 'AFTER the EOS boost')
refuses(lambda: fp._row_params([fp.make_rep_penalty(1.1, WINDOW),
                                fp.make_rep_penalty(1.2, WINDOW)], 1),
        'two repetition penalties', 'two repetition penalties')
refuses(lambda: fp._row_params([fp.make_eos_boost(1.0, 1.0, 1.0, 128009)], 2),
        'a boost on a token other than EOS', 'not END_OF_AUDIO')

# The per-step window bound.
tiny = FakeBatch(uids=[0], tokens=[list(range(fp.AUDIO_LO, fp.AUDIO_LO + 64))],
                 logits_processors=[[fp.make_rep_penalty(1.1, 8)]])
refuses(lambda: fp._bf_row_scalars(tiny, fp._bf_sync(tiny)),
        'a KV cache longer than the penalty window', 'repetition window is 8')

# ─────────────────────────────────────────────────────────────────────────────
# 5. install() refuses the models it cannot serve
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== install() refusals ===')

import types
import mlx.nn as nn
import mlx_lm


class FakeInner:
    def __init__(self, embed):
        self.embed_tokens = embed


class FakeModel:
    def __init__(self, embed, tied=True):
        self.args = types.SimpleNamespace(tie_word_embeddings=tied)
        self.model = FakeInner(embed)


good_embed = nn.Embedding(fp.SLICE_HI + 2, 8)
mx.eval(good_embed.weight)
good = FakeModel(good_embed)

real_version = mlx_lm.__version__
try:
    mlx_lm.__version__ = '0.32.0'
    refuses(lambda: fp.install(good, rep_window=8192, max_tokens=3700),
            'a different mlx-lm version', 'pinned to mlx-lm 0.31.3')
finally:
    mlx_lm.__version__ = real_version

refuses(lambda: fp.install(good, rep_window=4096, max_tokens=3700),
        'a repetition window that cannot cover prompt + generation',
        'does not cover a full generation')
refuses(lambda: fp.install(FakeModel(good_embed, tied=False),
                           rep_window=8192, max_tokens=3700),
        'an untied head', 'TIED input embedding')

quant = nn.QuantizedEmbedding(fp.SLICE_HI + 2, 64)
mx.eval(quant.weight, quant.scales)
refuses(lambda: fp.install(FakeModel(quant), rep_window=8192, max_tokens=3700),
        'a quantized embedding', 'QuantizedEmbedding')

refuses(lambda: fp.install(FakeModel(nn.Embedding(1024, 8)),
                           rep_window=8192, max_tokens=3700),
        'a vocabulary too small to hold the SNAC block', 'not an Orpheus checkpoint')

# The good case installs, is idempotent, and can be undone.
from mlx_lm.generate import GenerationBatch
stock_step = GenerationBatch._step
line = fp.install(good, rep_window=8192, max_tokens=3700)
check(GenerationBatch._step is fp._bf_step, 'install() replaces GenerationBatch._step')
check(good._bf_fastpath_head.shape == (fp.SLICE_N, 8),
      f'the head slice is [{fp.SLICE_N}, hidden] ({good._bf_fastpath_head.shape})')
check(isinstance(line, str) and 'fast path installed' in line,
      f'install() returns a log line: {line}')
fp.install(good, rep_window=8192, max_tokens=3700)
check(GenerationBatch._stock_step is stock_step, 'install() is idempotent')
fp.uninstall(good)
check(GenerationBatch._step is stock_step, 'uninstall() restores the stock step')
check(not hasattr(good, '_bf_fastpath_head'), 'uninstall() drops the head slice')

print('\n==================== RESULT ====================')
if failures:
    print(f'{len(failures)} case(s) FAILED')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('all MLX fast-path equivalence cases passed')
sys.exit(0)

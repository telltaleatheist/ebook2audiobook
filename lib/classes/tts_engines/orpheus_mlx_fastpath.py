"""Batched decode fast path for Orpheus on MLX.

Replaces mlx-lm's `GenerationBatch._step` with a version that removes two costs
the stock implementation pays on EVERY decode step. Nothing here changes the
model, the weights or the prompt framing; it is a re-expression of the same
arithmetic in a shape the GPU can do in one pass.

1. THE PER-ROW PYTHON LOGITS LOOP. mlx-lm 0.31.3 (generate.py:1330-1345) loops
   the batch in Python whenever ANY row carries a logits processor: per row it
   slices [1, V] out of the batch, gathers the repetition-penalty window,
   scatters it back, applies the EOS boost, then concatenates B slices again.
   Measured on an M1 Ultra at width 96 with 260-char rows: 151.7 ms/step with
   the processors, 142.1 ms/step with them removed entirely — i.e. the loop is
   ~9.6 ms/step, ~6% of decode, and it grows linearly with batch width.

2. THE DEAD 82% OF THE LM HEAD. Orpheus can only ever emit END_OF_AUDIO
   (128258) or a SNAC code ([128266, 128266 + 7*4096)); `_redistribute_codes`
   in orpheus.py keeps exactly that range and throws the rest away. The stock
   head still projects the hidden state onto all 156,940 rows of the tied
   embedding and discards 128,260 of the results. Slicing the tied embedding
   once to the contiguous block [128258, 156938) cuts the head's weight read
   from 964 MB to 176 MB per step and its FLOPs from 90.6 to 16.6 GFLOP at
   width 96 — and shrinks every op downstream of it (penalty, logsumexp,
   sampler) by 5.5x with it.

THE TWO NON-IDENTITIES, stated honestly:

(a) SAMPLING IS NOT BIT-IDENTICAL TO STOCK, even with the same RNG. top_p/min_p
    nucleus selection is computed over the emittable domain, so the softmax
    normaliser no longer includes the 128,260 dead ids. The measured mass out
    there is tiny but it is not zero, so a borderline nucleus cut can land
    differently.
(b) It cannot produce a token the stock path could not. The excluded ids are
    precisely the ones `_redistribute_codes` discards, so a sample that stock
    could have drawn and this cannot is a sample that would have been thrown
    away — the row would have produced a shorter clip, not a different one.

EXACTNESS OF THE PENALTY. The repetition penalty is reproduced with a per-row
SEEN MASK over the emittable block instead of a token history. That is exact
only while the penalty window covers prompt + whole generation: then "ids in the
window" and "ids this row has in its KV cache" are the same set. `install()`
refuses when the configured window cannot cover it, and the step re-checks the
exact per-row bound every step and raises rather than silently changing
sampling. Duplicate ids in the stock gather assign the same value, so a mask is
equivalent to the scatter.

WHAT IS NOT USED. `GenerationBatch._token_context` (the per-row `TokenBuffer`)
is never read by this step — the seen mask is rebuilt from `self.tokens`, which
holds exactly the tokens in the KV cache, so the buffers stay correct but idle.
That is also why `filter()`/`extend()` need no patch: any change to the live row
set changes `self.uids`, and the state is re-derived from `self.tokens` (or
permuted from the previous state) when it does.

`Response.logprobs` are now over the SLICED vocab (28,680 wide, index 0 ==
END_OF_AUDIO). orpheus.py never reads them — both MLX batch call sites use only
`r.uid`, `r.token` and `r.finish_reason` — so nothing downstream sees the
change. Anything that starts reading them must add SLICE_LO to the index.

THE HEAD SLICE AND LoRA. The slice is taken from `model.model.embed_tokens`,
an `nn.Embedding`. The MLX adapter path (`Orpheus._mlx_adapter_plan`) refuses by
name to wrap anything that is not an `nn.Linear`, so no adapter can ever change
the tied embedding under a cached slice; a voice switch leaves the head valid.
"""
import numpy as np
import mlx.core as mx

# Orpheus token geography (orpheus.py: END_OF_AUDIO_TOKEN, _redistribute_codes).
END_OF_AUDIO_TOKEN = 128258
AUDIO_LO = 128266
AUDIO_HI = 128266 + 4096 * 7          # 156938, exclusive
SLICE_LO = END_OF_AUDIO_TOKEN         # 128258 — EOS + every SNAC code, contiguous
SLICE_HI = AUDIO_HI                   # 156938
SLICE_N = SLICE_HI - SLICE_LO         # 28680
EOS_INDEX = END_OF_AUDIO_TOKEN - SLICE_LO   # 0

# The patch replaces a private method whose internals it depends on line for
# line. It is pinned, not feature-detected: a silently different `_step` is
# exactly the failure this module must not have.
REQUIRED_MLX_LM = '0.31.3'

# Generous upper bound on an Orpheus prompt, used by install()'s window check.
# A 500-char chunk frames to ~140 tokens; 512 is >3x that.
PROMPT_TOKEN_BOUND = 512


class FastPathUnsupported(RuntimeError):
    """The fast path cannot serve this model / configuration / processor set.

    Always names the reason. Never caught inside this module and never turned
    into a stock fallback: an Orpheus render that quietly costs 6% more is a
    bug that hides, and a penalty applied over the wrong set is worse.
    """


# ── marked processor factories ───────────────────────────────────────────────
#
# These behave EXACTLY like the stock closures when called, so a model the fast
# path was never installed on still generates correctly with them. The only
# thing they add is a `_bf_fast` marker the batched step reads its parameters
# out of, instead of trying to introspect an opaque closure.

def make_rep_penalty(penalty: float, window: int):
    """`mlx_lm.sample_utils.make_repetition_penalty`, marked for the fast path."""
    from mlx_lm.sample_utils import make_repetition_penalty
    fn = make_repetition_penalty(penalty, window)
    fn._bf_fast = ('rep', float(penalty), int(window))
    return fn


def make_eos_boost(base: float, start: float, expected: float,
                   eos: int = END_OF_AUDIO_TOKEN):
    """The `_boost` closure of `Orpheus._mlx_eos_boost_processor`, marked.

    Kept byte-for-byte identical in behaviour, including the fact that `n` is
    `len(tokens)` over a context that ALREADY contains the prompt (mlx-lm seeds
    the TokenBuffer with the KV-cache tokens, not with an empty list — see the
    note in orpheus.py). This module reproduces today's semantics; it does not
    fix them.
    """
    def _boost(tokens, logits):
        n = len(tokens)
        if n > start:
            bias = base * min(4.0, 1.0 + (n - start) / expected)
            return logits.at[:, eos].add(bias)
        return logits
    _boost._bf_fast = ('boost', float(base), float(start), float(expected), int(eos))
    return _boost


# ── per-batch state ──────────────────────────────────────────────────────────

class _BatchState:
    """Everything the batched step needs that is not in the logits.

    Rebuilt (or permuted) whenever the live uid tuple changes, which is the one
    event that can reorder or drop rows.
    """
    __slots__ = ('uids', 'seen', 'penalty', 'windows', 'pen_active',
                 'base', 'start', 'expected', 'has_boost')


def _row_params(procs, row: int):
    """(penalty, window, base, start, expected) for one row's processor list.

    Raises FastPathUnsupported for anything unmarked, doubled, or ordered so
    that the batched form would not reproduce the stock composition.
    """
    penalty, window = 1.0, None
    base, start, expected = 0.0, float('inf'), 1.0
    seen_rep = seen_boost = False
    for proc in procs:
        mark = getattr(proc, '_bf_fast', None)
        if mark is None:
            raise FastPathUnsupported(
                f'row {row} carries an unmarked logits processor {proc!r}. The MLX '
                'fast path can only reproduce processors built by '
                'orpheus_mlx_fastpath.make_rep_penalty / make_eos_boost; build it '
                'with those or set ORPHEUS_MLX_FASTPATH=0.')
        kind = mark[0]
        if kind == 'rep':
            if seen_rep:
                raise FastPathUnsupported(
                    f'row {row} carries two repetition penalties; the batched form '
                    'applies exactly one.')
            if seen_boost:
                raise FastPathUnsupported(
                    f'row {row} orders the repetition penalty AFTER the EOS boost. '
                    'The batched form applies penalty then boost, which is a '
                    'different function when reversed (the penalty would scale the '
                    'boosted EOS logit).')
            seen_rep = True
            penalty, window = mark[1], mark[2]
        elif kind == 'boost':
            if seen_boost:
                raise FastPathUnsupported(
                    f'row {row} carries two EOS boosts; the batched form applies '
                    'exactly one.')
            if mark[4] != END_OF_AUDIO_TOKEN:
                raise FastPathUnsupported(
                    f'row {row} boosts token {mark[4]}, not END_OF_AUDIO '
                    f'({END_OF_AUDIO_TOKEN}); the batched form biases the EOS column '
                    'of the emittable slice and nothing else.')
            seen_boost = True
            base, start, expected = mark[1], mark[2], mark[3]
        else:
            raise FastPathUnsupported(
                f'row {row} carries a processor marked {mark!r}, which this version '
                'does not know how to batch.')
    return penalty, window, base, start, expected


def _build_seen(token_lists):
    """[B, SLICE_N] bool: True where an emittable id occurs in a row's KV cache.

    `self.tokens[i]` is mlx-lm's own record of the tokens in row i's cache
    (generate.py keeps it that way deliberately), so this is exact — no need to
    track `filter()`/`extend()`.
    """
    seen = np.zeros((len(token_lists), SLICE_N), dtype=bool)
    for i, toks in enumerate(token_lists):
        if not toks:
            continue
        a = np.fromiter(toks, dtype=np.int64, count=len(toks))
        a = a[(a >= SLICE_LO) & (a < SLICE_HI)]
        if a.size:
            seen[i, a - SLICE_LO] = True
    return mx.array(seen)


def _bf_sync(self):
    """Return this GenerationBatch's fast-path state, rebuilding it if stale."""
    uids = tuple(self.uids)
    state = getattr(self, '_bf_state', None)
    if state is not None and state.uids == uids:
        return state

    n_rows = len(uids)
    procs = self.logits_processors
    # Stock skips the whole processor block unless `any(self.logits_processors)`,
    # and `filter()` leaves the list stale when none are set — so mirror the same
    # test rather than trusting the list's shape.
    if not (procs and any(procs)):
        procs = [()] * n_rows
    elif len(procs) != n_rows:
        # Stock indexes self.logits_processors[e] per live row, so a length
        # mismatch here would already be an IndexError there. Never quietly
        # decode a row without the penalty it was inserted with.
        raise FastPathUnsupported(
            f'{len(procs)} logits-processor lists for {n_rows} live rows; the batch '
            'state is inconsistent and the fast path will not guess which row lost '
            'its processors.')

    penalty, windows, base, start, expected = [], [], [], [], []
    for i in range(n_rows):
        p, w, b, s, e = _row_params(procs[i], i)
        penalty.append(p)
        windows.append(w)
        base.append(b)
        start.append(s)
        expected.append(e)

    seen = None
    if state is not None and state.seen is not None:
        # The common case: a row retired, so the new uid set is a subset of the
        # old one in the same relative order. Permuting the mask is exact and
        # costs one gather instead of re-walking every row's history.
        old = {u: i for i, u in enumerate(state.uids)}
        if all(u in old for u in uids) and n_rows:
            seen = state.seen[mx.array([old[u] for u in uids], dtype=mx.int32)]
    if seen is None:
        seen = _build_seen(self.tokens)

    new = _BatchState()
    new.uids = uids
    new.seen = seen
    new.penalty = penalty
    new.windows = windows
    new.pen_active = any(p != 1.0 for p in penalty)
    new.base = base
    new.start = start
    new.expected = expected
    new.has_boost = any(b > 0.0 for b in base)
    self._bf_state = new
    return new


# ── the batched step, in three testable pieces ───────────────────────────────
#
# The forward pass is the only part of `_bf_step` that needs a real model, so
# everything that is not the forward pass lives in these three functions. The
# equivalence test drives exactly this code against the stock per-row loop.

def _bf_mark_inputs(state, inputs, n_rows):
    """Append the current input token to each row's seen mask, in place.

    This mirrors `TokenBuffer.update_and_fetch(inputs)`, which stock calls
    BEFORE the processors run — so the token being conditioned on is already in
    the penalty window. Rows whose input is outside the emittable block (the
    first step's last prompt token, 128257, or any text id) mark nothing, which
    is what a gather over them finds inside the slice: nothing.
    """
    col = inputs.astype(mx.int32) - SLICE_LO
    in_slice = (col >= 0) & (col < SLICE_N)
    col = mx.clip(col, 0, SLICE_N - 1)
    rows = mx.arange(n_rows)
    seen = state.seen
    seen[rows, col] = seen[rows, col] | in_slice
    state.seen = seen
    return seen


def _bf_row_scalars(self, state):
    """Per-row EOS bias, and the exactness check for the penalty window.

    Computed in Python floats over Python ints so the arithmetic is bit-for-bit
    what the stock `_boost` closure does. `n` is `len(self.tokens[i]) + 1`,
    which is exactly `len(TokenBuffer)` at the moment stock's processors run:
    the KV-cache tokens plus the input just appended. That INCLUDES THE PROMPT —
    the semantics the boost has today, not the ones its docstring describes.
    """
    n_rows = len(state.uids)
    bias = [0.0] * n_rows
    any_bias = False
    for i in range(n_rows):
        n = len(self.tokens[i]) + 1
        window = state.windows[i]
        if window is not None and n > window:
            raise FastPathUnsupported(
                f'row {i} has {n} tokens in its KV cache but the repetition window is '
                f'{window}. The seen-mask form is only equal to the stock gather while '
                'the window covers prompt + generation; raise ORPHEUS_MLX_REP_WINDOW '
                '(or lower ORPHEUS_MLX_MAX_TOKENS) rather than render with a penalty '
                'applied over the wrong set.')
        if state.base[i] > 0.0:
            s = state.start[i]
            if n > s:
                bias[i] = state.base[i] * min(4.0, 1.0 + (n - s) / state.expected[i])
                any_bias = True
    return bias, any_bias


def _bf_apply(logits, state, bias, any_bias):
    """Repetition penalty + EOS boost over the whole batch at once.

    The penalty is stock's `logits[:, tokens] = where(l < 0, l*p, l/p)` written
    as a mask: duplicate ids in stock's gather assign the same value, so the set
    of touched columns is all that matters, and the seen mask IS that set
    restricted to the emittable block.
    """
    if state.pen_active:
        pen = mx.array(state.penalty, dtype=logits.dtype)[:, None]
        logits = mx.where(state.seen,
                          mx.where(logits < 0, logits * pen, logits / pen),
                          logits)
    if state.has_boost and any_bias:
        logits = logits.at[:, EOS_INDEX].add(mx.array(bias, dtype=logits.dtype))
    return logits


def _bf_step(self):
    """Batched replacement for `mlx_lm.generate.GenerationBatch._step`.

    Mirrors the stock control flow line for line; only the logits-processor
    block and the head projection differ.
    """
    W = getattr(self.model, '_bf_fastpath_head', None)
    if W is None:
        # The ONE deliberate delegation. `_step` is a CLASS attribute, so
        # patching it patches every GenerationBatch in the process — including
        # ones over a model the fast path was never installed on (the bench
        # harness without --fastpath, or any other mlx-lm user in this
        # interpreter). Those must keep their stock behaviour exactly.
        return type(self)._stock_step(self)

    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens

    # Forward pass, stopping one op short of the stock head: run the decoder
    # stack, then project only onto the emittable block. Mathematically
    # `embed_tokens.as_linear(h)` (x @ weight.T) restricted to those rows.
    h = self.model.model(inputs[:, None], cache=self.prompt_cache)[:, -1, :]
    logits = h @ W.T                                     # [B, SLICE_N]

    n_rows = len(self.uids)
    state = _bf_sync(self)
    _bf_mark_inputs(state, inputs, n_rows)
    bias, any_bias = _bf_row_scalars(self, state)
    logits = _bf_apply(logits, state, bias, any_bias)

    # Normalize the logits
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Sample (stock branching, verbatim)
    if any(self.samplers):
        all_samples = []
        for e in range(n_rows):
            sample_sampler = self.samplers[e] or self.fallback_sampler
            sampled = sample_sampler(logprobs[e : e + 1])
            all_samples.append(sampled)
        sampled = mx.concatenate(all_samples, axis=0)
    else:
        sampled = self.fallback_sampler(logprobs)

    # Back to real token ids. Both samplers return uint32; +SLICE_LO keeps that
    # dtype and stays far inside its range (max id 156937).
    sampled = sampled + SLICE_LO

    self._next_tokens = sampled
    self._next_logprobs = list(logprobs)
    mx.async_eval(self._next_tokens, self._next_logprobs, state.seen)

    mx.eval(inputs, self._current_logprobs)
    inputs = inputs.tolist()
    for sti, ti in zip(self.tokens, inputs):
        sti.append(ti)
    return inputs, self._current_logprobs


# ── installation ─────────────────────────────────────────────────────────────

def install(model, *, rep_window: int, max_tokens: int) -> str:
    """Make `model` decode through the batched step. Returns a log line.

    Raises FastPathUnsupported, by name, for every condition the patch cannot
    honour. The caller decides what to do about it; there is no env switch in
    here (orpheus.py owns ORPHEUS_MLX_FASTPATH).
    """
    import mlx.nn as nn
    import mlx_lm
    from mlx_lm.generate import GenerationBatch

    version = getattr(mlx_lm, '__version__', '<unknown>')
    if version != REQUIRED_MLX_LM:
        raise FastPathUnsupported(
            f'the Orpheus MLX fast path replaces GenerationBatch._step and depends on '
            f'its internals line for line, so it is pinned to mlx-lm '
            f'{REQUIRED_MLX_LM}; this environment has {version}. Re-verify the step '
            'against the new source and move the pin, or set ORPHEUS_MLX_FASTPATH=0.')

    args = getattr(model, 'args', None)
    if args is None or not getattr(args, 'tie_word_embeddings', False):
        raise FastPathUnsupported(
            'the fast path reads the LM head out of the TIED input embedding; this '
            f'model reports tie_word_embeddings='
            f'{getattr(args, "tie_word_embeddings", None)!r}, so its head is a separate '
            'lm_head this patch does not slice.')

    embed = model.model.embed_tokens
    if not isinstance(embed, nn.Embedding):
        raise FastPathUnsupported(
            f'model.model.embed_tokens is a {type(embed).__name__}, not a plain '
            'nn.Embedding. A quantized embedding stores packed weights and group '
            'scales, so slicing its `.weight` would not be the head at all.')
    weight = embed.weight
    if weight.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise FastPathUnsupported(
            f'the tied embedding is {weight.dtype}, not bf16/fp16/fp32; the fast path '
            'does not carry a dequantisation path (quantization is out by ruling).')
    if weight.ndim != 2 or weight.shape[0] < SLICE_HI:
        raise FastPathUnsupported(
            f'the tied embedding is {tuple(weight.shape)}; Orpheus needs at least '
            f'{SLICE_HI} rows to cover EOS + every SNAC code. This is not an Orpheus '
            'checkpoint.')

    if rep_window <= max_tokens + PROMPT_TOKEN_BOUND:
        raise FastPathUnsupported(
            f'repetition window {rep_window} does not cover a full generation plus its '
            f'prompt ({max_tokens} + {PROMPT_TOKEN_BOUND}). The seen-mask form is only '
            'equal to the stock gather when it does. Raise ORPHEUS_MLX_REP_WINDOW, '
            'lower ORPHEUS_MLX_MAX_TOKENS, or set ORPHEUS_MLX_FASTPATH=0.')

    head = weight[SLICE_LO:SLICE_HI]
    mx.eval(head)
    model._bf_fastpath_head = head

    if getattr(GenerationBatch, '_stock_step', None) is None:
        GenerationBatch._stock_step = GenerationBatch._step
        GenerationBatch._step = _bf_step

    mb = head.size * head.dtype.size / 1e6
    return (f'Orpheus MLX fast path installed: head sliced to '
            f'[{SLICE_LO}, {SLICE_HI}) = {SLICE_N} emittable ids '
            f'({mb:.0f} MB of {weight.size * weight.dtype.size / 1e6:.0f} MB read per '
            f'step), repetition penalty + EOS boost batched (mlx-lm {version}, '
            f'rep window {rep_window} > max tokens {max_tokens})')


def uninstall(model=None) -> None:
    """Undo `install` (tests; nothing in production calls it)."""
    from mlx_lm.generate import GenerationBatch
    if getattr(GenerationBatch, '_stock_step', None) is not None:
        GenerationBatch._step = GenerationBatch._stock_step
        del GenerationBatch._stock_step
    if model is not None and hasattr(model, '_bf_fastpath_head'):
        del model._bf_fastpath_head

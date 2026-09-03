#!/usr/bin/env python
"""Separate the two hypotheses for Orpheus MLX decode cost.

The production log cannot distinguish them: live width and KV context are almost
perfectly anti-correlated there (early steps = wide+short, late steps = narrow+long),
so step time reads flat and a fit is unidentifiable.

Here every row in a batch gets the SAME text, so all rows retire together and the
live width is CONSTANT for the whole run. Sweeping width at fixed text length gives
ms/step as a clean function of (width, ctx), which identifies:

    step_ms = FIXED + PER_ROW*width + KV*width*ctx

  H1 (width-bound):  FIXED dominates  -> throughput scales with width; batch
                     scheduling / fill is the lever.
  H2 (KV-bound):     KV dominates     -> throughput is flat in width; only shorter
                     context or cheaper KV helps.

Also measures the batched-logits-processor variant and (optionally) a quantized model.

  python sweep_mlx.py --model <dir> --widths 8,24,48,96 --chars 300

--fastpath drives the SAME run through lib/classes/tts_engines/orpheus_mlx_fastpath
(batched processors + sliced head). Everything else about the run is unchanged, so
`dense_ms_per_step` between the two is the whole measurement.

--outside-slice-mass replaces the sweep with a short probe: it decodes ~N steps of a
real prompt and reports logsumexp(full vocab) - logsumexp(emittable slice) per row,
which is the exact size of the fast path's one non-identity (the softmax normaliser
no longer includes the ids Orpheus cannot emit).
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def build_texts(n, chars):
    base = ("The heretic stood before the assembled council and spoke plainly of what "
            "he had seen beyond the mountain, knowing the price of the telling. ")
    s = (base * ((chars // len(base)) + 2))[:chars]
    return [s] * n


def outside_slice_probe(a, model, mx, BatchGenerator, make_sampler,
                        make_logits_processors, fastpath, EOS):
    """How much probability mass does the emittable slice leave out?

    The fast path normalises over [128258, 156938) instead of the full 156,940-wide
    vocabulary, so top_p/min_p see a slightly different nucleus. The size of that
    difference is logsumexp(full logits) - logsumexp(sliced logits), in nats.

    It is read straight off the STOCK step's own output, with no second forward
    pass and no touching of the generator's cache: mlx-lm hands back
    `Response.logprobs` already normalised over the full vocab, so
    logsumexp(logprobs restricted to the slice) is exactly -(that gap). These are
    the post-penalty, post-boost logprobs the sampler actually sees, which is the
    distribution the non-identity is about.
    """
    import math
    width = int(a.widths.split(",")[0])
    texts = build_texts(width, a.chars)
    ptoks = [model.prepare_input_ids(t, a.voice)[0].tolist() for t in texts]
    bg = BatchGenerator(
        model, max_tokens=a.max_tokens, stop_tokens=[[EOS]],
        sampler=make_sampler(0.6, top_p=0.8),
        logits_processors=make_logits_processors(None, 1.1, a.rep_window),
        completion_batch_size=width, prefill_batch_size=width)
    bg.insert([list(p) for p in ptoks])

    per_step = []
    step = 0
    while step < a.outside_slice_mass:
        responses = bg.next_generated()
        if not responses:
            break
        step += 1
        lp = mx.stack([r.logprobs for r in responses]).astype(mx.float32)
        gap = -mx.logsumexp(lp[:, fastpath.SLICE_LO:fastpath.SLICE_HI], axis=-1)
        mx.eval(gap)
        per_step.append(max(float(x) for x in gap.tolist()))
    bg.close()
    worst = max(per_step) if per_step else float("nan")
    print("RESULT:" + json.dumps(dict(
        tag=a.tag, probe="outside_slice_mass", width=width, chars=a.chars,
        steps=len(per_step),
        max_log_mass_nats=worst,
        max_excluded_prob=1.0 - math.exp(-worst),
        per_step_max=[round(x, 10) for x in per_step])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", default="deathstalker")
    ap.add_argument("--widths", default="8,24,48,96")
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=3700)
    ap.add_argument("--rep-window", type=int, default=4096)
    ap.add_argument("--no-processors", action="store_true",
                    help="drop logits processors entirely: isolates their cost")
    ap.add_argument("--cache-limit-gb", type=float, default=8.0)
    ap.add_argument("--fastpath", action="store_true",
                    help="install the Orpheus MLX batched-decode fast path")
    ap.add_argument("--outside-slice-mass", type=int, default=0, metavar="STEPS",
                    help="probe mode: decode this many steps and report the log-mass "
                         "the emittable slice leaves out (needs --widths with one width)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from lib.classes.tts_engines import orpheus_mlx_fastpath as fastpath

    mx.set_cache_limit(int(a.cache_limit_gb * 1e9))
    t0 = time.time()
    model = load_model(a.model)
    # match production framing: fine-tunes were trained with START_OF_AI, START_OF_SPEECH
    orig = model.prepare_input_ids
    AI_SPEECH = mx.array([[128261, 128257]], dtype=mx.int64)
    def prepare_input_ids(prompt, voice=None, zeroprompt=None, ref_audio=None, ref_text=None, *ar, **kw):
        ids = orig(prompt, voice, zeroprompt, ref_audio, ref_text, *ar, **kw)
        if voice is not None and zeroprompt is None and ref_audio is None and ref_text is None:
            ids = mx.concatenate([ids, AI_SPEECH], axis=1)
        return ids
    model.prepare_input_ids = prepare_input_ids
    print(f"# model loaded in {time.time()-t0:.1f}s: {a.model}", flush=True)

    if a.fastpath:
        print("# " + fastpath.install(model, rep_window=a.rep_window,
                                      max_tokens=a.max_tokens), flush=True)

    EOS = 128258

    if a.outside_slice_mass:
        outside_slice_probe(a, model, mx, BatchGenerator, make_sampler,
                            make_logits_processors, fastpath, EOS)
        return
    for width in [int(w) for w in a.widths.split(",")]:
        texts = build_texts(width, a.chars)
        ptoks = [model.prepare_input_ids(t, a.voice)[0].tolist() for t in texts]
        kw = {}
        if not a.no_processors:
            kw["logits_processors"] = (
                [fastpath.make_rep_penalty(1.1, a.rep_window)] if a.fastpath
                else make_logits_processors(None, 1.1, a.rep_window))
        bg = BatchGenerator(
            model, max_tokens=a.max_tokens, stop_tokens=[[EOS]],
            sampler=make_sampler(0.6, top_p=0.8),
            completion_batch_size=width, prefill_batch_size=width, **kw)
        bg.insert([list(p) for p in ptoks])
        mx.reset_peak_memory()
        step = 0; retired = 0; marks = []
        MARK = 50
        t_gen = time.time()
        while responses := bg.next_generated():
            step += 1
            for r in responses:
                if r.finish_reason is not None:
                    retired += 1
            if step % MARK == 0:
                marks.append((step, retired, time.time() - t_gen))
        wall = time.time() - t_gen
        bg.close()
        # Per-segment ms/step while the batch is still FULL (retired == 0): this is
        # ms/step as a clean function of context at CONSTANT live width.
        seg = []
        prev = (0, 0, 0.0)
        for m in marks:
            if m[1] == 0:
                ms = (m[2] - prev[2]) / (m[0] - prev[0]) * 1000
                seg.append({"ctx": m[0], "ms": round(ms, 1)})
            prev = m
        dense = [m for m in marks if m[1] == 0]
        dense_ms = (dense[-1][2] / dense[-1][0] * 1000) if dense else float("nan")
        peak = mx.get_peak_memory() / 1e9
        out = dict(tag=a.tag, width=width, chars=a.chars, steps=step, wall_s=round(wall, 1),
                   dense_ms_per_step=round(dense_ms, 1),
                   dense_steps=(dense[-1][0] if dense else 0),
                   tok_per_s=round(width * step / wall, 1),
                   rows_per_min=round(width / wall * 60, 2),
                   peak_gb=round(peak, 1), processors=(not a.no_processors),
                   fastpath=a.fastpath, segments=seg)
        print("RESULT:" + json.dumps(out), flush=True)
        del bg
        mx.clear_cache()

if __name__ == "__main__":
    main()

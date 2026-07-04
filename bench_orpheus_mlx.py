#!/usr/bin/env python
"""Orpheus MLX throughput/memory benchmark (Mac Studio tuning pass, Jul 2026).

Replicates lib/classes/tts_engines/orpheus.py::_convert_mlx_batch exactly
(chunk mode), plus a 'continuous' mode that keeps ONE BatchGenerator alive and
streams sentences through it (the hypothesized fix for tail-idle + per-chunk
rebuild). Measures sentences/min, generation vs SNAC-decode split, and MLX
active/cache/peak memory + RSS per chunk.

Usage:
  python bench_orpheus_mlx.py --mode chunk --width 48 --n 96 [--no-clear-cache]
  python bench_orpheus_mlx.py --mode continuous --width 48 --n 96
  python bench_orpheus_mlx.py --mode single --n 12          # per-sentence baseline
Emits one JSON line per run to stdout (prefixed RESULT:) for easy scraping.
"""
import argparse
import json
import os
import re
import resource
import sys
import time

VTT_DEFAULT = "/Volumes/Callisto/Shared/BookForge/projects/The_Mysterious_Stranger_-_Mark_Twain/output/subtitles.vtt"
MODEL = "mlx-community/orpheus-3b-0.1-ft-bf16"
VOICE = "zac"
TEMPERATURE, TOP_P, REP_PENALTY = 0.6, 0.8, 1.1
END_OF_AUDIO_TOKEN = 128258
MAX_TOKENS = 2048  # matches _convert_mlx_batch


def parse_vtt(path, limit):
    sents = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line or re.fullmatch(r"\d+", line):
            continue
        if len(line) > 10:
            sents.append(line)
        if len(sents) >= limit:
            break
    return sents


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mem_snapshot(mx):
    return {
        "active_gb": round(mx.get_active_memory() / 1e9, 2),
        "cache_gb": round(mx.get_cache_memory() / 1e9, 2),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "rss_peak_gb": round(rss_mb() / 1024, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chunk", "continuous", "single"], default="chunk")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--vtt", default=VTT_DEFAULT)
    ap.add_argument("--offset", type=int, default=0, help="skip this many sentences first")
    ap.add_argument("--no-clear-cache", action="store_true",
                    help="chunk mode: skip the per-chunk mx.clear_cache()")
    ap.add_argument("--prefill", type=int, default=0,
                    help="continuous mode: prefill_batch_size (default width//4, min 4)")
    ap.add_argument("--decode-batch", action="store_true",
                    help="decode SNAC per finished row instead of after the full drain")
    ap.add_argument("--cache-limit-gb", type=float, default=0,
                    help="bound the MLX buffer cache via mx.set_cache_limit")
    args = ap.parse_args()

    import numpy as np
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.models.llama.llama import decode_audio_from_codes

    all_sents = parse_vtt(args.vtt, args.offset + args.n)
    sents = all_sents[args.offset:args.offset + args.n]
    if len(sents) < args.n:
        print(f"FATAL: only {len(sents)} sentences available, need {args.n}", file=sys.stderr)
        sys.exit(1)
    avg_chars = sum(len(s) for s in sents) / len(sents)

    if args.cache_limit_gb:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    t0 = time.perf_counter()
    model = load_model(MODEL)
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s; {len(sents)} sentences, avg {avg_chars:.0f} chars", flush=True)

    mx.reset_peak_memory()
    chunk_logs = []
    gen_s = decode_s = 0.0
    total_tokens = 0
    audio_samples = 0
    run_t0 = time.perf_counter()

    def decode_row(ptoks, toks):
        nonlocal decode_s, audio_samples
        d0 = time.perf_counter()
        ids = mx.array([ptoks + toks])
        code_lists = model.parse_output(ids)
        if code_lists and len(code_lists[0]) > 0:
            wav = np.array(decode_audio_from_codes(code_lists[0])[0], dtype=np.float32)
            audio_samples += len(wav)
        decode_s += time.perf_counter() - d0

    if args.mode == "single":
        for i, s in enumerate(sents):
            g0 = time.perf_counter()
            audio = None
            for result in model.generate(s, voice=VOICE, temperature=TEMPERATURE,
                                         top_p=TOP_P, repetition_penalty=REP_PENALTY,
                                         max_tokens=MAX_TOKENS):
                audio = result.audio
            gen_s += time.perf_counter() - g0
            if audio is not None:
                audio_samples += len(np.array(audio))
            if (i + 1) % 4 == 0:
                print(f"  {i+1}/{len(sents)} {json.dumps(mem_snapshot(mx))}", flush=True)

    elif args.mode == "chunk":
        # Exact replica of _convert_mlx_batch: new BatchGenerator per chunk,
        # completion==prefill==chunk len, decode after full drain, optional clear.
        for c0 in range(0, len(sents), args.width):
            chunk = sents[c0:c0 + args.width]
            ptoks_list = [model.prepare_input_ids(s, VOICE)[0].tolist() for s in chunk]
            g0 = time.perf_counter()
            bg = BatchGenerator(
                model, max_tokens=MAX_TOKENS, stop_tokens={END_OF_AUDIO_TOKEN},
                sampler=make_sampler(TEMPERATURE, top_p=TOP_P),
                logits_processors=make_logits_processors(None, REP_PENALTY, 20),
                completion_batch_size=len(chunk), prefill_batch_size=len(chunk),
            )
            uids = bg.insert([list(p) for p in ptoks_list])
            out = {u: [] for u in uids}
            while responses := bg.next():
                for r in responses:
                    if r.finish_reason != "stop":
                        out[r.uid].append(r.token)
            bg.close()
            gen_s += time.perf_counter() - g0
            for ptoks, uid in zip(ptoks_list, uids):
                total_tokens += len(out[uid])
                decode_row(ptoks, out[uid])
            snap = mem_snapshot(mx)  # BEFORE clear: shows intra-chunk cache growth
            if not args.no_clear_cache:
                mx.clear_cache()
            chunk_logs.append(snap)
            done = min(c0 + args.width, len(sents))
            rate = done / (time.perf_counter() - run_t0) * 60
            print(f"  chunk done {done}/{len(sents)} {rate:.1f} sent/min {json.dumps(snap)}", flush=True)

    else:  # continuous
        prefill = args.prefill or max(4, args.width // 4)
        ptoks_list = [model.prepare_input_ids(s, VOICE)[0].tolist() for s in sents]
        g0 = time.perf_counter()
        bg = BatchGenerator(
            model, max_tokens=MAX_TOKENS, stop_tokens={END_OF_AUDIO_TOKEN},
            sampler=make_sampler(TEMPERATURE, top_p=TOP_P),
            logits_processors=make_logits_processors(None, REP_PENALTY, 20),
            completion_batch_size=args.width, prefill_batch_size=prefill,
        )
        uids = bg.insert([list(p) for p in ptoks_list])
        uid2idx = {u: i for i, u in enumerate(uids)}
        out = {u: [] for u in uids}
        finished = 0
        last_log = time.perf_counter()
        while responses := bg.next():
            done_rows = []
            for r in responses:
                if r.finish_reason != "stop":
                    out[r.uid].append(r.token)
                if r.finish_reason is not None:
                    done_rows.append(r.uid)
            if args.decode_batch and done_rows:
                gen_s += time.perf_counter() - g0
                for u in done_rows:
                    total_tokens += len(out[u])
                    decode_row(ptoks_list[uid2idx[u]], out.pop(u))
                g0 = time.perf_counter()
            finished += len(done_rows)
            if time.perf_counter() - last_log > 15:
                snap = mem_snapshot(mx)
                chunk_logs.append(snap)
                rate = finished / (time.perf_counter() - run_t0) * 60
                print(f"  {finished}/{len(sents)} {rate:.1f} sent/min {json.dumps(snap)}", flush=True)
                last_log = time.perf_counter()
        bg.close()
        gen_s += time.perf_counter() - g0
        for u, toks in out.items():
            total_tokens += len(toks)
            decode_row(ptoks_list[uid2idx[u]], toks)

    wall = time.perf_counter() - run_t0
    result = {
        "mode": args.mode, "width": args.width, "n": len(sents),
        "clear_cache": not args.no_clear_cache,
        "prefill": (args.prefill or max(4, args.width // 4)) if args.mode == "continuous" else None,
        "decode_batch": args.decode_batch,
        "wall_s": round(wall, 1),
        "sent_per_min": round(len(sents) / wall * 60, 1),
        "gen_s": round(gen_s, 1), "decode_s": round(decode_s, 1),
        "gen_tokens": total_tokens,
        "tok_per_s": round(total_tokens / gen_s, 1) if gen_s else None,
        "audio_min": round(audio_samples / 24000 / 60, 1),
        "realtime_x": round((audio_samples / 24000) / wall, 2),
        "mem": mem_snapshot(mx),
        "avg_chars": round(avg_chars),
    }
    print("RESULT: " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Validate the modified PRODUCTION Orpheus MLX path end-to-end.

Imports the real Orpheus engine (cache limit at load, no per-chunk flush,
cap-hit re-render ladder) and drives convert_batch() in worker-sized chunks
over many real sentences, logging throughput + MLX memory per chunk. A healthy
run shows flat active memory, cache pinned at the limit, no RSS creep.

Run from the e2a repo root:
  ORPHEUS_MLX_CACHE_LIMIT_GB=8 ORPHEUS_BATCH_SIZE=96 python bench_validate_real.py --n 576
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VTT = "/Volumes/Callisto/Shared/BookForge/projects/The_Mysterious_Stranger_-_Mark_Twain/output/subtitles.vtt"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=576)
    ap.add_argument("--width", type=int, default=int(os.environ.get("ORPHEUS_BATCH_SIZE", "96")))
    args = ap.parse_args()

    import mlx.core as mx
    from lib.classes.tts_engines.orpheus import Orpheus

    sents = parse_vtt(VTT, args.n)
    if len(sents) < args.n:
        print(f"FATAL: only {len(sents)} sentences, need {args.n}", file=sys.stderr)
        sys.exit(1)

    outdir = tempfile.mkdtemp(prefix="orpheus_validate_")
    session = {
        "tts_engine": "orpheus",
        "fine_tuned": "zac",
        "sentences_dir": outdir,
        "device": "mlx",
    }
    engine = Orpheus(session)
    print(f"backend={engine.backend} voice={engine.voice} outdir={outdir}", flush=True)
    assert engine.backend == "mlx", f"expected mlx backend, got {engine.backend}"

    def rss_gb():
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)

    t0 = time.perf_counter()
    ok_total = 0
    for c0 in range(0, len(sents), args.width):
        items = [(c0 + j, s) for j, s in enumerate(sents[c0:c0 + args.width])]
        results = engine.convert_batch(items)
        ok_total += sum(bool(r) for r in results)
        done = c0 + len(items)
        el = time.perf_counter() - t0
        print(f"chunk@{done}/{len(sents)} ok={ok_total} {done/el*60:.1f} sent/min "
              f"active={mx.get_active_memory()/1e9:.2f}GB cache={mx.get_cache_memory()/1e9:.2f}GB "
              f"peak={mx.get_peak_memory()/1e9:.2f}GB rss={rss_gb():.2f}GB", flush=True)

    wall = time.perf_counter() - t0
    wavs = len([f for f in os.listdir(outdir) if f.endswith((".wav", ".flac"))])
    print("RESULT: " + json.dumps({
        "n": len(sents), "width": args.width, "ok": ok_total, "files": wavs,
        "wall_s": round(wall, 1), "sent_per_min": round(len(sents) / wall * 60, 1),
        "active_gb": round(mx.get_active_memory() / 1e9, 2),
        "cache_gb": round(mx.get_cache_memory() / 1e9, 2),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "rss_gb": round(rss_gb(), 2),
    }), flush=True)


if __name__ == "__main__":
    main()

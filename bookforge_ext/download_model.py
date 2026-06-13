"""
Download a TTS model/voice into e2a's HuggingFace cache — on demand.

BookForge ships a barebones core (one bundled voice). Every other XTTS voice is
fetched on demand by spawning this helper in the bundled python env. It resolves
a voice the SAME way the XTTS engine does (lib/.../presets + lib.conf's HF cache
dir), so a downloaded voice lands byte-identical to where xtts.py's load_engine()
will look — no special-casing at inference time.

Usage:
    python -m bookforge_ext.download_model --engine xtts --preset ScarlettJohansson
        [--cache-dir DIR]      # override the HF cache (used by the seed builder)
        [--bf-progress]        # emit BF_PROGRESS lines on stdout for the UI
        [--repo R --sub S/ --files config.json model.pth vocab.json]   # bypass presets

Output contract:
    - Progress: with --bf-progress, lines "BF_PROGRESS <received> <total> <desc>"
      on stdout (huggingface_hub's own tqdm still goes to stderr).
    - Final line on stdout: one JSON object —
        {"ok": true,  "snapshotDir": "...", "subDir": "...", "files": {name: path}}
        {"ok": false, "error": "..."}
      Exit code is 0 on success, non-zero on failure.
"""

import argparse
import json
import os
import sys


def _resolve_preset(engine, preset):
    """Look up (repo, sub, files) from e2a's own preset tables (single source)."""
    from lib.classes.tts_engines.common.preset_loader import load_engine_presets
    presets = load_engine_presets(engine)
    entry = presets.get(preset)
    if not entry:
        raise KeyError(f"Unknown {engine} preset: {preset!r}")
    return entry["repo"], entry["sub"], list(entry["files"])


def _make_bf_tqdm():
    """A tqdm subclass that mirrors byte progress to stdout as BF_PROGRESS lines."""
    from tqdm.auto import tqdm as _tqdm

    class BFTqdm(_tqdm):
        def update(self, n=1):
            ret = super().update(n)
            try:
                total = self.total or 0
                desc = (self.desc or "").strip().rstrip(":")
                sys.stdout.write(f"BF_PROGRESS {self.n} {total} {desc}\n")
                sys.stdout.flush()
            except Exception:
                pass
            return ret

    return BFTqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="xtts")
    ap.add_argument("--preset")
    ap.add_argument("--repo")
    ap.add_argument("--sub", default="")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--cache-dir", dest="cache_dir")
    ap.add_argument("--bf-progress", action="store_true")
    args = ap.parse_args()

    # bookforge_ext sits at the e2a root, so root = parent of this package dir.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    # Importing lib.conf sets HUGGINGFACE_HUB_CACHE/HF_HOME = <root>/models/tts —
    # the exact dir xtts.py downloads into. Fall back to that path if conf can't
    # be imported for some reason.
    try:
        from lib.conf import tts_dir
        default_cache = tts_dir
    except Exception:
        default_cache = os.path.join(root, "models", "tts")

    repo, sub, files = args.repo, args.sub, args.files
    if not repo:
        try:
            repo, sub, files = _resolve_preset(args.engine, args.preset)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            return 2

    # Download every repo file except ref.wav (a local reference clip, not in the
    # HF repo). Fine-tuned voices have 3 (config/model/vocab); the base model adds
    # speakers_xtts.pth.
    dl_files = [f for f in (files or []) if f != "ref.wav"]
    if not dl_files:
        print(json.dumps({"ok": False, "error": "no files to download"}))
        return 2

    cache_dir = args.cache_dir or default_cache
    os.makedirs(cache_dir, exist_ok=True)
    patterns = [f"{sub}{f}" for f in dl_files]

    tqdm_class = _make_bf_tqdm() if args.bf_progress else None

    try:
        from huggingface_hub import snapshot_download

        kwargs = dict(repo_id=repo, allow_patterns=patterns, cache_dir=cache_dir)
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        try:
            snap = snapshot_download(**kwargs)
        except TypeError:
            # Older huggingface_hub without tqdm_class — retry with default bars.
            kwargs.pop("tqdm_class", None)
            snap = snapshot_download(**kwargs)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1

    sub_parts = [p for p in sub.strip("/").split("/") if p]
    sub_dir = os.path.join(snap, *sub_parts) if sub_parts else snap
    resolved = {f: os.path.join(sub_dir, f) for f in dl_files}

    # Sanity: the headline checkpoint must actually be on disk.
    model_file = next((f for f in dl_files if f.endswith(".pth")), dl_files[-1])
    if not os.path.exists(resolved[model_file]):
        print(json.dumps({"ok": False, "error": f"download finished but {model_file} is missing"}))
        return 1

    print(json.dumps({"ok": True, "snapshotDir": snap, "subDir": sub_dir, "files": resolved}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

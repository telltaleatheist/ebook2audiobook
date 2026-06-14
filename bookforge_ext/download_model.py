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
import hashlib
import json
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
import zipfile


# BookForge download mirror — a FALLBACK for when the upstream home (HuggingFace
# for voices, Stanford/HF for Stanza) is unreachable. Everything here also has an
# upstream home; the mirror only kicks in after upstream fails. Override for
# testing with BOOKFORGE_MIRROR_BASE.
MIRROR_BASE = os.environ.get("BOOKFORGE_MIRROR_BASE", "https://owenmorgan.com/bookforge").rstrip("/")

# The two XTTS repos the mirror hosts, mapped to their mirror sub-paths. Keyed on
# the exact repo ids from electron/xtts-voices.ts (BASE_REPO / FINE_TUNED_REPO).
_BASE_REPO = "coqui/XTTS-v2"
_FINE_TUNED_REPO = "drewThomasson/fineTunedTTSModels"


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


def _dir_bytes(d):
    """Sum the byte size of every file under d (best-effort, never raises)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(d):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _http_download(url, dest, bf_progress=False, desc=""):
    """Stream a URL to `dest`, emitting BF_PROGRESS lines. Returns True on a 200
    that fully downloads, False on 404/HTTP error (so callers can treat a missing
    mirror file as "not available"). Raises only on unexpected I/O errors."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "BookForge/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            received = 0
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    received += len(chunk)
                    if bf_progress:
                        sys.stdout.write(f"BF_PROGRESS {received} {total} {desc}\n")
                        sys.stdout.flush()
            os.replace(tmp, dest)
            return True
    except urllib.error.HTTPError as e:
        # 404 (file not mirrored) / other HTTP error → not available.
        try:
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
        except OSError:
            pass
        if e.code == 404:
            return False
        raise


def _mirror_prefix(repo, sub):
    """The mirror URL prefix (ending in '/') that holds a repo's files, or None
    when this repo isn't mirrored. The base model lives under /xtts-v2/; each
    fine-tuned voice under /voices/<id>/ where <id> is the last path component of
    its sub (e.g. xtts-v2/eng/ScarlettJohansson/ → ScarlettJohansson)."""
    if repo == _BASE_REPO:
        return f"{MIRROR_BASE}/xtts-v2/"
    if repo == _FINE_TUNED_REPO:
        name = [p for p in sub.strip("/").split("/") if p]
        if not name:
            return None
        return f"{MIRROR_BASE}/voices/{name[-1]}/"
    return None


def _hf_revision(repo):
    """The real `main` commit sha for `repo`, or a deterministic fake when the
    hub is unreachable. Using the real sha makes the fabricated cache entry
    resolve transparently online too (hf_hub_download returns the snapshot file
    without re-downloading); the fake-sha entry only resolves while offline."""
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(repo).sha
        if sha:
            return sha
    except Exception:
        pass
    return hashlib.sha1(repo.encode("utf-8")).hexdigest()


def _download_xtts_from_mirror(repo, sub, dl_files, cache_dir, bf_progress):
    """Fetch an XTTS voice/base from the BookForge mirror into a VALID HF-cache
    layout (refs/main + snapshots/<rev>/<sub>/<file>), so the engine's
    hf_hub_download() resolves it with no special-casing. Returns the snapshot
    dir on success, or None when the repo isn't mirrored / a file is missing."""
    prefix = _mirror_prefix(repo, sub)
    if prefix is None:
        return None

    rev = _hf_revision(repo)
    repo_dir = os.path.join(cache_dir, "models--" + repo.replace("/", "--"))
    snap_dir = os.path.join(repo_dir, "snapshots", rev)
    sub_parts = [p for p in sub.strip("/").split("/") if p]
    target_sub = os.path.join(snap_dir, *sub_parts) if sub_parts else snap_dir

    try:
        for f in dl_files:
            ok = _http_download(prefix + f, os.path.join(target_sub, f), bf_progress, desc=f)
            if not ok:
                # A required file isn't on the mirror — abandon the fallback.
                shutil.rmtree(snap_dir, ignore_errors=True)
                return None
    except Exception:
        shutil.rmtree(snap_dir, ignore_errors=True)
        return None

    # refs/main lets hf_hub_download resolve the snapshot by its default revision.
    refs_dir = os.path.join(repo_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, "main"), "w", encoding="utf-8") as fh:
        fh.write(rev)

    return snap_dir


def _download_stanza_from_mirror(lang, stanza_dir, bf_progress):
    """Fetch a Stanza language pack's default.zip from the mirror and extract it
    into <stanza_dir>/<lang>/. The full resources.json catalog already ships with
    the bundled core languages, so Pipeline(REUSE_RESOURCES) finds the new lang
    once its model files are present. Returns the lang dir, or None if unavailable."""
    url = f"{MIRROR_BASE}/stanza/{lang}/default.zip"
    tmp_zip = os.path.join(stanza_dir, f".{lang}.default.zip")
    try:
        ok = _http_download(url, tmp_zip, bf_progress, desc=lang)
        if not ok:
            return None
        lang_dir = os.path.join(stanza_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(lang_dir)
        return lang_dir
    except Exception:
        return None
    finally:
        try:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)
        except OSError:
            pass


def _download_stanza(args, root):
    """Fetch a Stanza language pack into <root>/models/stanza/<lang>/."""
    stanza_dir = os.path.join(root, "models", "stanza")
    os.makedirs(stanza_dir, exist_ok=True)
    lang_dir = os.path.join(stanza_dir, args.lang)

    # Poll the language dir's on-disk size and mirror it as BF_PROGRESS lines.
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            try:
                recv = _dir_bytes(lang_dir)
                sys.stdout.write(f"BF_PROGRESS {recv} {args.total} {args.lang}\n")
                sys.stdout.flush()
            except Exception:
                pass
            stop.wait(0.5)

    poller = None
    if args.bf_progress:
        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()

    try:
        import stanza
        stanza.download(args.lang, model_dir=stanza_dir)
    except Exception as e:
        # Upstream (Stanford/HF) failed — fall back to the BookForge mirror's
        # default.zip. The poller keeps reporting on-disk progress meanwhile.
        mirror = _download_stanza_from_mirror(args.lang, stanza_dir, args.bf_progress)
        if mirror is None:
            stop.set()
            if poller is not None:
                poller.join(timeout=1)
            print(json.dumps({"ok": False, "error": f"Stanza download failed ({e}); no mirror fallback available"}))
            return 1

    stop.set()
    if poller is not None:
        poller.join(timeout=1)

    if not os.path.isdir(lang_dir) or not os.listdir(lang_dir):
        print(json.dumps({"ok": False, "error": f"download finished but {lang_dir} is missing or empty"}))
        return 1

    print(json.dumps({"ok": True, "dir": lang_dir, "lang": args.lang}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="xtts")
    ap.add_argument("--preset")
    ap.add_argument("--repo")
    ap.add_argument("--sub", default="")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--cache-dir", dest="cache_dir")
    ap.add_argument("--lang")
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--bf-progress", action="store_true")
    args = ap.parse_args()

    # bookforge_ext sits at the e2a root, so root = parent of this package dir.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    # Stanza language packs are segmentation models, not HF voices — different
    # mechanism, same output contract (BF_PROGRESS + a single final JSON line).
    if args.engine == "stanza":
        return _download_stanza(args, root)

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
        # Upstream HuggingFace failed — fall back to the BookForge mirror, which
        # writes the same HF-cache layout so the engine resolves it identically.
        snap = _download_xtts_from_mirror(repo, sub, dl_files, cache_dir, args.bf_progress)
        if snap is None:
            print(json.dumps({"ok": False, "error": f"HuggingFace download failed ({e}); no mirror fallback available"}))
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

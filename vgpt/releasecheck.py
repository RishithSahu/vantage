"""Is this the release it says it is?  Reads the manifest that travels with the application (`RELEASE.json`, written by tools_release_manifest.py) and compares it with the files that are really there.

`info(root)` is cheap (the manifest and what the running process resolved), `verify(root, manifest, models=False)` hashes the source and vocabulary files (a second or two) and, with models=True, the model files too (hundreds of megabytes); source=False leaves the source out (an installed bundle).
The doctor runs the cheap check at every start; `tools_release_manifest.py verify` runs everything."""
import hashlib
import json
import os

SOURCE_DIRS = ("vgpt", "ui")
SOURCE_FILES = ("vehiclegpt.py", "app.py")
SKIP = {"__pycache__", ".pytest_cache"}
SKIP_FILES = {"research.py", "research.html", "review.html"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def source_files(root):
    out = []
    for d in SOURCE_DIRS:
        for base, dirs, files in os.walk(os.path.join(root, d)):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            for f in sorted(files):
                if f.endswith((".pyc", ".pyo")) or f in SKIP_FILES: continue
                out.append(os.path.relpath(os.path.join(base, f), root).replace(os.sep, "/"))
    for f in SOURCE_FILES:
        if os.path.exists(os.path.join(root, f)): out.append(f)
    return sorted(out)


def load(root):
    try: return json.load(open(os.path.join(root, "RELEASE.json"), encoding="utf-8"))
    except (OSError, ValueError): return None


def verify(root, manifest, models=False, source=True):
    """-> a list of differences between the tree at `root` and its manifest (empty = the same release)."""
    bad = []; want = manifest.get("source", {}).get("files", {})
    if want and source:                                                                                        # a frozen bundle keeps its code inside an archive: there the source files cannot be listed, and the models and the vocabulary are what is checked
        cur = {rel: sha256(os.path.join(root, rel)) for rel in source_files(root)}
        for rel in sorted(set(cur) | set(want)):
            if rel not in cur: bad.append(f"missing source file {rel}")
            elif rel not in want: bad.append(f"source file not in the release: {rel}")
            elif cur[rel] != want[rel]: bad.append(f"source file changed: {rel}")
    for rel, e in (manifest.get("vocabulary") or {}).get("files", {}).items():
        if e and (not os.path.exists(os.path.join(root, rel)) or sha256(os.path.join(root, rel)) != e["sha256"]): bad.append(f"vocabulary file differs: {rel}")
    for rel, e in (manifest.get("models") or {}).items():
        if not isinstance(e, dict) or "sha256" not in e: continue
        p = os.path.join(root, rel)
        if not os.path.exists(p): bad.append(f"missing model file {rel}")
        elif models and sha256(p) != e["sha256"]: bad.append(f"model file changed: {rel}")
        elif not models and os.path.getsize(p) != e.get("bytes", os.path.getsize(p)): bad.append(f"model file has another size: {rel}")
    return bad


def info(root, resolved=None):
    """What a status page shows about the release: -> {"present": False} or the manifest's headline and whether the model files the process chose are the ones the manifest names."""
    m = load(root)
    if m is None: return {"present": False}
    d = {"present": True, "version": m.get("version"), "made": m.get("made"), "source_digest": (m.get("source") or {}).get("digest"), "vcs": {k: (m.get("vcs") or {}).get(k) for k in ("kind", "commit", "tag", "dirty", "releasable_from_tag")},
         "installer": m.get("installer"), "models_match": None}
    if resolved and resolved.get("adapter_sha256"):
        want = (m.get("models") or {}).get("models/" + str(resolved.get("adapter")))
        d["models_match"] = bool(want and want.get("sha256") == resolved["adapter_sha256"])
    return d

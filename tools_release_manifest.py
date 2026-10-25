"""The provenance of a release, written down so that it can be checked: which source, which models, which vocabulary, which test data, which backend, which platform, which installer.

    python tools_release_manifest.py make     [--root DIR] [--out FILE] [--installer FILE]     write the manifest of the tree at DIR (default: this project; for a bundle, its `_internal` folder)
    python tools_release_manifest.py verify   [--root DIR] [--manifest FILE]                   check a tree (an installed copy, a bundle) against its manifest; exit code 1 on any difference
    python tools_release_manifest.py show     [--manifest FILE]

The manifest holds
    source        SHA-256 of every runtime source file (vgpt/**, ui/**, vehiclegpt.py, app.py) and one digest over all of them: two copies of the project are the same program exactly when this digest is the same
    models        the manifest models/product.json, the adapter, the ONNX Runtime build, the base model and the confidence calibration, each with its SHA-256 and size
    vocabulary    the ontology, the glossary and the signal registry's data files, each with its SHA-256, and the frame version
    test_data     the test-data manifest, the corpus manifest and the driving fixture, with their SHA-256
    runtime       Python, ONNX Runtime and llama.cpp versions, platform and machine
    vcs           the git commit and tag when the tree is a git worktree, otherwise {"kind": "none"} - and `releasable_from_tag` says whether the release can be rebuilt from a named commit
    installer     file name, size and SHA-256 (added after the installer exists: `make --installer`)

A copy of the manifest travels inside the installed application (`RELEASE.json` beside `models/`), `/api/health` reports it, and `verify` is how a device (or a person) proves it runs the release it claims to."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
VOCAB = ("vgpt/meaning/ontology_data.json", "vgpt/meaning/glossary.jsonl", "vgpt/meaning/calibration.json")


from vgpt.releasecheck import sha256, source_files, verify  # noqa: E402  (the same code the doctor runs)


def entry(root, rel):
    p = os.path.join(root, rel)
    return {"sha256": sha256(p), "bytes": os.path.getsize(p)} if os.path.isfile(p) else None


def vcs(root):
    git = os.path.join(root, ".git")
    if not os.path.exists(git): return {"kind": "none", "releasable_from_tag": False, "note": "this tree is not a git worktree: the release cannot be rebuilt from a named commit"}

    def run(*a):
        try: return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception: return ""
    commit = run("rev-parse", "HEAD"); tag = run("describe", "--exact-match", "--tags", "HEAD"); dirty = bool(run("status", "--porcelain"))
    return {"kind": "git", "commit": commit, "tag": tag or None, "dirty": dirty, "releasable_from_tag": bool(commit and tag and not dirty)}


def runtime():
    d = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "system": platform.system()}
    for mod, key in (("onnxruntime", "onnxruntime"), ("llama_cpp", "llama_cpp"), ("numpy", "numpy")):
        try: d[key] = __import__(mod).__version__
        except Exception: d[key] = None
    return d


def make(root, installer=None, version=None, source_root=None):
    """`root`: where the models, the vocabulary and the test data are (the project, or the `_internal` folder of a bundle); `source_root`: where the source is (always the project: a bundle holds its code in an archive)."""
    source_root = source_root or root; sys.path.insert(0, source_root)
    files = {rel: sha256(os.path.join(source_root, rel)) for rel in source_files(source_root)}
    digest = hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()).hexdigest()
    models = {}; cfg = os.path.join(root, "models", "product.json"); models["models/product.json"] = entry(root, "models/product.json")
    if models["models/product.json"]:
        j = json.load(open(cfg, encoding="utf-8")); models["selected"] = j
        for rel in ("models/" + str(j.get("adapter") or "-"), "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"): models[rel] = entry(root, rel)
        if j.get("ort"):
            for f in ("model.onnx", "model.onnx.data", "draft.json"): models[f"models/{j['ort']}/{f}"] = entry(root, f"models/{j['ort']}/{f}")
    onto = {}
    try:
        from vgpt.meaning import ontology as ON
        onto["frame_version"] = 2 if (models.get("selected") or {}).get("frame", 1) == 2 else 1; onto["ontology_version"] = int(ON.load(onto["frame_version"]).version)
    except Exception as e: onto["error"] = type(e).__name__
    vocab = {rel: entry(root, rel) for rel in VOCAB}
    td = {rel: entry(root, rel) for rel in ("tests/testdata_manifest.json", "tests/corpus/MANIFEST.json", "tests/fixtures/driving_segments.npz")}
    try: from vgpt.version import VERSION as V
    except Exception: V = version
    m = {"name": "Vantage", "version": version or V, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "source": {"digest": digest, "files": files}, "models": models, "vocabulary": dict(onto, files=vocab), "test_data": td, "runtime": runtime(), "vcs": vcs(source_root)}
    if installer: m["installer"] = {"file": os.path.basename(installer), "bytes": os.path.getsize(installer), "sha256": sha256(installer)}
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    for n in ("make", "verify", "show"):
        p = sub.add_parser(n); p.add_argument("--root", default=HERE); p.add_argument("--out", default=None); p.add_argument("--manifest", default=None); p.add_argument("--installer", default=None); p.add_argument("--version", default=None); p.add_argument("--source-root", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        m = make(os.path.abspath(a.root), a.installer, a.version, os.path.abspath(a.source_root) if a.source_root else None); out = a.out or os.path.join(a.root, "RELEASE.json"); os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(m, open(out, "w", encoding="utf-8"), indent=1); print(f"wrote {out}: source {m['source']['digest'][:16]}, {len(m['source']['files'])} files, vcs {m['vcs']['kind']}" + ("" if m["vcs"]["releasable_from_tag"] else " (NOT releasable from a tag)")); return 0
    mp = a.manifest or os.path.join(a.root, "RELEASE.json")
    if a.cmd == "show": print(json.dumps({k: v for k, v in json.load(open(mp, encoding="utf-8")).items() if k != "source"}, indent=1)); return 0
    if a.cmd == "verify":
        bad = verify(os.path.abspath(a.root), json.load(open(mp, encoding="utf-8")), models=True, source=not a.source_root)
        print("the tree matches its release manifest" if not bad else "DIFFERENCES:\n  " + "\n  ".join(bad[:40])); return 1 if bad else 0
    ap.print_help(); return 2


if __name__ == "__main__": sys.exit(main())

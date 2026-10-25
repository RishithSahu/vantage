"""Reproducibility manifest: exact identity of everything a result depends on.
Source hashes (vgpt/, bench/, training/ python files), model and adapter hashes, database hashes, question-set hashes, runtime versions, and the configuration a run was given.
The acceptance set is never hashed here: its hash is recorded once, at freeze time (bench/freeze_acceptance.py).
    python bench/manifest.py            prints the manifest of the working tree
Large-file hashes are cached in results/_cache/hashes.json keyed by (path, size, mtime)."""
import hashlib, json, os, platform, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "results", "_cache", "hashes.json")
SETS = ["bench/indep_dev.tsv", "bench/indep_final.tsv", "bench/diag_v1.tsv"]                       # acceptance_v1.tsv deliberately absent
MODELS = ["models/LFM2.5-350M-Q4_K_M.gguf", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "models/ir-lora-qwen25-05b.gguf", "models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf", "models/bge-small-en-v1.5/model_int8.onnx", "models/bge-small-en-v1.5/model.onnx"]


def _file_hash(path):
    st = os.stat(path); key = f"{os.path.relpath(path, HERE)}|{st.st_size}|{int(st.st_mtime)}"
    try: cache = json.load(open(CACHE))
    except Exception: cache = {}
    if key in cache: return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    cache[key] = h.hexdigest()[:16]
    try: os.makedirs(os.path.dirname(CACHE), exist_ok=True); json.dump(cache, open(CACHE, "w"))
    except OSError: pass
    return cache[key]


def source_hash():
    h = hashlib.sha256(); files = []
    for sub in ("vgpt", "bench", "training"):
        for dp, dn, fn in os.walk(os.path.join(HERE, sub)):
            dn[:] = sorted(d for d in dn if d not in ("__pycache__", "_cache", "data"))
            for f in sorted(fn):
                if f.endswith(".py") and not f.startswith("_make_"): files.append(os.path.join(dp, f))
    for p in sorted(files):
        h.update(os.path.relpath(p, HERE).encode()); h.update(open(p, "rb").read())
    return h.hexdigest()[:16], len(files)


def manifest(config=None, with_models=True):
    sh, n = source_hash(); m = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "source_hash": sh, "source_files": n, "python": platform.python_version(), "platform": platform.platform(), "config": config or {}}
    try:
        import llama_cpp; m["llama_cpp_python"] = llama_cpp.__version__
    except Exception: m["llama_cpp_python"] = None
    try:
        import numpy; m["numpy"] = numpy.__version__
    except Exception: pass
    m["sqlite"] = __import__("sqlite3").sqlite_version
    m["question_sets"] = {p: _file_hash(os.path.join(HERE, p)) for p in SETS if os.path.exists(os.path.join(HERE, p))}
    m["databases"] = {os.path.basename(p): _file_hash(p) for p in sorted(__import__("glob").glob(os.path.join(HERE, "data", "gen_*.db")))}
    if with_models: m["models"] = {p: _file_hash(os.path.join(HERE, p)) for p in MODELS if os.path.exists(os.path.join(HERE, p))}
    return m


if __name__ == "__main__": print(json.dumps(manifest(), indent=1))

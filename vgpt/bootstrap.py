"""One start-up for every way the assistant is run.

The launcher (`vehiclegpt.py run`, `smoke`), the interface module (`python ui/server.py`) and the terminal chat (`app.py`) all begin here, so the process that answers questions is the same process whichever door it was started from:

    1. hardening          no core dumps, not dumpable, private files - before any vehicle data is read
    2. the product model  `models/product.json` names the adapter, the frame version and the ONNX Runtime build; an explicit environment setting of the person still wins
    3. the memory guard   the case study's budget (about 2 GB for everything) as a rule the process keeps
    4. the doctor         the install is checked; a failed check stops the start with the plain report

`resolved()` says which model was chosen (adapter file and SHA-256, frame version, backend), and `describe_model(model)` says what a loaded model really is; `/api/health` shows both, so a status page can prove which model answers.

Importing this module changes nothing in the environment of the importing process (a test or a tool); only `adopt_product_model()` and `startup()` do."""
import hashlib
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = getattr(sys, "_MEIPASS", os.path.dirname(HERE))

_RESOLVED = {"source": "defaults", "adapter": None, "adapter_sha256": None, "frame": None, "ort_dir": None, "ort_sha256": None, "vehicle_hint": False, "product_json": None}
_HASH_CACHE = {}
_LOCK = threading.Lock()
_HASHING = []
_DONE = []                                                                                                         # the start-up ran in this process: a second door (the launcher, then the interface module) does not repeat it


def _persist_path():
    try:
        from . import paths
        return os.path.join(paths.home(), "hashes.json")
    except Exception: return None


def file_sha256(path, chunk=1 << 20, throttle=0.0):
    """SHA-256 of a file, computed once per (path, size, modification time): in this process and, in the application's private folder, across starts (a 600 MB model is not read again at every start of a small board).
    `throttle` sleeps that many seconds after every chunk, so that hashing in the background does not take the storage from the assistant."""
    try: st = os.stat(path)
    except OSError: return None
    key = (os.path.abspath(path), st.st_size, int(st.st_mtime))
    with _LOCK:
        if key in _HASH_CACHE: return _HASH_CACHE[key]
    pp = _persist_path(); disk = {}
    if pp:
        try: disk = json.load(open(pp, encoding="utf-8"))
        except (OSError, ValueError): disk = {}
        hit = disk.get(key[0])
        if isinstance(hit, list) and hit[:2] == [key[1], key[2]] and len(hit) == 3:
            with _LOCK: _HASH_CACHE[key] = hit[2]
            return hit[2]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
                if throttle: time.sleep(throttle)
    except OSError: return None
    d = h.hexdigest()
    with _LOCK: _HASH_CACHE[key] = d
    if pp:
        try:
            disk[key[0]] = [key[1], key[2], d]; tmp = pp + ".tmp"; json.dump(disk, open(tmp, "w", encoding="utf-8")); os.replace(tmp, pp)
        except OSError: pass
    return d


def ort_sha256():
    """The hash of the ONNX build, or None while it is still being computed (a background thread, throttled, started by the first call): reading a 600 MB file is not done at start-up."""
    if _RESOLVED.get("ort_sha256"): return _RESOLVED["ort_sha256"]
    od = os.environ.get("VGPT_ORT_DIR")
    if not od or not os.path.exists(os.path.join(od, "model.onnx")): return None
    if not _HASHING:
        _HASHING.append(1); threading.Thread(target=lambda: _RESOLVED.__setitem__("ort_sha256", file_sha256(os.path.join(od, "model.onnx"), throttle=0.002)), daemon=True, name="hash-ort").start()
    return _RESOLVED.get("ort_sha256")


def adopt_product_model(root=None):
    """The language model the product uses is whatever `models/product.json` says ({"adapter": ..., "frame": 2, "ort": ..., "vehicle_hint": ...}).  Without the file, or with VGPT_FRAME already set, nothing changes in the
    environment.  -> the description of what was chosen (also kept for `resolved()`)."""
    root = root or ROOT; r = dict(_RESOLVED, source="defaults")
    cfg = os.path.join(root, "models", "product.json")
    try:
        if os.environ.get("VGPT_FRAME") is not None: r["source"] = "environment"
        elif os.path.exists(cfg):
            j = json.load(open(cfg, encoding="utf-8")); r["product_json"] = file_sha256(cfg)
            ad = os.path.join(root, "models", str(j.get("adapter") or ""))
            if j.get("adapter") and os.path.exists(ad):
                os.environ["VGPT_FRAME"] = str(int(j.get("frame", 1))); os.environ.setdefault("VGPT_MEANING_ADAPTER", ad); r["source"] = "models/product.json"
            od = os.path.join(root, "models", str(j.get("ort") or ""))                                              # the same model as an int8 ONNX Runtime build (tools/build_ort_model.py)
            if j.get("ort") and os.path.exists(os.path.join(od, "model.onnx")): os.environ.setdefault("VGPT_ORT_DIR", od)
            if j.get("vehicle_hint"): os.environ.setdefault("VGPT_VEHICLE_HINT", "1")                              # the adapter was trained with the car type in its system prompt
    except (OSError, ValueError, TypeError): pass
    ad = os.environ.get("VGPT_MEANING_ADAPTER"); od = os.environ.get("VGPT_ORT_DIR")
    r.update(adapter=os.path.basename(ad) if ad else None, adapter_sha256=file_sha256(ad) if ad and os.path.exists(ad) else None, frame=int(os.environ["VGPT_FRAME"]) if os.environ.get("VGPT_FRAME", "").isdigit() else None,
             ort_dir=os.path.basename(od) if od else None, vehicle_hint=os.environ.get("VGPT_VEHICLE_HINT") == "1", ort_sha256=None)
    _RESOLVED.clear(); _RESOLVED.update(r)
    return dict(_RESOLVED)


def resolved():
    """What `adopt_product_model` chose (a copy)."""
    return dict(_RESOLVED)


def describe_model(model):
    """What a loaded frame model really is: the backend that runs it, the frame version, the vocabulary version, the decoder and the thread count.  None-safe."""
    if model is None: return {"loaded": False}
    kind = type(model).__name__
    d = {"loaded": True, "class": kind, "backend": "onnxruntime-int8" if kind == "OrtFrameModel" else "llama.cpp" if hasattr(model, "llm") else kind, "frame": getattr(model, "version", None),
         "decoder": getattr(model, "decoder", None), "provider": getattr(model, "provider", None), "vehicle_hint": bool(getattr(model, "vehicle_hint", False)), "vehicle": getattr(model, "vehicle", None)}
    try: d["ontology"] = int(model.onto.version)
    except Exception: pass
    return d


def rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"): return int(line.split()[1]) / 1024.0
    except OSError: pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        from . import memuse
        return memuse.peak_rss_mb()                                                                           # neither /proc nor psutil (QNX): the peak the C library keeps, 0.0 when even that is missing


def memory_guard(limit_mb, rss=None):
    """The case study's budget as a rule the process keeps, not a number it reports: past the limit the process ends itself (the service manager starts it again, the stored history stays on disk)."""
    rss = rss or rss_mb

    def watch():
        peak = 0.0
        while True:
            time.sleep(3); r = rss(); peak = max(peak, r)
            if r > limit_mb:
                print(f"MEMORY LIMIT: {r:.0f} MB resident is more than the {limit_mb:.0f} MB budget (peak {peak:.0f} MB); restarting", file=sys.stderr, flush=True); os._exit(75)
    threading.Thread(target=watch, daemon=True, name="memory-guard").start()


def memory_limit_from(arg=None):
    """The limit in MB: the explicit argument, else VGPT_MEMORY_LIMIT_MB, else 0 (none)."""
    try: v = int(arg) if arg else int(os.environ.get("VGPT_MEMORY_LIMIT_MB", "0") or 0)
    except ValueError: v = 0
    return max(0, v)


def preflight(root=None, model="0.5b"):
    """-> (ok, report text, checks): the doctor's checks of this install."""
    from . import doctor
    checks = doctor.run_checks(root or ROOT, None, model)
    return (not doctor.summarize(checks)["fail"]), doctor.format_report(checks), checks


def startup(root=None, model="0.5b", memory_limit_mb=None, doctor_checks=True, guard=None):
    """The one start: harden, adopt the product model, start the memory guard, check the install.  -> (ok, report).  When `ok` is False nothing was started and the report says what to fix."""
    if _DONE: return True, ""
    from . import hardening
    hardening.harden()
    adopt_product_model(root)
    if doctor_checks:
        ok, report, checks = preflight(root, model)
        if not ok: return False, report
        from . import doctor
        for c in checks:
            if c["status"] == doctor.WARN and c["name"] == "Language model": print("note:", c["detail"])
    lim = memory_limit_from(memory_limit_mb)
    if lim: (guard or memory_guard)(lim)
    _DONE.append(time.time()); return True, ""

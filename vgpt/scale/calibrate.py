"""Measure, do not assume: how many cores does ONE model context still gain from on THIS machine?

On a Cortex-A72 (no dot-product instructions) two threads are twice as fast as one and four are nearly twice as fast again; on an x86 core with AVX-VNNI the model is already near its best at two.  The planner
needs that knee to decide between 'one context on all cores' and 'several contexts on two cores each', so it is measured once (a few seconds per thread count) and remembered per hardware signature.

    knee = calibrate(lambda t: OrtFrameModel(dir, threads=t), cores=8)       # -> {'times': {1: 0.18, 2: 0.11, 4: 0.09}, 'knee': 2}
    cached = load_cached(hw)                                                  # the answer measured earlier on this hardware, or None"""
import hashlib
import json
import os
import time

QUESTIONS = ["what is the average battery temperature over the last trip", "highest speed last week", "is my tyre pressure fine", "how many trips did I take this week", "what was my average speed yesterday", "how much energy did I use last week"]
GAIN = 0.25                                  # a further doubling of threads must make a parse at least this much faster to be worth the cores


def signature(hw):
    return hashlib.sha1(f"{hw.arch}|{sorted(hw.isa)}|{hw.cores}|{round(hw.mem_mb, -3)}".encode()).hexdigest()[:12]


def _path():
    base = os.environ.get("VEHICLEGPT_HOME") or os.path.join(os.path.expanduser("~"), ".vehiclegpt"); os.makedirs(base, exist_ok=True); return os.path.join(base, "calibration.json")


def load_cached(hw):
    try: return json.load(open(_path())).get(signature(hw))
    except (OSError, ValueError): return None


def save(hw, result):
    try: db = json.load(open(_path()))
    except (OSError, ValueError): db = {}
    db[signature(hw)] = dict(result, when=time.strftime("%Y-%m-%d"), hardware=hw.describe())
    with open(_path(), "w") as f: json.dump(db, f, indent=1)


def calibrate(make_model, cores, candidates=(1, 2, 4), reps=1, hw=None):
    """make_model(threads) -> an object with parse(text).  Returns {'times': {threads: median seconds}, 'knee': threads}."""
    times = {}
    for t in [c for c in candidates if c <= max(1, cores)]:
        m = make_model(t); m.parse("how fast am I going"); m.parse("how fast am I going"); ts = []
        for q in QUESTIONS * reps:
            t0 = time.perf_counter(); m.parse(q); ts.append(time.perf_counter() - t0)
        times[t] = sorted(ts)[len(ts) // 2]; del m
    ks = sorted(times); knee = ks[0]
    for a, b in zip(ks, ks[1:]):
        if times[b] <= times[a] * (1 - GAIN): knee = b
        else: break
    res = {"times": {str(k): round(v, 4) for k, v in times.items()}, "knee": knee}
    if hw is not None: save(hw, res)
    return res

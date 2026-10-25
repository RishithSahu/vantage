"""Latency of the whole assistant on the target board: every question is timed from text in to answer out.

    taskset -c 0,1 python bench/pi_latency.py [--n 300] [--db data/world_aster.db] [--out results/pi_latency.json]

Each run is one slice (--start/--count, 100 questions by default, about 10 minutes) so no run is long.  Questions: the acceptance set, the 500-question list and an evenly spread sample of the frozen meaning sets.  One warm-up question is asked first and reported separately."""
import argparse, csv, glob, json, os, statistics, sys, time
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def questions(n):
    out = []
    for ln in open(os.path.join(ROOT, "bench", "acceptance_v1.tsv"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2: out.append(("acc", p[2]))
    pool = []
    for f in sorted(glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv"))):
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if len(p) > 1: pool.append((os.path.basename(f)[:-4], p[1]))
    step = max(1, len(pool) // max(1, n)); out += pool[::step][:n]
    f = os.path.join(ROOT, "VehicleGPT_500_QA.csv")
    if os.path.exists(f):
        with open(f, encoding="utf-8-sig") as fh: out += [("qa500", r["question"]) for r in csv.DictReader(fh)][::5]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--db", default="data/world_aster.db"); ap.add_argument("--out", default="results/pi_latency.json"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=100); ap.add_argument("--limit", type=float, default=2.0)
    a = ap.parse_args()
    import vehiclegpt; vehiclegpt._adopt_product_model()                                       # the model the product ships
    from ui import server as S
    if os.environ.get("NO_RIG"): S.CameraRig.start = lambda self: None                      # the linked product does not decode a camera clip in the application: the car sends its own
    ns = SimpleNamespace(db=a.db, model="0.5b", camera="clip", webcam_index=0, live=None, linked=False)
    STAGE = {}
    try:                                                                                              # stage times of the frame model, when it is the ONNX one
        from vgpt.meaning import ort_backend as OB
        _p = OB.OrtFrameModel.parse
        def _parse(self, *a, **k):
            t = time.perf_counter(); r = _p(self, *a, **k); STAGE.clear(); STAGE.update(self.stats, parse_s=time.perf_counter() - t); return r
        OB.OrtFrameModel.parse = _parse
    except Exception: pass
    t = time.monotonic(); app = S.App(ns); print("loaded in %.1fs" % (time.monotonic() - t), flush=True)
    t = time.monotonic(); app.ask("how fast am I going", "driver"); warm = time.monotonic() - t; print("warm-up %.2fs" % warm, flush=True)
    if not app.brain.model_loaded: sys.exit("the language model did not load: %s" % app.brain.load_error)
    rows = []
    for i, (src, q) in enumerate(questions(a.n)[a.start:a.start + a.count]):
        t = time.monotonic()
        try: r = app.ask(q, "driver"); ok = bool(r.get("answer"))
        except Exception as e: r, ok = {"answer": repr(e)}, False
        dt = time.monotonic() - t; rows.append({"src": src, "q": q, "s": round(dt, 3), "ok": ok, "a": str(r.get("answer"))[:120], **{k: round(v, 3) for k, v in STAGE.items()}}); STAGE.clear()
        if dt > a.limit: print("SLOW %.2fs [%s] %s" % (dt, src, q), flush=True)
    ts = sorted(r["s"] for r in rows)
    print("n=%d median %.2f p95 %.2f max %.2f over-limit %d no-answer %d" % (len(ts), statistics.median(ts), ts[int(.95 * len(ts))], ts[-1], sum(x > a.limit for x in ts), sum(not r["ok"] for r in rows)))
    try: print("memory:", " ".join(l.split(":")[0] + l.split(":")[1].strip() for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM"))))
    except OSError: pass
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True); json.dump({"warm": warm, "rows": rows}, open(os.path.join(ROOT, a.out), "w"), indent=1)


if __name__ == "__main__": main()

"""Latency of the whole assistant on the target board, for one TYPE of car: every question is timed from text in to answer out, and split into reading the question (the model) and answering it (the data).

    taskset -c 0,1 python bench/pi_cartype.py --db data/kestrel_petrol_4yr.db --type petrol --out results/pi_petrol.json [--limit 2.0] [--repeat 1] [--home DIR]

Questions: the type's own probe files in bench/probe_questions (an electric car: the EV and broad files; a petrol or diesel car: the combustion files), the owner's two real sessions, and the trend / why questions.  Nothing is cached anywhere:
a repeated question is a real run.  The assistant is the application's own App (the same code the window uses), with the camera rig switched off (the linked product has no camera decoding in the application)."""
import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
PQ = os.path.join(ROOT, "bench", "probe_questions")
FILES = {"ev": ["ev_broad.txt", "hard_mix_ev.txt", "owner_round2_1024.txt", "user_session_1024.txt", "trend_complex.txt", "anomaly_scope.txt"],
         "petrol": ["ice_basic.txt", "ice_digest.txt", "ice_frames.txt", "hard_mix_ice.txt", "owner_round2_ice.txt", "user_session_1024.txt", "trend_complex.txt"]}
FILES["diesel"] = FILES["petrol"]


def questions(kind):
    seen, out = set(), []
    for f in FILES[kind]:
        p = os.path.join(PQ, f)
        if not os.path.exists(p): continue
        for ln in open(p, encoding="utf-8"):
            q = ln.strip()
            if not q or q.startswith("#") or q.lower() in seen: continue
            seen.add(q.lower()); out.append((f[:-4], q))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--type", required=True, choices=sorted(FILES)); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=float, default=2.0)
    ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--pipeline", default="auto", choices=["auto", "wire", "records", "off"]); ap.add_argument("--max", type=int, default=0); a = ap.parse_args()
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from ui import server as S
    S.CameraRig.start = lambda self: None
    ns = SimpleNamespace(db=a.db, model="0.5b", camera="clip", webcam_index=0, live=None, linked=False, pipeline=a.pipeline)
    t = time.monotonic(); app = S.App(ns); print("loaded in %.1fs" % (time.monotonic() - t), flush=True)
    t = time.monotonic(); app.ask("how fast am I going", "driver"); warm = time.monotonic() - t; print("warm-up %.2fs" % warm, flush=True)
    if not app.brain.model_loaded: sys.exit("the language model did not load: %s" % app.brain.load_error)
    h = app.health(); print("model:", h["model"].get("backend"), h["model"].get("adapter"), "frame", h["model"].get("frame"), "kind:", a.type, flush=True)
    qs = questions(a.type) * a.repeat
    if a.max: qs = qs[:a.max]
    rows = []
    for src, q in qs:
        t = time.monotonic()
        try: r = app.ask(q, "driver"); ok = bool(r.get("answer"))
        except Exception as e: r, ok = {"answer": repr(e)}, False
        dt = time.monotonic() - t; ms = r.get("ms") or {}
        rows.append({"src": src, "q": q, "s": round(dt, 3), "parse_ms": ms.get("parse"), "tools_ms": ms.get("tools"), "route": r.get("route"), "ok": ok, "a": str(r.get("answer"))[:100]})
        if dt > a.limit: print("SLOW %.2fs (parse %s, tools %s) [%s] %s" % (dt, ms.get("parse"), ms.get("tools"), r.get("route"), q), flush=True)
    ts = sorted(r["s"] for r in rows)
    print("n=%d median %.2f p95 %.2f max %.2f over-%.1fs %d (%.0f%%) no-answer %d" % (len(ts), statistics.median(ts), ts[int(.95 * len(ts))], ts[-1], a.limit, sum(x > a.limit for x in ts), 100 * sum(x > a.limit for x in ts) / len(ts), sum(not r["ok"] for r in rows)))
    try: print("memory:", " ".join(l.split(":")[0] + l.split(":")[1].strip() for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM"))))
    except OSError: pass
    os.makedirs(os.path.dirname(os.path.abspath(os.path.join(ROOT, a.out))), exist_ok=True); json.dump({"type": a.type, "db": a.db, "warm": warm, "model": h["model"], "rows": rows}, open(os.path.join(ROOT, a.out), "w"), indent=1)


if __name__ == "__main__": main()

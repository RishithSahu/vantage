"""Run the frozen 500-question QA benchmark (VehicleGPT_500_QA.csv) through the full application, one fresh brain per question (no dialogue context).
   python bench/qa500_run.py --mode learned_only --adapter models/qp-lora-v14d_s143.gguf --name v14d_s143_lo --out results/semantic_recovery/v15/runs
Inference only.  Raw answers, route, path, frontend tag and latency are stored; scoring is a separate step (bench/qa500_score.py)."""
import argparse, csv, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["legacy", "learned", "learned_only"]); ap.add_argument("--adapter", default=None); ap.add_argument("--name", required=True)
ap.add_argument("--csv", default=os.path.join(HERE, "VehicleGPT_500_QA.csv")); ap.add_argument("--out", required=True); ap.add_argument("--now", type=int, default=1787625900)
ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_demo.db")); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
os.environ["VGPT_FRONTEND"] = a.mode; os.environ["VGPT_THREADS"] = str(a.threads)
if a.adapter: os.environ["VGPT_QP_LORA"] = os.path.abspath(a.adapter)
if a.mode == "learned_only": os.environ["VGPT_TRIPWIRE"] = "1"
from vgpt.product import make_legacy_brain
rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
if a.limit: rows = rows[:a.limit]
os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, a.name + ".jsonl")
done = set()
if os.path.exists(path):
    for l in open(path, encoding="utf8"): done.add(json.loads(l)["id"])
brain = None
with open(path, "a", encoding="utf8") as fh:
    for i, r in enumerate(rows):
        if r["id"] in done: continue
        t = time.perf_counter(); err = None; ans = ""; route = None; fe = None; path_ = None; intent = None; plan = None
        try:
            if brain is None: brain = make_legacy_brain(a.db, slm=None, now=a.now)
            brain.qcs = type(brain.qcs)(); brain.qcs.last = None
            if getattr(brain, "_lf", None) is not None: brain._lf.last_plan = None
            res = brain.ask(r["question"], use_slm=False)
            ans = res.get("answer") or ""; f = res.get("facts") if isinstance(res.get("facts"), dict) else {}
            route = f.get("route"); fe = f.get("frontend"); path_ = res.get("path"); intent = res.get("intent"); plan = f.get("plan") or f.get("query")
        except Exception as e: err = f"{type(e).__name__}: {e}"
        fh.write(json.dumps({"id": r["id"], "mode": a.mode, "name": a.name, "answer": ans, "route": route, "frontend": fe, "path": path_, "intent": intent, "plan": plan,
                             "error": err, "ms": round((time.perf_counter() - t) * 1000, 1)}, ensure_ascii=False) + "\n"); fh.flush()
        if (i + 1) % 50 == 0: print(f"{a.name} {i + 1}/{len(rows)}", flush=True)
print("done", a.name)

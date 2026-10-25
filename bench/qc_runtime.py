"""Runtime cost of the query-compiler path under the target's limits: pinned to 2 CPU cores, one Brain, rules only (no model).
    python bench/qc_runtime.py [--db data/gen_courier_ext.db] [--out results/qc_runtime.json]
Reports per-stage latency (parse / validate / execute / compose / verify), end-to-end p50 / p95, and the peak memory of the process."""
import argparse, json, os, statistics, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import psutil
psutil.Process().cpu_affinity([0, 1])
os.environ["VGPT_QC"] = "first"; os.environ["VGPT_IR_SLM"] = "0"
from vgpt.brain import Brain
ap = argparse.ArgumentParser(); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); ap.add_argument("--out", default=os.path.join(HERE, "results", "qc_runtime.json")); a = ap.parse_args()
qs = [l.split("\t")[0] for l in open(os.path.join(HERE, "bench", "ir_gold.tsv"), encoding="utf-8") if "\t" in l and not l.rstrip().endswith("op=none")]
rss0 = psutil.Process().memory_info().rss / 1e6; b = Brain(a.db, now=1790406000, slm_ir=None); rows = []
for q in qs * 3:
    t = time.perf_counter(); r = b.ask(q); ms = (time.perf_counter() - t) * 1000
    if r["path"] == "qc": rows.append((ms, r["facts"].get("timings_ms", {}), r["facts"].get("parse_ms", 0)))
tot = [x[0] for x in rows]; stages = {k: statistics.mean(x[1].get(k, 0) for x in rows) for k in ("validate", "execute", "compose", "verify")}; stages["parse"] = statistics.mean(x[2] for x in rows)
mi = psutil.Process().memory_info(); res = {"questions": len(rows), "cores": 2, "end_to_end_ms": {"mean": statistics.mean(tot), "p50": statistics.median(tot), "p95": sorted(tot)[int(0.95 * (len(tot) - 1))], "max": max(tot)},
       "stage_mean_ms": {k: round(v, 2) for k, v in stages.items()}, "rss_start_mb": round(rss0), "peak_mb": round(getattr(mi, "peak_wset", mi.rss) / 1e6)}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))

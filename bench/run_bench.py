"""One benchmark configuration, run INSIDE the simulated 2 GB / 2-core target (Windows Job Object). Spawned by bench/orchestrate.py so that a
configuration which does not fit simply crashes its own process and is recorded as 'did not fit'.
  python bench/run_bench.py --model models/x.gguf --mmap 0 --slowdown 1 --os-mb 700 --out results/x.json"""
import argparse, json, os, subprocess, sys, time, statistics as st
# Footprint discipline for a 2-core target: BLAS thread buffers scale with detected cores (354 MB at import on a 12-core host). We do no BLAS maths.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt import governor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="none"); ap.add_argument("--fmt", default="words"); ap.add_argument("--mmap", type=int, default=0); ap.add_argument("--slowdown", type=float, default=1.0)
ap.add_argument("--mem", type=int, default=2048); ap.add_argument("--os-mb", type=int, default=700); ap.add_argument("--cores", default="0,2"); ap.add_argument("--ingest", type=int, default=1)
ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--label", default="")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = {"config": vars(a)}
lim = governor.apply(mem_mb=a.mem, cores=tuple(int(x) for x in a.cores.split(",")), slowdown=a.slowdown); res["limits"] = lim

ballast = governor.start_os_ballast(a.os_mb)
stats_file = f"{ROOT}/data/ingest_stats_{os.getpid()}.json"
ingest = None
if a.ingest:
    ingest = subprocess.Popen([sys.executable, "-m", "vgpt.ingestd", "--replay", f"{ROOT}/data/replay.pkl", "--db", f"{ROOT}/data/live_bench_{os.getpid()}.db", "--speed", "1", "--loop", "--stats", stats_file],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)
res["mem_peak_after_os_and_ingest_mb"] = round(governor.peak_job_mb(), 1)

from vgpt.brain import Brain, Slm, FAST_MIN
slm = None
if a.model != "none":
    slm = Slm(os.path.join(ROOT, a.model) if not os.path.isabs(a.model) else a.model, threads=a.threads, mmap=bool(a.mmap), fmt=a.fmt)
    res["model_load_s"] = round(slm.load_s, 2)
    t = time.perf_counter(); slm.warmup(); res["warmup_s"] = round(time.perf_counter() - t, 2)
brain = Brain(f"{ROOT}/data/vehicle.db", slm=slm)
res["mem_peak_after_model_mb"] = round(governor.peak_job_mb(), 1)

Q = json.load(open(f"{ROOT}/bench/questions.json")); T = json.load(open(f"{ROOT}/data/truth.json"))
RELEVANT = {"EVENTS": ("event", "w"), "BEFORE_EVENT": ("event",), "TREND": ("metric", "w"), "STATUS": ("metric",), "WHY_RANGE": (), "COMPARE": (), "LARGEST": (), "NONE": ()}


def parse_ok(q, r):
    p = r["params"]
    return r["intent"] == q["intent"] and all({"event": p["event"], "metric": p["metric"], "w": p["w"]}[k] == (q[k] if q[k] != "-" else None) for k in RELEVANT[q["intent"]])


def numbers_ok(q, r):
    """Independent numeric check against simulator ground truth; None = no numeric truth for this family."""
    f = r["facts"]; it = q["intent"]
    if not f.get("ok"): return it == "NONE"
    if it == "EVENTS": return f["count"] == T["events_counts"][f["type"]][f["when"]]
    if it == "WHY_RANGE": return abs(f["first"] - T["why14"]["first"]) <= 1.0 and abs(f["last"] - T["why14"]["last"]) <= 1.0
    if it == "LARGEST": return f["day"] == T["largest_eff_drop_day"]
    if it == "COMPARE": return abs(f["city"] - T["city_whkm"]) / T["city_whkm"] < 0.06 and abs(f["highway"] - T["highway_whkm"]) / T["highway_whkm"] < 0.06
    if it == "BEFORE_EVENT":
        d = {x["signal"]: x for x in f["deviations"]}
        return f["ts"] == T["overtemp_ts"] and abs(d["coolant_flow"]["end"] - T["flow_at_overtemp"]) <= 0.3 and abs(d["batt_temp"]["end"] - T["batt_at_overtemp"]) <= 0.3
    return None


rows = []; cpu0 = time.process_time(); wall0 = time.perf_counter()
for rep in range(1):
    for q in Q:
        t = time.perf_counter(); r = brain.ask(q["q"]); wall = (time.perf_counter() - t) * 1000
        rows.append({"q": q["q"], "split": q["split"], "intent_true": q["intent"], "intent": r["intent"], "path": r["path"], "parse_ok": parse_ok(q, r), "numbers_ok": numbers_ok(q, r) if parse_ok(q, r) else (False if q["intent"] != "NONE" else None),
                     "ms_total": round(wall, 1), "ms_parse": r["ms"]["parse"], "ms_tools": r["ms"]["tools"], "slm_out_tokens": (r["slm"] or {}).get("out_tokens"), "answer": r["answer"][:260]})
res["queries"] = rows
res["query_cpu_s"] = round(time.process_time() - cpu0, 2); res["query_wall_s"] = round(time.perf_counter() - wall0, 2)
res["mem_peak_job_mb"] = round(governor.peak_job_mb(), 1)
res["working_sets_mb"] = {f"{v[0]}": round(v[1], 1) for v in governor.working_sets_mb().values()}
if ingest:
    time.sleep(5.5)
    try: res["ingest"] = json.load(open(stats_file))
    except (OSError, ValueError): res["ingest"] = None
    ingest.kill()
    for ext in ("", "-wal", "-shm"):
        try: os.remove(f"{ROOT}/data/live_bench_{os.getpid()}.db{ext}")
        except OSError: pass
    try: os.remove(stats_file)
    except OSError: pass
ballast.stdin.write("\n"); ballast.stdin.flush()
json.dump(res, open(a.out, "w"), indent=1)


def summ(split=None):
    rs = [r for r in rows if split in (None, r["split"])]; n = len(rs); ms = sorted(r["ms_total"] for r in rs)
    nums = [r["numbers_ok"] for r in rs if r["numbers_ok"] is not None]
    return (f"parse {sum(r['parse_ok'] for r in rs)}/{n} | numbers {sum(nums)}/{len(nums)} | latency median {st.median(ms):.0f} ms p95 {ms[int(.95 * (n - 1))]:.0f} ms max {ms[-1]:.0f} ms | "
            f"slm used {sum(r['path'] == 'slm' for r in rs)}/{n}")


print(f"[{a.label or a.model}] K={a.slowdown} mmap={a.mmap} os={a.os_mb}MB limit={a.mem}MB")
print("  all :", summ()); print("  test:", summ("test"))
print(f"  memory: peak job {res['mem_peak_job_mb']} MB of {a.mem} MB (after OS+ingest {res['mem_peak_after_os_and_ingest_mb']}, after model {res['mem_peak_after_model_mb']}); working sets {res['working_sets_mb']}")
if ingest: print("  ingest:", res.get("ingest"))

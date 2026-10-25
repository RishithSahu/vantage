"""Benchmark run INSIDE the ARM64 QEMU guest (Linux, 2 GB RAM, 2 cores). Reports guest-visible memory (the real OS shares the 2 GB), CPU model, latency and accuracy.
  python bench/run_guest.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --mmap 1 [--splits dev,test,blind,field,typo] [--ingest 1]
Prints one JSON object between @@RESULT_BEGIN@@ / @@RESULT_END@@ markers (the host harness captures it from the serial console)."""
import argparse, json, os, platform, random, statistics as st, subprocess, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="none"); ap.add_argument("--mmap", type=int, default=1); ap.add_argument("--splits", default="dev,test,blind,field,typo")
ap.add_argument("--ingest", type=int, default=1); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--state", default=""); ap.add_argument("--label", default=""); a = ap.parse_args()


def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":"); d[k] = int(v.split()[0]) / 1024.0
    return {"total_mb": round(d["MemTotal"]), "free_mb": round(d["MemFree"]), "available_mb": round(d["MemAvailable"]), "used_mb": round(d["MemTotal"] - d["MemAvailable"]), "swap_total_mb": round(d.get("SwapTotal", 0))}


def proc_mem(pid="self"):
    out = {}
    for l in open(f"/proc/{pid}/status"):
        if l.startswith(("VmRSS", "VmHWM")): out[l.split(":")[0]] = round(int(l.split()[1]) / 1024)
    return out


res = {"platform": {"machine": platform.machine(), "python": platform.python_version(), "cpus_online": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))}}
for l in open("/proc/cpuinfo"):
    if l.startswith(("CPU part", "model name", "Features")) and l.split(":")[0].strip() not in res["platform"]: res["platform"][l.split(":")[0].strip()] = l.split(":", 1)[1].strip()[:90]
res["mem_os_idle"] = meminfo()

ingest = None
if a.ingest and os.path.exists(os.path.join(ROOT, "data", "replay.pkl")):
    ingest = subprocess.Popen([sys.executable, "-m", "vgpt.ingestd", "--replay", "data/replay.pkl", "--db", "/tmp/live_guest.db", "--speed", "1", "--loop", "--stats", "/tmp/ingest_stats.json"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5); res["mem_with_ingest"] = meminfo()

from vgpt.brain import Brain, Slm
slm = None
if a.model != "none":
    t = time.perf_counter(); slm = Slm(os.path.join(ROOT, a.model) if not os.path.isabs(a.model) else a.model, threads=a.threads, mmap=bool(a.mmap), state_path=(os.path.join(ROOT, a.state) if a.state else None)); res["model_load_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter(); slm.warmup(); res["warmup_s"] = round(time.perf_counter() - t, 1); res["state_source"] = slm.state_source
    res["ready_s"] = round(res["model_load_s"] + res["warmup_s"], 1)
res["mem_after_model"] = meminfo(); res["assistant_after_model"] = proc_mem()
brain = Brain(os.path.join(ROOT, "data", "vehicle.db"), slm=slm)

Q = json.load(open(os.path.join(ROOT, "bench", "questions.json")))
RELEVANT = {"EVENTS": ("event", "w"), "BEFORE_EVENT": ("event",), "TREND": ("metric", "w"), "STATUS": ("metric",), "WHY_RANGE": (), "COMPARE": (), "LARGEST": (), "NONE": (), "USAGE": ("w",), "SUMMARY": ("metric", "w"), "TYRES": (), "WEATHER": (), "SERVICE": (), "HEALTH": (), "DRIVING": (), "PLACES": ()}
def full_ok(q, intent, p): return intent == q["intent"] and all(q[k] == "-" or {"event": p["event"], "metric": p["metric"], "w": p["w"]}[k] == q[k] for k in RELEVANT[q["intent"]])
def typo(s, rng):
    out = []
    for w in s.split(" "):
        if len(w) >= 5 and w.isalpha() and rng.random() < 0.25:
            i = rng.randrange(1, len(w) - 1); k = rng.randrange(3); w = (w[:i] + w[i + 1] + w[i] + w[i + 2:]) if k == 0 else (w[:i] + w[i + 1:]) if k == 1 else (w[:i] + w[i] + w[i:])
        out.append(w)
    return " ".join(out)

rng = random.Random(3); per = {}; min_avail = res["mem_after_model"]["available_mb"]; rows = []
for split in a.splits.split(","):
    qs = [dict(q, q=typo(q["q"], rng), split="typo") for q in Q if q["split"] in ("test", "blind")] if split == "typo" else [q for q in Q if q["split"] == split]
    ok = ab = 0
    for q in qs:
        t = time.perf_counter(); r = brain.ask(q["q"]); ms = (time.perf_counter() - t) * 1000
        g = full_ok(q, r["intent"], r["params"]); ok += g; ab += (not g and r["path"] == "clarify" and q["intent"] != "NONE")
        min_avail = min(min_avail, meminfo()["available_mb"]); rows.append({"split": split, "ms": round(ms, 1), "path": r["path"], "ok": g, "q": q["q"][:70]})
    per[split] = {"n": len(qs), "correct": ok, "abstain": ab}
# ---- extended domains on a GENERATED vehicle (tyres, rain/light, service, health, driving, places) + a tester's own chat questions
ext_db = os.path.join(ROOT, "data", "gen_demo.db"); ext_q = os.path.join(ROOT, "bench", "questions_ext.json")
if os.path.exists(ext_db) and os.path.exists(ext_q):
    b2 = Brain(ext_db, slm=slm); rows2 = []; ok2 = 0; EQ = json.load(open(ext_q, encoding="utf-8"))
    for q in EQ:
        t = time.perf_counter(); r = b2.ask(q["q"]); ms_ = (time.perf_counter() - t) * 1000
        g = (r["intent"] == q["intent"]) if q["intent"] != "NONE" else (r["intent"] == "NONE" and r["path"] in ("gate", "unsupported", "clarify")); ok2 += g
        min_avail = min(min_avail, meminfo()["available_mb"]); rows2.append({"ms": round(ms_, 1), "path": r["path"], "ok": g, "q": q["q"][:60]})
    m2 = sorted(x["ms"] for x in rows2); per["ext"] = {"n": len(EQ), "correct": ok2, "abstain": 0}
    res["ext"] = {"db": "data/gen_demo.db (generated)", "n": len(EQ), "correct": ok2, "median_ms": round(st.median(m2), 1), "p95_ms": m2[int(.95 * (len(m2) - 1))], "max_ms": m2[-1], "under_2s": sum(1 for m in m2 if m < 2000), "misses": [x["q"] for x in rows2 if not x["ok"]][:8]}
# ---- the analysis engine: the 120-question stress list + labelled paraphrases, on a generated vehicle that has a cooling-pump fault and an over-temperature warning
tort_db = os.path.join(ROOT, "data", "gen_courier.db")
if os.path.exists(tort_db):
    b3 = Brain(tort_db, slm=slm); rows3 = []; routed = labelled = via_qa = 0
    for fn in ("bench/torture_questions.txt", "bench/torture2_questions.txt", "bench/torture_dev.tsv", "bench/torture_blind_used.tsv"):
        p3 = os.path.join(ROOT, fn)
        if not os.path.exists(p3): continue
        for line in open(p3, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"): continue
            lab, q = line.split("\t", 1) if fn.endswith(".tsv") else (None, line)
            t = time.perf_counter(); r = b3.ask(q); ms_ = (time.perf_counter() - t) * 1000; f = r["facts"] if isinstance(r["facts"], dict) else {}
            via_qa += r["path"] == "qa"
            if lab: labelled += 1; routed += (f.get("route") == lab) if r["path"] == "qa" else (lab in ("legacy", "why_range2") and r["path"] != "gate")
            min_avail = min(min_avail, meminfo()["available_mb"]); rows3.append({"ms": round(ms_, 1), "q": q[:60], "path": r["path"]})
    m3 = sorted(x["ms"] for x in rows3)
    res["torture"] = {"db": "data/gen_courier.db (generated)", "n": len(rows3), "answered_by_analysis_engine": via_qa, "labelled": labelled, "routed_as_intended": routed, "median_ms": round(st.median(m3), 1), "p95_ms": m3[int(.95 * (len(m3) - 1))],
                      "max_ms": m3[-1], "under_2s": sum(1 for m in m3 if m < 2000), "under_5s": sum(1 for m in m3 if m < 5000), "slowest": sorted(rows3, key=lambda x: -x["ms"])[:5], "assistant_mb": proc_mem()}
res["accuracy"] = per; res["mem_min_available_during_queries_mb"] = min_avail; res["mem_after_queries"] = meminfo(); res["assistant_after_queries"] = proc_mem()
ms = sorted(x["ms"] for x in rows); slm_ms = sorted(x["ms"] for x in rows if x["path"] == "slm")
res["latency_ms"] = {"n": len(ms), "median": round(st.median(ms), 1), "p95": ms[int(.95 * (len(ms) - 1))], "max": ms[-1], "answers_under_2s": sum(1 for m in ms if m < 2000),
                     "slm_calls": len(slm_ms), "slm_median": round(st.median(slm_ms), 1) if slm_ms else None, "slm_max": slm_ms[-1] if slm_ms else None}
res["slowest"] = sorted(rows, key=lambda x: -x["ms"])[:5]
if ingest:
    time.sleep(1)
    try: res["ingest"] = json.load(open("/tmp/ingest_stats.json"))
    except (OSError, ValueError): res["ingest"] = None
    ingest.kill()
if a.ingest and os.path.exists(os.path.join(ROOT, "data", "replay.pkl")):               # unpaced throughput of the REAL wire-format decode path inside the guest
    subprocess.run([sys.executable, "-m", "vgpt.ingestd", "--replay", "data/replay.pkl", "--db", "/tmp/live_max.db", "--speed", "max", "--stats", "/tmp/ingest_max.json", "--max-seconds", "25"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try: res["ingest_unpaced"] = json.load(open("/tmp/ingest_max.json"))
    except (OSError, ValueError): res["ingest_unpaced"] = None
print("@@RESULT_BEGIN@@"); print(json.dumps(res)); print("@@RESULT_END@@")

"""Complete-answer benchmark for the VANTAGE pipeline, meant to run INSIDE the ARM64 QEMU guest (Linux, 2 GB, 2 cores) but runnable anywhere.
Per question it times every stage of the real path: rules parse -> (when the rules are unsure) frozen-model planner with the schema prompt -> interpretation guard -> validate -> execute -> evidence -> compose -> verify.
Reports per-stage and total latency (mean / p50 / p95), peak resident memory of the process, whole-guest memory (the OS and the page cache share the same 2 GB), and the one-time cost of building the prompt state.
EMULATION NOTICE: under QEMU TCG the guest timings are software-emulation timings on an x86 host, not Cortex-A55 timings.
    python bench/run_guest_v2.py --model models/LFM2.5-350M-Q4_K_M.gguf --questions 24 [--bank 4 --draft] [--db data/gen_courier_ext.db]
Prints one JSON object between @@RESULT_BEGIN@@ / @@RESULT_END@@."""
import argparse, json, os, platform, statistics as st, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
NOW = 1790406000

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/LFM2.5-350M-Q4_K_M.gguf"); ap.add_argument("--questions", type=int, default=24); ap.add_argument("--bank", type=int, default=0); ap.add_argument("--draft", action="store_true")
ap.add_argument("--db", default="data/gen_courier_ext.db"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--set", default="bench/indep_dev.tsv"); ap.add_argument("--cache", default=os.path.join(ROOT, "cache")); ap.add_argument("--label", default=""); ap.add_argument("--n_ctx", type=int, default=4608); a = ap.parse_args()


def meminfo():
    try:
        d = {}
        for l in open("/proc/meminfo"):
            k, v = l.split(":"); d[k] = int(v.split()[0]) / 1024.0
        return {"total_mb": round(d["MemTotal"]), "available_mb": round(d["MemAvailable"]), "used_mb": round(d["MemTotal"] - d["MemAvailable"]), "cached_mb": round(d.get("Cached", 0))}
    except OSError: return {}


def proc_peak_mb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmHWM"): return round(int(l.split()[1]) / 1024)
    except OSError: pass
    try:
        import psutil; mi = psutil.Process().memory_info(); return round(getattr(mi, "peak_wset", mi.rss) / 1e6)
    except Exception: return None


def pct(xs, p): xs = sorted(xs); return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None
def summ(xs): return {"n": len(xs), "mean": round(st.mean(xs), 1), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": round(max(xs), 1)} if xs else {"n": 0}


res = {"label": a.label, "platform": {"machine": platform.machine(), "python": platform.python_version(), "cpus": os.cpu_count()}, "mem_idle": meminfo(), "config": {"model": a.model, "bank": a.bank, "draft": a.draft, "threads": a.threads, "n_ctx": a.n_ctx}}
for l in (open("/proc/cpuinfo") if os.path.exists("/proc/cpuinfo") else []):
    if l.startswith(("CPU part", "model name")): res["platform"][l.split(":")[0].strip()] = l.split(":", 1)[1].strip()[:80]

from vgpt import nl2ir
from vgpt.brain import Brain
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate
from vgpt import interpretation
from vgpt.schema_planner import SchemaPlanner
from bench.indep_io import load

t = time.perf_counter(); brain = Brain(os.path.join(ROOT, a.db), slm_ir=None, now=NOW); qc = brain.qc(); res["engine_load_s"] = round(time.perf_counter() - t, 2); res["mem_after_engine"] = meminfo()
t = time.perf_counter(); os.makedirs(a.cache, exist_ok=True)
planner = SchemaPlanner(os.path.join(ROOT, a.model), mode="full", bank_k=a.bank, n_ctx=a.n_ctx, threads=a.threads, cache_dir=a.cache); res["model_load_and_prefix_s"] = round(time.perf_counter() - t, 1)
res["prefix"] = {"tokens": planner.be.prefix_tokens, "from_disk": getattr(planner.be, "prefix_from_disk", None), "seconds": round(getattr(planner.be, "prefix_s", 0.0), 2)}; res["mem_after_model"] = meminfo(); res["proc_peak_mb_after_model"] = proc_peak_mb()

rows = [r for r in load(os.path.join(ROOT, a.set)) if not r["prev_text"]][: a.questions]
per = []; stage = {"rules": [], "model": [], "execute": [], "total": [], "total_model_path": [], "total_rules_path": []}; statuses = {}
for r in rows:
    text = r["text"]; t0 = time.perf_counter(); pr = nl2ir.parse(text, now=NOW, tz=3600, profile=qc.profile, state=None); t_rules = (time.perf_counter() - t0) * 1000; q = pr.query if pr.ok else None; src = "rules"; t_model = 0.0; info = {}
    if q is None:
        src = "model"; t1 = time.perf_counter(); draft = pr.query.to_text() if (a.draft and pr.query is not None and pr.query.op) else None
        mq, info = planner.compile(text, None, draft=draft); t_model = (time.perf_counter() - t1) * 1000
        if mq is not None and not interpretation.interpret(text, mq, None, source="model", now=NOW, tz=3600).issues(): q = mq
    t2 = time.perf_counter(); status = "refused"
    if q is not None:
        try:
            if validate(q, qc.profile).valid: status = qc.execute(q).status
            else: status = "invalid"
        except IRError: status = "invalid"
    t_exec = (time.perf_counter() - t2) * 1000; total = t_rules + t_model + t_exec
    per.append({"id": r["id"], "source": src, "status": status, "rules_ms": round(t_rules, 1), "model_ms": round(t_model, 1), "exec_ms": round(t_exec, 1), "total_ms": round(total, 1), "prompt_tokens": info.get("prompt_tokens"), "out_tokens": info.get("out_tokens")})
    stage["rules"].append(t_rules); stage["execute"].append(t_exec); stage["total"].append(total); statuses[status] = statuses.get(status, 0) + 1
    if src == "model": stage["model"].append(t_model); stage["total_model_path"].append(total)
    else: stage["total_rules_path"].append(total)
res["questions"] = len(rows); res["statuses"] = statuses; res["latency_ms"] = {k: summ(v) for k, v in stage.items()}; res["per_question"] = per
res["proc_peak_mb"] = proc_peak_mb(); res["mem_end"] = meminfo()
print("@@RESULT_BEGIN@@"); print(json.dumps(res)); print("@@RESULT_END@@")

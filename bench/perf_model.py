"""The product (ModelBrain: retrieval + one frozen-model plan + validation + typed pipeline) INSIDE the simulated target: a Windows Job Object with a hard 2048 MB committed-memory limit shared by everything in the "car" (an OS-ballast process standing for the vehicle OS and middleware, the ingest daemon and the assistant), CPU affinity pinned to 2 logical cores. No QEMU. The laptop's cores are not Cortex-A5x cores: nothing here is an ARM timing, and the 700 MB OS ballast is a local stress ASSUMPTION, not a measurement of any vehicle OS.
    python bench/perf_model.py --cores 10,11 --os-mb 700 --set dev --limit 40 --out results/perf_model_lfm.json [--model models/granite-4.0-1b-Q4_K_M.gguf --n-ctx 8192]
Reported separately: cold start (embedding model, model load, the one static prefix: built or loaded from disk), per-stage latency over warm questions (retrieval + embedding, per-message render, state restore, prompt evaluation, generation, compile, execution, total), the peak memory of the WHOLE job at each phase (not the assistant alone), the ingestion keep-up, and whether anything failed to fit. Cached and ordinary inference are compared on a few questions after the run. A warm cache never hides the cold start."""
import argparse, json, os, subprocess, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import governor

ap = argparse.ArgumentParser(); ap.add_argument("--cores", default="10,11"); ap.add_argument("--mem", type=int, default=2048); ap.add_argument("--os-mb", type=int, default=700); ap.add_argument("--ingest", type=int, default=1)
ap.add_argument("--set", default="dev"); ap.add_argument("--limit", type=int, default=40); ap.add_argument("--out", required=True); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db"))
ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")); ap.add_argument("--n-ctx", type=int, default=6144)
a = ap.parse_args(); res = {"config": vars(a)}
res["limits"] = governor.apply(mem_mb=a.mem, cores=tuple(int(x) for x in a.cores.split(",")), slowdown=1.0)
ballast = governor.start_os_ballast(a.os_mb); ingest = None; stats = os.path.join(HERE, "data", f"ingest_stats_model_{os.getpid()}.json")
if a.ingest:
    ingest = subprocess.Popen([sys.executable, "-m", "vgpt.ingestd", "--replay", os.path.join(HERE, "data", "replay.pkl"), "--db", os.path.join(HERE, "data", f"live_model_{os.getpid()}.db"), "--speed", "1", "--loop", "--stats", stats], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)
res["job_mb_after_os_and_ingest"] = round(governor.peak_job_mb(), 1)
try:
    import sqlite3
    from bench.indep_io import load
    from training.build_dataset import DEFAULT_NOW
    from vgpt.modelbrain import ModelBrain
    from vgpt.modelplan import Planner, State
    t = time.perf_counter(); conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); P = Planner(conn, model_path=a.model, n_ctx=a.n_ctx)
    res["cold_s"] = {"embedder_and_index": round(P.retrieval_load_s, 2), "model_load_and_prefix": round(P.model_load_s, 1), "prefix_seconds": round(P.prefix_s, 1), "prefix_from_disk": P.lm.prefixes["plan"]["from_disk"], "total": round(time.perf_counter() - t, 1)}
    res["prefix_tokens"] = P.lm.prefixes["plan"]["tokens"]; res["dialect"] = P.dialect.name; res["model_identity"] = P.lm.identity; res["job_mb_after_model"] = round(governor.peak_job_mb(), 1)
    b = ModelBrain(a.db, planner=P, now=DEFAULT_NOW)
    rows = [r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv") if a.set == "dev" else a.set) if not r["prev_text"]][: a.limit]; per = []
    for r in rows:
        b.reset(); t0 = time.perf_counter(); out = b.ask(r["text"]); ms = (time.perf_counter() - t0) * 1000
        per.append({"id": r["id"], "ms": round(ms, 1), "outcome": b.outcome, "timing": dict(b.timing)})
    res["questions"] = per; res["job_mb_peak"] = round(governor.peak_job_mb(), 1); res["fits"] = True
    xs = sorted(p["ms"] for p in per); q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1)))]
    st = [p["timing"] for p in per if p["timing"]]; mean = lambda k: round(sum(s.get(k) or 0 for s in st) / max(1, len(st)), 1); p95 = lambda k: sorted(s.get(k) or 0 for s in st)[int(0.95 * (len(st) - 1))]
    res["latency_ms"] = {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p50": q(0.5), "p95": q(0.95), "max": xs[-1], "over_2s": sum(1 for x in xs if x > 2000)}
    res["stage_ms"] = {k: {"mean": mean(k), "p95": p95(k)} for k in ("retrieve_ms", "render_ms", "state_ms", "prefill_ms", "decode_ms", "grammar_ms", "compile_ms", "execute_ms", "total_ms")}
    res["tokens"] = {"prompt_mean": mean("prompt_tokens"), "out_mean": mean("out_tokens")}
    # cached versus ordinary inference after the run (same call expected)
    agree = []
    for r in rows[:3]:
        msg = P.build_message(r["text"], State(), DEFAULT_NOW)[0]; cached, _ = P.lm.generate(msg, grammar=P.grammar_new, max_tokens=130, stop_text=None, prefix="plan", assistant_prefix=P.dialect.open_text)
        plain = P.lm.generate_uncached(P.system, msg, P.shots, grammar=P.grammar_new, max_tokens=130, stop_text=None, assistant_prefix=P.dialect.open_text, tools=P.tools); agree.append(cached == plain)
    res["cached_equals_ordinary"] = agree
except MemoryError: res["fits"] = False; res["error"] = "MemoryError"
except Exception as e: res["fits"] = False; res["error"] = f"{type(e).__name__}: {e}"
finally:
    try:
        if ingest: ingest.terminate()
        if os.path.exists(stats): res["ingest"] = json.load(open(stats))
    except Exception: pass
    res["job_mb_peak_final"] = round(governor.peak_job_mb(), 1); res["working_sets_mb"] = governor.working_sets_mb()
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "questions"}, indent=1))

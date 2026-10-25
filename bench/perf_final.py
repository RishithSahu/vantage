"""Final-configuration performance on the laptop (2 pinned x86 cores; NOT ARM). Per question the complete assistant (Brain, policy="clarify"; with --config select the frozen-LFM candidate selector, with --config none the product configuration vgpt.product.make_brain) is asked the question; stages are timed by wrapping the real functions.
  python tools_pin.py 8,9 bench/perf_final.py [--out results/perf_final.json]
Reports: cold start (model load, prompt-state construction from scratch vs restored), warm per-question latency by family / path / outcome, the common-vs-analytic split pre-registered in VANTAGE_STATUS.md, time to clarification vs time to final answer, peak memory.
The questions are the dev set (follow-ups are preceded by their previous question, untimed). A path that exceeds 2 s is reported as such."""
import argparse, json, os, shutil, statistics as st, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ["VGPT_QC"] = "all"
from bench.indep_io import load
from bench.manifest import manifest
from vgpt.product import make_legacy_brain as make_brain
NOW = 1790406000
COMMON = {"LK", "EVT", "THR", "CMP", "SEL", "CAP", "FU", "MSY", "REF"}; ANALYTIC = {"EXPL", "CHG", "CTX"}


def peak_mb():
    try:
        import psutil; mi = psutil.Process().memory_info(); return round(getattr(mi, "peak_wset", mi.rss) / 1e6)
    except Exception: return None


def pct(xs, p): xs = sorted(xs); return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None
def summ(xs): return {"n": len(xs), "mean": round(st.mean(xs), 1), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": round(max(xs), 1), "over_2s": sum(1 for x in xs if x > 2000)} if xs else {"n": 0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-350M-Q4_K_M.gguf")); ap.add_argument("--out", default=os.path.join(HERE, "results", "perf_final.json")); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); ap.add_argument("--config", default="select", choices=["select", "none"], help="select = frozen-model candidate selection; none = deterministic candidates + clarification policy, no model"); a = ap.parse_args()
    res = {"manifest": manifest({"policy": "clarify", "selector": os.path.basename(a.model) if a.config == "select" else None, "n_ctx": 4608, "retrieval": False, "pinned_cores": 2}), "note": "x86 laptop, 2 pinned cores; not ARM"}
    from vgpt.candidate_planner import CandidateSelector
    cache = tempfile.mkdtemp(); sel = None
    if a.config == "select":
        t = time.perf_counter(); cold = CandidateSelector(a.model, cache_dir=cache); res["cold_start_s"] = {"model_load_and_prefix_build": round(time.perf_counter() - t, 2), "prefix_tokens": cold.be.prefix_tokens, "model_load_only": round(cold.be.load_s, 2), "prefix_build": round(cold.be.prefix_s, 2)}
        del cold; t = time.perf_counter(); sel = CandidateSelector(a.model, cache_dir=cache); res["warm_start_s"] = {"model_load_and_prefix_restore": round(time.perf_counter() - t, 2), "prefix_from_disk": sel.be.prefix_from_disk, "prefix_restore": round(sel.be.prefix_s, 3)}
    res["peak_mb_after_model"] = peak_mb()
    from vgpt import nl2ir, candidates as CD, policy as POL
    from vgpt.brain import Brain
    acc = {"parse": 0.0, "candidates": 0.0, "select": 0.0, "execute": 0.0}
    def wrap(mod, name, key):
        f = getattr(mod, name)
        def g(*x, **k):
            t0 = time.perf_counter()
            try: return f(*x, **k)
            finally: acc[key] += (time.perf_counter() - t0) * 1000
        setattr(mod, name, g)
    wrap(nl2ir, "parse", "parse"); wrap(CD, "generate", "candidates")
    if sel is not None:
        orig_select = sel.select
        def sel_wrapped(*x, **k):
            t0 = time.perf_counter()
            try: return orig_select(*x, **k)
            finally: acc["select"] += (time.perf_counter() - t0) * 1000
        sel.select = sel_wrapped
    rows = load(os.path.join(HERE, "bench", "indep_dev.tsv")); per = []; b = Brain(a.db, slm_ir=None, now=NOW, policy="clarify", selector=sel, qc_mode="all") if sel is not None else make_brain(a.db, now=NOW)            # config none == the product configuration
    from vgpt import qc as QC
    orig_exec = QC.QueryCompiler.execute
    def ex_wrapped(self, *x, **k):
        t0 = time.perf_counter()
        try: return orig_exec(self, *x, **k)
        finally: acc["execute"] += (time.perf_counter() - t0) * 1000
    QC.QueryCompiler.execute = ex_wrapped
    for r in rows:
        b.qcs = nl2ir.ConvState(); b.trace.__init__()
        if r["prev_text"]: b.ask(r["prev_text"], use_slm=False)
        for k in acc: acc[k] = 0.0
        n0 = len(b.trace.records); t0 = time.perf_counter(); out = b.ask(r["text"], use_slm=False); ms = (time.perf_counter() - t0) * 1000
        outcome = b.trace.records[-1].outcome if len(b.trace.records) > n0 else "legacy_or_refused"
        path = "candidate+model" if acc["select"] > 0 else ("candidates" if acc["candidates"] > 0 and outcome != "legacy_or_refused" else "rules" if outcome in ("answered", "no_data", "limitation") and acc["candidates"] == 0 else "legacy/other" if outcome == "legacy_or_refused" else "candidates")
        per.append({"id": r["id"], "family": r["family"], "group": "common" if r["family"] in COMMON else "analytic" if r["family"] in ANALYTIC else "other", "ms": round(ms, 1), "outcome": outcome, "route": out.get("path"), "path": path, **{k: round(v, 1) for k, v in acc.items()}})
    res["peak_mb_end"] = peak_mb(); res["per_question"] = per; cut = lambda f: [p["ms"] for p in per if f(p)]
    res["latency_ms"] = {"all": summ(cut(lambda p: True)), "common (pre-registered)": summ(cut(lambda p: p["group"] == "common")), "analytic": summ(cut(lambda p: p["group"] == "analytic"))}
    res["by_outcome"] = {o: summ(cut(lambda p, o=o: p["outcome"] == o)) for o in sorted({p["outcome"] for p in per})}
    res["by_path"] = {o: summ(cut(lambda p, o=o: p["path"] == o)) for o in sorted({p["path"] for p in per})}
    res["by_family"] = {f: summ(cut(lambda p, f=f: p["family"] == f)) for f in sorted({p["family"] for p in per})}
    res["stage_ms_mean"] = {k: round(st.mean(p[k] for p in per), 1) for k in acc}; res["time_to_clarification_ms"] = summ(cut(lambda p: p["outcome"] == "clarify")); res["time_to_final_answer_ms"] = summ(cut(lambda p: p["outcome"] in ("answered", "no_data", "limitation")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); shutil.rmtree(cache, ignore_errors=True)
    print(json.dumps({k: res.get(k) for k in ("cold_start_s", "warm_start_s", "peak_mb_after_model", "peak_mb_end", "latency_ms", "by_outcome", "by_path", "stage_ms_mean", "time_to_clarification_ms", "time_to_final_answer_ms")}, indent=1))


if __name__ == "__main__": main()

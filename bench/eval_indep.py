"""Evaluate a query planner on an independent question file (bench/indep_dev.tsv or bench/indep_final.tsv), under the target's limits (pin cores with tools_pin.py).

    python tools_pin.py 8,9 bench/eval_indep.py --system rules --set final
    python tools_pin.py 8,9 bench/eval_indep.py --system qwen --set dev
    python tools_pin.py 8,9 bench/eval_indep.py --system schema --model models/LFM2.5-350M-Q4_K_M.gguf --mode full --set dev --perturb 0

What is measured (per question, then per family with 95 % Wilson intervals):
  ir_exact       the predicted query equals the gold query as data (condition order ignored)
  outcome_ok     executing the prediction gives the same verified result as executing the gold query (status, value, first item, count). Rows whose gold uses an operation the engine cannot execute yet are scored by ir_exact.
  answered       the planner produced a query that validates (it did not refuse / fail)
  wrong_answer   the planner produced a validating query whose result differs from the gold result: the dangerous error ("confidently wrong")
  false_action   the gold is op=none (refuse) but the planner produced a query
  false_refusal  the gold is a query but the planner wrote op=none or nothing parseable
  latency        planner milliseconds per question (model + retrieval), tokens
--perturb N  evaluates under a RENAMED catalog (seed N): every signal, event and condition has a different identifier the model has never seen (schema transfer); gold queries are mapped, execution still uses the registry."""
import argparse, json, math, os, sqlite3, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW, gold_of
from training.evaluate import parts, same_gold
from vgpt import nl2ir
from vgpt.qc import QueryCompiler
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)


class Rules:
    name = "rules"
    def plan(self, row): t = time.perf_counter(); p = nl2ir.parse(row["text"], now=DEFAULT_NOW, tz=3600); return (p.query.to_text() if p.ok else "op=none"), {"ms": (time.perf_counter() - t) * 1000, "out_tokens": 0}


class Hybrid:
    """The production cascade: deterministic rules when they are confident (with the previous turn as follow-up state), otherwise the language-model planner; a model query that fails the grounding guard becomes op=none."""
    def __init__(self, fallback, guard=True): self.fb = fallback; self.guard = guard; self.name = f"hybrid(rules -> {fallback.name}{'+guard' if guard else ''})"
    def plan(self, row):
        t = time.perf_counter(); st = nl2ir.ConvState()
        if row["prev_ir"]:
            try: st.last = Query.from_text(row["prev_ir"])
            except IRError: pass
        p = nl2ir.parse(row["text"], now=DEFAULT_NOW, tz=3600, state=st); ms = (time.perf_counter() - t) * 1000
        if p.ok: return p.query.to_text(), {"ms": ms, "out_tokens": 0, "source": "rules"}
        row = dict(row, draft=(p.query.to_text() if (p.query is not None and getattr(p.query, "op", "")) else None)); ir, info = self.fb.plan(row); info["ms"] += ms; info["source"] = "model"
        if self.guard and ir != "op=none":
            from vgpt.ir_slm import grounding_issues
            try:
                if grounding_issues(Query.from_text(ir), row["text"]): ir = "op=none"; info["guarded"] = True
            except IRError: ir = "op=none"
        return ir, info


class QwenLora:
    name = "qwen0.5b+lora"
    def __init__(self, model, lora, threads=2):
        from vgpt.ir_slm import IrSlm
        self.m = IrSlm(model, lora_path=lora, threads=threads, mode="tuned"); self.m.warmup()
    def plan(self, row): q, info = self.m.compile(row["text"]); return (info["raw"] if q is not None or info["raw"] == "op=none" else "op=none"), {"ms": info["ms"], "out_tokens": info.get("out_tokens") or 0}


class Schema:
    def __init__(self, model, mode, view, threads=2, cache=None, k=8, examples=True, bank_k=0):
        from vgpt.schema_planner import SchemaPlanner
        self.p = SchemaPlanner(model, view=view, mode=mode, threads=threads, cache_dir=cache, k_signals=k, examples=examples, bank_k=bank_k); self.view = view; self.name = f"schema[{os.path.basename(model)}|{mode}|{view.name}|static_ex={examples}|bank_k={bank_k}]"
    def plan(self, row):
        prev = (row["prev_text"], row["prev_ir"]) if row["prev_text"] else None
        q, info = self.p.compile(row["text"], prev, draft=row.get("draft")); ir = info["ir"] if (q is not None or info["ir"] == "op=none") else "op=none"
        return ir, {"ms": info["ms"], "out_tokens": info["out_tokens"], "prompt_tokens": info["prompt_tokens"], "raw": info["raw"]}


def outcome(qc, text, cache):
    if text in cache: return cache[text]
    try: q = Query.from_text(text); v = validate(q, qc.profile)
    except IRError: cache[text] = ("unparsed", None); return cache[text]
    if v.codes(): cache[text] = ("invalid", None); return cache[text]
    try: res = qc.execute(q); cache[text] = (res.status, gold_of(res))
    except Exception: cache[text] = ("invalid", None)
    return cache[text]


def run(system, rows, qc, progress=False):
    out = []; cache = {}
    for i, r in enumerate(rows):
        pred, info = system.plan(r); gold = r["ir"]; rec = {"id": r["id"], "family": r["family"], "text": r["text"], "gold": gold, "pred": pred, "ms": info["ms"], "tok": info.get("out_tokens", 0), "ptok": info.get("prompt_tokens", 0)}
        gnone = gold == "op=none"; pnone = pred == "op=none"
        try: pq = Query.from_text(pred) if not pnone else None
        except IRError: pq = None; pnone = True
        rec["answered"] = not pnone and pq is not None and not validate(pq, qc.profile).codes()
        if gnone: rec["ir_exact"] = pnone; rec["outcome_ok"] = pnone; rec["false_action"] = not pnone; rec["false_refusal"] = False; rec["wrong_answer"] = False
        else:
            gq = Query.from_text(gold); rec["ir_exact"] = bool(pq is not None and parts(pq)["full"] == parts(gq)["full"]); gs, gg = outcome(qc, gold, cache)
            if pnone or pq is None: rec["outcome_ok"] = False; rec["false_refusal"] = True; rec["false_action"] = False; rec["wrong_answer"] = False
            else:
                ps, pg = outcome(qc, pred, cache); rec["false_refusal"] = False; rec["false_action"] = False
                rec["outcome_ok"] = rec["ir_exact"] if gs in ("invalid", "unparsed") else (ps == gs and (gg is None or same_gold(pg, gg)))        # gold engine cannot run this operation yet: fall back to exact match
                rec["wrong_answer"] = rec["answered"] and not rec["outcome_ok"]
        out.append(rec)
        if progress and (i + 1) % 25 == 0: print(f"  {i + 1}/{len(rows)}", flush=True)
    return out


def summarize(rows):
    def rate(key, subset=None):
        s = subset if subset is not None else rows; k = sum(1 for r in s if r[key]); lo, hi = wilson(k, len(s)); return {"k": k, "n": len(s), "rate": round(k / max(len(s), 1), 4), "ci95": [round(lo, 3), round(hi, 3)]}
    answerable = [r for r in rows if r["gold"] != "op=none"]; refuse = [r for r in rows if r["gold"] == "op=none"]; fams = {}
    for r in rows: fams.setdefault(r["family"], []).append(r)
    lat = sorted(r["ms"] for r in rows); answered = [r for r in answerable if r["answered"]]
    return {"n": len(rows), "outcome_ok": rate("outcome_ok"), "ir_exact": rate("ir_exact"), "outcome_ok_answerable": rate("outcome_ok", answerable), "ir_exact_answerable": rate("ir_exact", answerable),
            "wrong_answer_among_answered": {"k": sum(1 for r in answered if r["wrong_answer"]), "n": len(answered), "rate": round(sum(1 for r in answered if r["wrong_answer"]) / max(len(answered), 1), 4)},
            "false_action_on_refusal_set": rate("false_action", refuse), "false_refusal_on_answerable": rate("false_refusal", answerable),
            "by_family": {f: {"n": len(v), "outcome_ok": round(sum(1 for r in v if r["outcome_ok"]) / len(v), 3), "ir_exact": round(sum(1 for r in v if r["ir_exact"]) / len(v), 3), "ci95_outcome": [round(x, 3) for x in wilson(sum(1 for r in v if r["outcome_ok"]), len(v))]} for f, v in sorted(fams.items())},
            "latency_ms": {"mean": round(statistics.mean(lat), 1), "p50": round(lat[len(lat) // 2], 1), "p95": round(lat[int(0.95 * (len(lat) - 1))], 1), "max": round(lat[-1], 1)}, "out_tokens_mean": round(statistics.mean(r["tok"] for r in rows), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["rules", "qwen", "schema", "hybrid"], required=True); ap.add_argument("--set", default="dev", help="dev | final | path"); ap.add_argument("--model"); ap.add_argument("--lora", default=os.path.join(HERE, "models", "ir-lora-qwen25-05b.gguf"))
    ap.add_argument("--mode", default="full", choices=["full", "slice"]); ap.add_argument("--perturb", type=int, default=-1, help="seed of a renamed catalog (schema transfer); -1 = the registry's own names"); ap.add_argument("--drop", default="", help="comma list of registry ids withheld from the schema")
    ap.add_argument("--fallback", default="qwen", choices=["qwen", "schema"]); ap.add_argument("--no-guard", action="store_true"); ap.add_argument("--max", type=int, default=0); ap.add_argument("--families", default=""); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); ap.add_argument("--out"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cache", default=os.path.join(HERE, "results", "_cache")); ap.add_argument("--no-examples", action="store_true"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--bank", type=int, default=0, help="retrieved catalog-generated examples per question"); ap.add_argument("--progress", action="store_true"); a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set); rows = load(path)
    if a.families: rows = [r for r in rows if r["family"] in a.families.split(",")]
    if a.max: rows = rows[: a.max]
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); qc = QueryCompiler(conn, now=DEFAULT_NOW)
    if a.system == "rules": sysm = Rules()
    elif a.system == "hybrid" and a.fallback == "qwen": sysm = Hybrid(QwenLora(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), a.lora, a.threads), not a.no_guard)
    elif a.system == "qwen": sysm = QwenLora(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), a.lora, a.threads)
    else:
        a_sys = a.system; from vgpt.schema_view import SchemaView, renamed_view
        view = renamed_view(a.perturb, drop=[x for x in a.drop.split(",") if x]) if a.perturb >= 0 else SchemaView(drop=[x for x in a.drop.split(",") if x])
        if a.perturb >= 0:                                                                                  # show the model gold examples under the renamed catalog: rows keep registry ids for execution, the planner works in display names
            pass
        sysm = Schema(a.model, a.mode, view, a.threads, a.cache, a.k, not a.no_examples, a.bank)
        if a_sys == "hybrid": sysm = Hybrid(sysm, not a.no_guard)
        if a.perturb >= 0:
            for r in rows:
                r["prev_ir_display"] = view.to_display(r["prev_ir"]) if r["prev_ir"] else None
    t0 = time.time(); res = run(sysm, rows, qc, a.progress); summ = summarize(res); summ["system"] = sysm.name; summ["set"] = os.path.basename(path); summ["wall_s"] = round(time.time() - t0)
    try:
        import psutil
        mi = psutil.Process().memory_info(); summ["peak_rss_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1e6)
    except Exception: pass
    if hasattr(sysm, "fb") and hasattr(sysm.fb, "p"): sysm = sysm.fb
    if hasattr(sysm, "p"): summ["prefix"] = {"tokens": sysm.p.be.prefix_tokens, "seconds": round(sysm.p.prefix_s, 2), "from_disk": sysm.p.be.prefix_from_disk}
    print(json.dumps({k: v for k, v in summ.items() if k != "by_family"}, indent=1)); print("family  n  outcome  ir_exact  (95% CI of outcome)")
    for f, v in summ["by_family"].items(): print(f"  {f:5s} {v['n']:3d}  {v['outcome_ok']:.2f}  {v['ir_exact']:.2f}  {v['ci95_outcome']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump({"summary": summ, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()

"""Evaluation harness v2 (supersedes bench/eval_indep.py for any new claim; the old numbers are historical and used a weaker comparison).

What is different, and why
  * OUTCOME EQUALITY IS OPERATION-SPECIFIC AND MULTI-DATABASE. A planned query counts as correct only if it produces the same signature as the gold query on EVERY vehicle database in `data/` (five vehicles with different signal sets, weather and
    stories): same status, operation, signal, unit, resolved time window, value, and the identity (timestamp, value) of every listed event / ranked trip / explained factor. A coincidentally equal number on one vehicle no longer passes.
  * NOTHING IS CREDITED BY AN UNSUPPORTED RESPONSE. The product's event-explanation engine is attached (the QueryCompiler is built through Brain, as in the product). A gold query that still cannot execute is scored by exact plan match and
    reported under scored_by='plan'.
  * ERRORS ARE CLASSIFIED BY WHICH FIELD IS WRONG (operation / signal / statistic / period / conditions / event / group-selection / unit), so interpretation errors are visible separately from execution and evidence errors.
  * FOLLOW-UPS: --history gold uses the correct previous query (an upper bound); --history own uses the system's OWN interpretation of the previous turn (what really happens in a conversation).
  * TWO KINDS OF SYSTEM, never mixed: PLANNERS (rules | qwen-lora | schema) see the same question, the same context and the same optional draft; the COMPLETE SYSTEM (cascade: rules, then a model, then the grounding guard) is a separate run.
  * RISK-COVERAGE: answered coverage, wrong-among-answered with exact counts and abstention are reported together; a lower error rate that comes with lower coverage is shown as such.
    python tools_pin.py 8,9 bench/eval_v2.py --system schema --model models/LFM2.5-350M-Q4_K_M.gguf --set dev --bank 4 --draft rules --history own --out results/v2_x.json"""
import argparse, json, math, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import nl2ir
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DBS = [("gen_courier_ext", DEFAULT_NOW), ("gen_demo", None), ("gen_cold", None), ("gen_roadwarrior", None), ("gen_courier", None)]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


from bench.evalcore import Engines as _Engines, FixtureError, parts, error_fields, score, same, signature


def Engines(names=None):
    return _Engines([d for d in DBS if not names or d[0] in names])


# ------------------------------------------------------------------------------------------------ planners (same inputs for all: question, previous turn, optional draft)
class RulesPlanner:
    name = "rules"

    def plan(self, text, prev=None, draft=None):
        t = time.perf_counter(); st = nl2ir.ConvState()
        if prev and prev[1]:
            try: st.last = Query.from_text(prev[1])
            except IRError: pass
        p = nl2ir.parse(text, now=DEFAULT_NOW, tz=3600, state=st)
        return (p.query.to_text() if p.ok else "op=none"), {"ms": (time.perf_counter() - t) * 1000, "tok": 0, "ok": bool(p.ok), "partial": p.query.to_text() if (p.query is not None and p.query.op) else None}


class QwenLoraPlanner:
    """The earlier trained adapter: HISTORICAL BASELINE. Its interface takes the question only; context and draft are ignored (said so in the output)."""
    name = "qwen0.5b+lora(question only)"

    def __init__(self, threads=2):
        from vgpt.ir_slm import IrSlm
        self.m = IrSlm(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), lora_path=os.path.join(HERE, "models", "ir-lora-qwen25-05b.gguf"), threads=threads, mode="tuned"); self.m.warmup()

    def plan(self, text, prev=None, draft=None):
        q, info = self.m.compile(text); return (info["raw"] if q is not None or info["raw"] == "op=none" else "op=none"), {"ms": info["ms"], "tok": info.get("out_tokens") or 0}


class SchemaModelPlanner:
    def __init__(self, model, mode, bank, use_draft, threads=2, cache=None, examples=True, view=None):
        from vgpt.schema_planner import SchemaPlanner
        from vgpt.schema_view import SchemaView
        self.p = SchemaPlanner(model, view=view or SchemaView(), mode=mode, threads=threads, cache_dir=cache, bank_k=bank, examples=examples); self.use_draft = use_draft
        self.name = f"schema[{os.path.basename(model)}|{mode}|bank={bank}|draft={'rules' if use_draft else 'none'}|static_examples={examples}]"

    def plan(self, text, prev=None, draft=None):
        if self.use_draft and draft is None:                                                                  # planner-only run: the draft comes from the same rules parse the cascade would have made
            draft = RulesPlanner().plan(text, prev)[1].get("partial")
        q, info = self.p.compile(text, prev if prev and prev[1] else None, draft=draft if self.use_draft else None); ir = info["ir"] if (q is not None or info["ir"] == "op=none") else "op=none"
        return ir, {"ms": info["ms"], "tok": info["out_tokens"], "ptok": info["prompt_tokens"], "prefix_tokens": info["prefix_tokens"]}


class Cascade:
    """THE COMPLETE SYSTEM: rules when they are confident, else the model (given the rules' partial parse as a draft if the model is a schema planner), then a guard on the model's query.
    guard: none | numbers (the older number/signal/condition grounding check) | interp (the interpretation contract: every field of the query must come from the question, the previous turn or a disclosed default) | both."""
    def __init__(self, model_planner, guard="numbers", complete=False, rules_guard=False):
        self.complete = complete; self.rules_guard = rules_guard
        self.rules = RulesPlanner(); self.m = model_planner; self.guard = {True: "numbers", False: "none"}.get(guard, guard); self.name = f"cascade(rules -> {model_planner.name}{'' if self.guard == 'none' else '+guard:' + self.guard}{'+complete' if complete else ''}{'+rulesguard' if rules_guard else ''})"

    def plan(self, text, prev=None, draft=None):
        ir, info = self.rules.plan(text, prev)
        pq0 = None
        if prev and prev[1]:
            try: pq0 = Query.from_text(prev[1])
            except IRError: pq0 = None
        if info["ok"] and self.rules_guard:                                                       # the semantic check applies to the rules' own answers too
            from vgpt import interpretation
            try:
                if interpretation.interpret(text, Query.from_text(ir), pq0, source="rules", now=DEFAULT_NOW, tz=3600).ungrounded(): info["ok"] = False; info["rules_flagged"] = True
            except IRError: pass
        if info["ok"]: info["source"] = "rules"; return ir, info
        ir2, i2 = self.m.plan(text, prev, draft=info.get("partial")); i2["ms"] += info["ms"]; i2["source"] = "model"
        if info.get("rules_flagged"): i2["rules_flagged"] = True
        if self.complete and ir2 != "op=none":
            from vgpt import completion
            try:
                dq = Query.from_text(info["partial"]) if info.get("partial") else None
                cq, notes = completion.complete(text, Query.from_text(ir2), dq, pq0, now=DEFAULT_NOW, tz=3600)
                if notes: ir2 = cq.to_text(); i2["completed"] = notes
            except IRError: pass
        if self.guard != "none" and ir2 != "op=none":
            try:
                q = Query.from_text(ir2)
                if self.guard in ("numbers", "both"):
                    from vgpt.ir_slm import grounding_issues
                    if grounding_issues(q, text): ir2 = "op=none"; i2["guarded"] = "numbers"
                if ir2 != "op=none" and self.guard in ("interp", "both"):
                    from vgpt import interpretation
                    pq = Query.from_text(prev[1]) if prev and prev[1] and prev[1] != "op=none" else None
                    iss = interpretation.interpret(text, q, pq, source="model", now=DEFAULT_NOW, tz=3600).issues()
                    if iss: ir2 = "op=none"; i2["guarded"] = "interp"; i2["issues"] = iss
            except IRError: ir2 = "op=none"
        return ir2, i2


class PolicyCascade:
    """The clarification-first system: rules -> (model proposal) -> vgpt.policy.decide. The model never answers by itself; a proposal is accepted only when it passes the support and coverage checks, otherwise its reading is offered."""
    def __init__(self, model_planner=None, check_rules=True):
        self.m = model_planner; self.check_rules = check_rules
        self.name = "policy(rules" + (f" -> {model_planner.name}" if model_planner is not None else "") + ("" if check_rules else ", unchecked rules") + ")"

    def plan(self, text, prev=None, draft=None):
        from vgpt import policy
        t0 = time.perf_counter(); st = nl2ir.ConvState(); pq0 = None
        if prev and prev[1]:
            try: pq0 = Query.from_text(prev[1]); st.last = pq0
            except IRError: pq0 = None
        pr = nl2ir.parse(text, now=DEFAULT_NOW, tz=3600, state=st); mq = None; info = {"ms": 0.0, "tok": 0}
        if not pr.ok and not pr.clarify and self.m is not None:
            ir2, i2 = self.m.plan(text, prev, draft=pr.query.to_text() if (pr.query is not None and pr.query.op) else None); info.update(tok=i2.get("tok", 0), ptok=i2.get("ptok"))
            if ir2 != "op=none":
                try: mq = Query.from_text(ir2)
                except IRError: mq = None
        d = policy.decide(text, pr, mq, pq0, None, DEFAULT_NOW, 3600, self.check_rules); info["ms"] = (time.perf_counter() - t0) * 1000; info["source"] = d.source; info["decision"] = d.kind
        if d.kind == "answer": return d.query.to_text(), info
        if d.kind == "clarify" and d.candidates: info["clarify"] = [c.to_text() for c in d.candidates]; info["question"] = d.question; return "op=none", info
        return "op=none", info


class SelectPolicy:
    """Rules, then (when they cannot answer) deterministic candidate generation + frozen-model SELECTION, then the support/coverage checks (vgpt.policy.decide_by_selection)."""
    def __init__(self, selector, check_rules=True): self.sel = selector; self.check_rules = check_rules; self.name = f"policy-select(rules -> {selector.name})"

    def plan(self, text, prev=None, draft=None):
        from vgpt import policy
        t0 = time.perf_counter(); st = nl2ir.ConvState(); pq0 = None
        if prev and prev[1]:
            try: pq0 = Query.from_text(prev[1]); st.last = pq0
            except IRError: pq0 = None
        pr = nl2ir.parse(text, now=DEFAULT_NOW, tz=3600, state=st)
        d, inf = policy.decide_by_selection(text, pr, self.sel, pq0, None, DEFAULT_NOW, 3600, self.check_rules)
        info = {"ms": (time.perf_counter() - t0) * 1000, "tok": 0, "ptok": inf.get("model_ptok"), "source": d.source, "decision": d.kind, "candidates": inf.get("candidates"), "selected": inf.get("selected"), "model_ms": inf.get("model_ms", 0.0)}
        if d.kind == "answer": return d.query.to_text(), info
        if d.kind == "clarify" and d.candidates: info["clarify"] = [c.to_text() for c in d.candidates]; info["question"] = d.question; return "op=none", info
        return "op=none", info


# ------------------------------------------------------------------------------------------------ run and summarise
def run(planner, rows, eng, history="gold", progress=False):
    out = []; own = {}
    for i, r in enumerate(rows):
        prev = None
        if r["prev_text"]:
            pir = r["prev_ir"]
            if history == "own":
                if r["prev_text"] not in own: own[r["prev_text"]] = planner.plan(r["prev_text"], None)[0]
                pir = own[r["prev_text"]]
            prev = (r["prev_text"], pir if pir != "op=none" else None)
        pred, info = planner.plan(r["text"], prev, draft=None); gold = r["ir"]
        rec = {"id": r["id"], "family": r["family"], "text": r["text"], "gold": gold, "pred": pred, "prev_text": r["prev_text"], "ms": info["ms"], "tok": info.get("tok", 0), "ptok": info.get("ptok"), "source": info.get("source", ""), "history": history if r["prev_text"] else ""}
        rec.update(score(eng, gold, pred, info.get("clarify"))); rec["answered"] = rec["kind"] == "answer"
        if r.get("alts") and not rec["correct"] and rec["kind"] in ("answer", "explained"):                      # a reviewer-approved alternative interpretation is as correct as the primary one
            for alt in r["alts"]:
                if score(eng, alt, pred)["correct"]: rec.update(correct=True, wrong_answer=False, wrong_non_answer=False, fields=[], accepted_alternative=alt); break
        if rec["gold_kind"] in ("answer", "explained", "invalid"): rec["single_db_match"] = rec["correct"]
        if info.get("candidates") is not None and gold != "op=none":                                       # generator quality, measured apart from the selector
            rec["cand_n"] = len(info["candidates"]); rec["cand_available"] = any(score(eng, gold, c)["correct"] for c in info["candidates"]); rec["selected"] = info.get("selected"); rec["model_ms"] = info.get("model_ms", 0.0)
        out.append(rec)
        if progress and (i + 1) % 25 == 0: print(f"  {i + 1}/{len(rows)}", flush=True)
    return out


def summarize(rows):
    def rate(sub, key):
        k = sum(1 for r in sub if r[key]); lo, hi = wilson(k, len(sub)); return {"k": k, "n": len(sub), "rate": round(k / max(len(sub), 1), 4), "ci95": [round(lo, 3), round(hi, 3)]}
    ans = [r for r in rows if r["gold"] != "op=none"]; ref = [r for r in rows if r["gold"] == "op=none"]; answered = [r for r in ans if r["answered"]]; fams = {}
    for r in rows: fams.setdefault(r["family"], []).append(r)
    lat = sorted(r["ms"] for r in rows); fields = {}
    for r in ans:
        for f in r.get("fields", []): fields[f] = fields.get(f, 0) + 1
    coincid = sum(1 for r in ans if r.get("single_db_match") and not r["correct"] and r["scored_by"] == "execution")
    outcomes = {k: sum(1 for r in rows if r.get("kind") == k) for k in ("answer", "explained", "refusal", "invalid", "clarify")}
    cl = [r for r in rows if r.get("kind") == "clarify"]; clar = {"asked": len(cl), "contains_intended_reading": sum(1 for r in cl if r.get("clarify_hit")), "on_answerable": sum(1 for r in cl if r["gold"] != "op=none"), "unneeded": sum(1 for r in cl if r.get("unneeded_clarification")),
            "mean_readings_offered": round(statistics.mean([r["clarify_n"] for r in cl]), 2) if cl else None, "correct_after_one_extra_turn": sum(1 for r in rows if r.get("correct")) + sum(1 for r in cl if r.get("resolved_correct")),
            "candidate_questions": sum(1 for r in rows if r.get("cand_n") is not None), "candidate_available": sum(1 for r in rows if r.get("cand_available")), "model_selected_correct": sum(1 for r in rows if r.get("cand_available") and r.get("selected") is not None and r.get("kind") == "answer" and r.get("correct")), "model_ms_mean": round(statistics.mean([r["model_ms"] for r in rows if r.get("cand_n") and r.get("model_ms")]), 1) if any(r.get("cand_n") and r.get("model_ms") for r in rows) else None}
    return {"n": len(rows), "outcomes": outcomes, "clarification": clar, "plan_exact": rate(ans, "plan_exact"), "wrong_non_answers": sum(1 for r in rows if r.get("wrong_non_answer")), "correct_all": rate(rows, "correct"), "correct_answerable": rate(ans, "correct"), "answered_coverage": {"k": len(answered), "n": len(ans), "rate": round(len(answered) / max(len(ans), 1), 4)},
            "wrong_among_answered": {"k": sum(1 for r in answered if r["wrong_answer"]), "n": len(answered), "rate": round(sum(1 for r in answered if r["wrong_answer"]) / max(len(answered), 1), 4)},
            "false_action_on_refusal_set": rate(ref, "false_action"), "false_refusal_on_answerable": rate(ans, "false_refusal"), "scored_by": {k: sum(1 for r in rows if r.get("scored_by") == k) for k in ("execution", "plan", "refusal")},
            "single_database_match_that_failed_on_other_vehicles": coincid, "wrong_fields": dict(sorted(fields.items(), key=lambda x: -x[1])),
            "by_family": {f: {"n": len(v), "correct": round(sum(1 for r in v if r["correct"]) / len(v), 3), "ci95": [round(x, 3) for x in wilson(sum(1 for r in v if r["correct"]), len(v))]} for f, v in sorted(fams.items())},
            "latency_ms": {"mean": round(statistics.mean(lat), 1), "p50": round(lat[len(lat) // 2], 1), "p95": round(lat[int(0.95 * (len(lat) - 1))], 1)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["rules", "qwen", "schema", "cascade-qwen", "cascade-schema", "policy-rules", "policy-schema", "policy-qwen", "policy-select"], required=True); ap.add_argument("--set", default="dev"); ap.add_argument("--model"); ap.add_argument("--mode", default="full", choices=["full", "slice"])
    ap.add_argument("--bank", type=int, default=0); ap.add_argument("--draft", default="none", choices=["none", "rules"]); ap.add_argument("--history", default="gold", choices=["gold", "own"]); ap.add_argument("--no-examples", action="store_true"); ap.add_argument("--guard", default="numbers", choices=["none", "numbers", "interp", "both"]); ap.add_argument("--complete", action="store_true"); ap.add_argument("--no-rules-check", action="store_true"); ap.add_argument("--rules-guard", action="store_true")
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--families", default=""); ap.add_argument("--out"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--cache", default=os.path.join(HERE, "results", "_cache")); ap.add_argument("--progress", action="store_true"); a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set); rows = load(path)
    if a.families: rows = [r for r in rows if r["family"] in a.families.split(",")]
    if a.max: rows = rows[: a.max]
    eng = Engines()
    if a.system == "rules": pl = RulesPlanner()
    elif a.system == "policy-rules": pl = PolicyCascade(None, not a.no_rules_check)
    elif a.system == "policy-select":
        from vgpt.candidate_planner import CandidateSelector
        pl = SelectPolicy(CandidateSelector(a.model, threads=a.threads, cache_dir=a.cache), not a.no_rules_check)
    elif a.system == "qwen": pl = QwenLoraPlanner(a.threads)
    else:
        m = SchemaModelPlanner(a.model, a.mode, a.bank, a.draft == "rules", a.threads, a.cache, not a.no_examples) if a.system in ("schema", "cascade-schema", "policy-schema") else QwenLoraPlanner(a.threads)
        pl = m if a.system == "schema" else PolicyCascade(m, not a.no_rules_check) if a.system.startswith("policy") else Cascade(m, a.guard, a.complete, a.rules_guard)
    t0 = time.time(); res = run(pl, rows, eng, a.history, a.progress); summ = summarize(res); summ.update({"system": pl.name, "set": os.path.basename(path), "history": a.history, "databases": [n for n, _ in eng.qcs], "wall_s": round(time.time() - t0)})
    try:
        import psutil
        mi = psutil.Process().memory_info(); summ["peak_rss_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1e6)
    except Exception: pass
    print(json.dumps({k: v for k, v in summ.items() if k != "by_family"}, indent=1)); print("family  n  correct  (95% CI)")
    for f, v in summ["by_family"].items(): print(f"  {f:5s} {v['n']:3d}  {v['correct']:.2f}  {v['ci95']}")
    if a.out: os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump({"summary": summ, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()

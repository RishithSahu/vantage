"""End-to-end evaluation of the PRODUCT (vgpt.modelbrain.ModelBrain: retrieval + one frozen-model plan + structural validation + the typed pipeline), scored with the same evaluator as the baselines (bench/eval_v2.py, multi-database result signatures, clarification scoring).
    python tools_pin.py 8,9 bench/eval_model.py --set dev --out results/m1_dev.json [--model models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf] [--limit N] [--ids d001,d002]
Follow-ups use the SYSTEM'S OWN conversation state: the previous question of the row is asked first and whatever the system resolved is what the follow-up continues (no gold history is supplied).
Every question that is not scored correct is attributed to EXACTLY ONE earliest failing stage, so effort goes where the failure is:
    retrieval        a capability the gold reading needs was not retrieved (all required capabilities, not just one of them)
    prompt           retrieved but not shown to the model (not among the cards of the rendered message)
    grammar          the gold reading cannot be written in the plan language (not representable / not valid structurally)
    model            the model's plan differs from the gold reading (wrong operation, signal, period, condition, state edit ...)
    compiler         the model's plan equalled the gold reading but the compiled query differs
    execution        the compiled query equals the gold reading but the answer / rendering is scored wrong
Timings are reported in separate stages (retrieval + embedding, per-message render, state restore, prompt evaluation, generation, validation + execution, total). The model sees no gold data of any kind."""
import argparse, collections, hashlib, json, os, statistics, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.eval_v2 import Engines, run, summarize
from bench.indep_io import load
from bench.ledger import source_revision
from training.build_dataset import DEFAULT_NOW
from vgpt import plancompile as PC
from vgpt.modelbrain import ModelBrain
from vgpt.modelplan import Planner
from vgpt.query_ir import Query
from vgpt.toolspec import plan_to_call

STAGES = ("retrieval", "prompt", "grammar", "grounding", "model", "compiler", "execution")


def required_ids(q):
    ids = []
    for x in (q.target, q.target2, q.event.type if q.event else None):
        if x: ids.append(x)
    for c in q.scope.conditions:
        if c.signal: ids.append(c.signal)
    return list(dict.fromkeys(ids))


def representable(q, reg):
    """None if the plan language can express the gold reading, else the reason."""
    if q.event is not None and q.event.selector not in ("last", "first", "all"): return f"event selector {q.event.selector}"
    for t in (q.scope.time, q.scope_b.time if q.scope_b is not None else None):
        if t is not None and t.type not in ("current", "relative", "calendar", "date", "all"): return f"period form {t.type}"
    p = PC.query_to_plan(q)
    if p is None or not plan_to_call(p): return "no function"
    iss = PC.validate(p, reg)
    return "; ".join(m for _, m in iss) if iss else None


def norm_plan(p):
    out = {}
    for k in PC.FN_FIELDS.get(p.get("fn"), ()):
        v = p.get(k)
        if v in (None, "", [], {}): continue
        out[k] = sorted(map(repr, map(PC._norm_cond, v))) if k == "cond" else v
    out["fn"] = p.get("fn")
    return out


class ModelPlanner:
    def __init__(self, db, planner, repair=False, expand=False):
        self.b = ModelBrain(db, planner=planner, now=DEFAULT_NOW, repair=repair, expand=expand); self.P = planner; self.name = "model-plan (" + os.path.basename(planner.model_path) + ")"; self.records = []; self.timings = []; self.outcomes = collections.Counter()

    def plan(self, text, prev=None, draft=None):
        b = self.b; b.reset()
        if prev and prev[0]: b.ask(prev[0])                                                        # own history: whatever the system made of the previous turn
        state_plan = b.state.prev_plan; t0 = time.perf_counter(); r = b.ask(text); ms = (time.perf_counter() - t0) * 1000; pr = b.last_plan_result; out = b.last_outcome; lq = b.last_qc; oc = b.outcome; self.outcomes[oc] += 1; self.timings.append(dict(b.timing, outcome=oc))
        info = {"ms": ms, "tok": (pr.timing.get("out_tokens") if pr else 0) or 0, "ptok": pr.timing.get("prompt_tokens") if pr else None, "source": f"outcome:{oc}"}
        self.records.append({"text": text, "prev_text": prev[0] if prev else None, "call": pr.raw if pr else None, "error": pr.error if pr else "exception", "outcome": oc, "hit_ids": [h.id for h in pr.hits] if pr else [], "card_ids": pr.card_ids if pr else [],
                             "plan": (out.plan if out is not None and out.kind in ("query", "tool") else None), "kind": out.kind if out is not None else None, "query": lq.query.to_text() if (lq is not None and getattr(lq, "query", None) is not None) else None, "timing": dict(b.timing), "repair": b.repair, "expansion": b.expansion, "support_diag": b.support_diag, "fn": pr.fn if pr else None, "model_plan": pr.plan if pr else None, "state_plan": state_plan, "ungrounded": (r.get("facts") or {}).get("ungrounded"), "proposed": (r.get("facts") or {}).get("proposed")})
        if oc == "ambiguous": info["clarify"] = ["op=none"]; info["confirm"] = False; return "op=none", info                       # a specific question
        if oc in ("failed_plan", "invalid_plan"): info["source"] = "failed"; return "op=none", info
        if r.get("path") == "qc" and lq is not None and getattr(lq, "query", None) is not None: info["status"] = lq.status; return lq.query.to_text(), info
        return "op=none", info


def classify(rec, row, planner_rec, reg):
    """Earliest failing stage of a question that was not scored correct (None if correct)."""
    if rec["correct"]: return None
    gold = row["ir"]
    if gold == "op=none": return "model"                                                                 # a question that should have been refused / was not answerable
    try: gq = Query.from_text(gold)
    except Exception: return "model"
    need = required_ids(gq); hit = set(planner_rec["hit_ids"]); cards = set(planner_rec["card_ids"])
    if planner_rec.get("proposed") and planner_rec.get("ungrounded"):                                       # the model's own plan was right but a quotation could not be grounded
        try:
            if norm_plan(PC.merge(None, dict(planner_rec["proposed"]))) == norm_plan(PC.query_to_plan(gq)): return "grounding"
        except Exception: pass
    prev_ids = []
    if need and not all((i in hit) or (row.get("prev_text") and False) for i in need):
        if not (row.get("prev_text") and all(i in cards for i in need)): return "retrieval"                # a follow-up may rely on the ids carried from the previous turn
    if need and not all(i in cards for i in need): return "prompt"
    why = representable(gq, reg)
    if why: return "grammar"
    gp = PC.query_to_plan(gq)
    mp = planner_rec["plan"]
    if mp is None or norm_plan(mp) != norm_plan(gp): return "model"
    if planner_rec["query"] is None or planner_rec["query"] != gold: return "compiler"
    return "execution"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--set", default="dev"); ap.add_argument("--out"); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db"))
    ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")); ap.add_argument("--n-ctx", default="6144", help="an integer or auto (sized to the measured prompt)"); ap.add_argument("--repair", action="store_true", help="ablation: ONE repair pass when a quotation cannot be grounded"); ap.add_argument("--expand", action="store_true", help="ablation: ONE bounded retrieval expansion on an unresolved target"); ap.add_argument("--schema", default="rich", choices=["rich", "compact"]); ap.add_argument("--shots", default="all", choices=["all", "core"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ids", default=""); ap.add_argument("--stride", type=int, default=0, help="every k-th question (a quick diagnostic subset)")
    a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set)
    if os.path.basename(path).startswith("acceptance"): raise SystemExit("the acceptance set is sealed and is not read during development")
    rows = load(path)
    if a.ids: rows = [r for r in rows if r["id"] in a.ids.split(",")]
    if a.stride: rows = rows[:: a.stride]
    if a.limit: rows = rows[: a.limit]
    import sqlite3
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); P = Planner(conn, model_path=a.model, n_ctx=(a.n_ctx if a.n_ctx == "auto" else int(a.n_ctx)), schema_level=a.schema, core_shots=(a.shots == "core"))
    eng = Engines(); pl = ModelPlanner(a.db, P, repair=a.repair, expand=a.expand)
    t0 = time.time(); res = run(pl, rows, eng, "own", progress=True); summ = summarize(res)
    # the record of a question is the one produced when IT was asked (a follow-up row asks its previous question first): take the last record per text
    last = {}
    for r in pl.records: last[(r["text"], r["prev_text"])] = r
    stages = collections.Counter(); per = []
    for rec, row in zip(res, rows):
        pr = last.get((row["text"], row["prev_text"]))
        st = classify(rec, row, pr, P.reg) if pr else "model"; rec["stage"] = st
        if st: stages[st] += 1
    # all-required-capability recall (retrieval only)
    need_rows = [(r, last.get((r["text"], r["prev_text"]))) for r in rows if r["ir"] != "op=none"]
    def req(r):
        try: return required_ids(Query.from_text(r["ir"]))
        except Exception: return []
    rec_all = [all(i in set(p["hit_ids"]) for i in req(r)) for r, p in need_rows if p and req(r)]; rec_any = [any(i in set(p["hit_ids"]) for i in req(r)) for r, p in need_rows if p and req(r)]
    ts = [t for t in pl.timings if t.get("generate_ms") is not None]; mean = lambda k: round(statistics.mean([t.get(k) or 0 for t in ts]), 1) if ts else None; q95 = lambda k: sorted([t.get(k) or 0 for t in ts])[int(0.95 * (len(ts) - 1))] if ts else None
    summ.update({"source_rev": source_revision(), "prompt_id": hashlib.sha256((P.system + json.dumps(P.shots) + json.dumps(P.tools, default=str)).encode()).hexdigest()[:10], "system": pl.name, "dialect": P.dialect.name, "n_ctx": P.n_ctx if hasattr(P, "n_ctx") else a.n_ctx, "repair_pass": a.repair, "expansion": a.expand, "schema": a.schema, "shots": a.shots, "model_sha256_16": P.lm.identity, "set": os.path.basename(path), "history": "own", "wall_s": round(time.time() - t0), "outcomes_model_route": dict(pl.outcomes), "earliest_failing_stage": {s: stages.get(s, 0) for s in STAGES},
                 "retrieval_recall": {"all_required": [sum(rec_all), len(rec_all)], "at_least_one_required": [sum(rec_any), len(rec_any)]},
                 "timing_ms": {k: {"mean": mean(k), "p95": q95(k)} for k in ("retrieve_ms", "render_ms", "state_ms", "prefill_ms", "decode_ms", "grammar_ms", "generate_ms", "compile_ms", "execute_ms", "total_ms")},
                 "tokens": {"prompt_mean": mean("prompt_tokens"), "out_mean": mean("out_tokens")}, "load_s": {"model_and_prefix": round(P.model_load_s, 1), "retrieval": round(P.retrieval_load_s, 2)}, "prefix_tokens": P.lm.prefixes["plan"]["tokens"]})
    rp = [r for r in pl.records if r.get("repair")]
    summ["grounding"] = {"repair_passes": len(rp), "still_ungrounded": sum(1 for r in pl.records if r.get("ungrounded")), "expansions": sum(1 for r in pl.records if (r.get("expansion") or {}).get("added")), "expansion_replans": sum(1 for r in pl.records if (r.get("expansion") or {}).get("second_call"))}
    print(json.dumps({k: v for k, v in summ.items() if k not in ("by_family",)}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump({"summary": summ, "rows": res, "records": pl.records}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()

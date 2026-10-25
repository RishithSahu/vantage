"""Diagnostic: does the frozen model UNDERSTAND the question, or does it fail to EXPRESS its understanding in the requested format? Same questions, same capability cards, same instruction, same conversation context (for a follow-up the GOLD previous plan is supplied as the state - a diagnostic only, never in the product), different OUTPUT INTERFACE:
    --format toolcall   the model's own tool-call dialect (production format)
    --format frame      the compact semantic frame with explicit `none` for absent fields
    --decode grammar    constrained to the plan language (production)
    --decode free       no grammar: the model's own text, parsed leniently; text that does not parse is counted as malformed and is never executed
Each question is scored by compiling the model's plan (no grounding check here - this measures the plan, not the quotation) and comparing the resulting query with the gold query, and attributed to what differs: operation / target / period / conditions added / conditions missing / statistic / event / selection / unit / state update / malformed. A fixed slice (every k-th development question) keeps the runs short.
    python tools_pin.py 8,9 bench/diag_format.py --model PATH [--n-ctx N] --format toolcall|frame --decode grammar|free [--stride 2] --out results/diag_x.json"""
import argparse, collections, json, os, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import plancompile as PC
from vgpt.modelplan import Planner, State
from vgpt.query_ir import Query
from vgpt import toolspec as T

EQUIV = {"count_events": {"did_event_happen"}, "did_event_happen": {"count_events"}}
REFUSAL_FNS = ("decline", "ask_user", "unresolved", "say")        # kept for the lenient view; the strict view below accepts only `decline` for a refusal


def normalise(gq, cq):
    if gq.op == "count" and gq.event is not None and cq.event is not None: gq.event.selector = cq.event.selector = "all"
    if gq.op == "event_context" and gq.event is not None and cq.event is not None and not gq.event.before and not gq.event.after: gq.event.before = cq.event.before


def diff_fields(gq, cq):
    """Names of what differs between the gold query and the compiled one, as diagnostic categories."""
    gf, cf = PC.query_fields(gq), PC.query_fields(cq); cats = []
    if gq.op != cq.op: cats.append("operation")
    if gf.get("tgt") != cf.get("tgt") or gf.get("tgt2") != cf.get("tgt2"): cats.append("target")
    if gf.get("per") != cf.get("per") or gf.get("per2") != cf.get("per2"): cats.append("period")
    gc = sorted(map(PC._norm_cond, gf.get("cond", []))); cc = sorted(map(PC._norm_cond, cf.get("cond", [])))
    if gc != cc: cats.append("conditions_added" if (not gc and cc) else "conditions_missing" if (gc and not cc) else "conditions_wrong")
    if gf.get("grp") != cf.get("grp"): cats.append("groups")
    if gf.get("stat") != cf.get("stat"): cats.append("statistic")
    if gf.get("evt") != cf.get("evt") or (gq.op not in ("count",) and gf.get("ev") != cf.get("ev")) or gf.get("bef") != cf.get("bef") or gf.get("aft") != cf.get("aft"): cats.append("event")
    if (gf.get("ent") or {}) != (cf.get("ent") or {}): cats.append("selection")
    if gf.get("unit") != cf.get("unit"): cats.append("unit")
    return cats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")); ap.add_argument("--n-ctx", default="auto")
    ap.add_argument("--format", default="toolcall", choices=["toolcall", "frame", "intent", "xlam", "xlam_rev", "qwen"]); ap.add_argument("--decode", default="grammar", choices=["grammar", "free"]); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--evidence", default="none", choices=["none", "spans", "spans_ops"], help="a frozen pretrained extractor supplies spans / absences (and optionally ranked operations) as evidence in the prompt"); ap.add_argument("--extractor", default="fastino/gliner2-base-v1"); ap.add_argument("--schema", default="rich", choices=["rich", "compact"]); ap.add_argument("--shots", default="all", choices=["all", "core", "none"]); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--result-source", default="none", choices=["none", "oracle"], help="oracle: the annotated requested result is given to the planner (a CEILING measurement of the factored design, never a product path)"); ap.add_argument("--demos", type=int, default=0, help="k nearest generated demonstrations (vgpt/demos.py) shown as earlier chat turns"); ap.add_argument("--demo-cards", default="yes", choices=["yes", "no"]); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--set", default="dev"); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); ap.add_argument("--out"); a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set)
    if os.path.basename(path).startswith("acceptance"): raise SystemExit("the acceptance set is sealed")
    rows = load(path)[:: a.stride]
    if a.limit: rows = rows[: a.limit]
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    ext = None
    if a.evidence != "none":
        from vgpt.extractor import Extractor
        ext = Extractor(a.extractor)
    P = Planner(conn, model_path=a.model, threads=a.threads, extractor=ext, evidence_ops=(a.evidence == "spans_ops"), n_ctx=(a.n_ctx if a.n_ctx == "auto" else int(a.n_ctx)), dialect=(a.format if a.format in ("frame", "intent", "xlam", "xlam_rev", "qwen") else None), decode=a.decode, schema_level=a.schema, core_shots=({"core": True, "none": "none"}.get(a.shots, False)), demo_k=a.demos, demo_cards=(a.demo_cards == "yes"))
    from bench.diag_gliner import annotations
    ann = annotations() if os.path.exists(os.path.join(HERE, "bench", "slice_v1_annotations.tsv")) else {}
    res = []; cats = collections.Counter(); ok = 0; mal = 0; strict = 0; t0 = time.time()
    for i, r in enumerate(rows):
        st = State()
        if r["prev_ir"]:
            try: st.prev_plan = PC.query_to_plan(Query.from_text(r["prev_ir"]))
            except Exception: pass
        pr = P.plan(r["text"], st, DEFAULT_NOW, result=((ann.get(r["id"]) or {}).get("result") if a.result_source == "oracle" else None)); rec = {"id": r["id"], "text": r["text"], "gold": r["ir"], "raw": pr.raw, "error": pr.error, "ms": pr.timing.get("plan_ms")}
        refusal = r["ir"].startswith("op=none")
        if pr.plan is None: rec.update(ok=False, cats=["malformed"]); mal += 1
        elif refusal: good = pr.fn in REFUSAL_FNS; rec.update(ok=good, cats=[] if good else ["should_refuse"], fn=pr.fn)
        else:
            gq = Query.from_text(r["ir"]); turn = pr.plan.get("turn", "new")
            if pr.fn in REFUSAL_FNS: rec.update(ok=False, cats=["false_refusal"], fn=pr.fn)
            else:
                p = PC.merge(st.prev_plan if turn != "new" else None, pr.plan); o = PC.compile_plan(pr.plan, st.prev_plan, None, P.reg, DEFAULT_NOW, 0)
                if o.kind != "query": rec.update(ok=False, cats=["state_update" if (turn != "new" and not st.prev_plan) or o.kind == "invalid" and o.say else "invalid:" + o.kind], fn=pr.fn)
                else:
                    cq = o.query; gq2 = Query.from_text(r["ir"]); normalise(gq2, cq); cs = diff_fields(gq2, cq)
                    if cs and r["prev_ir"]: cs = [c for c in cs] + (["state_update"] if len(cs) == 1 and cs[0] in ("period", "conditions_missing", "conditions_wrong", "conditions_added", "target", "statistic") else [])
                    rec.update(ok=not cs, cats=cs, fn=pr.fn)
        want = (ann.get(r["id"]) or {}).get("result"); got = T.RESULT_OF_FN.get(pr.fn) if pr.plan is not None else None
        rec["result_want"], rec["result_got"] = want, got
        rec["strict_ok"] = bool(rec["ok"] and (want is None or (got == want if not refusal else pr.fn == "decline")))                 # strict: the requested RESULT must also be the annotated one (count / existence / list are not interchangeable); a refusal only counts as `decline`
        strict += rec["strict_ok"]
        ok += rec["ok"]; cats.update(rec["cats"]); res.append(rec)
        if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(rows)}  correct so far {ok}", flush=True)
    summ = {"model": os.path.basename(a.model), "format": a.format, "decode": a.decode, "schema": a.schema, "evidence": a.evidence, "shots": a.shots, "n": len(rows), "plan_correct": ok, "strict_correct": strict, "malformed": mal, "categories": dict(cats.most_common()), "wall_s": round(time.time() - t0), "dialect": P.dialect.name, "prefix_tokens": P.lm.prefixes["plan"]["tokens"],
            "mean_ms": round(sum(x["ms"] or 0 for x in res) / max(len(res), 1))}
    print(json.dumps(summ, indent=1))
    if a.out: os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump({"summary": summ, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()

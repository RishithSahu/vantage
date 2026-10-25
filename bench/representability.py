"""Offline representability check: can the plan language express every supported development interpretation, end to end?
For each gold interpretation (never shown to the model; this is a check of the REPRESENTATION, not of any model) the script writes the call a perfect model would write, and tests, in order:
    1 plan       the gold Query has a plan form (an operation of the plan language)
    2 grammar    the generation grammar accepts that exact call text (with a conversation state when the row is a follow-up)
    3 compiler   compiling the parsed call returns a Query equal to the gold Query (documented defaults only for omitted fields)
    4 executor   the compiled Query validates and executes (answered / no_data / missing_data) rather than being rejected
It reports the first failing step per question with the reason, so representation gaps are extended with general typed fields instead of being blamed on model capacity.
    python bench/representability.py [--set dev|final|both] [--dialect py|json] [-v]"""
import argparse, collections, os, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import capreg, plancompile as PC, plangrammar, signal_registry as R, toolspec as T
from vgpt.gbnf import Grammar
from vgpt.query_ir import Query


def dummy_sources(plan):
    plan = dict(plan); plan["src"] = {k: "the words" for k in T.GROUNDED.values() if plan.get(k) not in (None, "", [], {})}; return plan


def edit_call(prev_plan, gold_plan):
    """The follow-up a perfect model would write to get from the previous resolved plan to the gold plan: only what CHANGES, `drop` for removed parts, conditions added / replaced as needed. -> (function, args)."""
    fn = gold_plan["fn"]; fields = PC.FN_FIELDS[fn]; carried = {k: v for k, v in prev_plan.items() if k in fields and k in PC.KEEP}; change = {"fn": fn, "op": T.FN_OP[fn], "turn": "follow_up"}; drops = []; cedit = None
    for k in fields:
        gv, pv = gold_plan.get(k), carried.get(k)
        if gv in (None, "", [], {}):
            if pv not in (None, "", [], {}) and k not in ("bef", "aft"): drops.append(next(d for d, ks in PC.DROP_FIELDS.items() if k in ks))      # an absent event window means "the default window", which the carried one already equals
            continue
        if k == "cond" and pv:
            gn = {repr(PC._norm_cond(c)) for c in gv}; pn = {repr(PC._norm_cond(c)) for c in pv}
            if gn == pn: continue
            if pn < gn: change["cond"] = [c for c in gv if repr(PC._norm_cond(c)) not in pn]; cedit = "add"
            else: change["cond"] = list(gv); cedit = "replace"
        elif k == "ent" and pv:
            diff = {a: b for a, b in gv.items() if pv.get(a) != b}
            if diff: change["ent"] = diff
        elif gv != pv: change[k] = gv
    if cedit: change["cedit"] = cedit
    if drops: change["drop"] = sorted(set(drops))
    return change


def followup_check(rows, reg, G, qc):
    """Can the FOLLOW-UP (state edit applied to the previous resolved plan) express each gold follow-up? A gold query turned into a complete stand-alone plan does not prove this."""
    n = ok = 0; fails = []
    for r in rows:
        if not r["prev_ir"] or r["ir"].startswith("op=none") or r["prev_ir"].startswith("op=none"): continue
        n += 1
        try: pq, gq = Query.from_text(r["prev_ir"]), Query.from_text(r["ir"]); pp, gp = PC.query_to_plan(pq), PC.query_to_plan(gq)
        except Exception as e: fails.append((r["id"], r["text"], f"unparsable {e}")); continue
        if pp is None or gp is None: fails.append((r["id"], r["text"], "no plan form")); continue
        ch = edit_call(pp, gp); fa = T.plan_to_fn_args(dict(ch, **{k: v for k, v in ch.items()})) if False else None
        args = {"turn": "follow_up"}
        for k, v in ch.items():
            if k in ("fn", "op", "turn", "cedit", "drop"): continue
            pn = {tgt: nm for nm, tgt in T.PARAM_KEY.items() if tgt}.get(k)
            args[pn if pn else k] = v
        # build the call from a plan so that period / duration / selection are written in their call forms
        plan_for_call = {k: v for k, v in ch.items() if k not in ("turn", "cedit", "drop")}; fa = T.plan_to_fn_args(dict(plan_for_call, src={}))
        if fa is None: fails.append((r["id"], r["text"], "no call form")); continue
        fn, cargs = fa; cargs = {k: v for k, v in cargs.items() if k != "turn" and not k.endswith("_words")}
        if set(cargs) | set(ch.get("drop") or []) == set(): cargs = {}
        cargs = {"turn": "follow_up", **{k: v for k, v in cargs.items() if k in {p for p, _ in T.frame_fields(fn)}}}
        for k in list(cargs):                                                                          # keep only what the edit changes
            key = T.PARAM_KEY.get(k)
            if key and key not in ch and k not in ("turn",) and not (k == "which"): cargs.pop(k)
        if ch.get("cedit"): cargs["conditions_edit"] = ch["cedit"]
        if ch.get("drop"): cargs["drop"] = ch["drop"]
        for pn in T.GROUNDED:                                                                          # a grounded argument is written with its (dummy) quoted words
            if pn in cargs: cargs = {**{k: v for k, v in cargs.items() if k != pn}, T.words_name(pn): "the words", pn: cargs[pn]}
        order = [pn for pn, _, _ in T.all_params(fn)]; cargs = {k: cargs[k] for k in sorted(cargs, key=lambda k: order.index(k) if k in order else 99)}              # the call writes arguments in the schema's order
        text = T.args_to_pythonic(fn, cargs) + "]"
        if not G.accepts(text): fails.append((r["id"], r["text"], "grammar: " + text[:200])); continue
        fn2, a2 = T.parse_call("[" + text); o = PC.compile_plan(T.call_to_plan(fn2, a2), pp, None, reg, DEFAULT_NOW, 0)
        if o.kind != "query": fails.append((r["id"], r["text"], f"{o.kind}: {o.reason or o.say} | {text[:160]}")); continue
        cq = o.query; gq2 = Query.from_text(r["ir"])
        if gq2.op == "count" and gq2.event is not None: gq2.event.selector = cq.event.selector = "all"
        if gq2.op == "event_context" and gq2.event is not None and not gq2.event.before and not gq2.event.after: gq2.event.before = cq.event.before
        if gq2.to_text() != cq.to_text(): fails.append((r["id"], r["text"], f"gold {gq2.to_text()} != applied {cq.to_text()} | edit {text[:140]}")); continue
        ok += 1
    return n, ok, fails


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="both"); ap.add_argument("--dialect", default="py"); ap.add_argument("-v", action="store_true"); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); reg = capreg.CapabilityRegistry(R.Profile(conn))
    G = {ws: Grammar(plangrammar.build(reg, with_state=ws, style=a.dialect)) for ws in (True, False)}
    files = {"dev": ["indep_dev.tsv"], "final": ["indep_final.tsv"], "both": ["indep_dev.tsv", "indep_final.tsv"]}[a.set]; rows = []
    for f in files: rows += load(os.path.join(HERE, "bench", f))
    from vgpt.qc import QueryCompiler
    qc = QueryCompiler(conn, now=DEFAULT_NOW); fails = collections.defaultdict(list); n = 0
    for r in rows:
        if r["ir"].startswith("op=none"): continue
        n += 1
        try: q = Query.from_text(r["ir"])
        except Exception as e: fails["plan"].append((r["id"], r["text"], f"gold unparsable: {e}")); continue
        plan = PC.query_to_plan(q)
        if plan is None: fails["plan"].append((r["id"], r["text"], f"op {q.op} has no plan form")); continue
        fa = T.plan_to_fn_args(dummy_sources(plan))
        if fa is None: fails["plan"].append((r["id"], r["text"], "no call form")); continue
        text = T.args_to_pythonic(*fa) + "]" if a.dialect == "py" else __import__("json").dumps({"name": fa[0], "arguments": fa[1]})
        if not G[bool(r["prev_ir"])].accepts(text): fails["grammar"].append((r["id"], r["text"], text[:240])); continue
        fn, args = T.parse_call("[" + text) if a.dialect == "py" else (fa[0], fa[1])
        o = PC.compile_plan(T.call_to_plan(fn, args), None, None, reg, DEFAULT_NOW, 0)
        if o.kind != "query": fails["compiler"].append((r["id"], r["text"], f"{o.kind}: {o.reason or o.say}")); continue
        gq = Query.from_text(r["ir"]); cq = o.query
        if gq.op == "count" and gq.event is not None: gq.event.selector = cq.event.selector = "all"                      # a count ignores which occurrence is "selected"
        if gq.op == "event_context" and gq.event is not None and not gq.event.before and not gq.event.after: gq.event.before = cq.event.before    # the documented default window
        if gq.op in ("event_context", "explain_event", "list_events") and gq.event is not None and gq.event.selector == "last" and cq.event is not None and cq.event.selector == "last": pass
        if gq.to_text() != cq.to_text(): fails["compiler"].append((r["id"], r["text"], f"gold {gq.to_text()} != compiled {cq.to_text()}")); continue
        try: res = qc.execute(o.query)
        except Exception as e: fails["executor"].append((r["id"], r["text"], f"exception {type(e).__name__}")); continue
        if res.status in ("invalid",): fails["executor"].append((r["id"], r["text"], f"status {res.status}"))
    fn_, fok, ffails = followup_check(rows, reg, G[True], qc)
    bad = sum(len(v) for v in fails.values())
    print(f"representable end to end: {n - bad}/{n}   (first failing step: " + ", ".join(f"{k} {len(v)}" for k, v in fails.items()) + ")")
    print(f"follow-up state edits (previous resolved plan + edit -> gold): {fok}/{fn_} representable" + ("" if fok == fn_ else f"  ({fn_ - fok} not)"))
    for x in ffails[: (len(ffails) if a.v else 8)]: print(f"  [follow-up] {x[0]} {x[1][:50]} | {x[2][:230]}")
    for k, v in fails.items():
        for x in v[: (len(v) if a.v else 8)]: print(f"  [{k}] {x[0]} {x[1][:55]} | {x[2]}")


if __name__ == "__main__": main()

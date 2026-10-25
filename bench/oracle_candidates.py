"""ORACLE-CANDIDATE sets for the selector diagnostic. For each stand-alone, answerable question of the frozen slice: the GOLD interpretation (the evaluator's gold Query) and up to N valid nearby alternatives that differ from it in ONE semantic respect - operation, statistic/polarity, target, time, scope (an added or removed restriction) or event-vs-signal attachment. Every candidate is a typed Query rendered in plain English by vgpt.paraphrase.describe (deterministic, registry labels only). The period is rendered only when the independent annotation says the question states one (a defaulted period is not something the person said); an alternative that ADDS a period asserts something unstated and is wrong for that reason.
This is a DIAGNOSTIC bank: mutations of the gold are produced from the typed IR, never from wording, and are never used by the product. Deterministic for a seed."""
import copy, os, random, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_gliner import annotations, has
from bench.indep_io import load
from vgpt import capreg, paraphrase, signal_registry as R
from vgpt.query_ir import Query, TimeSpec, Condition

TIMES = ["all", "last:7:day", "cal:yesterday", "cal:today", "last:30:day", "cal:this_week", "cal:this_month", "last:24:hour"]
EVENT_OPS = {"count": "count", "list_events": "list_events", "exists": "list_events"}
AGG_SWAP = {"mean": ["max", "min", "sum"], "max": ["min", "mean"], "min": ["max", "mean"], "sum": ["mean", "max"], "median": ["mean"], "std": ["mean", "delta"], "delta": ["mean", "std"], "p95": ["max", "mean"]}
OP_SWAP = {"aggregate": ["extreme", "trend", "detect_change", "current_value"], "extreme": ["aggregate", "detect_change", "current_value"], "count": ["list_events", "explain_event", "event_context"], "list_events": ["count", "explain_event", "event_context"],
           "exists": ["count", "list_events"], "percent_time": ["count", "exists", "aggregate"], "compare": ["aggregate", "trend"], "trend": ["detect_change", "aggregate", "explain_change"], "detect_change": ["trend", "extreme", "explain_change"],
           "explain_change": ["trend", "detect_change", "compare"], "explain_event": ["event_context", "list_events", "count"], "event_context": ["explain_event", "list_events", "count"], "current_value": ["aggregate", "extreme", "capability"],
           "rank": ["aggregate", "list_events", "count"], "capability": ["current_value", "aggregate"], "correlation": ["compare", "trend"]}


def _words(q):
    """paraphrase.describe, plus the yes/no query (which that renderer leaves as IR text): 'whether <event> occurred' / 'whether <the condition> held'."""
    if q.op == "exists":
        per = paraphrase.label_of_period(q.scope.time.to_text()); per = "" if q.scope.time.type == "current" else f" over {per}" if q.scope.time.type != "calendar" else f" {per}"
        if q.event: return f"whether any {q.event.type.replace('_', ' ').lower()} event occurred{per}"
        conds = ", ".join(paraphrase._cond(c) for c in q.scope.conditions) or "the condition"; return f"whether the condition ({conds}) held{per}"
    return paraphrase.describe(q)


def describe(q, stated_period):
    """The candidate in words; the period only when the question states one (or when the candidate asserts a period the gold does not)."""
    if q.scope.time.type == "current": return _words(q)
    return _words(q) if stated_period else _words(_no_time(q))


def _no_time(q):
    c = copy.deepcopy(q); c.scope.time = TimeSpec("current"); return c


def mutations(q, reg, rng, stated_period):
    """-> [(kind, Query, stated_period flag for its description)]"""
    out = []
    def add(kind, mq, sp): out.append((kind, mq, sp))
    if q.agg in AGG_SWAP and q.op in ("aggregate", "extreme", "compare"):
        for a in AGG_SWAP[q.agg][:2]:
            m = copy.deepcopy(q); m.agg = a; m.op = "extreme" if (a in ("max", "min") and q.op != "compare") else ("aggregate" if q.op != "compare" else "compare"); add("statistic", m, stated_period)
    if q.op == "extreme" and q.agg in ("max", "min"):
        m = copy.deepcopy(q); m.agg = "min" if q.agg == "max" else "max"; add("polarity", m, stated_period)
    for o in OP_SWAP.get(q.op, [])[:3]:
        if o in ("list_events", "explain_event", "event_context") and not q.event: continue
        if o in ("trend", "detect_change", "explain_change", "current_value", "capability", "extreme", "aggregate") and not q.target: continue
        m = copy.deepcopy(q); m.op = o
        if o == "extreme" and not m.agg in ("max", "min"): m.agg = "max"
        if o in ("aggregate",) and m.agg in (None, "count", "percent_time"): m.agg = "mean"
        if o in ("trend", "detect_change", "explain_change", "current_value", "capability") and not m.target:
            if q.event and q.event.type: m.target = None
            else: continue
        add("operation", m, stated_period)
    if q.target and q.target in R.CONCEPTS:
        cat = R.CONCEPTS[q.target].category if hasattr(R.CONCEPTS[q.target], "category") else None
        pool = [i for i, c in R.CONCEPTS.items() if i != q.target and (cat is None or getattr(c, "category", None) == cat)] or [i for i in R.CONCEPTS if i != q.target]
        for t in rng.sample(pool, min(2, len(pool))):
            m = copy.deepcopy(q); m.target = t; add("target", m, stated_period)
    for tt in rng.sample([t for t in TIMES if t != q.scope.time.to_text()], 2):
        m = copy.deepcopy(q); m.scope.time = TimeSpec.from_text(tt); add("time", m, True)
    if q.scope.conditions:
        m = copy.deepcopy(q); m.scope.conditions = []; add("scope", m, stated_period)
        c = q.scope.conditions[0]
        if c.op in ("gt", "lt"):
            m = copy.deepcopy(q); m.scope.conditions[0].op = "lt" if c.op == "gt" else "gt"; add("scope", m, stated_period)
    else:
        m = copy.deepcopy(q); m.scope.conditions = [Condition(predicate=rng.choice(["highway", "city", "charging", "night"]))]; add("scope", m, stated_period)
    if q.event and q.event.type:
        m = copy.deepcopy(q); m.event.selector = {"last": "first", "first": "last", "all": "last", "most_severe": "last"}.get(m.event.selector, "last"); add("operation", m, stated_period)
    return out


def build(n_alt=6, seed=7, db=os.path.join(HERE, "data", "gen_courier_ext.db")):
    reg = None; rng = random.Random(seed); ann = annotations()
    ids = [l.strip() for l in open(os.path.join(HERE, "bench", "slice_v1.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}
    out = []
    for i in ids:
        r = rows[i]
        if r["prev_text"] or r["ir"].startswith("op=none"): continue
        q = Query.from_text(r["ir"]); stated = has(ann[i]["interval"]); gold = describe(q, stated); cands = [("gold", gold)]; seen = {gold}
        ms = mutations(q, reg, rng, stated); rng.shuffle(ms); kinds = set()
        for kind, mq, sp in sorted(ms, key=lambda x: x[0] in kinds):                       # prefer a spread of kinds
            try: d = describe(mq, sp)
            except (AttributeError, KeyError, TypeError): continue
            if d in seen or d.strip() == "": continue
            seen.add(d); cands.append((kind, d)); kinds.add(kind)
            if len(cands) > n_alt: break
        out.append({"id": i, "text": r["text"], "candidates": cands})
    return out


if __name__ == "__main__":
    for e in build()[:14]:
        print(e["id"], "|", e["text"])
        for k, d in e["candidates"]: print("    ", k.ljust(10), d)

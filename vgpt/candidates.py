"""Deterministic candidate generation: every well-formed query the question can support, built from the question text, the accepted conversation state and the runtime schema (registry + this vehicle's profile).
No model is involved. Each candidate keeps the phrases that support its fields (via the interpretation contract), so a selector - the small model, or a person answering a clarification - chooses among readings instead of writing a query.
Candidate AVAILABILITY (is the intended query among them?) is measured separately from selection, so generator failures are not charged to the model.
Sources of evidence: strict lexicon hits, describe-by-parts matches, cue words, unit/referent words for thresholds, the question's period phrases, named conditions, group words, selection words, and (for continuations) the previous accepted query."""
import copy
import itertools
import re

from . import interpretation as I, nl2ir, signal_registry as R
from .query_ir import Condition, EntitySelect, EventRef, Query, Scope, TimeSpec
from .query_validator import validate
from .vocab_parts import _described_by_parts, _phrases_of, _signal_hits, _tokens, event_by_parts, signal_by_parts

SIGNAL_OPS = ("current_value", "aggregate", "extreme", "trend", "explain_change", "detect_change", "percent_time", "capability")
EVENT_OPS = ("count", "list_events", "event_context", "explain_event")


def _targets(t, tok, prev, fu):
    out = []; seen = _signal_hits(t)
    for n in seen:
        if n not in out: out.append(n)
    sp = signal_by_parts(t)
    if sp and sp[0] not in out: out.append(sp[0])
    for n, rx in I.TARGET_CUES.items():
        if re.search(rx, t) and n in R.CONCEPTS and n not in out and (n != "drive_mode"): out.append(n)
    for n in R.CONCEPTS:
        if n not in out and _described_by_parts(tok, _phrases_of(n)) and len(_tokens(R.label_of(n))) >= 2: out.append(n)
    if prev is not None and fu:
        for n in (prev.target, prev.target2):
            if n and n not in out: out.append(n)
    return out[:6]


def _wheel_specialise(targets, t):
    mw = nl2ir.WHEEL_RX.search(t); out = []
    for n in targets:
        v = nl2ir._wheel_variant(n, (mw.group(1), mw.group(2))) if mw else None
        out.append(v if (v and v in R.CONCEPTS) else n)
    return out


def _events(t, tok):
    ev = [e for _, e in nl2ir._events(nl2ir._S(t)) if e]
    h = event_by_parts(t)
    if h and h[0] not in ev: ev.append(h[0])
    if re.search(r"\b(warning|alert|alarm|fault|problem|issue)s?\b", t) and "ANY_WARNING" not in ev: ev.append("ANY_WARNING")
    return list(dict.fromkeys(ev))[:3]


def _conditions(t, targets, prev):
    """-> list of alternative condition sets [[Condition,...], ...]: thresholds (comparator + number + unit + referent) and named conditions."""
    named = [Condition(predicate=p) for _, p in nl2ir._predicates(nl2ir._S(t), [])]
    named = [c for i, c in enumerate(named) if c.predicate not in [x.predicate for x in named[:i]]]
    trs = I.thresholds(t); options = []
    for tr in trs:
        near = I._near_words(t, tr["span"]); refs = set().union(*[I.REFERENT[w] for w in near + I._words(t) if w in I.REFERENT]) if any(w in I.REFERENT for w in near + I._words(t)) else set()
        sigs = []
        for s in list(refs) + ([I.UNIT_CUE_SIGNAL[tr["unit"]]] if tr["unit"] in I.UNIT_CUE_SIGNAL else []) + [x for x in targets] + ([c.signal for c in prev.scope.conditions if c.signal] if prev is not None else []):
            if s in R.CONCEPTS and s not in sigs:
                dim = R.unit_family(R.CONCEPTS[s].unit)
                if tr["unit"] in (None, "degree") or R.unit_family(tr["unit"]) == dim: sigs.append(s)
        if not sigs and tr["unit"] is not None:                                                            # a unit gives a dimension, not a sensor: offer each sensor of that dimension as a reading (the person chooses)
            dim = "temperature" if tr["unit"] == "degree" else R.unit_family(tr["unit"])
            sigs = [n for n in I.CANDIDATE_SIGNALS if n in R.CONCEPTS and R.unit_family(R.CONCEPTS[n].unit) == dim]
        conds = []
        for s in sigs[:4]:
            v, conv = nl2ir._to_unit(str(tr["num"]), tr["unit"], s) if tr["unit"] not in (None, "degree") else (tr["num"], None)
            if v is not None: conds.append(Condition(signal=s, op=tr["cmp"], value=float(tr["num"]) if conv else v, unit=conv))
        options.append(conds or [None])
    combos = [list(c) for c in itertools.product(*options)] if options else [[]]
    return [([x for x in combo if x is not None] + named) for combo in combos[:4]]


def _periods(t, now, tz):
    return nl2ir._times(nl2ir._S(t), now, tz)


def generate(text, prev=None, profile=None, now=None, tz=0, rules_draft=None, limit=24):
    """-> list of (Query, Interpretation), best supported first."""
    try:
        from . import nlu
        text = nlu.shorthand(text); fixed, fixes = nlu.normalize(str(text), repair_typos=True); text = fixed if fixes else text
    except Exception: pass
    t = nl2ir.normalise(text); S = nl2ir._S(t); tok = _tokens(t)
    cont = bool(I.CONT_RX.match(t)); fu = prev is not None and (cont or bool(I.REF_RX.search(t)) or len(t.split()) <= 6)
    targets = _wheel_specialise(_targets(t, tok, prev, fu), t); events = _events(t, tok); times = _periods(t, now, tz); condsets = _conditions(t, targets, prev)
    sel = nl2ir._selection(nl2ir._S(t), t); aggs = [a for rx, a in nl2ir.AGG_WORDS if re.search(rx, t)]
    ops = [k for k, rx in I.OP_CUES.items() if rx and k in SIGNAL_OPS + EVENT_OPS + ("rank", "compare", "correlation") and re.search(rx, t)] + ["aggregate", "current_value"]
    if events and not any(o in ops for o in EVENT_OPS): ops = ["count", "list_events"] + ops                        # an event with no operation word: counting and listing are both readings
    ops = list(dict.fromkeys(ops)); out = []
    def add(q):
        try:
            if profile is not None and validate(q, profile).codes(): return
            elif profile is None and validate(q, None).codes(): return
        except Exception: return
        out.append(q)
    def base(op, **kw):
        q = Query(op=op); q.scope = Scope(); q.source = "candidate"
        for k, v in kw.items(): setattr(q, k, v)
        return q
    def period(q, conds):
        q.scope.select = copy.deepcopy(sel) if sel is not None else None; q.scope.conditions = copy.deepcopy(conds)
        q.scope.time = copy.deepcopy(times[0]) if times else _spec(I.documented_default_period(q, t))
        return q
    for conds in condsets:
        for op in ops:
            if op == "percent_time":
                if conds: add(period(base(op), conds))
            elif op in SIGNAL_OPS and targets:
                for tg in targets[:4]:
                    if op in ("aggregate",):
                        for agg in (aggs[:1] or [("sum" if tg in R.EXTENSIVE else "mean")]): add(period(base(op, target=tg, agg=agg), conds))
                    elif op == "extreme":
                        for agg in (["max", "min"] if not aggs or aggs[0] not in ("max", "min") else [aggs[0]]): add(period(base(op, target=tg, agg=agg), conds))
                    elif op in ("current_value", "trend", "explain_change", "detect_change", "capability"): add(period(base(op, target=tg), conds if op in ("current_value", "trend") else []))
            elif op in EVENT_OPS and events:
                for ev in events[:3]:
                    q = period(base(op), conds); q.event = EventRef(type=ev, selector="all" if op == "list_events" and re.search(r"\b(all|every|list|show|each)\b", t) else "last" if op in ("list_events", "event_context", "explain_event") else "last")
                    if op == "event_context": q.event.before = 600
                    add(q)
            elif op == "compare" and targets:
                preds = [c.predicate for c in conds if c.predicate]
                if len(preds) >= 2: q = period(base(op, target=targets[0], agg="mean"), []); q.groups = preds[:2]; add(q)
                elif len(times) >= 2: q = period(base(op, target=targets[0], agg="mean"), []); q.scope_b = Scope(copy.deepcopy(times[1]), None, []); add(q)
            elif op == "correlation" and len(targets) >= 2: add(period(base(op, target=targets[0], target2=targets[1]), []))
            elif op == "rank" and sel is not None: add(period(base(op), []))
    if rules_draft is not None and rules_draft.op: out.append(rules_draft)
    if prev is not None and fu:                                                                           # continuation: the previous accepted query with the stated slots replaced
        q = copy.deepcopy(prev); q.source = "candidate"
        if times: q.scope.time = copy.deepcopy(times[0])
        if targets and targets[0] != prev.target: q.target = targets[0]
        add(q)
    seen = set(); res = []
    for q in out:
        k = q.to_text()
        if k in seen: continue
        seen.add(k); it = I.interpret(text, q, prev, source="candidate", now=now, tz=tz); res.append((q, it))
    res.sort(key=lambda p: (len(p[1].issues()), -sum(1 for f in p[1].fields if f.origin in ("question", "parts", "cue", "prior_turn"))))
    return res[:limit]


def _spec(s):
    if s == "all": return TimeSpec("all")
    if s == "now": return TimeSpec("current")
    if s.startswith("cal:"): return TimeSpec("calendar", name=s[4:])
    _, n, u = s.split(":"); return TimeSpec("relative", int(n), u)

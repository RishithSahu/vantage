"""Plain-English reading of a Query, so that a clarification can offer concrete readings ("the number of harsh-braking events yesterday") instead of asking the person to rephrase.
Deterministic, built only from the registry's own labels; never model-written text."""
import re
from . import signal_registry as R


def label_of_period(text):
    if text in ("all", "now"): return {"all": "the whole recorded history", "now": "the current moment"}[text]
    if text.startswith("cal:"): return text[4:].replace("_", " ")
    if text.startswith("last:"):
        _, n, u = text.split(":"); return f"the last {n} {u}s" if n != "1" else f"the last {u}"
    if text.startswith("date:"): return text[5:]
    return text

STAT = {"mean": "the average", "max": "the highest", "min": "the lowest", "sum": "the total", "median": "the median", "std": "the variability of", "p95": "the 95th percentile of", "delta": "the change in", "count": "the number of", "percent_time": "the share of time with"}
OPS = {"current_value": "the current value of", "trend": "whether there is a trend in", "explain_change": "what explains the change in", "detect_change": "when the largest change happened in", "correlation": "how", "capability": "whether the vehicle records"}


def _cond(c):
    if c.predicate:
        ph = R.PREDICATES.get(c.predicate, (c.predicate.replace("_", " "),))[0]
        return ph if c.op != "not" else "excluding the time " + ph
    sym = {"gt": "above", "ge": "at least", "lt": "below", "le": "at most", "eq": "equal to"}.get(c.op, c.op)
    return f"while {R.label_of(c.signal)} is {sym} {c.value:g}" + (f" {c.unit}" if c.unit else (f" {R.CONCEPTS[c.signal].unit}" if c.signal in R.CONCEPTS else ""))


def describe(q):
    """-> a short phrase describing what the query would compute."""
    per = label_of_period(q.scope.time.to_text()); conds = [_cond(c) for c in q.scope.conditions]; tail = ((" " + ", ".join(conds)) if conds else "") + ("" if q.scope.time.type == "current" else (f" {per}" if q.scope.time.type == "calendar" and q.scope.time.name in ("today", "yesterday", "day_before_yesterday", "last_night", "this_morning") else f" over {per}"))
    ev = q.event.type.replace("_", " ").lower() if q.event else ""
    sel = q.scope.select; seltxt = ""
    if sel is not None:
        seltxt = f" on the {'' if not sel.k else str(sel.k) + ' '}{'longest' if sel.rank_by in ('distance_driven', 'duration') and sel.order == 'desc' else 'shortest' if sel.rank_by in ('distance_driven', 'duration') else 'selected'} {sel.entity.replace('_', ' ')}{'s' if sel.k and sel.k > 1 else ''}"
    if q.op == "count": return f"the number of {ev or R.label_of(q.target) if (ev or q.target) else 'items'} events{tail}".replace(" events events", " events") if ev else f"the number of {(sel.entity.replace('_', ' ') + 's') if sel else 'items'}{tail}"
    if q.op == "list_events":
        what = {"all": f"the list of {ev} events", "page": f"the list of {ev} events (page {q.event.n or 1})", "last": f"the most recent {ev} event", "first": f"the earliest {ev} event", "nth": f"event number {q.event.n or 1} of {ev}", "most_severe": f"the most severe {ev} event"}.get(q.event.selector, f"{ev} events")
        return what + tail
    if q.op == "event_context": return f"what happened around the {ev} event" + (f" ({R.label_of(q.target)})" if q.target else "")
    if q.op == "explain_event": return f"what explains the {ev} event"
    if q.op == "rank": return f"the ranking of {sel.entity.replace('_', ' ') + 's' if sel else 'items'}" + (f" by {R.label_of(sel.rank_by) if sel and sel.rank_by in R.CONCEPTS else (sel.rank_by or '')}" if sel else "") + tail
    if q.op == "compare": return f"{STAT.get(q.agg, 'the value of')} {R.label_of(q.target)} compared between " + (" and ".join(R.PREDICATES.get(g, (g,))[0] for g in q.groups) if q.groups else f"{per} and {label_of_period(q.scope_b.time.to_text()) if q.scope_b else 'another period'}")
    if q.op == "percent_time": return f"the share of time{tail}"
    if q.op == "correlation": return f"how {R.label_of(q.target)} relates to {R.label_of(q.target2)}{tail}"
    if q.op in OPS: return f"{OPS[q.op]} {R.label_of(q.target)}" + ("" if q.op in ("current_value", "capability") else tail)
    if q.op in ("aggregate", "extreme"): return f"{STAT.get(q.agg, 'the value of')} {R.label_of(q.target)}{seltxt}{tail}"
    return q.to_text()

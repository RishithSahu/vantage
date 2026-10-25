"""Meaning frame (+ literal values) -> typed Query for the existing deterministic engine.

A table-driven mapping: the frame's (act, op, ent) select a query form; the frame's concept supplies the target / event; literals supply the period and numeric limits.  No user text is read
here: `lit` is the already-parsed Literals object.  Returns (Query, None) or (None, reason) where reason names the reason in words the resolver can show."""
from .. import signal_registry as R
from ..query_ir import Condition, EntitySelect, EventRef, Query, Scope, TimeSpec

AGG = {"mean": "mean", "median": "median", "sum": "sum", "spread": "std", "count": "count", "max": "max", "min": "min"}
ACCUMULATING = ("energy_used", "distance_driven", "regen_energy")
MODIFIERS = ("raining", "night", "hot_weather", "cold", "high_speed", "hard_braking", "hard_acceleration")                  # conditions that hold during part of the driving: "against driving" = against the rest of it
STATE_ON = ("rain", "adas_state")                                                                   # on / off states: a duration or a share is the time the state was on
RANK_METRIC = {"trip_record": "distance_driven", "distance_driven": "distance_driven", "energy_used": "energy_used", "wh_per_km": "wh_per_km", "speed": "avg_speed"}
PREVIOUS_CALENDAR = {"this_week": "last_week", "this_month": "last_month", "today": "yesterday", "last_week": None, "yesterday": "day_before_yesterday"}


def _cond(token):
    if token in ("none", "threshold"): return None
    return Condition(predicate=token[4:], op="not") if token.startswith("not_") else Condition(predicate=token)


def _threshold(lit, concept):
    """-> Condition (a numeric limit the user typed) or None."""
    for c in lit.thresholds:
        if c.signal in (None, concept) or c.signal in R.CONCEPTS: return c
    if lit.cmp and lit.numbers and concept in R.CONCEPTS:                          # a bare number after a comparator word ("when above 50") is a limit on the concept itself
        return Condition(signal=concept, op=lit.cmp, value=lit.numbers[0], unit=R.CONCEPTS[concept].unit)
    return None


def _time(lit, default):
    return lit.times[0] if lit.times else default


def to_query(frame, lit, onto):
    c = onto.concepts.get(frame.concept)
    if c is None: return None, "unknown concept"
    act, op = frame.act, frame.op
    if any(t.type == "range" for t in lit.times): return None, "an explicit interval (from ... to ... ago) is not accepted by the typed engine yet (its validator only takes named windows)"
    rel7, whole = TimeSpec("relative", 7, "day"), TimeSpec("all")
    conds = [x for x in (_cond(frame.cond),) if x is not None]
    if frame.cond == "threshold":
        th = _threshold(lit, c.id)
        if th is None: return None, "I need the limit you mean (for example 'above 100 km/h')"
    numeric = c.kind in ("signal", "derived", "declared")

    if act == "now" and c.id == "wh_per_km": return Query(op="aggregate", target=c.id, agg="mean", scope=Scope(_time(lit, rel7))), None      # a ratio over time has no instantaneous reading: its recent average
    if act == "now" and c.id in ACCUMULATING: return Query(op="aggregate", target=c.id, agg="sum", scope=Scope(_time(lit, TimeSpec("calendar", name="today")))), None
    if act == "now" and numeric:
        return Query(op="current_value", target=c.id, scope=Scope(TimeSpec("current"))), None

    if act == "stat":
        if c.id == "trip_record":
            if op == "count" and frame.ent == "trip": return Query(op="count", scope=Scope(_time(lit, rel7), EntitySelect("trip", None, None, "desc", None))), None
            return None, "trip statistics other than the number of trips are not wired to an executor yet"
        if c.id == "drive_mode" and op in ("share", "duration") and conds:
            return Query(op="percent_time" if op == "share" else "duration", scope=Scope(_time(lit, rel7), None, conds)), None
        if not numeric: return None, "no statistic is defined for this kind of concept"
        if c.id in STATE_ON and op in ("share", "duration", "count", "mean", "max"):               # a yes/no state (rain, lane keeping engaged): "how long / how much of the time" is the time with the state on
            return Query(op="share" if op == "share" else "duration", target=c.id, scope=Scope(_time(lit, rel7), None, conds), params={"thr": "gt:0.5"}), None
        if op == "count" and frame.cond == "threshold":                                           # "how many times did X go below N": separate occasions
            th = _threshold(lit, c.id)
            if th is not None: return Query(op="episodes", target=c.id, scope=Scope(_time(lit, rel7)), params={"thr": f"{th.op}:{format(float(th.value), '.15g')}"}, unit=th.unit), None
        if op in ("share", "duration"):
            if frame.cond != "threshold": return None, "I need a numeric limit to measure time above or below"
            th = _threshold(lit, c.id); q = Query(op=op, target=c.id, scope=Scope(_time(lit, rel7)), params={"thr": f"{th.op}:{format(float(th.value), '.15g')}"}, unit=th.unit)
            return q, None
        agg = AGG.get(op)
        if agg is None or op == "count": return None, "this statistic needs an executor that is not wired yet"
        if frame.ent in ("trip", "charge"):
            entity = "trip" if frame.ent == "trip" else "charge_session"
            return Query(op="aggregate", target=c.id, agg=agg, scope=Scope(lit.times[0] if lit.times else TimeSpec("all"), EntitySelect(entity, None, "recency", "desc", 1), conds)), None
        if c.id in ACCUMULATING and agg in ("mean", "median", "std"): return None, "an average of an accumulating quantity (for example distance per day) is not wired to an executor yet"
        scope = Scope(_time(lit, rel7), None, conds + ([_threshold(lit, c.id)] if frame.cond == "threshold" else []))
        return Query(op="aggregate", target=c.id, agg=agg, scope=scope), None

    if act == "extreme" and numeric or (act == "extreme" and c.id == "trip_record"):
        if frame.ent == "day": return None, "per-day ranking is not wired to an executor yet"
        if frame.ent == "trip" or c.id == "trip_record":
            metric = RANK_METRIC.get(c.id)
            if metric is None and numeric and frame.ent == "trip":                                      # the peak / trough of a signal inside the chosen trip
                return Query(op="aggregate", target=c.id, agg=AGG[op], scope=Scope(lit.times[0] if lit.times else TimeSpec("all"), EntitySelect("trip", None, "recency", "desc", 1), conds)), None
            if metric is None: return None, "ranking trips by this quantity is not wired to an executor yet"
            return Query(op="rank", scope=Scope(_time(lit, whole), EntitySelect("trip", None, metric, "desc" if op == "max" else "asc", 1))), None
        return Query(op="extreme", target=c.id, agg=AGG[op] if op in ("max", "min") else "max", scope=Scope(_time(lit, rel7), None, conds)), None

    if act == "trend" and numeric:
        time = _time(lit, TimeSpec("relative", 14, "day"))
        if c.id == "range_estimate": return Query(op="explain_change", target=c.id, scope=Scope(time)), None
        return Query(op="trend" if op in ("change", "none") else "detect_change", target=c.id, scope=Scope(time, None, conds)), None

    if act == "compare" and numeric:
        agg = "sum" if c.id in ACCUMULATING else AGG.get(op, "mean") if op != "none" else "mean"
        if frame.vs == "period":
            if len(lit.times) >= 2: t1, t2 = lit.times[0], lit.times[1]
            elif len(lit.times) == 1 and lit.times[0].type == "calendar" and PREVIOUS_CALENDAR.get(lit.times[0].name):
                t1 = lit.times[0]; t2 = TimeSpec("calendar", name=PREVIOUS_CALENDAR[t1.name])
            elif len(lit.times) == 1 and lit.times[0].type == "relative": t1 = lit.times[0]; t2 = TimeSpec("previous", t1.value, t1.unit)
            else: return None, "I need the two periods to compare (for example 'this week versus last week')"
            q = Query(op="compare", target=c.id, agg=agg, scope=Scope(t1, None, conds)); q.scope_b = Scope(t2, None, conds); return q, None
        if frame.cond != "none" and frame.vs != "none":
            return Query(op="compare", target=c.id, agg=agg, scope=Scope(_time(lit, whole if {frame.cond, frame.vs} & {"cold", "hot_weather"} else rel7), None, [Condition(predicate="driving")] if frame.vs == "driving" and frame.cond in MODIFIERS else []), groups=[frame.cond, "not_" + frame.cond if frame.vs == "driving" and frame.cond in MODIFIERS else frame.vs]), None   # winter against summer needs the whole history, not a week; "in the rain" against "driving" means against the dry driving
        return None, "a comparison needs two conditions or two periods"

    if act == "events" and c.kind == "event":
        ev = c.registry_key; time = lit.times[0] if lit.times else None
        if op in ("last", "first"): return Query(op="list_events", scope=Scope(time or whole), event=EventRef(ev, op)), None
        if op == "list": return Query(op="list_events", scope=Scope(time or rel7, None, conds), event=EventRef(ev, "all")), None
        if op in ("count", "exists"): return Query(op="count", scope=Scope(time or (whole if op == "exists" else rel7), None, conds), event=EventRef(ev, "last")), None

    if act == "context" and c.kind == "event":
        return Query(op="event_context", scope=Scope(whole), event=EventRef(c.registry_key, "last", None, 600, 0)), None

    if act == "why":
        if c.kind == "event": return Query(op="explain_event", scope=Scope(_time(lit, whole)), event=EventRef(c.registry_key, "last")), None
        if numeric: return Query(op="explain_change", target=c.id, scope=Scope(_time(lit, TimeSpec("relative", 14, "day")), None, conds)), None

    return None, "no typed query form is defined for this meaning yet"

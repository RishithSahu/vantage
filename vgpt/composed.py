"""Compositions of typed queries.

The typed IR answers one statistic over one window.  Several ordinary questions are a *composition* of such statistics rather than one more statistic:

  * which day had the highest / lowest average (or peak) X           -> the same typed query per local day, then the arg-extremum
  * how many days did X exceed N                                     -> the same typed query per local day, then a count
  * how much did X change between window A and window B            -> two typed queries, then their difference
  * how much hotter / faster / higher was X than Y                    -> two typed queries (one per quantity), then their difference
  * the range / spread between the highest and the lowest X           -> two typed queries (max and min), then their difference

Nothing here reads the database itself: every number is the typed engine's own answer for an explicitly constructed window, so these operations inherit its
unit handling, conditions, data-quality rules and refusals.  If any part cannot be computed the whole composition declines to answer rather than reporting a partial
result as if it were complete.
"""
import datetime as dt
import re

from . import signal_registry as R
from .query_ir import Query, TimeSpec, Condition
from .query_planner import resolve_time, local_midnight, DAY

HIGH = r"highest|hottest|warmest|fastest|most|maximum|max|peak|biggest|largest|greatest|longest|top|worst|strongest|higher|hotter|warmer"
LOW = r"lowest|coldest|coolest|slowest|least|minimum|min|smallest|shortest|best|weakest|lower|colder|cooler"
THRESH = {"gt": "above", "ge": "at least", "lt": "below", "le": "at most", "eq": "exactly"}


def _result(answer, route, **facts):
    return {"answer": answer, "brief": answer.split("\n", 1)[0], "intent": "ANALYSIS", "facts": dict({"ok": True, "route": route}, **facts)}


def _in_label(label):
    return "in " + label if label.startswith("the ") else label


def _day_label(ts, tz):
    d = dt.datetime.fromtimestamp(ts + tz, dt.timezone.utc)
    return f"{d.strftime('%A')} {d.day} {d.strftime('%b')}"


def _fmt(target, v):
    u = R.CONCEPTS[target].unit
    if u == "%": return f"{v:.1f}%"
    if u in ("rpm", "W"): return f"{v:,.0f} {u}"
    return f"{v:.1f} {u}" + ("" if u else "")


def _value(qc, q):
    """The typed engine's number for q, or (None, reason)."""
    from .query_validator import validate
    if q.scope.time.type != "range":                                                            # an ordinary window goes through the full validator
        v = validate(q, qc.profile)
        if not v.valid or not v.capability.ok: return None, "not computable"
    try: res = qc.executor.run(q.normalized())                                                  # internal windows (one local day) are built here, never taken from text
    except Exception as e: return None, type(e).__name__
    if res.status == "ok" and isinstance(res.value, (int, float)): return float(res.value), ""
    return None, res.reason or res.status


def _days(qc, spec):
    """Local calendar days of a window: [(label_ts, TimeSpec(range))...]; the first / last day may be partial."""
    lo, hi, label = resolve_time(spec, qc.ctx); tz = qc.ctx.tz; d = local_midnight(lo, tz); out = []
    while d < hi and len(out) < 62:
        a, b = max(lo, d), min(hi, d + DAY)
        if b > a: out.append((d, TimeSpec("range", start=a, end=b)))
        d += DAY
    return out, label


def _base(brain, qc, raw):
    from . import nl2ir
    try: p = nl2ir.parse(raw, now=brain.now, tz=brain.tz, profile=qc.profile, state=None)
    except Exception: return None
    return p


def _aggregate_query(target, agg, time, conds):
    q = Query(op="extreme" if agg in ("max", "min") else "aggregate", target=target, agg=agg)
    q.scope.time = time; q.scope.conditions = list(conds)
    return q


def _implicit_measure(t):
    """The quantity that the wording itself names when no signal noun is given: 'fastest day' (speed), 'which day was the battery coolest' (battery temperature), 'which day did I drive the most' (distance).
    -> (concept, statistic) or (None, None)."""
    nouns = [n for n in _NOUNS if re.search(r"\b" + n + r"\b", t)]
    if re.search(r"\b(?:hottest|coldest|warmest|coolest|hotter|colder|warmer|cooler)\b", t):
        for n in nouns:
            c = _concept_for(n, "temperature")
            if c: return c, "mean"
        return None, None
    if re.search(r"\b(?:fastest|slowest|faster|slower)\b", t): return "speed", "mean"
    if re.search(r"\b(?:drive|drove|driven|travel\w*|cover\w*|go|went)\b.{0,25}\b(?:most|farthest|furthest|longest|least|shortest|less|more)\b|\b(?:longest|farthest|furthest|shortest) (?:day|distance)\b", t): return "distance_driven", "sum"
    return None, None


def _argday(brain, qc, t, raw):
    """'which day had the highest average speed', 'on which day last week was the battery hottest', 'which was my fastest day this week', 'which day did I drive the most'."""
    if not re.search(r"\b(?:which|what) (?:day|date|weekday)\b|\bon which day\b|\bwhich of the (?:last |past )?\w+ days\b|\b\w+est day\b|\bthe (?:day|date) (?:with|on which|when|that|of)\b|\bwhich was (?:my|the) \w+ day\b", t): return None
    if re.search(r"\b(?:was|is) (?:the )?(?:last|first) \b", t): return None
    hi_w, lo_w = re.search(r"\b(?:" + HIGH + r")\b", t), re.search(r"\b(?:" + LOW + r")\b", t)
    if not hi_w and not lo_w: return None
    direction = "max" if (hi_w and (not lo_w or hi_w.start() < lo_w.start())) else "min"
    if re.search(r"\b(?:hottest|warmest|fastest|longest|farthest|furthest)\b", t): direction = "max"
    if re.search(r"\b(?:coldest|coolest|slowest|shortest)\b", t): direction = "min"
    plain = re.sub(r"\b(?:which|what) (?:day|date|weekday)\b|\bon which day\b|\bthe (?:day|date) (?:with|on which|when|that|of)\b|\bwhich was (?:my|the) \w+ day\b", "what was", t)
    plain = re.sub(r"\b(?:the )?(?:hottest|coldest|warmest|coolest|fastest|slowest|longest|shortest) day\b", "the day", plain)
    target = stat = None; q0 = None
    p = _base(brain, qc, plain)
    if p is not None and p.query is not None and p.query.target and R.CONCEPTS[p.query.target].dtype == "numeric": q0 = p.query; target = q0.target
    if target is None:
        target, imp_stat = _implicit_measure(t)
        if target is None: return None
        if q0 is None:                                                                               # the window and conditions still come from the typed reading of the sentence
            p2 = _base(brain, qc, "what was the average " + R.label_of(target) + " " + (re.search(r"\b(?:in|over|during|for|within) the (?:last|past) \w+ \w+|\bthis (?:week|month)\b|\blast (?:week|month)\b|\byesterday\b|\bin the past (?:week|month)\b", t) or [""])[0])
            if p2 is None or p2.query is None: return None
            q0 = p2.query; q0.target = target
        stat = imp_stat
    adj_style = re.search(r"\b(?:hottest|coldest|warmest|coolest|fastest|slowest)\b", t) and not re.search(r"\b(?:highest|lowest|peak|maximum|minimum|max|min)\b", t)      # "the hottest day" is the day that was hottest overall
    if stat is None or (adj_style and stat in ("max", "min")): stat = "mean" if (adj_style or re.search(r"\b(?:average|mean|typical|on average|avg)\b", t) or q0.agg == "mean") else direction
    days, label = _days(qc, q0.scope.time)
    if len(days) < 2: return None
    vals = []; part = set()
    for d, spec in days:
        v, _ = _value(qc, _aggregate_query(target, stat, spec, q0.scope.conditions))
        if v is not None:
            vals.append((d, v))
            if spec.end - spec.start < DAY - 60: part.add(d)                                       # the first / last day of a window may be only part of a day
    if len(vals) < 2: return None
    pool = [x for x in vals if x[0] not in part] if (stat == "sum" and len([x for x in vals if x[0] not in part]) >= 2) else vals     # a part day cannot be compared on a total
    best = (max if direction == "max" else min)(pool, key=lambda x: x[1])
    word = ("highest" if direction == "max" else "lowest") + (" average" if stat == "mean" else " total" if stat == "sum" else "")
    cond = "".join(f" {c.predicate.replace('_', ' ')}" if c.predicate else "" for c in q0.scope.conditions)
    brief = f"The day with the {word} {R.label_of(target)}{cond} {label} was {_day_label(best[0], qc.ctx.tz)}, at {_fmt(target, best[1])}."
    kind = {"mean": "average", "max": "highest", "min": "lowest", "sum": "total"}[stat]
    det = [f"Per day ({kind}): " + "; ".join(f"{_day_label(d, qc.ctx.tz)} {_fmt(target, v)}" + (" (part of the day)" if d in part else "") for d, v in sorted(vals)) + "."]
    from . import dialogue_state
    q0.op = "extreme" if stat in ("max", "min") else "aggregate"; q0.agg = stat; q0.target = target; dialogue_state.remember(brain, q0)
    return _result(brief + "\n" + "\n".join(det), "composed_argday", target=target, best_day=best[0], value=best[1], per_day=len(vals))


def _hits_per_day(brain, qc, rest):
    """For a clause such as 'my speed exceed 90 km/h in the last 10 days', 'the car was charging last week', 'i have a hard brake this week':
    -> ([(day, hit, partial)], window label, clause that completes 'On N of M days ...') or None."""
    rest = re.sub(r"^(?:did|was|were|had|has|have|is|are|do|does) ", "", rest.strip())
    p = _base(brain, qc, "what percentage of the time " + rest)                  # a threshold or a state parses as a share-of-time question; only its condition and window are used
    sig = [c for c in p.query.scope.conditions if c.signal] if (p is not None and p.query is not None) else []
    preds = [c for c in p.query.scope.conditions if c.predicate] if (p is not None and p.query is not None) else []
    if not sig and not preds:
        p = _base(brain, qc, "how many " + re.sub(r"^(?:i |there |the car |my car )?(?:have |has |get |got |see |record |recorded )?(?:a |an |any )?", "", rest))
        if p is None or p.query is None or p.query.event is None: return None
        q0 = p.query; days, label = _days(qc, q0.scope.time)
        if len(days) < 2: return None
        hits = []
        for d, spec in days:
            q = Query(op="count", event=q0.event); q.scope.time = spec; q.scope.conditions = list(q0.scope.conditions)
            v, _ = _value(qc, q)
            if v is not None: hits.append((d, v > 0, spec.end - spec.start < DAY - 60))
        return hits, label, f"there was at least one {R.EVENTS[q0.event.type][0]}"
    q0 = p.query; days, label = _days(qc, q0.scope.time)
    if len(days) < 2: return None
    hits = []
    if sig:
        c = sig[0]; target = c.signal; others = [x for x in q0.scope.conditions if x is not c]
        stat = "max" if c.op in ("gt", "ge") else "min" if c.op in ("lt", "le") else "mean"
        thr = c.value
        if c.unit and R.unit_family(c.unit) == R.unit_family(R.CONCEPTS[target].unit): thr = R.convert(c.value, c.unit, R.CONCEPTS[target].unit)
        cmp_ = {"gt": lambda v: v > thr, "ge": lambda v: v >= thr, "lt": lambda v: v < thr, "le": lambda v: v <= thr, "eq": lambda v: v == thr}[c.op]
        for d, spec in days:
            v, _ = _value(qc, _aggregate_query(target, stat, spec, others))
            if v is not None: hits.append((d, bool(cmp_(v)), spec.end - spec.start < DAY - 60))
        unit = c.unit or R.CONCEPTS[target].unit
        return hits, label, f"the {R.label_of(target)} {({'gt': 'went above', 'ge': 'reached at least', 'lt': 'dropped below', 'le': 'dropped to at most', 'eq': 'averaged exactly'})[c.op]} {c.value:g} {unit}"
    for d, spec in days:                                                                           # a named state: some time spent in it that day
        q = Query(op="percent_time"); q.scope.time = spec; q.scope.conditions = list(q0.scope.conditions)
        v, _ = _value(qc, q)
        if v is not None: hits.append((d, v > 0.0, spec.end - spec.start < DAY - 60))
    from .answer_composer import _pred_phrase
    return hits, label, "the car spent time " + " and ".join(_pred_phrase(c) for c in preds)


def _count_days(brain, qc, t, raw):
    """'how many days in the last 10 days did my speed exceed 90 km/h'."""
    if not re.search(r"\bhow many days\b|\bon how many days\b|\bnumber of days\b", t): return None
    rest = re.sub(r"\b(?:on )?how many days\b|\bnumber of days\b", "", t).strip()
    h = _hits_per_day(brain, qc, rest)
    if h is None or len(h[0]) < 2: return None
    hits, label, clause = h
    n_hit = sum(1 for _, x, _ in hits if x)
    part = " (the first or last day is only part of a day)" if any(pt for _, _, pt in hits) else ""
    return _result(f"On {n_hit} of {len(hits)} calendar days {_in_label(label)} {clause}{part}.", "composed_count_days", n=n_hit, of=len(hits))


def _which_days(brain, qc, t, raw):
    """'on which days last week did I charge', 'which days did the motor go above 60 degrees'."""
    if not re.search(r"\bon which days\b|\bwhich days\b|\bwhat days\b|\bwhich dates\b", t): return None
    rest = re.sub(r"\b(?:on )?(?:which|what) (?:days|dates)\b", "", t).strip()
    rest = re.sub(r"\b(?:did|do|does) (?:i|we|the car|my car) (charge|park|drive)\b", lambda m: {"charge": "the car was charging", "park": "the car was parked", "drive": "the car was driving"}[m.group(1)], rest)
    rest = re.sub(r"\b(?:did|do|does) (?:i|we) (?:have|get|see) (?:a |an |any )?", "i have a ", rest)
    h = _hits_per_day(brain, qc, rest)
    if h is None or len(h[0]) < 2: return None
    hits, label, clause = h
    days = [d for d, x, _ in hits if x]
    if not days: return _result(f"On none of the {len(hits)} calendar days {_in_label(label)} {clause}.", "composed_which_days", n=0)
    names = ", ".join(_day_label(d, qc.ctx.tz) for d in days)
    return _result(f"{clause[0].upper() + clause[1:]} on {len(days)} of {len(hits)} calendar days {_in_label(label)}: {names}.", "composed_which_days", n=len(days), of=len(hits))


def _group_counts(brain, qc, t, raw):
    """'how many hard brakes on the highway versus the city last week': one typed count per named state."""
    from . import nl2ir
    if not re.search(r"\b(?:versus|vs|compared (?:to|with)|against|or|and|than|split by|broken down by)\b", t): return None
    S = nl2ir._S(nl2ir.normalise(t)); names = []
    for _, name in nl2ir._predicates(S, []):
        if name not in names and name in ("city", "highway", "charging", "parked", "driving"): names.append(name)
    if len(names) != 2: return None
    stripped = re.sub(r"\b(?:on |in |during |while )?(?:the )?(?:highway|motorway|city|town|charging|parked|driving)(?: driving| roads?| streets?)?\b", " ", t)
    stripped = re.sub(r"\b(?:versus|vs\.?|compared (?:to|with)|against|or|and|than|split by|broken down by)\b", " ", stripped)
    p = _base(brain, qc, re.sub(r"\s+", " ", stripped).strip())
    if p is None or p.query is None or p.query.event is None: return None
    q0 = p.query; out = []
    for nm in names:
        q = Query(op="count", event=q0.event); q.scope.time = q0.scope.time; q.scope.conditions = [Condition(predicate=nm)]
        v, _ = _value(qc, q)
        if v is None: return None
        out.append((nm, int(v)))
    from .answer_composer import _pred_phrase
    lab = R.EVENTS[q0.event.type][0]; label = resolve_time(q0.scope.time, qc.ctx)[2]
    return _result(f"{label.capitalize() if not label.startswith('the ') else 'Over ' + label}: " + "; ".join(f"{n} {lab}{'' if n == 1 else 's'} {_pred_phrase(Condition(predicate=nm))}" for nm, n in out) + ".", "composed_group_counts", counts=dict(out))


def _extent(brain, qc, t, raw):
    """'the difference between the highest and the lowest battery temperature over the last 10 days'."""
    if not (re.search(r"\b(?:range|spread|difference|gap|swing|variation)\b", t) and re.search(r"\b(?:highest|maximum|max|peak)\b", t) and re.search(r"\b(?:lowest|minimum|min)\b", t)):
        if not re.search(r"\bhow much (?:did|does|do|has|have)\b.*\b(?:vary|varied|swing|swung|fluctuate\w*)\b", t): return None
    plain = re.sub(r"\b(?:the )?(?:range|spread|difference|gap|swing|variation)(?: (?:between|of|in))?\b", " ", t)
    plain = re.sub(r"\b(?:the )?(?:highest|maximum|max|peak)\b(?: and (?:the )?(?:lowest|minimum|min))?", " ", plain); plain = re.sub(r"\b(?:and )?(?:the )?(?:lowest|minimum|min)\b", " ", plain)
    p = _base(brain, qc, "what was the highest " + re.sub(r"^(?:what (?:is|was|are|were) |how (?:big|large|much) (?:is|was) )", "", plain.strip()))
    if p is None or p.query is None or not p.query.target or R.CONCEPTS[p.query.target].dtype != "numeric": return None
    q0 = p.query; hi_v, _ = _value(qc, _aggregate_query(q0.target, "max", q0.scope.time, q0.scope.conditions)); lo_v, _ = _value(qc, _aggregate_query(q0.target, "min", q0.scope.time, q0.scope.conditions))
    if hi_v is None or lo_v is None: return None
    lab = R.label_of(q0.target); u = R.CONCEPTS[q0.target].unit; _, _, period = resolve_time(q0.scope.time, qc.ctx)
    return _result(f"The {lab} varied by {_fmt(q0.target, hi_v - lo_v)} {period}: from {_fmt(q0.target, lo_v)} (lowest) to {_fmt(q0.target, hi_v)} (highest).", "composed_extent", target=q0.target, high=hi_v, low=lo_v)


_ADJ_TEMP = r"hotter|colder|warmer|cooler|higher|lower"
_NOUNS = ("motor", "battery", "pack", "coolant", "outside", "ambient", "cabin", "inverter", "charger")


def _concept_for(noun, kind):
    for key in (f"{noun} {kind}", f"{noun} temp"):
        c = R.LEXICON.get(key)
        if c: return sorted(c)[0]
    return None


def _signal_gap(brain, qc, t, raw):
    """'how much hotter was the motor than the battery on average yesterday', 'how much faster was the highway average than the city average'."""
    m = re.search(r"\bhow much (" + _ADJ_TEMP + r"|faster|slower|more|less)\b(?: (?:was|is|were|are|did|does))? (?:the |my )?(.+?) (?:than|compared (?:to|with)|relative to) (?:the |my )?(.+)", t)
    if not m: return None
    adj, left, right = m.group(1), m.group(2), m.group(3)
    tail = re.search(r"\b(?:on average|yesterday|today|last|over|during|in the last|this week|this month|in the past|for the last|all week|on \w+day)\b.*$", right)
    window = tail.group(0) if tail else ""
    right_core = right[: tail.start()].strip() if tail else right.strip()
    left_t = left.strip(); right_t = right_core.strip(" ?.!")
    for noun_l in _NOUNS:
        pass
    kind = "temperature" if adj in _ADJ_TEMP.split("|") else "speed" if adj in ("faster", "slower") else None
    ca = cb = None
    if kind == "temperature":
        na = next((n for n in _NOUNS if re.search(r"\b" + n + r"\b", left_t)), None); nb = next((n for n in _NOUNS if re.search(r"\b" + n + r"\b", right_t)), None)
        if na and nb and na != nb: ca, cb = _concept_for(na, "temperature"), _concept_for(nb, "temperature")
    if not ca or not cb:
        for x in (left_t, right_t):
            pass
        return None
    stat = "max" if re.search(r"\b(?:highest|peak|maximum|max)\b", t) else "min" if re.search(r"\b(?:lowest|minimum|min)\b", t) else "mean"
    rest = (window + " " + (left.split(" on average")[0] if "on average" in left else "")).strip()
    p = _base(brain, qc, "what was the average " + R.label_of(ca) + " " + (window or "over the last 7 days"))
    if p is None or p.query is None: return None
    time = p.query.scope.time; conds = p.query.scope.conditions
    va, _ = _value(qc, _aggregate_query(ca, stat, time, conds)); vb, _ = _value(qc, _aggregate_query(cb, stat, time, conds))
    if va is None or vb is None: return None
    _, _, period = resolve_time(time, qc.ctx); d = va - vb; u = R.CONCEPTS[ca].unit
    word = {"mean": "on average", "max": "at its highest", "min": "at its lowest"}[stat]
    if abs(d) < 0.05: return _result(f"The {R.label_of(ca)} and the {R.label_of(cb)} were the same {word} {period} ({_fmt(ca, va)}).", "composed_signal_gap", a=ca, b=cb, diff=d)
    hot = "higher" if d > 0 else "lower"
    return _result(f"The {R.label_of(ca)} was {abs(d):.1f} {u} {hot} than the {R.label_of(cb)} {word} {period}: {_fmt(ca, va)} against {_fmt(cb, vb)}.", "composed_signal_gap", a=ca, b=cb, diff=d)


_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fourteen": 14, "thirty": 30}
_WIN_RX = re.compile(r"\b(?:(?:the )?(?:week|7 days|seven days) before(?: that| last week| the last)?|(?:the )?previous week|(?:the )?week before last|two weeks ago|(?:the )?(?:month|30 days) before(?: that| last month)?|(?:the )?previous month|(?:the )?(\w+) days before(?: that)?|"
                     r"(?:the )?(?:last|past) (\w+) (day|week|month)s?|last week|this week|last month|this month|yesterday|today|(?:the )?day before(?: yesterday)?|day before yesterday)\b")


def _windows(t, qc):
    """The distinct time windows named in a sentence, in order of mention, as TimeSpecs.  Offsets ("the week before") are anchored on the window mentioned before them."""
    now = qc.ctx.now; out = []; prev_len = None
    for m in _WIN_RX.finditer(t):
        w = m.group(0)
        # Only explicit anaphora ("before that") is anchored to the
        # preceding window.  Unanchored "two weeks ago" remains an offset
        # from now, preserving its ordinary meaning.
        anchor_start = None
        if out and "before that" in w:
            anchor_start = resolve_time(out[-1], qc.ctx)[0]
        if re.search(r"(?:week|7 days|seven days) before|previous week|week before last|two weeks ago", w):
            end = anchor_start if anchor_start is not None else now - 7 * DAY
            spec = TimeSpec("range", start=end - 7 * DAY, end=end)
        elif re.search(r"(?:month|30 days) before|previous month", w):
            end = anchor_start if anchor_start is not None else now - 30 * DAY
            spec = TimeSpec("range", start=end - 30 * DAY, end=end)
        elif m.group(1) and m.group(1) in _NUM or (m.group(1) or "").isdigit():
            n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
            end = anchor_start if anchor_start is not None else now - n * DAY
            spec = TimeSpec("range", start=end - n * DAY, end=end)
        elif m.group(2) and m.group(3):
            n = int(m.group(2)) if m.group(2).isdigit() else _NUM.get(m.group(2))
            if n is None: continue
            spec = TimeSpec("relative", n, m.group(3))
        elif w == "last week": spec = TimeSpec("relative", 7, "day")
        elif w == "last month": spec = TimeSpec("relative", 30, "day")
        elif w == "this week": spec = TimeSpec("calendar", name="this_week")
        elif w == "this month": spec = TimeSpec("calendar", name="this_month")
        elif w == "yesterday": spec = TimeSpec("calendar", name="yesterday")
        elif w == "today": spec = TimeSpec("calendar", name="today")
        elif "day before" in w: spec = TimeSpec("calendar", name="day_before_yesterday")
        else: continue
        if spec.to_text() not in [x.to_text() for x in out]: out.append(spec)
    if re.search(r"\bthis week\b", t) and re.search(r"\blast week\b", t):                  # "this week against last week": the calendar week before the one in progress, not an overlapping 7 days
        start = resolve_time(TimeSpec("calendar", name="this_week"), qc.ctx)[0]
        out = [TimeSpec("range", start=start - 7 * DAY, end=start) if (x.type == "relative" and x.value == 7 and x.unit == "day") else x for x in out]
    if re.search(r"\bthis month\b", t) and re.search(r"\blast month\b", t):
        start = resolve_time(TimeSpec("calendar", name="this_month"), qc.ctx)[0]; prev = resolve_time(TimeSpec("calendar", name="last_month"), qc.ctx)
        out = [TimeSpec("range", start=prev[0], end=prev[1]) if (x.type == "relative" and x.value == 30 and x.unit == "day") else x for x in out]
    return out


def _window_change(brain, qc, t, raw):
    """'by how much did my average speed change between last week and the week before'."""
    if not re.search(r"\b(?:change[ds]?|differ(?:ence|ent|s)?|increase[ds]?|decrease[ds]?|gone (?:up|down)|went (?:up|down)|risen|fallen|dropped|improved|worse|better|compare[ds]?|versus|vs|more|less|higher|lower|hotter|colder|warmer|cooler|faster|slower)\b", t): return None
    from . import nl2ir
    if len({nm for _, nm in nl2ir._predicates(nl2ir._S(nl2ir.normalise(t)), [])}) >= 2: return None          # "while charging versus while driving": two states of one window
    uniq = _windows(t, qc)
    if len(uniq) < 2 and re.search(r"\bthan \d|\d ?(?:km/h|kph|mph|%|degrees?|deg|°|c\b|kpa|psi|kw|rpm|km\b|minutes?|hours?)", t): return None      # a stated threshold ("faster than 100 km/h") is a condition, not a second period
    if len(uniq) == 1 and uniq[0].type == "range" and uniq[0].end == qc.ctx.now - (uniq[0].end - uniq[0].start) and (uniq[0].end - uniq[0].start) in (7 * DAY, 30 * DAY):
        uniq = [TimeSpec("relative", round((uniq[0].end - uniq[0].start) / DAY), "day"), uniq[0]]                    # "compared to the week before": the window just ended is the one being compared
    if len(uniq) == 1 and re.search(r"\b(?:compared (?:to|with)|than|versus|vs\.?|against|since)\b", t) and uniq[0].type == "relative" and (uniq[0].value, uniq[0].unit) in ((7, "day"), (30, "day")) and re.search(r"\blast (?:week|month)\b", t):
        cur, prev = ("this_week", "last_week") if uniq[0].value == 7 else ("this_month", "last_month")      # "compared to last month": the month so far against the month before it
        b_ = resolve_time(TimeSpec("calendar", name=prev), qc.ctx)
        uniq = [TimeSpec("calendar", name=cur), TimeSpec("range", start=b_[0], end=b_[1])]
    if len(uniq) != 2: return None
    stripped = _WIN_RX.sub(" ", t)
    stripped = re.sub(r"\b(?:between|compared (?:to|with)|versus|vs\.?|and|from|to|than|by how much|how much|did|does|has|have|change[ds]?|differ\w*|increase[ds]?|decrease[ds]?)\b", " ", stripped)
    p = _base(brain, qc, "what was the " + re.sub(r"\s+", " ", stripped).strip() + " yesterday")
    if p is None or p.query is None or not p.query.target or R.CONCEPTS[p.query.target].dtype != "numeric":
        imp, _st = _implicit_measure(t)
        if imp is None or imp == "distance_driven": return None
        p = _base(brain, qc, "what was the average " + R.label_of(imp) + " yesterday")
        if p is None or p.query is None: return None
        p.query.target = imp
    q0 = p.query; stat = "max" if re.search(r"\b(?:highest|maximum|max|peak)\b", t) else "min" if re.search(r"\b(?:lowest|minimum|min)\b", t) else "mean"
    ta, tb = uniq
    va, _ = _value(qc, _aggregate_query(q0.target, stat, ta, q0.scope.conditions)); vb, _ = _value(qc, _aggregate_query(q0.target, stat, tb, q0.scope.conditions))
    la = _label(ta, qc); lb = _label(tb, qc)
    if va is None or vb is None:
        miss = la if va is None else lb
        return _result(f"I can't compare the two periods: there is no recorded {R.label_of(q0.target)} {miss}, so I won't offer a difference.", "composed_window_change", target=q0.target, missing=miss)
    u = R.CONCEPTS[q0.target].unit; d = va - vb; word = {"mean": "average", "max": "highest", "min": "lowest"}[stat]
    pct = f" ({d / vb * 100:+.0f}%)" if vb not in (0, 0.0) and u not in ("%",) else ""
    up = re.search(r"\b(?:higher|hotter|warmer|faster|more|greater|larger|bigger|gone up|go up|increased?|risen|rose|up)\b", t); down = re.search(r"\b(?:lower|colder|cooler|slower|less|smaller|gone down|go down|decreased?|fallen|fell|dropped|down)\b", t)
    polar = re.match(r"^(?:is|was|has|have|did|does|are|were|do|can)\b", t.strip()) and (up or down)
    verdict = ""
    if polar:
        same = abs(d) < max(0.05, abs(vb) * 0.005)
        if same: verdict = "No, they were essentially the same. "
        else: verdict = ("Yes. " if ((d > 0) == bool(up and not down)) else "No. ")
    notes = [x for x in (_coverage_note(qc, ta, la), _coverage_note(qc, tb, lb)) if x]
    return _result(verdict + f"The {word} {R.label_of(q0.target)} was {_fmt(q0.target, va)} {la} and {_fmt(q0.target, vb)} {lb}: a difference of {abs(d):.1f} {u}, {'higher' if d > 0 else 'lower'} {la}{pct}." + ("".join(" " + x for x in notes)), "composed_window_change", target=q0.target, a=va, b=vb)


def _coverage_note(qc, spec, label):
    """A comparison over a period that is mostly not recorded must say so (a month with three recorded days is not a month)."""
    lo, hi, _ = resolve_time(spec, qc.ctx); hi = min(hi, qc.ctx.now + 1); tz = qc.ctx.tz
    span = max(1, round((hi - lo) / DAY))
    if span < 3: return ""
    try: n = qc.conn.execute("SELECT COUNT(DISTINCT (ts + ?) / ?) FROM sig_1m WHERE ts >= ? AND ts < ?", (tz, DAY, lo, hi)).fetchone()[0]
    except Exception: return ""
    return f"(Only {n} of the {span} days {label} have recorded data.)" if n < 0.6 * span else ""


def _label(spec, qc):
    if spec.type == "range":
        lo, hi = int(spec.start), int(spec.end); tz = qc.ctx.tz
        f = lambda x: (lambda d: f"{d.day} {d.strftime('%b')}")(dt.datetime.fromtimestamp(x + tz, dt.timezone.utc))
        return f"from {f(lo)} to {f(hi - 1)}"
    return resolve_time(spec, qc.ctx)[2]


def _exists(brain, qc, t, raw):
    """'did I ever go over 100 km/h yesterday', 'has the battery temperature been below 15 degrees this week': an existence question about a threshold."""
    if not re.match(r"^(?:did|have|has|was|were|is|are|do|does|could|can)\b", t.strip()): return None
    if re.search(r"\bhow\b|\bwhen\b|\bwhat\b|\bwhich\b|\bwhy\b", t): return None
    if not re.search(r"\b(?:above|over|exceed\w*|below|under|beyond|more than|less than|higher than|lower than|at least|at most|reach\w*|hit|top\w*|pass\w*)\b", t): return None
    ever = bool(re.search(r"\bever\b|\bat any (?:point|time)\b|\bon record\b", t))
    # "was it never above X?" asks whether the ordinary existence claim is
    # false.  Keep that polarity as an explicit semantic property instead of
    # letting a later maximum-speed fallback answer the opposite question.
    negated = bool(re.search(r"\bnever\b", t))
    rest = re.sub(r"\b(?:ever|at any point|at any time|on record)\b", " ", t.strip())
    rest = re.sub(r"^(?:did|have|has|was|were|is|are|do|does|could|can) (?:i |we |you |the car |my car |it |there |my |the )?", "", rest)
    rest = re.sub(r"\bnever\b", " ", rest)
    p = _base(brain, qc, "what percentage of the time was " + rest)
    if p is None or p.query is None: return None
    sig = [c for c in p.query.scope.conditions if c.signal]
    if not sig: return None
    c = sig[0]; target = c.signal; others = [x for x in p.query.scope.conditions if x is not c]
    no_window = any("No period given" in n for n in (p.query.notes or []))
    time = TimeSpec("all") if (no_window and ever) else p.query.scope.time
    hi = c.op in ("gt", "ge", "eq")
    q = Query(op="extreme", target=target, agg="max" if hi else "min"); q.scope.time = time; q.scope.conditions = others
    r = qc.execute(q)
    if r.status != "answered" or r.result is None or not isinstance(r.result.value, (int, float)):
        return _result(r.answer.text, "composed_exists", ok=False) if r.status in ("no_data", "missing_data") and getattr(r, "answer", None) else None
    thr = c.value
    if c.unit and R.unit_family(c.unit) == R.unit_family(R.CONCEPTS[target].unit): thr = R.convert(c.value, c.unit, R.CONCEPTS[target].unit)
    v = float(r.result.value)
    ok = {"gt": v > thr, "ge": v >= thr, "lt": v < thr, "le": v <= thr, "eq": abs(v - thr) < 1e-9}[c.op]
    unit = c.unit or R.CONCEPTS[target].unit
    word = THRESH[c.op]
    head = r.answer.text.split(chr(10) + "Based on")[0].strip()
    if negated:
        verdict = (f"No - it did go {word} {c.value:g} {unit}. " if ok
                   else f"Yes. It never went {word} {c.value:g} {unit}. ")
    else:
        verdict = "Yes. " if ok else f"No - it never went {word} {c.value:g} {unit}. "
    from . import dialogue_state
    return _result(verdict + head, "composed_exists", yes=bool(not ok if negated else ok), value=v, target=target, negated=negated)


_CLOCK_RX = re.compile(r"\b(?:between|from) (\d{1,2})(?::(\d{2}))? ?(am|pm)? ?(?:and|to|-|until|till) ?(\d{1,2})(?::(\d{2}))? ?(am|pm)\b|\b(?:at|around|about) (\d{1,2})(?::(\d{2}))? ?(am|pm)\b|\b(after|since|before) (\d{1,2})(?::(\d{2}))? ?(am|pm)\b|\bin the (morning|afternoon|evening)\b|\b(?:at night|overnight|during the night)\b")


def _clock_slots(m):
    """The matched clock phrase -> (start minute, end minute) of the local day, or a pair of slots across midnight; plus a readable label."""
    def mins(h, mi, ap):
        h = int(h) % 12 if ap else int(h); return (h + (12 if ap == "pm" else 0)) * 60 + int(mi or 0)
    g = m.groups()
    if g[3] is not None:                                                                       # between A and B
        ap2 = g[5]; ap1 = g[2] or ap2
        a, b = mins(g[0], g[1], ap1), mins(g[3], g[4], ap2)
        if b <= a: b += 24 * 60
        return [(a, b)], f"between {a // 60 % 24:02d}:{a % 60:02d} and {b // 60 % 24:02d}:{b % 60:02d}"
    if g[6] is not None:                                                                       # at 8 am: that hour
        a = mins(g[6], g[7], g[8]); return [(a, a + 60)], f"from {a // 60:02d}:{a % 60:02d} to {(a + 60) // 60 % 24:02d}:{(a + 60) % 60:02d}"
    if g[9] is not None:                                                                       # after / before
        a = mins(g[10], g[11], g[12])
        if g[9] == "before": return [(0, a)], f"before {a // 60:02d}:{a % 60:02d}"
        return [(a, 24 * 60)], f"after {a // 60:02d}:{a % 60:02d}"
    if g[13] is not None:
        a, b = {"morning": (6, 12), "afternoon": (12, 17), "evening": (17, 21)}[g[13]]; return [(a * 60, b * 60)], f"in the {g[13]} ({a:02d}:00 to {b:02d}:00)"
    return [(0, 6 * 60), (21 * 60, 24 * 60)], "at night (21:00 to 06:00)"


def _clock(brain, qc, t, raw):
    """A statistic over part of each day: 'the average speed between 8 am and 10 am yesterday', 'the highest battery temperature in the evening last week'.  Evaluated per local day on that slot and combined."""
    m = _CLOCK_RX.search(t)
    if not m: return None
    slots, label = _clock_slots(m)
    plain = (t[:m.start()] + " " + t[m.end():]).strip()
    p = _base(brain, qc, plain)
    if p is None or p.query is None or not p.query.target or p.query.op not in ("aggregate", "extreme", "count") and not p.query.event: return None
    q0 = p.query; days, wlabel = _days(qc, q0.scope.time)
    if not days: return None
    tz = qc.ctx.tz; parts = []
    agg = q0.agg or "mean"
    if q0.op == "aggregate" and agg not in ("mean", "max", "min", "sum"): return None
    for d, spec in days:
        for a, b in slots:
            lo, hi = max(spec.start, d + a * 60), min(spec.end, d + b * 60)
            if hi <= lo: continue
            q = Query(op="extreme" if agg in ("max", "min") else "aggregate", target=q0.target, agg=agg); q.scope.time = TimeSpec("range", start=lo, end=hi); q.scope.conditions = list(q0.scope.conditions)
            try: res = qc.executor.run(q.normalized())
            except Exception: continue
            if res.status == "ok" and isinstance(res.value, (int, float)): parts.append((float(res.value), int(res.samples or 1)))
    if not parts: return _result(f"There is no recorded {R.label_of(q0.target)} {label} {wlabel}.", "composed_clock", ok=False)
    if agg == "mean": val = sum(v * n for v, n in parts) / sum(n for _, n in parts)
    elif agg == "max": val = max(v for v, _ in parts)
    elif agg == "min": val = min(v for v, _ in parts)
    else: val = sum(v for v, _ in parts)
    word = {"mean": "average", "max": "highest", "min": "lowest", "sum": "total"}[agg]
    return _result(f"The {word} {R.label_of(q0.target)} {label} {wlabel} was {_fmt(q0.target, val)} (from {sum(n for _, n in parts):,} samples).", "composed_clock", target=q0.target, value=val)


_SESSION_RX = re.compile(r"\b(?:last|latest|previous|most recent|final|first|earliest)\b(?: \w+)? (?:charge|charging session|charge session|charging|plug[- ]?in)\b|\bcharged (?:last|most recently)\b|\bwhen i (?:last )?charged\b|\b(?:charge|charging session) (?:before|prior to) (?:that|the last)\b")
_BEFORE_RX = re.compile(r"\b(?:compared (?:to|with)|versus|vs\.?|against|than|and|relative to)\b.{0,25}\b(?:the )?(?:one|charge|session|charging session|time) (?:before|prior|previously)\b|\b(?:the )?(?:one|charge|session) before (?:that|it|the last)\b|\bprevious (?:one|charge|session)\b.*\b(?:compared|versus|vs)\b|\b(?:compared (?:to|with)|versus|vs\.?|against|than) (?:the |my )?(?:previous|prior|earlier) (?:one|charge|session|charging session)\b")


def _fmt_dur(sec):
    m = int(round(sec / 60.0))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def _session_attr(brain, qc, t, raw):
    """Attributes of a particular charging session: 'how long did my last charge take compared to the one before', 'how much did the battery temperature rise during my last charge',
    'what was the temperature outside when I last charged'.  The session comes from the sessions ledger; signal statistics inside it are typed queries over its exact interval."""
    if not _SESSION_RX.search(t): return None
    if "charges" not in qc.profile.tables: return None
    two = bool(_BEFORE_RX.search(t))
    first = bool(re.search(r"\b(?:first|earliest)\b", t))
    rows = qc.conn.execute("SELECT start, end, kind, kw, kwh, soc0, soc1, grid_kwh FROM charges WHERE end <= ? ORDER BY start " + ("ASC" if first else "DESC") + " LIMIT 2", (qc.ctx.now + 1,)).fetchall()
    if not rows or (two and len(rows) < 2): return None
    sess = rows[:2] if two else rows[:1]
    tz = qc.ctx.tz
    def stamp(ts):
        d = dt.datetime.fromtimestamp(ts + tz, dt.timezone.utc); return f"{d.day} {d.strftime('%b')}, {d:%H:%M}"
    names = ["the last charge", "the one before"] if not first else ["the first charge", "the next one"]
    temp = re.search(r"\b(?:temperature|temp|hot|hotter|warm|cold)\b", t)
    stat = "delta" if re.search(r"\b(?:rise|rose|risen|increase\w*|go up|went up|climb\w*|change[ds]?|gain\w*|drop\w*|fall\w*|fell)\b", t) else "max" if re.search(r"\b(?:highest|peak|maximum|max|hottest)\b", t) else "min" if re.search(r"\b(?:lowest|minimum|min|coldest)\b", t) else "mean"
    out = []; vals = []; label = None
    for s in sess:
        start, end, kind, kw, kwh, soc0, soc1, grid = s
        if temp:
            noun = next((n for n in _NOUNS if re.search(r"\b" + n + r"\b", t)), None)
            if re.search(r"\b(?:outside|ambient|weather|external|air)\b", t): concept = "ambient_temperature"
            else: concept = _concept_for(noun or "battery", "temperature") or "battery_temperature"
            q = Query(op="extreme" if stat in ("max", "min") else "aggregate", target=concept, agg=stat); q.scope.time = TimeSpec("range", start=int(start), end=int(end) + 1)
            v, _ = _value(qc, q)
            if v is None: return None
            vals.append(v); label = (f"{stat if stat != 'mean' else 'average'} {R.label_of(concept)}", R.CONCEPTS[concept].unit, concept)
        elif re.search(r"\b(?:how long|duration|take|took|lasted?|time)\b", t) and not re.search(r"\bwhat time\b", t): vals.append(end - start); label = ("duration", "s", None)
        elif re.search(r"\b(?:energy|kwh|added|delivered|how much (?:power|electricity)|put in|gained)\b", t) and not re.search(r"\bsoc\b|state of charge", t): vals.append(kwh); label = ("energy added to the battery", "kWh", None)
        elif re.search(r"\b(?:power|kw|rate|speed|how fast)\b", t):
            if re.search(r"\b(?:average|mean|typical|on average)\b", t) and end > start: vals.append(kwh / ((end - start) / 3600.0)); label = ("average charging power (energy added over the session's duration)", "kW", None)
            else: vals.append(kw); label = ("peak charging power", "kW", None)
        elif re.search(r"\b(?:soc|state of charge|battery level|percent\w*|%|how full|charge level)\b", t): vals.append(soc1 - soc0); label = ("state-of-charge gain", "%", None)
        else: vals.append(start); label = ("start", "ts", None)
    what, unit, concept = label
    def show(v):
        if unit == "s": return _fmt_dur(v)
        if unit == "ts": return stamp(v)
        return f"{v:.1f} {unit}" if what != "state-of-charge gain" else f"{v:.0f} percentage points"
    if re.search(r"\bhow long ago\b|\bhow (?:many|much) (?:minutes|hours|days)? ?ago\b|\bhow recently\b|\bplug(?:ged)?[- ]?in\b", t) and not temp:
        ago = lambda sec: (f"{int(sec // 86400)} days {int(sec % 86400 // 3600)} h" if sec >= 86400 else _fmt_dur(sec))
        parts_ = [f"{n.capitalize()} was plugged in {ago(qc.ctx.now - sx[0])} ago (on {stamp(sx[0])}) and finished {ago(qc.ctx.now - sx[1])} ago" for n, sx in zip(names, sess)]
        return _result("; ".join(parts_) + ".", "last_charge", n=len(sess))
    if what == "start":
        def one(sx):
            st, en, kind_, kw_, kwh_, s0_, s1_, _g = sx
            return f"started on {stamp(st)} and ended at {stamp(en)[-5:]} ({_fmt_dur(en - st)}): {s0_:.0f}% to {s1_:.0f}%, +{kwh_:.1f} kWh at up to {kw_:.0f} kW"
        txt = "; ".join(f"{n.capitalize()} {one(sx)}" for n, sx in zip(names, sess)) + "."
        return _result(txt, "last_charge", n=len(vals))
    if concept and stat == "delta": what = f"change in {R.label_of(concept)}"
    if len(sess) == 1:
        s0 = sess[0]
        return _result(f"The {what} during {names[0]} ({stamp(s0[0])} to {stamp(s0[1])[-5:]}) was {show(vals[0])}.", "composed_session", value=vals[0], what=what)
    d = vals[0] - vals[1]
    diff = _fmt_dur(abs(d)) if unit == "s" else (f"{abs(d):.1f} {unit}" if unit != "%" else f"{abs(d):.0f} percentage points")
    cmpw = ("longer" if d > 0 else "shorter") if unit == "s" else ("higher" if d > 0 else "lower")
    tail = "the same" if abs(d) < 1e-9 else f"{diff} {cmpw}"
    return _result(f"The {what} was {show(vals[0])} during {names[0]} ({stamp(sess[0][0])}) and {show(vals[1])} during {names[1]} ({stamp(sess[1][0])}): {'identical' if tail == 'the same' else tail}.", "composed_session", a=vals[0], b=vals[1], what=what)


_ADJ_PRESS = r"higher|lower|greater|more|less|bigger|smaller|larger|faster|slower"


def _vs_normal(brain, qc, t, raw):
    """'was the coolant flow lower than normal yesterday', 'was my motor temperature unusual last week': a past window against this vehicle's own daily averages in the 14 days before it."""
    if not re.search(r"\b(?:normal|usual|abnormal|unusual\w*|as expected|than usual|than normal|out of the ordinary|outlier)\b", t) and not (re.match(r"^(?:is|was|were|are|did|has|have|does)\b", t.strip()) and re.search(r"\btypical\b", t)): return None   # "my typical X" is a statistic, not a normality question
    if re.search(r"\b(?:chang\w*|trend\w*|creep\w*|drift\w*|increas\w*|decreas\w*|rising|falling)\b", t): return None
    if re.search(r"\b(?:right now|currently|at the moment|now)\b", t): return None
    plain = re.sub(r"\b(?:lower|higher|less|more|hotter|colder|faster|slower|warmer|cooler|bigger|smaller|better|worse) than (?:normal|usual|typical|expected)\b|\b(?:abnormal|unusual\w*|normal|usual|typical|as expected|expected|out of the ordinary)\b", " ", t)
    plain = re.sub(r"^(?:was|is|were|are|has|have|did|does|do)\b", "what was", plain.strip())
    p = _base(brain, qc, re.sub(r"\s+", " ", plain))
    if p is None or p.query is None or not p.query.target or R.CONCEPTS[p.query.target].dtype != "numeric" or p.query.scope.time.type == "current": return None
    q0 = p.query; stat = "max" if re.search(r"\b(?:peak|highest|maximum|max)\b", t) else "min" if re.search(r"\b(?:lowest|minimum|min)\b", t) else "mean"
    lo, hi, label = resolve_time(q0.scope.time, qc.ctx)
    cur, _ = _value(qc, _aggregate_query(q0.target, stat, q0.scope.time, q0.scope.conditions))
    if cur is None: return None
    base = []
    d = local_midnight(lo, qc.ctx.tz) - DAY
    for _ in range(14):
        v, _ = _value(qc, _aggregate_query(q0.target, stat, TimeSpec("range", start=d, end=d + DAY), q0.scope.conditions))
        if v is not None: base.append(v)
        d -= DAY
    if len(base) < 5: return _result(f"I can't judge whether that was normal: fewer than 5 of the 14 days before {label} have recorded {R.label_of(q0.target)}.", "composed_vs_normal", ok=False)
    mean = sum(base) / len(base); sd = (sum((x - mean) ** 2 for x in base) / (len(base) - 1)) ** 0.5
    n_days = max(1, round((hi - lo) / DAY)); se = sd / (n_days ** 0.5) if stat == "mean" else sd
    z = (cur - mean) / se if se > 1e-9 else 0.0
    u = R.CONCEPTS[q0.target].unit; lab = R.label_of(q0.target); word = {"mean": "average", "max": "highest", "min": "lowest"}[stat]
    kind = "unusually high" if z > 1.5 else "unusually low" if z < -1.5 else "within its usual range"
    asked_low = re.search(r"\b(?:lower|less|colder|slower|cooler|smaller|worse)\b", t); asked_high = re.search(r"\b(?:higher|more|hotter|faster|warmer|bigger)\b", t)
    lead = ""
    if asked_low or asked_high:
        yes = (z < -1.5) if asked_low else (z > 1.5)
        lead = "Yes. " if yes else "No. "
    return _result(lead + f"The {word} {lab} {label} was {_fmt(q0.target, cur)}, {kind} for this vehicle: in the {len(base)} days before it the daily {word} was {_fmt(q0.target, mean)} (typical day-to-day spread +/-{sd:.1f} {u}).", "composed_vs_normal", value=cur, mean=mean, sd=sd, z=z)


_POS = {"fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right"}


def _wheel_values(qc, base, stat, time, conds):
    out = {}
    for pos in _POS:
        c = f"{base}_{pos}"
        if c not in R.CONCEPTS: return None
        if time.type == "current":
            q = Query(op="current_value", target=c); r = qc.execute(q)
            v = float(r.result.value) if r.status == "answered" and r.result is not None and isinstance(r.result.value, (int, float)) else None
        else: v, _ = _value(qc, _aggregate_query(c, stat, time, conds))
        if v is None: return None
        out[pos] = v
    return out


def _wheel(brain, qc, t, raw):
    """'which tyre has the lowest pressure right now', 'is the front left tyre pressure lower than the front right', 'which wheel was fastest yesterday': one typed value per wheel position."""
    base = "tyre_pressure" if re.search(r"\b(?:tyre|tire|tyres|tires)\b.*\bpressure\b|\bpressure\b", t) else "wheel_speed" if re.search(r"\bwheel speed\b", t) else None
    if base is None: return None
    m_which = re.search(r"\b(?:which|what) (?:tyre|tire|wheel)\b.*\b(" + HIGH + "|" + LOW + r")\b", t)
    m_pair = re.search(r"\b(front|rear)[ -]*(left|right)\b.*?\b(" + _ADJ_PRESS + r")\b.*?\bthan (?:the )?(front|rear)[ -]*(left|right)\b", t)
    if not m_which and not m_pair: return None
    time = None; conds = []
    ws = _windows(t, qc)
    if re.search(r"\b(?:right now|currently|at the moment|now)\b", t) or not ws: time = TimeSpec("current") if re.search(r"\b(?:right now|currently|at the moment|now)\b", t) else TimeSpec("calendar", name="today") if not ws else None
    if time is None: time = ws[0]
    stat = "max" if re.search(r"\b(?:highest|peak|maximum|max)\b", t) else "min" if re.search(r"\b(?:lowest|minimum|min)\b", t) else "mean"
    vals = _wheel_values(qc, base, stat if time.type != "current" else "mean", time, conds)
    if vals is None: return None
    unit = R.CONCEPTS[f"{base}_fl"].unit; wl = "tyre pressure" if base == "tyre_pressure" else "wheel speed"
    label = "right now" if time.type == "current" else (_label(time, qc) if time.type == "range" else resolve_time(time, qc.ctx)[2])
    word = {"mean": "average ", "max": "highest ", "min": "lowest "}[stat] if time.type != "current" else ""
    if m_which:
        hi_w = re.search(r"\b(?:" + HIGH + r")\b", t); lo_w = re.search(r"\b(?:" + LOW + r")\b", t)
        want_max = bool(hi_w and (not lo_w or hi_w.start() < lo_w.start()))
        pos = (max if want_max else min)(vals, key=vals.get)
        allv = "; ".join(f"{_POS[k]} {vals[k]:.0f} {unit}" for k in vals)
        basis = "" if time.type == "current" else (" (by its average over the period)" if stat == "mean" else "")
        return _result(f"The {'highest' if want_max else 'lowest'} {wl} {label} was the {_POS[pos]} wheel at {vals[pos]:.0f} {unit}{basis}. All four: {allv}.", "composed_wheel", position=pos, value=vals[pos])
    a = m_pair.group(1)[0] + m_pair.group(2)[0]; b = m_pair.group(4)[0] + m_pair.group(5)[0]
    adj = m_pair.group(3); d = vals[a] - vals[b]
    up = adj in ("higher", "greater", "more", "bigger", "larger", "faster")
    yes = (d > 0) if up else (d < 0)
    lead = "Yes. " if abs(d) >= 0.5 and yes else ("No, they are the same to within 0.5 " + unit + ". " if abs(d) < 0.5 else "No. ")
    return _result(lead + f"The {word}{_POS[a]} {wl} {label} was {vals[a]:.0f} {unit} and the {_POS[b]} was {vals[b]:.0f} {unit}: a difference of {abs(d):.1f} {unit}.", "composed_wheel", a=vals[a], b=vals[b])


_PER_RX = re.compile(r"\b(average|mean|typical|avg)\b.*\bper (trip|journey|drive|day|charge|charging session|session)\b|\bper (trip|journey|drive|day)\b.*\b(average|mean|typical|avg)\b|\bon average (?:per|for each|every) (trip|journey|drive|day)\b|\b(?:for|on) (?:a|an|each|every) (?:typical |average )?(trip|journey|drive|day)\b")


def _per_unit(brain, qc, t, raw):
    """'my average regen energy per trip', 'average distance per day last week': the total over the window divided by the number of trips (or days) in it."""
    m = _PER_RX.search(t)
    if not m: return None
    unit_word = next((g for g in m.groups() if g in ("trip", "journey", "drive", "day", "charge", "charging session", "session")), None)
    if unit_word is None: return None
    plain = re.sub(r"\bper (?:trip|journey|drive|day|charge|charging session|session)\b|\b(?:on average )?(?:for|on) (?:a|an|each|every) (?:typical |average )?(?:trip|journey|drive|day)\b|\bon average\b", " ", t)
    p = _base(brain, qc, "what was the total " + re.sub(r"\b(average|mean|typical|avg|total|my|the)\b", " ", plain).strip())
    if p is None or p.query is None or not p.query.target or p.query.target not in ("energy_used", "distance_driven", "regen_energy"): return None
    q0 = p.query; time = q0.scope.time
    tot, _ = _value(qc, _aggregate_query(q0.target, "sum", time, q0.scope.conditions))
    if tot is None: return None
    if unit_word in ("day",):
        days, label = _days(qc, time)
        n = len([d for d, spec in days if _value(qc, _aggregate_query(q0.target, "sum", spec, q0.scope.conditions))[0] is not None]) or 0
        what = "recorded day"
    else:
        ent = "charge_session" if unit_word.startswith("charg") or unit_word in ("charge", "session") else "trip"
        from .query_ir import EntitySelect
        q = Query(op="count"); q.scope.time = time; q.scope.select = EntitySelect(ent, None, None, "desc", None)
        n, _ = _value(qc, q); n = int(n or 0); what = "trip" if ent == "trip" else "charging session"
    if not n: return _result(f"There is no recorded {what} {resolve_time(time, qc.ctx)[2]} to average over.", "composed_per_unit", ok=False)
    label = resolve_time(time, qc.ctx)[2]; u = R.CONCEPTS[q0.target].unit
    return _result(f"The {R.label_of(q0.target)} per {what} {label} averaged {tot / n:.2f} {u}: {tot:.1f} {u} in total over {n} {what}{'' if n == 1 else 's'}.", "composed_per_unit", value=tot / n, n=n)


_NUMW = r"one|two|three|four|five|six|seven|eight|\d+"


def _what_changed(brain, qc, t, raw):
    """'which three things about the car changed most this week': every measured numeric signal's mean in the window against the window before it, ranked by standardised change."""
    m = re.search(r"\b(?:which|what)\b(?: (" + _NUMW + r"))? (?:things?|signals?|measurements?|readings?|values?|metrics?|parameters?|sensors?)\b.{0,40}\bchang\w*\b|\bwhat (?:has |have )?chang\w* (?:the )?most\b|\bbiggest (?:changes?|differences?)\b", t)
    if not m: return None
    k = 3
    mk = re.search(r"\b(" + _NUMW + r")\b", t[m.start(): m.start() + 30])
    if mk and mk.group(1) is not None:
        k = int(mk.group(1)) if mk.group(1).isdigit() else _NUM.get(mk.group(1), 3)
    k = max(1, min(8, k))
    ws = [] if re.search(r"\bthis (?:week|month)\b", t) else _windows(t, qc)          # "this week" in a ranking of changes is the rolling week, not the days since Monday
    span_days = 30 if re.search(r"\bmonth\b", t) else 7
    now = qc.ctx.now
    cur = ws[0] if ws else TimeSpec("relative", span_days, "day")
    clo, chi, clabel = resolve_time(cur, qc.ctx)
    length = chi - clo
    prev = TimeSpec("range", start=clo - length, end=clo)
    rows = []
    for name, c in R.CONCEPTS.items():
        if c.dtype != "numeric" or getattr(c, "kind", "measured") != "measured": continue
        if name in ("rated_max_speed", "tyre_tread_depth") or not qc.profile.concept_available(name): continue
        va, _ = _value(qc, _aggregate_query(name, "mean", cur, []))
        vb, _ = _value(qc, _aggregate_query(name, "mean", prev, []))
        sd, _ = _value(qc, _aggregate_query(name, "std", TimeSpec("range", start=clo - 2 * length, end=chi), []))
        if va is None or vb is None or not sd or sd <= 1e-9: continue
        rows.append((abs(va - vb) / sd, name, va, vb))
    if len(rows) < 2: return None
    rows.sort(reverse=True)
    top = rows[:k]
    lab = _label(prev, qc)
    lines = [f"{i + 1}. {R.label_of(n)}: {_fmt(n, a)} against {_fmt(n, b)} ({'up' if a > b else 'down'} {abs(a - b):.1f} {R.CONCEPTS[n].unit})" for i, (z, n, a, b) in enumerate(top)]
    return _result(f"Compared with the previous {round(length / DAY)} day{'' if round(length / DAY) == 1 else 's'} ({lab}), the {len(top)} measured quantities that changed most {clabel} (by size of change relative to their own variation) were:\n" + "\n".join(lines), "composed_what_changed", top=[n for _, n, _, _ in top])


def handle(brain, raw, t):
    qc = brain.qc()
    if qc is False or qc is None: return None
    try:
        from . import temporal_ops
        r = temporal_ops.handle(brain, qc, t)
        if r: return r
    except Exception as e:
        import logging; logging.getLogger("vehiclegpt").warning("temporal layer failed: %s: %s", type(e).__name__, e)
    for fn in (_session_attr, _what_changed, _clock, _exists, _wheel, _vs_normal, _per_unit, _argday, _count_days, _which_days, _group_counts, _extent, _signal_gap, _window_change):
        try: r = fn(brain, qc, t, raw)
        except Exception: r = None
        if r: return r
    return None

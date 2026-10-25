"""Answer composer: turns deterministic evidence into natural language with TEMPLATES. No language model is involved in stating a result - the numbers come from the executor, the words from here.

Rules the templates follow (and the verifier re-checks):
  * the brief (one or two sentences) comes first; details (period, sample count, source tables, quality, method, per-item rows) come under it;
  * every number printed is an evidence item (or a date / time / count taken from the same result);
  * estimates say "estimated" and list their assumptions; measured values never do;
  * missing data gets a plain explanation (from capability.py) - never a number;
  * causal language is limited to 'preceded', 'associated with' and 'likely contributor'; a bare 'caused' is never used."""
import datetime as dt
import re
from dataclasses import dataclass, field

from . import signal_registry as R
from .evidence import Evidence, graph_from_context, graph_from_diagnosis


@dataclass
class Answer:
    brief: str
    details: list = field(default_factory=list)
    followups: list = field(default_factory=list)
    kind: str = "value"               # value | missing | no_data | analysis | unsupported

    @property
    def text(self): return self.brief + ("\n" + "\n".join(self.details) if self.details else "")


# ------------------------------------------------------------------------------------------------ formatting
def decimals(unit, v):
    a = abs(v)
    if unit in ("%", "km", "km/h", "mph", "kPa", "psi", "W", "rpm", "Nm", "Wh/km", "Wh/mi", "m", "hp", "lux", "s", "min"): return 0 if a >= 10 or unit in ("Wh/km", "Wh/mi", "kPa", "psi", "rpm", "Nm", "W", "m", "hp", "lux") else 1
    if unit in ("C", "F", "kW", "L/min", "deg", "deg/s", "bar", "A", "C/min", "kW/min"): return 1 if a < 1000 else 0
    if unit in ("kWh", "Wh", "m/s2", "g", "V", "ratio", "r"): return 2 if a < 10 else 1
    return 1


def _pred_state(c):
    """A named state as something the time was spent in: 'while charging', or for its complement 'while not charging'."""
    ph = R.PREDICATES[c.predicate][0]
    if c.op != "not": return ph
    return ph.replace("while ", "while not ", 1) if ph.startswith("while ") else "not " + ph


def _pred_phrase(c):
    """The wording of a named condition; a negated one reads as the exclusion of that state."""
    ph = R.PREDICATES[c.predicate][0]
    if c.op != "not": return ph
    return "excluding the time " + ph.replace("on the motorway / highway", "on the motorway / highway", 1)


def fmt(v, unit=""):
    if v is None: return "n/a"
    if isinstance(v, str): return v
    if unit == "count": return f"{int(round(v)):,}"
    if v == 0: return "0"
    d = decimals(unit, v); s = f"{v:,.{d}f}" if abs(v) >= 1000 else f"{v:.{d}f}"
    if s in ("-0", "-0.0", "-0.00"): s = s[1:]
    return s


def uname(unit): return {"C": "°C", "F": "°F", "deg": "°", "deg/s": "°/s", "count": "", "none": "", "ratio": "", "r": ""}.get(unit, unit)


def vu(v, unit):
    u = uname(unit); return fmt(v, unit) + (" " + u if u else "")


def when(ts, tz):
    d = dt.datetime.fromtimestamp(ts + tz, dt.timezone.utc); return f"{d.day} {d.strftime('%b')}, {d.strftime('%H:%M')}"


def day(ts, tz):
    d = dt.datetime.fromtimestamp(ts + tz, dt.timezone.utc); return f"{d.day} {d.strftime('%b')}"


AGG_WORD = {"mean": "average", "min": "lowest", "max": "highest", "median": "median", "std": "variability (standard deviation) of", "first": "first recorded", "last": "latest", "delta": "change in", "count": "number of samples of", "p95": "95th-percentile", "sum": "total", "percent_time": "share of time for", "mode": "most common state of"}


def _cap(s): return s[:1].upper() + s[1:]


def _in(period):
    """'the last 14 days' -> 'over the last 14 days'; 'yesterday' / 'last week' / 'on 2026-09-21' stay as they are."""
    return "over " + period if period.startswith(("the last", "the whole")) else period


def _quality_lines(ev):
    out = []
    if ev.tier == "live": out.append("Taken live from the vehicle’s bus as it arrives; it is also being stored in the dataset.")
    elif ev.samples and ev.tier == "trip ledger": out.append(f"Based on {ev.samples:,} {'trip' if ev.samples == 1 else 'trips'} from the trip ledger.")
    elif ev.samples and ev.tier == "events": out.append(f"Based on {ev.samples:,} recorded {'event' if ev.samples == 1 else 'events'}.")
    elif ev.samples:
        src = list(dict.fromkeys("the vehicle's own recording of this signal" if re.match(r"lkv?_\d+", x) else x for x in ev.sources))                  # the per-signal view names (lkv_7_1s.v) are storage detail, not something to tell the driver
        out.append(f"Based on {ev.samples:,} {'sample' if ev.samples == 1 else 'samples'}" + (f" ({ev.tier})" if ev.tier else "") + (f" from {', '.join(src)}" if src else "") + ".")
    if ev.method: out.append(f"Method: {ev.method}.")
    if ev.kind == "estimated": out.append("This is an estimate" + (": " + "; ".join(ev.assumptions) if ev.assumptions else "") + ".")
    elif ev.kind == "derived" and ev.assumptions: out.append("Assumptions: " + "; ".join(ev.assumptions) + ".")
    for f in ev.flags[:3]: out.append(f"Data quality: {f['detail']}.")
    if ev.quality == "low" and not ev.flags: out.append("Data quality: low.")
    return out


def _est_prefix(ev): return "About " if ev.kind == "estimated" else ""


# ------------------------------------------------------------------------------------------------ the composer
def compose(res, ev, q, tz=0, profile=None, capability=None):
    """-> Answer. `res` = executor Result, `ev` = Evidence built from it, `q` = Query, `capability` = CapabilityReport when the query could not run."""
    if capability is not None and not capability.ok:
        a = Answer(capability.message, list(capability.details), kind="missing")
        if capability.status == "UNKNOWN": a.kind = "unsupported"
        return a
    if res.status == "unsupported": return Answer("I can't answer that with the engines attached.", [res.reason], kind="unsupported")
    if res.status == "no_data":
        lab = res.period_label
        return Answer(f"I couldn't find any data for that{(' ' + _in(lab)) if lab else ''}: {res.reason}.", [f"Recorded history: {res.reason}."], kind="no_data")
    fn = {"current_value": _current, "aggregate": _aggregate, "acceleration_band": _acceleration_band, "extreme": _extreme, "trend": _trend, "compare": _compare, "count": _count, "list_events": _list_events, "event_context": _context, "rank": _rank, "correlation": _correlation,
          "percent_time": _percent, "explain_event": _explain, "capability": _capability, "explain_change": _explain_change, "detect_change": _detect_change}[res.op]
    a = fn(res, ev, q, tz)
    a.details = a.details + _quality_lines(ev)
    return a


def _current(r, ev, q, tz):
    t = r.target; lab = R.label_of(t); c = R.CONCEPTS[t]; stale = r.extra.get("stale_s")
    if t == "range_estimate":
        brief = f"Estimated range is about {vu(r.value, r.unit)} (battery {r.extra['soc']:.1f}%, at your recent {r.extra['wh_per_km']:.0f} Wh/km)."
        return Answer(brief, [f"Estimate: {a_}" for a_ in r.assumptions], ["How was my driving this week?", "Why has my range dropped lately?"])
    if isinstance(r.value, str): brief = f"The vehicle is {r.value}" + (f" (as of {when(r.when, tz)})." if stale and stale > 300 else " right now.")
    elif t == "tyre_pressure": brief = f"The {'lowest tyre pressure' if any(e[1] == 'tp_*' for e in R.STORE.get('tyre_pressure', {}).values()) or not R.STORE.get('tyre_pressure') else 'tyre pressure (as this vehicle stores it)'} is {vu(r.value, r.unit)}" + (f" (last reading {when(r.when, tz)}, the car is off)." if stale and stale > 300 else " right now.")
    elif stale and stale > 300: brief = f"{_cap(lab)} was {vu(r.value, r.unit)} as of {when(r.when, tz)}, the most recent reading on record."
    else: brief = f"{_cap(lab)} is {vu(r.value, r.unit)} right now."
    det = []
    if t == "speed" and r.extra.get("gps") is not None and any(f["code"] == "DISAGREE" for f in r.flags):
        f = [x for x in r.flags if x["code"] == "DISAGREE"][0]; brief += f" {f['detail'].capitalize()}; I used the vehicle speed signal."
        det.append(f"GPS speed: {vu(r.extra['gps'], 'km/h')}.")
    return Answer(brief, det, [f"What was my average {lab} yesterday?"] if c.dtype == "numeric" else [])


def _aggregate(r, ev, q, tz):
    t = r.target; lab = R.label_of(t); c = R.CONCEPTS[t]; sel = q.scope.select; per = r.period_label; agg = q.agg or "mean"
    if sel and per == "the whole recorded history": per = "on record"                                  # a selected trip is not a window of time
    scope = ""; sel_note = ""
    if sel:
        n = len(r.extra.get("entities") or []) or r.extra.get("trip_count") or 0
        order = "least" if (sel.order == "desc" and sel.rank_by == "wh_per_km") else "most" if sel.rank_by == "wh_per_km" else ("longest" if sel.order == "desc" else "shortest") if sel.rank_by in ("distance_driven", "duration") else ("most recent" if sel.order == "desc" else "earliest") if sel.rank_by == "recency" else ("most energy-consuming" if sel.order == "desc" else "least energy-consuming") if sel.rank_by == "energy_used" else "highest" if sel.order == "desc" else "lowest"
        kind = f"{(order + ' efficient ') if sel.rank_by == 'wh_per_km' else (order + ' ') if sel.rank_by else ''}{(sel.klass + ' ') if sel.klass else ''}{sel.entity.replace('_', ' ').replace('charge session', 'charging session')}"
        scope = (f" during the only {kind}" if n == 1 and sel.k and sel.k > 1 else f" during the {n} {kind}s" if n > 1 else f" during the {kind}").replace("  ", " ")
        if sel.k and n and n < sel.k: sel_note = f"Only {n} matching {sel.entity}{'' if n == 1 else 's'} found {per} (you asked for {sel.k})."
        else: sel_note = ""
    cond = "".join(f" {_pred_phrase(x)}" if x.predicate else f" while {R.label_of(x.signal)} was {({'gt': 'above', 'lt': 'below', 'ge': 'at least', 'le': 'at most', 'eq': 'exactly'})[x.op]} {x.value:g} {x.unit or R.CONCEPTS[x.signal].unit}" for x in q.scope.conditions)
    if t == "energy_used": brief = f"You used {vu(r.value, r.unit)}{scope}{cond} {per}."
    elif t == "distance_driven": brief = f"You drove {vu(r.value, r.unit)}{scope}{cond} {per}."
    elif t == "regen_energy": brief = f"Regenerative braking recovered {vu(r.value, r.unit)}{scope}{cond} {per}."
    elif t == "wh_per_km": brief = f"Your energy consumption{scope}{cond} {per} was {vu(r.value, r.unit)}."
    elif agg == "count": brief = f"There are {fmt(r.value, 'count')} samples of {lab}{scope}{cond} {per}."
    else: brief = f"The {AGG_WORD[agg]} {lab}{scope}{cond} {per} was {vu(r.value, r.unit)}."
    brief = _est_prefix(ev) + brief[:1].lower() + brief[1:] if ev.kind == "estimated" and not brief.startswith("About") else brief
    if r.items and len(r.items) > 1: det = ["Per trip: " + "; ".join(f"{day(i['start'], tz)} {when(i['start'], tz)[-5:]}: {fmt(i['wh_per_km'], 'Wh/km')} Wh/km over {fmt(i['km'], 'km')} km" for i in r.items[:10] if i.get('wh_per_km') is not None)]
    else: det = []
    if sel_note: det.insert(0, sel_note)
    return Answer(brief, det, ["What about highway driving?", "And while charging?"])


def _acceleration_band(r, ev, q, tz):
    x = r.extra; n = len(x.get("runs", []))
    brief = f"Average acceleration from {fmt(x['low_kmh'], 'km/h')} to {fmt(x['high_kmh'], 'km/h')} km/h was {vu(r.value, r.unit)} {_in(r.period_label)}, across {n} valid run{'' if n == 1 else 's'}."
    det = ["Each run uses interpolated threshold-crossing times; invalid samples, vehicle-off periods and telemetry gaps are excluded."]
    return Answer(brief, det, [], "analysis")


def _extreme(r, ev, q, tz):
    lab = R.label_of(r.target); word = "highest" if r.agg == "max" else "lowest"
    ops = {"gt": "above", "lt": "below", "ge": "at least", "le": "at most", "eq": "exactly"}
    cond = "".join(f" {_pred_phrase(c)}" if c.predicate else f" while {R.label_of(c.signal)} was {ops[c.op]} {c.value:g} {c.unit or R.CONCEPTS[c.signal].unit}" for c in q.scope.conditions)       # the restriction the answer was computed under is part of the answer
    brief = f"The {word} {lab}{cond} {_in(r.period_label)} was {vu(r.value, r.unit)}, on {when(r.when, tz)}."
    st = r.extra.get("state", {}); det = ["At that moment: " + ", ".join(f"{R.label_of(k)} {vu(v, R.CONCEPTS[k].unit)}" for k, v in st.items()) + "."] if st else []
    return Answer(brief, det, ["What happened before that?"])


def _trend(r, ev, q, tz):
    x = r.extra; lab = R.label_of(r.target); c = R.CONCEPTS[r.target]
    word = "rose" if x["change"] > 0 else "fell"
    if x["significant"]: brief = f"{_cap(lab)} {word} by {vu(abs(x['change']), c.unit)} {_in(r.period_label)}, from {fmt(x['first'], c.unit)} to {vu(x['last'], c.unit)} - a clear trend of {vu(abs(r.value), c.unit)} per day."
    else: brief = f"No clear trend in {lab} {_in(r.period_label)}: it went from {fmt(x['first'], c.unit)} to {vu(x['last'], c.unit)}, which is within normal day-to-day variation."
    det = [f"Slope {fmt(r.value, c.unit)} {uname(c.unit)} per day over {x['days']} days" + (f"; p-value {x['p']:.3f}" if x.get("p") is not None else "") + "."]
    return Answer(brief, det, ["Is that significant?", "What changed at the same time?"], "analysis")


def _compare(r, ev, q, tz):
    x = r.extra; lab = R.label_of(r.target); u = r.unit; la, lb = x["labels"]; a, b = x["a"], x["b"]
    bare = lambda s: s[3:] if str(s).startswith("in ") else s                                         # "in 2024" reads "for 2024" in a comparison
    names = {"A": bare(x["periods"][0]), "B": bare(x["periods"][1])}
    pretty = lambda n: n.replace("not_", "not ").replace("_", " ") if isinstance(n, str) else n
    na, nb = (names.get(la, pretty(la)), names.get(lb, pretty(lb))); hi, lo = (lb, la) if b > a else (la, lb)
    pct = abs(x["ratio_pct"]) if x.get("ratio_pct") is not None else None
    if abs(a - b) < 1e-9: brief = f"{_cap(lab)} was the same for {na} and {nb}: {vu(a, u)}."
    else:
        word = "higher" if b > a else "lower"
        brief = f"{_cap(lab)} was {vu(b, u)} for {nb}" + (f", {fmt(pct, '%')}% {word} than {vu(a, u)} for {na}." if pct is not None else f" against {vu(a, u)} for {na}.")
    if q.groups and r.period_label: brief = brief.rstrip(".") + f" {_in(r.period_label)}."
    det = [f"{na}: {vu(a, u)}", f"{nb}: {vu(b, u)}", f"Difference: {vu(b - a, u)}"]
    if x.get("p") is not None: det.append(f"Day-to-day variation test: p-value {x['p']:.3f} ({'a clear difference' if x['p'] < 0.05 else 'not clearly different'}).")
    return Answer(brief, det, [], "analysis")


def _count(r, ev, q, tz):
    if q.event is not None:
        lab = R.EVENTS[q.event.type][0]; n = int(r.value)
        where = "".join(f" {_pred_phrase(c)}" if c.predicate else f" while {R.label_of(c.signal)} was {({'gt': 'above', 'lt': 'below', 'ge': 'at least', 'le': 'at most', 'eq': 'exactly'})[c.op]} {c.value:g} {c.unit or R.CONCEPTS[c.signal].unit}" for c in q.scope.conditions if not (c.predicate and R.EVENT_AS_PREDICATE.get(q.event.type) == c.predicate and c.op != "not"))
        brief = f"There {'was' if n == 1 else 'were'} {n} {lab}{'' if n == 1 else 's'}{where} {_in(r.period_label)}."
        det = ["Latest: " + "; ".join(f"{when(i['ts'], tz)} {i['detail']}" for i in r.items[-3:])] if r.items else []
        lb = (r.extra or {}).get("latest_before")
        if lb and not n:
            brief += f" The latest earlier one was on {when(lb['ts'], tz)}, before that period."
        return Answer(brief, det, ["When was the last one?"])
    if q.scope.select is not None:
        e_ = q.scope.select; plain = not (e_.klass or e_.rank_by or e_.k); name = e_.entity.replace("_", " ").replace("charge session", "charging session")
        return Answer(f"There {'was' if r.value == 1 else 'were'} {int(r.value)} {'' if plain else 'matching '}{(e_.klass + ' ') if e_.klass else ''}{name}{'' if r.value == 1 else 's'} {_in(r.period_label)}.", [], [])
    sig = [c for c in q.scope.conditions if c.signal]
    if sig and q.target == sig[0].signal:                                                          # "how many times did the speed go above 100 km/h": separate stretches above the threshold
        c = sig[0]; word = {"gt": "went above", "ge": "reached at least", "lt": "dropped below", "le": "dropped to at most", "eq": "was exactly"}[c.op]
        n = int(r.value); u = c.unit or R.CONCEPTS[c.signal].unit
        return Answer(f"{_cap(R.label_of(c.signal))} {word} {c.value:g} {u} {n} time{'' if n == 1 else 's'} {_in(r.period_label)}" + (" (separate stretches, each counted once)." if n else "."), [], [])
    return Answer(f"{_cap(R.label_of(q.target) if q.target else 'the condition')} occurred {int(r.value)} time{'' if r.value == 1 else 's'} {_in(r.period_label)}.", [], [])


def _event_where(q):
    ops = {"gt": "above", "lt": "below", "ge": "at least", "le": "at most", "eq": "exactly"}
    bits = [(_pred_phrase(c) if c.predicate else f"while {R.label_of(c.signal)} was {ops[c.op]} {c.value:g} {c.unit or R.CONCEPTS[c.signal].unit}") for c in q.scope.conditions
            if not (c.predicate and q.event is not None and R.EVENT_AS_PREDICATE.get(q.event.type) == c.predicate and c.op != "not")]
    return (" " + " and ".join(bits)) if bits else ""


def _list_events(r, ev, q, tz):
    lab = R.EVENTS[q.event.type][0]; where = _event_where(q)
    if not r.items: return Answer(f"No {lab} was recorded{where} {_in(r.period_label)}.", [], [])
    illustrative = any("illustrative" in str(i.get("detail", "")).lower() or "simulat" in str(i.get("detail", "")).lower() for i in r.items)
    disclosure = "These are illustrative records from the simulated demo history, not confirmed production-vehicle faults." if illustrative else None
    sel = q.event.selector
    if sel in ("last", "first", "nth", "most_severe"):
        i = r.items[0]; brief = f"The {'first' if sel == 'first' else 'last' if sel == 'last' else 'requested'} {lab}{where} was on {when(i['ts'], tz)}: {i['detail']}."
        if sel == "most_severe": brief = f"The highest-severity recorded {lab} was {i.get('sev', 'unrated')} on {when(i['ts'], tz)}: {i['detail']}."
        det = [f"{int(r.value)} {lab}{'s' if r.value != 1 else ''} in total {r.period_label}."]
        if disclosure: det.append(disclosure)
        return Answer(brief, det, ["What happened before it?"])
    pg = (r.extra or {}).get("page")
    if pg and pg["first"] > 1: brief = f"Events {pg['first']} to {pg['last']} of {pg['total']} {lab}{'s' if pg['total'] != 1 else ''} {_in(r.period_label)}, newest first."
    else: brief = f"{int(r.value)} {lab}{'s' if r.value != 1 else ''}{where} {_in(r.period_label)}; the latest was on {when(r.items[0]['ts'], tz)}: {r.items[0]['detail']}."
    det = ([disclosure] if disclosure else []) + ["  - " + f"{when(i['ts'], tz)}: {i['detail']}" for i in r.items[:25]]
    return Answer(brief, det, ["Show the next events"] if pg and pg["last"] < pg["total"] else [])


def _context(r, ev, q, tz):
    x = r.extra; lab = x["event_label"]; moves = sorted([s for s in r.items], key=lambda s: -abs(s["change"]) / ((s["max"] - s["min"]) or 1.0 if False else max(abs(s["first"]), abs(s["last"]), 1e-6)))
    top = [s for s in moves if abs(s["change"]) > 1e-9][:3]
    parts = [f"{s['label']} went from {fmt(s['first'], s['unit'])} to {vu(s['last'], s['unit'])}" for s in top] or ["the recorded signals stayed steady"]
    bef, aft = x.get("window", (0, 0)); mins = lambda s: (f"{s // 60} minutes" if s % 60 == 0 and s >= 120 else f"{s} seconds")
    span = f"the {mins(bef)} before" + (f" and the {mins(aft)} after" if aft else "")
    brief = f"In {span} the {lab} on {when(r.when, tz)} (the interval {when(r.when - bef, tz)} to {when(r.when + aft, tz)}{'' if aft else '; the event itself is the anchor and is not part of these figures'}): " + "; ".join(parts) + "."
    det = [f"  - {s['label']}: {fmt(s['first'], s['unit'])} to {vu(s['last'], s['unit'])} (min {fmt(s['min'], s['unit'])}, max {fmt(s['max'], s['unit'])})" for s in r.items]
    if x.get("near_events"): det.append("Other events in the window: " + "; ".join(f"{when(i['ts'], tz)} {i['detail']}" for i in x["near_events"][:4]) + ".")
    g = graph_from_context(r, lab); ev.graph = g
    lines = g.text_lines()
    if lines: det.append("Order of events in the recorded data (association, not proof of cause):"); det += ["  - " + l for l in lines]
    if x.get("frozen_context"): det.append("Source: the 1 Hz context stored with the event.")
    return Answer(brief, det, ["What evidence supports that?", "Could there be another explanation?"], "analysis")


def _rank(r, ev, q, tz):
    e = q.scope.select; unit = r.items[0].get("unit", "") if r.items else ""; lab = R.label_of(e.rank_by) if e.rank_by in R.CONCEPTS else e.rank_by
    desc = e.order == "desc"
    word = ("least efficient" if desc else "most efficient") if e.rank_by == "wh_per_km" else ("most recent" if desc else "earliest") if e.rank_by == "recency" else ("most energy-consuming" if desc else "least energy-consuming") if e.rank_by == "energy_used" else (("longest" if desc else "shortest") if e.rank_by in ("distance_driven", "duration") else ("highest " + lab if desc else "lowest " + lab))
    val = lambda i: (fmt(i['value'] / 60, 'min') + ' min') if unit == 's' else vu(i['value'], unit)
    ent = f"{e.klass + ' ' if e.klass else ''}{e.entity}"
    if len(r.items) == 1:
        i = r.items[0]
        brief = f"The {word} {ent} {_in(r.period_label) if r.period_label != 'the whole recorded history' else 'on record'} was on {when(i['start'], tz)}: {val(i)}" + (f", {fmt(i['km'], 'km')} km" if i.get("km") is not None and e.rank_by != "distance_driven" else "") + (f", lasting {fmt((i['end'] - i['start']) / 60, 'min')} min" if e.rank_by == "distance_driven" and i.get("end") and i.get("start") else "") + "."
    else:
        brief = f"The {len(r.items)} {word} {ent}s {_in(r.period_label)}: " + "; ".join(f"{day(i['start'], tz)} {val(i)}" for i in r.items[:5]) + "."
    det = [f"  {n}. {when(i['start'], tz)}: {val(i)}" + (f", {fmt(i['km'], 'km')} km" if i.get("km") is not None and e.rank_by != "distance_driven" else "") for n, i in enumerate(r.items, 1)]
    return Answer(brief, det, ["What was battery temperature during those trips?"], "analysis")


def _correlation(r, ev, q, tz):
    a, b = R.label_of(q.target), R.label_of(q.target2); v = r.value; s = "strongly" if abs(v) >= 0.7 else "moderately" if abs(v) >= 0.4 else "weakly" if abs(v) >= 0.15 else "hardly"
    brief = f"{_cap(a)} and {b} are {s} {'positively' if v > 0 else 'negatively'} correlated {_in(r.period_label)} (r = {fmt(v, 'r')}); that is an association, not proof that one drives the other."
    det = ([f"Spearman rank correlation: {fmt(r.extra['spearman'], 'r')}."] if r.extra.get("spearman") is not None else []) + [f"Paired samples used: {r.extra.get('samples_used', r.samples):,}."]
    return Answer(brief, det, [], "analysis")


def _percent(r, ev, q, tz):
    cond = " and ".join(_pred_state(c) if c.predicate else f"with {R.label_of(c.signal)} {({'gt': 'above', 'lt': 'below', 'ge': 'at least', 'le': 'at most', 'eq': 'exactly'})[c.op]} {c.value:g} {c.unit or R.CONCEPTS[c.signal].unit}" for c in q.scope.conditions)
    den = r.extra.get("denominator", "the period"); of = "of the driving time" if den == "driving time" else "of the time"
    mins = int(round((r.value or 0.0) / 100.0 * (r.samples or 0)))
    span = f"about {mins:,} minutes ({mins / 60.0:.1f} hours)" if mins >= 60 else f"about {mins} minutes"
    if re.search(r"\bhow (?:many (?:minutes|mins|hours|hrs|seconds)|long|much time)\b", (getattr(q, "text", "") or "").lower()):        # a duration was asked for: lead with the duration, give the share second
        return Answer(f"{_cap(span)} {_in(r.period_label)} {cond} ({fmt(r.value, '%')}% {of}).", [f"Share of {den}."], [])
    return Answer(f"{fmt(r.value, '%')}% {of} {_in(r.period_label)} was spent {cond}, {span}.", [f"Share of {den}."], [])


def _explain(r, ev, q, tz):
    d = r.extra["diag"]; e = r.extra["event"]; lab = R.EVENTS[q.event.type][0]; alts = d.get("alts") or []
    ev.graph = graph_from_diagnosis(d, lab, e["ts"])
    if not alts: return Answer(f"I can see the {lab} on {when(e['ts'], tz)} but the recorded signals don't point to a likely contributor.", [], [], "analysis")
    top = alts[0]; name = top["name"].split(" (")[0]
    brief = f"For the {lab} on {when(e['ts'], tz)}, the most likely contributor is {name} ({top['strength']} evidence); that is what the data supports, not a proven cause."
    det = [f"  - {l}" for l in ev.graph.text_lines()] + [f"Confidence: {d.get('confidence', 'moderate')}."]
    return Answer(brief, det, ["What evidence supports that?", "What would prove it?"], "analysis")


def _capability(r, ev, q, tz): return Answer(str(r.value), [], [], "value")


# ------------------------------------------------------------------------------------------------ change operators
def _detect_change(r, ev, q, tz):
    x = r.extra; t = r.target; lab = R.label_of(t); u = x["unit"]; sh = x.get("shift"); jp = x.get("jump"); bad = x.get("direction_bad")
    if sh and sh["clear"]:
        word = "rose" if sh["change"] > 0 else "fell"
        brief = f"The best candidate for when {lab} changed is {day(sh['day'], tz)}: it {word} from {fmt(sh['before'], u)} to {vu(sh['after'], u)}" + (f" ({fmt(abs(sh['pct']), '%')}%)" if sh.get("pct") is not None else "") + f" - {sh['shape']}."
    else:
        brief = f"I found no clear sustained change in {lab} {_in(r.period_label)}; the biggest single move was on {day(jp['day'], tz)}."
    det = []
    gap_days = jp.get("gap_days", round((jp["day"] - jp["from_day"]) / 86400)) if jp and jp.get("from_day") else 1
    if jp: det.append((f"Largest day-to-day move{' in the unfavourable direction' if bad else ''}: {day(jp['day'], tz)}," if gap_days <= 1 else f"Largest move between two observed days{' in the unfavourable direction' if bad else ''} ({gap_days} days apart, no readings in between): {day(jp['from_day'], tz)} to {day(jp['day'], tz)},")+f" {fmt(jp['from'], u)} to {vu(jp['to'], u)} ({'+' if jp['change'] > 0 else ''}{fmt(jp['change'], u)} {uname(u)}).")
    if sh: det.append(f"Candidate split: {fmt(sh['before'], u)} before and {vu(sh['after'], u)} after, over {x['days']} days (daily values); effect size {sh['effect_size']:.1f} standard deviations; adjusted p-value {sh['p_adj']:.3f} (it allows for having searched every split, and treats days as independent)" + ("" if sh["clear"] else " - not clear enough to call a change") + ".")
    return Answer(brief, det, ["Why did it change?", "What else changed on that day?"], "analysis")


def _explain_change(r, ev, q, tz):
    x = r.extra; o = x["obs"]; t = r.target; lab = R.label_of(t); u = x["unit"]; cb = x.get("contrib"); w = x["windows"]
    word = "rose" if o["change"] > 0 else "fell"; per = f"{_in(r.period_label)}" if r.period_label else ""
    if t == "range_estimate" and o.get("range_before") is not None: head = f"Your estimated range {word} from about {fmt(o['range_before'], 'km')} to {vu(o['range_after'], 'km')} {per}"
    else: head = f"{_cap(lab)} {word} from {fmt(o['before'], u)} to {vu(o['after'], u)}{(' (' + fmt(abs(o['pct']), '%') + '%)') if o.get('pct') is not None else ''} {per}, compared with the period just before"
    if not o["clear"] and (o.get("pct") is None or abs(o["pct"]) < 8):
        brief = f"{_cap(lab)} was about the same: {fmt(o['before'], u)} to {vu(o['after'], u)} {per}, which is within normal day-to-day variation, so there is little to explain."; cb = None; x = dict(x, assoc=[], unexplained=False); x["open"] = []
    elif cb and abs(cb["gap"]) > 1e-6:
        share = 100 * cb["explained"] / cb["gap"]; brief = head + (f". The recorded conditions account for about {fmt(abs(share), '%')}% of the change in energy use per kilometre (a distance-weighted trip model); the remaining {fmt(abs(100 - share), '%')}% is whatever they do not capture." if 0 <= share <= 100 else f". The recorded conditions together explain more than the whole change ({fmt(abs(share), '%')}% of it): some of them pushed energy use up and others pulled it down, so they are listed one by one below.")
    elif x["assoc"] and x["unexplained"] is False: brief = head + ". " + _assoc_summary(x)
    else: brief = head + (". None of the recorded signals that could account for it changed in a matching way." if x["assoc"] else ". The recorded data show the change but not what is behind it.")
    det = []
    when_txt = f" A candidate start is {day(o['start_day'], tz)} ({o['shape']}; adjusted p-value {o['start_p_adj']:.3f})." if o.get("start_day") else ""
    sig = "clear" if o["clear"] else "not clear"
    if o.get("p") is not None: det.append(f"1. Observed ({o['basis']}): {lab} was {vu(o['before'], u)} over {plural_d(o['days'][0])} before and {vu(o['after'], u)} over {plural_d(o['days'][1])} after (p-value {o['p']:.3f}, effect size {o['effect_size']:.1f}; {sig})." + when_txt)
    else: det.append(f"1. Observed ({o['basis']}): {lab} was {vu(o['before'], u)} before and {vu(o['after'], u)} after.")
    if cb:
        det.append(f"2. Contributions (an estimate from a trip-level model over {cb['n_trips'][0] + cb['n_trips'][1]} trips, R2 = {cb['r2']:.2f}; total energy / total distance {fmt(cb['trip_before'], 'Wh/km')} to {vu(cb['trip_after'], 'Wh/km')}, a gap of {SIGNED_(cb['gap'])} Wh/km):")
        for f in cb["factors"][:4]: det.append(f"   - {f['name']}: {fmt(f['before'], f['unit'])} to {vu(f['after'], f['unit'])} accounts for {SIGNED_(f['contribution'])} Wh/km ({fmt(abs(f['share_pct']), '%')}% of the gap)" + ("; this condition moved together with the period itself, so its share may stand in for a cause that was not recorded" if abs(f.get("period_corr", 0.0)) >= 0.5 else ""))
        det.append(f"   - not captured by the model and the recorded conditions: {SIGNED_(cb['residual'])} Wh/km ({fmt(abs(100 * cb['residual'] / cb['gap']), '%')}%) - kept separate, not spread over the factors above; it can hold missing causes, noise and model error")
    elif x["assoc"]:
        det.append("2. Associated changes (these changed over the same periods; an association, not proof of cause):")
        for a in x["assoc"]:
            if a["moved"]: det.append(f"   - {a['label']}: {fmt(a['before'], a['unit'])} to {vu(a['after'], a['unit'])} (p-value {a['p']:.3f}); {a['mechanism']}" + ("; this fits the direction of the change" if a["consistent"] else "; this does not fit the direction of the change"))
            else: det.append(f"   - {a['label']}: no clear change ({fmt(a['before'], a['unit'])} to {vu(a['after'], a['unit'])})")
        if o.get("normalised"):
            n = o["normalised"]; det.append(f"   - corrected to a common temperature (days with both readings only) the pressure went from {fmt(n['before'], 'kPa')} to {vu(n['after'], 'kPa')}; a change that survives the correction points to a leak rather than the weather" if (n.get("p") or 1) < 0.01 else f"   - corrected to a common temperature (days with both readings only) the pressure is unchanged ({fmt(n['before'], 'kPa')} to {vu(n['after'], 'kPa')}); the weather accounts for the drop")
    elif o["clear"] or (o.get("pct") is not None and abs(o["pct"]) >= 8): det.append("2. No recorded signal is declared to drive this measure, so no contributions can be given.")
    if x["open"]: det.append("3. Not resolved by the recorded data; these measurements would help: " + "; ".join(x["open"]) + ".")
    return Answer(brief, det, ["What evidence supports that?", "Could there be another explanation?"], "analysis")


def _assoc_summary(x):
    parts = [f"{a['label']} {'rose' if a['change'] > 0 else 'fell'} ({fmt(a['before'], a['unit'])} to {vu(a['after'], a['unit'])})" for a in x["assoc"] if a["consistent"]][:2]
    return "Changes in " + " and ".join(parts) + " fit the direction of the change (associated, not proof)." if parts else ""


def SIGNED_(v): return ("+" if v > 0 else "") + fmt(v, "Wh/km")


def plural_d(n): return f"{n} day{'s' if n != 1 else ''}"

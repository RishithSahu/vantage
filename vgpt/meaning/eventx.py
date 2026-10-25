"""Executors for questions about HOW OFTEN things happened and about the vehicle's text logs - the part of "find historical events, trends and anomalies" that a single count cannot answer.

    trend    <event>  change | rise | drop       are ECU resets getting more frequent / in which week did hard braking rise most
    compare  <event>  count  vs=period | vs=<condition>   resets this week versus last / ADAS disengagements on the highway versus in the city (counts AND rates per driving hour)
    extreme  <event>  max | min   ent=day | week the day (week) with the most / fewest hard-braking events
    events   <event>  count | exists | list   ent=day | week   on how many days did X happen
    events   service_log  list | count | exists | first | last  search the WARN+ log lines of the vehicle's services by words and by named source

Every number comes from SQL over the events / logs / ledger tables or from the typed engine's own condition machinery (named predicates, time in a condition); tests/test_eventx.py recomputes them independently.
A statement about a change says how many events it rests on and uses an exact binomial test (the two halves of a window are equally long, so under "no change" an event is equally likely to fall in either); it
is exploratory and says so.  No text of the question is inspected here: the frame names the event, `lit` carries the already-parsed period, and `literals.log_terms` supplies the words of a log search."""
import math

from .. import analytics as A
from .. import signal_registry as R
from ..plan_exec import resolve_time
from ..qa_core import safe_text
from ..query_ir import Condition, EventRef, Query, Scope, TimeSpec
from . import literals as L

DAY = 86400
PREVIOUS_CALENDAR = {"this_week": "last_week", "this_month": "last_month", "today": "yesterday", "yesterday": "day_before_yesterday"}
LOG_LIMIT = 8


def _r(resolver, answer, route, frame, ok=True, **facts):
    from .resolve import _result
    return _result(answer, route, "event_series", frame, **dict(facts, **({} if ok else {"ok": False})))


def _types(c):
    ev = c.registry_key
    return tuple(R.EVENT_GROUPS.get(ev, (ev,)))


def _win(b, qc, spec):
    lo, hi, label = resolve_time(spec, qc.ctx)
    return int(lo), min(int(hi), int(b.now) + 1), label


def _window(b, qc, lit, default):
    """The period of an event question.  "Now" ("do dropouts happen more often now than before") is an instant: it names no period of events, so the default window is used."""
    t = lit.times[0] if lit.times else None
    return _win(b, qc, default if t is None or t.type == "current" else t)


def _events(b, types, lo, hi):
    q = ",".join("?" * len(types))
    return b.c.execute(f"SELECT ts, type, detail FROM events WHERE type IN ({q}) AND ts>=? AND ts<? ORDER BY ts", (*types, lo, hi)).fetchall()


def _stored(b, c):
    """Is this event type something the events table can hold (detected events such as COOLANT_FLOW_DROP are computed from signals)?"""
    return c.registry_key not in R.DETECTED_EVENTS


def _gap(frame, c, why):
    from .resolve import _result
    return _result(f"I understood this as a question about how often {c.label} happened, but {why}, so I will not guess.", f"gap:{frame.act}:{c.id}", "none", frame, intent="CLARIFY", ok=False, coverage_gap=f"{frame.act}:{c.id}")


def _coverage(b, lo, hi):
    """(days of the window that have recorded data, days in the window)."""
    days = max((hi - lo) / DAY, 1e-9); tz = int(b.tz)
    got = b.c.execute("SELECT COUNT(DISTINCT (ts+?)/86400) FROM sig_1m WHERE ts>=? AND ts<?", (tz, lo, hi)).fetchone()[0]
    return int(got or 0), days


def _binom_p(k, n, p0):
    """Exact two-sided binomial probability of an outcome at least as unlikely as k out of n when each event falls in the first group with probability p0."""
    if n <= 0: return 1.0
    if n > 1500:                                                                                    # normal approximation with continuity correction
        z = (abs(k - n * p0) - 0.5) / math.sqrt(n * p0 * (1 - p0)); return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))
    pm = [math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i)) for i in range(n + 1)]
    return min(1.0, sum(x for x in pm if x <= pm[k] * (1 + 1e-9)))


def _plural(n, word): return f"{n} {word}" + ("" if n == 1 else "s")
def _cap(s): return s[:1].upper() + s[1:]
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
NO_DIGITS = str.maketrans("", "", "0123456789.")


def _date(ts):
    """'27 Sep' for a UTC timestamp, in the vehicle's local time (analytics.set_tz has been called)."""
    s = A.stamp(ts); return f"{int(s[8:10])} {MONTHS[int(s[5:7]) - 1]}"


def _wording(c, n=1):
    return c.label if n == 1 else (c.label + "s" if not c.label.endswith("s") else c.label)


def _against_previous(res, c, frame, b, qc, ps, rows, lo, hi, label):
    """How often an event happened in the period asked about, against the same kind of period before it (this month so far against the whole of last month): events per day, and an exact test that the two rates differ."""
    lo2, hi2, label2 = _win(b, qc, ps[1])
    if hi2 <= lo2: return None
    rows2 = _events(b, _types(c), lo2, hi2); n1, n2 = len(rows), len(rows2); d1, d2 = max((hi - lo) / DAY, 1e-9), (hi2 - lo2) / DAY; r1, r2 = n1 / d1, n2 / d2
    if n1 + n2 < 6: return _r(res, f"There were only {n1} {_wording(c, n1)} {label} and {n2} {label2}, too few to say whether {c.label} is becoming more or less frequent.", f"trend:{c.id}:previous", frame, count=n1, previous=n2)
    p = _binom_p(n1, n1 + n2, d1 / (d1 + d2)); more = r1 > r2
    rd = frame.op in ("rise", "drop")
    if p < 0.10 and abs(r1 - r2) > 1e-9: lead = (("Yes - " if more == (frame.op == "rise") else "No - ") if rd else "") + c.label[:1].upper() * (not rd) + (c.label if rd else c.label[1:]); head = f"{lead} became {'more' if more else 'less'} frequent: {n1} {label} ({r1:.2f} per day) against {n2} {label2} ({r2:.2f} per day; exact test p = {p:.3f})."
    else: head = (("No - " + c.label) if rd else _cap(c.label)) + f" did not clearly change in frequency: {n1} {label} ({r1:.2f} per day) against {n2} {label2} ({r2:.2f} per day; exact test p = {p:.2f})."
    return _r(res, head + f"{chr(10)}Based on {_plural(n1 + n2, 'recorded event')} in the two periods; a count of recorded events per day, not a rate per hour driven. The period so far ({d1:.0f} days) is compared with the whole earlier period ({d2:.0f} days).",
              f"trend:{c.id}:previous", frame, count=n1, previous=n2, p_value=p, window=[lo, hi], previous_window=[lo2, hi2])


# ------------------------------------------------------------------------------------------------------------------ trend
def event_trend(res, c, frame, lit):
    b = res.brain; qc = b.qc()
    if not _stored(b, c): return _gap(frame, c, "that event is detected from signals and has no stored occurrences to count over time")
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 28, "day")); A.set_tz(b.tz)
    rows = _events(b, _types(c), lo, hi); n = len(rows); days = (hi - lo) / DAY
    ps = _periods(lit) if (len(lit.times) == 1 and lit.times[0].type in ("calendar", "relative")) else None
    if ps is not None and frame.op in ("rise", "drop", "change") and lit.times[0].type == "calendar" and lit.times[0].name in PREVIOUS_CALENDAR:
        r_ = _against_previous(res, c, frame, b, qc, ps, rows, lo, hi, label)                                        # "increased this month": this month against last month, not one half of this month against the other
        if r_ is not None: return r_
    if n == 0: return _r(res, f"No {c.label} was recorded {label}, so there is no change in how often it happens.", f"trend:{c.id}", frame, ok=False, count=0, window=[lo, hi])
    mid = lo + (hi - lo) // 2; n1 = sum(1 for r in rows if r[0] < mid); n2 = n - n1; have, span = _coverage(b, lo, hi)
    p = _binom_p(n2, n, 0.5); r1, r2 = n1 / (days / 2), n2 / (days / 2); more = n2 > n1
    if n < 6: head = f"There were only {n} {_wording(c, n)} {label} ({n1} in the first half, {n2} in the second), too few to say whether this is becoming more or less frequent."
    elif p < 0.10 and n2 != n1:
        rd = frame.op in ("rise", "drop"); lead = (("Yes - " if more == (frame.op == "rise") else "No - ") + c.label) if rd else _cap(c.label)
        head = f"{lead} became {'more' if more else 'less'} frequent {label}: {n1} in the first half, {n2} in the second ({r1:.2f} to {r2:.2f} per day; exact binomial p = {p:.3f})."
    else: head = (("No - " + c.label) if frame.op in ("rise", "drop") else _cap(c.label)) + f" did not clearly change in frequency {label}: {n1} in the first half, {n2} in the second ({r1:.2f} to {r2:.2f} per day; exact binomial p = {p:.2f})."
    cov = f"Data was recorded on {have} of the {span:.0f} days of the period." if have < span * 0.9 else ""
    weekly = ""; cnt = None; edge = None
    if frame.op in ("rise", "drop"):
        lo7 = ((lo + int(b.tz) + DAY - 1) // DAY) * DAY - int(b.tz); weeks = int((hi - lo7) // (7 * DAY))                                  # whole 7-day stretches from the first local midnight in the window
        if weeks >= 2:
            edges = [lo7 + i * 7 * DAY for i in range(weeks + 1)]; cnt = [sum(1 for r in rows if edges[i] <= r[0] < edges[i + 1]) for i in range(weeks)]
            steps = [(cnt[i] - cnt[i - 1], i) for i in range(1, weeks)]; d, i = (max if frame.op == "rise" else min)(steps); word = "rose" if frame.op == "rise" else "fell"
            if (frame.op == "rise" and d > 0) or (frame.op == "drop" and d < 0): weekly = f"Week by week the counts were {', '.join(map(str, cnt))}; the largest {'rise' if frame.op == 'rise' else 'fall'} was in the week starting {_date(edges[i])}, {cnt[i]} against {cnt[i - 1]} the week before."; edge = edges[i]
            else: weekly = f"Week by week the counts were {', '.join(map(str, cnt))}; no week was {'higher' if frame.op == 'rise' else 'lower'} than the one before."
    return _r(res, head + (("\n" + weekly) if weekly else "") + f"\nBased on {_plural(n, 'recorded event')}; this is a count of recorded events, not a rate per hour driven. The first/second-half comparison is exploratory and does not separate a change in the vehicle from a change in how much it was driven." + (("\n" + cov) if cov else ""),
              f"trend:{c.id}:{frame.op}" if frame.op in ("rise", "drop") else f"trend:{c.id}", frame, count=n, first_half=n1, second_half=n2, p_value=p, window=[lo, hi], **({"counts": cnt} if cnt else {}), **({"stretch_start": edge} if edge else {}))


# ------------------------------------------------------------------------------------------------------------------ compare
def _periods(lit):
    if len(lit.times) >= 2: return lit.times[0], lit.times[1]
    if len(lit.times) == 1 and lit.times[0].type == "calendar" and PREVIOUS_CALENDAR.get(lit.times[0].name): return lit.times[0], TimeSpec("calendar", name=PREVIOUS_CALENDAR[lit.times[0].name])
    if len(lit.times) == 1 and lit.times[0].type == "relative": return lit.times[0], TimeSpec("previous", lit.times[0].value, lit.times[0].unit)
    return None


def event_compare(res, c, frame, lit):
    b = res.brain; qc = b.qc()
    if not _stored(b, c): return _gap(frame, c, "that event is detected from signals and has no stored occurrences to compare")
    A.set_tz(b.tz); types = _types(c)
    if frame.vs == "period":
        ps = _periods(lit)
        if ps is None: return _gap(frame, c, "I need the two periods to compare (for example 'this week versus last week')")
        (lo1, hi1, l1), (lo2, hi2, l2) = [_win(b, qc, p) for p in ps]
        n1, n2 = len(_events(b, types, lo1, hi1)), len(_events(b, types, lo2, hi2)); d1, d2 = (hi1 - lo1) / DAY, (hi2 - lo2) / DAY
        if min(d1, d2) <= 0: return _gap(frame, c, "one of the two periods has not started yet")
        r1, r2 = n1 / d1, n2 / d2; p = _binom_p(n1, n1 + n2, d1 / (d1 + d2)) if n1 + n2 else 1.0
        rel = "the same" if (abs(r1 - r2) < 1e-9) else ("more" if r1 > r2 else "less")
        head = f"{_cap(c.label)}: {n1} {l1} against {n2} {l2} ({r1:.2f} against {r2:.2f} per day) - {'equally often' if rel == 'the same' else rel + ' often'} in the first period."
        if n1 + n2 >= 6 and rel != "the same": head += f" Exact binomial p = {p:.3f} ({'a clear difference' if p < 0.05 else 'not a clear difference'})."
        elif n1 + n2 < 6: head += " Too few events to call the difference real."
        if abs(d1 - d2) > 0.05 * max(d1, d2): head += f" The periods differ in length ({d1:.1f} and {d2:.1f} days), so rates per day are the fair comparison."
        return _r(res, head + f"\nBased on {_plural(n1 + n2, 'recorded event')}.", f"compare:{c.id}:period", frame, counts=[n1, n2], rates_per_day=[r1, r2], p_value=p)
    if frame.cond not in ("none", "threshold") and frame.vs not in ("none", "period"):
        lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 60, "day")); out = []
        for cond in (frame.cond, frame.vs):
            cd = Condition(predicate=cond[4:], op="not") if cond.startswith("not_") else Condition(predicate=cond); sc = Scope(lit.times[0] if lit.times else TimeSpec("relative", 60, "day"), None, [cd])
            cnt = qc.execute(Query(op="count", scope=sc, event=EventRef(c.registry_key, "last"))); share = qc.execute(Query(op="percent_time", scope=sc))
            if cnt.status != "answered" or share.status != "answered" or not getattr(share.result, "samples", 0): return _gap(frame, c, f"the time spent {R.PREDICATES.get(cond, (cond,))[0]} cannot be measured from the recorded data")
            minutes = share.result.value / 100.0 * share.result.samples; out.append((cond, int(cnt.result.value), minutes / 60.0))
        (ca, na, ha), (cb, nb, hb) = out
        if ha <= 0 or hb <= 0: return _r(res, f"The vehicle spent no recorded time {R.PREDICATES[ca if ha <= 0 else cb][0]} {label}, so the two rates cannot be compared.", f"compare:{c.id}:cond", frame, ok=False)
        ra, rb = na / ha, nb / hb; p = _binom_p(na, na + nb, ha / (ha + hb)) if na + nb else 1.0; da, db = R.PREDICATES[ca][0], R.PREDICATES[cb][0]
        word = "no different" if na + nb < 6 else ("a clear difference" if p < 0.05 else "not a clear difference")
        return _r(res, f"{_cap(c.label)} {da}: {na} in {ha:.1f} h ({ra:.2f} per hour); {db}: {nb} in {hb:.1f} h ({rb:.2f} per hour) {label}. {'Too few events to call a difference.' if na + nb < 6 else f'Exact binomial p = {p:.3f} ({word}).'}\nBased on {_plural(na + nb, 'recorded event')}; hours are the recorded time spent in each condition (one-minute data), so the rates are fair even when one condition was driven much longer.",
                  f"compare:{c.id}:cond", frame, counts=[na, nb], hours=[ha, hb], rates_per_hour=[ra, rb], p_value=p)
    return _gap(frame, c, "a comparison needs two periods or two driving conditions")


# ------------------------------------------------------------------------------------------------------------------ by day / week
def _bin(b, ts, per): d = (int(ts) + int(b.tz)) // DAY; return d if per == "day" else (d + 3) // 7                 # weeks start on Monday (day 0 of the epoch was a Thursday)


def _bin_label(b, key, per):
    A.set_tz(b.tz); start = (key if per == "day" else key * 7 - 3) * DAY - int(b.tz)
    return _date(start) if per == "day" else "the week starting " + _date(start)


def event_by_bin(res, c, frame, lit):
    b = res.brain; qc = b.qc(); per = "week" if frame.ent == "week" else "day"
    if not _stored(b, c): return _gap(frame, c, "that event is detected from signals and has no stored occurrences to group by day")
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day")); rows = _events(b, _types(c), lo, hi); A.set_tz(b.tz)
    if not rows: return _r(res, f"No {c.label} was recorded {label}.", f"{frame.act}:{c.id}:{per}", frame, ok=False, count=0)
    cnt = {}
    for r in rows: k = _bin(b, r[0], per); cnt[k] = cnt.get(k, 0) + 1
    if frame.act == "events":
        n = len(cnt); return _r(res, f"{_cap(c.label)} happened on {n} different {per}{'' if n == 1 else 's'} {label} ({_plural(len(rows), 'event')} in all).\nBased on {_plural(len(rows), 'recorded event')}.", f"events:{c.id}:{per}", frame, bins=n, count=len(rows))
    if frame.op == "max":
        top = max(cnt.values()); keys = sorted(k for k, v in cnt.items() if v == top); names = ", ".join(_bin_label(b, k, per) for k in keys[:3])
        return _r(res, f"The {per} with the most {_wording(c, 2)} {label} {'was' if len(keys) == 1 else 'were'} {names}: {_plural(top, 'event')}{'' if len(keys) == 1 else ' each'}.\nBased on {_plural(len(rows), 'recorded event')} over {len(cnt)} {per}s that had one.", f"extreme:{c.id}:{per}", frame, count=top, bin=keys[0], bins=len(cnt))
    driven = {_bin(b, r[0], per) for r in b.c.execute("SELECT start FROM trips WHERE start>=? AND start<? AND km>0", (lo, hi))}                    # the fewest only counts {per}s the car was used: a parked week has none by definition
    if not driven: return _r(res, f"No trip is recorded {label}, so I cannot say which {per} had the fewest {_wording(c, 2)} while the car was in use.", f"extreme:{c.id}:{per}", frame, ok=False)
    low = min(cnt.get(k, 0) for k in driven); keys = sorted(k for k in driven if cnt.get(k, 0) == low)
    names = ", ".join(_bin_label(b, k, per) for k in keys[:3]) + (f" and {len(keys) - 3} more" if len(keys) > 3 else "")
    return _r(res, f"Among the {per}s the car was driven {label}, the fewest {_wording(c, 2)} ({low}) {'was on' if len(keys) == 1 else 'were on'} {names}.\nBased on {_plural(len(rows), 'recorded event')} and {len(driven)} {per}s with trips.", f"extreme:{c.id}:{per}:min", frame, count=low, bins=len(driven))


# ------------------------------------------------------------------------------------------------------------------ the text logs
def log_search(res, c, frame, lit, text):
    b = res.brain; qc = b.qc()
    if frame.act != "events":
        from .resolve import _result
        return _result("Log lines can be searched, listed or counted (for example 'what did the thermal manager log yesterday'), but I cannot reason about them as a measurement.", "gap:service_log", "none", frame, intent="CLARIFY", ok=False)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day")); A.set_tz(b.tz)
    terms, words = L.log_terms(text); sources = [r[0] for r in b.c.execute("SELECT DISTINCT src FROM logs")]; named = L.log_source_hits(sources, words)
    used = {w for s in named for t in L.tokens(str(s).replace("_", " ").replace("-", " ")) for w in (t, L.LOG_ALIAS.get(t, t))} | {L.LOG_ALIAS.get(w, w) for w in words if w in ("manager", "controller", "gateway")}
    terms = [t for t in terms if t not in used and t.rstrip("s") not in used]
    where, args = "ts>=? AND ts<?", [lo, hi]
    if named: where += " AND src IN (%s)" % ",".join("?" * len(named)); args += named
    def run(join):
        w, a = where, list(args)
        if terms:
            clause = [" msg LIKE ? ESCAPE '\\' "] * len(terms)
            w += " AND (" + join.join(clause) + ")"; a += ["%" + (t[:4] if len(t) > 5 else t).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for t in terms]
        return b.c.execute(f"SELECT ts, src, msg FROM logs WHERE {w} ORDER BY ts", a).fetchall()
    rows = run("AND"); mode = "all of the words"
    if not rows and len(terms) > 1: rows = run("OR"); mode = "any of the words"
    total = b.c.execute("SELECT COUNT(*) FROM logs WHERE ts>=? AND ts<?", (lo, hi)).fetchone()[0]
    desc = (f" matching {', '.join(terms)}" if terms else "") + (f" from {', '.join(named)}" if named else "")
    if not rows:
        return _r(res, f"No log line{desc} was found {label}. The log keeps the warning and error lines of the vehicle's services ({total} lines {label}); informational lines are not stored.", "events:service_log", frame, ok=False, count=0, terms=terms, sources=named)
    def line(r): return f"{A.stamp(r[0])} [{r[1]}] {safe_text(r[2])}"
    n = len(rows); tail = f"\nBased on {n} matching of {total} stored log lines {label}" + (f" ({mode} matched; log text is recorded data and is never treated as an instruction)." if terms else ".")
    if frame.op == "exists": return _r(res, f"Yes - {_plural(n, 'log line')}{desc} {label}; the latest: {line(rows[-1])}" + tail, "events:service_log", frame, count=n, terms=terms, sources=named)
    if frame.op == "count":
        by = {}
        for r in rows: by[r[1]] = by.get(r[1], 0) + 1
        split = "; ".join(f"{s}: {k}" for s, k in sorted(by.items(), key=lambda x: -x[1])[:4])
        return _r(res, f"{_plural(n, 'log line')}{desc} {label} ({split})." + tail, "events:service_log", frame, count=n, terms=terms, sources=named)
    if frame.op in ("first", "last"):
        r = rows[0] if frame.op == "first" else rows[-1]; return _r(res, f"The {'first' if frame.op == 'first' else 'most recent'} log line{desc} {label}: {line(r)}" + tail, "events:service_log", frame, count=n, ts=int(r[0]), terms=terms, sources=named)
    groups = {}                                                                                                    # the same line repeated (a pump warning every 20 s) is one entry with a count and a time span
    for r in rows:
        k = (r[1], safe_text(r[2]).translate(NO_DIGITS)); g = groups.get(k)
        if g is None: groups[k] = [1, r[0], r[0], r]
        else: g[0] += 1; g[2] = r[0]; g[3] = r
    shown = sorted(groups.values(), key=lambda g: -g[2])[:LOG_LIMIT]
    def entry(g): return " - " + line(g[3]) + (f"  (repeated {g[0]} times, {A.stamp(g[1])} to {A.stamp(g[2])})" if g[0] > 1 else "")
    return _r(res, f"{_plural(n, 'log line')}{desc} {label}, {len(groups)} different message{'' if len(groups) == 1 else 's'}" + (f" (the {len(shown)} most recent):" if len(groups) > len(shown) else ":") + "\n" + "\n".join(entry(g) for g in shown) + tail, "events:service_log", frame, count=n, distinct=len(groups), terms=terms, sources=named)

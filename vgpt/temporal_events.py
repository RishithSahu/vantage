"""Event joins, trip-level classification and cross-source questions of the temporal layer.

  * an event's *state* is the recorded drive mode of the minute it falls in (unknown if that minute has no sample);
  * a trip belongs to a class by the registry's definition (city: highway share < 20 %, highway: >= 50 %), and a trip owns the events whose time lies inside [start, end];
  * "no events" is only reported for a period the events table covers - an empty result is not confused with missing evidence.
"""
import datetime as dt
import math
import re

import numpy as np

from . import signal_registry as R
from .query_ir import Query, TimeSpec, Condition, EventRef
from .query_planner import resolve_time, local_midnight, DAY
from . import composed as C
from .temporal import (STEP, BREAK, MODE, NUMRX, _num, threshold, to_signal_unit, Data, runs, run_seconds, fmt_dur, stamp, day_label, states_of, describe_states, _ans)
from .temporal_ops import Ctx, _nodata, pct_ok

EVENT_WORDS = [(r"hard[- ]?brak\w*|harsh[- ]?brak\w*|emergency stop", "HARSH_BRAKE"), (r"hard[- ]?accel\w*|harsh[- ]?accel\w*", "HARSH_ACCEL"), (r"hard[- ]?corner\w*", "HARSH_CORNER"),
               (r"battery[- ]?temp\w* warning\w*|over[- ]?temp\w*|overheat\w*|thermal warning\w*", "WARN_BATT_OVERTEMP"), (r"tyre[- ]?pressure warning\w*|tire[- ]?pressure warning\w*|tpms|tyre warning\w*", "TPMS_LOW"),
               (r"coolant (?:flow )?(?:fault|drop|warning)\w*|coolant[- ]flow (?:fault|drop)\w*", "COOLANT_FLOW_DROP"), (r"fault codes?|trouble codes?|dtcs?|diagnostic codes?", "DTC_SET"),
               (r"adas disengage\w*|lane[- ]assist disengage\w*", "ADAS_DISENGAGE"), (r"camera (?:block\w*|fault\w*)", "CAMERA_BLOCKED"), (r"\bwarnings?\b|\balerts?\b|\bfaults?\b", "ANY_WARNING")]


def event_type(t):
    for rx, ty in EVENT_WORDS:
        if re.search(rx, t): return ty
    return None


def fetch_events(qc, etype, lo, hi):
    """-> list of {'ts','type','detail','sev'} in [lo, hi) for a stored, grouped or detected event type."""
    q = Query(op="list_events", event=EventRef(type=etype, selector="all")); q.scope.time = TimeSpec("range", start=int(lo), end=int(hi))
    items, _label, _per, error = qc.executor._events(q)
    # A missing telemetry frame for an explicit condition is unknown, not an
    # empty event population.  The legacy helper has no error channel, so do
    # not manufacture a zero-count result.
    if error:
        return []
    return [dict(i) for i in items]


def modes_at(qc, ts_list, lo, hi):
    """Drive mode of the minute containing each timestamp (NaN where that minute is not recorded)."""
    d = Data(qc, [], lo - 120, hi + 120)
    out = []
    for ts in ts_list:
        if d.n == 0 or d.mode is None: out.append(float("nan")); continue
        i = int(np.searchsorted(d.ts, ts, side="right")) - 1
        out.append(float(d.mode[i]) if 0 <= i < d.n and ts < d.ts[i] + STEP + 5 else float("nan"))
    return out


def apply_states(mode, conds):
    ok = True
    for c in conds:
        if np.isnan(mode): return False
        name = c.predicate
        inside = {"parked": mode == 0, "city": mode == 1, "highway": mode == 2, "charging": mode == 3, "driving": mode in (1, 2), "moving": mode in (1, 2)}.get(name)
        if inside is None: return False
        ok = ok and ((not inside) if c.op == "not" else inside)
    return ok


def _double_negated(t):
    """"don't count warnings that weren't during charging": the events counted are those that were during charging."""
    return bool(re.search(r"\b(?:don't|do not|never|exclude|ignore|skip|without)\b.{0,25}\b(?:that|which)\b (?:weren't|were not|wasn't|was not|didn't|did not|aren't|are not|didn't happen)\b", t))


# ------------------------------------------------------------------ D: events under state conditions
def op_events_by_state(c):
    t = c.t; ty = event_type(t)
    if not ty: return None
    if re.search(r"\btrips?\b", t): return None
    lo, hi, label = c.window()
    if c.spec.type == "relative" and c.spec.value == 7 and not re.search(r"\b(?:last|past|this|yesterday|today|week|days?)\b", t): lo, hi, label = (c.qc.ctx.lo, c.qc.ctx.now + 1, "over the whole recorded history")
    conds = list(c.states)
    if _double_negated(t) and conds:
        conds = [Condition(predicate=x.predicate) if x.op == "not" else x for x in conds]
    if not conds: return None
    items = fetch_events(c.qc, ty, lo, hi)
    positives = [x for x in conds if x.op != "not" and x.predicate in MODE]
    if len(positives) >= 2:                                                                           # two mutually exclusive states named together are two groups to count, not one impossible conjunction
        modes_ = modes_at(c.qc, [i["ts"] for i in items], lo, hi)
        rest = [x for x in conds if x not in positives]
        parts = []; counts = {}
        for pst in positives:
            n_ = sum(1 for m_ in modes_ if not np.isnan(m_) and apply_states(m_, [pst] + rest))
            counts[pst.predicate] = n_
            parts.append(f"{n_} {R.EVENTS[ty][0]}{'' if n_ == 1 else 's'} {describe_states([pst])}")
        return _ans(f"{label.capitalize() if not label.startswith('the ') and not label.startswith('over') else 'Over ' + label.replace('over ', '', 1)}: " + "; ".join(parts) + (f", with the car {describe_states(rest)}" if rest else "") + ".", "temporal_event_groups", counts=counts)
    keep = []; unknown = 0
    modes = modes_at(c.qc, [i["ts"] for i in items], lo, hi)
    for it, m in zip(items, modes):
        if np.isnan(m): unknown += 1
        elif apply_states(m, conds): keep.append(it)
    lab = R.EVENTS[ty][0]
    cond_txt = describe_states(conds).replace("not driving", "not moving")
    notes = []
    if unknown: notes.append(f"{unknown} event{'s' if unknown != 1 else ''} fell in a minute with no recorded drive state and could not be classified")
    if len(conds) >= 2 and len({x.predicate for x in conds if x.predicate in MODE and x.op != "not"}) >= 2: notes.append("this vehicle records one drive state at a time, so two of those states cannot hold together")
    nt = (" Note: " + "; ".join(notes) + ".") if notes else ""
    n = len(keep)
    if re.search(r"\blist\b|\bshow\b|\bdisplay\b", t) and not re.search(r"\bhow many\b|\bcount\b", t):
        if not keep: return _ans(f"No {lab} was recorded {label} while the car was {cond_txt}.{nt}", "temporal_events", n=0)
        rows = "; ".join(f"{stamp(i['ts'], c.tz)} {i.get('detail', '')}".strip() for i in keep[:12])
        return _ans(f"{n} {lab}{'s' if n != 1 else ''} {label} while the car was {cond_txt}: {rows}{' ...' if n > 12 else ''}.{nt}", "temporal_events", n=n)
    if re.match(r"^(?:were|was|did|is|are|have|has)\b", t.strip()) and re.search(r"\bany\b", t):
        return _ans(f"{'Yes' if n else 'No'}. {n} {lab}{'s' if n != 1 else ''} {label} while the car was {cond_txt}.{nt}", "temporal_events", n=n, verdict=bool(n))
    return _ans(f"{n} {lab}{'s' if n != 1 else ''} {label} while the car was {cond_txt}.{nt}", "temporal_events", n=n)


# ------------------------------------------------------------------ trips
def trips_between(qc, lo, hi):
    cur = qc.conn.execute("SELECT start, end, km, kwh, wh_km, avg_speed, hwy_share FROM trips WHERE start >= ? AND start < ? AND km > 0.2 ORDER BY start", (int(lo), int(hi)))
    return [dict(zip(("start", "end", "km", "kwh", "wh_km", "avg_speed", "hwy"), r)) for r in cur.fetchall()]


def trip_class(tr): return "city" if (tr["hwy"] is not None and tr["hwy"] < 0.2) else "highway" if (tr["hwy"] is not None and tr["hwy"] >= 0.5) else "mixed"


def _window_all(c, t):
    lo, hi, label = c.window()
    if c.spec.type == "relative" and c.spec.value == 7 and not re.search(r"\b(?:last|past|this|yesterday|today|week|days?|month)\b", t): return c.qc.ctx.lo, c.qc.ctx.now + 1, "over the whole recorded history"
    return lo, hi, label


def op_trip_events(c):
    t = c.t
    if not re.search(r"\btrips?\b", t): return None
    ty = event_type(t) or ("HARSH_BRAKE" if re.search(r"\bbrak\w*", t) else None)
    if not ty: return None
    lo, hi, label = _window_all(c, t)
    trips = trips_between(c.qc, lo, hi)
    if not trips: return _ans(f"There are no recorded trips {label}.", "temporal_trip_events", ok=False)
    evs = fetch_events(c.qc, ty, lo, hi + 1)
    lab = R.EVENTS[ty][0]
    for tr in trips: tr["n"] = sum(1 for e in evs if tr["start"] <= e["ts"] <= tr["end"])
    th = None
    mw = re.search(r"\b(?:average |mean )?(?:energy )?consumption (?:exceeded|above|over|greater than|higher than|of more than|more than)\s*" + NUMRX + r"\s*wh/km", t) or re.search(r"\b(?:exceed\w*|above|over|greater than|more than)\s*" + NUMRX + r"\s*wh/km", t)
    if mw: th = _num(mw.group(1))
    cls = "city" if re.search(r"\bcity\b", t) else "highway" if re.search(r"\bhighway|motorway\b", t) else None
    sel = trips
    desc = []
    if th is not None: sel = [x for x in sel if x["wh_km"] is not None and x["wh_km"] > th]; desc.append(f"trips with average consumption above {th:g} Wh/km")
    if cls: sel = [x for x in sel if trip_class(x) == cls]; desc.append(f"{cls} trips")
    scope = " among ".join(reversed(desc)) if desc else "trips"
    notes = f" (events table covers the whole period; {len(trips)} trips of at least 0.2 km {label})"
    if re.search(r"\bno\b.*\b(?:brak\w*|events?)\b|\bwithout\b|\bnever had\b|\bzero\b|\bnot have\b", t) and not re.search(r"\bexclude\b", t) or re.search(r"\bhad no\b|\bwith no\b", t):
        none = [x for x in sel if x["n"] == 0]
        return _ans(f"{len(none)} of {len(sel)} {scope} {label} had no {lab} at all{notes[:-1] if False else ''}. ({sum(1 for x in sel if x['n'] > 0)} had at least one.)", "temporal_trip_events", n=len(none), of=len(sel))
    if re.search(r"\bexclude\b.*\bnever\b|\bonly trips? (?:that|which) had\b|\btrips? that (?:had|have) (?:any )?\w+ ?\w*events?\b", t):
        some = [x for x in sel if x["n"] > 0]
        rows = "; ".join(f"{stamp(x['start'], c.tz)} ({x['n']} {lab}{'s' if x['n'] != 1 else ''}, {x['km']:.0f} km)" for x in some[-10:])
        return _ans(f"After excluding trips that never had a {lab}, {len(some)} of {len(sel)} {scope} {label} remain" + (f"; the most recent: {rows}." if some else "."), "temporal_trip_events", n=len(some), of=len(sel))
    total = sum(x["n"] for x in sel)
    return _ans(f"{total} {lab}{'s' if total != 1 else ''} occurred on {len(sel)} {scope} {label} (an event belongs to the trip whose start-to-end interval contains it).{notes}", "temporal_trip_events", n=total, trips=len(sel))


def op_trip_extreme_excluding_latest(c):
    t = c.t
    if not (re.search(r"\bexcluding the (?:most recent|latest|last) trip\b|\bexcept (?:for )?the (?:most recent|latest|last) trip\b", t) and re.search(r"\b(?:worst|highest|most|least|best|lowest)\b", t)): return None
    lo, hi, label = _window_all(c, t)
    trips = trips_between(c.qc, lo, hi)
    if len(trips) < 2: return _ans("There are fewer than two recorded trips, so nothing remains after excluding the most recent.", "temporal_trip_extreme", ok=False)
    rest = trips[:-1]
    worst = re.search(r"\b(?:worst|highest|most|least efficient)\b", t) is not None
    best = (max if worst else min)([x for x in rest if x["wh_km"] is not None], key=lambda x: x["wh_km"])
    return _ans(f"Excluding the most recent trip ({stamp(trips[-1]['start'], c.tz)}), the trip with the {'worst' if worst else 'best'} energy consumption {label} was on {stamp(best['start'], c.tz)}: {best['wh_km']:.0f} Wh/km over {best['km']:.1f} km ({len(rest)} trips considered, each at least 0.2 km).", "temporal_trip_extreme", wh_km=best["wh_km"])


def op_trip_vs_category(c):
    t = c.t
    if not (re.search(r"\bfor each trip\b|\beach trip\b", t) and re.search(r"\b(?:same|category|class)\b", t) and re.search(r"\bconsumption|energy|efficien\w*\b", t)): return None
    lo, hi, label = _window_all(c, t)
    trips = [x for x in trips_between(c.qc, lo, hi) if x["wh_km"] is not None]
    if not trips: return _ans(f"There are no recorded trips {label}.", "temporal_trip_vs_category", ok=False)
    means = {}
    for k in ("city", "highway", "mixed"):
        v = [x["wh_km"] for x in trips if trip_class(x) == k]
        if v: means[k] = (sum(v) / len(v), len(v))
    rows = []
    for x in trips[-8:]:
        k = trip_class(x); mu, n = means[k]
        rows.append(f"{stamp(x['start'], c.tz)} {k}: {x['wh_km']:.0f} Wh/km vs {mu:.0f} ({x['wh_km'] - mu:+.0f})")
    cat = "; ".join(f"{k} {m:.0f} Wh/km over {n} trips" for k, (m, n) in means.items())
    return _ans(f"Category averages {label}: {cat}. The {len(rows)} most recent trips against their own category average (city = under 20% motorway, highway = 50% or more, otherwise mixed): " + " | ".join(rows) + f". {len(trips)} trips in total.", "temporal_trip_vs_category", means={k: v[0] for k, v in means.items()})


def op_events_between(c):
    t = c.t
    m = re.search(r"\bafter the (last|latest|most recent) charg\w*(?: session)?\b.*\bbefore the (first|next) trip\b", t)
    m2 = re.search(r"\bbetween the (last|latest) (?:over[- ]?temp\w*|battery[- ]?temp\w*)? ?warning\b.*\bnext (?:recorded )?trip\b|\bpassed between the (last|latest) (?:over[- ]?temp\w*|battery[- ]?temp\w*) warning and the next\b", t)
    if m:
        ch = c.qc.conn.execute("SELECT start, end FROM charges WHERE end <= ? ORDER BY end DESC LIMIT 1", (c.qc.ctx.now + 1,)).fetchone()
        if not ch: return _ans("There is no recorded charging session.", "temporal_events_between", ok=False)
        tr = c.qc.conn.execute("SELECT start FROM trips WHERE start >= ? AND km > 0.2 ORDER BY start LIMIT 1", (ch[1],)).fetchone()
        if not tr: return _ans(f"The last charging session ended {stamp(ch[1], c.tz)} and no trip is recorded after it, so the interval has no end.", "temporal_events_between", ok=False)
        ty = event_type(t) or "ANY_WARNING"
        evs = fetch_events(c.qc, ty, ch[1], tr[0])
        next_day = local_midnight(ch[1], c.tz) != local_midnight(tr[0], c.tz)
        lab = R.EVENTS[ty][0]
        body = "; ".join(f"{stamp(e['ts'], c.tz)} {e.get('detail', '')}".strip() for e in evs[:10]) if evs else "none"
        return _ans(f"Between the end of the last charging session ({stamp(ch[1], c.tz)}) and the next recorded trip ({stamp(tr[0], c.tz)}{'' if next_day else ', the same day'}) there {'was' if len(evs) == 1 else 'were'} {len(evs)} {lab}{'s' if len(evs) != 1 else ''}: {body}.", "temporal_events_between", n=len(evs))
    if m2 or re.search(r"\bminutes? passed between\b.*\bwarning\b.*\bnext\b.*\btrip\b", t):
        ty = event_type(t) or "WARN_BATT_OVERTEMP"
        evs = fetch_events(c.qc, ty, c.qc.ctx.lo, c.qc.ctx.now + 1)
        if not evs: return _ans("No such warning is recorded.", "temporal_events_between", ok=False)
        last = max(evs, key=lambda e: e["ts"])
        tr = c.qc.conn.execute("SELECT start FROM trips WHERE start >= ? AND km > 0.2 ORDER BY start LIMIT 1", (last["ts"],)).fetchone()
        if not tr: return _ans(f"The last {R.EVENTS[ty][0]} was at {stamp(last['ts'], c.tz)} and no trip is recorded after it.", "temporal_events_between", ok=False)
        mins = (tr[0] - last["ts"]) / 60.0
        return _ans(f"{mins:.0f} minutes ({fmt_dur(tr[0] - last['ts'])}) passed between the last {R.EVENTS[ty][0]} ({stamp(last['ts'], c.tz)}) and the start of the next recorded trip ({stamp(tr[0], c.tz)}).", "temporal_events_between", minutes=mins)
    return None


# ------------------------------------------------------------------ E: composed arithmetic
def op_first_last_days(c):
    t = c.t
    m = re.search(r"\b(?:in )?the (last|first) (\w+) days? .{0,60}\b(?:than|compared (?:to|with)|versus)\b.{0,30}\bthe (first|last) (\w+) days?\b", t)
    if not m or not c.signal: return None
    n1 = int(_num(m.group(2))); n2 = int(_num(m.group(4)))
    p = re.search(r"\b(?:same )?(\w+)[- ]day period\b", t)
    span = int(_num(p.group(1))) if p else 14
    now = c.qc.ctx.now; lo = now - span * DAY
    def win(which, n): return (lo, lo + n * DAY) if which == "first" else (now - n * DAY, now + 1)
    a = win(m.group(1), n1); b = win(m.group(3), n2)
    da = Data(c.qc, [c.signal], *a); db = Data(c.qc, [c.signal], *b)
    va, vb = da.vals.get(c.signal), db.vals.get(c.signal)
    if va is None or vb is None or da.n == 0 or db.n == 0: return _ans(f"One of the two periods has no recorded {c.sig_label()}, so they cannot be compared.", "temporal_first_last", ok=False)
    ma, mb = float(np.nanmean(va)), float(np.nanmean(vb))
    pct = (ma - mb) / mb * 100.0 if mb else float("nan")
    thr = re.search(r"\bat least " + NUMRX + r"%", t)
    unit = R.CONCEPTS[c.signal].unit
    lead = ""
    if thr and re.search(r"\blower\b", t):
        yes = pct <= -_num(thr.group(1)); lead = "Yes. " if yes else "No. "
    return _ans(lead + f"The average {c.sig_label()} was {ma:.2f} {unit} in the {m.group(1)} {n1} days and {mb:.2f} {unit} in the {m.group(3)} {n2} days of the same {span}-day period: {pct:+.1f}% ({da.n:,} and {db.n:,} samples).", "temporal_first_last", pct=pct)


def op_pair_difference_by_day(c):
    t = c.t
    if not (re.search(r"\bdifference\b", t) and re.search(r"\bbattery\b", t) and re.search(r"\bmotor\b", t) and re.search(r"\btemperature\b", t)): return None
    per_day = re.search(r"\beach day\b|\bper day\b|\bdaily\b|\bevery day\b", t); which = re.search(r"\bwhich day\b|\bon which day\b|\bwhat day\b", t)
    if not (per_day or which): return None
    d, label = c.load(["battery_temperature", "motor_temperature"])
    if d.n == 0 or not (d.has("battery_temperature") and d.has("motor_temperature")): return _nodata(c, label, "battery and motor temperature")
    diff = np.abs(d.vals["battery_temperature"] - d.vals["motor_temperature"]); ok = ~np.isnan(diff)
    local = ((d.ts + c.tz) // DAY) * DAY - c.tz
    rows = []
    for day in np.unique(local):
        m = (local == day) & ok
        if m.any():
            i = int(np.argmax(np.where(m, diff, -1))); rows.append((int(day), float(diff[i]), int(d.ts[i]), int(m.sum())))
    if not rows: return _nodata(c, label, "paired readings")
    unit = R.CONCEPTS["battery_temperature"].unit
    if which and not per_day:
        best = max(rows, key=lambda r: r[1])
        return _ans(f"The battery-to-motor temperature difference was greatest on {day_label(best[0], c.tz)}: {best[1]:.1f} {unit} at {stamp(best[2], c.tz)}. Each day's value is the largest absolute difference between the two temperatures recorded in the same minute ({best[3]} paired samples that day).", "temporal_pair_day", day=best[0], diff=best[1])
    return _ans(f"Largest battery-to-motor temperature difference per day {label} (absolute, same-minute pairs): " + "; ".join(f"{day_label(d_, c.tz)} {v:.1f} {unit}" for d_, v, _ts, _n in rows) + ".", "temporal_pair_day", days=len(rows))


def op_weekday(c):
    t = c.t
    if not (re.search(r"\bweekday\b|\bday of the week\b", t) and c.signal): return None
    lo, hi, label = _window_all(c, t)
    d = Data(c.qc, [c.signal], lo, hi)
    arr = d.vals.get(c.signal)
    if arr is None or d.n == 0: return _nodata(c, label, c.sig_label())
    wd = ((d.ts + c.tz) // DAY + 3) % 7
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows = []
    for k in range(7):
        m = (wd == k) & ~np.isnan(arr)
        if m.any(): rows.append((k, float(arr[m].mean()), int(m.sum()), len(np.unique(((d.ts[m] + c.tz) // DAY))) ))
    if len(rows) < 2: return _ans(f"Fewer than two weekdays have recorded data {label}.", "temporal_weekday", ok=False)
    hi_r = max(rows, key=lambda r: r[1]); lo_r = min(rows, key=lambda r: r[1]); unit = R.CONCEPTS[c.signal].unit
    allr = "; ".join(f"{names[k]} {v:.1f}" for k, v, _n, _d in rows)
    return _ans(f"The {names[hi_r[0]]} had the highest average {c.sig_label()} {label}: {hi_r[1]:.1f} {unit}, which is {hi_r[1] - lo_r[1]:.1f} {unit} higher than the lowest weekday ({names[lo_r[0]]}, {lo_r[1]:.1f} {unit}). All weekdays: {allr}. Each weekday is averaged over the {sum(r[3] for r in rows)} recorded calendar days, so unequal coverage per weekday matters ({', '.join(f'{names[k][:3]} {nd} day{chr(115) if nd != 1 else chr(0)[:0]}' for k, _v, _n, nd in rows)}).", "temporal_weekday", best=names[hi_r[0]], worst=names[lo_r[0]])


def op_charge_groups(c):
    t = c.t
    if not (re.search(r"\blast (?:three|3|two|2|four|4|five|5) charg\w*", t) and re.search(r"\b(?:before them|previous|preceding|earlier|the (?:three|3) before)\b", t)): return None
    n = int(_num(re.search(r"\blast (\w+) charg", t).group(1)))
    rows = c.qc.conn.execute("SELECT start, end, kw, kwh FROM charges WHERE end <= ? ORDER BY start DESC LIMIT ?", (c.qc.ctx.now + 1, 2 * n)).fetchall()
    if len(rows) < 2 * n: return _ans(f"Only {len(rows)} charging sessions are recorded, fewer than the {2 * n} needed to compare two groups of {n}.", "temporal_charge_groups", ok=False)
    A, B = rows[:n], rows[n:]
    # An energy comparison is only meaningful when every session in both
    # groups has a complete, finite interval and energy reading.  Silently
    # treating an absent value as zero would bias one group.
    required = A + B
    if any(r[0] is None or r[1] is None or r[3] is None for r in required):
        return _ans("One or more of the charging sessions needed for this comparison has incomplete timing or energy data, so I cannot calculate a fair group comparison.", "temporal_charge_groups", ok=False)
    if any(not math.isfinite(float(r[3])) or r[1] < r[0] for r in required):
        return _ans("One or more of the charging sessions needed for this comparison has invalid timing or energy data, so I cannot calculate a fair group comparison.", "temporal_charge_groups", ok=False)
    def agg(g):
        e = sum(r[3] for r in g); hrs = sum(max(r[1] - r[0], 1) for r in g) / 3600.0
        return e, (e / hrs if hrs else float("nan"))
    ea, pa = agg(A); eb, pb = agg(B)
    return _ans(f"Last {n} sessions (from {stamp(A[-1][0], c.tz)}): {ea:.1f} kWh in total, average power {pa:.1f} kW (energy over time spent charging). The {n} before them (from {stamp(B[-1][0], c.tz)}): {eb:.1f} kWh in total, average power {pb:.1f} kW. Difference: {ea - eb:+.1f} kWh and {pa - pb:+.1f} kW.", "temporal_charge_groups", a=(ea, pa), b=(eb, pb))


def op_tyre_deviation(c):
    t = c.t
    if not (re.search(r"\btyre|tire\b", t) and re.search(r"\bdeviat\w*|\bdiffer\w*|\baway from\b|\bfurthest\b|\bfarthest\b", t) and re.search(r"\brecommended|\bspec\w*|\bnominal\b|\bplacard\b", t)): return None
    meta = dict(c.qc.conn.execute("SELECT k, v FROM meta WHERE k LIKE 'veh.tyre_spec%'").fetchall())
    if not meta: return _ans("This vehicle's profile has no recommended tyre pressures recorded, so I cannot compute a deviation from them (I will not pick a reference).", "temporal_tyre_dev", ok=False)
    try:
        # Front and rear placards are separate vehicle facts.  A front value
        # is not a valid substitute for a missing rear recommendation.
        front = float(meta["veh.tyre_spec_front"])
        rear = float(meta["veh.tyre_spec_rear"])
    except (KeyError, TypeError, ValueError):
        return _ans("This vehicle's profile does not contain both front and rear recommended tyre pressures, so I cannot calculate deviations from them.", "temporal_tyre_dev", ok=False)
    if not math.isfinite(front) or not math.isfinite(rear):
        return _ans("This vehicle's recommended tyre pressures are invalid, so I cannot calculate deviations from them.", "temporal_tyre_dev", ok=False)
    vals = {}
    for pos in ("fl", "fr", "rl", "rr"):
        r = c.qc.execute(Query(op="current_value", target=f"tyre_pressure_{pos}"))
        v = float(r.result.value) if r.status == "answered" and r.result is not None and isinstance(r.result.value, (int, float)) else None
        if v is None:
            d = Data(c.qc, [f"tyre_pressure_{pos}"], c.qc.ctx.now - 3 * DAY, c.qc.ctx.now + 1)
            a = d.vals.get(f"tyre_pressure_{pos}")
            if a is not None and (~np.isnan(a)).any(): v = float(a[~np.isnan(a)][-1])
        vals[pos] = v
    if any(v is None for v in vals.values()): return _ans("A recent reading of every tyre is needed and is not available.", "temporal_tyre_dev", ok=False)
    names = {"fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right"}
    dev = {p: vals[p] - (front if p in ("fl", "fr") else rear) for p in vals}
    worst = max(dev, key=lambda p: abs(dev[p]))
    return _ans(f"The {names[worst]} tyre is furthest from its recommended pressure: {vals[worst]:.0f} kPa against {(front if worst in ('fl', 'fr') else rear):.0f} kPa recommended ({dev[worst]:+.0f} kPa). All four: " + "; ".join(f"{names[p]} {vals[p]:.0f} kPa ({dev[p]:+.0f})" for p in vals) + f". Recommended pressures come from the vehicle profile (front {front:.0f}, rear {rear:.0f} kPa, cold); the readings are the latest recorded and include warm-up.", "temporal_tyre_dev", worst=worst, deviation=dev[worst])


def op_overnight_daytime(c):
    t = c.t
    if not (re.search(r"\bovernight\b", t) and re.search(r"\bdaytime\b|\bduring the day\b", t) and re.search(r"\bincrease|rise\b", t)): return None
    sig = c.signal or "battery_temperature"
    if c.spec.type == "calendar" and c.spec.name in ("last_night", "this_morning"): c.spec = TimeSpec("relative", 14, "day")      # "overnight" names the part of the day, not the period asked about
    lo, hi, label = c.window()
    d = Data(c.qc, [sig], lo, hi); arr = d.vals.get(sig)
    if arr is None or d.n == 0: return _nodata(c, label, R.label_of(sig))
    hour = ((d.ts + c.tz) % DAY) / 3600.0
    def change(mask):
        idx = np.where(mask & ~np.isnan(arr))[0]
        if len(idx) < 4: return None
        # per contiguous period (a night / a day): last quarter-hour mean minus first quarter-hour mean, averaged over periods
        groups = []; cur = [idx[0]]
        for i in idx[1:]:
            if d.ts[i] - d.ts[cur[-1]] > 3 * 3600: groups.append(cur); cur = [i]
            else: cur.append(i)
        groups.append(cur)
        out = []
        for g in groups:
            if len(g) < 6: continue
            k = max(2, len(g) // 4); out.append(float(arr[g[-k:]].mean() - arr[g[:k]].mean()))
        return (float(np.mean(out)), len(out)) if out else None
    night = (hour >= 21) | (hour < 6); day = ~night
    cn, cd = change(night), change(day); unit = R.CONCEPTS[sig].unit
    if cn is None or cd is None: return _ans(f"There is not enough recorded {R.label_of(sig)} in both the overnight (21:00-06:00) and daytime (06:00-21:00) periods {label} to compare their changes.", "temporal_overnight", ok=False)
    if cn[0] >= 0 and cd[0] >= 0:
        conclusion = f"The {'overnight' if cn[0] > cd[0] else 'daytime'} increase was greater"
    elif cn[0] <= 0 and cd[0] <= 0:
        conclusion = f"Both periods cooled; {'overnight' if cn[0] < cd[0] else 'daytime'} cooled more"
    else:
        rising = "overnight" if cn[0] > 0 else "daytime"
        conclusion = f"{rising.capitalize()} increased while the other period decreased"
    return _ans(f"{conclusion}: average {R.label_of(sig)} changed by {cn[0]:+.1f} {unit} overnight (21:00 to 06:00, {cn[1]} nights) against {cd[0]:+.1f} {unit} during the day (06:00 to 21:00, {cd[1]} days). A change is the last quarter of a period's samples minus the first quarter, averaged over periods {label}.", "temporal_overnight", night=cn[0], day=cd[0])


OPS = [op_events_between, op_trip_extreme_excluding_latest, op_trip_vs_category, op_trip_events, op_events_by_state, op_first_last_days, op_pair_difference_by_day, op_weekday, op_charge_groups,
       op_tyre_deviation, op_overnight_daytime]
# Event-specialist routes do not all implement the typed condition mask.  Do
# not let one of them answer an explicitly conditioned request with an
# unfiltered population; the shared planner can apply the condition or
# return an honest clarification.
_CONDITION_UNSAFE = {op_first_last_days, op_pair_difference_by_day, op_weekday, op_overnight_daytime}


def handle(brain, qc, t, c=None):
    c = c or Ctx(brain, qc, t)
    for fn in OPS:
        if fn in _CONDITION_UNSAFE and c.states:
            continue
        try: r = fn(c)
        except Exception as e:
            import logging; logging.getLogger("vehiclegpt").warning("temporal event op %s failed: %s: %s", fn.__name__, type(e).__name__, e); r = None
        if r: return r
    return None

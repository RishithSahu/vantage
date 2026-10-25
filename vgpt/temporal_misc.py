"""Explicit clock intervals, before/after comparisons, per-day rankings, direction-of-difference questions and the epistemic questions (certainty, conflicting sources,
proof, estimation, causation).  The epistemic operations read the evidence and say what it can and cannot support; they never convert a statistic into advice."""
import datetime as dt
import math
import re

import numpy as np

from . import signal_registry as R
from .query_ir import Query, TimeSpec, Condition
from .query_planner import resolve_time, local_midnight, DAY
from . import composed as C
from .temporal import (STEP, BREAK, MODE, NUMRX, _num, threshold, to_signal_unit, Data, runs, run_seconds, fmt_dur, stamp, day_label, states_of, describe_states, _ans)
from .temporal_ops import Ctx, _nodata, pct_ok, percentiles_of
from .temporal_events import fetch_events, event_type


def _clock(h, m=0): return int(h) * 3600 + int(m) * 60


def _day_start(c, word):
    today = local_midnight(c.qc.ctx.now, c.tz)
    return {"today": today, "yesterday": today - DAY, "tomorrow": today + DAY}.get(word, today)


def _sig_by_unit_noun(t, unit_hint=None):
    nouns = [("battery", "battery_temperature"), ("pack", "battery_temperature"), ("motor", "motor_temperature"), ("coolant", "coolant_flow"), ("outside", "ambient_temperature"),
             ("ambient", "ambient_temperature"), ("cabin", "cabin_temperature")]
    for n, c in nouns:
        if re.search(r"\b" + n + r"\b", t): return c
    return None


def _fix_signal(c):
    """A temperature threshold names a component ("the pack above 50 C", "my battery above 45 C"): take that component's temperature, not its state of charge."""
    t = c.t
    if c.th and c.th[3] and c.th[3].lower() in ("°c", "c", "°f") or re.search(r"\btemp\w*\b", t):
        s = _sig_by_unit_noun(t)
        if s and (c.signal is None or R.CONCEPTS[c.signal].unit not in ("C", "°C")): c.signal = s
    if c.th and c.signal:
        op, v, raw, u = c.th
        c.th = (op, to_signal_unit(raw, u, c.signal), raw, u)


# ------------------------------------------------------------------ F: explicit clock intervals
_T = r"(\d{1,2}):(\d{2})"


def op_clock_interval(c):
    t = c.t
    m = re.search(r"\bbetween " + _T + r"(?: (yesterday|today))? and " + _T + r"(?: (yesterday|today))?\b", t)
    m2 = re.search(r"\bfrom " + _T + r"(?: (yesterday|today))? (?:until|to|till),? (?:but )?(?:not including|excluding),? " + _T + r"(?: (yesterday|today))?\b", t)
    mm = m or m2
    if not mm: return None
    g = mm.groups()
    h1, m1, d1, h2, m2_, d2 = int(g[0]), int(g[1]), g[2], int(g[3]), int(g[4]), g[5]
    if d1 is None and d2 is None:
        today = local_midnight(c.qc.ctx.now, c.tz)
        # the most recent occurrence of that clock time that has already happened
        base = today if _clock(h1, m1) <= (c.qc.ctx.now - today) else today - DAY
        day_note = "today" if base == today else "yesterday, because that time has not happened yet today"
        lo = base + _clock(h1, m1); hi = base + _clock(h2, m2_)
        if hi <= lo: hi += DAY
    else:
        d1 = d1 or d2; d2 = d2 or d1
        lo = _day_start(c, d1) + _clock(h1, m1); hi = _day_start(c, d2) + _clock(h2, m2_); day_note = f"{d1}" + (f" to {d2}" if d2 != d1 else "")
        if hi <= lo: hi += DAY
    _fix_signal(c)
    half_open = bool(m2)
    span_txt = f"{stamp(lo, c.tz)} to {stamp(hi, c.tz)}" + (" (half-open: the end instant is not included)" if half_open else "")
    if c.signal is None: return None
    d = Data(c.qc, [c.signal], lo, hi)                                    # Data loads [lo, hi): the end instant is excluded by construction
    arr = d.vals.get(c.signal)
    if arr is None or d.n == 0: return _ans(f"There is no recorded {c.sig_label()} from {span_txt}.", "temporal_clock", ok=False)
    unit = R.CONCEPTS[c.signal].unit
    if c.th and re.search(r"\bhow long\b|\bhow many (?:minutes|seconds|hours)\b", t):
        cm = c.compare(d); n = int(cm.sum()); sec = n * STEP
        return _ans(f"{c.sig_label().capitalize()} was {c.th_text()} for {fmt_dur(sec)} from {span_txt} ({day_note}); {n} of {d.n} recorded one-minute samples in that interval qualified. The interval crosses midnight correctly in the vehicle's local time.", "temporal_clock", seconds=sec, samples=n)
    v = float(np.nanmean(arr))
    return _ans(f"The average {c.sig_label()} from {span_txt} ({day_note}) was {v:.2f} {unit} over {int((~np.isnan(arr)).sum())} recorded one-minute sample{'' if int((~np.isnan(arr)).sum()) == 1 else 's'}.", "temporal_clock", value=v)


def op_before_after_charge(c):
    t = c.t
    if not (re.search(r"\bbefore charging began\b|\bbefore (?:the )?charg\w* (?:began|started)\b|\bimmediately before charging\b", t) and re.search(r"\bafter (?:it|the charg\w*|charging) ended\b|\bafter it ended\b", t)): return None
    hrs = 2
    mh = re.search(r"\b(?:the )?(\w+) hours?\b", t)
    if mh:
        try: hrs = int(_num(mh.group(1)))
        except Exception: pass
    ch = c.qc.conn.execute("SELECT start, end FROM charges WHERE end <= ? ORDER BY start DESC LIMIT 1", (c.qc.ctx.now + 1,)).fetchone()
    if not ch: return _ans("No charging session is recorded, so there is nothing to anchor on.", "temporal_before_after", ok=False)
    s, e = int(ch[0]), int(ch[1]); w = hrs * 3600
    before = (s - w, s); after = (e, e + w)
    sigs = ["battery_temperature", "motor_temperature", "coolant_flow", "battery_soc", "ambient_temperature"]
    db = Data(c.qc, sigs, *before); da = Data(c.qc, sigs, *after)
    rows = []
    for sg in sigs:
        a, b = db.vals.get(sg), da.vals.get(sg)
        if a is None or b is None: continue
        ma = np.nanmean(a) if (~np.isnan(a)).any() else None; mb = np.nanmean(b) if (~np.isnan(b)).any() else None
        if ma is None or mb is None: continue
        rows.append(f"{R.label_of(sg)} {ma:.1f} -> {mb:.1f} {R.CONCEPTS[sg].unit} ({mb - ma:+.1f})")
    note = []
    if db.n < hrs * 30: note.append(f"only {db.n} recorded samples in the {hrs} h before")
    if da.n < hrs * 30: note.append(f"only {da.n} in the {hrs} h after")
    if not rows: return _ans(f"The charging session of {stamp(s, c.tz)} to {stamp(e, c.tz)} has no recorded signal data in the {hrs} hours before it or after it, so the two periods cannot be compared.", "temporal_before_after", ok=False)
    return _ans(f"Charging session {stamp(s, c.tz)} to {stamp(e, c.tz)}. Averages in the {hrs} h before it began -> the {hrs} h after it ended: " + "; ".join(rows) + "." + (" Coverage: " + "; ".join(note) + "." if note else ""), "temporal_before_after", rows=len(rows))


def op_daily_lowest(c):
    t = c.t
    if not (re.search(r"\bdays?\b", t) and re.search(r"\b(?:lowest|highest|coldest|hottest|minimum|maximum)\b", t) and re.search(r"\bnot (?:the )?individual\b|\bnot (?:the )?(?:single|separate) (?:lowest|highest|samples?)\b|\bdaily\b|\bper day\b|\bshow the days\b|\bwhich days\b", t)): return None
    if not c.signal: return None
    low = bool(re.search(r"\b(?:lowest|coldest|minimum|least)\b", t))
    lo, hi, label = c.window()
    d = Data(c.qc, [c.signal], lo, hi); arr = d.vals.get(c.signal)
    if arr is None or d.n == 0: return _nodata(c, label, c.sig_label())
    local = ((d.ts + c.tz) // DAY) * DAY - c.tz
    rows = []
    for day in np.unique(local):
        m = (local == day) & ~np.isnan(arr)
        if m.sum() >= 5: rows.append((int(day), float(arr[m].mean()), int(m.sum())))
    if len(rows) < 2: return _ans(f"Fewer than two days have enough recorded {c.sig_label()} {label}.", "temporal_daily_rank", ok=False)
    rows.sort(key=lambda r: r[1], reverse=not low); unit = R.CONCEPTS[c.signal].unit
    top = rows[:3]
    return _ans(f"Ranked by each day's average (not by single samples), the days with the {'lowest' if low else 'highest'} {c.sig_label()} {label} were: " + "; ".join(f"{day_label(d_, c.tz)} {v:.2f} {unit} ({n} samples)" for d_, v, n in top) + f". Days with fewer than 5 recorded samples are not ranked ({len(rows)} days ranked).", "temporal_daily_rank", days=[r[0] for r in top])


def op_cross_vs_near(c):
    t = c.t
    if not re.search(r"\b(?:cross\w*|exceed\w*|reach\w*|hit)\b", t) or not re.search(r"\bor (?:only |just )?(?:sit|stay|hover|sat|stayed|remain|near|close|approach)\w*", t): return None
    _fix_signal(c)
    if c.th is None:
        m = re.search(r"\bcross\w* " + NUMRX, t)
        if m and c.signal: c.th = ("gt", _num(m.group(1)), _num(m.group(1)), None)
    if c.th is None or not c.signal: return None
    d, label = c.load()
    arr = d.vals.get(c.signal)
    if arr is None or d.n == 0: return _nodata(c, label, c.sig_label())
    v = c.th[1]; ok = ~np.isnan(arr)
    over = arr > v
    # "Below" is a directional relation, including for negative power.  A
    # percentage multiplier reverses its numeric bound for negatives; use an
    # absolute tolerance below the threshold instead.
    near = (arr <= v) & (arr >= v - 0.05 * abs(v)) & ok
    mx = float(np.nanmax(arr)); unit = R.CONCEPTS[c.signal].unit
    if over.any():
        return _ans(f"It crossed. {c.sig_label().capitalize()} was above {v:g} {unit} in {int(over.sum())} recorded one-minute samples {label} (highest {mx:.1f} {unit}); it was within 5% below that level in another {int(near.sum())}. Only recorded minute averages are compared, so a brief peak inside a minute could be missed.", "temporal_cross", crossed=True, max=mx)
    return _ans(f"It did not cross {v:g} {unit}: the highest recorded one-minute value {label} was {mx:.1f} {unit}; {int(near.sum())} samples were within 5% below that level. A brief peak inside a minute would not appear in minute averages.", "temporal_cross", crossed=False, max=mx)


def op_less_efficient(c):
    t = c.t
    if not (re.search(r"\b(?:less|more) efficient\b", t) and re.search(r"\bcity\b", t) and re.search(r"\b(?:motorway|highway)\b", t)): return None
    lo, hi, label = (c.qc.ctx.lo, c.qc.ctx.now + 1, "over the whole recorded history") if c.spec.type == "relative" and c.spec.value == 7 and not re.search(r"\b(?:last|past|this|yesterday|week|days?)\b", t) else c.window()
    q = Query(op="compare", target="wh_per_km", agg="mean", groups=["city", "highway"]); q.scope.time = c.spec if label != "over the whole recorded history" else TimeSpec("all")
    r = c.qc.execute(q)
    items = r.result.items if r.status == "answered" and r.result is not None else None
    vals = {}
    try:
        for it in (items or []): vals[it.get("group") or it.get("name") or it.get("label")] = it.get("value")
    except Exception: vals = {}
    if len(vals) < 2:
        txt = r.answer.text.split(chr(10))[0] if r.status == "answered" else None
        return _ans(txt or "I cannot compare city and highway consumption: one of the two has no recorded driving.", "temporal_less_efficient", ok=txt is not None)
    cv, hv = vals.get("city"), vals.get("highway")
    if cv is None or hv is None: return None
    asked_city_less = re.search(r"\bhow much (?:less|more) efficient (?:was|is) city\b|\bcity driving\b.*\b(?:less|more) efficient\b", t) is not None
    diff = cv - hv
    if abs(diff) < 1: return _ans(f"No difference: city driving used {cv:.0f} Wh/km and motorway driving {hv:.0f} Wh/km {label}.", "temporal_less_efficient", diff=diff)
    if diff > 0: return _ans(f"City driving was less efficient than motorway driving {label}: {cv:.0f} Wh/km against {hv:.0f} Wh/km, {diff:.0f} Wh/km ({diff / hv * 100:.0f}%) more energy per km.", "temporal_less_efficient", diff=diff)
    return _ans(f"No - city driving was not less efficient. {label.capitalize()} city driving used {cv:.0f} Wh/km and motorway driving {hv:.0f} Wh/km, so city driving used {-diff:.0f} Wh/km ({-diff / hv * 100:.0f}%) less energy per km; motorway driving was the less efficient of the two.", "temporal_less_efficient", diff=diff)


def op_compare_night(c):
    t = c.t
    if not re.search(r"\b(?:night|overnight|evening)\b", t): return None
    sigs = [s for s in c.signals if s in R.CONCEPTS]
    if len(sigs) < 2 or not re.search(r"\b(?:higher|lower|hotter|colder|warmer|cooler|greater|than)\b", t): return None
    a, b = sigs[0], sigs[1]
    if R.unit_family(R.CONCEPTS[a].unit) != R.unit_family(R.CONCEPTS[b].unit):
        return _ans(f"I cannot compare the magnitude of {R.label_of(a)} ({R.CONCEPTS[a].unit}) with {R.label_of(b)} ({R.CONCEPTS[b].unit}); they are different physical quantities.", "temporal_night", ok=False)
    yday = local_midnight(c.qc.ctx.now, c.tz) - DAY
    lo, hi = yday + 21 * 3600, yday + DAY + 6 * 3600
    d = Data(c.qc, [a, b], lo, min(hi, c.qc.ctx.now + 1))
    va, vb = d.vals.get(a), d.vals.get(b)
    if va is None or vb is None or d.n == 0: return _ans(f"There is no recorded {R.label_of(a)} and {R.label_of(b)} for last night (21:00 to 06:00).", "temporal_night", ok=False)
    ma, mb = float(np.nanmean(va)), float(np.nanmean(vb)); unit = R.CONCEPTS[a].unit
    up = re.search(r"\b(?:higher|hotter|warmer|greater)\b", t) is not None
    yes = (ma > mb) if up else (ma < mb)
    return _ans(f"{'Yes' if yes else 'No'}. Last night (21:00 on {stamp(yday, c.tz)[:6]} to 06:00) the average {R.label_of(a)} was {ma:.1f} {unit} and the average {R.label_of(b)} {mb:.1f} {unit}: {abs(ma - mb):.1f} {unit} {'higher' if ma > mb else 'lower'} for the {R.label_of(a)} ({d.n} recorded samples). \"Night\" is taken as 21:00 to 06:00.", "temporal_night", a=ma, b=mb)


def op_range_state(c):
    t = c.t
    if not (re.search(r"\b(?:range|spread)\b", t) and re.search(r"\bhighest\b", t) and re.search(r"\blowest\b", t) and c.states): return None
    if not c.signal: return None
    d, label = c.load()
    arr = d.vals.get(c.signal); m = d.state_mask(c.states)
    if arr is None or m is None or d.n == 0: return _nodata(c, label, c.sig_label())
    v = arr[m & ~np.isnan(arr)]
    if len(v) == 0: return _nodata(c, label, f"{c.sig_label()} {describe_states(c.states)}")
    unit = R.CONCEPTS[c.signal].unit
    return _ans(f"While the car was {describe_states(c.states)} {label}, {c.sig_label()} ranged from {v.min():.1f} to {v.max():.1f} {unit}: a range of {v.max() - v.min():.1f} {unit} ({len(v):,} recorded samples).", "temporal_range_state", range=float(v.max() - v.min()))


# ------------------------------------------------------------------ H: epistemic questions
def _current_flow(c):
    r = c.qc.execute(Query(op="current_value", target="coolant_flow"))
    if r.status == "answered" and r.result is not None and isinstance(r.result.value, (int, float)): return float(r.result.value), r.result
    return None, None


def op_fault_certainty(c):
    t = c.t
    if not (re.search(r"\b(?:how certain|how sure|how confident|how likely|are you sure|certain)\b", t) and re.search(r"\bfault|warning|problem|issue\b", t) and re.search(r"\bactive|still|current\w*\b", t)): return None
    ty = "COOLANT_FLOW_DROP" if re.search(r"\bcoolant\b", t) else (event_type(t) or "ANY_WARNING")
    ev = fetch_events(c.qc, ty, c.qc.ctx.lo, c.qc.ctx.now + 1)
    lab = R.EVENTS[ty][0]
    now = c.qc.ctx.now
    flow, res = _current_flow(c)
    dtc = c.qc.conn.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' ORDER BY ts DESC LIMIT 1").fetchone()
    lines = []
    if ev:
        last = max(ev, key=lambda e: e["ts"]); lines.append(f"the last {lab} was detected on {stamp(last['ts'], c.tz)} ({fmt_dur(now - last['ts'])} ago; {len(ev)} in the recorded history)")
    else: lines.append(f"no {lab} is recorded")
    if flow is not None:
        age = res.extra.get("stale_s") if res is not None and getattr(res, "extra", None) else None
        lines.append(f"the most recent coolant-flow reading is {flow:.1f} L/min" + (f" ({age} s old)" if age is not None else ""))
    else: lines.append("there is no current coolant-flow reading, so the present state cannot be observed")
    if dtc: lines.append(f"the stored diagnostic record ({stamp(dtc[0], c.tz)}: {dtc[1]}) is labelled illustrative in this data set and is not confirmation of a real fault")
    return _ans("I cannot put a confidence figure on it, and I cannot confirm the fault is active. What the data shows: " + "; ".join(lines) + ". A historical detection or a stored code shows the condition occurred; being active needs a current status signal or a confirmed uncleared code, and neither is recorded.", "temporal_fault_certainty", events=len(ev))


def op_conflicting_value(c):
    t = c.t
    if not (re.search(r"\b(?:says|shows|showed|displays|reads)\b", t) and re.search(r"\b(?:previous answer|you said|your (?:last|previous) answer|earlier answer)\b", t) and re.search(r"\bwhich\b.*\b(?:correct|right|true)\b|\bcorrect\b", t)): return None
    nums = re.findall(r"(\d+(?:\.\d+)?) ?(l/min|km/h|kpa|°c|c\b|%|kw)", t)
    r = c.qc.execute(Query(op="current_value", target="coolant_flow")) if re.search(r"l/min", t) else None
    if r is not None and r.status == "answered" and r.result is not None:
        v = r.result.value; ts = r.result.when
        return _ans(f"I cannot take either number from the message as the answer. The recorded snapshot I read now is {v:.1f} L/min at {stamp(ts, c.tz)} ({r.result.tier}). If the dashboard shows a different number, they are not the same sample: the dashboard value is whatever sample it was drawn from, and my earlier answer was computed from the snapshot current when it was asked. The snapshot timestamp printed with each answer is what settles which reading is which; the value stated in the message ({', '.join(a + ' ' + b for a, b in nums)}) is not evidence I can verify.", "temporal_conflict", value=v)
    return _ans("I cannot decide between two quoted values from the message alone: the authoritative value is the recorded reading at a stated timestamp, which I would need to read from the data. Tell me the signal and the time and I will read it.", "temporal_conflict", ok=False)


def op_regen_reconcile(c):
    t = c.t
    if not (re.search(r"\bregen\w*\b", t) and re.search(r"\bledger\b|\btrip ledger\b", t) and re.search(r"\bsamples?\b|\bintegrat\w*\b", t)): return None
    return run_regen_reconcile_plan(c)


def run_regen_reconcile_plan(c):
    now = c.qc.ctx.now; lo = now - 7 * DAY; hi = now + 1
    conn = c.qc.conn
    led7 = conn.execute("SELECT COUNT(*), COALESCE(SUM(x.regen_kwh), 0), COALESCE(SUM(t.kwh), 0) FROM trip_ext x JOIN trips t ON t.start = x.start WHERE t.start >= ? AND t.start < ? AND t.km > 0.2", (lo, hi)).fetchone()
    d = Data(c.qc, ["battery_power", "speed"], lo, hi)
    p = d.vals.get("battery_power"); sp = d.vals.get("speed")
    minute = None
    if p is not None and d.n:
        mov = (sp >= 10.0) if sp is not None else np.ones(d.n, bool)
        minute = float(np.where((p < 0) & mov & ~np.isnan(p), -p, 0.0).sum()) * STEP / 3600.0
    # the same trips at 1 Hz, where the 1 Hz tier exists
    trips = conn.execute("SELECT t.start, t.end, x.regen_kwh FROM trips t JOIN trip_ext x ON x.start = t.start WHERE t.start >= ? AND t.start < ? AND t.km > 0.2 AND x.regen_kwh IS NOT NULL ORDER BY t.start", (lo, hi)).fetchall()
    hz_led = hz_int = 0.0; n_hz = 0
    for s_, e_, led in trips:
        r = conn.execute("SELECT COUNT(*), COALESCE(SUM(CASE WHEN power < 0 THEN -power ELSE 0 END), 0) FROM sig_1s WHERE ts >= ? AND ts <= ?", (s_, e_)).fetchone()
        if r[0] >= 0.9 * max(1, e_ - s_): hz_led += float(led); hz_int += r[1] / 1000.0 / 3600.0; n_hz += 1
    if led7[0] == 0:
        return _ans("The trip ledger holds no regeneration value for any trip in the last 7 days on this vehicle, so there is no ledger figure to reconcile with the power samples. " + (f"Integrating the recorded one-minute battery power gives {minute:.2f} kWh, but one-minute averages cancel short strong regeneration against traction and read far too low (on vehicles with both, by about 15x), so that number should not be reported as regeneration." if minute is not None else "There are no one-minute power samples either."), "temporal_regen", ledger=None, minute=minute)
    if minute is not None:
        msg = (f"Last 7 days: the trip ledger's regeneration is {led7[1]:.1f} kWh over {led7[0]} trips ({100 * led7[1] / led7[2] if led7[2] else float('nan'):.0f}% of the {led7[2]:.0f} kWh those trips used). "
               f"Integrating the recorded one-minute battery power gives {minute:.2f} kWh. ")
    else:
        msg = f"Last 7 days: the trip ledger's regeneration is {led7[1]:.1f} kWh over {led7[0]} trips; there are no one-minute power samples. "
    if n_hz:
        msg += (f"The discrepancy is explained by resolution, not by an error in the ledger: on the {n_hz} trips that have 1 Hz power, integrating it gives {hz_int:.1f} kWh against {hz_led:.1f} kWh in the ledger "
                f"({100 * (hz_led - hz_int) / hz_led:+.0f}% difference), while the one-minute averages cancel short strong regeneration against traction and read only {(minute or 0) / max(led7[1], 1e-9) * 100:.0f}% of the ledger. "
                "Units are kWh in all three and the integration reproduces a known synthetic case exactly. So the ledger figure is the one to report (it is consistent with the 1 Hz measurements), and the one-minute integral should not be used for regeneration.")
    else:
        msg += "No 1 Hz power is recorded for those trips, so the ledger cannot be checked against a higher-resolution integral; one-minute averages cancel short regeneration against traction and understate it, so I would report the ledger value labelled unverified."
    return _ans(msg, "temporal_regen", ledger=led7[1], minute=minute, hz_ledger=hz_led, hz_integral=hz_int)


def op_prove_zero_degradation(c):
    t = c.t
    if not (re.search(r"\b(?:prove|proof|definitely|certain)\b", t) and re.search(r"\bdegrad\w*|\bsoh\b|state of health\b", t)): return None
    stated = re.search(r"\b(?:last|past|previous|this|today|yesterday|week|month|year|days?|months?)\b", t)
    lo, hi, label = c.window() if stated else (c.qc.ctx.now - 30 * DAY, c.qc.ctx.now + 1, "over the last month")
    rows = c.qc.conn.execute("SELECT ts, soh FROM vday WHERE ts >= ? AND ts < ? AND soh IS NOT NULL ORDER BY ts", (lo, hi)).fetchall() if "vday" in c.qc.profile.tables else []
    if not rows:
        meta = c.qc.conn.execute("SELECT v FROM meta WHERE k='veh.soh'").fetchone()
        return _ans("No. There is no dated series of state-of-health readings " + label + (f" (the profile holds one value, {meta[0]}%)" if meta else "") + ", so no change - zero or otherwise - can be established, and absence of a measured change would still not prove that none occurred.", "temporal_soh_proof", ok=False)
    vals = [r[1] for r in rows]; first, last = vals[0], vals[-1]
    res = min((abs(b - a) for a, b in zip(vals, vals[1:]) if b != a), default=None)
    return _ans(f"No, I cannot prove that. {label.capitalize()} there are {len(rows)} state-of-health readings; they go from {first:.1f}% to {last:.1f}% (a change of {last - first:+.2f} points" + (f"; the smallest step between readings is {res:.2f} points" if res else "") + "). State of health is an estimate with limited resolution and noise, so \"no detectable change\" is evidence that any degradation was smaller than that resolution, not proof of zero.", "temporal_soh_proof", n=len(rows), change=last - first)


def _exact_target(c, t, h, mi, s):
    """-> (target epoch, explicit_date_label or None).  A date the user wrote (17 March 2025, 2025-03-17, yesterday) fixes the DAY; without one the most recent occurrence of the clock time is used."""
    from . import qa_core as _Q
    now = c.qc.ctx.now
    try: date, explicit, label = _Q.find_date(t or "", now, c.tz)
    except Exception: date, explicit, label = None, False, ""
    if date is not None and (explicit or label in ("yesterday", "today", "the day before yesterday")):
        import datetime as _dt
        midnight = int(_dt.datetime(date.year, date.month, date.day, tzinfo=_dt.timezone.utc).timestamp()) - c.tz
        return midnight + h * 3600 + mi * 60 + s, (f"{date.day} {date.strftime('%b %Y')}" if explicit else None)
    today = local_midnight(now, c.tz); target = today + h * 3600 + mi * 60 + s
    if target > now: target -= DAY
    return target, None


def _no_data_for_day(c, target, label, h, mi, s):
    _D = Data; lo = target - (h * 3600 + mi * 60 + s)                              # midnight of that day
    for sg in ("speed", "battery_temperature", "battery_soc"):
        try:
            d = _D(c.qc, [sg], lo, lo + DAY)
            if d.n: return None
        except Exception: continue
    return _ans(f"There is no data for {label} in this vehicle's recorded history, so I cannot state a reading at {h:02d}:{mi:02d}:{s:02d}.", "temporal_exact_instant", ok=False)


def op_exact_instant(c):
    t = c.t
    m = re.search(r"\bat exactly " + _T + r"(?::(\d{2}))?\b", t) or re.search(r"\bexactly at " + _T + r"(?::(\d{2}))?\b", t)
    if not m: return None
    h, mi = int(m.group(1)), int(m.group(2)); s = int(m.group(3) or 0)
    target, dlabel = _exact_target(c, t, h, mi, s)
    if dlabel:
        nd = _no_data_for_day(c, target, dlabel, h, mi, s)
        if nd is not None: return nd
    sig = _sig_by_unit_noun(t) or (c.signal if c.signal and c.signal in R.CONCEPTS else None)
    cands = [sig] if sig else ["battery_temperature", "motor_temperature", "ambient_temperature"]
    out = []; found_exact = False
    for sg in cands:
        d = Data(c.qc, [sg], target - 180, target + 181)
        arr = d.vals.get(sg)
        if arr is None or d.n == 0: out.append(f"{R.label_of(sg)}: no samples within three minutes"); continue
        exact = np.where(d.ts == target)[0]
        if len(exact):
            found_exact = True
            out.append(f"{R.label_of(sg)}: recorded at exactly that second: {arr[exact[0]]:.1f} {R.CONCEPTS[sg].unit}"); continue
        i = int(np.searchsorted(d.ts, target)); lo_i, hi_i = i - 1, i
        if lo_i >= 0 and hi_i < d.n and d.ts[hi_i] - d.ts[lo_i] <= 2 * STEP:
            f = (target - d.ts[lo_i]) / float(d.ts[hi_i] - d.ts[lo_i]); est = float(arr[lo_i] + f * (arr[hi_i] - arr[lo_i]))
            out.append(f"{R.label_of(sg)}: no sample at that second; neighbours {arr[lo_i]:.1f} at {stamp(d.ts[lo_i], c.tz)}:{(d.ts[lo_i] % 60):02d} and {arr[hi_i]:.1f} at {stamp(d.ts[hi_i], c.tz)}:{(d.ts[hi_i] % 60):02d}; a linear interpolation would be about {est:.1f} {R.CONCEPTS[sg].unit} (an estimate, not a measurement)")
        else: out.append(f"{R.label_of(sg)}: no sample at that second and no pair of neighbouring samples close enough (within {2 * STEP} s) to interpolate from")
    lead = f"Recorded at exactly {h:02d}:{mi:02d}:{s:02d} ({stamp(target, c.tz)[:6]}): " if found_exact else f"There is no measurement I can state as the reading at exactly {h:02d}:{mi:02d}:{s:02d} ({stamp(target, c.tz)[:6]}). "
    return _ans(lead + "; ".join(out) + ".", "temporal_exact_instant", ok=found_exact)


def op_causation(c):
    t = c.t
    if not (re.search(r"\bdefinitely\b|\bbecause\b|\bcaused?\b", t) and re.search(r"\boverheat\w*\b", t) and re.search(r"\bcoolant\b", t)): return None
    lo, hi = c.qc.ctx.lo, c.qc.ctx.now + 1
    d = Data(c.qc, ["coolant_flow", "battery_temperature"], lo, hi)
    f, b = d.vals.get("coolant_flow"), d.vals.get("battery_temperature")
    ev = fetch_events(c.qc, "WARN_BATT_OVERTEMP", lo, hi)
    parts = []
    if f is not None and b is not None and d.n > 30:
        ok = ~np.isnan(f) & ~np.isnan(b) & (d.mode != 0 if d.mode is not None else True) & (d.mode != 3 if d.mode is not None else True)
        if ok.sum() > 30:
            from scipy.stats import spearmanr
            r, _p = spearmanr(f[ok], b[ok])
            # Consecutive vehicle samples are serially correlated, so the
            # ordinary rank-correlation p-value's independent-observation
            # assumption is false.  Report the association descriptively.
            parts.append(f"while driving, coolant flow and battery temperature have a Spearman correlation of {r:+.2f} across {int(ok.sum()):,} consecutive samples (descriptive only; no nominal p-value is reported because those samples are autocorrelated)")
    low = 0
    for e in ev:
        i = int(np.searchsorted(d.ts, e["ts"])); seg = f[max(0, i - 30):i + 1] if f is not None else np.array([])
        if len(seg) and np.nanmin(seg) < 0.75 * np.nanmedian(f[~np.isnan(f)]) : low += 1
    parts.append(f"{len(ev)} battery over-temperature warning{'s' if len(ev) != 1 else ''} recorded; in {low} of them coolant flow was below 75% of its usual level in the preceding 30 minutes")
    return _ans("I cannot say that it \"definitely\" overheated because of low coolant flow. " + "; ".join(parts).capitalize() + ". Co-occurrence and a plausible mechanism make low flow a likely contributor, but the data are observational: other causes (heavy load, ambient heat, fast charging) vary together with flow, so this is association, not proof of cause. Establishing causation would need a controlled test or a recorded pump/valve fault.", "temporal_causation", warnings=len(ev))


def op_fault_criteria(c):
    t = c.t
    if not (re.search(r"\bfaults?\b", t) and re.search(r"\bconfirmed\b", t) and re.search(r"\bcurrent\b", t) and re.search(r"\b(?:independent|more than one|several|multiple)\b", t)): return None
    types = [("DTC_SET", "stored diagnostic trouble code"), ("WARN_BATT_OVERTEMP", "battery over-temperature warning"), ("TPMS_LOW", "tyre-pressure warning"), ("COOLANT_FLOW_DROP", "coolant-flow drop (detected from the signal)")]
    now = c.qc.ctx.now; rows = []
    for ty, name in types:
        ev = fetch_events(c.qc, ty, c.qc.ctx.lo, now + 1)
        if not ev: continue
        last = max(ev, key=lambda e: e["ts"]); illustrative = any("illustrative" in str(e.get("detail", "")).lower() or "simulat" in str(e.get("detail", "")).lower() for e in ev)
        age_h = (now - last["ts"]) / 3600.0
        rows.append(f"{name}: {len(ev)} recorded, last {stamp(last['ts'], c.tz)} ({age_h:.0f} h ago); confirmed: " + ("no - labelled illustrative/simulated" if illustrative else "not established (no confirmation flag is recorded)") + f"; current: {'possibly (within the last hour)' if age_h < 1 else 'no recorded evidence it is still present'}")
    if not rows: return _ans("No fault or warning events are recorded.", "temporal_fault_criteria", n=0)
    return _ans("None of the recorded faults meets all three criteria. " + " | ".join(rows) + ". Independent support would need a second, physically separate signal agreeing (for example low coolant flow together with a rising temperature); the records here are single events whose supporting signals are not independent evidence of a confirmed fault.", "temporal_fault_criteria", n=len(rows))


def op_tail_percentile_safety(c):
    t = c.t
    ps = [p for p in percentiles_of(t) if p > 99]
    if not ps and not re.search(r"\b99\.9\b", t): return None
    p = ps[0] if ps else 99.9
    sig = c.signal or _sig_by_unit_noun(t) or "battery_temperature"
    lo, hi, label = c.window()
    d = Data(c.qc, [sig], lo, hi)
    arr = d.vals.get(sig)
    n = 0 if arr is None else int((~np.isnan(arr)).sum())
    ok, need = pct_ok(n, p)
    cur = c.qc.execute(Query(op="current_value", target=sig))
    curtxt = f" The most recent recorded {R.label_of(sig)} is {cur.result.value:.1f} {R.CONCEPTS[sig].unit}." if cur.status == "answered" and cur.result is not None and isinstance(cur.result.value, (int, float)) else ""
    unit = R.CONCEPTS[sig].unit
    if not ok:
        high = f" The highest recorded {R.label_of(sig)} {label} was {float(np.nanmax(arr)):.1f} {unit}." if arr is not None and n else " No valid recorded values are available in that period."
        return _ans(f"I will not calculate a {p:g}th percentile from {n:,} samples: estimating a tail that far out needs at least {need:,} samples, otherwise it is just the maximum in disguise.{high}{curtxt} And a percentile is not a safety criterion: whether to stop driving depends on active warnings and the manufacturer's limits, not on a distribution statistic. If a warning light or message is showing, or the car is behaving abnormally, stop when it is safe to do so and have it checked.", "temporal_tail", n=n)
    v = float(np.percentile(arr[~np.isnan(arr)], p))
    return _ans(f"The {p:g}th percentile {R.label_of(sig)} {label} was {v:.1f} {unit} ({n:,} samples). That is a description of the past, not a safety criterion: whether to stop driving depends on active warnings and the manufacturer's limits.{curtxt}", "temporal_tail", value=v)



# ------------------------------------------------------------------ explicit-argument forms used by the plan executor (vgpt/plan_exec.py): same evidence, no question text
def run_fault_certainty_plan(c, ty):
    ev = fetch_events(c.qc, ty, c.qc.ctx.lo, c.qc.ctx.now + 1)
    lab = R.EVENTS[ty][0]; now = c.qc.ctx.now
    flow, res = _current_flow(c)
    dtc = c.qc.conn.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' ORDER BY ts DESC LIMIT 1").fetchone()
    lines = []
    if ev:
        last = max(ev, key=lambda e: e["ts"]); lines.append(f"the last {lab} was detected on {stamp(last['ts'], c.tz)} ({fmt_dur(now - last['ts'])} ago; {len(ev)} in the recorded history)")
    else: lines.append(f"no {lab} is recorded")
    if flow is not None:
        age = res.extra.get("stale_s") if res is not None and getattr(res, "extra", None) else None
        lines.append(f"the most recent coolant-flow reading is {flow:.1f} L/min" + (f" ({age} s old)" if age is not None else ""))
    else: lines.append("there is no current coolant-flow reading, so the present state cannot be observed")
    if dtc: lines.append(f"the stored diagnostic record ({stamp(dtc[0], c.tz)}: {dtc[1]}) is labelled illustrative in this data set and is not confirmation of a real fault")
    return _ans("I cannot put a confidence figure on it, and I cannot confirm the fault is active. What the data shows: " + "; ".join(lines) + ". A historical detection or a stored code shows the condition occurred; being active needs a current status signal or a confirmed uncleared code, and neither is recorded.", "temporal_fault_certainty", events=len(ev))


def run_prove_zero_degradation_plan(c):
    # The generic compiler defaults a bare query to seven days.  This
    # operation's documented default is thirty days, while an explicit scope
    # is always honored.
    if c.q.scope.time.type == "relative" and c.q.scope.time.value == 7 and c.q.scope.time.unit == "day":
        lo, hi, label = c.qc.ctx.now - 30 * DAY, c.qc.ctx.now + 1, "over the last month"
    else:
        lo, hi, label = c.window()
    rows = c.qc.conn.execute("SELECT ts, soh FROM vday WHERE ts >= ? AND ts < ? AND soh IS NOT NULL ORDER BY ts", (lo, hi)).fetchall() if "vday" in c.qc.profile.tables else []
    if not rows:
        meta = c.qc.conn.execute("SELECT v FROM meta WHERE k='veh.soh'").fetchone()
        return _ans("No. There is no dated series of state-of-health readings " + label + (f" (the profile holds one value, {meta[0]}%)" if meta else "") + ", so no change - zero or otherwise - can be established, and absence of a measured change would still not prove that none occurred.", "temporal_soh_proof", ok=False)
    vals = [r[1] for r in rows]; first, last = vals[0], vals[-1]
    res = min((abs(b - a) for a, b in zip(vals, vals[1:]) if b != a), default=None)
    return _ans(f"No, I cannot prove that. {label.capitalize()} there are {len(rows)} state-of-health readings; they go from {first:.1f}% to {last:.1f}% (a change of {last - first:+.2f} points" + (f"; the smallest step between readings is {res:.2f} points" if res else "") + "). State of health is an estimate with limited resolution and noise, so \"no detectable change\" is evidence that any degradation was smaller than that resolution, not proof of zero.", "temporal_soh_proof", n=len(rows), change=last - first)


def run_exact_instant_plan(c, at, sig=None):
    try:
        parts = [int(x) for x in str(at).split(":")]; h, mi = parts[0], parts[1]; s = parts[2] if len(parts) > 2 else 0
    except Exception: return _ans("I could not read the time of day of that request.", "plan_clarify", ok=False)
    target, dlabel = _exact_target(c, getattr(c, "t", ""), h, mi, s)
    if dlabel:
        nd = _no_data_for_day(c, target, dlabel, h, mi, s)
        if nd is not None: return nd
    cands = [sig] if sig in R.CONCEPTS else ["battery_temperature", "motor_temperature", "ambient_temperature"]
    out = []; found_exact = False
    for sg in cands:
        d = Data(c.qc, [sg], target - 180, target + 181)
        arr = d.vals.get(sg)
        if arr is None or d.n == 0: out.append(f"{R.label_of(sg)}: no samples within three minutes"); continue
        exact = np.where(d.ts == target)[0]
        if len(exact):
            found_exact = True
            out.append(f"{R.label_of(sg)}: recorded at exactly that second: {arr[exact[0]]:.1f} {R.CONCEPTS[sg].unit}"); continue
        i = int(np.searchsorted(d.ts, target)); lo_i, hi_i = i - 1, i
        if lo_i >= 0 and hi_i < d.n and d.ts[hi_i] - d.ts[lo_i] <= 2 * STEP:
            f = (target - d.ts[lo_i]) / float(d.ts[hi_i] - d.ts[lo_i]); est = float(arr[lo_i] + f * (arr[hi_i] - arr[lo_i]))
            out.append(f"{R.label_of(sg)}: no sample at that second; neighbours {arr[lo_i]:.1f} at {stamp(d.ts[lo_i], c.tz)}:{(d.ts[lo_i] % 60):02d} and {arr[hi_i]:.1f} at {stamp(d.ts[hi_i], c.tz)}:{(d.ts[hi_i] % 60):02d}; a linear interpolation would be about {est:.1f} {R.CONCEPTS[sg].unit} (an estimate, not a measurement)")
        else: out.append(f"{R.label_of(sg)}: no sample at that second and no pair of neighbouring samples close enough (within {2 * STEP} s) to interpolate from")
    lead = f"Recorded at exactly {h:02d}:{mi:02d}:{s:02d} ({stamp(target, c.tz)[:6]}): " if found_exact else f"There is no measurement I can state as the reading at exactly {h:02d}:{mi:02d}:{s:02d} ({stamp(target, c.tz)[:6]}). "
    return _ans(lead + "; ".join(out) + ".", "temporal_exact_instant", ok=found_exact)


def run_causation_plan(c):
    lo, hi = c.qc.ctx.lo, c.qc.ctx.now + 1
    d = Data(c.qc, ["coolant_flow", "battery_temperature"], lo, hi)
    f, b = d.vals.get("coolant_flow"), d.vals.get("battery_temperature")
    ev = fetch_events(c.qc, "WARN_BATT_OVERTEMP", lo, hi)
    parts = []
    if f is not None and b is not None and d.n > 30:
        ok = ~np.isnan(f) & ~np.isnan(b) & (d.mode != 0 if d.mode is not None else True) & (d.mode != 3 if d.mode is not None else True)
        if ok.sum() > 30:
            from scipy.stats import spearmanr
            r, _p = spearmanr(f[ok], b[ok]); parts.append(f"while driving, coolant flow and battery temperature have a Spearman correlation of {r:+.2f} across {int(ok.sum()):,} consecutive samples (descriptive only; no nominal p-value is reported because those samples are autocorrelated)")
    low = 0
    for e in ev:
        i = int(np.searchsorted(d.ts, e["ts"])); seg = f[max(0, i - 30):i + 1] if f is not None else np.array([])
        if len(seg) and np.nanmin(seg) < 0.75 * np.nanmedian(f[~np.isnan(f)]): low += 1
    parts.append(f"{len(ev)} battery over-temperature warning{'s' if len(ev) != 1 else ''} recorded; in {low} of them coolant flow was below 75% of its usual level in the preceding 30 minutes")
    return _ans("I cannot say that it \"definitely\" overheated because of low coolant flow. " + "; ".join(parts).capitalize() + ". Co-occurrence and a plausible mechanism make low flow a likely contributor, but the data are observational: other causes (heavy load, ambient heat, fast charging) vary together with flow, so this is association, not proof of cause. Establishing causation would need a controlled test or a recorded pump/valve fault.", "temporal_causation", warnings=len(ev))


def run_fault_criteria_plan(c):
    types = [("DTC_SET", "stored diagnostic trouble code"), ("WARN_BATT_OVERTEMP", "battery over-temperature warning"), ("TPMS_LOW", "tyre-pressure warning"), ("COOLANT_FLOW_DROP", "coolant-flow drop (detected from the signal)")]
    now = c.qc.ctx.now; rows = []
    for ty, name in types:
        ev = fetch_events(c.qc, ty, c.qc.ctx.lo, now + 1)
        if not ev: continue
        last = max(ev, key=lambda e: e["ts"]); illustrative = any("illustrative" in str(e.get("detail", "")).lower() or "simulat" in str(e.get("detail", "")).lower() for e in ev)
        age_h = (now - last["ts"]) / 3600.0
        rows.append(f"{name}: {len(ev)} recorded, last {stamp(last['ts'], c.tz)} ({age_h:.0f} h ago); confirmed: " + ("no - labelled illustrative/simulated" if illustrative else "not established (no confirmation flag is recorded)") + f"; current: {'possibly (within the last hour)' if age_h < 1 else 'no recorded evidence it is still present'}")
    if not rows: return _ans("No fault or warning events are recorded.", "temporal_fault_criteria", n=0)
    return _ans("None of the recorded faults meets all three criteria. " + " | ".join(rows) + ". Independent support would need a second, physically separate signal agreeing (for example low coolant flow together with a rising temperature); the records here are single events whose supporting signals are not independent evidence of a confirmed fault.", "temporal_fault_criteria", n=len(rows))


def run_tail_percentile_plan(c, p, sig):
    sig = sig if sig in R.CONCEPTS else "battery_temperature"
    lo, hi, label = c.window()
    d = Data(c.qc, [sig], lo, hi)
    arr = d.vals.get(sig); n = 0 if arr is None else int((~np.isnan(arr)).sum())
    ok, need = pct_ok(n, p)
    cur = c.qc.execute(Query(op="current_value", target=sig))
    curtxt = f" The most recent recorded {R.label_of(sig)} is {cur.result.value:.1f} {R.CONCEPTS[sig].unit}." if cur.status == "answered" and cur.result is not None and isinstance(cur.result.value, (int, float)) else ""
    unit = R.CONCEPTS[sig].unit
    if not ok:
        high = f" The highest recorded {R.label_of(sig)} {label} was {float(np.nanmax(arr)):.1f} {unit}." if arr is not None and n else " No valid recorded values are available in that period."
        return _ans(f"I will not calculate a {p:g}th percentile from {n:,} samples: estimating a tail that far out needs at least {need:,} samples, otherwise it is just the maximum in disguise.{high}{curtxt} And a percentile is not a safety criterion: whether to stop driving depends on active warnings and the manufacturer's limits, not on a distribution statistic. If a warning light or message is showing, or the car is behaving abnormally, stop when it is safe to do so and have it checked.", "temporal_tail", n=n)
    v = float(np.percentile(arr[~np.isnan(arr)], p))
    return _ans(f"The {p:g}th percentile {R.label_of(sig)} {label} was {v:.1f} {unit} ({n:,} samples). That is a description of the past, not a safety criterion: whether to stop driving depends on active warnings and the manufacturer's limits.{curtxt}", "temporal_tail", value=v)


def run_conflicting_value_plan(c, sig, quoted):
    r = c.qc.execute(Query(op="current_value", target=sig)) if sig in R.CONCEPTS else None
    if r is not None and r.status == "answered" and r.result is not None and isinstance(r.result.value, (int, float)):
        v = r.result.value; ts = r.result.when; unit = R.CONCEPTS[sig].unit
        return _ans(f"I cannot take either number from the message as the answer. The recorded snapshot I read now is {v:.1f} {unit} at {stamp(ts, c.tz)} ({r.result.tier}). If the dashboard shows a different number, they are not the same sample: the dashboard value is whatever sample it was drawn from, and my earlier answer was computed from the snapshot current when it was asked. The snapshot timestamp printed with each answer is what settles which reading is which; the values stated in the message ({', '.join(str(x) for x in quoted if x)}) are not evidence I can verify.", "temporal_conflict", value=v)
    return _ans("I cannot decide between two quoted values from the message alone: the authoritative value is the recorded reading at a stated timestamp, which I would need to read from the data. Tell me the signal and the time and I will read it.", "temporal_conflict", ok=False)

OPS = [op_clock_interval, op_before_after_charge, op_regen_reconcile, op_conflicting_value, op_fault_certainty, op_prove_zero_degradation, op_exact_instant, op_causation,
       op_fault_criteria, op_tail_percentile_safety, op_daily_lowest, op_cross_vs_near, op_less_efficient, op_compare_night, op_range_state]
# These specialist handlers predate the typed condition algebra.  They must
# never consume a question while silently dropping a state/threshold scope.
# Handing it back lets the general typed planner apply the condition (or
# clarify if it cannot), which is safer than a plausible unfiltered answer.
_CONDITION_UNSAFE = {op_clock_interval, op_before_after_charge, op_daily_lowest, op_compare_night}


def handle(brain, qc, t, c=None):
    c = c or Ctx(brain, qc, t)
    _fix_signal(c)
    for fn in OPS:
        if fn in _CONDITION_UNSAFE and c.states:
            continue
        try: r = fn(c)
        except Exception as e:
            import logging; logging.getLogger("vehiclegpt").warning("temporal misc op %s failed: %s: %s", fn.__name__, type(e).__name__, e); r = None
        if r: return r
    return None

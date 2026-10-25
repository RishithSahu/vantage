"""Calendar breakdowns that need no training: weekends against weekdays, one line per day / week / month, and the pattern in WHEN events happen.

The model already writes the usual frame for these questions ("compare wh_per_km mean", "stat distance_driven sum", "events any_warning count"); what the frame cannot carry is the calendar part - which kind of day, which grain
of breakdown, whether a time pattern is asked.  Those are values read from the words like a date (`literals.day_grouping` -> `lit.daykinds`, `lit.per`, `lit.pattern`), and this module answers from the databases:

    weekend / weekday   compare | stat | trend on a trip-ledger quantity, a stored signal, an event rate, charging, or the fuel economy of a combustion car: the value per weekend day against per weekday, with a rank test of the daily values
    per day | week | month   stat | events | trend | extreme (month): one line per bin and the total or overall figure; first and last bins that are only partly inside the window are marked
    pattern in time     any event: share by part of the day, by weekday, and by what the car was doing (driving / charging / parked) against the share of time it spent so, with a chi-square test; "no clear pattern" when none holds

Every number is SQL over trips / charges / events / the one-minute tier; tests/test_round11_executors.py recomputes them.  Associations only.  The text of the question is not read here."""
import datetime as dt
import math
import re

from .. import analytics as A
from .. import signal_registry as R
from ..query_ir import TimeSpec
from . import digest as DG
from . import fuelx as FX
from ..qa_core import minmax_ts
from .eventx import DAY, _events, _r, _stored, _types, _window

LEDGER = ("wh_per_km", "energy_used", "distance_driven", "trip_record")
FUEL = ("fuel_rate", "wh_per_km", "energy_used")
MODES = {0: "parked", 1: "driving in town", 2: "driving on the highway", 3: "charging"}
BANDS = (("night (0-6 h)", 0, 6), ("morning (6-12 h)", 6, 12), ("afternoon (12-18 h)", 12, 18), ("evening (18-24 h)", 18, 24))
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MIN_KM = {"day": 5.0, "week": 20.0, "month": 50.0}
DEFAULT_DAYS = {"day": 14, "week": 56, "month": 365}
NL = chr(10)


# ------------------------------------------------------------------------------------------------------------------ small statistics
def _gammaincc(a, x):
    """Regularised upper incomplete gamma function Q(a, x) (series below a + 1, continued fraction above)."""
    if x <= 0: return 1.0
    if x < a + 1:
        term = total = 1.0 / a; n = a
        for _ in range(500):
            n += 1; term *= x / n; total += term
            if abs(term) < abs(total) * 1e-12: break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b = x + 1 - a; c = 1e300; d = 1.0 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2; d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d; c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c; d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1) < 1e-12: break
    return min(1.0, h * math.exp(-x + a * math.log(x) - math.lgamma(a)))


def chi_square_p(observed, expected):
    """p-value of the chi-square goodness of fit of `observed` counts to `expected` counts (same total)."""
    cells = [(o, e) for o, e in zip(observed, expected) if e > 0]
    if len(cells) < 2: return 1.0
    x2 = sum((o - e) ** 2 / e for o, e in cells); return _gammaincc((len(cells) - 1) / 2.0, x2 / 2.0)


def rank_test_p(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation with tie correction) of two lists of daily values."""
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3: return None
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b]); ranks = [0.0] * len(allv); i = 0; ties = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]: j += 1
        r = (i + j) / 2.0 + 1; t = j - i + 1; ties += t ** 3 - t
        for k in range(i, j + 1): ranks[k] = r
        i = j + 1
    r1 = sum(r for r, (v, g) in zip(ranks, allv) if g == 0); u1 = r1 - n1 * (n1 + 1) / 2.0; n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))) if n > 1 else 0.0
    if sd == 0: return 1.0
    z = (abs(u1 - n1 * n2 / 2.0) - 0.5) / sd; return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


def _num(v, nd=None):
    if nd is not None: return f"{v:,.{nd}f}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


# ------------------------------------------------------------------------------------------------------------------ calendar helpers (local days)
def _date(day): return dt.date.fromordinal(int(day) + 719163)                  # day 0 = 1970-01-01
def _dow(day): return (int(day) + 3) % 7                                       # Monday = 0 (1 Jan 1970 was a Thursday)


def _key(day, grain):
    if grain == "day": return int(day)
    if grain == "week": return (int(day) + 3) // 7
    d = _date(day); return d.year * 12 + d.month - 1


def _bin_days(key, grain):
    """(first local day, last local day) of a bin."""
    if grain == "day": return key, key
    if grain == "week": return key * 7 - 3, key * 7 + 3
    y, m = divmod(key, 12); a = dt.date(y, m + 1, 1); b = dt.date(y + (m == 11), (m + 1) % 12 + 1, 1)
    return a.toordinal() - 719163, b.toordinal() - 719163 - 1


def _label(key, grain):
    if grain == "day": return str(_date(key))
    if grain == "week": return "week of " + str(_date(key * 7 - 3))
    y, m = divmod(key, 12); return dt.date(y, m + 1, 1).strftime("%b %Y")


# ------------------------------------------------------------------------------------------------------------------ what a concept is made of
def _kind(res, c):
    """'ledger' | 'fuel' | 'charge' | 'event' | 'signal' | None."""
    cp = FX.caps_of(res); ice = FX.CAPS.is_combustion(cp)
    if c.id in ("fuel_rate", "wh_per_km", "energy_used") and ice and cp.get("fuel"): return "fuel"
    if c.id in LEDGER: return "ledger"
    if c.id == "charging_session": return "charge"
    if c.kind == "event": return "event" if c.registry_key and _stored_ok(c) else None
    if c.kind in ("signal", "declared", "derived"): return "signal"
    return None


def _stored_ok(c): return c.registry_key not in R.DETECTED_EVENTS


def _daily(res, c, frame, lit, text, lo, hi, agg):
    """{local day: value} of a stored quantity over [lo, hi): the daily tiers when the vehicle has them, else its minute rows."""
    from .pairs import minute_series
    cx = DG.Ctx(res, c, frame, lit, text); cx.lo, cx.hi = int(lo), int(hi)
    d = cx.daily(c.id, agg)
    if d: return {k: v for k, v in d.items() if (lo + int(res.brain.tz)) // DAY <= k <= (hi - 1 + int(res.brain.tz)) // DAY}
    s = minute_series(res.brain, c, lo, hi)
    if not s: return None
    tz = int(res.brain.tz); by = {}
    for t, v in s.items(): by.setdefault((t + tz) // DAY, []).append(v)
    f = {"AVG": lambda x: sum(x) / len(x), "MAX": max, "MIN": min}[agg]
    return {k: f(v) for k, v in by.items() if len(v) >= 5}


def _agg_of(frame): return "MAX" if frame.op == "max" else "MIN" if frame.op == "min" else "AVG"


def collect(res, c, frame, lit, text, lo, hi):
    """-> (kind, per-day data) where the data is {local day: record}; None when the vehicle cannot supply the quantity.
    ledger: (km, kwh, trips); fuel: (litres, km); charge: (sessions, kwh); event: count; signal: value."""
    b = res.brain; tz = int(b.tz); kind = _kind(res, c)
    if kind == "ledger":
        from .trendv import TRIP_COND
        w = TRIP_COND.get(frame.cond, ""); wa = (" AND " + w) if w else ""
        try: rows = b.c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, SUM(km), SUM(kwh), COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3" + wa + " GROUP BY d", (tz, int(lo), int(hi))).fetchall()
        except Exception: return None, None
        return kind, {int(d): (float(km or 0), float(kwh or 0), int(n)) for d, km, kwh, n in rows}
    if kind == "fuel":
        out = {}
        for d in range((int(lo) + tz) // DAY, (int(hi) - 1 + tz) // DAY + 1):
            e = FX.economy(b, max(d * DAY - tz, lo), min((d + 1) * DAY - tz, hi))
            if e and e["km"] >= 1.0: out[d] = (e["litres"], e["km"])
        return kind, out
    if kind == "charge":
        fast = " AND kind<>'AC'" if frame.cond == "fast_charging" else ""
        try: rows = b.c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, COUNT(*), SUM(kwh) FROM charges WHERE start>=? AND start<?" + fast + " GROUP BY d", (tz, int(lo), int(hi))).fetchall()
        except Exception: return None, None
        return kind, {int(d): (int(n), float(k or 0)) for d, n, k in rows}
    if kind == "event":
        out = {}
        for r in _events(b, _types(c), int(lo), int(hi)):
            d = (int(r[0]) + tz) // DAY; out[d] = out.get(d, 0) + 1
        return kind, out
    if kind == "signal":
        d = _daily(res, c, frame, lit, text, lo, hi, _agg_of(frame))
        return (kind, d) if d else (None, None)
    return None, None


def _clamp(b, lo, hi):
    """The period cut to the recorded history: months before the first record would only show zeros."""
    try: first, last = minmax_ts(b.c, "sig_1m")
    except Exception: return int(lo), int(hi)
    if first is None: return int(lo), int(hi)
    return max(int(lo), int(first)), min(int(hi), int(last) + 1)


def days_of(lo, hi, tz): return range((int(lo) + tz) // DAY, (int(hi) - 1 + tz) // DAY + 1)


def value_of(kind, c, frame, days, data, all_days):
    """-> (value, unit text, per-day values for the rank test, number of days it rests on) of a group of local days.  Totals are averaged per calendar day of the group (`all_days`: zero-activity days included)."""
    if kind == "ledger":
        have = [data[d] for d in days if d in data]
        if c.id == "wh_per_km":
            km = sum(x[0] for x in have); kwh = sum(x[1] for x in have)
            return (1000.0 * kwh / km if km >= 5 else None), " Wh/km", [1000.0 * data[d][1] / data[d][0] for d in days if d in data and data[d][0] >= 5 and data[d][1] > 0], len(have)
        i, u = {"distance_driven": (0, " km per day"), "energy_used": (1, " kWh per day"), "trip_record": (2, " trips per day")}[c.id]
        vals = [data[d][i] if d in data else 0.0 for d in days]
        return (sum(vals) / len(vals) if vals else None), u, vals, len(vals)
    if kind == "fuel":
        have = [data[d] for d in days if d in data]; km = sum(x[1] for x in have); lit_ = sum(x[0] for x in have)
        if c.id in ("fuel_rate", "wh_per_km") and frame.op != "sum": return (100.0 * lit_ / km if km >= 5 else None), " L/100 km", [100.0 * data[d][0] / data[d][1] for d in days if d in data and data[d][1] >= 5], len(have)
        vals = [data[d][0] if d in data else 0.0 for d in days]; return (sum(vals) / len(vals) if vals else None), " L per day", vals, len(vals)
    if kind == "charge":
        vals = [data[d][0] if d in data else 0 for d in days]; return (sum(vals) / len(vals) if vals else None), " charging sessions per day", vals, len(vals)
    if kind == "event":
        vals = [data.get(d, 0) for d in days]; return (sum(vals) / len(vals) if vals else None), f" {c.label}s per day", vals, len(vals)
    vals = [data[d] for d in days if d in data]
    return (sum(vals) / len(vals) if vals else None), "", vals, len(vals)


def _unit_of(res, c):
    from ..investigate.primitives import Data
    return DG._unit(Data(res.brain).unit(c.id) or getattr(c, "unit", "") or "") if c.kind != "event" else ""


# ------------------------------------------------------------------------------------------------------------------ weekend against weekday
def weekend_weekday(res, c, frame, lit, text, kinds):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); tz = int(b.tz)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 90, "day")); lo, hi = _clamp(b, lo, min(int(hi), int(b.now) + 1))
    kind, data = collect(res, c, frame, lit, text, lo, hi)
    if kind is None: return None
    ds = list(days_of(lo, hi, tz)); wk = [d for d in ds if _dow(d) >= 5]; wd = [d for d in ds if _dow(d) < 5]
    if len(wk) < 4 or len(wd) < 8: return _r(res, f"The period {label} has only {len(wk)} weekend days and {len(wd)} weekdays, too few to compare them.", "calx:weekend", frame, ok=False)
    vw, uw, xw, nw = value_of(kind, c, frame, wk, data, True); vd, ud, xd, nd = value_of(kind, c, frame, wd, data, True)
    if vw is None or vd is None: return _r(res, f"There is not enough recorded {c.label} on weekends or on weekdays {label} to compare them.", "calx:weekend", frame, ok=False)
    unit = uw or _unit_of(res, c); name = "fuel consumption" if kind == "fuel" and c.id != "energy_used" else "fuel used" if kind == "fuel" else c.label
    p = rank_test_p(xw, xd); pct = (vw - vd) / abs(vd) * 100.0 if abs(vd) > 1e-9 else None
    if len(kinds) == 1:                                                                                      # one kind of day named: its figure first, the other as the yardstick
        first, second = ("weekends", "weekdays") if kinds[0] == "weekend" else ("weekdays", "weekends"); a_, b_ = (vw, vd) if kinds[0] == "weekend" else (vd, vw)
        head = f"{_cap(name)} {label}: {_num(a_)}{unit} on {first}, against {_num(b_)}{unit} on {second}"
    else:
        head = f"{_cap(name)} {label}: {_num(vw)}{unit} on weekends, against {_num(vd)}{unit} on weekdays"
    if pct is not None and abs(pct) >= 0.5: head += f" ({abs(pct):.0f} % {'higher' if pct > 0 else 'lower'} on weekends)"
    head += "."
    test = ("" if p is None else f" Day-level rank test: p = {p:.3f} ({'a clear difference' if p < 0.05 else 'not clearly different'}).")
    base = f"Based on {len(wk)} weekend days and {len(wd)} weekdays of local time (Saturday and Sunday are the weekend)."
    cn = cond_note(frame, kind, c)
    return _r(res, head + test + NL + base + ((NL + cn) if cn else "") + NL + "Evidence: computed from recorded measurements; an association with the calendar, not a cause.", f"calx:weekend:{c.id}", frame, weekend=vw, weekday=vd, p=p, weekend_days=len(wk), weekdays=len(wd))


def _cap(s): return s[:1].upper() + s[1:]


def cond_note(frame, kind, c):
    """A condition the breakdown cannot apply is said, not dropped."""
    from .trendv import TRIP_COND
    cond = frame.cond
    if cond in ("none", "driving", "threshold"): return ""
    if kind == "ledger" and cond in TRIP_COND: return ""
    if kind == "charge" and cond == "fast_charging": return ""
    return f"The condition '{cond.replace('_', ' ')}' cannot be applied to this breakdown, so every {'event' if kind == 'event' else 'record'} in the period is used."


# ------------------------------------------------------------------------------------------------------------------ one line per day / week / month
def series(res, c, frame, lit, text, grain):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); tz = int(b.tz)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", DEFAULT_DAYS[grain], "day")); lo, hi = _clamp(b, lo, min(int(hi), int(b.now) + 1))
    kind, data = collect(res, c, frame, lit, text, lo, hi)
    if kind is None: return None
    ds = list(days_of(lo, hi, tz)); keys = sorted({_key(d, grain) for d in ds}); first_d, last_d = ds[0], ds[-1]; items = []
    for k in keys:
        a, z = _bin_days(k, grain); days = [d for d in range(a, z + 1) if first_d <= d <= last_d]; partial = a < first_d or z > last_d
        if kind == "ledger":
            have = [data[d] for d in days if d in data]
            if not have and c.id != "trip_record": continue
            km = sum(x[0] for x in have); kwh = sum(x[1] for x in have); n = sum(x[2] for x in have)
            if c.id == "wh_per_km":
                if km < MIN_KM[grain]: continue
                v, u = 1000.0 * kwh / km, " Wh/km"
            else: v, u = {"distance_driven": (km, " km"), "energy_used": (kwh, " kWh"), "trip_record": (float(n), " trips")}[c.id]
            extra = f" over {km:.0f} km" if c.id == "wh_per_km" else (f" ({n} trip{'s' if n != 1 else ''})" if c.id != "trip_record" else "")
        elif kind == "fuel":
            have = [data[d] for d in days if d in data]; km = sum(x[1] for x in have); lt = sum(x[0] for x in have)
            if not have: continue
            if c.id == "energy_used" or frame.op == "sum": v, u, extra = lt, " L", f" over {km:.0f} km"
            else:
                if km < MIN_KM[grain]: continue
                v, u, extra = 100.0 * lt / km, " L/100 km", f" ({lt:.0f} L over {km:.0f} km)"
        elif kind == "charge":
            v = float(sum(data.get(d, (0, 0))[0] for d in days)); kwh = sum(data.get(d, (0, 0))[1] for d in days); u, extra = " charging session" + ("" if v == 1 else "s"), f" ({kwh:.0f} kWh)"
        elif kind == "event":
            v = float(sum(data.get(d, 0) for d in days)); u, extra = f" {c.label}" + ("" if v == 1 else "s"), ""
        else:
            vals = [data[d] for d in days if d in data]
            if not vals: continue
            v = max(vals) if frame.op == "max" else min(vals) if frame.op == "min" else sum(vals) / len(vals); u, extra = _unit_of(res, c), ""
        items.append((k, v, u, extra, partial))
    if not items: return _r(res, f"There is not enough recorded {c.label} {label} to give it for each {grain}.", f"calx:{grain}", frame, ok=False)
    name = ("fuel used" if (c.id == "energy_used" or frame.op == "sum") else "fuel consumption") if kind == "fuel" else c.label + ("s" if kind == "event" and not c.label.endswith("s") else ""); unit0 = items[0][2]; cnt = kind in ("event", "charge") or c.id == "trip_record"
    if frame.act == "extreme" and frame.op in ("max", "min"):
        best = (max if frame.op == "max" else min)(items, key=lambda x: x[1]); rest = sorted(items, key=lambda x: x[1], reverse=frame.op == "max")[1:3]
        txt = f"The {grain} with the {('most' if frame.op == 'max' else 'fewest') if cnt else ('highest' if frame.op == 'max' else 'lowest')} {name} {label} was {_label(best[0], grain)}: {_num(best[1], 0 if cnt else None)}{best[2]}{best[3]}." + (NL + "Next: " + "; ".join(f"{_label(x[0], grain)}: {_num(x[1], 0 if cnt else None)}{x[2]}" for x in rest) + "." if rest else "")
        return _r(res, txt + NL + f"Based on {len(items)} {grain}s with data." + (" The first or last one is only partly inside the period." if any(x[4] for x in items) else "") + NL + "Evidence: computed from recorded measurements.", f"calx:{grain}:extreme", frame, value=best[1], key=best[0])
    shown = items[-14:]
    head = f"{_cap(name)} for each {grain} {label}" + (f" (the latest {len(shown)} of {len(items)} {grain}s)" if len(items) > len(shown) else "") + ":"
    lines = [" - " + _label(x[0], grain) + (" (partial)" if x[4] else "") + f": {_num(x[1], 0 if cnt or x[2].strip() == 'km' else None)}{x[2]}{x[3]}" for x in shown]
    vals = [x[1] for x in items]
    if kind == "ledger" and c.id == "wh_per_km" or (kind == "fuel" and unit0.strip() != "L") or kind == "signal": foot = f"Highest: {_label(max(items, key=lambda x: x[1])[0], grain)}; lowest: {_label(min(items, key=lambda x: x[1])[0], grain)}."
    else: foot = f"Total: {_num(sum(vals), 0 if cnt or unit0.strip() == 'km' else None)}{unit0 if kind != 'event' else ' ' + name} over {len(items)} {grain}s (average {_num(sum(vals) / len(vals), 1 if cnt else None)} per {grain})."
    txt = head + NL + NL.join(lines) + NL + foot
    cn = cond_note(frame, kind, c)
    return _r(res, txt + NL + ("A bin marked partial is only partly inside the period." + NL if any(x[4] for x in items) else "") + (cn + NL if cn else "") + "Evidence: computed from recorded measurements.", f"calx:{grain}:{c.id}", frame, values=[x[1] for x in items], keys=[x[0] for x in items])


# ------------------------------------------------------------------------------------------------------------------ the pattern in WHEN events happen
def when_pattern(res, c, frame, lit, text):
    if c.kind != "event" or not _stored_ok(c): return None
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); tz = int(b.tz)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 90, "day")); lo, hi = _clamp(b, lo, min(int(hi), int(b.now) + 1))
    rows = _events(b, _types(c), int(lo), int(hi)); n = len(rows)
    if n == 0: return _r(res, f"No {c.label} was recorded {label}, so there is no pattern in when it happens.", "calx:pattern", frame, count=0)
    if n < 12: return _r(res, f"Only {n} {c.label}{'' if n == 1 else 's'} {label}: too few to see a pattern in when they happen. The times: " + "; ".join(A.stamp(r[0]) for r in rows[-8:]) + ".", "calx:pattern", frame, count=n)
    ts = [int(r[0]) for r in rows]; days = list(days_of(lo, hi, tz)); parts = []; found = []
    hours = [((t + tz) % DAY) // 3600 for t in ts]; band = [sum(1 for h in hours if a <= h < z) for _, a, z in BANDS]
    base_w, base_name = [0.25] * 4, "if they were spread evenly over the day"
    try:                                                                                                  # most events happen while driving: judge their hour against the hours the car was driven, not against the clock
        drv = [0] * 4
        for (t,) in b.c.execute("SELECT ts FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2)", (int(lo), int(hi))).fetchall():
            h = ((int(t) + tz) % DAY) // 3600; drv[h // 6] += 1
        pick = ts[:: len(ts) // 400 + 1]; on_drive = 0
        for t in pick:
            r = b.c.execute("SELECT mode FROM sig_1m WHERE ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (t, t - 180)).fetchone(); on_drive += bool(r and r[0] in (1, 2))
        if sum(drv) >= 120 and on_drive / max(len(pick), 1) >= 0.7: base_w, base_name = [x / sum(drv) for x in drv], "if they followed the hours the car was driven"
    except Exception: pass
    p_band = chi_square_p(band, [n * w for w in base_w]); top = max(range(4), key=lambda i: band[i] - n * base_w[i])
    parts.append("time of day: " + ", ".join(f"{BANDS[i][0]} {100 * band[i] / n:.0f} %" for i in range(4)))
    if p_band < 0.01 and band[top] / n >= 1.4 * base_w[top] and band[top] / n >= 0.3: found.append(f"they cluster in the {BANDS[top][0]} ({100 * band[top] / n:.0f} % of them, against {100 * base_w[top]:.0f} % {base_name})")
    dow = [0] * 7
    for t in ts: dow[_dow((t + tz) // DAY)] += 1
    nd = [sum(1 for d in days if _dow(d) == k) for k in range(7)]; exp = [n * x / max(sum(nd), 1) for x in nd]; p_dow = chi_square_p(dow, exp)
    parts.append("weekday: " + ", ".join(f"{WEEKDAYS[k][:3]} {dow[k]}" for k in range(7)))
    if p_dow < 0.01:
        hot = [k for k in range(7) if exp[k] > 0 and dow[k] >= 1.5 * exp[k]]
        if hot: found.append("they come on " + ", ".join(WEEKDAYS[k] + "s" for k in hot) + " more than the number of such days would give")
    wk_share = sum(dow[5:]) / n
    exposure = {m: 0 for m in MODES}
    try:
        for m, k in b.c.execute("SELECT mode, COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? GROUP BY mode", (int(lo), int(hi))).fetchall():
            if m in exposure: exposure[m] = k
        sample = ts if len(ts) <= 2500 else ts[:: len(ts) // 2500 + 1]; at = {m: 0 for m in MODES}
        for t in sample:
            r = b.c.execute("SELECT mode FROM sig_1m WHERE ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (t, t - 180)).fetchone()
            if r and r[0] in at: at[r[0]] += 1
        tot_at = sum(at.values()); tot_ex = sum(exposure.values())
        if tot_at >= 10 and tot_ex > 0:
            expm = [tot_at * exposure[m] / tot_ex for m in MODES]; obs = [at[m] for m in MODES]; p_mode = chi_square_p(obs, expm)
            parts.append("what the car was doing: " + ", ".join(f"{MODES[m]} {100 * at[m] / tot_at:.0f} % (of its time {100 * exposure[m] / tot_ex:.0f} %)" for m in MODES if at[m] or exposure[m] / tot_ex >= 0.05))
            if p_mode < 0.01:
                over = [m for i, m in enumerate(MODES) if expm[i] > 0 and obs[i] >= 1.5 * expm[i] and obs[i] / tot_at >= 0.2]
                if over: found.append("they happen mostly while " + " or ".join(MODES[m] for m in over) + f" ({', '.join(f'{100 * at[m] / tot_at:.0f} % against {100 * exposure[m] / tot_ex:.0f} % of the time' for m in over)})")
    except Exception: pass
    concl = ("Pattern: " + "; ".join(found) + ".") if found else "No clear pattern: they are not concentrated in a part of the day, on particular weekdays, or in one driving state beyond what the time spent in it would give."
    txt = f"{_plural_(n, c.label)} {label}. " + NL + NL.join(" - " + x for x in parts) + NL + concl
    return _r(res, txt + NL + "Based on the recorded events (chi-square tests against an even spread, p < 0.01 to call a pattern); an association with the clock and the driving state, not a cause.", "calx:pattern", frame, count=n, found=found, weekend_share=wk_share)


def by_hour(res, c, frame, lit, text):
    """A quantity by the hour of the day it was recorded in ("which hour of the day has the highest average consumption"): the trip ledger grouped by the hour each trip started, or a stored signal averaged by clock hour.  Hours with too little
    data are left out and said so."""
    from .pairs import minute_series
    b = res.brain; qc = b.qc(); tz = int(b.tz); A.set_tz(b.tz); kind = _kind(res, c)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 90, "day")); lo, hi = _clamp(b, lo, min(int(hi), int(b.now) + 1)); MIN = 3
    if kind == "ledger":
        try: rows = b.c.execute("SELECT CAST(((start + ?) % 86400) / 3600 AS INTEGER) h, SUM(km), SUM(kwh), COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3 GROUP BY h", (tz, int(lo), int(hi))).fetchall()
        except Exception: return None
        if c.id == "wh_per_km": per = {int(h): (1000.0 * float(kwh) / float(km), int(n)) for h, km, kwh, n in rows if km and kwh and n >= MIN}; unit, nd = " Wh/km", 0
        elif c.id == "energy_used": per = {int(h): (float(kwh) / n, int(n)) for h, km, kwh, n in rows if kwh and n >= MIN}; unit, nd = " kWh per trip", 1
        elif c.id == "distance_driven" and re.search(r"\bmost\b|\btotal\b|\bsum\b|\bbusiest\b", " " + str(text).lower() + " "): per = {int(h): (float(km), int(n)) for h, km, kwh, n in rows if km and n >= MIN}; unit, nd = " km in total", 0
        elif c.id == "distance_driven": per = {int(h): (float(km) / n, int(n)) for h, km, kwh, n in rows if km and n >= MIN}; unit, nd = " km per trip", 1
        else: per = {int(h): (float(n), int(n)) for h, km, kwh, n in rows if n >= 1}; unit, nd = " trips", 0
        what, per_note = "trips that started in that hour", f"hours with fewer than {MIN} trips are left out"
    elif kind == "signal":
        s_ = minute_series(b, c, lo, hi)
        if not s_: return None
        if c.id not in DG.ANY_TIME and c.id not in DG.STATES:                                              # a quantity of driving is read while driving: parked minutes would pull every hour towards zero
            try: drv = {int(t) for (t,) in b.c.execute("SELECT ts FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2)", (int(lo), int(hi))).fetchall()}
            except Exception: drv = None
            if drv: s_ = {t: v for t, v in s_.items() if int(t) in drv}
        by = {}
        for t, v in s_.items(): by.setdefault(((int(t) + tz) % DAY) // 3600, []).append(v)
        per = {h: (sum(v) / len(v), len(v)) for h, v in by.items() if len(v) >= 30}; u = _unit_of(res, c); unit = (" " + u) if u else ""; nd = 1; what, per_note = "minutes", "hours with fewer than 30 recorded minutes are left out"
    else: return None
    if len(per) < 2: return _r(res, f"There is not enough data {label} to compare the hours of the day (at least {MIN} per hour are needed).", "calx:by_hour", frame, hours=len(per))
    top = max(per, key=lambda h: per[h][0]); low = min(per, key=lambda h: per[h][0]); nm = c.label
    pick = low if frame.op == "min" else top
    txt = f"{_cap(nm)} by hour of the day, {label}: the {'lowest' if pick == low else 'highest'} is between {pick:02d}:00 and {pick + 1:02d}:00 ({_num(per[pick][0], nd)}{unit}, {per[pick][1]} {what})."
    other = top if pick == low else low
    txt += f" The {'highest' if pick == low else 'lowest'} is between {other:02d}:00 and {other + 1:02d}:00 ({_num(per[other][0], nd)}{unit})."
    txt += NL + "By hour: " + "; ".join(f"{h:02d}h {_num(per[h][0], nd)}" for h in sorted(per)) + "." + NL + f"Based on the recorded {'trip ledger' if kind == 'ledger' else 'one-minute rows'}; {per_note}. An association with the clock, not a cause."
    return _r(res, txt, "calx:by_hour", frame, hours=len(per), best=int(pick))


def _plural_(n, w): return f"{n} {w}" + ("" if n == 1 else "s")


# ------------------------------------------------------------------------------------------------------------------ entry
def maybe(res, text, frame, lit, kinds, per, pattern):
    """-> an answer dict or None (the ordinary executors then take the frame)."""
    c = res.onto.concepts.get(frame.concept)
    if c is None or frame.act in ("define", "howto", "spec", "why", "context", "health", "advice", "explain", "at", "now", "estimate") or frame.c2 != "none": return None
    cp = FX.caps_of(res)
    if FX.CAPS.is_combustion(cp) and (c.id in FX.NO_BATTERY or c.id == "battery_soc"): return None                # a battery quantity on a fuel car is answered as "this vehicle has no traction battery" (fuelx)
    if pattern and c.kind == "event": return when_pattern(res, c, frame, lit, text)
    if pattern and c.kind != "event" and frame.act in ("extreme", "stat", "compare", "trend") and re.search(r"\bhour|time of day|part of (?:the )?day|morning|evening", " " + str(text).lower() + " "): return by_hour(res, c, frame, lit, text)
    if kinds and frame.act in ("compare", "stat", "trend", "extreme", "events"): return weekend_weekday(res, c, frame, lit, text, kinds)
    if per and frame.act in ("stat", "events", "trend", "extreme"):
        if frame.act == "extreme" and per != "month": return None                                         # per day / week extremes are binx / fuelx
        r = series(res, c, frame, lit, text, per)
        return r
    return None

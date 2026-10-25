"""Which day / which week was the highest or the lowest, for the quantities that are sums or ratios over trips: consumption (Wh/km), energy used, distance driven.

A stored signal's best day comes from the daily tier (resolve._by_day); these three are not stored signals - they exist only as the trip ledger - so they are computed from it, per local day or per week
(Monday to Sunday).  A ratio is only ranked on days / weeks with enough driving to mean something (5 km a day, 20 km a week), and the answer says how much driving the winner had.
Combustion vehicles' fuel consumption is the fuel executors' (fuelx.py)."""
import sqlite3

from .. import analytics as A
from ..query_ir import TimeSpec
from .eventx import DAY, _r, _window

MIN_KM = {"day": 5.0, "week": 20.0}


def _d(ts):
    t = A.stamp(ts)
    return t[:10] if t[:4].isdigit() else t.split(",")[0]
NAMES = {"wh_per_km": ("energy consumption", "Wh/km", 0), "energy_used": ("energy used", "kWh", 1), "distance_driven": ("distance driven", "km", 0)}


def extreme_by_bin(res, c, frame, lit):
    if c.id not in NAMES or frame.ent not in ("day", "week") or frame.op not in ("max", "min"): return None
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); tz = int(b.tz); unit_bin = frame.ent
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 90 if unit_bin == "week" else 30, "day")); hi = min(int(hi), int(b.now) + 1)
    expr = "CAST((start + ?) / 86400 AS INTEGER)" if unit_bin == "day" else "CAST((CAST((start + ?) / 86400 AS INTEGER) + 3) / 7 AS INTEGER)"          # weeks start on Monday (1 Jan 1970 was a Thursday)
    try: rows = b.c.execute(f"SELECT {expr} g, SUM(km), SUM(kwh), COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3 GROUP BY g", (tz, int(lo), hi)).fetchall()
    except sqlite3.Error: return None
    name, unit, nd = NAMES[c.id]; items = []
    for g, km, kwh, n in rows:
        if not km: continue
        if c.id == "wh_per_km":
            if km < MIN_KM[unit_bin] or not kwh: continue
            v = float(kwh) * 1000.0 / float(km)
        elif c.id == "energy_used":
            if kwh is None: continue
            v = float(kwh)
        else: v = float(km)
        start = (int(g) * 86400 - tz) if unit_bin == "day" else ((int(g) * 7 - 3) * 86400 - tz)
        items.append((v, start, float(km), int(n)))
    if not items: return _r(res, f"There is not enough driving {label} to rank {unit_bin}s by {name}" + (f" (at least {MIN_KM[unit_bin]:.0f} km is needed for a ratio)." if c.id == "wh_per_km" else "."), f"extreme:{c.id}:{unit_bin}", frame, ok=False)
    lowest = frame.op == "min"; items.sort(key=lambda x: x[0], reverse=not lowest); best = items[0]; word = "lowest" if lowest else "highest"
    when = (_d(best[1]) if unit_bin == "day" else "the week starting " + _d(best[1]))
    top = "; ".join(f"{_d(x[1])}: {x[0]:.{nd}f} {unit}" for x in items[1:3])
    return _r(res, f"The {unit_bin} with the {word} {name} {label} was {when}: {best[0]:.{nd}f} {unit}" + (f" over {best[2]:.0f} km in {best[3]} trip{'s' if best[3] != 1 else ''}." if c.id != "distance_driven" else f" in {best[3]} trip{'s' if best[3] != 1 else ''}.") +
              (f"\nNext: {top}." if top else "") + f"\nBased on the trip ledger, {len(items)} {unit_bin}s with enough driving.\nEvidence: computed from measurements, one record per trip.", f"extreme:{c.id}:{unit_bin}", frame, value=best[0], start=best[1])


def series_by_bin(res, c, frame, lit):
    """'Distance per week this month', 'energy each day last week', 'consumption per week': one line per local day or Monday-to-Sunday week from the trip ledger, newest last, with the total (or the overall figure) at the end."""
    if c.id not in NAMES or frame.ent not in ("day", "week") or frame.op not in ("sum", "mean", "count", "none"): return None
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); tz = int(b.tz); unit_bin = frame.ent; NL = chr(10)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 56 if unit_bin == "week" else 14, "day")); hi = min(int(hi), int(b.now) + 1)
    expr = "CAST((start + ?) / 86400 AS INTEGER)" if unit_bin == "day" else "CAST((CAST((start + ?) / 86400 AS INTEGER) + 3) / 7 AS INTEGER)"
    try: rows = b.c.execute(f"SELECT {expr} g, SUM(km), SUM(kwh), COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3 GROUP BY g ORDER BY g", (tz, int(lo), hi)).fetchall()
    except sqlite3.Error: return None
    name, unit, nd = NAMES[c.id]; items = []
    for g, km, kwh, n in rows:
        if not km: continue
        if c.id == "wh_per_km":
            if km < MIN_KM[unit_bin] or not kwh: continue
            v = float(kwh) * 1000.0 / float(km)
        elif c.id == "energy_used":
            if kwh is None: continue
            v = float(kwh)
        else: v = float(km)
        items.append((v, (int(g) * 86400 - tz) if unit_bin == "day" else ((int(g) * 7 - 3) * 86400 - tz), float(km), float(kwh or 0.0), int(n)))
    if not items: return _r(res, f"There is not enough driving {label} to give {name} for each {unit_bin}.", f"stat:{c.id}:{unit_bin}", frame, ok=False)
    shown = items[-14:]; head = f"{name.capitalize()} for each {unit_bin} {label}" + (f" (the latest {len(shown)} of {len(items)} {unit_bin}s)" if len(items) > len(shown) else "") + ":"
    lines = [" - " + ("week of " if unit_bin == "week" else "") + f"{_d(x[1])}: {x[0]:.{nd}f} {unit}" + (f" ({x[4]} trip{'s' if x[4] != 1 else ''})" if c.id != "wh_per_km" else f" over {x[2]:.0f} km") for x in shown]
    if c.id == "wh_per_km": tot = 1000.0 * sum(x[3] for x in items) / max(sum(x[2] for x in items), 1e-9); foot = f"Overall: {tot:.0f} {unit}."
    else: tot = sum(x[0] for x in items); foot = f"Total: {tot:.{nd}f} {unit} over {len(items)} {unit_bin}{'s' if len(items) != 1 else ''} with driving (average {tot / len(items):.{nd}f} {unit})."
    return _r(res, head + NL + NL.join(lines) + NL + foot + NL + "Based on the trip ledger." + NL + "Evidence: computed from measurements, one record per trip.", f"stat:{c.id}:{unit_bin}", frame, values=[x[0] for x in items])

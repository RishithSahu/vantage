"""Frame version 2: questions that need a second quantity, or the vehicle's own history as the yardstick.

    stat X corr c2=Y         "is my consumption related to my speed"            -> the correlation of two quantities over a period
    compare X c2=Y           "compare the battery and the motor temperature"     -> the two side by side over the same period
    events E exists c2=F     "did the battery or the motor overheat"            -> how often each happened, and whether either did
    compare X vs=typical     "is my consumption higher than usual"              -> this period against the vehicle's own usual values (a rolling history of windows of the same length)

The model reads WHICH quantities and WHAT is asked (the frame); everything here is arithmetic over the stored minute values.  Nothing is inferred beyond what the numbers say: a correlation is an association (never
a cause), a "usual" needs enough history to exist, and a quantity the vehicle does not store is said to be missing."""
import sqlite3
import statistics

from .. import analytics as A
from .. import fastcols as FC
from .. import signal_registry as R
from ..query_ir import TimeSpec
from . import fuelx as FX
from .eventx import DAY, _plural, _r, _window

MIN_PAIRS = 30                                   # fewer paired minutes than this say nothing reliable about a relationship
MIN_WINDOWS = 6                                  # fewer earlier windows than this make no "usual"


def minute_series(b, c, lo, hi):
    """{minute timestamp: value in the concept's unit} for a stored quantity, from the one-minute tier (or the vehicle's own lake), or None when the vehicle does not store it as a series."""
    store = R.STORE.get(c.id); tier = (store or {}).get("1m")
    try:
        if tier:
            table, col, scale = tier
            fast = FC.series(b.c, table, col, scale, int(lo), int(hi))
            if fast is not None: return fast
            return {int(ts): v / scale for ts, v in b.c.execute(f"SELECT ts, {col} FROM {table} WHERE ts>=? AND ts<? AND {col} IS NOT NULL", (int(lo), int(hi))).fetchall()}
        sid, _ = FX._sid(b, c.id)
        if sid is not None: return {int(ts): float(v) for ts, v in b.c.execute("SELECT ts, v FROM lk_1m WHERE sid=? AND ts>=? AND ts<? AND v IS NOT NULL", (sid, int(lo), int(hi))).fetchall()}
    except sqlite3.Error: return None
    return None


def _missing(res, frame, c, why="is not stored as a series by this vehicle"):
    return _r(res, f"{c.label.capitalize()} {why}, so I cannot answer this. Nothing was estimated from another quantity.", "pairs:missing", frame, ok=False)


def _unit(c): return "" if c.unit in ("none", "category", "count") else " " + c.unit


def strength(r):
    a = abs(r)
    return "no clear relationship" if a < 0.2 else "a weak relationship" if a < 0.4 else "a moderate relationship" if a < 0.7 else "a strong relationship"


TRIP_COLS = {"speed": ("avg_speed", "average trip speed"), "ambient_temperature": ("avg_amb", "average outside temperature of the trip"), "hvac_power": ("avg_hvac_w", "average climate power of the trip")}


def trip_corr(res, c, other, frame, lit):
    """Energy consumption is a ratio per trip, not a stored series: its relationship to a trip average (speed, outside temperature, climate power) is read over the trips of the trip ledger."""
    col, what = TRIP_COLS[other.id]; b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day")); hi = min(int(hi), int(b.now) + 1)
    try: rows = b.c.execute(f"SELECT {col}, wh_km FROM trips WHERE start>=? AND start<? AND wh_km IS NOT NULL AND {col} IS NOT NULL AND km>=2", (int(lo), hi)).fetchall()
    except sqlite3.Error: return None
    if len(rows) < 8: return _r(res, f"Only {len(rows)} trips {label} have both a consumption and {what} recorded; at least 8 are needed to say how they relate.", "pairs:corr", frame, ok=False, n=len(rows))
    xs = [float(r[0]) for r in rows]; ys = [float(r[1]) for r in rows]
    if statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0: return None
    r = statistics.correlation(xs, ys); way = "tend to rise and fall together" if r > 0 else "tend to move in opposite directions"
    return _r(res, f"Over {label}, energy consumption and the {what} show {strength(r)} (correlation {r:+.2f}, across {len(rows)} trips)" + (f": they {way}." if abs(r) >= 0.2 else ".") +
              "\nBased on the trip ledger: one consumption figure and one average per trip.\nEvidence: computed from measurements. A correlation is an association: it does not show that one causes the other.", "pairs:corr", frame, r=r, n=len(rows))


def correlate(res, c, c2, frame, lit):
    if "wh_per_km" in (c.id, c2.id):
        other = c2 if c.id == "wh_per_km" else c
        if other.id in TRIP_COLS and not minute_series(res.brain, c if c.id == "wh_per_km" else c2, 0, 1):
            r_ = trip_corr(res, c, other, frame, lit)
            if r_ is not None: return r_
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day")); hi = min(int(hi), int(b.now) + 1)
    s1, s2 = minute_series(b, c, lo, hi), minute_series(b, c2, lo, hi)
    if s1 is None: return _missing(res, frame, c)
    if s2 is None: return _missing(res, frame, c2)
    ts = sorted(set(s1) & set(s2)); xs = [s1[t] for t in ts]; ys = [s2[t] for t in ts]
    if len(ts) < MIN_PAIRS: return _r(res, f"Only {len(ts)} minutes {label} have both {c.label} and {c2.label} recorded; at least {MIN_PAIRS} are needed to say anything about how they relate.", "pairs:corr", frame, ok=False, n=len(ts))
    if statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return _r(res, f"{(c if statistics.pstdev(xs) == 0 else c2).label.capitalize()} did not change {label}, so there is no relationship to measure with the other.", "pairs:corr", frame, ok=False, n=len(ts))
    r = statistics.correlation(xs, ys); way = "tend to rise and fall together" if r > 0 else "tend to move in opposite directions"
    txt = (f"Over {label}, {c.label} and {c2.label} show {strength(r)} (correlation {r:+.2f})" + (f": they {way}." if abs(r) >= 0.2 else ".") +
           f"\nBased on {len(ts):,} minutes in which both were recorded (one-minute averages).\nEvidence: computed from measurements. A correlation is an association: it does not show that one causes the other.")
    return _r(res, txt, "pairs:corr", frame, r=r, n=len(ts))


def side_by_side(res, c, c2, frame, lit):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day")); hi = min(int(hi), int(b.now) + 1)
    rows = []
    for x in (c, c2):
        s = minute_series(b, x, lo, hi)
        if s is None: return _missing(res, frame, x)
        v = list(s.values())
        if not v: return _r(res, f"No {x.label} is recorded {label}, so the two cannot be put side by side.", "pairs:compare", frame, ok=False)
        rows.append((x, statistics.mean(v), max(v), min(v), len(v)))
    word = {"max": "highest", "min": "lowest"}.get(frame.op, "average"); lines = [f"{label.capitalize()}, side by side ({word}):"]
    for x, mean_, hi_, lo_, n in rows:
        v = {"max": hi_, "min": lo_}.get(frame.op, mean_)
        lines.append(f" - {x.label}: {v:.1f}{_unit(x)}" + ("" if frame.op in ("max", "min") else f" (highest {hi_:.1f}, lowest {lo_:.1f})"))
    return _r(res, "\n".join(lines) + f"\nBased on {rows[0][4]:,} and {rows[1][4]:,} one-minute samples.\nEvidence: computed from measurements.", "pairs:compare", frame, values=[{"max": r[2], "min": r[3]}.get(frame.op, r[1]) for r in rows])


def either(res, c, c2, frame, lit):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day")); hi = min(int(hi), int(b.now) + 1)
    counts = []
    for x in (c, c2):
        types = list(R.EVENT_GROUPS.get(x.registry_key, (x.registry_key,))) if x.registry_key else []
        if not types: return _missing(res, frame, x, "is not an event this vehicle records")
        marks = ",".join("?" * len(types)); n, last = b.c.execute(f"SELECT COUNT(*), MAX(ts) FROM events WHERE type IN ({marks}) AND ts>=? AND ts<?", (*types, int(lo), int(hi))).fetchone(); counts.append((x, n, last))
    total = sum(n for _, n, _ in counts)
    lines = [("Yes - " if total else "No - ") + f"{_plural(total, 'event')} of either kind {label}."] + [f" - {x.label}: {_plural(n, 'time')}" + (f", the latest {A.stamp(last)}" if n else "") for x, n, last in counts]
    return _r(res, "\n".join(lines) + "\nBased on the recorded events.", "pairs:either", frame, counts=[n for _, n, _ in counts])


def against_usual_trip(res, c, frame, lit):
    """'Did the battery heat up more than usual on my last trip': the latest trip's highest (or average) value of a stored signal against the same figure of the trips before it (at least six, the last 60 days)."""
    b = res.brain; A.set_tz(b.tz); NL = chr(10)
    try: trips = b.c.execute("SELECT start, end FROM trips WHERE end>start AND km>0.3 AND start>=? ORDER BY start", (int(b.now) - 60 * DAY,)).fetchall()
    except sqlite3.Error: return None
    if len(trips) < 7: return _r(res, "Fewer than seven trips are recorded in the last 60 days, which is too few to say what a usual trip is.", "pairs:usual_trip", frame, ok=False)
    s = minute_series(b, c, trips[0][0] - 60, trips[-1][1] + 60)
    if s is None: return _missing(res, frame, c)
    peak = frame.op == "max"; vals = []
    for st, en in trips:
        x = [v for t, v in s.items() if st <= t <= en]
        vals.append((max(x) if peak else statistics.mean(x)) if len(x) >= 3 else None)
    now_v, hist = vals[-1], sorted(v for v in vals[:-1] if v is not None)
    if now_v is None: return _r(res, f"No {c.label} is recorded during the last trip, so it cannot be compared with the usual.", "pairs:usual_trip", frame, ok=False)
    if len(hist) < 6: return _r(res, f"Only {len(hist)} earlier trips have {c.label} recorded, and at least 6 are needed to say what is usual. I will not guess.", "pairs:usual_trip", frame, ok=False)
    q = lambda p: hist[min(len(hist) - 1, max(0, int(round(p * (len(hist) - 1)))))]; where = "higher than usual" if now_v > q(0.9) else "lower than usual" if now_v < q(0.1) else "within the usual range"; below = sum(1 for x in hist if x < now_v) / len(hist)
    fmt = lambda v: f"{v:.1f}{_unit(c)}"; what = "highest" if peak else "average"
    return _r(res, f"On the last trip ({A.stamp(trips[-1][0])}) the {what} {c.label} was {fmt(now_v)}, which is {where} for a trip in this vehicle: the usual is about {fmt(q(0.5))} (the middle 80 % of the {len(hist)} earlier trips lie between {fmt(q(0.1))} and {fmt(q(0.9))}); this trip is higher than {below * 100:.0f} % of them." + NL + "Evidence: computed from measurements; usual means this vehicle's own recent trips.", "pairs:usual_trip", frame, value=now_v, typical=q(0.5), trips=len(hist))


def against_usual(res, c, frame, lit):
    """This period (default: the last 7 days) against the vehicle's own history: the same-length windows of the 90 days before it."""
    if frame.ent == "trip" and c.kind != "event" and c.id not in ("energy_used", "distance_driven"):
        r_ = against_usual_trip(res, c, frame, lit)
        if r_ is not None: return r_
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day")); hi = min(int(hi), int(b.now) + 1); span = max(hi - lo, 3600)
    event = c.kind == "event"
    if event:
        types = list(R.EVENT_GROUPS.get(c.registry_key, (c.registry_key,))) if c.registry_key else []
        if not types: return _missing(res, frame, c, "is not an event this vehicle records")
        marks = ",".join("?" * len(types)); count = lambda a, z: b.c.execute(f"SELECT COUNT(*) FROM events WHERE type IN ({marks}) AND ts>=? AND ts<?", (*types, int(a), int(z))).fetchone()[0]
        now_v = count(lo, hi); hist = [count(lo - k * span, lo - (k - 1) * span) for k in range(1, 13) if lo - k * span >= b.now - 120 * DAY]
    elif c.id in ("energy_used", "distance_driven") and b.c.execute("SELECT 1 FROM sqlite_master WHERE name='trips'").fetchone():                        # totals from the trip ledger: this period's total against the totals of the same-length periods before it
        col = "kwh" if c.id == "energy_used" else "km"
        tot = lambda a, z: (lambda r: float(r[0]) if r[1] else None)(b.c.execute(f"SELECT SUM({col}), COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.2", (int(a), int(z))).fetchone())
        now_v = tot(lo, hi)
        if now_v is None: return _r(res, f"No trips are recorded {label}, so {c.label} cannot be compared with the usual.", "pairs:usual", frame, ok=False)
        hist = [m for m in (tot(lo - k * span, lo - (k - 1) * span) for k in range(1, 13)) if m is not None]
    else:
        s = minute_series(b, c, lo - 12 * span, hi)
        if s is None: return _missing(res, frame, c)
        moving = c.id == "speed"                                                                  # an average speed means the speed while driving, not the parked minutes
        mean = lambda a, z: (lambda v: statistics.mean(v) if len(v) >= 10 else None)([x for t, x in s.items() if a <= t < z and (not moving or x > 1.0)])
        now_v = mean(lo, hi)
        if now_v is None: return _r(res, f"No {c.label} is recorded {label}, so it cannot be compared with the usual.", "pairs:usual", frame, ok=False)
        hist = [m for m in (mean(lo - k * span, lo - (k - 1) * span) for k in range(1, 13)) if m is not None]
    if len(hist) < MIN_WINDOWS: return _r(res, f"Only {len(hist)} earlier periods of the same length are recorded, and at least {MIN_WINDOWS} are needed to say what is usual for this vehicle. I will not guess.", "pairs:usual", frame, ok=False, windows=len(hist))
    hs = sorted(hist); q = lambda p: hs[min(len(hs) - 1, max(0, int(round(p * (len(hs) - 1)))))]; typical = statistics.median(hs); below = sum(1 for x in hs if x < now_v) / len(hs)
    where = "higher than usual" if now_v > q(0.9) else "lower than usual" if now_v < q(0.1) else "within the usual range"
    fmt = (lambda v: f"{v:.0f}") if event else (lambda v: f"{v:.1f}{_unit(c)}")
    what = f"there were {_plural(int(now_v), c.label)}" if event else f"the total {c.label} was {fmt(now_v)}" if c.id in ("energy_used", "distance_driven") else f"the {'average speed while driving' if c.id == 'speed' else 'average ' + c.label} was {fmt(now_v)}"
    txt = (f"{label.capitalize()}, {what}, which is {where} for this vehicle: the usual is about {fmt(typical)} (the middle 80 % of earlier periods lie between {fmt(q(0.1))} and {fmt(q(0.9))}); this period is higher than {below * 100:.0f} % of them.\n"
           f"Based on {len(hist)} earlier periods of the same length from this vehicle's own history.\nEvidence: computed from measurements; \"usual\" means this vehicle's own past, not other vehicles.")
    return _r(res, txt, "pairs:usual", frame, value=now_v, typical=typical, windows=len(hist))


def maybe(res, c, frame, lit):
    """-> an answer, or None when the frame is not one of the version-2 shapes (the ordinary executors then take it)."""
    O = res.onto; c2 = O.concepts.get(frame.c2) if frame.c2 != "none" else None
    if frame.c2 != "none" and c2 is None: return None
    if frame.act == "events" and c2 is not None: return either(res, c, c2, frame, lit)
    if frame.op == "corr" and c2 is not None: return correlate(res, c, c2, frame, lit)
    if frame.act == "compare" and c2 is not None: return side_by_side(res, c, c2, frame, lit)
    if frame.vs == "typical": return against_usual(res, c, frame, lit)
    return None

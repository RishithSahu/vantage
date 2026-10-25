"""A direct answer to a trend question: "is X rising?", "did X fall?", "has X changed?" are answered yes or no FIRST, from the daily values of the window, and then by whatever the typed executor found.

The statistics of the typed executor (a candidate change point with an adjusted p-value) are kept for the detail; what was missing is (a) the asked direction answered plainly and (b) a size below which a change does not
count however "significant" it is - a battery health of 91 % that moves to 91.1 % has not degraded.  A change counts only when the daily means of the second half of the window differ from the first half by at least 3 % of
their size AND by at least 2 % of the quantity's usual range, and the slope is not noise (Kendall's tau of the daily values against time, normal approximation, p < 0.10).  Nothing here reads the question: the
frame (rise / drop / change) says which direction was asked."""
import math

from . import digest as DG

DAY = 86400
MIN_REL, MIN_RANGE, MAX_P = 0.03, 0.02, 0.10
STEADY_TAU, STEADY_P, STEADY_REL = 0.5, 0.01, 0.005                    # a slow drift that is not "big" still counts when it is consistent: Kendall tau of at least 0.5 at p < 0.01 and a change of at least 0.5 %
TRIP_COND = {"cold": "avg_amb<5.0", "hot_weather": "avg_amb>28.0", "highway": "hwy_share>=0.5", "city": "hwy_share<0.5"}          # conditions a whole trip can answer (trip ledger columns)
DAY_COND = {"cold": ("<", 5.0), "hot_weather": (">", 28.0)}            # conditions a whole day can answer for a stored signal (the day's mean outside temperature)
LEDGER = ("wh_per_km", "energy_used", "distance_driven", "regen_energy")


def kendall(y):
    """(tau, p) of the values against their order; normal approximation without tie correction (daily means rarely tie)."""
    n = len(y)
    if n < 6: return 0.0, 1.0
    s = sum((y[j] > y[i]) - (y[j] < y[i]) for i in range(n) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18.0; z = (abs(s) - (1 if s else 0)) / math.sqrt(var)
    p = math.erfc(z / math.sqrt(2)); return s / (n * (n - 1) / 2.0), p


def derived_daily(cx, concept, per_trip=False):
    """{local day: value} for the quantities that are not a stored signal: consumption, energy and distance per day from the trip ledger (days with at least 5 km for the ratio), battery health from the daily table."""
    c = cx.b.c; tz = int(cx.tz); lo = int(cx.lo) - 2 * DAY; hi = int(cx.hi); w = TRIP_COND.get(getattr(cx.frame, "cond", "none"), ""); wa = (" AND " + w) if w else ""
    try:
        if concept in ("wh_per_km", "energy_used", "distance_driven"):
            rows = c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, SUM(km), SUM(kwh) FROM trips WHERE start>=? AND start<? AND km>0.3" + wa + " GROUP BY d", (tz, lo, hi)).fetchall()
            if concept == "distance_driven" and per_trip:
                rows = c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, AVG(km) FROM trips WHERE start>=? AND start<? AND km>0.3" + wa + " GROUP BY d", (tz, lo, hi)).fetchall()
                return {int(d): float(km) for d, km in rows if km}
            if concept == "distance_driven": return {int(d): float(km) for d, km, kwh in rows if km}
            if concept == "energy_used": return {int(d): float(kwh) for d, km, kwh in rows if kwh}
            return {int(d): float(kwh) * 1000.0 / float(km) for d, km, kwh in rows if km and km >= 5.0 and kwh}
        if concept == "motor_temperature":                                    # a combustion vehicle's drivetrain temperature = the engine's coolant (the same column an electric car uses for the motor)
            rows = c.execute("SELECT CAST((ts + ?) / 86400 AS INTEGER) d, AVG(motor) / 10.0 FROM sig_1m WHERE ts>=? AND ts<? AND motor IS NOT NULL GROUP BY d", (tz, lo, hi)).fetchall()
            return {int(d): float(v) for d, v in rows if v is not None}
        if concept == "regen_energy":
            rows = c.execute("SELECT CAST((t.start + ?) / 86400 AS INTEGER) d, SUM(x.regen_kwh) FROM trips t JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<?" + wa + " GROUP BY d", (tz, lo, hi)).fetchall()
            return {int(d): float(v) for d, v in rows if v is not None}
        if concept == "charging_power":                                       # the average power while charging, per day with a charge: the all-hours mean of the signal is mostly zeros
            rows = c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, SUM(kwh), SUM(end - start) FROM charges WHERE start>=? AND start<? AND end>start GROUP BY d", (tz, lo, hi)).fetchall()
            return {int(d): float(kwh) * 3600.0 / float(sec) for d, kwh, sec in rows if sec and kwh and sec >= 600}
        if concept == "battery_soh":
            rows = c.execute("SELECT CAST((ts + ?) / 86400 AS INTEGER) d, AVG(soh) FROM vday WHERE ts>=? AND ts<? AND soh IS NOT NULL GROUP BY d", (tz, lo, hi)).fetchall()
            return {int(d): float(v) for d, v in rows}
    except Exception: return None
    return None


def facts(cx, concept, per_trip=False):
    """-> dict(first, second, change, rel, tau, p, days, direction, slope_per_day, cond_note) from the daily means of the window, or None when the vehicle does not store the quantity or too few days are there.
    direction: rise / fall = a change of at least 3 % AND 2 % of the quantity's usual range that is not noise, or a SLOW but consistent drift (tau >= 0.5 at p < 0.01, at least 0.5 %: a tyre leaking, a battery ageing);
    noisy = big but the daily values vary too much; flat = neither."""
    d = (derived_daily(cx, concept, per_trip) if concept == "charging_power" else None) or (None if per_trip else cx.daily(concept, "AVG")) or derived_daily(cx, concept, per_trip)
    if not d: return None
    cond = getattr(cx.frame, "cond", "none"); note = ""
    if cond in DAY_COND and concept not in LEDGER and concept != "ambient_temperature":                       # the days on which the condition held (the day's mean outside temperature)
        amb = cx.daily("ambient_temperature", "AVG")
        if amb:
            op, lim = DAY_COND[cond]; keep = {k for k, v in amb.items() if (v < lim if op == "<" else v > lim)}; d = {k: v for k, v in d.items() if k in keep}
            if not d: return {"empty_condition": cond}
        else: note = f"The condition '{cond}' could not be applied (no outside temperature is recorded), so all days are used."
    elif cond not in ("none", "driving", "threshold") and not (cond in TRIP_COND and concept in LEDGER):
        note = f"The condition '{cond.replace('_', ' ')}' cannot be applied to a trend of {cx.label_of(concept)}, so all driving is used."
    win = cx.window(d, cx.lo, cx.hi); days = sorted(win)
    if len(days) < 6: return None
    y = [win[k] for k in days]; h = len(y) // 2; a, b = sum(y[:h]) / h, sum(y[len(y) - h:]) / h
    lo_, hi_ = cx.D.rng(concept); span = max(hi_ - lo_, 1e-9); ys = sorted(y); spread = ys[int(0.9 * (len(ys) - 1))] - ys[int(0.1 * (len(ys) - 1))]
    if spread > 0: span = min(span, 4.0 * spread)                             # the quantity's usual range: its sensor range, but no more than 4 times the spread (10th to 90th percentile) its own daily values show (a ratio's sensor range is absurdly wide)
    diff = b - a; rel = abs(diff) / max(abs(a), 1e-9); tau, p = kendall(y)
    big = rel >= MIN_REL and abs(diff) >= MIN_RANGE * span; steady = (not big) and len(days) >= 12 and p < STEADY_P and abs(tau) >= STEADY_TAU and rel >= STEADY_REL; real = (big and p < MAX_P) or steady
    xm = sum(days) / len(days); ym = sum(y) / len(y); var = sum((k - xm) ** 2 for k in days); slope = sum((k - xm) * (v - ym) for k, v in zip(days, y)) / var if var else 0.0
    return {"first": a, "second": b, "change": diff, "rel": rel, "tau": tau, "p": p, "days": len(days), "steady": steady and real, "slope_per_day": slope, "cond_note": note, "span_days": days[-1] - days[0] + 1,
            "direction": ("rise" if diff > 0 else "fall") if real else ("noisy" if big else "flat")}


def _num(v):
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def verdict(res, c, frame, lit, text):
    """-> (headline, detail) for a rise / drop / change question about a stored numeric quantity, or None."""
    if c.kind not in ("signal", "derived", "declared") or frame.op not in ("rise", "drop", "change"): return None
    cx = DG.Ctx(res, c, frame, lit, text)
    if not lit.times: cx.lo, cx.hi, cx.label = (cx.now - 3650 * DAY, cx.now + 1, "over the whole recorded history") if c.id == "battery_soh" else (cx.now - 28 * DAY, cx.now + 1, "in the last 28 days")          # ageing is read over the whole history
    f = facts(cx, c.id, per_trip=(frame.ent == "trip" and c.id == "distance_driven"))
    if f is None: return None
    if "empty_condition" in f:
        nm = f["empty_condition"].replace("_", " ")
        return f"No day {cx.label} had {nm} weather, so I cannot say how {cx.label_of(c.id)} changed under that condition.", "The condition was applied to the days (the day's mean outside temperature).", "flat", {"first": 0.0, "second": 0.0, "days": 0}
    lab = "the average trip length" if (frame.ent == "trip" and c.id == "distance_driven") else cx.label_of(c.id); u = DG._unit(cx.D.unit(c.id) or getattr(c, "unit", "") or ""); where = cx.label; want = {"rise": "rise", "drop": "fall", "change": None}[frame.op]; d = f["direction"]
    span = f"from {_num(f['first'])}{u} to {_num(f['second'])}{u} ({f['change'] / max(abs(f['first']), 1e-9) * 100:+.0f} %)"
    if d == "noisy":
        head = f"No reliable trend - {lab} averaged {_num(f['second'])}{u} in the second half {where} against {_num(f['first'])}{u} in the first ({f['change'] / max(abs(f['first']), 1e-9) * 100:+.0f} %), but the daily values vary too much to call that a change."
    elif d == "flat" and f["rel"] >= 0.25:                                                                       # a large relative move that is small next to the signal's whole range (a few kW of a 0-200 kW charger): not "the same"
        head = f"No reliable trend - {lab} averaged {_num(f['second'])}{u} in the second half {where} against {_num(f['first'])}{u} in the first ({f['change'] / max(abs(f['first']), 1e-9) * 100:+.0f} %), but that is small next to the usual range of the quantity."
    elif d == "flat":
        head = f"No - {lab} stayed about the same {where}: {_num(f['first'])}{u} in the first half, {_num(f['second'])}{u} in the second."
        if frame.op == "change": head = f"No - {lab} did not change in any meaningful way {where}: {_num(f['first'])}{u} in the first half, {_num(f['second'])}{u} in the second."
    elif want is None or want == d: head = f"Yes - {lab} {'rose' if d == 'rise' else 'fell'}{' steadily' if f['steady'] else ''} {where}, {span}."
    else: head = f"No - {lab} did not {want}; it {'rose' if d == 'rise' else 'fell'}{' steadily' if f['steady'] else ''} {where}, {span}."
    if d in ("rise", "fall") and f["span_days"] >= 40:                                                           # how fast: the least-squares slope of the daily means
        per, word = (f["slope_per_day"] * 365.25, "year") if f["span_days"] >= 150 else (f["slope_per_day"] * 30.4, "month")
        head += f" That is about {per:+.{1 if abs(per) < 10 else 0}f}{u} per {word}."
    detail = f"Based on {f['days']} daily averages; a change counts when the second half differs from the first by at least {MIN_REL * 100:.0f} % and {MIN_RANGE * 100:.0f} % of the usual range of the quantity and the trend is not noise (Kendall tau {f['tau']:+.2f}, p {f['p']:.3f}); a slow drift also counts when it is consistent (tau of at least 0.5 at p < 0.01)." + ((" " + f["cond_note"]) if f.get("cond_note") else "")
    return head, detail, d, f


def trip_trend(res, text, frame, c, lit):
    """-> a result dict: are there more / fewer trips per day (the trip ledger, days without a trip count as 0), answered yes / no first.  Same practical-size rule as verdict(): half-window means differ by 10 % and the slope is not noise."""
    from .resolve import _result
    cx = DG.Ctx(res, c, frame, lit, text)
    if not lit.times: cx.lo, cx.hi, cx.label = cx.now - 28 * DAY, cx.now + 1, "in the last 28 days"
    tz = int(cx.tz); lo, hi = int(cx.lo), int(min(cx.hi, cx.now + 1))
    try: rows = cx.b.c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3 GROUP BY d", (tz, lo, hi)).fetchall()
    except Exception: return None
    d0, d1 = (lo + tz) // DAY, (hi + tz) // DAY
    if d1 - d0 < 12: return _result(f"The period {cx.label} is too short (under 12 days) to say whether trips became more or less frequent.", "trend:trip_record", "trendv", frame, ok=False)
    cnt = {int(k): int(v) for k, v in rows}; y = [cnt.get(k, 0) for k in range(d0, d1)]; h = len(y) // 2; a, b = sum(y[:h]) / h, sum(y[len(y) - h:]) / h
    tau, p = kendall(y); rel = abs(b - a) / max(a, 1e-9); real = rel >= 0.10 and abs(b - a) >= 0.2 and p < MAX_P; d = ("rise" if b > a else "fall") if real else "flat"; want = {"rise": "rise", "drop": "fall", "change": None}[frame.op]
    span = f"from {a:.2f} to {b:.2f} trips a day ({(b - a) / max(a, 1e-9) * 100:+.0f} %)"
    if d == "flat" and rel >= 0.10: head = f"No reliable trend - trips per day averaged {b:.2f} in the second half {cx.label} against {a:.2f} in the first ({(b - a) / max(a, 1e-9) * 100:+.0f} %), but the daily counts vary too much to call that a change."
    elif d == "flat": head = f"No - the number of trips per day {'did not change in any meaningful way' if frame.op == 'change' else 'stayed about the same'} {cx.label}: {a:.2f} a day in the first half, {b:.2f} in the second."
    elif want is None or want == d: head = f"Yes - the number of trips per day {'rose' if d == 'rise' else 'fell'} {cx.label}, {span}."
    else: head = f"No - trips per day did not {want}; they {'rose' if d == 'rise' else 'fell'} {cx.label}, {span}."
    return _result(head + f"\nBased on {len(y)} days of the trip ledger, {sum(y)} trips; a change counts when the second half differs by at least 10 % and the trend is not noise (Kendall tau {tau:+.2f}, p {p:.3f}).\nEvidence: computed from measurements, one record per trip.", "trend:trip_record", "trendv", frame, trips=sum(y), first=a, second=b)


def add_verdict(res, text, frame, r, c, lit):
    """Put the verdict in front of the executor's answer; when the change is below the practical size, the candidate-change-point text (which would describe noise) is replaced by the one-line basis."""
    v = verdict(res, c, frame, lit, text)
    if v is None or not isinstance(r, dict) or not r.get("answer"): return r
    head, detail, d, f = v
    body = r["answer"]
    r = dict(r); stable = d in ("flat", "noisy") or "no clear" in body.lower()[:60]
    r["answer"] = head + "\n" + (detail if stable else detail + "\n" + body); r["brief"] = head
    facts_ = dict(r.get("facts") or {}); facts_["trend_verdict"] = {"direction": d, "first": f["first"], "second": f["second"], "days": f["days"], "steady": f.get("steady", False)}; r["facts"] = facts_
    return r

"""Advice that is made from the vehicle's own habits, not from a stock paragraph.

    battery_care   "how do I make the battery last longer", "should I charge to 100 every day": how the last 30 days were actually spent (time at a high charge, at a very low charge, the share of DC fast charges,
                   the hottest the pack got) and, next to each habit, the general rule that goes with it.  The general rules are labelled as such; the figures are the vehicle's.
Nothing is guessed: a figure the vehicle does not record is left out, and with no battery history the answer says so."""
import sqlite3

from .. import analytics as A
from .eventx import DAY, _r

HIGH, LOW, WARM = 90.0, 10.0, 35.0                                           # percent charge, percent charge, degrees C


def battery_care(res, frame, lit):
    b = res.brain; A.set_tz(b.tz); now = int(b.now); lo = now - 30 * DAY; c = b.c; facts = {}; lines = []
    try:
        soc = c.execute("SELECT COUNT(*), SUM(soc > ? ), SUM(soc < ?) FROM sig_1m WHERE ts>=? AND ts<=? AND soc IS NOT NULL", (HIGH * 10, LOW * 10, lo, now)).fetchone()
        tmax = c.execute("SELECT MAX(batt_max) / 10.0, SUM(batt_max > ?) FROM sig_1m WHERE ts>=? AND ts<=? AND batt_max IS NOT NULL", (WARM * 10, lo, now)).fetchone()
        ch = c.execute("SELECT COUNT(*), SUM(kind != 'AC'), SUM(soc1 >= 99) FROM charges WHERE start>=? AND start<=?", (lo, now)).fetchone()
    except sqlite3.Error: return None
    if not soc or not soc[0]: return _r(res, "This vehicle has no battery charge history recorded, so I cannot base battery-care advice on how it is used.", "advice:battery_care", frame, ok=False)
    n = soc[0]; hi_pct, lo_pct = 100.0 * (soc[1] or 0) / n, 100.0 * (soc[2] or 0) / n; facts.update(high_pct=hi_pct, low_pct=lo_pct)
    lines.append(f"From your last 30 days: the battery spent {hi_pct:.0f} % of the time above {HIGH:.0f} % charge and {lo_pct:.0f} % below {LOW:.0f} %." + (" Staying high for long periods ages a battery faster; keep the everyday limit near 80 % unless you need the range for the day." if hi_pct >= 10 else " That is a gentle pattern for the battery.") + (" Running very low is also hard on it; recharge before it gets that low." if lo_pct >= 5 else ""))
    if ch and ch[0]:
        dc = ch[1] or 0; lines.append(f"You charged {ch[0]} times, {dc} of them on DC fast chargers ({100.0 * dc / ch[0]:.0f} %)" + (f" and {ch[2] or 0} up to a full 100 %." if ch[2] else ".") + (" Frequent fast charging adds heat and wear; use it when you need it and charge slowly the rest of the time." if dc / ch[0] >= 0.3 else " Mostly slow charging is the easy way on the battery."))
        facts.update(charges=ch[0], dc=dc)
    if tmax and tmax[0] is not None:
        warm = tmax[1] or 0; lines.append(f"The pack peaked at {tmax[0]:.0f} °C" + (f" and was above {WARM:.0f} °C for {warm} minutes." if warm else ".") + (" Heat is the other thing that ages a battery: avoid leaving it parked at a high charge in the sun, and avoid fast charging a hot pack." if warm >= 30 or tmax[0] >= 40 else " That is comfortable."))
        facts["peak_c"] = tmax[0]
    try:
        soh = c.execute("SELECT soh FROM vday WHERE soh IS NOT NULL ORDER BY ts DESC LIMIT 1").fetchone()
        if soh: lines.append(f"Its health is {soh[0]:.0f} % of the original capacity."); facts["soh"] = soh[0]
    except sqlite3.Error: pass
    lines += ["The figures are from your vehicle's own records; the sentences about what ages a battery are general guidance, not a measurement of your battery.", "Evidence: computed from the vehicle's recorded charge, temperature and charging data."]
    return _r(res, "\n".join(lines), "advice:battery_care", frame, **facts)


def tyre_extreme(res, frame, lit):
    """The lowest (or highest) tyre pressure of the period AND which wheel it was: the per-wheel minute values, not the average of the four."""
    from ..query_ir import TimeSpec
    from .eventx import _window
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day")); hi = min(int(hi), int(b.now) + 1)
    names = (("front left", "tp_fl"), ("front right", "tp_fr"), ("rear left", "tp_rl"), ("rear right", "tp_rr")); fn = "MIN" if frame.op == "min" else "MAX"; best = []
    for nm, col in names:
        try: r = b.c.execute(f"SELECT ts, {col} FROM ext_1m WHERE ts>=? AND ts<? AND {col} IS NOT NULL ORDER BY {col} {'ASC' if fn == 'MIN' else 'DESC'} LIMIT 1", (int(lo), hi)).fetchone()
        except sqlite3.Error: return None
        if r and r[1] is not None: best.append((float(r[1]) / 10.0, nm, int(r[0])))
    if not best: return None
    pick = (min if fn == "MIN" else max)(best); word = "lowest" if fn == "MIN" else "highest"
    others = "; ".join(f"{nm} {v:.0f} kPa" for v, nm, ts in sorted(best, reverse=(fn == "MAX")) if nm != pick[1])
    return _r(res, f"The {word} tyre pressure {label} was {pick[0]:.0f} kPa, on the {pick[1]} tyre, on {A.stamp(pick[2])}.\nThe {word} of the other tyres: {others}.\nBased on the per-wheel one-minute readings.\nEvidence: measured, one-minute averages.", "extreme:tyre_wheel", frame, wheel=pick[1], kpa=pick[0])

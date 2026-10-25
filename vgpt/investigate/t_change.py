"""Change analyses: why a quantity changed, when it began, gradual versus one-off, where the extra energy went, the largest change in behaviour, joint changes, correlations, outlier trips, heat rise at high
power, and period comparisons.  All of it is computed from the trips, the one-minute tier and the events; the numbers carry their sample sizes and the method is stated."""
import math
from collections import Counter

import numpy as np

from .. import signal_registry as R
from ..query_ir import Query, Scope, TimeSpec
from .primitives import effect, robust_z
from .report import Report
from .t_events import period, specific
from .t_trips import FEATS, attribute, daily, daily_whkm, day_str, feat_label, feat_nd, feat_unit, split_change, trips_feat
from .templates import EVLABEL, dur, fmtv, hms


def _fmt(v, nd=1): return f"{v:.{nd}f}"


def _contrib_lines(att, ta, tb, top=5):
    if not att: return ["too few trips for a trip-level attribution"]
    order = np.argsort(-np.abs(att["contrib"])); out = []
    for j in order[:top]:
        f = att["names"][j]; out.append(f"{feat_label(f)}: {_fmt(att['ma'][j], feat_nd(f))} -> {_fmt(att['mb'][j], feat_nd(f))} {feat_unit(f)}; contributes about {att['contrib'][j]:+.1f} Wh/km")
    out.append(f"together they explain {att['contrib'].sum():+.1f} of the {att['total']:+.1f} Wh/km change; the other {att['unexplained']:+.1f} Wh/km is not captured by the recorded features (linear trip model, R2 = {att['r2']:.2f}, {att['n']} trips)")
    return out


# ================================================================================================ why did it change, and when did it begin
def change_explain(req, D, focus=None):
    lo, hi, label = period(req, D, 14); rep = Report("", "change_explain"); concept = "range_estimate" if any(c in req.concepts for c in ("range_estimate", "reported_range")) else "wh_per_km"
    q = Query(op="explain_change", target=concept, scope=Scope(req.times[0] if req.times else TimeSpec("relative", 14, "day"))); res = D.qc.execute(q)
    rep.step(f"typed causal attribution (explain_change) for {concept}")
    text = res.answer.text.split("\n") if res.status == "answered" else []
    rep.title = text[0] if text else f"Change in {D.label(concept)} {label}"
    if len(text) > 1: rep.add("Causal attribution (the assistant's change-explanation engine)", *[t for t in text[1:] if t.strip()][:8])
    mid = (lo + hi) // 2; ta, tb = trips_feat(D, lo, mid), trips_feat(D, mid, hi)
    if ta and tb: rep.add(f"Trip-level contributions: first half of {label} against the second", *_contrib_lines(attribute(ta, tb), ta, tb))
    d, v = daily_whkm(D, max(D.t0, hi - 60 * 86400), hi); sc = split_change(v) if len(v) >= 10 else None
    if sc: rep.add("When it began", f"the daily energy per km steps from {sc[1]:.0f} to {sc[2]:.0f} Wh/km at {day_str(d[sc[0]])} (effect size {sc[3]:+.1f}, p = {sc[4]:.3f}); this is the best single change point of the last {len(v)} driving days")
    rep.add("What cannot be determined", "anything the vehicle does not record (payload, road gradient, driver behaviour beyond hard braking / acceleration, wind) is inside the unexplained part", "an attribution is an association in this vehicle's own history, not a controlled experiment")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(ta) + len(tb), "sources": ["trips", "sig_1m", "events"], "tier": "trips + 1-minute"}; return rep


def efficiency_onset(req, D, focus=None):
    lo, hi, label = period(req, D, 60); rep = Report("", "efficiency_onset"); d, v = daily_whkm(D, lo, hi)
    if len(v) < 10: rep.title = "Not enough driving days to find when efficiency changed."; rep.ok = False; return rep
    sc = split_change(v, min_seg=5)
    if sc is None: rep.title = "No change point found."; rep.ok = False; return rep
    i, m1, m2, es, p = sc; day = day_str(d[i])
    rep.title = f"Energy per km changed from {m1:.0f} to {m2:.0f} Wh/km ({100 * (m2 - m1) / m1:+.0f} %) starting {day} (best single change point over {len(v)} driving days, p = {p:.3f})"
    rep.step(f"daily Wh/km over {label}; least-squares two-segment split"); w = 7 * 86400; t_on = int(d[i] * 86400 - D.tz)
    ta, tb = trips_feat(D, t_on - 14 * 86400, t_on), trips_feat(D, t_on, t_on + 14 * 86400)
    if ta and tb: rep.add("What differs before and after (14 days each side)", *_contrib_lines(attribute(ta, tb), ta, tb, 4))
    rows = []
    for c in ("hvac_power", "ambient_temperature", "motor_temperature", "tyre_pressure", "battery_temperature", "speed", "lane_confidence", "coolant_flow"):
        if not D.available(c): continue
        dd, vv = daily(D, c, t_on - 28 * 86400, t_on + 28 * 86400)
        if len(vv) < 12: continue
        s2 = split_change(vv, 4)
        if s2 and abs(s2[3]) >= 1.0: rows.append((dd[s2[0]], c, s2))
    rows.sort(key=lambda r: r[0])
    if rows: rep.add("Which signals changed first (day of each signal's own change point within 4 weeks of the onset)", *[f"{day_str(dc)}: {D.label(c)} {s2[1]:.1f} -> {s2[2]:.1f} {D.unit(c)} (effect {s2[3]:+.1f})" for dc, c, s2 in rows[:6]])
    day_trips = D.trips(t_on, t_on + 86400, 1.0); base = float(np.median([x["wh_km"] for x in D.trips(t_on - 14 * 86400, t_on, 1.0)] or [0]))
    first = next((t for t in day_trips if t["wh_km"] > base * 1.12), None)
    if first:
        bef = D.c.execute("SELECT AVG(hvac), AVG(amb)/10.0, AVG(motor)/10.0 FROM sig_1m WHERE ts>=? AND ts<?", (first["start"] - 1800, first["start"])).fetchone(); aft = D.c.execute("SELECT AVG(hvac), AVG(amb)/10.0, AVG(motor)/10.0 FROM sig_1m WHERE ts>=? AND ts<?", (first["start"], first["start"] + 1800)).fetchone()
        rep.add("The first trip above baseline, 30 minutes before and after it starts", f"{D.stamp(first['start'])}: {first['wh_km']:.0f} Wh/km against a recent median of {base:.0f}", f"climate power {fmtv(bef[0] or 0, 'W', 0)} before, {fmtv(aft[0] or 0, 'W', 0)} after; outside {fmtv(bef[1] or 0, 'C')} / {fmtv(aft[1] or 0, 'C')}; motor {fmtv(bef[2] or 0, 'C')} / {fmtv(aft[2] or 0, 'C')}")
    rep.add("How to read the onset", "a change point is the day the series is best split into two stable levels; a gradual drift is placed where the average changes most, not where it started")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(v), "sources": ["trips", "sig_1m"], "tier": "trips + 1-minute"}; return rep


def trend_vs_outliers(req, D, focus=None):
    from scipy.stats import kendalltau, theilslopes
    lo, hi, label = period(req, D, 30); rep = Report("", "trend_vs_outliers"); trips = trips_feat(D, lo, hi, 2.0)
    if len(trips) < 12: rep.title = f"Only {len(trips)} trips {label}: too few to separate a trend from outliers."; rep.ok = False; return rep
    t = np.array([(x["start"] - lo) / 86400.0 for x in trips]); y = np.array([x["wh_km"] for x in trips]); z = robust_z(y, y); out = np.abs(z) > 3.0
    sl, ic, _, _ = theilslopes(y, t); tau, p = kendalltau(t, y); keep = ~out; sl2, _, _, _ = theilslopes(y[keep], t[keep]) if keep.sum() >= 8 else (float("nan"),) * 4; tau2, p2 = kendalltau(t[keep], y[keep]) if keep.sum() >= 8 else (float("nan"), float("nan"))
    span = float(t.max() - t.min()); total = sl * span
    verdict = "a gradual deterioration: it persists after the unusual trips are removed" if p2 < 0.05 and sl2 > 0 else "caused by a few unusual trips: without them there is no significant trend" if out.any() and not (p2 < 0.05) else "no significant change"
    rep.title = f"Energy per km {label}: {verdict}"
    rep.add("The trend", f"{len(trips)} trips; median-based (Theil-Sen) slope {sl:+.2f} Wh/km per day = {total:+.0f} Wh/km over {span:.0f} days (Kendall p = {p:.3f})", (f"after removing the {int(out.sum())} unusual trips: slope {sl2:+.2f} Wh/km per day (p = {p2:.3f})" if out.any() and keep.sum() >= 8 else "no trip is unusual enough (more than 3 robust spreads from the typical value) to remove"))
    if out.any():
        idx = np.argsort(-np.abs(z))[:int(min(5, out.sum()))]; ref = [x for i, x in enumerate(trips) if keep[i]]
        lines = []
        for i in idx:
            x = trips[i]; diffs = []
            for f, lab, u, nd in FEATS:
                a = [r[f] for r in ref if r.get(f) is not None]
                if a and x.get(f) is not None:
                    e = effect([x[f]] * 3, a)
                    if e: diffs.append((abs(e["d"]), f"{lab} {_fmt(x[f], nd)} vs typical {_fmt(e['b'], nd)} {u}"))
            diffs.sort(reverse=True); lines.append(f"{D.stamp(x['start'])}: {x['wh_km']:.0f} Wh/km over {x['km']:.1f} km (typical {np.median(y):.0f}); differs mainly in " + "; ".join(t for _, t in diffs[:3]))
        rep.add("Largest contributors (unusual trips)", *lines)
        extra = float(np.sum((y[out] - np.median(y)) * np.array([trips[i]["km"] for i in range(len(trips)) if out[i]])) / 1000.0); rep.add("Weight of the unusual trips", f"together they used about {extra:.1f} kWh more than typical trips of the same length, {100 * extra / max(sum(x['kwh'] for x in trips), 1):.1f} % of all energy in the period")
    rep.add("Method", "a robust trend (Theil-Sen) is compared with and without trips more than 3 robust spreads from the median; the verdict needs a significant trend after the removal to call it gradual")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(trips), "sources": ["trips", "sig_1m"], "tier": "trips + 1-minute"}; return rep


def energy_attribution(req, D, focus=None):
    n = next((k for k, noun in req.counts if noun == "trip"), 10); rep = Report("", "energy_attribution"); alltr = trips_feat(D, D.t0, D.end + 1, 1.0)
    if len(alltr) < n + 20: rep.title = "Not enough earlier trips to estimate a baseline."; rep.ok = False; return rep
    last, base = alltr[-n:], alltr[-(n + 100):-n]
    ba = [x for x in base if x.get("minutes")]; X = np.array([[x["avg_speed"], x["avg_hvac_w"], x["avg_amb"], x["hwy_share"], x["idle_share"], x["harsh_per_100km"]] for x in ba], dtype=float); y = np.array([x["wh_km"] for x in ba], dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; A = np.c_[Z, np.ones(len(Z))]; coef, *_ = np.linalg.lstsq(A.T @ A + 0.5 * np.eye(7), A.T @ y, rcond=None); r2 = 1 - float(np.sum((y - A @ coef) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))
    idle_kwh = 0.0; acc_kwh = 0.0; total = 0.0; rows = []
    for x in last:
        z = (np.array([x["avg_speed"], x["avg_hvac_w"], x["avg_amb"], x["hwy_share"], x["idle_share"], x["harsh_per_100km"]]) - mu) / sd; ref = np.array(z)
        r = D.c.execute("SELECT COALESCE(SUM(CASE WHEN speed<30 THEN power END),0)/1000.0/60.0 FROM sig_1m WHERE ts>=? AND ts<?", (x["start"] - 59, x["end"])).fetchone()[0] or 0.0
        z_idle = ref.copy(); z_idle[4] = (0.0 - mu[4]) / sd[4]; z_acc = ref.copy(); z_acc[5] = (0.0 - mu[5]) / sd[5]
        d_idle = float(coef[4] * (ref[4] - z_idle[4])); d_acc = float(coef[5] * (ref[5] - z_acc[5])); km = x["km"]
        idle_kwh += max(d_idle, 0) * km / 1000.0; acc_kwh += max(d_acc, 0) * km / 1000.0; total += x["kwh"]; rows.append((x, d_idle, d_acc))
    conf = "low" if r2 < 0.4 or len(ba) < 40 else "medium" if r2 < 0.7 else "high"
    rep.title = f"Extra energy over the last {n} trips: about {idle_kwh:.2f} kWh from standstill / crawling and {acc_kwh:.2f} kWh from hard acceleration and braking ({100 * (idle_kwh + acc_kwh) / max(total, 0.1):.1f} % of the {total:.1f} kWh used) - confidence {conf}"
    rep.step(f"baseline model of Wh/km fitted on the {len(ba)} trips before these {n}; each behaviour set to zero to see what it adds")
    rep.add("Calculation method", "1. fit Wh/km of earlier trips on average speed, climate power, outside temperature, motorway share, share of time below 30 km/h, and hard-braking / acceleration events per 100 km", "2. for each of the last trips, replace the idle share, then the harsh-event rate, with zero and take the difference of the model's prediction", "3. multiply by the trip's distance and add up")
    rep.add("Assumptions", "the effects are linear and independent; 'aggressive acceleration' is represented by recorded hard-acceleration and hard-braking events (0.3 g and above); idling is time below 30 km/h in the one-minute data", "everything else (payload, gradient, wind) stays in the unexplained part")
    rep.add("Largest individual trips", *[f"{D.stamp(x['start'])}: {x['km']:.1f} km, standstill share {x['idle_share']:.0%}, {d_i * x['km'] / 1000:.2f} kWh from it; hard events {x['harsh_per_100km']:.1f} per 100 km, {d_a * x['km'] / 1000:.2f} kWh" for x, d_i, d_a in sorted(rows, key=lambda r: -(r[1] + r[2]) * r[0]['km'])[:3]])
    rep.add("Confidence", f"{conf}: model R2 = {r2:.2f} on {len(ba)} trips; this is an estimate, not a measurement")
    rep.evidence = {"period": f"the last {n} trips", "period_ts": [last[0]["start"], last[-1]["end"]], "samples": n, "sources": ["trips", "sig_1m"], "tier": "trips + 1-minute"}; return rep


# ================================================================================================ the largest change, joint changes, correlations
def largest_change(req, D, focus=None):
    lo, hi, label = period(req, D, 30); rep = Report("", "largest_change"); rows = []
    for c in ("battery_temperature", "coolant_flow", "motor_temperature", "aux_battery_voltage", "hvac_power", "lane_confidence", "camera_blockage", "tyre_pressure", "reported_range", "speed", "ambient_temperature"):
        if not D.available(c): continue
        dd, vv = daily(D, c, lo, hi)
        if len(vv) < 10: continue
        s = split_change(vv, 4)
        if s and s[4] < 0.05: rows.append((abs(s[3]), c, dd, s))
    w = daily_whkm(D, lo, hi)
    if len(w[1]) >= 10:
        s = split_change(w[1], 4)
        if s and s[4] < 0.05: rows.append((abs(s[3]), "wh_per_km", w[0], s))
    rows.sort(reverse=True)
    if not rows: rep.title = f"No signal changed in a statistically clear way {label}."; rep.add("What was checked", "daily means of eleven signals and the energy per km, each split at its best change point"); return rep
    rep.title = f"Largest change {label}: {D.label(rows[0][1])} " + f"{rows[0][3][1]:.1f} -> {rows[0][3][2]:.1f} {D.unit(rows[0][1])} from {day_str(rows[0][2][rows[0][3][0]])}"
    for es, c, dd, s in rows[:4]:
        day = day_str(dd[s[0]]); t_day = int(dd[s[0]] * 86400 - D.tz); near = [e for e in D.all_events(t_day - 2 * 86400, t_day + 2 * 86400) if e.type not in ("HARSH_BRAKE", "HARSH_ACCEL", "CHARGE_START")]
        rep.add(f"{D.label(c)} (effect {s[3]:+.1f}, p = {s[4]:.3f})", f"daily mean {s[1]:.1f} -> {s[2]:.1f} {D.unit(c)} from {day}", ("recorded events within 2 days of the change: " + ", ".join(f"{EVLABEL(t)} x{n}" for t, n in Counter(e.type for e in near).most_common(3))) if near else "no recorded event within 2 days of it - unexplained by the record")
    rep.add("What can and cannot be established", "the date and size of each change are measured; why it changed is not recorded unless an event sits next to it", "daily means hide short spikes")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": sum(len(r[2]) for r in rows), "sources": ["sig_1m", "trips"], "tier": "1-minute daily means"}; return rep


def joint_change(req, D, focus=None):
    lo, hi, label = period(req, D, 30); rep = Report("", "joint_change"); names = [c for c in req.concepts if c in ("battery_temperature", "wh_per_km", "range_estimate", "reported_range")] or ["battery_temperature", "wh_per_km", "reported_range"]; series = {}
    for c in names:
        if c in ("wh_per_km", "range_estimate"): d, v = daily_whkm(D, lo, hi)
        else: d, v = daily(D, c, lo, hi)
        if len(v) >= 8: series[c if c != "range_estimate" else "wh_per_km"] = (d, v)
    if len(series) < 2: rep.title = "Fewer than two of the named quantities have enough daily data."; rep.ok = False; return rep
    days = sorted(set.intersection(*[set(map(int, d)) for d, v in series.values()])); Z = {}
    for c, (d, v) in series.items(): m = dict(zip(map(int, d), robust_z(v, v))); Z[c] = np.array([m[x] for x in days])
    hit = [(x, [c for c in Z if abs(Z[c][i]) >= 2.0]) for i, x in enumerate(days)]; joint = [(x, cs) for x, cs in hit if len(cs) >= 2]
    names_ = list(Z); corr = {(a, b): float(np.corrcoef(Z[a], Z[b])[0, 1]) for i, a in enumerate(names_) for b in names_[i + 1:]}
    rep.title = f"{len(joint)} day{'s' if len(joint) != 1 else ''} {label} on which {', '.join(D.label(c) if c in R.CONCEPTS else c for c in names_)} were unusual together"
    rep.add("Unusual days (robust z of 2 or more)", *[f"{day_str(x)}: " + ", ".join(f"{c} {Z[c][days.index(x)]:+.1f}" for c in cs) for x, cs in joint[:8]] or ["no day with two quantities unusual at once"])
    rep.add("Do they move together?", *[f"{a} and {b}: daily correlation {r:+.2f}" for (a, b), r in corr.items()], ("the timing supports a shared driver" if corr and max(abs(r) for r in corr.values()) > 0.5 and joint else "the timing does not support a single shared driver" ) )
    rep.add("Reading this", "unusual days that coincide are an association; an independent explanation for each cannot be ruled out from timing alone")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(days), "sources": ["sig_1m", "trips"], "tier": "daily means"}; return rep


def correlate_changes(req, D, focus=None):
    from scipy.stats import spearmanr
    lo, hi, label = period(req, D, 30); rep = Report("", "correlate_changes"); trips = trips_feat(D, lo, hi, 2.0)
    if len(trips) < 12: rep.title = "Too few trips for a correlation."; rep.ok = False; return rep
    y = np.array([t["wh_km"] for t in trips]); rows = []
    for f, lab, u, nd in FEATS:
        x = np.array([t[f] if t.get(f) is not None else np.nan for t in trips], dtype=float); m = ~np.isnan(x)
        if m.sum() >= 12 and np.std(x[m]) > 0:
            r, p = spearmanr(x[m], y[m]); rows.append((abs(r), lab, r, p, f))
    rows.sort(reverse=True); direct = {"avg_hvac_w", "avg_speed", "regen_share"}
    rep.title = f"What energy use per km tracks across {len(trips)} trips {label}: " + (f"{rows[0][1]} (rank correlation {rows[0][2]:+.2f})" if rows else "nothing clear")
    rep.add("Correlation with Wh/km (strongest first)", *[f"{lab}: {r:+.2f} (p = {p:.3f})" + ("  - direct physical contributor (it is part of the energy balance)" if f in direct else "  - correlation only") for _, lab, r, p, f in rows[:6]])
    d, v = daily_whkm(D, lo, hi)
    if len(v) >= 8:
        z = robust_z(np.diff(v), np.diff(v)); idx = np.where(np.abs(z) > 2.0)[0]
        jumps = [(day_str(d[i + 1]), float(v[i + 1] - v[i])) for i in idx][:4]
        if jumps:
            lines = []
            for day, dv in jumps:
                t_day = int((np.datetime64(day).astype("datetime64[s]").astype("int64")) - D.tz); ev = [e for e in D.all_events(t_day, t_day + 86400) if e.type not in ("HARSH_BRAKE", "HARSH_ACCEL", "CHARGE_START")]
                lines.append(f"{day}: {dv:+.0f} Wh/km day-on-day; recorded events that day: " + (", ".join(f"{EVLABEL(t)} x{n}" for t, n in Counter(e.type for e in ev).most_common(3)) or "none"))
            rep.add("Sudden day-to-day changes in consumption", *lines)
    rep.add("Direct evidence versus correlation", "speed, climate power and energy recovery enter the energy balance directly; temperatures, hard events and trip shape only correlate with it", "the vehicle records no engine / motor load channel separate from battery power, so those cannot be separated further")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(trips), "sources": ["trips", "sig_1m", "events"], "tier": "trips + 1-minute"}; return rep


def thermal_rise(req, D, focus=None):
    lo, hi, label = period(req, D, 30); rep = Report("", "thermal_rise")
    rows = D.c.execute("SELECT ts, batt/10.0, power/1000.0, speed/10.0, soc/10.0, flow/10.0, mode FROM sig_1m WHERE ts>=? AND ts<? AND batt IS NOT NULL ORDER BY ts", (lo, hi)).fetchall()
    if len(rows) < 100: rep.title = "Too little battery-temperature data."; rep.ok = False; return rep
    a = np.array(rows, dtype=float); t = a[:, 0]; rate = np.full(len(a), np.nan); k = 3
    for i in range(len(a) - k):
        if t[i + k] - t[i] == 60 * k and a[i, 6] in (1, 2) and a[i + k, 6] in (1, 2): rate[i] = (a[i + k, 1] - a[i, 1]) / k
    P = np.array([np.mean(a[i:i + k + 1, 2]) for i in range(len(a))]); m = ~np.isnan(rate)
    bins = [(0, 10), (10, 30), (30, 60), (60, 400)]; stats = []
    for b0, b1 in bins:
        s = m & (P >= b0) & (P < b1)
        if s.sum() >= 20: stats.append((b0, b1, int(s.sum()), float(np.median(rate[s])), float(np.percentile(rate[s], 95))))
    hi_p = m & (P >= 30); unusual = hi_p & (rate > 0.15) & (rate > (np.percentile(rate[hi_p], 90) if hi_p.sum() > 20 else 0.15)); n_un = int(unusual.sum())
    rep.title = f"Battery temperature rise versus power {label}: " + (f"{n_un} one-minute-step episodes rose unusually fast at 30 kW or more" if n_un else "no unusually fast rise at high power")
    rep.add("Median and 95th-percentile rise rate by average battery power (driving only)", *[f"{b0}-{b1} kW: median {md:+.3f} C/min, 95th percentile {p95:+.3f} C/min ({n} windows)" for b0, b1, n, md, p95 in stats])
    if n_un:
        idx = np.where(unusual)[0]; idx = idx[np.argsort(-rate[idx])][:4]
        rep.add("Fastest rises at high power", *[f"{D.stamp(t[i])}: +{rate[i]:.2f} C/min at {P[i]:.0f} kW, speed {a[i, 3]:.0f} km/h, state of charge {a[i, 4]:.0f} %, coolant flow {a[i, 5]:.1f} L/min" for i in idx])
        normal = hi_p & ~unusual; rep.add("Cooling during those episodes", f"coolant flow {np.median(a[unusual, 5]):.1f} L/min in the fast-rise episodes against {np.median(a[normal, 5]):.1f} L/min in other high-power driving" if normal.sum() > 10 else "no comparison group")
    rep.add("Reading this", "rise rates use three-minute differences of the one-minute battery temperature; 'unusually fast' means above the 90th percentile of high-power driving and above 0.15 C/min", "current and voltage enter through power; the cooling flow is the physical suspect when the rise is fast at modest power")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": int(m.sum()), "sources": ["sig_1m"], "tier": "1-minute"}; return rep


# ================================================================================================ outlier trips and comparisons
def trip_outliers(req, D, focus=None):
    lo, hi, label = period(req, D, 30); rep = Report("", "trip_outliers"); text = req.text.lower(); k = next((n for n, noun in req.counts if noun == "trip" and n <= 6), 3); trips = trips_feat(D, lo, hi, 1.5)
    if len(trips) < 10: rep.title = f"Only {len(trips)} trips {label}."; rep.ok = False; return rep
    if "no recorded faults" in text or "contained major" in text or "ten trips" in text:
        n = next((n for n, noun in req.counts if noun == "trip"), 10)
        for t in trips: t["warn"] = len(D.events(["ANY_WARNING"], t["start"], t["end"]))
        clean = [t for t in trips if not t["warn"]][-n:]; faulty = [t for t in trips if t["warn"]][-n:]
        if len(faulty) < 3: rep.title = f"Only {len(faulty)} trips {label} contained a warning: too few to compare with the {len(clean)} clean ones."; rep.ok = False; return rep
        rows = []
        for f, lab, u, nd in FEATS:
            a = [t[f] for t in faulty if t.get(f) is not None]; b = [t[f] for t in clean if t.get(f) is not None]; e = effect(a, b)
            if e: rows.append((abs(e["d"]), lab, e, u, nd))
        rows.sort(reverse=True); rep.title = f"{len(faulty)} most recent trips with a warning compared with {len(clean)} recent trips without one ({label})"
        rep.add("Features that differ most (median in trips with a warning vs without)", *[f"{lab}: {_fmt(e['a'], nd)} vs {_fmt(e['b'], nd)} {u}; {max(e['above75'], e['below25']):.0%} of the faulty trips were " + ("above" if e["d"] > 0 else "below") + " the clean trips' " + ("75th" if e["d"] > 0 else "25th") + " percentile" for _, lab, e, u, nd in rows[:5]])
        rep.add("Which differences may just be different driving conditions", "distance, duration and motorway share differ because warnings cluster on long motorway trips; compare the speed and temperature rows only after allowing for that", "ten trips per group is small: treat the effect sizes as indications")
        rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(clean) + len(faulty), "sources": ["trips", "events", "sig_1m"], "tier": "trips + 1-minute"}; return rep
    if "two trips" in text:
        pair = None; key = "adas_share" if "adas" in text else "wh_km"
        for i, a in enumerate(trips):
            for b in trips[i + 1:]:
                if a.get(key) is None or b.get(key) is None: continue
                if abs(a["km"] - b["km"]) <= 0.1 * a["km"] and abs(a["duration_min"] - b["duration_min"]) <= 0.15 * a["duration_min"]:
                    diff = abs(a[key] - b[key])
                    if pair is None or diff > pair[0]: pair = (diff, a, b)
        if pair is None: rep.title = "No two trips of similar distance and duration were found."; rep.ok = False; return rep
        _, a, b = pair; a, b = sorted((a, b), key=lambda t: -(t[key] or 0)); rep.title = f"Two trips of about {a['km']:.0f} km and {a['duration_min']:.0f} min: {'ADAS use' if key == 'adas_share' else 'energy'} differs most ({a[key]:.0f} vs {b[key]:.0f})"
        rows = []
        for f, lab, u, nd in FEATS + [("harsh_brake", "hard braking events", "", 0), ("rain_share", "share of the trip in rain", "", 2), ("batt_max", "peak battery temperature", "C", 1)]:
            va, vb = a.get(f), b.get(f)
            if va is not None and vb is not None: rows.append((abs(va - vb) / (abs(va) + abs(vb) + 1e-9), f"{lab}: {_fmt(va, nd)} vs {_fmt(vb, nd)} {u}"))
        rows.sort(reverse=True); rep.add(f"Side by side ({D.stamp(a['start'])} vs {D.stamp(b['start'])})", f"{a['wh_km']:.0f} vs {b['wh_km']:.0f} Wh/km", *[t for _, t in rows[:7]])
        for t_, name in ((a, "first"), (b, "second")):
            ev = D.all_events(t_["start"], t_["end"]); rep.add(f"Events during the {name} trip", *[f"{hms(D, e.ts)} {EVLABEL(e.type)}: {e.detail}" for e in ev[:5]] or ["none"])
        rep.add("Reading this", "similar length and duration, different outcome: the rows above are the recorded differences; the vehicle does not record road type, traffic or payload", "speed pattern is summarised from one-minute averages")
        rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": 2, "sources": ["trips", "sig_1m", "events"], "tier": "trips + 1-minute"}; return rep
    km = np.array([t["km"] for t in trips]); out = []
    for i, t in enumerate(trips):
        ref = [r["wh_km"] for j, r in enumerate(trips) if j != i and abs(r["km"] - t["km"]) <= 0.35 * t["km"]]
        if len(ref) >= 6: out.append((float(robust_z([t["wh_km"]], ref)[0]), i, ref))
    out.sort(reverse=True); sel = out[:k]
    if not sel: rep.title = "No trip has enough similar-distance trips to compare against."; rep.ok = False; return rep
    rep.title = f"The {len(sel)} trips {label} with the highest energy per km for their distance"
    for z, i, ref in sel:
        t = trips[i]; peers = [r for j, r in enumerate(trips) if j != i and abs(r["km"] - t["km"]) <= 0.35 * t["km"]]; rows = []
        for f, lab, u, nd in FEATS:
            a = [r[f] for r in peers if r.get(f) is not None]
            if a and t.get(f) is not None:
                e = effect([t[f]] * 3, a)
                if e: rows.append((abs(e["d"]), f"{lab} {_fmt(t[f], nd)} vs typical {_fmt(e['b'], nd)} {u}"))
        rows.sort(reverse=True); rep.add(f"{D.stamp(t['start'])}: {t['wh_km']:.0f} Wh/km over {t['km']:.1f} km (similar trips median {np.median(ref):.0f}; {z:+.1f} robust spreads above)", *[x for _, x in rows[:4]], *[f"{hms(D, e.ts)} {EVLABEL(e.type)}" for e in D.all_events(t["start"], t["end"])[:3]])
    rep.add("Reading this", "'similar distance' means within 35 % of the trip's length; differences are the recorded features, not proof of what caused the extra energy; road conditions are available only as the share of the trip in rain")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(trips), "sources": ["trips", "sig_1m", "trip_ext"], "tier": "trips + 1-minute"}; return rep


def compare_periods(req, D, focus=None):
    rep = Report("", "compare_periods"); now = D.end + 1; n = 7 * 86400; a0, a1, b0, b1 = now - 2 * n, now - n, now - n, now; ta = trips_feat(D, a0, a1, 1.0); tb = trips_feat(D, b0, b1, 1.0)
    if "city" in req.text.lower():
        ta = [t for t in ta if t["hwy_share"] < 0.3]; tb = [t for t in tb if t["hwy_share"] < 0.3]
    if len(ta) < 4 or len(tb) < 4: rep.title = f"Too few trips in the two weeks ({len(ta)} and {len(tb)}) to compare."; rep.ok = False; return rep
    wa = 1000 * sum(t["kwh"] for t in ta) / max(sum(t["km"] for t in ta), 1); wb = 1000 * sum(t["kwh"] for t in tb) / max(sum(t["km"] for t in tb), 1)
    rep.title = f"{'City trips' if 'city' in req.text.lower() else 'Trips'}: {wb:.0f} Wh/km in the last 7 days against {wa:.0f} in the 7 days before ({wb - wa:+.0f}, {100 * (wb - wa) / wa:+.0f} %)"
    rep.add("Trips compared", f"{len(tb)} trips ({sum(t['km'] for t in tb):.0f} km) against {len(ta)} trips ({sum(t['km'] for t in ta):.0f} km)")
    rep.add("Which changes contributed most", *_contrib_lines(attribute(ta, tb), ta, tb, 5))
    ev_a = len(D.events(["HARSH_BRAKE", "HARSH_ACCEL"], a0, a1)); ev_b = len(D.events(["HARSH_BRAKE", "HARSH_ACCEL"], b0, b1)); rep.add("Driving behaviour", f"hard braking / acceleration events: {ev_a} in the earlier week, {ev_b} in the last week")
    wrn = D.events(["ANY_WARNING"], b0, b1); rep.add("Vehicle conditions", *[f"{hms(D, e.ts)} {D.stamp(e.ts)[:10]} {EVLABEL(e.type)}: {e.detail}" for e in wrn[:3]] or ["no warning in the last week"])
    rep.evidence = {"period": "the last 7 days against the 7 days before", "period_ts": [a0, b1], "samples": len(ta) + len(tb), "sources": ["trips", "sig_1m", "events"], "tier": "trips + 1-minute"}; return rep

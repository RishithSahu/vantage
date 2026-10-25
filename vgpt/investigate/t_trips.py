"""Journey- and trend-centred analyses: operating conditions of a fault, heat after stops, cold starts, the range estimate, efficiency change and its cause, outlier trips, energy attribution,
the largest change in behaviour, joint changes, correlations, and period comparisons."""
import math
from collections import Counter, defaultdict

import numpy as np

from .. import signal_registry as R
from .primitives import effect
from .report import Report
from .t_events import SIGNAL_OF_CODE, dtc_code, period, specific
from .templates import CONTEXT_ONLY, EVLABEL, dur, fmtv, hms

FEATS = [("avg_speed", "average speed", "km/h", 0), ("speed_std", "speed variability", "km/h", 0), ("idle_share", "share of time at standstill / crawling", "", 2), ("avg_hvac_w", "climate-control power", "W", 0), ("avg_amb", "outside temperature", "C", 1),
         ("motor_mean", "mean motor temperature", "C", 0), ("hwy_share", "motorway share", "", 2), ("harsh_per_100km", "hard braking / acceleration per 100 km", "", 1), ("duration_min", "duration", "min", 0), ("km", "distance", "km", 1), ("regen_share", "share of energy recovered", "", 2)]


def trips_feat(D, lo, hi, min_km=1.0):
    return [D.trip_features(t) for t in D.trips(lo, hi, min_km)]


def daily(D, concept, lo, hi):
    ent = (R.STORE.get(concept) or {}).get("1m")
    if not ent or ent[0] not in D.have: return np.array([]), np.array([])
    t, col, scale = ent
    rows = D.c.execute(f"SELECT CAST((ts+?)/86400 AS INTEGER) d, AVG({col})/? FROM {t} WHERE ts>=? AND ts<? AND {col} IS NOT NULL GROUP BY d ORDER BY d", (D.tz, float(scale), int(lo), int(hi))).fetchall()
    return np.array([r[0] for r in rows], dtype=float), np.array([r[1] for r in rows], dtype=float)


def daily_whkm(D, lo, hi):
    rows = D.c.execute("SELECT CAST((start+?)/86400 AS INTEGER) d, SUM(kwh), SUM(km), COUNT(*) FROM trips WHERE start>=? AND start<? AND km>=1 GROUP BY d ORDER BY d", (D.tz, int(lo), int(hi))).fetchall()
    rows = [r for r in rows if r[2] and r[2] > 5]
    return np.array([r[0] for r in rows], dtype=float), np.array([1000 * r[1] / r[2] for r in rows], dtype=float)


def day_str(d): import datetime as dt; return (dt.date(1970, 1, 1) + dt.timedelta(days=int(d))).isoformat()


def split_change(v, min_seg=4):
    """Best two-segment split of a series (least squares): -> (index, mean_before, mean_after, effect size, p-value) or None."""
    from scipy.stats import ttest_ind
    n = len(v)
    if n < 2 * min_seg: return None
    best = None; cs = np.cumsum(v); cs2 = np.cumsum(v * v)
    for i in range(min_seg, n - min_seg + 1):
        a = cs[i - 1]; b = cs[-1] - a; sse = (cs2[i - 1] - a * a / i) + ((cs2[-1] - cs2[i - 1]) - b * b / (n - i))
        if best is None or sse < best[0]: best = (sse, i)
    i = best[1]; a, b = v[:i], v[i:]; sp = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(1, n - 2)) or 1e-9
    return i, float(a.mean()), float(b.mean()), float((b.mean() - a.mean()) / sp), float(ttest_ind(a, b, equal_var=False).pvalue)


def attribute(ta, tb, feats=FEATS):
    """Linear trip model of Wh/km on standardised features over both groups; contribution of each feature to the change of the mean between group a and group b."""
    rows = [t for t in ta + tb if t.get("minutes") and t.get("wh_km") is not None]; names = [f for f, *_ in feats if all(f in r and r[f] is not None for r in rows)]
    if len(rows) < 12 or len(names) < 2: return None
    X = np.array([[r[f] for f in names] for r in rows], dtype=float); y = np.array([r["wh_km"] for r in rows], dtype=float); mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    A = np.c_[Z, np.ones(len(Z))]; coef, *_ = np.linalg.lstsq(A.T @ A + 0.5 * np.eye(A.shape[1]), A.T @ y, rcond=None); pred = A @ coef; r2 = 1 - float(np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))
    ma = np.array([np.mean([t[f] for t in ta]) for f in names]); mb = np.array([np.mean([t[f] for t in tb]) for f in names]); contrib = coef[:-1] * ((mb - mu) / sd - (ma - mu) / sd)
    total = float(np.mean([t["wh_km"] for t in tb]) - np.mean([t["wh_km"] for t in ta]))
    return {"names": names, "contrib": contrib, "total": total, "r2": r2, "unexplained": total - float(contrib.sum()), "n": len(rows), "ma": ma, "mb": mb}


def feat_label(f): return next((l for k, l, u, nd in FEATS if k == f), f)
def feat_unit(f): return next((u for k, l, u, nd in FEATS if k == f), "")
def feat_nd(f): return next((nd for k, l, u, nd in FEATS if k == f), 1)


# ================================================================================================ conditions under which a fault occurs
def conditions(req, D, focus=None):
    lo, hi, label = period(req, D, 120); rep = Report("", "conditions"); types = specific(req) or ["DTC_SET"]; evs = D.events(types[:1] if types[0] != "DTC_SET" or len(types) == 1 else types[:1], lo, hi)
    if types[0] in ("WARN_OVERTEMP", "POWER_DERATE", "WARN_BATT_OVERTEMP") and "DTC_SET" in types[1:]: evs = D.events([types[0]], lo, hi)
    code = None
    if evs and evs[0].type == "DTC_SET":
        top = Counter(dtc_code(e) for e in evs).most_common(1)[0][0]; evs = [e for e in evs if dtc_code(e) == top]; code = top
    if len(evs) < 3: rep.title = f"Only {len(evs)} matching event{'s' if len(evs) != 1 else ''} {label}: too few to say under which conditions it occurs."; rep.ok = False; return rep
    name = f"{code} ({D.dtc().get(code, {}).get('text', '')})" if code else EVLABEL(evs[0].type); trips = trips_feat(D, lo, hi, 0.3); starts = np.array([t["start"] for t in trips]); ends = np.array([t["end"] for t in trips])
    def at(ts):
        r = D.c.execute("SELECT speed/10.0, motor/10.0, amb/10.0, power/1000.0, batt/10.0, flow/10.0, mode FROM sig_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (int(ts) // 60 * 60,)).fetchone(); i = int(np.searchsorted(starts, ts, side="right")) - 1
        return {"speed": r[0], "motor temperature": r[1], "outside temperature": r[2], "battery power": r[3], "battery temperature": r[4], "trip elapsed (min)": (ts - starts[i]) / 60.0 if 0 <= i < len(starts) and ts <= ends[i] else float("nan"), "trip length (min)": (ends[i] - starts[i]) / 60.0 if 0 <= i < len(starts) and ts <= ends[i] else float("nan")} if r else None
    ev_f = [f for f in (at(e.ts) for e in evs) if f]; rng = np.random.default_rng(3)
    rows = D.c.execute("SELECT ts FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2) AND speed>50", (lo, hi)).fetchall(); pick = [r[0] for r in rows]; pick = [pick[i] for i in rng.choice(len(pick), size=min(600, len(pick)), replace=False)] if pick else []
    ctl = [f for f in (at(t) for t in pick) if f]; rep.step(f"{len(evs)} events of {name}; {len(ctl)} control driving minutes (speed above 50 km/h) from the same period")
    rows_e = []
    for k in ev_f[0]:
        a = np.array([f[k] for f in ev_f], dtype=float); b = np.array([f[k] for f in ctl], dtype=float); a, b = a[~np.isnan(a)], b[~np.isnan(b)]; e = effect(a, b) if len(a) >= 3 and len(b) >= 10 else None
        if e: rows_e.append((abs(e["d"]), k, e))
    rows_e.sort(reverse=True)
    rep.title = f"{name}: {len(evs)} occurrences {label}; conditions compared with {len(ctl)} ordinary motorway-speed driving minutes"
    rep.add("Conditions at the moment it occurred (typical value at the events vs in the control minutes)", *[f"{k}: {fmtv(e['a'], '', 1)} vs {fmtv(e['b'], '', 1)}; {max(e['above75'], e['below25']):.0%} of the occurrences were " + ("above" if e["d"] > 0 else "below") + f" the control's " + ("75th" if e["d"] > 0 else "25th") + " percentile" + ("  <- consistent" if abs(e["d"]) >= 0.8 and max(e["above75"], e["below25"]) >= 0.7 else "") for _, k, e in rows_e[:6]])
    key = [(k, e) for _, k, e in rows_e if abs(e["d"]) >= 0.8 and max(e["above75"], e["below25"]) >= 0.7 and k in ("speed", "motor temperature", "outside temperature", "battery power", "battery temperature")][:3]
    if key:
        thr = {k: (float(np.percentile([f[k] for f in ev_f if not math.isnan(f[k])], 15 if e["d"] > 0 else 85)), e["d"] > 0) for k, e in key}
        data = D.c.execute("SELECT ts, speed/10.0, motor/10.0, amb/10.0, power/1000.0, batt/10.0 FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2) ORDER BY ts", (lo, hi)).fetchall(); cols = {"speed": 1, "motor temperature": 2, "outside temperature": 3, "battery power": 4, "battery temperature": 5}
        meet = [r[0] for r in data if all((r[cols[k]] >= v) if up else (r[cols[k]] <= v) for k, (v, up) in thr.items() if k in cols and r[cols[k]] is not None)]
        eps = []; cur = []
        for t in meet:
            if cur and t - cur[-1] > 300: eps.append(cur); cur = []
            cur.append(t)
        if cur: eps.append(cur)
        eps = [e for e in eps if len(e) >= 2]; hit = sum(1 for e in eps if any(abs(ev.ts - e[0]) <= (e[-1] - e[0]) + 180 and e[0] - 180 <= ev.ts for ev in evs))
        cond = " and ".join(f"{k} {'above' if up else 'below'} {v:.0f}" for k, (v, up) in thr.items())
        rep.add("Check of the rule that these conditions suggest", f"conditions: {cond}", f"such episodes (two or more minutes): {len(eps)}; the fault occurred in {hit} of them ({hit / max(1, len(eps)):.0%}); in {len(eps) - hit} similar episodes it did not" + (" - so the conditions favour it but do not decide it" if 0 < hit < len(eps) else ""))
    tl = [f["trip length (min)"] for f in ev_f if not math.isnan(f["trip length (min)"])]
    if tl and trips:
        alld = [t["duration_min"] for t in trips]; rep.add("Journey length", f"median length of the trips it occurred in: {np.median(tl):.0f} min; all trips: {np.median(alld):.0f} min; share of occurrences in trips longer than 60 min: {np.mean([x > 60 for x in tl]):.0%} (all trips: {np.mean([x > 60 for x in alld]):.0%})")
    rep.add("Limits", "an association between operating conditions and a fault is not a diagnosis; the controls are ordinary driving minutes at similar speed, not a matched experiment")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(evs) + len(ctl), "sources": ["events", "sig_1m", "trips"], "tier": "events + 1-minute"}; return rep


# ================================================================================================ heat after stops, cold starts, the range estimate
def thermal_stop(req, D, focus=None):
    lo, hi, label = period(req, D, 30); rep = Report("", "thermal_stop"); trips = D.trips(lo, hi, 1.0); rows = []
    for t in trips:
        e = t["end"]; s = D.series("battery_temperature", e - 300, e + 1500)
        if s is None or len(s.t) < 6: continue
        before = s.t <= e; after = s.t > e
        if before.sum() < 2 or after.sum() < 3: continue
        t0 = float(s.v[before][-1]); post = s.v[after]; pt = s.t[after]; peak = float(np.max(post)); rise = peak - t0
        rows.append((rise, e, float(pt[np.argmax(post)] - e), t0, peak, s.res))
    if not rows: rep.title = f"No stop with enough battery-temperature data was found {label}."; rep.ok = False; return rep
    risers = [r for r in rows if r[0] >= 0.6]; rep.step(f"{len(rows)} stops checked {label}; a rise of 0.6 C or more after the stop counts")
    rep.title = f"The battery kept heating after the vehicle stopped on {len(risers)} of {len(rows)} stops {label}"
    rep.add("Largest rises after a stop", *[f"{D.stamp(e)}: {t0:.1f} -> {peak:.1f} C (+{rise:.1f} C), the peak came {dur(dt)} after the stop ({R_res(res)})" for rise, e, dt, t0, peak, res in sorted(risers, reverse=True)[:6]])
    if risers:
        hot = [r for r in rows if r[3] >= 31]; cold = [r for r in rows if r[3] < 31]
        rep.add("When it happens", f"typical rise {np.median([r[0] for r in risers]):.1f} C, lasting {dur(float(np.median([r[2] for r in risers])))} until the peak", f"stops that began hot (31 C or more): {sum(1 for r in hot if r[0] >= 0.6)} of {len(hot)} rose; stops that began cooler: {sum(1 for r in cold if r[0] >= 0.6)} of {len(cold)}")
        pump = []
        for rise, e, dt, t0, peak, res in sorted(risers, reverse=True)[:5]:
            f = D.series("coolant_flow", e - 120, e + 600)
            if f is not None and len(f.v): pump.append(f"{D.stamp(e)[5:]}: coolant flow {fmtv(float(np.median(f.v[f.t <= e][-10:])) if (f.t <= e).any() else float('nan'), 'L/min')} before the stop, {fmtv(float(np.median(f.v[f.t > e])) if (f.t > e).any() else float('nan'), 'L/min')} after")
        rep.add("Did the cooling system keep working?", *pump)
    rep.add("Reading this", "heat soak is expected after hard driving or fast charging (the cooling pump stops while the pack is still warm); it is a concern only when it is large or repeated", "older stops are seen at one-minute resolution, so the peak time is approximate")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(rows), "sources": ["sig_1s", "sig_1m", "trips"], "tier": "1 Hz / 1-minute"}; return rep


def R_res(res): return "1 Hz" if res <= 1.5 else "1-minute averages"


def cold_start(req, D, focus=None):
    lo, hi, label = period(req, D, 14); rep = Report("", "cold_start"); trips = trips_feat(D, lo, hi, 1.0); firsts = []; prev_end = None
    for t in trips:
        if prev_end is None or t["start"] - prev_end >= 6 * 3600: firsts.append(t)
        prev_end = t["end"]
    if len(firsts) < 3: rep.title = f"Only {len(firsts)} cold starts {label}: too few to compare."; rep.ok = False; return rep
    warn = {t["start"]: len(D.events(["ANY_WARNING"], t["start"], t["end"])) for t in firsts}
    rows = []
    for t in firsts:
        b = D.series("battery_temperature", t["start"] - 120, t["start"] + 60); a = t["avg_amb"]; early = D.c.execute("SELECT SUM(power)/1000.0/60.0, SUM(speed)/10.0/60.0 FROM sig_1m WHERE ts>=? AND ts<?", (t["start"], t["start"] + 600)).fetchone()
        e_wh = 1000 * early[0] / early[1] if early and early[1] and early[1] > 0.3 else None
        rows.append((t, float(b.v[0]) if b is not None and len(b.v) else None, a, e_wh))
    rep.step(f"{len(firsts)} first trips after at least 6 h parked {label}; early = first 10 minutes")
    rep.title = f"Cold starts {label}: {len(firsts)} first trips of the day, {sum(1 for t in firsts if warn[t['start']])} with a warning during the trip"
    ok = [r for r in rows if not warn[r[0]["start"]]]; bad = [r for r in rows if warn[r[0]["start"]]]
    def summ(rs, name):
        init = [r[1] for r in rs if r[1] is not None]; amb = [r[2] for r in rs]; early = [r[3] for r in rs if r[3] is not None]; allw = [r[0]["wh_km"] for r in rs if r[0].get("wh_km")]
        return f"{name} ({len(rs)}): start battery temperature {np.median(init):.1f} C vs outside {np.median(amb):.1f} C; first-10-minutes consumption {np.median(early):.0f} Wh/km vs whole trip {np.median(allw):.0f} Wh/km" if init and early and allw else f"{name} ({len(rs)}): too little data"
    rep.add("Initial temperature and early-trip energy", summ(ok, "trips without a warning"), summ(bad, "trips with a warning") if bad else "no first trip of the day contained a warning")
    cold = [r for r in rows if r[2] is not None and r[2] < 10]; warm = [r for r in rows if r[2] is not None and r[2] >= 10]
    if cold and warm: rep.add("Cold versus mild mornings", f"outside below 10 C ({len(cold)} starts): whole-trip consumption {np.median([r[0]['wh_km'] for r in cold]):.0f} Wh/km; mild ({len(warm)}): {np.median([r[0]['wh_km'] for r in warm]):.0f} Wh/km")
    rep.add("Not covered", "warm-up duration is not a recorded quantity; the time the pack takes to reach its working temperature is read from the battery temperature series where it exists")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(firsts), "sources": ["trips", "sig_1m"], "tier": "1-minute"}; return rep


def range_drop(req, D, focus=None):
    lo, hi, label = period(req, D, 120); rep = Report("", "range_drop"); concept = "reported_range" if D.available("reported_range") else None
    if concept is None: rep.title = "This vehicle does not record the range estimate it displays, so a revision of that estimate cannot be separated from real energy use."; rep.ok = False; return rep
    best = None; short = "short" in req.text.lower() or "only" in req.text.lower() and "distance" in req.text.lower()
    for t in D.trips(lo, hi, 1.0):
        if short and not (3.0 <= t["km"] <= 25.0): continue
        s = D.series(concept, t["start"] - 60, t["end"] + 60)
        if s is None or len(s.v) < 5: continue
        start = float(s.v[0]); low = float(np.min(s.v)); drop = start - low; i = int(np.argmin(s.v))
        km_at_low = max(t["km"] * float(s.t[i] - t["start"]) / max(t["end"] - t["start"], 1), 0.1)
        if best is None or drop / max(start, 1) > best[0]: best = (drop / max(start, 1), t, s, start, low, i, km_at_low)
    if best is None: rep.title = f"No trip with range-estimate data {label}."; rep.ok = False; return rep
    frac, t, s, start, low, i, km = best; e = D.c.execute("SELECT SUM(power)/1000.0/60.0 FROM sig_1m WHERE ts>=? AND ts<?", (t["start"], int(s.t[i]))).fetchone()[0] or 0.0
    cap = float(D.c.execute("SELECT v FROM meta WHERE k='veh.cap_kwh'").fetchone()[0]); soc = D.series("battery_soc", t["start"] - 60, t["end"] + 60); dsoc = float(soc.v[0] - soc.v[min(i, len(soc.v) - 1)]) if soc is not None and len(soc.v) > 2 else float("nan")
    expected = km
    rep.title = f"Largest drop of the dashboard range estimate {label}: {start:.0f} -> {low:.0f} km ({100 * frac:.0f} %) during the trip of {D.stamp(t['start'])}"
    rep.add("What the vehicle displayed", f"estimate at the start {start:.0f} km, lowest {low:.0f} km at {hms(D, s.t[i])} ({dur(s.t[i] - t['start'])} into the trip), at the end {float(s.v[-1]):.0f} km")
    rep.add("What actually happened to the battery", f"distance driven to that point: {km:.1f} km; energy used: {e:.2f} kWh; state of charge fell {dsoc:.1f} points" if not math.isnan(dsoc) else f"distance driven to that point: {km:.1f} km; energy used: {e:.2f} kWh", f"if the estimate were only following consumption it would have dropped by about the distance driven, {expected:.1f} km; it dropped {start - low:.0f} km")
    ratio = (start - low) / max(expected, 0.5)
    verdict = "a revised prediction: the display fell far more than the distance or the energy used can explain" if ratio > 3 else "mostly real energy use" if ratio < 1.6 else "partly a revision of the prediction"
    rep.add("Conclusion (inferred)", verdict, "energy figures come from the battery power signal, the range figure from the vehicle's own estimate; an inconsistent sensor reading would show up as a state-of-charge jump, and there is none" if not math.isnan(dsoc) and dsoc < 6 else "the state of charge changed a lot too, so a sensor offset cannot be excluded")
    others = D.all_events(t["start"], t["end"]); rep.add("Other events during that trip", *[f"{hms(D, e_.ts)} {EVLABEL(e_.type)}: {e_.detail}" for e_ in others[:4]]) if others else None
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(s.v), "sources": ["vhal range", "sig_1m", "trips"], "tier": "1-minute"}; rep.facts.update(drop_pct=100 * frac, ratio=ratio); return rep

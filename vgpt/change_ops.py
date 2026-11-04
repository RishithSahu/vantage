"""Two composable analysis operators over the stored history. Each states exactly which quantity it answers.

  detect_change(target, window)   Where in the window did the DAILY VALUE of the target change? A CANDIDATE change point (the best two-level split of the daily values) with its size and an adjusted p-value, and the single largest
                                  day-to-day move in the unfavourable direction. For range the series is transformed first (range = usable capacity / consumption) and searched directly, so the reported move is the largest FALL IN RANGE.
  explain_change(target, window)  Why does the target differ from the period just before (or the period t2)? Three levels, never merged:
                                    1 OBSERVED   the change between the two periods, on this quantity: energy per kilometre is TOTAL energy / TOTAL distance of each period; every other signal is the AVERAGE OF ITS DAILY VALUES;
                                    2 CONTRIBUTIONS   only for energy per kilometre / range, only for an unrestricted scope: a trip-level regression on the recorded conditions, on a per-trip-average basis (stated), with the unexplained part
                                      (residual) shown separately and never spread over the factors. A residual is whatever the model and the recorded conditions do not capture - missing causes, noise, interactions, model error - not proof of a missing cause.
                                      For other signals the declared physical dependencies are checked one by one and reported as ASSOCIATIONS (this driver changed in a direction that would raise / lower the target);
                                    3 OPEN   what the recorded data cannot separate and which measurement would.
Scope: the primary question's conditions and selection apply to every series of the recent period; the comparison period uses `scope_b` (its own conditions and selection) when it was given, else the primary scope.
Statistics: Welch t-test for the two fixed windows (their boundaries come from the question, not from the data); for the best split the p-value is a max-statistic permutation test that accounts for having searched all split points (days treated as
exchangeable: serial correlation is NOT modelled, which is why the wording says 'candidate' and gives the effect size next to it). Nothing is learned."""
import math

import numpy as np

from . import analytics as A, qa_core as Q, signal_registry as R
from .query_ir import Scope, TimeSpec
from .query_planner import resolve_time, DAY

BAD = {"wh_per_km": +1, "range_estimate": -1, "battery_temperature": +1, "motor_temperature": +1, "coolant_flow": -1, "hvac_power": +1, "tyre_pressure": -1, "aux_battery_voltage": -1, "battery_soh": -1, "cell_voltage_min": -1, "energy_used": +1,
       "battery_power": None, "speed": None, "ambient_temperature": None, "regen_energy": -1, "charging_power": -1, "battery_soc": None}


def _pct_change(before, after, unit):
    """Percentage change only for a positive ratio-scale baseline.

    Celsius/Fahrenheit and signed power have arbitrary or meaningful zero
    crossings; percentages such as "-5 C rose by -44%" are dimensionally
    misleading.  Absolute change remains available for those quantities.
    """
    if unit in ("C", "F", "°C", "°F") or before <= 1e-9:
        return None
    return 100 * (after - before) / before

DEPENDS = {
    "battery_temperature": [("ambient_temperature", +1, "a warmer environment leaves less cooling margin"), ("coolant_flow", -1, "less coolant flow removes less heat"), ("battery_power", +1, "more power through the pack makes more heat"), ("charging_power", +1, "fast charging heats the pack")],
    "motor_temperature": [("speed", +1, "higher speed means higher motor load"), ("ambient_temperature", +1, "a warmer environment leaves less cooling margin"), ("motor_torque", +1, "more torque makes more heat")],
    "coolant_flow": [("battery_temperature", +1, "the pump runs harder when the pack is hot"), ("ambient_temperature", +1, "the pump runs harder in warm weather")],
    "hvac_power": [("ambient_temperature", -1, "colder weather needs more heating")],
    "tyre_pressure": [("ambient_temperature", +1, "tyre pressure rises and falls with temperature (about 1 % per 3 C)")],
    "aux_battery_voltage": [("ambient_temperature", +1, "battery voltage depends on temperature")],
    "battery_voltage": [("battery_current", -1, "voltage sags under load"), ("battery_soc", +1, "voltage follows state of charge")],
}
MISSING = {
    "battery_temperature": ["coolant temperature at the pack inlet and outlet", "pump speed", "the cell-level temperature spread"],
    "coolant_flow": ["pump speed or current", "coolant level", "coolant temperature"],
    "tyre_pressure": ["the temperature of the tyres themselves", "individual-wheel pressure history"],
    "motor_temperature": ["motor coolant temperature", "gradient and payload"],
    "hvac_power": ["cabin set-point and occupants"],
    "default": ["the load and route conditions (payload, wind, road gradient)", "a controlled capacity or calibration test"],
}
GAUGE_NOTE = "the stored pressure value is used as given in the temperature correction"


# ------------------------------------------------------------------------------------------------ series
def _wstats(x, w):
    w = np.asarray(w, float); x = np.asarray(x, float); m = float(np.sum(w * x) / np.sum(w)); sd = math.sqrt(float(np.sum(w * (x - m) ** 2) / np.sum(w))); return m, sd


def _weighted_attribution(c, T_all, bad, good):
    """Distance-WEIGHTED trip-level decomposition of the change in energy per kilometre. Weighting every trip by its distance makes the weighted mean of trip Wh/km exactly total energy / total distance,
    so the quantity explained is the SAME quantity as the headline (not a per-trip average). Weighted ridge regression on the standardised conditions of Q.FEATURES; coefficient x (weighted mean of the condition in the
    later period - in the earlier period). -> dict like qa_core.eff_period, or None."""
    T = [t for t in T_all if t["km"] > 0]
    if len(T) < 12 or len(bad) < 2 or len(good) < 2: return None
    w = [t["km"] for t in T]; feats = [f for f in Q.FEATURES if _wstats([f[2](t) for t in T], w)[1] > 1e-6]
    if not feats: return None
    X = np.array([[f[2](t) for f in feats] for t in T], float); y = np.array([t["wh_km"] for t in T], float); W = np.array(w, float)
    mu = [_wstats(X[:, i], W)[0] for i in range(len(feats))]; sg = [_wstats(X[:, i], W)[1] or 1.0 for i in range(len(feats))]; Z = (X - np.array(mu)) / np.array(sg)
    A_ = np.hstack([np.ones((len(T), 1)), Z]); lam = 0.5 * float(np.sum(W)) / len(T); P = np.eye(A_.shape[1]) * lam; P[0, 0] = 0.0
    beta = np.linalg.solve(A_.T @ (W[:, None] * A_) + P, A_.T @ (W * y)); fitted = A_ @ beta; ybar = float(np.sum(W * y) / np.sum(W)); r2 = 1 - float(np.sum(W * (y - fitted) ** 2)) / max(float(np.sum(W * (y - ybar) ** 2)), 1e-9)
    bb = [float(beta[i + 1] / sg[i]) for i in range(len(feats))]; bb = [0.0 if (f[0] in Q.PHYS and b < 0) else b for f, b in zip(feats, bb)]          # a physically impossible sign is noise
    out = []
    for i, f in enumerate(feats):
        xb = _wstats([f[2](t) for t in bad], [t["km"] for t in bad])[0]; xg = _wstats([f[2](t) for t in good], [t["km"] for t in good])[0]
        both = bad + good; xs_ = np.array([f[2](t) for t in both], float); ind = np.array([1.0] * len(bad) + [0.0] * len(good)); ww = np.array([t["km"] for t in both], float)
        mx, sx = _wstats(xs_, ww); mi, si = _wstats(ind, ww); pc = float(np.sum(ww * (xs_ - mx) * (ind - mi)) / np.sum(ww) / (sx * si)) if sx > 1e-9 and si > 1e-9 else 0.0
        out.append({"key": f[0], "name": f[1], "unit": f[3], "bad": xb, "good": xg, "dx": xb - xg, "per_unit": bb[i], "contrib": bb[i] * (xb - xg), "period_corr": pc})
    out.sort(key=lambda r: -abs(r["contrib"]))
    gm = lambda L: 1000.0 * sum(t["kwh"] for t in L) / sum(t["km"] for t in L)                              # total energy / total distance of the period's trips
    return {"factors": out, "gap": gm(bad) - gm(good), "explained": sum(r["contrib"] for r in out), "bad_mean": gm(bad), "good_mean": gm(good), "r2": r2, "n": len(T)}


def _series(ex, q, target, lo, hi, scope=None):
    """Daily values of `target` over EXACTLY [lo, hi) (a partial first or last day contributes only its in-interval samples), under `scope` (default: the question's primary scope): mean for signals, energy / distance for accumulated measures. -> (list of (day_start, value, n), error)."""
    sc0 = scope or q.scope; sc = Scope(TimeSpec("range", start=int(lo), end=int(hi)), sc0.select, list(sc0.conditions))                 # the exact interval is enforced BEFORE aggregation
    q2 = q.clone(); q2.op = "trend"; q2.target = target; q2.target2 = None; q2.agg = None; q2.scope = sc; q2.scope_b = None
    out, label, period, err, fr = ex._daily(q2, sc, target, allow_tier=True)                                    # a long window reads the daily tier (same numbers, a few hundred rows instead of millions of minutes)
    if out is None: return None, err
    return [d for d in out if d[0] < hi], ""                                                                              # a partial first or last day keeps only the samples inside the interval


def _total_ratio(ex, q, lo, hi, scope):
    """Energy per kilometre of a PERIOD = total energy in Wh / total distance in km over the period's days (not the average of daily ratios)."""
    e, _ = _series(ex, q, "energy_used", lo, hi, scope); d, _ = _series(ex, q, "distance_driven", lo, hi, scope)
    if not e or not d: return None
    ed = {x[0]: x[1] for x in e}; dd = {x[0]: x[1] for x in d}; days = [k for k in ed if k in dd and dd[k] > 0]
    if not days: return None
    return 1000.0 * sum(ed[k] for k in days) / sum(dd[k] for k in days)


def _tstat(ys, k):
    a, b = ys[:k], ys[k:]; va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    return abs(np.mean(a) - np.mean(b)) / math.sqrt(va + vb) if va + vb > 0 else 0.0


def _tstats_all(ys, ks):
    """The two-sample (Welch) statistic of every split `k` at once, from cumulative sums (identical to _tstat per split; O(n) instead of O(n) per split)."""
    y = np.asarray(ys, float); n = len(y); c1 = np.concatenate([[0.0], np.cumsum(y)]); c2 = np.concatenate([[0.0], np.cumsum(y * y)]); k = np.asarray(ks, float); ki = np.asarray(ks, int)
    na, nb = k, n - k; sa, sb = c1[ki], c1[n] - c1[ki]; qa, qb = c2[ki], c2[n] - c2[ki]; ma, mb = sa / na, sb / nb
    va = np.maximum((qa - na * ma * ma) / (na - 1), 0.0) / na; vb = np.maximum((qb - nb * mb * mb) / (nb - 1), 0.0) / nb; den = np.sqrt(va + vb)
    return np.where(va + vb > 0, np.abs(ma - mb) / np.where(den > 0, den, 1.0), 0.0)


def _tstats_rows(P, ks):
    """`_tstats_all` for every row of P at once (row-wise cumulative sums: the same numbers as one row at a time)."""
    n = P.shape[1]; B = P.shape[0]; ki = np.asarray(ks, int); k = np.asarray(ks, float); na, nb = k, n - k
    c1 = np.concatenate([np.zeros((B, 1)), np.cumsum(P, axis=1)], axis=1); c2 = np.concatenate([np.zeros((B, 1)), np.cumsum(P * P, axis=1)], axis=1)
    sa, sb = c1[:, ki], c1[:, [n]] - c1[:, ki]; qa, qb = c2[:, ki], c2[:, [n]] - c2[:, ki]; ma, mb = sa / na, sb / nb
    va = np.maximum((qa - na * ma * ma) / (na - 1), 0.0) / na; vb = np.maximum((qb - nb * mb * mb) / (nb - 1), 0.0) / nb; den = np.sqrt(va + vb)
    return np.where(va + vb > 0, np.abs(ma - mb) / np.where(den > 0, den, 1.0), 0.0)


def _split(ys, B=400):
    """Best two-level split (>= 3 days each side) with a permutation p-value of the LARGEST statistic over all splits (accounts for the search; days exchangeable). -> dict or None."""
    n = len(ys)
    if n < 6: return None
    ys = np.asarray(ys, float); ks = list(range(3, n - 2)); stats = _tstats_all(ys, ks); k = ks[int(np.argmax(stats))]; obs = float(np.max(stats))
    rng = np.random.default_rng(7); P = np.empty((B, n))
    for i in range(B): P[i] = rng.permutation(ys)                                                          # the same random stream, in the same order, as shuffling one at a time
    ge = int(np.sum(_tstats_rows(P, ks).max(axis=1) >= obs - 1e-12))                                    # all the shuffles at once (400 calls of 1 ms each were 0.4 s on the Raspberry Pi)
    a, b = ys[:k], ys[k:]; ma, mb = float(np.mean(a)), float(np.mean(b)); xs = np.arange(n)
    sse_step = float(np.sum((a - ma) ** 2) + np.sum((b - mb) ** 2)); lin = np.polyfit(xs, ys, 1); sse_lin = float(np.sum((ys - np.polyval(lin, xs)) ** 2))
    sp = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(n - 2, 1)) or 1e-9
    return {"k": k, "p_adj": (1 + ge) / (1 + B), "d": abs(mb - ma) / sp, "mean_before": ma, "mean_after": mb, "shape": "a step (the level changed on one day and stayed)" if sse_step < 0.8 * sse_lin else "a gradual drift"}


def _range_k(ex): return A.cap_kwh(ex.c) * 1000.0


# ------------------------------------------------------------------------------------------------ detect_change
def detect_change(ex, q):
    from .query_executor import Result
    t = q.target; lo, hi, label = resolve_time(q.scope.time, ex.ctx); st = "wh_per_km" if t == "range_estimate" else t
    ser, err = _series(ex, q, st, lo, hi)
    if ser is None: return ex._no_data("detect_change", err, q, label, (lo, hi))
    if len(ser) < 6: return ex._no_data("detect_change", f"only {len(ser)} day(s) of data - too few to find a change", q, label, (lo, hi))
    days = [d for d, _, _ in ser]; ys = [v for _, v, _ in ser]; k = _range_k(ex)
    if t == "range_estimate":                                                                              # search the RANGE series itself, so the largest move is the largest fall in range
        keep = [i for i, v in enumerate(ys) if v]; days = [days[i] for i in keep]; ys = [k / ys[i] for i in keep]
    sgn = BAD.get(t); sp = _split(ys); jumps = [(ys[i] - ys[i - 1], i) for i in range(1, len(ys))]
    jump = max(jumps, key=lambda j: (sgn * j[0]) if sgn else abs(j[0])); c = R.CONCEPTS[t]; sd_day = float(np.std(ys[:max(3, len(ys) // 2)])) or 1e-9; shift = None
    if sp: shift = {"day": int(days[sp["k"]]), "before": sp["mean_before"], "after": sp["mean_after"], "change": sp["mean_after"] - sp["mean_before"], "pct": _pct_change(sp["mean_before"], sp["mean_after"], c.unit),
                    "p_adj": sp["p_adj"], "effect_size": sp["d"], "shape": sp["shape"], "clear": bool(sp["p_adj"] < 0.05 and sp["d"] >= 1.0)}
    r = Result("detect_change", value=float(shift["day"]) if shift else float(days[jump[1]]), unit="", target=t, period=(lo, hi), period_label=label, samples=sum(n for *_, n in ser), items=[{"day": d, "value": v} for d, v in zip(days, ys)],
               extra={"days": len(ys), "unit": c.unit, "direction_bad": sgn, "shift": shift, "jump": {"day": int(days[jump[1]]), "from_day": int(days[jump[1] - 1]), "gap_days": int(round((days[jump[1]] - days[jump[1] - 1]) / 86400)), "from": ys[jump[1] - 1], "to": ys[jump[1]], "change": jump[0], "sigmas": abs(jump[0]) / sd_day},
                      "basis": "daily values" + (" of range (capacity / consumption)" if t == "range_estimate" else "")})
    r.tier = "daily"; r.sources = ["daily values of " + R.label_of(t)] if st == t else ["daily consumption values (range = usable capacity / consumption)"]; r.quality_kind = "derived"; return r


# ------------------------------------------------------------------------------------------------ explain_change
def _driver_change(ex, q, d, lo, hi, lo2, hi2, scope_a, scope_b):
    if not ex.p.concept_available(d): return None
    a, _ = _series(ex, q, d, lo2, hi2, scope_b); b, _ = _series(ex, q, d, lo, hi, scope_a)
    if not a or not b or len(a) < 3 or len(b) < 3: return None
    xa, xb = [v for _, v, _ in a], [v for _, v, _ in b]; w = Q.welch(xa, xb); ci = Q.welch_ci(xa, xb)
    if not w: return None
    return {"driver": d, "label": R.label_of(d), "unit": R.CONCEPTS[d].unit, "before": Q.mean(xa), "after": Q.mean(xb), "change": Q.mean(xb) - Q.mean(xa), "p": w[2], "d": (ci or {}).get("d", 0.0), "n": (len(xa), len(xb))}


def explain_change(ex, q):
    from .query_executor import Result
    t = q.target; lo, hi, label = resolve_time(q.scope.time, ex.ctx); scope_a = q.scope; scope_b = q.scope_b if q.scope_b is not None else q.scope
    if q.scope_b is not None: lo2, hi2, _ = resolve_time(q.scope_b.time, ex.ctx)
    else: lo2, hi2 = lo - (hi - lo), lo
    lo2 = max(lo2, ex.ctx.lo)
    if hi2 - lo2 < 3 * DAY: return ex._no_data("explain_change", "there is not enough earlier history to compare with", q, label, (lo, hi))
    st = "wh_per_km" if t == "range_estimate" else t; k = _range_k(ex) if t == "range_estimate" else None
    a, err = _series(ex, q, st, lo2, hi2, scope_b); b, err2 = _series(ex, q, st, lo, hi, scope_a)
    if a is None or b is None: return ex._no_data("explain_change", err or err2, q, label, (lo, hi))
    if len(a) < 3 or len(b) < 3: return ex._no_data("explain_change", f"too few days of data to compare ({len(a)} before, {len(b)} after)", q, label, (lo, hi))
    xa, xb = [v for _, v, _ in a], [v for _, v, _ in b]; w = Q.welch(xa, xb); ci = Q.welch_ci(xa, xb)
    if st == "wh_per_km":                                                                                 # consumption of a period = total energy / total distance
        ra, rb = _total_ratio(ex, q, lo2, hi2, scope_b), _total_ratio(ex, q, lo, hi, scope_a); basis = "total energy / total distance of each period"
        if ra is None or rb is None: ra, rb, basis = Q.mean(xa), Q.mean(xb), "average of the daily values"
    else: ra, rb, basis = Q.mean(xa), Q.mean(xb), "average of the daily values"
    if t == "range_estimate" and basis.startswith("total energy"): basis = "range = usable capacity / (total energy / total distance of each period)"
    ma, mb = (k / ra, k / rb) if k else (ra, rb)
    c = R.CONCEPTS[t]; sgn = BAD.get(t); both = _series(ex, q, st, lo2, hi, scope_a)[0] or []; ys_both = [v for _, v, _ in both]
    days_both = [d for d, _, _ in both]
    if k: keep = [i for i, v in enumerate(ys_both) if v]; days_both = [days_both[i] for i in keep]; ys_both = [k / ys_both[i] for i in keep]
    sp = _split(ys_both) if len(ys_both) >= 6 else None
    obs = {"before": ma, "after": mb, "change": mb - ma, "pct": _pct_change(ma, mb, c.unit), "p": w[2] if w else None, "effect_size": abs((ci or {}).get("d", 0.0)), "clear": bool(w and w[2] < 0.05 and abs((ci or {}).get("d", 0.0)) >= 0.8),
           "days": (len(a), len(b)), "start_day": int(days_both[sp["k"]]) if sp else None, "start_p_adj": sp["p_adj"] if sp else None, "shape": sp["shape"] if sp else None, "worse": None if not sgn else bool(sgn * (mb - ma) > 0),
           "ci": None if not ci else [ci["lo"], ci["hi"]], "basis": basis}
    contrib = None; assoc = []; restricted = bool(q.scope.conditions or q.scope.select or (q.scope_b is not None and (q.scope_b.conditions or q.scope_b.select)))
    if t in ("wh_per_km", "range_estimate") and not restricted:
        Tall = Q.trips_between(ex.c, ex.ctx.lo, ex.ctx.hi + 1); Ta, Tb = Q.trips_between(ex.c, lo2, hi2), Q.trips_between(ex.c, lo, hi)
        Q.attach_tyre(ex.c, Tall + Ta + Tb)
        res = _weighted_attribution(ex.c, Tall, Tb, Ta); fit = {"r2": res["r2"]} if res else None
        if res and abs(res["gap"]) > 1e-6:
            contrib = {"gap": res["gap"], "explained": res["explained"], "residual": res["gap"] - res["explained"], "r2": fit["r2"], "n_trips": (len(Ta), len(Tb)), "basis": "distance-weighted trips (total energy / total distance)", "trip_before": res["good_mean"], "trip_after": res["bad_mean"],
                       "factors": [{"name": f["name"], "unit": f["unit"], "before": f["good"], "after": f["bad"], "contribution": f["contrib"], "share_pct": 100 * f["contrib"] / res["gap"], "per_unit": f["per_unit"], "period_corr": f.get("period_corr", 0.0)} for f in res["factors"] if abs(f["contrib"]) >= 0.05], "unit": "Wh/km"}
            if t == "range_estimate": obs["range_before"], obs["range_after"] = ma, mb
    else:
        for d, sign, why in DEPENDS.get(t, []):
            ch = _driver_change(ex, q, d, lo, hi, lo2, hi2, scope_a, scope_b)
            if ch is None: continue
            moved = ch["p"] < 0.05 and abs(ch["d"]) >= 0.8; direction = (1 if ch["change"] > 0 else -1) * sign; toward = (direction * (1 if obs["change"] > 0 else -1) > 0) if abs(obs["change"]) > 0 else False
            assoc.append(dict(ch, moved=moved, consistent=bool(moved and toward), mechanism=why))
        if t == "tyre_pressure" and ex.p.concept_available("ambient_temperature"):                          # temperature-normalised pressure, joined by DAY: a real leak survives the correction
            at, _ = _series(ex, q, "ambient_temperature", lo2, hi2, scope_b); bt, _ = _series(ex, q, "ambient_temperature", lo, hi, scope_a)
            ta = {d: v for d, v, _ in (at or [])}; tb = {d: v for d, v, _ in (bt or [])}
            na = [v * 293.15 / (ta[d] + 273.15) for d, v, _ in a if d in ta]; nb = [v * 293.15 / (tb[d] + 273.15) for d, v, _ in b if d in tb]
            if len(na) >= 3 and len(nb) >= 3:
                wn = Q.welch(na, nb); obs["normalised"] = {"before": Q.mean(na), "after": Q.mean(nb), "change": Q.mean(nb) - Q.mean(na), "p": wn[2] if wn else None, "days": (len(na), len(nb)), "convention": GAUGE_NOTE}
    unexplained = None
    if contrib is not None: unexplained = abs(contrib["residual"]) > 0.3 * abs(contrib["gap"])
    elif t in DEPENDS: unexplained = not any(x["consistent"] for x in assoc)
    open_items = MISSING.get(t, MISSING["default"]) if (unexplained or contrib is None) else MISSING["default"][:1]
    if restricted and t in ("wh_per_km", "range_estimate"): open_items = ["a contribution breakdown for this restricted selection (the trip-level model is fitted on all trips)"] + open_items
    r = Result("explain_change", value=obs["pct"] if obs["pct"] is not None else obs["change"], unit="%" if obs["pct"] is not None else c.unit, target=t, period=(lo, hi), period_label=label, samples=sum(n for *_, n in a + b),
               extra={"obs": obs, "contrib": contrib, "assoc": assoc, "unexplained": unexplained, "open": open_items, "windows": {"earlier": [lo2, hi2], "recent": [lo, hi]}, "unit": c.unit}, tier="daily + trip ledger" if contrib else "daily")
    r.sources = (["daily values of " + R.label_of(t)] if st == t else ["daily consumption values (range = usable capacity / consumption)"]) + (["trips"] if contrib else []) + [R.label_of(x["driver"]) for x in assoc]; r.quality_kind = "estimated" if contrib else "derived"; r.estimated = bool(contrib)
    if contrib: r.assumptions = [f"a linear trip-level model weighted by distance (R2 = {contrib['r2']:.2f}) attributes the change to the recorded conditions", "anything the model and the recorded conditions do not capture appears in the unexplained part"] + (["low tyre pressure is the shortfall against the spec cold pressure, corrected to 20 C with the recorded tyre temperature (the ambient temperature is not used as a substitute)"] if any("tyre" in f["name"] for f in contrib["factors"]) else []) + ["daily values are serially correlated, so the change-point p-value assumes independent days and is exploratory"]
    return r

"""Cause diagnosis: from what changed in the history to the physical cause most likely behind it, with the evidence, what was ruled out, and how sure the engine is.

How it works (docs/CAUSES.md):
  1. CHANGES.  Every quantity the vehicle stores is turned into one value per day.  Each series gets its change points, their size in units of the series' own day-to-day spread (made stricter where the series
     drifts slowly), whether the change stayed or passed (a few days of a dirty windscreen) and when it started.  Signals that depend on conditions (battery temperature on the weather and the load, cabin
     climate power on the weather) are first adjusted for them with a robust fit, so a season is not mistaken for a fault.  Event rates (harsh driving, warnings) are tested as Poisson counts against the
     distance driven.
  2. CONSUMPTION.  A change in energy per kilometre is split with a physical trip model (aerodynamic drag ~ speed squared, heating per kilometre, cold-weather losses, highway share), fitted ONLY on the days before
     the suspected change, so the fit cannot absorb it.  What the recorded conditions explain is credited to them; the remainder is asked which shape it has - a step on one day (something added: drag) or a
     creep (friction at the wheels).
  3. HYPOTHESES.  A library of vehicle causes (library.py), each with the signature it leaves.  Each is scored on the changes above.
  4. RANKING.  Scores become confidences against an explicit "no known cause fits" alternative; a history with no significant change says so instead of inventing a cause.  Everything that could not be
     assessed (the vehicle does not store the signal) is listed.

The engine reads only the vehicle's stored data; it never reads how a history was made."""
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import stats as S
from .data import History

UNIT = {"coolant_flow": "L/min", "battery_temperature": "°C", "aux_battery_voltage": "V", "ambient_temperature": "°C", "hvac_power": "W", "speed": "km/h", "tyre_pressure": "kPa"}
STEP_Z = 4.0                 # a change must exceed this many day-to-day spreads ...
STEP_P = 0.01                # ... and be unlikely under shuffling the days


@dataclass
class Change:
    """One quantity's change: signed size in the series' own spread units, when, and what shape."""
    name: str
    label: str
    ok: bool = False
    effect: float = 0.0
    p: float = 1.0
    onset: int = -1
    end: int = -1                        # last day of a transient change (-1 = persists)
    shape: str = ""                      # step | ramp | transient
    before: float = float("nan")
    after: float = float("nan")
    unit: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def rel(self):
        """Size of the change as a fraction of the usual level."""
        if not (np.isfinite(self.before) and np.isfinite(self.after)) or abs(self.before) < 1e-9: return 0.0
        return float((self.after - self.before) / abs(self.before))

    def strength(self, sign):
        if not self.ok: return 0.0
        return max(0.0, sign * self.effect)


def _floor(x, rel, absolute):
    m = abs(float(np.nanmedian(x))) if np.isfinite(x).any() else 0.0; return max(rel * m, absolute)


def huber_adjust(y, X, iters=8):
    """Remove from y what the condition columns X explain, with a robust (Huber) fit over all days so that a change in a minority of days is not absorbed.  -> residual (same scale as y), r2 explained."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < 14: return y.copy(), 0.0
    Z = (X[ok] - X[ok].mean(0)) / np.maximum(X[ok].std(0), 1e-9); A = np.column_stack([np.ones(ok.sum()), Z]); yy = y[ok]; w = np.ones(len(yy))
    for _ in range(iters):
        beta = np.linalg.lstsq(A * np.sqrt(w)[:, None], yy * np.sqrt(w), rcond=None)[0]; r = yy - A @ beta; s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9; u = np.abs(r) / (1.5 * s); w = np.where(u <= 1, 1.0, 1.0 / u)
    full = np.column_stack([np.ones(len(y)), (np.where(np.isfinite(X), X, np.nanmean(X, 0)) - X[ok].mean(0)) / np.maximum(X[ok].std(0), 1e-9)]); res = y - full @ beta
    tot = float(np.var(yy)); r2 = 1.0 - float(np.var(yy - A @ beta)) / tot if tot > 0 else 0.0; return res, r2


def analyse(y, name, label, unit="", rel_floor=0.004, abs_floor=0.0, ctx=None, min_rel_change=0.0):
    """Change features of one daily series (see module docstring)."""
    y = S.despike(np.asarray(y, float)); c = Change(name, label, unit=unit)
    if np.isfinite(y).sum() < 14: return c
    raw = y; r2 = 0.0
    if ctx is not None:
        adj, r2 = huber_adjust(y, ctx)
        if r2 >= 0.25: y = adj                                                                        # adjust only for conditions that really explain the series; a weak fit just smears a real change over the days
    c.extra["r2"] = r2; st = S.best_step(y, min_seg=3)
    if st is None: return c
    k = st["k"]; before = y[:k]; sd, rho = S.noise(before, floor=_floor(raw[:k], rel_floor, abs_floor) + 1e-9)
    if rho > 0.2: sd *= min(2.0, math.sqrt((1 + min(rho, 0.9)) / (1 - min(rho, 0.9))))                    # a series whose noise itself drifts slowly: days are not independent, so a given jump is less surprising
    b = float(np.median(S.valid(before))); a = float(st["after"]); eff = (a - b) / sd
    c.before, c.after, c.p = float(np.median(S.valid(raw[:k]))), float(np.median(S.valid(raw[k:]))), st["p"]
    practical = abs(c.after - c.before) >= min_rel_change * max(abs(c.before), 1e-9)
    if st["p"] < STEP_P and abs(eff) >= STEP_Z and practical and not st["transient"]: c.ok = True; c.effect = float(eff); c.onset = int(k); c.shape = st["shape"]
    exc = S.excursion(y, b, sd, k=4.0, min_days=2)
    if exc and not (exc["peak_z"] >= 6.0 and exc["days"] >= 2 and S.excursion_p(y, b, sd, exc) < STEP_P): exc = None
    if exc and (not c.ok or exc["peak_z"] > abs(c.effect) * 1.5):
        seg = raw[exc["start"]:exc["end"] + 1]
        if abs(float(np.nanmedian(seg)) - c.before) >= min_rel_change * max(abs(c.before), 1e-9):
            c.ok = True; c.effect = float(exc["direction"] * exc["peak_z"]); c.onset = int(exc["start"]); c.end = -1 if exc["persistent"] else int(exc["end"]); c.shape = "transient" if not exc["persistent"] else "step"; c.after = float(np.nanmedian(seg))
    elif st["transient"] and st["p"] < STEP_P and abs(eff) >= STEP_Z and practical: c.ok = True; c.effect = float(eff); c.onset = int(k); c.shape = "transient"
    if c.ok and c.shape in ("step", "ramp") and c.end < 0:                                           # the change began before the day the level settled: walk back to where the series first left its usual band for good
        j = c.onset; sg = 1 if c.effect > 0 else -1
        while j - 1 >= 0 and (not np.isfinite(y[j - 1]) or sg * (y[j - 1] - b) > 2.0 * sd): j -= 1
        c.onset = int(max(j, 0))
    if not c.ok:                                                                                          # a creep that began in the last days has no earlier level to compare with: test the trend itself
        er = S.end_ramp(y, sd_floor_for_ramp(y, sd, raw, rel_floor, abs_floor))
        if er is not None and abs(er["total"]) >= min_rel_change * max(abs(float(np.nanmedian(raw[: er["start"]]))), 1e-9):
            c.ok = True; c.effect = float(er["direction"] * min(abs(er["z"]) * 1.6, 12.0)); c.onset = int(er["start"]); c.shape = "ramp"; c.p = float(math.erfc(abs(er["z"]) / math.sqrt(2)))
            c.before = float(np.nanmedian(raw[: er["start"]])); c.after = float(np.nanmedian(raw[-3:]))
    return c


def sd_floor_for_ramp(y, sd, raw, rel_floor, abs_floor):
    """Noise level for judging a trend over the last days: the series' own spread over everything before the trend, never below the physical floor."""
    ok = S.valid(y[: max(len(y) - 8, 6)]); base = S.robust_sd(ok, floor=_floor(raw, rel_floor, abs_floor) + 1e-9) if len(ok) > 5 else sd
    return max(base, 1e-9)


def poisson_window(counts, exposure, name, label, unit="", min_total=8, max_len=14, z_min=4.5):
    """The run of days where the event rate (per unit of exposure, e.g. per 100 km driven) is highest compared with the other days: a transient burst or a persistent shift.  Poisson likelihood ratio; z corrected
    for having searched every window."""
    n = len(counts); c = Change(name, label, unit=unit); ex = np.where(np.isfinite(exposure) & (exposure > 0), exposure, 0.0)
    if counts.sum() < min_total or (ex > 0).sum() < 14: c.extra["total"] = float(counts.sum()); return c
    tot_c, tot_e = float(counts.sum()), float(ex.sum()); r0 = tot_c / max(tot_e, 1e-9); best = (0.0, None, None, 0)
    cs = np.concatenate([[0.0], np.cumsum(counts)]); es = np.concatenate([[0.0], np.cumsum(ex)]); nwin = 0
    for L in range(2, max_len + 1):
        for i in range(0, n - L + 1):
            nwin += 1; cin, ein = cs[i + L] - cs[i], es[i + L] - es[i]; cout, eout = tot_c - cin, tot_e - ein
            if ein < 5 or eout < 5: continue
            rin, rout = cin / ein, cout / eout
            llr = (cin * math.log(max(rin, 1e-12) / max(r0, 1e-12)) if cin > 0 else 0.0) + (cout * math.log(max(rout, 1e-12) / max(r0, 1e-12)) if cout > 0 else 0.0) + (ein * r0 - cin) * 0 if False else 0.0
            llr = (cin * math.log(max(rin, 1e-12) / max(rout, 1e-12)) if cin > 0 else 0.0) - (ein * (rin - rout))
            z = math.sqrt(max(2 * llr, 0.0)) * (1 if rin > rout else -1)
            if abs(z) > abs(best[0]): best = (z, i, L, 1)
    if best[1] is None: return c
    z, i, L, _ = best; zc = abs(z) - math.sqrt(2 * math.log(max(nwin, 2))) * 0.6
    c.before = float(tot_c / tot_e); c.after = float((cs[i + L] - cs[i]) / max(es[i + L] - es[i], 1e-9)); c.effect = float(math.copysign(zc, z)); c.onset = int(i); c.end = int(i + L - 1) if i + L < n - 1 else -1; c.shape = "transient" if c.end >= 0 else "step"
    c.ok = abs(zc) >= z_min - 1.5 and abs(z) >= z_min; c.p = float(math.erfc(abs(z) / math.sqrt(2))); c.extra["total"] = tot_c; return c


# =========================================================================================== features
def build(H: History):
    """All changes the vehicle's data allows, by name.  Missing quantities are absent from `F` and recorded in `F['_missing']`."""
    F = {}; missing = []
    amb = H.daily("ambient_temperature") if H.available("ambient_temperature") else None; power = H.daily("battery_power") if H.available("battery_power") else None

    def ctx(*cols):
        cols = [c for c in cols if c is not None]
        return np.column_stack(cols) if cols else None

    def series(name, concept, agg="AVG", label=None, ctx_=None, **kw):
        if not H.available(concept): missing.append(concept); return None
        y = H.daily(concept, agg)
        if concept == "aux_battery_voltage": y = np.where(y > 13.0, np.nan, y)                           # above 13 V the charger is on: that is not the battery's own (resting) level
        F[name] = analyse(y, name, label or concept.replace("_", " "), UNIT.get(concept, ""), ctx=ctx_, **kw); F[name].extra["y"] = y; return y
    series("flow", "coolant_flow", abs_floor=0.05, label="coolant flow", min_rel_change=0.15)
    series("batt", "battery_temperature", abs_floor=0.2, label="battery temperature", ctx_=ctx(amb, power), min_rel_change=0.03)
    series("amb", "ambient_temperature", abs_floor=0.4, label="outside temperature")
    series("hvac", "hvac_power", rel_floor=0.05, abs_floor=25, label="cabin climate power", ctx_=ctx(amb), min_rel_change=0.25)
    series("v12", "aux_battery_voltage", "MIN", abs_floor=0.02, label="12 V battery voltage", min_rel_change=0.015)
    series("soh", "battery_soh", abs_floor=0.05, label="battery health", min_rel_change=0.01)
    tp = {w: H.daily(f"tyre_pressure_{w}") for w in ("fl", "fr", "rl", "rr") if H.available(f"tyre_pressure_{w}")}
    if len(tp) == 4:
        M = np.vstack([tp[w] for w in tp]); names = list(tp)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                                                # a wheel with no readings yet is an empty slice, not a problem
            for i, w in enumerate(names):
                others = np.nanmedian(np.delete(M, i, axis=0), axis=0); rel = tp[w] - others; c = analyse(rel, f"tyre_{w}", f"{w.upper()} tyre pressure relative to the other tyres", "kPa", rel_floor=0.0, abs_floor=2.0)
                F[f"tyre_{w}"] = c; c.extra["y"] = rel; c.extra["abs"] = tp[w]; base = np.nanmedian(rel[: max(8, len(rel) // 3)]); c.extra["depth"] = float(base - np.nanmin(rel)) if np.isfinite(rel).sum() > 8 else 0.0
            mean4 = np.nanmean(M, axis=0); F["tyre_mean"] = analyse(mean4, "tyre_mean", "tyre pressure (all four)", "kPa", abs_floor=1.5, ctx=ctx(amb)); F["tyre_mean"].extra["y"] = mean4
    else: missing.append("tyre_pressure (all four wheels)")
    tr = H.trips(); km_d = H.by_day(tr["km"], tr["day"], how="sum")
    if H.available("camera_blockage"):
        cb = H.daily("camera_blockage"); F["cam_block"] = analyse(cb, "cam_block", "camera view blocked", "", rel_floor=0.05, abs_floor=0.012, min_rel_change=0.3); F["cam_block"].extra["y"] = cb
    else: missing.append("camera_blockage")
    ncam = H.event_counts("CAMERA_BLOCKED"); F["ev_cam"] = poisson_window(ncam, np.where(km_d > 0, 1.0, 0.0) * 1.0, "ev_cam", "camera-blocked events", "per day", min_total=1, z_min=3.0)
    F["ev_overtemp"] = poisson_window(H.event_counts("WARN_BATT_OVERTEMP", "COOLANT_FLOW_DROP", "WARN_OVERTEMP"), np.ones(H.D), "ev_overtemp", "over-temperature / low-flow warnings", "per day", min_total=6, z_min=4.5)
    F["ev_12v"] = poisson_window(H.event_counts("LOW_12V", "VOLTAGE_DIP", "ECU_RESET"), np.ones(H.D), "ev_12v", "low-voltage events and ECU resets", "per day", min_total=4, z_min=3.5)
    harsh_n = H.event_counts("HARSH_BRAKE", "HARSH_ACCEL", "HARSH_CORNER")
    if np.isfinite(km_d).any(): F["harsh"] = poisson_window(harsh_n, np.where(np.isfinite(km_d), km_d / 100.0, 0.0), "harsh", "harsh braking, acceleration and cornering", "per 100 km", min_total=12, max_len=14, z_min=4.5)
    F["_trips"] = tr; F["_missing"] = missing; F["_days"] = H.D; F["_amb"] = amb
    wh = H.by_day(tr["wh"], tr["day"], weights=tr["km"]) if len(tr["km"]) else np.full(H.D, np.nan)
    if np.isfinite(wh).sum() >= 14: F["wh"] = analyse(wh, "wh", "energy per kilometre", "Wh/km", rel_floor=0.01, abs_floor=1.0, ctx=ctx(amb), min_rel_change=0.03); F["wh"].extra["y"] = wh
    if len(tr["km"]) and np.isfinite(tr["hwy"]).any():
        hw = H.by_day(tr["hwy"], tr["day"], weights=tr["km"]); F["hwy"] = analyse(hw, "hwy", "highway share of driving", "", rel_floor=0.05, abs_floor=0.03, min_rel_change=0.2)
    if len(tr["km"]) and np.isfinite(tr["speed"]).any():
        sp = H.by_day(tr["speed"], tr["day"], weights=tr["km"]); F["spd"] = analyse(sp, "spd", "average speed", "km/h", rel_floor=0.05, abs_floor=1.5, min_rel_change=0.1)
    if "wh" in F: F["wh_raw"] = analyse(F["wh"].extra["y"], "wh_raw", "energy per kilometre (before adjusting for the weather)", "Wh/km", rel_floor=0.01, abs_floor=1.0, min_rel_change=0.03)
    F.update(_consumption(H, F))
    return F


def _wls(X, y, w):
    A = np.column_stack([np.ones(len(y)), X]) * w[:, None]; return np.linalg.lstsq(A, y * w, rcond=None)[0]


def _huber_wls(X, y, km, iters=6):
    """Weighted (by distance) robust fit of energy per kilometre on the physical regressors."""
    w0 = np.sqrt(np.maximum(km, 0.5)); w = w0.copy(); A1 = np.column_stack([np.ones(len(y)), X])
    for _ in range(iters):
        sol = np.linalg.lstsq(A1 * w[:, None], y * w, rcond=None)[0]; r = y - A1 @ sol; sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9; u = np.abs(r) / (1.8 * sc); w = w0 * np.where(u <= 1, 1.0, 1.0 / u)
    return sol


def _day_means(resid, day, km, D):
    out = np.full(D, np.nan)
    for d in range(D):
        m = day == d
        if m.any(): out[d] = float(np.average(resid[m], weights=km[m]))
    return out


def _consumption(H, F):
    """Trip-level physical model of energy per kilometre (aerodynamic drag ~ speed squared, climate power per km, cold-weather losses, highway share).
    1. fit it on ALL days with a robust fit: weather varies a lot over a window, so its effect is learned from the data;
    2. look for a change in what the model does not explain (the daily residual), and when it starts;
    3. confirm by refitting only on the days before that start (so the fit cannot have absorbed the change) and re-measuring;
    4. ask what shape the unexplained part has (a step on one day, or a creep) and what the recorded conditions explain of the raw change."""
    out = {}; tr = F["_trips"]
    if len(tr["km"]) < 40: return out
    day = tr["day"].astype(int); n = len(day); v = tr["speed"]; amb = tr["amb"]; hv = tr["hvac"]; wh = tr["wh"]; km = tr["km"]; D = H.D
    if not (np.isfinite(v).all() and np.isfinite(amb).all() and np.isfinite(hv).all()): return out
    names = ("speed squared", "climate power per km", "cold weather", "highway share"); hwy = np.where(np.isfinite(tr["hwy"]), tr["hwy"], 0.0)
    X = np.column_stack([v ** 2 / 1000.0, hv / np.maximum(v, 5.0), np.clip(15.0 - amb, 0, 40), hwy]); A1 = np.column_stack([np.ones(n), X])
    sol = _huber_wls(X, wh, km); resid = wh - A1 @ sol; eday = _day_means(resid, day, km, D)
    base = float(np.average(wh, weights=km)); sd_e, rho = S.noise(eday, floor=0.5)
    # (2) strongest change in the unexplained part
    ok = np.isfinite(eday); d_ix = np.arange(D)[ok]; e = eday[ok]; best = None
    for k0 in range(10, D - 3):
        a = e[d_ix < k0]; b = e[d_ix >= k0]
        if len(a) < 6 or len(b) < 4: continue
        z = (b.mean() - a.mean()) / (sd_e * math.sqrt(1.0 / len(a) + 1.0 / len(b)))
        if best is None or abs(z) > abs(best[0]): best = (z, k0)
    er = S.end_ramp(eday, sd_e, z_min=3.5) if best is None or abs(best[0]) < 4.0 else None
    if best is None and er is None: return out
    k = best[1] if best else er["start"]; z = best[0] if best else er["direction"] * er["z"]
    # (3) confirm with a fit that has not seen the days after the onset
    ref = day < k; post = day >= k; z_pre = z
    if ref.sum() >= 30 and post.sum() >= 6:
        sol_p = _huber_wls(X[ref], wh[ref], km[ref]); r_p = wh - A1 @ sol_p; ed = _day_means(r_p, day, km, D); sd_p, _ = S.noise(ed[:k], floor=0.5); pre = S.valid(ed[:k]); pst = S.valid(ed[k:])
        if len(pre) >= 6 and len(pst) >= 4: z_pre = (pst.mean() - pre.mean()) / (sd_p * math.sqrt(1.0 / len(pre) + 1.0 / len(pst)))
    pm = float(np.nanmean(eday[k:]) - np.nanmean(eday[:k]))
    # (4) shape of the unexplained part
    sse_step = sse_ramp = float("inf"); k_step = k_ramp = k
    for k0 in range(8, D - 3):
        a = e[d_ix < k0]; b = e[d_ix >= k0]; x = (d_ix[d_ix >= k0] - k0).astype(float)
        if len(a) >= 5 and len(b) >= 4:
            s_ = float(np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2))
            if s_ < sse_step: sse_step, k_step = s_, k0
            if x.std() > 0:
                slope = float(np.sum((x - x.mean()) * (b - b.mean())) / np.sum((x - x.mean()) ** 2)); fit = a.mean() + slope * x; s2 = float(np.sum((a - a.mean()) ** 2) + np.sum((b - fit) ** 2))
                if s2 < sse_ramp: sse_ramp, k_ramp = s2, k0
    shape = "ramp" if sse_ramp < 0.85 * sse_step else "step"
    drag_t = 0.0; pm_mask = post
    if pm_mask.sum() >= 12:
        Zd = np.column_stack([np.ones(pm_mask.sum()), (v[pm_mask] ** 2) / 1000.0 - np.mean(v[pm_mask] ** 2) / 1000.0]); wd = np.sqrt(np.maximum(km[pm_mask], 0.5)); bd = np.linalg.lstsq(Zd * wd[:, None], resid[pm_mask] * wd, rcond=None)[0]
        rd = resid[pm_mask] * wd - (Zd * wd[:, None]) @ bd; s2 = float(np.sum(rd ** 2)) / max(pm_mask.sum() - 2, 1); cv = s2 * np.linalg.inv((Zd * wd[:, None]).T @ (Zd * wd[:, None]) + 1e-9 * np.eye(2)); drag_t = float(bd[1] / math.sqrt(max(cv[1, 1], 1e-12)))
    # what the recorded conditions explain of the change between the recent three weeks and the days before them (the usual "why has it changed lately")
    rec = day >= D - 21; old_ = day < D - 21; win = None
    if rec.sum() >= 8 and old_.sum() >= 30:
        dx = np.array([np.average(X[rec, i], weights=km[rec]) - np.average(X[old_, i], weights=km[old_]) for i in range(X.shape[1])]); terms = sol[1:] * dx; cov = np.linalg.inv((A1 * np.sqrt(np.maximum(km, 0.5))[:, None]).T @ (A1 * np.sqrt(np.maximum(km, 0.5))[:, None]) + 1e-9 * np.eye(A1.shape[1])) * float(np.var(resid) + 1e-9)
        se = float(math.sqrt(max(dx @ cov[1:, 1:] @ dx, 1e-12))); win = {"gap": float(np.average(wh[rec], weights=km[rec]) - np.average(wh[old_], weights=km[old_])), "explained": float(terms.sum()), "se": se, "terms": {nm: float(t) for nm, t in zip(names, terms)}, "amb_change": float(np.average(amb[rec], weights=km[rec]) - np.average(amb[old_], weights=km[old_]))}
    contrib = {}; gap = 0.0; rawc = F.get("wh_raw")
    if rawc is not None and rawc.ok and rawc.onset >= 8:
        k2 = rawc.onset; r2_ = day < k2; p2_ = day >= k2
        if r2_.sum() >= 20 and p2_.sum() >= 5:
            gap = float(np.average(wh[p2_], weights=km[p2_]) - np.average(wh[r2_], weights=km[r2_])); contrib = {nm: float(sol[i] * (np.average(X[p2_, i - 1], weights=km[p2_]) - np.average(X[r2_, i - 1], weights=km[r2_]))) for i, nm in enumerate(names, start=1)}
            out["wh_raw_model"] = {"gap": gap, "contrib": contrib, "onset": int(k2)}
    zc = min(abs(z), abs(z_pre)) * (1 if z * z_pre > 0 else 0)
    out["wh_model"] = {"gap": gap, "resid": float(pm), "z": float(zc), "z_global": float(z), "z_pre": float(z_pre), "onset": int(k_step if shape == "step" else k_ramp), "scan_onset": int(k), "shape": shape, "sse_step": sse_step, "sse_ramp": sse_ramp,
                       "contrib": contrib, "sd_day": float(sd_e), "base": base, "drag_t": drag_t, "window": win, "rel": float(pm / base) if base else 0.0, "ok": bool(zc >= 4.0 and pm > 0.03 * base) if pm > 0 else bool(zc >= 4.0 and -pm > 0.03 * base), "up": bool(pm > 0)}
    return out


# =========================================================================================== ranking
ONSET_FROM = {"cooling_pump_degradation": ("flow",), "tyre_slow_leak": ("tyre_fl", "tyre_fr", "tyre_rl", "tyre_rr"), "weak_12v_battery": ("v12",), "camera_obstruction": ("cam_block", "ev_cam"), "added_drag": ("wh_model",),
              "rolling_resistance_increase": ("wh_model",), "aggressive_driving": ("harsh",), "cold_weather": ("wh_raw", "amb"), "climate_overuse": ("hvac",), "higher_speeds": ("spd", "hwy"), "battery_aging": ("soh",)}
BETA = 9.0                  # how fast confidence grows with evidence (calibrated on the development set)
NO_CAUSE_EVIDENCE = 0.30    # the "nothing known fits" alternative, in units of evidence
NO_CAUSE_PRIOR = 0.45


def score(cause, F):
    """-> (evidence in [0,1] or None when the cause cannot be assessed from the stored data, supporting notes)."""
    tot = w = 0.0; notes = []; first = True
    for name, wt, fn in cause.tests:
        r = fn(F)
        if r is None:
            if first: return None, []
            continue
        v, note = r; tot += wt * v; w += wt; first = False
        if v >= 0.25 and note: notes.append(note)
    return (tot / w if w else None), notes


def _onset(F, feat):
    c = F.get(feat)
    if isinstance(c, Change): return c.onset if c.ok else None
    if isinstance(c, dict): return c.get("onset") if c.get("ok", True) else None
    return None


def diagnose(H: History, symptom=None, F=None, library=None, since_day=None):
    """Rank the causes of what changed.  `since_day` (a day index in the window): the question is about the time since then - a change that settled long before is an older finding, reported but not given as the reason
    for something that moved lately."""
    from .library import LIBRARY
    F = F or build(H); library = library or LIBRARY; rows = []; unassessable = []; older = []
    if since_day is not None:
        for k, c in list(F.items()):
            if isinstance(c, Change) and c.ok and c.onset >= 0 and c.onset < since_day - 4 and c.end < 0: older.append(c.label); c.ok = False
        m = F.get("wh_model")
        if m and m.get("ok") and m["onset"] < since_day - 4: older.append("energy per kilometre"); m["ok"] = False
        if m and m.get("window"):
            F["wh_model"] = dict(m)
    for cz in library:
        e, notes = score(cz, F)
        if e is None: unassessable.append(cz.id); continue
        rows.append({"id": cz.id, "title": cz.title, "evidence": float(e), "notes": notes, "cause": cz})
    unk = math.log(NO_CAUSE_PRIOR) + BETA * NO_CAUSE_EVIDENCE; p_none = 1.0
    for g in {r["cause"].group or r["id"] for r in rows}:                                                  # causes of one symptom compete; independent causes are each judged against "no cause" alone
        mem = [r for r in rows if (r["cause"].group or r["id"]) == g]; lg = [math.log(r["cause"].prior) + BETA * r["evidence"] for r in mem]; mx = max(lg + [unk]); den = sum(math.exp(x - mx) for x in lg) + math.exp(unk - mx)
        for r, x in zip(mem, lg): r["confidence"] = float(math.exp(x - mx) / den)
        p_none = min(p_none, float(math.exp(unk - mx) / den))
    rows.sort(key=lambda r: -r["confidence"])
    changes = [c for k, c in F.items() if isinstance(c, Change) and c.ok]
    out = []
    for r in rows:
        cz = r["cause"]; on = None
        for f in ONSET_FROM.get(cz.id, ()):
            o = _onset(F, f)
            if o is not None and o >= 0 and (on is None or o < on): on = o
        out.append({"id": r["id"], "title": cz.title, "confidence": round(r["confidence"], 3), "evidence": round(r["evidence"], 3), "notes": r["notes"], "mechanism": cz.mechanism, "check": cz.check, "severity": cz.severity,
                    "onset_day": on, "onset_date": H.date(on) if on is not None else None})
    top = out[0] if out else None
    nothing = (not changes and not (F.get("wh_model") or {}).get("ok")) or top is None                      # "nothing to explain" is about the data: no significant change anywhere.  A change with an uncertain cause is a different answer.
    return {"hypotheses": out, "p_no_known_cause": round(p_none, 3), "nothing_to_explain": bool(nothing), "changes": [{"name": c.name, "label": c.label, "effect": round(c.effect, 1), "onset_day": c.onset, "shape": c.shape, "before": c.before, "after": c.after, "unit": c.unit} for c in changes],
            "missing": sorted(set(F.get("_missing", []))), "unassessable": unassessable, "older_changes": older, "days": H.D, "model": F.get("wh_model"), "_features": F}


def diagnose_db(path, symptom=None, horizon=70):
    """Diagnose a stored history (read-only)."""
    import sqlite3
    conn = sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)
    try: return diagnose(History(conn, horizon), symptom)
    finally: conn.close()

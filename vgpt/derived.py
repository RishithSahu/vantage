"""Derived metrics: deterministic physics / bookkeeping on top of recorded signals. The language model never computes any of this; it only asks for a metric by name.

Every derived metric declares its dependencies in signal_registry (DERIVED_REQUIRES / DERIVED_ALTERNATIVES). `compute` picks the first alternative whose inputs are available in the current database and returns the values together with
HOW they were obtained (via, formula, quality, assumptions). Rules that are enforced here, not left to the caller:
  * measured beats derived: if the vehicle records motor torque, that is what is used;
  * an electrical quantity is never used as a mechanical one (battery power is not motor power);
  * a value that rests on an assumption (nominal gear ratio, wheel radius) is returned with quality "estimated" and the assumptions listed, so the composer must say "estimated";
  * no input -> no value (NaN), never a made-up number.
Two kinds: POINTWISE metrics (a number per sample, e.g. torque, acceleration, turn radius) and ACCUMULATED metrics (energy, distance, regenerated energy, Wh/km: an integral over a scope)."""
import math
from dataclasses import dataclass, field

import numpy as np

G = 9.80665
KW_TO_HP = 1.34102


@dataclass
class DerivedSeries:
    name: str
    values: np.ndarray
    unit: str
    quality: str = "derived"                     # derived | estimated
    via: tuple = ()                              # the concepts the values were computed from
    formula: str = ""
    assumptions: list = field(default_factory=list)


def _col(frame, name):
    v = frame.cols.get(name)
    return None if v is None else np.asarray(v, dtype=float)


def _dt(frame):
    ts = frame.ts.astype(float); d = np.diff(ts, prepend=ts[0] - frame.res if len(ts) else 0.0)
    d[(d <= 0) | (d > 3 * frame.res)] = np.nan                          # gaps (car off) are not integrated across
    return d


def _series_for(name, frame, profile):
    """Series for a concept that is either measured (already in the frame) or derivable. -> (values, quality, assumptions) or None."""
    v = _col(frame, name)
    if v is not None: return v, "measured", []
    res = profile.resolve(name)
    if res.status in ("derived", "estimated"):
        ds = compute(name, frame, profile)
        if ds is not None: return ds.values, ds.quality, ds.assumptions
    return None


# ------------------------------------------------------------------------------------------------ pointwise metrics
def compute(name, frame, profile):
    """-> DerivedSeries or None when no dependency set is available."""
    fn = _POINTWISE.get(name)
    if fn is None: return None
    n = len(frame.ts)
    if n == 0: return DerivedSeries(name, np.array([]), "", "derived")
    return fn(frame, profile)


def _motor_rpm(frame, profile):
    direct = _col(frame, "motor_rpm")
    sp = _col(frame, "speed")
    if direct is not None and (sp is None or np.isfinite(direct).all()): return DerivedSeries("motor_rpm", direct, "rpm", "measured", ("motor_rpm",))
    if sp is None: return None if direct is None else DerivedSeries("motor_rpm", direct, "rpm", "measured", ("motor_rpm",))
    ratio, p1 = profile.param("gear_ratio"); r, p2 = profile.param("wheel_radius_m")
    ass = [f"gear_ratio = {ratio:g} ({'vehicle' if p1 == 'vehicle' else 'project nominal, not read from the vehicle'})", f"wheel_radius_m = {r:g} ({'vehicle' if p2 == 'vehicle' else 'project nominal, not read from the vehicle'})"]
    q = "derived" if p1 == "vehicle" and p2 == "vehicle" else "estimated"
    calc = sp / 3.6 / (2 * math.pi * r) * 60.0 * ratio
    if direct is not None:
        use = np.isfinite(direct); calc = np.where(use, direct, calc); q = "estimated" if q == "estimated" else "derived"
        return DerivedSeries("motor_rpm", calc, "rpm", q, ("motor_rpm", "speed"), "recorded rpm, otherwise speed / (2 pi r) x gear ratio", ass if q == "estimated" else [])
    return DerivedSeries("motor_rpm", calc, "rpm", q, ("speed",), "speed / (2 pi r) x gear ratio", ass if q == "estimated" else [])


def _motor_power(frame, profile):
    direct = _col(frame, "motor_power")
    t = _col(frame, "motor_torque"); rp = _motor_rpm(frame, profile)
    if direct is not None and (t is None or rp is None or np.isfinite(direct).all()): return DerivedSeries("motor_power", direct, "kW", "measured", ("motor_power",))
    if t is None or rp is None: return None                              # NOT battery power: electrical != mechanical
    calc = t * rp.values / 9549.0
    if direct is not None:
        calc = np.where(np.isfinite(direct), direct, calc)
        return DerivedSeries("motor_power", calc, "kW", "estimated" if rp.quality == "estimated" else "derived", ("motor_power", "motor_torque", "motor_rpm"), "recorded power, otherwise torque x rpm / 9549", rp.assumptions)
    return DerivedSeries("motor_power", calc, "kW", rp.quality if rp.quality != "measured" else "derived", ("motor_torque", "motor_rpm"), "torque x rpm / 9549", rp.assumptions)


def _motor_torque(frame, profile):
    direct = _col(frame, "motor_torque")
    p = _col(frame, "motor_power"); rp = _motor_rpm(frame, profile)
    if direct is not None and (p is None or rp is None or np.isfinite(direct).all()): return DerivedSeries("motor_torque", direct, "Nm", "measured", ("motor_torque",))
    if p is None or rp is None: return None
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(rp.values) > 50, p * 9549.0 / rp.values, np.nan)                  # torque is undefined at (near) standstill
    if direct is not None:
        t = np.where(np.isfinite(direct), direct, t)
        return DerivedSeries("motor_torque", t, "Nm", "estimated" if rp.quality == "estimated" else "derived", ("motor_torque", "motor_power", "motor_rpm"), "recorded torque, otherwise power x 9549 / rpm", rp.assumptions)
    return DerivedSeries("motor_torque", t, "Nm", rp.quality if rp.quality != "measured" else "derived", ("motor_power", "motor_rpm"), "power x 9549 / rpm", rp.assumptions)


def _horsepower(frame, profile):
    mp = _motor_power(frame, profile)
    return None if mp is None else DerivedSeries("horsepower", mp.values * KW_TO_HP, "hp", mp.quality, mp.via, "kW x 1.341", mp.assumptions)


def _accel(frame, profile):
    sp = _col(frame, "speed")
    if sp is None: return None
    d = _dt(frame); a = np.full(len(sp), np.nan)
    a[1:] = np.diff(sp) / 3.6 / d[1:]
    return DerivedSeries("acceleration", a, "m/s2", "derived", ("speed",), "d(speed)/dt")


def _long_acc(frame, profile):
    direct = _col(frame, "longitudinal_acceleration")
    a = _accel(frame, profile)
    if direct is not None and (a is None or np.isfinite(direct).all()): return DerivedSeries("longitudinal_acceleration", direct, "m/s2", "measured", ("longitudinal_acceleration",))
    if a is None: return None
    if direct is not None: return DerivedSeries("longitudinal_acceleration", np.where(np.isfinite(direct), direct, a.values), "m/s2", "derived", ("longitudinal_acceleration", "speed"), "recorded acceleration, otherwise d(speed)/dt")
    return DerivedSeries("longitudinal_acceleration", a.values, "m/s2", "derived", ("speed",), "d(speed)/dt")


def _accel_magnitude(frame, profile):
    a = _long_acc(frame, profile)
    return None if a is None else DerivedSeries("acceleration_magnitude", np.abs(a.values), "m/s2", a.quality, a.via, "abs(a_long)", a.assumptions)


def _braking(frame, profile):
    a = _long_acc(frame, profile)
    return None if a is None else DerivedSeries("braking_intensity", np.maximum(0.0, -a.values) / G, "g", "derived", a.via, "max(0, -a_long) / g")


def _turn_radius(frame, profile):
    sp = _col(frame, "speed"); n = len(frame.ts); values = np.full(n, np.nan); via = []; assumptions = []
    sa = _col(frame, "steering_angle"); wb, prov = profile.param("wheelbase_m"); sr, _ = profile.param("steering_ratio")
    with np.errstate(divide="ignore", invalid="ignore"):
        if sa is not None and wb is not None:
            delta = np.radians(sa / (sr or 1.0)); r = wb / np.tan(delta); r[np.abs(np.degrees(delta)) < 0.3] = np.nan                    # driving straight: no finite radius
            ok = np.isfinite(r); values[ok] = np.abs(r[ok]); via.append("steering_angle"); assumptions.append(f"steering ratio = {sr:g}" if sr else "road-wheel angle = steering-wheel angle (no steering ratio recorded)")
        if sp is not None and _col(frame, "yaw_rate") is not None:
            yr = np.radians(_col(frame, "yaw_rate")); r = (sp / 3.6) / yr; r[(np.abs(yr) < 0.005) | (sp < 3)] = np.nan
            ok = ~np.isfinite(values) & np.isfinite(r); values[ok] = np.abs(r[ok]); via += ["speed", "yaw_rate"]
        if sp is not None and _col(frame, "lateral_acceleration") is not None:
            al = np.abs(_col(frame, "lateral_acceleration")); r = (sp / 3.6) ** 2 / al; r[(al < 0.2) | (sp < 3)] = np.nan
            ok = ~np.isfinite(values) & np.isfinite(r); values[ok] = r[ok]; via += ["speed", "lateral_acceleration"]
    if not via: return None
    return DerivedSeries("turn_radius", values, "m", "derived", tuple(dict.fromkeys(via)), "direct-first: R=L/tan(delta), then v/yaw rate, then v^2/a_lat", assumptions)


def _slip(frame, profile):
    sp = _col(frame, "speed"); ws = [_col(frame, f"wheel_speed_{w}") for w in ("fl", "fr", "rl", "rr")]
    if sp is None or any(w is None for w in ws): return None
    with np.errstate(divide="ignore", invalid="ignore"):
        slip = np.where(sp > 3, (np.mean(ws, axis=0) - sp) / sp, np.nan)
    return DerivedSeries("wheel_slip", slip, "ratio", "derived", ("speed", "wheel_speed_*"), "(mean wheel speed - vehicle speed) / vehicle speed")


def _thermal_rate(frame, profile):
    bt = _col(frame, "battery_temperature")
    if bt is None: return None
    d = _dt(frame); r = np.full(len(bt), np.nan); r[1:] = np.diff(bt) / (d[1:] / 60.0)
    return DerivedSeries("thermal_rise_rate", r, "C/min", "derived", ("battery_temperature",), "d(temperature)/dt")


def _batt_eff(frame, profile):
    mp = _motor_power(frame, profile); bp = _col(frame, "battery_power")
    if mp is None or bp is None: return None
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(np.abs(bp) > 5.0, mp.values / bp, np.nan)
    return DerivedSeries("battery_efficiency", e, "ratio", mp.quality, ("motor_power", "battery_power"), "motor power / battery power", mp.assumptions)


def _battery_power(frame, profile):
    direct = _col(frame, "battery_power")
    v, i = _col(frame, "battery_voltage"), _col(frame, "battery_current")
    if direct is not None and (v is None or i is None or np.isfinite(direct).all()): return DerivedSeries("battery_power", direct, "kW", "measured", ("battery_power",))
    if v is None or i is None: return None
    calc = v * i / 1000.0
    if direct is not None:
        return DerivedSeries("battery_power", np.where(np.isfinite(direct), direct, calc), "kW", "derived", ("battery_power", "battery_voltage", "battery_current"), "recorded power, otherwise V x I / 1000")
    return DerivedSeries("battery_power", calc, "kW", "derived", ("battery_voltage", "battery_current"), "V x I / 1000")


_POINTWISE = {"motor_rpm": _motor_rpm, "motor_power": _motor_power, "motor_torque": _motor_torque, "horsepower": _horsepower, "acceleration": _accel, "acceleration_magnitude": _accel_magnitude, "longitudinal_acceleration": _long_acc, "braking_intensity": _braking,
              "turn_radius": _turn_radius, "wheel_slip": _slip, "thermal_rise_rate": _thermal_rate, "battery_efficiency": _batt_eff, "battery_power": _battery_power}
POINTWISE = frozenset(_POINTWISE)


# ------------------------------------------------------------------------------------------------ accumulated metrics (integrals over a set of samples)
def _driving(frame):
    mode = _col(frame, "drive_mode")
    speed = _col(frame, "speed"); n = len(frame.ts); moving = np.zeros(n, bool); known = np.zeros(n, bool); inferred = np.zeros(n, bool)
    if mode is not None:
        valid_mode = np.isfinite(mode) & np.isin(mode, (0, 1, 2, 3)); moving[valid_mode] = np.isin(mode[valid_mode], (1, 2)); known |= valid_mode
    if speed is not None:
        use = ~known & np.isfinite(speed); moving[use] = speed[use] > 1.0; known[use] = True; inferred[use] = True
    return moving, known, inferred


def prepare(frame, profile):
    """The whole-frame arrays accumulate() reads (sample spacing, speed, battery power, driving state, regeneration power), computed once: a window of many days is split into days without redoing them for every day."""
    bp = _battery_power(frame, profile); drv, known, inferred = _driving(frame)
    return (_dt(frame), _col(frame, "speed"), None if bp is None else bp.values, drv, known, inferred, _col(frame, "regen_power"))


def accumulate(name, frame, profile, mask=None, pre=None, part=None):
    """-> (value, unit, quality, via, assumptions, sample_count) for energy_used / distance_driven / regen_energy / wh_per_km over the (masked) samples of a frame, or None.
    `pre` = prepare(frame, profile) and `part` = a slice of the frame (one day of it): the same arithmetic over those samples only, where `mask` is still the mask of the whole frame."""
    n = len(frame.ts)
    if n == 0: return None
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    arr = pre if pre is not None else prepare(frame, profile)
    if part is not None: arr = tuple(None if a is None else a[part] for a in arr); m = m[part]
    return _accumulate(name, *arr, m)


def _accumulate(name, dt, sp, bp, drv, state_known, state_inferred, rp, m):
    interval_ok = np.isfinite(dt) & (dt > 0)
    if name == "distance_driven" and sp is not None:
        sel = m & interval_ok & np.isfinite(sp); return float(np.sum(sp[sel] / 3.6 * dt[sel]) / 1000.0), "km", "derived", ("speed",), [], int(sel.sum())
    if name == "energy_used" and bp is not None:
        sel = m & interval_ok & state_known & drv & np.isfinite(bp)
        if not np.any(sel) and not np.any(m & interval_ok & state_known): return None
        ass = [f"moving state inferred from speed for {int((m & interval_ok & state_inferred).sum())} eligible sample(s)"] if np.any(m & interval_ok & state_inferred) else []
        unknown = int((m & interval_ok & ~state_known & np.isfinite(bp)).sum())
        if unknown: ass.append(f"excluded {unknown} sample(s) because moving/parked state was unknown")
        return float(np.sum(bp[sel] * dt[sel]) / 3600.0), "kWh", "derived", ("battery_power",), ass, int(sel.sum())             # net: regeneration reduces it
    if name == "regen_energy":
        if rp is not None:
            sel = m & interval_ok & np.isfinite(rp); return float(np.sum(rp[sel] * dt[sel]) / 3600.0), "kWh", "derived", ("regen_power",), [], int(sel.sum())
        if bp is not None and sp is not None:
            sel = m & interval_ok & state_known & drv & (sp > 1.0) & (bp < 0) & np.isfinite(bp); return float(np.sum(-bp[sel] * dt[sel]) / 3600.0), "kWh", "derived", ("battery_power", "speed"), [], int(sel.sum())
    if name == "wh_per_km":
        eligible = m & interval_ok & state_known & drv
        e = _accumulate("energy_used", dt, sp, bp, drv, state_known, state_inferred, rp, eligible); d = _accumulate("distance_driven", dt, sp, bp, drv, state_known, state_inferred, rp, eligible)
        if e and d and d[0] > 0.2: return e[0] * 1000.0 / d[0], "Wh/km", "derived", ("battery_power", "speed"), [], min(e[5], d[5])
    return None


ACCUMULATED = frozenset({"energy_used", "distance_driven", "regen_energy", "wh_per_km"})

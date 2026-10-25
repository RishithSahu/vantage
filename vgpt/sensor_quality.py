"""Sensor-quality engine: is the data a number was computed from trustworthy?  Deterministic checks on a Frame, reported as flags that travel with the result (evidence.quality) so the answer can say
"according to the vehicle speed signal; GPS briefly reported an invalid zero" instead of silently picking whichever value arrived last.

Checks: freshness (how old is the newest sample), dropout (gaps), range violation (outside the registry's physical range), stuck value (flat while the vehicle is moving), spikes (isolated jumps),
timestamp problems (out of order / duplicates), and cross-sensor disagreement (vehicle speed vs GPS speed vs wheel speeds).
No check changes a value; they only describe it. Levels: high (no flags) / medium (minor flags) / low (a flag that makes the number unreliable)."""
from dataclasses import dataclass

import numpy as np

from . import signal_registry as R


@dataclass
class Flag:
    code: str                  # FRESHNESS | DROPOUT | RANGE | STUCK | SPIKE | TIMESTAMP | DISAGREE | FEW_SAMPLES
    severity: str              # minor | major
    signal: str
    detail: str

    def to_dict(self): return {"code": self.code, "severity": self.severity, "signal": self.signal, "detail": self.detail}


def freshness(last_ts, now, res=1, signal=""):
    age = now - last_ts
    limit = max(10 * res, 30)
    if age > limit: return [Flag("FRESHNESS", "major" if age > 20 * limit else "minor", signal, f"the newest sample is {age:.0f} s old")]
    return []


def timestamps(ts):
    ts = np.asarray(ts)
    if len(ts) < 2: return []
    d = np.diff(ts); out = []
    if (d < 0).any(): out.append(Flag("TIMESTAMP", "major", "", f"{int((d < 0).sum())} samples are out of time order"))
    if (d == 0).any(): out.append(Flag("TIMESTAMP", "minor", "", f"{int((d == 0).sum())} duplicate timestamps"))
    return out


def dropout(ts, res, signal=""):
    """Short gaps INSIDE activity (5 sample periods .. 15 min). Long gaps are the vehicle being switched off - normal, not a fault."""
    ts = np.asarray(ts)
    if len(ts) < 30: return []                                       # too few samples to say anything about gaps
    gaps = np.diff(ts); big = gaps[(gaps > 5 * res) & (gaps <= max(900, 6 * res))]
    if len(big) == 0: return []
    lost = float(np.sum(big - res)); active = float(np.sum(gaps[gaps <= max(900, 6 * res)])) or 1.0
    frac = lost / active
    return [Flag("DROPOUT", "major" if frac > 0.2 else "minor", signal, f"{len(big)} short gap(s) inside driving or charging, {lost / 60:.0f} min lost ({100 * frac:.0f}% of the active time)")] if frac > 0.01 else []


def column_dropout(values, res, signal="", min_seconds=30):
    """One sensor missing (NULL) for a stretch while the others keep reporting."""
    v = np.asarray(values, dtype=float); nan = np.isnan(v)
    if not nan.any() or nan.all(): return []
    edges = np.diff(np.concatenate([[0], nan.astype(int), [0]])); starts = np.where(edges == 1)[0]; ends = np.where(edges == -1)[0]
    lens = (ends - starts) * res; runs = lens[lens >= min_seconds]
    if len(runs) == 0: return []
    frac = float(nan.sum()) / len(v)
    return [Flag("DROPOUT", "major" if frac > 0.1 else "minor", signal, f"{signal.replace('_', ' ')} was missing for {len(runs)} stretch(es), {runs.sum() / 60:.0f} min in total")]


def range_violation(values, concept):
    c = R.CONCEPTS.get(concept); v = np.asarray(values, dtype=float); v = v[~np.isnan(v)]
    if c is None or c.lo is None or len(v) == 0: return []
    bad = int(((v < c.lo) | (v > c.hi)).sum())
    return [Flag("RANGE", "major" if bad / len(v) > 0.01 else "minor", concept, f"{bad} of {len(v)} samples are outside the plausible range {c.lo:g}..{c.hi:g} {c.unit}")] if bad else []


FAST_SIGNALS = {"speed", "battery_power", "battery_voltage", "battery_current", "steering_angle", "yaw_rate", "accelerator_position", "motor_rpm", "motor_torque", "motor_power", "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr", "lateral_acceleration", "longitudinal_acceleration"}


def bias(values, yaw, speed, signal="steering_angle", limit=1.5):
    """A steering angle that is not near zero while the vehicle drives straight (yaw rate ~ 0, speed > 30 km/h) has an offset."""
    if yaw is None or speed is None: return []
    v = np.asarray(values, dtype=float); y = np.asarray(yaw, dtype=float); sp = np.asarray(speed, dtype=float)
    ok = (np.abs(y) < 1.0) & (sp > 30) & ~np.isnan(v)
    if ok.sum() < 60: return []
    m = float(np.median(v[ok]))
    return [Flag("BIAS", "major", signal, f"{signal.replace('_', ' ')} reads {m:+.1f} deg while the vehicle drives straight (it should be about 0)")] if abs(m) > limit else []


def stuck(values, moving=None, min_run=30, signal=""):
    """A value that does not change at all for a long run while the vehicle is active. Only for signals that must move when driving (slow ones - temperatures, state of charge, flow - are legitimately flat)."""
    if signal and signal not in FAST_SIGNALS: return []
    v = np.asarray(values, dtype=float)
    if len(v) < min_run: return []
    same = np.concatenate([[False], v[1:] == v[:-1]]); run = best = 0; val = None
    for i, s in enumerate(same):
        if s and (moving is None or moving[i]) and not (moving is None and v[i] == 0):                  # a flat zero without a speed reference is just a signal at rest
            run += 1
            if run > best: best, val = run, v[i]
        else: run = 0
    if best >= min_run and np.nanstd(v) > 0: return [Flag("STUCK", "major", signal, f"the value did not change for {best} consecutive samples while the vehicle was active")]
    return []


def spikes(values, signal="", k=8.0):
    v = np.asarray(values, dtype=float)
    if len(v) < 20: return []
    ok = ~np.isnan(v)
    if ok.sum() < 20: return []
    med = np.nanmedian(np.abs(np.diff(v[ok]))) or 1e-9; d = np.abs(np.diff(v[ok])); idx = np.where(d > k * max(med, 1e-6) * 10)[0]
    sd3 = 6 * np.nanstd(v[ok]) / 3                                                                                  # loop-invariant: computed once, not once per candidate
    iso = [i for i in idx if i + 2 < len(d) and d[i + 1] > 0.8 * d[i] and d[i] > sd3]
    return [Flag("SPIKE", "minor", signal, f"{len(iso)} isolated jump(s) that reverse immediately")] if len(iso) >= 1 and len(iso) / max(len(v), 1) < 0.02 else []


def cross_check_speed(vehicle, gps=None, wheels=None):
    """Compare vehicle speed with GPS speed and (if present) the mean wheel speed over the SAME samples. -> (consensus_value_kmh, flags). The consensus is the vehicle-speed signal unless it is the odd one out."""
    v = np.asarray(vehicle, dtype=float); flags = []
    if len(v) == 0: return float("nan"), flags
    ref = v[-1]
    if gps is not None and len(gps) == len(v):
        g = np.asarray(gps, dtype=float); n = min(len(v), 30)
        d = np.abs(v[-n:] - g[-n:])
        if np.isnan(d).all(): return ref, flags
        dv = np.nanmedian(d)
        if dv > max(8.0, 0.15 * max(ref, 1.0)):
            kind = "reported an invalid zero" if np.nanmedian(g[-n:]) < 1.0 and ref > 10 else f"disagreed with the vehicle speed by about {dv:.0f} km/h"
            flags.append(Flag("DISAGREE", "major", "gps_speed", f"GPS speed {kind}"))
    if wheels:
        w = np.nanmean(np.vstack([np.asarray(x, dtype=float) for x in wheels]), axis=0)
        n = min(len(v), 30); dw = np.nanmedian(np.abs(v[-n:] - w[-n:]))
        if dw > max(6.0, 0.1 * max(ref, 1.0)): flags.append(Flag("DISAGREE", "major", "wheel_speed", f"the mean wheel speed differs from the vehicle speed by about {dw:.0f} km/h"))
    return float(ref), flags


def assess_frame(frame, concepts, now=None, min_samples=5):
    """-> (level, [Flag]) for the columns a result used."""
    flags = []
    if len(frame) < min_samples: flags.append(Flag("FEW_SAMPLES", "major" if len(frame) == 0 else "minor", "", f"only {len(frame)} sample(s)"))
    if len(frame) >= 3: flags += timestamps(frame.ts) + dropout(frame.ts, frame.res)
    sp = frame.column("speed"); moving = (sp > 3) if sp is not None else None
    for c in concepts:
        v = frame.column(c)
        if v is None: continue
        flags += range_violation(v, c) + column_dropout(v, frame.res, c)
        flags += stuck(v, moving, min_run=max(20, int(180 / frame.res)), signal=c)
        if c == "steering_angle": flags += bias(v, frame.column("yaw_rate"), sp, c)
        if c in FAST_SIGNALS or c in ("battery_temperature", "motor_temperature"): flags += spikes(v, c)
    if now is not None and len(frame): flags += freshness(int(frame.ts[-1]), now, frame.res)
    level = "high"
    if any(f.severity == "minor" for f in flags): level = "medium"
    if any(f.severity == "major" for f in flags): level = "low"
    return level, flags

"""Numerically safe longitudinal-motion calculations.

This module deliberately contains no language handling.  It consumes timestamped
measurements and returns typed numerical evidence, so every front end gets the
same gap, validity and interpolation behaviour.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandRun:
    start: float
    end: float
    seconds: float
    acceleration: float


def interval_acceleration(ts, speed_kmh, valid=None, max_gap_s=5.0):
    """Return (signed, magnitude, dt) for valid adjacent samples.

    A missing/off sample or a time gap breaks the series.  Values are not
    resampled and a missing value is never replaced by zero.
    """
    t = np.asarray(ts, dtype=float); v = np.asarray(speed_kmh, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    if valid is not None: ok &= np.asarray(valid, dtype=bool)
    if len(t) < 2: return np.array([]), np.array([]), np.array([])
    dt = np.diff(t); pair = ok[:-1] & ok[1:] & (dt > 0) & (dt <= float(max_gap_s))
    a = np.full(len(dt), np.nan, dtype=float)
    a[pair] = np.diff(v)[pair] / 3.6 / dt[pair]
    good = np.isfinite(a)
    return a[good], np.abs(a[good]), dt[good]


def band_runs(ts, speed_kmh, low_kmh, high_kmh, valid=None, max_gap_s=5.0,
              min_accel=0.03, max_accel=8.0):
    """Find valid upward low->high crossings using linear interpolation.

    Each run starts at the interpolated low-threshold crossing and ends at the
    interpolated high-threshold crossing.  Duplicate timestamps, invalid/off
    samples and gaps terminate an unfinished run.  Results outside configurable
    physical bounds are excluded rather than clipped.
    """
    lo, hi = sorted((float(low_kmh), float(high_kmh)))
    if not np.isfinite(lo + hi) or hi <= lo: return []
    t = np.asarray(ts, dtype=float); v = np.asarray(speed_kmh, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    if valid is not None: ok &= np.asarray(valid, dtype=bool)
    order = np.argsort(t, kind="stable"); t, v, ok = t[order], v[order], ok[order]
    out, began = [], None
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if not (ok[i - 1] and ok[i]) or dt <= 0 or dt > float(max_gap_s):
            began = None; continue
        a, b = v[i - 1], v[i]
        if began is not None and b < lo:
            began = None
        if b > a:
            if began is None and a <= lo <= b:
                began = t[i - 1] + (lo - a) * dt / (b - a)
            if began is not None and a <= hi <= b:
                ended = t[i - 1] + (hi - a) * dt / (b - a)
                secs = ended - began
                if secs > 0:
                    acc = (hi - lo) / 3.6 / secs
                    if min_accel <= acc <= max_accel:
                        out.append(BandRun(float(began), float(ended), float(secs), float(acc)))
                began = None
        elif b <= lo:
            began = None
    return out


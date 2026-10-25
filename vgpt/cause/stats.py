"""Small robust statistics for daily series: change points, context adjustment, effect sizes.  numpy only; every function tolerates gaps (NaN) and short series."""
import math

import numpy as np

_RNG = np.random.default_rng(12345)


def valid(x): return x[np.isfinite(x)]


def robust_sd(x, floor=0.0):
    x = valid(np.asarray(x, float))
    if len(x) < 4: return max(float(np.std(x)) if len(x) else 0.0, floor)
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826; sd = float(np.std(x))
    return max(mad if mad > 0 else sd, 0.25 * sd, floor)


def ols(X, y):
    """-> coefficients (with intercept first) fitted on the finite rows, and the fitted function."""
    X = np.asarray(X, float); y = np.asarray(y, float); ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    A = np.column_stack([np.ones(int(ok.sum())), X[ok]])
    if A.shape[0] < A.shape[1] + 2: return None
    lam = 1e-6 * np.eye(A.shape[1]); lam[0, 0] = 0
    beta = np.linalg.solve(A.T @ A + lam, A.T @ y[ok]); return beta


def predict(beta, X):
    X = np.asarray(X, float); return beta[0] + X @ beta[1:]


def adjust(y, X, fit_mask):
    """Residual of y after removing what the conditions X explain, the relation being fitted ONLY on `fit_mask` rows (the period before the suspected change), so a real change is not absorbed by the fit."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    beta = ols(X[fit_mask], y[fit_mask])
    if beta is None: return y - np.nanmedian(y[fit_mask]) if np.isfinite(y[fit_mask]).any() else y, None
    return y - predict(beta, X), beta


def best_step(y, min_seg=3, B=300, lo=0):
    """The one day after which the level of the series changed the most (two-level fit), with a permutation p-value of the largest statistic over every possible split.
    -> None, or dict(k, before, after, effect (in robust SDs of the days before), t, p, shape ('step' | 'ramp'), transient (the level came back))."""
    y = np.asarray(y, float); idx = np.where(np.isfinite(y))[0]
    if len(idx) < 2 * min_seg + 2: return None
    v = y[idx]; n = len(v); ks = list(range(max(min_seg, lo), n - min_seg + 1))
    if not ks: return None

    def stats(arr):
        c1 = np.concatenate([[0.0], np.cumsum(arr)]); c2 = np.concatenate([[0.0], np.cumsum(arr * arr)]); k = np.asarray(ks); na, nb = k.astype(float), n - k.astype(float)
        sa, sb = c1[k], c1[n] - c1[k]; qa, qb = c2[k], c2[n] - c2[k]; ma, mb = sa / na, sb / nb
        va = np.maximum((qa - na * ma * ma) / np.maximum(na - 1, 1), 0) / na; vb = np.maximum((qb - nb * mb * mb) / np.maximum(nb - 1, 1), 0) / nb
        s = np.sqrt(va + vb); return np.where(s > 0, np.abs(ma - mb) / np.where(s > 0, s, 1), 0.0), ma, mb
    t, ma, mb = stats(v); j = int(np.argmax(t)); obs = float(t[j]); k = ks[j]
    P = np.empty((B, n)); 
    for i in range(B): P[i] = _RNG.permutation(v)                                                   # the same random stream, in the same order, as the one-by-one loop it replaces
    kk = np.asarray(ks); na, nb = kk.astype(float), n - kk.astype(float)
    c1 = np.concatenate([np.zeros((B, 1)), np.cumsum(P, axis=1)], axis=1); c2 = np.concatenate([np.zeros((B, 1)), np.cumsum(P * P, axis=1)], axis=1)
    sa, sb = c1[:, kk], c1[:, [n]] - c1[:, kk]; qa, qb = c2[:, kk], c2[:, [n]] - c2[:, kk]; mA, mB = sa / na, sb / nb
    vA = np.maximum((qa - na * mA * mA) / np.maximum(na - 1, 1), 0) / na; vB = np.maximum((qb - nb * mB * mB) / np.maximum(nb - 1, 1), 0) / nb
    sS = np.sqrt(vA + vB); tt = np.where(sS > 0, np.abs(mA - mB) / np.where(sS > 0, sS, 1), 0.0)
    ge = int(np.sum(tt.max(axis=1) >= obs - 1e-12))
    p = (1 + ge) / (1 + B); a, b = v[:k], v[k:]; sd = robust_sd(a, floor=1e-9); eff = (float(np.median(b)) - float(np.median(a))) / sd
    xs = np.arange(len(b)); lin = np.polyfit(xs, b, 1) if len(b) >= 3 else (0.0, float(np.mean(b)))
    sse_const = float(np.sum((b - np.mean(b)) ** 2)); sse_lin = float(np.sum((b - np.polyval(lin, xs)) ** 2)); ramp = len(b) >= 5 and sse_lin < 0.6 * sse_const and abs(lin[0]) * len(b) > 1.0 * sd
    tail = v[-min(3, len(v)):]; back = abs(float(np.median(tail)) - float(np.median(a))) < 0.4 * abs(float(np.median(b)) - float(np.median(a))) if len(b) > 6 else False
    return {"k": int(idx[k]), "before": float(np.median(a)), "after": float(np.median(b)), "effect": eff, "t": obs, "p": p, "shape": "ramp" if ramp else "step", "transient": bool(back), "n_before": len(a), "n_after": len(b)}


def window_change(y, recent=14):
    """Level of the last `recent` days against the days before them (no search): effect in robust SDs of the earlier days and a Welch-type t."""
    y = np.asarray(y, float); a = valid(y[:-recent]); b = valid(y[-recent:])
    if len(a) < 5 or len(b) < 3: return None
    sd = robust_sd(a, floor=1e-9); eff = (float(np.median(b)) - float(np.median(a))) / sd
    t = (float(np.mean(b)) - float(np.mean(a))) / math.sqrt(float(np.var(a, ddof=1)) / len(a) + float(np.var(b, ddof=1)) / max(len(b), 2) + 1e-18); return {"before": float(np.median(a)), "after": float(np.median(b)), "effect": eff, "t": t}


def slope_per_day(y):
    y = np.asarray(y, float); ok = np.isfinite(y); x = np.arange(len(y))[ok]
    if ok.sum() < 5: return 0.0
    yy = y[ok]; s = [(yy[j] - yy[i]) / (x[j] - x[i]) for i in range(0, len(yy), max(1, len(yy) // 12)) for j in range(i + 1, len(yy), max(1, len(yy) // 12)) if x[j] != x[i]]
    return float(np.median(s)) if s else 0.0


def excursion(y, before, sd, k=3.0, min_days=2):
    """How far and for how long the series left its usual level (`before`, with day-to-day spread `sd`): the strongest run of >= min_days consecutive days beyond k spreads.
    -> None, or dict(peak_z, start, end, days, persistent (reaches the last day), direction (+1/-1))."""
    y = np.asarray(y, float); z = (y - before) / max(sd, 1e-12); best = None; i = 0; n = len(z)
    while i < n:
        if np.isfinite(z[i]) and abs(z[i]) > k:
            sgn = 1 if z[i] > 0 else -1; j = i
            while j + 1 < n and (not np.isfinite(z[j + 1]) or (z[j + 1] * sgn > k)): j += 1
            run = z[i:j + 1]; run = run[np.isfinite(run)]
            if len(run) >= min_days:
                peak = float(np.max(np.abs(run)))
                if best is None or peak * math.sqrt(len(run)) > best["peak_z"] * math.sqrt(best["days"]): best = {"peak_z": peak, "start": i, "end": j, "days": int(len(run)), "persistent": bool(j >= n - 2), "direction": sgn}
            i = j + 1
        else: i += 1
    return best


def excursion_p(y, before, sd, found, B=300, k=4.0, min_days=2):
    """How often shuffling the days gives an excursion at least as strong as `found` (strength = peak z x sqrt(days)).  Small = a real run of unusual days, not a few odd ones.
    All B shuffles are drawn in the original order and their runs are found in one vectorised pass (gaps are transparent to a run, so only the recorded days are shuffled and scanned)."""
    obs = found["peak_z"] * math.sqrt(found["days"]); v = np.asarray(y, float); idx = np.where(np.isfinite(v))[0]; vals = v[idx]; m = len(vals)
    if m == 0: return 1.0 / (1 + B)
    P = np.empty((B, m))
    for i in range(B): P[i] = _RNG.permutation(vals)
    Z = (P - before) / max(sd, 1e-12)
    S = (Z > k).astype(np.int8) - (Z < -k).astype(np.int8)
    S = np.concatenate([S, np.zeros((B, 1), np.int8)], axis=1).ravel(); A = np.concatenate([np.abs(Z), np.zeros((B, 1))], axis=1).ravel()
    prev = np.concatenate([[0], S[:-1]]); nxt = np.concatenate([S[1:], [0]])
    starts = np.flatnonzero((S != 0) & (S != prev)); ends = np.flatnonzero((S != 0) & (S != nxt))
    if len(starts) == 0: return (1 + 0) / (1 + B)
    lens = ends - starts + 1; keep = lens >= min_days
    if not keep.any(): return (1 + 0) / (1 + B)
    starts, ends, lens = starts[keep], ends[keep], lens[keep]
    bounds = np.empty(2 * len(starts), np.int64); bounds[0::2] = starts; bounds[1::2] = ends + 1
    peak = np.maximum.reduceat(A, bounds)[0::2]; strength = peak * np.sqrt(lens.astype(float)); rows = starts // (m + 1)
    best = np.zeros(B); np.maximum.at(best, rows, strength)
    return (1 + int(np.sum(best >= obs))) / (1 + B)


def _med(a):
    """Median of a short list of floats (same value as np.median: the mean of the two middle values for an even count)."""
    a = sorted(a); n = len(a); h = n // 2
    return a[h] if n % 2 else (a[h - 1] + a[h]) / 2.0


def despike(y, k=6.0, half=3):
    """Blank single odd days (a charger day, a sensor glitch) that stand far outside both their neighbours on either side; a real change lasts several days and is untouched."""
    y = np.asarray(y, float).copy(); out = y.copy(); n = len(y); yl = y.tolist(); fin = [math.isfinite(x) for x in yl]
    def valid_l(lo, hi): return [yl[t] for t in range(max(0, lo), min(n, hi)) if fin[t]]
    for i in range(n):
        if not fin[i]: continue
        left = valid_l(i - half - 1, i); right = valid_l(i + 1, i + 1 + half + 1)
        if len(left) >= 2 and len(right) >= 2:
            nb = left + right; med = _med(nb); spread = max(_med([abs(x - med) for x in nb]) * 1.4826, 0.02 * abs(med), 1e-9)
            if abs(yl[i] - med) > k * spread and abs(_med(left) - _med(right)) < 0.5 * abs(yl[i] - med): out[i] = np.nan
        else:                                                                                              # the first or last days have neighbours on one side only: a stricter test
            nb = left if len(left) >= 3 else right if len(right) >= 3 else None
            if nb is None: continue
            med = _med(nb); spread = max(_med([abs(x - med) for x in nb]) * 1.4826, 0.02 * abs(med), 1e-9)
            if abs(yl[i] - med) > 1.6 * k * spread: out[i] = np.nan
    return out


def noise(x, floor=0.0):
    """Day-to-day noise of the days before a suspected change: spread and lag-1 correlation of what is left after a straight line is removed (a creep that has already started must not count as noise)."""
    x = np.asarray(x, float); ok = np.isfinite(x); idx = np.arange(len(x))[ok]; v = x[ok]
    if len(v) < 6: return max(float(np.std(v)) if len(v) else 0.0, floor), 0.0
    try: line = np.polyfit(idx, v, 1); r = v - np.polyval(line, idx)
    except (np.linalg.LinAlgError, ValueError): r = v - np.median(v)
    sd = robust_sd(r, floor=floor); rho = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 8 and np.std(r) > 0 else 0.0
    return sd, (0.0 if not np.isfinite(rho) else rho)


def mann_kendall(x):
    """-> (S statistic's z, Kendall tau) for a monotone trend in x (NaN dropped)."""
    v = valid(np.asarray(x, float)); n = len(v)
    if n < 6: return 0.0, 0.0
    s_ = int(np.triu(np.sign(v[None, :] - v[:, None]), 1).sum())                                      # S statistic: sum over all pairs i < j of sign(v[j] - v[i])
    var = n * (n - 1) * (2 * n + 5) / 18.0; z = (s_ - np.sign(s_)) / math.sqrt(var) if s_ != 0 else 0.0
    return float(z), float(s_ / (n * (n - 1) / 2))


def end_ramp(y, sd, min_len=8, max_len=24, z_min=3.5):
    """A creep that has only been going on for the last days: the longest-evidence monotone run ending at the last day.  -> None or dict(start, z, total (change over the run), direction)."""
    y = np.asarray(y, float); best = None
    for L in range(min_len, min(max_len, len(y) - 5) + 1):
        seg = y[-L:]
        if np.isfinite(seg).sum() < max(6, int(0.6 * L)): continue
        z, tau = mann_kendall(seg); idx = np.arange(L)[np.isfinite(seg)]; slope = np.polyfit(idx, seg[np.isfinite(seg)], 1)[0]; total = slope * (L - 1)
        if abs(z) >= z_min and abs(total) >= 4.0 * sd and (best is None or abs(z) > abs(best["z"]) + 1e-9): best = {"start": len(y) - L, "z": float(z), "total": float(total), "direction": 1 if total > 0 else -1}
    return best

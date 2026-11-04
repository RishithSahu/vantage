"""The per-minute questions the whole history is asked ("is my car ok", daily means, the first day the coolant flow dropped, the latest abnormal reading) answered from the columnar copy (vgpt/colstore.py) instead of by SQL.

Why: each of them used to scan or group millions of rows through SQLite (the Raspberry Pi 4 hands a row to Python in 3.5 microseconds: 7 seconds for four years of minutes).  Here the same numbers come from a few vectorised passes over arrays, in tens of milliseconds.

Every function returns exactly what the SQL it replaces returned (same rows, same order, same arithmetic: the sums are over whole numbers, which are exact in float64 in any order) or None when the copy cannot be used (short window, not built yet, not a plain column): the caller then runs its SQL as before.
tests/test_fastcols.py compares each one with its SQL on real databases, also with rows appended after the copy was made.  VGPT_FASTCOLS=0 turns them all off."""
import functools, os, re
import numpy as np
from . import analytics as A, colstore

_PLAIN = re.compile(r"^[a-z0-9_]+$")
_DIFF = re.compile(r"^\(([a-z0-9_]+)-([a-z0-9_]+)\)$")                                  # "(batt-amb)": a temperature as the excess over the outside temperature


def _safe(f):
    """A fast path never breaks a question: any error in it means 'use the SQL' (None)."""
    @functools.wraps(f)
    def g(*a, **k):
        try: return f(*a, **k)
        except Exception: return None
    return g


def on(): return os.environ.get("VGPT_FASTCOLS", "1") == "1"


def _get(conn, cols, lo, hi, table="sig_1m"):
    if not on(): return None
    cs = colstore.of(conn)
    return cs.columns(conn, table, list(cols), lo, hi) if cs else None


def _value(expr):
    """-> (names of the columns to read, function from their arrays to the expression), or None when `expr` is neither a plain column nor (a-b)."""
    m = _DIFF.match(expr)
    if m: return [m[1], m[2]], lambda x: x[0] - x[1]
    if _PLAIN.match(expr): return [expr], lambda x: x[0]
    return None


def _num(x): return int(x) if x == int(x) else float(x)                                  # what SQLite hands back for the stored number


def _in(mode, modes):
    m = mode == modes[0]
    for x in modes[1:]: m |= mode == x
    return m


@_safe
def daily(conn, col, lo, hi, modes, minv, scale, agg, minn):
    """-> [(day_start, mean / scale, rows)] as qa_b.MixB.daily (mean of `col` per local day over the minutes whose mode is in `modes` [and col > minv * scale]; days with fewer than `minn` rows or no value are left out)."""
    if agg != "AVG" or not _PLAIN.match(col) or not modes: return None
    a = _get(conn, ["mode", col], lo, hi)
    if a is None: return None
    ts, mode, v = a
    m = _in(mode, modes)
    if minv is not None: m &= v > int(minv * scale)
    t = ts[m]; w = v[m]
    if not len(t): return []
    if t[0] < 0: return None
    day = (t.astype(np.int64) + A.TZ) // 86400; k = day - day[0]; nn = ~np.isnan(w)
    cnt = np.bincount(k); got = np.bincount(k, weights=nn); tot = np.bincount(k, weights=np.where(nn, w, 0.0))
    out = []
    for i in np.flatnonzero((cnt >= minn) & (got > 0)).tolist(): out.append((int(day[0] + i) * 86400 - A.TZ, float(tot[i] / got[i]) / scale, int(cnt[i])))
    return out


@_safe
def flow_days(conn, lo, hi, minn):
    """-> [(day_start, median coolant flow L/min, rows)] as qa_core.flow_days."""
    a = _get(conn, ["mode", "flow"], lo, hi)
    if a is None: return None
    ts, mode, f = a
    m = (f > 30) & _in(mode, (1, 2, 3)); t = ts[m]; f = f[m]
    if not len(t): return []
    if t[0] < 0: return None
    day = (t.astype(np.int64) + A.TZ) // 86400; cut = np.flatnonzero(np.diff(day)) + 1; starts = np.concatenate([[0], cut]).tolist(); ends = np.concatenate([cut, [len(day)]]).tolist(); out = []
    for s, e in zip(starts, ends):
        if e - s >= minn: out.append((int(day[s]) * 86400 - A.TZ, float(np.sort(f[s:e])[(e - s) // 2]) / 10.0, e - s))
    return out


@_safe
def baseline_values(conn, col, lo, hi, modes, scale, minv):
    """-> the sorted values (/ scale) that qa_core.baseline_stats takes its statistics over, as a list."""
    got = _value(col)
    if got is None or not modes: return None
    names, fn = got; a = _get(conn, ["mode"] + names, lo, hi)
    if a is None: return None
    mode = a[1]; v = fn(a[2:]); m = _in(mode, modes) & ~np.isnan(v)
    if minv is not None: m &= v > int(minv * scale)
    return (np.sort(v[m]) / scale).tolist()


@_safe
def latest_above(conn, col, lo, hi_incl, lim):
    """-> (ts, raw value) of the newest minute with ts in [lo, hi_incl] and `col` > lim, False when there is none, None when the copy cannot be used."""
    got = _value(col)
    if got is None: return None
    names, fn = got; a = _get(conn, names, lo, hi_incl + 1)
    if a is None: return None
    v = fn(a[1:]); idx = np.flatnonzero(v > lim)
    if not len(idx): return False
    i = int(idx[-1]); return (int(a[0][i]), _num(v[i]))


@_safe
def recent_below(conn, col, lo, hi_incl, lim, floor, limit):
    """-> [(ts, value)] of the newest `limit` minutes in [lo, hi_incl] with floor < col < lim, newest first; None when the copy cannot be used."""
    if not _PLAIN.match(col): return None
    a = _get(conn, [col], lo, hi_incl + 1)
    if a is None: return None
    v = a[1]; idx = np.flatnonzero((v < lim) & (v > floor))[::-1][:limit]
    return [(int(a[0][i]), _num(v[i])) for i in idx.tolist()]


@_safe
def count_per_day(conn, table, col, lo, hi, tz):
    """-> {local day number: rows with `col` not NULL} for ts in [lo, hi), as `SELECT (ts + tz) / 86400, COUNT(*) ... WHERE col IS NOT NULL GROUP BY 1` (days without such a row are left out)."""
    if not _PLAIN.match(col) or not _PLAIN.match(table): return None
    a = _get(conn, [col], lo, hi, table)
    if a is None: return None
    ts = a[0][~np.isnan(a[1])]
    if not len(ts): return {}
    if ts[0] < 0: return None
    day = (ts.astype(np.int64) + tz) // 86400; k = np.bincount(day - day[0])
    return {int(day[0]) + i: int(k[i]) for i in np.flatnonzero(k).tolist()}


@_safe
def series(conn, table, col, scale, lo, hi):
    """-> {int(ts): value / scale} for the rows of the per-minute `table` in [lo, hi) with `col` not NULL (what pairs.minute_series reads), or None when the copy cannot be used."""
    if not _PLAIN.match(col) or not _PLAIN.match(table): return None
    a = _get(conn, [col], lo, hi, table)
    if a is None: return None
    keep = ~np.isnan(a[1])
    return dict(zip(a[0][keep].astype(np.int64).tolist(), (a[1][keep] / scale).tolist()))


@_safe
def min_count_last(conn, table, col, lo, hi):
    """-> (smallest value, number of values, newest value) of `col` over the per-minute rows in [lo, hi) that are not NULL, as stored numbers; (None, 0, None) when there are none; None when the copy cannot be used."""
    if not _PLAIN.match(col) or not _PLAIN.match(table): return None
    a = _get(conn, [col], lo, hi, table)
    if a is None: return None
    v = a[1][~np.isnan(a[1])]
    return (None, 0, None) if not len(v) else (_num(v.min()), int(len(v)), _num(v[-1]))


@_safe
def rows_per_day(conn, table, lo, hi, tz, step=86400):
    """-> {local day number: rows} for ts in [lo, hi), as `SELECT CAST((ts + tz) / step AS INTEGER), COUNT(*) ... GROUP BY 1`."""
    if not _PLAIN.match(table): return None
    a = _get(conn, [], lo, hi, table)
    if a is None: return None
    ts = a[0]
    if not len(ts): return {}
    if ts[0] < 0: return None
    day = (ts.astype(np.int64) + tz) // step; k = np.bincount(day - day[0])
    return {int(day[0]) + i: int(k[i]) for i in np.flatnonzero(k).tolist()}


@_safe
def minute_columns(conn, cols, lo, hi):
    """-> [ts, col, ...] float64 arrays of the per-minute signal rows in [lo, hi) (NULL = NaN), or None when the copy cannot be used."""
    return _get(conn, cols, lo, hi)

"""Read per-day aggregates (mean / max / min / count of a column over the minutes of each local day) from the DAILY TIER where it covers them, from the minute rows where it does not.

The tier (table agg_1d, written by vgpt.store.roll_daily) holds every finished day; a day that is only partly inside the requested time range, the day still filling and anything newer than the last
rolled day are aggregated from the minute table with the SAME statement the callers used to run over the whole range - so the numbers are identical, only the number of rows read changes
(about 100 tier rows per quantity per year instead of 1,440 minutes per day).  `masked`: only the minutes the car was driving (sig_1m.mode in 1, 2)."""
import sqlite3

DAY = 86400


def _tier_state(conn, table, masked, tz):
    try:
        st = conn.execute("SELECT through, sig_max_ts FROM agg_1d_state WHERE tbl=? AND masked=? AND tz=?", (table, int(masked), tz)).fetchone()
        if st is None: return None
        cur = conn.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]
        return (int(st[0]), int(st[1])) if cur is not None and int(cur) >= int(st[1]) else None            # the tier belongs to this data (it is only ever extended)
    except sqlite3.Error: return None


def _scan(conn, tz, table, masked, cols, lo, hi, has_mode):
    """{col: {day: (n, avg, mx, mn)}} aggregated from the minute rows with lo <= ts < hi (days with no value of a column are absent)."""
    if hi <= lo: return {}
    join = " JOIN sig_1m s ON s.ts = t.ts" if masked and table != "sig_1m" else ""
    where = ((" AND s.mode IN (1,2)" if table != "sig_1m" else " AND t.mode IN (1,2)") if masked and has_mode else "")
    sel = ", ".join(f"AVG(t.{c}), MAX(t.{c}), MIN(t.{c}), COUNT(t.{c})" for c in cols); out = {c: {} for c in cols}
    try:
        for row in conn.execute(f"SELECT CAST((t.ts + ?) / 86400 AS INTEGER) d, {sel} FROM {table} t{join} WHERE t.ts>=? AND t.ts<?{where} GROUP BY d", (tz, int(lo), int(hi))):
            for ci, c in enumerate(cols):
                av, mx, mn, n = row[1 + 4 * ci: 5 + 4 * ci]
                if n: out[c][int(row[0])] = (int(n), av, mx, mn)
    except sqlite3.Error: pass
    return out


def per_day(conn, tz, table, masked, cols, lo, hi, has_mode=True):
    """{col: {day: (n, avg, mx, mn)}} for the minute rows with lo <= ts < hi (same days, same numbers as one scan of the minute table)."""
    cols = list(cols); st = _tier_state(conn, table, masked, tz)
    if st is None: return _scan(conn, tz, table, masked, cols, lo, hi, has_mode)
    through = st[0]
    d_a = -((-(int(lo) + tz)) // DAY)                                   # first day that starts at or after lo
    d_b = min(through, (int(hi) + tz) // DAY - 1)                      # last day that ends at or before hi (and is rolled)
    if d_b < d_a: return _scan(conn, tz, table, masked, cols, lo, hi, has_mode)
    out = {c: {} for c in cols}
    try:
        q = ",".join("?" * len(cols))
        for c, d, n, av, mx, mn in conn.execute(f"SELECT col, d, n, avg, mx, mn FROM agg_1d WHERE tbl=? AND masked=? AND tz=? AND d>=? AND d<=? AND col IN ({q})", (table, int(masked), tz, d_a, d_b, *cols)):
            out[c][int(d)] = (int(n), av, mx, mn)
    except sqlite3.Error: return _scan(conn, tz, table, masked, cols, lo, hi, has_mode)
    for part in (_scan(conn, tz, table, masked, cols, lo, d_a * DAY - tz, has_mode), _scan(conn, tz, table, masked, cols, (d_b + 1) * DAY - tz, hi, has_mode)):
        for c in cols: out[c].update(part.get(c, {}))
    return out

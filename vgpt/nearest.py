"""The row of a time-keyed table closest to an instant, found with two index probes (the newest row at or before it, the oldest after it) instead of sorting the whole table by distance.

`SELECT <what> FROM <table> [WHERE <notnull> IS NOT NULL] ORDER BY ABS(ts - ?) LIMIT 1` reads every row (0.4 s for four years of minutes on a laptop, seconds on the Raspberry Pi 4); this reads two.
It gives the same row, including the same choice on a tie (SQLite keeps the earlier one).  `what` starts with ts."""


def nearest(conn, table, what, target, notnull=None):
    w = f" AND {notnull} IS NOT NULL" if notnull else ""
    before = conn.execute(f"SELECT {what} FROM {table} WHERE ts<=?{w} ORDER BY ts DESC LIMIT 1", (target,)).fetchone()
    after = conn.execute(f"SELECT {what} FROM {table} WHERE ts>?{w} ORDER BY ts LIMIT 1", (target,)).fetchone()
    if before is None: return after
    if after is None: return before
    return before if abs(before[0] - target) <= abs(after[0] - target) else after

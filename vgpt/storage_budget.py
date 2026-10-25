"""A car's history has a size limit, and it is kept: `ret.budget_gb` (default 6) is the most the database file plus its archive folder may take, and the nightly housekeeping enforces it.

    usage(conn)            -> {"database": bytes, "archive": bytes, "total": bytes, "archive_files": n, "oldest": ts}   (files on the storage, not rows)
    report(conn)           -> the same plus the size of each table (needs dbstat; else an estimate from row counts) and a projection of the size after five years at the present growth
    enforce(conn, now)     -> {"removed": n, "freed": bytes, "total": bytes, "budget": bytes}   removes the OLDEST archive files, one month of one table at a time, until the total is under the budget;
                              the database itself is never touched (the hot and warm tiers, the daily tier and the records are the answers to most questions); returns what it did so that it can be logged

What the order means: the oldest seconds go first, then the oldest minutes; the daily tier (`agg_1d`), the events, the trips and the logs are not in the archive and stay.  A car that records more signals than
the budget allows therefore loses the finest resolution of its OLDEST years before it loses anything recent, and says so (the `source_log` row and the return value)."""
import os
import sqlite3

DEFAULT_GB = 6.0


def _db_file(conn):
    for _seq, name, f in conn.execute("PRAGMA database_list").fetchall():
        if name == "main": return f or None
    return None


def _size(path):
    try: return os.path.getsize(path)
    except OSError: return 0


def budget_bytes(conn):
    try: gb = float(conn.execute("SELECT v FROM meta WHERE k='ret.budget_gb'").fetchone()[0])
    except (TypeError, ValueError, sqlite3.Error): gb = DEFAULT_GB
    return int(gb * 1e9)


def usage(conn):
    f = _db_file(conn); db = sum(_size(f + e) for e in ("", "-wal", "-shm")) if f else 0
    arch = 0; n = 0; oldest = None
    try:
        rows = conn.execute("SELECT file, lo FROM archive_idx").fetchall(); d = os.path.join(os.path.dirname(os.path.abspath(f)), "archive") if f else None
        for name, lo in rows:
            arch += _size(os.path.join(d, name)) if d else 0; n += 1; oldest = lo if oldest is None else min(oldest, lo)
    except sqlite3.Error: pass
    return {"database": db, "archive": arch, "total": db + arch, "archive_files": n, "oldest": oldest}


def report(conn, years=5):
    u = usage(conn); tables = {}
    try: tables = {n: int(s) for n, s in conn.execute("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name").fetchall()}
    except sqlite3.Error: pass
    try: lo, hi = (int(x) for x in conn.execute("SELECT MIN(ts), MAX(ts) FROM sig_1m").fetchone()); have_days = max(1.0, (hi - lo) / 86400.0)
    except (TypeError, sqlite3.Error): have_days = 1.0
    try: a_lo = conn.execute("SELECT MIN(lo) FROM archive_idx").fetchone()[0]; hist_days = max(have_days, (hi - a_lo) / 86400.0) if a_lo else have_days
    except (TypeError, sqlite3.Error, NameError): hist_days = have_days
    u.update(tables=tables, history_days=round(hist_days, 1), per_year=int(u["total"] / hist_days * 365.0), projected_total=int(u["total"] / hist_days * 365.0 * years), years=years)
    return u


def enforce(conn, now_ts=None):
    """Remove the oldest archive files until the database and the archive together fit the budget."""
    f = _db_file(conn); budget = budget_bytes(conn); u = usage(conn); out = {"removed": 0, "freed": 0, "total": u["total"], "budget": budget}
    if f is None or u["total"] <= budget: return out
    d = os.path.join(os.path.dirname(os.path.abspath(f)), "archive")
    for i, name, tbl, lo in conn.execute("SELECT id, file, tbl, lo FROM archive_idx ORDER BY (tbl LIKE '%_1m'), lo").fetchall():           # seconds before minutes, oldest first
        if out["total"] <= budget * 0.9: break
        size = _size(os.path.join(d, name))
        try: os.remove(os.path.join(d, name))
        except OSError: continue
        conn.execute("DELETE FROM archive_idx WHERE id=?", (i,)); out["removed"] += 1; out["freed"] += size; out["total"] -= size
        if out["removed"] == 1:
            try: conn.execute("INSERT INTO source_log VALUES(?,?,?,?)", (int(now_ts or 0), "storage", "budget", f"the history exceeds its {budget / 1e9:.1f} GB budget: the oldest archived data is being removed"))
            except sqlite3.Error: pass
    return out

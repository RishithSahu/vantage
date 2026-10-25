"""What the SQLite of this system can do, and what the application does when it cannot.

SQLite is built differently on every system: the copy inside a desktop Python has the FTS5 full-text module, the copy on a stripped-down embedded system (a QNX port, a minimal Yocto image) often has not.  Log lines are searched through
an FTS5 index when the system has it; when it has not, the same search runs as a plain substring match over the log table (slower on a very long log, the same answers for whole words, a few more for word fragments), and nothing
else in the application changes.  The history database is built on a laptop and may arrive with an index this SQLite cannot read: that is detected, not trusted."""
import sqlite3

_FTS5 = None


def has_fts5():
    """True when this SQLite can create an FTS5 table (probed once)."""
    global _FTS5
    if _FTS5 is None:
        try:
            c = sqlite3.connect(":memory:"); c.execute("CREATE VIRTUAL TABLE t USING fts5(x)"); c.close(); _FTS5 = True
        except sqlite3.Error: _FTS5 = False
    return _FTS5


def log_index_usable(conn):
    """True when `conn` holds the log index table and this SQLite can read it."""
    if not has_fts5(): return False
    try: return conn.execute("SELECT 1 FROM sqlite_master WHERE name='logs_fts'").fetchone() is not None
    except sqlite3.Error: return False


def rebuild_log_index(conn):
    if log_index_usable(conn): conn.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')")


def search_logs(conn, tokens, limit=5):
    """-> [(ts, src, msg)], newest first, of the log lines containing any of `tokens` (words of letters and digits)."""
    if not tokens: return []
    if log_index_usable(conn):
        return conn.execute("SELECT l.ts, l.src, l.msg FROM logs_fts f JOIN logs l ON l.rowid=f.rowid WHERE logs_fts MATCH ? ORDER BY l.ts DESC LIMIT ?", (" OR ".join(tokens), limit)).fetchall()
    where = " OR ".join("lower(msg) LIKE ? ESCAPE '\\'" for _ in tokens)
    like = [f"%{t.lower().replace(chr(92), chr(92) * 2).replace('%', chr(92) + '%').replace('_', chr(92) + '_')}%" for t in tokens]
    return conn.execute(f"SELECT ts, src, msg FROM logs WHERE {where} ORDER BY ts DESC LIMIT ?", (*like, limit)).fetchall()

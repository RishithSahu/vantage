"""The names of the tables and views of a database, read once per schema version (the list is 7 ms on the Raspberry Pi 4 with a few hundred lake views, and several modules ask for it in every question).
SQLite's schema_version changes whenever a table, view or index is created or dropped, so the list can never be stale; this is metadata, not data."""
import threading

_LOCK = threading.Lock(); _CACHE = {}


def table_names(conn):
    try:
        row = conn.execute("PRAGMA database_list").fetchone(); path = row[2] if row and row[2] else None
        ver = conn.execute("PRAGMA schema_version").fetchone()[0]
    except Exception: path = ver = None
    if path:
        with _LOCK: hit = _CACHE.get(path)
        if hit and hit[0] == ver: return set(hit[1])
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    if path:
        with _LOCK: _CACHE[path] = (ver, frozenset(names))
    return names

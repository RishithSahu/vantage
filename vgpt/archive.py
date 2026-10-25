"""The lossless archive of the seconds that expired from the database.

The database keeps the once-per-second rows (`sig_1s`, and `lk_1s` for the signals beyond the core ones) for `ret.fine_days` (7 by default) and the per-minute tier for a year.  Before the nightly housekeeping deletes the
seconds it WRITES THEM HERE: one compressed file per run and table, losslessly (every value, every second, NULLs included), next to the database in `archive/`, and one row in the table `archive_idx` that says which seconds
the file holds.  The files are kept for `ret.archive_days` (a year, the product's floor).

    write(conn, cut)        archive the rows older than `cut` that are not archived yet (called by store.compact just before it deletes them)
    coverage(conn)          (first second, last second) the archive holds, or None
    hydrate(conn, lo, hi)   make the archived seconds of [lo, hi) readable as if they had never expired, for THIS connection only: the live tables are shadowed by temporary views that add the archived rows
                            (the assistant's handle stays read-only; nothing is written to the database).  Returns {"rows", "segments", "damaged"}.
    purge(conn)             delete every archive file (the person asked for the history to be erased)

Only questions that are exact over the seconds themselves (a highest speed, how long a limit was exceeded) load it, and only for a window the planner limits to 45 days; everything else is answered from the minute tier.
Integrity: every file's SHA-256 is in the index and the index rows are chained (each hash covers the previous one), so a changed, removed or reordered file is found when it is loaded and the question says the seconds are
not available rather than answering from damaged data.  Plain standard library (array, lzma); the codec is column-wise deltas, which is what makes a day of one-second rows a few hundred kilobytes."""
import array
import hashlib
import json
import lzma
import os
import sqlite3
import struct
import time

MAGIC = b"VGA1"
TABLES = {"sig_1s": ("ts", "iiiiiiiiii"), "lk_1s": ("sid, ts", "iif")}                       # table -> (order, column kinds: i integer, f float)
MAX_HYDRATED_ROWS = 6_000_000                                                                 # more than this and the earlier windows are dropped first (a 45-day window of one table is about 3.9 M rows)
_LAST = {}


# ------------------------------------------------------------------------------------------------ codec
def _zig(v): return (v << 1) ^ (v >> 63)
def _unzig(z): return (z >> 1) ^ -(z & 1)


MAGIC2 = b"VGA2"
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def _np():
    import numpy as np
    return np


def _pack_ints(arr, mask):
    """Deltas of the column (a NULL repeats the previous value, so its delta is 0), zig-zagged and cut into byte planes of the narrowest width: the high planes of small deltas are runs of zeros that LZMA removes."""
    np = _np(); n = len(arr)
    if n == 0: return b"\x01"
    idx = np.where(mask == 0, np.arange(n), -1); np.maximum.accumulate(idx, out=idx); filled = np.where(idx >= 0, arr[np.maximum(idx, 0)], 0).astype(np.int64)
    d = np.diff(filled, prepend=np.int64(0)); z = ((d << 1) ^ (d >> 63)).astype(np.uint64); top = int(z.max()) if n else 0
    w = 1 if top < 1 << 8 else 2 if top < 1 << 16 else 4 if top < 1 << 32 else 8
    planes = z.astype({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[w]).view(np.uint8).reshape(n, w).T
    return bytes([w]) + np.ascontiguousarray(planes).tobytes()


def _unpack_ints(payload, n, mask):
    np = _np()
    if n == 0: return []
    w = payload[0]; planes = np.frombuffer(payload, np.uint8, offset=1).reshape(w, n)
    z = np.ascontiguousarray(planes.T).view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[w]).reshape(n).astype(np.int64)         # the byte planes back into one integer per row
    d = (z >> 1) ^ -(z & 1); vals = np.cumsum(d).tolist()
    return [None if m else v for v, m in zip(vals, mask.tolist())] if mask.any() else vals


def _pack_floats(arr, mask):
    """A column of REAL values that are really fixed-point numbers (the lake rounds to three decimals) is stored as integers with the scale; any other column is stored as doubles.  Exact either way: the
    integers are accepted only if dividing them by the scale gives back every double bit for bit."""
    np = _np(); n = len(arr)
    if n == 0: return b"\x00"
    for sc in SCALES:
        with np.errstate(all="ignore"): q = np.rint(arr * sc); ok = bool(np.abs(q).max() < 2.0 ** 52) and bool(np.all(q / sc == arr))
        if ok: return b"\x01" + struct.pack("<I", sc) + _pack_ints(q.astype(np.int64), mask)
    return b"\x00" + np.ascontiguousarray(arr.astype("<f8").view(np.uint8).reshape(n, 8).T).tobytes()


def _unpack_floats(payload, n, mask):
    np = _np()
    if n == 0: return []
    if payload[0] == 1:
        sc = struct.unpack_from("<I", payload, 1)[0]; ints = _unpack_ints(payload[5:], n, np.zeros(n, np.uint8)); vals = (np.array(ints, np.int64) / sc).tolist()
    else:
        vals = np.ascontiguousarray(np.frombuffer(payload, np.uint8, offset=1).reshape(8, n).T).view("<f8").reshape(n).tolist()
    return [None if m else v for v, m in zip(vals, mask.tolist())] if mask.any() else vals


def encode(rows, kinds):
    """rows: list of tuples (None allowed) -> bytes.  Columns are stored one after the other (integers as byte-plane deltas, floats as scaled integers when they are exactly that, else doubles; NULLs as a bit
    mask) and the whole is LZMA-compressed.  Lossless: decode() returns the same rows."""
    np = _np(); n = len(rows); cols = list(zip(*rows)) if rows else [()] * len(kinds); parts = [struct.pack("<4sIH", MAGIC2, n, len(kinds)), kinds.encode("ascii")]
    for k, col in zip(kinds, cols):
        mask = np.fromiter((v is None for v in col), np.uint8, n)
        if k == "i": payload = _pack_ints(np.array([0 if v is None else v for v in col], np.int64), mask)
        else: payload = _pack_floats(np.array([0.0 if v is None else v for v in col], np.float64), mask)
        mb = np.packbits(mask).tobytes() if mask.any() else b""
        parts += [struct.pack("<II", len(mb), len(payload)), mb, payload]
    return lzma.compress(b"".join(parts), preset=6, check=lzma.CHECK_CRC32)


def _encode_v1(rows, kinds):
    """The first file format (kept for the tests that read old archives)."""
    n = len(rows); parts = [struct.pack("<4sIH", MAGIC, n, len(kinds)), kinds.encode("ascii")]
    for j, k in enumerate(kinds):
        col = [r[j] for r in rows]; mask = bytearray(1 if v is None else 0 for v in col)
        if k == "i":
            prev = 0; d = array.array("q")
            for v in col:
                if v is None: d.append(0); continue
                d.append(v - prev); prev = v
            data = d.tobytes()
        else:
            data = array.array("d", [0.0 if v is None else float(v) for v in col]).tobytes()
        parts += [struct.pack("<II", len(mask), len(data)), bytes(mask), data]
    return lzma.compress(b"".join(parts), preset=6, check=lzma.CHECK_CRC32)


def decode(blob):
    raw = lzma.decompress(blob); magic, n, nc = struct.unpack_from("<4sIH", raw, 0)
    if magic not in (MAGIC, MAGIC2): raise ValueError("not an archive file")
    off = 10; kinds = raw[off:off + nc].decode("ascii"); off += nc; cols = []
    if magic == MAGIC2:
        np = _np()
        for k in kinds:
            lm, ld = struct.unpack_from("<II", raw, off); off += 8; mb = raw[off:off + lm]; off += lm; payload = raw[off:off + ld]; off += ld
            mask = np.unpackbits(np.frombuffer(mb, np.uint8))[:n] if lm else np.zeros(n, np.uint8)
            cols.append(_unpack_ints(payload, n, mask) if k == "i" else _unpack_floats(payload, n, mask))
        return list(zip(*cols)) if cols else []
    for k in kinds:
        lm, ld = struct.unpack_from("<II", raw, off); off += 8; mask = raw[off:off + lm]; off += lm; data = raw[off:off + ld]; off += ld
        if k == "i":
            d = array.array("q"); d.frombytes(data); vals = []; prev = 0
            for x, m in zip(d, mask):
                if m: vals.append(None); continue
                prev += x; vals.append(prev)
        else:
            d = array.array("d"); d.frombytes(data); vals = [None if m else x for x, m in zip(d, mask)]
        cols.append(vals)
    return list(zip(*cols)) if cols else []


# ------------------------------------------------------------------------------------------------ the files
def _db_file(conn):
    for _seq, name, f in conn.execute("PRAGMA database_list").fetchall():
        if name == "main": return f or None
    return None


def folder(conn):
    f = _db_file(conn)
    return os.path.join(os.path.dirname(os.path.abspath(f)), "archive") if f else None


def _ensure(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS archive_idx(id INTEGER PRIMARY KEY, tbl TEXT, lo INT, hi INT, rows INT, bytes INT, file TEXT, sha TEXT, chain TEXT, made INT)")


def _has_table(conn, name): return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


WINDOW_S = 6 * 3600                                                                          # one file holds at most this much time of one table (a day of a rich vehicle's lake is millions of rows)
COLS = {"sig_1s": "ts, speed, soc, power, batt, flow, amb, hvac, motor, mode", "lk_1s": "sid, ts, v"}


def _schema(conn, tbl):
    """(column names, kinds, order) of a table of the database, from its own definition: INTEGER columns are 'i', REAL columns 'f'."""
    info = conn.execute(f"PRAGMA main.table_info({tbl})").fetchall()
    names = [r[1] for r in info]; kinds = "".join("f" if any(t in (r[2] or "").upper() for t in ("REAL", "FLOA", "DOUB")) else "i" for r in info)
    pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
    return names, kinds, ", ".join(pk) if pk else names[0]


def write(conn, cut, now_ts=None, tables=None):
    """Archive the rows of every archived table older than `cut` that are not archived yet, in files of at most six hours.  Returns the number of rows written.  Safe to repeat: what the
    index holds is skipped.  `tables` defaults to the seconds tables."""
    d = folder(conn)
    if d is None: return 0
    os.makedirs(d, exist_ok=True); _ensure(conn); total = 0
    for tbl in (tables or TABLES):
        if not _has_table(conn, tbl): continue
        if tbl in TABLES: order, kinds = TABLES[tbl]; names = [c.strip() for c in COLS[tbl].split(",")]; window = WINDOW_S
        else: names, kinds, order = _schema(conn, tbl); window = WINDOW_S
        tcol = names.index("ts"); sel = ", ".join(names)
        done = conn.execute("SELECT MAX(hi) FROM archive_idx WHERE tbl=?", (tbl,)).fetchone()[0]
        cur = (done + 1) if done is not None else conn.execute(f"SELECT MIN(ts) FROM {tbl}").fetchone()[0]
        while cur is not None and cur < cut:
            end = min((cur // window + 1) * window, int(cut))
            rows = conn.execute(f"SELECT {sel} FROM {tbl} WHERE ts >= ? AND ts < ? ORDER BY {order}", (int(cur), int(end))).fetchall()
            if rows:
                first = min(r[tcol] for r in rows); last = max(r[tcol] for r in rows)
                blob = encode(rows, kinds); name = f"{tbl}_{first}_{last}.vga"; path = os.path.join(d, name); tmp = path + ".part"
                with open(tmp, "wb") as f: f.write(blob); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, path)
                sha = hashlib.sha256(blob).hexdigest(); prev = conn.execute("SELECT chain FROM archive_idx ORDER BY id DESC LIMIT 1").fetchone()
                chain = hashlib.sha256(((prev[0] if prev else "") + sha + f"{tbl}{first}{last}{len(rows)}").encode()).hexdigest()
                conn.execute("INSERT INTO archive_idx(tbl, lo, hi, rows, bytes, file, sha, chain, made) VALUES(?,?,?,?,?,?,?,?,?)", (tbl, first, last, len(rows), len(blob), name, sha, chain, int(now_ts if now_ts is not None else time.time())))
                total += len(rows)
            cur = end
    return total


def expire(conn, now_ts):
    """Delete the archive files (and their index rows) older than `ret.archive_days`."""
    d = folder(conn)
    if d is None or not _has_table(conn, "archive_idx"): return 0
    try: days = int(float(conn.execute("SELECT v FROM meta WHERE k='ret.archive_days'").fetchone()[0]))
    except (TypeError, ValueError, sqlite3.Error): days = 365
    old = conn.execute("SELECT id, file FROM archive_idx WHERE hi < ?", (int(now_ts) - days * 86400,)).fetchall()
    for _i, name in old:
        try: os.remove(os.path.join(d, name))
        except OSError: pass
    conn.execute("DELETE FROM archive_idx WHERE hi < ?", (int(now_ts) - days * 86400,)); return len(old)


def coverage(conn):
    """(first, last) second the archive holds (sig_1s), or None."""
    try:
        r = conn.execute("SELECT MIN(lo), MAX(hi) FROM archive_idx WHERE tbl IN ('sig_1s', 'sig_1m')").fetchone()
    except sqlite3.Error: return None
    return (r[0], r[1]) if r and r[0] is not None else None


def verify(conn):
    """-> (ok, problems): every file present, its hash as indexed, and the chain unbroken."""
    d = folder(conn); bad = []; prev = ""
    if d is None or not _has_table(conn, "archive_idx"): return True, bad
    for i, tbl, lo, hi, n, nb, name, sha, chain in conn.execute("SELECT id, tbl, lo, hi, rows, bytes, file, sha, chain FROM archive_idx ORDER BY id").fetchall():
        p = os.path.join(d, name)
        if not os.path.exists(p): bad.append(f"{name}: missing"); prev = chain; continue
        blob = open(p, "rb").read()
        if hashlib.sha256(blob).hexdigest() != sha: bad.append(f"{name}: changed")
        if hashlib.sha256((prev + sha + f"{tbl}{lo}{hi}{n}").encode()).hexdigest() != chain: bad.append(f"{name}: the chain is broken")
        prev = chain
    return not bad, bad


# ------------------------------------------------------------------------------------------------ reading
def _seg_state(conn):
    """{segment id: rows} already loaded on this connection (kept in its TEMP schema, so it dies with the connection)."""
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS arch_seg(id INTEGER PRIMARY KEY, rows INT)")
    return dict(conn.execute("SELECT id, rows FROM arch_seg").fetchall())


def _prepare(conn, tbl):
    """Temporary table holding archived rows of `tbl`, and a temporary view with the table's own name that adds them to the live rows (the live rows win)."""
    tmp = f"arch_{tbl}"
    if conn.execute("SELECT 1 FROM sqlite_temp_master WHERE name=?", (tmp,)).fetchone(): return tmp
    info = conn.execute(f"PRAGMA main.table_info({tbl})").fetchall(); names = [r[1] for r in info]; pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
    cols = ", ".join(f"{r[1]} {r[2] or ''}" for r in info) + (f", PRIMARY KEY({', '.join(pk)})" if pk else "")
    conn.execute(f"CREATE TEMP TABLE {tmp}({cols}) WITHOUT ROWID" if pk else f"CREATE TEMP TABLE {tmp}({cols})")
    sel = ", ".join(names); cond = f"ts < (SELECT COALESCE(MIN(ts), 9223372036854775807) FROM main.{tbl})"
    conn.execute(f"CREATE TEMP VIEW {tbl} AS SELECT {sel} FROM main.{tbl} UNION ALL SELECT {sel} FROM temp.{tmp} WHERE {cond}")
    for name, sql in conn.execute("SELECT name, sql FROM main.sqlite_master WHERE type='view' AND sql LIKE ?", (f"%{tbl}%",)).fetchall():          # a view of the main schema reads the main table, not the shadow: the views over this table are
        conn.execute(f"DROP VIEW IF EXISTS temp.{name}"); conn.execute(f"CREATE TEMP VIEW {name} AS {sql[sql.upper().index(' AS ') + 4:]}")             # made again in the TEMP schema, where the name resolves to the shadow
    return tmp


def hydrate(conn, lo, hi, tables=None):
    """Make the archived rows of [lo, hi) readable on this connection.  The archived SECONDS (`sig_1s`, `lk_1s`).
    {"rows": archived rows in the window (sig_1s, or the first minute table), "segments": files used, "damaged": [files that failed their hash]}."""
    out = {"rows": 0, "segments": 0, "damaged": []}
    d = folder(conn); tables = tuple(tables or TABLES)
    if d is None: return out
    try: segs = conn.execute(f"SELECT id, tbl, lo, hi, rows, file, sha FROM archive_idx WHERE hi >= ? AND lo < ? AND tbl IN ({','.join('?' * len(tables))}) ORDER BY id", (int(lo), int(hi), *tables)).fetchall()
    except sqlite3.Error: return out
    if not segs: return out
    was_ro = bool(conn.execute("PRAGMA query_only").fetchone()[0])
    if was_ro:
        try: conn.execute("PRAGMA query_only=OFF"); conn.set_authorizer(None)                                    # the assistant's handle is read-only; only its TEMP schema is written, here and nowhere else
        except sqlite3.Error: pass
    try:
        st = _seg_state(conn); need = [s for s in segs if s[0] not in st]
        if need:
            if sum(st.values()) + sum(s[4] for s in need) > MAX_HYDRATED_ROWS:
                for (t,) in conn.execute("SELECT name FROM sqlite_temp_master WHERE type='table' AND name LIKE 'arch\\_%' ESCAPE '\\' AND name <> 'arch_seg'").fetchall(): conn.execute(f"DELETE FROM temp.{t}")
                conn.execute("DELETE FROM temp.arch_seg"); st = {}
            for sid, tbl, slo, shi, n, name, sha in need:
                p = os.path.join(d, name)
                try:
                    blob = open(p, "rb").read()
                    if hashlib.sha256(blob).hexdigest() != sha: out["damaged"].append(name); continue
                    rows = decode(blob)
                except (OSError, ValueError, lzma.LZMAError): out["damaged"].append(name); continue
                tmp = _prepare(conn, tbl); ph = ",".join("?" * len(rows[0])) if rows else "?"
                if rows: conn.executemany(f"INSERT OR IGNORE INTO temp.{tmp} VALUES({ph})", rows)
                conn.execute("INSERT OR REPLACE INTO temp.arch_seg VALUES(?,?)", (sid, len(rows)))
    finally:
        if was_ro:
            try: conn.execute("PRAGMA query_only=ON")
            except sqlite3.Error: pass
            from . import store
            conn.set_authorizer(store._read_only_authorizer)
    first = tables[0]
    out["segments"] = sum(1 for s in segs if s[5] not in out["damaged"] and s[1] == first)
    if conn.execute("SELECT 1 FROM sqlite_temp_master WHERE name=?", (f"arch_{first}",)).fetchone():
        out["rows"] = conn.execute(f"SELECT COUNT(*) FROM temp.arch_{first} WHERE ts>=? AND ts<?", (int(lo), int(hi))).fetchone()[0]
    return out


def purge(conn):
    """The history was erased: the archive goes with it."""
    d = folder(conn); n = 0
    if d and os.path.isdir(d):
        for name in os.listdir(d):
            if name.endswith((".vga", ".part")):
                try: os.remove(os.path.join(d, name)); n += 1
                except OSError: pass
    if _has_table(conn, "archive_idx"): conn.execute("DELETE FROM archive_idx")
    return n

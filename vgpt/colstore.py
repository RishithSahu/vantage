"""Columnar copies of the per-minute tables, next to the database (`<database>.cols/`), so that a long window is read as arrays and not as millions of rows.

Why: SQLite hands rows to Python one tuple at a time (about 3.5 microseconds a row on the Raspberry Pi 4); four years of minutes is 2.1 million rows, so one question that needed a long window spent 7 seconds just receiving them.
A column file is the same numbers laid out end to end (float64, NaN for NULL); a window is two binary searches on the time column and a slice of the memory-mapped file.  Nothing is cached between questions in the sense of answers:
this is an index of the stored data, rebuilt from it, and it never holds anything the database does not.

Keeping it true to the database:
  * rows that arrive after the copy was made (the live writer appends one row a minute) are read from the database at every call and added at the end (at most TAIL_MAX of them; more, and the copy is rebuilt in the background);
  * before every use the copy is checked against the database: the first time, the last stored time and fourteen rows spread over the history must still be there; a copy that fails is ignored and rebuilt;
  * a table that was not copied yet, or a column that is missing, is read from the database as before while a background thread builds the copy (one table at a time, from its own read-only connection).
It is only used for tables named *_1m (append-only history) and for windows of MIN_WINDOW seconds or more; a folder that cannot be written disables it (everything then reads from the database as before)."""
import json, os, re, threading, time
import numpy as np

TAIL_MAX = 200_000
REFRESH_MIN_S = float(os.environ.get("VGPT_COLSTORE_REFRESH_S", "0.5"))      # the files are brought up to date at most this often per table (rows change once a minute; one question calls the copy a dozen times)
GUARD = 30                             # the newest minutes are rewritten by the live writer when a minute finishes: the last GUARD rows are re-read from the database at every call
EXTEND_AT = 1_000                      # rows that arrived since the copy: more than this and the copy is extended in the background (the live writer adds one row a minute per table)
MIN_WINDOW = 3 * 86400
_NAME = re.compile(r"^[a-z0-9_]+$")
_LOCK = threading.Lock(); _BUILDING = set(); _STORES = {}
SKIPS = {}                                                                                  # why a build ended without publishing a copy (diagnostics: reason -> times)


def _skip(why): SKIPS[why] = SKIPS.get(why, 0) + 1


_RETRY = {}                                                                                 # (database, table) -> earliest time a failed build is tried again
RETRY_AFTER = 300.0


def _backoff(key, secs=None): _RETRY[key] = time.time() + (RETRY_AFTER if secs is None else secs)


def _path_of(conn):
    try: row = conn.execute("PRAGMA database_list").fetchone()
    except Exception: return None
    return row[2] if row and row[2] else None


class ColStore:
    def __init__(self, db_path):
        self.db = db_path; self.dir = db_path + ".cols"; self._meta = {}; self._mm = {}; self._locks = {}; self._refreshed = {}; self._top = None; self._dbmin = 0; self.ok = os.environ.get("VGPT_COLSTORE", "1") == "1"
        if self.ok:
            try: os.makedirs(self.dir, exist_ok=True); self.ok = os.access(self.dir, os.W_OK)
            except OSError: self.ok = False

    def _lock(self, tb):
        with _LOCK: return self._locks.setdefault(tb, threading.Lock())                       # one per table: a long build of one table must not hold up another

    # ------------------------------------------------------------------ files
    def _f(self, tb, name, ext="f8"): return os.path.join(self.dir, f"{tb}.{name}.{ext}")

    def _load_meta(self, tb):
        try:
            with open(os.path.join(self.dir, tb + ".json")) as f: return json.load(f)
        except (OSError, ValueError): return None

    def _col(self, tb, name, n):
        key = (tb, name, n)
        a = self._mm.get(key)
        if a is None:
            for k in [k for k in self._mm if k[0] == tb and k[1] == name]: del self._mm[k]
            a = self._mm[key] = np.memmap(self._f(tb, name), dtype=np.float64, mode="r", shape=(n,))
        return a

    # ------------------------------------------------------------------ reading
    def read(self, conn, tb, cols, lo, hi):
        """-> float64 array [rows, 1 + len(cols)] (time first, NaN for NULL) for the rows of `tb` with lo <= ts < hi, or None when the copy cannot be used (the caller reads the database)."""
        a = self.columns(conn, tb, cols, lo, hi)
        if a is None: return None
        out = np.empty((len(a[0]), len(a)), dtype=np.float64)
        for j, x in enumerate(a): out[:, j] = x
        return out

    def columns(self, conn, tb, cols, lo, hi, _again=True):
        """-> [time, col, col, ...] as 1-D float64 arrays (NULL = NaN) of the rows of `tb` with lo <= ts < hi, or None when the copy cannot be used.  With no rows newer than the copy these are slices of the memory-mapped files (read only, nothing is copied)."""
        if not self.ok or not _NAME.match(tb) or not tb.endswith("_1m") or hi - lo < MIN_WINDOW: return None
        m = self._meta.get(tb)
        if m is None: m = self._meta[tb] = self._load_meta(tb)
        if not m or any(c not in m["cols"] for c in cols):
            if _again and self._build_now(tb, cols, m): return self.columns(conn, tb, cols, lo, hi, False)
            self.build_async(tb, sorted(set(cols) | set((m or {}).get("cols", [])))); return None
        n = m["n"]
        try:
            if not self._valid(conn, tb, m):
                self._meta[tb] = None
                if _again and self._build_now(tb, cols, m): return self.columns(conn, tb, cols, lo, hi, False)
                self.build_async(tb, sorted(set(cols) | set(m["cols"]))); return None
            lk = self._lock(tb); fresh = False
            if time.monotonic() - self._refreshed.get(tb, -1e9) < REFRESH_MIN_S: fresh = True              # brought up to date a moment ago (rows newer than the files are still read from the database below)
            elif lk.acquire(blocking=False):                                                  # never wait for a build or an extension in a question
                try: self._extend(tb); fresh = True; self._refreshed[tb] = time.monotonic()
                except Exception: pass
                finally: lk.release()
            if fresh:
                self._meta.pop(tb, None); m2 = self._meta[tb] = self._load_meta(tb)
                if m2 and all(c in m2["cols"] for c in cols): m = m2; n = m["n"]
            ts = self._col(tb, "ts", n); tail = []
            if not fresh or m["last"] < self._seen_top(conn, tb):                           # rows the files do not have yet (the files could not be brought up to date just now, or rows arrived since): read them from the database
                tail = conn.execute(f"SELECT ts, {', '.join(cols)} FROM {tb} WHERE ts>? ORDER BY ts LIMIT {TAIL_MAX + 1}", (int(m["last"]),)).fetchall() if cols else conn.execute(f"SELECT ts FROM {tb} WHERE ts>? ORDER BY ts LIMIT {TAIL_MAX + 1}", (int(m["last"]),)).fetchall()
                if len(tail) > TAIL_MAX: self.build_async(tb, sorted(set(cols) | set(m["cols"]))); return None
                if len(tail) > EXTEND_AT: self.extend_async(tb)
            top = self._top
            if top is not None and top < m["last"]: hi = min(hi, int(top) + 1)             # this connection's snapshot: rows the files have that it cannot see yet are not part of its answer
            lo = max(lo, self._dbmin)                                                        # rows the database has dropped (its retention) are not part of the answer either
            i0 = int(np.searchsorted(ts, lo, side="left")); i1 = int(np.searchsorted(ts, hi, side="left"))
            out = [ts[i0:i1]] + [self._col(tb, c, n)[i0:i1] for c in cols]
            if tail:
                t = np.array(tail, dtype=np.float64).reshape(len(tail), 1 + len(cols)); t = t[(t[:, 0] >= lo) & (t[:, 0] < hi)]
                if len(t): out = [np.concatenate([x, t[:, j]]) for j, x in enumerate(out)]
            return out
        except Exception:                                                                   # whatever goes wrong with the copy, the caller reads the database
            self._meta[tb] = None; _backoff((self.db, tb), 60); return None

    def _seen_top(self, conn, tb):
        self._top = conn.execute(f"SELECT MAX(ts) FROM {tb}").fetchone()[0]
        return -1 if self._top is None else int(self._top)

    def _build_now(self, tb, cols, m):
        """No usable copy of this table yet: make it here, in one pass over the table, and let the question use it.  The alternative was to answer this question by scanning the whole table through Python (20 to 60 s on the Raspberry Pi 4) while the same pass ran in the background.
        Waits for a build of this table that is already running instead of starting a second; gives up (and the caller reads the database) when a build of it ended without a copy a short while ago."""
        if os.environ.get("VGPT_COLSTORE_SYNC", "1") != "1" or _RETRY.get((self.db, tb), 0) > time.time(): return False
        try:
            with self._lock(tb):
                m2 = self._load_meta(tb)
                fresh = m2 is not None and all(c in m2["cols"] for c in cols) and (m is None or m2 != m)                        # somebody finished it while this waited for the lock
                if not fresh:
                    before = dict(SKIPS); self._build(tb, sorted(set(cols) | set((m2 or m or {}).get("cols", []))))
                    if SKIPS != before: _backoff((self.db, tb)); return False
            self._meta.pop(tb, None); return True
        except Exception:
            _backoff((self.db, tb)); return False

    def _valid(self, conn, tb, m):
        """The copy still describes this database.  The history is not only appended to: the newest minutes are rewritten when they finish (the last GUARD rows are re-read at every call, see _extend) and the oldest rows are dropped by the retention (rows before the database's first are ignored).
        So the checks are on the rest: the database does not start before the copy does, the spread-out check rows that still exist are there, and a row older than the guard has the stored value."""
        r = conn.execute(f"SELECT MIN(ts) FROM {tb}").fetchone()
        if not r or r[0] is None or int(r[0]) < m["first"]: return False
        self._dbmin = int(r[0]); self._top = None
        for t in m["check"]:
            if t >= self._dbmin and conn.execute(f"SELECT 1 FROM {tb} WHERE ts=?", (int(t),)).fetchone() is None: return False
        if not m["cols"]: return True
        anchor = int(m.get("anchor", m["first"]))
        if anchor < self._dbmin: return True
        ts = self._col(tb, "ts", m["n"]); i = int(np.searchsorted(ts, anchor))
        if i >= m["n"] or int(ts[i]) != anchor: return False
        c0 = m["cols"][0]; row = conn.execute(f"SELECT {c0} FROM {tb} WHERE ts=?", (anchor,)).fetchone(); ref = self._col(tb, c0, m["n"])[i]
        return row is not None and ((row[0] is None and np.isnan(ref)) or (row[0] is not None and float(row[0]) == float(ref)))

    # ------------------------------------------------------------------ building
    @staticmethod
    def _want(tb, cols):
        """Every column of the table a question may ask for (the registry's whitelist), so one pass over the table serves all later questions."""
        from . import signal_registry as R
        return sorted((set(R.TABLE_COLUMNS.get(tb, ())) - {"ts"}) | set(cols))

    def build_async(self, tb, cols):
        if not self.ok or not _NAME.match(tb) or not tb.endswith("_1m"): return
        key = (self.db, tb)
        with _LOCK:
            if key in _BUILDING or _RETRY.get(key, 0) > time.time(): return
            _BUILDING.add(key)
        threading.Thread(target=self._build_thread, args=(tb, cols, key), daemon=True, name="colstore-" + tb).start()

    def extend_async(self, tb):
        key = (self.db, tb)
        with _LOCK:
            if key in _BUILDING or _RETRY.get(key, 0) > time.time(): return
            _BUILDING.add(key)
        threading.Thread(target=self._build_thread, args=(tb, None, key), daemon=True, name="colstore-ext-" + tb).start()

    def extend(self, tb):
        with self._lock(tb): self._extend(tb)

    def _extend(self, tb):
        """Bring the files up to the database: re-read the last GUARD rows (the newest minutes may have been rewritten) and everything after them, overwrite from there, and replace the description last.  Files are never shortened (a reader may have them mapped)."""
        import sqlite3
        m = self._load_meta(tb)
        if not m: return
        names = ["ts"] + m["cols"]; n = m["n"]; s0 = max(0, n - GUARD); t0 = int(np.fromfile(self._f(tb, "ts"), dtype=np.float64, count=1, offset=8 * s0)[0]) if n else 0
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=30)
        try:
            con.execute("PRAGMA query_only=ON")
            rows = con.execute(f"SELECT {', '.join(names)} FROM {tb} WHERE ts>=? ORDER BY ts LIMIT 500000", (t0,)).fetchall()
        finally: con.close()
        if not rows: return
        if int(rows[0][0]) != t0: s0 = int(np.searchsorted(self._col(tb, "ts", n), rows[0][0]))      # the row the rewrite starts at is gone (dropped by the retention): start at the first one that is left
        arr = np.array(rows, dtype=np.float64)
        for j, nm in enumerate(names):
            with open(self._f(tb, nm), "r+b") as f: f.seek(8 * s0); f.write(arr[:, j].tobytes())
        m["n"] = s0 + len(rows); m["last"] = int(arr[-1, 0]); m["anchor"] = int(self._col_ts(tb, m["n"], max(0, m["n"] - 1 - GUARD))); tag = f".tmp{os.getpid()}"
        with open(os.path.join(self.dir, tb + ".json") + tag, "w") as f: json.dump(m, f)
        os.replace(os.path.join(self.dir, tb + ".json") + tag, os.path.join(self.dir, tb + ".json")); self._meta.pop(tb, None)

    def _col_ts(self, tb, n, i): return np.fromfile(self._f(tb, "ts"), dtype=np.float64, count=1, offset=8 * i)[0]

    def _build_thread(self, tb, cols, key):
        try:
            before = dict(SKIPS)
            self.build(tb, cols) if cols is not None else self.extend(tb)
            if SKIPS != before: _backoff(key)                                              # ended without publishing: do not start again at once
        except Exception: _backoff(key)
        finally:
            with _LOCK: _BUILDING.discard(key)

    def build(self, tb, cols):
        with self._lock(tb): self._build(tb, cols)

    def _build(self, tb, cols):
        """Copy the whole table's `cols` (and time) to column files, then publish them by writing the description last."""
        import sqlite3
        cols = self._want(tb, cols); con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=30)
        try:
            con.execute("PRAGMA query_only=ON")
            kind = con.execute("SELECT type, sql FROM sqlite_master WHERE name=?", (tb,)).fetchone()
            if kind and kind[0] == "view" and re.search(r"group\s+by|having", kind[1] or "", re.I): _backoff((self.db, tb), 86400); return _skip("grouped view")        # seconds to materialise, and it changes as rows complete
            top = con.execute(f"SELECT MAX(ts) FROM {tb}").fetchone()[0]
            if top is None: return _skip("empty")
            n = con.execute(f"SELECT COUNT(*) FROM {tb} WHERE ts<=?", (int(top),)).fetchone()[0]            # the live writer keeps adding rows while this runs: the copy is of the rows up to the newest one at the start
            names = ["ts"] + cols; arrs = [np.empty(n, dtype=np.float64) for _ in names]; i = 0
            cur = con.execute(f"SELECT {', '.join(names)} FROM {tb} WHERE ts<=? ORDER BY ts", (int(top),))
            while True:
                rows = cur.fetchmany(200_000)
                if not rows: break
                block = np.array(rows, dtype=np.float64)
                if i + len(block) > n: return _skip("grew")                               # the table grew while it was copied: the next call tries again
                for j in range(len(names)): arrs[j][i:i + len(block)] = block[:, j]
                i += len(block)
            if i != n: return _skip("short")
            self._mm = {k: v for k, v in self._mm.items() if k[0] != tb}; import gc; gc.collect()       # a file that is still mapped cannot be replaced on Windows
            tag = f".tmp{os.getpid()}"
            for nm, a in zip(names, arrs):
                with open(self._f(tb, nm) + tag, "wb") as f: f.write(a.tobytes())
            for nm in names: os.replace(self._f(tb, nm) + tag, self._f(tb, nm))
            ts = arrs[0]; check = sorted({int(ts[int(k)]) for k in np.linspace(0, n - 1, 14)})
            meta = {"n": int(n), "first": int(ts[0]), "last": int(ts[-1]), "anchor": int(ts[max(0, n - 1 - GUARD)]), "cols": cols, "check": check}
            with open(os.path.join(self.dir, tb + ".json") + tag, "w") as f: json.dump(meta, f)
            os.replace(os.path.join(self.dir, tb + ".json") + tag, os.path.join(self.dir, tb + ".json"))
            self._meta.pop(tb, None)
        finally: con.close()


def warm_async(db_path):
    """At start-up, in the background: read the column files into the operating system's file cache (the first question after a boot would otherwise wait for the SD card, 2 seconds on the Raspberry Pi 4), and start the copy of the per-minute signal table when there is none yet.
    This holds no answers and predicts no question: it is the same file read the database does on its own first use."""
    try: _warm(db_path)
    except Exception: pass


def _warm(db_path):
    if os.environ.get("VGPT_COLSTORE", "1") != "1" or not db_path or not os.path.isabs(os.path.abspath(db_path)) or not os.path.exists(db_path): return
    d = db_path + ".cols"
    if not os.path.exists(os.path.join(d, "sig_1m.json")):
        st = of_path(db_path)
        if st.ok:
            key = (db_path, "*warm")
            with _LOCK:
                if key in _BUILDING: return
                _BUILDING.add(key)

            def make():
                try:
                    for tb in ("sig_1m", "ext_1m", "cam_1m"):                                         # the real per-minute tables, one after the other, so a question that needs one finds it built
                        try: st.build(tb, ["flow"] if tb == "sig_1m" else [])
                        except Exception: pass
                    try:                                                                                # then the car's own lake signals (one view each, not the grouped ones): the first question about a tyre or a 12 V reading would otherwise make its copy on the spot (2 s on the Pi)
                        import sqlite3
                        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
                        try: views = sorted(n for n, q in con.execute("SELECT name, sql FROM sqlite_master WHERE type='view' AND name LIKE 'lkv%'") if re.match(r"^lkv_[a-z0-9]+_1m$", n) and "group by" not in (q or "").lower())
                        finally: con.close()
                        for tb in views:
                            try: st.build(tb, ["v"])
                            except Exception: pass
                            time.sleep(0.2)                                                              # a pause between the copies: the questions come first
                    except Exception: pass
                finally:
                    with _LOCK: _BUILDING.discard(key)
            threading.Thread(target=make, daemon=True, name="colstore-start").start()
        return

    def run():
        try:
            for n in sorted(os.listdir(d), key=lambda n: (not n.startswith("sig_1m."), n)):
                if not n.endswith(".f8"): continue
                with open(os.path.join(d, n), "rb", buffering=0) as f:
                    if hasattr(os, "posix_fadvise"): os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
                    else:
                        while f.read(1 << 22): pass
        except Exception: pass
    threading.Thread(target=run, daemon=True, name="colstore-warm").start()


def of_path(p):
    s = _STORES.get(p)
    if s is None:
        with _LOCK: s = _STORES.setdefault(p, ColStore(p))
    return s


def of(conn):
    """The store of a connection's database file (None for an in-memory or unnamed database)."""
    p = _path_of(conn)
    if not p or not os.path.isabs(p): return None
    return of_path(p)

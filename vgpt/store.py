"""Tiered on-device store (SQLite, stdlib only). Designed for a ~2 GB system shared with the vehicle OS:
  sig_1s   last 7 days at 1 Hz while driving/charging (1/min while parked)        - scaled integers, WITHOUT ROWID (compact)
  sig_1m   per-minute avg/min/max for 90 days                                       - what makes 'several weeks' cheap
  agg_1d   per-day mean/max/min of every column, forever (about 100 rows per quantity per year) - what makes 'years' cheap
  trips    one summary row per trip (energy/km, hvac, highway share ...)            - kept for the whole retention window
  events   ADAS / DTC / warnings / charge sessions                                  - kept long
  event_ctx +-15 min of 1 Hz key signals frozen (zlib) when an important event fires - so root cause survives the 1 Hz expiry
  logs     WARN+ log lines of services, with FTS5 full-text index                   - lexical retrieval; no embedding model needed (saves RAM)
Writers: ONLY the ingest daemon. The assistant opens the DB with  mode=ro  + PRAGMA query_only (one-way isolation, tested).
"""
import datetime as dt, json, math, os, sqlite3, struct, time, zlib
from collections import deque

from . import sqlfeatures

SIG_COLS = ["speed", "soc", "power", "batt", "flow", "amb", "hvac", "motor", "mode"]       # scaled ints in sig_1s
SCALE = {"speed": 10, "soc": 10, "power": 1, "batt": 10, "flow": 10, "amb": 10, "hvac": 1, "motor": 10, "mode": 1}   # units: 0.1 km/h, 0.1 %, W, 0.1 C ...
SCHEMA = """
CREATE TABLE IF NOT EXISTS sig_1s(ts INTEGER PRIMARY KEY, speed INT, soc INT, power INT, batt INT, flow INT, amb INT, hvac INT, motor INT, mode INT) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS sig_1m(ts INTEGER PRIMARY KEY, speed INT, soc INT, power INT, batt INT, batt_max INT, flow INT, flow_min INT, amb INT, hvac INT, motor INT, mode INT) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS trips(id INTEGER PRIMARY KEY, start INT, end INT, km REAL, kwh REAL, wh_km REAL, avg_speed REAL, hwy_share REAL, avg_amb REAL, avg_hvac_w REAL, soc_start REAL, soc_end REAL);
CREATE TABLE IF NOT EXISTS events(ts INT, type TEXT, sev TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS i_ev ON events(type, ts);
CREATE TABLE IF NOT EXISTS event_ctx(ts INT PRIMARY KEY, type TEXT, blob BLOB);
CREATE TABLE IF NOT EXISTS event_ctx_v2(ts INT, type TEXT, blob BLOB, PRIMARY KEY(ts,type));
CREATE TABLE IF NOT EXISTS logs(ts INT, src TEXT, msg TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(msg, src, content='logs', content_rowid='rowid');
CREATE TABLE IF NOT EXISTS cam_1m(ts INTEGER PRIMARY KEY, frames INT, block_avg REAL, block_max REAL, sharp_avg REAL, luma_avg REAL, lane_avg REAL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS event_thumb(ts INTEGER PRIMARY KEY, type TEXT, png BLOB);   -- 64x48 grey thumbnail at camera fault events (metadata-only policy)
CREATE TABLE IF NOT EXISTS event_thumb_v2(ts INTEGER, type TEXT, png BLOB, PRIMARY KEY(ts,type));
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS agg_1d(tbl TEXT, col TEXT, masked INT, d INT, tz INT, n INT, avg REAL, mx REAL, mn REAL, PRIMARY KEY(tbl, col, masked, d, tz)) WITHOUT ROWID;   -- the DAILY tier: per local day mean / max / min of every stored column (kept for the whole life of the vehicle)
CREATE TABLE IF NOT EXISTS agg_1d_state(tbl TEXT, masked INT, tz INT, through INT, sig_max_ts INT, PRIMARY KEY(tbl, masked, tz)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS gps_1s(ts INTEGER PRIMARY KEY, lat INT, lon INT, hdg INT, spd INT, hdop INT) WITHOUT ROWID;   -- GNSS fixes: deg*1e6, course deg*10, km/h*10, HDOP*10 (1 Hz while the car is awake, 1/min parked; last 7 days)
-- extension tables (generated / richer vehicles; every reader tolerates their absence)
CREATE TABLE IF NOT EXISTS ext_1m(ts INTEGER PRIMARY KEY, tp_fl INT, tp_fr INT, tp_rl INT, tp_rr INT, rain INT, lux INT, v12 INT, adas INT, odo INT, lat INT, lon INT, tt INT) WITHOUT ROWID;   -- per active minute: tyre kPa*10, rain 0-3, lux, 12V*100, ADAS share %, odo km*10, lat/lon *1e5, tyre temperature C*10 (TPMS sensors report it)
CREATE TABLE IF NOT EXISTS vday(ts INTEGER PRIMARY KEY, odo REAL, pad REAL, soh REAL, v12 REAL) WITHOUT ROWID;                                                                      -- daily: odometer, brake-pad %, battery state of health %, 12V rest voltage
CREATE TABLE IF NOT EXISTS places(id INTEGER PRIMARY KEY, name TEXT, kind TEXT, lat REAL, lon REAL);
CREATE TABLE IF NOT EXISTS charges(start INTEGER PRIMARY KEY, end INTEGER, kind TEXT, kw REAL, kwh REAL, soc0 REAL, soc1 REAL, place INT, lat REAL, lon REAL, grid_kwh REAL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS trip_ext(start INTEGER PRIMARY KEY, from_id INT, to_id INT, harsh_brake INT, harsh_accel INT, harsh_corner INT, regen_kwh REAL, rain_share REAL, adas_share REAL, lat0 REAL, lon0 REAL, lat1 REAL, lon1 REAL) WITHOUT ROWID;
"""
CTX_TYPES = {"WARN_BATT_OVERTEMP": 900, "DTC_SET": 900, "ADAS_DISENGAGE": 120, "CAMERA_BLOCKED": 120, "CAMERA_DEGRADED": 120, "ECU_RESET": 120, "UNEXPECTED_SHUTDOWN": 300, "VOLTAGE_DIP": 120,
             "SENSOR_CONFLICT": 120, "AEB_ACTIVATION": 120, "CAN_ERROR": 120, "POWER_DERATE": 300, "WARN_OVERTEMP": 600, "LOW_12V": 300, "TPMS_LOW": 300}      # seconds of context either side


def schema_for(fts5):
    """The schema this SQLite can build: without FTS5 (a stripped-down embedded build) there is no log index and the log search falls back to a substring match (vgpt/sqlfeatures.py)."""
    return SCHEMA if fts5 else SCHEMA.replace("CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(msg, src, content='logs', content_rowid='rowid');\n", "")


SCHEMA_HERE = schema_for(sqlfeatures.has_fts5())
RET_1S_DAYS, RET_1M_DAYS, RET_LOG_DAYS = 7, 365, 1825


def _tier_cache(cache_kb):
    """(page cache KiB, mmap bytes): explicit argument, else the resource tier's (VGPT_TIER), else the 4 MiB / no-mmap default of the 2 GB board."""
    from . import tiers
    if cache_kb is not None: return int(cache_kb), 0
    if os.environ.get("VGPT_TIER"): t = tiers.get(); return int(t.sqlite.get("cache_kb", 4096)), int(t.sqlite.get("mmap_mb", 0)) * 2 ** 20
    return 4096, 0


def open_rw(path, cache_kb=None, threads=False):
    """Writer handle.  `threads=True` lets several threads share it (the live runtime serialises them with its own lock)."""
    cache_kb, mmap = _tier_cache(cache_kb)
    c = sqlite3.connect(path, isolation_level=None, timeout=10, check_same_thread=not threads)
    c.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA mmap_size=%d; PRAGMA cache_size=-%d; PRAGMA temp_store=FILE;" % (mmap, cache_kb))
    c.executescript(SCHEMA_HERE); return c


_DENIED = tuple(getattr(sqlite3, n) for n in ("SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE", "SQLITE_CREATE_INDEX",
                                               "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW",
                                               "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER", "SQLITE_DROP_TEMP_VIEW",
                                               "SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE") if hasattr(sqlite3, n))


def _read_only_authorizer(action, a, b, db, source):
    """The third lock on the assistant's handle (after mode=ro and query_only): a statement that would write, or reach another file by ATTACH, is refused when it is prepared."""
    return sqlite3.SQLITE_DENY if action in _DENIED else sqlite3.SQLITE_OK


def open_ro(path, cache_kb=None):
    """Assistant-side handle: read-only at the OS/URI level, at the SQL level, and by an authorizer that refuses writes and ATTACH."""
    cache_kb, mmap = _tier_cache(cache_kb)
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    c.execute("PRAGMA query_only=ON"); c.execute("PRAGMA mmap_size=%d" % mmap); c.execute("PRAGMA cache_size=-%d" % cache_kb); c.set_authorizer(_read_only_authorizer); return c


def q_int(name, v):
    if v is None: return None
    return int(round(v * SCALE[name]))


# Maximum age at which a last value still describes the current second.  Fast
# motion signals expire quickly; slow thermal/SOC sensors may legitimately
# publish less often.  Absence is stored as NULL, never fabricated as zero.
TTL = {"speed": 2, "speed_vhal": 2, "ign": 2, "chg_state": 2, "chg_kw": 2,
       "pack_v": 5, "pack_a": 5, "soc": 120, "batt_temp": 120,
       "coolant_flow": 10, "amb_temp": 300, "amb_vhal": 300,
       "hvac_w": 10, "motor_temp": 120}


class Ingestor:
    """Live path: decoded records -> 1 Hz rows + events + logs. Keeps a 30 min RAM ring of 1 Hz rows for event-context freezing."""
    def __init__(self, conn, router, lake=None, track_meta=False):
        self.c, self.router = conn, router; self.lake = lake; self.track_meta = track_meta; self._meta_at = 0
        self.alias = dict(getattr(router, "alias", None) or {})
        self.inv = {internal: concept for concept, internal in self.alias.items()}          # internal column name -> the concept name the router emits (profile routers)
        self.ttl_over = getattr(router, "ttl", None) or {}
        self.latest, self.sec, self.ring = {}, None, deque(maxlen=1800)
        self.pending_events = []; self.gps_dirty = False; self.gps_fix_ts = None; self.gps_last = 0; self.msgs = 0; self.rows = 0; self.park_last = 0; self.cam_acc = {}; self.cam_finalized = set(); self.thumb = None
        self.dropped_late = self.dropped_invalid = 0; self.fts = sqlfeatures.log_index_usable(conn)
        prof = getattr(router, "p", None)                                                  # a vehicle that never states its ignition (an OBD-II dongle, a learned profile): the drive state is derived from motion
        self.derive_ign = prof is not None and not any(b.concept == "_ignition" for b in getattr(prof, "bindings", ())); self._moved_at = None

    def decode(self, bus, payload, ts):
        """The collector's half: raw traffic -> canonical records (the decoder layer; it keeps no database state)."""
        return self.router.decode(bus, payload, ts) if getattr(self.router, "wants_ts", False) else self.router.decode(bus, payload)

    def feed(self, ts, bus, payload):
        return self.feed_recs(ts, self.decode(bus, payload, ts))

    def feed_recs(self, ts, recs):
        """The store's half: canonical records -> 1 Hz rows, events, logs.  This is the only code that writes, and it never sees raw bus traffic."""
        self.msgs += 1
        s = int(ts)
        if self.sec is None: self.sec = s
        if s < self.sec:
            self.dropped_late += 1
            return
        if s > self.sec:
            self._flush(self.sec); self.sec = s
        for rec in recs:
            k = rec[0]
            if k in ("sig", "sig_ts"):
                name, value = rec[1], rec[2]; source_s = int(rec[3]) if k == "sig_ts" and rec[3] is not None else s
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    self.dropped_invalid += 1; continue
                if name == "gps_valid" and value <= 0:
                    for x in ("lat", "lon", "hdg", "gspeed", "hdop"): self.latest.pop(x, None)
                    self.gps_dirty = False; self.gps_fix_ts = None
                    self.latest[name] = (float(value), source_s); continue
                self.latest[name] = (float(value), source_s)
                if name in ("lat", "lon"): self.gps_dirty = True; self.gps_fix_ts = source_s
                internal = self.alias.get(name, name)
                if internal in ("cam_block", "cam_sharp", "cam_luma", "lane_conf"): self._cam(s, internal, float(value))
            elif k == "thumb": self.thumb = rec[1]
            elif k == "event": self._event(s, rec[1], rec[2], rec[3])
            elif k == "log":
                log_s = s
                if len(rec) > 3 and rec[3]:
                    try: log_s = int(dt.datetime.fromisoformat(str(rec[3]).replace("Z", "+00:00")).timestamp())
                    except (ValueError, TypeError): pass
                cur = self.c.execute("INSERT INTO logs(ts,src,msg) VALUES(?,?,?)", (log_s, rec[1], rec[2]))
                if self.fts: self.c.execute("INSERT INTO logs_fts(rowid,msg,src) VALUES(?,?,?)", (cur.lastrowid, rec[2], rec[1]))

    def _value(self, name, sec):
        key = self.inv.get(name, name); item = self.latest.get(key)
        if item is None: return None
        value, source_s = item
        return value if 0 <= sec - source_s <= self.ttl_over.get(key, TTL.get(name, 2)) else None

    def _row(self, sec):
        speed = self._value("speed", sec)
        soc = self._value("soc", sec); batt = self._value("batt_temp", sec); flow = self._value("coolant_flow", sec)
        amb = self._value("amb_temp", sec); hvac = self._value("hvac_w", sec); motor = self._value("motor_temp", sec)
        volt, amps = self._value("pack_v", sec), self._value("pack_a", sec)
        pw = volt * amps if volt is not None and amps is not None else None
        if pw is None:
            kw = self._value("pack_kw", sec)                                                   # a vehicle that publishes battery power directly
            if kw is not None: pw = kw * 1000.0
        charge_state = self._value("chg_state", sec); charge_kw = self._value("chg_kw", sec); ign = self._value("ign", sec)
        if ign is None and self.derive_ign: ign = self._derived_ign(sec, speed)
        if charge_state == 1.0: mode = 3
        elif charge_state == 0.0:
            mode = 0 if ign == 0.0 else (None if speed is None else (2 if speed > 75 else 1))
        elif charge_kw is not None and charge_kw > 0.5: mode = 3
        elif ign == 0.0: mode = 0
        elif ign == 1.0 and speed is not None: mode = 2 if speed > 75 else 1
        else: mode = None
        return (q_int("speed", speed), q_int("soc", soc), int(round(pw)) if pw is not None else None, q_int("batt", batt), q_int("flow", flow),
                q_int("amb", amb), int(round(hvac)) if hvac is not None else None, q_int("motor", motor), mode)

    def _derived_ign(self, sec, speed):
        """1 while the vehicle has moved or its engine has turned within the last 120 s (a stop at a red light does not end the trip), 0 after that; None before anything is known."""
        rpm = self.latest.get("engine_rpm")
        if (speed is not None and speed > 1.0) or (rpm is not None and rpm[0] > 300 and 0 <= sec - rpm[1] <= 5): self._moved_at = sec
        if self._moved_at is None: return None if speed is None and rpm is None else 0.0
        return 1.0 if sec - self._moved_at <= 120 else 0.0

    def _cam(self, ts, name, v):
        """Per-minute camera metadata aggregate (no pixels): frames, blocked share avg/max, sharpness, luma, lane confidence."""
        m = ts // 60 * 60; a = self.cam_acc.get(m)
        if m in self.cam_finalized:
            self.dropped_late += 1
            return
        if a is None:
            for old in [k for k in self.cam_acc if k < m]: self._cam_write(old)
            a = self.cam_acc[m] = {x: {"sum": 0.0, "n": 0} for x in ("cam_block", "cam_sharp", "cam_luma", "lane_conf")}; a["bmax"] = None
        a[name]["sum"] += v; a[name]["n"] += 1
        if name == "cam_block": a["bmax"] = v if a["bmax"] is None else max(a["bmax"], v)

    def _cam_write(self, m):
        a = self.cam_acc.pop(m, None)
        if a:
            avg = lambda k: a[k]["sum"] / a[k]["n"] if a[k]["n"] else None
            n = a["cam_block"]["n"]
            self.c.execute("INSERT OR REPLACE INTO cam_1m VALUES(?,?,?,?,?,?,?)", (m, n, avg("cam_block"), a["bmax"], avg("cam_sharp"), avg("cam_luma"), avg("lane_conf")))
            self.cam_finalized.add(m)

    def snapshot(self):
        """Make the partial current-minute camera aggregate and any pending events visible to readers WITHOUT closing the minute (used by live UIs)."""
        self._flush_events(final=False)
        for m, a in self.cam_acc.items():
            avg = lambda k: a[k]["sum"] / a[k]["n"] if a[k]["n"] else None
            self.c.execute("INSERT OR REPLACE INTO cam_1m VALUES(?,?,?,?,?,?,?)", (m, a["cam_block"]["n"], avg("cam_block"), a["bmax"], avg("cam_sharp"), avg("cam_luma"), avg("lane_conf")))

    def _gps_write(self, sec):
        """One GNSS row per second while moving, one per minute while parked (what a car would keep); a fix needs both latitude and longitude."""
        if not self.gps_dirty or "lat" not in self.latest or "lon" not in self.latest: return
        fix = int(self.gps_fix_ts if self.gps_fix_ts is not None else sec)
        lat, lts = self.latest["lat"]; lon, lots = self.latest["lon"]
        if lts != fix or lots != fix: return
        self.gps_dirty = False
        def gv(n):
            x = self.latest.get(n); return x[0] if x is not None and x[1] == fix else None
        sp = gv("gspeed")
        if sp is not None and sp < 1.5 and fix - self.gps_last < 60: return
        self.gps_last = fix
        hdg, hdop = gv("hdg"), gv("hdop")
        self.c.execute("INSERT OR REPLACE INTO gps_1s VALUES(?,?,?,?,?,?)", (fix, int(round(lat * 1e6)), int(round(lon * 1e6)), None if hdg is None else int(round(hdg * 10)), None if sp is None else int(round(sp * 10)), None if hdop is None else int(round(hdop * 10))))

    def _flush(self, sec):
        self._gps_write(sec)
        row = self._row(sec)
        if self.lake is not None: self.lake.tick(sec, self.latest, row[-1] != 0 if row[-1] is not None else any(v is not None for v in row))
        if self.track_meta and sec - self._meta_at >= 10:
            self._meta_at = sec; self.c.execute("INSERT OR IGNORE INTO meta VALUES('t0',?)", (str(sec),)); self.c.execute("INSERT OR REPLACE INTO meta VALUES('end',?)", (str(sec),))
        if not any(v is not None for v in row):
            self._flush_events(final=False); return
        self.ring.append((sec,) + row)
        self._flush_events(final=False)
        if row[-1] != 0 or sec - self.park_last >= 60:                   # parked: 1 row per minute
            self.c.execute("INSERT OR REPLACE INTO sig_1s VALUES(?,?,?,?,?,?,?,?,?,?)", (sec,) + row); self.rows += 1
            if row[-1] == 0: self.park_last = sec

    def _flush_events(self, final=False):
        """Freeze pending event contexts only after their post-event window exists."""
        keep = []
        for (ts, ty, sv, de, th) in self.pending_events:
            if not final and self.sec is not None and self.sec < ts + 5:
                keep.append((ts, ty, sv, de, th)); continue
            if ty in CTX_TYPES and self.ring: self._freeze(ts, ty)
        self.pending_events = keep

    def _event(self, ts, ty, sv, de):
        th = self.thumb if ty.startswith("CAMERA_") else None
        if th is not None: self.thumb = None
        # The event is visible immediately; only its forensic context is
        # delayed until following samples have arrived.
        self.c.execute("INSERT INTO events VALUES(?,?,?,?)", (ts, ty, sv, de))
        if th:
            self.c.execute("INSERT OR REPLACE INTO event_thumb VALUES(?,?,?)", (ts, ty, th))  # legacy reader compatibility
            self.c.execute("INSERT OR REPLACE INTO event_thumb_v2 VALUES(?,?,?)", (ts, ty, th))
        if ty in CTX_TYPES: self.pending_events.append((ts, ty, sv, de, th))

    def _freeze(self, ts, ty):
        win = CTX_TYPES[ty]; rows = [r for r in self.ring if ts - win <= r[0] <= ts + 5]
        if rows:
            blob = zlib.compress(json.dumps(rows, separators=(",", ":")).encode(), 6)
            self.c.execute("INSERT OR REPLACE INTO event_ctx VALUES(?,?,?)", (ts, ty, blob))  # legacy reader compatibility
            self.c.execute("INSERT OR REPLACE INTO event_ctx_v2 VALUES(?,?,?)", (ts, ty, blob))

    def close(self):
        if self.lake is not None: self.lake.close()
        if self.sec is not None: self._flush(self.sec)
        self._flush_events(final=True)
        for m in list(self.cam_acc): self._cam_write(m)


DAILY_TABLES = ("sig_1m", "ext_1m")


def ensure_daily_schema(conn):
    """Create the daily-tier tables in a database that was made before they existed (the writer only)."""
    conn.execute("CREATE TABLE IF NOT EXISTS agg_1d(tbl TEXT, col TEXT, masked INT, d INT, tz INT, n INT, avg REAL, mx REAL, mn REAL, PRIMARY KEY(tbl, col, masked, d, tz)) WITHOUT ROWID")
    conn.execute("CREATE TABLE IF NOT EXISTS agg_1d_state(tbl TEXT, masked INT, tz INT, through INT, sig_max_ts INT, PRIMARY KEY(tbl, masked, tz)) WITHOUT ROWID")


def tz_of(conn):
    try: return int(float(dict(conn.execute("SELECT k, v FROM meta WHERE k LIKE 'env.tz_h'").fetchall()).get("env.tz_h", 0)) * 3600)
    except (sqlite3.Error, ValueError): return 0


def _daily_targets(conn):
    """[(table_or_view, [columns])] that get a daily tier: the minute tables (sig_1m, ext_1m, cam_1m: every column but ts) and the vehicle's own lake signals (views lkv_<sid>_1m: column v)."""
    import re
    out = []
    for name, typ in conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view')").fetchall():
        if name in ("sig_1m", "ext_1m", "cam_1m"):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({name})") if r[1] != "ts" and (r[2] or "").upper() in ("INT", "INTEGER", "REAL", "FLOAT", "NUMERIC")]
            if cols: out.append((name, cols))
        elif re.fullmatch(r"lkv_(?:\d+|tyre)_1m", name):
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({name})")}; out.append((name, [c for c in ("v", "vmin", "vmax") if c in have]))
    return sorted(out)


def roll_daily(conn, now_ts=None):
    """The daily tier: for every finished local day, the mean / maximum / minimum / count of each column of the minute tables and of the vehicle's own lake signals, once with only the minutes the car was
    driving (mode 1 or 2) and once with all minutes.  Incremental (only days after the last rolled day), written by the ONE writer (the ingest daemon, in `compact`), read by the cause engine and by
    long-range questions.  The statements are the same SQL the engine used to run over the whole window per question, so the numbers are identical; what changes is that a question about 70 days, or
    3 years, reads ~100 rows per quantity instead of scanning every minute.  The day that contains the newest sample is never rolled (it is still filling)."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "sig_1m" not in tables or "agg_1d" not in tables: return 0
    tz = tz_of(conn); hi = conn.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]
    if hi is None: return 0
    through = (int(hi) + tz) // 86400 - 1; written = 0; has_mode = "mode" in {r[1] for r in conn.execute("PRAGMA table_info(sig_1m)")}
    for table, cols in _daily_targets(conn):
        for masked in (1, 0):
            if masked and not has_mode: continue
            st = conn.execute("SELECT through FROM agg_1d_state WHERE tbl=? AND masked=? AND tz=?", (table, masked, tz)).fetchone()
            if st:                                                                                           # a column added to the tier later (or never rolled) is rolled from the first day
                have_c = {r[0] for r in conn.execute("SELECT DISTINCT col FROM agg_1d WHERE tbl=? AND masked=? AND tz=?", (table, masked, tz))}
                if any(c not in have_c and conn.execute(f"SELECT 1 FROM {table} WHERE {c} IS NOT NULL LIMIT 1").fetchone() for c in cols):
                    conn.execute("DELETE FROM agg_1d WHERE tbl=? AND masked=? AND tz=?", (table, masked, tz)); conn.execute("DELETE FROM agg_1d_state WHERE tbl=? AND masked=? AND tz=?", (table, masked, tz)); st = None
            lo_ts = None if st else conn.execute(f"SELECT MIN(ts) FROM {table}").fetchone()[0]
            first = (st[0] + 1) if st else ((int(lo_ts) + tz) // 86400 if lo_ts is not None else None)
            if first is None or first > through: continue
            join = " JOIN sig_1m s ON s.ts = t.ts" if masked and table != "sig_1m" else ""
            where = (" AND s.mode IN (1,2)" if table != "sig_1m" else " AND t.mode IN (1,2)") if masked else ""
            sel = ", ".join(f"AVG(t.{c}), MAX(t.{c}), MIN(t.{c}), COUNT(t.{c})" for c in cols); rows = []
            for row in conn.execute(f"SELECT CAST((t.ts + ?) / 86400 AS INTEGER) d, {sel} FROM {table} t{join} WHERE t.ts >= ? AND t.ts < ?{where} GROUP BY d", (tz, first * 86400 - tz, (through + 1) * 86400 - tz)):
                for ci, c in enumerate(cols):
                    av, mx, mn, n = row[1 + 4 * ci: 5 + 4 * ci]
                    if n: rows.append((table, c, masked, int(row[0]), tz, int(n), av, mx, mn))
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany("INSERT OR REPLACE INTO agg_1d VALUES(?,?,?,?,?,?,?,?,?)", rows)
                conn.execute("INSERT OR REPLACE INTO agg_1d_state VALUES(?,?,?,?,?)", (table, masked, tz, through, int(hi))); conn.execute("COMMIT")
            except Exception: conn.execute("ROLLBACK"); raise
            written += len(rows)
    return written


def compact(conn, now_ts):
    """Nightly housekeeping: 1 s -> 1 min tier, expiry of old tiers. Pure SQL (no numpy), bounded transaction size."""
    ret = {}
    try: ret = dict(conn.execute("SELECT k, v FROM meta WHERE k LIKE 'ret.%'").fetchall())             # a provisioned vehicle carries its own retention
    except sqlite3.Error: pass
    fine, coarse, logs = (int(float(ret.get(k, d))) for k, d in (("ret.fine_days", RET_1S_DAYS), ("ret.coarse_days", RET_1M_DAYS), ("ret.log_days", RET_LOG_DAYS)))
    cut = now_ts - fine * 86400
    complete_cut = (cut // 60) * 60                                # never split a minute across the two retention tiers
    from . import archive
    arch = archive.write(conn, complete_cut, now_ts)               # the seconds that are about to expire are written, losslessly, to the archive first: a crash here loses nothing (the rows are still there)
    try: roll_daily(conn, now_ts)                                    # the daily tier is written BEFORE any minute of the old tier can expire
    except sqlite3.Error: pass
    conn.execute("BEGIN")
    conn.execute("""WITH base AS (
      SELECT *, ts/60*60 AS m, LEAD(ts,1,ts/60*60+60) OVER (PARTITION BY ts/60 ORDER BY ts) AS nt
      FROM sig_1s WHERE ts < ?), w AS (
      SELECT *, MAX(1,MIN(nt,m+60)-ts) AS dt FROM base)
      INSERT OR REPLACE INTO sig_1m
      SELECT m,
       CAST(SUM(CASE WHEN speed IS NOT NULL THEN speed*dt END)/SUM(CASE WHEN speed IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN soc IS NOT NULL THEN soc*dt END)/SUM(CASE WHEN soc IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN power IS NOT NULL THEN power*dt END)/SUM(CASE WHEN power IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN batt IS NOT NULL THEN batt*dt END)/SUM(CASE WHEN batt IS NOT NULL THEN dt END) AS INT), MAX(batt),
       CAST(SUM(CASE WHEN flow IS NOT NULL THEN flow*dt END)/SUM(CASE WHEN flow IS NOT NULL THEN dt END) AS INT), MIN(flow),
       CAST(SUM(CASE WHEN amb IS NOT NULL THEN amb*dt END)/SUM(CASE WHEN amb IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN hvac IS NOT NULL THEN hvac*dt END)/SUM(CASE WHEN hvac IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN motor IS NOT NULL THEN motor*dt END)/SUM(CASE WHEN motor IS NOT NULL THEN dt END) AS INT),
       (SELECT mode FROM w w2 WHERE w2.m=w.m AND mode IS NOT NULL GROUP BY mode ORDER BY SUM(dt) DESC, mode ASC LIMIT 1)
      FROM w GROUP BY m""", (complete_cut,))
    conn.execute("DELETE FROM sig_1s WHERE ts < ?", (complete_cut,))
    archive.expire(conn, now_ts)
    from . import storage_budget
    storage_budget.enforce(conn, now_ts)                           # the history keeps within its size budget (ret.budget_gb, default 6): the oldest archived files go first
    conn.execute("DELETE FROM sig_1m WHERE ts < ?", (now_ts - coarse * 86400,))
    conn.execute("DELETE FROM logs WHERE ts < ?", (now_ts - logs * 86400,))
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='lk_1m'").fetchone():
        conn.execute("DELETE FROM lk_1s WHERE ts < ?", (complete_cut,)); conn.execute("DELETE FROM lk_1m WHERE ts < ?", (now_ts - coarse * 86400,))
        for t in ("cam_1m", "ext_1m"): conn.execute(f"DELETE FROM {t} WHERE ts < ?", (now_ts - coarse * 86400,))
    ctx = int(float(ret.get("ret.context_days", 1825))); conn.execute("DELETE FROM event_ctx WHERE ts < ?", (now_ts - ctx * 86400,)); conn.execute("DELETE FROM event_ctx_v2 WHERE ts < ?", (now_ts - ctx * 86400,))
    conn.execute("DELETE FROM gps_1s WHERE ts < ?", (cut,))
    sqlfeatures.rebuild_log_index(conn)
    conn.execute("COMMIT"); conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def rollup(conn, lo, hi):
    """The minute rows of the finished minutes in [lo, hi): the same duration-weighted averages and modal state the daily compaction writes, produced as soon as a minute is complete so that a question about
    'this week' or 'the last hour' includes what has just arrived.  The 1 Hz rows stay where they are (the compaction expires them later)."""
    lo, hi = int(lo) // 60 * 60, int(hi) // 60 * 60
    if hi <= lo: return 0
    conn.execute("""WITH base AS (
      SELECT *, ts/60*60 AS m, LEAD(ts,1,ts/60*60+60) OVER (PARTITION BY ts/60 ORDER BY ts) AS nt
      FROM sig_1s WHERE ts >= ? AND ts < ?), w AS (
      SELECT *, MAX(1,MIN(nt,m+60)-ts) AS dt FROM base)
      INSERT OR REPLACE INTO sig_1m
      SELECT m,
       CAST(SUM(CASE WHEN speed IS NOT NULL THEN speed*dt END)/SUM(CASE WHEN speed IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN soc IS NOT NULL THEN soc*dt END)/SUM(CASE WHEN soc IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN power IS NOT NULL THEN power*dt END)/SUM(CASE WHEN power IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN batt IS NOT NULL THEN batt*dt END)/SUM(CASE WHEN batt IS NOT NULL THEN dt END) AS INT), MAX(batt),
       CAST(SUM(CASE WHEN flow IS NOT NULL THEN flow*dt END)/SUM(CASE WHEN flow IS NOT NULL THEN dt END) AS INT), MIN(flow),
       CAST(SUM(CASE WHEN amb IS NOT NULL THEN amb*dt END)/SUM(CASE WHEN amb IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN hvac IS NOT NULL THEN hvac*dt END)/SUM(CASE WHEN hvac IS NOT NULL THEN dt END) AS INT),
       CAST(SUM(CASE WHEN motor IS NOT NULL THEN motor*dt END)/SUM(CASE WHEN motor IS NOT NULL THEN dt END) AS INT),
       (SELECT mode FROM w w2 WHERE w2.m=w.m AND mode IS NOT NULL GROUP BY mode ORDER BY SUM(dt) DESC, mode ASC LIMIT 1)
      FROM w GROUP BY m""", (lo, hi))
    return (hi - lo) // 60


# ---------------------------------------------------------------------- DEV-ONLY bulk seeding from the simulator (numpy)
def seed_from_sim(path, veh, keep_1s_days=RET_1S_DAYS):
    import numpy as np
    import os
    for ext in ("", "-wal", "-shm"):
        try: os.remove(path + ext)
        except OSError: pass
    c = open_rw(path); n = veh.n; t0 = veh.t0
    sec = np.arange(n); drive = veh.ign | (veh.mode == 3)
    keep = drive | (sec % 60 == 0)
    cols = [veh.speed * 10, veh.soc * 10, veh.power * 1000.0, veh.batt * 10, veh.flow * 10, veh.amb * 10, veh.hvac, veh.motor * 10, veh.mode]
    A = np.stack([np.rint(x).astype(np.int64) for x in cols], 1)
    cut_s = n - keep_1s_days * 86400
    idx = np.where(keep & (sec >= cut_s))[0]
    c.execute("BEGIN"); c.executemany("INSERT INTO sig_1s VALUES(?,?,?,?,?,?,?,?,?,?)", ((int(t0 + i), *map(int, A[i])) for i in idx)); c.execute("COMMIT")
    # minute tier for everything (what compaction would have produced)
    m = np.arange(0, n - n % 60, 60); R = A[: n - n % 60].reshape(-1, 60, A.shape[1])
    # Seed every represented minute using the same duration-weighted modal
    # state rule as compact(), including parked periods.
    sel = np.arange(len(m))
    rows = []
    for k in sel:
        b = R[k]; rows.append((int(t0 + m[k]), int(b[:, 0].mean()), int(b[:, 1].mean()), int(b[:, 2].mean()), int(b[:, 3].mean()), int(b[:, 3].max()), int(b[:, 4].mean()),
                               int(b[:, 4].min()), int(b[:, 5].mean()), int(b[:, 6].mean()), int(b[:, 7].mean()), int(np.min(np.flatnonzero(np.bincount(b[:, 8].astype(int)) == np.bincount(b[:, 8].astype(int)).max())))))
    c.execute("BEGIN"); c.executemany("INSERT OR REPLACE INTO sig_1m VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows); c.execute("COMMIT")
    c.execute("BEGIN")
    for tr in veh.trips:
        c.execute("INSERT INTO trips(start,end,km,kwh,wh_km,avg_speed,hwy_share,avg_amb,avg_hvac_w,soc_start,soc_end) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (tr["start"], tr["end"], tr["km"], tr["kwh"], tr["wh_km"], tr["avg_speed"], tr["hwy_share"], tr["avg_amb"], tr["avg_hvac_w"], tr["soc_start"], tr["soc_end"]))
    for (ts, ty, sv, de) in veh.events:
        c.execute("INSERT INTO events VALUES(?,?,?,?)", (int(ts), ty, sv, de))
        if ty in CTX_TYPES:
            w = CTX_TYPES[ty]; i0 = max(0, int(ts - t0) - w); i1 = min(n, int(ts - t0) + 5)
            ctx = [(int(t0 + i), *map(int, A[i])) for i in range(i0, i1)]
            blob = zlib.compress(json.dumps(ctx, separators=(",", ":")).encode(), 6)
            c.execute("INSERT OR REPLACE INTO event_ctx VALUES(?,?,?)", (int(ts), ty, blob))
            c.execute("INSERT OR REPLACE INTO event_ctx_v2 VALUES(?,?,?)", (int(ts), ty, blob))
        if ty in ("DTC_SET", "WARN_BATT_OVERTEMP"):
            src = "ivi-hmi" if ty == "WARN_BATT_OVERTEMP" else ("body-tpms" if "yre" in de else "power-mgr" if "oltage" in de else "qnx-thermal")
            c.execute("INSERT INTO logs VALUES(?,?,?)", (int(ts), src, ("WARN " if ty == "DTC_SET" else "CRIT ") + de))
    # service-log breadcrumbs from the degraded pump (WARN lines, one per 20 s below 8 L/min)
    low = np.where((veh.flow > 0.3) & (veh.flow < 8.0))[0]
    for i in low[::20]: c.execute("INSERT INTO logs VALUES(?,?,?)", (int(t0 + i), "qnx-thermal", f"WARN coolant pump flow low ({veh.flow[i]:.1f} L/min)"))
    c.execute("COMMIT"); sqlfeatures.rebuild_log_index(c)
    c.execute("INSERT OR REPLACE INTO meta VALUES('t0',?),('n',?)", (str(t0), str(n))) if False else None
    c.execute("INSERT OR REPLACE INTO meta VALUES('t0',?)", (str(t0),)); c.execute("INSERT OR REPLACE INTO meta VALUES('end',?)", (str(t0 + n),))
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)"); c.execute("VACUUM"); c.close()

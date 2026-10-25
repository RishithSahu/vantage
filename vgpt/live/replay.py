"""A recorded vehicle as a DATA SOURCE: the data lives somewhere else, arrives at the application, and is stored by the application as it arrives.

    recorded history (read-only)  --- what the vehicle sent, second by second --->  Runtime (the ONE writer)  --->  the application's own database  --->  the assistant answers from it

Until now the application opened a finished database and a clock decided which part of it the assistant could see: the future was already there, only hidden.  Here the application starts with the history
up to the moment of the replay (the weeks it has "already stored") and NOTHING after it.  The source then does what the vehicle does:

  wire     (a vehicle with a profile: the four generated vehicles)  every second of the recording is sent as raw bus traffic of THIS vehicle - CAN / CAN-FD frames, SOME/IP notifications, Android VHAL
           parcelables, GNSS sentences, camera metadata, UDS diagnostic reports, QNX log lines - by the same profile-driven emitter the virtual car used (`vgpt/world/emit.py`).  The runtime decodes it with
           the vehicle's own mappings and stores it (`vgpt/live/runtime.py`): signals, events the vehicle reports, and the events the ingest derives itself (hard braking, a tyre going low, a warm battery),
           trips and charging sessions, the bus monitor, source health.  The assistant never sees anything that has not come through that path.
  records  (a recording with no profile: the Case Study 5 files)  rows of the recording are delivered to the database at the moment they were recorded - the application still only knows what has arrived.

The schedule (real time unless the person moves the Time speed slider) says which second of the recording is "now"; the application's clock follows the data that has actually arrived, so it can never run
ahead of what is stored.  Nothing here fabricates a value: every value sent is one the recording holds.

    cut_after(conn, t)            the history as it stood at second t (everything later is removed from the working copy)
    Recorded(path, profile)       what the vehicle sent in second t
    WireSource / RecordSource     threads that deliver the recording on a schedule;  headless:  WireSource.deliver(t0, t1)"""
import re
import sqlite3
import string
import threading
import time
from pathlib import Path

from ..protocols import nmea
from ..world.emit import Emitter
from .runtime import Runtime

# every time-indexed table of a vehicle history and the column that places a row in time
TIMED = (("sig_1s", "ts"), ("sig_1m", "ts"), ("lk_1s", "ts"), ("lk_1m", "ts"), ("gps_1s", "ts"), ("ext_1m", "ts"), ("cam_1m", "ts"), ("events", "ts"), ("logs", "ts"), ("event_ctx", "ts"), ("event_ctx_v2", "ts"),
         ("event_ctx_x", "ts"), ("event_thumb", "ts"), ("event_thumb_v2", "ts"), ("source_log", "ts"), ("trips", "end"), ("trip_ext", "start"), ("charges", "start"), ("vday", "ts"))
CORE_MAP = (("speed", 0, 10.0), ("battery_soc", 1, 10.0), ("battery_temperature", 3, 10.0), ("coolant_flow", 4, 10.0), ("ambient_temperature", 5, 10.0), ("hvac_power", 6, 1.0), ("motor_temperature", 7, 10.0))
CHUNK = 600


def _tables(conn): return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def cut_after(conn, t):
    """The history as it stood at second `t`: everything recorded after it is removed from this (working) copy.  The odometer reading in the vehicle's meta is rewound to where it was."""
    have = _tables(conn); t = int(t)
    try:
        odo = conn.execute("SELECT odo FROM ext_1m WHERE ts<=? AND odo IS NOT NULL ORDER BY ts DESC LIMIT 1", (t,)).fetchone()
        if odo is None: odo = conn.execute("SELECT odo FROM vday WHERE ts<=? ORDER BY ts DESC LIMIT 1", (t,)).fetchone()
    except sqlite3.Error: odo = None
    conn.execute("BEGIN")
    for tb, col in TIMED:
        if tb in have: conn.execute(f"DELETE FROM {tb} WHERE {col}>?", (t,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES('end',?)", (str(t),))
    if "agg_1d" in have and "agg_1d_state" in have:                                       # the daily tier knows nothing of what came after `t`: days from the cut day on are dropped, earlier days stay valid
        try:
            tz = int(float(dict(conn.execute("SELECT k, v FROM meta WHERE k='env.tz_h'").fetchall()).get("env.tz_h", 0)) * 3600); day = (t + tz) // 86400
            conn.execute("DELETE FROM agg_1d WHERE d >= ?", (day,)); hi = conn.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0] if "sig_1m" in have else None
            conn.execute("UPDATE agg_1d_state SET through = MIN(through, ?), sig_max_ts = MIN(sig_max_ts, ?)", (day - 1, int(hi) if hi is not None else t))
        except sqlite3.Error: pass
    if odo is not None and odo[0] is not None: conn.execute("INSERT OR REPLACE INTO meta VALUES('veh.odo_end',?)", (str(odo[0] / 10.0 if odo[0] > 1e6 else odo[0]),))
    conn.execute("COMMIT")
    if "logs_fts" in have:
        try: conn.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')")
        except sqlite3.Error: pass
    try: conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error: pass


def invert(template, text):
    """The fields of an event whose text was made from `template` ('12 V supply dipped to {min_v:.1f} V for {duration_ms} ms'): the inverse of the profile's own detail template.  None when it does not fit."""
    parts = []; names = []
    for lit, name, spec, conv in string.Formatter().parse(template):
        parts.append(re.escape(lit))
        if name: parts.append(f"(?P<{name}>.+?)"); names.append(name)
    m = re.fullmatch("".join(parts), str(text).strip())
    if not m: return None
    out = {}
    for k, v in m.groupdict().items():
        if k.endswith("_text"): out[k] = v
        else:
            try: out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError: out[k] = v
    return out


class Recorded:
    """What the recorded vehicle sent in each second, read back from its stored history (read-only)."""
    def __init__(self, path, profile):
        self.c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10, check_same_thread=False); self.p = profile; self.have = _tables(self.c)
        self.concept = {sid: concept for sid, concept in self.c.execute("SELECT sid, concept FROM sig_catalog")} if "sig_catalog" in self.have else {}
        self.rules = {r.type: r for r in profile.events if r.origin.startswith("someip:")}                       # events the vehicle itself reports as messages
        self._a = self._b = -1; self._sig = {}; self._lake = {}; self._gps = {}; self._ev = {}; self._log = {}; self._cam = {}
        self.has_uds = any(s.type == "uds" for s in profile.sources.values()); self.has_gnss = any(s.type == "gnss" for s in profile.sources.values())

    # ------------------------------------------------------------------------------------------------ chunked reads (one second is never a query)
    def _load(self, a):
        a = a // CHUNK * CHUNK; b = a + CHUNK; c = self.c; self._a, self._b = a, b
        self._sig = {r[0]: r[1:] for r in c.execute("SELECT * FROM sig_1s WHERE ts>=? AND ts<?", (a, b))} if "sig_1s" in self.have else {}
        lake = {}
        if "lk_1s" in self.have:
            for sid, concept in self.concept.items():
                for ts, v in c.execute("SELECT ts, v FROM lk_1s WHERE sid=? AND ts>=? AND ts<?", (sid, a, b)): lake.setdefault(ts, {})[concept] = v
        self._lake = lake
        self._gps = {r[0]: r[1:] for r in c.execute("SELECT ts, lat, lon, hdg, spd, hdop FROM gps_1s WHERE ts>=? AND ts<?", (a, b))} if "gps_1s" in self.have else {}
        self._cam = {r[0]: r[1:] for r in c.execute("SELECT ts, block_avg, sharp_avg, luma_avg, lane_avg FROM cam_1m WHERE ts>=? AND ts<?", (a - 60, b))} if "cam_1m" in self.have else {}
        self._ev = {}; self._log = {}
        for ts, ty, sev, de in c.execute("SELECT ts, type, sev, detail FROM events WHERE ts>=? AND ts<? ORDER BY rowid", (a, b)): self._ev.setdefault(ts, []).append((ty, sev, de))
        for ts, src, msg in c.execute("SELECT ts, src, msg FROM logs WHERE ts>=? AND ts<? ORDER BY rowid", (a, b)): self._log.setdefault(ts, []).append((src, msg))

    def _at(self, t):
        if not (self._a <= t < self._b): self._load(t)

    def state(self, t):
        """The canonical readings of second t ({concept: value}); empty when the vehicle sent nothing (a parked car sleeps)."""
        self._at(t); row = self._sig.get(t); lake = self._lake.get(t)
        if row is None and not lake: return {}
        st = dict(lake or {})
        if row is not None:
            for concept, i, scale in CORE_MAP:
                if row[i] is not None: st[concept] = row[i] / scale
            mode = row[8]
            if mode is not None: st["_ignition"] = 1.0 if mode in (1, 2) else 0.0; st["_charging_active"] = 1.0 if mode == 3 else 0.0
        cam = self._cam.get(t // 60 * 60)                                                                                         # the camera's own report is a per-minute summary: it is sent every second the car is awake
        if cam is not None and row is not None and row[8] in (1, 2):
            for concept, v in zip(("camera_blockage", "camera_sharpness", "camera_brightness", "lane_confidence"), cam):
                if v is not None: st.setdefault(concept, v)
        return st

    def gps(self, t):
        self._at(t); r = self._gps.get(t)
        return None if r is None or r[0] is None or r[1] is None else (r[0] / 1e6, r[1] / 1e6, None if r[2] is None else r[2] / 10.0, None if r[3] is None else r[3] / 10.0, None if r[4] is None else r[4] / 10.0)

    def logs(self, t): self._at(t); return self._log.get(t, [])

    def wire_events(self, t, st):
        """The events the vehicle itself REPORTS in second t, as event dicts for the emitter: diagnostic codes (a UDS report), driver-assistance and supervisor events (SOME/IP), CAN error frames.
        Events the ingest derives from the signals (hard braking, a low tyre, a warm battery ...) are not sent: the application derives them itself, as it would in the car."""
        self._at(t); out = []
        for ty, sev, de in self._ev.get(t, []):
            if ty in ("DTC_SET", "DTC_CLEAR") and self.has_uds and de: out.append({"type": ty, "code": str(de).split(" ")[0]})
            elif ty == "CAN_ERROR": out.append({"type": ty})
            elif ty in self.rules:
                f = invert(self.rules[ty].detail, de)
                if f is not None:
                    f.setdefault("speed_kmh", int(round(st.get("speed", 0) or 0))); out.append(dict(f, type=ty))
        return out

    def next_data(self, t):
        """The first second at or after `t` in which the recording holds anything (the source jumps over the silence in between)."""
        best = None
        for tb in ("sig_1s", "gps_1s", "events", "logs"):
            if tb not in self.have: continue
            r = self.c.execute(f"SELECT MIN(ts) FROM {tb} WHERE ts>=?", (int(t),)).fetchone()[0]
            if r is not None and (best is None or r < best): best = r
        return best

    def span(self):
        r = self.c.execute("SELECT MIN(ts), MAX(ts) FROM sig_1s").fetchone(); return r if r and r[0] is not None else (None, None)

    def close(self):
        try: self.c.close()
        except sqlite3.Error: pass


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- the sources
class Source(threading.Thread):
    """Delivers a recording to the application on a schedule.  `sched` is a Clock holding the replay plan (real time by default); `self.t` is the last second that has ARRIVED."""
    kind = "records"

    def __init__(self, sched, begin):
        super().__init__(daemon=True, name=f"source-{self.kind}"); self.sched = sched; self.begin = int(begin); self.t = self.begin; self.stop_flag = False; self.laps = sched.laps; self.lag = 0.0
        self.err = None; self.t_start = time.time(); self.rate = 0.0; self._rt = (time.monotonic(), self.t)

    def now(self): return float(self.t)

    def deliver(self, upto, deadline=None): raise NotImplementedError
    def restart(self): raise NotImplementedError
    def publish(self): pass
    def close(self): pass

    def run(self):
        try:
            while not self.stop_flag:
                target = int(self.sched.schedule_now())
                if self.sched.laps != self.laps: self.laps = self.sched.laps; self.restart(); continue                       # the recording has played to its end: the history is as it was at the start, and it plays again
                if target <= self.t: time.sleep(0.02); continue
                self.t = self.deliver(target, time.perf_counter() + 0.25); self.publish(); self.lag = target - self.t
                if self.lag > 3: self.sched.rebase(self.t)                                                                 # the ingest cannot keep up with the speed asked for: the schedule slows to what actually arrives
                m = time.monotonic()
                if m - self._rt[0] >= 1.0: self.rate = 0.7 * self.rate + 0.3 * (self.t - self._rt[1]) / (m - self._rt[0]); self._rt = (m, self.t)
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"
        finally: self.close()

    def stop(self):
        self.stop_flag = True
        if self.is_alive() and threading.current_thread() is not self: self.join(timeout=5.0)

    def status(self): return {"kind": self.kind, "t": self.t, "lag_s": round(self.lag, 1), "speed": round(self.rate, 1), "error": self.err}


class WireSource(Source):
    """A vehicle with a profile: every second goes out as the raw bus traffic of that vehicle, into the runtime, which stores it."""
    kind = "wire"

    def __init__(self, sched, src_path, work_path, profile, begin, monitor=True, runtime=True):
        """`runtime=False`: the traffic is only produced, for `sink` to carry elsewhere (the car side of vgpt/link.py: the ingest runs on Vantage)."""
        super().__init__(sched, begin); self.src, self.work, self.p = src_path, work_path, profile; self.monitor = monitor; self.rec = Recorded(src_path, profile); self.em = Emitter(profile)
        self.log_fmt = next((s.raw.get("format", "slog2info") for s in profile.sources.values() if s.type == "log"), "slog2info"); self.body = next((s.raw.get("channel", "") for s in profile.sources.values() if s.id == "body"), "")
        self.msgs = 0; self._fresh = True; self.rt = Runtime(work_path, profile=profile, monitor=monitor) if runtime else None; self.sink = self.rt.feed if runtime else None

    def _messages(self, sec, full=False):
        st = self.rec.state(sec); out = []
        if st: out += self.em.tick(sec, st, full)
        for k, ev in enumerate(self.rec.wire_events(sec, st)):
            ts = sec + 0.3 + 0.01 * k
            if ev["type"] == "CAN_ERROR":
                from ..protocols import canframes as cf
                out += [(ts + 0.01 * j, "can", cf.Frame(ts + 0.01 * j, 0, b"", error_frame=True, error_class=0x8, channel=self.body)) for j in range(3)]
            else: out += self.em.event(ts, ev)
        for k, (src, msg) in enumerate(self.rec.logs(sec)):
            ts = sec + 0.6 + 0.01 * k; sev = "critical" if msg.startswith("CRIT") else "error" if msg.startswith("ERR") else "warning" if msg.startswith("WARN") else "info"
            out.append((ts, "log", self.em.log_line(ts, src, msg, sev, self.log_fmt)))
        g = self.rec.gps(sec)
        if g is not None and self.rec.has_gnss:
            lat, lon, hdg, spd, hdop = g
            out.append((sec + 0.2, "gnss", nmea.encode_rmc(sec + 0.2, lat, lon, spd or 0.0, hdg or 0.0, valid=True)))
            if sec % 5 == 0: out.append((sec + 0.25, "gnss", nmea.encode_gga(sec + 0.25, lat, lon, quality=1, sats=9 + sec % 3, hdop=hdop if hdop is not None else 0.9)))
        out.sort(key=lambda r: r[0]); return out

    def deliver(self, upto, deadline=None):
        """Make every second after `self.t` up to `upto` arrive, jumping over the stretches in which the recording holds nothing.  Returns the last second that has arrived."""
        t = self.t
        while t < upto:
            nxt = self.rec.next_data(t + 1)
            if nxt is None or nxt > upto: t = upto; break                                                             # nothing was sent in between: the time simply passes
            gap = nxt > t + 1 or self._fresh
            if nxt > t + 1: t = nxt - 1
            t += 1; self._fresh = False
            for ts, bus, payload in self._messages(t, gap): self.sink(ts, bus, payload); self.msgs += 1
            if deadline is not None and time.perf_counter() > deadline and t < upto: break
        self.t = t; return t

    def publish(self):
        if self.rt is not None: self.rt.publish()

    def restart(self):
        if self.rt is None: self.em = Emitter(self.p); self.t = self.begin; self._fresh = True; return                       # the receiving side restarts its own storage
        self.rt.close(); c = sqlite3.connect(self.work, isolation_level=None, timeout=30)
        try: cut_after(c, self.begin)
        finally: c.close()
        self.rt = Runtime(self.work, profile=self.p, monitor=self.monitor); self.sink = self.rt.feed; self.em = Emitter(self.p); self.t = self.begin; self._fresh = True

    def close(self):
        self.rec.close()
        try:
            if self.rt is not None: self.rt.close()
        except Exception: pass

    def status(self):
        d = super().status(); s = self.rt.stats()
        with self.rt.lock: src = self.rt.sources()
        d.update(msgs=s["msgs"], by_bus=s["by_bus"], rows=s["rows_1hz"], lake_rows=s["lake_rows"], decode_us=s["decode_us"], dropped=s["dropped_late"] + s["router_dropped"], sources=[{"id": x["id"], "name": x["name"], "type": x["type"], "state": x["state"], "frames": x["frames"]} for x in src], new_trips=s["new_trips"], new_charges=s["new_charges"])
        return d


class RecordSource(Source):
    """A recording with no vehicle profile: its rows are delivered to the database at the moment they were recorded."""
    kind = "records"

    def __init__(self, sched, src_path, work_path, begin):
        super().__init__(sched, begin); self.src, self.work = src_path, work_path; self.rows = 0
        self.c = sqlite3.connect(Path(work_path).as_uri(), uri=True, isolation_level=None, timeout=30, check_same_thread=False); self._attach()

    def _attach(self):
        self.c.execute("ATTACH DATABASE ? AS o", (Path(self.src).as_uri() + "?mode=ro",))                                  # the recording is only ever read
        self.mine = {r[0] for r in self.c.execute("SELECT name FROM main.sqlite_master WHERE type='table'")}; self.orig = {r[0] for r in self.c.execute("SELECT name FROM o.sqlite_master WHERE type='table'")}

    def deliver(self, upto, deadline=None):
        lo = self.t; c = self.c; c.execute("BEGIN")
        try:
            for tb, col in TIMED:
                if tb not in self.mine or tb not in self.orig: continue
                if tb == "trips":                                                                                           # a drive in progress is shown as far as it has gone
                    c.execute("DELETE FROM main.trips WHERE start>?", (self.begin,))
                    c.execute("INSERT OR REPLACE INTO main.trips(id,start,end,km,kwh,wh_km,avg_speed,hwy_share,avg_amb,avg_hvac_w,soc_start,soc_end) SELECT id,start,MIN(end,?),km,kwh,wh_km,avg_speed,hwy_share,avg_amb,avg_hvac_w,soc_start,soc_end FROM o.trips WHERE start>? AND start<=?", (upto, self.begin, upto))
                    continue
                cur = c.execute(f"INSERT OR REPLACE INTO main.{tb} SELECT * FROM o.{tb} WHERE {col}>? AND {col}<=?", (lo, upto)); self.rows += max(cur.rowcount, 0)
            c.execute("INSERT OR REPLACE INTO main.meta VALUES('end',?)", (str(upto),)); c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK"); raise
        if "logs" in self.mine and "logs_fts" in self.mine:
            try: c.execute("INSERT INTO main.logs_fts(rowid, msg, src) SELECT rowid, msg, src FROM main.logs WHERE ts>? AND ts<=?", (lo, upto))
            except sqlite3.Error: pass
        self.t = upto; return upto

    def restart(self):
        self.c.execute("DETACH DATABASE o")
        try: cut_after(self.c, self.begin)
        finally: self._attach()
        self.t = self.begin

    def close(self):
        try: self.c.close()
        except sqlite3.Error: pass

    def status(self):
        d = super().status(); d.update(rows=self.rows, sources=[{"id": "recorder", "name": "Recorded drive", "type": "records", "state": "ok" if not self.err else "error", "frames": self.rows}]); return d

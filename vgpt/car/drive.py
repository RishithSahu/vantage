"""Where the car's data comes from: a simulated drive, or a recorded dataset played back.  Both give the same thing: `step()` returns the bus traffic of the next second of the car's life, as
(time, bus, payload) in the car's own time, produced by walking the vehicle's profile in reverse (vgpt/world/emit.py).

    SimDrive       the virtual car (vgpt/live/car.py): a route with traffic lights and bends, a battery or an engine, temperatures, tyres, a camera, faults that can be injected, a driver you can command
    DatasetDrive   a recorded vehicle history (a data/*.db): every second of the recording is what THIS car would have put on its buses.  Any dataset can be played through any car: the car carries the
                   quantities it has a signal for (an electric history played through a petrol car sends the speed, temperatures and tyres, and nothing about a battery it does not have)
The dataset's own time is moved so that the first second played is now."""
import os
import sqlite3
import time

from ..live.car import FAULTS, VirtualCar
from ..live.replay import Recorded
from ..live.sim import Simulation
from ..protocols import canframes as cf, nmea
from ..world.emit import Emitter
from .dashcam import Dashcam


def local_tz_hours():
    return -(time.timezone if time.daylight == 0 else time.altzone) / 3600.0


class SimDrive:
    kind = "simulation"

    def __init__(self, profile, seed=7, t0=None, tz_h=None, home=(12.9716, 77.5946), camera=True, start="driving", init=None):
        t0 = float(int(t0 if t0 is not None else time.time()))
        self.p = profile; self.car = VirtualCar(profile, seed=seed, t0=t0, tz_h=local_tz_hours() if tz_h is None else tz_h, home=home, init=init)
        self.sink_buf = []; self.sim = Simulation(None, self.car, sink=lambda ts, bus, payload: self.sink_buf.append((ts, bus, payload)))
        self.cam = Dashcam() if (camera and any(s.type == "camera" for s in profile.sources.values())) else None; self._last_pic = -1e9
        if start == "driving": self.sim.control("start_trip")
        self.t0 = t0

    @property
    def t(self): return self.car.t

    def step(self):
        self.sink_buf = []
        with self.sim.lock: self.sim.step()
        out, self.sink_buf = self.sink_buf, []
        if self.cam is not None and self.car.t - self._last_pic >= 2.0 and self.car.awake():
            self._last_pic = self.car.t; jb = self.cam.jpeg(self.car, quality=60)
            if jb: out.append((self.car.t + 0.7, "cam", jb))
        return out

    def control(self, cmd, **kw): return self.sim.control(cmd, **kw)
    def inject(self, name, **kw): return self.sim.inject(name, **kw)
    def clear(self, name=None): return self.sim.clear(name)
    def faults(self): return {k: {"label": v[0], "kind": v[1], "text": v[2]} for k, v in FAULTS.items()}

    def skip(self, seconds):
        """Advance the car's clock without sending anything (a night passes): used to move the car to a time of day."""
        with self.sim.lock:
            for _ in range(int(seconds)): self.car.step(); self.car.drain()

    def describe(self):
        with self.sim.lock: return self.car.describe()


class DatasetDrive:
    kind = "dataset"

    def __init__(self, profile, path, start=None, t0=None, loop=True):
        self.loop = loop; self.p = profile; self.path = path; self.rec = Recorded(path, profile); self.em = Emitter(profile); self.log_fmt = next((s.raw.get("format", "slog2info") for s in profile.sources.values() if s.type == "log"), "slog2info")
        self.body = next((s.raw.get("channel", "") for s in profile.sources.values() if s.id == "body"), "")
        lo, hi = self.rec.span()
        if lo is None: raise ValueError("this dataset holds no second-by-second history")
        self.ended = False; self.lo, self.hi = int(lo), int(hi); self.cursor = int(start if start is not None else self._first_trip(lo)); self.shift = float(int(t0 if t0 is not None else time.time())) - self.cursor; self._fresh = True; self.sent_s = 0

    def _first_trip(self, lo):
        try:
            c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            try:
                r = c.execute("SELECT MIN(start) FROM trips WHERE start>=?", (int(lo),)).fetchone()
                return max(int(lo), int(r[0]) - 120) if r and r[0] else int(lo)
            finally: c.close()
        except sqlite3.Error: return int(lo)

    @property
    def t(self): return self.cursor + self.shift

    def step(self):
        """The next second of the recording.  Silence in the recording is passed over quickly (a car that sleeps sends nothing): the cursor jumps to the next second that holds data."""
        nxt = self.rec.next_data(self.cursor + 1)
        if (nxt is None or nxt > self.hi) and not self.loop: self.shift += 1.0; self.sent_s += 0; self.ended = True; return []                  # the recording has ended and the car stays parked: its clock runs on, nothing is sent
        if nxt is None or nxt > self.hi: self.cursor = self.lo; self._fresh = True; nxt = self.cursor          # the recording has ended: it starts again
        gap = nxt > self.cursor + 1
        if gap and nxt - self.cursor > 5: self.shift += 0; self.cursor = nxt - 1                           # do not wait through hours of nothing
        else: self.cursor = max(self.cursor + 1, nxt) - 1
        self.cursor += 1; sec = self.cursor; self.sent_s += 1
        st = self.rec.state(sec); out = []
        if st: out += self.em.tick(sec, st, gap or self._fresh)
        for k, ev in enumerate(self.rec.wire_events(sec, st)):
            ts = sec + 0.3 + 0.01 * k
            if ev["type"] == "CAN_ERROR": out += [(ts + 0.01 * j, "can", cf.Frame(ts + 0.01 * j, 0, b"", error_frame=True, error_class=0x8, channel=self.body)) for j in range(3)]
            else: out += self.em.event(ts, ev)
        for k, (src, msg) in enumerate(self.rec.logs(sec)):
            ts = sec + 0.6 + 0.01 * k; sev = "critical" if msg.startswith("CRIT") else "error" if msg.startswith("ERR") else "warning" if msg.startswith("WARN") else "info"
            out.append((ts, "log", self.em.log_line(ts, src, msg, sev, self.log_fmt)))
        g = self.rec.gps(sec)
        if g is not None and self.rec.has_gnss:
            lat, lon, hdg, spd, hdop = g
            out.append((sec + 0.2, "gnss", nmea.encode_rmc(sec + 0.2, lat, lon, spd or 0.0, hdg or 0.0, valid=True)))
            if sec % 5 == 0: out.append((sec + 0.25, "gnss", nmea.encode_gga(sec + 0.25, lat, lon, quality=1, sats=9 + sec % 3, hdop=hdop if hdop is not None else 0.9)))
        self._fresh = False; sh = self.shift
        out = [(ts + sh, bus, self._shifted(payload, sh)) for ts, bus, payload in out]; out.sort(key=lambda r: r[0]); return out

    @staticmethod
    def _shifted(payload, sh):
        if isinstance(payload, cf.Frame): payload.ts += sh
        elif isinstance(payload, dict) and "timestamp" in payload: payload["timestamp"] = int(payload["timestamp"] + sh * 1e9)
        return payload

    def end_state(self):
        """What the recording says about its last moment, for a car that carries on from it: {tz_h, home, soc, odometer, charging}."""
        c = self.rec.c; out = {}
        try:
            meta = dict(c.execute("SELECT k, v FROM meta WHERE k IN ('env.tz_h','env.lat','env.lon','veh.odo_end')").fetchall())
            out["tz_h"] = float(meta.get("env.tz_h", 0)); out["odometer"] = float(meta["veh.odo_end"]) if "veh.odo_end" in meta else None
            if "env.lat" in meta and "env.lon" in meta: out["home"] = (float(meta["env.lat"]), float(meta["env.lon"]))
            row = c.execute("SELECT soc, mode FROM sig_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (self.hi,)).fetchone()
            if row: out["soc"] = (row[0] / 10.0) if row[0] is not None else None; out["charging"] = (row[1] == 3)
        except sqlite3.Error: pass
        return out

    def control(self, cmd, **kw):
        if cmd == "seek": self.cursor = max(self.lo, min(int(kw.get("t", self.lo)), self.hi)); self._fresh = True
        else: raise KeyError(cmd)
        return self.describe()

    def inject(self, name, **kw): raise KeyError("a recorded dataset cannot be changed")
    def clear(self, name=None): return self.describe()
    def faults(self): return {}
    def describe(self): return {"t": self.t, "dataset_t": self.cursor, "progress": round((self.cursor - self.lo) / max(self.hi - self.lo, 1), 4), "span": [self.lo, self.hi], "dataset": os.path.basename(self.path)}
    def close(self): self.rec.close()

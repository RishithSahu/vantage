"""The simulation loop: virtual car -> the profile's bus emitter -> the live runtime, on a clock that runs in real time or faster.

    sim = Simulation.from_db("data/live_aster.db"); sim.start()          # continues the history in that database, in real time
    sim.set_speed(30)                                                    # 30 simulated seconds per real second
    sim.control("start_trip"); sim.inject("coolant_pump")

Every message the runtime receives is wire-format traffic of THIS vehicle (CAN / CAN-FD frames, SOME/IP notifications, VHAL parcelables, UDS reports, log lines, camera metadata, NMEA) produced by walking
the profile's bindings in reverse (vgpt/world/emit.py); nothing reaches the database except through the profile router.  Replace `sink` and the same traffic goes over UDP to a runtime in another process."""
import heapq
import sqlite3
import threading
import time

from ..protocols import nmea
from ..vehicle.profile import Profile
from ..world.emit import Emitter
from .car import FAULTS, VirtualCar
from .runtime import Runtime


def car_from_db(db, profile=None, seed=7):
    """A virtual car that continues the history in `db`: same vehicle, place, odometer, charge, temperatures, tyres - and the clock starts where the record ends."""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10); meta = dict(c.execute("SELECT k, v FROM meta").fetchall())
    p = profile or Profile.load(meta.get("vehicle.id", "aster_ev"))
    end = int(float(meta.get("end") or c.execute("SELECT MAX(ts) FROM sig_1s").fetchone()[0] or 1_790_000_000)); init = {}
    row = c.execute("SELECT soc, batt, amb, motor FROM sig_1s WHERE ts<=? AND soc IS NOT NULL ORDER BY ts DESC LIMIT 1", (end,)).fetchone()
    if row: init.update(battery_soc=row[0] / 10.0, battery_temperature=(row[1] or 150) / 10.0, ambient_temperature=(row[2] or 150) / 10.0, motor_temperature=(row[3] or 150) / 10.0)
    if "veh.odo_end" in meta: init["odometer"] = float(meta["veh.odo_end"])
    try:
        x = c.execute("SELECT tp_fl, tp_fr, tp_rl, tp_rr, v12 FROM ext_1m WHERE ts<=? AND tp_fl IS NOT NULL ORDER BY ts DESC LIMIT 1", (end,)).fetchone()
        if x: init.update(tyre_pressure_fl=x[0] / 10.0, tyre_pressure_fr=x[1] / 10.0, tyre_pressure_rl=x[2] / 10.0, tyre_pressure_rr=x[3] / 10.0)
        if x and x[4]: init["aux_battery_voltage"] = x[4] / 100.0
    except sqlite3.Error: pass
    home = work = None
    try:
        pl = {k: (la, lo) for k, la, lo in c.execute("SELECT kind, lat, lon FROM places WHERE kind IN ('home','work')")}
        home, work = pl.get("home"), pl.get("work")
    except sqlite3.Error: pass
    if home is None and "env.lat" in meta: home = (float(meta["env.lat"]), float(meta["env.lon"]))
    tz = float(meta.get("env.tz_h", 0.0)); c.close()
    return VirtualCar(p, seed=seed, t0=end, tz_h=tz, init=init, home=home, work=work)


class Simulation(threading.Thread):
    def __init__(self, runtime, car, emitter=None, speed=1.0, sink=None):
        super().__init__(daemon=True, name="virtual-car"); self.rt, self.car = runtime, car; self.em = emitter or Emitter(car.p); self.sink = sink or runtime.feed
        self.speed = float(speed); self.paused = False; self.stop_flag = False; self.lock = threading.RLock(); self.heap = []; self.seq = 0; self.steps = 0; self.msgs = 0; self.budget = 0.0
        self.log_fmt = next((s.raw.get("format", "slog2info") for s in car.p.sources.values() if s.type == "log"), "slog2info"); self.has_gnss = any(s.type == "gnss" for s in car.p.sources.values())
        self.t_wall = 0.0; self.lag = 0.0; self.step_ms = 0.0

    @classmethod
    def from_db(cls, db, speed=1.0, seed=7, sink=None, runtime=None):
        car = car_from_db(db, seed=seed); rt = runtime or Runtime(db, profile=car.p); return cls(rt, car, speed=speed, sink=sink)

    # ---------------------------------------------------------------------------------------------------------- one simulated second
    def step(self):
        car, em = self.car, self.em; state = car.step(); sec = int(car.t); out = []
        if state: out += em.tick(sec, state)
        ev, lg = car.drain()
        for (ts, e) in ev: self.seq += 1; heapq.heappush(self.heap, (max(ts, float(sec)), self.seq, "event", e))
        for (ts, src, msg, sev) in lg: self.seq += 1; heapq.heappush(self.heap, (max(ts, float(sec)), self.seq, "log", (src, msg, sev)))
        while self.heap and self.heap[0][0] < sec + 1.0:
            ts, _, kind, payload = heapq.heappop(self.heap); ts = min(ts + 0.02, sec + 0.99)
            if kind == "event": out += self._event(ts, payload)
            else: out.append((ts, "log", em.log_line(ts, payload[0], payload[1], payload[2], self.log_fmt)))
        if self.has_gnss and state:
            out.append((sec + 0.2, "gnss", nmea.encode_rmc(sec + 0.2, car.lat, car.lon, car.v * 3.6, car.heading, valid=True)))
            if sec % 5 == 0: out.append((sec + 0.25, "gnss", nmea.encode_gga(sec + 0.25, car.lat, car.lon, quality=1, sats=9 + sec % 3, hdop=0.9 + (sec % 7) * 0.05)))
        out.sort(key=lambda r: r[0])
        for ts, bus, payload in out: self.sink(ts, bus, payload)
        self.msgs += len(out); self.steps += 1; return len(out)

    def _event(self, ts, ev):
        if ev["type"] == "CAN_ERROR":                                                                                   # an error frame on the body bus: the router counts it and raises the event itself
            from ..protocols import canframes as cf
            chan = next((s.raw.get("channel", "") for s in self.car.p.sources.values() if s.id == "body"), "")
            return [(ts + 0.01 * k, "can", cf.Frame(ts + 0.01 * k, 0, b"", error_frame=True, error_class=0x8, channel=chan)) for k in range(3)]
        return self.em.event(ts, ev)

    # ---------------------------------------------------------------------------------------------------------- the clock
    def run(self):
        last = time.monotonic()
        while not self.stop_flag:
            now = time.monotonic(); dt = now - last; last = now
            if self.paused or self.speed <= 0: time.sleep(0.05); continue
            self.budget += dt * self.speed; n = int(self.budget); self.budget -= n; n = min(n, 600)
            if n:
                t0 = time.perf_counter()
                with self.lock:
                    for _ in range(n): self.step()
                    if self.rt is not None: self.rt.publish()
                self.step_ms = 0.8 * self.step_ms + 0.2 * (time.perf_counter() - t0) * 1000.0 / n
                self.lag = max(0.0, self.budget + (dt * self.speed - n))
            time.sleep(0.02)

    def stop(self):
        self.stop_flag = True

    # ---------------------------------------------------------------------------------------------------------- controls (thread-safe)
    def set_speed(self, x): self.speed = max(0.0, min(float(x), 600.0))
    def pause(self, on=True): self.paused = bool(on)

    def skip_to(self, t, max_s=129600):
        """Run the clock forward at full speed (at most 36 h).  A parked car sleeps through the time and is recorded as a sleeping car is: one wake-up a minute."""
        with self.lock:
            for _ in range(int(min(max(t - self.car.t, 0), max_s))): self.step()
            if self.rt is not None: self.rt.publish()

    def control(self, cmd, **kw):
        with self.lock:
            c = self.car
            if cmd == "start_trip": c.start_trip(kw.get("reverse"))
            elif cmd == "park": c.park(); c.set(ignition=False) if c.v < 0.2 else None
            elif cmd == "ignition": c.set(ignition=bool(kw.get("on")))
            elif cmd == "target": c.set(ignition=True, drive="manual", target_kmh=float(kw.get("kmh", 0)), brake=0.0)
            elif cmd == "brake": c.set(brake=float(kw.get("level", 0)))
            elif cmd == "plug": c.set(plug=kw.get("kind", ""), ignition=False if kw.get("kind") else c.c["ignition"], charge_to=float(kw.get("to", 100 if kw.get("kind") == "ac" else 80)))
            elif cmd == "hvac": c.set(hvac=float(kw.get("level", 0.5)))
            elif cmd == "lka": c.set(lka=bool(kw.get("on")))
            elif cmd == "rain": c.set(rain=None if kw.get("level") in (None, "auto") else int(kw["level"]))
            else: raise KeyError(cmd)
            return c.describe()

    def inject(self, name, **kw):
        with self.lock: self.car.inject(name, **kw); return self.car.describe()

    def clear(self, name=None):
        with self.lock: self.car.clear(name); return self.car.describe()

    def status(self):
        with self.lock: d = self.car.describe()
        d.update(sim={"speed": self.speed, "paused": self.paused, "steps": self.steps, "msgs": self.msgs, "step_ms": round(self.step_ms, 2), "t": self.car.t}, fault_catalogue={k: {"label": v[0], "kind": v[1], "text": v[2]} for k, v in FAULTS.items()})
        return d

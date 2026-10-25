"""The car.  A vehicle model (its profile: which buses it has and what each carries), a drive (a simulated route or a recorded dataset) and a pipe.  `Car.start()` runs the car in real time: every message goes
onto the pipe at the moment it would appear on the bus.  The car knows nothing about Vantage except that a pipe is there; it sends what its buses carry, in the form each bus carries it.

    car = Car(model_dir, writer, dataset=None)           dataset: a data/*.db to play instead of a simulated drive
    car.start(); car.control("ignition", on=False); car.inject("tyre_leak"); car.set_speed(10); car.stop()

`set_speed(x)`: the car's clock runs x times faster than the wall clock (1 = real time).  At high speeds the pipe is the limit: when Vantage cannot keep up the writes block, so the car slows to the pace Vantage
can take, nothing is dropped."""
import os
import threading
import time

from ..vehicle.profile import Profile
from .drive import DatasetDrive, SimDrive
from .transmit import Transmitter, make_vin
from ..pipe.wire import LinkClosed

ANNOUNCE_EVERY_S = 60.0


class Car(threading.Thread):
    def __init__(self, model_dir, writer, dataset=None, seed=7, speed=1.0, start="driving", t0=None, camera=True, channel_seed=None, vin_seed=None, instant=False):
        super().__init__(daemon=True, name="car")
        self.model_dir = model_dir; self.profile = Profile.load(model_dir); self.writer = writer; self.dataset = dataset; self.speed = float(speed); self.stop_flag = threading.Event(); self.paused = False
        self.tx = Transmitter(writer, self.profile, seed=channel_seed if channel_seed is not None else seed + 1000)
        self.instant = bool(instant and dataset); self.seed = seed; self.camera = camera; self.instant_carried_on = False
        if self.instant:                                                                                  # the history is loaded by Vantage in one copy; the car carries on from its last ten minutes, in the history's own calendar
            self.drive = DatasetDrive(self.profile, dataset, loop=False); self.drive.cursor = max(self.drive.lo, self.drive.hi - 600); self.drive.shift = 0.0; self.drive._fresh = True
        else: self.drive = DatasetDrive(self.profile, dataset, t0=t0) if dataset else SimDrive(self.profile, seed=seed, t0=t0, camera=camera, start=start)
        self.vin = make_vin(vin_seed if vin_seed is not None else self.profile.id + str(seed)); self.sent = 0; self.error = ""; self.started_wall = None; self._wall0 = None; self._car0 = None; self.lag = 0.0
        self.link_closed = False; self._announced = -1e9; self.steps = 0
        self.doip = next((s.raw for s in self.profile.sources.values() if s.type == "uds" and s.raw.get("transport") == "doip"), None)
        self.has_uds = any(s.type == "uds" for s in self.profile.sources.values()); self.has_vhal = any(s.type == "vhal" for s in self.profile.sources.values())

    # ---------------------------------------------------------------------------------------------------------- the clock
    def _rebase(self):
        self._wall0 = time.monotonic(); self._car0 = self.drive.t

    def set_speed(self, x):
        self.speed = max(0.05, min(float(x), 600.0)); self._rebase()

    def pause(self, on=True):
        self.paused = bool(on)
        if not on: self._rebase()

    def _wait(self, ts):
        if self.speed > 40: return
        target = self._wall0 + (ts - self._car0) / self.speed
        while not self.stop_flag.is_set():
            d = target - time.monotonic()
            if d <= 0: break
            time.sleep(min(d, 0.25))
            if self.paused: return

    # ---------------------------------------------------------------------------------------------------------- the loop
    def run(self):
        self.started_wall = time.time()
        try:
            self.writer.control(proto=1, kind="vehicle data link", clock="calendar")
            if self.instant: self.writer.control(history={"path": os.path.abspath(self.dataset), "name": os.path.basename(self.dataset)})
            self._rebase()
            while not self.stop_flag.is_set():
                if self.paused: time.sleep(0.05); continue
                if self.drive.t - self._announced >= ANNOUNCE_EVERY_S: self._announce()
                if self.instant and getattr(self.drive, "ended", False): self._carry_on()
                out = self.drive.step(); self.steps += 1
                for ts, bus, payload in out:
                    if self.stop_flag.is_set(): break
                    self._wait(ts)
                    self.tx.send(ts, bus, payload); self.sent += 1
                self._wait(self.drive.t + 1.0 if self.speed <= 40 else self.drive.t)
                self.lag = max(0.0, time.monotonic() - (self._wall0 + (self.drive.t - self._car0) / self.speed)) if self.speed <= 40 else 0.0
        except LinkClosed: self.link_closed = True; self.error = "Vantage closed the link"
        except Exception as e: self.error = f"{type(e).__name__}: {e}"
        finally:
            try: self.writer.close()
            except Exception: pass

    def _carry_on(self):
        """The recorded history is finished (instant load): the car does not freeze in the last second of it - a car that was charging goes on charging, a parked one stays parked, and it can be driven - from the history's own calendar, place and charge."""
        end = self.drive.end_state(); t0 = self.drive.t
        init = {k: v for k, v in (("battery_soc", end.get("soc")), ("odometer", end.get("odometer"))) if v is not None}
        self.drive = SimDrive(self.profile, seed=self.seed + 1, t0=t0, tz_h=end.get("tz_h"), home=end.get("home") or (12.9716, 77.5946), camera=self.camera, start="parked", init=init)
        if end.get("charging"): self.drive.control("plug", kind="ac", to=100)
        self.instant_carried_on = True; self._rebase()

    def _announce(self):
        self._announced = self.drive.t
        if self.doip is not None: self.tx.announce_vin_doip(self.drive.t, self.vin, int(str(self.doip.get("tester", "0x0E80")), 16), int(str(self.doip.get("entity", "0x1001")), 16), self.doip.get("eid", "0050c2a1b2c3"), self.doip.get("gid", "0050c2a1b2c3"))
        if self.has_uds and self.doip is None or self.has_vhal: self.tx.announce_vin(self.drive.t, self.vin, uds=self.has_uds and self.doip is None, vhal=self.has_vhal)

    def stop(self, join=True):
        self.stop_flag.set()
        if join and self.is_alive() and threading.current_thread() is not self: self.join(timeout=5.0)

    # ---------------------------------------------------------------------------------------------------------- what the person can do to the car
    def control(self, cmd, **kw): return self.drive.control(cmd, **kw)
    def inject(self, name, **kw): return self.drive.inject(name, **kw)
    def clear(self, name=None): return self.drive.clear(name)

    def status(self):
        d = {"model": self.profile.specs.get("veh.model", self.profile.id), "drive": self.drive.kind, "t": self.drive.t, "speed_x": self.speed, "paused": self.paused, "sent": self.sent, "records": self.writer.records, "bytes": self.writer.bytes,
             "by_medium": dict(self.tx.counts), "channels": len(self.tx.chan), "error": self.error, "link_closed": self.link_closed, "alive": self.is_alive(), "lag_s": round(self.lag, 2), "vin": self.vin, "instant": self.instant}
        try: d["state"] = self.drive.describe()
        except Exception: d["state"] = {}
        return d

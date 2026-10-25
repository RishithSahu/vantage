"""The live car inside the head-unit simulator: the virtual car drives, the profile's buses carry its signals, the runtime stores them, and the same assistant answers questions about them.

    LiveCar(db)          Runtime (the one writer of `db`) + Simulation (virtual car on a clock) + helpers for the page
    snapshot()           what the Live page shows: car state, sources, bus monitor, signals, statistics, controls
    core()               the cluster's numbers straight from the runtime (no database round trip), merged into the telemetry
    Dashcam              (moved to vgpt/car/dashcam.py: the camera is the car's)

Everything the page can press goes through `Simulation.control / inject / clear / set_speed / skip_to`; the car cannot be given a value, only a situation."""
import math
import time

try: import cv2                                                                                    # only the picture of the virtual car needs it
except ImportError: cv2 = None
import numpy as np

from vgpt.car.dashcam import Dashcam
from vgpt.live.car import FAULTS
from vgpt.live.sim import Simulation, car_from_db


class LiveCar:
    def __init__(self, db, speed=1.0, seed=7, start_local_hour=7.4):
        self.db = db; self.sim = Simulation.from_db(db, speed=speed, seed=seed); self.rt = self.sim.rt; self.car = self.sim.car; self.cam = Dashcam()
        t0 = self.car.t
        if start_local_hour is not None:                                                                       # the demo starts the next morning: the night is slept through at full speed (and recorded as a parked car is)
            tz = self.car.tz; day = int((t0 + tz) // 86400); target = day * 86400 - tz + start_local_hour * 3600
            if target <= t0 + 600: target += 86400
            self.sim.skip_to(target)
        self.sim.start()

    # ---------------------------------------------------------------------------------------------------------- the clock the rest of the server follows
    def now(self): return self.car.t
    def speed(self): return 0.0 if self.sim.paused else self.sim.speed

    def close(self):
        self.sim.stop(); self.sim.join(timeout=3.0)
        try: self.rt.close()
        except Exception: pass

    # ---------------------------------------------------------------------------------------------------------- numbers for the cluster
    def core(self):
        """Fresh values for the instrument cluster, taken from what the buses just delivered (the database lags by the second being written)."""
        c = self.car; v = {s["concept"]: s["value"] for s in self.rt.signals()}; mode = 3 if c.charge_state == 1 else (0 if not c.c["ignition"] else (2 if (v.get("speed") or 0) > 75 else 1))
        p = v.get("battery_current"); volt = v.get("battery_voltage")
        out = {"live": True, "ok": True, "age_s": 0, "tier": "bus", "sample_ts": int(c.t), "speed": v.get("speed"), "soc": v.get("battery_soc"), "batt_c": v.get("battery_temperature"), "amb_c": v.get("ambient_temperature"), "hvac_w": v.get("hvac_power"),
               "motor_c": v.get("motor_temperature"), "flow": v.get("coolant_flow"), "mode": mode, "power_kw": (p * volt / 1000.0) if (p is not None and volt is not None) else None}
        if c.ev is False:
            out.update(soc=v.get("fuel_level"), power_kw=(v.get("fuel_rate") or 0.0) * 9.0 if c.c["ignition"] else 0.0, batt_c=v.get("engine_oil_temperature"))
        if v.get("reported_range") is not None: out["range_km"] = v["reported_range"]
        return {k: x for k, x in out.items() if x is not None}

    # ---------------------------------------------------------------------------------------------------------- the Live page
    def snapshot(self, full=True):
        with self.sim.lock: car = self.car.describe()
        snap = self.rt.snapshot(bus_rows=full); sim = self.sim.status()
        snap.update(car=car, sim=sim["sim"], faults=sim["fault_catalogue"], collectors=[])
        if full:
            for r in snap.get("bus", []): r["raw"] = r["raw"][:96]
        return snap

    def camera_state(self):
        c = self.car; blocked = c.blockage > 0.5; deg = c.sharp < 0.5
        state = "BLOCKED" if blocked else ("DEGRADED" if deg else "OK")
        return {"state": state, "reason": "lens obstruction" if blocked else ("low sharpness" if deg else ""), "cond": "rain" if c.rain_now else "", "blocked_frac": round(c.blockage, 3), "sharp_ratio": round(c.sharp, 2), "luma": round(c.luma, 0),
                "lane_conf": round(c.lane, 2), "low_light": c.lux < 400.0, "lost": None, "fps": 8.0, "ms": 0.0, "source": "Virtual forward camera (procedural)", "fault": "cover" if blocked else "none", "error": ""}


__all__ = ["LiveCar", "Dashcam", "FAULTS", "car_from_db"]

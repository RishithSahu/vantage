"""The virtual car: a physical model that produces the canonical state of a vehicle second by second, and the events a vehicle would raise.

It knows only concepts (speed, battery_soc, tyre_pressure_fl ...) - never a bus.  How a particular vehicle publishes them is the profile pack's business (vgpt/world/emit.py walks the bindings in reverse and
produces CAN / CAN-FD frames, SOME/IP notifications, VHAL parcelables, UDS reports, log lines, camera metadata).  So the same car drives an electric hatchback, a second EV with another bus layout, or a diesel truck.

  kinematics    a driver (route mode: speed limits, junction bends, traffic lights, stops; manual mode: a target speed) -> speed, acceleration, steering, yaw, lateral acceleration along a real route
  powertrain    battery-electric: road load + drivetrain/regen efficiency -> pack power, voltage sag, current, state of charge, motor torque / rpm, AC and DC charging with a taper
                diesel: road load -> engine power, gear and rpm, load, fuel rate and level; exhaust heat, boost, rail pressure, the particulate filter filling and regenerating (logged), AdBlue use
                petrol: the same drivetrain; manifold pressure and air flow, lambda, fuel trims, catalyst heat, ignition timing
  thermal       pack and motor temperature with a coolant loop (pump flow follows demand), brake discs, tyre temperature and pressure (ideal gas, leaks), 12 V battery
  assistance    lane keeping (engages on motorway, disengages for the reasons a real system gives), lead-vehicle distance, camera quality
  faults        injectable and removable: coolant pump degradation, tyre leak, blocked camera, weak 12 V, ECU watchdog resets, brown-out, unexpected shutdown, body-bus dropout, speed / outside-temperature sensor
                disagreement, emergency braking, a dashboard range revision, an arbitrary diagnostic code

    car = VirtualCar(profile, seed=1, t0=1792364400, init={"battery_soc": 80, "odometer": 52557})
    car.set(ignition=True, drive="route")         # or car.set(ignition=True, drive="manual", target_kmh=90)
    state = car.step()                            # one simulated second -> {concept: value}; car.events / car.logs hold what was raised"""
import math
import random

from .road import Road, haversine

G = 9.80665
SPEC_TYRE_KPA = 230.0
FAULTS = {
    "coolant_pump": ("Coolant pump degraded", "persistent", "The battery coolant pump delivers about a third of its flow: the pack heats up on fast charging and long drives, then the thermal warning and the pump code follow."),
    "tyre_leak": ("Slow puncture, rear left", "persistent", "The rear-left tyre loses pressure steadily; the low-pressure warning and its diagnostic code follow."),
    "camera_blocked": ("Front camera blocked", "persistent", "Mud or a cover over the lens: blockage rises, sharpness collapses, lane keeping drops out."),
    "weak_12v": ("Weak 12 V battery", "persistent", "The 12 V battery sags when the car wakes and at rest; low-voltage warnings and brown-outs appear."),
    "speed_conflict": ("Android speed reads 6 % high", "persistent", "The infotainment speed and the CAN speed disagree: a sensor conflict is recorded."),
    "ecu_reset": ("ADAS controller watchdog reset", "once", "The ADAS domain controller resets and lane keeping disengages ('system limit')."),
    "brownout": ("12 V brown-out", "once", "A voltage dip and an infotainment reset."),
    "shutdown": ("Unexpected shutdown", "once", "12 V supply loss: every signal stops for ~12 s, then the gateway restarts and logs P0562 and U0155."),
    "can_burst": ("Body-bus dropout", "once", "A loose body-CAN connector: tyre, light, rain, 12 V and odometer signals vanish for ~30 s, error frames are counted, U0140 is stored."),
    "ambient_glitch": ("Dashboard shows -40 C outside", "once", "The Android outside temperature reads -40 C for ten seconds while CAN is fine: a sensor conflict with no diagnostic code."),
    "aeb": ("Automatic emergency braking", "once", "A lead vehicle stops suddenly: AEB acts, hard braking is recorded."),
    "range_glitch": ("Dashboard range revised -15 %", "persistent", "The dashboard range estimate drops 15 % for twenty minutes while the energy use is normal."),
    "overheat": ("Pack heater stuck on", "persistent", "An extra 4 kW of heat in the pack; the temperature climbs until the over-temperature warning."),
}
DTC_FOR = {"coolant_pump": "P1A00", "tyre_leak": "C0752", "camera_blocked": "C1A10", "weak_12v": "P0562"}
ENGINE_GEARS = (14.0, 9.2, 6.4, 4.5, 3.2, 2.35, 1.7, 1.2, 0.86)


def _clip(x, lo, hi): return lo if x < lo else hi if x > hi else x


class VirtualCar:
    def __init__(self, profile, seed=1, t0=1_790_000_000, tz_h=0.0, init=None, road=None, home=None, work=None):
        self.p = profile; sp = profile.specs; self.rng = random.Random(seed); self.t = float(t0); self.tz = float(tz_h) * 3600.0
        self.ev = profile.spec["vehicle"].get("powertrain", "bev") not in ("diesel", "petrol", "gasoline", "ice"); self.tank = float(profile.specs.get("veh.tank_l", 90.0))
        self.kind = "bev" if self.ev else ("petrol" if str(profile.spec["vehicle"].get("powertrain", "")).lower() in ("petrol", "gasoline") else "diesel")
        self.cap = float(sp.get("veh.cap_kwh", 60.0)); self.soh = float(sp.get("veh.soh", 100.0)); self.r_wheel = float(sp.get("veh.wheel_radius_m", 0.33)); self.gear = float(sp.get("veh.gear_ratio", 9.0)); self.wheelbase = float(sp.get("veh.wheelbase_m", 2.7))
        self.steer_ratio = float(sp.get("veh.steering_ratio", 14.5)); self.nominal_v = float(sp.get("veh.nominal_voltage", 380)); self.top_kmh = float(sp.get("veh.rated_max_speed", 160))
        self.home_kw = float(sp.get("veh.home_charger_kw", 7.4)); self.tyre_spec = (float(sp.get("veh.tyre_spec_front", SPEC_TYRE_KPA)), float(sp.get("veh.tyre_spec_rear", SPEC_TYRE_KPA)))
        self.mass = (1850.0 if self.ev else 6500.0) + 150.0; self.cda = 0.62 if self.ev else 3.4; self.crr = 0.010 if self.ev else 0.0075; self.ncell = round(96 * self.nominal_v / 380.0)
        home = home or (51.5074, -0.1278); work = work or (home[0] + 0.12, home[1] - 0.2)
        self.road = road or Road(home, work, seed=seed + 11); self.s = 0.0; self.rev = False
        from .. import signal_registry as R
        self.bound = {b.concept for b in profile.bindings}
        init = dict(init or {}); g = lambda k, d: float(init.get(k, d))
        self.v = 0.0; self.odo = g("odometer", 12000.0); self.soc = g("battery_soc", 82.0); self.fuel = g("fuel_level", 80.0)
        self.t_amb_mean = g("ambient_temperature", 15.0) - 4.0 * math.sin(2 * math.pi * (self._hour() - 9.0) / 24.0); self.amb = g("ambient_temperature", 15.0)
        self.t_batt = g("battery_temperature", self.amb + 1.0); self.t_motor = g("motor_temperature", self.amb + 2.0); self.t_oil = self.t_motor; self.t_cabin = self.amb
        self.pc = [g("tyre_pressure_fl", self.tyre_spec[0]), g("tyre_pressure_fr", self.tyre_spec[0]), g("tyre_pressure_rl", self.tyre_spec[1]), g("tyre_pressure_rr", self.tyre_spec[1])]
        self.t_tyre = self.amb; self.t_brake = [self.amb] * 2; self.aux = g("aux_battery_voltage", 12.6)
        self.egt = self.amb; self.cat = self.amb; self.soot = g("dpf_soot_load", 12.0); self.regen = 0; self.regen_left = 0; self.def_lvl = g("def_level", 88.0); self.stft = 0.0                 # the parts a combustion engine adds to rpm and fuel
        self.ewma_e = 165.0 * 0.05; self.ewma_d = 0.05; self.a_lon = 0.0; self.steer = 0.0; self.yaw = 0.0; self.a_lat = 0.0
        self.c = {"ignition": False, "drive": "manual", "target_kmh": 0.0, "brake": 0.0, "plug": "", "hvac": 0.5, "lka": True, "charge_to": 100.0, "rain": None}
        self.faults = {}; self.events = []; self.logs = []; self.dtcs = set()
        self.charge_state = 0; self.charge_kw = 0.0; self.charge_kind = ""; self.t_off = -1e9; self.blackout = 0.0; self.dwell = 0.0; self.red_until = {}
        self.adas_on = False; self.adas_since = 0.0; self.adas_next = 0.0; self.thermal_warned = False; self.surprise = 0.0; self.pending_dtc = None; self.pending_dtcs = None; self.dtc_at = None; self.resume = None; self.aeb_until = 0.0; self.glitch_aux = None; self.load = 0.0; self.fuel_rate = 0.0; self.p_loss = 0.0; self.lux = 20000.0; self.luma = 120.0; self.lead = 80.0; self.rain_now = 0; self.cloud = 0.6; self.route_active = False; self.arrived = False
        self.trip_km = 0.0; self.trip_kwh = 0.0; self.gear_i = 0; self.rpm = 0.0; self.engine_p = 0.0; self.p_batt = 0.0; self.p_fric = 0.0; self.i_batt = 0.0; self.v_batt = self.voc(); self.hvac_w = 0.0
        self.flow = 0.0; self.lane = 0.9; self.blockage = 0.02; self.sharp = 1.2; self.glitch_until = 0.0; self.alive = True; self.torque = 0.0; self.motor_rpm = 0.0; self.lat, self.lon = self.road.position(0.0)
        self.heading = self.road.heading(0.0); self.accel_pedal = 0.0; self.brake_pedal = 0.0; self.wheel = [0.0] * 4
        self.cvt = str(sp.get("veh.cvt", "0")).lower() in ("1", "true", "yes"); self.rng_body = random.Random(seed * 7919 + 13); self.door_until = 0.0; self.belt_at = None; self.ign_prev = False; self.prev_steer = 0.0; self.steer_rate = 0.0; self.door_next = None

    # ---------------------------------------------------------------------------------------------------------- controls
    def set(self, **kw):
        for k, v in kw.items():
            if k not in self.c: raise KeyError(k)
            self.c[k] = v
        if kw.get("ignition") is True and not self.alive: self.alive = True
        if kw.get("ignition") is False: self.route_active = False

    def start_trip(self, reverse=None):
        """Ignition on and drive the route to its end (reverse: B -> A).  After arriving the car parks and switches off."""
        if reverse is not None and reverse != self.rev: self.rev = reverse; self.s = 0.0
        elif self.s > self.road.length - 60.0: self.rev = not self.rev; self.s = 0.0
        self.c.update(ignition=True, drive="route", plug=""); self.route_active = True; self.arrived = False; self.dwell = 0.0; self.trip_km = 0.0; self.trip_kwh = 0.0; self.charge_state = 0

    def park(self): self.c.update(drive="manual", target_kmh=0.0, brake=0.0); self.route_active = False

    def inject(self, name, **kw):
        if name not in FAULTS: raise KeyError(name)
        kind = FAULTS[name][1]; t = self.t
        if name == "coolant_pump": self.faults[name] = {"factor": kw.get("factor", 0.30)}
        elif name == "tyre_leak": self.faults[name] = {"wheel": kw.get("wheel", 2), "kpa_min": kw.get("kpa_min", 4.0)}
        elif name == "camera_blocked": self.faults[name] = {}
        elif name == "weak_12v": self.faults[name] = {}
        elif name == "speed_conflict": self.faults[name] = {"gain": kw.get("gain", 1.06)}
        elif name == "overheat": self.faults[name] = {"kw": kw.get("kw", 4.0)}
        elif name == "range_glitch": self.faults[name] = {"until": t + 1200.0, "since": t}
        elif name == "ecu_reset":
            self.events.append((t, {"type": "ECU_RESET", "ecu": "ADAS domain controller", "cause": "watchdog timeout"})); self.logs.append((t, "adas_ctrl", "ERR heartbeat to the gateway lost, restarting", "error")); self.logs.append((t + 7, "adas_ctrl", "WARN boot complete, calibration pending", "warning"))
            if self.adas_on: self._disengage("system limit", t + 2)
            self.adas_next = t + self.rng.uniform(45, 100)
        elif name == "brownout":
            vmin = self.rng.uniform(8.2, 10.4); ms = int(self.rng.uniform(120, 400)); self.events.append((t, {"type": "VOLTAGE_DIP", "min_v": round(vmin, 1), "duration_ms": ms})); self.logs.append((t, "power_mgr", f"WARN 12V rail dipped to {vmin:.1f} V ({ms} ms) during start-up", "warning"))
            self.events.append((t + 1, {"type": "ECU_RESET", "ecu": "infotainment", "cause": "brown-out"})); self.logs.append((t + 1, "ivi", "ERR brown-out reset of the infotainment unit", "error")); self.glitch_aux = (t, t + 2, vmin)
        elif name == "shutdown":
            t1 = t + 1.0                                                                                      # the first second without data: the supply was lost between the last sample and this one
            self.events.append((t1, {"type": "UNEXPECTED_SHUTDOWN", "cause": "12 V supply loss", "last_valid_ms": 120})); self.events.append((t1, {"type": "VOLTAGE_DIP", "min_v": 6.4, "duration_ms": 900}))
            self.logs.append((t1, "power_mgr", "CRIT 12V supply lost, high-voltage contactors opened", "critical")); self.logs.append((t1 + 1, "bms", "ERR contactor open while current flowing", "error"))
            gap = int(self.rng.uniform(9, 16)); self.blackout = t1 + gap; self.resume = {"gap": gap, "t": t1}; self.adas_on = False
        elif name == "can_burst":
            dur = int(self.rng.uniform(20, 40)); n = int(self.rng.uniform(40, 400)); self.faults[name] = {"until": t + dur}
            self.events.append((t, {"type": "CAN_ERROR", "detail": f"CAN error frames on the body bus: {n} in {dur} s (error-passive)"})); self.logs.append((t, "can_gw", f"WARN body bus error counter 128 ({n} error frames)", "warning"))
            self.dtc_at = (t + 20, "U0140")
        elif name == "ambient_glitch": self.faults[name] = {"until": t + 10.0}
        elif name == "aeb":
            sp = self.v * 3.6; self.events.append((t, {"type": "AEB_ACTIVATION", "speed_kmh": int(sp), "decel_g": 0.61})); self.logs.append((t, "adas_ctrl", f"WARN AEB triggered, closing speed {self.v * 0.6:.1f} m/s", "warning"))
            self.aeb_until = t + 6.0; self.lead = 25.0
        if kind == "persistent" and name in DTC_FOR and name not in ("tyre_leak", "weak_12v"): self.pending_dtc = (t + 45.0, DTC_FOR[name])
        return name

    def clear(self, name=None):
        for n in ([name] if name else list(self.faults)): self.faults.pop(n, None)
        if name in DTC_FOR and DTC_FOR[name] in self.dtcs: self.set_dtc(DTC_FOR[name], False)

    def set_dtc(self, code, on=True):
        t = self.t
        if on and code not in self.dtcs: self.dtcs.add(code); self.events.append((t, {"type": "DTC_SET", "codes": [code]}))
        elif not on and code in self.dtcs: self.dtcs.discard(code); self.events.append((t, {"type": "DTC_CLEAR", "codes": [code]}))

    # ---------------------------------------------------------------------------------------------------------- environment
    def _hour(self): return ((self.t + self.tz) % 86400.0) / 3600.0

    def _environment(self):
        h = self._hour(); self.amb += (self.t_amb_mean + 4.0 * math.sin(2 * math.pi * (h - 9.0) / 24.0) + self.rng.gauss(0, 0.05) - self.amb) * 0.01
        r = self.c["rain"]
        if r is None:
            if self.rain_now == 0 and self.rng.random() < 0.0002: self.rain_now = self.rng.choice((1, 2))
            elif self.rain_now and self.rng.random() < 0.0012: self.rain_now = 0
        else: self.rain_now = int(r)
        sun = max(0.0, math.sin(math.pi * (h - 6.0) / 13.5)) if 6.0 <= h <= 19.5 else 0.0; self.cloud = _clip(self.cloud + self.rng.gauss(0, 0.005), 0.25, 1.0)
        self.lux = max(40.0, 90000.0 * sun * (1.0 - 0.7 * (self.cloud - 0.25)) * (0.35 if self.rain_now else 1.0))

    # ---------------------------------------------------------------------------------------------------------- driver and dynamics
    def _target_speed(self):
        """(cruise speed, required deceleration): the speed the driver wants now and the braking the road ahead (bends, red lights, the destination) already demands."""
        c = self.c; r = self.road; s, rev = self.s, self.rev
        if not c["ignition"] or self.arrived: return 0.0, 0.0
        if c["drive"] == "route": tgt = r.limit_at(s, rev) / 3.6 * 0.97 * (1.0 + 0.02 * math.sin(self.t / 90.0))
        else: tgt = c["target_kmh"] / 3.6
        tgt = min(tgt, self.top_kmh / 3.6); v = self.v; need = 0.0; A = 1.0                              # comfortable deceleration: the speed envelope that still lets the driver brake at A
        def envelope(d, vallow):
            nonlocal tgt, need
            d = max(d - v * 1.2, 0.0)                                                                            # the driver decides once a second: plan for where the car will be a moment from now
            tgt = min(tgt, math.sqrt(vallow * vallow + 2.0 * A * max(d, 0.0)))
            if v > vallow:
                x = (v * v - vallow * vallow) / (2.0 * max(d, 1.0))
                if x >= 2.0: need = max(need, x)                                                       # the envelope is already violated: brake harder
        for look in range(0, 190, 5):                                                                  # bends: lateral acceleration kept near 2 m/s2
            ahead = max(abs(r.curvature(s + look, rev)), abs(r.curvature(s + look + 3, rev))); envelope(look + 2.0, max(math.sqrt(2.0 / max(ahead, 1e-5)), 3.0))
        if c["drive"] == "route":
            for (d, red, p) in r.stops_ahead(s, rev, 260.0):
                key = round(p)
                if key not in self.red_until: self.red_until[key] = self.t + (self.rng.uniform(14, 40) if self.rng.random() < 0.5 else -1.0)
                if self.red_until[key] > self.t: envelope(max(d - 4.0, 0.0), 0.0)
            envelope(max(r.length - s - 3.0, 0.0), 0.0)
        return tgt, need

    def _driver(self):
        c = self.c; tgt, need = self._target_speed(); vmax_acc = (2.5 if self.ev else 1.1)
        if c["brake"] > 0.01: a = -c["brake"] * 8.0
        else:
            err = tgt - self.v; a = _clip(err * (0.45 if err > 0 else 0.7), -2.6, vmax_acc * (1.0 if self.v < 14 else 14.0 / self.v))
            if self.v < 0.3 and tgt < 0.3: a = 0.0
            if self.v < 0.5 and tgt < 0.3 and a < 0: a = -self.v
            a += self.rng.gauss(0, 0.05)
        if c["drive"] == "route" and self.v > 9.0 and self.surprise <= self.t and self.rng.random() < 1.0 / 1800.0: self.surprise = self.t + 3.0 + self.rng.random() * 2.0          # a car cuts in: a short firm braking
        if self.surprise > self.t and self.v > 5.0: a = min(a, -3.4)
        if self.aeb_until > self.t: a = -6.0
        if self.blackout > self.t: a = -0.15
        self.accel_pedal = _clip(a / vmax_acc * 60.0 + 12.0, 0, 100) if a > 0.05 and not c["brake"] else 0.0
        self.brake_pedal = _clip(-a / 8.0 * 100.0, 0, 100) if a < -0.4 or c["brake"] else 0.0
        return a

    def _dynamics(self, a):
        v0 = self.v; self.v = max(0.0, v0 + a); ds = (v0 + self.v) / 2.0; self.a_lon = self.v - v0 + self.rng.gauss(0, 0.02)
        self.s += ds; self.odo += ds / 1000.0; self.trip_km += ds / 1000.0
        if self.s >= self.road.length:
            if self.route_active:
                self.s = self.road.length; self.v = 0.0; self.arrived = True; self.route_active = False; self.c.update(drive="manual", target_kmh=0.0, ignition=False); self.t_off = self.t
            else: self.s = 0.0; self.rev = not self.rev
        kap = self.road.curvature(self.s, self.rev); self.yaw = math.degrees(self.v * kap) + self.rng.gauss(0, 0.03); self.a_lat = self.v * self.v * kap
        self.steer = math.degrees(math.atan(self.wheelbase * kap)) * self.steer_ratio + self.rng.gauss(0, 0.3); self.heading = self.road.heading(self.s, self.rev); self.lat, self.lon = self.road.position(self.s, self.rev)
        w = self.v * 3.6; tw = 1.55 / 2.0 * math.radians(self.yaw) * 3.6
        self.wheel = [w - tw + self.rng.gauss(0, 0.1), w + tw + self.rng.gauss(0, 0.1), w - tw + self.rng.gauss(0, 0.1), w + tw + self.rng.gauss(0, 0.1)]

    # ---------------------------------------------------------------------------------------------------------- powertrain
    def voc(self): return self.ncell * (3.20 + 0.0095 * self.soc)

    def _road_power(self, a):
        v = self.v; f = self.crr * self.mass * G + 0.5 * 1.2 * self.cda * v * v + self.mass * a * 1.04
        return f, f * v

    def _hvac(self):
        dT = abs(21.0 - self.amb); return _clip(self.c["hvac"] * (180.0 * dT + 250.0), 0.0, 5200.0) * (1.0 if self.c["ignition"] else 0.0)

    def _ev(self, a):
        F, Pw = self._road_power(a); self.hvac_w = self._hvac(); aux = 320.0 + self.hvac_w + (240.0 if self.rain_now else 0.0)
        regen_lim = 55000.0 * _clip(self.v / 5.0, 0.0, 1.0)
        if Pw >= 0: p_elec = Pw / 0.90; self.p_fric = 0.0
        else:
            reg = min(-Pw * 0.62, regen_lim); self.p_fric = max(0.0, -Pw - reg / 0.62); p_elec = -reg
        if self.blackout > self.t: p_elec = 0.0; aux = 0.0
        p = p_elec + aux; chg = self.charge_state == 1
        if chg: p = -self.charge_kw * 1000.0 + aux * 0.2
        self.p_batt = p; voc = self.voc(); rint = 0.045 * 60.0 / self.cap; i0 = p / voc; self.v_batt = voc - rint * i0; self.i_batt = p / max(self.v_batt, 50.0)
        self.soc = _clip(self.soc - p * 1.0 / 3.6e6 / (self.cap * self.soh / 100.0) * 100.0, 0.0, 100.0)
        self.torque = F * self.r_wheel / self.gear if self.v > 0.2 or F > 0 else 0.0
        if Pw < 0: self.torque = -min(abs(F) * self.r_wheel / self.gear, 300.0)
        self.motor_rpm = self.v / self.r_wheel * self.gear * 60.0 / (2 * math.pi)
        if self.v > 0.5 and self.v_batt: self.trip_kwh += max(p, -1e9) / 3.6e6
        if self.v > 0.8: self.ewma_e += (max(p, -20000.0) / 3600.0 - self.ewma_e) / 18000.0 * 1.0; self.ewma_d += (self.v / 1000.0 - self.ewma_d) / 18000.0
        self._charging()
        if chg: self.p_loss = 0.04 * self.charge_kw * 1000.0 + self.i_batt ** 2 * rint
        else: self.p_loss = self.i_batt ** 2 * rint
        return p

    def _charging(self):
        c = self.c; plug = c["plug"]
        if self.charge_state == 1:
            lim = min(c["charge_to"], 100.0)
            if not plug or self.soc >= lim - 0.05 or not self.alive:
                self.charge_state = 2 if self.soc >= lim - 0.05 else 0; self.charge_kw = 0.0
            else:
                if plug == "dc":
                    p = 110.0 if self.soc < 50 else (110.0 - (self.soc - 50) * 2.25 if self.soc < 90 else 20.0 - (self.soc - 90) * 1.5)
                    tf = 0.5 if self.t_batt < 5 else (1.0 if self.t_batt < 43 else max(0.3, 1.0 - (self.t_batt - 43) * 0.09)); self.charge_kw = max(3.0, p * tf)
                else: self.charge_kw = self.home_kw * (0.97 if self.soc < 95 else 0.6)
        elif plug and not c["ignition"] and self.charge_state == 0 and self.soc < min(c["charge_to"], 100.0) - 0.5 and self.v < 0.1:
            self.charge_state = 1; self.charge_kind = "DC" if plug == "dc" else "AC"; self.charge_kw = 7.0 if plug == "ac" else 30.0
            self.events.append((self.t, {"type": "CHARGE_START", "kind": self.charge_kind, "kw": self.home_kw if plug == "ac" else 110.0}))
        elif not plug and self.charge_state in (1, 2): self.charge_state = 0; self.charge_kw = 0.0
        elif plug and self.charge_state == 2 and self.soc < min(c["charge_to"], 100.0) - 3: self.charge_state = 0

    def _diesel(self, a):
        F, Pw = self._road_power(a)
        if self.blackout > self.t: Pw = 0.0
        self.p_fric = max(0.0, -Pw) * 0.7; p_eng = max(Pw, 0.0) / 0.88; self.engine_p = p_eng; wheel_w = self.v / self.r_wheel
        run = self.c["ignition"] and self.alive
        gi = 0
        for i, ratio in enumerate(ENGINE_GEARS):
            gi = i
            if wheel_w * ratio * 60 / (2 * math.pi) <= 1750.0: break
        self.gear_i = gi if self.v > 1.0 else 0; self.rpm = max(wheel_w * ENGINE_GEARS[gi] * 60 / (2 * math.pi), 650.0) if run else 0.0
        w = max(self.rpm * 2 * math.pi / 60.0, 1.0); self.torque = _clip(p_eng / w if run else 0.0, 0.0, 800.0)
        self.load = _clip(self.torque / 850.0 * 100.0, 0, 100) if run else 0.0
        self._engine_parts(run)
        self.fuel_rate = _clip(((2.2 + 0.255 * p_eng / 1000.0) * (1.12 if self.regen else 1.0)) if run else 0.0, 0.0, 100.0); self.fuel = _clip(self.fuel - self.fuel_rate / 3600.0 / self.tank * 100.0, 0.0, 100.0)
        self.hvac_w = self._hvac(); self.p_batt = 0.0; self.i_batt = 0.0; self.p_loss = 0.0
        if self.v > 0.8: self.ewma_e += (self.fuel_rate * 10.0 / 3600.0 - self.ewma_e) / 18000.0; self.ewma_d += (self.v / 1000.0 - self.ewma_d) / 18000.0

    def _engine_parts(self, run):
        """What a combustion engine adds to rpm, load and fuel - the same rules the world builder lives by, one second at a time.  A diesel heats its exhaust, fills its particulate filter with soot and burns it off in a
        regeneration cycle (logged, and it burns a little more fuel), and uses AdBlue; a petrol engine heats its catalyst and trims its mixture."""
        ld, t = self.load, int(self.t)
        if self.kind == "diesel":
            if run:
                self.egt += (self.amb + 120.0 + 4.2 * ld + (420.0 if self.regen else 0.0) - self.egt) / 90.0
                if self.regen: self.regen_left -= 1; self.soot = max(self.soot - 0.065, 2.0)
                else: self.soot = min(self.soot + 0.0028 + 0.004 * ld / 100.0, 100.0)
                if not self.regen and self.soot >= 78.0 and self.v * 3.6 > 55.0 and self.regen_left <= 0: self.regen = 1; self.regen_left = 900; self.logs.append((t, "ecm", f"WARN DPF regeneration started (soot load {self.soot:.0f} %)", "warning"))
                elif self.regen and (self.regen_left <= 0 or self.soot <= 3.0): self.regen = 0; self.logs.append((t, "ecm", f"WARN DPF regeneration complete (soot load {self.soot:.0f} %)", "warning"))
                self.def_lvl = max(self.def_lvl - self.fuel_rate / 3600.0 * 0.05 / 20.0 * 100.0, 0.0)             # a 20 L AdBlue tank used at 5 % of the fuel burned
            else:
                self.egt += (self.amb - self.egt) / 1500.0; self.regen = 0; self.regen_left = 0
        else:
            self.cat += ((self.amb + 300.0 + 4.5 * ld + 0.04 * self.rpm if run else self.amb) - self.cat) / (240.0 if run else 900.0); self.stft = 0.97 * self.stft + self.rng.gauss(0, 0.9)

    # ---------------------------------------------------------------------------------------------------------- thermal, tyres, supply
    def _thermal(self):
        f = self.faults; awake = self.c["ignition"] or self.charge_state == 1 or (self.t - self.t_off) < 120.0
        if self.ev:
            load = abs(self.i_batt) / (self.cap / 60.0)
            if awake and self.blackout <= self.t:
                demand = 3.5 + 0.55 * max(0.0, self.t_batt - 24.0) + 0.020 * load + (2.0 if self.charge_kw > 30 else 0.0)
                if self.t_batt > 40.0: demand += 2.0 * (self.t_batt - 40.0)
                flow = _clip(demand, 3.0, 18.0) * (f["coolant_pump"]["factor"] if "coolant_pump" in f else 1.0)
            else: flow = 0.0
            self.flow = flow; heat = self.p_loss + (f["overheat"]["kw"] * 1000.0 if "overheat" in f else 0.0)
            cool = flow * (14.0 * max(0.0, self.t_batt - self.amb - 2.0) + 55.0 * max(0.0, self.t_batt - 30.0)); passive = 6.0 * (self.t_batt - self.amb)
            self.t_batt += (heat - cool - passive) / (190e3 * self.cap / 60.0)
            eq = self.amb + 8.0 + 0.00045 * abs(self.p_batt) * (1.4 if "coolant_pump" in f else 1.0)
            self.t_motor += (eq - self.t_motor) / (240.0 if (self.c["ignition"] and self.v > 0.5) else 900.0)
        else:
            run = self.c["ignition"] and self.alive; eq = (90.0 + 0.06 * self.load) if run else self.amb
            self.t_motor += (eq - self.t_motor) / (300.0 if run else 1500.0); self.t_oil += (self.t_motor + 8.0 - self.t_oil) / 400.0; self.flow = 0.0
            self.t_batt = self.t_motor
        self.t_cabin += ((21.0 if self.c["hvac"] > 0.1 and self.c["ignition"] else self.amb) - self.t_cabin) / 600.0
        fric = self.p_fric
        for i in (0, 1): self.t_brake[i] += (fric * (0.6 if i == 0 else 0.4) / 9000.0 - (self.t_brake[i] - self.amb) * (0.004 + 0.0006 * self.v))
        eq_t = self.amb + (0.16 * self.v * 3.6 if self.v > 0.5 else 0.0); self.t_tyre += (eq_t - self.t_tyre) / (700.0 if eq_t > self.t_tyre else 1800.0)
        if "tyre_leak" in f:
            lk = f["tyre_leak"]; self.pc[lk["wheel"]] = max(60.0, self.pc[lk["wheel"]] - lk["kpa_min"] / 60.0)
        if "weak_12v" in f: self.aux = 11.2 + self.rng.gauss(0, 0.04) if not awake else 12.9 + self.rng.gauss(0, 0.1)
        elif awake: self.aux += (13.8 - self.aux) * 0.2 + self.rng.gauss(0, 0.03)
        else: self.aux += (12.55 - self.aux) * 0.001
        ga = self.glitch_aux
        if ga and ga[0] <= self.t < ga[1]: self.aux = ga[2]
        if self.blackout > self.t: self.aux = 5.0

    def _assist(self):
        f = self.faults; c = self.c; moving = self.v > 3.0
        self.blockage = 0.85 + self.rng.gauss(0, 0.02) if "camera_blocked" in f else _clip(0.02 + self.rng.gauss(0, 0.005) + (0.04 if self.rain_now else 0.0), 0.0, 1.0)
        self.sharp = _clip((1.25 - 0.25 * self.rain_now) * (1.0 - 0.95 * min(1.0, self.blockage / 0.85)) + self.rng.gauss(0, 0.03), 0.0, 5.0)
        base = 0.93 - 0.12 * self.rain_now - (0.0 if self.road.is_motorway(self.s, self.rev) else 0.15)
        self.lane = _clip(base * (1.0 - self.blockage * 0.9) + self.rng.gauss(0, 0.02), 0.0, 1.0) if moving else 0.0
        self.luma = _clip(25.0 * math.log10(self.lux + 1.0) * (1.0 - 0.7 * self.blockage), 5.0, 235.0)
        want = c["lka"] and self.v * 3.6 >= 65.0 and self.lane >= 0.55 and self.road.is_motorway(self.s, self.rev) and self.blackout <= self.t and c["ignition"] and "camera_blocked" not in f
        if self.adas_on:
            why = None
            if c["brake"] > 0.15: why = "driver override"
            elif "camera_blocked" in f: why = "camera blocked"
            elif self.lane < 0.5: why = "lane marking lost"
            elif self.v * 3.6 < 55.0 or not c["lka"]: why = "driver override" if not c["lka"] else "system limit"
            elif self.rng.random() < 1.0 / 2400.0: why = "lane marking lost"
            if why: self._disengage(why, self.t)
        elif want and self.t >= self.adas_next: self.adas_on = True; self.adas_since = self.t
        if self.v > 3.0:
            hw = self.rng.uniform(1.6, 2.4) if self.road.is_motorway(self.s, self.rev) else self.rng.uniform(0.8, 1.6); tgt = _clip(hw * self.v + self.rng.gauss(0, 2), 6.0, 160.0)
            self.lead += (tgt - self.lead) * 0.08
        else: self.lead = 60.0

    def _disengage(self, why, t):
        self.adas_on = False; self.adas_next = t + self.rng.uniform(20, 60)
        self.events.append((t, {"type": "ADAS_DISENGAGE", "reason": why, "speed_kmh": int(self.v * 3.6)}))

    def _supervise(self):
        t = self.t; pd = self.pending_dtc
        if pd and t >= pd[0] and "coolant_pump" in self.faults: self.set_dtc(pd[1]); self.pending_dtc = None
        elif pd and "coolant_pump" not in self.faults and pd[1] == "P1A00": self.pending_dtc = None
        if "tyre_leak" in self.faults and self.pc[self.faults["tyre_leak"]["wheel"]] < 175.0: self.set_dtc("C075" + str(self.faults["tyre_leak"]["wheel"] + 0))
        if "weak_12v" in self.faults and self.aux < 11.6 and t % 90 == 0: self.logs.append((t, "power_mgr", f"WARN 12V battery at {self.aux:.2f} V", "warning"))
        da = self.dtc_at
        if da and t >= da[0]: self.set_dtc(da[1]); self.dtc_at = None
        if not self.ev: pass
        elif self.t_batt > 45.0 and not self.thermal_warned: self.thermal_warned = True; self.logs.append((t, "qnx-thermal", f"CRIT battery temperature warning: pack at {self.t_batt:.1f} C", "critical"))
        elif self.t_batt < 42.0: self.thermal_warned = False
        if self.ev and self.t_batt > 50.0 and t % 30 == 0: self.logs.append((t, "qnx-thermal", "WARN thermal derate, charging power limited", "warning"))
        for n in [n for n, v in self.faults.items() if "until" in v and v["until"] <= t]:
            self.faults.pop(n)
        rs = self.resume
        if rs and self.blackout and t >= self.blackout:
            gap = rs["gap"]; self.events.append((t, {"type": "ECU_RESET", "ecu": "gateway", "cause": "power-on"})); self.logs.append((t, "can_gw", "WARN restart after power loss", "warning"))
            self.logs.append((t + 3, "diag", "WARN stored U0155 after restart", "warning")); self.pending_dtcs = (t + 2, ["P0562", "U0155"]); self.resume = None; self.blackout = 0.0
        pdt = self.pending_dtcs
        if pdt and t >= pdt[0]:
            for code in pdt[1]: self.set_dtc(code)
            self.pending_dtcs = None

    # ---------------------------------------------------------------------------------------------------------- one second
    def step(self):
        """Advance one second.  Returns the canonical state of that second (a channel that is asleep or lost is absent), appends raised events to `self.events` and log lines to `self.logs`."""
        self.t += 1.0; self._environment(); a = self._driver(); self._dynamics(a)
        if self.ev: self._ev(a)
        else: self._diesel(a)
        self._thermal(); self._assist(); self._supervise(); self._body()
        return self.state()

    def _body(self):
        """Doors, belt, steering rate: the body-bus quantities a vehicle publishes.  The driver's door opens a few seconds after the car is switched off and the belt unlatches with it; the belt latches a few seconds after the ignition goes on."""
        ign = bool(self.c["ignition"]); t = self.t
        if ign and not self.ign_prev: self.belt_at = t + 8.0; self.door_next = None
        if not ign and self.ign_prev: self.belt_at = None; self.door_next = t + 6.0
        self.ign_prev = ign
        if self.door_next is not None and t >= self.door_next: self.door_until = t + 12.0; self.door_next = None
        self.steer_rate = self.steer - self.prev_steer; self.prev_steer = self.steer

    def gear_number(self):
        """The gear the selector shows: 0 park / neutral, 1 and up a gear (a battery-electric vehicle has one: 1 while it drives)."""
        if not self.c["ignition"] or self.v < 0.3: return 0
        return 1 if (self.ev or self.cvt) else self.gear_i + 1

    def drain(self):
        ev, lg = self.events, self.logs; self.events, self.logs = [], []; return ev, lg

    def awake(self):
        return self.c["ignition"] or self.charge_state == 1 or (self.t - self.t_off) < 120.0 or int(self.t) % 60 < 3

    def reported_range(self):
        whkm = _clip(1000.0 * self.ewma_e / max(self.ewma_d, 0.02), 110.0, 320.0)
        r = self.soc / 100.0 * self.cap * self.soh / 100.0 / whkm * 1000.0 if self.ev else self.fuel / 100.0 * self.tank / max(_clip(1000.0 * self.ewma_e / max(self.ewma_d, 0.02) / 1000.0, 15.0, 60.0), 1.0) * 100.0
        g = self.faults.get("range_glitch")
        if g:
            x = (self.t - g["since"]) / max(g["until"] - g["since"], 1.0); r *= 1.0 - 0.15 * math.sin(math.pi * _clip(x, 0.0, 1.0)) ** 0.5
        return r

    def state(self):
        if self.blackout > self.t or not self.alive: return {}
        if not self.awake(): return {}
        f = self.faults; rng = self.rng; n = lambda s: rng.gauss(0, s); chg = self.charge_state == 1; drv = bool(self.c["ignition"])
        s = {"_ignition": 1 if drv else 0, "_charging_active": 1 if chg else 0, "speed": self.v * 3.6, "odometer": self.odo, "ambient_temperature": self.amb + n(0.05), "aux_battery_voltage": self.aux, "motor_temperature": self.t_motor,
             "brake_pedal": self.brake_pedal, "accelerator_position": self.accel_pedal, "steering_angle": self.steer, "yaw_rate": self.yaw, "lateral_acceleration": self.a_lat, "longitudinal_acceleration": self.a_lon,
             "tyre_temperature": self.t_tyre, "brake_temperature": max(self.t_brake), "rain": self.rain_now, "light": self.lux, "adas_state": 1 if self.adas_on else 0, "lead_vehicle_distance": self.lead,
             "lane_confidence": self.lane, "camera_blockage": self.blockage, "camera_sharpness": self.sharp, "camera_brightness": self.luma, "brake_pressure": self.p_fric / 1000.0 / 40.0 * 10.0 + self.brake_pedal * 0.2,
             "cabin_temperature": self.t_cabin, "hvac_power": self.hvac_w, "wheel_speed_fl": self.wheel[0], "wheel_speed_fr": self.wheel[1], "wheel_speed_rl": self.wheel[2], "wheel_speed_rr": self.wheel[3]}
        k = (273.15 + self.t_tyre) / (273.15 + 20.0)
        for i, nm in enumerate(("fl", "fr", "rl", "rr")): s["tyre_pressure_" + nm] = self.pc[i] * k + n(0.3)
        if self.ev:
            s.update({"battery_soc": self.soc, "battery_voltage": self.v_batt, "battery_current": self.i_batt, "battery_temperature": self.t_batt, "cell_temperature_max": self.t_batt + 0.9 + 0.012 * abs(self.p_batt) / 1000.0,
                      "cell_temperature_min": self.t_batt - 0.7, "cell_voltage_min": self.v_batt / self.ncell - 0.035 - 0.00008 * abs(self.i_batt), "coolant_flow": self.flow, "charging_power": self.charge_kw if chg else 0.0,
                      "motor_torque": self.torque, "motor_rpm": self.motor_rpm, "reported_range": self.reported_range()})
        else:
            s.update({"engine_rpm": self.rpm, "engine_torque": self.torque, "engine_load": self.load, "fuel_rate": self.fuel_rate, "fuel_level": self.fuel, "engine_oil_temperature": self.t_oil, "barometric_pressure": 100.4 + n(0.05),
                      "fuel_temperature": self.amb + 14.0 if drv else self.amb, "reported_range": self.reported_range()})
            if self.kind == "diesel":
                s.update({"boost_pressure": max(self.load - 12.0, 0.0) * 1.7 + n(0.6) if drv else 0.0, "exhaust_gas_temperature": self.egt, "dpf_soot_load": self.soot, "dpf_regeneration": self.regen, "def_level": self.def_lvl,
                          "fuel_rail_pressure": 28000.0 + 1320.0 * self.load + n(300.0) if drv else 0.0})
            else:
                mapk = 30.0 + 0.62 * self.load + n(0.5) if drv else 100.4; cut = drv and self.fuel_rate < 1.0
                s.update({"intake_manifold_pressure": mapk, "intake_air_temperature": self.amb + (8.0 + 0.1 * self.load if drv else 0.0), "mass_air_flow": 1.5 * self.rpm / 120.0 * 0.85 * 1.19 * mapk / 101.3 if drv else 0.0,
                          "lambda": (1.55 if cut else 0.88 if self.load > 92 else 1.0 + n(0.008)) if drv else 1.0, "short_term_fuel_trim": _clip(self.stft, -12.0, 12.0) if drv and not cut else 0.0, "long_term_fuel_trim": 3.5 + n(0.15),
                          "catalyst_temperature": self.cat + n(2.0), "ignition_timing": 10.0 + 24.0 * (1 - self.load / 100.0) + n(0.4) if drv else 0.0})
        belt = 1 if (drv and self.belt_at is not None and self.t >= self.belt_at) else 0
        s.update({"gear": self.gear_number(), "parking_brake": 1 if (not drv and self.v < 0.2) else 0, "door_state": 1 if self.t < self.door_until else 0, "seatbelt_state": belt, "headlight_state": (1 if (drv and self.lux < 3000.0) else 0),
                  "wiper_state": (self.rain_now if drv else 0), "steering_torque": _clip(0.35 * self.a_lat + self.rng_body.gauss(0, 0.05), -8.0, 8.0), "steering_rate": self.steer_rate, "acc_state": 1 if self.adas_on else 0, "heading": self.heading,
                  "throttle_position": _clip(self.accel_pedal * 0.92 + (3.0 if drv else 0.0), 0.0, 100.0), "transmission_temperature": (self.t_oil + 6.0) if not self.ev else self.t_motor + 4.0})
        if self.ev:
            s.update({"battery_power": self.p_batt / 1000.0, "cell_voltage_max": self.v_batt / self.ncell + 0.035 + 0.00005 * abs(self.i_batt), "inverter_temperature": self.t_motor + 7.0 + 0.0004 * abs(self.p_batt),
                      "charging_voltage": self.v_batt if chg else 0.0, "charging_current": (self.charge_kw * 1000.0 / max(self.v_batt, 50.0)) if chg else 0.0, "motor_power": self.p_batt / 1000.0 * 0.92})
        elif self.kind == "diesel": s.update({"intake_air_temperature": self.amb + (10.0 + 0.12 * self.load if drv else 0.0), "mass_air_flow": 1.9 * self.rpm / 120.0 * 0.9 * 1.19 * (1.0 + self.load / 40.0) if drv else 0.0})
        if "speed_conflict" in f: s["speed@2"] = s["speed"] * f["speed_conflict"]["gain"]
        if "ambient_glitch" in f: s["ambient_temperature@2"] = -40.0
        if "camera_blocked" in f: s["lane_confidence@2"] = max(0.0, self.lane - 0.4)
        if "can_burst" in f:
            for key in ("tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr", "tyre_temperature", "light", "rain", "aux_battery_voltage", "odometer"): s.pop(key, None)
        s["ChargingService.ChargeState.state"] = 1 if chg else (2 if self.charge_state == 2 else 0)
        s["AdasState.SystemState.level"] = 2 if self.adas_on else 0
        return s

    # ---------------------------------------------------------------------------------------------------------- reporting for the UI
    def describe(self):
        return {"t": self.t, "ignition": self.c["ignition"], "drive": self.c["drive"], "route_active": self.route_active, "arrived": self.arrived, "speed": round(self.v * 3.6, 1), "odometer": round(self.odo, 2),
                "soc": round(self.soc, 1), "fuel": round(self.fuel, 1), "pack_c": round(self.t_batt, 1), "motor_c": round(self.t_motor, 1), "amb_c": round(self.amb, 1), "flow": round(self.flow, 1), "power_kw": round(self.p_batt / 1000.0, 1),
                "tyres": [round(p * (273.15 + self.t_tyre) / 293.15, 0) for p in self.pc], "aux_v": round(self.aux, 2), "adas": self.adas_on, "rain": self.rain_now, "lat": round(self.lat, 6), "lon": round(self.lon, 6), "heading": round(self.heading, 0),
                "charge": {"state": self.charge_state, "kw": round(self.charge_kw, 1), "kind": self.charge_kind, "plug": self.c["plug"]}, "progress": round(self.s / max(self.road.length, 1.0), 3), "reverse": self.rev, "range_km": round(self.reported_range(), 0),
                "faults": sorted(self.faults), "dtcs": sorted(self.dtcs), "powertrain": self.kind, "gear": self.gear_i + 1 if not self.ev and self.v > 1 else None, "rpm": round(self.rpm if not self.ev else self.motor_rpm),
                "controls": dict(self.c)}

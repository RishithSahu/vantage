"""Canonical per-second signals of a simulated vehicle, derived day by day from the generator's life (vgpt/datagen_life.py).

The life generator simulates what the physics decides (speed, battery power, state of charge, temperatures, pump flow, trips, charging, weather, ADAS use, camera health).  The platform layer
needs MORE channels than that - steering, yaw, accelerations, wheel speeds, motor torque / rpm, pack voltage / current, cell temperatures, brake pressure, pedals, brake-disc temperature, tyre
pressure and temperature per wheel, 12 V voltage, odometer, lead-vehicle distance, lane / camera quality, the dashboard's range estimate - so they are derived here, by the same physics a car obeys
(bicycle model for steering, wheel radius and gear ratio for rpm, torque = power / angular speed, a lumped thermal mass for the brake discs), with the REAL accelerometer traces of the driving data
supplying the longitudinal and lateral acceleration.  Day-by-day, so a 105-day history never needs more than one day of channels in memory.

    S = Signals(gv, profile); for d in range(S.days): arrays = S.day(d)   # {concept: float32[86400]}, NaN where the signal is not available (parked sensors that sleep, source dropouts)"""
import math

import numpy as np

from . import ice as ICE

G = 9.80665


def _rep(x, k=60, n=86400):
    return np.repeat(np.asarray(x, np.float32), k)[:n]


class Signals:
    def __init__(self, gv, profile, seed=0, faults=None):
        self.gv, self.p, self.days = gv, profile, gv.n // 86400
        self.rng = np.random.default_rng(seed + 777); self.faults = faults
        V = gv.V; self.mass = V["mass"] * 1.1; self.cap = float(gv.life.cap)
        self.wheelbase = float(profile.specs.get("veh.wheelbase_m", 2.7)); self.radius = float(profile.specs.get("veh.wheel_radius_m", 0.33)); self.gear = float(profile.specs.get("veh.gear_ratio", 9.0))
        self.steer_ratio = float(profile.specs.get("veh.steering_ratio", 14.5))
        n = gv.n; self.a_lon = np.zeros(n, np.float32); self.a_lat = np.zeros(n, np.float32)                  # g, from the real accelerometer traces of each trip
        for tr in gv.trips:
            s, L = tr["s"], len(tr["v"]); self.a_lon[s:s + L] = tr["lon"][:L]; self.a_lat[s:s + L] = tr["lat"][:L]
        self.brake_t = float(gv.amb[0]) + 5.0; self.heading = 0.0; self.lead = 60.0; self.lead_on = False; self.reported = self._reported_range()
        self.fine_days = 7; self.ice = ICE.Engine(profile, np.random.default_rng(seed + 4242), float(gv.amb[0]), gv.t0) if ICE.is_ice(profile) else None      # a combustion vehicle: the engine channels replace the electric ones

    # ------------------------------------------------------------------------------------------------ the dashboard's range estimate (per minute)
    def _reported_range(self):
        gv = self.gv; m = gv.n // 60; P = gv.power[: m * 60].reshape(m, 60).astype(np.float64).mean(1); v = gv.speed[: m * 60].reshape(m, 60).astype(np.float64).mean(1); act = (gv.mode[: m * 60].reshape(m, 60) % 3 != 0).any(1) & (v > 3)
        e = np.where(act, P / 60.0, 0.0); d = np.where(act, v / 60.0, 0.0)                                       # kWh and km per minute while driving
        alpha = 1.0 / 300.0; ne, nd, whkm = 160.0 * 0.05, 0.05, np.empty(m)                                    # exponentially weighted consumption over ~5 driving hours, held while parked
        el, dl = e.tolist(), d.tolist()
        for i in range(m):
            if dl[i] > 0.02: ne = (1 - alpha) * ne + alpha * el[i]; nd = (1 - alpha) * nd + alpha * dl[i]
            whkm[i] = float(np.clip(1000.0 * ne / nd, 110.0, 320.0))
        soc = gv.soc[: m * 60].reshape(m, 60).astype(np.float64).mean(1)
        return (soc / 100.0 * self.cap / whkm * 1000.0 * float(self.rng.uniform(0.97, 1.03))).astype(np.float32)

    # ------------------------------------------------------------------------------------------------ one day of canonical channels
    def day(self, d):
        gv, rng = self.gv, self.rng; a, b = d * 86400, (d + 1) * 86400; n = 86400; sl = slice(a, b); m0 = d * 1440
        speed = gv.speed[sl].astype(np.float64); mode = gv.mode[sl]; ign = gv.ign[sl]; drv = (mode == 1) | (mode == 2); awake = mode != 0
        soc = gv.soc[sl].astype(np.float64); P = gv.power[sl].astype(np.float64); batt = gv.batt[sl].astype(np.float64); mm = gv.minute
        al = self.a_lon[sl].astype(np.float64); at = self.a_lat[sl].astype(np.float64); v = speed / 3.6
        out = {}
        out["speed"] = speed.astype(np.float32); out["battery_soc"] = soc.astype(np.float32); out["battery_power"] = P.astype(np.float32); out["battery_temperature"] = batt.astype(np.float32)
        out["coolant_flow"] = gv.flow[sl]; out["ambient_temperature"] = gv.amb[sl]; out["hvac_power"] = gv.hvac[sl]; out["motor_temperature"] = gv.motor[sl]; out["odometer"] = gv.odo[sl]
        out["charging_power"] = gv.chg[sl]; out["regen_power"] = np.where(drv & (P < 0), -P, 0.0).astype(np.float32)
        # pack electrical: 96 cells in series, open-circuit voltage from state of charge, sag with current
        voc = 96 * (3.20 + 0.0095 * soc); cur = P * 1000.0 / voc; volt = voc - 0.045 * cur * (self.cap and 60.0 / self.cap); cur = P * 1000.0 / volt
        out["battery_voltage"] = volt.astype(np.float32); out["battery_current"] = cur.astype(np.float32)
        out["cell_temperature_max"] = (batt + 0.9 + 0.012 * np.abs(P)).astype(np.float32); out["cell_temperature_min"] = (batt - 0.7).astype(np.float32); out["cell_voltage_min"] = (volt / 96.0 - 0.035 - 0.00008 * np.abs(cur)).astype(np.float32)
        # dynamics: real accelerometer + bicycle model
        g_lon = al * G; g_lat = at * G; yaw = np.where(v > 2.0, g_lat / np.maximum(v, 2.0), 0.0); yaw = np.clip(yaw, -0.5, 0.5)
        delta = np.arctan(self.wheelbase * yaw / np.maximum(v, 1.0)); steer = np.degrees(delta) * self.steer_ratio; steer = np.where(v > 1.0, steer, 0.0)
        out["steering_angle"] = steer.astype(np.float32); out["steering_rate"] = np.gradient(steer).astype(np.float32); out["yaw_rate"] = np.degrees(yaw).astype(np.float32)
        out["longitudinal_acceleration"] = g_lon.astype(np.float32); out["lateral_acceleration"] = g_lat.astype(np.float32)
        dv = 0.5 * 1.52 * yaw * 3.6; noise = lambda s: rng.normal(0, 0.1, n)
        for name, off in (("fl", -dv), ("fr", dv), ("rl", -dv), ("rr", dv)):
            out["wheel_speed_" + name] = np.where(speed < 0.5, 0.0, np.maximum(speed + off + noise(0), 0.0)).astype(np.float32)
        rpm = v / self.radius * self.gear * 60.0 / (2 * math.pi); aux = out["hvac_power"].astype(np.float64) / 1000.0 + 0.35
        mech = np.where(drv & (P > 0), np.maximum(P - aux, 0) * 0.9, np.where(drv & (P < 0), P * 0.9, 0.0)); torque = np.where(rpm > 60, mech * 9549.0 / np.maximum(rpm, 60), 0.0)
        out["motor_rpm"] = np.where(drv, rpm, 0.0).astype(np.float32); out["motor_torque"] = np.clip(torque, -450, 450).astype(np.float32); out["motor_power"] = mech.astype(np.float32)
        decel = np.clip(-al, 0, None)
        bp = np.where(drv, np.clip((decel - 0.13) * 380.0, 0, 120) + np.where(decel > 0.025, 3.0 + 60.0 * decel, 0.0), 0.0)
        out["brake_pressure"] = bp.astype(np.float32); out["brake_pedal"] = np.clip(bp * 0.8, 0, 100).astype(np.float32)
        out["accelerator_position"] = np.where(drv, np.clip(np.where(al > 0, 12 + 120 * al, 5 + 3 * np.clip(v / 3, 0, 8) * (P > 0)) + rng.normal(0, 0.6, n), 0, 100), 0.0).astype(np.float32)
        out["brake_temperature"] = self._brake(drv, decel, speed, out["ambient_temperature"])
        # tyres, 12 V, light, rain (per-minute channels of the life generator, repeated)
        pr = mm["pr"][:, m0:m0 + 1440]
        for w, name in enumerate(("fl", "fr", "rl", "rr")): out["tyre_pressure_" + name] = _rep(pr[w])
        out["tyre_temperature"] = gv.life.tyre_t[sl]; out["rain"] = _rep(mm["rain"][m0:m0 + 1440]); out["light"] = _rep(mm["lux"][m0:m0 + 1440]); out["aux_battery_voltage"] = _rep(np.where(mm["act"][m0:m0 + 1440], mm["v12"][m0:m0 + 1440], mm["v12_rest"][min(d, len(mm["v12_rest"]) - 1)]))
        # ADAS and camera
        adas = gv.adas_on[sl]; out["adas_state"] = adas.astype(np.float32)
        out["lead_vehicle_distance"] = self._lead(drv, speed, d)
        out["lane_confidence"] = _rep(np.where(mm["drv"][m0:m0 + 1440], mm["lane"][m0:m0 + 1440], np.nan)); out["camera_blockage"] = _rep(mm["block"][m0:m0 + 1440]); out["camera_sharpness"] = _rep(mm["sharp"][m0:m0 + 1440])
        out["camera_brightness"] = _rep(mm["luma"][m0:m0 + 1440]); out["reported_range"] = _rep(self.reported[m0:m0 + 1440])
        out["cabin_temperature"] = (np.where(ign, 21.5 + 0.12 * (out["ambient_temperature"] - 21.5), out["ambient_temperature"] + 2.0)).astype(np.float32)
        for key in ("_ignition",): out[key] = ign.astype(np.float32)
        out["_charging_active"] = (mode == 3).astype(np.float32)
        out["_awake"] = awake
        self._body(out, mode, drv, ign, speed, yaw)
        if self.ice is not None: self.ice.day(out, d, int(self.gv.t0) + a, mode, d >= self.days - self.fine_days)
        if self.faults is not None: self.faults.shape_day(d, a, out)
        return out

    def _body(self, out, mode, drv, ign, speed, yaw):
        """The channels a car carries besides its drivetrain, derived from when it drives (so a longer, denser history exists for every make, not only those whose database is limited to the drivetrain):
        doors (open for a moment before a drive and after it), seat belt (latched a few seconds into a drive, unlatched after), parking brake (set while parked), headlights (dark or rain), wipers (rain),
        steering torque (from the angle and the speed), a GNSS speed and heading (the heading follows the yaw), a gear (D while driving, P parked; a combustion vehicle overrides it with its own) and the
        throttle."""
        n = len(speed); rng = self.rng; park = ~drv; door = np.zeros(n, np.float32); belt = np.zeros(n, np.float32)
        for s0, s1 in _runs(drv):
            door[max(0, s0 - 25):s0] = 1.0; door[s1:min(n, s1 + 20)] = 1.0; belt[min(n, s0 + 6):s1] = 1.0 if rng.random() > 0.015 else 0.0       # one trip in sixty-odd is made unbelted
            if 30 < s1 - s0 and rng.random() < 0.08: k = s0 + int(rng.integers(20, max(21, s1 - s0))); door[k:k + 3] = 1.0                           # a door opened at a stop
        out["door_state"] = door; out["seatbelt_state"] = belt
        out["parking_brake"] = np.where(mode == 0, 1.0, 0.0).astype(np.float32)
        dark = np.nan_to_num(out["light"].astype(np.float64), nan=50000.0) < 1500.0; rainy = np.nan_to_num(out["rain"].astype(np.float64), nan=0.0) > 0.0
        out["headlight_state"] = (drv & (dark | rainy)).astype(np.float32); out["wiper_state"] = (drv & rainy).astype(np.float32)
        v = speed / 3.6; out["steering_torque"] = np.where(drv, np.clip(out["steering_angle"].astype(np.float64) * (0.012 + 0.0009 * np.minimum(v, 30.0)), -8.0, 8.0) + rng.normal(0, 0.05, n), 0.0).astype(np.float32)
        out["gps_speed"] = np.where(speed < 0.5, 0.0, np.round(speed + rng.normal(0, 0.12, n), 1)).astype(np.float32)
        h = self.heading + np.cumsum(np.nan_to_num(yaw.astype(np.float64)) * (180.0 / math.pi)) ; self.heading = float(h[-1] % 360.0); out["heading"] = (h % 360.0).astype(np.float32)
        out["gear"] = np.where(drv & (speed > 0.5), 1.0, 0.0).astype(np.float32)
        out["humidity"] = np.clip(58.0 - 0.9 * (out["ambient_temperature"].astype(np.float64) - 12.0) + 9.0 * np.nan_to_num(out["rain"].astype(np.float64), nan=0.0) + rng.normal(0, 0.8, n), 12.0, 96.0).astype(np.float32)
        out["charging_voltage"] = np.where(out["charging_power"].astype(np.float64) > 0, out["battery_voltage"].astype(np.float64) + 4.0, 0.0).astype(np.float32)
        out["charging_current"] = np.where(out["charging_power"].astype(np.float64) > 0, out["charging_power"].astype(np.float64) * 1000.0 / np.maximum(out["battery_voltage"].astype(np.float64), 50.0), 0.0).astype(np.float32)
        out["throttle_position"] = np.clip(out["accelerator_position"].astype(np.float64) * 0.92, 0, 100).astype(np.float32)

    def _brake(self, drv, decel, speed, amb):
        """Brake-disc temperature: friction power heats a lumped mass, airflow (grows with speed) and conduction cool it."""
        n = len(speed); T = np.empty(n, np.float64); t = self.brake_t; amb = amb.astype(np.float64); P = np.where(drv & (decel > 0.16), self.mass * (decel - 0.14) * G * speed / 3.6, 0.0)
        Pl, vl, al = P.tolist(), speed.tolist(), amb.tolist()
        for i in range(n):
            t += (Pl[i] * 0.55 / 9000.0) - (6.0 + 0.55 * vl[i] / 3.6 * 3.0) * (t - al[i]) / 9000.0 if (Pl[i] > 0 or t > al[i] + 0.3) else 0.0
            T[i] = t
        self.brake_t = float(t); return T.astype(np.float32)

    def _lead(self, drv, speed, d):
        """Distance to the vehicle ahead: a slowly wandering gap when there is one (motorway most of the time, city less), NaN when the road ahead is free or the car is parked."""
        n = len(speed); out = np.full(n, np.nan, np.float32); rng = self.rng
        for s0, s1 in _runs(drv):
            L = s1 - s0
            if L < 60: continue
            gap = np.empty(L); x = self.lead if self.lead_on else float(rng.uniform(30, 80))
            noise = rng.normal(0, 0.6, L); sp = speed[s0:s1]
            for i in range(L):
                x += 0.02 * (max(15.0, 0.9 * sp[i] / 3.6 * 1.6) - x) + noise[i]; gap[i] = x
            present = np.ones(L, bool)
            for i0, i1 in _runs(rng.random(L // 60 + 1) < 0.25):                                          # stretches with a free road ahead
                present[i0 * 60:i1 * 60] = False
            out[s0:s1] = np.where(present & (sp > 8), np.clip(gap, 4, 250), np.nan); self.lead = float(gap[-1]); self.lead_on = True
        return out


def _runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(np.int8)); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

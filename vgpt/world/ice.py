"""The internal-combustion overlay of the world builder: the same life (trips, weather, driver, faults, stories) lived by a diesel truck or a petrol car instead of a battery-electric car.

The life generator decides WHEN the vehicle drives and how hard (speed, road load).  For an ICE profile this module turns that into what an engine does with it: gear and engine speed, torque and load,
fuel rate, the fuel level (with a refuelling when the tank is nearly empty, written as an event), the coolant and oil temperatures (warm-up, thermostat, cool-down) - and the signals that make a diesel
different from a petrol engine:

    diesel   boost pressure, exhaust gas temperature, rail pressure, the particulate filter filling with soot and burning it off in regeneration cycles (logged), the AdBlue / DEF tank emptying with the fuel burned
    petrol   intake manifold pressure, mass air flow, intake air temperature, lambda (closed loop around 1.0, rich at full load, lean on over-run), short- and long-term fuel trims, catalyst temperature, ignition timing

Only the channels the profile declares are produced.  The battery-electric channels of the life are removed from what the database holds (no state of charge, no pack, no charging session, no regeneration):
the head unit and the assistant decide what the vehicle is from the data it contains, so a combustion database must not carry them.

    E = Engine(profile, rng, amb0, t0);  E.day(out, d, base, mode, keep_1s)    # inside Signals.day: adds the engine channels to `out`
    finish(conn, gv, E)                                                        # after the lake is written: refuel events, logs, no charging, wide tables without battery columns, engine coolant as `motor`
"""
import math

import numpy as np

ICE_POWERTRAINS = ("diesel", "petrol", "gasoline", "ice")
DIESEL = dict(gears=np.array((14.0, 9.2, 6.4, 4.5, 3.2, 2.35, 1.7, 1.2, 0.86)), shift=1750.0, idle=780.0, min_rpm=650.0, tank=200.0, tmax=800.0, tref=850.0, base_l=2.2, per_kw=0.255, cut_l=0.8, redline=3200.0)
PETROL = dict(gears=np.array((14.6, 8.4, 5.6, 4.2, 3.3, 2.7)), shift=2900.0, idle=820.0, min_rpm=720.0, tank=50.0, tmax=250.0, tref=260.0, base_l=0.8, per_kw=0.33, cut_l=0.25, redline=6200.0)


def is_ice(profile):
    return str(profile.vehicle.get("powertrain", "bev")).lower() in ICE_POWERTRAINS


class Engine:
    def __init__(self, profile, rng, amb0, t0):
        sp = profile.specs; self.kind = "diesel" if str(profile.vehicle.get("powertrain")).lower() == "diesel" else "petrol"; self.P = DIESEL if self.kind == "diesel" else PETROL
        self.tank = float(sp.get("veh.tank_l", self.P["tank"])); self.radius = float(sp.get("veh.wheel_radius_m", 0.34)); self.rng = rng; self.t0 = int(t0); self.has = set(profile.concept_names())
        self.fuel = 76.0; self.cool = float(amb0); self.oil = float(amb0); self.egt = float(amb0); self.cat = float(amb0); self.soot = 12.0; self.regen = 0; self.regen_left = 0; self.def_level = 88.0; self.stft = 0.0
        self.refuels = []; self.logs = []; self.cool_min = {}; self.cool_1s = {}; self.def_tank = 20.0

    def day(self, out, d, base, mode, keep_1s):
        n = 86400; rng = self.rng; P = self.P; speed = out["speed"].astype(np.float64); v = speed / 3.6; run = (mode == 1) | (mode == 2); mech = out["motor_power"].astype(np.float64)
        amb = out["ambient_temperature"].astype(np.float64); gears = P["gears"]
        p_eng = np.where(run, np.maximum(mech, 0.0) / 0.88 + 0.6, 0.0)                                        # kW at the crank: wheel power over the drivetrain efficiency, plus the accessories
        wheel_w = v / self.radius
        rpm_g = wheel_w[:, None] * gears[None, :] * 60.0 / (2 * math.pi); ok = rpm_g <= P["shift"]; gi = np.where(ok.any(1), ok.argmax(1), len(gears) - 1)
        rpm = rpm_g[np.arange(n), gi]; rpm = np.where(run, np.where(v < 0.3, P["idle"], np.maximum(rpm, P["min_rpm"])), 0.0)
        omega = np.maximum(rpm * 2 * math.pi / 60.0, 1.0); torque = np.where(run, np.clip(p_eng * 1000.0 / omega, 0.0, P["tmax"]), 0.0); load = np.where(run, np.clip(torque / P["tref"] * 100.0, 0.0, 100.0), 0.0)
        coast = mech <= -0.5
        rate = np.where(run, np.where(~coast, P["base_l"] + P["per_kw"] * p_eng, P["cut_l"]), 0.0)                # L/h; fuel cut-off while coasting
        # ---- the loop over the seconds of the day: coolant and oil warm-up, and for a diesel the exhaust, the particulate filter and the AdBlue tank
        cool = np.empty(n); oil = np.empty(n); egt = np.empty(n); soot = np.empty(n); regen = np.zeros(n, np.float32); defl = np.empty(n); rate_l = rate.tolist()
        c, o, e, sl, rg, rleft, dl, ct = self.cool, self.oil, self.egt, self.soot, self.regen, self.regen_left, self.def_level, self.cat
        tgt = (86.0 + 0.12 * load).tolist(); rl, al, ld, vl, sp = run.tolist(), amb.tolist(), load.tolist(), speed.tolist(), rpm.tolist(); noise = rng.normal(0, 0.12, n).tolist(); dsl = self.kind == "diesel"
        cat = np.empty(n); fast = 0
        for i in range(n):
            if rl[i]:
                c += (tgt[i] - c) / 360.0; o += (c + 9.0 - o) / 700.0
                if dsl:
                    et = al[i] + 120.0 + 4.2 * ld[i] + (420.0 if rg else 0.0); e += (et - e) / 90.0
                    if rg: rleft -= 1; sl = max(sl - 0.065, 2.0); rate_l[i] *= 1.12
                    else: sl = min(sl + 0.0028 + 0.004 * ld[i] / 100.0, 100.0)
                    if not rg and sl >= 78.0 and vl[i] > 55.0 and rleft <= 0: rg = 1; rleft = 900; self.logs.append((int(base + i), "ecm", f"WARN DPF regeneration started (soot load {sl:.0f} %)"))
                    elif rg and (rleft <= 0 or sl <= 3.0): rg = 0; self.logs.append((int(base + i), "ecm", f"WARN DPF regeneration complete (soot load {sl:.0f} %)"))
                    dl -= rate_l[i] / 3600.0 * 0.05 / self.def_tank * 100.0
                else: ct += (al[i] + 300.0 + 4.5 * ld[i] + 0.04 * sp[i] - ct) / 240.0
            else:
                c += (al[i] - c) / 2600.0; o += (al[i] - o) / 3600.0
                if dsl: e += (al[i] - e) / 1500.0
                else: ct += (al[i] - ct) / 900.0
                if rg: rg = 0; rleft = 0
            cool[i] = c + noise[i]; oil[i] = o + noise[i] * 0.5; egt[i] = e + noise[i] * 3.0; soot[i] = sl; regen[i] = rg; defl[i] = dl; cat[i] = ct + noise[i] * 2.0
            if dsl and dl < 12.0 and rl[i] and fast == 0 and i % 600 == 0: dl = float(rng.uniform(94, 100)); fast = 1; self.logs.append((int(base + i), "ecm", "WARN AdBlue refill detected"))
        self.cool, self.oil, self.egt, self.cat, self.soot, self.regen, self.regen_left, self.def_level = c, o, e, ct, sl, rg, rleft, dl
        use = np.asarray(rate_l) / 3600.0 / self.tank * 100.0; cum = np.cumsum(use); level = self.fuel - cum
        d0 = np.diff(np.r_[False, run, False].astype(np.int8)); starts = np.where(d0 == 1)[0]
        for s0 in starts:                                                                                        # a nearly empty tank is filled before the next drive
            cur = float(level[max(s0 - 1, 0)])
            if cur < 16.0:
                r = max(0, int(s0) - 300); fill = float(rng.uniform(86.0, 99.0)); lvl_r = float(level[r]); level[r:] += fill - lvl_r
                self.refuels.append((int(base + r), round((fill - lvl_r) / 100.0 * self.tank, 1), round(lvl_r, 1), round(fill, 1)))
        level = np.clip(level, 0.0, 100.0); self.fuel = float(level[-1])
        f32 = lambda a: np.asarray(a).astype(np.float32); has = self.has
        out["engine_rpm"] = f32(rpm); out["engine_torque"] = f32(torque); out["engine_load"] = f32(load); out["fuel_rate"] = f32(np.asarray(rate_l)); out["fuel_level"] = f32(level)
        out["gear"] = f32(np.where(run & (v > 0.3), gi + 1, 0)); out["throttle_position"] = f32(np.where(run, np.clip(8.0 + 0.85 * load + rng.normal(0, 0.5, n), 0.0, 100.0), 0.0))
        tt = np.empty(n); tcur = self.tr if hasattr(self, "tr") else float(amb[0]); a_ = amb.tolist(); r_ = run.tolist(); l_ = ld if False else load.tolist()
        for i in range(n):
            tcur += ((a_[i] + 38.0 + 0.45 * l_[i] - tcur) / 900.0) if r_[i] else ((a_[i] - tcur) / 3000.0); tt[i] = tcur
        self.tr = float(tcur); out["transmission_temperature"] = f32(tt)
        out["motor_temperature"] = f32(cool); out["engine_oil_temperature"] = f32(oil); out["fuel_temperature"] = f32(np.where(run, amb + 14.0, amb))
        dd = d / 9.0; baro = 101.3 + 0.9 * math.sin(2 * math.pi * dd + 0.7) + np.cumsum(rng.normal(0, 0.0004, n)); out["barometric_pressure"] = f32(baro)
        if dsl:
            out["boost_pressure"] = f32(np.where(run, np.maximum(load - 12.0, 0.0) * 1.7 + rng.normal(0, 0.6, n), 0.0)); out["exhaust_gas_temperature"] = f32(egt); out["dpf_soot_load"] = f32(soot); out["dpf_regeneration"] = f32(regen)
            out["intake_air_temperature"] = f32(amb + np.where(run, 6.0 + 0.06 * load, 0.0)); out["intake_manifold_pressure"] = f32(baro + out["boost_pressure"]); out["mass_air_flow"] = f32(np.where(run, 1.9 * rpm / 120.0 * 0.9 * 1.19 * (baro + out["boost_pressure"]) / 101.3 * 0.6, 0.0)); out["def_level"] = f32(np.clip(defl, 0.0, 100.0)); out["fuel_rail_pressure"] = f32(np.where(run, 28000.0 + 1320.0 * load + rng.normal(0, 300, n), 0.0))
        else:
            mapk = np.where(run, 30.0 + 0.62 * load + rng.normal(0, 0.5, n), baro); iat = amb + np.where(run, 8.0 + 0.1 * load, 0.0)
            maf = np.where(run, 1.5 * rpm / 120.0 * 0.85 * 1.19 * mapk / 101.3, 0.0)
            lam = np.where(run, np.where(coast, 1.55, np.where(load > 92, 0.88, 1.0 + rng.normal(0, 0.008, n))), 1.0)
            st = np.empty(n); s_ = self.stft; z = rng.normal(0, 0.9, n).tolist()
            for i in range(n):
                s_ = 0.97 * s_ + z[i]; st[i] = s_
            self.stft = float(s_); lt = 3.5 + 0.02 * d + 0.3 * math.sin(d / 3.0)
            out["intake_manifold_pressure"] = f32(mapk); out["intake_air_temperature"] = f32(iat); out["mass_air_flow"] = f32(maf); out["lambda"] = f32(lam)
            out["short_term_fuel_trim"] = f32(np.where(run & ~coast, np.clip(st, -12, 12), 0.0)); out["long_term_fuel_trim"] = f32(np.where(run, lt + rng.normal(0, 0.15, n), lt)); out["catalyst_temperature"] = f32(cat)
            out["boost_pressure"] = f32(np.where(run, np.maximum(load - 55.0, 0.0) * 1.2 + rng.normal(0, 0.5, n), 0.0))
            out["ignition_timing"] = f32(np.where(run, 10.0 + 24.0 * (1 - load / 100.0) + rng.normal(0, 0.4, n), 0.0))
        self.cool_min[d] = cool.reshape(1440, 60).mean(1).astype(np.float32)
        if keep_1s: self.cool_1s[d] = cool.astype(np.float32)


def finish(conn, gv, E):
    """Make the wide tables and the event list describe the combustion vehicle the lake describes."""
    t0 = int(gv.t0)
    conn.execute("DELETE FROM charges")
    conn.execute("DELETE FROM events WHERE type IN ('CHARGE_START','CHARGE_END','WARN_BATT_OVERTEMP')")
    conn.executemany("INSERT INTO events(ts, type, sev, detail) VALUES(?,?,?,?)", [(ts, "REFUEL", "info", f"{litres:.0f} L added (tank {a:.0f}% to {b:.0f}%)") for ts, litres, a, b in E.refuels])
    for (ts, src, msg) in E.logs:
        cur = conn.execute("INSERT INTO logs(ts,src,msg) VALUES(?,?,?)", (int(ts), src, msg)); conn.execute("INSERT INTO logs_fts(rowid,msg,src) VALUES(?,?,?)", (cur.lastrowid, msg, src))
    for tb in ("sig_1m", "sig_1s"):
        conn.execute(f"UPDATE {tb} SET soc=NULL, power=NULL, batt=NULL, flow=NULL, hvac=NULL, mode=CASE WHEN mode=3 THEN 0 ELSE mode END")
    conn.execute("UPDATE sig_1m SET batt_max=NULL, flow_min=NULL")
    conn.executemany("UPDATE sig_1m SET motor=? WHERE ts=?", [(int(round(float(E.cool_min[(ts - t0) // 86400][((ts - t0) % 86400) // 60]) * 10)), ts) for (ts,) in conn.execute("SELECT ts FROM sig_1m").fetchall() if (ts - t0) // 86400 in E.cool_min])
    rows = []
    for (ts,) in conn.execute("SELECT ts FROM sig_1s").fetchall():
        d = (ts - t0) // 86400
        if d in E.cool_1s: rows.append((int(round(float(E.cool_1s[d][(ts - t0) % 86400]) * 10)), ts))
    conn.executemany("UPDATE sig_1s SET motor=? WHERE ts=?", rows)
    conn.execute("UPDATE trips SET kwh=NULL, wh_km=NULL, soc_start=NULL, soc_end=NULL, avg_hvac_w=NULL")
    conn.execute("UPDATE trip_ext SET regen_kwh=NULL")
    conn.execute("DELETE FROM meta WHERE k IN ('veh.cap_kwh','veh.soh','veh.home_charger_kw')")                      # a combustion vehicle has no traction battery, no state of health, no home charger
    conn.execute("INSERT OR REPLACE INTO meta VALUES('veh.powertrain',?)", ("diesel_ice" if E.kind == "diesel" else "petrol_ice",)); conn.execute("INSERT OR REPLACE INTO meta VALUES('veh.tank_l',?)", (str(E.tank),))

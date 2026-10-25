"""Simulator extension: richer, physically consistent signals and fault scenarios WITH ground truth - purely additive.

An existing database (gen_*.db, vehicle.db, a real vehicle's) is never altered in the tables it already has. `migrate()` only CREATEs two new tables if they are missing:
    xsig_1s  (ts, steering_angle, steering_rate, yaw_rate, longitudinal_acceleration, lateral_acceleration, wheel_speed_fl..rr, motor_rpm, motor_torque, motor_power, battery_voltage, battery_current,
              brake_pressure, accelerator_position, brake_pedal, charging_power, regen_power, cell_temperature_max, cell_temperature_min, cell_voltage_min, gps_speed_valid)   one row per sig_1s second
    xsig_1m  the same columns, per-minute means (older history keeps only what can be reconstructed from minute means; the dynamic signals - accelerations, yaw, steering - are NULL there, honestly)
and a few meta rows (veh.wheelbase_m, veh.wheel_radius_m, veh.gear_ratio, veh.steering_ratio, veh.drivetrain_eff, schema.ext). Databases without these tables keep working exactly as before: the capability
engine simply reports the signals as not recorded.

Signals are DERIVED from the data the generator already produced (speed, battery power, state of charge, heading), through the same physics the derived-metric engine inverts (bicycle model for steering and yaw,
wheel radius and gear ratio for rpm, power = torque x rpm for the motor), so a derived answer on this data can be checked against these 'measured' columns.

Fault scenarios are injected into the NEW columns only and written to <db>_faults.json with exact intervals, so the sensor-quality engine can be scored:
    stuck_wheel_fr      wheel_speed_fr frozen for 25 minutes while driving            -> expect STUCK on wheel_speed_fr
    wheel_rl_dropout    wheel_speed_rl missing in 3 bursts of 4 minutes               -> expect DROPOUT
    steering_offset     steering angle offset +4 deg from the middle of the window    -> expect BIAS on steering_angle
    yaw_spikes          isolated single-sample yaw-rate spikes                         -> expect SPIKE on yaw_rate
    brake_overrange     brake pressure reading 250 bar for 30 s (sensor fault)        -> expect RANGE on brake_pressure"""
import json
import math
import os
import sqlite3

import numpy as np

COLS = ("steering_angle", "steering_rate", "yaw_rate", "longitudinal_acceleration", "lateral_acceleration", "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr", "motor_rpm", "motor_torque", "motor_power",
        "battery_voltage", "battery_current", "brake_pressure", "accelerator_position", "brake_pedal", "charging_power", "regen_power", "cell_temperature_max", "cell_temperature_min", "cell_voltage_min", "gps_speed_valid")
DYNAMIC = {"steering_angle", "steering_rate", "yaw_rate", "longitudinal_acceleration", "lateral_acceleration", "brake_pressure", "accelerator_position", "brake_pedal"}
VEHICLE_PARAMS = {"city": dict(wheelbase=2.45, radius=0.29, gear=9.4, steer=14.5, eff=0.90), "compact": dict(wheelbase=2.55, radius=0.30, gear=9.0, steer=15.0, eff=0.90), "family": dict(wheelbase=2.70, radius=0.33, gear=9.0, steer=15.5, eff=0.91),
                  "suv": dict(wheelbase=2.85, radius=0.36, gear=9.2, steer=16.0, eff=0.91)}
SCENARIOS = ("stuck_wheel_fr", "wheel_rl_dropout", "steering_offset", "yaw_spikes", "brake_overrange")


def migrate(conn):
    """Idempotent, additive: create the extension tables and meta rows if absent. Never touches an existing table."""
    cols = ", ".join(f"{c} REAL" for c in COLS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS xsig_1s(ts INTEGER PRIMARY KEY, {cols}) WITHOUT ROWID")
    conn.execute(f"CREATE TABLE IF NOT EXISTS xsig_1m(ts INTEGER PRIMARY KEY, {cols}) WITHOUT ROWID")
    conn.execute("INSERT OR IGNORE INTO meta VALUES('schema.ext','1')"); conn.commit()


def _smooth(x, k):
    if k <= 1: return x
    ker = np.ones(k) / k; pad = k // 2
    return np.convolve(np.pad(x, (pad, k - 1 - pad), mode="edge"), ker, mode="valid")


def derive(ts, speed_kmh, power_kw, soc, heading_deg, hvac_w, mode, batt_c, p, rng):
    """Vectorised physics -> dict of arrays (units = registry units). `p` = vehicle parameters."""
    n = len(ts); v = speed_kmh / 3.6; dt = np.maximum(np.diff(ts, prepend=ts[0] - 1), 1).astype(float)
    a = _smooth(np.gradient(v, ts.astype(float)) if n > 2 else np.zeros(n), 3)
    hd = np.unwrap(np.radians(heading_deg)); yaw = _smooth(np.gradient(hd, ts.astype(float)), 3)                       # rad/s
    gap = _smooth((dt > 3).astype(float), 5) > 0                                                                      # heading across a gap between two trips is not a rotation
    yaw[gap | (np.abs(yaw) > 0.5) | (v < 2.0)] = 0.0
    lat = v * yaw
    delta = np.clip(np.where(v > 1.0, np.arctan(p["wheelbase"] * yaw / np.maximum(v, 1.0)), 0.0), -0.5, 0.5)         # road-wheel angle, bicycle model
    steer = np.degrees(delta) * p["steer"]; srate = np.gradient(steer, ts.astype(float)) if n > 2 else np.zeros(n)
    track = 1.52; dv = 0.5 * track * yaw * 3.6                                                                          # km/h difference between left and right wheels in a turn
    driving = (mode == 1) | (mode == 2)
    slip = np.clip(a - 1.6, 0, None) * 0.012                                                                           # front-driven wheels slip a little under hard acceleration
    base = speed_kmh
    wfl = base * (1 + slip) - dv + rng.normal(0, 0.12, n); wfr = base * (1 + slip) + dv + rng.normal(0, 0.12, n)
    wrl = base - dv + rng.normal(0, 0.12, n); wrr = base + dv + rng.normal(0, 0.12, n)
    for w in (wfl, wfr, wrl, wrr): w[:] = np.where(speed_kmh < 0.5, 0.0, np.maximum(w, 0.0))
    rpm = v / p["radius"] * p["gear"] * 60 / (2 * math.pi)
    aux = hvac_w / 1000.0 + 0.35
    mech = np.where(driving & (power_kw > 0), np.maximum(power_kw - aux, 0) * p["eff"], np.where(driving & (power_kw < 0), power_kw * p["eff"], 0.0))
    torque = np.where(rpm > 60, mech * 9549.0 / np.maximum(rpm, 60), 0.0); torque = np.clip(torque, -320, 320)
    volt = 300.0 + 1.2 * soc; cur = power_kw * 1000.0 / volt; volt = volt - 0.045 * cur; cur = power_kw * 1000.0 / volt
    decel = np.clip(-a, 0, None); regen_cap = 1.3
    bp = np.where(driving, np.clip((decel - regen_cap) * 38.0, 0, 120) + np.where(decel > 0.25, 3.0 + 6.0 * decel, 0.0), 0.0)
    acc = np.where(driving, np.clip(np.where(a > 0, 14 + 24 * a, 6 + 3 * np.clip(v / 10, 0, 6) * (power_kw > 0)) + rng.normal(0, 0.6, n), 0, 100), 0.0)
    pedal = np.clip(bp * 0.7, 0, 100)
    chg = np.where(mode == 3, np.clip(-power_kw, 0, None), 0.0); regen = np.where(driving & (power_kw < 0), -power_kw, 0.0)
    cmax = batt_c + 0.9 + 0.012 * np.abs(power_kw); cmin = batt_c - 0.7; cvmin = volt / 96.0 - 0.035 - 0.00008 * np.abs(cur)
    return {"steering_angle": steer, "steering_rate": srate, "yaw_rate": np.degrees(yaw), "longitudinal_acceleration": a, "lateral_acceleration": lat, "wheel_speed_fl": wfl, "wheel_speed_fr": wfr, "wheel_speed_rl": wrl, "wheel_speed_rr": wrr,
            "motor_rpm": rpm, "motor_torque": torque, "motor_power": mech, "battery_voltage": volt, "battery_current": cur, "brake_pressure": bp, "accelerator_position": acc, "brake_pedal": pedal, "charging_power": chg, "regen_power": regen,
            "cell_temperature_max": cmax, "cell_temperature_min": cmin, "cell_voltage_min": cvmin, "gps_speed_valid": np.ones(n)}


def _inject(ts, X, rng, driving, truth):
    """Fault scenarios on the NEW columns; every one is recorded with its exact interval."""
    idx = np.where(driving)[0]
    if len(idx) < 600: return
    runs = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1); runs = [r for r in runs if len(r) >= 120]
    if not runs: return
    longest = max(runs, key=len)
    n = min(len(longest), 1500); a0 = longest[0] + (len(longest) - n) // 2; seg = np.arange(a0, a0 + n)
    X["wheel_speed_fr"][seg] = X["wheel_speed_fr"][seg[0]]; truth.append({"name": "stuck_wheel_fr", "signal": "wheel_speed_fr", "start": int(ts[seg[0]]), "end": int(ts[seg[-1]]), "expect": "STUCK"})
    bursts = []
    for k, r in enumerate(sorted(runs, key=len, reverse=True)[:3]):
        m = min(len(r) // 3, 240); s0 = r[len(r) // 3]; X["wheel_speed_rl"][s0:s0 + m] = np.nan; bursts.append((int(ts[s0]), int(ts[min(s0 + m, len(ts) - 1)])))
    truth.append({"name": "wheel_rl_dropout", "signal": "wheel_speed_rl", "intervals": bursts, "expect": "DROPOUT"})
    half = len(ts) // 2; X["steering_angle"][half:] += 4.0; truth.append({"name": "steering_offset", "signal": "steering_angle", "start": int(ts[half]), "end": int(ts[-1]), "offset_deg": 4.0, "expect": "BIAS"})
    pick = rng.choice(longest, size=min(6, len(longest)), replace=False); X["yaw_rate"][pick] += 70.0
    truth.append({"name": "yaw_spikes", "signal": "yaw_rate", "timestamps": [int(ts[i]) for i in sorted(pick)], "expect": "SPIKE"})
    r = sorted(runs, key=len)[len(runs) // 2]; s0 = r[len(r) // 2]; X["brake_pressure"][s0:s0 + 30] = 250.0
    truth.append({"name": "brake_overrange", "signal": "brake_pressure", "start": int(ts[s0]), "end": int(ts[min(s0 + 29, len(ts) - 1)]), "value": 250.0, "expect": "RANGE"})


def extend(db_path, seed=0, vehicle=None, scenarios=True, truth_path=None):
    """Add the extended signals (and optionally the fault scenarios) to a generated database. Returns the ground-truth dict."""
    rng = np.random.default_rng(seed + 4242)
    conn = sqlite3.connect(db_path); conn.execute("PRAGMA journal_mode=WAL")
    try:
        meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
        veh = vehicle or meta.get("gen.vehicle") or "family"; p = VEHICLE_PARAMS.get(veh, VEHICLE_PARAMS["family"])
        migrate(conn)
        for k, v in (("veh.wheelbase_m", p["wheelbase"]), ("veh.wheel_radius_m", p["radius"]), ("veh.gear_ratio", p["gear"]), ("veh.steering_ratio", p["steer"]), ("veh.drivetrain_eff", p["eff"])): conn.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))
        conn.execute("DELETE FROM xsig_1s"); conn.execute("DELETE FROM xsig_1m")
        rows = np.array(conn.execute("SELECT ts, speed, power, soc, batt, hvac, mode FROM sig_1s ORDER BY ts").fetchall(), dtype=float)
        truth = {"seed": seed, "vehicle": veh, "params": p, "scenarios": []}
        if len(rows):
            ts = rows[:, 0].astype(np.int64); gps = np.array(conn.execute("SELECT ts, hdg FROM gps_1s ORDER BY ts").fetchall(), dtype=float)
            hdg = np.interp(ts, gps[:, 0], gps[:, 1] / 10.0) if len(gps) else np.zeros(len(ts))
            X = derive(ts, rows[:, 1] / 10.0, rows[:, 2] / 1000.0, rows[:, 3] / 10.0, hdg, rows[:, 5], rows[:, 6].astype(int), rows[:, 4] / 10.0, p, rng)
            if scenarios: _inject(ts, X, rng, (rows[:, 6] == 1) | (rows[:, 6] == 2), truth["scenarios"])
            ph = ",".join("?" * (len(COLS) + 1)); cols = ", ".join(("ts",) + COLS)
            conn.executemany(f"INSERT INTO xsig_1s({cols}) VALUES({ph})", ((int(ts[i]),) + tuple(None if np.isnan(X[c][i]) else round(float(X[c][i]), 3) for c in COLS) for i in range(len(ts))))
            minute = ts // 60 * 60; order = np.argsort(minute, kind="stable"); u, start = np.unique(minute[order], return_index=True)
            def mean_of(c):
                arr = X[c][order]; out = np.add.reduceat(np.nan_to_num(arr), start) / np.maximum(np.add.reduceat(~np.isnan(arr), start), 1)
                cnt = np.add.reduceat(~np.isnan(arr), start); out[cnt == 0] = np.nan; return out
            M = {c: mean_of(c) for c in COLS}
            conn.executemany(f"INSERT INTO xsig_1m({cols}) VALUES({ph})", ((int(u[i]),) + tuple(None if np.isnan(M[c][i]) else round(float(M[c][i]), 3) for c in COLS) for i in range(len(u))))
            first_1s = int(ts[0])
            old = np.array(conn.execute("SELECT ts, speed, power, soc, batt, hvac, mode FROM sig_1m WHERE ts<? ORDER BY ts", (first_1s,)).fetchall(), dtype=float)
            if len(old):                                                                                             # older history: only what minute means can honestly support
                o = derive(old[:, 0].astype(np.int64), old[:, 1] / 10.0, old[:, 2] / 1000.0, old[:, 3] / 10.0, np.zeros(len(old)), old[:, 5], old[:, 6].astype(int), old[:, 4] / 10.0, p, rng)
                keep = [c for c in COLS if c not in DYNAMIC and c != "gps_speed_valid"]
                conn.executemany(f"INSERT OR IGNORE INTO xsig_1m({cols}) VALUES({ph})", ((int(old[i, 0]),) + tuple(round(float(o[c][i]), 3) if c in keep else None for c in COLS) for i in range(len(old))))
        conn.commit()
    finally: conn.close()
    tp = truth_path or (db_path[:-3] + "_faults.json" if db_path.endswith(".db") else db_path + "_faults.json")
    json.dump(truth, open(tp, "w"), indent=1); return truth

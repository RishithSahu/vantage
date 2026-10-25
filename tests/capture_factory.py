"""A known drive, rendered the way real vehicles and loggers write it - the test bench of the universal ingest (vgpt/ingest).

`truth(t)` is a deterministic drive (speed, engine speed, temperatures, fuel ...); every `render_*` function writes the same drive in one real-world format.  A test then reads the file with no hint but its
content and checks that the quantities come out right (name, unit, value).  Standard library only, so it can also run in an environment with python-can / cantools to make reference files.

    J1939 frames (EEC1, CCVS1, ET1, LFE, DD, AMB, EEC2, VEP1, IC1, VD)      candump -L, candump (default), Vector ASC, PEAK TRC
    OBD-II service 01 request / response pairs over ISO-TP (11-bit)         candump, ELM327 trace
    CSV like Torque / Car Scanner exports (metric and imperial), a messy logger CSV
    JSON lines: OpenXC, COVESA VSS paths, flat sensor objects, {"name","value"} lists
    NMEA 0183 (speed in knots)"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
T0 = 1_760_000_000.0
CATALOG = json.load(open(os.path.join(ROOT, "vgpt", "ingest", "catalog.json"), encoding="utf-8"))["map"]


def truth(t):
    """The drive: concept -> value in the registry's unit, at second t."""
    speed = max(0.0, 55 + 40 * math.sin(t / 12.0))
    rpm = 750 + speed * 21.0
    return {"speed": speed, "engine_rpm": rpm, "motor_temperature": 60 + 30 * min(1.0, t / 90.0), "engine_oil_temperature": 55 + 45 * min(1.0, t / 120.0),
            "fuel_level": 80 - t / 400.0, "fuel_rate": 4.0 + speed * 0.11, "ambient_temperature": 21.5, "accelerator_position": max(0.0, 20 + 18 * math.cos(t / 12.0)), "engine_load": 20 + speed * 0.5,
            "aux_battery_voltage": 13.8, "barometric_pressure": 101.0, "boost_pressure": 30 + speed * 0.6, "intake_air_temperature": 28.0, "odometer": 12345.0 + speed * t / 3600.0 * 0.6}


# ------------------------------------------------------------------------------------------------------------------ J1939
def _put(data, start, length, raw):
    raw = int(max(0, min((1 << (8 * length)) - 1, round(raw)))); data[start - 1:start - 1 + length] = raw.to_bytes(length, "little")


def j1939_frames(t, tr, sa=0):
    """-> [(ts, can_id, data)] for the parameter groups due at second t (10 Hz groups every tenth of a second)."""
    out = []
    def enc(pgn, pri, fields, when):
        data = bytearray(b"\xff" * 8)
        for concept, spn, start, ln, scale, offset in fields:
            if concept in tr: _put(data, start, ln, (tr[concept] - offset) / scale)
        out.append((T0 + t + when, (pri << 26) | (pgn << 8) | sa | 0x80000000, bytes(data)))
    table = {}
    for cid, m in CATALOG.items():
        for j in m.get("j1939", ()): table.setdefault(j["pgn"], []).append((cid, j["spn"], j["start"], j["len"], j["scale"], j["offset"]))
    for pgn, pri, rate in ((61444, 3, 10), (65265, 6, 10), (65266, 6, 10), (61443, 3, 10), (65262, 6, 1), (65276, 6, 1), (65269, 6, 1), (65271, 6, 1), (65270, 6, 1), (65248, 6, 1)):
        if pgn not in table: continue
        for k in range(rate): enc(pgn, pri, table[pgn], k / rate)
    return out


# ------------------------------------------------------------------------------------------------------------------ OBD-II
OBD_PIDS = {0x0D: ("speed", 1, 1.0, 0.0), 0x0C: ("engine_rpm", 2, 0.25, 0.0), 0x05: ("motor_temperature", 1, 1.0, -40.0), 0x5C: ("engine_oil_temperature", 1, 1.0, -40.0), 0x2F: ("fuel_level", 1, 100 / 255.0, 0.0),
            0x5E: ("fuel_rate", 2, 0.05, 0.0), 0x46: ("ambient_temperature", 1, 1.0, -40.0), 0x04: ("engine_load", 1, 100 / 255.0, 0.0), 0x42: ("aux_battery_voltage", 2, 0.001, 0.0), 0x33: ("barometric_pressure", 1, 1.0, 0.0),
            0x0F: ("intake_air_temperature", 1, 1.0, -40.0), 0x49: ("accelerator_position", 1, 100 / 255.0, 0.0)}


def obd_frames(t, tr, per_second=4):
    """Request / response pairs over 11-bit CAN: the scan tool asks (0x7DF), the engine ECU answers (0x7E8)."""
    out = []; pids = sorted(OBD_PIDS); base = int(t * per_second)
    for k in range(per_second):
        pid = pids[(base + k) % len(pids)]; name, n, scale, offset = OBD_PIDS[pid]; ts = T0 + t + k / per_second
        raw = round((tr[name] - offset) / scale); body = raw.to_bytes(n, "big")
        out.append((ts, 0x7DF, bytes([2, 1, pid]) + b"\x00" * 5)); out.append((ts + 0.01, 0x7E8, bytes([2 + n, 0x41, pid]) + body + b"\x00" * (5 - n)))
    return out


# ------------------------------------------------------------------------------------------------------------------ writers: CAN text formats
def _hexs(b): return b.hex().upper()


def write_candump_l(path, frames, channel="can0"):
    with open(path, "w", newline="\n") as f:
        for ts, cid, data in sorted(frames): f.write(f"({ts:.6f}) {channel} {cid & 0x1FFFFFFF:08X}#{_hexs(data)}\n" if cid & 0x80000000 else f"({ts:.6f}) {channel} {cid:03X}#{_hexs(data)}\n")


def write_candump_default(path, frames, channel="can0"):
    with open(path, "w", newline="\n") as f:
        for ts, cid, data in sorted(frames):
            ident = f"{cid & 0x1FFFFFFF:08X}" if cid & 0x80000000 else f"{cid:03X}"; f.write(f"  {channel}  {ident}   [{len(data)}]  {' '.join(f'{b:02X}' for b in data)}\n")


def write_asc(path, frames, channel=1):
    with open(path, "w", newline="\n") as f:
        f.write("date Mon Oct 20 10:00:00.000 am 2026\nbase hex  timestamps absolute\ninternal events logged\n")
        t0 = min(ts for ts, _, _ in frames)
        for ts, cid, data in sorted(frames):
            ident = f"{cid & 0x1FFFFFFF:X}x" if cid & 0x80000000 else f"{cid:X}"; f.write(f"   {ts - t0:.6f} {channel}  {ident:<15} Rx   d {len(data)} {' '.join(f'{b:02X}' for b in data)}\n")


def write_trc(path, frames):
    with open(path, "w", newline="\n") as f:
        f.write(";$FILEVERSION=2.0\n;   Start time: 20.10.2026 10:00:00.000.0\n;   Generated by capture_factory\n")
        t0 = min(ts for ts, _, _ in frames)
        for i, (ts, cid, data) in enumerate(sorted(frames), 1):
            ident = f"{cid & 0x1FFFFFFF:08X}" if cid & 0x80000000 else f"{cid:04X}"; f.write(f"{i:>7}) {(ts - t0) * 1000:12.3f}  DT  1  {ident:<8} Rx  {len(data)}  {' '.join(f'{b:02X}' for b in data)}\n")


def write_elm(path, frames):
    """An ELM327 trace with headers on: '7E8 03 41 0D 3C' lines (only the responses, as a scan tool shows them)."""
    with open(path, "w", newline="\n") as f:
        f.write("ATZ\nELM327 v1.5\n>ATH1\nOK\n")
        for ts, cid, data in sorted(frames):
            if cid == 0x7E8: f.write(f"{cid:03X} {' '.join(f'{b:02X}' for b in data[:1 + data[0]])}\n")


# ------------------------------------------------------------------------------------------------------------------ writers: tables
OBD_APP_COLS = [("Speed (OBD)(km/h)", "speed", 1.0, 0.0), ("Engine RPM(rpm)", "engine_rpm", 1.0, 0.0), ("Engine Coolant Temperature(°C)", "motor_temperature", 1.0, 0.0), ("Engine Oil Temperature(°C)", "engine_oil_temperature", 1.0, 0.0),
                ("Fuel Level (From Engine ECU)(%)", "fuel_level", 1.0, 0.0), ("Fuel flow rate/hour(l/hr)", "fuel_rate", 1.0, 0.0), ("Ambient air temp(°C)", "ambient_temperature", 1.0, 0.0), ("Engine Load(%)", "engine_load", 1.0, 0.0),
                ("Voltage (Control Module)(V)", "aux_battery_voltage", 1.0, 0.0), ("Throttle Position(Manifold)(%)", "accelerator_position", 1.0, 0.0), ("Intake Air Temperature(°C)", "intake_air_temperature", 1.0, 0.0)]
OBD_APP_IMPERIAL = [("Speed (OBD)(mph)", "speed", 0.621371, 0.0), ("Engine RPM(rpm)", "engine_rpm", 1.0, 0.0), ("Engine Coolant Temperature(°F)", "motor_temperature", 1.8, 32.0), ("Engine Oil Temperature(°F)", "engine_oil_temperature", 1.8, 32.0),
                    ("Fuel Level (From Engine ECU)(%)", "fuel_level", 1.0, 0.0), ("Fuel flow rate/hour(gal/hr)", "fuel_rate", 0.264172, 0.0), ("Ambient air temp(°F)", "ambient_temperature", 1.8, 32.0), ("Engine Load(%)", "engine_load", 1.0, 0.0),
                    ("Voltage (Control Module)(V)", "aux_battery_voltage", 1.0, 0.0), ("Barometer (on Android device)(psi)", "barometric_pressure", 0.145038, 0.0)]
LOGGER_COLS = [("vehSpd_kph", "speed", 1.0, 0.0), ("EngSpd", "engine_rpm", 1.0, 0.0), ("CltTemp_C", "motor_temperature", 1.0, 0.0), ("EOT", "engine_oil_temperature", 1.0, 0.0), ("FuelLvl_pct", "fuel_level", 1.0, 0.0),
               ("ThrPos", "accelerator_position", 1.0, 0.0), ("Baro_kPa", "barometric_pressure", 1.0, 0.0), ("Vbat", "aux_battery_voltage", 1.0, 0.0), ("MAP_kPa", "boost_pressure", 1.0, 0.0), ("Odo_km", "odometer", 1.0, 0.0)]


def write_csv(path, cols, duration, step=1.0, delimiter=",", time_name="Time", iso=False, decimal="."):
    import datetime as dt
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(delimiter.join([time_name] + [c[0] for c in cols]) + "\n"); t = 0.0
        while t < duration:
            tr = truth(t); stamp = dt.datetime.fromtimestamp(T0 + t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if iso else f"{T0 + t:.3f}"
            f.write(delimiter.join([stamp] + [f"{tr[c] * k + o:.3f}".replace(".", decimal) for _, c, k, o in cols]) + "\n"); t += step


OPENXC = {"vehicle_speed": ("speed", 1.0), "engine_speed": ("engine_rpm", 1.0), "fuel_level": ("fuel_level", 1.0), "odometer": ("odometer", 1.0), "accelerator_pedal_position": ("accelerator_position", 1.0)}
VSS = {"Vehicle.Speed": ("speed", 1.0), "Vehicle.Powertrain.CombustionEngine.Speed": ("engine_rpm", 1.0), "Vehicle.Powertrain.CombustionEngine.ECT": ("motor_temperature", 1.0), "Vehicle.Powertrain.CombustionEngine.EOT": ("engine_oil_temperature", 1.0),
       "Vehicle.Powertrain.FuelSystem.RelativeLevel": ("fuel_level", 1.0), "Vehicle.Exterior.AirTemperature": ("ambient_temperature", 1.0), "Vehicle.Chassis.Accelerator.PedalPosition": ("accelerator_position", 1.0),
       "Vehicle.LowVoltageBattery.CurrentVoltage": ("aux_battery_voltage", 1.0), "Vehicle.TraveledDistance": ("odometer", 1.0)}


def write_openxc(path, duration):
    with open(path, "w", newline="\n") as f:
        for i in range(int(duration)):
            tr = truth(i)
            for name, (concept, k) in OPENXC.items(): f.write(json.dumps({"name": name, "value": round(tr[concept] * k, 3), "timestamp": round(T0 + i + 0.1, 3)}) + "\n")


def write_vss(path, duration, wrapper="flat"):
    with open(path, "w", newline="\n") as f:
        for i in range(int(duration)):
            tr = truth(i); vals = {p: round(tr[c] * k, 3) for p, (c, k) in VSS.items()}
            if wrapper == "flat": f.write(json.dumps(dict(vals, timestamp=round(T0 + i, 3))) + "\n")
            else: f.write(json.dumps({"ts": round((T0 + i) * 1000), "updates": [{"path": p, "value": v} for p, v in vals.items()]}) + "\n")


def write_flat_json(path, duration, nested=False):
    with open(path, "w", newline="\n") as f:
        for i in range(int(duration)):
            tr = truth(i)
            o = {"time": round((T0 + i) * 1000), "vehicleSpeed_kph": round(tr["speed"], 2), "engine": {"rpm": round(tr["engine_rpm"]), "coolantTemp_C": round(tr["motor_temperature"], 1), "oilTemp_C": round(tr["engine_oil_temperature"], 1)}, "fuel": {"level_pct": round(tr["fuel_level"], 2)}}
            f.write(json.dumps(o) + "\n")


def write_nmea(path, duration):
    def cs(s):
        c = 0
        for ch in s: c ^= ord(ch)
        return f"{c:02X}"
    with open(path, "w", newline="\n") as f:
        for i in range(int(duration)):
            tr = truth(i); knots = tr["speed"] / 1.852; t = T0 + i; hh, mm, ss = int(t % 86400 // 3600), int(t % 3600 // 60), int(t % 60); body = f"GPRMC,{hh:02d}{mm:02d}{ss:02d}.00,A,4807.038,N,01131.000,E,{knots:.2f},84.4,201026,,,A"
            f.write(f"${body}*{cs(body)}\n")


def can_all(duration):
    frames = []
    for i in range(int(duration)): frames += j1939_frames(i, truth(i))
    return frames


def obd_all(duration):
    frames = []
    for i in range(int(duration)): frames += obd_frames(i, truth(i))
    return frames

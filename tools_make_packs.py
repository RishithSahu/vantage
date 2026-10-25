"""Writes the two demonstration profile packs that differ most from the reference vehicle (packs/aster_ev is hand-written):

  meridian_ev    US-market battery SUV: ONE CAN-FD bus with 29-bit ids and Motorola signals, imperial units (mph, psi, degF, miles), a body CAN, brake-disc temperatures, SOME/IP ADAS, Android VHAL, logcat
  rhino_diesel   light commercial diesel truck: SAE J1939 on 29-bit ids (real PGNs and SPN scalings), engine + fuel signals and the diesel-only ones (boost, exhaust gas temperature, DPF soot load and regeneration,
                 AdBlue / DEF level, rail pressure), no battery / ADAS / camera, a Linux telematics unit (syslog), UDS/OBD
  kestrel_petrol petrol hatchback: 11-bit powertrain CAN at 500 kbit/s with OBD-II-style engine data (manifold pressure, mass air flow, lambda, short / long fuel trims, catalyst temperature, ignition timing) and none of
                 the diesel ones, a body CAN, UDS/OBD, a Linux telematics unit, GNSS

The packs exist to prove that the same assistant, ingest and store run unchanged on vehicles that differ in buses, names, units, byte order and powertrain.
    python tools_make_packs.py"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgpt", "vehicle", "packs")
EXT = 0x80000000


def dbc(version, nodes, messages, vals=()):
    """messages: [(id, name, dlc, sender, [(sig, start, len, order, sign, factor, offset, lo, hi, unit)], mux)]"""
    out = [f'VERSION "{version}"', "", "NS_ :", "", "BS_:", "", "BU_: " + " ".join(nodes), ""]
    for fid, name, dlc, sender, sigs in messages:
        out.append(f"BO_ {fid} {name}: {dlc} {sender}")
        for sg in sigs:
            sig, start, ln, order, sign, f, o, lo, hi, unit = sg[:10]; mux = sg[10] if len(sg) > 10 else ""
            out.append(f" SG_ {sig}{(' ' + mux) if mux else ''} : {start}|{ln}@{order}{sign} ({f},{o}) [{lo}|{hi}] \"{unit}\" Vector__XXX")
        out.append("")
    out += list(vals)
    return "\n".join(out) + "\n"


def write(pack, name, text):
    d = os.path.join(ROOT, pack); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, name), "w", encoding="utf-8", newline="\n").write(text)


# ================================================================================================ meridian_ev
def meridian():
    main = dbc("Meridian E5 main CAN-FD (29-bit ids, Motorola)", ["GW", "BMS", "THERM", "CHG", "HVAC", "BCM"], [
        (0x18FF1000 | EXT, "VehDynamics", 16, "GW", [
            ("VehSpdAvgDrvn_mph", 7, 16, 0, "+", 0.01, 0, 0, 300, "mph"), ("AccelPedalPct", 23, 8, 0, "+", 0.4, 0, 0, 100, "%"), ("BrkPedalPct", 31, 8, 0, "+", 0.4, 0, 0, 100, "%"),
            ("SteerWhlAngle_deg", 39, 16, 0, "-", 0.1, 0, -780, 780, "deg"), ("YawRate_dps", 55, 16, 0, "-", 0.01, 0, -120, 120, "deg/s"), ("LatAcc_g", 71, 16, 0, "-", 0.0005, 0, -3, 3, "g"),
            ("LongAcc_g", 87, 16, 0, "-", 0.0005, 0, -3, 3, "g")]),
        (0x18FF1100 | EXT, "HVBattery", 16, "BMS", [
            ("HVBatt_SOC_Disp", 7, 8, 0, "+", 0.5, 0, 0, 100, "%"), ("HVBatt_Volt", 15, 16, 0, "+", 0.05, 0, 0, 1000, "V"), ("HVBatt_Curr", 31, 16, 0, "-", 0.05, 0, -1500, 1500, "A"),
            ("HVBatt_Temp_F", 47, 16, 0, "-", 0.1, 0, -40, 250, "degF"), ("HVBatt_CellMaxTemp_F", 63, 16, 0, "-", 0.1, 0, -40, 250, "degF"), ("HVBatt_CellMinTemp_F", 79, 16, 0, "-", 0.1, 0, -40, 250, "degF"),
            ("HVBatt_CellMinV", 95, 16, 0, "+", 0.001, 0, 0, 6, "V")]),
        (0x18FF1200 | EXT, "ThermalSys", 16, "THERM", [
            ("CoolPumpFlow_lpm", 7, 16, 0, "+", 0.1, 0, 0, 100, "L/min"), ("MotorTemp_F", 23, 16, 0, "-", 0.1, 0, -40, 400, "degF"), ("MotorTorque_Nm", 39, 16, 0, "-", 0.1, 0, -600, 600, "Nm"),
            ("MotorSpeed_rpm", 55, 16, 0, "-", 1, 0, -2000, 20000, "rpm"), ("BrkDiscTempMax_C", 71, 16, 0, "+", 0.1, 0, 0, 1000, "degC")]),
        (0x18FF1300 | EXT, "ClimateSys", 8, "HVAC", [
            ("OutsideTemp_F", 7, 16, 0, "-", 0.1, 0, -60, 150, "degF"), ("CabinTemp_F", 23, 16, 0, "-", 0.1, 0, -40, 180, "degF"), ("HvacPwr_kW", 39, 16, 0, "+", 0.01, 0, 0, 10, "kW")]),
        (0x18FF1400 | EXT, "ChargeSys", 8, "CHG", [("ChgPwr_kW", 7, 16, 0, "+", 0.1, 0, 0, 400, "kW"), ("ChgSt", 23, 8, 0, "+", 1, 0, 0, 3, "")]),
    ], ["VAL_ %d ChgSt 0 \"unplugged\" 1 \"charging\" 2 \"complete\" 3 \"fault\" ;" % (0x18FF1400 | EXT)])
    body = dbc("Meridian E5 body CAN 250 kbit/s", ["BCM", "CLU"], [
        (0x3A0, "TpmsData", 8, "BCM", [("TyrPres_FL_psi", 7, 8, 0, "+", 0.25, 0, 0, 63.75, "psi"), ("TyrPres_FR_psi", 15, 8, 0, "+", 0.25, 0, 0, 63.75, "psi"), ("TyrPres_RL_psi", 23, 8, 0, "+", 0.25, 0, 0, 63.75, "psi"),
                                       ("TyrPres_RR_psi", 31, 8, 0, "+", 0.25, 0, 0, 63.75, "psi"), ("TyrTemp_F", 39, 8, 0, "+", 1, -40, -40, 215, "degF")]),
        (0x3A8, "EnvSensors", 8, "BCM", [("RainLvl", 7, 8, 0, "+", 1, 0, 0, 3, ""), ("AmbLight_lux", 15, 16, 0, "+", 1, 0, 0, 65535, "lux"), ("Batt12V", 31, 16, 0, "+", 0.001, 0, 0, 20, "V")]),
        (0x3B0, "Cluster", 8, "CLU", [("Odometer_mi", 7, 32, 0, "+", 0.1, 0, 0, 400000, "mi")])])
    write("meridian_ev", "main.dbc", main); write("meridian_ev", "body.dbc", body)
    write("meridian_ev", "services.json", json.dumps({
        "0x4A11": {"name": "AdasState", "interface_version": 1, "events": {
            "0x8001": {"name": "SystemState", "fmt": ">BB", "fields": ["engaged", "level"]},
            "0x8002": {"name": "Disengaged", "fmt": ">BH", "fields": ["reason", "speed_kmh"], "enums": {"reason": {"0": "driver override", "1": "lane marking lost", "2": "camera blocked", "3": "system limit", "4": "sensor conflict", "5": "radar fault"}}},
            "0x8003": {"name": "AebActivation", "fmt": ">Hf", "fields": ["speed_kmh", "decel_g"]}}},
        "0x4B01": {"name": "ChargingService", "interface_version": 1, "events": {"0x8002": {"name": "SessionStarted", "fmt": ">Bf", "fields": ["kind", "kw"], "enums": {"kind": {"0": "AC", "1": "DC"}}}}},
        "0x4C01": {"name": "EcuSupervisor", "interface_version": 1, "events": {
            "0x8001": {"name": "EcuReset", "fmt": ">BB", "fields": ["ecu", "cause"], "enums": {"ecu": {"0": "ADAS domain controller", "1": "battery management", "2": "gateway", "3": "infotainment", "4": "body controller", "5": "thermal manager"}, "cause": {"0": "power-on", "1": "watchdog timeout", "2": "software request", "3": "brown-out"}}},
            "0x8003": {"name": "UnexpectedShutdown", "fmt": ">BH", "fields": ["cause", "last_valid_ms"], "enums": {"cause": {"0": "unknown", "1": "12 V supply loss", "2": "watchdog timeout", "3": "high-voltage interlock"}}}}},
        "0x4D01": {"name": "PowerManager", "interface_version": 1, "events": {"0x8001": {"name": "VoltageDip", "fmt": ">Hf", "fields": ["duration_ms", "min_v"]}}}}, indent=1))
    write("meridian_ev", "dtc.json", json.dumps({
        "P0A7F": {"text": "Hybrid/EV battery pack deterioration", "subsystem": "battery", "severity": "warn"}, "P0AFA": {"text": "Hybrid/EV battery system voltage low", "subsystem": "battery", "severity": "warn"},
        "P0AA6": {"text": "Hybrid/EV battery voltage system isolation fault", "subsystem": "battery", "severity": "critical"}, "P0562": {"text": "System voltage low", "subsystem": "power", "severity": "warn"},
        "C0035": {"text": "Left front wheel speed sensor circuit", "subsystem": "brakes", "severity": "warn"}, "U0121": {"text": "Lost communication with anti-lock brake system module", "subsystem": "network", "severity": "warn"},
        "U0100": {"text": "Lost communication with powertrain control module", "subsystem": "network", "severity": "warn"}, "B1000": {"text": "ECU internal fault", "subsystem": "electronics", "severity": "warn"},
        "C1A10": {"text": "Forward camera misaligned or blocked (illustrative code)", "subsystem": "adas", "severity": "warn"}, "P1A00": {"text": "Battery coolant flow low (illustrative code)", "subsystem": "thermal", "severity": "warn"}}, indent=1))
    S = lambda concept, frm, unit, rate=1.0, **kw: dict({"concept": concept, "from": frm, "unit": unit, "rate_hz": rate}, **kw)
    sig = [S("speed", "can:main/VehDynamics.VehSpdAvgDrvn_mph", "mph", 10), S("speed", "vhal:hal/PERF_VEHICLE_SPEED", "m/s", 1, priority=2, conflict_abs=3),
           S("accelerator_position", "can:main/VehDynamics.AccelPedalPct", "%", 10), S("brake_pedal", "can:main/VehDynamics.BrkPedalPct", "%", 10), S("steering_angle", "can:main/VehDynamics.SteerWhlAngle_deg", "deg", 10),
           S("yaw_rate", "can:main/VehDynamics.YawRate_dps", "deg/s", 10), S("lateral_acceleration", "can:main/VehDynamics.LatAcc_g", "g", 10), S("longitudinal_acceleration", "can:main/VehDynamics.LongAcc_g", "g", 10),
           S("battery_soc", "can:main/HVBattery.HVBatt_SOC_Disp", "%", 10), S("battery_voltage", "can:main/HVBattery.HVBatt_Volt", "V", 10), S("battery_current", "can:main/HVBattery.HVBatt_Curr", "A", 10),
           S("battery_temperature", "can:main/HVBattery.HVBatt_Temp_F", "F", 10), S("cell_temperature_max", "can:main/HVBattery.HVBatt_CellMaxTemp_F", "F", 10), S("cell_temperature_min", "can:main/HVBattery.HVBatt_CellMinTemp_F", "F", 10),
           S("cell_voltage_min", "can:main/HVBattery.HVBatt_CellMinV", "V", 10), S("coolant_flow", "can:main/ThermalSys.CoolPumpFlow_lpm", "L/min", 10), S("motor_temperature", "can:main/ThermalSys.MotorTemp_F", "F", 10),
           S("motor_torque", "can:main/ThermalSys.MotorTorque_Nm", "Nm", 10), S("motor_rpm", "can:main/ThermalSys.MotorSpeed_rpm", "rpm", 10), S("brake_temperature", "can:main/ThermalSys.BrkDiscTempMax_C", "C", 10),
           S("ambient_temperature", "can:main/ClimateSys.OutsideTemp_F", "F", 1), S("cabin_temperature", "can:main/ClimateSys.CabinTemp_F", "F", 1), S("hvac_power", "can:main/ClimateSys.HvacPwr_kW", "kW", 1),
           S("charging_power", "can:main/ChargeSys.ChgPwr_kW", "kW", 10), dict(S("_charging_active", "can:main/ChargeSys.ChgSt", "none", 10), map={"charging": 1, "unplugged": 0, "complete": 0, "fault": 0}),
           dict(S("_charging_active", "vhal:hal/EV_CHARGE_STATE", "none", 0.2, priority=2), map={"1": 1, "3": 0, "2": 0, "4": 0}), dict(S("_ignition", "vhal:hal/IGNITION_STATE", "none", 0.2), map={"4": 1, "5": 1, "3": 0, "2": 0, "1": 0}),
           S("tyre_pressure_fl", "can:body/TpmsData.TyrPres_FL_psi", "psi", 1), S("tyre_pressure_fr", "can:body/TpmsData.TyrPres_FR_psi", "psi", 1), S("tyre_pressure_rl", "can:body/TpmsData.TyrPres_RL_psi", "psi", 1),
           S("tyre_pressure_rr", "can:body/TpmsData.TyrPres_RR_psi", "psi", 1), S("tyre_temperature", "can:body/TpmsData.TyrTemp_F", "F", 1), S("rain", "can:body/EnvSensors.RainLvl", "none", 1),
           S("light", "can:body/EnvSensors.AmbLight_lux", "lux", 1), S("aux_battery_voltage", "can:body/EnvSensors.Batt12V", "V", 1), S("odometer", "can:body/Cluster.Odometer_mi", "mi", 1),
           S("odometer", "vhal:hal/PERF_ODOMETER", "km", 0.2, priority=2, conflict_abs=8), S("adas_state", "someip:eth/AdasState.SystemState.engaged", "none", 1),
           S("camera_blockage", "camera:cam/blockage", "ratio", 1), S("camera_sharpness", "camera:cam/sharpness", "ratio", 1), S("camera_brightness", "camera:cam/luma", "none", 1), S("lane_confidence", "camera:cam/lane_conf", "ratio", 1)]
    ev = [{"id": "adas_dis", "from": "someip:AdasState.Disengaged", "type": "ADAS_DISENGAGE", "severity": "info", "detail": "{reason_text}"},
          {"id": "aeb", "from": "someip:AdasState.AebActivation", "type": "AEB_ACTIVATION", "severity": "warn", "detail": "automatic emergency braking at {speed_kmh} km/h ({decel_g:.2f} g)"},
          {"id": "chg_start", "from": "someip:ChargingService.SessionStarted", "type": "CHARGE_START", "severity": "info", "detail": "{kind_text} {kw:.0f} kW"},
          {"id": "ecu_reset", "from": "someip:EcuSupervisor.EcuReset", "type": "ECU_RESET", "severity": "warn", "detail": "{ecu_text} reset ({cause_text})"},
          {"id": "shutdown", "from": "someip:EcuSupervisor.UnexpectedShutdown", "type": "UNEXPECTED_SHUTDOWN", "severity": "critical", "detail": "unexpected shutdown ({cause_text}); last valid data {last_valid_ms} ms before"},
          {"id": "vdip", "from": "someip:PowerManager.VoltageDip", "type": "VOLTAGE_DIP", "severity": "warn", "detail": "12 V supply dipped to {min_v:.1f} V for {duration_ms} ms"},
          {"id": "batt_sig", "from": "signal:battery_temperature", "when": ">45", "hold_s": 10, "type": "WARN_BATT_OVERTEMP", "severity": "critical", "detail": "Battery temperature above 45 °C ({value:.1f} °C)"}]
    for w, n in (("fl", "front left"), ("fr", "front right"), ("rl", "rear left"), ("rr", "rear right")):
        ev.append({"id": f"tpms_{w}", "from": f"signal:tyre_pressure_{w}", "when": "<207", "hold_s": 60, "type": "TPMS_LOW", "severity": "warn", "detail": n + " tyre pressure low ({value:.0f} kPa, spec 241 kPa)"})
    ev += [{"id": "low12v", "from": "signal:aux_battery_voltage", "when": "<11.8", "hold_s": 20, "type": "LOW_12V", "severity": "warn", "detail": "12 V battery at {value:.2f} V"}, {"id": "dtc", "from": "uds:dtc"}]
    prof = {"schema": "vehiclegpt.profile/1", "id": "meridian_ev",
            "vehicle": {"make": "Meridian", "model": "E5 SUV", "year": 2026, "powertrain": "bev",
                        "specs": {"veh.model": "Meridian E5 SUV 82 kWh", "veh.cap_kwh": 82.0, "veh.soh": 100.0, "veh.tyre_spec_front": 241, "veh.tyre_spec_rear": 241, "veh.service_km": 12000, "veh.home_charger_kw": 11.0,
                                  "veh.sw": "3.4.2", "veh.wheelbase_m": 2.89, "veh.wheel_radius_m": 0.35, "veh.gear_ratio": 10.2, "veh.steering_ratio": 15.0, "veh.rated_max_speed": 200, "veh.seats": 5, "veh.nominal_voltage": 400}},
            "sources": [{"id": "main", "type": "can", "name": "Main CAN-FD (29-bit, Motorola)", "dbc": "main.dbc", "fd": True, "channel": "can0"}, {"id": "body", "type": "can", "name": "Body CAN 250 kbit/s", "dbc": "body.dbc", "channel": "can1"},
                        {"id": "diag", "type": "uds", "name": "UDS / OBD-II", "response_ids": ["0x7E8..0x7EF"], "dtc": "dtc.json"}, {"id": "eth", "type": "someip", "name": "SOME/IP", "catalog": "services.json"},
                        {"id": "hal", "type": "vhal", "name": "Android Automotive VHAL"}, {"id": "logs", "type": "log", "name": "QNX + logcat", "format": "slog2info"},
                        {"id": "cam", "type": "camera", "name": "Forward camera", "cams": ["front"]}, {"id": "gnss", "type": "gnss", "name": "GNSS"}],
            "signals": sig, "events": ev, "transitions": ["adas_state"],
            "thresholds": {"overtemp_c": 45, "tyre_low_kpa": 207, "low_12v": 11.8},
            "retention": {"fine_days": 14, "coarse_days": 1830, "log_days": 1825, "context_days": 1825, "archive_days": 1830, "records_days": 1825}}
    write("meridian_ev", "profile.json", json.dumps(prof, indent=1, ensure_ascii=False))


# ================================================================================================ rhino_diesel (SAE J1939)
def j1939(pri, pgn, sa=0): return ((pri << 26) | (pgn << 8) | sa) | EXT


def rhino():
    can = dbc("Rhino T4 diesel - SAE J1939 (250 kbit/s, 29-bit)", ["ENG", "ABS", "DASH", "BCM"], [
        (j1939(3, 0xF004), "EEC1", 8, "ENG", [("EngDemandTrq", 8, 8, 1, "+", 1, -125, -125, 125, "%"), ("EngActualTrq", 16, 8, 1, "+", 1, -125, -125, 125, "%"), ("EngSpeed", 24, 16, 1, "+", 0.125, 0, 0, 8031.875, "rpm")]),
        (j1939(3, 0xF003), "EEC2", 8, "ENG", [("AccelPedal", 8, 8, 1, "+", 0.4, 0, 0, 100, "%"), ("EngPctLoad", 24, 8, 1, "+", 1, 0, 0, 250, "%")]),
        (j1939(6, 0xFEEE), "ET1", 8, "ENG", [("EngCoolantTemp", 0, 8, 1, "+", 1, -40, -40, 210, "degC"), ("FuelTemp", 8, 8, 1, "+", 1, -40, -40, 210, "degC"), ("EngOilTemp", 16, 16, 1, "+", 0.03125, -273, -273, 1735, "degC")]),
        (j1939(6, 0xFEF1), "CCVS1", 8, "ABS", [("WheelBasedSpeed", 8, 16, 1, "+", 0.00390625, 0, 0, 250.996, "km/h")]),
        (j1939(6, 0xFEF2), "LFE1", 8, "ENG", [("FuelRate", 0, 16, 1, "+", 0.05, 0, 0, 3212.75, "L/h")]),
        (j1939(6, 0xFEFC, 0x17), "DD", 8, "DASH", [("FuelLevel1", 8, 8, 1, "+", 0.4, 0, 0, 100, "%")]),
        (j1939(6, 0xFEF5), "AMB", 8, "BCM", [("BaroPress", 0, 8, 1, "+", 0.5, 0, 0, 125, "kPa"), ("AmbAirTemp", 24, 16, 1, "+", 0.03125, -273, -273, 1735, "degC")]),
        (j1939(6, 0xFEF7), "VEP1", 8, "BCM", [("BatteryPotential", 32, 16, 1, "+", 0.05, 0, 0, 3212.75, "V")]),
        (j1939(6, 0xFEF4), "TIRE", 8, "BCM", [("TireLoc", 0, 8, 1, "+", 1, 0, 0, 255, "", "M"), ("PressFL", 8, 8, 1, "+", 4, 0, 0, 1000, "kPa", "m0"), ("PressFR", 8, 8, 1, "+", 4, 0, 0, 1000, "kPa", "m1"),
                                                ("PressRL", 8, 8, 1, "+", 4, 0, 0, 1000, "kPa", "m2"), ("PressRR", 8, 8, 1, "+", 4, 0, 0, 1000, "kPa", "m3"), ("TireTemp", 16, 16, 1, "+", 0.03125, -273, -273, 1735, "degC")]),
        (j1939(6, 0xFEC1), "VDHR", 8, "DASH", [("HiResDistance", 0, 32, 1, "+", 5, 0, 0, 21055406000, "m")]),
        (j1939(3, 0xF001), "EBC1", 8, "ABS", [("BrakePedalPos", 8, 8, 1, "+", 0.4, 0, 0, 100, "%")]),
        (j1939(6, 0xFEF6), "IC1", 8, "ENG", [("BoostPress", 8, 8, 1, "+", 2, 0, 0, 500, "kPa"), ("ExhaustGasTemp", 40, 16, 1, "+", 0.03125, -273, -273, 1735, "degC")]),
        (j1939(6, 0xFE56), "AT1T1I", 8, "ENG", [("DEFLevel", 0, 8, 1, "+", 0.4, 0, 0, 100, "%")]),
        (j1939(7, 0xFD7C), "DPF1S", 8, "ENG", [("DPFSootLoad", 0, 8, 1, "+", 0.4, 0, 0, 100, "%"), ("DPFRegenStatus", 8, 8, 1, "+", 1, 0, 0, 3, "")]),
        (j1939(6, 0xFEDB), "IMR1", 8, "ENG", [("RailPress", 16, 16, 1, "+", 100, 0, 0, 250000, "kPa")]),
        (j1939(3, 0xF009), "VDC2", 8, "ABS", [("SteerWheelAngle", 0, 16, 1, "+", 0.0009765625, -31.374, -31.374, 32.2, "rad"), ("YawRate", 16, 16, 1, "+", 0.0001220703, -3.92, -3.92, 4.0, "rad/s"),
                                                ("LateralAccel", 32, 16, 1, "+", 0.00048828, -15.687, -15.687, 16.4, "m/s2"), ("LongitudinalAccel", 48, 8, 1, "+", 0.1, -12.5, -12.5, 13.0, "m/s2")])])
    write("rhino_diesel", "j1939.dbc", can)
    write("rhino_diesel", "services.json", "{}\n")
    write("rhino_diesel", "dtc.json", json.dumps({
        "P0217": {"text": "Engine coolant over-temperature condition", "subsystem": "cooling", "severity": "critical"}, "P0299": {"text": "Turbocharger underboost", "subsystem": "air", "severity": "warn"},
        "P2002": {"text": "Diesel particulate filter efficiency below threshold", "subsystem": "aftertreatment", "severity": "warn"}, "P0087": {"text": "Fuel rail pressure too low", "subsystem": "fuel", "severity": "warn"},
        "P0562": {"text": "System voltage low", "subsystem": "power", "severity": "warn"}, "U0100": {"text": "Lost communication with engine control module", "subsystem": "network", "severity": "warn"},
        "U0121": {"text": "Lost communication with anti-lock brake system module", "subsystem": "network", "severity": "warn"}, "C0750": {"text": "Tyre pressure low - front left", "subsystem": "tyres", "severity": "warn"}}, indent=1))
    S = lambda concept, frm, unit, rate=1.0, **kw: dict({"concept": concept, "from": frm, "unit": unit, "rate_hz": rate}, **kw)
    sig = [S("speed", "can:j1939/CCVS1.WheelBasedSpeed", "km/h", 10), S("engine_rpm", "can:j1939/EEC1.EngSpeed", "rpm", 20), dict(S("engine_torque", "can:j1939/EEC1.EngActualTrq", "Nm", 20, scale=4.5), valid=[-650, 650]),
           S("engine_load", "can:j1939/EEC2.EngPctLoad", "%", 10), S("accelerator_position", "can:j1939/EEC2.AccelPedal", "%", 10), S("motor_temperature", "can:j1939/ET1.EngCoolantTemp", "C", 1),
           S("fuel_rate", "can:j1939/LFE1.FuelRate", "L/h", 10), S("fuel_level", "can:j1939/DD.FuelLevel1", "%", 1), S("ambient_temperature", "can:j1939/AMB.AmbAirTemp", "C", 1),
           S("aux_battery_voltage", "can:j1939/VEP1.BatteryPotential", "V", 1), S("brake_pedal", "can:j1939/EBC1.BrakePedalPos", "%", 10), S("steering_angle", "can:j1939/VDC2.SteerWheelAngle", "rad", 10),
           S("yaw_rate", "can:j1939/VDC2.YawRate", "rad/s", 10), S("lateral_acceleration", "can:j1939/VDC2.LateralAccel", "m/s2", 10), S("longitudinal_acceleration", "can:j1939/VDC2.LongitudinalAccel", "m/s2", 10),
           S("odometer", "can:j1939/VDHR.HiResDistance", "m", 1), S("tyre_pressure_fl", "can:j1939/TIRE.PressFL", "kPa", 1), S("tyre_pressure_fr", "can:j1939/TIRE.PressFR", "kPa", 1), S("tyre_pressure_rl", "can:j1939/TIRE.PressRL", "kPa", 1),
           S("tyre_pressure_rr", "can:j1939/TIRE.PressRR", "kPa", 1), S("tyre_temperature", "can:j1939/TIRE.TireTemp", "C", 1), dict(S("_ignition", "can:j1939/EEC1.EngSpeed", "rpm", 20), when=">300"),
           S("engine_oil_temperature", "can:j1939/ET1.EngOilTemp", "C", 1), S("barometric_pressure", "can:j1939/AMB.BaroPress", "kPa", 1), S("fuel_temperature", "can:j1939/ET1.FuelTemp", "C", 1),
           S("boost_pressure", "can:j1939/IC1.BoostPress", "kPa", 5), S("exhaust_gas_temperature", "can:j1939/IC1.ExhaustGasTemp", "C", 1), S("def_level", "can:j1939/AT1T1I.DEFLevel", "%", 1),
           S("dpf_soot_load", "can:j1939/DPF1S.DPFSootLoad", "%", 1), S("dpf_regeneration", "can:j1939/DPF1S.DPFRegenStatus", "none", 1), S("fuel_rail_pressure", "can:j1939/IMR1.RailPress", "kPa", 5)]
    ev = [{"id": "overtemp", "from": "signal:motor_temperature", "when": ">105", "hold_s": 10, "type": "WARN_OVERTEMP", "severity": "critical", "detail": "Engine coolant above 105 °C ({value:.1f} °C)"},
          {"id": "log_dpf", "from": "log:any", "match": "dpf regeneration", "type": "DTC_SET", "severity": "info", "detail": "DPF regeneration started"},
          {"id": "log_reset", "from": "log:any", "match": "watchdog reset", "max_severity": 3, "type": "ECU_RESET", "severity": "warn"},
          {"id": "log_brownout", "from": "log:any", "match": "brown-out", "max_severity": 3, "type": "VOLTAGE_DIP", "severity": "warn"},
          {"id": "low12v", "from": "signal:aux_battery_voltage", "when": "<11.6", "hold_s": 20, "type": "LOW_12V", "severity": "warn", "detail": "12 V battery at {value:.2f} V"}]
    for w, n in (("fl", "front left"), ("fr", "front right"), ("rl", "rear left"), ("rr", "rear right")):
        ev.append({"id": f"tpms_{w}", "from": f"signal:tyre_pressure_{w}", "when": "<200", "hold_s": 60, "type": "TPMS_LOW", "severity": "warn", "detail": n + " tyre pressure low ({value:.0f} kPa, spec 250 kPa)"})
    ev.append({"id": "dtc", "from": "uds:dtc"})
    prof = {"schema": "vehiclegpt.profile/1", "id": "rhino_diesel",
            "vehicle": {"make": "Rhino", "model": "T4 light truck", "year": 2025, "powertrain": "diesel",
                        "specs": {"veh.model": "Rhino T4 2.8 TD", "veh.tyre_spec_front": 250, "veh.tyre_spec_rear": 250, "veh.service_km": 20000, "veh.sw": "tcu 2.1", "veh.wheelbase_m": 3.2, "veh.wheel_radius_m": 0.37,
                                  "veh.rated_max_speed": 140, "veh.seats": 3, "veh.tank_l": 90}},
            "concepts": [{"name": "engine_oil_temperature", "label": "engine oil temperature", "unit": "C", "category": "powertrain", "synonyms": ["oil temperature", "oil temp", "engine oil temp", "engine oil temperature"], "lo": -40, "hi": 200},
                         {"name": "barometric_pressure", "label": "barometric pressure", "unit": "kPa", "category": "environment", "synonyms": ["barometric pressure", "air pressure", "atmospheric pressure", "baro pressure"], "lo": 30, "hi": 125},
                         {"name": "fuel_temperature", "label": "fuel temperature", "unit": "C", "category": "powertrain", "synonyms": ["fuel temperature", "fuel temp", "diesel temperature"], "lo": -40, "hi": 120},
                         {"name": "boost_pressure", "label": "boost pressure", "unit": "kPa", "category": "powertrain", "synonyms": ["boost pressure", "boost", "turbo pressure", "turbo boost", "intake manifold pressure"], "lo": 0, "hi": 500},
                         {"name": "exhaust_gas_temperature", "label": "exhaust gas temperature", "unit": "C", "category": "powertrain", "synonyms": ["exhaust gas temperature", "exhaust temperature", "egt", "exhaust temp", "exhaust"], "lo": -40, "hi": 1000},
                         {"name": "def_level", "label": "AdBlue level", "unit": "%", "category": "powertrain", "synonyms": ["adblue level", "adblue", "def level", "diesel exhaust fluid", "urea level", "scr fluid"], "lo": 0, "hi": 100},
                         {"name": "dpf_soot_load", "label": "particulate filter soot load", "unit": "%", "category": "powertrain", "synonyms": ["soot load", "dpf soot load", "particulate filter", "dpf load", "dpf", "particulate filter load", "soot"], "lo": 0, "hi": 100},
                         {"name": "dpf_regeneration", "label": "particulate filter regeneration", "unit": "none", "category": "powertrain", "synonyms": ["dpf regeneration", "filter regeneration", "regeneration status", "dpf regen"], "lo": 0, "hi": 3},
                         {"name": "fuel_rail_pressure", "label": "fuel rail pressure", "unit": "kPa", "category": "powertrain", "synonyms": ["fuel rail pressure", "rail pressure", "common rail pressure"], "lo": 0, "hi": 250000}],
            "sources": [{"id": "j1939", "type": "can", "name": "SAE J1939 powertrain CAN 250 kbit/s", "dbc": "j1939.dbc", "channel": "can0"}, {"id": "diag", "type": "uds", "name": "UDS / OBD-II", "response_ids": ["0x7E8..0x7EF"], "dtc": "dtc.json"},
                        {"id": "tcu", "type": "log", "name": "Linux telematics unit (syslog)", "format": "rfc3164"}, {"id": "gnss", "type": "gnss", "name": "GNSS"}],
            "signals": sig, "events": ev, "thresholds": {"overtemp_c": 105, "tyre_low_kpa": 200, "low_12v": 11.6},
            "retention": {"fine_days": 14, "coarse_days": 1830, "log_days": 1825, "context_days": 1825, "archive_days": 1830, "records_days": 1825}}
    write("rhino_diesel", "profile.json", json.dumps(prof, indent=1, ensure_ascii=False))


# ================================================================================================ kestrel_petrol (OBD-II style engine data on an 11-bit CAN)
def kestrel():
    pt = dbc("Kestrel C3 1.5 petrol - powertrain CAN 500 kbit/s (11-bit)", ["ECM", "TCM", "ABS", "BCM", "CLU"], [
        (0x201, "ENG_DATA", 8, "ECM", [("EngSpeed", 0, 16, 1, "+", 0.25, 0, 0, 8000, "rpm"), ("EngLoad", 16, 8, 1, "+", 0.4, 0, 0, 100, "%"), ("ThrottlePos", 24, 8, 1, "+", 0.4, 0, 0, 100, "%"), ("CoolantTemp", 32, 8, 1, "+", 1, -40, -40, 215, "degC"),
                                        ("OilTemp", 40, 8, 1, "+", 1, -40, -40, 215, "degC")]),
        (0x202, "ENG_AIR", 8, "ECM", [("MAP", 0, 8, 1, "+", 1, 0, 0, 255, "kPa"), ("IntakeAirTemp", 8, 8, 1, "+", 1, -40, -40, 215, "degC"), ("MAF", 16, 16, 1, "+", 0.01, 0, 0, 655, "g/s"), ("BaroPress", 32, 8, 1, "+", 1, 0, 0, 255, "kPa")]),
        (0x203, "ENG_FUEL", 8, "ECM", [("FuelRate", 0, 16, 1, "+", 0.01, 0, 0, 100, "L/h"), ("FuelLevel", 16, 8, 1, "+", 0.4, 0, 0, 100, "%"), ("FuelTrimST", 24, 8, 1, "+", 0.78125, -100, -100, 99.2, "%"), ("FuelTrimLT", 32, 8, 1, "+", 0.78125, -100, -100, 99.2, "%")]),
        (0x204, "ENG_EXH", 8, "ECM", [("Lambda", 0, 16, 1, "+", 0.001, 0, 0, 2, ""), ("CatalystTemp", 16, 16, 1, "+", 0.1, -40, -40, 1000, "degC"), ("IgnTiming", 32, 8, 1, "+", 0.5, -64, -64, 63.5, "deg")]),
        (0x300, "VEH_DYN", 8, "ABS", [("VehSpeed", 0, 16, 1, "+", 0.01, 0, 0, 300, "km/h"), ("BrakePedal", 16, 8, 1, "+", 0.4, 0, 0, 100, "%"), ("SteerAngle", 24, 16, 1, "-", 0.1, 0, -780, 780, "deg"), ("YawRate", 40, 16, 1, "-", 0.01, 0, -120, 120, "deg/s")]),
        (0x301, "VEH_ACC", 8, "ABS", [("LatAcc", 0, 16, 1, "-", 0.001, 0, -3, 3, "g"), ("LongAcc", 16, 16, 1, "-", 0.001, 0, -3, 3, "g")]),
        (0x3A0, "TPMS", 8, "BCM", [("PressFL", 0, 8, 1, "+", 2, 0, 0, 510, "kPa"), ("PressFR", 8, 8, 1, "+", 2, 0, 0, 510, "kPa"), ("PressRL", 16, 8, 1, "+", 2, 0, 0, 510, "kPa"), ("PressRR", 24, 8, 1, "+", 2, 0, 0, 510, "kPa"),
                                  ("TyreTemp", 32, 8, 1, "+", 1, -40, -40, 215, "degC")]),
        (0x3A8, "BODY_ENV", 8, "BCM", [("AmbientTemp", 0, 8, 1, "+", 0.5, -40, -40, 87, "degC"), ("Batt12V", 8, 16, 1, "+", 0.001, 0, 0, 20, "V"), ("Rain", 24, 8, 1, "+", 1, 0, 0, 3, "")]),
        (0x3B0, "CLUSTER", 8, "CLU", [("Odometer", 0, 32, 1, "+", 0.1, 0, 0, 999999, "km")])])
    write("kestrel_petrol", "powertrain.dbc", pt); write("kestrel_petrol", "services.json", "{}\n")
    write("kestrel_petrol", "dtc.json", json.dumps({
        "P0171": {"text": "System too lean (bank 1)", "subsystem": "fuel", "severity": "warn"}, "P0300": {"text": "Random / multiple cylinder misfire detected", "subsystem": "ignition", "severity": "warn"},
        "P0420": {"text": "Catalyst system efficiency below threshold (bank 1)", "subsystem": "emissions", "severity": "warn"}, "P0113": {"text": "Intake air temperature sensor high input", "subsystem": "air", "severity": "info"},
        "P0217": {"text": "Engine coolant over-temperature condition", "subsystem": "cooling", "severity": "critical"}, "P0562": {"text": "System voltage low", "subsystem": "power", "severity": "warn"},
        "U0100": {"text": "Lost communication with engine control module", "subsystem": "network", "severity": "warn"}, "C0750": {"text": "Tyre pressure low - front left", "subsystem": "tyres", "severity": "warn"}}, indent=1))
    S = lambda concept, frm, unit, rate=1.0, **kw: dict({"concept": concept, "from": frm, "unit": unit, "rate_hz": rate}, **kw)
    sig = [S("speed", "can:pt/VEH_DYN.VehSpeed", "km/h", 10), S("engine_rpm", "can:pt/ENG_DATA.EngSpeed", "rpm", 20), S("engine_load", "can:pt/ENG_DATA.EngLoad", "%", 10), S("accelerator_position", "can:pt/ENG_DATA.ThrottlePos", "%", 10),
           S("motor_temperature", "can:pt/ENG_DATA.CoolantTemp", "C", 1), S("engine_oil_temperature", "can:pt/ENG_DATA.OilTemp", "C", 1), S("fuel_rate", "can:pt/ENG_FUEL.FuelRate", "L/h", 10), S("fuel_level", "can:pt/ENG_FUEL.FuelLevel", "%", 1),
           S("intake_manifold_pressure", "can:pt/ENG_AIR.MAP", "kPa", 10), S("intake_air_temperature", "can:pt/ENG_AIR.IntakeAirTemp", "C", 1), S("mass_air_flow", "can:pt/ENG_AIR.MAF", "g/s", 10), S("barometric_pressure", "can:pt/ENG_AIR.BaroPress", "kPa", 1),
           S("short_term_fuel_trim", "can:pt/ENG_FUEL.FuelTrimST", "%", 5), S("long_term_fuel_trim", "can:pt/ENG_FUEL.FuelTrimLT", "%", 1), S("lambda", "can:pt/ENG_EXH.Lambda", "none", 10), S("catalyst_temperature", "can:pt/ENG_EXH.CatalystTemp", "C", 1),
           S("ignition_timing", "can:pt/ENG_EXH.IgnTiming", "deg", 10), S("brake_pedal", "can:pt/VEH_DYN.BrakePedal", "%", 10), S("steering_angle", "can:pt/VEH_DYN.SteerAngle", "deg", 10), S("yaw_rate", "can:pt/VEH_DYN.YawRate", "deg/s", 10),
           S("lateral_acceleration", "can:pt/VEH_ACC.LatAcc", "g", 10), S("longitudinal_acceleration", "can:pt/VEH_ACC.LongAcc", "g", 10), S("tyre_pressure_fl", "can:pt/TPMS.PressFL", "kPa", 1), S("tyre_pressure_fr", "can:pt/TPMS.PressFR", "kPa", 1),
           S("tyre_pressure_rl", "can:pt/TPMS.PressRL", "kPa", 1), S("tyre_pressure_rr", "can:pt/TPMS.PressRR", "kPa", 1), S("tyre_temperature", "can:pt/TPMS.TyreTemp", "C", 1), S("ambient_temperature", "can:pt/BODY_ENV.AmbientTemp", "C", 1),
           S("aux_battery_voltage", "can:pt/BODY_ENV.Batt12V", "V", 1), S("rain", "can:pt/BODY_ENV.Rain", "none", 1), S("odometer", "can:pt/CLUSTER.Odometer", "km", 1), dict(S("_ignition", "can:pt/ENG_DATA.EngSpeed", "rpm", 20), when=">300")]
    ev = [{"id": "overtemp", "from": "signal:motor_temperature", "when": ">105", "hold_s": 10, "type": "WARN_OVERTEMP", "severity": "critical", "detail": "Engine coolant above 105 °C ({value:.1f} °C)"},
          {"id": "log_reset", "from": "log:any", "match": "watchdog reset", "max_severity": 3, "type": "ECU_RESET", "severity": "warn"},
          {"id": "log_brownout", "from": "log:any", "match": "brown-out", "max_severity": 3, "type": "VOLTAGE_DIP", "severity": "warn"},
          {"id": "low12v", "from": "signal:aux_battery_voltage", "when": "<11.6", "hold_s": 20, "type": "LOW_12V", "severity": "warn", "detail": "12 V battery at {value:.2f} V"}]
    for w, n in (("fl", "front left"), ("fr", "front right"), ("rl", "rear left"), ("rr", "rear right")):
        ev.append({"id": f"tpms_{w}", "from": f"signal:tyre_pressure_{w}", "when": "<200", "hold_s": 60, "type": "TPMS_LOW", "severity": "warn", "detail": n + " tyre pressure low ({value:.0f} kPa, spec 230 kPa)"})
    ev.append({"id": "dtc", "from": "uds:dtc"})
    prof = {"schema": "vehiclegpt.profile/1", "id": "kestrel_petrol",
            "vehicle": {"make": "Kestrel", "model": "C3 hatchback", "year": 2024, "powertrain": "petrol",
                        "specs": {"veh.model": "Kestrel C3 1.5 petrol", "veh.tyre_spec_front": 230, "veh.tyre_spec_rear": 230, "veh.service_km": 15000, "veh.sw": "ecm 4.2", "veh.wheelbase_m": 2.6, "veh.wheel_radius_m": 0.31,
                                  "veh.rated_max_speed": 185, "veh.seats": 5, "veh.tank_l": 50}},
            "concepts": [{"name": "engine_oil_temperature", "label": "engine oil temperature", "unit": "C", "category": "powertrain", "synonyms": ["oil temperature", "oil temp", "engine oil temp", "engine oil temperature"], "lo": -40, "hi": 200},
                         {"name": "barometric_pressure", "label": "barometric pressure", "unit": "kPa", "category": "environment", "synonyms": ["barometric pressure", "air pressure", "atmospheric pressure", "baro pressure"], "lo": 30, "hi": 125},
                         {"name": "intake_manifold_pressure", "label": "intake manifold pressure", "unit": "kPa", "category": "powertrain", "synonyms": ["intake manifold pressure", "manifold pressure", "map", "manifold absolute pressure"], "lo": 0, "hi": 255},
                         {"name": "intake_air_temperature", "label": "intake air temperature", "unit": "C", "category": "powertrain", "synonyms": ["intake air temperature", "intake air temp", "air intake temperature", "iat"], "lo": -40, "hi": 215},
                         {"name": "mass_air_flow", "label": "mass air flow", "unit": "g/s", "category": "powertrain", "synonyms": ["mass air flow", "air flow", "maf", "airflow"], "lo": 0, "hi": 655},
                         {"name": "short_term_fuel_trim", "label": "short-term fuel trim", "unit": "%", "category": "powertrain", "synonyms": ["short term fuel trim", "short-term fuel trim", "stft", "fuel trim"], "lo": -100, "hi": 100},
                         {"name": "long_term_fuel_trim", "label": "long-term fuel trim", "unit": "%", "category": "powertrain", "synonyms": ["long term fuel trim", "long-term fuel trim", "ltft"], "lo": -100, "hi": 100},
                         {"name": "lambda", "label": "air-fuel ratio (lambda)", "unit": "none", "category": "powertrain", "synonyms": ["lambda", "air fuel ratio", "air-fuel ratio", "afr", "oxygen sensor"], "lo": 0, "hi": 2},
                         {"name": "catalyst_temperature", "label": "catalyst temperature", "unit": "C", "category": "powertrain", "synonyms": ["catalyst temperature", "catalytic converter temperature", "cat temperature", "catalyst temp", "catalyst", "catalytic converter"], "lo": -40, "hi": 1000},
                         {"name": "ignition_timing", "label": "ignition timing", "unit": "deg", "category": "powertrain", "synonyms": ["ignition timing", "spark timing", "spark advance", "timing advance"], "lo": -64, "hi": 64}],
            "sources": [{"id": "pt", "type": "can", "name": "Powertrain CAN 500 kbit/s (11-bit, OBD-II style)", "dbc": "powertrain.dbc", "channel": "can0", "bitrate": 500000}, {"id": "diag", "type": "uds", "name": "UDS / OBD-II", "response_ids": ["0x7E8..0x7EF"], "dtc": "dtc.json"},
                        {"id": "tcu", "type": "log", "name": "Linux telematics unit (syslog)", "format": "rfc3164"}, {"id": "gnss", "type": "gnss", "name": "GNSS"}],
            "signals": sig, "events": ev, "thresholds": {"overtemp_c": 105, "tyre_low_kpa": 200, "low_12v": 11.6},
            "retention": {"fine_days": 14, "coarse_days": 1830, "log_days": 1825, "context_days": 1825, "archive_days": 1830, "records_days": 1825}}
    write("kestrel_petrol", "profile.json", json.dumps(prof, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    meridian(); rhino(); kestrel()
    from vgpt.vehicle.profile import Profile
    for name in ("aster_ev", "meridian_ev", "rhino_diesel", "kestrel_petrol"):
        p = Profile.load(name); iss = p.validate(); print(name, p.summary()["concepts"], "concepts;", "issues:", [str(i) for i in iss] or "none")

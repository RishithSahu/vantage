"""The demonstration cars that carry the in-vehicle protocols the case study's makers use beyond CAN: DoIP, LIN, CANopen and FlexRay.  Each is derived from an existing pack (so it also has CAN, SOME/IP, VHAL, logs, GNSS)
and written as a normal profile pack + car model; the round trip of tools_make_cars (drive the car, put its state on the buses, decode with the profile's own router, compare) must pass.

    python tools_make_protocol_cars.py [--only nova_ev,...]

    nova_ev       battery-electric saloon, DoIP gateway for diagnostics (UDS over automotive Ethernet, ISO 13400), LIN body network (rain / light sensor, climate sensor, driver seat)
    volta_bus     city bus, CANopen (PDOs, heartbeat, emergency, SDO)
    prestige_gt   premium coupe, FlexRay chassis cluster + LIN body network
The makers are fictional ("synthetic maker"): the layouts are written for the test bench, the protocols are the real ones and are checked against independent implementations (docs/PROTOCOL_EVIDENCE.md)."""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from tools_make_cars import MODELS, PACKS, RETENTION, roundtrip  # noqa: E402
from vgpt.vehicle.profile import Profile, ProfileError  # noqa: E402

DEMO = os.path.join(ROOT, "vgpt", "vehicle", "protocol_demo")


def load(base):
    d = os.path.join(PACKS, base); return json.load(open(os.path.join(d, "profile.json"), encoding="utf-8")), d


def drop(spec, concepts):
    spec["signals"] = [s for s in spec["signals"] if not (s["concept"] in concepts and s["from"].startswith("can:"))]


def write(spec, base_dir, pid, meta):
    out = os.path.join(PACKS, pid); shutil.rmtree(out, ignore_errors=True); shutil.copytree(base_dir, out)
    if pid == "volta_bus":                                                                             # a bus has none of the car pack's files except the DTC table
        for f in os.listdir(out):
            if f != "dtc.json": os.remove(os.path.join(out, f))
    for extra in meta.pop("files", []): shutil.copyfile(os.path.join(DEMO, extra), os.path.join(out, extra))
    spec["id"] = pid; spec["retention"] = dict(RETENTION)
    json.dump(spec, open(os.path.join(out, "profile.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    p = Profile.load(out); issues = p.validate(); errors = [i for i in issues if i.level == "error"]
    if errors: raise ProfileError(f"{pid}: " + "; ".join(str(i) for i in errors))
    checked, bad = roundtrip(p)
    if bad: raise ProfileError(f"{pid}: round trip failed:\n  " + "\n  ".join(bad))
    for d in (os.path.join(MODELS, pid),):
        shutil.rmtree(d, ignore_errors=True); shutil.copytree(out, d)
    info = {"id": pid, "decoder_installed": True, "bound": len(p.bindings), "concepts": len(p.concept_names()), "extension": 0, **meta}
    json.dump(info, open(os.path.join(MODELS, pid + ".json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    return p, checked


def nova_ev():
    spec, base = load("meridian_ev"); v = spec["vehicle"]; v.update(make="Nova", model="N7 Ethernet", year=2026); v["specs"]["veh.model"] = "Nova N7 Ethernet 82 kWh"
    drop(spec, {"rain", "light", "cabin_temperature"})                                                 # these now come from the LIN body network
    spec["sources"] = [s for s in spec["sources"] if s["id"] != "diag"]
    spec["sources"].insert(2, {"id": "diag", "type": "uds", "name": "UDS over DoIP (ISO 13400, automotive Ethernet)", "transport": "doip", "tester": "0x0E80", "entity": "0x1001", "eid": "0050c2a1b2c3", "gid": "0050c2a1b2c3", "dtc": "dtc.json",
                                "dids": {"0xF40D": {"concept": "speed", "fmt": ">B", "unit": "km/h", "rate_hz": 2, "priority": 2}, "0xF446": {"concept": "ambient_temperature", "fmt": ">B", "offset": -40, "unit": "C", "rate_hz": 0.5, "priority": 2},
                                         "0xF442": {"concept": "aux_battery_voltage", "fmt": ">H", "scale": 0.001, "unit": "V", "rate_hz": 1, "priority": 2}}})
    spec["sources"].append({"id": "lin1", "type": "lin", "name": "LIN 2.1 body network, 19.2 kbit/s (rain / light, climate, driver seat)", "ldf": "body.ldf", "channel": "lin0", "enhanced": True})
    spec["signals"] += [{"concept": "rain", "from": "lin:lin1/RainLight.RainIntensity", "unit": "none", "rate_hz": 2},
                        {"concept": "light", "from": "lin:lin1/RainLight.AmbientLight", "unit": "lux", "rate_hz": 2},
                        {"concept": "cabin_temperature", "from": "lin:lin1/ClimateSensor.CabinTemp", "unit": "C", "rate_hz": 1},
                        {"concept": "humidity", "from": "lin:lin1/ClimateSensor.CabinHumidity", "unit": "%", "rate_hz": 0.5},
                        {"concept": "seatbelt_state", "from": "lin:lin1/SeatDriver.SeatBelt", "unit": "none", "rate_hz": 2, "map": {"on": 1, "off": 0}}]
    meta = dict(name="Nova N7 Ethernet", year=2026, region="anywhere", powertrain="bev", **{"class": "Battery-electric saloon: CAN-FD, SOME/IP, Android VHAL, DoIP diagnostics, LIN body network"}, files=["body.ldf"],
                interfaces=["CAN-FD main bus and body CAN", "SOME/IP on automotive Ethernet", "Android Vehicle HAL", "UDS over DoIP (ISO 13400): diagnostic trouble codes and data identifiers, announced with the VIN", "LIN 2.1 body network (rain / light sensor, climate sensor, driver seat)", "logcat / slog2info", "NMEA 0183 GNSS", "forward camera"],
                notes="A fictional maker. The DoIP gateway and the LIN network are written for the test bench; the protocols are the real ones.", source="written for the test bench (tools_make_protocol_cars.py)")
    return spec, base, meta


def prestige_gt():
    spec, base = load("kestrel_petrol"); v = spec["vehicle"]; v.update(make="Prestige", model="GT V8 FlexRay", year=2026); v["specs"]["veh.model"] = "Prestige GT V8 FlexRay"
    spec["sources"].append({"id": "chassis", "type": "flexray", "name": "FlexRay chassis cluster, 10 Mbit/s, 5 ms cycle (wheel speeds, dynamics, brake)", "description": "chassis.fx.json"})
    spec["sources"].append({"id": "lin1", "type": "lin", "name": "LIN 2.1 body network, 19.2 kbit/s (rain / light, climate, driver seat)", "ldf": "body.ldf", "channel": "lin0", "enhanced": True})
    drop(spec, {"cabin_temperature", "rain", "light", "steering_angle", "brake_pedal", "yaw_rate", "lateral_acceleration", "longitudinal_acceleration", "brake_pressure", "brake_temperature", "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr"})
    spec["signals"] += [
        {"concept": "wheel_speed_fl", "from": "flexray:chassis/WheelSpeeds.FL", "unit": "km/h", "rate_hz": 20}, {"concept": "wheel_speed_fr", "from": "flexray:chassis/WheelSpeeds.FR", "unit": "km/h", "rate_hz": 20},
        {"concept": "wheel_speed_rl", "from": "flexray:chassis/WheelSpeeds.RL", "unit": "km/h", "rate_hz": 20}, {"concept": "wheel_speed_rr", "from": "flexray:chassis/WheelSpeeds.RR", "unit": "km/h", "rate_hz": 20},
        {"concept": "yaw_rate", "from": "flexray:chassis/Dynamics.YawRate", "unit": "deg/s", "rate_hz": 20}, {"concept": "lateral_acceleration", "from": "flexray:chassis/Dynamics.LatAcc", "unit": "g", "rate_hz": 20},
        {"concept": "longitudinal_acceleration", "from": "flexray:chassis/Dynamics.LongAcc", "unit": "g", "rate_hz": 20}, {"concept": "steering_angle", "from": "flexray:chassis/Dynamics.SteerAngle", "unit": "deg", "rate_hz": 20},
        {"concept": "brake_pressure", "from": "flexray:chassis/Brake.Pressure", "unit": "bar", "rate_hz": 10}, {"concept": "brake_pedal", "from": "flexray:chassis/Brake.PedalPos", "unit": "%", "rate_hz": 10},
        {"concept": "brake_temperature", "from": "flexray:chassis/Brake.DiscTempMax", "unit": "C", "rate_hz": 2},
        {"concept": "rain", "from": "lin:lin1/RainLight.RainIntensity", "unit": "none", "rate_hz": 2}, {"concept": "light", "from": "lin:lin1/RainLight.AmbientLight", "unit": "lux", "rate_hz": 2},
        {"concept": "cabin_temperature", "from": "lin:lin1/ClimateSensor.CabinTemp", "unit": "C", "rate_hz": 1}, {"concept": "humidity", "from": "lin:lin1/ClimateSensor.CabinHumidity", "unit": "%", "rate_hz": 0.5},
        {"concept": "seatbelt_state", "from": "lin:lin1/SeatDriver.SeatBelt", "unit": "none", "rate_hz": 2, "map": {"on": 1, "off": 0}}]
    meta = dict(name="Prestige GT V8 FlexRay", year=2026, region="anywhere", powertrain="petrol", **{"class": "Premium grand tourer, petrol: powertrain CAN, FlexRay chassis cluster, LIN body network"}, files=["chassis.fx.json", "body.ldf"],
                interfaces=["Powertrain CAN 500 kbit/s (OBD-style)", "FlexRay chassis cluster 10 Mbit/s (wheel speeds, yaw, accelerations, steering angle, brake pressure)", "LIN 2.1 body network", "UDS / OBD-II diagnostics", "syslog", "NMEA 0183 GNSS"],
                notes="A fictional maker. The FlexRay cluster description is written for the test bench; the protocol is the real one (specification 2.1).", source="written for the test bench (tools_make_protocol_cars.py)")
    return spec, base, meta


def canopen_dbc():
    """The PDO mapping of the demonstration bus as a DBC (a TPDO is a CAN message whose identifier is 0x180 / 0x280 / .. plus the node number)."""
    def msg(cid, name, dlc, node, sigs):
        out = [f"BO_ {cid} {name}: {dlc} {node}"]
        for n, start, ln, signed, f, o, lo, hi, u in sigs: out.append(f' SG_ {n} : {start}|{ln}@1{"-" if signed else "+"} ({f},{o}) [{lo}|{hi}] "{u}" Vector__XXX')
        return "\n".join(out) + "\n"
    parts = ['VERSION "CANopen PDO mapping of the Volta city bus (written for the test bench)"', "", "NS_ :", "", "BS_:", "", "BU_: BMS INVERTER CHARGER HVAC TPMS", ""]
    parts += [msg(0x190, "BMS_TPDO1", 6, "BMS", [("PackSoc", 0, 16, False, 0.1, 0, 0, 100, "%"), ("PackVoltage", 16, 16, False, 0.1, 0, 0, 900, "V"), ("PackCurrent", 32, 16, True, 0.1, 0, -1000, 1000, "A")]),
              msg(0x290, "BMS_TPDO2", 6, "BMS", [("PackTemp", 0, 16, True, 0.1, 0, -40, 90, "degC"), ("CellTempMax", 16, 16, True, 0.1, 0, -40, 110, "degC"), ("CellTempMin", 32, 16, True, 0.1, 0, -40, 110, "degC")]),
              msg(0x390, "BMS_TPDO3", 4, "BMS", [("CellVoltMin", 0, 16, False, 0.001, 0, 0, 5, "V"), ("Charging", 16, 1, False, 1, 0, 0, 1, "")]),
              msg(0x1A0, "INV_TPDO1", 8, "INVERTER", [("MotorRpm", 0, 16, True, 1, 0, -20000, 20000, "rpm"), ("MotorTorque", 16, 16, True, 0.1, 0, -3000, 3000, "Nm"), ("MotorTemp", 32, 16, True, 0.1, 0, -40, 200, "degC"), ("VehSpeed", 48, 16, False, 0.01, 0, 0, 120, "km/h")]),
              msg(0x1B0, "CHG_TPDO1", 4, "CHARGER", [("ChargePower", 0, 16, False, 0.1, 0, 0, 450, "kW")]),
              msg(0x1C0, "HVAC_TPDO1", 6, "HVAC", [("OutsideTemp", 0, 16, True, 0.1, 0, -50, 60, "degC"), ("CabinTemp", 16, 16, True, 0.1, 0, -20, 80, "degC"), ("HvacPower", 32, 16, False, 1, 0, 0, 30000, "W")]),
              msg(0x1D0, "TPMS_TPDO1", 8, "TPMS", [("TyreFL", 0, 16, False, 0.1, 0, 0, 1000, "kPa"), ("TyreFR", 16, 16, False, 0.1, 0, 0, 1000, "kPa"), ("TyreRL", 32, 16, False, 0.1, 0, 0, 1000, "kPa"), ("TyreRR", 48, 16, False, 0.1, 0, 0, 1000, "kPa")])]
    return "\n".join(parts) + "\n"


def volta_bus():
    def b(concept, key, unit, rate, **kw): return dict({"concept": concept, "from": f"can:co/{key}", "unit": unit, "rate_hz": rate}, **kw)
    sources = [{"id": "co", "type": "can", "canopen": True, "name": "CANopen drive and battery network, 250 kbit/s (CiA 301; PDO mapping, heartbeat, emergency, SDO)", "dbc": "volta_canopen.dbc", "channel": "can0", "bitrate": 250000, "nodes": ["0x10", "0x20", "0x30", "0x40", "0x50"],
                "od": {"0x2000:0": {"node": "0x20", "concept": "odometer", "fmt": "<I", "scale": 0.1, "unit": "km", "rate_hz": 0.2}, "0x2001:0": {"node": "0x10", "concept": "aux_battery_voltage", "fmt": "<H", "scale": 0.001, "unit": "V", "rate_hz": 0.5}}},
               {"id": "diag", "type": "uds", "name": "UDS / OBD-II (ISO 15765-4)", "response_ids": ["0x7E8..0x7EF"], "dtc": "dtc.json"},
               {"id": "logs", "type": "log", "name": "Linux telematics unit (syslog)", "format": "rfc3164"}, {"id": "gnss", "type": "gnss", "name": "GNSS receiver (NMEA 0183)"}]
    signals = [b("battery_soc", "BMS_TPDO1.PackSoc", "%", 5), b("battery_voltage", "BMS_TPDO1.PackVoltage", "V", 5), b("battery_current", "BMS_TPDO1.PackCurrent", "A", 5), b("battery_temperature", "BMS_TPDO2.PackTemp", "C", 1),
               b("cell_temperature_max", "BMS_TPDO2.CellTempMax", "C", 1), b("cell_temperature_min", "BMS_TPDO2.CellTempMin", "C", 1), b("cell_voltage_min", "BMS_TPDO3.CellVoltMin", "V", 1),
               b("_charging_active", "BMS_TPDO3.Charging", "none", 5), b("motor_rpm", "INV_TPDO1.MotorRpm", "rpm", 20), b("motor_torque", "INV_TPDO1.MotorTorque", "Nm", 20), b("motor_temperature", "INV_TPDO1.MotorTemp", "C", 1),
               b("speed", "INV_TPDO1.VehSpeed", "km/h", 20), b("charging_power", "CHG_TPDO1.ChargePower", "kW", 1), b("ambient_temperature", "HVAC_TPDO1.OutsideTemp", "C", 1), b("cabin_temperature", "HVAC_TPDO1.CabinTemp", "C", 1),
               b("hvac_power", "HVAC_TPDO1.HvacPower", "W", 1), b("tyre_pressure_fl", "TPMS_TPDO1.TyreFL", "kPa", 1, valid=[0, 1000]), b("tyre_pressure_fr", "TPMS_TPDO1.TyreFR", "kPa", 1, valid=[0, 1000]), b("tyre_pressure_rl", "TPMS_TPDO1.TyreRL", "kPa", 1, valid=[0, 1000]),
               b("tyre_pressure_rr", "TPMS_TPDO1.TyreRR", "kPa", 1, valid=[0, 1000]), {"concept": "_ignition", "from": "can:co/INV_TPDO1.MotorRpm", "unit": "rpm", "rate_hz": 5, "when": ">-30000"}]
    spec = {"schema": "vehiclegpt.profile/1", "id": "volta_bus", "vehicle": {"make": "Volta", "model": "CityBus 12 m", "year": 2026, "powertrain": "bev", "specs": {
            "veh.model": "Volta CityBus 12 m 350 kWh", "veh.cap_kwh": 350.0, "veh.soh": 100.0, "veh.nominal_voltage": 650, "veh.tyre_spec_front": 800, "veh.tyre_spec_rear": 800, "veh.service_km": 20000, "veh.home_charger_kw": 150.0, "veh.sw": "1.8.0",
            "veh.wheelbase_m": 5.9, "veh.wheel_radius_m": 0.5, "veh.gear_ratio": 14.0, "veh.rated_max_speed": 80, "veh.seats": 38}}, "concepts": [], "sources": sources, "signals": signals,
            "events": [{"id": "dtc", "from": "uds:dtc"}, {"id": "overtemp", "from": "signal:battery_temperature", "when": ">55", "hold_s": 30, "type": "WARN_BATT_OVERTEMP", "severity": "critical", "detail": "Pack at {value:.1f} degC"}],
            "thresholds": {"overtemp_c": 105, "tyre_low_kpa": 700, "low_12v": 22.0}, "retention": dict(RETENTION)}
    open(os.path.join(DEMO, "volta_canopen.dbc"), "w", encoding="utf-8", newline="\n").write(canopen_dbc())
    meta = dict(name="Volta CityBus 12 m", year=2026, region="anywhere", powertrain="bev", **{"class": "Battery-electric city bus (simulated with the SUV drive profile) on a CANopen network"}, files=["volta_canopen.dbc"],
                interfaces=["CANopen 250 kbit/s (CiA 301): TPDOs of the battery, inverter, charger, HVAC and tyre-pressure nodes; heartbeat; emergency messages for faults; SDO reads of the odometer and 12 V supply", "UDS / OBD-II diagnostics", "syslog", "NMEA 0183 GNSS"],
                notes="A fictional maker. The object dictionary and PDO mapping are written for the test bench; the protocol is the real one (CiA 301).", source="written for the test bench (tools_make_protocol_cars.py)")
    return spec, os.path.join(PACKS, "aster_ev"), meta


BUILDERS = {"nova_ev": nova_ev, "prestige_gt": prestige_gt, "volta_bus": volta_bus}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); a = ap.parse_args(); bad = 0
    for pid, fn in BUILDERS.items():
        if a.only and pid not in a.only.split(","): continue
        try:
            spec, base, meta = fn(); p, n = write(spec, base, pid, meta); print(f"{pid:14s} {len(p.bindings):3d} bindings, {len(p.concept_names()):3d} quantities, round trip ok on {n}")
        except (ProfileError, KeyError, ValueError, OSError) as e:
            print(f"{pid:14s} FAILED: {e}"); bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

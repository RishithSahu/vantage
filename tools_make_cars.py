"""Builds the car models of the test bench: for each car a profile pack (profile.json + the CAN database(s) + dtc.json) that says which buses the car has and which quantity each signal carries.

    python tools_make_cars.py --dbc-dir <folder with the downloaded opendbc *.dbc> [--only id,id] [--describe file.dbc MSG.SIG ...]

The CAN databases of the real-vehicle models are the open-source opendbc files (MIT licence, https://github.com/commaai/opendbc; docs/THIRD_PARTY_NOTICES.md): the message identifiers, bit layouts, scales and
offsets are those of the real cars.  What a database does not publish (opendbc keeps what a driver-assistance add-on needs, so it often lacks the battery or the fuel level) is stated as a gap, and where the model
needs the quantity anyway it is carried by the standard the car class really has - OBD-II service 01 on the diagnostic port for combustion cars, a vendor UDS data identifier for a battery - or, last, by an
explicitly synthetic extension database (`<id>_ext.dbc`) so the test bench never pretends a real car publishes what it does not.

A model is data (tools_cars_data.py): `bind` lists (concept, "Message.Signal", options) rows that were read against the database.  Besides CAN models there are three standard-only kinds:
  kind="obd"     a car that only has the OBD-II port (a dongle on any car): no manufacturer database exists for it, Vantage learns it from the standard
  kind="j1939"   a heavy vehicle on SAE J1939 (the parameter groups of the standard, any source address)
  kind="vhal"    an Android Automotive car: the only data path an app has there is the Vehicle HAL, whose property ids are the AOSP standard
This tool checks every row against the database (the message, the signal, the unit conversion), then DRIVES the virtual car through the finished profile and requires every decoded quantity to come back
equal to what was sent (`roundtrip`), and refuses a model that does not pass.  The same files are the decoder file Vantage fingerprints against (vgpt/vehicle/packs/<id>, unless the model is meant to be
unknown to Vantage) and the car's own description (vgpt/car/models/<id>).
"""
import argparse
import json
import math
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from vgpt import signal_registry as R  # noqa: E402
from vgpt.ingest import catalog as K, standards as STD  # noqa: E402
from vgpt.protocols import dbc as DBC  # noqa: E402
from vgpt.vehicle.profile import Profile, ProfileError, norm_unit  # noqa: E402

PACKS = os.path.join(ROOT, "vgpt", "vehicle", "packs")
MODELS = os.path.join(ROOT, "vgpt", "car", "models")
SYNTH = os.path.join(ROOT, "vgpt", "vehicle", "dbc_synthetic")
RETENTION = {"fine_days": 14, "coarse_days": 1830, "log_days": 1825, "context_days": 1825, "archive_days": 1830, "records_days": 1825}      # 14 days at 1 Hz and the whole 5 years of minutes stay in the database (no archive step before a question: no added latency); expired seconds are archived losslessly
OBD_ICE = [("speed", "obd_speed", "km/h", 5), ("engine_rpm", "obd_rpm", "rpm", 5), ("motor_temperature", "obd_coolant_temp", "C", 1), ("engine_load", "obd_engine_load", "%", 2), ("intake_air_temperature", "obd_intake_air_temp", "C", 1),
           ("mass_air_flow", "obd_maf", "g/s", 2), ("throttle_position", "obd_throttle", "%", 2), ("fuel_level", "obd_fuel_level", "%", 1), ("ambient_temperature", "obd_ambient_temp", "C", 1),
           ("aux_battery_voltage", "obd_module_voltage", "V", 1), ("engine_oil_temperature", "obd_oil_temp", "C", 1), ("fuel_rate", "obd_fuel_rate", "L/h", 2), ("barometric_pressure", "obd_barometric", "kPa", 0.5),
           ("intake_manifold_pressure", "obd_intake_map", "kPa", 2), ("ignition_timing", "obd_timing_advance", "deg", 2), ("short_term_fuel_trim", "obd_fuel_trim_st1", "%", 1), ("long_term_fuel_trim", "obd_fuel_trim_lt1", "%", 0.5)]
OBD_EV = [("speed", "obd_speed", "km/h", 5), ("ambient_temperature", "obd_ambient_temp", "C", 1), ("aux_battery_voltage", "obd_module_voltage", "V", 1)]
PETROL_ONLY = {"intake_manifold_pressure", "ignition_timing", "short_term_fuel_trim", "long_term_fuel_trim"}
LOG_EVENTS = [{"id": "log_reset", "from": "log:any", "match": "watchdog reset", "max_severity": 3, "type": "ECU_RESET", "severity": "warn"},
              {"id": "log_brownout", "from": "log:any", "match": "brown-out", "max_severity": 3, "type": "VOLTAGE_DIP", "severity": "warn"}]
WHEELS = ("fl", "fr", "rl", "rr")
J1939_WANT = ["engine_rpm", "speed", "motor_temperature", "engine_load", "accelerator_position", "fuel_rate", "fuel_level", "boost_pressure", "intake_air_temperature", "exhaust_gas_temperature", "engine_oil_temperature",
              "barometric_pressure", "ambient_temperature", "aux_battery_voltage", "def_level", "dpf_soot_load", "fuel_rail_pressure", "odometer", "engine_torque_percent", "mass_air_flow"]

# What the synthetic extension database can carry: concept -> (bits, factor, offset, signed, low, high, unit, rate Hz).  Every row is a plain measurement; the extension exists only so that a model whose real
# database does not publish a quantity still has it on a bus, and it is labelled SYNTHETIC in the database, in the pack and in the car's description.
EXT = {"tyre_pressure_fl": (16, 0.1, 0, False, 0, 600, "kPa", 1), "tyre_pressure_fr": (16, 0.1, 0, False, 0, 600, "kPa", 1), "tyre_pressure_rl": (16, 0.1, 0, False, 0, 600, "kPa", 1),
       "tyre_pressure_rr": (16, 0.1, 0, False, 0, 600, "kPa", 1), "tyre_temperature": (12, 0.1, -40, False, -40, 120, "C", 1), "odometer": (24, 0.1, 0, False, 0, 1000000, "km", 1),
       "aux_battery_voltage": (12, 0.01, 0, False, 6, 16, "V", 1), "ambient_temperature": (11, 0.1, -50, False, -50, 60, "C", 1), "fuel_level": (10, 0.1, 0, False, 0, 100, "%", 1),
       "battery_soc": (12, 0.1, 0, False, 0, 100, "%", 1), "battery_temperature": (11, 0.1, -40, False, -40, 90, "C", 1), "battery_voltage": (14, 0.05, 0, False, 0, 800, "V", 5),
       "battery_current": (16, 0.1, 0, True, -900, 900, "A", 5), "motor_temperature": (11, 0.1, -40, False, -40, 200, "C", 1), "reported_range": (12, 1, 0, False, 0, 1500, "km", 0.2),
       "charging_power": (12, 0.1, 0, False, 0, 300, "kW", 1), "coolant_flow": (10, 0.1, 0, False, 0, 60, "L/min", 1), "motor_torque": (14, 0.1, 0, True, -500, 500, "Nm", 10),
       "engine_oil_temperature": (11, 0.1, -40, False, -40, 200, "C", 1), "barometric_pressure": (10, 0.1, 30, False, 30, 125, "kPa", 0.2), "rain": (2, 1, 0, False, 0, 3, "", 1),
       "light": (17, 1, 0, False, 0, 130000, "lux", 1), "cabin_temperature": (10, 0.1, -20, False, -20, 80, "C", 1), "hvac_power": (13, 1, 0, False, 0, 8000, "W", 1),
       "_ignition": (1, 1, 0, False, 0, 1, "", 5), "_charging_active": (1, 1, 0, False, 0, 1, "", 5)}
EXT_SPECIAL = {"tyre_pressure_fl": "EXT_TPMS", "tyre_pressure_fr": "EXT_TPMS", "tyre_pressure_rl": "EXT_TPMS", "tyre_pressure_rr": "EXT_TPMS", "_ignition": "EXT_KEYSTATE", "_charging_active": "EXT_KEYSTATE"}


def make_ext(car_id, concepts, taken_ids):
    """-> (dbc text, [binding dicts]) for a synthetic extension database carrying `concepts` (those the real database does not publish).  Identifiers are taken from the unused part of 0x6A0..0x6FF."""
    free = [i for i in range(0x6FF, 0x69F, -1) if i not in taken_ids]; groups = {}
    for c in concepts:
        rate = EXT[c][7]; g = EXT_SPECIAL.get(c) or "EXT_DATA_" + ("%g" % rate).replace(".", "_") + "HZ"; groups.setdefault(g, []).append(c)
    msgs = []
    for g, cs in groups.items():
        cur, bits, part = [], 0, 0
        for c in cs:
            n = EXT[c][0]
            if bits + n > 64: msgs.append((g if part == 0 else f"{g}_{part + 1}", cur)); cur, bits, part = [], 0, part + 1
            cur.append(c); bits += n
        if cur: msgs.append((g if part == 0 else f"{g}_{part + 1}", cur))
    lines = [f'VERSION "SYNTHETIC extension for {car_id}: quantities the open-source database of this vehicle does not publish. NOT the manufacturer\'s data."', "", "NS_ :", "", "BS_:", "", "BU_: EXT", ""]
    binds = []
    for k, (mname, cs) in enumerate(msgs):
        lines.append(f"BO_ {free[k]} {mname}: 8 EXT"); start = 0
        for c in cs:
            bits, fac, off, signed, lo, hi, unit, rate = EXT[c]; sig = "".join(w.capitalize() for w in c.strip("_").split("_"))
            lines.append(f' SG_ {sig} : {start}|{bits}@1{"-" if signed else "+"} ({fac:g},{off:g}) [{lo:g}|{hi:g}] "{unit}" Vector__XXX'); start += bits
            binds.append({"concept": c, "from": f"can:ext/{mname}.{sig}", "unit": unit or "none", "rate_hz": rate})
        lines.append("")
    return "\n".join(lines) + "\n", binds


def roundtrip(p, seconds=150, seed=5):
    """Drive the virtual car with this profile, put its state on the buses the profile describes, decode it again with the profile's own router, and compare.  Returns (checked, [mismatch lines])."""
    from vgpt.live.car import VirtualCar
    from vgpt.vehicle.router import ProfileRouter
    from vgpt.world.emit import Emitter
    car = VirtualCar(p, seed=seed, t0=1_790_000_000 + 8 * 3600); car.start_trip(); st = {}
    for _ in range(seconds): st = car.step()
    em = Emitter(p); r = ProfileRouter(p); got = {}
    for sec in range(int(car.t) - 3, int(car.t) + 1):
        for ts, bus, payload in em.tick(sec, st, full=True):
            for rec in r.decode(bus, payload, ts):
                if rec[0] == "sig": got[rec[1]] = rec[2]
    bad = []; checked = 0
    for b in p.bindings:
        if b.when or b.concept.startswith("_") or b.priority != 1: continue
        want = st.get(b.concept)
        if want is None: continue
        if b.concept not in got: bad.append(f"{b.concept}: sent {want:.3g} but nothing came back ({b.bus}:{b.key})"); continue
        checked += 1; have = got[b.concept]
        if b.map is not None or b.snap: continue
        tol = max(0.04 * abs(want), 0.8)
        if not math.isfinite(have) or abs(have - want) > tol: bad.append(f"{b.concept}: sent {want:.4g} came back {have:.4g} ({b.bus}:{b.key})")
    return checked, bad


def describe(dbc_path, names):
    """Print the definition of signals (factor, offset, range, unit, value table) so a binding is written against facts."""
    db = DBC.Database.load(dbc_path, lenient=True)
    for n in names:
        msg, _, sig = n.partition(".")
        m = db.by_name(msg)
        if m is None: print(f"{n}: no such message"); continue
        s = next((x for x in m.signals if x.name == sig), None)
        if s is None: print(f"{n}: no such signal in {msg} (has: {', '.join(x.name for x in m.signals[:14])})"); continue
        print(f"{n}: id 0x{m.frame_id:X}{'x' if m.extended else ''} dlc {m.dlc} start {s.start} len {s.length} {'LE' if s.little else 'BE'} {'signed' if s.signed else 'unsigned'} x{s.factor:g} {s.offset:+g} unit {s.unit!r} choices {dict(list(s.choices.items())[:10]) if s.choices else None}")


def _find(dbc_dir, fname):
    for d in (dbc_dir, SYNTH):
        p = os.path.join(d or "", fname)
        if os.path.isfile(p): return p
    raise FileNotFoundError(fname)


def build(car, dbc_dir, out_dirs):
    """-> (Profile, report lines).  Writes the pack to every directory in `out_dirs`."""
    pid = car["id"]; kind = car.get("kind", "can"); v = car.get("vehicle", {}); sources = []; signals = []; report = []; files = {}; dbs = {}
    for fname, sid, chan, fd, bitrate, label in car.get("can", ()):
        src = {"id": sid, "type": "can", "name": label, "dbc": fname, "channel": chan, "bitrate": bitrate}
        if fd: src.update(fd=True, data_bitrate=car.get("data_bitrate", 2000000))
        sources.append(src); path = _find(dbc_dir, fname); files[fname] = path; dbs[sid] = DBC.Database.load(path, lenient=True)
    for row in car.get("bind", ()):
        concept, ref = row[0], row[1]; opt = row[2] if len(row) > 2 else {}
        sid, key = ref.split(":", 1) if ":" in ref else (car["can"][0][1], ref)
        db = dbs[sid]; msg, _, sig = key.partition("."); m = db.by_name(msg)
        if m is None: raise ValueError(f"{pid}: message {msg!r} is not in the database of source {sid}")
        s = next((x for x in m.signals if x.name == sig), None)
        if s is None: raise ValueError(f"{pid}: signal {key!r} is not in the database of source {sid}")
        b = {"concept": concept, "from": f"can:{sid}/{key}", "unit": opt.get("unit") or norm_unit(s.unit) or "none", "rate_hz": opt.get("rate", 10)}
        for k in ("scale", "offset", "map", "invert", "when", "priority", "valid", "conflict_abs", "deadband", "ttl", "snap"):
            if k in opt: b[k] = opt[k]
        signals.append(b)
    if kind == "j1939":
        table = K.j1939_table(); pgns = {}; sup = set(J1939_WANT); done = {b["concept"] for b in signals}
        for pgn, rows in sorted(table.items()):
            for concept, spn, start, ln, scale, offset in rows:
                if concept in sup and concept in R.CONCEPTS and concept not in done: pgns.setdefault(pgn, []).append((concept, spn)); done.add(concept)           # a concept carried by two parameter groups is bound to the first
        text, names = STD.j1939_dbc(set(pgns))
        files["j1939.dbc"] = text; sources.append({"id": "j1939", "type": "can", "name": "SAE J1939 (250 kbit/s, 29-bit, any source address)", "dbc": "j1939.dbc", "channel": "can0", "bitrate": 250000, "j1939": True})
        for pgn, cs in sorted(pgns.items()):
            for concept, spn in cs: signals.append({"concept": concept, "from": f"can:j1939/J1939_{pgn}.SPN{spn}_{concept}", "unit": "none", "rate_hz": 5 if pgn in (61444, 65265) else 1})
        rpm = next((b for b in signals if b["concept"] == "engine_rpm"), None)
        if rpm: signals.append({"concept": "_ignition", "from": rpm["from"], "unit": "none", "rate_hz": 5, "when": ">300"})
    have = {b["concept"] for b in signals}
    if car.get("obd"):
        table = OBD_EV if car["powertrain"] == "bev" else [r for r in OBD_ICE if not (car["powertrain"] == "diesel" and r[0] in PETROL_ONLY)]
        for concept, key, unit, rate in table:
            if concept not in have: signals.append({"concept": concept, "from": f"uds:diag/{key}", "unit": unit, "rate_hz": rate}); have.add(concept)
    dids = {}
    for concept, did, fmt, scale, unit, rate in car.get("dids", ()):
        if concept in have: continue
        dids[did] = {"concept": concept, "fmt": fmt, "scale": scale, "unit": unit, "rate_hz": rate}; have.add(concept)
    for concept, key, opt in car.get("vhal", ()):
        b = {"concept": concept, "from": f"vhal:hal/{key}", "unit": opt.get("unit", "none"), "rate_hz": opt.get("rate", 1)}
        for k in ("scale", "offset", "map", "priority", "conflict_abs", "snap"):
            if k in opt: b[k] = opt[k]
        signals.append(b); have.add(concept)
    if car.get("vhal"): sources.append({"id": "hal", "type": "vhal", "name": "Android Vehicle HAL (AIDL, AOSP property ids)"})
    ext_concepts = [c for c in car.get("ext", ()) if c not in have]
    if ext_concepts:
        taken = set()
        for sid, db in dbs.items(): taken |= set(db.messages) | set(db.extended_messages)
        ext_text, ext_binds = make_ext(pid, ext_concepts, taken); signals += ext_binds; have |= set(ext_concepts); files[f"{pid}_ext.dbc"] = ext_text
        sources.append({"id": "ext", "type": "can", "name": f"{car['name']} extension CAN (SYNTHETIC: quantities the open database lacks)", "dbc": f"{pid}_ext.dbc", "channel": car.get("ext_channel", "can1"), "bitrate": 500000})
    if car.get("diag", True):
        sources.append({"id": "diag", "type": "uds", "name": car.get("diag_name", "UDS / OBD-II (ISO 15765-4)"), "response_ids": ["0x7E8..0x7EF"], "dtc": "dtc.json", **({"dids": dids} if dids else {})})
    sources.append({"id": car.get("log_id", "ivi"), "type": "log", "name": car["log_name"], "format": car["log_format"]})
    sources.append({"id": "gnss", "type": "gnss", "name": "GNSS receiver (NMEA 0183)"})
    if car.get("camera"): sources.append({"id": "cam", "type": "camera", "name": car["camera"], "cams": ["front"]})
    ev = []
    if "motor_temperature" in have and car["powertrain"] != "bev": ev.append({"id": "overtemp", "from": "signal:motor_temperature", "when": ">105", "hold_s": 10, "type": "WARN_OVERTEMP", "severity": "critical", "detail": "Coolant above 105 °C ({value:.1f} °C)"})
    if "aux_battery_voltage" in have: ev.append({"id": "low12v", "from": "signal:aux_battery_voltage", "when": "<11.6", "hold_s": 20, "type": "LOW_12V", "severity": "warn", "detail": "12 V battery at {value:.2f} V"})
    spec_kpa = v.get("specs", {}).get("veh.tyre_spec_front", 230)
    for w in WHEELS:
        if f"tyre_pressure_{w}" in have: ev.append({"id": f"tpms_{w}", "from": f"signal:tyre_pressure_{w}", "when": "<200", "hold_s": 60, "type": "TPMS_LOW", "severity": "warn", "detail": "tyre pressure low ({value:.0f} kPa, spec %d kPa)" % spec_kpa})
    ev += LOG_EVENTS + ([{"id": "dtc", "from": "uds:dtc"}] if car.get("diag", True) else [])
    vehicle = {"make": car["make"], "model": car["name"], "year": car["year"], "powertrain": car["powertrain"], "specs": dict(v.get("specs", {}), **{"veh.model": f"{car['make']} {car['name']}"})}
    spec = {"schema": "vehiclegpt.profile/1", "id": pid, "vehicle": vehicle, "concepts": list(car.get("concepts", [])), "sources": sources, "signals": signals, "events": ev,
            "thresholds": {"overtemp_c": 105, "tyre_low_kpa": 200, "low_12v": 11.6}, "retention": dict(RETENTION)}
    for out in out_dirs:
        os.makedirs(out, exist_ok=True)
        for fname, src in files.items():
            if os.path.isfile(src): shutil.copyfile(src, os.path.join(out, fname))
            else: open(os.path.join(out, fname), "w", encoding="utf-8", newline="\n").write(src)
        if car.get("diag", True): shutil.copyfile(os.path.join(PACKS, car.get("dtc_from", "kestrel_petrol" if car["powertrain"] != "bev" else "aster_ev"), "dtc.json"), os.path.join(out, "dtc.json"))
        with open(os.path.join(out, "profile.json"), "w", encoding="utf-8", newline="\n") as f: json.dump(spec, f, indent=1, ensure_ascii=False)
    p = Profile.load(out_dirs[0]); issues = p.validate()
    errors = [i for i in issues if i.level == "error"]
    if errors: raise ProfileError(f"{pid}: " + "; ".join(str(i) for i in errors))
    report += [str(i) for i in issues if i.level == "warning" and "retention" not in i.path and "DBC declares" not in i.message]
    checked, bad = roundtrip(p)
    if bad: raise ProfileError(f"{pid}: round trip failed:\n  " + "\n  ".join(bad))
    report.append(f"round trip ok on {checked} quantities")
    return p, report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dbc-dir"); ap.add_argument("--only", default=""); ap.add_argument("--describe", nargs="+")
    a = ap.parse_args()
    if a.describe: describe(a.describe[0], a.describe[1:]); return 0
    from tools_cars_data import CARS
    only = [x for x in a.only.split(",") if x]; os.makedirs(MODELS, exist_ok=True); failed = 0
    for car in CARS:
        if only and car["id"] not in only: continue
        car.setdefault("ext", [])
        outs = [os.path.join(PACKS, car["id"])] if car.get("decoder", True) else []
        model_dir = os.path.join(MODELS, car["id"])
        for d in outs + [model_dir]: shutil.rmtree(d, ignore_errors=True)
        try: p, rep = build(car, a.dbc_dir, outs + [model_dir])
        except (ProfileError, ValueError, FileNotFoundError, KeyError) as e:
            print(f"{car['id']:24s} FAILED: {e}"); failed += 1; continue
        meta = {"id": car["id"], "name": f"{car['make']} {car['name']}", "year": car["year"], "region": car["region"], "powertrain": car["powertrain"], "class": car["cls"], "decoder_installed": bool(car.get("decoder", True)),
                "interfaces": car["interfaces"], "notes": car.get("notes", ""), "source": car.get("source", ""), "bound": len(p.bindings), "concepts": len(p.concept_names()), "extension": sorted(car.get("ext", []))}
        with open(os.path.join(MODELS, car["id"] + ".json"), "w", encoding="utf-8", newline="\n") as f: json.dump(meta, f, indent=1, ensure_ascii=False)
        print(f"{car['id']:24s} {len(p.bindings):3d} bindings, {len(p.concept_names()):3d} quantities  {'; '.join(rep)[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

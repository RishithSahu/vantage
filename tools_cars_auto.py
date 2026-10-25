"""Add car models from open CAN databases without hand-written bindings: the signal mapper reads each database, a refuse-not-guess decision binds every signal it is sure of, and the rest of the car (OBD-II,
UDS, a labelled synthetic extension CAN for what the open database lacks, logs, GNSS, camera) is added exactly as for the hand-written models.  Writes tools_cars_data3.py, which tools_make_cars.py builds and
round-trips like the others.

    python tools_cars_auto.py [--dbc-dir tests/fixtures/opendbc]       then       python tools_make_cars.py --dbc-dir tests/fixtures/opendbc --only <ids>

What is real: the identifiers, bit layouts, scales and offsets (opendbc, MIT licence).  What is chosen by this tool: which signal is which quantity (the mapper; it leaves out what it cannot decide), the car's
class and the vehicle specifications (typical published values for the model, rounded), and which quantities ride the synthetic extension CAN (always labelled as such in the pack and the car's description)."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from vgpt.vehicle import onboard  # noqa: E402

LINUX = dict(log_format="rfc3164", log_name="Linux telematics / infotainment (syslog)")
ANDROID = dict(log_format="logcat", log_name="Android infotainment (logcat)")
QNX = dict(log_format="slog2info", log_name="QNX infotainment (slog2info)")
ICE_SPECS = {"veh.tyre_spec_front": 230, "veh.tyre_spec_rear": 230, "veh.service_km": 15000, "veh.wheelbase_m": 2.7, "veh.wheel_radius_m": 0.32, "veh.rated_max_speed": 190, "veh.seats": 5, "veh.tank_l": 55}
BEV_SPECS = {"veh.cap_kwh": 60, "veh.soh": 98, "veh.nominal_voltage": 360, "veh.tyre_spec_front": 250, "veh.tyre_spec_rear": 250, "veh.service_km": 20000, "veh.wheelbase_m": 2.7, "veh.wheel_radius_m": 0.33,
             "veh.gear_ratio": 9.0, "veh.rated_max_speed": 160, "veh.seats": 5, "veh.home_charger_kw": 7.4}

# id, make, model, year, region, powertrain, class, database, logging stack, camera, specifications that differ from the defaults
ROSTER = [
    ("honda_civic_2022", "Honda", "Civic EX (2022)", 2022, "USA / global", "petrol", "Compact saloon, petrol", "honda_civic_ex_2022_can_generated.dbc", LINUX, "Forward camera (Honda Sensing)", dict(**{"veh.tank_l": 47, "veh.wheelbase_m": 2.74, "veh.rated_max_speed": 200})),
    ("honda_clarity_phev", "Honda", "Clarity Plug-in Hybrid (2018)", 2018, "USA / Japan", "petrol", "Plug-in hybrid saloon (simulated with the petrol drive)", "honda_clarity_hybrid_2018_can_generated.dbc", ANDROID, "Forward camera (Honda Sensing)", {"veh.tank_l": 26, "veh.wheelbase_m": 2.75}),
    ("honda_odyssey_2018", "Honda", "Odyssey EX-L (2018)", 2018, "USA", "petrol", "Minivan, petrol V6", "honda_odyssey_exl_2018_generated.dbc", LINUX, "Forward camera (Honda Sensing)", {"veh.tank_l": 73, "veh.seats": 8, "veh.wheelbase_m": 3.0, "veh.rated_max_speed": 200}),
    ("acura_rdx_2018", "Acura", "RDX (2018)", 2018, "USA / Canada", "petrol", "Compact luxury SUV, petrol", "acura_rdx_2018_can_generated.dbc", LINUX, "Forward camera (AcuraWatch)", {"veh.tank_l": 62, "veh.wheelbase_m": 2.75}),
    ("ford_mondeo_2011", "Ford", "Mondeo 2.0 (CGEA 1.2, 2011)", 2011, "Europe", "petrol", "Large family car, petrol", "ford_cgea1_2_ptcan_2011.dbc", LINUX, None, {"veh.tank_l": 70, "veh.wheelbase_m": 2.85, "veh.rated_max_speed": 220}),
    ("chrysler_pacifica_phev", "Chrysler", "Pacifica Hybrid (2017)", 2017, "USA", "petrol", "Plug-in hybrid minivan (simulated with the petrol drive)", "chrysler_pacifica_2017_hybrid_generated.dbc", ANDROID, "Forward camera", {"veh.tank_l": 68, "veh.seats": 7, "veh.wheelbase_m": 3.08}),
    ("ram_1500_dt", "Ram", "1500 (DT, 2019)", 2019, "USA", "petrol", "Full-size pickup truck, petrol V8", "chrysler_ram_dt_generated.dbc", LINUX, "Forward camera", {"veh.tank_l": 98, "veh.wheelbase_m": 3.67, "veh.wheel_radius_m": 0.37, "veh.rated_max_speed": 180, "veh.tyre_spec_front": 260, "veh.tyre_spec_rear": 260}),
    ("subaru_outback_2015", "Subaru", "Outback (2015)", 2015, "USA / Japan / Australia", "petrol", "Estate SUV, petrol boxer", "subaru_outback_2015_generated.dbc", LINUX, "Forward camera (EyeSight)", {"veh.tank_l": 70, "veh.wheelbase_m": 2.74}),
    ("subaru_crosstrek_hybrid", "Subaru", "Crosstrek Hybrid (2020)", 2020, "USA", "petrol", "Plug-in hybrid crossover (simulated with the petrol drive)", "subaru_global_2020_hybrid_generated.dbc", LINUX, "Forward camera (EyeSight)", {"veh.tank_l": 55, "veh.wheelbase_m": 2.67}),
    ("mazda_cx5_2017", "Mazda", "CX-5 (2017)", 2017, "Japan / global", "petrol", "Compact SUV, petrol", "mazda_2017.dbc", LINUX, "Forward camera (i-Activsense)", {"veh.tank_l": 56, "veh.wheelbase_m": 2.7}),
    ("mazda_rx8", "Mazda", "RX-8 (rotary)", 2008, "Japan / global", "petrol", "Sports coupe, rotary petrol", "mazda_rx8.dbc", LINUX, None, {"veh.tank_l": 61, "veh.wheelbase_m": 2.7, "veh.seats": 4, "veh.rated_max_speed": 235}),
    ("vw_golf_mqbevo", "Volkswagen", "Golf (MQB evo)", 2020, "Europe / global", "petrol", "Compact hatchback, petrol", "vw_mqbevo.dbc", ANDROID, "Forward camera (Front Assist)", {"veh.tank_l": 50, "veh.wheelbase_m": 2.62}),
    ("nissan_leaf_2018", "Nissan", "Leaf (2018)", 2018, "Japan / global", "bev", "Battery-electric hatchback", "nissan_leaf_2018_generated.dbc", ANDROID, "Forward camera (ProPILOT)", dict(BEV_SPECS, **{"veh.cap_kwh": 40, "veh.nominal_voltage": 360, "veh.wheelbase_m": 2.7, "veh.rated_max_speed": 144})),
    ("nissan_xtrail_2017", "Nissan", "X-Trail (2017)", 2017, "Japan / global", "petrol", "Compact SUV, petrol", "nissan_x_trail_2017_generated.dbc", LINUX, "Forward camera (ProPILOT)", {"veh.tank_l": 60, "veh.wheelbase_m": 2.71}),
    ("kia_hyundai_generic", "Kia / Hyundai", "Generic Kia / Hyundai CAN (2019)", 2019, "Korea / global", "diesel", "Compact diesel SUV on the shared Kia / Hyundai CAN layout (AdBlue level included)", "hyundai_kia_generic.dbc", ANDROID, "Forward camera (SmartSense)", {"veh.tank_l": 54, "veh.wheelbase_m": 2.7}),
    ("hyundai_i30_2014", "Hyundai", "i30 (2014)", 2014, "Europe / Korea", "petrol", "Compact hatchback, petrol", "hyundai_i30_2014.dbc", ANDROID, None, {"veh.tank_l": 53, "veh.wheelbase_m": 2.65}),
    ("hyundai_santafe_2007", "Hyundai", "Santa Fe (2007)", 2007, "Korea / global", "petrol", "Mid-size SUV, petrol", "hyundai_santafe_2007.dbc", LINUX, None, {"veh.tank_l": 75, "veh.wheelbase_m": 2.7}),
    ("toyota_rav4_tnga", "Toyota", "RAV4 (TNGA, 2019)", 2019, "Japan / global", "petrol", "Compact SUV, petrol", "toyota_tnga_k_pt_generated.dbc", ANDROID, "Forward camera (Toyota Safety Sense)", {"veh.tank_l": 55, "veh.wheelbase_m": 2.69}),
    ("toyota_iq_2009", "Toyota", "iQ (2009)", 2009, "Japan / Europe", "petrol", "City car, petrol", "toyota_iQ_2009_can.dbc", LINUX, None, {"veh.tank_l": 32, "veh.wheelbase_m": 2.0, "veh.seats": 4, "veh.wheel_radius_m": 0.28, "veh.rated_max_speed": 170}),
    ("volvo_v40_2017", "Volvo", "V40 (2017)", 2017, "Sweden / global", "petrol", "Compact hatchback, petrol", "volvo_v40_2017_pt.dbc", QNX, "Forward camera (City Safety)", {"veh.tank_l": 52, "veh.wheelbase_m": 2.65}),
    ("alfa_romeo_giulia", "Alfa Romeo", "Giulia (Giorgio platform)", 2018, "Italy / global", "petrol", "Sports saloon, petrol", "fca_giorgio.dbc", LINUX, "Forward camera", {"veh.tank_l": 58, "veh.wheelbase_m": 2.82, "veh.rated_max_speed": 240}),
    ("luxgen_s5_2015", "Luxgen", "S5 (2015)", 2015, "Taiwan", "petrol", "Compact saloon, petrol", "luxgen_s5_2015.dbc", LINUX, None, {"veh.tank_l": 60, "veh.wheelbase_m": 2.7}),
    ("opel_omega_2001", "Opel", "Omega (2001)", 2001, "Germany / Europe", "petrol", "Executive saloon, petrol", "opel_omega_2001.dbc", LINUX, None, {"veh.tank_l": 70, "veh.wheelbase_m": 2.73, "veh.rated_max_speed": 230}),
    ("cadillac_ct6_phev", "Cadillac", "CT6 Plug-in (2017)", 2017, "USA / China", "petrol", "Plug-in hybrid luxury saloon (simulated with the petrol drive)", "cadillac_ct6_powertrain.dbc", ANDROID, "Forward camera", {"veh.tank_l": 56, "veh.wheelbase_m": 3.1, "veh.rated_max_speed": 210}),
    ("rivian_r1t", "Rivian", "R1T (primary actuator bus)", 2022, "USA", "bev", "Battery-electric pickup truck", "rivian_primary_actuator.dbc", ANDROID, "Surround cameras", dict(BEV_SPECS, **{"veh.cap_kwh": 135, "veh.nominal_voltage": 400, "veh.wheelbase_m": 3.45, "veh.wheel_radius_m": 0.4, "veh.rated_max_speed": 180, "veh.tyre_spec_front": 270, "veh.tyre_spec_rear": 270})),
]
TYRES = ["tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr", "tyre_temperature"]
ICE_EXT = TYRES + ["odometer", "aux_battery_voltage", "ambient_temperature", "fuel_level", "engine_oil_temperature", "barometric_pressure", "cabin_temperature"]
BEV_EXT = TYRES + ["odometer", "battery_soc", "battery_voltage", "battery_current", "battery_temperature", "motor_temperature", "reported_range", "charging_power", "coolant_flow", "motor_torque", "hvac_power",
                   "cabin_temperature", "ambient_temperature", "aux_battery_voltage", "_charging_active", "_ignition"]
KEEP = ("scale", "offset", "map", "invert", "when", "priority", "snap", "valid", "conflict_abs", "deadband", "ttl")


def entry(row, dbc_dir):
    pid, make, name, year, region, pt, cls, dbc, log, camera, specs = row
    spec, rep = onboard.draft(pid, [(os.path.join(dbc_dir, dbc), "can0", False)])
    sid = next(s["id"] for s in spec["sources"] if s["type"] == "can")
    binds = []
    for b in spec["signals"]:
        if not b["from"].startswith("can:"): continue
        ref = b["from"].split("/", 1)[1]; opt = {k: b[k] for k in KEEP if k in b}
        if b.get("unit") and b["unit"] != "none": opt["unit"] = b["unit"]
        opt["unit"] = opt.get("unit", "none"); opt["rate"] = 20 if b["concept"] in ("speed", "engine_rpm", "steering_angle", "yaw_rate", "lateral_acceleration", "longitudinal_acceleration") or b["concept"].startswith("wheel_speed") else b.get("rate_hz", 10)
        if b["concept"] == "gear" and opt.get("map"): opt["snap"] = True                                  # a gear NUMBER is shown as the nearest lever state (P / R / N / D)
        binds.append((b["concept"], ref, opt))
    have = {b[0] for b in binds}
    base = BEV_SPECS if pt == "bev" else ICE_SPECS
    sp = dict(base, **specs); sp.setdefault("veh.sw", "ecu 1.0")
    ext = [c for c in (BEV_EXT if pt == "bev" else ICE_EXT) if c not in have]
    bound = ", ".join(sorted(c for c in have if not c.startswith("_"))[:12])
    return dict(id=pid, make=make, name=name, year=year, region=region, powertrain=pt, obd=(pt != "bev"), cls=cls, **log, camera=camera, can=[(dbc, sid, "can0", False, 500000, f"CAN 500 kbit/s (OEM database {dbc[:-4]})")],
                bind=binds, ext=ext, vehicle=dict(specs=sp),
                notes=f"Bindings were drafted by the signal mapper (refuse-not-guess) from the open database and each one verified by driving the car through the pack: {len(binds)} drafted, {len(rep['unmapped'])} left unmapped. Quantities the database does not publish ride a labelled synthetic extension CAN.",
                interfaces=[f"CAN 500 kbit/s: the real {make} {name} database ({bound} ...)", "OBD-II service 01 over ISO 15765-4 CAN" if pt != "bev" else "OBD-II speed, ambient temperature and module voltage", "UDS diagnostic trouble codes", log["log_name"], "NMEA 0183 GNSS"]
                + (["forward camera pictures (JPEG)"] if camera else []) + ["tyre pressure, odometer and the quantities the open database lacks on a synthetic extension CAN"], source=f"opendbc (https://github.com/commaai/opendbc, MIT licence): {dbc}")


def verified(e, dbc_dir):
    """Build the model into a scratch folder and round-trip it; a binding whose decoded value does not come back equal to what was sent is dropped (the quantity moves to the extension CAN) and the model is built again."""
    import shutil, tempfile
    from tools_make_cars import build
    from vgpt.vehicle.profile import ProfileError
    dropped = []
    for _ in range(4):
        d = tempfile.mkdtemp(prefix="auto-")
        try:
            build(e, dbc_dir, [d]); return e, dropped
        except ProfileError as err:
            bad = [ln.strip().split(":", 1)[0] for ln in str(err).splitlines()[1:] if ln.startswith("  ") and ": sent " in ln]
            bad = [c for c in bad if any(b[0] == c for b in e["bind"])]
            if not bad: raise
            e["bind"] = [b for b in e["bind"] if b[0] not in bad]; e["ext"] = sorted(set(e["ext"]) | {c for c in bad if not c.startswith("_")}); dropped += bad
        finally: shutil.rmtree(d, ignore_errors=True)
    raise RuntimeError(f"{e['id']}: still failing after dropping {dropped}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dbc-dir", default=os.path.join(ROOT, "tests", "fixtures", "opendbc")); ap.add_argument("--out", default=os.path.join(ROOT, "tools_cars_data3.py")); a = ap.parse_args()
    cars = []
    for row in ROSTER:
        e = entry(row, a.dbc_dir); e, dropped = verified(e, a.dbc_dir); cars.append(e)
        print(f"{e['id']:26s} {len(e['bind']):3d} bound from the database, {len(e['ext']):2d} on the extension CAN" + (f"; dropped after the round trip: {dropped}" if dropped else ""))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write('"""Generated by tools_cars_auto.py (do not edit by hand): car models whose bindings the signal mapper drafted from open CAN databases."""\n\nMORE3 = ')
        import pprint
        f.write(pprint.pformat(cars, width=190, sort_dicts=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

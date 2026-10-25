"""The universal ingest: any capture, any naming, any unit -> the right quantity, in the registry's unit, stored by the same runtime that stores a vehicle's bus traffic.

Layers tested: units; the name matcher (a list of real-world signal names); the standards (J1939, OBD-II); the readers against files written by an independent library (python-can) and by tests/capture_factory.py;
and the whole path (autoprofile -> ingest runtime -> database) against a drive whose true values are known."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))

import capture_factory as F  # noqa: E402
from vgpt.ingest import autoprofile as AP, catalog as K, ingest as IG, readers as RD, standards as ST, units as U  # noqa: E402
from vgpt.protocols import obd  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures", "capture")


class Units(unittest.TestCase):
    def test_conversions_to_the_registry_unit(self):
        cases = [(100, "mph", "km/h", 160.9344), (10, "m/s", "km/h", 36.0), (54, "knots", "km/h", 100.008), (212, "degF", "C", 100.0), (300, "K", "C", 26.85), (32, "psi", "kPa", 220.632), (2.2, "bar", "kPa", 220.0), (1013, "hPa", "kPa", 101.3),
                 (5, "gal/h", "L/h", 18.927), (2, "kW", "W", 2000.0), (150, "hp", "kW", 111.855), (3600, "J", "kWh", 0.001), (500, "mV", "V", 0.5), (90, "mi", "km", 144.84), (1, "rad/s", "deg/s", 57.29578), (1.5, "g", "m/s2", 14.7), (60, "rps", "rpm", 3600.0)]
        for v, frm, to, want in cases:
            f = U.factor_between(frm, to); self.assertIsNotNone(f, (frm, to)); self.assertAlmostEqual(v * f[0] + f[1], want, delta=max(1e-3 * abs(want), 1e-4), msg=f"{v} {frm} -> {to}")

    def test_headers_carry_units(self):
        for header, base, unit in (("Speed (OBD)(km/h)", "Speed (OBD)", "km/h"), ("Engine Coolant Temperature(°C)", "Engine Coolant Temperature", "°C"), ("coolant [degF]", "coolant", "degF"), ("vehSpd_kph", "vehSpd", "kph"), ("Baro_kPa", "Baro", "kPa"), ("Odo_km", "Odo", "km"), ("EngSpd", "EngSpd", None)):
            self.assertEqual(U.split_header(header), (base, unit), header)

    def test_different_classes_do_not_convert(self):
        self.assertIsNone(U.factor_between("kPa", "C")); self.assertIsNone(U.factor_between("km/h", "kW")); self.assertIsNone(U.factor_between("widgets", "km/h"))


# (signal name as a vehicle, DBC, CSV or JSON writes it, unit if it states one, the quantity it is)
NAMES = [
    ("VehSpeed", "km/h", "speed"), ("vehicle_speed", None, "speed"), ("Speed (OBD)(km/h)", None, "speed"), ("v_ego", "m/s", "speed"), ("WheelBasedVehicleSpeed", None, "speed"), ("Vehicle.Speed", None, "speed"), ("DI_vehicleSpeed", "mph", "speed"),
    ("ESP_VehSpd", "kph", "speed"), ("SPD_KMH", None, "speed"), ("Fahrzeuggeschwindigkeit", None, None), ("EngSpeed", "rpm", "engine_rpm"), ("Engine RPM(rpm)", None, "engine_rpm"), ("engine_speed", None, "engine_rpm"), ("RPM", None, "engine_rpm"),
    ("Eng_n_actual", "rpm", "engine_rpm"), ("EngineSpeed", None, "engine_rpm"), ("Vehicle.Powertrain.CombustionEngine.Speed", None, "engine_rpm"), ("EngCoolantTemp", "degC", "motor_temperature"), ("Engine Coolant Temperature(°C)", None, "motor_temperature"),
    ("ECT", None, "motor_temperature"), ("Coolant_Temp", "C", "motor_temperature"), ("CltTemp_C", None, "motor_temperature"), ("Vehicle.Powertrain.CombustionEngine.ECT", None, "motor_temperature"), ("EngOilTemp", "C", "engine_oil_temperature"),
    ("Engine Oil Temperature(°C)", None, "engine_oil_temperature"), ("EOT", None, "engine_oil_temperature"), ("oil_temp", None, "engine_oil_temperature"), ("EngOilPress", "kPa", "engine_oil_pressure"), ("OilPressure_kPa", None, "engine_oil_pressure"),
    ("FuelLevel", "%", "fuel_level"), ("Fuel Level (From Engine ECU)(%)", None, "fuel_level"), ("fuel_level", None, "fuel_level"), ("TankLevel_pct", None, "fuel_level"), ("FuelRate", "L/h", "fuel_rate"), ("Fuel flow rate/hour(l/hr)", None, "fuel_rate"),
    ("EngFuelRate", None, "fuel_rate"), ("fuel_consumption_rate", "l/h", "fuel_rate"), ("AmbientAirTemp", "C", "ambient_temperature"), ("Ambient air temp(°C)", None, "ambient_temperature"), ("OutsideTemp", None, "ambient_temperature"),
    ("Vehicle.Exterior.AirTemperature", None, "ambient_temperature"), ("OAT", "degC", "ambient_temperature"), ("AmbTemp", None, "ambient_temperature"), ("ThrottlePos", "%", "throttle_position"), ("Throttle Position(Manifold)(%)", None, "throttle_position"),
    ("AccelPedalPos", "%", "accelerator_position"), ("accelerator_pedal_position", None, "accelerator_position"), ("APP", "%", "accelerator_position"), ("EngLoad", "%", "engine_load"), ("Engine Load(%)", None, "engine_load"), ("CalcLoad", None, "engine_load"),
    ("MAP_kPa", None, "intake_manifold_pressure"), ("IntakeManifoldPressure", None, "intake_manifold_pressure"), ("Manifold Pressure", "kPa", "intake_manifold_pressure"), ("IAT", "degC", "intake_air_temperature"), ("Intake Air Temperature(°C)", None, "intake_air_temperature"),
    ("MAF", "g/s", "mass_air_flow"), ("Mass Air Flow Rate(g/s)", None, "mass_air_flow"), ("BoostPressure", "kPa", "boost_pressure"), ("TurboBoost_kPa", None, "boost_pressure"), ("Baro_kPa", None, "barometric_pressure"), ("BarometricPressure", None, "barometric_pressure"),
    ("Lambda", None, "lambda"), ("AFR_equiv", None, "lambda"), ("STFT_B1", "%", "short_term_fuel_trim"), ("ShortTermFuelTrim", "%", "short_term_fuel_trim"), ("LTFT", "%", "long_term_fuel_trim"), ("LongTermFuelTrim_B1", None, "long_term_fuel_trim"),
    ("CatalystTemp", "C", "catalyst_temperature"), ("EGT", "degC", "exhaust_gas_temperature"), ("ExhaustGasTemp", None, "exhaust_gas_temperature"), ("TimingAdvance", "deg", "ignition_timing"), ("DPF_SootLoad", "%", "dpf_soot_load"), ("AdBlueLevel", "%", "def_level"),
    ("DEF_TankLevel", None, "def_level"), ("FuelRailPressure", "kPa", "fuel_rail_pressure"), ("RailPress", None, "fuel_rail_pressure"), ("EngineRunTime", "s", "engine_run_time"), ("TotalEngineHours", "h", "engine_hours"),
    ("Voltage (Control Module)(V)", None, "aux_battery_voltage"), ("BatteryPotential", "V", "aux_battery_voltage"), ("Vbat_12V", None, "aux_battery_voltage"), ("KL30_Voltage", None, "aux_battery_voltage"), ("LV_Battery_Voltage", None, "aux_battery_voltage"),
    ("Odometer", "km", "odometer"), ("TotalVehicleDistance", None, "odometer"), ("odo_km", None, "odometer"), ("Mileage", None, "odometer"), ("Vehicle.TraveledDistance", None, "odometer"), ("SOC", "%", "battery_soc"), ("BMS_SOC", None, "battery_soc"),
    ("State of Charge", "%", "battery_soc"), ("HV_Battery_Level", None, "battery_soc"), ("Vehicle.Powertrain.TractionBattery.StateOfCharge.Current", None, "battery_soc"), ("BMS_PackVoltage", "V", "battery_voltage"), ("HV_Voltage", None, "battery_voltage"),
    ("PackCurrent", "A", "battery_current"), ("BMS_PackCurrent", None, "battery_current"), ("HV_Current_A", None, "battery_current"), ("PackTemp", "C", "battery_temperature"), ("BMS_AvgCellTemp", None, "battery_temperature"), ("HV_Battery_Temperature", None, "battery_temperature"),
    ("BMS_MaxCellTemp", "C", "cell_temperature_max"), ("CellVoltMin", "V", "cell_voltage_min"), ("ChargePower", "kW", "charging_power"), ("ChargerPower_kW", None, "charging_power"), ("ChargeCurrent_A", None, "charging_current"), ("SOH", "%", "battery_soh"),
    ("MotorTorque", "Nm", "motor_torque"), ("EMotor_Torque", None, "motor_torque"), ("MotorSpeed", "rpm", "motor_rpm"), ("InverterTemp", "C", "inverter_temperature"), ("RegenPower", "kW", "regen_power"), ("Recuperation_kW", None, "regen_power"),
    ("CoolantFlow_lpm", None, "coolant_flow"), ("HVAC_Power", "W", "hvac_power"), ("CabinTemp", "C", "cabin_temperature"), ("SteeringAngle", "deg", "steering_angle"), ("SWA_deg", None, "steering_angle"), ("YawRate", "deg/s", "yaw_rate"),
    ("LongAccel", "m/s2", "longitudinal_acceleration"), ("LatAccel_g", None, "lateral_acceleration"), ("BrakePedalPos", "%", "brake_pedal"), ("BrakePressure", "bar", "brake_pressure"), ("WheelSpeed_FL", "km/h", "wheel_speed_fl"), ("WSS_RR", None, "wheel_speed_rr"),
    ("TPMS_FL_Press", "kPa", "tyre_pressure_fl"), ("TirePressureRearRight", None, "tyre_pressure_rr"), ("tyre_pressure_front_right", None, "tyre_pressure_fr"), ("TyreTemp_F", None, "tyre_temperature"), ("TransmissionOilTemp", "C", "transmission_temperature"),
    ("GearPosition", None, "gear"), ("CurrentGear", None, "gear"), ("GPS_Speed", "km/h", "gps_speed"), ("SpeedOverGround", "knots", "gps_speed"), ("Heading", "deg", "heading"), ("Altitude", "m", "altitude"), ("LeadDistance", "m", "lead_vehicle_distance"),
    ("LaneConfidence", None, "lane_confidence"), ("ParkingBrake", None, "parking_brake"),
]


class Names(unittest.TestCase):
    def test_signal_names_are_recognised(self):
        wrong = []; right = 0; refused = 0; scored = 0
        for name, unit, want in NAMES:
            if want is None: continue
            scored += 1; best, cands = AP._decide(name, unit, None, set())
            if best is None: refused += 1; wrong.append((name, unit, want, "REFUSED", [c[0] for c in cands[:2]]))
            elif best[0] == want: right += 1
            else: wrong.append((name, unit, want, best[0]))
        self.assertGreaterEqual(right / scored, 0.94, f"{right}/{scored} right; {refused} refused rather than guessed; misses: {wrong}")
        self.assertLessEqual(sum(1 for w in wrong if w[3] != "REFUSED") / scored, 0.03, f"too many WRONG answers (a refusal is better than a wrong quantity): {[w for w in wrong if w[3] != 'REFUSED']}")

    def test_a_quantity_is_never_read_in_the_wrong_class(self):
        for name, unit, bad in (("EngOilPress", "kPa", "engine_oil_temperature"), ("CoolantTemp", "kPa", "motor_temperature"), ("Speed", "degC", "speed")):
            best, _ = AP._decide(name, unit, None, set()); self.assertTrue(best is None or best[0] != bad, (name, unit, best))

    def test_the_values_decide_between_twelve_volts_and_the_traction_pack(self):
        hv = K.match("BatteryVoltage", "V", {"min": 340, "max": 400, "mean": 372}, floor=0.4)[0][0]; lv = K.match("BatteryVoltage", "V", {"min": 11.8, "max": 14.4, "mean": 13.6}, floor=0.4)[0][0]
        self.assertEqual((hv, lv), ("battery_voltage", "aux_battery_voltage"))

    def test_the_catalogue_names_are_all_known_concepts(self):
        from vgpt import signal_registry as R
        for cid in K.load()["map"]: self.assertIn(cid, R.CONCEPTS, cid)


class Packs(unittest.TestCase):
    def test_the_four_packs_dbc_signals_map_by_name_alone(self):
        import glob
        from vgpt.vehicle.profile import Profile
        right = wrong = refused = total = 0; bad = []
        for pack in sorted(os.path.join(ROOT, "vgpt", "vehicle", "packs", n) for n in ("aster_ev", "kestrel_petrol", "meridian_ev", "rhino_diesel")):         # the four packs written with the registry's own vocabulary; the real-vehicle databases are a harder benchmark (docs/CAR_MODELS.md)
            p = Profile.load(pack)
            for b in p.bindings:
                if b.bus != "can" or b.concept.startswith("_"): continue
                db = p.dbc(b.source); msg, _, sig = b.key.partition("."); m = next(m for d in (db.messages, db.extended_messages) for m in d.values() if m.name == msg); sg = next(x for x in m.signals if x.name == sig)
                best, _ = AP._decide(sig, sg.unit or None, None, set()); total += 1
                if best is None: refused += 1
                elif best[0] == b.concept or (b.concept, best[0]) in (("accelerator_position", "throttle_position"), ("engine_torque", "engine_torque_percent")): right += 1       # the pack's own choice is a coarser one in these two cases
                else: wrong += 1; bad.append((sig, b.concept, best[0]))
        self.assertGreaterEqual(right / total, 0.97, f"{right}/{total}; wrong {bad}"); self.assertEqual(wrong, 0, bad)


class Standards(unittest.TestCase):
    def test_obd_formulas_match_the_standard(self):
        for pid, raw, want in ((0x0D, bytes([60]), 60.0), (0x0C, bytes([0x1A, 0xF8]), 1726.0), (0x05, bytes([0x7B]), 83.0), (0x2F, bytes([128]), 50.196), (0x5E, bytes([0x03, 0xE8]), 50.0), (0x42, bytes([0x37, 0xDC]), 14.300), (0x06, bytes([160]), 25.0), (0x0F, bytes([65]), 25.0),
                               (0x3C, bytes([0x0C, 0xE8]), 290.4), (0xA6, bytes([0, 1, 0x86, 0xA0]), 10000.0)):
            ST.install_obd(); o = obd.decode(bytes([0x41, pid]) + raw); self.assertIsNotNone(o, hex(pid)); self.assertAlmostEqual(o["value"], want, delta=0.01 * max(1, abs(want)) + 0.01, msg=hex(pid))

    def test_j1939_identifiers(self):
        self.assertEqual(ST.j1939_fields(0x18FEF100)[1:], (0xFEF1, 0xF1, 0x00)); self.assertEqual(ST.j1939_fields(0x0CF00400)[:2], (3, 0xF004)); self.assertEqual(ST.j1939_fields(0x18EA0017)[1], 0xEA00)     # a request (PDU1): the destination is not part of the PGN
        self.assertEqual(ST.canonical_id(0x0CF00431), ST.canonical_id(0x18F00400))

    def test_not_available_is_never_a_reading(self):
        self.assertTrue(ST.is_na(0xFF, 8)); self.assertTrue(ST.is_na(0xFE, 8)); self.assertTrue(ST.is_na(0xFB, 8)); self.assertFalse(ST.is_na(0xFA, 8)); self.assertTrue(ST.is_na(0xFFFF, 16)); self.assertTrue(ST.is_na(0xFE12, 16)); self.assertFalse(ST.is_na(0xFAFF, 16)); self.assertTrue(ST.is_na(0xFB000000, 32)); self.assertFalse(ST.is_na(0xFAFFFFFF, 32)); self.assertFalse(ST.is_na(3, 2))

    def test_the_synthesised_dbc_decodes_a_group(self):
        from vgpt.protocols import dbc
        text, names = ST.j1939_dbc({65265}); db = dbc.Database.load(text); m = db.extended_messages[ST.canonical_id(0x18FEF100)]; data = bytearray(8); m.signals[0].encode(data, 72.5)
        self.assertAlmostEqual(m.decode(bytes(data))["SPN84_speed"], 72.5, delta=0.01)


def expected_from(cols):
    return {c: (c, 0.0) for _, c, _, _ in cols}


TOL = {"speed": 4.0, "engine_rpm": 90.0, "motor_temperature": 1.5, "engine_oil_temperature": 1.5, "fuel_level": 0.8, "fuel_rate": 0.7, "ambient_temperature": 0.4, "accelerator_position": 8.0, "throttle_position": 8.0, "engine_load": 2.5,
       "aux_battery_voltage": 0.15, "barometric_pressure": 0.7, "boost_pressure": 3.0, "intake_air_temperature": 1.5, "odometer": 0.3}


class WholePath(unittest.TestCase):
    """autoprofile -> ingest runtime -> database, for one drive written in many formats; values compared with the truth."""
    @classmethod
    def setUpClass(cls): cls.dir = tempfile.mkdtemp(prefix="ingest_test_")

    def run_case(self, name, writer, want, scale=1.0, duration=45, **kw):
        path = os.path.join(self.dir, name); writer(path); db = os.path.join(self.dir, name + ".db"); r = IG.ingest(path, db, anchor=F.T0, **kw); a = r["analysis"]
        for concept, (truth_key, extra) in want.items():
            rows = IG.read_back(db, concept); self.assertGreaterEqual(len(rows), 10, f"{name}: {concept} was not stored (mapped: {a.mapped()}; unmapped: {a.unmapped})")
            bad = []
            for ts, v in rows[8:-3:5]:
                tv = F.truth(ts - F.T0)[truth_key]; tol = TOL.get(concept, 1.5) * scale + extra
                if abs(v - tv) > tol: bad.append((round(ts - F.T0, 1), round(v, 3), round(tv, 3)))
            self.assertFalse(bad, f"{name}: {concept} differs from the truth: {bad[:4]}")
        return r

    J = {c: (c, 0.0) for c in ("speed", "engine_rpm", "motor_temperature", "engine_oil_temperature", "fuel_level", "fuel_rate", "ambient_temperature", "aux_battery_voltage", "engine_load", "boost_pressure", "odometer", "intake_air_temperature", "barometric_pressure")}
    J["accelerator_position"] = ("accelerator_position", 0.0)

    def test_j1939_candump(self): self.run_case("j1939.log", lambda p: F.write_candump_l(p, F.can_all(45)), self.J)

    def test_j1939_vector_asc(self): self.run_case("j1939.asc", lambda p: F.write_asc(p, F.can_all(45)), self.J)

    def test_j1939_peak_trc(self): self.run_case("j1939.trc", lambda p: F.write_trc(p, F.can_all(45)), self.J)

    def test_j1939_candump_default_view_is_read_and_mapped_without_times(self):
        path = os.path.join(self.dir, "nots.log"); F.write_candump_default(path, F.can_all(10)); a = AP.analyse(path, anchor=F.T0)
        self.assertEqual(a.format, "candump"); self.assertIn("speed", a.mapped()); self.assertIn("engine_rpm", a.mapped())

    def test_obd_ii_over_can(self):
        want = {c: (c, 0.0) for c in ("speed", "engine_rpm", "motor_temperature", "engine_oil_temperature", "fuel_level", "fuel_rate", "ambient_temperature", "engine_load", "aux_battery_voltage", "barometric_pressure", "intake_air_temperature")}
        self.run_case("obd.log", lambda p: F.write_candump_l(p, F.obd_all(60)), want, scale=6.0, duration=60)

    def test_obd_ii_elm327_trace_without_times(self):
        """An adapter trace has no clock: the quantities are recognised and every stored value lies in the range the drive really had."""
        path = os.path.join(self.dir, "elm.txt"); F.write_elm(path, F.obd_all(60)); db = os.path.join(self.dir, "elm.db"); r = IG.ingest(path, db, anchor=F.T0); self.assertEqual(r["analysis"].format, "elm")
        for concept, key in (("speed", "speed"), ("engine_rpm", "engine_rpm"), ("motor_temperature", "motor_temperature"), ("ambient_temperature", "ambient_temperature")):
            vals = [v for _, v in IG.read_back(db, concept)]; truth = [F.truth(t)[key] for t in range(0, 61)]; self.assertTrue(vals, concept)
            self.assertTrue(min(truth) - 3 <= min(vals) and max(vals) <= max(truth) + 3, (concept, min(vals), max(vals), min(truth), max(truth)))

    def test_torque_style_csv_metric(self):
        want = {c: (c, 0.0) for _, c, _, _ in F.OBD_APP_COLS if c != "accelerator_position"}; want["throttle_position"] = ("accelerator_position", 0.0)
        self.run_case("torque.csv", lambda p: F.write_csv(p, F.OBD_APP_COLS, 45), want)

    def test_torque_style_csv_imperial_units_are_converted_exactly(self):
        want = {c: (c, 0.0) for _, c, _, _ in F.OBD_APP_IMPERIAL}
        self.run_case("torque_us.csv", lambda p: F.write_csv(p, F.OBD_APP_IMPERIAL, 45), want, scale=1.0)

    def test_a_messy_logger_csv_with_semicolons_decimal_commas_and_iso_times(self):
        want = {"speed": ("speed", 0.0), "engine_rpm": ("engine_rpm", 0.0), "motor_temperature": ("motor_temperature", 0.0), "engine_oil_temperature": ("engine_oil_temperature", 0.0), "fuel_level": ("fuel_level", 0.0), "throttle_position": ("accelerator_position", 0.0),
                "barometric_pressure": ("barometric_pressure", 0.0), "aux_battery_voltage": ("aux_battery_voltage", 0.0), "intake_manifold_pressure": ("boost_pressure", 0.0), "odometer": ("odometer", 0.0)}
        self.run_case("logger.csv", lambda p: F.write_csv(p, F.LOGGER_COLS, 45, delimiter=";", decimal=",", iso=True, time_name="timestamp"), want)

    def test_openxc_json(self):
        self.run_case("openxc.json", lambda p: F.write_openxc(p, 45), {"speed": ("speed", 0.0), "engine_rpm": ("engine_rpm", 0.0), "fuel_level": ("fuel_level", 0.0), "odometer": ("odometer", 0.0), "accelerator_position": ("accelerator_position", 0.0)}, scale=2.0)

    def test_covesa_vss_json(self):
        want = {c: (c, 0.0) for c, _ in F.VSS.values()}
        self.run_case("vss.json", lambda p: F.write_vss(p, 45), want); self.run_case("vss2.json", lambda p: F.write_vss(p, 45, "updates"), want)

    def test_nested_json_with_unit_suffixed_names(self):
        self.run_case("flat.json", lambda p: F.write_flat_json(p, 45), {"speed": ("speed", 0.0), "engine_rpm": ("engine_rpm", 0.0), "motor_temperature": ("motor_temperature", 0.0), "engine_oil_temperature": ("engine_oil_temperature", 0.0), "fuel_level": ("fuel_level", 0.0)})

    def test_nmea_gnss(self):
        path = os.path.join(self.dir, "gps.nmea"); F.write_nmea(path, 40); a = AP.analyse(path)
        self.assertEqual(a.format, "nmea"); self.assertTrue(any(s["type"] == "gnss" for s in a.sources))

    def test_a_signal_the_catalogue_cannot_name_is_reported_not_guessed(self):
        path = os.path.join(self.dir, "odd.csv")
        with open(path, "w") as f:
            f.write("time,vehSpd_kph,zzqx_1,widget_flux\n"); [f.write(f"{F.T0 + i},{50 + i},{i},{i * 2}\n") for i in range(30)]
        a = AP.analyse(path, anchor=F.T0); self.assertEqual(a.mapped(), ["speed"]); self.assertTrue(any("zzqx" in u for u in a.unmapped))


class Reference(unittest.TestCase):
    """Files written by python-can (an independent implementation of the formats) read the same as the text files."""
    def test_every_reference_format_yields_the_same_frames(self):
        base = None
        for name in ("ref_j1939.blf", "ref_j1939.asc", "ref_j1939.trc", "ref_j1939_candump.log"):
            path = os.path.join(FIX, name)
            if not os.path.exists(path): self.skipTest("reference captures not generated (tests/fixtures/make_reference_captures.py)")
            got = sorted((r.payload.can_id, bytes(r.payload.data)) for r in RD.read(path, anchor=F.T0)); self.assertEqual(len(got), 920, name)
            if base is None: base = got
            self.assertEqual(got, base, name)

    def test_blf_timestamps_and_can_fd(self):
        p = os.path.join(FIX, "ref_fd.blf")
        if not os.path.exists(p): self.skipTest("reference captures not generated")
        recs = list(RD.read(p)); self.assertEqual(len(recs), 2); fd = recs[0].payload
        self.assertTrue(fd.fd and fd.brs and fd.extended and fd.can_id == 0x18FEF100 and fd.data == bytes(range(12))); self.assertAlmostEqual(recs[0].ts, F.T0 + 21.0, places=3); self.assertFalse(recs[1].payload.fd)

    def test_blf_ingests_like_the_text_log(self):
        p = os.path.join(FIX, "ref_j1939.blf")
        if not os.path.exists(p): self.skipTest("reference captures not generated")
        db = os.path.join(tempfile.mkdtemp(), "blf.db"); r = IG.ingest(p, db); rows = IG.read_back(db, "speed"); self.assertGreaterEqual(len(rows), 15)
        for ts, v in rows[3:-2:4]: self.assertAlmostEqual(v, F.truth(ts - F.T0)["speed"], delta=4.0)


if __name__ == "__main__":
    unittest.main()


class FromDbc(unittest.TestCase):
    """An electric vehicle's CAN traffic (three buses, three DBC files) and NOTHING else: no profile, no mapping.  What the autoprofile makes of it must agree with what the vehicle's own profile makes of the same bytes."""
    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "world_aster.db")), "the Aster world is not present")
    def test_dbc_only_agrees_with_the_vehicles_own_profile(self):
        import sqlite3
        from vgpt.live import replay as RP
        from vgpt.live.runtime import Runtime
        from vgpt.vehicle.profile import Profile
        class Sched: laps = 0
        src = os.path.join(ROOT, "data", "world_aster.db"); c = sqlite3.connect(f"file:{src}?mode=ro", uri=True); t = int(c.execute("SELECT MIN(ts) FROM sig_1s WHERE mode=1 AND ts>(SELECT MIN(ts)+86400 FROM sig_1s)").fetchone()[0]); c.close()
        prof = Profile.load("aster_ev"); w = RP.WireSource(Sched(), src, None, prof, t - 3, runtime=False); buf = []; w.sink = lambda ts, bus, p: buf.append((ts, bus, p)); w.deliver(t + 75); w.close()
        frames = [(ts, p.can_id | (0x80000000 if p.extended else 0), p.data, p.channel) for ts, bus, p in buf if bus == "can"]
        d = tempfile.mkdtemp(prefix="dbc_test_"); path = os.path.join(d, "ev.log")
        with open(path, "w", newline="\n") as f:
            for ts, cid, data, ch in sorted(frames, key=lambda x: x[0]):
                ident = f"{cid & 0x1FFFFFFF:08X}" if cid & 0x80000000 else f"{cid:03X}"
                f.write(f"({ts:.6f}) {ch or 'can0'} {ident}##1{data.hex().upper()}\n" if len(data) > 8 else f"({ts:.6f}) {ch or 'can0'} {ident}#{data.hex().upper()}\n")
        dbcs = [os.path.join(prof.base, n) for n in ("powertrain.dbc", "battery.dbc", "body.dbc")]
        db_auto = os.path.join(d, "auto.db"); r = IG.ingest(path, db_auto, dbc=dbcs); mapped = set(r["analysis"].mapped())
        for must in ("speed", "battery_soc", "battery_temperature", "battery_voltage", "motor_temperature", "coolant_flow", "hvac_power"): self.assertIn(must, mapped, f"{must} not recognised from the DBC names; unmapped: {r['analysis'].unmapped}")
        db_ref = os.path.join(d, "ref.db"); rt = Runtime(db_ref, profile=prof, monitor=False)
        for ts, cid, data, ch in sorted(frames, key=lambda x: x[0]):
            from vgpt.protocols import canframes
            rt.feed(ts, "can", canframes.Frame(ts, cid & 0x1FFFFFFF, data, extended=bool(cid & 0x80000000), fd=len(data) > 8, channel=ch))
        rt.publish(); rt.close(); checked = 0
        for concept in ("speed", "battery_soc", "battery_temperature", "motor_temperature"):
            a, b = dict(IG.read_back(db_auto, concept)), dict(IG.read_back(db_ref, concept)); common = sorted(set(a) & set(b)); self.assertGreater(len(common), 20, concept)
            diffs = [abs(a[k] - b[k]) for k in common]; self.assertLess(sorted(diffs)[len(diffs) // 2], 0.05 * (max(b.values()) - min(b.values()) + 1), concept); checked += 1
        self.assertEqual(checked, 4)


class OtherSources(unittest.TestCase):
    def test_android_vhal_json_lines_are_mapped_and_converted(self):
        import json
        from vgpt.protocols import vhal as V
        vh = V.Vhal(); d = tempfile.mkdtemp(); path = os.path.join(d, "hal.jsonl")
        with open(path, "w", newline="\n") as f:
            for i in range(40):
                tr = F.truth(i); ts = int((F.T0 + i) * 1e9)
                for name, val in (("PERF_VEHICLE_SPEED", tr["speed"] / 3.6), ("ENGINE_RPM", tr["engine_rpm"]), ("ENGINE_COOLANT_TEMP", tr["motor_temperature"]), ("ENV_OUTSIDE_TEMPERATURE", tr["ambient_temperature"])):
                    o = vh.encode(name, val, ts=ts); o["bytes"] = ""; f.write(json.dumps(o) + "\n")
        db = os.path.join(d, "hal.db"); r = IG.ingest(path, db, anchor=F.T0); self.assertEqual(r["analysis"].format, "jsonl"); self.assertIn("speed", r["analysis"].mapped())
        rows = IG.read_back(db, "speed"); self.assertGreater(len(rows), 10)
        for ts, v in rows[3:-2:4]: self.assertAlmostEqual(v, F.truth(ts - F.T0)["speed"], delta=2.0)

    def test_autosar_dlt_text_and_syslog_lines_are_read(self):
        import struct
        d = tempfile.mkdtemp(); path = os.path.join(d, "x.dlt"); out = bytearray()
        for k, text in enumerate(("brown-out detected on rail 3", "watchdog reset of the gateway")):
            arg = struct.pack("<L", 0x200) + struct.pack("<H", len(text) + 1) + text.encode() + b"\x00"
            ext = bytes([0x41, 1]) + b"APP1" + b"CTX1"; std = bytes([0x21, k]) + struct.pack(">H", 4 + len(ext) + len(arg)); out += b"DLT\x01" + struct.pack("<LL", int(F.T0) + k, 0) + b"ECU1" + std + ext + arg
        open(path, "wb").write(bytes(out)); recs = list(RD.read(path)); self.assertEqual([r.bus for r in recs], ["log", "log"]); self.assertIn("brown-out", recs[0].payload); self.assertAlmostEqual(recs[1].ts, F.T0 + 1, places=2)
        sys_ = os.path.join(d, "s.log")
        with open(sys_, "w") as f: f.write("2026-10-20 10:00:01 gateway: watchdog reset\n2026-10-20 10:00:02 tcu: link up\n")
        r = list(RD.read(sys_)); self.assertEqual(len(r), 2); self.assertEqual(RD.sniff(sys_), "log")

    def test_an_unreadable_file_says_why(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "x.bin"); open(p, "wb").write(os.urandom(2000))
        with self.assertRaises(ValueError): list(RD.read(p))
        with self.assertRaises(ValueError): AP.autoprofile(p, os.path.join(d, "pack"))

    def test_named_readings_travel_on_the_neutral_wire_format_of_the_collectors(self):
        from vgpt.live import collect
        rec = collect.encode_record(12.5, "table", {"vehicle_speed": 55.2, "Vehicle.Powertrain.TractionBattery.StateOfCharge": 81}); self.assertEqual(rec["bus"], "table")
        self.assertEqual(collect.decode_record(json.loads(json.dumps(rec))), (12.5, "table", {"vehicle_speed": 55.2, "Vehicle.Powertrain.TractionBattery.StateOfCharge": 81}))
        with self.assertRaises(ValueError): collect.decode_record({"ts": 1.0, "bus": "table", "payload": "vehicle_speed=55"})

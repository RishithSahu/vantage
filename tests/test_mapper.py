"""The signal mapper: which quantity is this signal?  It must refuse rather than guess.

Two layers: the rules, one by one, on made-up signals (so a failure names the rule); and the benchmark on real vehicle databases (tests/fixtures/opendbc, MIT licence) against a truth set in which every mapped signal was read by a
person: no wrong mapping, no mapping nobody reviewed, and a recall floor."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))

from vgpt.ingest import mapper as MP  # noqa: E402


def decide(name, unit=None, bits=16, factor=1.0, offset=0.0, signed=False, choices=(), message="", siblings=(), ice=None):
    return MP.decide(MP.Evidence(name=name, unit=unit, bits=bits, factor=factor, offset=offset, signed=signed, choices=list(choices), message=message, siblings=list(siblings), ice=ice))


class Rules(unittest.TestCase):
    def concept(self, *a, **k): return decide(*a, **k).concept

    def test_plain_measurements(self):
        self.assertEqual(self.concept("VehicleSpeed", "km/h", 16, 0.01), "speed")
        self.assertEqual(self.concept("EngineRPM", "rpm", 16, 0.25, ice=True), "engine_rpm")
        self.assertEqual(self.concept("COOLANT_TEMP", "C", 8), "motor_temperature")
        self.assertEqual(self.concept("WHEEL_SPEED_FL", "km/h", 16, 0.01), "wheel_speed_fl")
        self.assertEqual(self.concept("SteeringAngle", "deg", 16, 0.1, signed=True), "steering_angle")

    def test_counters_requests_and_validity_are_not_measurements(self):
        for name in ("VehicleSpeedCounter", "VehicleSpeedChecksum", "TargetVehicleSpeed", "VehicleSpeedRequest", "VehicleSpeedValid", "EngineSpeedFault", "VehicleSpeedWarning", "SpeedUnit"):
            self.assertIsNone(self.concept(name, "km/h", 16, 0.01), name)

    def test_an_update_bit_and_a_quality_flag_are_not_the_state(self):
        self.assertIsNone(self.concept("Smart_Wiper_Motor_Stat_UB", None, 1, choices=("Off", "On")))
        self.assertIsNone(self.concept("iB_BrakePedalApplied_Q", None, 2))

    def test_unit_class_must_agree(self):
        self.assertIsNone(self.concept("EngineSpeed", "C", 8))
        self.assertIsNone(self.concept("CoolantTemperature", "kPa", 8))

    def test_a_unit_the_mapper_cannot_convert_refuses(self):
        self.assertIsNone(self.concept("MAF", "mg/TDC", 8, 5.447))
        self.assertIsNone(self.concept("WheelSpeedFL", "yd/s", 16, 0.01))
        self.assertEqual(self.concept("VehicleSpeed", "Khp", 8), "speed")                      # a misspelling of kph in a published database
        self.assertEqual(self.concept("SteeringAngle", "yes/no", 16, 0.1, signed=True), "steering_angle")   # a field that holds no unit at all is not a unit

    def test_a_unit_written_at_the_end_of_the_name_is_read(self):
        d = decide("DATC_AmbientTemp_F", None, 8, siblings=["DATC_x", "DATC_y"], message="DATC_P_02")
        self.assertEqual(d.concept, "ambient_temperature"); self.assertEqual(d.transform.get("unit"), "F")

    def test_bits_cannot_cover_the_quantity(self):
        self.assertIsNone(self.concept("EngineRPM", "rpm", 5, 1, ice=True))
        self.assertIsNone(self.concept("ODOMETER", "m", 8, 10))                                 # 8 bits of 10 m wrap after 2.5 km
        self.assertIsNone(self.concept("immoTarget1Data", None, 40))

    def test_a_percentage_from_a_wide_field_with_no_unit_refuses(self):
        self.assertIsNone(self.concept("Brake_Pedal", None, 12, siblings=["Speed", "Other"], message="Brake_Pedal"))
        self.assertEqual(self.concept("AcceleratorPedal", None, 8), "accelerator_position")

    def test_signed_quantities_need_signed_fields(self):
        self.assertIsNone(self.concept("SteeringSpeed", None, 8, message="SAS_Data"))
        self.assertIsNone(self.concept("SAS_Speed", None, 8, 4, message="SAS1"))

    def test_an_angle_is_not_its_rate(self):
        self.assertIsNone(self.concept("CAN_RX_HIGH_YAW_ANGLE", "deg", 6, signed=True, message="SensorInput"))
        self.assertEqual(self.concept("YAW_RATE", "deg/s", 12, 0.0625, signed=True), "yaw_rate")

    def test_positions(self):
        self.assertIsNone(self.concept("SPEED_L", None, 16, signed=True, message="MOTORS_DATA"))     # a trailing L is the left side
        self.assertIsNone(self.concept("WHEEL_SPEED", "km/h", 16, 0.01, siblings=["WHEEL_ID"]))     # which wheel it is comes from another signal
        self.assertIsNone(self.concept("TirePress_Frame_Press", None, 8, siblings=["TirePress_Frame_ID", "TirePress_Frame_Temp"]))

    def test_engine_and_motor_follow_the_name_then_the_powertrain(self):
        self.assertEqual(self.concept("TORQUE", "Nm", 12, message="ENGINE_1", ice=True), "engine_torque")
        self.assertEqual(self.concept("TORQUE", "Nm", 12, signed=True, message="DRIVE_UNIT", ice=False), "motor_torque")
        self.assertIsNone(self.concept("MotorTorque", "Nm", 12, ice=True))                           # a combustion vehicle has no drive motor
        self.assertIsNone(self.concept("TORQUE_MOTOR", "Nm", 12, message="EPS_STATUS", ice=True))   # the steering motor's torque
        self.assertEqual(self.concept("EngineTorque", "Nm", 12, message="EngineAndBrake", ice=True), "engine_torque")

    def test_an_unreadable_prefix_in_front_of_a_bare_quantity_refuses(self):
        sib = ["TrnAin_Tq_Rq", "TrnAin_Tq_Actl", "TrnAin_Tq_Dsrd"]
        self.assertIsNone(self.concept("TrnAin_Tq_Actl", "Nm", 11, siblings=sib, message="EngineData_4", ice=True))   # transmission input torque
        self.assertEqual(self.concept("DI_torqueActual", "Nm", 14, 0.25, signed=True, siblings=["DI_torqueMax", "DI_x"], message="DI_torque", ice=False), "motor_torque")   # DI is in the message's own name

    def test_hours_must_be_the_engines(self):
        self.assertIsNone(self.concept("Hour", "h", 5, message="GTW_carState"))
        self.assertIsNone(self.concept("NV_Hour", None, 8, message="HU_CLU_P_02"))

    def test_identifiers_are_not_measurements(self):
        self.assertIsNone(self.concept("DteVehEId_No_Actl", None, 8, message="DTE_HPCMtoECG"))

    def test_states_need_every_word_read(self):
        self.assertEqual(self.concept("DOOR_OPEN_FL", None, 1), "door_state")
        self.assertIsNone(self.concept("RearDriverDoor", None, 2, choices=("Open", "Closed")))
        self.assertIsNone(self.concept("PassengerDoorOpen", None, 1))
        self.assertIsNone(self.concept("Smart_Wiper_Motor_Stat", None, 2))
        self.assertIsNone(self.concept("WprOutsdPkPosn", None, 1))
        self.assertIsNone(self.concept("SEATBELT_PASS_UNLATCHED", None, 1))
        self.assertEqual(decide("SEATBELT_DRIVER_UNLATCHED", None, 1).transform, {"invert01": True})

    def test_a_control_is_not_what_it_controls(self):
        for name, msg in (("HEAD_LIGHT_HANDLE_SWITCH", "CONSOLE"), ("HEADLIGHTS_ON", "STALK_STATUS"), ("WIPER_SWITCH", "STALK_STATUS"), ("CF_Clu_WiperLow", "CLU2"), ("CF_Clu_HeadLampTail", "CLU3"), ("AutoEpbButtnOn_B_Stat", "SelectDriveModeData"), ("ESP_EpbServiceMode", "ESP_EpbStatus")):
            self.assertIsNone(self.concept(name, None, 1, message=msg), name)

    def test_gear_tables_ignore_unusable_labels(self):
        d = decide("GearPos_D_Actl", None, 4, choices=("Unknown", "Reverse", "Undefined_5", "Neutral", "Park", "Drive"))
        self.assertEqual(d.concept, "gear"); self.assertEqual(d.transform["map"], {"Reverse": -1, "Neutral": 0, "Park": 0, "Drive": 1})

    def test_radar_and_cruise_quantities_are_not_the_vehicle_s(self):
        self.assertIsNone(self.concept("CAN_RX_RADAR_HEIGHT", "cm", 7))
        self.assertIsNone(self.concept("Throttle_Cruise", None, 8))

    def test_a_32_bit_odometer_and_altitude_are_fine(self):
        self.assertEqual(self.concept("VehOdo", "km", 32, 0.015625, message="Vehicle_Odo_LS"), "odometer")
        self.assertEqual(self.concept("MCU_elevation", "m", 32, 0.1, signed=True, siblings=["MCU_latitude", "MCU_longitude"], message="MCU_locationStatus2"), "altitude")

    def test_a_fraction_is_scaled_to_percent(self):
        d = decide("GAS_PEDAL", None, 8, 0.005, message="GAS_PEDAL_HYBRID")
        self.assertEqual(d.concept, "accelerator_position"); self.assertEqual(d.transform.get("scale"), 100.0)

    def test_every_refusal_says_why(self):
        d = decide("VehicleSpeedCounter", "km/h", 16, 0.01)
        self.assertIsNone(d.concept); self.assertTrue(d.refused)


class Benchmark(unittest.TestCase):
    """Real databases against the reviewed truth set.  A mistake is a wrong mapping, a mapping on a signal that must be refused, or a mapping nobody has reviewed."""

    @classmethod
    def setUpClass(cls):
        import mapper_benchmark as B
        cls.B = B
        items, over = B.load_truth()
        for k, (c, n) in over.items(): items.append((k[0], k[1], k[2], c, n))
        cls.items = items
        cls.res = B.run("new", items, B.DBC_DIR)
        cls.unreviewed = B.unreviewed(B.DBC_DIR, {(f, m, s) for f, m, s, c, n in items})

    def test_the_truth_set_is_big_and_every_database_in_it_exists(self):
        self.assertGreaterEqual(len(self.items), 800)
        self.assertEqual([r for r in self.res if r[4] == "MISSING"], [])
        self.assertGreaterEqual(len({i[0] for i in self.items}), 60)

    def test_no_wrong_mapping_and_no_kept_refusal(self):
        bad = [r for r in self.res if r[4] not in ("MISSING",) and ((r[3] is None and r[4] is not None) or (r[3] is not None and r[4] is not None and r[4] != r[3]))]
        self.assertEqual(bad, [], "\n".join(f"{r[0]} {r[1]}.{r[2]}: truth {r[3]} got {r[4]}" for r in bad))

    def test_nothing_is_mapped_that_nobody_reviewed(self):
        self.assertEqual([(f, m, s, d.concept) for f, m, s, d in self.unreviewed], [], "review these mappings, then add them to tests/fixtures/mapper_truth.json (bench/mapper_benchmark.py --export)")

    def test_recall_floor(self):
        pos = [r for r in self.res if r[3] is not None]; right = [r for r in pos if r[4] == r[3]]
        self.assertGreaterEqual(len(right) / len(pos), 0.99)


if __name__ == "__main__":
    unittest.main()

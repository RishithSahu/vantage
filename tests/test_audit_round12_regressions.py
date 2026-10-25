"""Focused adversarial regressions for the Round-12/13 audit findings."""
import math
import os
import sqlite3
import struct
import tempfile
import unittest

import numpy as np

from training.build_dataset import gold_of
from training.evaluate import same_gold
from vgpt import derived, store
from vgpt.frame import Frame as SignalFrame
from vgpt.protocols import canframes, dbc, isotp, logs, nmea, uds
from vgpt.qa_core import Sig


class Profile:
    def param(self, name):
        return {"gear_ratio": (9.0, "vehicle"), "wheel_radius_m": (0.33, "vehicle"), "wheelbase_m": (2.7, "vehicle"), "steering_ratio": (15.0, "vehicle")}[name]


class DerivedIntegrity(unittest.TestCase):
    def frame(self, ts, **cols): return SignalFrame(np.asarray(ts), {k: np.asarray(v, float) for k, v in cols.items()}, 1, "1s")

    def test_unknown_mode_uses_speed_per_sample_and_never_fabricates_zero(self):
        f = self.frame([0, 1, 2], speed=[30, 30, 30], battery_power=[10, 10, 10], drive_mode=[np.nan] * 3)
        r = derived.accumulate("energy_used", f, Profile())
        self.assertAlmostEqual(r[0], 30 / 3600); self.assertEqual(r[5], 3); self.assertTrue(r[4])
        missing = self.frame([0, 1], battery_power=[10, 10], drive_mode=[np.nan, np.nan])
        self.assertIsNone(derived.accumulate("energy_used", missing, Profile()))

    def test_sparse_direct_power_is_filled_from_equivalent_inputs(self):
        f = self.frame([0, 1, 2], battery_power=[2, np.nan, np.nan], battery_voltage=[400] * 3, battery_current=[10] * 3,
                       motor_power=[2, np.nan, np.nan], motor_torque=[100] * 3, motor_rpm=[1000] * 3)
        np.testing.assert_allclose(derived.compute("battery_power", f, Profile()).values, [2, 4, 4])
        np.testing.assert_allclose(derived.compute("motor_power", f, Profile()).values, [2, 100000 / 9549, 100000 / 9549])

    def test_invalid_timestamp_gap_is_not_integrated(self):
        f = self.frame([0, 1, 100], speed=[36] * 3, battery_power=[10] * 3, drive_mode=[1] * 3)
        d = derived.accumulate("distance_driven", f, Profile()); e = derived.accumulate("energy_used", f, Profile())
        self.assertAlmostEqual(d[0], .02); self.assertEqual(d[5], 2)
        self.assertAlmostEqual(e[0], 20 / 3600); self.assertEqual(e[5], 2)


class ProtocolBoundaries(unittest.TestCase):
    def test_nmea_requires_exact_checksum_ascii_finite_fields(self):
        good = nmea.encode_rmc(1700000000, 12.9, 77.6, 10, 20)
        self.assertEqual(nmea.decode(good)["type"], "RMC")
        for bad in (good + "XYZ", good[:-1], good + "$GPRMC", good.replace("GPRMC", "GPéMC")):
            with self.assertRaises(nmea.NmeaError): nmea.decode(bad)
        body = "GPRMC,120000.00,A,1254.0000,N,07736.0000,E,NaN,20.0,010126,,,A"
        with self.assertRaises(nmea.NmeaError): nmea.decode(f"${body}*{nmea.checksum(body)}")

    def test_isotp_expires_sessions_bounds_sizes_and_counts_reserved_pci(self):
        r = isotp.Reassembler(); self.assertIsNone(r.feed(bytes([0x10, 10]) + b"ABCDEF", now=0, session="a"))
        self.assertIsNone(r.feed(bytes([0x21]) + b"1234567", now=2, session="a")); self.assertEqual(r.error_counts["timeout"], 1)
        r.feed(bytes([0x10, 10]) + b"ABCDEF", now=3, session="a"); r.feed(bytes([0x21]) + b"1234567", now=3.1, session="b")
        self.assertEqual(r.error_counts["session"], 1)
        r.feed(bytes([0x40])); self.assertEqual(r.error_counts["reserved_pci"], 1)
        r.feed(bytes([0x10, 0]) + (isotp.MAX_MESSAGE + 1).to_bytes(4, "big")); self.assertEqual(r.error_counts["oversize"], 1)

    def test_can_lengths_ids_flags_and_error_identity_are_validated(self):
        raw = struct.pack("<IBBBB", 0x123, 65, 0, 0, 0) + bytes(64)
        with self.assertRaises(ValueError): canframes.parse_socketcan_struct(raw)
        self.assertIsNone(canframes.parse_candump("(1.0) can0 123#" + "00" * 9))
        self.assertIsNone(canframes.parse_candump("(1.0) can0 123456789#00"))
        err = struct.pack("<IBBBB", canframes.CAN_ERR | 4, 0, 0, 0, 0) + bytes(8)
        f = canframes.parse_socketcan_struct(err); self.assertTrue(f.error_frame); self.assertEqual(f.error_class, 4)

    def test_dbc_enums_identity_and_complete_dlc(self):
        text = '''BO_ 291 STD: 1 ECU\n SG_ A : 0|8@1+ (1,0) [0|255] "" ECU\nBO_ 2147483939 EXT: 1 ECU\n SG_ State : 0|8@1+ (1,0) [0|3] "" ECU\nVAL_ 2147483939 State 0 "OFF" 2 "FAULT" ;'''
        db = dbc.Database.load(text)
        self.assertEqual(db.decode(0x123, b"\x07", extended=False), {"A": 7.0})
        meta = db.decode_with_metadata(0x123, b"\x02", extended=True)["State"]
        self.assertEqual(meta.enum, "FAULT")
        self.assertEqual(db.decode_with_metadata(0x123, b"\x03", extended=True)["State"].enum, "UNKNOWN(3)")
        with self.assertRaises(ValueError): db.decode(0x123, b"", extended=False)
        with self.assertRaises(ValueError): db.decode(0x123, b"\x01")

    def test_uds_rejects_partial_records_and_bad_text_codes(self):
        with self.assertRaises(uds.UdsError): uds.decode(bytes([0x59, 0x02, 0xFF, 1]))
        for s in ("X1234", "P1GG0", "P1A", "P1A000", " P0133"):
            with self.assertRaisesRegex(ValueError, "invalid DTC code"): uds.parse_dtc(s)
        self.assertEqual(uds.dtc_text(*uds.parse_dtc("P0133")), "P0133")

    def test_slog_rejects_invalid_severity(self):
        self.assertIsNone(logs.parse(logs.format_slog2("Oct 15 10:00:00.000", "ecu", 9, "fatal")))


class EvaluationIntegrity(unittest.TestCase):
    def test_units_and_complete_lists_are_part_of_numeric_correctness(self):
        self.assertFalse(same_gold({"status": "answered", "value": 12, "unit": "kW", "n": 0}, {"status": "answered", "value": 12, "unit": "W", "n": 0}))
        self.assertTrue(same_gold({"status": "answered", "value": 1, "unit": "kW", "n": 0}, {"status": "answered", "value": 1000, "unit": "W", "n": 0}))
        a = {"status": "answered", "value": 0, "unit": "", "n": 5, "items_sha256": "a"}
        b = dict(a, items_sha256="b"); self.assertFalse(same_gold(a, b))


class LegacyAndRetention(unittest.TestCase):
    def test_legacy_acceleration_marks_first_and_gaps_unknown(self):
        s = Sig([0, 1, 100], {"speed": [10.0, 10.0, 30.0]}, 1, "1 Hz").accel()
        self.assertTrue(math.isnan(s[0])); self.assertEqual(s[1], 0); self.assertTrue(math.isnan(s[2]))

    def test_compaction_does_not_split_boundary_minute(self):
        with tempfile.TemporaryDirectory() as td:
            c = store.open_rw(os.path.join(td, "x.db")); now = store.RET_1S_DAYS * 86400 + 7290
            for ts, speed in ((7288, 100), (7289, 100), (7290, 900), (7291, 900)):
                c.execute("INSERT INTO sig_1s(ts,speed) VALUES(?,?)", (ts, speed))
            c.commit(); store.compact(c, now)
            self.assertIsNone(c.execute("SELECT speed FROM sig_1m WHERE ts=7260").fetchone())
            self.assertEqual(c.execute("SELECT COUNT(*) FROM sig_1s WHERE ts>=7260").fetchone()[0], 4)
            c.close()


if __name__ == "__main__": unittest.main()

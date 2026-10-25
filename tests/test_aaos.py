"""An Android Automotive device as the car: the car service's text is read into Vehicle HAL parcels (sample lines are real output of an AAOS 17 Cuttlefish device), and what Vantage learns from those parcels."""
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.car import aaos  # noqa: E402

SAMPLE = """**All property values**
HalPropValue{Property ID: INFO_VIN(0x11100100), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 553977452058, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: 1GCARVIN123456789}
Can not get property value for property: POWER_POLICY_REQ(0x11100f21) in areaId: GLOBAL(0x0)
HalPropValue{Property ID: PARKING_BRAKE_ON(0x11200402), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 159035504219, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: TRUE}
HalPropValue{Property ID: INFO_MODEL_YEAR(0x11400103), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 554191514150, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: 2023 YEAR}
HalPropValue{Property ID: CURRENT_GEAR(0x11400401), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 554235481835, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: GEAR_PARK(0x4)}
HalPropValue{Property ID: PERF_VEHICLE_SPEED(0x11600207), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 557389054148, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: 25.5 METER_PER_SEC}
HalPropValue{Property ID: TIRE_PRESSURE(0x392881ff), Area ID: WHEEL(0x4), ElapsedRealtimeNanos: 557389054148, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: 241.5 KILOPASCAL}
HalPropValue{Property ID: INFO_MODEL(0x11100102), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 553987441669, SystemStatus: UNAVAILABLE(0x1), VendorStatus: 0, Value: gone}
"""


class Parse(unittest.TestCase):
    def test_values_become_the_forms_the_hal_uses(self):
        got = {p["name"]: p for p in aaos.parse_dump(SAMPLE)}
        self.assertEqual(set(got), {"INFO_VIN", "PARKING_BRAKE_ON", "INFO_MODEL_YEAR", "CURRENT_GEAR", "PERF_VEHICLE_SPEED", "TIRE_PRESSURE"})
        self.assertEqual(got["PARKING_BRAKE_ON"]["int32"], [1]); self.assertEqual(got["INFO_MODEL_YEAR"]["int32"], [2023]); self.assertEqual(got["CURRENT_GEAR"]["int32"], [4])
        self.assertEqual(got["PERF_VEHICLE_SPEED"]["float"], [25.5]); self.assertEqual(got["INFO_VIN"]["string"], "1GCARVIN123456789"); self.assertEqual((got["TIRE_PRESSURE"]["area"], got["TIRE_PRESSURE"]["float"]), (4, [241.5]))
        self.assertEqual(got["PERF_VEHICLE_SPEED"]["prop"], 0x11600207)

    def test_a_property_the_device_says_is_unavailable_is_not_sent(self):
        self.assertNotIn("INFO_MODEL", {p["name"] for p in aaos.parse_dump(SAMPLE)})

    def test_the_drive_script_cycles_the_one_writable_property(self):
        self.assertEqual({aaos.drive_script(t)["TURN_SIGNAL_SWITCH"] for t in range(0, 40, 3)}, {0, 1, 2})


class Learned(unittest.TestCase):
    """What the device sends is read by Vantage with no decoder file and no knowledge that it is Android."""
    def test_vantage_learns_the_standard_properties_of_such_a_device(self):
        import shutil
        from vgpt.car.writer import Writer
        from vgpt.pipe import receiver
        from vgpt.car.transmit import Transmitter
        home = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, home, True); svc = receiver.LinkService(home, roots=[]); self.addCleanup(svc.stop)
        r, w = os.pipe(); svc.attach(r); wr = Writer(w); tx = Transmitter(wr, None, seed=3); t0 = 1_790_000_000.0
        for i in range(80):
            for p in aaos.parse_dump(SAMPLE.replace("25.5 METER", f"{20 + i * 0.1:.1f} METER")): tx.send(t0 + i * 0.5, "vhal", {k: v for k, v in p.items() if k != "name"})
        wr.close(); svc.join(20); svc.drain(30); st = svc.status(); self.assertIsNotNone(st.get("vehicle"), st.get("decision")); self.assertTrue(st["decision"]["mode"].startswith("learned"))
        import sqlite3
        c = sqlite3.connect(st["vehicle"]["db"]); got = {x[0] for x in c.execute("SELECT concept FROM sig_catalog")}; c.close()
        self.assertIn("speed", got); self.assertIn("gear", got)


class StateOfCharge(unittest.TestCase):
    def test_a_battery_level_in_watt_hours_is_a_percentage_of_the_packs_size_and_not_a_percentage_without_it(self):
        import shutil
        import sqlite3
        from vgpt.car.transmit import Transmitter
        from vgpt.car.writer import Writer
        from vgpt.pipe import receiver
        level = "HalPropValue{Property ID: EV_BATTERY_LEVEL(0x11600309), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 1, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: 112500.0 WATT_HOUR}\n"
        cap = "HalPropValue{Property ID: INFO_EV_BATTERY_CAPACITY(0x11600106), Area ID: GLOBAL(0x0), ElapsedRealtimeNanos: 1, SystemStatus: AVAILABLE(0x0), VendorStatus: 0, Value: 150000.0 WATT_HOUR}\n"
        for text, want in ((SAMPLE + level + cap, 75.0), (SAMPLE + level, None)):
            home = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, home, True); svc = receiver.LinkService(home, roots=[]); self.addCleanup(svc.stop)
            r, w = os.pipe(); svc.attach(r); wr = Writer(w); tx = Transmitter(wr, None, seed=3)
            for i in range(80):
                for p in aaos.parse_dump(text): tx.send(1_790_000_000.0 + i * 0.5, "vhal", {k: v for k, v in p.items() if k != "name"})
            wr.close(); svc.join(20); svc.drain(30); c = sqlite3.connect(svc.status()["vehicle"]["db"])
            try:
                got = {x[0] for x in c.execute("SELECT concept FROM sig_catalog")}; soc = c.execute("SELECT soc / 10.0 FROM sig_1s WHERE soc IS NOT NULL ORDER BY ts DESC LIMIT 1").fetchone()
            finally: c.close()
            if want is None: self.assertNotIn("battery_soc", got, "an energy in Wh with no pack size is not a state of charge")
            else: self.assertEqual(soc[0], want)


if __name__ == "__main__":
    unittest.main()

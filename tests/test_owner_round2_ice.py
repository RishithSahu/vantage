"""The owner's questions (rounds 1 and 2, 2026-10-24) put to the petrol and the diesel sample car: every variation that works on the electric car works on a fuel car too, in fuel words, from the vehicle's own tables.

Each check is against separate SQL on the car's own four-year database: litres per 100 km per trip from the fuel-rate and speed minutes, the hour of the day with the highest consumption, the two periods before and after the last service, the level
at the start and end of the last trip, the tank still to fill.  The words of an electric car put to a fuel car (state of charge, battery level, battery consumption, charge time) are read as the matching fuel quantity and the first line says so."""
import os
import re
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CARS = {"petrol": os.path.join(ROOT, "data", "world4", "kestrel_petrol", "kestrel_petrol_4yr.db"), "diesel": os.path.join(ROOT, "data", "world4", "rhino_diesel", "rhino_diesel_4yr.db")}


class OnFuelCar:
    DB = None

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import vehiclegpt; vehiclegpt._adopt_product_model()
        from vgpt.product import make_meaning_brain
        from vgpt.vehicle import lake as LK
        LK.adopt_path(cls.DB); cls.c = sqlite3.connect("file:" + cls.DB.replace("\\", "/") + "?mode=ro", uri=True)
        cls.now = cls.c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0] + 1; cls.tz = int(float(cls.c.execute("SELECT v FROM meta WHERE k='env.tz_h'").fetchone()[0]) * 3600)
        cls.sid = dict(cls.c.execute("SELECT concept, sid FROM sig_catalog").fetchall())["fuel_rate"]
        cls.b = make_meaning_brain(cls.DB, now=cls.now, threads=4)

    def ask(self, q): self.b.dialogue.reset(); r = self.b.ask(q); r["route"] = (r.get("facts") or {}).get("route"); return r["answer"], r

    def l100(self, lo, hi):
        l = self.c.execute("SELECT SUM(v) / 60.0 FROM lk_1m WHERE sid=? AND ts>=? AND ts<?", (self.sid, lo, hi)).fetchone()[0]
        km = self.c.execute("SELECT SUM(speed) / 600.0 FROM sig_1m WHERE ts>=? AND ts<? AND speed IS NOT NULL", (lo, hi)).fetchone()[0]
        return None if not l or not km or km < 1.0 else l / km * 100.0

    def test_each_of_the_last_trips_has_litres_per_100_km_not_litres_per_hour(self):
        a, r = self.ask("Compare fuel consumption across my last five trips"); self.assertEqual(r["route"], "trip:last_n"); self.assertNotIn("L/h", a)
        self.assertEqual(len(re.findall(r"^ - ", a, re.M)), 5)
        for st, en in self.c.execute("SELECT start, end FROM trips WHERE km>0.3 ORDER BY start DESC LIMIT 5").fetchall():
            v = self.l100(st, en + 60)
            if v is not None: self.assertIn(f"{v:.1f} L/100 km", a)
        a, _ = self.ask("what was my fuel consumption on each of my last 4 trips"); self.assertEqual(len(re.findall(r"^ - ", a, re.M)), 4)

    def test_the_hour_of_the_day_with_the_highest_fuel_consumption_is_the_one_in_the_minute_tables(self):
        a, r = self.ask("Which hour of the day has the highest average fuel consumption"); self.assertEqual(r["route"], "calx:by_hour"); lo = self.now - 90 * 86400
        rows = self.c.execute("SELECT CAST(((l.ts + ?) % 86400) / 3600 AS INTEGER) h, SUM(l.v) / 60.0, SUM(s.speed) / 600.0 FROM lk_1m l JOIN sig_1m s ON s.ts = l.ts WHERE l.sid=? AND l.ts>=? AND l.ts<? AND s.speed IS NOT NULL AND s.mode IN (1, 2) GROUP BY h", (self.tz, self.sid, lo, self.now)).fetchall()
        per = {h: l / k * 100.0 for h, l, k in rows if l and k and k >= 5.0}; top = max(per, key=per.get)
        self.assertIn(f"between {top:02d}:00 and {top + 1:02d}:00 ({per[top]:.1f} L/100 km", a)
        a, _ = self.ask("what time of day do I use the least fuel"); low = min(per, key=per.get); self.assertIn(f"lowest is between {low:02d}:00 and {low + 1:02d}:00", a)

    def test_before_and_after_my_last_service_are_two_periods_named_by_the_record(self):
        a, r = self.ask("Compare fuel consumption before and after my last service"); ts = int(float(self.c.execute("SELECT v FROM meta WHERE k='veh.last_service_ts'").fetchone()[0]))
        self.assertEqual(r["route"], "trend:fuel_economy"); self.assertNotIn("same length before it", a); self.assertRegex(a, r"^Fuel consumption since \d+ \w+ \d{4} was [\d.]+ L/100 km against [\d.]+ L/100 km from ")
        after = self.l100(((ts + self.tz) // 86400 + 1) * 86400 - self.tz, self.now)                       # from the first whole local day after the service
        self.assertIsNotNone(after); self.assertAlmostEqual(float(re.search(r"was ([\d.]+) L/100 km", a).group(1)), after, delta=0.15)

    def test_the_last_time_a_tyre_was_below_the_recommended_pressure_is_not_read_as_autumn(self):
        a, r = self.ask("When did my tyre pressure last fall below the recommended threshold"); self.assertEqual(r["route"], "tyre:below_spec"); self.assertIn("The last time a tyre was below", a); self.assertNotRegex(a, r"September \d{4} to November")

    def test_the_level_at_the_start_and_end_of_the_last_trip_is_the_fuel_level_with_a_note(self):
        for q in ("What was my fuel level at the beginning and end of my last trip", "what was my battery state of charge at the beginning and end of my last trip"):
            a, r = self.ask(q); self.assertEqual(r["route"], "trip:bounds"); self.assertIn("at the start", a); self.assertIn("at the end", a); self.assertIn("fuel level", a)
        self.assertIn("no traction battery", a)

    def test_state_of_charge_and_battery_level_are_the_fuel_level(self):
        for q in ("what is my state of charge", "what is my current battery level", "how much battery left"):
            a, _ = self.ask(q); self.assertIn("Fuel level is", a, q)
        a, _ = self.ask("lowest battery level last week"); self.assertIn("lowest fuel level last week", a); self.assertIn("no traction battery", a)
        a, _ = self.ask("what was my state of charge yesterday at 6 pm"); self.assertIn("Fuel level at", a)

    def test_battery_health_on_a_fuel_car_is_the_12_volt_battery(self):
        for q in ("how is my battery", "how is batety", "how healthy is the battery"):
            a, r = self.ask(q); self.assertEqual(r["route"], "v12", q); self.assertIn("12 V battery reads", a)

    def test_how_long_to_fill_the_tank_gives_the_litres_and_does_not_invent_a_time(self):
        a, r = self.ask("how long to fill the tank"); self.assertEqual(r["route"], "now:fill_tank")
        pct = self.c.execute("SELECT v FROM lk_1m WHERE sid=(SELECT sid FROM sig_catalog WHERE concept='fuel_level') AND ts<=? ORDER BY ts DESC LIMIT 1", (self.now,)).fetchone()[0]
        self.assertIn(f"The tank is at {pct:.0f} %", a); self.assertIn("I cannot give a time", a)

    def test_a_trip_estimate_in_fuel_words_in_either_word_order(self):
        for q in ("how much fuel will 50 km use", "how much petrol will 50 km take", "how much will fuel drop if i travel 50km", "how much battery will 50 km use", "how many litres for 50 km"):
            a, _ = self.ask(q); self.assertRegex(a, r"50 km would use about [\d.]+ L", q)
        a, _ = self.ask("can i drive 5000 km"); self.assertTrue(a.startswith("No - 5000 km needs"), a)
        a, _ = self.ask("can i drive 2 km"); self.assertTrue(a.startswith("Yes - 2 km needs"), a)

    def test_start_counts_say_starts_and_a_week_compare_names_both_weeks(self):
        a, _ = self.ask("How many times have I started the engine yesterday"); self.assertTrue(a.startswith("You started the car "), a)
        a, _ = self.ask("Compare my average fuel consumption this week with last week"); self.assertRegex(a, r"this week was [\d.]+ L/100 km against [\d.]+ L/100 km last week")

    def test_a_day_without_driving_is_a_zero_speed_not_an_error(self):
        a, _ = self.ask("what was highest speed on 11th october"); self.assertIn("11 Oct", a)


for name, path in CARS.items():
    globals()["OnThe" + name.capitalize() + "Car"] = unittest.skipUnless(os.path.exists(path), "needs the four-year " + name + " sample car")(type("OnThe" + name.capitalize() + "Car", (OnFuelCar, unittest.TestCase), {"DB": path}))
del OnFuelCar


if __name__ == "__main__": unittest.main()

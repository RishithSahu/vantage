"""Bugs found by auditing the assistant's answers on every vehicle (bench/audit_answers.py), each with the answer it used to give.

* a diesel truck answered "how many mAh is our battery" with "battery capacity: 60 kWh" (the world builder left electric entries in a combustion vehicle's specification),
* "is the battery charging" on a diesel was answered "I cannot answer that kind of question about that quantity",
* "why is the range shrinking" on a diesel was answered in terms of battery state of charge and battery power."""
import os
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
NOW = 1792348500


def ask(db, table, q):
    path = os.path.join(ROOT, "data", db)
    if not os.path.exists(path): raise unittest.SkipTest(db + " is not present")
    from test_followups import Scripted
    from vgpt.meaning.pipeline import MeaningBrain
    from vgpt.vehicle import lake as LK
    for k in ("VGPT_LAYERS_OFF", "VGPT_ARBITRATE", "VGPT_MODEL_ONLY"): os.environ.pop(k, None)
    LK.adopt_path(path)
    try: return MeaningBrain(path, Scripted(table), now=NOW).ask(q)
    finally: LK.release()


class CombustionVehicles(unittest.TestCase):
    def frame(self, *a):
        from vgpt.meaning.frame import Frame
        return Frame(*a)

    def test_the_specification_of_a_fuel_vehicle_has_no_battery_entries(self):
        for db in ("world_rhino.db", "world_kestrel.db"):
            p = os.path.join(ROOT, "data", db)
            if not os.path.exists(p): continue
            keys = {k for (k,) in sqlite3.connect(f"file:{p}?mode=ro", uri=True).execute("SELECT k FROM meta")}
            self.assertFalse(keys & {"veh.cap_kwh", "veh.soh", "veh.home_charger_kw"}, db)

    def test_a_battery_question_on_a_diesel_says_there_is_no_traction_battery(self):
        for q, f in (("is the battery charging", ("now", "charging_session")), ("how many mAh is our battery", ("spec", "battery_capacity")), ("has the charging speed been getting worse", ("trend", "charging_session", "change")),
                     ("how big is the home charger", ("spec", "home_charger_power"))):
            r = ask("world_rhino.db", {q: self.frame(*f)}, q); self.assertEqual(r["facts"]["route"], "outcome:no_traction_battery", q); self.assertIn("no traction battery", r["answer"]); self.assertNotIn("kWh", r["answer"].split("\n")[0])

    def test_the_range_of_a_fuel_vehicle_is_discussed_in_fuel_terms(self):
        for q, f in (("why is the range shrinking", ("why", "range_estimate", "drop")), ("how did range change across the two weeks", ("trend", "range_estimate", "change"))):
            r = ask("world_rhino.db", {q: self.frame(*f)}, q); a = r["answer"]
            self.assertEqual(r["facts"]["route"], "trend:fuel_economy"); self.assertIn("L/100 km", a); self.assertNotIn("battery_soc", a); self.assertNotIn("state of charge", a)
            speeds = [float(x) for x in a.split("Average speed while moving: ")[1].split(" km/h")[0:1]] if "Average speed while moving" in a else []
            self.assertTrue(all(0 < s < 200 for s in speeds), "speed is stored in tenths of km/h and must be shown in km/h")

    def test_a_recording_that_only_labels_itself_diesel_says_what_it_holds_and_nothing_else(self):
        r = ask("cs5_diesel_ice.db", {"why is the range shrinking": self.frame("why", "range_estimate", "drop")}, "why is the range shrinking")
        self.assertEqual(r["facts"]["route"], "now:not_reported"); self.assertIn("holds only speed", r["answer"])

    def test_how_hot_is_the_motor_on_a_fuel_vehicle_is_the_engine_coolant_temperature(self):
        for q, f in (("how hot is the motor", ("now", "motor_temperature")), ("highest motor temperature this week", ("extreme", "motor_temperature", "max"))):
            r = ask("world_rhino.db", {q: self.frame(*f)}, q); self.assertIn("engine coolant temperature", r["answer"].lower(), q); self.assertEqual(r["facts"]["route"].split(":")[1], "engine_temperature")

    def test_a_follow_up_after_a_question_about_a_missing_system_does_not_break(self):
        from test_followups import Scripted
        from vgpt.meaning.pipeline import MeaningBrain
        from vgpt.vehicle import lake as LK
        path = os.path.join(ROOT, "data", "world_rhino.db")
        if not os.path.exists(path): self.skipTest("world_rhino.db is not present")
        LK.adopt_path(path)
        try:
            b = MeaningBrain(path, Scripted({"is the battery temperature normal": self.frame("now", "battery_temperature"), "and yesterday": self.frame("ambiguous", "none")}), now=NOW)
            b.ask("is the battery temperature normal"); r = b.ask("and yesterday"); self.assertNotIn("could not compute", r["answer"])
        finally: LK.release()

    def test_a_definition_of_something_a_fuel_vehicle_lacks_is_still_given(self):
        r = ask("world_rhino.db", {"what is soh": self.frame("define", "battery_soh")}, "what is soh")
        self.assertEqual(r["facts"]["route"], "define:battery_soh"); self.assertIn("usable capacity", r["answer"]); self.assertIn("does not have or record it", r["answer"])
        r = ask("world_rhino.db", {"lowest the battery charge has been": self.frame("extreme", "battery_soc", "min")}, "lowest the battery charge has been")
        self.assertEqual(r["facts"]["route"], "outcome:no_traction_battery")

    def test_the_engine_temperature_at_an_instant(self):
        r = ask("world_rhino.db", {"motor temp 4pm yday": self.frame("at", "motor_temperature")}, "motor temp 4pm yday")
        self.assertEqual(r["facts"]["route"], "at:engine_temperature"); self.assertIn("Engine coolant temperature at", r["answer"])


if __name__ == "__main__":
    unittest.main()

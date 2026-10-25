"""Combustion vehicles (diesel and petrol): the vehicle is recognised from the data it sends - never from a setting - and the assistant answers in fuel terms.

Two short worlds (16 days) are built once: a diesel truck (J1939, AdBlue, particulate filter) and a petrol hatchback (OBD-II style engine data).  The shipped 105-day datasets are checked when present."""
import os as _os; _os.environ.setdefault("VGPT_DRIVING_FIXTURE", "1")                                  # the small derived driving fixture stands in for the real traces when they are absent (tools_testdata.py)
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from vgpt import datagen as DG  # noqa: E402,F401
from vgpt.meaning import ontology  # noqa: E402
from vgpt.meaning.frame import Frame  # noqa: E402
from vgpt.meaning.pipeline import MeaningBrain  # noqa: E402
from vgpt.vehicle import caps as CAPS, lake as LK  # noqa: E402
from vgpt.world import build as WB  # noqa: E402
from test_eventx import Scripted  # noqa: E402

_W = {}


def world(profile):
    if profile not in _W:
        path = os.path.join(tempfile.gettempdir(), f"combustion_{profile}.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(path + e)
            except OSError: pass
        LK.release(); gv = WB.build_world(profile, days=16, seed=7, end="2026-10-18", verbose=False); WB.write_world(gv, path, verbose=False); LK.release()
        _W[profile] = {"path": path, "now": int(gv.t0 + gv.n) - 3600}
    return _W[profile]


def tearDownModule():
    LK.release()


class Recognition(unittest.TestCase):
    def caps(self, profile):
        c = sqlite3.connect(world(profile)["path"]); k = CAPS.detect(c); c.close(); return k

    def test_a_diesel_is_recognised_by_its_diesel_only_signals(self):
        k = self.caps("rhino_diesel"); self.assertEqual((k["energy"], k["fuel_type"]), ("ice", "diesel")); self.assertFalse(k["battery"]); self.assertTrue(k["fuel"] and k["refuel"])
        self.assertEqual(sorted(k["aftertreatment"]), ["AdBlue / SCR", "particulate filter"]); self.assertIn("def_level", k["signals"]); self.assertNotIn("lambda", k["signals"]); self.assertIn("Diesel vehicle", k["summary"])

    def test_a_petrol_car_is_recognised_by_its_spark_ignition_signals(self):
        k = self.caps("kestrel_petrol"); self.assertEqual((k["energy"], k["fuel_type"]), ("ice", "petrol")); self.assertFalse(k["battery"]); self.assertEqual(k["aftertreatment"], ["catalytic converter"])
        self.assertIn("lambda", k["signals"]); self.assertNotIn("dpf_soot_load", k["signals"]); self.assertIn("Petrol vehicle", k["summary"])

    def test_an_electric_vehicle_and_a_speed_only_file_are_told_apart(self):
        for name, energy in (("world_aster.db", "ev"), ("cs5_petrol_ice.db", "unknown"), ("cs5_diesel_ice.db", "unknown")):
            p = os.path.join(ROOT, "data", name)
            if not os.path.exists(p): continue
            c = sqlite3.connect(f"file:{p}?mode=ro", uri=True); k = CAPS.detect(c); c.close(); self.assertEqual(k["energy"], energy, name)
            if energy == "unknown": self.assertFalse(k["battery"] or k["fuel"] or k["engine"]); self.assertEqual(k["declared"]["energy"], "ice"); self.assertIn("nothing in the data confirms it", k["summary"]); self.assertTrue(CAPS.is_combustion(k))      # the file's own label counts only because its signals say nothing

    def test_the_fuel_type_is_inferred_from_the_rev_range_when_nothing_else_tells(self):
        c = sqlite3.connect(world("kestrel_petrol")["path"]); sid = dict(c.execute("SELECT concept, sid FROM sig_catalog").fetchall())
        for concept in CAPS.PETROL_ONLY: c.execute("DELETE FROM lk_1m WHERE sid=?", (sid[concept],)); c.execute("DELETE FROM lk_1s WHERE sid=?", (sid[concept],))
        c.execute("UPDATE lk_1m SET vmax=6400 WHERE sid=?", (sid["engine_rpm"],)); k = CAPS.detect(c); c.close()
        self.assertEqual(k["fuel_type"], "petrol"); self.assertIn("speed range", k["summary"])


class Builder(unittest.TestCase):
    def test_a_combustion_world_carries_no_battery_and_does_refuel(self):
        for profile, extra, absent in (("rhino_diesel", "dpf_soot_load", "lambda"), ("kestrel_petrol", "lambda", "dpf_soot_load")):
            c = sqlite3.connect(world(profile)["path"]); cat = dict(c.execute("SELECT concept, sid FROM sig_catalog").fetchall())
            self.assertIn(extra, cat); self.assertNotIn(absent, cat); self.assertNotIn("battery_soc", cat)
            self.assertEqual(c.execute("SELECT COUNT(soc) + COUNT(power) + COUNT(batt) FROM sig_1m").fetchone()[0], 0); self.assertEqual(c.execute("SELECT COUNT(*) FROM charges").fetchone()[0], 0)
            lo, hi = c.execute("SELECT MIN(v), MAX(v) FROM lk_1m WHERE sid=?", (cat["fuel_level"],)).fetchone(); self.assertTrue(0 <= lo < hi <= 100)
            lit = c.execute("SELECT SUM(v) / 60.0 FROM lk_1m WHERE sid=?", (cat["fuel_rate"],)).fetchone()[0]; km = c.execute("SELECT SUM(speed) / 600.0 FROM sig_1m").fetchone()[0]; self.assertTrue(6 <= lit / km * 100 <= 16, lit / km * 100)
            mo = c.execute("SELECT MIN(motor), MAX(motor) FROM sig_1m").fetchone(); self.assertTrue(100 <= mo[0] and 700 <= mo[1] <= 1100, "the engine coolant is stored in tenths of a degree like every temperature of the wide table"); c.close()

    def test_a_diesel_particulate_filter_fills_and_regenerates(self):
        c = sqlite3.connect(world("rhino_diesel")["path"]); cat = dict(c.execute("SELECT concept, sid FROM sig_catalog").fetchall())
        hi = c.execute("SELECT MAX(v) FROM lk_1m WHERE sid=?", (cat["dpf_soot_load"],)).fetchone()[0]; reg = c.execute("SELECT MAX(v) FROM lk_1m WHERE sid=?", (cat["dpf_regeneration"],)).fetchone()[0]
        self.assertGreater(hi, 60); self.assertEqual(reg, 1.0); self.assertTrue(c.execute("SELECT 1 FROM logs WHERE msg LIKE '%DPF regeneration started%'").fetchone()); c.close()


class Answers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onto = ontology.load()

    def brain(self, profile):
        w = world(profile); LK.release(); LK.adopt_path(w["path"]); m = Scripted(); b = MeaningBrain(w["path"], m); b.now = w["now"]; return b, m

    def ask(self, b, m, text, frame):
        m.t[text] = frame; b.dialogue.reset(); b.frame_model = m; r = b.ask(text); return r, (r.get("facts") or {})

    def test_energy_left_is_the_fuel_level_and_the_range_it_gives(self):
        for profile in ("rhino_diesel", "kestrel_petrol"):
            b, m = self.brain(profile)
            for frame in (Frame("estimate", "range_estimate"), Frame("now", "fuel_level"), Frame("now", "battery_soc")):
                r, f = self.ask(b, m, "how much fuel is left " + frame.concept, frame); self.assertIn("Fuel level is", r["answer"]); self.assertIn("L/100 km", r["answer"]); self.assertGreater(f["range_km"], 100)
            r, f = self.ask(b, m, "how much battery is left", Frame("now", "battery_capacity")) if "battery_capacity" in self.onto.concepts else (self.ask(b, m, "how much battery is left", Frame("now", "battery_soc")))
            self.assertIn("no traction battery", r["answer"])

    def test_consumption_and_fuel_used_and_refuels(self):
        b, m = self.brain("rhino_diesel")
        r, f = self.ask(b, m, "what is my fuel consumption", Frame("stat", "wh_per_km", "mean")); self.assertIn("L/100 km", r["answer"]); self.assertTrue(6 <= f["l100"] <= 16)
        r, f = self.ask(b, m, "how much fuel did i use over the last 14 days", Frame("stat", "fuel_level", "sum")); self.assertIn("L of fuel were used", r["answer"]); self.assertGreater(f["litres"], 20)
        r, f = self.ask(b, m, "when did i last refuel", Frame("events", "charging_session", "last")); self.assertIn("L added", r["answer"]); self.assertGreaterEqual(f["count"], 1)
        r, f = self.ask(b, m, "how many times did i refuel over the last 14 days", Frame("events", "charging_session", "count")); self.assertIn("refuel", r["answer"])

    def test_engine_health_names_the_signals_this_engine_has(self):
        b, m = self.brain("rhino_diesel"); r, f = self.ask(b, m, "how is the engine", Frame("health", "motor_condition")); self.assertIn("particulate filter", r["answer"]); self.assertIn("AdBlue", r["answer"]); self.assertNotIn("catalyst", r["answer"])
        b, m = self.brain("kestrel_petrol"); r, f = self.ask(b, m, "how is the engine", Frame("health", "motor_condition")); self.assertIn("fuel trim", r["answer"]); self.assertIn("catalyst", r["answer"]); self.assertNotIn("AdBlue", r["answer"])

    def test_what_is_this_vehicle_says_what_the_data_says(self):
        b, m = self.brain("kestrel_petrol"); r, f = self.ask(b, m, "what kind of vehicle is this", Frame("spec", "vehicle_model")); self.assertIn("Kestrel", r["answer"]); self.assertIn("Petrol vehicle", r["answer"]); self.assertIn("spark-ignition", r["answer"])

    def running(self, b, path):
        """A moment when the engine is running (so torque and speed are both stored)."""
        c = sqlite3.connect(path); sid = c.execute("SELECT sid FROM sig_catalog WHERE concept='engine_rpm'").fetchone()[0]
        ts = c.execute("SELECT MAX(ts) FROM lk_1m WHERE sid=? AND v>600", (sid,)).fetchone()[0]; c.close(); b.now = int(ts) + 30

    def test_power_is_torque_times_speed_whatever_the_model_called_it(self):
        b, m = self.brain("kestrel_petrol"); self.running(b, world("kestrel_petrol")["path"])                                                                       # an OBD-II style petrol car reports engine LOAD, not torque: power cannot be worked out, and it says so
        r, f = self.ask(b, m, "how much power do we produce", Frame("now", "battery_power")); self.assertIn("does not report its torque", r["answer"]); self.assertIn("engine load", r["answer"]); self.assertNotIn("Engine power is about", r["answer"])
        for profile in ("rhino_diesel",):
            b, m = self.brain(profile); self.running(b, world(profile)["path"])
            for text, frame in (("how much power do we produce", Frame("now", "battery_power")), ("what is the engine power", Frame("now", "engine_load")), ("how many kilowatts is it making", Frame("now", "battery_capacity")),
                                ("how much horsepower", Frame("now", "horsepower")), ("what is my power output", Frame("now", "motor_power"))):
                r, f = self.ask(b, m, text, frame); self.assertIn("Engine power is about", r["answer"], (profile, text)); self.assertIn(" hp)", r["answer"]); self.assertAlmostEqual(f["kw"], f["torque_nm"] * f["rpm"] / 9549.0, places=6)
            r, f = self.ask(b, m, "what is the engine load", Frame("now", "engine_load")); self.assertIn("Engine load is", r["answer"])                  # no power word: the model's reading stands
            r, f = self.ask(b, m, "what was my peak power", Frame("extreme", "battery_power", "max")); self.assertIn("highest engine power", r["answer"]); self.assertGreater(f["kw"], 5)
            r, f = self.ask(b, m, "what is my average power", Frame("stat", "motor_power", "mean")); self.assertIn("average power while running", r["answer"])

    def test_an_engine_that_reports_no_torque_says_so_instead_of_guessing(self):
        b, m = self.brain("rhino_diesel"); c = sqlite3.connect(world("rhino_diesel")["path"]); sid = c.execute("SELECT sid FROM sig_catalog WHERE concept='engine_torque'").fetchone()[0]; c.close()
        saved = b._vreader if getattr(b, "_vreader", None) else None; self.running(b, world("rhino_diesel")["path"]); b._vreader = CAPS.Reader(b.c); b._vreader.sid.pop("engine_torque", None)
        r, f = self.ask(b, m, "how much power do we produce", Frame("now", "battery_power")); self.assertIn("does not report its torque", r["answer"]); self.assertIn("Nothing was estimated", r["answer"])

    def test_a_vehicle_that_declares_an_engine_but_sends_only_speed_is_not_answered_in_battery_terms(self):
        src = os.path.join(ROOT, "data", "cs5_diesel_ice.db")
        if not os.path.exists(src): self.skipTest("Case Study 5 diesel file not present")
        path = os.path.join(tempfile.gettempdir(), "combustion_cs5_diesel.db"); shutil.copy(src, path); LK.release(); LK.adopt_path(path); m = Scripted(); b = MeaningBrain(path, m); b.now = 1789120740
        for text, frame in (("how much power do we produce", Frame("now", "battery_power")), ("what is my battery level", Frame("now", "battery_soc")), ("how is the engine", Frame("health", "motor_condition"))):
            r, f = self.ask(b, m, text, frame); self.assertIn("diesel vehicle", r["answer"], text); self.assertNotIn("battery power", r["answer"]); self.assertIn("Nothing was estimated", r["answer"]); self.assertIn("only speed", r["answer"])
            if frame.concept.startswith("battery"): self.assertIn("no traction battery", r["answer"])                                     # the missing battery is mentioned when the question is about the battery, not when it is about the engine

    def test_the_diesel_file_announces_its_fault_codes_once(self):
        src = os.path.join(ROOT, "data", "cs5_diesel_ice.db")
        if not os.path.exists(src): self.skipTest("Case Study 5 diesel file not present")
        c = sqlite3.connect(f"file:{src}?mode=ro", uri=True); n, d = c.execute("SELECT COUNT(*), COUNT(DISTINCT detail) FROM events WHERE type='DTC_SET'").fetchone(); c.close()
        self.assertLessEqual(n, 5, "a code that stays active is one event, not one per minute"); self.assertGreaterEqual(d, 1)

    def test_a_signal_only_this_vehicle_declares_is_found_by_its_own_vocabulary(self):
        b, m = self.brain("rhino_diesel"); r, f = self.ask(b, m, "how full is the adblue tank", Frame("now", "unlisted")); self.assertIn("AdBlue level is", r["answer"])
        b, m = self.brain("kestrel_petrol"); r, f = self.ask(b, m, "what is the long term fuel trim", Frame("now", "longitudinal_acceleration")); self.assertIn("Long-term fuel trim", r["answer"])


if __name__ == "__main__":
    unittest.main()

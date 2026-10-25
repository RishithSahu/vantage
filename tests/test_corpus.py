"""The versioned test corpus (tests/corpus/, built by `tools_testdata.py corpus`): every kind of vehicle the case study names - electric, petrol, diesel, hybrid - and the situations that matter - protocol variation, missing signals,
faults, stale data, no data - answered by the real product model and checked against plain SQL on the same file.

The corpus is an integration-test bundle: when it is absent the tests skip with the way to build it, and under VGPT_REQUIRE_TESTDATA=1 (the release gate) they fail."""
import json
import os
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests import testdata  # noqa: E402

CORPUS = os.path.join(ROOT, "tests", "corpus")
HAVE, WHY = testdata.gate("corpus")
MANIFEST = json.load(open(os.path.join(CORPUS, "MANIFEST.json"), encoding="utf-8")) if HAVE else {"files": []}
BY = {e["name"]: e for e in MANIFEST["files"]}
KINDS = {"ev": "ev", "petrol": "petrol", "diesel": "diesel", "hybrid": "hybrid", "bus": "ev", "obd": "petrol", "prop": "diesel", "missing_signals": "ev", "faults": "ev", "stale": "ev"}


def path(name): return os.path.join(CORPUS, BY[name]["file"])


def sql(name, q, *a):
    c = sqlite3.connect("file:" + path(name).replace("\\", "/") + "?mode=ro", uri=True)
    try: return c.execute(q, a).fetchall()
    finally: c.close()


def latest(name, col): return sql(name, f"SELECT {col} FROM sig_1s WHERE {col} IS NOT NULL ORDER BY ts DESC LIMIT 1")[0][0]


def fuel_pct(name):
    sid = sql(name, "SELECT sid FROM sig_catalog WHERE concept='fuel_level'")[0][0]
    return sql(name, "SELECT v FROM lk_1s WHERE sid=? ORDER BY ts DESC LIMIT 1", sid)[0][0]


_BRAINS = {}


def ask(name, q):
    from vgpt.vehicle import lake as LK
    if "model" not in _BRAINS:
        import vehiclegpt; vehiclegpt._adopt_product_model()
        from vgpt.product import _frame_model
        _BRAINS["model"] = _frame_model(os.environ.get("VGPT_MEANING_ADAPTER"), 4)
    from vgpt.meaning.pipeline import MeaningBrain
    from vgpt.product import set_vehicle_of
    LK.release(); LK.adopt_path(path(name)); set_vehicle_of(_BRAINS["model"], path(name)); _BRAINS["model"].refresh()
    b = _BRAINS.get(name)
    if b is None: b = _BRAINS[name] = MeaningBrain(path(name), _BRAINS["model"], now=BY[name]["now"])
    LK.adopt_path(path(name)); b.dialogue.reset(); r = b.ask(q); return r["answer"], r


@unittest.skipUnless(HAVE, WHY)
class Files(unittest.TestCase):
    def test_every_file_matches_its_recorded_hash_size_and_vehicle_kind(self):
        from tools_testdata import sha256
        from vgpt.vehicle import caps
        self.assertGreaterEqual(len(BY), 11)
        for name, e in BY.items():
            p = path(name); self.assertTrue(os.path.exists(p), name); self.assertEqual(os.path.getsize(p), e["bytes"], name); self.assertEqual(sha256(p), e["sha256"], name + " changed: rebuild the corpus and the manifest")
            c = sqlite3.connect("file:" + p.replace("\\", "/") + "?mode=ro", uri=True)
            try: self.assertEqual(caps.vehicle_kind(c) or "none", e["kind"], name)
            finally: c.close()

    def test_the_case_study_scope_is_covered(self):
        kinds = {e["kind"] for e in BY.values()}; self.assertTrue({"ev", "petrol", "diesel", "hybrid"} <= kinds)
        scen = " ".join(e["scenario"] for e in BY.values()); [self.assertIn(w, scen) for w in ("protocol", "missing signals", "faults", "stale data", "no data", "hybrid")]
        self.assertLess(sum(e["bytes"] for e in BY.values()), 80e6, "the corpus must stay small enough to ship")


@unittest.skipUnless(HAVE, WHY)
class Vehicles(unittest.TestCase):
    def test_the_electric_car_reads_its_state_of_charge_and_answers_a_fuel_question_with_it(self):
        a, _ = ask("ev", "what is my state of charge"); self.assertIn(f"{latest('ev', 'soc') / 10.0:.0f} %", a)
        a, _ = ask("ev", "how much fuel is left"); self.assertIn("electric car", a); self.assertIn(f"{latest('ev', 'soc') / 10.0:.0f} %", a)

    def test_the_petrol_and_the_diesel_car_read_their_fuel_and_have_no_traction_battery(self):
        for name in ("petrol", "diesel", "obd", "prop"):
            a, _ = ask(name, "how much fuel is left"); self.assertIn(f"Fuel level is {fuel_pct(name):.0f} %", a, name)
            a, _ = ask(name, "what is the battery temperature"); self.assertIn("no traction battery", a, name)

    def test_the_hybrid_has_both_a_battery_and_a_tank(self):
        a, r = ask("hybrid", "what is my state of charge"); self.assertIn(f"{latest('hybrid', 'soc') / 10.0:.0f} %", a); self.assertNotIn("does not have that system", a); self.assertNotIn("no traction battery", a)
        a, _ = ask("hybrid", "how much fuel is left"); self.assertIn(f"Fuel level is {fuel_pct('hybrid'):.0f} %", a); self.assertNotIn("electric car", a)
        a, _ = ask("hybrid", "what is the battery temperature"); self.assertIn("won't estimate", a)                                          # a hybrid that records no pack temperature says so, it does not invent one

    def test_every_bus_answers_the_speed_in_kilometres_per_hour(self):
        for name in ("bus", "obd", "prop", "ev", "petrol", "diesel"):
            a, _ = ask(name, "what is my speed right now"); self.assertRegex(a, r"km/h", name)


@unittest.skipUnless(HAVE, WHY)
class Situations(unittest.TestCase):
    def test_an_impossible_tyre_pressure_is_a_faulty_sensor_not_a_flat_tyre(self):
        a, _ = ask("bus", "what is my tyre pressure"); self.assertEqual(sql("bus", "SELECT COUNT(*) FROM ext_1m WHERE tp_fr < 0 AND ts = (SELECT MAX(ts) FROM ext_1m)")[0][0], 1, "this file really carries a negative pressure")
        self.assertIn("faulty reading", a); self.assertNotIn("front right -", a); self.assertNotIn("187 %", a)

    def test_signals_the_vehicle_does_not_record_are_said_to_be_missing_and_never_invented(self):
        for q in ("what is the battery temperature", "how is the coolant flow"):
            a, _ = ask("missing_signals", q); self.assertIn("couldn't find any data", a, q); self.assertNotRegex(a, r"\d+(\.\d+)? (°C|°C|L/min)")
        a, _ = ask("missing_signals", "what is my state of charge"); self.assertIn("%", a)                                                 # what it does record still answers

    def test_injected_faults_are_counted_exactly(self):
        f = BY["faults"]["facts"]; a, _ = ask("faults", "how many ECU resets in the last 24 hours"); self.assertIn(f"There were {f['ecu_resets_last_24h']} ECU resets", a); self.assertEqual(f["ecu_resets_last_24h"], 3)
        a, _ = ask("faults", "any warnings today"); self.assertIn("There were 5 warnings today", a)

    def test_a_record_that_ended_two_days_ago_is_not_a_current_reading(self):
        for q in ("what is my speed right now", "what is my state of charge"):
            a, _ = ask("stale", q); self.assertIn("not a current reading", a, q); self.assertIn("48 h", a, q)

    def test_a_file_with_no_data_says_so_and_estimates_nothing(self):
        for q in ("what is my speed right now", "what is my state of charge", "what is my tyre pressure"):
            a, _ = ask("no_data", q); self.assertIn("won't estimate", a, q)
        a, _ = ask("no_data", "how many trips last week"); self.assertIn("There were 0 trips", a)


if __name__ == "__main__": unittest.main()

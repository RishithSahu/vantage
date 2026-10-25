"""The digest answers (summary, what changed, unusual, usual behaviour, why one trip / why the energy fell while parked), run on the shipped vehicles of three kinds - an electric car, a petrol car, a diesel truck - and
on a recorded file that holds almost nothing.  The frames are given to the resolver directly (the language model is not involved), so what is tested is what the vehicle's data makes of a meaning:

  * every figure is checked against plain SQL on the same database;
  * what is said depends on the vehicle (energy and kWh for the electric car, fuel and L/100 km for the others, nothing about a battery on a diesel);
  * a quantity or period with no data says so and invents nothing;
  * a premise that is not true in the data ("why was it so hot") is corrected, not explained."""
import os
import re
import sqlite3
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_NO_QA_CLF"] = "1"

DATA = os.path.join(ROOT, "data")
DAY = 86400


class V2(unittest.TestCase):
    """Frame version 2 for the length of one test only: an environment variable set at import time would reach every other test module of the session."""
    def setUp(self): self._env = mock.patch.dict(os.environ, {"VGPT_FRAME": "2"}); self._env.start()
    def tearDown(self): self._env.stop()


class NoModel:
    """Stands in for the frame model: the resolver is driven with frames, nothing is parsed."""
    def glossary(self, *a, **k): return ""
    def parse(self, *a, **k): return None, {}
    def refresh(self): pass


def have(*names): return all(os.path.exists(os.path.join(DATA, n)) for n in names)


_cache = {}


def brain(db):
    if db not in _cache:
        from vgpt.product import make_meaning_brain
        from vgpt.vehicle import lake as LK
        p = os.path.join(DATA, db); LK.adopt_path(p); c = sqlite3.connect(f"file:{p}?mode=ro", uri=True); end = int(c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]) + 120; c.close()
        _cache[db] = (make_meaning_brain(p, slm=NoModel(), now=end), end, p)
    b, end, p = _cache[db]
    from vgpt.vehicle import lake as LK
    LK.adopt_path(p); return b, end, p


def ask(db, text, frame):
    from vgpt.meaning.frame import Frame
    b, end, p = brain(db); b.qcs.last = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []; b._last_data = None
    return b.resolver.answer(text, Frame.from_text(frame))


def sql(db, q, *a):
    c = sqlite3.connect(f"file:{os.path.join(DATA, db)}?mode=ro", uri=True)
    try: return c.execute(q, a).fetchall()
    finally: c.close()


@unittest.skipUnless(have("world_aster.db"), "needs the Aster world")
class ElectricCar(V2):
    db = "world_aster.db"

    def test_a_week_summary_gives_trips_distance_energy_and_what_stands_out(self):
        r = ask(self.db, "summarise the last week", "summary none none none none none none none"); a = r["answer"]; b, end, p = brain(self.db)
        self.assertEqual(r["facts"]["route"], "digest:summary"); self.assertRegex(a, r"\d+ trips?, \d+ km"); self.assertIn("kWh", a); self.assertIn("Wh/km", a); self.assertNotIn("L/100", a); self.assertNotIn("lkv_", a)
        n = int(re.search(r"(\d+) trips?, ", a)[1]); km = float(re.search(r", ([\d,]+) km", a)[1].replace(",", ""))
        rows = sql(self.db, "SELECT COUNT(*), COALESCE(SUM(km),0) FROM trips WHERE start>=? AND start<? AND km>=0.3", end - 7 * DAY, end)[0]
        self.assertLessEqual(abs(n - rows[0]), 1, (n, rows)); self.assertLess(abs(km - rows[1]), max(3, 0.03 * rows[1]), (km, rows))
        self.assertIn("Based on", a); self.assertTrue(r["facts"]["steps"])

    def test_what_changed_lists_only_real_changes_and_each_has_a_before_and_after(self):
        r = ask(self.db, "what changed since last week", "summary none change none none none none none"); a = r["answer"]
        self.assertEqual(r["facts"]["route"], "digest:changes")
        if r["facts"]["changes"]: self.assertLessEqual(a.count("\n- "), 5); self.assertTrue(all(re.search(r"from [\d.,]+ to [\d.,]+|[\d,]+ now, \d+|from [\d,]+ to [\d,]+", ln) for ln in a.splitlines() if ln.startswith("- ")), a)
        else: self.assertIn("Nothing has changed", a)

    def test_the_usual_range_of_a_quantity_comes_from_its_own_history(self):
        r = ask(self.db, "what is a normal battery temperature", "stat battery_temperature usual none none none none none"); a = r["answer"]; self.assertEqual(r["facts"]["route"], "digest:usual")
        m = re.search(r"usually between ([-\d.]+) and ([-\d.]+) .*median ([-\d.]+)", a); self.assertTrue(m, a); lo, hi, med = (float(x) for x in m.groups()); self.assertLess(lo, med); self.assertLess(med, hi)
        b, end, p = brain(self.db); v = [x[0] / 10.0 for x in sql(self.db, "SELECT batt FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2) AND batt IS NOT NULL", end - 28 * DAY, end)]
        v.sort(); self.assertAlmostEqual(med, v[len(v) // 2], delta=0.6)

    def test_unusual_readings_name_the_day_and_the_usual_value(self):
        r = ask(self.db, "anything unusual this month", "anomaly none none none none none none none"); a = r["answer"]; self.assertEqual(r["facts"]["route"], "digest:anomaly")
        if r["facts"]["anomalies"]: self.assertRegex(a, r"against [-\d.,]+ .*usual|first in the previous|against about"); self.assertNotIn("lkv_", a)
        else: self.assertIn("Nothing looks unusual", a)

    def test_why_was_the_longest_trip_costly_compares_it_with_the_other_trips(self):
        r = ask(self.db, "why was my longest trip so costly", "why wh_per_km none none none trip none none"); a = r["answer"]
        self.assertIn("longest trip", a); self.assertTrue("associations" in a or "Nothing about the trip stands out" in a or "does not show anything unusual" in a or "against your other" in a or "different" in a, a); self.assertNotIn("proves", a.lower())

    def test_a_false_premise_is_corrected_not_explained(self):
        r = ask(self.db, "why was the battery so hot on my longest trip", "why battery_temperature none none none trip none none"); a = r["answer"]
        self.assertTrue(re.search(r"in line with a usual trip peak|clearly higher than|clearly lower than", a), a)
        if "in line with" in a: self.assertTrue("not unusual" in a or "does not show anything unusual" in a, a)

    def test_a_why_about_a_quantity_says_where_the_quantity_stands(self):
        r = ask(self.db, "why is the battery temperature high", "why battery_temperature none none none none none none"); a = r["answer"]
        m = re.search(r"For reference: battery temperature averaged ([-\d.]+) .*? (within|above|below) its usual range \(([-\d.]+) to ([-\d.]+)", a); self.assertTrue(m, a)
        b, end, p = brain(self.db); v = [x[0] / 10.0 for x in sql(self.db, "SELECT batt FROM sig_1m WHERE ts>=? AND ts<? AND batt IS NOT NULL", end - 7 * DAY, end)]
        self.assertAlmostEqual(float(m[1]), sum(v) / len(v), delta=1.0); lo, hi = float(m[3]), float(m[4]); x = float(m[1]); self.assertEqual(m[2], "within" if lo <= x <= hi else ("above" if x > hi else "below"))

    def test_the_energy_lost_while_parked_is_the_difference_between_the_two_readings(self):
        r = ask(self.db, "why did the battery drain overnight", "why battery_soc drop parked none none none none"); a = r["answer"]; self.assertEqual(r["facts"]["route"], "digest:parked")
        m = re.search(r"parked from (.+?) to (.+?) \(", a); self.assertTrue(m, a); self.assertTrue(re.search(r"fell from ([\d.]+)% to ([\d.]+)%|did not fall", a), a)
        self.assertLess(r["facts"]["hours"], 25); self.assertGreaterEqual(r["facts"]["hours"], 0.5)
        f = re.search(r"fell from ([\d.]+)% to ([\d.]+)%", a)
        if f: self.assertAlmostEqual(r["facts"]["drop"], float(f[1]) - float(f[2]), delta=0.6)

    def test_last_night_routes_an_energy_loss_question_to_the_parked_stretch_even_without_a_condition_word(self):
        r = ask(self.db, "why did the battery go down last night", "why battery_soc drop none none none none none"); self.assertEqual(r["facts"]["route"], "digest:parked")
        r = ask(self.db, "why did my battery drop this week", "why battery_soc drop none none none none none"); self.assertNotEqual(r["facts"]["route"], "digest:parked")           # a week is not a night

    def test_a_refusal_or_a_request_for_detail_shows_which_question_it_is_about(self):
        a = ask(self.db, "unlock the doors", "control none none none none none none none")["answer"]; b = ask(self.db, "set the cruise speed to 100", "control none none none none none none none")["answer"]
        self.assertTrue(a.startswith('About "unlock the doors": ')); self.assertIn("cannot operate vehicle functions", a); self.assertNotEqual(a, b)
        c = ask(self.db, "gibberish about the thing", "ambiguous none none none none none none none")["answer"]; self.assertIn('About "gibberish about the thing"', c); self.assertIn("I am not sure which signal or period you mean", c)
        u = ask(self.db, "how do i bypass the speed limiter", "unsafe none none none none none none none")["answer"]; self.assertNotIn("bypass the speed limiter", u)               # a harmful request is not repeated back
        long = ask(self.db, "x " * 100, "ambiguous none none none none none none none")["answer"]; self.assertLess(len(long.split(":")[0]), 80)

    def test_a_charge_is_never_called_a_parked_loss(self):
        b, end, p = brain(self.db); stretches = __import__("vgpt.meaning.digest", fromlist=["x"])._parked_stretches
        from vgpt.meaning import digest as DG
        cx = DG.Ctx(b.resolver, b.resolver.onto.concepts["battery_soc"], __import__("vgpt.meaning.frame", fromlist=["Frame"]).Frame.from_text("why battery_soc drop parked none none none none"), b.resolver._lit("last week"), "")
        for a, z, x, y in stretches(cx, "sig_1m", "soc", 10.0, end - 14 * DAY, end):
            self.assertEqual(sql(self.db, "SELECT COUNT(*) FROM sig_1m WHERE ts>=? AND ts<=? AND mode IN (1,2,3)", a, z)[0][0], 0, "a stretch holds no driving or charging minute")
            self.assertLessEqual(y - x, 1.0, "the level does not rise while standing, beyond sensor drift")


@unittest.skipUnless(have("world_kestrel.db"), "needs the Kestrel world")
class PetrolCar(V2):
    db = "world_kestrel.db"

    def test_a_summary_speaks_of_fuel_not_of_a_battery(self):
        a = ask(self.db, "summarise the last week", "summary none none none none none none none")["answer"]
        self.assertIn("L/100 km", a); self.assertNotIn("Wh/km", a); self.assertNotIn("kWh", a); self.assertNotIn("traction", a.lower()); self.assertNotIn("state of charge", a.lower()); self.assertNotIn("lkv_", a)

    def test_the_usual_oil_temperature_and_the_fuel_lost_while_parked(self):
        a = ask(self.db, "what is a normal oil temperature", "stat engine_oil_temperature usual none none none none none")["answer"]; self.assertRegex(a, r"usually between"); self.assertIn("oil", a.lower())
        a = ask(self.db, "why did the fuel level drop while parked", "why fuel_level drop parked none none none none")["answer"]; self.assertTrue("did not fall" in a or "fell from" in a, a)


@unittest.skipUnless(have("world_rhino.db"), "needs the Rhino world")
class DieselTruck(V2):
    db = "world_rhino.db"

    def test_a_summary_and_the_unusual_readings_know_the_exhaust_side(self):
        a = ask(self.db, "summarise the last week", "summary none none none none none none none")["answer"]; self.assertIn("L/100 km", a); self.assertNotIn("traction", a.lower()); self.assertNotIn("state of charge", a.lower()); self.assertNotIn("kwh", a.lower())
        r = ask(self.db, "anything unusual this month", "anomaly none none none none none none none"); self.assertEqual(r["facts"]["route"], "digest:anomaly")

    def test_a_quantity_the_truck_does_not_have_is_never_mentioned(self):
        a = ask(self.db, "what changed since last week", "summary none change none none none none none")["answer"].lower(); self.assertNotIn("state of charge", a); self.assertNotIn("regen", a); self.assertNotIn("traction", a); self.assertNotIn("kwh", a)


@unittest.skipUnless(have("cs5_diesel_ice.db"), "needs the Case Study 5 diesel file")
class AlmostNothingRecorded(V2):
    db = "cs5_diesel_ice.db"

    def test_every_digest_says_what_it_has_and_invents_nothing(self):
        for q, f in (("summarise the last week", "summary none none none none none none none"), ("what changed since last week", "summary none change none none none none none"), ("anything unusual", "anomaly none none none none none none none"),
                     ("what is a normal battery temperature", "stat battery_temperature usual none none none none none"), ("why did the battery drain overnight", "why battery_soc drop parked none none none none")):
            r = ask(self.db, q, f); a = r["answer"].lower()
            for bad in ("kwh", "wh/km", "l/100", "oil temperature", "battery temperature of", "state of charge was"): self.assertNotIn(bad, a, (q, a))
            self.assertFalse(re.search(r"\b(?:fell|rose) from", a) and "speed" not in a, (q, a))


if __name__ == "__main__":
    unittest.main()

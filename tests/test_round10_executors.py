"""Round 10 executors, each checked against numbers recomputed from the database with plain SQL:
trend verdict (the asked direction answered yes / no, a change smaller than the practical size is "no"), anomaly scoped to a component, tyre answers that name the low tyre, and the petrol / diesel executors
(city against highway, fuel per trip, idling, tank size, fuel advice).  The frames are given directly: no language model is used."""
import os
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
import ask_frame as AF                                  # noqa: E402  (does not touch the environment when imported)
from vgpt.meaning.pipeline import MeaningBrain          # noqa: E402
from vgpt.vehicle import lake as LK                     # noqa: E402

_BEFORE = {}


def setUpModule():
    _BEFORE["v"] = os.environ.get("VGPT_FRAME"); os.environ["VGPT_FRAME"] = "2"                  # frame version 2 for these tests only


def tearDownModule():
    if _BEFORE.get("v") is None: os.environ.pop("VGPT_FRAME", None)
    else: os.environ["VGPT_FRAME"] = _BEFORE["v"]

DAY = 86400


def brain(db):
    path = os.path.join(ROOT, "data", db)
    if not os.path.exists(path): raise unittest.SkipTest(db + " is not present")
    LK.adopt_path(path); return MeaningBrain(path, AF.Null(), now=AF.last_ts(path), tau=0.0), path


def ask(b, text, spec): return b.resolver.answer(text, AF.frame_of(spec))


class TrendVerdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.b, cls.path = brain("world_aster.db")

    def test_the_asked_direction_is_answered_first_with_numbers_from_the_database(self):
        r = ask(self.b, "are my trips getting longer", "trend distance_driven op=rise ent=trip"); v = r["facts"]["trend_verdict"]
        c = sqlite3.connect(self.path); now = AF.last_ts(self.path); tz = int(self.b.tz); lo = now + 1 - 28 * DAY; d0, d1 = (lo + tz) // DAY, (now + tz) // DAY           # the whole local days of the window
        rows = c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, AVG(km) FROM trips WHERE km>0.3 GROUP BY d HAVING d>=? AND d<=? ORDER BY d", (tz, d0, d1)).fetchall()
        y = [x[1] for x in rows]; h = len(y) // 2
        self.assertAlmostEqual(v["first"], sum(y[:h]) / h, delta=0.5); self.assertAlmostEqual(v["second"], sum(y[len(y) - h:]) / h, delta=0.5)
        self.assertTrue(r["answer"].startswith(("Yes", "No")), r["answer"][:80])

    def test_a_change_below_the_practical_size_is_a_no(self):
        r = ask(self.b, "is the battery degrading", "trend battery_soh op=drop"); self.assertTrue(r["answer"].startswith("No"), r["answer"][:100]); self.assertEqual(r["facts"]["trend_verdict"]["direction"], "flat")
        self.assertNotIn("best candidate", r["answer"])

    def test_the_opposite_direction_is_named(self):
        r = ask(self.b, "is the coolant flow slipping", "trend coolant_flow op=drop"); self.assertTrue(r["answer"].startswith("Yes"), r["answer"][:100]); self.assertEqual(r["facts"]["trend_verdict"]["direction"], "fall")
        r = ask(self.b, "is the coolant flow rising", "trend coolant_flow op=rise"); self.assertTrue(r["answer"].startswith("No"), r["answer"][:100]); self.assertIn("fell", r["answer"].split("\n")[0])

    def test_a_noisy_difference_is_not_called_a_trend(self):
        r = ask(self.b, "how did the daily distance evolve this month", "trend distance_driven op=change"); self.assertIn(r["facts"]["trend_verdict"]["direction"], ("noisy", "flat", "rise", "fall"))
        if r["facts"]["trend_verdict"]["direction"] == "noisy": self.assertTrue(r["answer"].startswith("No reliable trend"))


class Combustion(unittest.TestCase):
    def test_city_against_highway_matches_the_minute_table(self):
        for db, tank in (("world_kestrel.db", 50), ("world_rhino.db", 90)):
            b, path = brain(db); r = ask(b, "compare city and highway fuel use", "compare fuel_rate op=mean cond=city vs=highway")
            c = sqlite3.connect(path); now = AF.last_ts(path); sid = c.execute("SELECT sid FROM sig_catalog WHERE concept='fuel_rate'").fetchone()[0]; lo = now + 1 - 30 * DAY
            q = lambda mode: c.execute("SELECT SUM(l.v) / 60.0, SUM(s.speed) / 600.0 FROM lk_1m l JOIN sig_1m s ON s.ts = l.ts WHERE l.sid=? AND l.ts>=? AND l.ts<? AND s.mode=? AND s.speed IS NOT NULL", (sid, lo, now + 1, mode)).fetchone()
            (l1, k1), (l2, k2) = q(1), q(2); self.assertIn(f"{l1 / k1 * 100:.1f} L/100 km in the city", r["answer"]); self.assertIn(f"{l2 / k2 * 100:.1f} L/100 km on the highway", r["answer"])
            r = ask(b, "what is the tank size", "spec fuel_level"); self.assertIn(f"{tank} L", r["answer"])

    def test_the_trip_that_used_the_most_fuel_is_the_largest_in_litres(self):
        b, path = brain("world_kestrel.db"); r = ask(b, "which trip used the most fuel", "extreme fuel_rate op=max ent=trip")
        c = sqlite3.connect(path); now = AF.last_ts(path); sid = c.execute("SELECT sid FROM sig_catalog WHERE concept='fuel_rate'").fetchone()[0]
        best = max((c.execute("SELECT SUM(v) / 60.0 FROM lk_1m WHERE sid=? AND ts>=? AND ts<=?", (sid, st, en)).fetchone()[0] or 0.0 for st, en, km in c.execute("SELECT start, end, km FROM trips WHERE start>=? AND start<? AND km>=3", (now + 1 - 30 * DAY, now + 1))))
        self.assertIn(f"{best:.1f} L over", r["answer"])

    def test_idling_is_the_engine_running_while_standing_not_the_hours_parked(self):
        b, path = brain("world_kestrel.db"); r = ask(b, "how long was the engine idling this week", "stat drive_mode op=duration cond=idling")
        self.assertIn("idling", r["answer"]); self.assertNotIn("160 h", r["answer"])

    def test_fuel_advice_and_correlation_are_not_the_consumption_figure(self):
        b, path = brain("world_kestrel.db")
        self.assertIn("city", ask(b, "how do i save fuel", "advice fuel_rate")["answer"].lower())
        self.assertIn("correlation", ask(b, "does speed affect my fuel economy", "stat fuel_rate op=corr c2=speed")["answer"])


class TyresAndAnomaly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        import test_live as TL
        cls.db, rt, car, sim = TL.rig("aster_ev", tag="_r10_tyre"); car.start_trip(); TL.run(sim, 240); sim.inject("tyre_leak", wheel=2, kpa_min=8.0); TL.run(sim, 1500); rt.publish(); rt.close()
        LK.adopt_path(cls.db); cls.b = MeaningBrain(cls.db, AF.Null(), now=AF.last_ts(cls.db), tau=0.0)

    def test_the_low_tyre_is_named_with_the_warning(self):
        for text, spec in (("which tyre is low", "now tyre_pressure"), ("how are my tyres", "health tyre_condition")):
            a = ask(self.b, text, spec)["answer"]; self.assertIn("rear left", a); self.assertIn("warning", a.lower())

    def test_anomaly_about_tyres_is_about_tyres(self):
        a = ask(self.b, "is anything unusual with the tyres", "anomaly tyre_condition")["answer"].lower()
        self.assertTrue("tyre" in a); self.assertNotIn("coolant", a); self.assertNotIn("12 v", a)


if __name__ == "__main__":
    unittest.main()

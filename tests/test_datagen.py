"""Data generator: determinism, physical consistency, real-data provenance, and - most important - that the assistant's answers on a generated vehicle match the generator's ground truth."""
import os, sys, tempfile, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import datasets as D

HAVE = bool(D.load_vif()) and os.path.exists(os.path.join(D.REF, "epa", "uddscol.txt"))


@unittest.skipUnless(HAVE, "needs the VIF-AEGIS zip (Downloads) and data/ref/epa")
class Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt import datagen_life as DL, datagen_events as DE, brain, store, analytics as A
        cls.DL, cls.DE, cls.brain, cls.A = DL, DE, brain, A
        cls.kw = dict(seed=11, days=14, city="graz", driver="commuter", vehicle="family", end="2026-08-20", offline=True, verbose=False, work_km=22,
                      stories=["tyre_leak", "cooling_pump", "dirty_windscreen", "roadworks", "harsh_week"])
        cls.gv = DL.build(**cls.kw); cls.path = os.path.join(tempfile.mkdtemp(), "g.db"); DE.seed(cls.path, cls.gv); cls.T = cls.gv.truth
        cls.b = brain.Brain(cls.path)                                             # rules only: no model needed

    def ask(self, q): return self.b.ask(q)

    # ---------------------------------------------------------------- generator itself
    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        g2 = self.DL.build(**self.kw); self.assertEqual([t["km"] for t in g2.trips], [t["km"] for t in self.gv.trips]); self.assertTrue(np.array_equal(g2.soc, self.gv.soc))
        g3 = self.DL.build(**dict(self.kw, seed=12)); self.assertNotEqual([t["km"] for t in g3.trips][:8], [t["km"] for t in self.gv.trips][:8])

    def test_physics_is_sane(self):
        gv = self.gv; self.assertTrue(np.isfinite(gv.soc).all() and np.isfinite(gv.batt).all())
        self.assertGreaterEqual(float(gv.soc.min()), 0.0); self.assertLessEqual(float(gv.soc.max()), 100.0001); self.assertTrue(-25 < float(gv.batt.min()) and float(gv.batt.max()) < 70)
        self.assertLessEqual(float(gv.speed.max()), self.gv.V["top"] + 1)
        self.assertTrue(np.all(np.diff(gv.odo) >= -1e-3))                                                   # odometer never goes backwards
        for tr in gv.trips[:30]:                                                                            # energy balance: SOC change x capacity == integral of power, per trip
            s = tr["s"]; L = len(tr["v"]); d_soc = (gv.soc[s] - gv.soc[s + L - 1]) / 100 * gv.life.cap; e_pow = float(gv.power[s:s + L].sum() / 3600.0)
            self.assertAlmostEqual(d_soc, e_pow, delta=max(0.15, 0.06 * abs(e_pow)))
        wh = [t["wh_km"] for t in gv.trips if t["km"] > 5]; self.assertTrue(110 < float(np.median(wh)) < 260, np.median(wh))             # real-world EV consumption range

    def test_harsh_events_come_from_real_accelerometer_data_at_a_realistic_rate(self):
        km = sum(t["km"] for t in self.gv.trips); hb = sum(t["harsh_brake"] for t in self.gv.trips)
        self.assertTrue(0.5 < 100 * hb / km < 20, 100 * hb / km)
        self.assertTrue(any(s["src"] == "vif" for s in D.segments()["city"]) and any(s["src"] == "udds" for s in D.segments()["city"] + D.segments()["hwy"]))

    def test_weather_is_used(self):
        w = self.gv.world; self.assertEqual(float(w.amb.dtype == np.float32), 1.0); self.assertTrue(((w.rain >= 0) & (w.rain <= 3)).all())               # rain intensity levels are 0..3 (offline synthetic weather may be dry, so presence of rain is not asserted)
        self.assertEqual(self.gv.world.source.split(" ")[0], "synthetic")                                   # this test runs offline

    # ---------------------------------------------------------------- assistant answers vs ground truth
    def _window_days(self, n): return self.T["daily"][-n:]

    def test_driving_counts_match_truth(self):
        r = self.ask("How many times did I brake hard in the last 7 days?"); f = r["facts"]; self.assertEqual(r["intent"], "DRIVING")
        d = self._window_days(7); self.assertEqual(f["brake"], sum(x["harsh_brake"] for x in d)); self.assertEqual(f["accel"], sum(x["harsh_accel"] for x in d)); self.assertEqual(f["trips"], sum(x["trips"] for x in d))
        self.assertAlmostEqual(f["km"], sum(x["km"] for x in d), delta=0.3)

    def test_yesterday_duration_and_places(self):
        r = self.ask("How long did I drive yesterday?"); self.assertEqual(r["intent"], "DRIVING"); y = self.T["daily"][-1]; self.assertEqual(r["facts"]["minutes"], round(y["duration_min"]))
        r = self.ask("Where did I go yesterday?"); self.assertEqual(r["intent"], "PLACES"); got = [(t["from"], t["to"]) for t in r["facts"]["trips"]]
        want = [(a if a != "stop" else "a charging stop", b if b != "stop" else "a charging stop") for a, b in y["places"]]; self.assertEqual(got, want)

    def test_rain_minutes_match_truth(self):
        r = self.ask("Was it raining last week?"); self.assertEqual(r["intent"], "WEATHER"); f = r["facts"]
        if f.get("ok"): self.assertEqual(f["rain_min"], sum(x["rain_active_min"] for x in self._window_days(7))); self.assertEqual(f["heavy_min"], sum(x["rain_heavy_active_min"] for x in self._window_days(7)))

    def test_odometer_and_service(self):
        r = self.ask("What is the odometer reading?"); self.assertEqual(r["intent"], "SERVICE"); self.assertAlmostEqual(r["facts"]["odo_km"], self.T["odo_end"], delta=1.5)
        f = self.ask("When is my next service?")["facts"]; self.assertEqual(f["km_to_service"], round(self.T["last_service_km"] + self.T["service_km"] - self.T["odo_end"]), msg=f)

    def test_tyres_report_the_leak_and_the_warning(self):
        T = self.T; r = self.ask("How are my tyres?"); f = r["facts"]; self.assertEqual(r["intent"], "TYRES"); self.assertTrue(f["ok"])
        warned = "TPMS_LOW" in T["event_ts"]; refilled = "TYRE_INFLATED" in T["event_ts"]
        if warned and not refilled: self.assertTrue(f["active_warnings"], f)
        if refilled: self.assertFalse(f["active_warnings"], f)
        for w, cold, spec in zip(f["wheels"], T["tyre_cold_end_kpa"], T["tyre_spec_kpa"]): self.assertAlmostEqual(w["kpa"], cold, delta=9, msg=(w, cold))              # temperature-compensated reading == the generator's cold pressure
        self.assertIn("tyre", r["answer"].lower())

    def test_health_finds_the_planted_cooling_fault_and_nothing_invented(self):
        f = self.ask("Is my car ok?")["facts"]; areas = {i["area"] for i in f["items"] if i["level"] == "problem"}
        flow_dtc = any("P1A00" in e for e in [x[3] for x in self.gv.events if x[1] == "DTC_SET"])
        if flow_dtc: self.assertIn("cooling", areas)
        else: self.assertNotIn("cooling", areas)
        self.assertEqual("battery" in {i["area"] for i in f["items"] if i["level"] == "problem"}, "WARN_BATT_OVERTEMP" in self.T["event_ts"])

    def test_charging_energy_matches_truth(self):
        r = self.ask("How much did I charge last week?"); f = r["facts"]; self.assertEqual(r["intent"], "EVENTS"); d = self._window_days(7)
        self.assertEqual(f["count"], sum(x["charge_sessions"] for x in d)); self.assertAlmostEqual(f.get("kwh", 0), sum(x["charge_kwh"] for x in d), delta=0.3)

    def test_local_time_zone_is_honoured(self):
        from vgpt import analytics as A
        self.assertEqual(self.b.tz, int(self.gv.tzh * 3600)); r = self.ask("How far did I drive yesterday?")
        self.assertAlmostEqual(r["facts"]["km"], self.T["daily"][-1]["km"], delta=0.3)

    def test_negative_utc_offsets_work(self):
        gv = self.DL.build(**dict(self.kw, city="phoenix", days=10, stories=[])); path = os.path.join(tempfile.mkdtemp(), "p.db"); self.DE.seed(path, gv); b = self.brain.Brain(path)
        self.assertLess(b.tz, 0)
        for q in ("Why has my range dropped?", "How far did I drive yesterday?", "What is my range?"): self.assertNotEqual(b.ask(q)["path"], "error", q)
        self.assertAlmostEqual(b.ask("How far did I drive yesterday?")["facts"]["km"], gv.truth["daily"][-1]["km"], delta=0.3)

    def test_unknown_data_degrades_gracefully_on_the_legacy_database(self):
        legacy = os.path.join(ROOT, "data", "vehicle.db")
        if not os.path.exists(legacy): self.skipTest("legacy db missing")
        b = self.brain.Brain(legacy)
        for q in ("How are my tyres?", "Where did I park?", "When is my next service?", "Was it raining last week?"):
            r = b.ask(q); self.assertNotEqual(r["path"], "error", q); self.assertRegex(r["answer"].lower(), r"couldn't|can't|cannot|\bno\b.*\b(?:recorded|readings?|record)\b|not recorded", q)   # the answer names what the legacy database lacks


if __name__ == "__main__":
    unittest.main()

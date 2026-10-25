"""Cross-layer regressions for semantic substitutions and motion arithmetic."""
import os
import unittest

import numpy as np

from vgpt import interpretation, kinematics, nl2ir
from vgpt.brain import Brain
from vgpt.query_ir import Query
from vgpt.signal_registry import Profile
from vgpt.store import open_ro


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "gen_demo.db")
NOW = 1787109669


class MotionArithmetic(unittest.TestCase):
    def test_twenty_to_thirty_in_five_seconds(self):
        runs = kinematics.band_runs([0, 5], [20, 30], 20, 30)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0].acceleration, 10 / 3.6 / 5, places=6)

    def test_irregular_samples_use_interpolated_crossing_times(self):
        runs = kinematics.band_runs([0, 2, 5, 7], [18, 22, 28, 32], 20, 30)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0].start, 1.0, places=6)
        self.assertAlmostEqual(runs[0].end, 6.0, places=6)
        self.assertAlmostEqual(runs[0].acceleration, 10 / 3.6 / 5, places=6)

    def test_gap_off_and_missing_samples_break_a_run(self):
        self.assertEqual(kinematics.band_runs([0, 20], [20, 30], 20, 30, max_gap_s=5), [])
        self.assertEqual(kinematics.band_runs([0, 5], [20, 30], 20, 30, valid=[True, False]), [])
        self.assertEqual(kinematics.band_runs([0, 5], [20, np.nan], 20, 30), [])

    def test_signed_mean_and_magnitude_are_not_conflated(self):
        signed, magnitude, elapsed = kinematics.interval_acceleration([0, 5, 10], [0, 18, 0])
        self.assertAlmostEqual(float(signed.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(magnitude.mean()), 1.0, places=6)
        self.assertEqual(elapsed.tolist(), [5.0, 5.0])


@unittest.skipUnless(os.path.exists(DB), "fixture database missing")
class TypedSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = open_ro(DB); cls.profile = Profile(cls.c)

    @classmethod
    def tearDownClass(cls): cls.c.close()

    def supported(self, text):
        parsed = nl2ir.parse(text, now=NOW, tz=19800, profile=self.profile)
        self.assertTrue(parsed.ok, (text, parsed.reason, parsed.unexplained))
        self.assertEqual(interpretation.interpret(text, parsed.query, None, now=NOW, tz=19800).issues(), [])
        return parsed.query

    def test_speed_band_is_a_first_class_operation(self):
        for text in ("acceleration between 20 to 30 kmph", "acceleration from 20 to 30 km/h"):
            q = self.supported(text)
            self.assertEqual(q.op, "acceleration_band")
            self.assertEqual(q.target, "acceleration")
            self.assertEqual([(c.signal, c.op, c.value) for c in q.scope.conditions], [("speed", "ge", 20.0), ("speed", "le", 30.0)])
            self.assertEqual(Query.from_text(q.to_text()).to_dict(), q.to_dict())

    def test_component_and_specification_concepts_remain_distinct(self):
        self.assertEqual(self.supported("how is brake fluid").target, "brake_fluid_condition")
        rated = self.supported("How fast can this car go?")
        observed = self.supported("highest speed I reached this week")
        self.assertEqual((rated.op, rated.target), ("capability", "rated_max_speed"))
        self.assertEqual((observed.op, observed.target), ("extreme", "speed"))

    def test_critical_diagnostic_compiles_to_severity_selection(self):
        q = self.supported("What are my critical diagnostics?")
        self.assertEqual(q.event.type, "DTC_SET")
        self.assertEqual(q.event.selector, "most_severe")

    def test_signed_and_absolute_acceleration_have_distinct_targets(self):
        signed = self.supported("average signed acceleration last week")
        magnitude = self.supported("average absolute acceleration last week")
        self.assertEqual(signed.target, "acceleration")
        self.assertEqual(magnitude.target, "acceleration_magnitude")


@unittest.skipUnless(os.path.exists(DB), "fixture database missing")
class EndToEndAnswers(unittest.TestCase):
    def setUp(self): self.b = Brain(DB, slm=None, slm_ir=None, policy="clarify", qc_mode="all", now=NOW)
    def tearDown(self): self.b.c.close()

    def test_acceleration_executes_without_efficiency_or_clarification(self):
        r = self.b.ask("acceleration between 20 to 30 kmph", use_slm=False)
        self.assertIn((r.get("facts") or {}).get("route"), ("qc:acceleration_band", "acceleration_bands"))      # the interpolated 1 Hz band-run executor answers it
        self.assertRegex(r["answer"].lower(), r"acceleration from 20(?:-| to )30 km/h")
        self.assertNotIn("energy consumption", r["answer"].lower())
        self.assertNotEqual(r["intent"], "CLARIFY")

    def test_brake_fluid_never_substitutes_brake_pad_life(self):
        r = self.b.ask("how is brake fluid", use_slm=False)
        self.assertIn("brake fluid condition", r["answer"].lower())
        self.assertIn("isn't recorded", r["answer"].lower())
        self.assertNotIn("brake pad", r["answer"].lower())

    def test_rated_speed_does_not_use_historical_peak(self):
        rated = self.b.ask("How fast can this car go?", use_slm=False)
        observed = self.b.ask("highest speed I reached this week", use_slm=False)
        self.assertIn("rated maximum speed", rated["answer"].lower())
        self.assertIn("isn't recorded", rated["answer"].lower())
        self.assertIn("highest vehicle speed", observed["answer"].lower())

    def test_charging_speed_uses_power_not_road_speed(self):
        r = self.b.ask("how is charging speed", use_slm=False)
        self.assertIn("charging power", r["answer"].lower())
        self.assertNotIn("vehicle speed", r["answer"].lower())
        self.assertNotIn("km/h", r["answer"].lower())

    def test_critical_diagnostics_cannot_become_trip_analytics(self):
        r = self.b.ask("What are my critical diagnostics?", use_slm=False)
        self.assertEqual((r.get("facts") or {}).get("route"), "qc:list_events")
        self.assertIn("highest-severity", r["answer"].lower())
        self.assertNotIn("longest trip", r["answer"].lower())

if __name__ == "__main__": unittest.main()

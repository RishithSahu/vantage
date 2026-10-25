"""Major semantic recovery: regression tests for defects found while triaging the full-suite failures (each has a reproducer, a positive and a negative case)."""
import os as _os; _os.environ.setdefault("VGPT_DRIVING_FIXTURE", "1")                                  # the small derived driving fixture stands in for the real traces when they are absent (tools_testdata.py)
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)


class Schema(unittest.TestCase):
    def test_every_operation_has_a_help_line(self):
        from vgpt.query_ir import OPS
        from vgpt.schema_view import OP_HELP
        self.assertEqual([o for o in OPS if o not in OP_HELP], [])


class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt import datagen_life as DL, datagen_events as DE, brain
        cls.brain = brain
        gv = DL.build(seed=11, days=21, city="graz", driver="commuter", vehicle="family", end="2026-08-20", offline=True, verbose=False, work_km=22, stories=["cooling_pump", "tyre_leak"])
        cls.path = os.path.join(tempfile.mkdtemp(), "q.db"); DE.seed(cls.path, gv)

    def ask(self, q): return self.brain.Brain(self.path).ask(q, use_slm=False)

    def route(self, r): return (r.get("facts") or {}).get("route")

    # an explicit calendar date fixes the day of an exact-instant question
    def test_explicit_date_outside_the_data_is_reported_as_no_data(self):
        r = self.ask("What was the vehicle speed on 17 March 2025 at exactly 09:13:27?")
        self.assertIn("no data for 17 Mar 2025", r["answer"]); self.assertNotIn("(20 Aug)", r["answer"])

    def test_explicit_date_inside_the_data_is_used(self):
        r = self.ask("What was the battery temperature on 15 August 2026 at exactly 09:13:27?")
        self.assertNotIn("no data for", r["answer"]); self.assertIn("15 Aug", r["answer"])

    # a statement about a world that did not happen is not a measurement query
    def test_proof_of_a_counterfactual_is_not_an_event_count(self):
        r = self.ask("Can you prove that the last fault would not have happened under a different driving style?")
        self.assertEqual(self.route(r), "cf_generic"); self.assertIn("No one can prove", r["answer"])

    def test_a_factual_event_count_while_driving_is_unchanged(self):
        r = self.ask("How many warnings were there while the car was driving in the last 7 days?")
        self.assertTrue(str(self.route(r)).startswith(("temporal_events", "qc")))

    # "how long ago" asks for the time since an event, not for the length of a state
    def test_how_long_ago_is_recency_not_duration(self):
        self.assertEqual(self.route(self.ask("How long ago was the car last plugged in?")), "last_charge")

    def test_how_long_was_charging_is_still_a_duration(self):
        self.assertEqual(self.route(self.ask("For how long was the car charging in the last 7 days?")), "temporal_duration")

    # a share question with no comparable level must not crash the temporal operation
    def test_proportion_of_drives_does_not_raise(self):
        import logging
        with self.assertNoLogs("vehiclegpt", level="WARNING"):
            self.ask("what proportion of drives had at least one alarm")

    def test_share_of_driving_time_above_a_speed_still_works(self):
        r = self.ask("What percentage of driving time was spent above 80 km/h?")
        self.assertEqual(self.route(r), "temporal_share"); self.assertRegex(r["answer"], r"\d+\.\d% of the driving time")


if __name__ == "__main__":
    unittest.main()

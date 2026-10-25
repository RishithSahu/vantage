"""Regression tests for the six Round-52 semantic_ops behaviours that were rebuilt from the compiled artefact (results/round52/semantic_ops.cpython-311.pyc.bak)
and verified identical to it over a 186-question corpus (results/semantic_recovery/overnight/compare_semantic_ops.py)."""
import os, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ.setdefault("VGPT_FRONTEND", "legacy")
DB = os.path.join(HERE, "data", "gen_demo.db")
NOW = 1787625900


@unittest.skipUnless(os.path.exists(DB), "needs data/gen_demo.db")
class Round52SemanticOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt.product import make_legacy_brain
        cls.make = staticmethod(lambda: make_legacy_brain(DB, slm=None, now=NOW))

    def ask(self, q, brain=None):
        from vgpt import nlu, semantic_ops
        b = brain or self.make(); t = nlu.normalize(q, repair_typos=True)[0]
        return semantic_ops.handle(b, q, t)

    # ---- freshness
    def test_freshness_current_sample_is_called_current(self):
        r = self.ask("Is the speed reading stale?")
        self.assertEqual(r["facts"]["route"], "freshness"); self.assertIn("reading is current:", r["answer"])

    def test_freshness_old_sample_is_not_called_current(self):
        from vgpt import semantic_ops
        orig = semantic_ops._fresh_snapshot
        def stale(brain):
            s = orig(brain); s = dict(s); s["age_s"] = 600; return s
        semantic_ops._fresh_snapshot = stale
        try: r = self.ask("Is the speed reading stale?")
        finally: semantic_ops._fresh_snapshot = orig
        self.assertIn("not current (the newest valid sample is 600 s old)", r["answer"])
        self.assertNotIn("reading is current:", r["answer"])

    def test_age_question_about_power_is_a_freshness_question(self):
        r = self.ask("How old is the battery power reading?")
        self.assertEqual(r["facts"]["route"], "freshness")

    # ---- brake fluid
    def test_brake_fluid_is_named_and_never_replaced_by_pad_life(self):
        r = self.ask("Is my brake fluid okay?")
        self.assertEqual(r["facts"]["route"], "component_unavailable")
        self.assertIn("Brake fluid condition isn't recorded", r["answer"]); self.assertIn("not a substitute", r["answer"])

    # ---- stated efficiency
    def test_stated_efficiency_is_arithmetic_on_the_given_figure(self):
        r = self.ask("How many kWh for 300 km at 160 Wh/km?")
        self.assertEqual(r["facts"]["route"], "distance_to_energy")
        self.assertIn("300 x 160 / 1000 = 48.0 kWh", r["answer"]); self.assertTrue(r["facts"]["stated"])

    def test_stated_efficiency_beats_the_recorded_one(self):
        r = self.ask("What energy would 120 km need at an efficiency of 140 Wh per km?")
        self.assertIn("16.8 kWh", r["answer"]); self.assertIn("140", r["answer"])

    def test_without_a_stated_efficiency_the_recorded_one_is_used(self):
        r = self.ask("How much energy does a 250 km trip need at my efficiency?")
        self.assertEqual(r["facts"]["route"], "distance_to_energy"); self.assertNotIn("stated", r["facts"])

    # ---- warning context
    def test_safety_context_is_the_latest_problem_not_any_event(self):
        r = self.ask("Can I safely drive 200 km with this warning")
        self.assertEqual(r["facts"]["route"], "safety_assessment")
        self.assertIn("The most recent recorded problem is", r["answer"]); self.assertNotIn("CHARGE_START", r["answer"])

    def test_advice_style_safety_question_is_not_an_assessment(self):
        r = self.ask("What are some tips to drive safely?")
        self.assertTrue(r is None or r["facts"]["route"] != "safety_assessment")

    # ---- rated vs recorded maximum speed
    def test_rated_speed_is_not_substituted_but_the_record_is_shown_as_observation(self):
        r = self.ask("What is the vehicle's rated maximum speed?")
        self.assertEqual(r["facts"]["route"], "capability"); self.assertFalse(r["facts"]["available"])
        self.assertIn("I will not substitute the highest speed observed in telemetry.", r["answer"])
        self.assertIn("(an observation, not a rating)", r["answer"])

    # ---- relation / multi-task guards
    def test_a_coordination_is_not_a_two_concept_relation(self):
        r = self.ask("Is state of charge the same as state of health and battery temperature?")
        self.assertTrue(r is None or r["facts"]["route"] != "semantic_relation")

    def test_a_plain_difference_question_is_still_a_relation(self):
        r = self.ask("What is the difference between state of charge and state of health?")
        self.assertEqual(r["facts"]["route"], "semantic_relation")

    def test_multi_task_report_request_is_not_one_route(self):
        self.assertIsNone(self.ask("Find, reconstruct, identify and explain every anomaly and list the faults"))

    # ---- the typed engine's comparison is reported as the typed path
    def test_city_highway_comparison_is_tagged_as_the_typed_engine(self):
        r = self.ask("Compare energy consumption between city and highway driving.")
        self.assertEqual(r["facts"]["route"], "city_highway_energy"); self.assertEqual(r["facts"].get("engine"), "qc")

    def test_a_significance_test_request_is_not_tagged_as_typed(self):
        r = self.ask("Compare energy consumption between city and highway driving with a significance test.")
        self.assertTrue(r is None or r["facts"].get("engine") != "qc")

    # ---- state-duration comparisons
    def test_driving_and_charging_duration_are_report_items_not_an_intersection(self):
        r = self.ask("How much time did I spend driving versus charging yesterday?")
        self.assertEqual(r["facts"]["route"], "state_duration")
        self.assertIn("driving duration was", r["answer"])
        self.assertIn("charging duration was", r["answer"])
        self.assertIn("separate state durations", r["answer"])
        self.assertNotIn("driving and charging for", r["answer"])


if __name__ == "__main__": unittest.main()

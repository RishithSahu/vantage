"""End-to-end regressions for broad everyday driver semantic families.

The cases intentionally use paraphrase groups rather than one accepted
sentence.  They protect the ontology/speech-act composition that sits between
the 0.5B semantic classifier and the typed executor.
"""
import os
import unittest

from vgpt import nl2ir, nlu
from vgpt.brain import Brain
from vgpt.product import make_legacy_brain
from vgpt.signal_registry import Profile
from vgpt.store import open_ro


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "gen_demo.db")
NOW = 1787625900


@unittest.skipUnless(os.path.exists(DB), "needs data/gen_demo.db")
class DriverSemanticFamilies(unittest.TestCase):
    def setUp(self):
        self.b = make_legacy_brain(DB, slm=None, now=NOW)

    def tearDown(self):
        self.b.c.close()

    def ask(self, text):
        return self.b.ask(text, use_slm=False)

    def assert_route(self, text, route):
        r = self.ask(text)
        self.assertEqual((r.get("facts") or {}).get("route"), route, (text, r["answer"]))
        return r

    def test_capacity_is_profile_semantics_not_vehicle_health(self):
        for text in ("how many ppl can fit in my car", "how many passengers can this vehicle carry", "what is the seating capacity"):
            r = self.assert_route(text, "profile_capability")
            self.assertNotIn("vehicle health", r["answer"].lower())

    def test_range_capability_is_not_speed_capability(self):
        for text in ("how much can we travel", "how far can this car drive", "what distance can we go"):
            r = self.assert_route(text, "remaining_range")
            self.assertIn("range", r["answer"].lower())
            self.assertNotIn("maximum speed", r["answer"].lower())

    def test_coarse_model_frame_cannot_preempt_exact_semantics(self):
        class WrongFrame:
            def __init__(self): self.calls = 0
            def parse(self, _text):
                self.calls += 1
                return "capability", {"task": "capability", "metric": "speed", "operator": "maximum", "event": None}, {"pmax": 0.99}

        old = os.environ.get("VGPT_SEMANTIC_FRONTDOOR")
        os.environ["VGPT_SEMANTIC_FRONTDOOR"] = "1"
        model = WrongFrame()
        b = Brain(DB, slm=model, slm_ir=None, policy="clarify", qc_mode="all", now=NOW)
        try:
            r = b.ask("how much can we travel", use_slm=True)
            self.assertEqual((r.get("facts") or {}).get("route"), "remaining_range")
            self.assertEqual(model.calls, 0)
        finally:
            b.c.close()
            if old is None: os.environ.pop("VGPT_SEMANTIC_FRONTDOOR", None)
            else: os.environ["VGPT_SEMANTIC_FRONTDOOR"] = old

    def test_rated_speed_is_not_current_or_observed_speed(self):
        r = self.assert_route("how fast can this vehicle go", "profile_capability")
        self.assertIn("rated maximum speed", r["answer"].lower())
        self.assertIn("not a rating", r["answer"].lower())

    def test_lifetime_distance_uses_odometer_and_respects_miles(self):
        r = self.assert_route("how many miles have we gone", "odometer")
        self.assertIn("miles", r["answer"].lower())
        self.assertIn("odometer", r["answer"].lower())

    def test_service_paraphrases_share_one_operation(self):
        for text in ("when to service", "when to get servicing", "when should I go for servicing", "when to service our car"):
            self.assert_route(text, "service_due")

    def test_schema_absence_does_not_substitute_a_neighbouring_measure(self):
        r = self.assert_route("how is wheel alignment", "unrecorded_measure")
        self.assertIn("does not include wheel alignment", r["answer"].lower())
        self.assertNotIn("kpa", r["answer"].lower())

    def test_domain_typo_repair_uses_joint_component_and_state(self):
        text, fixes = nlu.normalize("did betry get hoy")
        self.assertEqual(text, "did battery get hot")
        self.assertEqual(dict(fixes), {"betry": "battery", "hoy": "hot"})
        self.assert_route("did betry get hoy", "thermal_history")
        # A valid question word must never be repaired into the state "hot".
        self.assertEqual(nlu.normalize("how is break pad")[0], "how is brake pad")

    def test_battery_event_speech_acts_are_distinct(self):
        self.assert_route("why did the battery overheat", "why_event")
        self.assert_route("when did the battery overheat", "thermal_event")
        self.assert_route("were there any overheating issues", "thermal_history")

    def test_bare_why_uses_typed_battery_context(self):
        self.ask("how is battery")
        self.assert_route("why", "why_event")

    def test_engine_is_not_silently_replaced_by_battery(self):
        r = self.assert_route("why did the engine overheat", "powertrain_distinction")
        self.assertIn("no combustion engine", r["answer"].lower())

    def test_plural_tyres_preserve_quantifier_and_followup(self):
        for text in ("tell all tyres", "show every tire", "how are tyres", "tell tyres"):
            r = self.assert_route(text, "tyre_inventory")
            for wheel in ("front left", "front right", "rear left", "rear right"):
                self.assertIn(wheel, r["answer"].lower())
        self.ask("tell all tyres")
        self.assert_route("all of them", "tyre_inventory")
        self.assert_route("list all of them", "tyre_inventory")

    def test_last_drive_is_entity_selection_not_unknown_speed_since(self):
        r = self.assert_route("what is avg speed since last drive", "last_trip_speed")
        self.assertIn("most recent completed trip", r["answer"].lower())

    def test_external_vehicle_comparison_is_not_driving_score(self):
        r = self.assert_route("is my car better than the Tesla Roadster", "external_comparison")
        self.assertNotIn("driving score", r["answer"].lower())
        self.assertIn("needs a criterion", r["answer"].lower())

    def test_compact_duration_parses_and_dialogue_window_edits(self):
        c = open_ro(DB)
        try:
            p = Profile(c)
            x = nl2ir.parse("average speed in the last 2hrs", now=NOW, tz=19800, profile=p)
            self.assertTrue(x.ok, (x.reason, x.unexplained))
            self.assertEqual(x.query.scope.time.to_text(), "last:2:hour")
        finally:
            c.close()
        self.ask("average vehicle speed over the last 7 days")
        r = self.ask("in the last 2hrs")
        self.assertEqual(r["intent"], "QC_AGGREGATE")
        self.assertIn("last 2 hours", r["answer"].lower())

    def test_care_and_whole_health_are_not_single_signal_queries(self):
        self.assert_route("how can I take care of my car", "maintenance_advice")
        r = self.assert_route("what are the issues in my vehicle", "vehicle_health")
        self.assertIn("recorded events and signals", r["answer"].lower())

    def test_battery_dimensions_do_not_collapse_into_battery_health(self):
        self.assert_route("how much is the battery charge now", "current_soc")
        self.assert_route("I want to know the charge left in the battery", "current_soc")
        self.assert_route("what is our battery capacity", "battery_capacity")
        r = self.assert_route("how many mAh is our battery", "battery_capacity")
        self.assertIn("without the pack's verified nominal voltage", r["answer"])
        r = self.assert_route("total number of charging cycles", "charge_cycles")
        self.assertIn("sessions are not equivalent", r["answer"])

    def test_capability_and_power_quantities_are_not_substituted(self):
        self.assert_route("can I charge my laptop using the car", "export_power_capability")
        r = self.assert_route("what is our motor power", "motor_power")
        self.assertNotIn("motor is at", r["answer"].lower())
        r = self.assert_route("how to calculate motor power", "motor_power_definition")
        self.assertIn("9549", r["answer"])

    def test_single_acceleration_band_is_a_complete_query(self):
        r = self.assert_route("acceleration between speeds 20 to 30 km per hour", "acceleration_bands")
        self.assertIn("20-30 km/h", r["answer"])
        r = self.assert_route("acceleration going 10 to 20 m per second", "acceleration_bands")
        self.assertIn("36-72 km/h", r["answer"])

    def test_calendar_points_and_ranges_keep_their_scope(self):
        r = self.assert_route("what was my RPM yesterday at 10:00 night", "signal_point")
        self.assertIn("24 Aug, 22:00", r["answer"])
        self.assertIn("derived", r["answer"])
        r = self.assert_route("where was I at 10:00 yesterday", "location_point")
        self.assertIn("GPS", r["answer"])
        r = self.assert_route("average speed from 3 days ago to 5 days ago", "calendar_range")
        self.assertIn("20 Aug", r["answer"])

    def test_driving_assessment_and_adas_have_human_semantics(self):
        self.assert_route("do I drive properly", "driving_assessment")
        self.assert_route("ADAS", "adas_definition")
        r = self.assert_route("driver assistant state zero mean", "adas_state_definition")
        self.assertIn("off or inactive", r["answer"])


if __name__ == "__main__":
    unittest.main()

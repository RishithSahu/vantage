"""The interpretation contract (v2): provenance, support and coverage. Cases are chosen to show the MECHANISM (compositional units, comparators, referents, negation, defaults, operand order), not individual sentences."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import interpretation as I
from vgpt.query_ir import Query

NOW = 1790406000


def interp(text, ir, prev=None, source="model"): return I.interpret(text, Query.from_text(ir), Query.from_text(prev) if prev else None, source=source, now=NOW, tz=3600)
def origin(it, name): return [f.origin for f in it.of(name)]


class Support(unittest.TestCase):
    def test_named_fields_come_from_the_question(self):
        it = interp("what was my highest battery temperature in the last 3 days", "op=extreme tgt=battery_temperature agg=max t=last:3:day")
        self.assertEqual(origin(it, "target"), ["question"]); self.assertEqual(origin(it, "time"), ["question"]); self.assertEqual(origin(it, "statistic"), ["question"]); self.assertEqual(it.issues(), []); self.assertEqual(it.disclosures(), [])

    def test_a_default_period_is_disclosed_with_the_actual_value_per_operation(self):
        it = interp("average battery temperature", "op=aggregate tgt=battery_temperature agg=mean t=last:7:day"); self.assertEqual(origin(it, "time"), ["default"]); self.assertIn("No period was given, so I used the last 7 days.", it.disclosures())
        it = interp("is the coolant flow going down", "op=trend tgt=coolant_flow t=last:14:day"); self.assertIn("No period was given, so I used the last 14 days.", it.disclosures())
        it = interp("when did the coolant flow change", "op=detect_change tgt=coolant_flow t=all"); self.assertIn("No period was given, so I used the whole recorded history.", it.disclosures())

    def test_a_period_that_is_not_the_documented_default_for_the_operation_is_not_accepted_as_a_default(self):
        it = interp("how does speed relate to consumption", "op=correlation tgt=speed tgt2=wh_per_km t=last:30:day"); self.assertEqual(origin(it, "time"), ["ungrounded"]); self.assertTrue(it.issues())

    def test_a_vague_word_justifies_only_the_documented_window_of_the_operation(self):
        it = interp("why has my range dropped lately", "op=explain_change tgt=range_estimate t=last:14:day"); self.assertEqual(origin(it, "time"), ["vague"]); self.assertIn("'lately' was read as the last 14 days.", it.disclosures())
        it = interp("why has my range dropped lately", "op=explain_change tgt=range_estimate t=last:90:day"); self.assertEqual(origin(it, "time"), ["ungrounded"])

    def test_a_stated_period_that_the_query_replaced_is_rejected_and_reported(self):
        it = interp("average battery temperature in the last 3 days", "op=aggregate tgt=battery_temperature agg=mean t=last:30:day"); self.assertEqual(origin(it, "time"), ["ungrounded"]); self.assertTrue(any("last 3 days" in u for u in it.unresolved))

    def test_last_week_is_recorded_as_an_ambiguity_and_disclosed(self):
        it = interp("how many times did i brake hard last week", "op=count t=last:7:day ev=HARSH_BRAKE"); self.assertTrue(any("previous calendar week" in a for a in it.ambiguities)); self.assertTrue(any("last 7 days" in d for d in it.disclosures()))

    def test_an_operation_needs_wording_that_supports_it(self):
        self.assertIn(origin(interp("average speed last week", "op=aggregate tgt=speed agg=mean t=last:7:day"), "operation")[0], ("default", "question"))      # a plain lookup needs no cue word of its own
        self.assertEqual(origin(interp("how many hard brakes last week", "op=count t=last:7:day ev=HARSH_BRAKE"), "operation"), ["question"])
        self.assertEqual(origin(interp("what is the average speed", "op=explain_change tgt=speed t=last:14:day"), "operation"), ["ungrounded"])

    def test_a_substituted_signal_or_condition_is_ungrounded(self):
        self.assertTrue(interp("what was the peak steering torque yesterday", "op=extreme tgt=motor_torque agg=max t=cal:yesterday").issues())
        self.assertFalse(interp("what was the peak motor torque yesterday", "op=extreme tgt=motor_torque agg=max t=cal:yesterday").issues())
        it = interp("what was my fastest trip last month", "op=aggregate tgt=speed agg=max t=last:30:day c=fast_charging"); self.assertIn("ungrounded", origin(it, "condition"))


class Thresholds(unittest.TestCase):
    """Number, unit, comparator and referent are recognised separately, so equivalent expressions and several thresholds need no per-sentence handler."""
    def test_equivalent_unit_expressions_support_the_same_condition(self):
        for text in ("share of time above 60 mph this month", "share of time over 60 miles per hour this month", "share of time faster than 60 mph this month"):
            it = interp(text, "op=percent_time t=cal:this_month c=sig:speed:gt:60:mph"); self.assertEqual(origin(it, "condition"), ["cue"], text); self.assertEqual(it.issues(), [], text)
        it = interp("share of time above 100 km/h this month", "op=percent_time t=cal:this_month c=sig:speed:gt:100:km_per_h"); self.assertEqual(it.issues(), [])

    def test_the_same_number_with_the_wrong_comparator_is_rejected(self):
        it = interp("share of time above 60 mph this month", "op=percent_time t=cal:this_month c=sig:speed:lt:60:mph"); self.assertEqual(origin(it, "condition"), ["ungrounded"])

    def test_a_temperature_unit_gives_a_dimension_not_a_sensor(self):
        it = interp("how often was it colder than 5 degrees last month", "op=percent_time t=last:30:day c=sig:ambient_temperature:lt:5"); self.assertEqual(origin(it, "condition"), ["ambiguous"])
        self.assertTrue(it.clarifications() and "temperature" in it.clarifications()[0]["question"])
        it = interp("how often was it colder than 5 degrees outside last month", "op=percent_time t=last:30:day c=sig:ambient_temperature:lt:5"); self.assertEqual(origin(it, "condition"), ["question"])
        it = interp("how often was the pack colder than 5 degrees last month", "op=percent_time t=last:30:day c=sig:battery_temperature:lt:5"); self.assertEqual(origin(it, "condition"), ["question"])
        it = interp("how often was the pack colder than 5 degrees last month", "op=percent_time t=last:30:day c=sig:ambient_temperature:lt:5"); self.assertEqual(origin(it, "condition"), ["ungrounded"])         # the referent is the pack, not the air

    def test_fahrenheit_is_converted_and_checked(self):
        it = interp("peak speed while it was colder than 32 fahrenheit outside", "op=extreme tgt=speed agg=max t=last:30:day c=sig:ambient_temperature:lt:32:F"); self.assertEqual(origin(it, "condition"), ["question"])
        it = interp("peak speed while it was colder than 32 fahrenheit outside", "op=extreme tgt=speed agg=max t=last:30:day c=sig:ambient_temperature:lt:32"); self.assertEqual(origin(it, "condition"), ["question"])        # 32 read as the signal's own unit is a different threshold (disclosure is the contract's other half)

    def test_every_stated_threshold_must_be_represented(self):
        it = interp("average speed while charging power was over 20 kw and the pack above 40 degrees", "op=aggregate tgt=speed agg=mean t=last:7:day c=sig:charging_power:gt:20:kW")
        self.assertTrue(any("40" in u for u in it.unresolved), it.unresolved)

    def test_two_thresholds_are_matched_one_to_one(self):
        ir = "op=percent_time t=last:7:day c=sig:speed:gt:60:mph,sig:battery_soc:lt:20:%"
        it = interp("share of the week above 60 mph and below 20 percent charge", ir); self.assertEqual(it.issues(), [], it.issues())


class Coverage(unittest.TestCase):
    def test_an_omitted_named_condition_is_reported(self):
        it = interp("average speed on the highway last week", "op=aggregate tgt=speed agg=mean t=last:7:day"); self.assertTrue(any("highway" in u for u in it.unresolved))

    def test_a_negated_condition_must_not_be_applied(self):
        it = interp("average speed without the rain restriction", "op=aggregate tgt=speed agg=mean t=last:7:day c=raining", "op=aggregate tgt=speed agg=mean t=last:7:day c=raining"); self.assertIn("ungrounded", origin(it, "condition"))
        it = interp("average speed without the rain restriction", "op=aggregate tgt=speed agg=mean t=last:7:day", "op=aggregate tgt=speed agg=mean t=last:7:day c=raining"); self.assertEqual(it.unresolved, [])

    def test_both_periods_of_a_comparison_are_checked_and_ordered(self):
        ok = interp("average speed today versus yesterday", "op=compare tgt=speed agg=mean t=cal:today t2=cal:yesterday"); self.assertEqual(ok.issues(), [])
        sw = interp("average speed today versus yesterday", "op=compare tgt=speed agg=mean t=cal:yesterday t2=cal:today"); self.assertTrue(sw.unresolved)
        one = interp("average speed today versus yesterday", "op=compare tgt=speed agg=mean t=cal:today"); self.assertTrue(any("yesterday" in u for u in one.unresolved))

    def test_group_order_is_checked(self):
        self.assertEqual(interp("is it cheaper in the city than on the highway", "op=compare tgt=wh_per_km agg=mean t=last:7:day grp=city,highway").unresolved, [])
        self.assertTrue(interp("is it cheaper in the city than on the highway", "op=compare tgt=wh_per_km agg=mean t=last:7:day grp=highway,city").unresolved)

    def test_a_specific_sensor_must_stay_specific(self):
        self.assertEqual(interp("what is the tyre pressure on the front left", "op=current_value tgt=tyre_pressure_fl t=now").issues(), [])
        self.assertTrue(interp("tyre pressure now", "op=current_value tgt=tyre_pressure_fl t=now").issues())                       # a position nobody stated
        self.assertTrue(any("front left" in u for u in interp("what is the tyre pressure on the front left", "op=current_value tgt=tyre_pressure t=now").unresolved))   # the stated position was dropped

    def test_ranking_order_and_metric_follow_the_wording(self):
        self.assertEqual(interp("my three longest trips", "op=rank t=all sel=trip:distance_driven:desc:3:-").issues(), [])
        self.assertTrue(interp("my three shortest trips", "op=rank t=all sel=trip:distance_driven:desc:3:-").issues())
        self.assertTrue(interp("my three longest trips", "op=rank t=all sel=trip:energy_used:desc:3:-").issues())


class Conversation(unittest.TestCase):
    prev = "op=aggregate tgt=speed agg=max t=cal:yesterday c=highway"

    def test_a_continuation_inherits_only_what_it_leaves_unsaid(self):
        it = interp("and for the motor", "op=extreme tgt=motor_temperature agg=max t=cal:yesterday c=highway", self.prev); self.assertEqual(origin(it, "target"), ["prior_turn"]); self.assertEqual(origin(it, "time"), ["prior_turn"])

    def test_a_short_utterance_alone_does_not_inherit_context(self):
        it = interp("battery health", "op=current_value tgt=battery_soh t=now", self.prev); self.assertNotIn("prior_turn", origin(it, "target") + origin(it, "time"))
        it = interp("what is the tyre pressure", "op=aggregate tgt=tyre_pressure agg=max t=cal:yesterday", self.prev); self.assertEqual(origin(it, "target"), ["question"]); self.assertNotEqual(origin(it, "time"), ["prior_turn"])

    def test_without_a_previous_turn_nothing_can_be_inherited(self):
        it = interp("and for tomorrow", "op=aggregate tgt=speed agg=max t=last:7:day"); self.assertTrue(it.ungrounded() or origin(it, "time") != ["prior_turn"])

    def test_then_is_ambiguous_when_two_antecedents_are_plausible(self):
        prev = "op=count t=cal:yesterday ev=HARSH_BRAKE"; it = interp("what was the temperature then", "op=extreme tgt=battery_temperature agg=max t=cal:yesterday", prev)
        self.assertEqual(origin(it, "reference"), ["ambiguous"]); self.assertEqual(len(it.of("reference")[0].alternatives), 2)

    def test_energy_use_is_read_as_consumption_and_says_so(self):
        it = interp("what changed in my energy usage lately", "op=explain_change tgt=wh_per_km t=last:14:day")
        self.assertTrue(any("energy per kilometre" in a for a in it.ambiguities) or origin(it, "target") == ["cue"])


if __name__ == "__main__": unittest.main()

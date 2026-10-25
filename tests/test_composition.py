"""Generic compositional steps of the rule parser: wheel position, comparatives with units, follow-up revisions, superlatives, existence questions, describe-by-parts."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import nl2ir
from vgpt.query_ir import Query

NOW = 1790406000


def parse(text, prev=None):
    st = nl2ir.ConvState()
    if prev: st.last = Query.from_text(prev)
    p = nl2ir.parse(text, now=NOW, tz=3600, state=st); return p.query.to_text().split(" ref=")[0] if p.ok else None


class Composition(unittest.TestCase):
    def test_a_wheel_position_keeps_the_specific_sensor(self):
        self.assertEqual(parse("what is the tyre pressure on the front left"), "op=current_value tgt=tyre_pressure_fl t=now")
        self.assertEqual(parse("and the front right", "op=aggregate tgt=tyre_pressure_rr agg=mean t=last:7:day"), "op=aggregate tgt=tyre_pressure_fr agg=mean t=last:7:day")

    def test_a_comparative_with_a_unit_is_a_condition_on_the_signal_of_that_unit(self):
        self.assertIn("c=sig:speed:gt:60:mph", parse("average speed while going faster than 60 mph last week"))

    def test_a_follow_up_removes_or_replaces_a_restriction_and_keeps_the_rest(self):
        prev = "op=aggregate tgt=speed agg=mean t=cal:this_week c=raining"
        self.assertEqual(parse("without the rain restriction", prev), "op=aggregate tgt=speed agg=mean t=cal:this_week")
        self.assertEqual(parse("actually city only", "op=aggregate tgt=wh_per_km agg=mean t=cal:this_month c=highway"), "op=aggregate tgt=wh_per_km agg=mean t=cal:this_month c=city")

    def test_the_period_before_shifts_a_calendar_period_and_keeps_the_condition(self):
        self.assertEqual(parse("and for the day before", "op=extreme tgt=speed agg=max t=cal:yesterday c=highway"), "op=extreme tgt=speed agg=max t=cal:day_before_yesterday c=highway")

    def test_superlatives_and_how_hot_did_it_get_are_maxima(self):
        self.assertEqual(parse("how fast was the car going at its fastest yesterday"), "op=extreme tgt=speed agg=max t=cal:yesterday")
        self.assertEqual(parse("how hot did the pack get this month"), "op=extreme tgt=battery_temperature agg=max t=cal:this_month")

    def test_existence_questions_are_counts(self):
        self.assertEqual(parse("did the camera get blocked at all last week"), "op=count t=cal:last_week ev=CAMERA_BLOCKED")

    def test_describe_by_parts_needs_every_word_and_a_unique_candidate(self):
        self.assertEqual(parse("how many times did i brake hard last month"), "op=count t=cal:last_month ev=HARSH_BRAKE")
        self.assertIsNone(parse("how many times did the thing happen last month"))                 # nothing is described: refused, not guessed

    def test_a_substituted_signal_is_not_invented(self):
        q = parse("what was the peak steering torque yesterday"); self.assertTrue(q is None or "tgt=steering_torque" in q)


if __name__ == "__main__": unittest.main()

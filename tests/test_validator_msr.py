"""Validator false rejections found by running the runtime grounding check over the training labels (msr/label_vs_validator.py).
Each repair has a positive case (a correct plan is accepted) and a negative case (a plan that invents or contradicts the argument is still rejected)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import learned_front as LF
from vgpt.query_ir import Query


class _Q: last = None


class _B: qcs = _Q()


def check(plan, text, context=None):
    return LF.LearnedFront(_B(), None).check(Query.from_text(plan), text, context=context)


class Numbers(unittest.TestCase):
    def test_negative_threshold_is_grounded(self):
        self.assertEqual(check("op=exists tgt=ambient_temperature t=last:7:day p.thr=lt:-5 unit=C", "was the outside air below -5 degrees in the last week"), [])
        self.assertEqual(check("op=exists tgt=ambient_temperature t=last:7:day p.thr=lt:-5 unit=C", "did it get below minus five degrees outside last week"), [])

    def test_wrong_negative_threshold_is_rejected(self):
        self.assertTrue(check("op=exists tgt=ambient_temperature t=last:7:day p.thr=lt:-7 unit=C", "was the outside air below -5 degrees in the last week"))

    def test_singular_which_day_needs_no_number(self):
        plan = "op=by_period tgt=battery_temperature agg=max t=last:7:day p.by=day p.k=1 p.ord=desc"
        self.assertEqual(check(plan, "which day last week had the highest pack temperature"), [])

    def test_a_count_of_days_still_needs_a_number(self):
        self.assertTrue(check("op=by_period tgt=battery_temperature agg=max t=last:7:day p.by=day p.k=3 p.ord=desc", "which day last week had the highest pack temperature"))

    def test_ordinary_words_are_not_number_words(self):
        self.assertNotIn(8.0, LF._numbers_in("Right, only excluding the time it was charging"))
        ctx = "op=exists tgt=battery_temperature t=all p.thr=le:40 unit=C"
        self.assertEqual(check("op=exists tgt=battery_temperature t=all c=not:charging p.thr=le:40 unit=C", "Right, only excluding the time it was charging", context=ctx), [])

    def test_misspelt_number_word_is_still_a_number(self):
        self.assertIn(80.0, LF._numbers_in("how many days was the motor temp at least eighyt degrees"))


class Windows(unittest.TestCase):
    def test_hyphenated_number_words(self):
        self.assertIn(29 * 3600, LF._question_windows("across the trailing twenty-nine hours")[0])
        self.assertEqual(check("op=aggregate tgt=speed agg=mean t=last:29:hour", "average speed across the trailing twenty-nine hours"), [])
        self.assertTrue(check("op=aggregate tgt=speed agg=mean t=last:31:hour", "average speed across the trailing twenty-nine hours"))

    def test_hour_and_minutes_compound(self):
        self.assertEqual(check("op=aggregate tgt=speed agg=max t=last:91:minute", "what was the top speed across the last hour and thirty-one minutes"), [])
        self.assertTrue(check("op=aggregate tgt=speed agg=max t=last:45:minute", "what was the top speed across the last hour and thirty-one minutes"))

    def test_a_quarter_of_an_hour(self):
        self.assertIn(900, LF._question_windows("for a quarter of an hour")[0]); self.assertIn(1800, LF._question_windows("for half an hour")[0])

    def test_fortnight_names_a_count_of_14(self):
        self.assertEqual(check("op=topic tgt=battery_temperature t=last:14:day p.n=14 p.topic=daily_series", "pack temperature for every day of the last fortnight"), [])

    def test_equal_length_comparison_window(self):
        q = "Compare total distance on the day before yesterday with the immediately preceding window of equal length"
        self.assertEqual(check("op=window_change tgt=distance_driven agg=sum t=cal:day_before_yesterday t2=prev:1:day", q), [])
        self.assertTrue(check("op=window_change tgt=distance_driven agg=sum t=cal:day_before_yesterday t2=prev:3:day", q))

    def test_number_naming_the_plans_own_window_is_not_an_ignored_number(self):
        ctx = "op=duration tgt=speed t=last:7:day c=highway p.thr=gt:100 unit=km_per_h"
        self.assertEqual(check("op=duration tgt=speed t=last:14:day c=highway p.thr=gt:100 unit=km_per_h", "over two weeks instead", context=ctx), [])
        self.assertTrue(check("op=duration tgt=speed t=last:7:day c=highway p.thr=gt:100 unit=km_per_h", "make the threshold 120 instead", context=ctx))


if __name__ == "__main__":
    unittest.main()

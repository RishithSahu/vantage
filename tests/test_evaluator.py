"""The evaluator is tested against deliberately wrong predictions: a scoring function that accepts them would inflate every number in the project.
Uses the real evaluation databases (a missing one is an error, not a skip: the experiment must not shrink silently)."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from bench import evalcore as E
from tests import testdata
from bench.eval_v2 import DBS

_ENG = None
def eng():
    global _ENG
    if _ENG is None: _ENG = E.Engines(DBS)
    return _ENG


@testdata.needs("legacy_worlds")
class Positive(unittest.TestCase):
    def test_an_identical_query_is_correct_and_plan_exact(self):
        r = E.score(eng(), "op=aggregate tgt=speed agg=mean t=last:7:day", "op=aggregate tgt=speed agg=mean t=last:7:day"); self.assertTrue(r["correct"]); self.assertTrue(r["plan_exact"]); self.assertEqual(r["kind"], "answer")

    def test_refusals(self):
        self.assertTrue(E.score(eng(), "op=none", "op=none")["correct"])
        r = E.score(eng(), "op=none", "op=aggregate tgt=speed agg=mean t=last:7:day"); self.assertTrue(r["false_action"]); self.assertFalse(r["correct"])
        r = E.score(eng(), "op=aggregate tgt=speed agg=mean t=last:7:day", "op=none"); self.assertTrue(r["false_refusal"]); self.assertFalse(r["correct"])

    def test_answers_and_explained_non_answers_are_different_outcomes(self):
        a = E.score(eng(), "op=aggregate tgt=speed agg=mean t=last:7:day", "op=aggregate tgt=speed agg=mean t=last:7:day"); self.assertEqual(a["kind"], "answer")
        n = E.score(eng(), "op=aggregate tgt=fuel_rate agg=mean t=last:7:day", "op=aggregate tgt=fuel_rate agg=mean t=last:7:day"); self.assertEqual(n["kind"], "explained"); self.assertEqual(n["gold_kind"], "explained")


@testdata.needs("legacy_worlds")
class Rejects(unittest.TestCase):
    """Each case: a prediction that a lenient scorer might accept."""
    def assertWrong(self, gold, pred):
        r = E.score(eng(), gold, pred); self.assertFalse(r["correct"], (gold, pred)); return r

    def test_correct_value_wrong_period(self):
        r = self.assertWrong("op=aggregate tgt=battery_soc agg=max t=last:30:day", "op=aggregate tgt=battery_soc agg=max t=last:60:day")      # the maximum of a full range is the same number over a longer window on a short history
        self.assertIn("period", r["fields"])

    def test_correct_count_wrong_event_type(self):
        gold, pred = "op=count t=last:14:day ev=HARSH_BRAKE", "op=count t=last:14:day ev=CHARGE_START"
        r = self.assertWrong(gold, pred); self.assertIn("event", r["fields"])
        sg, sp = eng().run(gold), eng().run(pred)
        forged = [dict(a, value=b.get("value")) for a, b in zip(sg, sp)]                      # even if the counts were identical on every database, the event identity differs
        self.assertFalse(all(E.same(x, y) for x, y in zip(forged, sp)) and all(x.get("asked") == y.get("asked") for x, y in zip(forged, sp)))

    def test_correct_number_wrong_unit(self):
        a = {"status": "answered", "value": 22.0, "unit": "C", "items": [], "n": 0}; b = dict(a, unit="F")
        self.assertTrue(E.same(a, dict(a))); self.assertFalse(E.same(a, b))

    def test_correct_first_page_wrong_later_items(self):
        items = [[1000 + i, "HARSH_BRAKE", None] for i in range(30)]; a = {"status": "answered", "value": 30.0, "items": items, "n": 30, "total": 30.0}
        b = dict(a, items=items[:25] + [[9999, "HARSH_BRAKE", None]] + items[26:]); c = dict(a, items=items[:25], n=25)
        self.assertFalse(E.same(a, b)); self.assertFalse(E.same(a, c))                          # a changed later item and a truncated page are both differences

    def test_missing_versus_numeric_is_a_difference(self):
        a = {"status": "answered", "value": None, "items": [["x", None, None]], "n": 1}; b = {"status": "answered", "value": None, "items": [["x", None, 5.0]], "n": 1}
        self.assertFalse(E.same(a, b)); self.assertFalse(E.same(dict(a, value=3.0), a)); self.assertTrue(E.same(a, dict(a)))

    def test_correct_unsupported_status_wrong_requested_capability(self):
        gold, pred = "op=aggregate tgt=fuel_rate agg=mean t=last:7:day", "op=aggregate tgt=engine_rpm agg=mean t=last:7:day"
        sg, sp = eng().run(gold), eng().run(pred); r = self.assertWrong(gold, pred)
        self.assertIn("signal", r["fields"])
        if [s["status"] for s in sg] == [s["status"] for s in sp]: self.assertFalse(r["correct"])           # same non-answer status for the wrong signal must still be wrong

    def test_omitted_highway_condition(self):
        r = self.assertWrong("op=aggregate tgt=speed agg=mean t=last:7:day c=highway", "op=aggregate tgt=speed agg=mean t=last:7:day"); self.assertIn("conditions", r["fields"])

    def test_swapped_comparison_operands(self):
        gold = "op=compare tgt=wh_per_km agg=mean t=last:7:day grp=city,highway"; r = self.assertWrong(gold, "op=compare tgt=wh_per_km agg=mean t=last:7:day grp=highway,city")
        self.assertIn("group/selection", r["fields"])
        a, b = E.parts(E.Query.from_text(gold)), E.parts(E.Query.from_text("op=compare tgt=wh_per_km agg=mean t=last:7:day grp=highway,city")); self.assertNotEqual(a, b)

    def test_swapped_period_operands_in_a_comparison(self):
        self.assertWrong("op=compare tgt=speed agg=mean t=cal:today t2=cal:yesterday", "op=compare tgt=speed agg=mean t=cal:yesterday t2=cal:today")

    def test_explanation_factor_values_matter(self):
        f = {"before": 200.0, "after": 230.0, "windows": [1, 2], "residual": 5.0, "gap": 30.0, "factors": [["cold weather", 15.0], ["speed", 10.0]]}
        a = {"status": "answered", "explain": f}; self.assertTrue(E.same(a, {"status": "answered", "explain": dict(f)}))
        self.assertFalse(E.same(a, {"status": "answered", "explain": dict(f, factors=[["cold weather", 5.0], ["speed", 10.0]])}))
        self.assertFalse(E.same(a, {"status": "answered", "explain": dict(f, residual=20.0)}))


class Fixtures(unittest.TestCase):
    def test_a_missing_database_fails_immediately(self):
        with self.assertRaises(E.FixtureError): E.Engines([("no_such_database", None)])


if __name__ == "__main__": unittest.main()

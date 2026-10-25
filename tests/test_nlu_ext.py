"""The tester's own chat transcript + extended-domain phrasings must be understood by the RULES alone (no model needed) and typos must not turn ordinary words into assistant vocabulary."""
import json, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import brain, nlu

Q = json.load(open(os.path.join(ROOT, "bench", "questions_ext.json"), encoding="utf-8"))


class Routing(unittest.TestCase):
    def test_every_non_refusal_question_routes_to_the_expected_intent_with_fast_confidence(self):
        for q in Q:
            if q["intent"] in ("NONE", "HELP", "GLOSSARY", "UNITS", "FOLLOWUP", "ABOUT", "FEEDBACK"): continue
            t = nlu.normalize(q["q"])[0]; intent, p, conf = brain.parse_clean(t)
            self.assertEqual(intent, q["intent"], q["q"]); self.assertGreaterEqual(conf, brain.FAST_MIN, q["q"])

    def test_new_rules_do_not_steal_older_questions(self):
        for q, want in (("Is my highway efficiency worse than in the city?", "COMPARE"), ("When did fuel efficiency show the largest drop?", "LARGEST"), ("How much charge do I have left?", "STATUS"),
                        ("Something is wrong - I barely get any distance out of a full battery, why?", "WHY_RANGE")):
            intent, p, conf = brain.parse_clean(nlu.normalize(q)[0]); self.assertEqual(intent, want, q); self.assertGreaterEqual(conf, brain.FAST_MIN, q)

    def test_refusals_stay_refusals(self):
        for q in Q:
            if q["intent"] != "NONE": continue
            t = nlu.normalize(q["q"])[0]
            self.assertTrue(nlu.unsupported(t) or not brain.in_domain_clean(t) or brain.parse_clean(t)[2] < brain.FAST_MIN, q["q"])

    def test_ordinary_words_are_never_repaired_into_vocabulary(self):
        for w in ("raining", "working", "seeing", "recently", "cities", "wheels"):
            self.assertEqual(nlu.repair(w), (w, False), w)
        self.assertEqual(nlu.normalize("is it raining")[0], "is it raining")                      # was once turned into "remaining"
        self.assertEqual(nlu.normalize("which camera is not working")[0], "which camera is not working")      # was once turned into "weeking"

    def test_real_typos_are_still_repaired(self):
        for typo, want in (("recentky", "recently"), ("maintainance", "maintenance"), ("statur", "status"), ("temprature", "temperature"), ("tyers", "tyres")):
            self.assertIn(nlu.normalize(typo)[0], (want, "tire"), typo)                                  # "tyres" is a synonym of "tire" in the assistant


if __name__ == "__main__":
    unittest.main()

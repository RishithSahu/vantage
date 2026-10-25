"""Everyday driver questions: each must get a real answer (no 'did you mean' / out-of-domain gate) and the answers to the common ones must contain the right facts, with a short spoken form."""
import os, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import datasets as D

HAVE = bool(D.load_vif()) and os.path.exists(os.path.join(D.REF, "epa", "uddscol.txt"))


@unittest.skipUnless(HAVE, "needs the VIF-AEGIS zip (Downloads) and data/ref/epa")
class Everyday(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt import datagen_life as DL, datagen_events as DE, brain
        cls.brain = brain; gv = DL.build(seed=41, days=28, city="london", driver="delivery", vehicle="city", end="2026-09-25", offline=True, verbose=False, stories=["cooling_pump", "tyre_leak"])
        cls.path = os.path.join(tempfile.mkdtemp(), "ev.db"); DE.seed(cls.path, gv)

    def test_every_everyday_question_is_answered(self):
        b = self.brain.Brain(self.path); qs = [l.strip() for l in open(os.path.join(ROOT, "bench", "everyday_questions.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]; bad = []
        for q in qs:
            r = b.ask(q, use_slm=False)
            if r["path"] in ("clarify", "gate") or (r["brief"].startswith("I couldn't") and "current state is unknown" not in r["brief"]): bad.append((q, r["path"], r["brief"][:80]))   # stale camera evidence is stated, not guessed
            self.assertLessEqual(len(r["brief"]), 360, q); self.assertTrue(r["speech"], q)
        self.assertEqual(bad, [])

    def test_the_common_answers_say_the_right_thing(self):
        b = self.brain.Brain(self.path)
        cases = [("how's my battery", ["Battery", "%", "km"]), ("when did I last charge", ["last charge", "kWh"]), ("how much did I spend on charging", ["don't know your electricity price", "kWh"]), ("how much did I spend on charging at 0.30 per kWh", ["at 0.3 per kWh"]),
                 ("am I driving too fast", ["speed limits"]), ("how can I improve my range", ["Cruise at 100", "Brake"]), ("was the battery ever hot", ["pack"]), ("is the coolant pump ok", ["coolant flow"]), ("which day was most efficient", ["most efficient day"]),
                 ("how long to charge to 100%", ["kWh"]), ("is it safe to keep driving", ["problem"]), ("what is my driving score", ["out of 100"]), ("what does that code mean", ["fault", "It was set"]), ("how old is the battery", ["original capacity"]),
                 ("give me a summary of the week", ["trips", "Wh/km"]), ("can I make it 150 km", ["range"]), ("do I need to charge", ["%"]), ("explain the last warning simply", ["worth having it checked"]), ("how many trips did I make", ["trip"]), ("what was my longest trip", ["longest trip"])]
        for q, want in cases:
            r = b.ask(q, use_slm=False); a = r["answer"] + " " + r["brief"]
            for w in want: self.assertIn(w, a, q)
        self.assertEqual(b.qa.errors, [])

    def test_the_pasted_transcript_is_answered_properly(self):
        b = self.brain.Brain(self.path)
        for q in ("is it advised to go on a long trip", "is it advised for me to go out in this condition of the car", "I asked if it's safe for me to go out or not"):
            r = b.ask(q, use_slm=False); self.assertEqual(r["path"], "qa", q); self.assertEqual(r["intent"], "HEALTH", q); self.assertLessEqual(len(r["brief"]), 170, q);
        self.assertIn("negative", b.ask("why is my battery power negative", use_slm=False)["answer"])
        self.assertIn("Safe-driving tips", b.ask("how do I drive this car safely", use_slm=False)["answer"])
        self.assertEqual(b.ask("what is a question about this vehicle", use_slm=False)["intent"], "HELP")
        r = b.ask("bhai kya bol rahe ho bhai", use_slm=False); self.assertEqual(r["intent"], "FOREIGN"); self.assertIn("English", r["answer"])
        r = b.ask("gaadi ka kya haal hai", use_slm=False); self.assertIn("->", r["understood"] or "")
        for q, w in (("hello", "Hello"), ("thanks", "welcome"), ("good job", "Glad")): self.assertIn(w, b.ask(q, use_slm=False)["answer"], q)

    def test_questions_that_are_not_about_the_car_are_not_answered_as_if_they_were(self):
        b = self.brain.Brain(self.path)
        for q in ("tell me a joke", "who won the cricket match", "what is the capital of France", "how do I cook pasta", "recommend a good movie", "what time is it in Tokyo", "how do I lose weight", "write me a poem", "my phone is not charging"):
            r = b.ask(q, use_slm=False); self.assertIn(r["path"], ("gate", "unsupported", "rules", "clarify"), q); self.assertNotEqual(r["intent"], "HEALTH", q)

    def test_unclear_questions_get_related_suggestions_for_the_ui(self):
        b = self.brain.Brain(self.path); r = b.ask("the thing is doing the other thing again", use_slm=False)
        self.assertIn(r["path"], ("gate", "clarify")); self.assertTrue(r["suggest"] and len(r["suggest"]) == 3)


if __name__ == "__main__":
    unittest.main()

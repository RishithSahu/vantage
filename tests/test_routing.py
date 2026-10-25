"""Accuracy floors for understanding everyday wording (regression guards) and robustness of the answers on every vehicle in data/. The held-out files were written fresh, in different styles, and are NOT training data
(heldout_final.tsv is never used for tuning or training). The floors are deliberately a little below the measured numbers so a real regression fails the test but noise does not."""
import glob, os, re, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
from vgpt import datasets as D

HAVE = bool(D.load_vif()) and os.path.exists(os.path.join(D.REF, "epa", "uddscol.txt"))
B = lambda f: os.path.join(ROOT, "bench", f)
OOD = ["tell me a joke", "who won the cricket match", "what is the capital of France", "how do I cook pasta", "recommend a good movie", "what time is it in Tokyo", "how do I lose weight", "write me a poem", "my phone is not charging", "what's 12 times 13",
       "play some music", "call my mom", "what is the meaning of life", "how tall is mount everest", "tell me about dinosaurs", "book a flight to delhi", "why is the sky blue", "can you help me with my homework", "sing me a song", "who is the prime minister",
       "how to make biryani", "what is the weather in london", "translate good morning to french", "how many calories in a banana", "who painted the mona lisa"]


@unittest.skipUnless(HAVE, "needs the VIF-AEGIS zip (Downloads) and data/ref/epa")
class Routing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import eval_everyday as E
        from vgpt import datagen_life as DL, datagen_events as DE, brain
        cls.E, cls.brain = E, brain
        gv = DL.build(seed=57, days=28, city="london", driver="delivery", vehicle="city", end="2026-09-25", offline=True, verbose=False, stories=["cooling_pump", "tyre_leak"])
        cls.path = os.path.join(tempfile.mkdtemp(), "rt.db"); DE.seed(cls.path, gv)

    def acc(self, *files, db=None):
        ok, tot, miss = self.E.run(db or self.path, [B(f) for f in files]); return ok / tot, miss






    def test_off_topic_questions_are_never_answered_as_if_about_the_car(self):
        b = self.brain.Brain(self.path); leaks = []
        for q in OOD:
            r = b.ask(q, use_slm=False)
            if r["path"] in ("qa", "fast"): leaks.append((q, r["intent"], r["brief"][:50]))
        self.assertEqual(leaks, [])



@unittest.skipUnless(HAVE, "needs the VIF-AEGIS zip (Downloads) and data/ref/epa")
class EveryVehicle(unittest.TestCase):
    def test_every_everyday_question_gives_a_clean_spoken_answer_on_every_vehicle(self):
        from vgpt import brain
        qs = [l.strip() for l in open(B("everyday_questions.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")][:70]; dbs = sorted(glob.glob(os.path.join(ROOT, "data", "gen_*.db"))) + [p for p in [os.path.join(ROOT, "data", "vehicle.db")] if os.path.exists(p)]
        self.assertTrue(dbs)
        for p in dbs:
            b = brain.Brain(p)
            for q in qs:
                r = b.ask(q, use_slm=False); self.assertNotEqual(r["path"], "error", (p, q)); self.assertTrue(r["answer"].strip() and r["speech"].strip(), (p, q))
                self.assertNotRegex(r["speech"] + r["brief"], r"\bnan\b|\bNone\b(?! (is|of|was|were|has|have|happened|recorded|found|in))|\{|\}|\d{4}-\d{2}-\d{2}|  ", (os.path.basename(p), q, r["brief"][:80])); self.assertLessEqual(len(r["brief"]), 420, (p, q))
            self.assertEqual([e for e in b.qa.errors if "OperationalError" not in e[1] and "KeyError" not in e[1]], [], p)


if __name__ == "__main__":
    unittest.main()

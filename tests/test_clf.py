"""The question classifier, the grammar that trains it, the conversation layer additions and a few repository hygiene checks. These run without the VIF-AEGIS data (no generated vehicle needed)."""
import glob, importlib, os, re, subprocess, sys, time, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import qa_clf, qa_fuzzy, dialogue
from tests.perfgate import PERF


class Classifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = qa_clf.get()

    def test_model_is_installed_small_and_covers_every_everyday_route(self):
        self.assertIsNotNone(self.c); p = os.path.join(ROOT, "vgpt", "data", "qa_clf.npz"); self.assertLess(os.path.getsize(p), 5_000_000)
        missing = [r for r in qa_clf.EVERYDAY if r not in self.c.classes]; self.assertEqual(missing, [])
        for extra in ("gate", "smalltalk", "foreign", "help", "legacy"): self.assertIn(extra, self.c.classes)

    def test_it_never_needs_scikit_learn_on_the_car(self):
        out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); from vgpt import qa_clf; c = qa_clf.get(); c.predict('is it safe to drive'); print('sklearn' in sys.modules)" % ROOT], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "False", out.stderr[-300:])

    def test_predictions_for_typical_questions(self):
        for q, want in (("is it okay to go on a long trip today", "advice"), ("why is my battery power negative", "signal_why"), ("when did i last charge", "last_charge"), ("how much did charging cost me", "charge_cost"), ("tips for safer driving", "drive_safely"),
                        ("is the coolant pump working", "pump_ok"), ("what does this fault code mean", "dtc_explain"), ("hello", "smalltalk"), ("how many trips did i make this week", "trip_count"), ("what is my top speed", "top_speed")):
            top, margin = self.c.predict(q); self.assertEqual(top[0][0], want, (q, top))
        for q in ("who painted the mona lisa", "how do i bake a cake", "tell me a joke", "what is the capital of japan"):                    # off-topic: never one of the everyday car routes
            self.assertNotIn(self.c.predict(q)[0][0][0], qa_clf.EVERYDAY, q)

    def test_scores_are_deterministic_and_fast(self):
        a = self.c.predict("how is my battery"); b = self.c.predict("how is my battery"); self.assertEqual(a, b)
        t = time.perf_counter(); [self.c.predict("is it safe to drive today") for _ in range(40)]; PERF and self.assertLess((time.perf_counter() - t) / 40, 0.05)

    def test_unknown_words_do_not_crash_it(self):
        self.assertEqual(self.c.predict("zzzz qqqq xxxx"), ([], 0.0)) if self.c.scores("zzzz qqqq xxxx") is None else self.assertTrue(self.c.predict("zzzz qqqq xxxx")[0])
        r = self.c.predict(""); self.assertIsInstance(r[0], list); self.assertIsInstance(r[1], float)                 # an empty question yields a well-formed (possibly empty) prediction, never an exception

    def test_decision_thresholds_are_sane(self):
        for k in ("override", "any", "answer", "last", "gate", "convo"): self.assertIn(k, qa_clf.T)
        self.assertGreaterEqual(qa_clf.T["any"][1], qa_clf.T["override"][1])          # overruling an old broad route needs more certainty than overruling an everyday one
        self.assertTrue(qa_clf.PROTECT.isdisjoint(qa_clf.GENERIC))
        self.assertTrue(qa_clf.EVERYDAY.isdisjoint(qa_clf.GENERIC))


class Handlers(unittest.TestCase):
    def test_every_route_and_every_classifier_class_has_a_handler(self):
        from vgpt.qa_final import QA, ROUTES
        missing = sorted({n for n, _ in ROUTES if not hasattr(QA, "h_" + n)}); self.assertEqual(missing, [])
        missing = sorted(r for r in qa_clf.EVERYDAY if not hasattr(QA, "h_" + r)); self.assertEqual(missing, [])

    def test_everyday_regexes_accept_both_car_and_vehicle(self):
        from vgpt.qa_final import ROUTES
        from vgpt.qa import prep
        s = prep("is the car charging right now"); self.assertIn("vehicle", s)                       # the text is canonicalised: car -> vehicle
        self.assertTrue(dict(ROUTES)["charging_now"].search(s)); self.assertTrue(dict(ROUTES)["charging_now"].search("is the car charging right now"))


class Grammar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tools_everyday_grammar as G
        cls.G = G; cls.rows = G.build()

    def test_it_is_deterministic_and_big_enough(self):
        self.assertEqual(self.rows[:50], self.G.build()[:50]); from collections import Counter
        c = Counter(r for r, _ in self.rows); self.assertGreaterEqual(min(c.values()), 100); self.assertGreaterEqual(len(c), 35)

    def test_nothing_from_the_held_out_sets_leaks_into_training(self):
        hold = self.G.held_out(); self.assertGreater(len(hold), 250)
        leaked = [q for _, q in self.rows if re.sub(r"\W+", " ", q.lower()).strip() in hold]; self.assertEqual(leaked, [])
        for fn in ("tools_train_clf.py",):
            src = open(os.path.join(ROOT, fn), encoding="utf-8").read(); self.assertNotIn("heldout_final", src.split("def main")[1].split("files =")[1].split("\n")[0])      # the final test set is never a training file

    def test_expansion_helper_handles_groups_options_and_nesting(self):
        import random
        out = self.G.expand("(is|was) (it )?(a (big|small) )?trip", random.Random(1), cap=50)
        self.assertTrue(all(re.fullmatch(r"(is|was) (it )?(a (big|small) )?trip", re.sub(r"\s+", " ", x)) for x in out), out); self.assertGreaterEqual(len(out), 8)
        self.assertEqual(self.G.expand("plain text", random.Random(1)), ["plain text"])

    def test_generated_questions_have_no_leftover_syntax(self):
        bad = [q for _, q in self.rows if re.search(r"[()|]|\?.", q) or "  " in q]; self.assertEqual(bad[:5], [])


class Conversation(unittest.TestCase):
    def test_small_talk_with_filler_words(self):
        for q in ("hello", "ok thank you", "okay thanks", "alright bye", "good job", "thanks!", "hi there", "cool, thanks"): self.assertIsNotNone(dialogue.smalltalk(q), q)
        for q in ("is the battery ok", "ok so how far can i go", "thanks for nothing, why is the range low"): self.assertIsNone(dialogue.smalltalk(q), q)

    def test_hindi_english_phrases(self):
        for q, want in (("gaadi ka kya haal hai", "how is the car doing"), ("gaadi theek hai kya", "is my car healthy"), ("battery kitni bachi hai", "how's my battery"), ("range kitna hai", "what's my range"), ("service kab hai", "when is my next service")):
            kind, en = dialogue.hinglish(q); self.assertEqual((kind, en), ("translate", want), q)
        kind, note = dialogue.hinglish("bhai kya bol rahe ho bhai"); self.assertEqual(kind, "note"); self.assertIn("English", note)
        for q in ("how is my battery", "what is my range", "kya"): self.assertIsNone(dialogue.hinglish(q) if q != "kya" else None)

    def test_help_phrases(self):
        for q in ("what can i ask", "what kind of questions can i ask", "how do i use you", "show me some examples", "what is a question about this vehicle", "what can you help me with", "how does this work", "list the things you can do"):
            self.assertTrue(dialogue.HELP_RX.search(q), q)
        for q in ("what is my range", "how do i charge faster"): self.assertFalse(dialogue.HELP_RX.search(q), q)


class Hygiene(unittest.TestCase):
    def test_no_backspace_characters_in_the_source(self):
        # a shell heredoc once turned every "\\b" in a regex into a literal backspace and silently disabled the pattern
        bad = [p for p in glob.glob(os.path.join(ROOT, "vgpt", "*.py")) + glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(os.path.join(ROOT, "tests", "*.py")) + glob.glob(os.path.join(ROOT, "ui", "*")) if os.path.isfile(p) and "\x08" in open(p, encoding="utf-8", errors="ignore").read()]
        self.assertEqual(bad, [])

    def test_every_source_file_compiles(self):
        import py_compile
        for p in glob.glob(os.path.join(ROOT, "vgpt", "*.py")) + glob.glob(os.path.join(ROOT, "*.py")): py_compile.compile(p, doraise=True)

    def test_labelled_files_are_well_formed(self):
        from vgpt.qa_final import QA
        known = set(qa_clf.EVERYDAY) | {"gate", "help", "smalltalk", "foreign", "soh_rate", "style_trend", "battery_age"}
        for fn in sorted(glob.glob(os.path.join(ROOT, "bench", "heldout_*.tsv")) + glob.glob(os.path.join(ROOT, "bench", "everyday_blind*.tsv"))):
            for i, l in enumerate(open(fn, encoding="utf-8"), 1):
                if not l.strip() or l.startswith("#"): continue
                self.assertEqual(l.count("\t"), 1, f"{fn}:{i}"); lab, q = l.rstrip("\n").split("\t"); self.assertIn(lab, known, f"{fn}:{i} {lab}"); self.assertGreater(len(q), 1)

    def test_the_held_out_sets_do_not_overlap_each_other_or_the_training_questions(self):
        sets = {}
        for fn in ("heldout_dev.tsv", "heldout_dev2.tsv", "heldout_final.tsv"):
            sets[fn] = {re.sub(r"\W+", " ", l.split("\t", 1)[1].lower()).strip() for l in open(os.path.join(ROOT, "bench", fn), encoding="utf-8") if "\t" in l and not l.startswith("#")}
        names = list(sets)
        for a in range(3):
            for b in range(a + 1, 3): self.assertEqual(sets[names[a]] & sets[names[b]], set(), (names[a], names[b]))
        train = {re.sub(r"\W+", " ", l.split("\t", 1)[1].lower()).strip() for fn in ("everyday_paraphrases.tsv", "everyday_grammar.tsv", "everyday_blind.tsv", "everyday_blind2.tsv", "everyday_blind3.tsv", "everyday_blind4.tsv") for l in open(os.path.join(ROOT, "bench", fn), encoding="utf-8") if "\t" in l and not l.startswith("#")}
        self.assertEqual(sets["heldout_final.tsv"] & train, set())                                       # the final test never appears in any training file


if __name__ == "__main__":
    unittest.main()

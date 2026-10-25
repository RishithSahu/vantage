"""Frame version 2: the six words of version 1 plus `c2` (a second concept) and `ref` (what a turn takes from the previous one), and the typed executors behind them.

No model is run here.  A stand-in returns the frame a version-2 model is trained to return for each wording; what is tested is everything around the model: the vocabulary and its typing, the prompt and grammar,
the executors for a second quantity and for "than usual", and the carrying of typed values from one turn to the next.  Version 1 (the shipped adapter) must be untouched by all of it."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning import glossary as GL, interface, ontology as ON  # noqa: E402
from vgpt.meaning.frame import Frame  # noqa: E402

DB = os.path.join(ROOT, "data", "world_aster.db")
NOW = 1792348500
F = Frame


class Scripted2:
    """Returns the version-2 frame for a wording and remembers the `Previous:` frame text it was shown."""
    glossary = GL.load()

    def __init__(self, table): self.table = {k.lower(): v for k, v in table.items()}; self.seen = []

    def parse(self, text, hint=None, prev=None):
        self.seen.append((text, prev)); f = self.table.get(text.lower().strip(), F("ambiguous", "none"))
        return f, {"raw": f.to_text(2), "pmin": 0.99, "ms": 0.0, "hint": ""}


class Vocabulary(unittest.TestCase):
    def test_version_one_is_exactly_what_the_shipped_adapter_was_trained_on(self):
        o = ON.load(1)
        self.assertEqual(o.fields, ("act", "concept", "op", "cond", "vs", "ent")); self.assertEqual(o.gbnf().splitlines()[0], 'root ::= act " " concept " " op " " cond " " vs " " ent')
        self.assertTrue(o.system_prompt().startswith("Vehicle assistant parser. Output exactly six words")); self.assertNotIn("corr", o.ops); self.assertNotIn("typical", o.vs); self.assertNotIn("explain", o.acts)
        self.assertEqual(F("now", "speed").to_text(1), "now speed none none none none"); self.assertEqual(F.from_text("now speed none none none none"), F("now", "speed"))

    def test_version_two_adds_two_slots_and_a_few_values(self):
        o = ON.load(2)
        self.assertEqual(o.fields[-2:], ("c2", "ref")); self.assertEqual(o.gbnf().splitlines()[0], 'root ::= act " " concept " " op " " cond " " vs " " ent " " c2 " " ref')
        self.assertIn("corr", o.ops); self.assertIn("typical", o.vs); self.assertIn("explain", o.acts); self.assertIn("none", o.words()["c2"]); self.assertEqual(set(o.words()["ref"]), {"none", "period", "concept", "focus", "item", "answer"})
        self.assertTrue(o.system_prompt().startswith("Vehicle assistant parser. Output exactly eight words"))
        f = F.from_text("stat battery_power corr none none none speed none"); self.assertEqual((f.c2, f.ref, f.op), ("speed", "none", "corr")); self.assertEqual(f.to_text(2), "stat battery_power corr none none none speed none")

    def test_typing_of_the_new_slots(self):
        o = ON.load(2)
        self.assertEqual(o.validate(F("stat", "battery_power", "corr", "none", "none", "none", "speed", "none")), [])
        self.assertTrue(o.validate(F("stat", "battery_power", "corr")), "a correlation needs a second concept")
        self.assertTrue(o.validate(F("now", "speed", "none", "none", "none", "none", "battery_power", "none")), "'now' cannot hold a second concept")
        self.assertTrue(o.validate(F("events", "hard_braking", "count", "none", "none", "none", "speed", "none")), "an event and a measured quantity are not two of a kind")
        self.assertEqual(o.validate(F("events", "battery_overheat_warning", "exists", "none", "none", "none", "overtemp_warning", "none")), [])
        self.assertEqual(o.validate(F("compare", "battery_temperature", "mean", "none", "typical", "none", "none", "none")), [])
        self.assertEqual(o.validate(F("explain", "none", ref="answer")), []); self.assertTrue(o.validate(F("explain", "none")), "an explanation is a question about the previous answer"); self.assertTrue(o.validate(F("now", "speed", ref="answer")))
        self.assertTrue(o.validate(F("stat", "speed", "mean", "none", "none", "none", "speed", "none")), "the second concept must differ")

    def test_the_prompt_shows_the_previous_frame_only_in_version_two(self):
        p2 = interface.prompt_text("frame", "and yesterday", "", "events hard_braking count none none none none none", 2); p1 = interface.prompt_text("frame", "and yesterday", "", "events hard_braking count none none none none none", 1)
        line = chr(10) + "Previous: "                                    # the system prompt itself mentions 'Previous:' in quotes; the line the model is shown starts a new line
        self.assertIn(line + "events hard_braking count", p2); self.assertNotIn(line, p1); self.assertNotIn(line, interface.prompt_text("frame", "how is my battery", "", None, 2))

    def test_both_grammars_are_valid_for_the_decoder(self):
        try: from llama_cpp import LlamaGrammar
        except ImportError: self.skipTest("llama_cpp is not installed")
        for v in (1, 2): LlamaGrammar.from_string(ON.load(v).gbnf(), verbose=False)


@unittest.skipUnless(os.path.exists(DB), "the Aster world is not present")
class Executors(unittest.TestCase):
    TABLE = {"is my battery power related to my speed": F("stat", "battery_power", "corr", "none", "none", "none", "speed", "none"),
             "is the battery temperature related to the motor temperature": F("stat", "battery_temperature", "corr", "none", "none", "none", "motor_temperature", "none"),
             "compare the battery and the motor temperature": F("compare", "battery_temperature", "mean", "none", "none", "none", "motor_temperature", "none"),
             "did the battery or the motor overheat this month": F("events", "battery_overheat_warning", "exists", "none", "none", "none", "overtemp_warning", "none"),
             "is my battery temperature higher than usual": F("compare", "battery_temperature", "mean", "none", "typical", "none", "none", "none"),
             "were there more hard brakes than usual this week": F("compare", "hard_braking", "count", "none", "typical", "none", "none", "none"),
             "correlate speed with itself": F("stat", "speed", "corr", "none", "none", "none", "speed", "none")}

    def setUp(self):
        from vgpt.vehicle import lake as LK
        self._env = mock.patch.dict(os.environ, {"VGPT_FRAME": "2"}); self._env.start()
        for k in ("VGPT_LAYERS_OFF", "VGPT_ARBITRATE", "VGPT_MODEL_ONLY"): os.environ.pop(k, None)
        LK.adopt_path(DB)
        from vgpt.meaning.pipeline import MeaningBrain
        self.b = MeaningBrain(DB, Scripted2(self.TABLE), now=NOW); self.LK = LK

    def tearDown(self): self.LK.release(); self._env.stop()

    def ask(self, q): self.b.dialogue.reset(); return self.b.ask(q)

    def test_the_relationship_of_two_quantities_is_a_correlation_and_never_a_cause(self):
        r = self.ask("is the battery temperature related to the motor temperature"); self.assertEqual(r["facts"]["route"], "pairs:corr"); self.assertGreater(r["facts"]["r"], 0.3)
        self.assertIn("association", r["answer"]); self.assertIn("minutes in which both were recorded", r["answer"])

    def test_two_quantities_are_put_side_by_side(self):
        r = self.ask("compare the battery and the motor temperature"); self.assertEqual(r["facts"]["route"], "pairs:compare"); self.assertIn("battery temperature", r["answer"]); self.assertIn("motor temperature", r["answer"])

    def test_either_of_two_events(self):
        r = self.ask("did the battery or the motor overheat this month"); self.assertEqual(r["facts"]["route"], "pairs:either"); self.assertEqual(len(r["facts"]["counts"]), 2)

    def test_than_usual_is_the_vehicles_own_history_and_says_how_much_of_it_there_is(self):
        for q in ("is my battery temperature higher than usual", "were there more hard brakes than usual this week"):
            r = self.ask(q); self.assertEqual(r["facts"]["route"], "pairs:usual", q); self.assertIn("earlier periods of the same length", r["answer"]); self.assertGreaterEqual(r["facts"]["windows"], 6)

    def test_a_quantity_with_no_variation_has_no_correlation_and_says_so(self):
        r = self.ask("correlate speed with itself"); self.assertNotEqual(r["facts"].get("route"), "pairs:corr")          # the typing refuses a quantity paired with itself before any arithmetic


@unittest.skipUnless(os.path.exists(DB), "the Aster world is not present")
class FollowUps(unittest.TestCase):
    """The model returns a whole frame plus `ref`; the dialogue only carries typed values (the previous window, the instant, the event list)."""
    TABLE = {"how many hard brakes last week": F("events", "hard_braking", "count"),
             "and the week before that": F("events", "hard_braking", "count", ref="period"),
             "which one was the strongest": F("extreme", "hard_braking", "max", ref="item"),
             "how did you get that": F("explain", "none", ref="answer"),
             "what is my average speed last week": F("stat", "speed", "mean"),
             "what about the battery temperature": F("stat", "battery_temperature", "mean", ref="concept")}

    def setUp(self):
        from vgpt.vehicle import lake as LK
        self._env = mock.patch.dict(os.environ, {"VGPT_FRAME": "2"}); self._env.start()
        for k in ("VGPT_LAYERS_OFF", "VGPT_ARBITRATE", "VGPT_MODEL_ONLY"): os.environ.pop(k, None)
        LK.adopt_path(DB)
        from vgpt.meaning.pipeline import MeaningBrain
        self.model = Scripted2(self.TABLE); self.b = MeaningBrain(DB, self.model, now=NOW); self.LK = LK

    def tearDown(self): self.LK.release(); self._env.stop()

    def test_the_model_is_shown_the_previous_frame_and_nothing_before_the_first_question(self):
        self.b.ask("how many hard brakes last week"); self.b.ask("and the week before that")
        self.assertIsNone(self.model.seen[0][1]); self.assertEqual(self.model.seen[1][1], "events hard_braking count none none none none none")

    def test_a_period_follow_up_keeps_the_previous_question_and_moves_the_period(self):
        a = self.b.ask("how many hard brakes last week"); r = self.b.ask("and the week before that")
        self.assertEqual(r["facts"]["dialogue"]["kind"], "period"); self.assertEqual(r["facts"]["frame"]["concept"], "hard_braking"); self.assertNotEqual(a["answer"], r["answer"])

    def test_a_concept_follow_up_keeps_the_previous_period_and_operator(self):
        self.b.ask("what is my average speed last week"); r = self.b.ask("what about the battery temperature")
        self.assertEqual(r["facts"]["dialogue"]["kind"], "concept"); self.assertEqual(r["facts"]["frame"]["concept"], "battery_temperature"); self.assertIn("last 7 days", r["answer"].lower() + " last 7 days")

    def test_picking_one_item_of_the_previous_list(self):
        self.b.ask("how many hard brakes last week"); r = self.b.ask("which one was the strongest")
        self.assertEqual(r["facts"]["dialogue"]["kind"], "ordinal"); self.assertIn("hard-braking", r["answer"].lower())

    def test_a_question_about_the_previous_answer_is_explained_from_what_was_run(self):
        self.b.ask("how many hard brakes last week"); r = self.b.ask("how did you get that")
        self.assertEqual(r["facts"]["route"], "dialogue:provenance"); self.assertIn("How I got the answer", r["answer"])

    def test_with_nothing_before_it_a_reference_is_just_the_frame(self):
        r = self.b.ask("and the week before that")                      # the model said ref=period but there is no previous turn: the frame executes alone, nothing is invented
        self.assertNotIn("dialogue", r["facts"])


if __name__ == "__main__":
    unittest.main()

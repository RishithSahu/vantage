"""The typed non-executing outcomes keep their own meaning from model output to rendered text (vgpt/outcomes.py).
v15 regression: `p.why=offtopic` was rendered as a greeting, so an unsafe request got "Hello - I answer questions..." from the hybrid renderer."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import outcomes as O
from vgpt import learned_front as LF
from vgpt.query_ir import Query
from vgpt.query_grammar import PARAM_ENUMS, gbnf


class _Qcs: last = None


class _Brain: qcs = _Qcs()


class Stub:
    def __init__(self, out): self.out = out; self.lora_path = None
    def compile(self, text, context=None): return Query.from_text(self.out), {"raw": self.out, "ms": 0.0, "min_p": 0.99}


def answer(plan, text="x", contract=2):
    f = LF.LearnedFront(_Brain(), Stub(plan)); f.contract = contract
    return f.answer(text)[0]


class Categories(unittest.TestCase):
    def test_every_category_has_a_distinct_rendering_in_contract_2(self):
        texts = {c: O.render(c, 2)[0] for c in O.CATEGORIES if c != "definition"}
        self.assertEqual(len(set(texts.values())), len(texts), texts)

    def test_only_greeting_is_a_greeting(self):
        for c in ("offtopic", "unsupported", "unrecorded", "control", "unsafe", "ambiguous"):
            self.assertFalse(O.render(c, 2)[0].startswith("Hello"), c)
        self.assertTrue(O.render("greeting", 2)[0].startswith("Hello"))

    def test_unsafe_is_a_refusal_that_reveals_nothing(self):
        t = O.render("unsafe", 2)[0]
        self.assertIn("cannot help", t); self.assertFalse(any(ch.isdigit() for ch in t))

    def test_contract_1_keeps_the_frozen_wording(self):
        self.assertTrue(O.render("offtopic", 1)[0].startswith("Hello - I answer questions"))     # frozen adapters were evaluated with this rendering
        self.assertEqual(O.render("unsupported", 1)[0], "That is outside what this vehicle's recorded data can tell me, so I will not guess.")

    def test_unknown_category_is_not_another_category(self):
        self.assertEqual(O.render("nonsense", 2), (None, None)); self.assertEqual(O.render("unsafe", 1), (None, None))

    def test_grammar_and_parser_accept_the_new_categories(self):
        for w in ("greeting", "unsafe"):
            self.assertIn(w, PARAM_ENUMS["why"]); self.assertEqual(Query.from_text(f"op=none t=all p.why={w}").params["why"], w)
        self.assertIn("unsafe", gbnf())


class Rendering(unittest.TestCase):
    def test_outcomes_survive_the_front_end(self):
        want = {"greeting": "social", "offtopic": "out_of_scope", "unsupported": "unsupported", "unrecorded": "unrecorded", "control": "control", "unsafe": "unsafe", "ambiguous": "ambiguous"}
        seen = set()
        for why, kind in want.items():
            r = answer(f"op=none t=all p.why={why}", text="what was the cabin CO2 level")      # names an unmodelled quantity: a grounded "not recorded"
            self.assertEqual(r["facts"]["outcome"], kind); self.assertEqual(r["facts"]["why"], why); self.assertEqual(r["intent"], "NONE"); self.assertFalse(r["facts"]["ok"])
            seen.add(r["answer"])
        self.assertEqual(len(seen), len(want))

    def test_offtopic_and_unsafe_are_never_the_greeting(self):
        for why in ("offtopic", "unsafe", "unsupported", "control"):
            self.assertFalse(answer(f"op=none t=all p.why={why}")["answer"].startswith("Hello"))

    def test_unknown_category_abstains_instead_of_defaulting(self):
        f = LF.LearnedFront(_Brain(), Stub("op=none t=all p.why=banana")); f.contract = 2
        r, it = f.answer("x"); self.assertIsNone(r); self.assertTrue(any("unknown outcome" in i for i in it.issues))

    def test_frozen_contract_still_renders_the_old_greeting_for_offtopic(self):
        self.assertTrue(answer("op=none t=all p.why=offtopic", contract=1)["answer"].startswith("Hello"))


class Sidecar(unittest.TestCase):
    def test_sidecar_selects_the_contract_and_default_is_1(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "adapter.gguf"); open(a, "w").close()
            self.assertEqual(O.contract_for(a), 1)
            json.dump({"outcome_contract": 2}, open(a + ".contract.json", "w")); self.assertEqual(O.contract_for(a), 2)
        self.assertEqual(O.contract_for(None), 1)

    def test_frozen_adapters_have_no_sidecar(self):
        for n in ("qp-lora.gguf", "qp-lora-v13.gguf", "qp-lora-v14d_s143.gguf", "qp-lora-v15.gguf"):
            p = os.path.join(HERE, "models", n)
            self.assertFalse(os.path.exists(p + ".contract.json")); self.assertEqual(O.contract_for(p), 1)


if __name__ == "__main__":
    unittest.main()

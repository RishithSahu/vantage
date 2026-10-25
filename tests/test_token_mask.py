"""The token-mask decoder (vgpt/meaning/interface.py TokenMask) must allow exactly what the generated grammar allows, and give the model's answer, only faster."""
import json
import os
import sys
import time
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning.interface import TokenMask  # noqa: E402


class CharLlm:
    """A tokenizer with one token per byte, enough to test the state logic."""
    def token_eos(self): return 999
    def tokenize(self, b, add_bos=False, special=False): return list(b)


def run(mask, text):
    """Greedy-by-preference walk: at each step ask which tokens are open, check `text`'s next token is among them, advance.  -> the ids open after the whole text."""
    st = mask.start()
    for ch in text.encode():
        assert ch in set(mask.open_ids(st).tolist()), (text, chr(ch))
        st = mask.advance(st, ch)
    return set(mask.open_ids(st).tolist())


class Logic(unittest.TestCase):
    def setUp(self): self.m = TokenMask(CharLlm(), [["a", "ab", "none"], ["x", "xy", "none"], ["p", "none"]])

    def test_the_first_slot_opens_with_its_own_first_tokens_only(self):
        self.assertEqual(set(self.m.open_ids(self.m.start()).tolist()), {ord("a"), ord("n")})

    def test_a_word_that_is_a_prefix_of_another_keeps_both_ways_open(self):
        o = run(self.m, "a"); self.assertIn(ord("b"), o); self.assertIn(ord(" "), o); self.assertNotIn(999, o)             # continue the word, or end it and take the space of the next slot

    def test_the_next_slot_starts_with_a_space_and_its_own_words(self):
        o = run(self.m, "ab "); self.assertEqual(o, {ord("x"), ord("n")}); o = run(self.m, "ab x"); self.assertIn(ord("y"), o); self.assertIn(ord(" "), o)

    def test_only_the_end_of_the_answer_is_open_after_the_last_word(self):
        self.assertEqual(run(self.m, "a x p"), {999}); self.assertEqual(run(self.m, "none none none"), {999})

    def test_a_word_not_in_a_slot_is_never_open(self):
        with self.assertRaises(AssertionError): run(self.m, "z")
        with self.assertRaises(AssertionError): run(self.m, "a q")

    def test_the_processor_masks_the_scores_and_follows_the_generated_tokens(self):
        p = self.m.processor(); scores = np.zeros(1000, dtype=np.float32); out = p([1, 2, 3], scores)
        self.assertEqual(set(np.flatnonzero(np.isfinite(out)).tolist()), {ord("a"), ord("n")}); out = p([1, 2, 3, ord("a")], np.zeros(1000, dtype=np.float32))
        self.assertTrue({ord("b"), ord(" ")} <= set(np.flatnonzero(np.isfinite(out)).tolist()))


def _product():
    """The adapter and frame version the product itself uses (models/product.json): this comparison must be made with the model that is shipped, not with a file name written into the test."""
    try:
        j = json.load(open(os.path.join(ROOT, "models", "product.json"), encoding="utf-8")); return os.path.join(ROOT, "models", j["adapter"]), int(j.get("frame", 2))
    except (OSError, ValueError, KeyError): return os.path.join(ROOT, "models", "meaning-lora-frame2.gguf"), 2


MODEL, FRAME = _product()
BASE = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")


@unittest.skipUnless(os.path.exists(MODEL) and os.path.exists(BASE), "needs the frame-2 adapter and the base model")
class AgainstTheGrammar(unittest.TestCase):
    def test_the_same_frames_and_the_same_confidence_as_the_grammar_and_faster(self):
        import io
        os.environ["HF_HUB_OFFLINE"] = "1"
        with mock.patch.dict(os.environ, {"VGPT_FRAME": str(FRAME)}):
            from vgpt.meaning.interface import FrameModel
            from vgpt.vehicle import lake as LK
            qs = []
            for f in ("novel_v1", "digest_v1", "catalog_v1"):
                for ln in io.open(os.path.join(ROOT, "bench", "meaning_eval", f + ".tsv"), encoding="utf-8"):
                    if not ln.startswith("#") and ln.strip(): qs.append(ln.split("\t")[1])
            qs = qs[::9][:40]; LK.adopt_path(os.path.join(ROOT, "data", "world_aster.db")) if os.path.exists(os.path.join(ROOT, "data", "world_aster.db")) else None
            m = FrameModel(MODEL, threads=2, version=FRAME); self.assertIsNotNone(m.mask); m.parse("warm up")
            t0 = time.perf_counter(); a = [m.parse(q)[1] for q in qs]; ta = time.perf_counter() - t0
            mask, m.mask = m.mask, None; t0 = time.perf_counter(); b = [m.parse(q)[1] for q in qs]; tb = time.perf_counter() - t0; m.mask = mask
            self.assertEqual([x["raw"] for x in a], [x["raw"] for x in b]); self.assertTrue(all(abs((x["pmin"] or 0) - (y["pmin"] or 0)) < 0.02 for x, y in zip(a, b)))                         # the mask reads the confidence over the tokens it allows, the grammar over the whole vocabulary: the same to about a hundredth
            self.assertEqual(getattr(m, "mask_misses", 0), 0); self.assertLess(ta, 0.8 * tb, (ta, tb))


if __name__ == "__main__":
    unittest.main()

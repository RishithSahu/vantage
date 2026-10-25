"""The decoder and the confidence: one bounded decode that fails closed, a confidence read over the tokens the decoder may allow (the same in both backends), a monotone calibration from development questions only, and an answer
that is withheld - not given as if certain - when the calibrated confidence is under the line.  Also: ONNX Runtime does not busy-wait on a board that has no cores to spare."""
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vgpt.meaning import calibration as CB  # noqa: E402
from vgpt.meaning.interface import TokenMask  # noqa: E402

BASE = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
WORLD = os.path.join(ROOT, "data", "world_aster.db")


class FakeLlm:
    """Tokens are the characters of the word: the mask only needs tokenize() and token_eos()."""
    def token_eos(self): return 999
    def tokenize(self, b, add_bos=False, special=False): return [ord(c) for c in b.decode()]


class Mask(unittest.TestCase):
    def mask(self): return TokenMask(FakeLlm(), [["now", "events"], ["speed", "battery_soc"], ["none"]])

    def test_the_longest_valid_frame_bounds_the_decode(self):
        m = self.mask(); self.assertEqual(m.max_tokens, len("events") + len(" battery_soc") + len(" none") + 1)

    def test_the_confidence_is_read_over_the_allowed_tokens_only(self):
        m = self.mask(); steps = []; state = {}; proc = m.processor(steps, state); scores = np.full(1000, -50.0); scores[ord("n")] = 2.0; scores[ord("e")] = 1.0; scores[ord("x")] = 30.0       # x is not allowed, and huge
        out = proc(np.array([], dtype=np.int64), scores); self.assertAlmostEqual(steps[0], 1.0 / (1.0 + np.exp(1.0 - 2.0)), places=9)                       # not 1/(1+...+e^(30-2)): the vocabulary outside the mask does not count
        self.assertTrue(np.isfinite(out[ord("n")]) and np.isfinite(out[ord("e")]) and np.isinf(out[ord("x")]))

    def test_no_open_token_fails_closed_with_only_the_end_token(self):
        m = self.mask(); m.open_ids = lambda states: np.array([], dtype=np.int64); state = {}; proc = m.processor([], state); out = proc(np.array([], dtype=np.int64), np.zeros(1000))
        self.assertTrue(state["dead"]); self.assertEqual(np.flatnonzero(np.isfinite(out)).tolist(), [999])


@unittest.skipUnless(os.path.exists(BASE), "needs the base model's tokenizer")
class RealTokenizer(unittest.TestCase):
    def test_the_bound_covers_every_word_of_the_vocabulary_with_the_real_tokenizer(self):
        from llama_cpp import Llama
        from vgpt.meaning import ontology as ON
        llm = Llama(model_path=BASE, vocab_only=True, verbose=False); onto = ON.load(2); w = onto.words(); m = TokenMask(llm, [w[k] for k in onto.fields])
        longest = sum(max(len(llm.tokenize((("" if i == 0 else " ") + x).encode(), add_bos=False, special=False)) for x in w[k]) for i, k in enumerate(onto.fields)) + 1
        self.assertEqual(m.max_tokens, longest); self.assertLess(m.max_tokens, 40 + 30)


class OrtSpin(unittest.TestCase):
    def spin(self, cores, threads, env=None):
        from vgpt.meaning import ort_backend as O
        with mock.patch.dict(os.environ, env or {}, clear=False):
            if not env: os.environ.pop("VGPT_ORT_SPIN", None)
            with mock.patch("os.sched_getaffinity", return_value=set(range(cores)), create=True): return O.spin_default(threads)

    def test_spinning_is_off_when_the_board_has_no_cores_to_spare(self):
        self.assertEqual(self.spin(4, 2), "0"); self.assertEqual(self.spin(2, 2), "0"); self.assertEqual(self.spin(4, 4), "0")

    def test_spinning_is_on_with_cores_to_spare_and_the_person_can_decide(self):
        self.assertEqual(self.spin(12, 2), "1"); self.assertEqual(self.spin(4, 2, {"VGPT_ORT_SPIN": "1"}), "1"); self.assertEqual(self.spin(12, 2, {"VGPT_ORT_SPIN": "0"}), "0")


class Calibration(unittest.TestCase):
    def test_the_map_is_monotone_and_within_zero_and_one(self):
        xs = [i / 200 for i in range(0, 201)]; ys = [CB.calibrated(x) for x in xs]
        self.assertTrue(all(0.0 <= y <= 1.0 for y in ys)); self.assertTrue(all(b >= a - 1e-12 for a, b in zip(ys, ys[1:]))); self.assertEqual(CB.calibrated(None), 0.0)

    def test_the_file_records_how_it_was_made_and_that_it_was_checked_on_other_questions(self):
        j = json.load(open(CB.PATH, encoding="utf-8")); r = j["report"]
        self.assertGreater(r["questions"], 1000); self.assertGreater(r["checked_on"], 500); self.assertLessEqual(r["calibrated"]["ece"], r["raw"]["ece"]); self.assertIn("crc32", j["split"])
        for bad in ("sealed", "holdout", "acceptance"): self.assertNotIn(bad, " ".join(j["sets"]), "the sealed sets are never used for calibration"); self.assertIn(bad, j["excluded"])

    def test_the_line_comes_from_the_environment_and_zero_switches_it_off(self):
        with mock.patch.dict(os.environ, {"VGPT_MEANING_REFUSE_BELOW": "0.6"}): self.assertEqual(CB.refuse_below(), 0.6)
        with mock.patch.dict(os.environ, {"VGPT_MEANING_REFUSE_BELOW": "0"}): self.assertEqual(CB.refuse_below(), 0.0)
        with mock.patch.dict(os.environ, {"VGPT_MEANING_REFUSE_BELOW": "junk"}): self.assertEqual(CB.refuse_below(), CB.DEFAULT_REFUSE_BELOW)


@unittest.skipUnless(os.path.exists(WORLD), "needs data/world_aster.db")
class Withholding(unittest.TestCase):
    def ask(self, pmin, env=None):
        from vgpt.meaning import glossary as GL
        from vgpt.meaning.frame import Frame
        from vgpt.meaning.pipeline import MeaningBrain
        from vgpt.vehicle import lake as LK

        class Stub:
            glossary = GL.load()
            def parse(self, text, hint=None, prev=None):
                f = Frame("now", "speed"); return f, {"raw": f.to_text(), "pmin": pmin, "ms": 0.0, "hint": ""}
        with mock.patch.dict(os.environ, env or {}):
            LK.adopt_path(WORLD)
            try:
                b = MeaningBrain(WORLD, Stub(), now=1792360800); return b.ask("what is my speed right now")
            finally: LK.release()

    def test_a_reading_the_model_was_unsure_of_is_not_answered_as_certain(self):
        r = self.ask(0.30); self.assertEqual(r["facts"]["route"], "clarify:low_confidence"); self.assertIn("will not guess", r["answer"]); self.assertLess(r["conf"], 0.5); self.assertEqual(r["facts"]["refuse_below"], CB.DEFAULT_REFUSE_BELOW)

    def test_a_confident_reading_is_answered_and_the_reported_confidence_is_the_calibrated_one(self):
        r = self.ask(0.999); self.assertNotEqual(r["facts"]["route"], "clarify:low_confidence"); self.assertAlmostEqual(r["conf"], round(CB.calibrated(0.999), 2), places=2); self.assertEqual(r["facts"]["pmin"], 0.999)
        r2 = self.ask(0.55); self.assertAlmostEqual(r2["conf"], round(CB.calibrated(0.55), 2), places=2); self.assertLess(r2["conf"], 0.9)                  # a middling score is reported as middling: no floor of 0.5, no ceiling

    def test_no_confidence_information_is_not_taken_for_low_confidence(self):
        r = self.ask(None); self.assertNotEqual(r["facts"]["route"], "clarify:low_confidence"); self.assertIsNone(r["facts"]["confidence"])

    def test_the_refusal_can_be_switched_off(self):
        r = self.ask(0.30, {"VGPT_MEANING_REFUSE_BELOW": "0"}); self.assertNotEqual(r["facts"]["route"], "clarify:low_confidence")


if __name__ == "__main__": unittest.main()

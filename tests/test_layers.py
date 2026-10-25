"""The layers between the learned frame and the answer can be switched off one by one or all at once, so that what the model alone contributes can be measured (vgpt/meaning/layers.py)."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning import layers  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tests"))
DB = os.path.join(ROOT, "data", "gen_courier.db")


class Switches(unittest.TestCase):
    def test_by_default_only_the_retired_arbitration_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("VGPT_LAYERS_OFF", "VGPT_ARBITRATE", "VGPT_MODEL_ONLY"): os.environ.pop(k, None)
            self.assertEqual(layers.off(), {"arbitrate"}); self.assertTrue(layers.on("discourse")); self.assertFalse(layers.model_only())

    def test_a_layer_is_switched_off_by_name_and_all_at_once(self):
        with mock.patch.dict(os.environ, {"VGPT_LAYERS_OFF": "own_vocab, investigate"}):
            self.assertFalse(layers.on("own_vocab")); self.assertFalse(layers.on("investigate")); self.assertTrue(layers.on("unsafe_floor"))
        for v in ({"VGPT_LAYERS_OFF": "all"}, {"VGPT_MODEL_ONLY": "1"}):
            with mock.patch.dict(os.environ, v): self.assertTrue(layers.model_only()); self.assertFalse(layers.on("fuel_remap"))

    def test_the_old_arbitration_variable_still_switches_it_on(self):
        with mock.patch.dict(os.environ, {"VGPT_ARBITRATE": "1"}): os.environ.pop("VGPT_LAYERS_OFF", None); self.assertTrue(layers.on("arbitrate"))


@unittest.skipUnless(os.path.exists(DB), "generated history not present")
class InThePipeline(unittest.TestCase):
    def brain(self):
        from test_followups import Scripted, TABLE, NOW
        from vgpt.meaning.pipeline import MeaningBrain
        return MeaningBrain(DB, Scripted(TABLE), now=NOW)

    def test_with_every_layer_off_the_models_frame_is_executed_as_it_is_and_a_fragment_is_not_completed(self):
        with mock.patch.dict(os.environ, {"VGPT_LAYERS_OFF": "all"}):
            b = self.brain(); b.ask("how many hard brakes last week"); r = b.ask("and the week before that")
            self.assertEqual(r["facts"]["frame"]["act"], "ambiguous", "no follow-up completion: the model's reading of the fragment stands"); self.assertNotIn("dialogue", r["facts"])

    def test_with_the_layers_on_the_same_fragment_continues_the_previous_question(self):
        with mock.patch.dict(os.environ, {}):
            for k in ("VGPT_LAYERS_OFF", "VGPT_ARBITRATE", "VGPT_MODEL_ONLY"): os.environ.pop(k, None)
            b = self.brain(); b.ask("how many hard brakes last week"); r = b.ask("and the week before that")
            self.assertEqual(r["facts"]["frame"]["concept"], "hard_braking"); self.assertEqual(r["facts"]["dialogue"]["kind"], "period")


class VehicleVocabulary(unittest.TestCase):
    """The layer that supplies words only ONE vehicle brings (AdBlue on the diesel truck, lambda on the petrol car) must not take over the model's honest 'this car does not record that' on any other vehicle."""
    NOW = 1792348500

    def ask(self, db, q, frame):
        path = os.path.join(ROOT, "data", db)
        if not os.path.exists(path): self.skipTest(db + " is not present")
        from test_followups import Scripted
        from vgpt.meaning.pipeline import MeaningBrain
        from vgpt.vehicle import lake as LK
        for k in ("VGPT_LAYERS_OFF", "VGPT_ARBITRATE", "VGPT_MODEL_ONLY", "VGPT_OWNVOCAB_ALL"): os.environ.pop(k, None)
        LK.adopt_path(path)
        try: return MeaningBrain(path, Scripted({q: frame}), now=self.NOW).ask(q)
        finally: LK.release()

    def test_a_name_only_this_vehicle_declares_is_found_through_its_own_vocabulary(self):
        from vgpt.meaning.frame import Frame
        r = self.ask("world_rhino.db", "what is the adblue level", Frame("now", "unlisted")); self.assertEqual(r["facts"]["frame"]["concept"], "def_level"); self.assertIn("own signal", r["facts"]["arbitrated"])

    def test_a_generic_word_never_takes_over_an_unlisted_reading_on_another_vehicle(self):
        from vgpt.meaning.frame import Frame
        for q, f in (("engine oil temperature", Frame("now", "unlisted")), ("how is the heated steering wheel", Frame("health", "unlisted")), ("is the wheel alignment ok", Frame("health", "unlisted"))):
            r = self.ask("world_aster.db", q, f); self.assertEqual(r["facts"]["frame"]["concept"], "unlisted", q); self.assertNotIn("arbitrated", r["facts"], q)


if __name__ == "__main__":
    unittest.main()

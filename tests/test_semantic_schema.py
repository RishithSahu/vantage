import json
import os
import tempfile
import unittest

from training import semantic_data
from vgpt.semantic_frame import SemanticFrame, TASK, OPERATION, CONCEPT, EVENT

class SemanticSchemaTests(unittest.TestCase):
    def test_every_wire_label_round_trips_to_one_canonical_frame(self):
        for task in TASK:
            for op in OPERATION:
                for concept in CONCEPT:
                    for event in EVENT:
                        frame = SemanticFrame.from_tokens(task, op, concept, event)
                        self.assertEqual(SemanticFrame.from_tokens(*frame.to_tokens().split()), frame)

    def test_generator_targets_decode_and_reencode_exactly(self):
        for row in semantic_data.rows():
            frame = SemanticFrame.from_tokens(*row["frame"].split())
            self.assertEqual(frame.as_dict(), row["canonical_frame"])
            self.assertEqual(frame.to_tokens(), row["frame"])

    def test_validation_holds_out_whole_templates_and_compositions(self):
        with tempfile.TemporaryDirectory() as root:
            audit = semantic_data.write(root)
            self.assertEqual(audit["template_overlap_train_val"], [])
            self.assertEqual(audit["composition_overlap_train_test_comp"], [])
            self.assertGreater(audit["counts"]["val"], 0)
            self.assertGreater(audit["counts"]["test_comp"], 0)
            with open(os.path.join(root, "split_audit.json"), encoding="utf-8") as f: persisted = json.load(f)
            self.assertEqual(persisted, audit)

    def test_coarse_frame_is_not_a_complete_query(self):
        frame = SemanticFrame.from_tokens("data", "compare", "energy", "none")
        self.assertFalse(hasattr(frame, "time"))
        self.assertFalse(hasattr(frame, "conditions"))
        self.assertFalse(hasattr(frame, "groups"))

if __name__ == "__main__": unittest.main()

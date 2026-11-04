"""vgpt/meaning/context.py: what a follow-up takes from the previous frame is copied after a bare reading (cases from the audit sets), and anything it cannot complete with certainty is left to the model."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.meaning import context as CX


class Complete(unittest.TestCase):
    def test_a_new_period_keeps_the_whole_previous_frame(self):
        w, why = CX.complete("ambiguous none none none none none none none", "events hard_braking count none none none none none", "same again for the week before", True)
        self.assertEqual((" ".join(w), why), ("events hard_braking count none none none none period", "period"))

    def test_a_new_quantity_keeps_act_op_and_the_rest(self):
        w, why = CX.complete("now motor_temperature none none none none none none", "stat speed mean none none none none none", "what about the motor temperature", False)
        self.assertEqual((" ".join(w), why), ("stat motor_temperature mean none none none none concept", "concept"))

    def test_pointers_are_left_to_the_model(self):
        for q in ("why did that happen", "which one was it", "and the battery level back then", "how did you get that"):
            self.assertFalse(CX.eligible(q), q)
            self.assertIsNone(CX.complete("why none none none none none none none", "events hard_braking last none none none none none", q, False), q)

    def test_eligibility_is_decided_from_the_wording(self):
        self.assertTrue(CX.eligible("and the motor")); self.assertTrue(CX.eligible("what about last month")); self.assertFalse(CX.eligible("how fast am I going"))

    def test_nothing_to_copy_from_means_no_completion(self):
        self.assertIsNone(CX.complete("ambiguous none none none none none none none", "none none none none none none none none", "same again for last week", True))
        self.assertIsNone(CX.complete("now speed none none none none none none", "stat speed mean none none none none none", "and the speed", False))        # the same quantity again: not a new one


if __name__ == "__main__":
    unittest.main()

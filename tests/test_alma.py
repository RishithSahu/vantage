"""Unit tests for the question-type-free ALMA feasibility prototype."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from alma.cards import generate_micro_examples, generate_roundtrip_scope_examples, gold_task_key
from alma.decoder import (
    Scored, ScopeProgram, exact_decode, execute_scope, minimal_scope_counterexample,
)
from alma.edge_data import challenge_rows, training_rows
from alma.operator_curriculum import (
    contrast_pairs as operator_contrast_pairs,
    verify_pairs as verify_operator_pairs,
)
from alma.joint_operator_integration import compatible, joint_training_rows, signature_vocabulary
from alma.field_arity_oracle import FIELD_SELECTION_FIELDS, substitute as substitute_field_arity
from alma.structural_suboracle import SUBFIELDS
from alma.entity_selection_suboracle import MASKS as ENTITY_SELECTION_MASKS, verify_masks
from alma.selection_projector import project_selection, selection_violation
from alma.selection_variant_experiment import (
    VARIANTS as SELECTION_VARIANTS, audit_rows as audit_selection_variants,
    curriculum as selection_curriculum, verify_curriculum as verify_selection_curriculum,
)
from alma.condition_shape_updated_oracle import error_type, grammar_record
from alma.condition_state_experiment import (
    audit_target as audit_condition_target, curriculum as condition_curriculum,
    verify_curriculum as verify_condition_curriculum,
)
from alma.omission_addition_oracle import presence_only, verify_family_partition
from alma.architecture_bakeoff import candidate_programs, safe_skeleton
from alma.supervised_parser import evaluation_rows, skeleton, training_pool
from alma.time_curriculum import contrast_pairs, verify_pairs
from vgpt.query_ir import Query, TimeSpec


class AlmaData(unittest.TestCase):
    def test_training_corpus_contains_no_complete_questions(self):
        rows = generate_micro_examples()
        self.assertGreater(len(rows), 500)
        for r in rows:
            low = r["text"].strip().lower()
            self.assertNotIn("?", low)
            self.assertFalse(low.startswith(("what ", "which ", "when ", "why ", "how ", "did ", "does ", "is ", "are ")), r)

    def test_roundtrip_scope_corpus_contains_no_questions(self):
        rows = generate_roundtrip_scope_examples()
        self.assertGreater(len(rows), 500)
        for r in rows:
            self.assertNotIn("?", r["text"])

    def test_gold_task_projection_is_compositional(self):
        self.assertEqual(gold_task_key(Query.from_text("op=extreme tgt=speed agg=max t=cal:yesterday")), "task:max")
        self.assertEqual(gold_task_key(Query.from_text("op=aggregate tgt=battery_temperature agg=mean t=last:7:day")), "task:mean")
        self.assertEqual(gold_task_key(Query.from_text("op=trend tgt=range_estimate t=last:2:week")), "task:trend")

    def test_edge_probe_is_question_free_and_construction_separated(self):
        train = training_rows()
        challenge = challenge_rows()
        self.assertEqual(len(train), 900)
        self.assertEqual(len(challenge), 300)
        self.assertFalse(any("?" in r["text"] for r in train))
        self.assertTrue(all(r["source"] != challenge[0]["source"] for r in train))
        self.assertEqual({r["label"] for r in challenge}, {"comparison", "left", "right"})

    def test_supervised_parser_keeps_final_and_acceptance_sets_out(self):
        train = training_pool()
        evaluation = evaluation_rows()
        self.assertEqual({r["source"] for r in train}, {"ir_gold", "diag_v1"})
        self.assertEqual({r["source"] for r in evaluation}, {"indep_dev"})
        self.assertGreaterEqual(len(train), 100)
        self.assertTrue(all(set(r["labels"]) == set(skeleton(Query.from_text(r["ir"]))) for r in train))

    def test_time_curriculum_contrasts_are_formally_distinguishable(self):
        train = verify_pairs(contrast_pairs("train"))
        held = verify_pairs(contrast_pairs("heldout"))
        self.assertEqual((train["pairs"], train["verified"]), (120, 120))
        self.assertEqual((held["pairs"], held["verified"]), (60, 60))
        self.assertFalse(train["failures"] or held["failures"])

    def test_operator_curriculum_is_valid_minimal_and_denotational(self):
        train = verify_operator_pairs(operator_contrast_pairs("train"))
        held = verify_operator_pairs(operator_contrast_pairs("heldout"))
        self.assertEqual((train["pairs"], train["verified"]), (56, 56))
        self.assertEqual((held["pairs"], held["verified"]), (56, 56))
        self.assertEqual(len(train["families"]), 14)
        self.assertFalse(train["failures"] or held["failures"])

    def test_joint_operator_vocabulary_contains_only_compatible_signatures(self):
        natural, synthetic, rows = joint_training_rows()
        signatures = signature_vocabulary(rows)
        self.assertEqual((len(natural), len(synthetic)), (100, 112))
        self.assertTrue(signatures)
        self.assertTrue(all(compatible(sig) for sig in signatures))

    def test_field_and_arity_oracles_are_disjoint(self):
        rows = evaluation_rows()[:2]
        base = [dict(row["labels"]) for row in rows]
        for prediction in base:
            prediction["condition_shape"] = "none"
            prediction["target_count"] = "none"
        fields = substitute_field_arity(base, rows, fields=True)
        arity = substitute_field_arity(base, rows, arity=True)
        self.assertTrue(all(p["target_count"] == "none" for p in fields))
        self.assertTrue(all(p["condition_shape"] == "none" for p in arity))
        self.assertNotIn("target_count", FIELD_SELECTION_FIELDS)

    def test_structural_suboracles_partition_the_field_oracle(self):
        flattened = [field for fields in SUBFIELDS.values() for field in fields]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(FIELD_SELECTION_FIELDS))

    def test_entity_selection_masks_are_disjoint_and_complete(self):
        verification = verify_masks()
        self.assertTrue(verification["disjoint"])
        self.assertTrue(verification["exact_fields"])
        flattened = [field for mask in ENTITY_SELECTION_MASKS.values() for field in mask]
        self.assertEqual(flattened, ["select_entity", "select_order", "select_k"])

    def test_selection_projector_is_conservative_and_text_free(self):
        orphan = {"select_entity": "none", "select_order": "desc", "select_k": "many"}
        fixed, audit = project_selection(orphan)
        self.assertEqual((fixed["select_entity"], fixed["select_order"], fixed["select_k"]),
                         ("none", "none", "none"))
        self.assertTrue(audit["changed"])
        malformed = {"select_entity": "trip", "select_order": "none", "select_k": "many"}
        unchanged, audit = project_selection(malformed)
        self.assertEqual(unchanged, malformed)
        self.assertTrue(audit["unresolved"])
        valid = {"select_entity": "charge_session", "select_order": "asc", "select_k": "one"}
        unchanged, audit = project_selection(valid)
        self.assertEqual(unchanged, valid)
        self.assertIsNone(selection_violation(valid))
        self.assertFalse(audit["changed"] or audit["unresolved"])

    def test_selection_variant_target_and_curriculum(self):
        audit = audit_selection_variants(training_pool())
        self.assertTrue(audit["unambiguous"])
        self.assertEqual(audit["class_counts"],
                         {"Absent": 91, "TripSelection": 8, "ChargeSessionSelection": 1})
        self.assertFalse(audit["full_selection_validity_expressible"])
        train = verify_selection_curriculum(selection_curriculum("train"))
        held = verify_selection_curriculum(selection_curriculum("heldout"))
        self.assertEqual((train["triples"], train["verified"]), (40, 40))
        self.assertEqual((held["triples"], held["verified"]), (20, 20))
        self.assertFalse(train["failures"] or held["failures"])
        self.assertEqual(set(SELECTION_VARIANTS), {"Absent", "TripSelection", "ChargeSessionSelection"})

    def test_condition_oracle_error_taxonomy_follows_typed_grammar(self):
        self.assertEqual(error_type("none", "named"), "spurious_condition")
        self.assertEqual(error_type("threshold", "none"), "omitted_condition")
        self.assertEqual(error_type("named", "threshold"), "named_threshold_kind_error")
        self.assertEqual(error_type("multi_named", "named"), "multiplicity_error")
        self.assertEqual(error_type("mixed", "multi_threshold"), "mixedness_error")
        row = {"ir": "op=count tgt=trip_count t=last:7:day c=highway,sig:speed:gt:80:km_per_h",
               "labels": {"condition_shape": "mixed"}}
        record = grammar_record(row, "multi_named")
        self.assertEqual(record["condition_count"], 2)
        self.assertEqual(record["condition_kinds"], ["named", "threshold"])
        self.assertEqual(record["error_type"], "mixedness_error")

    def test_condition_state_target_and_curriculum_are_bounded(self):
        train_audit = audit_condition_target(training_pool())
        dev_audit = audit_condition_target(evaluation_rows())
        self.assertTrue(train_audit["exactly_representable"])
        self.assertTrue(dev_audit["exactly_representable"])
        self.assertEqual(train_audit["state_counts"],
                         {"Absent": 72, "NamedCondition": 22, "ThresholdCondition": 6})
        self.assertEqual(dev_audit["state_counts"],
                         {"Absent": 108, "NamedCondition": 4, "ThresholdCondition": 8})
        for split, expected in (("train", 40), ("heldout", 20)):
            verification = verify_condition_curriculum(condition_curriculum(split))
            self.assertEqual((verification["triples"], verification["verified"]),
                             (expected, expected))
            self.assertFalse(verification["failures"])

    def test_presence_oracle_suppresses_but_never_fills(self):
        partition = verify_family_partition()
        self.assertTrue(partition["disjoint"] and partition["complete"])
        gold = dict(evaluation_rows()[0]["labels"])
        gold.update({"op": "none", "condition_shape": "none", "event": "present"})
        prediction = dict(gold)
        prediction.update({"op": "aggregate", "condition_shape": "named", "event": "none"})
        rows = [{"labels": gold}]
        corrected, audits = presence_only(rows, [prediction])
        self.assertEqual(corrected[0]["condition_shape"], "none")
        self.assertEqual(corrected[0]["event"], "none")
        self.assertIn("event", audits[0]["unresolved_missing_fields"])
        self.assertEqual(corrected[0]["op"], "aggregate")
        self.assertNotIn("op", audits[0]["suppressed_false_positive_fields"])

    def test_global_candidate_generator_emits_complete_valid_skeletons(self):
        structure = {field: [0.0] * len(values) for field, values in
                     __import__("alma.supervised_parser", fromlist=["LABELS"]).LABELS.items()}
        from alma.time_curriculum import LABELS as time_labels
        temporal = {field: [0.0] * len(values) for field, values in time_labels.items()}
        candidates = candidate_programs(structure, temporal, [0.0, 0.0, 0.0],
                                        {}, -10.0, interaction=False, limit=20)
        self.assertTrue(candidates)
        expected = set(__import__("alma.supervised_parser", fromlist=["LABELS"]).LABELS)
        self.assertTrue(all(set(candidate) == expected for candidate, _ in candidates))
        from alma.time_integration_ablation import invalid
        self.assertTrue(all(not invalid(candidate) for candidate, _ in candidates))
        self.assertEqual(safe_skeleton("not an ir"), {field: "none" for field in expected})


class ExactDecoder(unittest.TestCase):
    def test_oracle_atoms_reconstruct_without_beam_pruning(self):
        rows = exact_decode(
            [Scored("task:max", 1.0), Scored("task:mean", 0.1)],
            [Scored("battery_temperature", 1.0), Scored("motor_temperature", 0.1)],
            [Scored(TimeSpec("calendar", name="yesterday"), 1.0)],
        )
        self.assertGreater(len(rows), 1)
        self.assertEqual(rows[0][1].to_text(), "op=extreme tgt=battery_temperature agg=max t=cal:yesterday")


class DenotationalCounterexamples(unittest.TestCase):
    def test_scope_mutation_has_a_minimal_distinguishing_world(self):
        common = ScopeProgram(common_exclude=True)
        left = ScopeProgram(left_exclude=True)
        rows = minimal_scope_counterexample(common, left)
        self.assertIsNotNone(rows)
        self.assertNotEqual(execute_scope(common, rows), execute_scope(left, rows))

    def test_equivalent_scope_programs_have_no_counterexample(self):
        a = ScopeProgram(common_exclude=True)
        b = ScopeProgram(left_exclude=True, right_exclude=True)
        self.assertIsNone(minimal_scope_counterexample(a, b))


if __name__ == "__main__":
    unittest.main()

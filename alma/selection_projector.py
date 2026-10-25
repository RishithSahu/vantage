"""Deterministic grammar-constrained projection for selection skeleton fields."""
from __future__ import annotations

import copy
import json
import os
import statistics
from collections import Counter

import torch

from .operator_aggregation_oracle import score
from .supervised_parser import (
    ROOT, StructuralParser, diverse_prefix, evaluate as evaluate_structure,
    evaluation_rows, train as train_structure, training_pool,
)
from .time_curriculum import TimeParser, _natural, contrast_pairs, train as train_time
from .time_integration_ablation import predict_time, score as score_time


SEEDS = (701, 702, 703)
MATERIAL_FIELD_REGRESSION = .01
ENTITIES = {"trip", "charge_session"}
ORDERS = {"asc", "desc"}
K_VALUES = {"none", "one", "many"}


def selection_violation(pred):
    """Return violation class, or None.

    The current skeleton has no rank_by or variant-specific payload fields, so
    cross-variant leakage cannot be observed here. Unexpected enum values are
    classified as incompatible rather than repaired.
    """
    entity, order, k = pred["select_entity"], pred["select_order"], pred["select_k"]
    if entity == "none" and (order != "none" or k != "none"):
        return "absent_with_orphaned_dependents"
    if entity in ENTITIES and order == "none":
        return "malformed_present_selection"
    if entity not in ENTITIES | {"none"} or order not in ORDERS | {"none"} or k not in K_VALUES:
        return "incompatible_selection_variant"
    return None


def project_selection(pred):
    """Apply the predeclared conservative repair policy.

    Only orphan removal is recoverable from the structured prediction itself.
    Missing entities/orders are never invented.
    """
    out = copy.deepcopy(pred)
    violation = selection_violation(out)
    changed = unresolved = False
    if violation == "absent_with_orphaned_dependents":
        out["select_order"] = "none"
        out["select_k"] = "none"
        changed = True
    elif violation is not None:
        unresolved = True
    return out, {"violation": violation, "changed": changed, "unresolved": unresolved}


def selection_validity(predictions):
    counts = Counter(selection_violation(pred) or "valid" for pred in predictions)
    n = len(predictions)
    return {"valid_rate": counts["valid"] / n,
            "violation_rate": 1 - counts["valid"] / n,
            "cases": {name: {"count": count, "rate": count / n} for name, count in sorted(counts.items())}}


def summarize(runs, config):
    metrics = ("skeleton_exact", "field_omission_rate", "unsupported_addition_rate",
               "invalid_composition_rate")
    return {metric: {"mean": statistics.mean(run[config][metric] for run in runs),
                     "min": min(run[config][metric] for run in runs),
                     "max": max(run[config][metric] for run in runs)} for metric in metrics}


def main():
    output = os.path.join(ROOT, "results", "selection_projector")
    os.makedirs(output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    struct_train = diverse_prefix(training_pool())[:100]
    eval_rows = evaluation_rows()
    natural_time_train = _natural(training_pool())
    natural_time_eval = _natural(eval_rows)
    synthetic_time = [row for a, b, _ in contrast_pairs("train") for row in (a, b)]
    with open(os.path.join(ROOT, "results", "time_integration_ablation", "metrics.json"), encoding="utf-8") as f:
        reference = json.load(f)
    reference_by_seed = {run["seed"]: run["configurations"]["predicted_time"] for run in reference["runs"]}

    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if device == "cuda": torch.cuda.manual_seed_all(seed)
        structural = StructuralParser(model_path)
        train_structure(structural, struct_train, device, seed=seed)
        structural_eval = evaluate_structure(structural, eval_rows, device)

        torch.manual_seed(seed + 1000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 1000)
        time_model = TimeParser(model_path)
        train_time(time_model, natural_time_train + synthetic_time, device, seed + 1000)
        time_predictions = predict_time(time_model, natural_time_eval, device)
        integrated = score_time(eval_rows, [p["pred"] for p in structural_eval["predictions"]],
                                time_predictions, "predicted_time")
        baseline_predictions = [item["pred"] for item in integrated.pop("details")]
        projected, audit = [], []
        for pred in baseline_predictions:
            repaired, record = project_selection(pred)
            projected.append(repaired); audit.append(record)

        baseline = score(eval_rows, baseline_predictions)
        projection = score(eval_rows, projected)
        frozen = reference_by_seed[seed]
        for metric in ("skeleton_exact", "operator_accuracy", "field_omission_rate",
                       "unsupported_addition_rate", "invalid_composition_rate"):
            if abs(baseline[metric] - frozen[metric]) > 1e-12:
                raise RuntimeError(f"Baseline reproduction failed seed={seed} metric={metric}")
        repairs = sum((not a["exact"]) and b["exact"] for a, b in zip(baseline["details"], projection["details"]))
        regressions = sum(a["exact"] and not b["exact"] for a, b in zip(baseline["details"], projection["details"]))
        invalid_fixed = sum(a["invalid"] and not b["invalid"] for a, b in zip(baseline["details"], projection["details"]))
        invalid_created = sum((not a["invalid"]) and b["invalid"] for a, b in zip(baseline["details"], projection["details"]))
        baseline.pop("details"); projection.pop("details")
        runs.append({"seed": seed, "baseline": baseline,
                     "validate_only": {**baseline, "selection": selection_validity(baseline_predictions)},
                     "projection": {**projection, "selection": selection_validity(projected)},
                     "audit": {"changed": sum(item["changed"] for item in audit),
                               "changed_rate": sum(item["changed"] for item in audit) / len(audit),
                               "unresolved": sum(item["unresolved"] for item in audit),
                               "unresolved_rate": sum(item["unresolved"] for item in audit) / len(audit),
                               "input_cases": selection_validity(baseline_predictions)["cases"]},
                     "repairs": repairs, "regressions": regressions,
                     "invalid_fixed": invalid_fixed, "invalid_created": invalid_created,
                     "baseline_reproduction_passed": True})
        del structural, time_model
        if device == "cuda": torch.cuda.empty_cache()

    baseline_summary = summarize(runs, "baseline")
    projection_summary = summarize(runs, "projection")
    exact_deltas = [run["projection"]["skeleton_exact"] - run["baseline"]["skeleton_exact"] for run in runs]
    invalid_deltas = [run["projection"]["invalid_composition_rate"] - run["baseline"]["invalid_composition_rate"] for run in runs]
    omission_deltas = [run["projection"]["field_omission_rate"] - run["baseline"]["field_omission_rate"] for run in runs]
    addition_deltas = [run["projection"]["unsupported_addition_rate"] - run["baseline"]["unsupported_addition_rate"] for run in runs]
    gates = {"strict_exact_improves_all_seeds": all(value > 0 for value in exact_deltas),
             "invalid_not_increased_any_seed": all(value <= 0 for value in invalid_deltas),
             "omission_no_material_increase_any_seed": all(value <= MATERIAL_FIELD_REGRESSION for value in omission_deltas),
             "addition_no_material_increase_any_seed": all(value <= MATERIAL_FIELD_REGRESSION for value in addition_deltas)}
    semantic_retain = all(gates.values())
    selection_validity_improves = all(run["projection"]["selection"]["valid_rate"] >=
                                      run["validate_only"]["selection"]["valid_rate"] for run in runs) and any(
                                      run["projection"]["selection"]["valid_rate"] >
                                      run["validate_only"]["selection"]["valid_rate"] for run in runs)
    if semantic_retain:
        decision = "retain_as_deterministic_correction"
    elif selection_validity_improves:
        decision = "validation_or_rejection_only"
    else:
        decision = "reject_projector"

    violation_totals = Counter()
    for run in runs:
        for name, item in run["audit"]["input_cases"].items(): violation_totals[name] += item["count"]
    result = {"device": device, "seeds": list(SEEDS), "questions": len(eval_rows),
              "grammar": {"variants": ["Absent", "TripSelection", "ChargeSessionSelection"],
                          "observable_fields": ["select_entity", "select_order", "select_k"],
                          "unobservable_in_skeleton": ["rank_by", "variant_specific_payload"]},
              "repair_policy": {"absent_with_orphans": "clear order and k",
                                "malformed_present": "leave unchanged and mark unresolved",
                                "incompatible_variant": "leave unchanged and mark unresolved",
                                "question_text_used": False},
              "runs": runs, "summary": {"baseline": baseline_summary, "projection": projection_summary},
              "deltas": {"exact": exact_deltas, "invalid": invalid_deltas,
                         "omission": omission_deltas, "addition": addition_deltas},
              "violation_totals_over_seed_question_instances": dict(violation_totals),
              "material_field_regression_threshold": MATERIAL_FIELD_REGRESSION,
              "gates": gates, "semantic_correction_retained": semantic_retain,
              "selection_validity_improved": selection_validity_improves, "decision": decision,
              "trained_new_component": False, "actual_signal_schema_linking_evaluated": False,
              "rejected_joint_predictor_used": False, "opened_acceptance_or_final": False,
              "schema_linker_changed": False, "typed_ir_changed": False,
              "executor_changed": False, "time_component_changed": False,
              "development_set_status": "iteratively used; not untouched generalization evidence",
              "baseline_equivalence": "metric-exact deterministic reconstruction, not checkpoint bit identity"}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = ["# Grammar-Constrained Selection Projector", "",
             "The projector uses only existing structured predictions. It never reads question text or invents a missing entity/order. No model was trained.", "",
             "| Configuration | Skeleton exact | Omission | Addition | Overall invalid | Selection valid |",
             "|---|---:|---:|---:|---:|---:|",
             f"| A/B baseline + validate-only | {100*baseline_summary['skeleton_exact']['mean']:.1f}% | {100*baseline_summary['field_omission_rate']['mean']:.1f}% | {100*baseline_summary['unsupported_addition_rate']['mean']:.1f}% | {100*baseline_summary['invalid_composition_rate']['mean']:.1f}% | {100*statistics.mean(run['validate_only']['selection']['valid_rate'] for run in runs):.1f}% |",
             f"| C selection projection | {100*projection_summary['skeleton_exact']['mean']:.1f}% | {100*projection_summary['field_omission_rate']['mean']:.1f}% | {100*projection_summary['unsupported_addition_rate']['mean']:.1f}% | {100*projection_summary['invalid_composition_rate']['mean']:.1f}% | {100*statistics.mean(run['projection']['selection']['valid_rate'] for run in runs):.1f}% |",
             "", "## Per-seed effects", ""]
    for run, exact_delta, invalid_delta in zip(runs, exact_deltas, invalid_deltas):
        lines.append(f"- Seed {run['seed']}: exact {100*run['baseline']['skeleton_exact']:.1f}% → {100*run['projection']['skeleton_exact']:.1f}% ({100*exact_delta:+.1f}); invalid {100*run['baseline']['invalid_composition_rate']:.1f}% → {100*run['projection']['invalid_composition_rate']:.1f}% ({100*invalid_delta:+.1f}); repairs {run['repairs']}, regressions {run['regressions']}, changed {run['audit']['changed']}, unresolved {run['audit']['unresolved']}.")
    lines += ["", "## Detected input cases", ""]
    total = len(SEEDS) * len(eval_rows)
    for name, count in violation_totals.most_common():
        lines.append(f"- {name}: {count}/{total} ({100*count/total:.1f}%).")
    lines += ["", "## Gates", ""] + [f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items()]
    lines += ["", f"**Decision: {decision}**", "",
              "A validity improvement alone does not establish semantic correction. Unresolved cases are left untouched rather than filled from defaults or question phrases."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"deltas": result["deltas"], "gates": gates, "decision": decision,
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

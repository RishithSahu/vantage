"""Structural field-set and target-arity oracle study on the accepted time baseline."""
from __future__ import annotations

import copy
import json
import os
import statistics

import torch

from .operator_aggregation_oracle import score
from .supervised_parser import (
    ROOT, StructuralParser, diverse_prefix, evaluate as evaluate_structure,
    evaluation_rows, train as train_structure, training_pool,
)
from .time_curriculum import TimeParser, _natural, contrast_pairs, train as train_time
from .time_integration_ablation import predict_time, score as score_time


SEEDS = (701, 702, 703)
CONFIGS = ("A_baseline", "B_gold_structural_fields", "C_gold_target_arity", "D_gold_fields_and_arity")

# These are the structural presence/attachment fields predicted by the current
# skeleton. Actual signal identity is deliberately absent and remains the frozen
# schema linker's responsibility.
FIELD_SELECTION_FIELDS = (
    "condition_shape", "select_entity", "select_order", "select_k",
    "groups_count", "event", "before", "after", "refs",
)
ARITY_FIELD = "target_count"


def substitute(base, rows, fields=False, arity=False):
    out = []
    for pred, row in zip(base, rows):
        p = copy.deepcopy(pred)
        if fields:
            for name in FIELD_SELECTION_FIELDS:
                p[name] = row["labels"][name]
        if arity:
            p[ARITY_FIELD] = row["labels"][ARITY_FIELD]
        out.append(p)
    return out


def field_set_correct(pred, gold):
    return all(pred[name] == gold[name] for name in FIELD_SELECTION_FIELDS)


def interaction_diagnostic(rows, predictions):
    counts = {
        "correct_operator_incorrect_field_set": 0,
        "incorrect_operator_correct_field_set": 0,
        "correct_field_set_incorrect_arity": 0,
        "operator_and_field_set_incorrect": 0,
        "correct_operator_and_field_set": 0,
        "correct_operator_field_set_and_arity": 0,
    }
    for row, pred in zip(rows, predictions):
        gold = row["labels"]
        op_ok = pred["op"] == gold["op"]
        fields_ok = field_set_correct(pred, gold)
        arity_ok = pred[ARITY_FIELD] == gold[ARITY_FIELD]
        counts["correct_operator_incorrect_field_set"] += op_ok and not fields_ok
        counts["incorrect_operator_correct_field_set"] += (not op_ok) and fields_ok
        counts["correct_field_set_incorrect_arity"] += fields_ok and not arity_ok
        counts["operator_and_field_set_incorrect"] += (not op_ok) and not fields_ok
        counts["correct_operator_and_field_set"] += op_ok and fields_ok
        counts["correct_operator_field_set_and_arity"] += op_ok and fields_ok and arity_ok
    n = len(rows)
    return {name: {"count": value, "rate": value / n} for name, value in counts.items()}


def main():
    output = os.path.join(ROOT, "results", "field_arity_oracle")
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
        struct_eval = evaluate_structure(structural, eval_rows, device)

        torch.manual_seed(seed + 1000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 1000)
        time_model = TimeParser(model_path)
        train_time(time_model, natural_time_train + synthetic_time, device, seed + 1000)
        time_predictions = predict_time(time_model, natural_time_eval, device)
        integrated = score_time(eval_rows, [p["pred"] for p in struct_eval["predictions"]],
                                time_predictions, "predicted_time")
        baseline_predictions = [d["pred"] for d in integrated.pop("details")]
        configurations = {
            "A_baseline": baseline_predictions,
            "B_gold_structural_fields": substitute(baseline_predictions, eval_rows, fields=True),
            "C_gold_target_arity": substitute(baseline_predictions, eval_rows, arity=True),
            "D_gold_fields_and_arity": substitute(baseline_predictions, eval_rows, fields=True, arity=True),
        }
        scored = {name: score(eval_rows, predictions) for name, predictions in configurations.items()}
        frozen = reference_by_seed[seed]
        for metric in ("skeleton_exact", "operator_accuracy", "field_omission_rate",
                       "unsupported_addition_rate", "invalid_composition_rate"):
            if abs(scored["A_baseline"][metric] - frozen[metric]) > 1e-12:
                raise RuntimeError(f"Baseline reproduction failed seed={seed} metric={metric}")

        baseline_details = scored["A_baseline"]["details"]
        effects = {}
        for name in CONFIGS[1:]:
            details = scored[name]["details"]
            effects[name] = {
                "repairs": sum((not a["exact"]) and b["exact"] for a, b in zip(baseline_details, details)),
                "regressions": sum(a["exact"] and not b["exact"] for a, b in zip(baseline_details, details)),
                "invalid_fixed": sum(a["invalid"] and not b["invalid"] for a, b in zip(baseline_details, details)),
                "invalid_created": sum((not a["invalid"]) and b["invalid"] for a, b in zip(baseline_details, details)),
            }
        for item in scored.values(): item.pop("details")
        runs.append({"seed": seed, "configurations": scored, "effects": effects,
                     "interaction_diagnostic": interaction_diagnostic(eval_rows, baseline_predictions),
                     "eligible_gold_field_substitutions": len(eval_rows),
                     "eligible_gold_arity_substitutions": len(eval_rows),
                     "baseline_reproduction_passed": True})
        del structural, time_model
        if device == "cuda": torch.cuda.empty_cache()

    metrics = ("skeleton_exact", "operator_accuracy", "aggregation_accuracy", "target_arity_accuracy",
               "field_omission_rate", "unsupported_addition_rate", "invalid_composition_rate")
    summary = {}
    for config in CONFIGS:
        summary[config] = {}
        for metric in metrics:
            values = [run["configurations"][config][metric] for run in runs]
            summary[config][metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    base = summary["A_baseline"]["skeleton_exact"]["mean"]
    field_gain = summary["B_gold_structural_fields"]["skeleton_exact"]["mean"] - base
    arity_gain = summary["C_gold_target_arity"]["skeleton_exact"]["mean"] - base
    joint_gain = summary["D_gold_fields_and_arity"]["skeleton_exact"]["mean"] - base
    synergy = joint_gain - field_gain - arity_gain
    if max(field_gain, arity_gain, joint_gain) < .02:
        decision = "reassess_failure_taxonomy"
    elif synergy >= .02:
        decision = "prioritize_field_arity_interaction"
    elif field_gain >= arity_gain:
        decision = "investigate_structural_field_selection"
    else:
        decision = "isolate_target_arity"

    interaction_summary = {}
    for name in runs[0]["interaction_diagnostic"]:
        values = [run["interaction_diagnostic"][name]["rate"] for run in runs]
        interaction_summary[name] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    result = {"device": device, "seeds": list(SEEDS), "questions": len(eval_rows),
              "field_selection_definition": list(FIELD_SELECTION_FIELDS),
              "actual_signal_schema_linking_evaluated": False,
              "runs": runs, "summary": summary, "interaction_summary": interaction_summary,
              "oracle_gains": {"structural_fields": field_gain, "target_arity": arity_gain,
                               "joint": joint_gain, "joint_synergy": synergy},
              "next_decision": decision, "rejected_joint_predictor_used": False,
              "opened_acceptance_or_final": False, "schema_linker_changed": False,
              "typed_ir_changed": False, "executor_changed": False, "time_component_changed": False,
              "baseline_equivalence": "metric-exact deterministic reconstruction, not checkpoint bit identity"}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = ["# Structural Field-Set / Target-Arity Oracle", "",
             "The accepted predicted-time baseline was reconstructed with exact metric agreement for seeds 701–703. This establishes metric consistency, not bit-for-bit checkpoint identity. The rejected joint operator predictor was not used.", "",
             "`Structural fields` here means condition shape, entity-selection structure, grouping, event/window presence, and dialogue references. Actual signal identity remains the frozen schema linker's responsibility and is not evaluated by this oracle.", "",
             "| Configuration | Skeleton exact | Arity | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        s = summary[config]
        lines.append(f"| {config} | {100*s['skeleton_exact']['mean']:.1f}% ({100*s['skeleton_exact']['min']:.1f}–{100*s['skeleton_exact']['max']:.1f}) | {100*s['target_arity_accuracy']['mean']:.1f}% | {100*s['field_omission_rate']['mean']:.1f}% | {100*s['unsupported_addition_rate']['mean']:.1f}% | {100*s['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Oracle gains", "",
              f"- Gold structural field set: {100*field_gain:+.1f} skeleton-exact points.",
              f"- Gold target arity: {100*arity_gain:+.1f} points.",
              f"- Gold fields + arity: {100*joint_gain:+.1f} points.",
              f"- Joint synergy beyond individual gains: {100*synergy:+.1f} points.", "",
              "## Per-seed exactness", ""]
    for run in runs:
        c = run["configurations"]
        lines.append(f"- Seed {run['seed']}: A {100*c['A_baseline']['skeleton_exact']:.1f}%, B {100*c['B_gold_structural_fields']['skeleton_exact']:.1f}%, C {100*c['C_gold_target_arity']['skeleton_exact']:.1f}%, D {100*c['D_gold_fields_and_arity']['skeleton_exact']:.1f}%.")
    lines += ["", "## Baseline interaction diagnostic", ""]
    for name, values in interaction_summary.items():
        lines.append(f"- {name}: {100*values['mean']:.1f}% mean ({100*values['min']:.1f}–{100*values['max']:.1f}%).")
    lines += ["", f"**Next decision: {decision}**", "",
              "Oracle substitutions are diagnostic ceilings, not deployable model performance."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"oracle_gains": result["oracle_gains"], "decision": decision,
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

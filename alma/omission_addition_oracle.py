"""Presence-only and highest-count structural-family oracle diagnostics."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict

from .condition_shape_updated_oracle import load_frozen_predictions
from .operator_aggregation_oracle import changes, score
from .selection_variant_experiment import SEEDS
from .supervised_parser import LABELS, ROOT, evaluation_rows


FAMILIES = {
    "condition_structure": ("condition_shape",),
    "selection_structure": ("select_entity", "select_order", "select_k"),
    "event_window": ("time_type", "time_unit", "time2_type", "event", "before", "after"),
    "operator_aggregation": ("op", "agg", "target_count"),
    "other_structural": ("groups_count", "refs"),
}
METRICS = ("skeleton_exact", "field_omission_rate", "unsupported_addition_rate",
           "invalid_composition_rate")
MEANINGFUL_GAIN = .02
# `op=none` denotes an operation outside the recognized label vocabulary, not
# absence of an optional operation object. Primary time is likewise required.
# They remain in the diagnostic breakdown and family oracle, but are never
# changed by the presence-only intervention.
NON_PRESENCE_CONTENT_FIELDS = {"op", "time_type"}


def verify_family_partition():
    flattened = [field for fields in FAMILIES.values() for field in fields]
    return {"disjoint": len(flattened) == len(set(flattened)),
            "complete": set(flattened) == set(LABELS),
            "fields": flattened}


def field_error_counts(rows, predictions):
    omissions = Counter(); additions = Counter()
    family_omissions = Counter(); family_additions = Counter()
    family_missing_questions = Counter(); family_spurious_questions = Counter()
    field_to_family = {field: family for family, fields in FAMILIES.items() for field in fields}
    for row, prediction in zip(rows, predictions):
        gold = row["labels"]
        missing_families = set(); spurious_families = set()
        for field in LABELS:
            if gold[field] != "none" and prediction[field] == "none":
                omissions[field] += 1; family_omissions[field_to_family[field]] += 1
                missing_families.add(field_to_family[field])
            if gold[field] == "none" and prediction[field] != "none":
                additions[field] += 1; family_additions[field_to_family[field]] += 1
                spurious_families.add(field_to_family[field])
        family_missing_questions.update(missing_families)
        family_spurious_questions.update(spurious_families)
    families = {}
    for family, fields in FAMILIES.items():
        families[family] = {
            "omitted_field_instances": family_omissions[family],
            "unsupported_field_instances": family_additions[family],
            "total_field_instances": family_omissions[family] + family_additions[family],
            "questions_with_omission": family_missing_questions[family],
            "questions_with_addition": family_spurious_questions[family],
            "fields": {field: {"omitted": omissions[field], "unsupported": additions[field],
                               "total": omissions[field] + additions[field]}
                       for field in fields},
        }
    return {"families": families, "field_omissions": dict(omissions),
            "field_additions": dict(additions)}


def presence_only(rows, predictions):
    """Suppress false-positive fields; never synthesize a missing field's value."""
    output = []; audits = []
    for row, prediction in zip(rows, predictions):
        gold = row["labels"]; updated = copy.deepcopy(prediction)
        suppressed = []; unresolved = []
        for field in LABELS:
            if field in NON_PRESENCE_CONTENT_FIELDS:
                continue
            if gold[field] == "none" and prediction[field] != "none":
                updated[field] = "none"; suppressed.append(field)
            elif gold[field] != "none" and prediction[field] == "none":
                unresolved.append(field)
        output.append(updated)
        audits.append({"suppressed_false_positive_fields": suppressed,
                       "unresolved_missing_fields": unresolved})
    return output, audits


def family_oracle(rows, predictions, family):
    output = []
    for row, prediction in zip(rows, predictions):
        updated = copy.deepcopy(prediction)
        for field in FAMILIES[family]:
            updated[field] = row["labels"][field]
        output.append(updated)
    return output


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(runs, configuration):
    return {metric: {"mean": statistics.mean(run["configurations"][configuration][metric] for run in runs),
                     "min": min(run["configurations"][configuration][metric] for run in runs),
                     "max": max(run["configurations"][configuration][metric] for run in runs)}
            for metric in METRICS}


def main():
    output = os.path.join(ROOT, "results", "omission_addition_oracle")
    os.makedirs(output, exist_ok=True)
    partition = verify_family_partition()
    if not partition["disjoint"] or not partition["complete"]:
        raise RuntimeError(f"Error-family partition is invalid: {partition}")
    rows = evaluation_rows()
    selection_dir = os.path.join(ROOT, "results", "selection_variant")
    source_manifest, frozen_by_seed, _ = load_frozen_predictions(selection_dir, rows)

    aggregate_omissions = Counter(); aggregate_additions = Counter()
    baseline_by_seed = {}
    breakdown_by_seed = {}
    for seed in SEEDS:
        predictions = [item["accepted_prediction"] for item in frozen_by_seed[seed]]
        baseline_by_seed[seed] = predictions
        breakdown = field_error_counts(rows, predictions)
        breakdown_by_seed[seed] = breakdown
        aggregate_omissions.update(breakdown["field_omissions"])
        aggregate_additions.update(breakdown["field_additions"])
    family_totals = {
        family: sum(aggregate_omissions[field] + aggregate_additions[field] for field in fields)
        for family, fields in FAMILIES.items()
    }
    selected_family = max(FAMILIES, key=lambda family: (family_totals[family], family))

    runs = []; saved_predictions = []
    aggregate_suppressed = Counter(); aggregate_unresolved = Counter()
    for seed in SEEDS:
        baseline_predictions = baseline_by_seed[seed]
        presence_predictions, presence_audits = presence_only(rows, baseline_predictions)
        target_predictions = family_oracle(rows, baseline_predictions, selected_family)
        prediction_sets = {"A_frozen_baseline": baseline_predictions,
                           "B_presence_only": presence_predictions,
                           "C_targeted_family": target_predictions}
        scored = {name: score(rows, predictions) for name, predictions in prediction_sets.items()}
        effects = {name: changes(scored["A_frozen_baseline"]["details"], scored[name]["details"])
                   for name in ("B_presence_only", "C_targeted_family")}
        for index, (row, baseline, presence, target, audit) in enumerate(zip(
                rows, baseline_predictions, presence_predictions, target_predictions, presence_audits)):
            aggregate_suppressed.update(audit["suppressed_false_positive_fields"])
            aggregate_unresolved.update(audit["unresolved_missing_fields"])
            saved_predictions.append({"seed": seed, "index": index, "text": row["text"],
                                      "family": row["family"], "gold_ir": row["ir"],
                                      "gold_labels": row["labels"], "baseline": baseline,
                                      "presence_only": presence, "targeted_family": target,
                                      "presence_audit": audit,
                                      "baseline_exact": scored["A_frozen_baseline"]["details"][index]["exact"],
                                      "presence_exact": scored["B_presence_only"]["details"][index]["exact"],
                                      "targeted_family_exact": scored["C_targeted_family"]["details"][index]["exact"]})
        for value in scored.values(): value.pop("details")
        runs.append({"seed": seed, "configurations": scored, "effects": effects,
                     "error_breakdown": breakdown_by_seed[seed],
                     "presence_suppressed_field_instances": sum(len(a["suppressed_false_positive_fields"])
                                                                 for a in presence_audits),
                     "presence_unresolved_field_instances": sum(len(a["unresolved_missing_fields"])
                                                                 for a in presence_audits),
                     "presence_unresolved_questions": sum(bool(a["unresolved_missing_fields"])
                                                            for a in presence_audits)})

    configurations = ("A_frozen_baseline", "B_presence_only", "C_targeted_family")
    summary = {configuration: summarize(runs, configuration) for configuration in configurations}
    baseline_exact = summary["A_frozen_baseline"]["skeleton_exact"]["mean"]
    presence_gain = summary["B_presence_only"]["skeleton_exact"]["mean"] - baseline_exact
    target_gain = summary["C_targeted_family"]["skeleton_exact"]["mean"] - baseline_exact
    aggregate_breakdown = {
        "field_omissions": dict(aggregate_omissions),
        "field_additions": dict(aggregate_additions),
        "families": {family: {
            "fields": list(fields),
            "omitted_field_instances": sum(aggregate_omissions[field] for field in fields),
            "unsupported_field_instances": sum(aggregate_additions[field] for field in fields),
            "total_field_instances": family_totals[family],
            "seed_question_evaluations_with_omission": sum(
                breakdown_by_seed[seed]["families"][family]["questions_with_omission"] for seed in SEEDS),
            "seed_question_evaluations_with_addition": sum(
                breakdown_by_seed[seed]["families"][family]["questions_with_addition"] for seed in SEEDS)}
            for family, fields in FAMILIES.items()},
    }
    presence_invalid_delta = (summary["B_presence_only"]["invalid_composition_rate"]["mean"] -
                              summary["A_frozen_baseline"]["invalid_composition_rate"]["mean"])
    if presence_gain >= MEANINGFUL_GAIN and presence_invalid_delta > 0:
        next_target = "compatibility_aware_false_positive_control_with_calibrated_abstention"
    elif presence_gain >= MEANINGFUL_GAIN:
        next_target = "false_positive_control_and_calibrated_abstention"
    elif target_gain >= MEANINGFUL_GAIN:
        next_target = f"semantic_construction_and_interactions_for_{selected_family}"
    else:
        next_target = "representation_and_annotation_audit"
    result = {
        "experiment": "omission/addition decomposition with conservative presence oracle",
        "seeds": list(SEEDS), "questions": len(rows), "family_partition": FAMILIES,
        "family_partition_verified": partition,
        "aggregate_error_breakdown": aggregate_breakdown,
        "selected_target_family": selected_family,
        "target_selection_rule": "largest aggregate omitted + unsupported field-instance count",
        "runs": runs, "summary": summary,
        "oracle_gains": {"presence_only": presence_gain, "targeted_family": target_gain,
                         "presence_invalid_composition_delta": presence_invalid_delta},
        "presence_audit": {"suppressed_by_field": dict(aggregate_suppressed),
                           "unresolved_missing_by_field": dict(aggregate_unresolved),
                           "suppressed_field_instances": sum(aggregate_suppressed.values()),
                           "unresolved_missing_field_instances": sum(aggregate_unresolved.values())},
        "next_learning_target": next_target,
        "meaningful_gain_threshold": MEANINGFUL_GAIN,
        "presence_oracle_constraint": "suppression only; missing values are never copied from gold",
        "trained_new_component": False, "condition_state_predictor_used": False,
        "selection_variant_retained": True, "time_component_changed": False,
        "schema_linker_changed": False, "operator_aggregation_changed": False,
        "typed_ir_changed": False, "executor_changed": False,
        "acceptance_or_final_opened": False,
        "development_set_status": "iteratively used; not untouched generalization evidence",
        "source_selection_manifest_sha256": sha256_file(os.path.join(selection_dir, "MANIFEST.json")),
    }
    metrics_path = os.path.join(output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    predictions_path = os.path.join(output, "predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in saved_predictions: f.write(json.dumps(item, sort_keys=True) + "\n")
    manifest = {"format_version": 1, "experiment": "omission_addition_oracle",
                "source_selection_manifest_sha256": result["source_selection_manifest_sha256"],
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(predictions_path),
                "selected_target_family": selected_family,
                "trained_new_component": False, "acceptance_or_final_opened": False}
    with open(os.path.join(output, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    lines = ["# Omission/Addition and Presence-Only Oracle", "",
             "This diagnostic uses the exact persisted predictions of the accepted time + selection-variant parser. The rejected condition predictor is excluded. No model was trained.", "",
             "## Structural error breakdown", "",
             "Counts are field instances across 120 development questions × three seeds; one question may contribute multiple fields.", "",
             "| Family | Missing fields | Unsupported fields | Total fields | Missing cases | Unsupported cases |",
             "|---|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        entry = aggregate_breakdown["families"][family]
        lines.append(f"| {family} | {entry['omitted_field_instances']} | {entry['unsupported_field_instances']} | {entry['total_field_instances']} | {entry['seed_question_evaluations_with_omission']} | {entry['seed_question_evaluations_with_addition']} |")
    lines += ["", f"The mechanically selected highest-count family is **{selected_family}** ({family_totals[selected_family]} field errors).", "",
              "## Oracle configurations", "",
              "Presence-only suppresses fields proven absent by gold but never fills a missing value. Consequently, unresolved omissions receive no artificial exact-match credit.", "",
              "| Configuration | Skeleton exact | Omission | Addition | Invalid |",
              "|---|---:|---:|---:|---:|"]
    for configuration in configurations:
        values = summary[configuration]
        lines.append(f"| {configuration} | {100*values['skeleton_exact']['mean']:.1f}% ({100*values['skeleton_exact']['min']:.1f}–{100*values['skeleton_exact']['max']:.1f}) | {100*values['field_omission_rate']['mean']:.1f}% | {100*values['unsupported_addition_rate']['mean']:.1f}% | {100*values['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed exact-match effects", ""]
    for run in runs:
        base = run["configurations"]["A_frozen_baseline"]["skeleton_exact"]
        presence = run["configurations"]["B_presence_only"]["skeleton_exact"]
        target = run["configurations"]["C_targeted_family"]["skeleton_exact"]
        pe = run["effects"]["B_presence_only"]; te = run["effects"]["C_targeted_family"]
        lines.append(f"- Seed {run['seed']}: presence {100*base:.1f}%→{100*presence:.1f}% ({100*(presence-base):+.1f}; repairs/regressions {pe['repairs']}/{pe['regressions']}), targeted family {100*target:.1f}% ({100*(target-base):+.1f}; {te['repairs']}/{te['regressions']}); unresolved missing questions {run['presence_unresolved_questions']}.")
    lines += ["", "## Decision", "",
              f"Presence-only gain: {100*presence_gain:+.1f} points, with invalid composition changing by {100*presence_invalid_delta:+.1f} points. Complete {selected_family} oracle gain: {100*target_gain:+.1f} points.",
              f"Next target under the predefined 2-point threshold: **{next_target}**.", "",
              "Because independent suppression increases invalidity, the next experiment must couple false-positive control to typed compatibility and abstain/reject when a proposed deletion would leave an illegal signature. A free-standing per-field deletion head is not justified.", "",
              "### Selected bounded learning experiment", "",
              "Train one controller to choose `keep`, `suppress`, or `abstain` for predicted optional structures using the question together with the current typed signature. It must not insert missing structures or generate semantic contents. A suppression is executable only when the typed validator accepts the complete resulting signature; otherwise it becomes abstention/clarification. Report suppression precision and coverage separately, and integrate only if strict skeleton exactness improves in all three seeds while invalid composition does not increase in any seed.", "",
              "The targeted-family oracle copies gold values and is a diagnostic ceiling. The presence oracle does not. Development results are iterative; acceptance and final benchmarks remained sealed."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"family_totals": family_totals, "selected_family": selected_family,
                      "presence_gain": presence_gain, "target_gain": target_gain,
                      "next_target": next_target,
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

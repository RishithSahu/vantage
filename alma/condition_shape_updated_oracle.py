"""Condition-shape oracle on the persisted accepted selection-variant parser."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict

from vgpt.query_ir import Query
from .operator_aggregation_oracle import changes, score
from .selection_variant_experiment import SEEDS
from .supervised_parser import ROOT, evaluation_rows


METRICS = ("skeleton_exact", "field_omission_rate", "unsupported_addition_rate",
           "invalid_composition_rate")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def error_type(gold, predicted):
    if gold == predicted:
        return "correct"
    if gold == "none":
        return "spurious_condition"
    if predicted == "none":
        return "omitted_condition"
    if gold == "mixed" or predicted == "mixed":
        return "mixedness_error"
    gold_multi = gold.startswith("multi_")
    predicted_multi = predicted.startswith("multi_")
    gold_kind = gold.removeprefix("multi_")
    predicted_kind = predicted.removeprefix("multi_")
    if gold_multi != predicted_multi and gold_kind == predicted_kind:
        return "multiplicity_error"
    if gold_kind != predicted_kind and gold_multi == predicted_multi:
        return "named_threshold_kind_error"
    return "kind_and_multiplicity_error"


def grammar_record(row, predicted):
    query = Query.from_text(row["ir"])
    conditions = query.scope.conditions
    kinds = ["named" if condition.predicate else "threshold" for condition in conditions]
    return {
        "gold_shape": row["labels"]["condition_shape"],
        "predicted_shape": predicted,
        "condition_count": len(conditions),
        "condition_kinds": kinds,
        "condition_payloads": [condition.to_text() for condition in conditions],
        "error_type": error_type(row["labels"]["condition_shape"], predicted),
    }


def load_frozen_predictions(selection_dir, rows):
    manifest_path = os.path.join(selection_dir, "MANIFEST.json")
    predictions_path = os.path.join(selection_dir, "accepted_predictions.jsonl")
    metrics_path = os.path.join(selection_dir, "metrics.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if sha256_file(predictions_path) != manifest["accepted_predictions_sha256"]:
        raise RuntimeError("Accepted prediction hash does not match selection manifest")
    if sha256_file(metrics_path) != manifest["metrics_sha256"]:
        raise RuntimeError("Accepted metrics hash does not match selection manifest")
    dev_path = os.path.join(ROOT, "bench", "indep_dev.tsv")
    expected_dev_hash = manifest["dataset_sha256"]["bench/indep_dev.tsv"]
    if sha256_file(dev_path) != expected_dev_hash:
        raise RuntimeError("Development data changed after selection acceptance")
    grouped = defaultdict(list)
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            grouped[item["seed"]].append(item)
    for seed in SEEDS:
        grouped[seed].sort(key=lambda item: item["index"])
        if len(grouped[seed]) != len(rows):
            raise RuntimeError(f"Wrong persisted row count for seed {seed}")
        for index, (persisted, current) in enumerate(zip(grouped[seed], rows)):
            if persisted["index"] != index or persisted["text"] != current["text"]:
                raise RuntimeError(f"Persisted/current evaluation alignment failed at seed {seed}, row {index}")
            if persisted["gold_labels"] != current["labels"] or persisted["gold_ir"] != current["ir"]:
                raise RuntimeError(f"Persisted gold data mismatch at seed {seed}, row {index}")
    return manifest, grouped, metrics_path


def summarize(runs, configuration):
    fields = METRICS + ("condition_accuracy",)
    return {metric: {"mean": statistics.mean(run[configuration][metric] for run in runs),
                     "min": min(run[configuration][metric] for run in runs),
                     "max": max(run[configuration][metric] for run in runs)}
            for metric in fields}


def main():
    output = os.path.join(ROOT, "results", "condition_shape_updated_oracle")
    os.makedirs(output, exist_ok=True)
    selection_dir = os.path.join(ROOT, "results", "selection_variant")
    rows = evaluation_rows()
    manifest, grouped, accepted_metrics_path = load_frozen_predictions(selection_dir, rows)
    with open(accepted_metrics_path, encoding="utf-8") as f:
        accepted_metrics = json.load(f)
    accepted_by_seed = {run["seed"]: run["candidate"] for run in accepted_metrics["runs"]}

    runs = []
    oracle_rows = []
    confusion = Counter()
    error_types = Counter()
    gold_shape_counts = Counter()
    grammar_counts = Counter()
    for seed in SEEDS:
        frozen = grouped[seed]
        baseline_predictions = [item["accepted_prediction"] for item in frozen]
        oracle_predictions = []
        for row, prediction in zip(rows, baseline_predictions):
            updated = copy.deepcopy(prediction)
            updated["condition_shape"] = row["labels"]["condition_shape"]
            oracle_predictions.append(updated)

        baseline = score(rows, baseline_predictions)
        oracle = score(rows, oracle_predictions)
        baseline["condition_accuracy"] = sum(
            pred["condition_shape"] == row["labels"]["condition_shape"]
            for pred, row in zip(baseline_predictions, rows)) / len(rows)
        oracle["condition_accuracy"] = 1.0
        for metric in METRICS:
            if abs(baseline[metric] - accepted_by_seed[seed][metric]) > 1e-12:
                raise RuntimeError(f"Persisted accepted baseline differs for seed={seed}, metric={metric}")

        delta = changes(baseline["details"], oracle["details"])
        for index, (row, base_pred, oracle_pred, base_detail, oracle_detail) in enumerate(zip(
                rows, baseline_predictions, oracle_predictions, baseline["details"], oracle["details"])):
            grammar = grammar_record(row, base_pred["condition_shape"])
            confusion[(grammar["gold_shape"], grammar["predicted_shape"])] += 1
            error_types[grammar["error_type"]] += 1
            gold_shape_counts[grammar["gold_shape"]] += 1
            grammar_counts[(grammar["condition_count"], tuple(grammar["condition_kinds"]))] += 1
            oracle_rows.append({"seed": seed, "index": index, "text": row["text"],
                                "family": row["family"], "grammar": grammar,
                                "accepted_prediction": base_pred, "oracle_prediction": oracle_pred,
                                "baseline_exact": base_detail["exact"],
                                "oracle_exact": oracle_detail["exact"]})
        baseline.pop("details"); oracle.pop("details")
        runs.append({"seed": seed, "baseline": baseline, "condition_oracle": oracle,
                     "effects": delta,
                     "skeleton_gain": oracle["skeleton_exact"] - baseline["skeleton_exact"]})

    summary = {name: summarize(runs, name) for name in ("baseline", "condition_oracle")}
    gains = [run["skeleton_gain"] for run in runs]
    mean_gain = statistics.mean(gains)
    meaningful = mean_gain >= .02
    result = {
        "experiment": "updated condition-shape oracle after accepted selection-variant integration",
        "seeds": list(SEEDS), "questions": len(rows), "trained_new_component": False,
        "source_predictions": "persisted accepted selection-variant outputs",
        "source_manifest_sha256": sha256_file(os.path.join(selection_dir, "MANIFEST.json")),
        "source_prediction_hash_verified": True, "source_dataset_hash_verified": True,
        "source_baseline_metric_equivalence_verified": True,
        "runs": runs, "summary": summary, "skeleton_gains": gains,
        "mean_skeleton_gain": mean_gain,
        "opportunity_persists_at_two_point_threshold": meaningful,
        "condition_error_analysis": {
            "counts_across_seed_question_evaluations": dict(error_types),
            "gold_shape_counts_across_seed_question_evaluations": dict(gold_shape_counts),
            "confusion": {f"{gold}->{predicted}": count
                          for (gold, predicted), count in sorted(confusion.items())},
            "typed_grammar_patterns": {
                f"count={count}|kinds={','.join(kinds) if kinds else 'none'}": frequency
                for (count, kinds), frequency in sorted(grammar_counts.items())},
            "scope": "primary Scope.conditions shape only; condition identity, threshold operator/value/unit, and attachment are not corrected",
        },
        "selection_variant_changed": False, "time_component_changed": False,
        "schema_linker_changed": False, "operator_aggregation_changed": False,
        "typed_ir_changed": False, "executor_changed": False,
        "acceptance_or_final_opened": False,
        "development_set_status": "iteratively used; not untouched generalization evidence",
    }
    metrics_path = os.path.join(output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    prediction_path = os.path.join(output, "predictions.jsonl")
    with open(prediction_path, "w", encoding="utf-8") as f:
        for item in oracle_rows:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    oracle_manifest = {
        "format_version": 1, "experiment": "condition_shape_updated_oracle",
        "configuration": {"A": "accepted selection variant + predicted condition shape",
                          "B": "same frozen prediction with only condition_shape replaced by gold"},
        "seeds": list(SEEDS), "questions_per_seed": len(rows),
        "source_manifest_sha256": result["source_manifest_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(prediction_path),
        "accepted_or_final_opened": False,
    }
    with open(os.path.join(output, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(oracle_manifest, f, indent=2)

    lines = ["# Updated Condition-Shape Oracle", "",
             "This diagnostic starts from the exact persisted outputs of the accepted selection-variant parser. Their prediction, dataset, and metric hashes were verified before the substitution. No model was trained.", "",
             "| Configuration | Skeleton exact | Condition accuracy | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "condition_oracle"):
        s = summary[name]
        lines.append(f"| {name} | {100*s['skeleton_exact']['mean']:.1f}% ({100*s['skeleton_exact']['min']:.1f}–{100*s['skeleton_exact']['max']:.1f}) | {100*s['condition_accuracy']['mean']:.1f}% | {100*s['field_omission_rate']['mean']:.1f}% | {100*s['unsupported_addition_rate']['mean']:.1f}% | {100*s['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed complete-program effects", ""]
    for run in runs:
        lines.append(f"- Seed {run['seed']}: {100*run['baseline']['skeleton_exact']:.1f}% → {100*run['condition_oracle']['skeleton_exact']:.1f}% ({100*run['skeleton_gain']:+.1f}); repairs {run['effects']['repairs']}, regressions {run['effects']['regressions']}; invalid fixed/created {run['effects']['invalid_fixed']}/{run['effects']['invalid_created']}.")
    lines += ["", "## Typed-grammar diagnosis", "",
              f"Across 360 seed/question evaluations, error categories were: " + ", ".join(f"{name}={count}" for name, count in error_types.most_common()) + ".",
              "The shape label distinguishes no condition, one named predicate, one threshold, repeated named predicates, repeated thresholds, and mixed conditions. It does not evaluate the actual predicate/signal, comparison operator, value, unit, or attachment.", "",
              f"**Decision:** {'the condition-shape opportunity persists' if meaningful else 'the prior condition-shape opportunity has largely disappeared'}; mean oracle gain is {100*mean_gain:+.1f} skeleton-exact points. " + ("A bounded targeted condition-shape predictor is justified as the next experiment, but has not been trained here." if meaningful else "Reassess other structural errors before training a condition component."), "",
              "Oracle performance is a diagnostic ceiling, not deployable model performance. The development set is iterative; acceptance and final benchmarks remained sealed."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"gains": gains, "mean_gain": mean_gain, "meaningful": meaningful,
                      "error_types": dict(error_types),
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

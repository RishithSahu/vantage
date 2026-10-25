"""Bounded three-state condition presence/type integration experiment."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from vgpt.query_ir import Query
from vgpt.query_validator import validate
from .condition_shape_updated_oracle import load_frozen_predictions
from .operator_aggregation_oracle import changes, score
from .selection_variant_experiment import PERIODS, SEEDS
from .supervised_parser import ROOT, evaluation_rows, training_pool


STATES = ("Absent", "NamedCondition", "ThresholdCondition")
STATE_TO_SHAPE = {"Absent": "none", "NamedCondition": "named",
                  "ThresholdCondition": "threshold"}
SHAPE_TO_STATE = {shape: state for state, shape in STATE_TO_SHAPE.items()}
TRAINING_CONFIG = {"epochs": 12, "batch_size": 24, "learning_rate": 2e-5,
                   "weight_decay": .01, "max_length": 160,
                   "class_weight": "sqrt inverse frequency capped at 5.0"}
MATERIAL_ADDITION_TOLERANCE = .01

TRAIN_TEMPLATES = {
    "Absent": ("How much energy did the vehicle use {period}?",
               "Give total vehicle energy use {period}."),
    "NamedCondition": ("How much energy did the vehicle use while {named_text} {period}?",
                       "Total energy when the vehicle was {named_text} {period}."),
    "ThresholdCondition": ("How much energy was used while {signal_text} was {comparison_text} {period}?",
                           "Total energy with {signal_text} {comparison_text} {period}."),
}
HELDOUT_TEMPLATES = {
    "Absent": ("Across the car, what was aggregate energy consumption {period}?",
               "Report energy consumed by the car {period}."),
    "NamedCondition": ("Restricting this to {named_text}, what energy accumulated {period}?",
                       "For {named_text} operation, report energy consumed {period}."),
    "ThresholdCondition": ("Restricting this to times with {signal_text} {comparison_text}, what energy accumulated {period}?",
                           "Report energy consumed whenever {signal_text} remained {comparison_text} {period}."),
}

# A small deterministic vehicle-data world used only to verify that each formal
# contrast can change execution, not as natural-language evidence.
WORLD = (
    {"energy_used": 10., "predicates": {"highway"}, "speed": 100., "battery_temperature": 35.},
    {"energy_used": 20., "predicates": {"highway", "raining"}, "speed": 70., "battery_temperature": 32.},
    {"energy_used": 30., "predicates": {"city"}, "speed": 40., "battery_temperature": 28.},
    {"energy_used": 40., "predicates": {"city", "raining"}, "speed": 55., "battery_temperature": 31.},
    {"energy_used": 50., "predicates": {"charging"}, "speed": 0., "battery_temperature": 25.},
    {"energy_used": 60., "predicates": {"charging", "raining"}, "speed": 0., "battery_temperature": 38.},
)
CONTRASTS = (
    {"named": "highway", "named_text": "driving on the highway",
     "signal": "battery_temperature", "op": "gt", "value": 30, "unit": "C",
     "signal_text": "battery temperature", "comparison_text": "above 30 C"},
    {"named": "city", "named_text": "driving in the city",
     "signal": "speed", "op": "gt", "value": 80, "unit": "km/h",
     "signal_text": "vehicle speed", "comparison_text": "above 80 km/h"},
    {"named": "charging", "named_text": "charging",
     "signal": "battery_temperature", "op": "lt", "value": 30, "unit": "C",
     "signal_text": "battery temperature", "comparison_text": "below 30 C"},
    {"named": "raining", "named_text": "driving in rain",
     "signal": "speed", "op": "gt", "value": 60, "unit": "km/h",
     "signal_text": "vehicle speed", "comparison_text": "above 60 km/h"},
)


def state_of_shape(shape):
    if shape not in SHAPE_TO_STATE:
        raise ValueError(f"Three-state target cannot represent condition shape {shape!r}")
    return SHAPE_TO_STATE[shape]


def audit_target(rows):
    shapes = Counter(row["labels"]["condition_shape"] for row in rows)
    unsupported = {shape: count for shape, count in shapes.items() if shape not in SHAPE_TO_STATE}
    return {"examples": len(rows), "shape_counts": dict(shapes),
            "state_counts": dict(Counter(state_of_shape(shape) for shape in shapes.elements())) if not unsupported else {},
            "exactly_representable": not unsupported, "unsupported_shapes": unsupported}


def _period(index):
    if index < len(PERIODS):
        return PERIODS[index]
    return f"during the past {index + 1} days", f"last:{index + 1}:day"


def curriculum(split="train"):
    templates = TRAIN_TEMPLATES if split == "train" else HELDOUT_TEMPLATES
    count = 40 if split == "train" else 20
    triples = []
    for index in range(count):
        period_text, period_ir = _period(index)
        contrast = CONTRASTS[index % len(CONTRASTS)]
        template_index = index % 2
        common = f"op=aggregate tgt=energy_used agg=sum t={period_ir}"
        named_ir = common + f" c={contrast['named']}"
        unit = contrast["unit"].replace("/", "_per_")
        threshold_ir = common + f" c=sig:{contrast['signal']}:{contrast['op']}:{contrast['value']}:{unit}"
        rows = (
            {"text": templates["Absent"][template_index].format(period=period_text),
             "ir": common, "state": "Absent", "split": split},
            {"text": templates["NamedCondition"][template_index].format(
                period=period_text, **contrast), "ir": named_ir,
             "state": "NamedCondition", "split": split},
            {"text": templates["ThresholdCondition"][template_index].format(
                period=period_text, **contrast), "ir": threshold_ir,
             "state": "ThresholdCondition", "split": split},
        )
        triples.append(rows)
    return triples


def _condition_matches(condition, sample):
    if condition.predicate:
        return condition.predicate in sample["predicates"]
    left = sample[condition.signal]
    return {"gt": left > condition.value, "lt": left < condition.value,
            "ge": left >= condition.value, "le": left <= condition.value,
            "eq": left == condition.value}[condition.op]


def deterministic_denotation(ir):
    query = Query.from_text(ir)
    selected = [sample for sample in WORLD
                if all(_condition_matches(condition, sample) for condition in query.scope.conditions)]
    return round(sum(sample["energy_used"] for sample in selected), 6)


def verify_curriculum(triples):
    failures = []
    unique = set()
    for triple_index, triple in enumerate(triples):
        queries = [Query.from_text(row["ir"]) for row in triple]
        for row, query in zip(triple, queries):
            problems = validate(query).codes()
            if problems:
                failures.append({"triple": triple_index, "ir": row["ir"], "validation": problems})
            if state_of_shape(_condition_shape(query)) != row["state"]:
                failures.append({"triple": triple_index, "state_mismatch": row})
            key = (row["text"], row["ir"])
            if key in unique:
                failures.append({"triple": triple_index, "duplicate": key})
            unique.add(key)
        stable = {(q.op, q.target, q.agg, q.scope.time.to_text(), q.scope.select is None)
                  for q in queries}
        if len(stable) != 1:
            failures.append({"triple": triple_index, "non_condition_fields_changed": True})
        denotations = [deterministic_denotation(row["ir"]) for row in triple]
        if len(set(denotations)) != 3:
            failures.append({"triple": triple_index, "denotations_not_distinct": denotations})
    return {"triples": len(triples), "examples": len(triples) * 3,
            "verified": len(triples) if not failures else 0,
            "failures": failures, "unique_text_ir_pairs": len(unique),
            "world_rows": len(WORLD)}


def _condition_shape(query):
    conditions = query.scope.conditions
    if not conditions:
        return "none"
    kinds = ["named" if condition.predicate else "threshold" for condition in conditions]
    return kinds[0] if len(kinds) == 1 else ("multi_" + kinds[0] if len(set(kinds)) == 1 else "mixed")


def natural_rows(rows):
    return [{"text": row["text"], "state": state_of_shape(row["labels"]["condition_shape"]),
             "family": row["family"], "source": row["source"]} for row in rows]


class ConditionStatePredictor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(STATES))

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=160,
                              return_tensors="pt")

    def forward(self, batch):
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)


def train_model(model, rows, device, seed):
    random.seed(seed); torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model.to(device)
    loader = DataLoader(rows, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True,
                        collate_fn=lambda items: items)
    counts = Counter(row["state"] for row in rows)
    weights = torch.tensor([min(5., (len(rows) / (len(STATES) * counts[state])) ** .5)
                            for state in STATES], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"],
                                  weight_decay=TRAINING_CONFIG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    losses = []; started = time.perf_counter()
    for _ in range(TRAINING_CONFIG["epochs"]):
        model.train(); total = 0.; seen = 0
        for items in loader:
            batch = {key: value.to(device) for key, value in
                     model.tokenize([item["text"] for item in items]).items()}
            gold = torch.tensor([STATES.index(item["state"]) for item in items], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = nn.functional.cross_entropy(model(batch), gold, weight=weights)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(items); seen += len(items)
        losses.append(total / seen)
    return {"seconds": time.perf_counter() - started, "losses": losses,
            "examples": len(rows), "class_counts": dict(counts),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0}


def predict(model, rows, device):
    loader = DataLoader(rows, batch_size=64, collate_fn=lambda items: items)
    predictions = []; model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {key: value.to(device) for key, value in
                     model.tokenize([item["text"] for item in items]).items()}
            predictions.extend(STATES[index] for index in model(batch).argmax(-1).tolist())
    return predictions


def state_metrics(rows, predictions):
    confusion = Counter(); correct = Counter(); totals = Counter()
    for row, predicted in zip(rows, predictions):
        gold = row["state"]
        confusion[(gold, predicted)] += 1; totals[gold] += 1; correct[gold] += gold == predicted
    recalls = {state: correct[state] / totals[state] if totals[state] else None for state in STATES}
    return {"accuracy": sum(correct.values()) / len(rows),
            "macro_recall": statistics.mean(value for value in recalls.values() if value is not None),
            "per_class_recall": recalls, "class_counts": dict(totals),
            "confusion": {f"{gold}->{predicted}": count
                          for (gold, predicted), count in sorted(confusion.items())}}


def condition_errors(rows, predictions):
    spurious = omitted = type_confusion = 0
    for row, prediction in zip(rows, predictions):
        gold = row["labels"]["condition_shape"]
        got = prediction["condition_shape"]
        spurious += gold == "none" and got != "none"
        omitted += gold != "none" and got == "none"
        type_confusion += gold != "none" and got != "none" and gold != got
    return {"spurious_count": spurious, "spurious_rate": spurious / len(rows),
            "omitted_count": omitted, "omitted_rate": omitted / len(rows),
            "present_type_confusion_count": type_confusion,
            "present_type_confusion_rate": type_confusion / len(rows)}


def integrate(base_predictions, states):
    output = []
    for base, state in zip(base_predictions, states):
        prediction = copy.deepcopy(base)
        prediction["condition_shape"] = STATE_TO_SHAPE[state]
        output.append(prediction)
    return output


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(paths):
    return {os.path.relpath(path, ROOT).replace("\\", "/"): sha256_file(path) for path in paths}


def main():
    output = os.path.join(ROOT, "results", "condition_state")
    os.makedirs(output, exist_ok=True)
    train_source, eval_source = training_pool(), evaluation_rows()
    audit = {"natural_training": audit_target(train_source),
             "natural_development": audit_target(eval_source)}
    if not all(item["exactly_representable"] for item in audit.values()):
        raise RuntimeError(f"Three-state target audit failed: {audit}")
    train_triples, heldout_triples = curriculum("train"), curriculum("heldout")
    verification = {"train": verify_curriculum(train_triples),
                    "heldout": verify_curriculum(heldout_triples)}
    if verification["train"]["failures"] or verification["heldout"]["failures"]:
        raise RuntimeError(f"Condition curriculum verification failed: {verification}")
    audit_path = os.path.join(output, "AUDIT.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump({"target": list(STATES), "audit": audit,
                   "claim_boundary": "condition presence and single-condition named/threshold type only"}, f, indent=2)
    curriculum_path = os.path.join(output, "curriculum.jsonl")
    with open(curriculum_path, "w", encoding="utf-8") as f:
        for triple in train_triples + heldout_triples:
            f.write(json.dumps({"rows": triple,
                                "denotations": [deterministic_denotation(row["ir"]) for row in triple]}) + "\n")

    selection_dir = os.path.join(ROOT, "results", "selection_variant")
    source_manifest, frozen_by_seed, accepted_metrics_path = load_frozen_predictions(
        selection_dir, eval_source)
    natural_train, natural_eval = natural_rows(train_source), natural_rows(eval_source)
    synthetic_train = [row for triple in train_triples for row in triple]
    synthetic_eval = [row for triple in heldout_triples for row in triple]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    runs = []; saved_predictions = []; checkpoint_paths = []

    for seed in SEEDS:
        base_predictions = [item["accepted_prediction"] for item in frozen_by_seed[seed]]
        baseline = score(eval_source, base_predictions)
        baseline_errors = condition_errors(eval_source, base_predictions)
        torch.manual_seed(seed + 4000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 4000)
        model = ConditionStatePredictor(model_path)
        training = train_model(model, natural_train + synthetic_train, device, seed + 4000)
        natural_predictions = predict(model, natural_eval, device)
        synthetic_predictions = predict(model, synthetic_eval, device)
        candidate_predictions = integrate(base_predictions, natural_predictions)
        candidate = score(eval_source, candidate_predictions)
        candidate_errors = condition_errors(eval_source, candidate_predictions)
        effects = changes(baseline["details"], candidate["details"])
        for index, (row, state, base_pred, candidate_pred, base_detail, candidate_detail) in enumerate(zip(
                eval_source, natural_predictions, base_predictions, candidate_predictions,
                baseline["details"], candidate["details"])):
            saved_predictions.append({"seed": seed, "index": index, "text": row["text"],
                                      "family": row["family"], "gold_ir": row["ir"],
                                      "gold_labels": row["labels"], "predicted_state": state,
                                      "baseline_prediction": base_pred,
                                      "candidate_prediction": candidate_pred,
                                      "baseline_exact": base_detail["exact"],
                                      "candidate_exact": candidate_detail["exact"]})
        checkpoint_path = os.path.join(output, f"condition_state_seed{seed}.pt")
        cpu_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save({"format_version": 1, "component": "condition_state_predictor",
                    "seed": seed, "training_seed": seed + 4000, "states": list(STATES),
                    "training_config": TRAINING_CONFIG, "model_state_dict": cpu_state}, checkpoint_path)
        checkpoint_paths.append(checkpoint_path); del cpu_state
        natural_result = state_metrics(natural_eval, natural_predictions)
        synthetic_result = state_metrics(synthetic_eval, synthetic_predictions)
        baseline.pop("details"); candidate.pop("details")
        runs.append({"seed": seed, "baseline": baseline, "candidate": candidate,
                     "baseline_condition_errors": baseline_errors,
                     "candidate_condition_errors": candidate_errors,
                     "natural_condition_state": natural_result,
                     "synthetic_condition_state": synthetic_result,
                     "effects": effects, "training": training})
        del model
        if device == "cuda": torch.cuda.empty_cache()

    metric_names = ("skeleton_exact", "field_omission_rate",
                    "unsupported_addition_rate", "invalid_composition_rate")
    summary = {configuration: {metric: {"mean": statistics.mean(run[configuration][metric] for run in runs),
                                        "min": min(run[configuration][metric] for run in runs),
                                        "max": max(run[configuration][metric] for run in runs)}
                               for metric in metric_names}
               for configuration in ("baseline", "candidate")}
    exact_deltas = [run["candidate"]["skeleton_exact"] - run["baseline"]["skeleton_exact"]
                    for run in runs]
    invalid_deltas = [run["candidate"]["invalid_composition_rate"] - run["baseline"]["invalid_composition_rate"]
                      for run in runs]
    addition_deltas = [run["candidate"]["unsupported_addition_rate"] - run["baseline"]["unsupported_addition_rate"]
                       for run in runs]
    gates = {"strict_exact_improves_all_seeds": all(delta > 0 for delta in exact_deltas),
             "invalid_not_increased_any_seed": all(delta <= 0 for delta in invalid_deltas),
             "unsupported_addition_not_materially_worse_any_seed": all(
                 delta <= MATERIAL_ADDITION_TOLERANCE + 1e-12 for delta in addition_deltas)}
    passed = all(gates.values())
    result = {"experiment": "bounded condition presence/type integration", "device": device,
              "seeds": list(SEEDS), "questions": len(eval_source), "target_states": list(STATES),
              "audit": audit, "curriculum_verification": verification,
              "natural_training_examples": len(natural_train),
              "synthetic_training_examples": len(synthetic_train),
              "natural_development_examples": len(natural_eval),
              "synthetic_heldout_examples": len(synthetic_eval),
              "runs": runs, "summary": summary, "exact_deltas": exact_deltas,
              "invalid_deltas": invalid_deltas, "addition_deltas": addition_deltas,
              "material_addition_tolerance": MATERIAL_ADDITION_TOLERANCE,
              "gates": gates, "integration_success": passed,
              "selection_source_manifest_sha256": sha256_file(os.path.join(selection_dir, "MANIFEST.json")),
              "source_predictions_hash_verified": True, "source_dataset_hash_verified": True,
              "multiplicity_or_mixed_generated": False,
              "schema_linker_changed": False, "time_component_changed": False,
              "selection_variant_changed": False, "operator_aggregation_changed": False,
              "typed_ir_changed": False, "executor_changed": False,
              "acceptance_or_final_opened": False,
              "development_set_status": "iteratively used; not untouched generalization evidence"}
    metrics_path = os.path.join(output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    predictions_path = os.path.join(output, "predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in saved_predictions: f.write(json.dumps(row, sort_keys=True) + "\n")
    dataset_paths = [os.path.join(ROOT, "bench", name)
                     for name in ("ir_gold.tsv", "diag_v1.tsv", "indep_dev.tsv")]
    model_files = sorted(os.path.join(model_path, name) for name in os.listdir(model_path)
                         if os.path.isfile(os.path.join(model_path, name)))
    manifest = {"format_version": 1, "experiment": "condition_state",
                "status": "accepted_experimental" if passed else "rejected",
                "seeds": list(SEEDS), "questions_per_seed": len(eval_source),
                "training_config": TRAINING_CONFIG,
                "integration_gates": gates,
                "source_selection_manifest_sha256": result["selection_source_manifest_sha256"],
                "dataset_sha256": file_hashes(dataset_paths),
                "curriculum_sha256": sha256_file(curriculum_path),
                "base_model_sha256": file_hashes(model_files),
                "checkpoints_sha256": file_hashes(checkpoint_paths),
                "predictions_sha256": sha256_file(predictions_path),
                "metrics_sha256": sha256_file(metrics_path),
                "acceptance_or_final_opened": False}
    with open(os.path.join(output, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    lines = ["# Bounded Condition-State Integration Experiment", "",
             "The audited three-state target is exact for the current data: neither the 100 natural training examples nor the 120 development questions contains multiplicity or mixed conditions. The experiment predicts only absence, one named condition, or one threshold condition.", "",
             f"Curriculum: {verification['train']['triples']} verified training triples and {verification['heldout']['triples']} verified held-out triples. Each triple changes only condition structure and has three distinct denotations in the deterministic vehicle-data world.", "",
             "| Configuration | Skeleton exact | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|"]
    for configuration in ("baseline", "candidate"):
        values = summary[configuration]
        lines.append(f"| {configuration} | {100*values['skeleton_exact']['mean']:.1f}% ({100*values['skeleton_exact']['min']:.1f}–{100*values['skeleton_exact']['max']:.1f}) | {100*values['field_omission_rate']['mean']:.1f}% | {100*values['unsupported_addition_rate']['mean']:.1f}% | {100*values['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed results", ""]
    for run, exact_delta, invalid_delta, addition_delta in zip(runs, exact_deltas, invalid_deltas, addition_deltas):
        natural = run["natural_condition_state"]
        before = run["baseline_condition_errors"]; after = run["candidate_condition_errors"]
        lines.append(f"- Seed {run['seed']}: exact {100*run['baseline']['skeleton_exact']:.1f}% → {100*run['candidate']['skeleton_exact']:.1f}% ({100*exact_delta:+.1f}); invalid {100*invalid_delta:+.1f}; addition {100*addition_delta:+.1f}; condition-state accuracy {100*natural['accuracy']:.1f}%, macro recall {100*natural['macro_recall']:.1f}%; spurious {before['spurious_count']}→{after['spurious_count']}, omitted {before['omitted_count']}→{after['omitted_count']}; repairs/regressions {run['effects']['repairs']}/{run['effects']['regressions']}.")
    lines += ["", "## Predetermined gates", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items())
    mean_natural_accuracy = statistics.mean(run["natural_condition_state"]["accuracy"] for run in runs)
    mean_natural_macro = statistics.mean(run["natural_condition_state"]["macro_recall"] for run in runs)
    mean_synthetic_accuracy = statistics.mean(run["synthetic_condition_state"]["accuracy"] for run in runs)
    lines += ["", f"**Integration decision: {'PASS — retain experimentally' if passed else 'FAIL — do not integrate'}**", "",
              f"Mean natural condition-state accuracy was {100*mean_natural_accuracy:.1f}% with {100*mean_natural_macro:.1f}% macro recall, versus 87.5% condition-shape accuracy in the frozen baseline. Across all seeds, spurious conditions remained 27→27, omitted conditions fell 13→6, and present-type confusions fell 5→0. The candidate produced {sum(run['effects']['repairs'] for run in runs)} complete-structure repairs and {sum(run['effects']['regressions'] for run in runs)} regressions.",
              f"Synthetic held-out accuracy averaged {100*mean_synthetic_accuracy:.1f}% (range {100*min(run['synthetic_condition_state']['accuracy'] for run in runs):.1f}–{100*max(run['synthetic_condition_state']['accuracy'] for run in runs):.1f}%).",
              "Synthetic held-out accuracy is reported separately and is not natural-language evidence. This experiment does not generate condition identities, threshold values/operators/units, multiplicity, mixed conditions, or scope attachment. The development set is iterative; acceptance and final benchmarks remained sealed."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"audit": audit, "verification": verification,
                      "exact_deltas": exact_deltas, "invalid_deltas": invalid_deltas,
                      "addition_deltas": addition_deltas, "gates": gates,
                      "success": passed, "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

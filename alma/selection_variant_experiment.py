"""Audited three-class selection-variant curriculum and integration experiment."""
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
from .operator_aggregation_oracle import score
from .selection_projector import selection_validity
from .supervised_parser import (
    ROOT, StructuralParser, diverse_prefix, evaluate as evaluate_structure,
    evaluation_rows, train as train_structure, training_pool,
)
from .time_curriculum import TimeParser, _natural, contrast_pairs as time_contrast_pairs, train as train_time
from .time_integration_ablation import predict_time, score as score_time


SEEDS = (701, 702, 703)
VARIANTS = ("Absent", "TripSelection", "ChargeSessionSelection")
TRAINING_CONFIG = {"epochs": 12, "batch_size": 24, "learning_rate": 2e-5,
                   "weight_decay": .01, "max_length": 160,
                   "class_weight": "sqrt inverse frequency capped at 5.0"}
TRAIN_TEMPLATES = {
    "Absent": ("How much energy did the vehicle use {period}?",
               "Give total vehicle energy use {period}."),
    "TripSelection": ("How much energy was used by the {kword} trips ranked {direction} by {metric} {period}?",
                      "Total energy for the {kword} {direction} trips by {metric} {period}."),
    "ChargeSessionSelection": ("How much energy was used by the {kword} charge sessions ranked {direction} by {metric} {period}?",
                               "Total energy for the {kword} {direction} charging sessions by {metric} {period}."),
}
HELDOUT_TEMPLATES = {
    "Absent": ("What was the car's aggregate energy consumption {period}?",
               "Across the vehicle, how much energy accumulated {period}?"),
    "TripSelection": ("For the {kword} journeys ordered {direction} on {metric}, total their energy {period}.",
                      "Add up energy for the {kword} trips with {direction} {metric} {period}."),
    "ChargeSessionSelection": ("For the {kword} recharging sessions ordered {direction} on {metric}, total their energy {period}.",
                               "Add up energy for the {kword} charging sessions with {direction} {metric} {period}."),
}
PERIODS = (("last week", "cal:last_week"), ("this week", "cal:this_week"),
           ("last month", "cal:last_month"), ("this month", "cal:this_month"),
           ("yesterday", "cal:yesterday"), ("today", "cal:today"),
           ("during the past seven days", "last:7:day"),
           ("during the past thirty days", "last:30:day"),
           ("during the past two weeks", "last:2:week"),
           ("during the past twenty-four hours", "last:24:hour"))
TRIP_METRICS = (("efficiency", "wh_per_km"), ("distance", "distance_driven"),
                ("duration", "duration"), ("average speed", "avg_speed"),
                ("recency", "recency"))
CHARGE_METRICS = (("energy", "energy_used"), ("charging power", "charging_power"),
                  ("duration", "duration"), ("recency", "recency"))


def variant_of_query(query):
    selection = query.scope.select
    if selection is None: return "Absent"
    if selection.entity == "trip": return "TripSelection"
    if selection.entity == "charge_session": return "ChargeSessionSelection"
    raise ValueError(f"Unsupported selection entity: {selection.entity}")


def audit_rows(rows):
    classes, order, k_values = Counter(), Counter(), Counter()
    missing_payload = Counter(); ambiguous = []
    for row in rows:
        query = Query.from_text(row["ir"])
        try: variant = variant_of_query(query)
        except ValueError as exc:
            ambiguous.append({"ir": row["ir"], "reason": str(exc)}); continue
        classes[variant] += 1
        if query.scope.select:
            s = query.scope.select
            order[(variant, s.order)] += 1
            k_values[(variant, "none" if s.k is None else "one" if s.k == 1 else "many")] += 1
            if s.rank_by is not None: missing_payload["rank_by_not_in_skeleton"] += 1
            if s.klass is not None: missing_payload["klass_not_in_skeleton"] += 1
    return {"examples": len(rows), "class_counts": dict(classes),
            "unambiguous": len(ambiguous) == 0, "ambiguous": ambiguous,
            "order_counts": {"|".join(key): value for key, value in order.items()},
            "k_counts": {"|".join(key): value for key, value in k_values.items()},
            "representation_gaps": dict(missing_payload),
            "variant_target_expressible": True,
            "full_selection_validity_expressible": False,
            "reason": "select_entity expresses the variant; rank_by and klass are omitted from the skeleton"}


def _synthetic_row(text, ir, split, family):
    query = Query.from_text(ir)
    return {"text": text, "ir": ir, "variant": variant_of_query(query),
            "split": split, "family": family, "source": "formal_selection_variant"}


def curriculum(split="train"):
    templates = TRAIN_TEMPLATES if split == "train" else HELDOUT_TEMPLATES
    n = 40 if split == "train" else 20
    triples = []
    for i in range(n):
        if i < len(PERIODS):
            period_text, period_ir = PERIODS[i]
        else:
            period_text, period_ir = f"during the past {i + 1} days", f"last:{i + 1}:day"
        order = "desc" if i % 2 == 0 else "asc"
        direction = "highest" if order == "desc" else "lowest"
        k = i % 5 + 1
        kword = ("single" if k == 1 else str(k))
        trip_text, trip_metric = TRIP_METRICS[i % len(TRIP_METRICS)]
        charge_text, charge_metric = CHARGE_METRICS[i % len(CHARGE_METRICS)]
        ti = i % 2
        common = f"op=aggregate tgt=energy_used agg=sum t={period_ir}"
        absent = _synthetic_row(templates["Absent"][ti].format(period=period_text), common,
                                split, "selection_variant")
        trip = _synthetic_row(templates["TripSelection"][ti].format(
            kword=kword, direction=direction, metric=trip_text, period=period_text),
            common + f" sel=trip:{trip_metric}:{order}:{k}:-", split, "selection_variant")
        charge = _synthetic_row(templates["ChargeSessionSelection"][ti].format(
            kword=kword, direction=direction, metric=charge_text, period=period_text),
            common + f" sel=charge_session:{charge_metric}:{order}:{k}:-", split, "selection_variant")
        triples.append((absent, trip, charge))
    return triples


def deterministic_denotation(ir):
    query = Query.from_text(ir)
    variant = variant_of_query(query)
    return {"Absent": ("all_vehicle_energy", 210.0),
            "TripSelection": ("selected_trip_energy", 74.0),
            "ChargeSessionSelection": ("selected_charge_energy", 93.0)}[variant]


def verify_curriculum(triples):
    failures = []
    seen_text, seen_ir = set(), set()
    for triple in triples:
        variants = []
        denotations = []
        for row in triple:
            query = Query.from_text(row["ir"])
            problems = validate(query).codes()
            if problems: failures.append({"ir": row["ir"], "validation": problems})
            variants.append(row["variant"]); denotations.append(deterministic_denotation(row["ir"]))
            key = (row["text"], row["ir"])
            if key in seen_text: failures.append({"duplicate": key})
            seen_text.add(key); seen_ir.add(row["ir"])
        if set(variants) != set(VARIANTS): failures.append({"variants": variants})
        if len(set(denotations)) != 3: failures.append({"denotations": denotations})
        base = [Query.from_text(row["ir"]) for row in triple]
        stable = {(q.op, q.target, q.agg, q.scope.time.to_text()) for q in base}
        if len(stable) != 1: failures.append({"non_selection_fields_changed": [q.to_text() for q in base]})
    return {"triples": len(triples), "examples": 3 * len(triples),
            "verified": len(triples) if not failures else 0, "failures": failures,
            "unique_text_ir_pairs": len(seen_text), "unique_irs": len(seen_ir)}


class VariantPredictor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(VARIANTS))

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=160, return_tensors="pt")

    def forward(self, batch):
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)


def natural_variant_rows(rows):
    return [{"text": row["text"], "variant": variant_of_query(Query.from_text(row["ir"])),
             "source": row["source"], "family": row["family"]} for row in rows]


def train_variant(model, rows, device, seed, epochs=12):
    random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model.to(device)
    loader = DataLoader(rows, batch_size=24, shuffle=True, collate_fn=lambda x: x)
    counts = Counter(row["variant"] for row in rows)
    weights = torch.tensor([min(5.0, (len(rows) / (len(VARIANTS) * counts[v])) ** .5)
                            for v in VARIANTS], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    started = time.perf_counter(); losses = []
    for _ in range(epochs):
        model.train(); total = 0.; seen = 0
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokenize([x["text"] for x in items]).items()}
            gold = torch.tensor([VARIANTS.index(x["variant"]) for x in items], device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = nn.functional.cross_entropy(model(batch), gold, weight=weights)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss.detach()) * len(items); seen += len(items)
        losses.append(total / seen)
    return {"seconds": time.perf_counter() - started, "epochs": epochs, "examples": len(rows),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
            "losses": losses, "class_counts": dict(counts)}


def predict_variants(model, rows, device):
    loader = DataLoader(rows, batch_size=64, collate_fn=lambda x: x)
    out = []; model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokenize([x["text"] for x in items]).items()}
            out.extend(VARIANTS[i] for i in model(batch).argmax(-1).tolist())
    return out


def variant_metrics(rows, predictions):
    correct = Counter(); totals = Counter(); exact = 0
    for row, pred in zip(rows, predictions):
        gold = row["variant"]; totals[gold] += 1; correct[gold] += pred == gold; exact += pred == gold
    recalls = {v: correct[v] / totals[v] if totals[v] else None for v in VARIANTS}
    observed = [value for value in recalls.values() if value is not None]
    return {"accuracy": exact / len(rows), "macro_recall": statistics.mean(observed),
            "per_class_recall": recalls, "class_counts": dict(totals)}


def integrate_variant(base_predictions, variants):
    out = []; unresolved = 0
    for base, variant in zip(base_predictions, variants):
        pred = copy.deepcopy(base)
        if variant == "Absent":
            pred["select_entity"] = pred["select_order"] = pred["select_k"] = "none"
        else:
            pred["select_entity"] = "trip" if variant == "TripSelection" else "charge_session"
            # Existing order/k predictions are consumed only under a present variant.
            # Missing order is not filled from a default or question phrase.
            unresolved += pred["select_order"] == "none"
        out.append(pred)
    return out, unresolved


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(paths):
    return {os.path.relpath(path, ROOT).replace("\\", "/"): sha256_file(path)
            for path in paths}


def main():
    output = os.path.join(ROOT, "results", "selection_variant")
    os.makedirs(output, exist_ok=True)
    natural_train_source, natural_eval_source = training_pool(), evaluation_rows()
    audit = {"natural_training": audit_rows(natural_train_source),
             "natural_development": audit_rows(natural_eval_source)}
    audit["training_supported"] = audit["natural_training"]["unambiguous"] and audit["natural_training"]["variant_target_expressible"]
    audit["claim_boundary"] = "variant choice only; not rank_by, klass, or full selection validity"
    if not audit["training_supported"]: raise RuntimeError("Selection-variant target audit failed")
    train_triples, heldout_triples = curriculum("train"), curriculum("heldout")
    verification = {"train": verify_curriculum(train_triples), "heldout": verify_curriculum(heldout_triples)}
    if verification["train"]["failures"] or verification["heldout"]["failures"]:
        raise RuntimeError("Selection curriculum verification failed")
    with open(os.path.join(output, "AUDIT.json"), "w", encoding="utf-8") as f: json.dump(audit, f, indent=2)
    with open(os.path.join(output, "curriculum.jsonl"), "w", encoding="utf-8") as f:
        for triple in train_triples + heldout_triples:
            f.write(json.dumps({"rows": triple, "denotations": [deterministic_denotation(r["ir"]) for r in triple]}) + "\n")
    curriculum_path = os.path.join(output, "curriculum.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    natural_train = natural_variant_rows(natural_train_source)
    natural_eval = natural_variant_rows(natural_eval_source)
    synthetic_train = [row for triple in train_triples for row in triple]
    synthetic_eval = [row for triple in heldout_triples for row in triple]
    struct_train = diverse_prefix(natural_train_source)[:100]
    natural_time_train = _natural(natural_train_source)
    natural_time_eval = _natural(natural_eval_source)
    synthetic_time = [row for a, b, _ in time_contrast_pairs("train") for row in (a, b)]
    with open(os.path.join(ROOT, "results", "time_integration_ablation", "metrics.json"), encoding="utf-8") as f:
        frozen_reference = json.load(f)
    reference_by_seed = {run["seed"]: run["configurations"]["predicted_time"] for run in frozen_reference["runs"]}

    runs = []
    persisted_predictions = []
    checkpoint_paths = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if device == "cuda": torch.cuda.manual_seed_all(seed)
        structural = StructuralParser(model_path)
        train_structure(structural, struct_train, device, seed=seed)
        structural_eval = evaluate_structure(structural, natural_eval_source, device)
        torch.manual_seed(seed + 1000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 1000)
        time_model = TimeParser(model_path)
        train_time(time_model, natural_time_train + synthetic_time, device, seed + 1000)
        time_predictions = predict_time(time_model, natural_time_eval, device)
        integrated = score_time(natural_eval_source, [p["pred"] for p in structural_eval["predictions"]],
                                time_predictions, "predicted_time")
        baseline_predictions = [item["pred"] for item in integrated.pop("details")]
        baseline = score(natural_eval_source, baseline_predictions)
        frozen = reference_by_seed[seed]
        for metric in ("skeleton_exact", "operator_accuracy", "field_omission_rate",
                       "unsupported_addition_rate", "invalid_composition_rate"):
            if abs(baseline[metric] - frozen[metric]) > 1e-12:
                raise RuntimeError(f"Baseline reproduction failed seed={seed} metric={metric}")
        del structural, time_model
        if device == "cuda": torch.cuda.empty_cache()

        torch.manual_seed(seed + 3000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 3000)
        model = VariantPredictor(model_path)
        training = train_variant(model, natural_train + synthetic_train, device, seed + 3000)
        natural_predictions = predict_variants(model, natural_eval, device)
        synthetic_predictions = predict_variants(model, synthetic_eval, device)
        candidate_predictions, unresolved = integrate_variant(baseline_predictions, natural_predictions)
        candidate = score(natural_eval_source, candidate_predictions)
        repairs = sum((not a["exact"]) and b["exact"] for a, b in zip(baseline["details"], candidate["details"]))
        regressions = sum(a["exact"] and not b["exact"] for a, b in zip(baseline["details"], candidate["details"]))
        for index, (row, variant_pred, base_pred, candidate_pred, base_detail, candidate_detail) in enumerate(zip(
                natural_eval_source, natural_predictions, baseline_predictions, candidate_predictions,
                baseline["details"], candidate["details"])):
            persisted_predictions.append({
                "seed": seed, "index": index, "text": row["text"], "family": row["family"],
                "source": row["source"], "gold_ir": row["ir"], "gold_labels": row["labels"],
                "gold_variant": natural_eval[index]["variant"], "predicted_variant": variant_pred,
                "baseline_prediction": base_pred, "accepted_prediction": candidate_pred,
                "baseline_exact": base_detail["exact"], "accepted_exact": candidate_detail["exact"],
            })
        checkpoint_path = os.path.join(output, f"selection_variant_seed{seed}.pt")
        cpu_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save({"format_version": 1, "component": "selection_variant_predictor",
                    "accepted_for_experimental_parser": True, "seed": seed,
                    "training_seed": seed + 3000, "variants": list(VARIANTS),
                    "training_config": TRAINING_CONFIG, "model_state_dict": cpu_state}, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)
        del cpu_state
        baseline.pop("details"); candidate.pop("details")
        runs.append({"seed": seed, "baseline": baseline, "candidate": candidate,
                     "natural_variant": variant_metrics(natural_eval, natural_predictions),
                     "synthetic_variant": variant_metrics(synthetic_eval, synthetic_predictions),
                     "candidate_selection_validity": selection_validity(candidate_predictions),
                     "unresolved_present_order": unresolved,
                     "repairs": repairs, "regressions": regressions, "training": training})
        del model
        if device == "cuda": torch.cuda.empty_cache()

    metrics = ("skeleton_exact", "field_omission_rate", "unsupported_addition_rate", "invalid_composition_rate")
    summary = {config: {metric: {"mean": statistics.mean(run[config][metric] for run in runs),
                                 "min": min(run[config][metric] for run in runs),
                                 "max": max(run[config][metric] for run in runs)} for metric in metrics}
               for config in ("baseline", "candidate")}
    exact_deltas = [run["candidate"]["skeleton_exact"] - run["baseline"]["skeleton_exact"] for run in runs]
    invalid_deltas = [run["candidate"]["invalid_composition_rate"] - run["baseline"]["invalid_composition_rate"] for run in runs]
    gates = {"strict_exact_improves_all_seeds": all(value > 0 for value in exact_deltas),
             "invalid_not_increased_any_seed": all(value <= 0 for value in invalid_deltas)}
    passed = all(gates.values())
    result = {"device": device, "seeds": list(SEEDS), "questions": len(natural_eval),
              "audit": audit, "curriculum_verification": verification,
              "natural_training_examples": len(natural_train), "synthetic_training_examples": len(synthetic_train),
              "natural_development_examples": len(natural_eval), "synthetic_heldout_examples": len(synthetic_eval),
              "runs": runs, "summary": summary, "exact_deltas": exact_deltas,
              "invalid_deltas": invalid_deltas, "gates": gates, "integration_success": passed,
              "claim_boundary": audit["claim_boundary"], "development_set_status": "iteratively used; not untouched generalization evidence",
              "actual_signal_schema_linking_evaluated": False, "rank_by_evaluated": False,
              "klass_evaluated": False, "rejected_joint_predictor_used": False,
              "opened_acceptance_or_final": False, "schema_linker_changed": False,
              "typed_ir_changed": False, "executor_changed": False, "time_component_changed": False,
              "baseline_equivalence": "metric-exact deterministic reconstruction, not checkpoint bit identity",
              "accepted_component_persistence": "full checkpoint plus exact per-example integrated predictions"}
    metrics_path = os.path.join(output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)

    predictions_path = os.path.join(output, "accepted_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in persisted_predictions:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    source_paths = [os.path.join(ROOT, "bench", name)
                    for name in ("ir_gold.tsv", "diag_v1.tsv", "indep_dev.tsv")]
    model_files = sorted(os.path.join(model_path, name) for name in os.listdir(model_path)
                         if os.path.isfile(os.path.join(model_path, name)))
    manifest = {
        "format_version": 1, "experiment": "selection_variant", "status": "accepted_experimental",
        "seeds": list(SEEDS), "questions_per_seed": len(natural_eval),
        "training_config": TRAINING_CONFIG, "base_model_path": os.path.relpath(model_path, ROOT).replace("\\", "/"),
        "dataset_sha256": fingerprint_files(source_paths),
        "curriculum_sha256": sha256_file(curriculum_path),
        "base_model_sha256": fingerprint_files(model_files),
        "checkpoints_sha256": fingerprint_files(checkpoint_paths),
        "accepted_predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "accepted_predictions_rows": len(persisted_predictions),
        "evaluation_boundary": "indep_dev only; acceptance and final benchmarks unopened",
    }
    with open(os.path.join(output, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    lines = ["# Three-Class Selection-Variant Experiment", "",
             "Target audit passed for variant choice only. Full selection validity is not claimed because the skeleton omits `rank_by` and `klass`. The natural training distribution is 91 Absent / 8 TripSelection / 1 ChargeSessionSelection.", "",
             f"Synthetic curriculum: {verification['train']['triples']} verified training triples and {verification['heldout']['triples']} verified held-out triples; synthetic metrics are reported separately.", "",
             "| Configuration | Skeleton exact | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|"]
    for config in ("baseline", "candidate"):
        s = summary[config]
        lines.append(f"| {config} | {100*s['skeleton_exact']['mean']:.1f}% ({100*s['skeleton_exact']['min']:.1f}–{100*s['skeleton_exact']['max']:.1f}) | {100*s['field_omission_rate']['mean']:.1f}% | {100*s['unsupported_addition_rate']['mean']:.1f}% | {100*s['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed results", ""]
    for run, exact_delta, invalid_delta in zip(runs, exact_deltas, invalid_deltas):
        nv = run["natural_variant"]
        lines.append(f"- Seed {run['seed']}: exact {100*run['baseline']['skeleton_exact']:.1f}% → {100*run['candidate']['skeleton_exact']:.1f}% ({100*exact_delta:+.1f}); invalid {100*run['baseline']['invalid_composition_rate']:.1f}% → {100*run['candidate']['invalid_composition_rate']:.1f}% ({100*invalid_delta:+.1f}); natural variant accuracy {100*nv['accuracy']:.1f}%, macro recall {100*nv['macro_recall']:.1f}%; repairs {run['repairs']}, regressions {run['regressions']}.")
    lines += ["", "## Gates", ""] + [f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items()]
    lines += ["", f"**Integration decision: {'PASS — retain experimentally' if passed else 'FAIL — do not integrate'}**", "",
              f"Natural variant accuracy: {100*statistics.mean(run['natural_variant']['accuracy'] for run in runs):.1f}% overall; macro recall {100*statistics.mean(run['natural_variant']['macro_recall'] for run in runs):.1f}%.",
              f"Mean per-class recall: Absent {100*statistics.mean(run['natural_variant']['per_class_recall']['Absent'] for run in runs):.1f}%, TripSelection {100*statistics.mean(run['natural_variant']['per_class_recall']['TripSelection'] for run in runs):.1f}%, ChargeSessionSelection {100*statistics.mean(run['natural_variant']['per_class_recall']['ChargeSessionSelection'] for run in runs):.1f}%.",
              f"Complete-structure repairs/regressions: {sum(run['repairs'] for run in runs)}/{sum(run['regressions'] for run in runs)}. Conditional decoding left {sum(run['unresolved_present_order'] for run in runs)} present selections with unresolved order across all seed-question evaluations.",
              f"Training averaged {statistics.mean(run['training']['seconds'] for run in runs):.1f} seconds/seed with {max(run['training']['peak_vram_mb'] for run in runs):.0f} MB peak allocated CUDA memory.", "",
              "Synthetic held-out results are diagnostic for generated language only. This development set is iterative, and the two natural ChargeSessionSelection cases cannot support a strong generalization claim."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
    print(json.dumps({"audit": audit["natural_training"]["class_counts"], "verification": verification,
                      "exact_deltas": exact_deltas, "invalid_deltas": invalid_deltas,
                      "gates": gates, "success": passed, "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

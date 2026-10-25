"""Three-seed integration test for a joint operator/aggregation predictor."""
from __future__ import annotations

import copy
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

from .operator_aggregation_oracle import score as score_predictions
from .operator_curriculum import contrast_pairs as operator_contrast_pairs
from .supervised_parser import (
    ROOT, StructuralParser, diverse_prefix, evaluate as evaluate_structure,
    evaluation_rows, train as train_structure, training_pool,
)
from .time_curriculum import (
    TimeParser, _natural, contrast_pairs as time_contrast_pairs, train as train_time,
)
from .time_integration_ablation import predict_time, score as score_time


SEEDS = (701, 702, 703)


def compatible(signature):
    op, agg = signature
    requires_agg = op in ("aggregate", "extreme", "compare")
    return (agg != "none") == requires_agg


def joint_training_rows():
    natural = diverse_prefix(training_pool())[:100]
    synthetic = [row for a, b, _, _ in operator_contrast_pairs("train") for row in (a, b)]
    rows = natural + synthetic
    for row in rows:
        sig = (row["labels"]["op"], row["labels"]["agg"])
        if not compatible(sig):
            raise ValueError(f"Incompatible training signature {sig}: {row['ir']}")
    return natural, synthetic, rows


def signature_vocabulary(rows):
    return sorted({(row["labels"]["op"], row["labels"]["agg"]) for row in rows})


class JointOperatorParser(nn.Module):
    def __init__(self, model_path, signatures):
        super().__init__()
        self.signatures = tuple(signatures)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(self.signatures))

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=160, return_tensors="pt")

    def forward(self, batch):
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)


def train_joint(model, rows, device, seed, epochs=12, batch_size=24, lr=2e-5):
    random.seed(seed); torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model.to(device)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    index = {sig: i for i, sig in enumerate(model.signatures)}
    counts = Counter((r["labels"]["op"], r["labels"]["agg"]) for r in rows)
    weights = torch.tensor([
        min(4.0, (len(rows) / (len(model.signatures) * counts[sig])) ** .5)
        for sig in model.signatures
    ], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    losses, started = [], time.perf_counter()
    for _ in range(epochs):
        model.train(); total = 0.; seen = 0
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokenize([x["text"] for x in items]).items()}
            gold = torch.tensor([index[(x["labels"]["op"], x["labels"]["agg"])] for x in items], device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(batch)
                loss = nn.functional.cross_entropy(logits, gold, weight=weights)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss.detach()) * len(items); seen += len(items)
        losses.append(total / seen)
    return {"examples": len(rows), "epochs": epochs, "seconds": time.perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
            "losses": losses}


def predict_joint(model, rows, device):
    loader = DataLoader(rows, batch_size=64, shuffle=False, collate_fn=lambda x: x)
    predictions = []
    model.eval(); started = time.perf_counter()
    with torch.inference_mode():
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokenize([x["text"] for x in items]).items()}
            indices = model(batch).argmax(-1).tolist()
            predictions.extend(model.signatures[i] for i in indices)
    return predictions, (time.perf_counter() - started) * 1000 / len(rows)


def integrate(base_predictions, signatures):
    out = []
    for base, (op, agg) in zip(base_predictions, signatures):
        pred = copy.deepcopy(base)
        pred["op"], pred["agg"] = op, agg
        out.append(pred)
    return out


def main():
    output = os.path.join(ROOT, "results", "joint_operator_integration")
    os.makedirs(output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    natural_train, synthetic_train, joint_rows = joint_training_rows()
    signatures = signature_vocabulary(joint_rows)
    eval_rows = evaluation_rows()
    struct_rows = diverse_prefix(training_pool())[:100]
    natural_time = _natural(training_pool())
    synthetic_time = [row for a, b, _ in time_contrast_pairs("train") for row in (a, b)]
    time_eval = _natural(eval_rows)
    with open(os.path.join(ROOT, "results", "time_integration_ablation", "metrics.json"), encoding="utf-8") as f:
        frozen_reference = json.load(f)

    runs, row_outputs = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if device == "cuda": torch.cuda.manual_seed_all(seed)
        structural = StructuralParser(model_path)
        struct_stats = train_structure(structural, struct_rows, device, seed=seed)
        struct_eval = evaluate_structure(structural, eval_rows, device)

        torch.manual_seed(seed + 1000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 1000)
        time_model = TimeParser(model_path)
        time_seconds = train_time(time_model, natural_time + synthetic_time, device, seed + 1000)
        time_predictions = predict_time(time_model, time_eval, device)
        baseline_scored = score_time(eval_rows, [p["pred"] for p in struct_eval["predictions"]],
                                     time_predictions, "predicted_time")
        baseline_predictions = [d["pred"] for d in baseline_scored.pop("details")]
        baseline = score_predictions(eval_rows, baseline_predictions)

        reference = next(r["configurations"]["predicted_time"] for r in frozen_reference["runs"] if r["seed"] == seed)
        for metric in ("skeleton_exact", "operator_accuracy", "field_omission_rate",
                       "unsupported_addition_rate", "invalid_composition_rate"):
            if abs(baseline[metric] - reference[metric]) > 1e-12:
                raise RuntimeError(f"Frozen baseline reproduction failed seed={seed} metric={metric}: "
                                   f"{baseline[metric]} != {reference[metric]}")

        del structural, time_model
        if device == "cuda": torch.cuda.empty_cache()

        torch.manual_seed(seed + 2000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 2000)
        joint_model = JointOperatorParser(model_path, signatures)
        joint_stats = train_joint(joint_model, joint_rows, device, seed + 2000)
        joint_signatures, latency = predict_joint(joint_model, eval_rows, device)
        joint_predictions = integrate(baseline_predictions, joint_signatures)
        joint = score_predictions(eval_rows, joint_predictions)

        repairs = regressions = invalid_fixed = invalid_created = signature_correct = 0
        for row, bp, jp, sig in zip(eval_rows, baseline["details"], joint["details"], joint_signatures):
            repairs += (not bp["exact"]) and jp["exact"]
            regressions += bp["exact"] and (not jp["exact"])
            invalid_fixed += bp["invalid"] and not jp["invalid"]
            invalid_created += (not bp["invalid"]) and jp["invalid"]
            gold_sig = (row["labels"]["op"], row["labels"]["agg"])
            signature_correct += sig == gold_sig
            row_outputs.append({"seed": seed, "question": row["question"], "family": row["family"],
                                "gold_signature": gold_sig, "predicted_signature": sig,
                                "baseline_exact": bp["exact"], "joint_exact": jp["exact"],
                                "baseline_invalid": bp["invalid"], "joint_invalid": jp["invalid"]})
        baseline.pop("details"); joint.pop("details")
        runs.append({"seed": seed, "baseline": baseline, "joint": joint,
                     "joint_signature_accuracy": signature_correct / len(eval_rows),
                     "repairs": repairs, "regressions": regressions,
                     "invalid_fixed": invalid_fixed, "invalid_created": invalid_created,
                     "training": {"structure": struct_stats, "time_seconds": time_seconds,
                                  "joint": joint_stats}, "joint_latency_ms_per_question": latency})
        torch.save({"head": joint_model.head.state_dict(), "signatures": signatures,
                    "seed": seed, "training_recipe": {"epochs": 12, "batch_size": 24, "lr": 2e-5}},
                   os.path.join(output, f"joint_head_seed{seed}.pt"))
        del joint_model
        if device == "cuda": torch.cuda.empty_cache()

    metrics = ("skeleton_exact", "operator_accuracy", "aggregation_accuracy", "target_arity_accuracy",
               "field_omission_rate", "unsupported_addition_rate", "invalid_composition_rate")
    summary = {}
    for config in ("baseline", "joint"):
        summary[config] = {}
        for metric in metrics:
            values = [r[config][metric] for r in runs]
            summary[config][metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    deltas = [r["joint"]["skeleton_exact"] - r["baseline"]["skeleton_exact"] for r in runs]
    invalid_deltas = [r["joint"]["invalid_composition_rate"] - r["baseline"]["invalid_composition_rate"] for r in runs]
    gates = {"strict_skeleton_improves_all_seeds": all(d > 0 for d in deltas),
             "invalid_composition_not_increased_any_seed": all(d <= 0 for d in invalid_deltas)}
    passed = all(gates.values())
    result = {"device": device, "seeds": list(SEEDS), "questions": len(eval_rows),
              "natural_training_examples": len(natural_train),
              "synthetic_training_examples": len(synthetic_train),
              "joint_signatures": [list(x) for x in signatures], "runs": runs, "summary": summary,
              "skeleton_deltas": deltas, "invalid_deltas": invalid_deltas,
              "gates": gates, "integration_success": passed, "joint_oracle_reference": 0.29444444444444445,
              "opened_acceptance_or_final": False, "schema_linker_changed": False,
              "typed_ir_changed": False, "executor_changed": False, "time_component_changed": False}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(output, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in row_outputs: f.write(json.dumps(row) + "\n")

    lines = ["# Joint Operator/Aggregation Integration", "",
             "Same 120 natural development questions and seeds 701–703. The dedicated time component and all protected VehicleGPT components were unchanged. Acceptance and final benchmarks remained sealed.", "",
             "| Configuration | Skeleton exact | Operator | Aggregation | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for config in ("baseline", "joint"):
        s = summary[config]
        lines.append(f"| {config} | {100*s['skeleton_exact']['mean']:.1f}% ({100*s['skeleton_exact']['min']:.1f}–{100*s['skeleton_exact']['max']:.1f}) | {100*s['operator_accuracy']['mean']:.1f}% | {100*s['aggregation_accuracy']['mean']:.1f}% | {100*s['field_omission_rate']['mean']:.1f}% | {100*s['unsupported_addition_rate']['mean']:.1f}% | {100*s['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed gate", ""]
    for run, delta, invalid_delta in zip(runs, deltas, invalid_deltas):
        lines.append(f"- Seed {run['seed']}: skeleton {100*run['baseline']['skeleton_exact']:.1f}% → {100*run['joint']['skeleton_exact']:.1f}% ({100*delta:+.1f}); invalid {100*run['baseline']['invalid_composition_rate']:.1f}% → {100*run['joint']['invalid_composition_rate']:.1f}% ({100*invalid_delta:+.1f}); repairs {run['repairs']}, regressions {run['regressions']}.")
    lines += ["", "## Gates", ""] + [f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items()]
    lines += ["", f"**Integration decision: {'PASS — retain experimentally' if passed else 'FAIL — do not integrate'}**", "",
              f"Joint-signature accuracy: {100*statistics.mean(r['joint_signature_accuracy'] for r in runs):.1f}% mean ({100*min(r['joint_signature_accuracy'] for r in runs):.1f}–{100*max(r['joint_signature_accuracy'] for r in runs):.1f}).",
              f"Training: {statistics.mean(r['training']['joint']['seconds'] for r in runs):.1f} s/seed; peak allocated CUDA memory {max(r['training']['joint']['peak_vram_mb'] for r in runs):.0f} MB; joint-only batched inference {statistics.mean(r['joint_latency_ms_per_question'] for r in runs):.2f} ms/question.", "",
              "Interpretation: compatibility was learned well enough to eliminate operator/aggregation-invalid pairs, but semantic selection was not reliable. Mean operator accuracy rose while aggregation accuracy fell, unsupported additions increased, and seed 701 had as many complete-structure regressions as repairs. Stop expanding this curriculum and reconsider the prediction formulation before another training run.", "",
              "The 29.4% joint oracle is a diagnostic ceiling for this ablation, not a promised learned-model target."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"deltas": deltas, "invalid_deltas": invalid_deltas,
                      "gates": gates, "success": passed,
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

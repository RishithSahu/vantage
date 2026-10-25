"""Run the bounded ALMA-v2 frozen-representation edge experiment."""
from __future__ import annotations

import json
import os
import time

import torch

from .edge_data import challenge_rows, existing_natural_rows, training_rows
from .edge_model import FrozenEdgeParser, evaluate, train_heads
from .model import AlmaClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(ROOT, "results", "alma_v2_edge")
    os.makedirs(out, exist_ok=True)
    train = training_rows()
    challenge = challenge_rows()
    natural = existing_natural_rows()
    backbones = {
        "minilm": os.path.join(ROOT, "models", "alma-minilm-l6-v2"),
        "deberta_nli": os.path.join(ROOT, "models", "deberta_xsmall_nli"),
    }
    baseline_dir = os.path.join(ROOT, "results", "alma_scope_probe", "classifier_checkpoint")
    baseline_state = torch.load(os.path.join(baseline_dir, "classifier.pt"), map_location="cpu", weights_only=True)
    baseline = AlmaClassifier(os.path.join(baseline_dir, "encoder"), baseline_state["labels"]).to(device)
    baseline.classifier.load_state_dict(baseline_state["head"])
    baseline_rank, _ = baseline.predict([r["text"] for r in challenge], device=device)
    expected = ["scope:common" if r["label"] == "comparison" else "scope:" + r["label"] for r in challenge]
    baseline_accuracy = sum(rank[0] == gold for rank, gold in zip(baseline_rank, expected)) / len(challenge)
    del baseline
    if device == "cuda": torch.cuda.empty_cache()
    result = {"device": device, "opened_final_benchmark": False,
              "training_questions": 0, "training_examples": len(train),
              "challenge_examples": len(challenge),
              "same_set_sentence_baseline": baseline_accuracy, "models": {}}
    for name, path in backbones.items():
        started = time.perf_counter()
        model = FrozenEdgeParser(path)
        stats = train_heads(model, train, device=device)
        torch.save({"role_head": model.role_head.state_dict(),
                    "edge_head": model.edge_head.state_dict(),
                    "backbone": path}, os.path.join(out, f"{name}_heads.pt"))
        result["models"][name] = {
            "training": stats.__dict__, "challenge": evaluate(model, challenge, device),
            "existing_natural": evaluate(model, natural, device),
            "wall_seconds": time.perf_counter() - started,
        }
        del model
        if device == "cuda": torch.cuda.empty_cache()
    best_name = max(result["models"], key=lambda n: result["models"][n]["challenge"]["gold_spans_pred_edges"])
    best = result["models"][best_name]
    gates = {
        "attachment_at_least_80": best["challenge"]["gold_spans_pred_edges"] >= .80,
        "edge_improves_same_set_baseline_by_5_points": best["challenge"]["gold_spans_pred_edges"] >= baseline_accuracy + .05,
        "oracle_reconstruction_100": best["challenge"]["oracle_decoder"] == 1.0,
        "encoder_unchanged": all(v["training"]["encoder_unchanged"] for v in result["models"].values()),
        "existing_natural_above_previous_50": best["existing_natural"]["gold_spans_pred_edges"] > .50,
    }
    result["best_model"] = best_name
    result["gates"] = gates
    result["continue_alma"] = all(gates.values())
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lines = ["# ALMA-v2 Frozen Edge Experiment", "",
             "Training used 900 question-free declaratives. The 300-case challenge used held-out natural-question constructions. The independent final benchmark was not opened.", "",
             f"Same-set sentence-classifier baseline: **{100*baseline_accuracy:.1f}%**.", "",
             "| Backbone | Gold spans → edges | Predicted-span token F1 | Span exact | Predicted spans → edges | Full structure exact | Existing 12 gold spans → edges | Encoder changed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data in result["models"].items():
        c, n, s = data["challenge"], data["existing_natural"], data["training"]
        lines.append(f"| {name} | {100*c['gold_spans_pred_edges']:.1f}% | {100*c['predicted_span_token_f1']:.1f}% | {100*c['predicted_spans_exact']:.1f}% | {100*c['predicted_spans_pred_edges']:.1f}% | {100*c['complete_structure_exact']:.1f}% | {100*n['gold_spans_pred_edges']:.1f}% | {not s['encoder_unchanged']} |")
    lines += ["", "## Gates", ""] + [f"- {'PASS' if value else 'FAIL'}: {key}" for key, value in gates.items()]
    lines += ["", f"**Decision: {'CONTINUE' if result['continue_alma'] else 'STOP OR REDESIGN'} ALMA after this bounded experiment.**", "",
              "The 300-case set is automatically constructed with held-out templates and vocabulary; it is larger and construction-separated, but it is not a substitute for independently authored evaluation."]
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"best_model": best_name, "gates": gates, "continue_alma": result["continue_alma"],
                      "report": os.path.join(out, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

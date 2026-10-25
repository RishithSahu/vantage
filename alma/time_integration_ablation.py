"""Four-configuration integration ablation for the dedicated time head."""
from __future__ import annotations

import copy
import json
import os
import random
import statistics
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader

from vgpt.query_ir import Query
from .supervised_parser import (
    LABELS as STRUCT_LABELS, ROOT, StructuralParser, diverse_prefix,
    evaluate as evaluate_structure, evaluation_rows, train as train_structure,
    training_pool,
)
from .time_curriculum import (
    LABELS as TIME_LABELS, TimeParser, _natural, contrast_pairs,
    train as train_time,
)

SEEDS = (701, 702, 703)
CONFIGS = ("baseline", "oracle_time", "predicted_time", "shuffled_time")
TIME_FIELDS = ("time_type", "time_unit", "time2_type", "before", "after")


def predict_time(model, rows, device):
    loader = DataLoader(rows, batch_size=64, shuffle=False, collate_fn=lambda x: x)
    out = []
    model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokens([x["text"] for x in items]).items()}
            logits = model(batch)
            for i in range(len(items)):
                out.append({k: TIME_LABELS[k][int(logits[k][i].argmax())] for k in TIME_LABELS})
    return out


def event_fields(window):
    return {"before": "present" if window in ("before", "around") else "none",
            "after": "present" if window in ("after", "around") else "none"}


def structural_time(time_pred):
    return {"time_type": time_pred["time_type"], "time_unit": time_pred["time_unit"],
            "time2_type": time_pred["time2_type"], **event_fields(time_pred["event_window"])}


def baseline_time(struct_pred):
    before, after = struct_pred["before"] == "present", struct_pred["after"] == "present"
    window = "around" if before and after else ("before" if before else ("after" if after else "none"))
    return {"time_type": struct_pred["time_type"], "time_unit": struct_pred["time_unit"],
            "calendar": "none", "time2_type": struct_pred["time2_type"], "event_window": window}


def invalid(pred):
    return ((pred["op"] in ("aggregate", "extreme", "compare") and pred["agg"] == "none") or
            (pred["op"] not in ("aggregate", "extreme", "compare") and pred["agg"] != "none") or
            (pred["time2_type"] != "none" and pred["op"] != "compare") or
            ((pred["before"] != "none" or pred["after"] != "none") and pred["event"] == "none") or
            (pred["select_entity"] == "none" and
             (pred["select_order"] != "none" or pred["select_k"] != "none")))


def score(rows, struct_predictions, time_predictions, config):
    exact = strict_time = comparable_time = op_ok = omissions = additions = invalid_n = 0
    details = []
    for row, sp, tp in zip(rows, struct_predictions, time_predictions):
        gold = row["labels"]
        pred = copy.deepcopy(sp)
        if config == "oracle_time":
            for field in TIME_FIELDS: pred[field] = gold[field]
            strict_calendar_ok = True
        elif config in ("predicted_time", "shuffled_time"):
            pred.update(structural_time(tp))
            strict_calendar_ok = tp["calendar"] == (Query.from_text(row["ir"]).scope.time.name or "none")
        else:
            tp = baseline_time(sp)
            strict_calendar_ok = tp["calendar"] == (Query.from_text(row["ir"]).scope.time.name or "none")
        matches = {field: pred[field] == gold[field] for field in STRUCT_LABELS}
        is_exact = all(matches.values())
        comparable = all(matches[f] for f in TIME_FIELDS)
        exact += is_exact; comparable_time += comparable; strict_time += comparable and strict_calendar_ok
        op_ok += matches["op"]
        omissions += any(gold[f] != "none" and pred[f] == "none" for f in STRUCT_LABELS)
        additions += any(gold[f] == "none" and pred[f] != "none" for f in STRUCT_LABELS)
        invalid_n += invalid(pred)
        details.append({"exact": is_exact, "time_exact": comparable, "pred": pred})
    n = len(rows)
    return {"n": n, "skeleton_exact": exact / n, "strict_time_exact": strict_time / n,
            "comparable_time_exact": comparable_time / n, "operator_accuracy": op_ok / n,
            "field_omission_rate": omissions / n, "unsupported_addition_rate": additions / n,
            "invalid_composition_rate": invalid_n / n, "details": details}


def timed_predictions(struct_model, time_model, texts, device):
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    # Integrated inference runs both unchanged components.
    struct_model.eval(); time_model.eval()
    with torch.inference_mode():
        for model, tokenize in ((struct_model, struct_model.tokenize), (time_model, time_model.tokens)):
            for i in range(0, len(texts), 32):
                batch = {k: v.to(device) for k, v in tokenize(texts[i:i + 32]).items()}
                model(batch)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return {"total_ms": elapsed * 1000, "ms_per_question": elapsed * 1000 / len(texts),
            "peak_cuda_mb": peak}


def main():
    output = os.path.join(ROOT, "results", "time_integration_ablation")
    os.makedirs(output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    struct_train = diverse_prefix(training_pool())[:100]
    eval_rows = evaluation_rows()
    natural_time_train = _natural(training_pool())
    synthetic_train = [row for a, b, _ in contrast_pairs("train") for row in (a, b)]
    natural_time_eval = _natural(eval_rows)
    runs = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        struct_model = StructuralParser(model_path)
        struct_stats = train_structure(struct_model, struct_train, device, seed=seed)
        struct_eval = evaluate_structure(struct_model, eval_rows, device)
        struct_predictions = [row["pred"] for row in struct_eval["predictions"]]

        torch.manual_seed(seed + 1000)
        time_model = TimeParser(model_path)
        time_seconds = train_time(time_model, natural_time_train + synthetic_train, device, seed + 1000)
        time_predictions = predict_time(time_model, natural_time_eval, device)
        rng = random.Random(seed + 2000)
        shift = rng.randrange(1, len(time_predictions))
        shuffled = time_predictions[shift:] + time_predictions[:shift]

        configs = {
            "baseline": score(eval_rows, struct_predictions, [baseline_time(p) for p in struct_predictions], "baseline"),
            "oracle_time": score(eval_rows, struct_predictions, time_predictions, "oracle_time"),
            "predicted_time": score(eval_rows, struct_predictions, time_predictions, "predicted_time"),
            "shuffled_time": score(eval_rows, struct_predictions, shuffled, "shuffled_time"),
        }
        baseline_detail = configs["baseline"]["details"]
        predicted_detail = configs["predicted_time"]["details"]
        repairs = sum((not a["exact"]) and b["exact"] for a, b in zip(baseline_detail, predicted_detail))
        regressions = sum(a["exact"] and (not b["exact"]) for a, b in zip(baseline_detail, predicted_detail))
        for metrics in configs.values(): metrics.pop("details")
        latency = timed_predictions(struct_model, time_model, [r["text"] for r in eval_rows], device)
        runs.append({"seed": seed, "configurations": configs, "repairs": repairs,
                     "regressions": regressions, "latency": latency,
                     "training_seconds": {"structure": struct_stats["seconds"], "time": time_seconds}})
        del struct_model, time_model
        if device == "cuda": torch.cuda.empty_cache()

    summary = {}
    for config in CONFIGS:
        summary[config] = {}
        for metric in ("skeleton_exact", "strict_time_exact", "comparable_time_exact", "operator_accuracy",
                       "field_omission_rate", "unsupported_addition_rate", "invalid_composition_rate"):
            values = [r["configurations"][config][metric] for r in runs]
            summary[config][metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    deltas = [r["configurations"]["predicted_time"]["skeleton_exact"] -
              r["configurations"]["baseline"]["skeleton_exact"] for r in runs]
    gates = {"mean_strict_skeleton_improves": statistics.mean(deltas) > 0,
             "positive_in_all_seeds": all(d > 0 for d in deltas),
             "operator_no_regression": all(r["configurations"]["predicted_time"]["operator_accuracy"] ==
                                           r["configurations"]["baseline"]["operator_accuracy"] for r in runs),
             "invalid_composition_no_material_regression":
                 statistics.mean(r["configurations"]["predicted_time"]["invalid_composition_rate"] for r in runs) <=
                 statistics.mean(r["configurations"]["baseline"]["invalid_composition_rate"] for r in runs) + .01}
    result = {"device": device, "seeds": list(SEEDS), "questions": len(eval_rows),
              "opened_final_benchmark": False, "schema_linker_changed": False,
              "runs": runs, "summary": summary, "skeleton_deltas": deltas,
              "gates": gates, "integration_success": all(gates.values())}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lines = ["# Dedicated Time-Head Integration Ablation", "",
             "Same 120 natural development questions, three predefined seeds. Acceptance and final sets remained sealed.", "",
             "| Configuration | Skeleton exact | Strict time exact | Comparable time exact | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        s = summary[config]
        lines.append(f"| {config} | {100*s['skeleton_exact']['mean']:.1f}% | {100*s['strict_time_exact']['mean']:.1f}% | {100*s['comparable_time_exact']['mean']:.1f}% | {100*s['field_omission_rate']['mean']:.1f}% | {100*s['unsupported_addition_rate']['mean']:.1f}% | {100*s['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed skeleton exact", ""]
    for run, delta in zip(runs, deltas):
        c = run["configurations"]
        lines.append(f"- Seed {run['seed']}: baseline {100*c['baseline']['skeleton_exact']:.1f}%, oracle {100*c['oracle_time']['skeleton_exact']:.1f}%, predicted {100*c['predicted_time']['skeleton_exact']:.1f}%, shuffled {100*c['shuffled_time']['skeleton_exact']:.1f}%; delta {100*delta:+.1f} points; repairs {run['repairs']}, regressions {run['regressions']}.")
    lines += ["", "## Gates", ""] + [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in gates.items()]
    lines += ["", f"**Integration decision: {'PASS' if result['integration_success'] else 'FAIL'}**"]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"deltas": deltas, "gates": gates, "success": result["integration_success"],
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

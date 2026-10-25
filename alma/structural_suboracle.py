"""Five-way structural-field sub-oracle study with targeted overlap analysis."""
from __future__ import annotations

import copy
import itertools
import json
import os
import statistics
from collections import Counter

import torch

from .field_arity_oracle import FIELD_SELECTION_FIELDS
from .operator_aggregation_oracle import score
from .supervised_parser import (
    ROOT, StructuralParser, diverse_prefix, evaluate as evaluate_structure,
    evaluation_rows, train as train_structure, training_pool,
)
from .time_curriculum import TimeParser, _natural, contrast_pairs, train as train_time
from .time_integration_ablation import predict_time, score as score_time


SEEDS = (701, 702, 703)
SUBFIELDS = {
    "condition_shape": ("condition_shape",),
    "entity_selection": ("select_entity", "select_order", "select_k"),
    "grouping": ("groups_count",),
    "event_window": ("event", "before", "after"),
    "dialogue_references": ("refs",),
}
STRONG_INTERACTION_THRESHOLD = .02


def substitute(predictions, rows, categories):
    fields = tuple(field for category in categories for field in SUBFIELDS[category])
    out = []
    for pred, row in zip(predictions, rows):
        p = copy.deepcopy(pred)
        for field in fields: p[field] = row["labels"][field]
        out.append(p)
    return out


def wrong_categories(pred, gold):
    return tuple(name for name, fields in SUBFIELDS.items()
                 if any(pred[field] != gold[field] for field in fields))


def dominant_pattern(wrong):
    wrong = set(wrong)
    if not wrong: return "outside_structural_subfields"
    if {"event_window", "condition_shape"} <= wrong: return "interaction:event_window+condition_shape"
    if {"entity_selection", "condition_shape"} <= wrong: return "interaction:entity_selection+condition_shape"
    if len(wrong) > 1: return "interaction:" + "+".join(sorted(wrong))
    return next(iter(wrong))


def effects(baseline_details, candidate_details):
    return {
        "repairs": sum((not a["exact"]) and b["exact"] for a, b in zip(baseline_details, candidate_details)),
        "regressions": sum(a["exact"] and not b["exact"] for a, b in zip(baseline_details, candidate_details)),
        "invalid_fixed": sum(a["invalid"] and not b["invalid"] for a, b in zip(baseline_details, candidate_details)),
        "invalid_created": sum((not a["invalid"]) and b["invalid"] for a, b in zip(baseline_details, candidate_details)),
    }


def summarize(runs, configs):
    metric_names = ("skeleton_exact", "field_omission_rate", "unsupported_addition_rate",
                    "invalid_composition_rate")
    out = {}
    for config in configs:
        out[config] = {}
        for metric in metric_names:
            values = [run["configurations"][config][metric] for run in runs]
            out[config][metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    return out


def main():
    output = os.path.join(ROOT, "results", "structural_suboracle")
    os.makedirs(output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    struct_train = diverse_prefix(training_pool())[:100]
    eval_rows = evaluation_rows()
    natural_time_train = _natural(training_pool())
    natural_time_eval = _natural(eval_rows)
    synthetic_time = [row for a, b, _ in contrast_pairs("train") for row in (a, b)]
    with open(os.path.join(ROOT, "results", "time_integration_ablation", "metrics.json"), encoding="utf-8") as f:
        frozen_reference = json.load(f)
    reference_by_seed = {run["seed"]: run["configurations"]["predicted_time"]
                         for run in frozen_reference["runs"]}

    runs, baseline_predictions_by_seed = [], {}
    pattern_counts, pair_overlap = Counter(), Counter()
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
        baseline_predictions = [row["pred"] for row in integrated.pop("details")]
        baseline_predictions_by_seed[seed] = baseline_predictions
        prediction_sets = {"baseline": baseline_predictions}
        for category in SUBFIELDS:
            prediction_sets[category] = substitute(baseline_predictions, eval_rows, (category,))
        scored = {name: score(eval_rows, predictions) for name, predictions in prediction_sets.items()}
        frozen = reference_by_seed[seed]
        for metric in ("skeleton_exact", "operator_accuracy", "field_omission_rate",
                       "unsupported_addition_rate", "invalid_composition_rate"):
            if abs(scored["baseline"][metric] - frozen[metric]) > 1e-12:
                raise RuntimeError(f"Baseline reproduction failed seed={seed} metric={metric}")
        baseline_details = scored["baseline"]["details"]
        run_effects = {name: effects(baseline_details, scored[name]["details"]) for name in SUBFIELDS}
        for pred, row in zip(baseline_predictions, eval_rows):
            wrong = wrong_categories(pred, row["labels"])
            pattern_counts[dominant_pattern(wrong)] += 1
            for pair in itertools.combinations(wrong, 2): pair_overlap[tuple(sorted(pair))] += 1
        for item in scored.values(): item.pop("details")
        runs.append({"seed": seed, "configurations": scored, "effects": run_effects,
                     "baseline_reproduction_passed": True})
        del structural, time_model
        if device == "cuda": torch.cuda.empty_cache()

    individual_configs = ["baseline"] + list(SUBFIELDS)
    initial_summary = summarize(runs, individual_configs)
    base = initial_summary["baseline"]["skeleton_exact"]["mean"]
    gains = {name: initial_summary[name]["skeleton_exact"]["mean"] - base for name in SUBFIELDS}
    ranked = sorted(gains, key=lambda name: (-gains[name], name))
    top_pair = tuple(ranked[:2])
    pair_name = "combined:" + "+".join(top_pair)

    # Add only the strongest pair, as predefined. No exhaustive combination search.
    for run in runs:
        preds = substitute(baseline_predictions_by_seed[run["seed"]], eval_rows, top_pair)
        candidate = score(eval_rows, preds)
        baseline_details = score(eval_rows, baseline_predictions_by_seed[run["seed"]])["details"]
        run["effects"][pair_name] = effects(baseline_details, candidate["details"])
        candidate.pop("details")
        run["configurations"][pair_name] = candidate
    pair_summary = summarize(runs, [pair_name])[pair_name]
    pair_gain = pair_summary["skeleton_exact"]["mean"] - base
    pair_synergy = pair_gain - gains[top_pair[0]] - gains[top_pair[1]]

    triple_name = None
    triple_gain = triple_synergy = None
    if pair_synergy >= STRONG_INTERACTION_THRESHOLD:
        third = ranked[2]
        triple = top_pair + (third,)
        triple_name = "combined:" + "+".join(triple)
        for run in runs:
            preds = substitute(baseline_predictions_by_seed[run["seed"]], eval_rows, triple)
            candidate = score(eval_rows, preds)
            baseline_details = score(eval_rows, baseline_predictions_by_seed[run["seed"]])["details"]
            run["effects"][triple_name] = effects(baseline_details, candidate["details"])
            candidate.pop("details")
            run["configurations"][triple_name] = candidate
        triple_summary = summarize(runs, [triple_name])[triple_name]
        triple_gain = triple_summary["skeleton_exact"]["mean"] - base
        triple_synergy = triple_gain - sum(gains[name] for name in triple)

    all_configs = individual_configs + [pair_name] + ([triple_name] if triple_name else [])
    summary = summarize(runs, all_configs)
    dominance = gains[ranked[0]] - gains[ranked[1]] >= .02
    if max(gains.values()) < .02:
        decision = "reassess_field_taxonomy"
    elif pair_synergy >= STRONG_INTERACTION_THRESHOLD:
        decision = "investigate_constrained_joint:" + "+".join(top_pair)
    elif dominance:
        decision = "target_single_category:" + ranked[0]
    else:
        decision = "separate_targeted_diagnostics:" + "+".join(top_pair)

    total_instances = len(SEEDS) * len(eval_rows)
    overlap = {
        "dominant_patterns": {name: {"count": count, "rate": count / total_instances}
                              for name, count in pattern_counts.most_common()},
        "pairwise_coerrors": {"+".join(pair): {"count": count, "rate": count / total_instances}
                              for pair, count in pair_overlap.most_common()},
        "note": "Co-error and dominant-pattern labels describe observed overlap, not causal attachment proof.",
    }
    result = {"device": device, "seeds": list(SEEDS), "questions": len(eval_rows),
              "subfields": {k: list(v) for k, v in SUBFIELDS.items()},
              "runs": runs, "summary": summary, "individual_oracle_gains": gains,
              "ranked_categories": ranked,
              "top_pair": {"categories": list(top_pair), "gain": pair_gain, "synergy": pair_synergy},
              "third_combination_triggered": triple_name is not None,
              "triple": None if triple_name is None else {"name": triple_name, "gain": triple_gain,
                                                           "synergy": triple_synergy},
              "strong_interaction_threshold": STRONG_INTERACTION_THRESHOLD,
              "overlap": overlap, "next_decision": decision,
              "actual_signal_schema_linking_evaluated": False,
              "rejected_joint_predictor_used": False, "trained_new_component": False,
              "opened_acceptance_or_final": False, "schema_linker_changed": False,
              "typed_ir_changed": False, "executor_changed": False, "time_component_changed": False,
              "development_set_status": "iteratively used; must not be presented as untouched generalization evidence",
              "baseline_equivalence": "metric-exact deterministic reconstruction, not checkpoint bit identity"}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = ["# Structural-Field Sub-Oracle Study", "",
             "Five individual substitutions were evaluated on the same 120 natural development questions and seeds 701–703. No model was trained. The rejected joint predictor was excluded, and actual signal identity/schema linking remained outside the study.", "",
             "The accepted baseline was reconstructed with exact metric agreement, which establishes metric consistency rather than bit-for-bit checkpoint identity. This development set is now iterative and must not be described as untouched generalization evidence.", "",
             "| Oracle | Skeleton exact | Gain | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|---:|"]
    for config in list(SUBFIELDS) + [pair_name] + ([triple_name] if triple_name else []):
        s = summary[config]
        gain = s["skeleton_exact"]["mean"] - base
        lines.append(f"| {config} | {100*s['skeleton_exact']['mean']:.1f}% ({100*s['skeleton_exact']['min']:.1f}–{100*s['skeleton_exact']['max']:.1f}) | {100*gain:+.1f} | {100*s['field_omission_rate']['mean']:.1f}% | {100*s['unsupported_addition_rate']['mean']:.1f}% | {100*s['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed individual effects", ""]
    for run in runs:
        values = ", ".join(f"{name} {100*run['configurations'][name]['skeleton_exact']:.1f}%"
                           for name in SUBFIELDS)
        lines.append(f"- Seed {run['seed']} (baseline {100*run['configurations']['baseline']['skeleton_exact']:.1f}%): {values}.")
    lines += ["", "## Targeted combination", "",
              f"- Strongest pair: {' + '.join(top_pair)}; combined gain {100*pair_gain:+.1f} points; synergy {100*pair_synergy:+.1f} points.",
              f"- Strong-interaction threshold: {100*STRONG_INTERACTION_THRESHOLD:.1f} points.",
              f"- Third-category combination triggered: {'yes' if triple_name else 'no'}.", "",
              "## Dominant co-error patterns", ""]
    for name, item in list(overlap["dominant_patterns"].items())[:10]:
        lines.append(f"- {name}: {item['count']}/{total_instances} ({100*item['rate']:.1f}%).")
    lines += ["", "These are co-error patterns, not proof that one field causally attaches another incorrectly.", "",
              f"**Next decision: {decision}**", "",
              "Oracle substitutions are diagnostic ceilings, not deployable performance."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"gains": gains, "top_pair": result["top_pair"],
                      "third_triggered": triple_name is not None, "decision": decision,
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

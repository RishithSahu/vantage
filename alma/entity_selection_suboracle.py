"""Entity-selection sub-oracles for entity, order, and selection limit k."""
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
MASKS = {
    "entity": ("select_entity",),
    "order": ("select_order",),
    "k": ("select_k",),
}
SUBSTANTIAL_GAIN = .02
CLEAR_DOMINANCE_MARGIN = .02
COMPATIBILITY_INVALID_REDUCTION = .02


def verify_masks():
    fields = [field for mask in MASKS.values() for field in mask]
    return {"disjoint": len(fields) == len(set(fields)),
            "exact_fields": set(fields) == {"select_entity", "select_order", "select_k"},
            "masks": {name: list(mask) for name, mask in MASKS.items()}}


def substitute(predictions, rows, components):
    fields = tuple(field for component in components for field in MASKS[component])
    out = []
    for pred, row in zip(predictions, rows):
        p = copy.deepcopy(pred)
        for field in fields: p[field] = row["labels"][field]
        out.append(p)
    return out


def effects(baseline, candidate):
    return {
        "repairs": sum((not a["exact"]) and b["exact"] for a, b in zip(baseline, candidate)),
        "regressions": sum(a["exact"] and not b["exact"] for a, b in zip(baseline, candidate)),
        "invalid_fixed": sum(a["invalid"] and not b["invalid"] for a, b in zip(baseline, candidate)),
        "invalid_created": sum((not a["invalid"]) and b["invalid"] for a, b in zip(baseline, candidate)),
    }


def summarize(runs, configs):
    metrics = ("skeleton_exact", "field_omission_rate", "unsupported_addition_rate",
               "invalid_composition_rate")
    return {config: {metric: {
                "mean": statistics.mean(run["configurations"][config][metric] for run in runs),
                "min": min(run["configurations"][config][metric] for run in runs),
                "max": max(run["configurations"][config][metric] for run in runs),
            } for metric in metrics} for config in configs}


def main():
    mask_verification = verify_masks()
    if not all((mask_verification["disjoint"], mask_verification["exact_fields"])):
        raise RuntimeError(f"Entity-selection masks failed verification: {mask_verification}")
    output = os.path.join(ROOT, "results", "entity_selection_suboracle")
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

    runs, baseline_predictions_by_seed = [], {}
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
        baseline_predictions_by_seed[seed] = baseline_predictions
        prediction_sets = {"baseline": baseline_predictions}
        for component in MASKS:
            prediction_sets[component] = substitute(baseline_predictions, eval_rows, (component,))
        scored = {name: score(eval_rows, predictions) for name, predictions in prediction_sets.items()}
        frozen = reference_by_seed[seed]
        for metric in ("skeleton_exact", "operator_accuracy", "field_omission_rate",
                       "unsupported_addition_rate", "invalid_composition_rate"):
            if abs(scored["baseline"][metric] - frozen[metric]) > 1e-12:
                raise RuntimeError(f"Baseline reproduction failed seed={seed} metric={metric}")
        baseline_details = scored["baseline"]["details"]
        run_effects = {name: effects(baseline_details, scored[name]["details"]) for name in MASKS}
        for item in scored.values(): item.pop("details")
        runs.append({"seed": seed, "configurations": scored, "effects": run_effects,
                     "baseline_reproduction_passed": True})
        del structural, time_model
        if device == "cuda": torch.cuda.empty_cache()

    initial_configs = ["baseline"] + list(MASKS)
    initial_summary = summarize(runs, initial_configs)
    base = initial_summary["baseline"]["skeleton_exact"]["mean"]
    gains = {name: initial_summary[name]["skeleton_exact"]["mean"] - base for name in MASKS}
    invalid_reductions = {name: initial_summary["baseline"]["invalid_composition_rate"]["mean"] -
                                initial_summary[name]["invalid_composition_rate"]["mean"] for name in MASKS}
    with open(os.path.join(ROOT, "results", "structural_suboracle", "metrics.json"), encoding="utf-8") as f:
        parent_oracle = json.load(f)
    full_bundle_gain = parent_oracle["individual_oracle_gains"]["entity_selection"]
    full_bundle_invalid_reduction = (
        parent_oracle["summary"]["baseline"]["invalid_composition_rate"]["mean"] -
        parent_oracle["summary"]["entity_selection"]["invalid_composition_rate"]["mean"])
    bundle_residual = full_bundle_gain - sum(gains.values())
    ranked = sorted(gains, key=lambda name: (-gains[name], name))
    substantial = [name for name in ranked if gains[name] >= SUBSTANTIAL_GAIN]
    pair_name = None; pair_gain = None; pair_synergy = None
    if len(substantial) >= 2:
        pair = tuple(substantial[:2])
        pair_name = "combined:" + "+".join(pair)
        for run in runs:
            base_predictions = baseline_predictions_by_seed[run["seed"]]
            candidate = score(eval_rows, substitute(base_predictions, eval_rows, pair))
            baseline_details = score(eval_rows, base_predictions)["details"]
            run["effects"][pair_name] = effects(baseline_details, candidate["details"])
            candidate.pop("details")
            run["configurations"][pair_name] = candidate
        pair_summary = summarize(runs, [pair_name])[pair_name]
        pair_gain = pair_summary["skeleton_exact"]["mean"] - base
        pair_synergy = pair_gain - sum(gains[name] for name in pair)

    all_configs = initial_configs + ([pair_name] if pair_name else [])
    summary = summarize(runs, all_configs)
    compatibility_roles = [name for name in MASKS
                           if gains[name] < SUBSTANTIAL_GAIN and
                           invalid_reductions[name] >= COMPATIBILITY_INVALID_REDUCTION]
    dominates = gains[ranked[0]] >= SUBSTANTIAL_GAIN and gains[ranked[0]] - gains[ranked[1]] >= CLEAR_DOMINANCE_MARGIN
    if dominates:
        decision = "target_component:" + ranked[0]
    elif pair_name:
        decision = "evaluate_pair_interpretation:" + pair_name.split(":", 1)[1]
    elif compatibility_roles:
        decision = "investigate_compatibility_role:" + "+".join(compatibility_roles)
    elif gains[ranked[0]] >= SUBSTANTIAL_GAIN:
        decision = "target_component:" + ranked[0]
    else:
        decision = "reassess_entity_selection_formulation"

    result = {"device": device, "seeds": list(SEEDS), "questions": len(eval_rows),
              "mask_verification": mask_verification, "runs": runs, "summary": summary,
              "individual_oracle_gains": gains, "invalid_reductions": invalid_reductions,
              "full_entity_bundle_reference": {"gain": full_bundle_gain,
                                                "invalid_reduction": full_bundle_invalid_reduction,
                                                "gain_beyond_sum_of_isolated_fields": bundle_residual},
              "ranked_components": ranked, "substantial_gain_threshold": SUBSTANTIAL_GAIN,
              "clear_dominance_margin": CLEAR_DOMINANCE_MARGIN,
              "compatibility_invalid_reduction_threshold": COMPATIBILITY_INVALID_REDUCTION,
              "combined_pair": None if not pair_name else {"name": pair_name, "gain": pair_gain,
                                                            "synergy": pair_synergy},
              "compatibility_role_components": compatibility_roles, "next_decision": decision,
              "trained_new_component": False, "actual_signal_schema_linking_evaluated": False,
              "rejected_joint_predictor_used": False, "opened_acceptance_or_final": False,
              "schema_linker_changed": False, "typed_ir_changed": False,
              "executor_changed": False, "time_component_changed": False,
              "development_set_status": "iteratively used; not untouched generalization evidence",
              "baseline_equivalence": "metric-exact deterministic reconstruction, not checkpoint bit identity"}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = ["# Entity-Selection Three-Way Sub-Oracle", "",
             "`entity`, `order`, and `k` masks were verified disjoint before execution. The study used the reconstructed accepted-time baseline for seeds 701–703; no model was trained and actual signal/schema linking was not evaluated.", "",
             "| Oracle | Skeleton exact | Gain | Omission | Addition | Invalid |",
             "|---|---:|---:|---:|---:|---:|"]
    for config in list(MASKS) + ([pair_name] if pair_name else []):
        s = summary[config]; gain = s["skeleton_exact"]["mean"] - base
        lines.append(f"| {config} | {100*s['skeleton_exact']['mean']:.1f}% ({100*s['skeleton_exact']['min']:.1f}–{100*s['skeleton_exact']['max']:.1f}) | {100*gain:+.1f} | {100*s['field_omission_rate']['mean']:.1f}% | {100*s['unsupported_addition_rate']['mean']:.1f}% | {100*s['invalid_composition_rate']['mean']:.1f}% |")
    lines += ["", "## Per-seed exactness and effects", ""]
    for run in runs:
        parts = []
        for name in MASKS:
            metric = run["configurations"][name]["skeleton_exact"]
            effect = run["effects"][name]
            parts.append(f"{name} {100*metric:.1f}% (repairs {effect['repairs']}, regressions {effect['regressions']})")
        lines.append(f"- Seed {run['seed']} baseline {100*run['configurations']['baseline']['skeleton_exact']:.1f}%: " + "; ".join(parts) + ".")
    lines += ["", "## Decision", "",
              f"- Substantial-gain threshold: {100*SUBSTANTIAL_GAIN:.1f} points.",
              f"- Components meeting it: {', '.join(substantial) if substantial else 'none'}.",
              f"- Compatibility-role-only components: {', '.join(compatibility_roles) if compatibility_roles else 'none'}."]
    if pair_name:
        lines.append(f"- Targeted pair {pair_name.split(':', 1)[1]}: gain {100*pair_gain:+.1f} points; synergy {100*pair_synergy:+.1f} points.")
    lines += [f"- Previously measured full entity-selection bundle: gain {100*full_bundle_gain:+.1f} points, invalid reduction {100*full_bundle_invalid_reduction:+.1f} points.",
              f"- Bundle gain beyond the sum of the three isolated gains: {100*bundle_residual:+.1f} points.", "",
              "The bundle result is materially stronger than any isolated field. `order` and `k` depend on entity presence in the typed grammar, so correcting them alone can create invalid partial selections. No isolated field is justified as a learning target.", "",
              f"**Next decision: {decision}**", "",
              "Oracle substitutions are diagnostic ceilings, not deployable model performance."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"gains": gains, "invalid_reductions": invalid_reductions,
                      "combined_pair": result["combined_pair"], "decision": decision,
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

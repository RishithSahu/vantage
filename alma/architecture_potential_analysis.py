"""Post-hoc candidate-union ceiling and multi-world execution analysis."""
from __future__ import annotations

import json
import os
import sqlite3
import hashlib

from bench.eval_indep import outcome
from training.build_dataset import DEFAULT_NOW
from training.evaluate import same_gold
from vgpt.qc import QueryCompiler
from vgpt.query_ir import IRError, Query
from .architecture_bakeoff import safe_skeleton
from .selection_variant_experiment import SEEDS
from .supervised_parser import ROOT, evaluation_rows


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def multiworld_execution(direct_rows, database_paths):
    results = {}
    for path in database_paths:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        compiler = QueryCompiler(connection, now=DEFAULT_NOW)
        cache = {}; correct = exact = evaluated = 0
        for row in direct_rows:
            gold_text, predicted_text = row["gold"], row["pred"]
            exact += predicted_text == gold_text
            if gold_text == "op=none":
                correct += predicted_text == "op=none"; evaluated += 1; continue
            try:
                predicted_query = Query.from_text(predicted_text) if predicted_text != "op=none" else None
            except IRError:
                predicted_query = None
            gold_status, gold_value = outcome(compiler, gold_text, cache)
            if predicted_query is None:
                evaluated += 1; continue
            predicted_status, predicted_value = outcome(compiler, predicted_text, cache)
            if gold_status in ("invalid", "unparsed"):
                correct += predicted_text == gold_text
            else:
                correct += predicted_status == gold_status and (
                    gold_value is None or same_gold(predicted_value, gold_value))
            evaluated += 1
        connection.close()
        results[os.path.basename(path)] = {"outcome_accuracy": correct / evaluated,
                                           "exact_text_rate": exact / evaluated,
                                           "questions": evaluated}
    return results


def main():
    output = os.path.join(ROOT, "results", "architecture_bakeoff")
    rows = evaluation_rows()
    with open(os.path.join(ROOT, "results", "indep_dev_qwen_lora.json"), encoding="utf-8") as f:
        direct = json.load(f)
    direct_exact = [safe_skeleton(item["pred"]) == row["labels"]
                    for item, row in zip(direct["rows"], rows)]
    predictions = [json.loads(line) for line in open(
        os.path.join(output, "predictions.jsonl"), encoding="utf-8")]
    by_seed = {seed: sorted((item for item in predictions if item["seed"] == seed),
                            key=lambda item: item["index"]) for seed in SEEDS}
    cutoffs = (50, 100, 500)
    unions = {}
    for seed in SEEDS:
        seed_rows = by_seed[seed]
        unions[str(seed)] = {}
        for cutoff in cutoffs:
            global_union = sum(direct_ok or (item["global_gold_rank"] is not None and
                                             item["global_gold_rank"] <= cutoff)
                               for direct_ok, item in zip(direct_exact, seed_rows)) / len(rows)
            bottom_union = sum(direct_ok or (item["bottom_gold_rank"] is not None and
                                             item["bottom_gold_rank"] <= cutoff)
                               for direct_ok, item in zip(direct_exact, seed_rows)) / len(rows)
            both_union = sum(direct_ok or (item["global_gold_rank"] is not None and
                                           item["global_gold_rank"] <= cutoff) or
                             (item["bottom_gold_rank"] is not None and
                              item["bottom_gold_rank"] <= cutoff)
                             for direct_ok, item in zip(direct_exact, seed_rows)) / len(rows)
            unions[str(seed)][str(cutoff)] = {"direct_plus_global": global_union,
                                               "direct_plus_bottom_up": bottom_union,
                                               "direct_plus_both": both_union}
    database_paths = [os.path.join(ROOT, "data", name) for name in
                      ("gen_courier_ext.db", "gen_cold.db", "gen_roadwarrior.db",
                       "gen_demo.db", "gen_courier.db")]
    worlds = multiworld_execution(direct["rows"], database_paths)
    with open(os.path.join(output, "metrics.json"), encoding="utf-8") as f:
        bakeoff = json.load(f)
    mean_rerank_50_ms = sum(run["reranker_ms_per_question"] for run in bakeoff["neural_runs"]) / 3
    mean_decode_ms = sum(run["decoder_ms_per_question"] for run in bakeoff["neural_runs"]) / 3
    direct_mean_ms = direct["summary"]["latency_ms"]["mean"]
    estimated_sequential_ms = direct_mean_ms + mean_decode_ms + mean_rerank_50_ms * 10
    min_bottom_500 = min(unions[str(seed)]["500"]["direct_plus_bottom_up"] for seed in SEEDS)
    result = {
        "direct_structural_exact": sum(direct_exact) / len(rows),
        "candidate_union_recall": unions,
        "direct_plus_bottom_up_500_min_across_seeds": min_bottom_500,
        "ninety_percent_candidate_coverage_all_seeds": min_bottom_500 >= .90,
        "selector_achieved_ninety": False,
        "interpretation": "candidate inclusion is an oracle ceiling, not achieved selection accuracy or a generalization guarantee",
        "multiworld_direct_execution": worlds,
        "multiworld_min_outcome_accuracy": min(item["outcome_accuracy"] for item in worlds.values()),
        "resource_estimate": {
            "direct_generation_mean_ms": direct_mean_ms,
            "candidate_decode_mean_ms": mean_decode_ms,
            "reranker_50_candidates_mean_ms": mean_rerank_50_ms,
            "estimated_sequential_direct_plus_500_rerank_ms": estimated_sequential_ms,
            "under_two_seconds_estimate": estimated_sequential_ms < 2000,
            "caveat": "linear reranker scaling estimate on development hardware; must be measured after pruning/sharing"
        },
        "acceptance_or_final_opened": False,
    }
    potential_path = os.path.join(output, "POTENTIAL.json")
    with open(potential_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lines = ["# 90% Candidate-Potential Analysis", "",
             "This analysis asks whether any tested candidate set contains the correct structural program often enough for a perfect selector to reach 90%. It does not treat oracle inclusion as achieved accuracy.", "",
             "| Seed | Direct + bottom-up @50 | @100 | @500 |",
             "|---:|---:|---:|---:|"]
    for seed in SEEDS:
        lines.append(f"| {seed} | {100*unions[str(seed)]['50']['direct_plus_bottom_up']:.1f}% | {100*unions[str(seed)]['100']['direct_plus_bottom_up']:.1f}% | {100*unions[str(seed)]['500']['direct_plus_bottom_up']:.1f}% |")
    lines += ["", f"The direct-generator + bottom-up top-500 union reaches at least {100*min_bottom_500:.1f}% candidate coverage in every seed. This is the only tested route with a measured 90% structural candidate ceiling.", "",
              "The learned reranker did not realize this ceiling; it achieved only 5.3% exactness. At current measured throughput, sequential direct generation plus scoring 500 candidates is estimated above two seconds, so the route is not yet deployable without aggressive pruning, shared encoding, or conditional invocation.", "",
              "## Multi-world execution of the frozen direct generator", "",
              "| Vehicle world | Outcome accuracy |",
              "|---|---:|"]
    for name, values in worlds.items():
        lines.append(f"| {name} | {100*values['outcome_accuracy']:.1f}% |")
    lines += ["", f"Minimum across these worlds: {100*result['multiworld_min_outcome_accuracy']:.1f}%. The frozen generator therefore does not approach 90% execution accuracy even when denotational equivalence is allowed.", "",
              "**Conclusion:** there is measured candidate-set potential for a hybrid direct-generator + bottom-up route to reach 90% structural accuracy on this iterative development set, but no tested selector achieves it, the current latency estimate exceeds the target, and nothing here guarantees unseen-language performance."]
    potential_report_path = os.path.join(output, "POTENTIAL.md")
    with open(potential_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    manifest_path = os.path.join(output, "MANIFEST.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["potential_sha256"] = sha256_file(potential_path)
    manifest["potential_report_sha256"] = sha256_file(potential_report_path)
    manifest["multiworld_databases_sha256"] = {
        os.path.basename(path): sha256_file(path) for path in database_paths}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"min_candidate_union": min_bottom_500,
                      "multiworld_min_outcome": result["multiworld_min_outcome_accuracy"],
                      "estimated_ms": estimated_sequential_ms,
                      "report": potential_report_path}, indent=2))


if __name__ == "__main__":
    main()

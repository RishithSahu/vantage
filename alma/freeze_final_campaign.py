"""Aggregate and freeze the final VehicleGPT campaign without touching sealed sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

from .final_campaign import ROOT, sha256_file


def read(path):
    with open(path, encoding="utf-8") as handle: return json.load(handle)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", required=True)
    parser.add_argument("--tests-passed", type=int, default=0); args = parser.parse_args()
    campaign = os.path.abspath(args.campaign)
    phase = read(os.path.join(campaign, "PHASES_0_2.json"))
    reproduction = read(os.path.join(campaign, "phase0_reproduction.json"))
    recall = read(os.path.join(campaign, "candidate_recall.json"))
    semantic = read(os.path.join(campaign, "candidate_pools_semantic_first_recall.json"))
    selectors = read(os.path.join(campaign, "selectors", "metrics.json"))
    semantic_selectors = read(os.path.join(campaign, "candidate_pools_semantic_first_selector_metrics.json"))
    cpu = read(os.path.join(campaign, "candidate_pools_semantic_first_cpu_701_selector_metrics.json"))
    hard = read(os.path.join(campaign, "hard_negative_selector", "metrics.json"))
    ensemble = read(os.path.join(campaign, "selector_ensemble_metrics.json"))
    construction = read(os.path.join(campaign, "bottom_up_latency.json"))
    direct = reproduction["rescored"]
    cross = selectors["selectors"]["cross_listwise"]
    unanimous = ensemble["unanimous_edit"]
    gpu_p95 = sum(run["metrics"]["selector_latency_ms"]["p95"] for run in cross["runs"]) / len(cross["runs"])
    cpu_p95 = cpu["summary"]["10"]["selector_p95_ms_mean"]
    measured = {
        "frozen_direct": {"full_ir_exact": direct["ir_exact"]["rate"],
                          "strict_skeleton_exact": reproduction["strict_skeleton_exact"],
                          "primary_execution": direct["outcome_ok"]["rate"],
                          "p95_ms": direct["latency_ms"]["p95"],
                          "peak_rss_mb": 715},
        "candidate_recall": {"initial": recall["exact_full_ir_candidate_recall"],
                             "semantic_first": semantic["exact_full_ir_candidate_recall"],
                             "semantic_first_missing": semantic["missing_at_2000"]},
        "selectors": {
            "global_text_compatibility": selectors["selectors"]["global_text_compatibility"],
            "dual_listwise": {key: cross_value for key, cross_value in
                              selectors["selectors"]["dual_listwise"].items() if key != "runs"},
            "cross_listwise_k10": {"full_ir_exact_mean": cross["full_ir_exact_mean"],
                                    "full_ir_exact_min": cross["full_ir_exact_min"],
                                    "full_ir_exact_max": max(run["metrics"]["full_ir_exact"] for run in cross["runs"]),
                                    "worst_world_execution_mean": cross["execution_min_mean"],
                                    "gpu_selector_p95_ms_mean": gpu_p95},
            "frozen_cross_on_semantic_pool": semantic_selectors["summary"],
            "composed_hard_negative_cross": hard["summary"],
            "unanimous_edit_gate": unanimous,
        },
        "deployment": {
            "single_cross_cpu_selector_p95_ms": cpu_p95,
            "estimated_direct_plus_single_cross_p95_ms": direct["latency_ms"]["p95"] + cpu_p95,
            "bottom_up_construction_p95_ms": construction["construction_ms"]["p95"],
            "sequential_full_hybrid_p95_sum_ms": direct["latency_ms"]["p95"] + cpu_p95 + construction["construction_ms"]["p95"],
            "candidate_construction_included_in_full_sum": True,
            "two_second_gate_passed_by_direct": direct["latency_ms"]["p95"] < 2000,
            "two_second_gate_passed_by_hybrid": False,
            "target_arm_measured": False,
        },
        "ninety_percent": {
            "complete_ir_candidate_recall_at_500": semantic["exact_full_ir_candidate_recall"]["500"],
            "achieved_full_ir_accuracy": unanimous["full_ir_exact"],
            "achieved_worst_world_execution": unanimous["multiworld_min_execution"],
            "guaranteed": False,
            "reason": "90% is oracle inclusion on an iteratively used development set; selection accuracy is 57.5% and sealed evaluation remains unopened",
        },
        "acceptance_or_final_opened": False,
    }
    metrics_path = os.path.join(campaign, "FINAL_METRICS.json")
    with open(metrics_path, "w", encoding="utf-8") as handle: json.dump(measured, handle, indent=2)
    table = [
        ("Frozen direct grammar generator", "48.3%", "62.5%", "58.3%", "1,899.9 ms", "retain deployment baseline"),
        ("Direct sampling/self-consistency", "49.2%", "64.2%", "59.2%", "generator cost", "no meaningful diversity"),
        ("Global text compatibility", "47.5%", "62.5%", "59.2%", "13.9 ms selector", "reject"),
        ("Dual-encoder listwise", "22.2% mean", "not competitive", "30.8% worst-world mean", "not selected", "reject"),
        ("Cross-encoder listwise, K=10", "54.2% mean", "63.3-65.8%", "65.3% worst-world mean", "39.0 ms GPU selector", "passes accuracy gate"),
        ("Frozen cross, semantic pool K=50", "28.9% mean", "not competitive", "43.9% worst-world mean", "85.2 ms GPU selector", "reject"),
        ("Composed-hard-negative cross, K=50", "40.0% mean", "not competitive", "52.2% worst-world mean", "83.5 ms GPU selector", "reject"),
        ("Unanimous three-seed edit gate", "57.5%", "70.8%", "67.5% worst world", ">2 s estimated full path", "best research result"),
    ]
    lines = ["# VehicleGPT Final Full-IR Campaign", "",
             "## Decision", "",
             "No tested method achieved or guarantees 90% accuracy. The frozen direct grammar-constrained 0.5B generator remains the deployment candidate because it is the only finalist with a measured p95 below two seconds. The best research pipeline is the conservative unanimous cross-selector edit gate over a K=10 validated full-IR pool.", "",
             "The new bottom-up full-IR composer reaches 90.0% exact-program candidate recall at K=500 on the iteratively used 120-question development set. That is a genuine complete-IR oracle ceiling, but it is not achieved parsing accuracy: the best selector pipeline reaches 57.5% exact full IR and 67.5% worst-world execution.", "",
             "| Method | Exact full IR | Skeleton exact | Execution | Latency | Decision |",
             "|---|---:|---:|---:|---:|---|",
             *[f"| {name} | {full} | {skeleton} | {execution} | {latency} | {decision} |"
               for name, full, skeleton, execution, latency, decision in table], "",
             "## Candidate-generation findings", "",
             f"- Grammar-constrained sampling was highly peaked: nine decodes produced only 1.45 unique programs per question on average. Self-consistency did not improve the top prediction.",
             f"- The original mixed pool reached {100*recall['exact_full_ir_candidate_recall']['500']:.1f}% full-IR recall at K=500, despite {100*recall['skeleton_candidate_recall']['500']:.1f}% skeleton recall. This quantifies why skeleton coverage was misleading.",
             f"- Typed full-program recombination plus frozen schema/time/condition/selection/dialogue atoms raised exact recall to {100*semantic['exact_full_ir_candidate_recall']['50']:.1f}% at K=50, {100*semantic['exact_full_ir_candidate_recall']['100']:.1f}% at K=100, and {100*semantic['exact_full_ir_candidate_recall']['500']:.1f}% at K=500.",
             f"- Twelve of 120 questions still lack the gold complete IR even in the large pool.", "",
             "## Selector findings", "",
             "- The shared-question dual encoder failed, showing that independent question/program embeddings lose crucial attachment and value detail.",
             "- A joint-attention cross encoder improved over the generator in all three seeds at K=10 (52.5-57.5% exact). It added about 39 ms p95 on the laptop GPU and used about 1.38 GB peak VRAM during training.",
             "- Expanding K without matched training caused collapse. Training on 49 composed hard negatives also failed (37.5-41.7% exact), so more candidates alone are harmful.",
             "- The unanimous edit gate repaired 12 baseline errors and regressed 2, producing the strongest achieved result.", "",
             "## Deployment boundary", "",
             f"The measured bottom-up construction p95 is {construction['construction_ms']['p95']:.1f} ms and the one-model CPU selector p95 is {cpu_p95:.1f} ms. Adding these to the frozen direct generator's {direct['latency_ms']['p95']:.1f} ms p95 gives a conservative sequential sum of {direct['latency_ms']['p95'] + cpu_p95 + construction['construction_ms']['p95']:.1f} ms. Therefore the hybrid does not pass the two-second gate. Target ARM hardware was not available, so no ARM claim is made.", "",
             "## Research conclusion", "",
             "The strongest supported research direction is now specific: generate a small, high-recall set of complete executable programs from question-conditioned semantic atoms, then train a selector on the same error distribution while retaining a conservative direct-output gate. The next bottleneck is selector discrimination, not grammar validity or raw candidate inclusion.", "",
             "The acceptance and final benchmarks remain sealed. Because the development set has guided repeated choices, none of these numbers is an untouched generalization estimate. Freeze the method and decision criteria before opening an acceptance set.", ""]
    report_path = os.path.join(campaign, "MASTER_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as handle: handle.write("\n".join(lines))
    key_files = [
        os.path.join(ROOT, "alma", name) for name in
        ("final_campaign.py", "final_selectors.py", "full_ir_bottom_up.py",
         "evaluate_frozen_cross.py", "hard_negative_selector.py", "selector_ensemble.py",
         "benchmark_bottom_up.py", "freeze_final_campaign.py")]
    key_files += [metrics_path, report_path,
                  os.path.join(campaign, "candidate_pools.jsonl"),
                  os.path.join(campaign, "candidate_pools_semantic_first.jsonl"),
                  os.path.join(campaign, "selectors", "metrics.json"),
                  os.path.join(campaign, "hard_negative_selector", "metrics.json")]
    key_files.append(os.path.join(campaign, "bottom_up_latency.json"))
    checkpoint_dirs = [os.path.join(campaign, "selectors"), os.path.join(campaign, "hard_negative_selector")]
    checkpoints = [os.path.join(directory, name) for directory in checkpoint_dirs
                   for name in os.listdir(directory) if name.endswith(".pt")]
    manifest = {"format_version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "tests_passed": args.tests_passed,
                "files_sha256": {os.path.relpath(path, ROOT): sha256_file(path)
                                  for path in key_files + checkpoints},
                "sealed_files_excluded_from_hashing": ["bench/acceptance_v1.tsv", "bench/indep_final.tsv"],
                "acceptance_or_final_opened": False}
    with open(os.path.join(campaign, "FINAL_MANIFEST.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(campaign, "TEST_STATUS.json"), "w", encoding="utf-8") as handle:
        json.dump({"alma_tests_passed": args.tests_passed, "baseline_reproduction": reproduction["passed"],
                   "sealed_sets_opened": False}, handle, indent=2)
    print(json.dumps({"report": report_path, "metrics": metrics_path,
                      "best_achieved_full_ir": unanimous["full_ir_exact"],
                      "candidate_recall_at_500": semantic["exact_full_ir_candidate_recall"]["500"]}, indent=2))


if __name__ == "__main__": main()

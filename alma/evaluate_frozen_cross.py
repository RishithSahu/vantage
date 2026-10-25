"""Evaluate frozen cross-listwise selectors on a newly ordered, frozen pool."""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from bench.indep_io import load
from .final_campaign import DEV_PATH, ROOT, sha256_file
from .final_selectors import CrossListwise, SEEDS, cross_select, metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", required=True)
    parser.add_argument("--pool", default="candidate_pools_semantic_first.jsonl")
    parser.add_argument("--limits", default="10,50,100")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    args = parser.parse_args(); campaign = os.path.abspath(args.campaign)
    pool_path = os.path.join(campaign, args.pool)
    pools = [json.loads(line) for line in open(pool_path, encoding="utf-8")]
    rows = load(DEV_PATH); device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    seeds = tuple(int(value) for value in args.seeds.split(","))
    checkpoint_dir = os.path.join(campaign, "selectors")
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    result = {"pool": args.pool, "pool_sha256": sha256_file(pool_path), "device": device,
              "runs": {}, "acceptance_or_final_opened": False}
    prediction_rows = []
    for seed in seeds:
        model = CrossListwise(model_path)
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"cross_listwise_seed{seed}.pt"),
                                map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"]); model.to(device)
        result["runs"][str(seed)] = {}
        for limit in [int(value) for value in args.limits.split(",")]:
            predictions, latencies = cross_select(model, rows, pools, device, limit)
            result["runs"][str(seed)][str(limit)] = metrics(rows, predictions, latencies)
            for row, prediction in zip(rows, predictions):
                prediction_rows.append({"seed": seed, "limit": limit, "id": row["id"],
                                        "gold": row["ir"], "pred": prediction})
        del model
        if device == "cuda": torch.cuda.empty_cache()
    result["summary"] = {str(limit): {
        "full_ir_exact_mean": statistics.mean(result["runs"][str(seed)][str(limit)]["full_ir_exact"] for seed in seeds),
        "full_ir_exact_min": min(result["runs"][str(seed)][str(limit)]["full_ir_exact"] for seed in seeds),
        "multiworld_min_mean": statistics.mean(result["runs"][str(seed)][str(limit)]["multiworld_min_execution"] for seed in seeds),
        "selector_p95_ms_mean": statistics.mean(result["runs"][str(seed)][str(limit)]["selector_latency_ms"]["p95"] for seed in seeds),
    } for limit in [int(value) for value in args.limits.split(",")]}
    stem = os.path.splitext(args.pool)[0] + ("" if args.device == "auto" and seeds == SEEDS else f"_{device}_{'-'.join(map(str, seeds))}")
    with open(os.path.join(campaign, stem + "_selector_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(campaign, stem + "_selector_predictions.jsonl"), "w", encoding="utf-8") as handle:
        for item in prediction_rows: handle.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__": main()

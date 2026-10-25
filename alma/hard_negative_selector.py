"""Listwise cross-encoder trained on bottom-up complete-program hard negatives."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics

import torch

from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt.qc import QueryCompiler
from .final_campaign import DEV_PATH, ROOT, WORLD_NAMES, Pool, full_key, sha256_file
from .final_selectors import (CrossListwise, SEEDS, cross_select, metrics, train_cross,
                              training_candidate_sets)
from .full_ir_bottom_up import atoms, compose, kbest_combinations


def rich_training_sets(rows, base_sets, profile, maximum=50):
    output = []
    for row, base in zip(rows, base_sets):
        records = [item.as_dict() for item in base]
        atom_row = {"text": row["text"], "prev_ir": None}
        lists, ranks = atoms(atom_row, records, profile)
        pool = Pool(profile); pool.add(row["ir"], "gold")
        for item in base: pool.add(item.ir, "base_negative")
        for _, combination in kbest_combinations(lists, ranks, maximum=1200):
            try: candidate = compose(combination)
            except Exception: continue
            pool.add(candidate, "composed_negative")
            if len(pool._items) >= maximum * 5: break
        gold = full_key(row["ir"]); items = pool.ordered()
        positives = [item for item in items if full_key(item.ir) == gold]
        negatives = [item for item in items if full_key(item.ir) != gold]
        output.append([positives[0]] + negatives[:maximum - 1])
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", required=True)
    parser.add_argument("--pool", default="candidate_pools_semantic_first.jsonl")
    parser.add_argument("--limit", type=int, default=50); parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args(); campaign = os.path.abspath(args.campaign)
    pool_path = os.path.join(campaign, args.pool)
    pools = [json.loads(line) for line in open(pool_path, encoding="utf-8")]; dev = load(DEV_PATH)
    conn = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', WORLD_NAMES[0])}?mode=ro", uri=True)
    profile = QueryCompiler(conn, now=DEFAULT_NOW).profile; conn.close()
    train_rows, base_sets = training_candidate_sets(profile, maximum=32)
    train_sets = rich_training_sets(train_rows, base_sets, profile, args.limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    output = os.path.join(campaign, "hard_negative_selector"); os.makedirs(output, exist_ok=True)
    result = {"pool_sha256": sha256_file(pool_path), "candidate_limit": args.limit,
              "training_examples": len(train_rows), "device": device, "runs": [],
              "acceptance_or_final_opened": False}
    predictions_out = []
    for seed in SEEDS:
        model = CrossListwise(model_path)
        training = train_cross(model, train_rows, train_sets, device, seed, args.epochs, args.limit)
        predictions, latencies = cross_select(model, dev, pools, device, args.limit)
        run = {"seed": seed, "training": training, "metrics": metrics(dev, predictions, latencies)}
        result["runs"].append(run)
        torch.save({"seed": seed, "model_state_dict": {key: value.detach().cpu()
                    for key, value in model.state_dict().items()}},
                   os.path.join(output, f"hard_cross_seed{seed}.pt"))
        for row, prediction in zip(dev, predictions):
            predictions_out.append({"seed": seed, "id": row["id"], "gold": row["ir"], "pred": prediction})
        del model
        if device == "cuda": torch.cuda.empty_cache()
    result["summary"] = {
        "full_ir_exact_mean": statistics.mean(run["metrics"]["full_ir_exact"] for run in result["runs"]),
        "full_ir_exact_min": min(run["metrics"]["full_ir_exact"] for run in result["runs"]),
        "multiworld_min_mean": statistics.mean(run["metrics"]["multiworld_min_execution"] for run in result["runs"]),
        "selector_p95_ms_mean": statistics.mean(run["metrics"]["selector_latency_ms"]["p95"] for run in result["runs"]),
    }
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(output, "predictions.jsonl"), "w", encoding="utf-8") as handle:
        for item in predictions_out: handle.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__": main()

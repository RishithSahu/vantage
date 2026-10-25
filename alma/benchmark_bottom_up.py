"""Measure bottom-up candidate construction, excluding model generation and selection."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import time

from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt.qc import QueryCompiler
from .final_campaign import DEV_PATH, ROOT, WORLD_NAMES, Pool, percentile
from .full_ir_bottom_up import atoms, compose, kbest_combinations


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", required=True)
    parser.add_argument("--max-valid", type=int, default=4000); args = parser.parse_args()
    campaign = os.path.abspath(args.campaign)
    records = [json.loads(line) for line in open(os.path.join(campaign, "candidate_pools.jsonl"), encoding="utf-8")]
    rows = load(DEV_PATH)
    conn = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', WORLD_NAMES[0])}?mode=ro", uri=True)
    profile = QueryCompiler(conn, now=DEFAULT_NOW).profile
    durations = []; sizes = []
    for row, record in zip(rows, records):
        started = time.perf_counter(); pool = Pool(profile)
        for rank, item in enumerate(record["candidates"], 1):
            for source in item["sources"]: pool.add(item["ir"], source, rank)
        lists, ranks = atoms(row, record["candidates"], profile); added = 0
        for score, combination in kbest_combinations(lists, ranks):
            try: candidate = compose(combination)
            except Exception: continue
            before = len(pool._items); pool.add(candidate, "full_ir_bottom_up", int(score))
            added += len(pool._items) > before
            if added >= args.max_valid: break
        durations.append((time.perf_counter() - started) * 1000); sizes.append(len(pool._items))
    conn.close()
    result = {"n": len(rows), "max_valid": args.max_valid,
              "construction_ms": {"mean": statistics.mean(durations), "p50": statistics.median(durations),
                                  "p95": percentile(durations, .95), "max": max(durations)},
              "pool_size": {"mean": statistics.mean(sizes), "min": min(sizes), "max": max(sizes)},
              "includes": "atom extraction, composition, validation and deduplication",
              "excludes": "direct model generation, selector scoring and execution",
              "acceptance_or_final_opened": False}
    with open(os.path.join(campaign, "bottom_up_latency.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

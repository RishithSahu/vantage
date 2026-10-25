"""Gold-blind consensus gates over the three frozen cross-selector seeds."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

from bench.indep_io import load
from .final_campaign import DEV_PATH, full_key
from .final_selectors import metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", required=True)
    args = parser.parse_args(); campaign = os.path.abspath(args.campaign)
    path = os.path.join(campaign, "selectors", "predictions.jsonl")
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    base = {item["id"]: item["pred"] for item in items if item["selector"] == "generator_greedy"}
    votes = defaultdict(list)
    for item in items:
        if item["selector"] == "cross_listwise": votes[item["id"]].append(item["pred"])
    rows = load(DEV_PATH); result = {}; output = []
    for rule in ("majority", "two_vote_edit", "unanimous_edit"):
        predictions = []
        for row in rows:
            options = votes[row["id"]]; counts = Counter(full_key(value) for value in options)
            winning_key, count = counts.most_common(1)[0]
            winner = next(value for value in options if full_key(value) == winning_key)
            if rule == "unanimous_edit" and count < 3: winner = base[row["id"]]
            if rule == "two_vote_edit" and count < 2: winner = base[row["id"]]
            predictions.append(winner); output.append({"rule": rule, "id": row["id"], "pred": winner})
        measured = metrics(rows, predictions)
        measured["repairs_vs_greedy"] = sum(full_key(row["ir"]) != full_key(base[row["id"]]) and
                                             full_key(row["ir"]) == full_key(prediction)
                                             for row, prediction in zip(rows, predictions))
        measured["regressions_vs_greedy"] = sum(full_key(row["ir"]) == full_key(base[row["id"]]) and
                                                 full_key(row["ir"]) != full_key(prediction)
                                                 for row, prediction in zip(rows, predictions))
        result[rule] = measured
    with open(os.path.join(campaign, "selector_ensemble_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(campaign, "selector_ensemble_predictions.jsonl"), "w", encoding="utf-8") as handle:
        for item in output: handle.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

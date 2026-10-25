"""Three-seed stability check for the 100-example supervised parser."""
import json
import os
import statistics

import torch

from .supervised_parser import ROOT, StructuralParser, diverse_prefix, evaluate, evaluation_rows, train, training_pool


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    output = os.path.join(ROOT, "results", "supervised_structural_parser", "stability.json")
    rows = diverse_prefix(training_pool())[:100]
    held = evaluation_rows()
    runs = []
    for seed in (519, 520, 521):
        torch.manual_seed(seed)
        if device == "cuda": torch.cuda.manual_seed_all(seed)
        model = StructuralParser(model_path)
        stats = train(model, rows, device, seed=seed)
        metrics = evaluate(model, held, device)
        runs.append({"seed": seed, "skeleton_exact": metrics["skeleton_exact"],
                     "operation": metrics["field_accuracy"]["op"],
                     "aggregation": metrics["field_accuracy"]["agg"],
                     "time_type": metrics["field_accuracy"]["time_type"],
                     "seconds": stats["seconds"]})
        del model
        if device == "cuda": torch.cuda.empty_cache()
    summary = {name: {"mean": statistics.mean(r[name] for r in runs),
                      "min": min(r[name] for r in runs), "max": max(r[name] for r in runs)}
               for name in ("skeleton_exact", "operation", "aggregation", "time_type")}
    result = {"examples": 100, "evaluation_examples": len(held), "runs": runs, "summary": summary,
              "opened_final_benchmark": False}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

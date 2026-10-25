"""Compact natural-language-supervised structural parser experiment.

This predicts a compositional program skeleton from real questions. Open-schema
signal binding remains a separate frozen component, so this experiment cannot
damage schema-linking representations.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from bench.indep_io import load
from vgpt.query_ir import AGGS, OPS, Query

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = {
    "op": list(OPS) + ["none"],
    "agg": ["none"] + list(AGGS),
    "time_type": ["current", "relative", "calendar", "date", "all", "since_event", "range"],
    "time_unit": ["none", "minute", "hour", "day", "week", "month"],
    "time2_type": ["none", "current", "relative", "calendar", "date", "all", "since_event", "range"],
    "condition_shape": ["none", "named", "threshold", "multi_named", "multi_threshold", "mixed"],
    "select_entity": ["none", "trip", "charge_session"],
    "select_order": ["none", "asc", "desc"],
    "select_k": ["none", "one", "many"],
    "groups_count": ["none", "one", "two", "many"],
    "event": ["none", "present"],
    "before": ["none", "present"],
    "after": ["none", "present"],
    "target_count": ["none", "one", "two"],
    "refs": ["none", "present"],
}


def skeleton(q):
    conds = q.scope.conditions
    if not conds: cshape = "none"
    else:
        kinds = ["named" if c.predicate else "threshold" for c in conds]
        cshape = kinds[0] if len(kinds) == 1 else ("multi_" + kinds[0] if len(set(kinds)) == 1 else "mixed")
    sel = q.scope.select
    groups = len(q.groups)
    return {
        "op": q.op if q.op in LABELS["op"] else "none",
        "agg": q.agg or "none",
        "time_type": q.scope.time.type,
        "time_unit": q.scope.time.unit or "none",
        "time2_type": q.scope_b.time.type if q.scope_b else "none",
        "condition_shape": cshape,
        "select_entity": sel.entity if sel else "none",
        "select_order": sel.order if sel else "none",
        "select_k": "none" if not sel or sel.k is None else ("one" if sel.k == 1 else "many"),
        "groups_count": "none" if groups == 0 else ("one" if groups == 1 else ("two" if groups == 2 else "many")),
        "event": "present" if q.event else "none",
        "before": "present" if q.event and q.event.before else "none",
        "after": "present" if q.event and q.event.after else "none",
        "target_count": "two" if q.target2 else ("one" if q.target else "none"),
        "refs": "present" if q.refs else "none",
    }


def _item(text, ir, source, family="unknown", previous=None):
    q = Query.from_text(ir)
    model_text = f"Previous request: {previous}\nCurrent request: {text}" if previous else text
    return {"text": model_text, "question": text, "ir": ir, "source": source,
            "family": family, "labels": skeleton(q)}


def training_pool():
    rows = []
    with open(os.path.join(ROOT, "bench", "ir_gold.tsv"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip() or line.startswith("#"): continue
            text, ir = line.rstrip("\n").split("\t")[:2]
            rows.append(_item(text, ir, "ir_gold", f"ir_{i}"))
    for r in load(os.path.join(ROOT, "bench", "diag_v1.tsv")):
        rows.append(_item(r["text"], r["ir"], "diag_v1", r["family"],
                          None if r["prev_text"] == "-" else r["prev_text"]))
    return rows


def evaluation_rows():
    rows = []
    for r in load(os.path.join(ROOT, "bench", "indep_dev.tsv")):
        rows.append(_item(r["text"], r["ir"], "indep_dev", r["family"],
                          None if r["prev_text"] == "-" else r["prev_text"]))
    return rows


def diverse_prefix(rows, seed=409):
    """Greedy active-learning proxy: maximize unseen structural attributes."""
    remaining = list(rows)
    random.Random(seed).shuffle(remaining)
    chosen, seen = [], Counter()
    while remaining:
        def gain(row):
            signature = tuple(row["labels"].items())
            novelty = sum(1 / (1 + seen[(k, v)]) for k, v in signature)
            return novelty + 2 / (1 + seen[("signature", signature)])
        best = max(range(len(remaining)), key=lambda i: gain(remaining[i]))
        row = remaining.pop(best)
        chosen.append(row)
        signature = tuple(row["labels"].items())
        for item in signature: seen[item] += 1
        seen[("signature", signature)] += 1
    return chosen


class StructuralParser(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        dim = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({name: nn.Linear(dim, len(values)) for name, values in LABELS.items()})

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=160, return_tensors="pt")

    def forward(self, batch):
        out = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return {name: head(pooled) for name, head in self.heads.items()}


def train(model, rows, device, epochs=18, batch_size=16, lr=2e-5, seed=419):
    random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model.to(device)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    importance = {name: .5 for name in LABELS}
    importance.update({"op": 4.0, "agg": 2.0, "time_type": 2.0, "time_unit": 1.5,
                       "condition_shape": 1.5, "event": 1.5, "target_count": 1.5})
    class_weights = {}
    for name, values in LABELS.items():
        counts = Counter(r["labels"][name] for r in rows)
        weights = [0.0 if counts[v] == 0 else min(4.0, (len(rows) / (len(values) * counts[v])) ** .5)
                   for v in values]
        class_weights[name] = torch.tensor(weights, dtype=torch.float32, device=device)
    losses, started = [], time.perf_counter()
    for _ in range(epochs):
        model.train(); total = 0.; seen = 0
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokenize([x["text"] for x in items]).items()}
            gold = {name: torch.tensor([LABELS[name].index(x["labels"][name]) for x in items], device=device)
                    for name in LABELS}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(batch)
                terms = [importance[name] * nn.functional.cross_entropy(
                    logits[name], gold[name], weight=class_weights[name]) for name in LABELS]
                loss = sum(terms) / sum(importance.values())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss.detach()) * len(items); seen += len(items)
        losses.append(total / seen)
    return {"examples": len(rows), "epochs": epochs, "seconds": time.perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
            "losses": losses}


def evaluate(model, rows, device):
    loader = DataLoader(rows, batch_size=64, shuffle=False, collate_fn=lambda x: x)
    correct = Counter(); exact = 0; total = 0; errors = []; predictions = []
    model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokenize([x["text"] for x in items]).items()}
            logits = model(batch)
            pred = {name: logits[name].argmax(-1).tolist() for name in LABELS}
            for i, row in enumerate(items):
                got = {name: LABELS[name][pred[name][i]] for name in LABELS}
                matches = {name: got[name] == row["labels"][name] for name in LABELS}
                for name, ok in matches.items(): correct[name] += ok
                exact += all(matches.values()); total += 1
                predictions.append({"question": row["question"], "family": row["family"],
                                    "gold": row["labels"], "pred": got, "matches": matches})
                if not all(matches.values()) and len(errors) < 20:
                    errors.append({"question": row["question"], "gold": row["labels"], "pred": got,
                                   "wrong": [n for n, ok in matches.items() if not ok]})
    return {"n": total, "skeleton_exact": exact / total,
            "field_accuracy": {name: correct[name] / total for name in LABELS},
            "predictions": predictions, "errors": errors}


def diagnostic(predictions):
    groups = {
        "operator": ("op",),
        "aggregation": ("agg",),
        "time": ("time_type", "time_unit", "time2_type", "before", "after"),
        "scope_attachment": ("condition_shape", "groups_count"),
        "selection": ("select_entity", "select_order", "select_k"),
        "event": ("event", "before", "after"),
        "target_arity": ("target_count",),
        "dialogue_inheritance": ("refs",),
    }
    counts = {name: 0 for name in groups}
    omissions = additions = invalid = 0
    wrong_field_hist = Counter()
    for row in predictions:
        gold, pred = row["gold"], row["pred"]
        wrong = [name for name in LABELS if gold[name] != pred[name]]
        wrong_field_hist[str(len(wrong))] += 1
        for name, fields in groups.items():
            counts[name] += any(field in wrong for field in fields)
        omissions += any(gold[f] != "none" and pred[f] == "none" for f in LABELS)
        additions += any(gold[f] == "none" and pred[f] != "none" for f in LABELS)
        p = pred
        invalid_row = ((p["op"] in ("aggregate", "extreme", "compare") and p["agg"] == "none") or
                       (p["op"] not in ("aggregate", "extreme", "compare") and p["agg"] != "none") or
                       (p["time2_type"] != "none" and p["op"] != "compare") or
                       ((p["before"] != "none" or p["after"] != "none") and p["event"] == "none") or
                       (p["select_entity"] == "none" and
                        (p["select_order"] != "none" or p["select_k"] != "none")))
        invalid += invalid_row

    def oracle_exact(fields):
        return sum(all((name in fields) or row["matches"][name] for name in LABELS)
                   for row in predictions) / len(predictions)

    oracle = {name: oracle_exact(set(fields)) for name, fields in groups.items()}
    oracle["operator_plus_time"] = oracle_exact(set(groups["operator"] + groups["time"]))
    oracle["operator_time_scope"] = oracle_exact(set(groups["operator"] + groups["time"] + groups["scope_attachment"]))
    oracle["all_structure_except_op_agg"] = oracle_exact(set(LABELS) - {"op", "agg"})
    return {
        "n": len(predictions),
        "failure_categories": {name: {"count": count, "rate": count / len(predictions)}
                               for name, count in counts.items()},
        "field_omission": {"count": omissions, "rate": omissions / len(predictions)},
        "unsupported_addition": {"count": additions, "rate": additions / len(predictions)},
        "invalid_composition": {"count": invalid, "rate": invalid / len(predictions)},
        "wrong_field_count_histogram": dict(sorted(wrong_field_hist.items(), key=lambda x: int(x[0]))),
        "oracle_skeleton_exact": oracle,
        "schema_linking": {"evaluated_here": False,
                           "reason": "open-schema linking is an independent frozen component"},
    }


def majority_baseline(train_rows, eval_rows):
    major = {name: Counter(r["labels"][name] for r in train_rows).most_common(1)[0][0] for name in LABELS}
    exact = sum(all(r["labels"][name] == major[name] for name in LABELS) for r in eval_rows)
    return {"skeleton_exact": exact / len(eval_rows), "prediction": major}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    output = os.path.join(ROOT, "results", "supervised_structural_parser")
    os.makedirs(output, exist_ok=True)
    pool = diverse_prefix(training_pool())
    eval_rows = evaluation_rows()
    sizes = [n for n in (50, 100, 250, 500) if n <= len(pool)]
    result = {"device": device, "model": model_path, "training_pool": len(pool),
              "evaluation_examples": len(eval_rows), "opened_final_benchmark": False,
              "available_learning_curve_sizes": sizes,
              "unavailable_sizes": [n for n in (50, 100, 250, 500) if n > len(pool)], "runs": {}}
    for size in sizes:
        torch.manual_seed(419 + size)
        if device == "cuda": torch.cuda.manual_seed_all(419 + size)
        model = StructuralParser(model_path)
        stats = train(model, pool[:size], device)
        metrics = evaluate(model, eval_rows, device)
        result["runs"][str(size)] = {"training": stats, "evaluation": metrics,
                                     "majority_baseline": majority_baseline(pool[:size], eval_rows)}
        torch.save({"heads": model.heads.state_dict(), "size": size, "labels": LABELS},
                   os.path.join(output, f"heads_{size}.pt"))
        del model
        if device == "cuda": torch.cuda.empty_cache()
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if "100" in result["runs"]:
        result["diagnostic_100"] = diagnostic(result["runs"]["100"]["evaluation"]["predictions"])
        with open(os.path.join(output, "diagnostic.json"), "w", encoding="utf-8") as f:
            json.dump(result["diagnostic_100"], f, indent=2)
    lines = ["# Natural-Language-Supervised Structural Parser", "",
             "This development experiment uses only natural questions paired with executable IR. The independent final set and acceptance set were not opened.", "",
             f"Available safe training pool: {len(pool)} questions. Evaluation: {len(eval_rows)} independent-development questions.", "",
             "| Natural training examples | Skeleton exact | Operation | Aggregation | Condition shape | Time type | Seconds | Peak VRAM |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size in sizes:
        run = result["runs"][str(size)]; m = run["evaluation"]; s = run["training"]
        lines.append(f"| {size} | {100*m['skeleton_exact']:.1f}% | {100*m['field_accuracy']['op']:.1f}% | {100*m['field_accuracy']['agg']:.1f}% | {100*m['field_accuracy']['condition_shape']:.1f}% | {100*m['field_accuracy']['time_type']:.1f}% | {s['seconds']:.1f} | {s['peak_vram_mb']:.0f} MB |")
    lines += ["", "The 250/500 points were not fabricated: additional reviewed natural question/IR annotations are required.",
              "This phase measures structural skeletons, not full programs; open-schema signal/event/value binding remains a separate component.",
              "The detailed failure and oracle analysis is in `DIAGNOSTIC.md`; the collection protocol is in `bench/supervised_annotation`."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"pool": len(pool), "eval": len(eval_rows), "sizes": sizes,
                      "results": {n: result['runs'][str(n)]['evaluation']['skeleton_exact'] for n in sizes},
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

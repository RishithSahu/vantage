"""Formally verified contrast curriculum and dedicated time-structure probe."""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from vgpt.query_ir import CALENDAR, Query
from .supervised_parser import ROOT, evaluation_rows, training_pool

LABELS = {
    "time_type": ["current", "relative", "calendar", "date", "all", "since_event", "range"],
    "time_unit": ["none", "minute", "hour", "day", "week", "month"],
    "calendar": ["none"] + list(CALENDAR),
    "time2_type": ["none", "current", "relative", "calendar", "date", "all", "since_event", "range"],
    "event_window": ["none", "before", "after", "around"],
}


def time_labels(q):
    event_window = "none"
    if q.event:
        event_window = "around" if q.event.before and q.event.after else ("before" if q.event.before else ("after" if q.event.after else "none"))
    return {"time_type": q.scope.time.type, "time_unit": q.scope.time.unit or "none",
            "calendar": q.scope.time.name or "none",
            "time2_type": q.scope_b.time.type if q.scope_b else "none", "event_window": event_window}


def _row(text, ir, split, family):
    return {"text": text, "ir": ir, "labels": time_labels(Query.from_text(ir)),
            "source": "formal_contrast", "split": split, "family": family}


def contrast_pairs(split="train"):
    """Pairs are generated from IR specifications; held-out wording is disjoint."""
    train_templates = {
        "unit": ("Mean speed over the last {n} {unit}.", "The {n}-{unit} mean speed."),
        "calendar": ("Mean speed {period}.", "The {period} speed average."),
        "extent": ("Current speed.", "Mean speed across all recorded history."),
        "event": ("Mean speed in the ten minutes {direction} the latest warning.",
                  "The warning-relative speed window lies {direction} the newest warning."),
    }
    held_templates = {
        "unit": ("What was my average speed during the previous {n} {unit}?",
                 "How fast was I going on average across the past {n} {unit}?"),
        "calendar": ("What was the average speed {period}?", "How did speed average out {period}?"),
        "extent": ("How fast am I going right now?", "What is my mean speed over the entire history?"),
        "event": ("What was the average speed for ten minutes {direction} the most recent warning?",
                  "Show mean speed in the window {direction} the last warning."),
    }
    t = train_templates if split == "train" else held_templates
    pairs = []
    repetitions = 30 if split == "train" else 15
    for i in range(repetitions):
        template = t["unit"][i % 2]
        a = _row(template.format(n=2, unit="days"), "op=aggregate tgt=speed agg=mean t=last:2:day", split, "relative_unit")
        b = _row(template.format(n=2, unit="weeks"), "op=aggregate tgt=speed agg=mean t=last:2:week", split, "relative_unit")
        pairs.append((a, b, "relative_unit"))
        template = t["calendar"][i % 2]
        a = _row(template.format(period="today"), "op=aggregate tgt=speed agg=mean t=cal:today", split, "calendar_identity")
        b = _row(template.format(period="yesterday"), "op=aggregate tgt=speed agg=mean t=cal:yesterday", split, "calendar_identity")
        pairs.append((a, b, "calendar_identity"))
        a = _row(t["extent"][0], "op=current_value tgt=speed t=now", split, "current_vs_all")
        b = _row(t["extent"][1], "op=aggregate tgt=speed agg=mean t=all", split, "current_vs_all")
        pairs.append((a, b, "current_vs_all"))
        template = t["event"][i % 2]
        a = _row(template.format(direction="before"), "op=aggregate tgt=speed agg=mean t=all ev=ANY_WARNING bef=600", split, "event_direction")
        b = _row(template.format(direction="after"), "op=aggregate tgt=speed agg=mean t=all ev=ANY_WARNING aft=600", split, "event_direction")
        pairs.append((a, b, "event_direction"))
    return pairs


def deterministic_denotation(ir):
    """Tiny deterministic time-world: each distinct window has a distinct value."""
    q = Query.from_text(ir)
    if q.event and q.event.before and not q.event.after: return 101.0
    if q.event and q.event.after and not q.event.before: return 202.0
    t = q.scope.time
    if t.type == "current": return 999.0
    if t.type == "all": return 500.0
    if t.type == "relative": return {"minute": 10, "hour": 20, "day": 30, "week": 40, "month": 50}[t.unit] + (t.value or 0)
    if t.type == "calendar": return 600.0 + list(CALENDAR).index(t.name)
    return -1.0


def verify_pairs(pairs):
    failures = []
    for a, b, prop in pairs:
        changed = [k for k in LABELS if a["labels"][k] != b["labels"][k]]
        if not changed or deterministic_denotation(a["ir"]) == deterministic_denotation(b["ir"]):
            failures.append({"property": prop, "a": a, "b": b, "changed": changed})
    return {"pairs": len(pairs), "verified": len(pairs) - len(failures), "failures": failures}


class TimeParser(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        dim = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({k: nn.Linear(dim, len(v)) for k, v in LABELS.items()})

    def tokens(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=160, return_tensors="pt")

    def forward(self, batch):
        h = self.encoder(**batch).last_hidden_state
        m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return {k: head(p) for k, head in self.heads.items()}


def _natural(rows):
    return [{"text": r["text"], "labels": time_labels(Query.from_text(r["ir"])),
             "source": r["source"], "family": r["family"]} for r in rows]


def train(model, rows, device, seed, epochs=12):
    torch.manual_seed(seed); random.seed(seed)
    model.to(device); loader = DataLoader(rows, batch_size=24, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokens([x["text"] for x in items]).items()}
            gold = {k: torch.tensor([LABELS[k].index(x["labels"][k]) for x in items], device=device) for k in LABELS}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(batch)
                loss = sum(nn.functional.cross_entropy(logits[k], gold[k]) for k in LABELS) / len(LABELS)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return time.perf_counter() - started


def evaluate(model, rows, device):
    loader = DataLoader(rows, batch_size=64, collate_fn=lambda x: x)
    exact = comparable = 0; fields = {k: 0 for k in LABELS}; n = 0
    model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {k: v.to(device) for k, v in model.tokens([x["text"] for x in items]).items()}
            logits = model(batch)
            for i, row in enumerate(items):
                matches = {}
                for k in LABELS:
                    pred = LABELS[k][int(logits[k][i].argmax())]
                    matches[k] = pred == row["labels"][k]; fields[k] += matches[k]
                exact += all(matches.values())
                comparable += all(matches[k] for k in ("time_type", "time_unit", "time2_type", "event_window"))
                n += 1
    return {"n": n, "strict_time_exact": exact / n, "comparable_time_exact": comparable / n,
            "field_accuracy": {k: v / n for k, v in fields.items()}}


def main():
    output = os.path.join(ROOT, "results", "time_curriculum")
    os.makedirs(output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    train_pairs, test_pairs = contrast_pairs("train"), contrast_pairs("heldout")
    verification = {"train": verify_pairs(train_pairs), "heldout": verify_pairs(test_pairs)}
    synthetic_train = [row for a, b, _ in train_pairs for row in (a, b)]
    synthetic_test = [row for a, b, _ in test_pairs for row in (a, b)]
    natural_train, natural_test = _natural(training_pool()), _natural(evaluation_rows())
    runs = []
    for seed in (611, 612, 613):
        torch.manual_seed(seed)
        model = TimeParser(model_path)
        seconds = train(model, natural_train + synthetic_train, device, seed)
        runs.append({"seed": seed, "seconds": seconds,
                     "natural": evaluate(model, natural_test, device),
                     "synthetic": evaluate(model, synthetic_test, device)})
        del model
        if device == "cuda": torch.cuda.empty_cache()
    summary = {split: {metric: {"mean": statistics.mean(r[split][metric] for r in runs),
                                "min": min(r[split][metric] for r in runs),
                                "max": max(r[split][metric] for r in runs)}
                       for metric in ("strict_time_exact", "comparable_time_exact")}
               for split in ("natural", "synthetic")}
    result = {"device": device, "natural_training": len(natural_train),
              "synthetic_training": len(synthetic_train), "natural_evaluation": len(natural_test),
              "synthetic_evaluation": len(synthetic_test), "verification": verification,
              "runs": runs, "summary": summary, "opened_final_benchmark": False}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(output, "contrasts.jsonl"), "w", encoding="utf-8") as f:
        for a, b, prop in train_pairs + test_pairs:
            f.write(json.dumps({"property": prop, "a": a, "b": b,
                                "denotation_a": deterministic_denotation(a["ir"]),
                                "denotation_b": deterministic_denotation(b["ir"])}) + "\n")
    print(json.dumps({"verification": verification, "summary": summary,
                      "metrics": os.path.join(output, "metrics.json")}, indent=2))


if __name__ == "__main__":
    main()

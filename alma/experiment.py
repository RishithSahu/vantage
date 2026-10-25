"""Run the first falsifiable VOLT-Q/ALMA feasibility experiment.

Training data: algebra-generated declarative fragments only.
Evaluation: the existing independent development questions and a small natural
scope challenge.  indep_final.tsv is never opened by this module.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from alma.cards import (
    all_cards, generate_micro_examples, gold_task_key, held_micro_examples,
    scope_cards, signal_cards, task_cards,
)
from alma.model import (
    AlmaEncoder, AlmaPairMatcher, rank_cards, rank_pairs,
    train_matcher, train_pair_matcher,
)
from bench.indep_io import load
from vgpt.query_ir import Query


def _accuracy(model, rows, cards, device):
    if not rows:
        return {"n": 0, "top1": None, "top3": None, "errors": []}
    ranking, _ = rank_cards(model, [x["text"] for x in rows], cards, device=device)
    top1 = sum(r[0] == x["label"] for r, x in zip(ranking, rows))
    top3 = sum(x["label"] in r[:3] for r, x in zip(ranking, rows))
    errors = [
        {"text": x["text"], "gold": x["label"], "pred": r[0], "top3": r[:3]}
        for r, x in zip(ranking, rows) if r[0] != x["label"]
    ]
    return {"n": len(rows), "top1": top1 / len(rows), "top3": top3 / len(rows), "errors": errors[:20]}


def _pair_accuracy(model, rows, cards, device):
    if not rows:
        return {"n": 0, "top1": None, "top3": None, "errors": []}
    ranking, _ = rank_pairs(model, [x["text"] for x in rows], cards, device=device)
    top1 = sum(r[0] == x["label"] for r, x in zip(ranking, rows))
    top3 = sum(x["label"] in r[:3] for r, x in zip(ranking, rows))
    errors = [
        {"text": x["text"], "gold": x["label"], "pred": r[0], "top3": r[:3]}
        for r, x in zip(ranking, rows) if r[0] != x["label"]
    ]
    return {"n": len(rows), "top1": top1 / len(rows), "top3": top3 / len(rows), "errors": errors[:20]}


def _dev_rows():
    path = os.path.join(ROOT, "bench", "indep_dev.tsv")
    rows = []
    signal_rows = []
    for r in load(path):
        if r["prev_ir"]:
            continue
        try:
            q = Query.from_text(r["ir"])
        except Exception:
            continue
        key = gold_task_key(q)
        if key:
            rows.append({"id": r["id"], "text": r["text"], "label": key, "family": r["family"]})
        if q.target:
            signal_rows.append({"id": r["id"], "text": r["text"], "label": "signal:" + q.target, "family": r["family"]})
    return rows, signal_rows


def _scope_rows():
    out = []
    with open(os.path.join(HERE, "scope_dev.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _latency(model, cards, device, repeats=30):
    texts = ["largest battery temperature change before the latest warning while driving uphill"] * repeats
    started = time.perf_counter()
    rank_cards(model, texts, cards, device=device)
    elapsed = time.perf_counter() - started
    return 1000.0 * elapsed / repeats


def evaluate(model, device):
    dev, sig = _dev_rows()
    held = held_micro_examples()
    held_task = [x for x in held if x["kind"] == "task"]
    held_scope = [x for x in held if x["kind"] == "scope"]
    return {
        "held_micro_task": _accuracy(model, held_task, task_cards(), device),
        "held_micro_scope": _accuracy(model, held_scope, scope_cards(), device),
        "natural_dev_task": _accuracy(model, dev, task_cards(), device),
        "natural_scope_challenge": _accuracy(model, _scope_rows(), scope_cards(), device),
        "natural_dev_signal": _accuracy(model, sig, signal_cards(), device),
        "mean_batch_latency_ms_per_text": _latency(model, all_cards(), device),
    }


def evaluate_pair(model, device):
    dev, _ = _dev_rows()
    held = held_micro_examples()
    return {
        "held_micro_task": _pair_accuracy(model, [x for x in held if x["kind"] == "task"], task_cards(), device),
        "held_micro_scope": _pair_accuracy(model, [x for x in held if x["kind"] == "scope"], scope_cards(), device),
        "natural_dev_task": _pair_accuracy(model, dev, task_cards(), device),
        "natural_scope_challenge": _pair_accuracy(model, _scope_rows(), scope_cards(), device),
    }


def _fmt(v):
    return "n/a" if v is None else f"{100*v:.1f}%"


def write_report(path, result):
    fz, ft = result["frozen"], result["trained"]
    px = result["pair_trained"]
    lines = [
        "# VOLT-Q/ALMA First Feasibility Run", "",
        "This is a development experiment, not product or final-test evidence.",
        "The model saw algebra-generated declarative fragments and scope contrasts only.",
        "The existing independent final set was not opened.", "",
        "| Probe | Frozen bi-encoder | Trained bi-encoder | Pair matcher | Pair top-3 | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("held_micro_task", "Held microsemantic task phrases"),
        ("held_micro_scope", "Held microsemantic scope phrases"),
        ("natural_dev_task", "Natural development questions: task"),
        ("natural_scope_challenge", "Natural scope/attachment challenge"),
        ("natural_dev_signal", "Natural development questions: dynamic signal"),
    ):
        pair = px.get(key)
        lines.append(f"| {label} | {_fmt(fz[key]['top1'])} | {_fmt(ft[key]['top1'])} | "
                     f"{_fmt(pair['top1']) if pair else 'n/a'} | {_fmt(pair['top3']) if pair else 'n/a'} | {ft[key]['n']} |")
    tr = result["training"]
    lines += [
        "", "## Training",
        f"- Examples: {tr['examples']}",
        f"- Epochs: {tr['epochs']}",
        f"- Time: {tr['seconds']:.1f} seconds",
        f"- Peak allocated CUDA memory: {tr['peak_vram_mb']:.1f} MB",
        f"- Losses: {', '.join(f'{x:.4f}' for x in tr['losses'])}",
        f"- Pair-matcher examples: {result['pair_training']['examples']}",
        f"- Pair-matcher time: {result['pair_training']['seconds']:.1f} seconds",
        f"- Pair-matcher peak CUDA memory: {result['pair_training']['peak_vram_mb']:.1f} MB",
        "", "## Interpretation gate",
        "The method has initial potential only if natural-question task accuracy improves materially over the frozen model without destroying dynamic signal transfer, and if scope accuracy is substantially above chance.",
        "No result in this report supports a 95% product claim.",
        "", "## Sample trained-model errors",
    ]
    for e in ft["natural_dev_task"]["errors"][:12]:
        lines.append(f"- {e['text']} — gold {e['gold']}, predicted {e['pred']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "alma-minilm-l6-v2"))
    ap.add_argument("--output", default=os.path.join(ROOT, "results", "alma_probe"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frozen_model = AlmaEncoder(a.model).to(device)
    frozen = evaluate(frozen_model, device)
    del frozen_model
    if device == "cuda":
        torch.cuda.empty_cache()

    model = AlmaEncoder(a.model)
    train_rows = generate_micro_examples()
    trained_dir = os.path.join(a.output, "checkpoint")
    stats = train_matcher(
        model, train_rows, all_cards(), trained_dir,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=device,
    )
    trained = evaluate(model, device)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    pair_model = AlmaPairMatcher(a.model)
    pair_dir = os.path.join(a.output, "pair_checkpoint")
    pair_stats = train_pair_matcher(
        pair_model, train_rows, all_cards(), pair_dir,
        epochs=max(3, a.epochs - 1), batch_size=a.batch_size, lr=a.lr, device=device,
    )
    pair_trained = evaluate_pair(pair_model, device)
    result = {
        "model": os.path.abspath(a.model),
        "device": device,
        "training_sources": sorted(set(x["source"] for x in train_rows)),
        "opened_final_benchmark": False,
        "training": stats.__dict__,
        "pair_training": pair_stats.__dict__,
        "frozen": frozen,
        "trained": trained,
        "pair_trained": pair_trained,
    }
    with open(os.path.join(a.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_report(os.path.join(a.output, "REPORT.md"), result)
    print(json.dumps({
        "device": device,
        "examples": stats.examples,
        "seconds": round(stats.seconds, 1),
        "peak_vram_mb": round(stats.peak_vram_mb, 1),
        "frozen_dev_task": frozen["natural_dev_task"]["top1"],
        "trained_dev_task": trained["natural_dev_task"]["top1"],
        "frozen_scope": frozen["natural_scope_challenge"]["top1"],
        "trained_scope": trained["natural_scope_challenge"]["top1"],
        "pair_dev_task": pair_trained["natural_dev_task"]["top1"],
        "pair_scope": pair_trained["natural_scope_challenge"]["top1"],
        "trained_signal_top1": trained["natural_dev_signal"]["top1"],
        "report": os.path.join(a.output, "REPORT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()

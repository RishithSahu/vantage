"""Ablate scope learning: frozen NLI versus declarative round trips."""
from __future__ import annotations

import json
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .cards import generate_micro_examples, generate_roundtrip_scope_examples, scope_cards
from .experiment import _pair_accuracy, _scope_rows
from .model import AlmaClassifier, AlmaPairMatcher, train_classifier, train_pair_matcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nli_scope_accuracy(model_path: str, rows: list[dict], device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
    cards = scope_cards()
    rankings = []
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            batch = tokenizer(
                [row["text"]] * len(cards), [c.description for c in cards],
                padding=True, truncation=True, max_length=160, return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            scores = model(**batch).logits[:, 0]
            rankings.append([cards[i].key for i in torch.argsort(scores, descending=True).tolist()])
    correct = sum(rank[0] == row["label"] for rank, row in zip(rankings, rows))
    return {
        "n": len(rows), "top1": correct / len(rows), "seconds": time.perf_counter() - started,
        "predictions": [
            {"text": row["text"], "gold": row["label"], "pred": rank[0], "ranking": rank}
            for row, rank in zip(rows, rankings)
        ],
    }


def main():
    output = os.path.join(ROOT, "results", "alma_scope_probe")
    os.makedirs(output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = os.path.join(ROOT, "models", "alma-minilm-l6-v2")
    nli_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    eval_rows = _scope_rows()
    micro = [r for r in generate_micro_examples() if r["kind"] == "scope"]
    roundtrip = generate_roundtrip_scope_examples()

    frozen_nli = nli_scope_accuracy(nli_path, eval_rows, device)
    model = AlmaPairMatcher(base)
    stats = train_pair_matcher(
        model, micro + roundtrip, scope_cards(), os.path.join(output, "checkpoint"),
        epochs=6, batch_size=48, lr=2e-5, device=device,
    )
    learned = _pair_accuracy(model, eval_rows, scope_cards(), device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    labels = [c.key for c in scope_cards()]
    classifier = AlmaClassifier(base, labels)
    classifier_stats = train_classifier(
        classifier, micro + roundtrip, os.path.join(output, "classifier_checkpoint"),
        epochs=6, batch_size=48, lr=2e-5, device=device,
    )
    classifier_rankings, _ = classifier.predict([r["text"] for r in eval_rows], device=device)
    classifier_correct = sum(rank[0] == row["label"] for rank, row in zip(classifier_rankings, eval_rows))
    classifier_result = {
        "n": len(eval_rows), "top1": classifier_correct / len(eval_rows),
        "top3": 1.0,
        "errors": [
            {"text": row["text"], "gold": row["label"], "pred": rank[0], "ranking": rank}
            for row, rank in zip(eval_rows, classifier_rankings) if rank[0] != row["label"]
        ],
    }
    result = {
        "device": device, "opened_final_benchmark": False, "training_question_count": 0,
        "micro_examples": len(micro), "roundtrip_declarative_examples": len(roundtrip),
        "training": stats.__dict__, "frozen_nli": frozen_nli,
        "declarative_roundtrip_matcher": learned,
        "classifier_training": classifier_stats.__dict__,
        "declarative_roundtrip_classifier": classifier_result,
    }
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    report = [
        "# ALMA Scope-Supervision Ablation", "",
        "No complete or fragmentary questions were used for training. The sealed final benchmark was not opened.", "",
        f"- Frozen compact NLI: {100*frozen_nli['top1']:.1f}% ({frozen_nli['n']} examples)",
        f"- Declarative round-trip matcher: {100*learned['top1']:.1f}% top-1, {100*learned['top3']:.1f}% top-3 ({learned['n']} examples)",
        f"- Three-way scope classifier: {100*classifier_result['top1']:.1f}% top-1 ({classifier_result['n']} examples)",
        f"- Training: {stats.examples} text/card pairs in {stats.seconds:.1f}s; peak CUDA allocation {stats.peak_vram_mb:.1f} MB", "",
        "The round-trip condition is not strict microsemantic supervision: it includes complete declarative program descriptions. It is reported separately so it cannot be misrepresented as microsemantic transfer.", "",
        "## Errors", "",
    ]
    report.extend(f"- {e['text']} — gold {e['gold']}, predicted {e['pred']}" for e in classifier_result["errors"])
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(json.dumps({
        "frozen_nli": frozen_nli["top1"], "roundtrip": learned["top1"],
        "classifier": classifier_result["top1"],
        "roundtrip_top3": learned["top3"], "pairs": stats.examples,
        "seconds": round(stats.seconds, 1), "peak_vram_mb": round(stats.peak_vram_mb, 1),
        "report": os.path.join(output, "REPORT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()

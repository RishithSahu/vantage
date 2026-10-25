"""Selector bake-off over frozen full-IR pools produced by final_campaign."""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import statistics
import time
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch import nn
from transformers import AutoModel, AutoTokenizer

from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt.ir_slm import grounding_issues
from vgpt.qc import QueryCompiler
from vgpt.query_ir import IRError, Query
from .final_campaign import (DEV_PATH, ROOT, WORLD_NAMES, Pool, _mutations, full_key,
                             multiworld, percentile, sha256_file, strict_skeleton,
                             valid_program)
from .supervised_parser import diverse_prefix, training_pool


SEEDS = (701, 702, 703)


def program_gloss(text):
    if text == "op=none": return "refuse no vehicle query"
    try: q = Query.from_text(text)
    except IRError: return text.replace("_", " ").replace("=", " ").replace(":", " ")
    chunks = ["operation", q.op.replace("_", " ")]
    if q.target: chunks += ["target", q.target.replace("_", " ")]
    if q.target2: chunks += ["second target", q.target2.replace("_", " ")]
    if q.agg: chunks += ["aggregation", q.agg]
    chunks += ["time", q.scope.time.to_text().replace(":", " ").replace("_", " ")]
    if q.scope_b: chunks += ["second time", q.scope_b.time.to_text().replace(":", " ")]
    for condition in q.scope.conditions:
        chunks += ["condition", condition.to_text().replace(":", " ").replace("_", " ")]
    if q.scope.select:
        chunks += ["selection", q.scope.select.to_text().replace(":", " ").replace("_", " ")]
    if q.groups: chunks += ["groups"] + [item.replace("_", " ") for item in q.groups]
    if q.event: chunks += ["event", q.event.type.replace("_", " "), "before", str(q.event.before), "after", str(q.event.after)]
    if q.refs: chunks += ["inherited"] + q.refs
    return " ".join(chunks)


def training_candidate_sets(profile, maximum=24):
    rows = diverse_prefix(training_pool())[:100]
    questions = [row["text"] for row in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(questions)
    similarities = matrix @ matrix.T
    candidate_sets = []
    for index, row in enumerate(rows):
        pool = Pool(profile); pool.add(row["ir"], "gold")
        for candidate in _mutations(row["ir"]): pool.add(candidate, "hard_mutation")
        order = np.asarray(similarities[index].todense()).ravel().argsort()[::-1]
        for rank, other in enumerate(order, 1):
            if int(other) == index: continue
            pool.add(rows[int(other)]["ir"], "retrieval_negative", rank)
            if len(pool.ordered()) >= maximum * 2: break
        gold = full_key(row["ir"])
        items = pool.ordered()
        positives = [item for item in items if full_key(item.ir) == gold]
        negatives = [item for item in items if full_key(item.ir) != gold]
        # Formal mutations differ in executable IR; no execution-equivalent duplicates are retained.
        candidate_sets.append([positives[0]] + negatives[:maximum - 1])
    return rows, candidate_sets


class DualListwise(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        dim = self.encoder.config.hidden_size
        self.question_projection = nn.Linear(dim, 128)
        self.program_projection = nn.Linear(dim, 128)
        self.log_temperature = nn.Parameter(torch.tensor(2.3))

    def encode(self, texts, kind, device):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=160,
                               return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        projection = self.question_projection if kind == "question" else self.program_projection
        return nn.functional.normalize(projection(pooled), dim=-1)


class CrossListwise(nn.Module):
    """Question and complete program attend jointly; trained with a groupwise loss."""
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def score(self, questions, programs, device):
        text = [f"Question: {question}\nExecutable program: {program}"
                for question, program in zip(questions, programs)]
        batch = self.tokenizer(text, padding=True, truncation=True, max_length=192,
                               return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train_dual(model, rows, candidate_sets, device, seed, epochs=10):
    random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    order = list(range(len(rows))); losses = []; started = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(order); model.train(); total = 0.
        for offset in range(0, len(order), 4):
            indexes = order[offset:offset + 4]
            questions = [rows[index]["text"] for index in indexes]
            programs = [[program_gloss(item.ir) for item in candidate_sets[index]] for index in indexes]
            width = min(len(items) for items in programs)
            programs = [items[:width] for items in programs]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                q = model.encode(questions, "question", device)
                p = model.encode([item for group in programs for item in group], "program", device)
                p = p.view(len(indexes), width, -1)
                logits = torch.einsum("bd,bkd->bk", q, p) * model.log_temperature.exp().clamp(max=100)
                loss = nn.functional.cross_entropy(logits, torch.zeros(len(indexes), dtype=torch.long, device=device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += float(loss.detach()) * len(indexes)
        losses.append(total / len(rows))
    return {"seconds": time.perf_counter() - started, "losses": losses,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0}


def train_cross(model, rows, candidate_sets, device, seed, epochs=8, width=16):
    random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    order = list(range(len(rows))); losses = []; started = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(order); model.train(); total = 0.
        for index in order:
            candidates = candidate_sets[index][:width]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                scores = model.score([rows[index]["text"]] * len(candidates),
                                     [program_gloss(item.ir) for item in candidates], device)
                loss = nn.functional.cross_entropy(scores.unsqueeze(0),
                                                   torch.zeros(1, dtype=torch.long, device=device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(loss.detach())
        losses.append(total / len(rows))
    return {"seconds": time.perf_counter() - started, "losses": losses,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0}


def dual_select(model, rows, pools, device, limit=100):
    model.eval(); predictions = []; latencies = []
    with torch.inference_mode():
        for row, pool in zip(rows, pools):
            candidates = pool["candidates"][:limit]
            started = time.perf_counter()
            q = model.encode([row["text"]], "question", device)
            values = []
            for offset in range(0, len(candidates), 64):
                glosses = [program_gloss(item["ir"]) for item in candidates[offset:offset + 64]]
                p = model.encode(glosses, "program", device)
                values.extend((q @ p.T).squeeze(0).float().cpu().tolist())
            best = max(range(len(values)), key=values.__getitem__)
            predictions.append(candidates[best]["ir"])
            latencies.append((time.perf_counter() - started) * 1000)
    return predictions, latencies


def cross_select(model, rows, pools, device, limit=10):
    model.eval(); predictions = []; latencies = []
    with torch.inference_mode():
        for row, pool in zip(rows, pools):
            candidates = pool["candidates"][:limit]; values = []; started = time.perf_counter()
            for offset in range(0, len(candidates), 32):
                batch = candidates[offset:offset + 32]
                values.extend(model.score([row["text"]] * len(batch),
                                           [program_gloss(item["ir"]) for item in batch],
                                           device).float().cpu().tolist())
            best = max(range(len(values)), key=values.__getitem__)
            predictions.append(candidates[best]["ir"])
            latencies.append((time.perf_counter() - started) * 1000)
    return predictions, latencies


def text_similarity_select(rows, pools, limit=100):
    corpus = [row["text"] for row in rows]
    gloss_groups = [[program_gloss(item["ir"]) for item in pool["candidates"][:limit]] for pool in pools]
    all_glosses = [value for group in gloss_groups for value in group]
    word = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", sublinear_tf=True).fit(corpus + all_glosses)
    char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True).fit(corpus + all_glosses)
    predictions = []; latencies = []
    for row, pool, glosses in zip(rows, pools, gloss_groups):
        started = time.perf_counter()
        qword = word.transform([row["text"]]); pword = word.transform(glosses)
        qchar = char.transform([row["text"]]); pchar = char.transform(glosses)
        values = .6 * (pword @ qword.T).toarray().ravel() + .4 * (pchar @ qchar.T).toarray().ravel()
        # A small source prior is not language-specific and preserves the strong generator baseline.
        for index, candidate in enumerate(pool["candidates"][:limit]):
            if "direct_greedy" in candidate["sources"]: values[index] += .08
            values[index] += .01 * min(candidate.get("frequency", 1), 5)
        best = int(values.argmax()); predictions.append(pool["candidates"][best]["ir"])
        latencies.append((time.perf_counter() - started) * 1000)
    return predictions, latencies


def source_baselines(pools):
    greedy = []; consensus = []
    for pool in pools:
        candidates = pool["candidates"]
        greedy.append(next((item["ir"] for item in candidates if "direct_greedy" in item["sources"]), candidates[0]["ir"]))
        direct = [item for item in candidates if any(source.startswith("direct_") for source in item["sources"])]
        consensus.append(max(direct or candidates, key=lambda item: (item.get("frequency", 1),
                                                                      "direct_greedy" in item["sources"]))["ir"])
    return greedy, consensus


def metrics(rows, predictions, latencies=None):
    exact = sum(full_key(row["ir"]) == full_key(prediction) for row, prediction in zip(rows, predictions)) / len(rows)
    skeleton = sum(strict_skeleton(row["ir"]) == strict_skeleton(prediction)
                   for row, prediction in zip(rows, predictions)) / len(rows)
    result = {"full_ir_exact": exact, "strict_skeleton_exact": skeleton,
              "multiworld_execution": multiworld([(row["ir"], prediction)
                                                     for row, prediction in zip(rows, predictions)])}
    result["multiworld_min_execution"] = min(result["multiworld_execution"].values())
    if latencies:
        result["selector_latency_ms"] = {"mean": statistics.mean(latencies),
                                         "p50": statistics.median(latencies),
                                         "p95": percentile(latencies, .95)}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", required=True)
    parser.add_argument("--limit", type=int, default=100); parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cross-only", action="store_true")
    args = parser.parse_args(); campaign = os.path.abspath(args.campaign)
    pool_path = os.path.join(campaign, "candidate_pools.jsonl")
    if not os.path.exists(pool_path): raise RuntimeError("candidate pools are not frozen")
    rows = load(DEV_PATH); pools = [json.loads(line) for line in open(pool_path, encoding="utf-8")]
    conn = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', WORLD_NAMES[0])}?mode=ro", uri=True)
    profile = QueryCompiler(conn, now=DEFAULT_NOW).profile; conn.close()
    train_rows, train_sets = training_candidate_sets(profile)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = os.path.join(campaign, "selectors"); os.makedirs(output, exist_ok=True)
    prior_path = os.path.join(output, "metrics.json")
    if args.cross_only and os.path.exists(prior_path):
        result = json.load(open(prior_path, encoding="utf-8"))
    else:
        result = {"candidate_pool_sha256": sha256_file(pool_path), "candidate_limit": args.limit,
                  "device": device, "seeds": list(SEEDS), "selectors": {},
                  "acceptance_or_final_opened": False}
    greedy, consensus = source_baselines(pools)
    prediction_rows = ([json.loads(line) for line in open(os.path.join(output, "predictions.jsonl"), encoding="utf-8")]
                       if args.cross_only and os.path.exists(os.path.join(output, "predictions.jsonl")) else [])
    lexical, lexical_ms = text_similarity_select(rows, pools, args.limit)
    if not args.cross_only:
        result["selectors"]["generator_greedy"] = metrics(rows, greedy)
        result["selectors"]["direct_self_consistency"] = metrics(rows, consensus)
        result["selectors"]["global_text_compatibility"] = metrics(rows, lexical, lexical_ms)
        for name, values in (("generator_greedy", greedy), ("direct_self_consistency", consensus),
                             ("global_text_compatibility", lexical)):
            for row, prediction in zip(rows, values):
                prediction_rows.append({"selector": name, "id": row["id"], "gold": row["ir"], "pred": prediction})
    if not args.cross_only:
        model_path = os.path.join(ROOT, "models", "alma-minilm-l6-v2")
        dual_runs = []
        for seed in SEEDS:
            model = DualListwise(model_path)
            training = train_dual(model, train_rows, train_sets, device, seed, args.epochs)
            predictions, latencies = dual_select(model, rows, pools, device, args.limit)
            run = {"seed": seed, "training": training, "metrics": metrics(rows, predictions, latencies)}
            dual_runs.append(run)
            torch.save({"seed": seed, "model_state_dict": {key: value.detach().cpu()
                        for key, value in model.state_dict().items()}},
                       os.path.join(output, f"dual_listwise_seed{seed}.pt"))
            for row, prediction in zip(rows, predictions):
                prediction_rows.append({"selector": "dual_listwise", "seed": seed,
                                        "id": row["id"], "gold": row["ir"], "pred": prediction})
            del model
            if device == "cuda": torch.cuda.empty_cache()
        result["selectors"]["dual_listwise"] = {"runs": dual_runs,
            "full_ir_exact_mean": statistics.mean(run["metrics"]["full_ir_exact"] for run in dual_runs),
            "full_ir_exact_min": min(run["metrics"]["full_ir_exact"] for run in dual_runs),
            "execution_min_mean": statistics.mean(run["metrics"]["multiworld_min_execution"] for run in dual_runs)}
    cross_runs = []
    cross_model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    for seed in SEEDS:
        model = CrossListwise(cross_model_path)
        training = train_cross(model, train_rows, train_sets, device, seed, min(args.epochs, 8))
        predictions, latencies = cross_select(model, rows, pools, device, min(args.limit, 10))
        run = {"seed": seed, "training": training, "metrics": metrics(rows, predictions, latencies)}
        cross_runs.append(run)
        torch.save({"seed": seed, "model_state_dict": {key: value.detach().cpu()
                    for key, value in model.state_dict().items()}},
                   os.path.join(output, f"cross_listwise_seed{seed}.pt"))
        for row, prediction in zip(rows, predictions):
            prediction_rows.append({"selector": "cross_listwise", "seed": seed,
                                    "id": row["id"], "gold": row["ir"], "pred": prediction})
        del model
        if device == "cuda": torch.cuda.empty_cache()
    result["selectors"]["cross_listwise"] = {"candidate_limit": min(args.limit, 10), "runs": cross_runs,
        "full_ir_exact_mean": statistics.mean(run["metrics"]["full_ir_exact"] for run in cross_runs),
        "full_ir_exact_min": min(run["metrics"]["full_ir_exact"] for run in cross_runs),
        "execution_min_mean": statistics.mean(run["metrics"]["multiworld_min_execution"] for run in cross_runs)}
    metrics_path = os.path.join(output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle: json.dump(result, handle, indent=2)
    predictions_path = os.path.join(output, "predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for item in prediction_rows: handle.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps({name: value if name != "dual_listwise" else {
        "full_ir_exact_mean": value["full_ir_exact_mean"], "full_ir_exact_min": value["full_ir_exact_min"],
        "execution_min_mean": value["execution_min_mean"]}
        for name, value in result["selectors"].items()}, indent=2))


if __name__ == "__main__": main()

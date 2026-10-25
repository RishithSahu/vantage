"""Fixed six-family VehicleGPT structural-method bake-off on iterative development data."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from vgpt.query_ir import IRError, Query
from .condition_shape_updated_oracle import load_frozen_predictions
from .operator_aggregation_oracle import score
from .selection_variant_experiment import SEEDS, VARIANTS, VariantPredictor
from .supervised_parser import (LABELS, ROOT, StructuralParser, diverse_prefix,
                                evaluation_rows, skeleton, train as train_structure,
                                training_pool)
from .time_curriculum import (LABELS as TIME_LABELS, TimeParser, _natural,
                              contrast_pairs as time_contrast_pairs, train as train_time)
from .time_integration_ablation import invalid


VARIANT_TO_ENTITY = {"Absent": "none", "TripSelection": "trip",
                     "ChargeSessionSelection": "charge_session"}
ENTITY_TO_VARIANT = {value: key for key, value in VARIANT_TO_ENTITY.items()}
BEAM_SIZE = 500
RERANK_K = 50
INTERACTION_WEIGHT = .35
OPTIONAL_FIELDS = tuple(field for field in LABELS if field not in {"op", "time_type"})


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_probs(tensor):
    return torch.log_softmax(tensor.detach().float(), -1).cpu().tolist()


def structural_logits(model, rows, device):
    loader = DataLoader(rows, batch_size=64, collate_fn=lambda items: items)
    output = []; model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {key: value.to(device) for key, value in
                     model.tokenize([item["text"] for item in items]).items()}
            logits = model(batch)
            for index in range(len(items)):
                output.append({field: log_probs(logits[field][index]) for field in LABELS})
    return output


def time_logits(model, rows, device):
    loader = DataLoader(rows, batch_size=64, collate_fn=lambda items: items)
    output = []; model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {key: value.to(device) for key, value in
                     model.tokens([item["text"] for item in items]).items()}
            logits = model(batch)
            for index in range(len(items)):
                output.append({field: log_probs(logits[field][index]) for field in TIME_LABELS})
    return output


def variant_logits(model, rows, device):
    loader = DataLoader(rows, batch_size=64, collate_fn=lambda items: items)
    output = []; model.eval()
    with torch.inference_mode():
        for items in loader:
            batch = {key: value.to(device) for key, value in
                     model.tokenize([item["text"] for item in items]).items()}
            values = model(batch)
            output.extend(log_probs(values[index]) for index in range(len(items)))
    return output


def sorted_values(scores, labels):
    return sorted(((labels[index], score) for index, score in enumerate(scores)),
                  key=lambda item: item[1], reverse=True)


def merge_beam(beam, group, limit=BEAM_SIZE):
    merged = []
    for existing, existing_score in beam:
        for addition, addition_score in group:
            if set(existing).intersection(addition):
                continue
            merged.append(({**existing, **addition}, existing_score + addition_score))
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged[:limit]


def signature_prior(rows):
    counts = Counter((row["labels"]["op"], row["labels"]["agg"],
                      row["labels"]["target_count"]) for row in rows)
    vocabulary = len(LABELS["op"]) * len(LABELS["agg"]) * len(LABELS["target_count"])
    return {signature: math.log((count + 1) / (len(rows) + vocabulary))
            for signature, count in counts.items()}, math.log(1 / (len(rows) + vocabulary))


def candidate_programs(structure, temporal, variant, priors, unseen_prior,
                       interaction=True, limit=BEAM_SIZE):
    signature_group = []
    for op, op_score in sorted_values(structure["op"], LABELS["op"]):
        for agg, agg_score in sorted_values(structure["agg"], LABELS["agg"]):
            requires_agg = op in ("aggregate", "extreme", "compare")
            if (agg != "none") != requires_agg:
                continue
            for arity, arity_score in sorted_values(structure["target_count"], LABELS["target_count"]):
                signature = (op, agg, arity)
                interaction_score = INTERACTION_WEIGHT * priors.get(signature, unseen_prior) if interaction else 0.
                signature_group.append(({"op": op, "agg": agg, "target_count": arity},
                                        op_score + agg_score + arity_score + interaction_score))
    signature_group.sort(key=lambda item: item[1], reverse=True)

    time_event_group = []
    for time_type, a in sorted_values(temporal["time_type"], TIME_LABELS["time_type"]):
        for unit, b in sorted_values(temporal["time_unit"], TIME_LABELS["time_unit"]):
            if (time_type == "relative") != (unit != "none"):
                continue
            for time2, c in sorted_values(temporal["time2_type"], TIME_LABELS["time2_type"]):
                for window, d in sorted_values(temporal["event_window"], TIME_LABELS["event_window"]):
                    before = "present" if window in ("before", "around") else "none"
                    after = "present" if window in ("after", "around") else "none"
                    for event, e in sorted_values(structure["event"], LABELS["event"]):
                        if (before != "none" or after != "none") and event == "none":
                            continue
                        time_event_group.append(({
                            "time_type": time_type, "time_unit": unit, "time2_type": time2,
                            "before": before, "after": after, "event": event}, a + b + c + d + e))
    time_event_group.sort(key=lambda item: item[1], reverse=True)

    selection_group = []
    for variant_index, variant_name in enumerate(VARIANTS):
        entity = VARIANT_TO_ENTITY[variant_name]
        if entity == "none":
            selection_group.append(({"select_entity": "none", "select_order": "none",
                                     "select_k": "none"}, variant[variant_index]))
            continue
        for order, order_score in sorted_values(structure["select_order"], LABELS["select_order"]):
            if order == "none":
                continue
            for k_value, k_score in sorted_values(structure["select_k"], LABELS["select_k"]):
                selection_group.append(({"select_entity": entity, "select_order": order,
                                         "select_k": k_value},
                                        variant[variant_index] + order_score + k_score))
    selection_group.sort(key=lambda item: item[1], reverse=True)

    beam = [({}, 0.)]
    for group in (signature_group[:80], time_event_group[:100], selection_group[:20]):
        beam = merge_beam(beam, group, limit)
    for field in ("condition_shape", "groups_count", "refs"):
        group = [({field: value}, value_score)
                 for value, value_score in sorted_values(structure[field], LABELS[field])]
        beam = merge_beam(beam, group, limit)
    valid = [(prediction, value) for prediction, value in beam if not invalid(prediction)]
    valid.sort(key=lambda item: item[1], reverse=True)
    return valid[:limit]


def candidate_recall(rows, candidate_sets, cutoffs=(1, 10, 50, 100, 500)):
    return {str(cutoff): sum(any(candidate == row["labels"] for candidate, _ in candidates[:cutoff])
                             for row, candidates in zip(rows, candidate_sets)) / len(rows)
            for cutoff in cutoffs}


def score_top(rows, candidate_sets):
    predictions = [candidates[0][0] for candidates in candidate_sets]
    result = score(rows, predictions); result.pop("details")
    return result, predictions


def lexical_retrieval(train_rows, eval_rows):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform([row["text"] for row in train_rows] +
                                      [row["text"] for row in eval_rows])
    similarities = cosine_similarity(matrix[len(train_rows):], matrix[:len(train_rows)])
    predictions = [train_rows[int(row.argmax())]["labels"] for row in similarities]
    candidates = [[train_rows[int(index)]["labels"] for index in scores.argsort()[::-1][:10]]
                  for scores in similarities]
    metrics = score(eval_rows, predictions); metrics.pop("details")
    recall = {str(k): sum(any(candidate == row["labels"] for candidate in choices[:k])
                          for row, choices in zip(eval_rows, candidates)) / len(eval_rows)
              for k in (1, 3, 5, 10)}
    return metrics, recall, similarities


def compact_embeddings(rows, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            items = rows[start:start + 64]
            batch = tokenizer([row["text"] for row in items], padding=True, truncation=True,
                              max_length=160, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = nn.functional.normalize(pooled, dim=-1)
            output.append(pooled.cpu())
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return torch.cat(output)


def embedding_retrieval(train_rows, eval_rows, device):
    model_path = os.path.join(ROOT, "models", "alma-minilm-l6-v2")
    embedded = compact_embeddings(train_rows + eval_rows, model_path, device)
    similarities = embedded[len(train_rows):] @ embedded[:len(train_rows)].T
    predictions = [train_rows[int(row.argmax())]["labels"] for row in similarities]
    metrics = score(eval_rows, predictions); metrics.pop("details")
    recall = {}
    for k in (1, 3, 5, 10):
        recall[str(k)] = sum(any(train_rows[int(index)]["labels"] == row["labels"]
                                   for index in scores.topk(k).indices)
                               for row, scores in zip(eval_rows, similarities)) / len(eval_rows)
    return metrics, recall


def controller(train_rows, eval_rows, frozen_by_seed, similarities):
    results = []
    nearest = similarities.argsort(axis=1)[:, ::-1][:, :5]
    for seed in SEEDS:
        baseline_predictions = [item["accepted_prediction"] for item in frozen_by_seed[seed]]
        predictions = []; proposed = applied = correct = invalid_veto = 0
        for row_index, base in enumerate(baseline_predictions):
            gold_row = eval_rows[row_index]
            prediction = copy.deepcopy(base)
            for field in OPTIONAL_FIELDS:
                if prediction[field] == "none":
                    continue
                absent_vote = sum(train_rows[int(index)]["labels"][field] == "none"
                                  for index in nearest[row_index]) / len(nearest[row_index])
                if absent_vote < .8:
                    continue
                proposed += 1
                candidate = copy.deepcopy(prediction); candidate[field] = "none"
                if invalid(candidate):
                    invalid_veto += 1; continue
                prediction = candidate; applied += 1
                correct += gold_row["labels"][field] == "none"
            predictions.append(prediction)
        metrics = score(eval_rows, predictions); metrics.pop("details")
        results.append({"seed": seed, "metrics": metrics, "proposed": proposed,
                        "applied": applied, "invalid_veto": invalid_veto,
                        "suppression_precision": correct / applied if applied else None,
                        "suppression_coverage": applied / max(1, sum(
                            row["labels"][field] == "none" and base[field] != "none"
                            for row, base in zip(eval_rows, baseline_predictions)
                            for field in OPTIONAL_FIELDS))})
    return results


def safe_skeleton(text):
    try:
        return skeleton(Query.from_text(text))
    except (IRError, ValueError):
        return {field: "none" for field in LABELS}


def direct_generation_result(eval_rows):
    path = os.path.join(ROOT, "results", "indep_dev_qwen_lora.json")
    with open(path, encoding="utf-8") as f: result = json.load(f)
    if len(result["rows"]) != len(eval_rows):
        raise RuntimeError("Frozen direct-generation result is not aligned to development data")
    predictions = [safe_skeleton(item["pred"]) for item in result["rows"]]
    metrics = score(eval_rows, predictions); metrics.pop("details")
    summary = result["summary"]
    return {"source": os.path.relpath(path, ROOT).replace("\\", "/"),
            "source_sha256": sha256_file(path), "structural": metrics,
            "full_ir_exact": summary["ir_exact"]["rate"],
            "execution_outcome_accuracy": summary["outcome_ok"]["rate"],
            "latency_ms": summary["latency_ms"], "peak_rss_mb": summary.get("peak_rss_mb"),
            "seeds": 1, "existing_frozen_evaluation": True}


class Reranker(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def tokens(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=220,
                              return_tensors="pt")

    def forward(self, batch):
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def serialize_program(prediction):
    return " ".join(f"{field}={prediction[field]}" for field in LABELS)


def rerank_text(question, prediction):
    return f"Question: {question}\nCandidate typed structure: {serialize_program(prediction)}"


def train_reranker(model, train_rows, candidate_sets, device, seed):
    examples = []
    for row, candidates in zip(train_rows, candidate_sets):
        examples.append({"text": rerank_text(row["text"], row["labels"]), "label": 1.})
        negatives = []
        for prediction, _ in candidates:
            if prediction != row["labels"] and prediction not in negatives:
                negatives.append(prediction)
            if len(negatives) == 5: break
        if not negatives:
            corrupted = copy.deepcopy(row["labels"])
            corrupted["condition_shape"] = "named" if corrupted["condition_shape"] == "none" else "none"
            negatives = [corrupted]
        examples.extend({"text": rerank_text(row["text"], prediction), "label": 0.}
                        for prediction in negatives)
    random.Random(seed).shuffle(examples); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model.to(device); loader = DataLoader(examples, batch_size=24, shuffle=True, collate_fn=lambda x: x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    started = time.perf_counter()
    for _ in range(6):
        model.train()
        for items in loader:
            batch = {key: value.to(device) for key, value in
                     model.tokens([item["text"] for item in items]).items()}
            gold = torch.tensor([item["label"] for item in items], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(batch)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, gold)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    return {"seconds": time.perf_counter() - started, "examples": len(examples),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0}


def rerank(model, rows, candidate_sets, device):
    flat = []; boundaries = []
    for row, candidates in zip(rows, candidate_sets):
        start = len(flat)
        flat.extend(rerank_text(row["text"], prediction) for prediction, _ in candidates[:RERANK_K])
        boundaries.append((start, len(flat)))
    values = []; model.eval(); started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(flat), 64):
            batch = {key: value.to(device) for key, value in
                     model.tokens(flat[offset:offset + 64]).items()}
            values.extend(model(batch).detach().float().cpu().tolist())
    predictions = []
    for candidates, (start, end) in zip(candidate_sets, boundaries):
        best = max(range(start, end), key=lambda index: values[index]) - start
        predictions.append(candidates[best][0])
    return predictions, (time.perf_counter() - started) * 1000 / len(rows)


def mean_metrics(runs, key):
    return {metric: {"mean": statistics.mean(run[key][metric] for run in runs),
                     "min": min(run[key][metric] for run in runs),
                     "max": max(run[key][metric] for run in runs)}
            for metric in ("skeleton_exact", "field_omission_rate",
                           "unsupported_addition_rate", "invalid_composition_rate")}


def main():
    output = os.path.join(ROOT, "results", "architecture_bakeoff")
    os.makedirs(output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(ROOT, "models", "deberta_xsmall_nli")
    train_rows = diverse_prefix(training_pool())[:100]
    eval_rows = evaluation_rows()
    selection_dir = os.path.join(ROOT, "results", "selection_variant")
    selection_manifest, frozen_by_seed, _ = load_frozen_predictions(selection_dir, eval_rows)

    lexical_metrics, lexical_recall, lexical_similarities = lexical_retrieval(train_rows, eval_rows)
    embedding_metrics, embedding_recall = embedding_retrieval(train_rows, eval_rows, device)
    controller_runs = controller(train_rows, eval_rows, frozen_by_seed, lexical_similarities)
    direct_generation = direct_generation_result(eval_rows)
    priors, unseen_prior = signature_prior(train_rows)
    natural_time_train = _natural(training_pool())
    natural_time_eval = _natural(eval_rows)
    synthetic_time = [row for a, b, _ in time_contrast_pairs("train") for row in (a, b)]
    runs = []; all_predictions = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        if device == "cuda": torch.cuda.manual_seed_all(seed)
        structural = StructuralParser(model_path)
        struct_training = train_structure(structural, train_rows, device, seed=seed)
        struct_train_logits = structural_logits(structural, train_rows, device)
        struct_eval_logits = structural_logits(structural, eval_rows, device)
        struct_checkpoint = os.path.join(output, f"factorized_structural_seed{seed}.pt")
        torch.save({"seed": seed, "model_state_dict": {key: value.detach().cpu()
                    for key, value in structural.state_dict().items()}}, struct_checkpoint)
        del structural
        if device == "cuda": torch.cuda.empty_cache()

        torch.manual_seed(seed + 1000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 1000)
        temporal = TimeParser(model_path)
        time_seconds = train_time(temporal, natural_time_train + synthetic_time, device, seed + 1000)
        time_train_logits = time_logits(temporal, _natural(train_rows), device)
        time_eval_logits = time_logits(temporal, natural_time_eval, device)
        time_checkpoint = os.path.join(output, f"factorized_time_seed{seed}.pt")
        torch.save({"seed": seed, "model_state_dict": {key: value.detach().cpu()
                    for key, value in temporal.state_dict().items()}}, time_checkpoint)
        del temporal
        if device == "cuda": torch.cuda.empty_cache()

        variant = VariantPredictor(model_path)
        checkpoint = torch.load(os.path.join(selection_dir, f"selection_variant_seed{seed}.pt"),
                                map_location="cpu", weights_only=False)
        variant.load_state_dict(checkpoint["model_state_dict"]); variant.to(device)
        variant_train_logits = variant_logits(variant, train_rows, device)
        variant_eval_logits = variant_logits(variant, eval_rows, device)
        del variant
        if device == "cuda": torch.cuda.empty_cache()

        started = time.perf_counter()
        train_bottom = [candidate_programs(s, t, v, priors, unseen_prior, interaction=False)
                        for s, t, v in zip(struct_train_logits, time_train_logits, variant_train_logits)]
        eval_bottom = [candidate_programs(s, t, v, priors, unseen_prior, interaction=False)
                       for s, t, v in zip(struct_eval_logits, time_eval_logits, variant_eval_logits)]
        eval_global = [candidate_programs(s, t, v, priors, unseen_prior, interaction=True)
                       for s, t, v in zip(struct_eval_logits, time_eval_logits, variant_eval_logits)]
        train_global = [candidate_programs(s, t, v, priors, unseen_prior, interaction=True)
                        for s, t, v in zip(struct_train_logits, time_train_logits, variant_train_logits)]
        decode_ms = (time.perf_counter() - started) * 1000 / (len(train_rows) + 2 * len(eval_rows))
        bottom_metrics, bottom_predictions = score_top(eval_rows, eval_bottom)
        global_metrics, global_predictions = score_top(eval_rows, eval_global)
        bottom_recall = candidate_recall(eval_rows, eval_bottom)
        global_recall = candidate_recall(eval_rows, eval_global)

        torch.manual_seed(seed + 5000)
        if device == "cuda": torch.cuda.manual_seed_all(seed + 5000)
        reranker = Reranker(model_path)
        reranker_training = train_reranker(reranker, train_rows, train_global, device, seed + 5000)
        reranked_predictions, rerank_latency = rerank(reranker, eval_rows, eval_global, device)
        reranked_metrics = score(eval_rows, reranked_predictions); reranked_metrics.pop("details")
        reranker_checkpoint = os.path.join(output, f"reranker_seed{seed}.pt")
        torch.save({"seed": seed, "model_state_dict": {key: value.detach().cpu()
                    for key, value in reranker.state_dict().items()}, "candidate_k": RERANK_K},
                   reranker_checkpoint)
        del reranker
        if device == "cuda": torch.cuda.empty_cache()

        runs.append({"seed": seed, "factorized_global": global_metrics,
                     "bottom_up": bottom_metrics, "reranked": reranked_metrics,
                     "bottom_up_candidate_recall": bottom_recall,
                     "global_candidate_recall": global_recall,
                     "decoder_ms_per_question": decode_ms,
                     "reranker_ms_per_question": rerank_latency,
                     "training": {"structural": struct_training, "time_seconds": time_seconds,
                                  "reranker": reranker_training}})
        for index, row in enumerate(eval_rows):
            all_predictions.append({"seed": seed, "index": index, "text": row["text"],
                                    "gold": row["labels"],
                                    "factorized_global": global_predictions[index],
                                    "bottom_up": bottom_predictions[index],
                                    "reranked": reranked_predictions[index],
                                    "global_gold_rank": next((rank + 1 for rank, (candidate, _) in enumerate(eval_global[index])
                                                              if candidate == row["labels"]), None),
                                    "bottom_gold_rank": next((rank + 1 for rank, (candidate, _) in enumerate(eval_bottom[index])
                                                              if candidate == row["labels"]), None)})

    summary = {"factorized_global": mean_metrics(runs, "factorized_global"),
               "bottom_up": mean_metrics(runs, "bottom_up"),
               "reranked": mean_metrics(runs, "reranked")}
    controller_summary = {metric: {"mean": statistics.mean(run["metrics"][metric] for run in controller_runs),
                                   "min": min(run["metrics"][metric] for run in controller_runs),
                                   "max": max(run["metrics"][metric] for run in controller_runs)}
                          for metric in ("skeleton_exact", "field_omission_rate",
                                         "unsupported_addition_rate", "invalid_composition_rate")}
    recall_summary = {method: {cutoff: {"mean": statistics.mean(run[f"{method}_candidate_recall"][cutoff]
                                                                    for run in runs),
                                               "min": min(run[f"{method}_candidate_recall"][cutoff]
                                                          for run in runs),
                                               "max": max(run[f"{method}_candidate_recall"][cutoff]
                                                          for run in runs)}
                                        for cutoff in ("1", "10", "50", "100", "500")}
                      for method in ("bottom_up", "global")}
    methods = {
        "A_compatibility_controller": {"summary": controller_summary, "runs": controller_runs},
        "B_factorized_global": {"summary": summary["factorized_global"],
                                "candidate_recall": recall_summary["global"]},
        "C_bottom_up": {"summary": summary["bottom_up"],
                        "candidate_recall": recall_summary["bottom_up"]},
        "D_lexical_retrieval": {"metrics": lexical_metrics, "candidate_recall": lexical_recall},
        "D_embedding_retrieval": {"metrics": embedding_metrics, "candidate_recall": embedding_recall},
        "E_candidate_reranking": {"summary": summary["reranked"],
                                  "input_candidate_recall": recall_summary["global"]},
        "F_direct_grammar_generation": direct_generation,
    }
    observed = {
        "A_compatibility_controller": controller_summary["skeleton_exact"]["mean"],
        "B_factorized_global": summary["factorized_global"]["skeleton_exact"]["mean"],
        "C_bottom_up": summary["bottom_up"]["skeleton_exact"]["mean"],
        "D_lexical_retrieval": lexical_metrics["skeleton_exact"],
        "D_embedding_retrieval": embedding_metrics["skeleton_exact"],
        "E_candidate_reranking": summary["reranked"]["skeleton_exact"]["mean"],
        "F_direct_grammar_generation": direct_generation["structural"]["skeleton_exact"],
    }
    winner = max(observed, key=observed.get)
    max_candidate_recall = max(recall_summary["global"]["500"]["mean"],
                               recall_summary["bottom_up"]["500"]["mean"],
                               lexical_recall["10"], embedding_recall["10"])
    result = {"experiment": "six-family architecture bake-off", "device": device,
              "questions": len(eval_rows), "natural_training_examples": len(train_rows),
              "seeds": list(SEEDS), "methods": methods, "neural_runs": runs,
              "observed_structural_accuracy": observed, "observed_winner": winner,
              "best_observed_structural_accuracy": observed[winner],
              "best_tested_candidate_recall": max_candidate_recall,
              "ninety_percent_demonstrated": any(value >= .90 for value in observed.values()),
              "ninety_percent_candidate_ceiling_demonstrated": max_candidate_recall >= .90,
              "guarantee_possible": False,
              "guarantee_reason": "development performance and candidate recall cannot guarantee unseen natural-language accuracy",
              "execution_comparison_boundary": "only the frozen direct full-IR generator has execution metrics; skeleton candidates do not contain signal/value payloads",
              "selection_source_manifest_sha256": sha256_file(os.path.join(selection_dir, "MANIFEST.json")),
              "acceptance_or_final_opened": False,
              "development_set_status": "iteratively used; not untouched generalization evidence"}
    metrics_path = os.path.join(output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    predictions_path = os.path.join(output, "predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in all_predictions: f.write(json.dumps(item, sort_keys=True) + "\n")
    checkpoint_paths = [os.path.join(output, name) for name in os.listdir(output) if name.endswith(".pt")]
    manifest = {"format_version": 1, "experiment": "architecture_bakeoff",
                "dataset_sha256": {name: sha256_file(os.path.join(ROOT, "bench", name))
                                   for name in ("ir_gold.tsv", "diag_v1.tsv", "indep_dev.tsv")},
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(predictions_path),
                "checkpoints_sha256": {os.path.basename(path): sha256_file(path)
                                       for path in sorted(checkpoint_paths)},
                "direct_generation_source_sha256": direct_generation["source_sha256"],
                "acceptance_or_final_opened": False}
    with open(os.path.join(output, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    lines = ["# Six-Family VehicleGPT Architecture Bake-Off", "",
             "All methods use the same 100 natural training questions and 120-question iterative development set. Acceptance and final benchmarks remained sealed. Neural methods B/C/E use seeds 701–703. Method F is a pre-existing frozen single-checkpoint full-IR evaluation on the same questions.", "",
             "| Method | Strict skeleton exact | Invalid | Candidate recall |",
             "|---|---:|---:|---:|",
             f"| A Compatibility controller | {100*controller_summary['skeleton_exact']['mean']:.1f}% | {100*controller_summary['invalid_composition_rate']['mean']:.1f}% | n/a |",
             f"| B Factorized global decoder | {100*summary['factorized_global']['skeleton_exact']['mean']:.1f}% | {100*summary['factorized_global']['invalid_composition_rate']['mean']:.1f}% | {100*recall_summary['global']['500']['mean']:.1f}% @500 |",
             f"| C Bottom-up typed construction | {100*summary['bottom_up']['skeleton_exact']['mean']:.1f}% | {100*summary['bottom_up']['invalid_composition_rate']['mean']:.1f}% | {100*recall_summary['bottom_up']['500']['mean']:.1f}% @500 |",
             f"| D Lexical retrieval | {100*lexical_metrics['skeleton_exact']:.1f}% | {100*lexical_metrics['invalid_composition_rate']:.1f}% | {100*lexical_recall['10']:.1f}% @10 |",
             f"| D Compact embedding retrieval | {100*embedding_metrics['skeleton_exact']:.1f}% | {100*embedding_metrics['invalid_composition_rate']:.1f}% | {100*embedding_recall['10']:.1f}% @10 |",
             f"| E Candidate + learned reranker | {100*summary['reranked']['skeleton_exact']['mean']:.1f}% | {100*summary['reranked']['invalid_composition_rate']['mean']:.1f}% | input {100*recall_summary['global']['50']['mean']:.1f}% @50 |",
             f"| F Direct grammar-constrained generation | {100*direct_generation['structural']['skeleton_exact']:.1f}% | {100*direct_generation['structural']['invalid_composition_rate']:.1f}% | n/a |", "",
             f"Observed winner: **{winner}** at {100*observed[winner]:.1f}% structural exactness.",
             f"Direct generation full-IR exactness/execution outcome accuracy: {100*direct_generation['full_ir_exact']:.1f}% / {100*direct_generation['execution_outcome_accuracy']:.1f}%; p95 latency {direct_generation['latency_ms']['p95']:.1f} ms.", "",
             "## 90% conclusion", "",
             f"No tested method demonstrated 90% structural accuracy. Best tested candidate recall was {100*max_candidate_recall:.1f}%; this is an upper bound for a reranker restricted to that candidate set, not achieved accuracy.",
             "No empirical development experiment can guarantee 90% on unseen natural language. A credible 90% route requires both candidate recall above 90% on a sealed distribution and a selector whose lower confidence bound clears the target; neither condition is established here.", "",
             "Skeleton-only methods cannot be credited with execution accuracy because their representation omits signal identity, numeric values and other payloads. Method F is the only tested full-program system in this table."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"observed": observed, "winner": winner,
                      "best_candidate_recall": max_candidate_recall,
                      "ninety_demonstrated": result["ninety_percent_demonstrated"],
                      "report": os.path.join(output, "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()

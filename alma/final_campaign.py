"""Reproducible full-IR candidate campaign for the VehicleGPT iterative dev set.

The sealed acceptance/final sets are deliberately not addressable from this module.
Every candidate is generated without development gold and is validated before it can
enter a selector pool.  Gold is read only by the evaluator after pools are frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bench.eval_indep import outcome, run as evaluate_rows, summarize
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from training.evaluate import parts, same_gold
from vgpt import nl2ir, query_grammar as G
from vgpt.qc import QueryCompiler
from vgpt.query_ir import AGGS, OPS, IRError, Query
from vgpt.query_validator import validate
from .architecture_bakeoff import safe_skeleton
from .supervised_parser import diverse_prefix, evaluation_rows, training_pool


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_PATH = os.path.join(ROOT, "bench", "indep_dev.tsv")
BASELINE_PATH = os.path.join(ROOT, "results", "indep_dev_qwen_lora.json")
MODEL_PATH = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
LORA_PATH = os.path.join(ROOT, "models", "ir-lora-qwen25-05b.gguf")
BAKEOFF_DIR = os.path.join(ROOT, "results", "architecture_bakeoff")
WORLD_NAMES = ("gen_courier_ext.db", "gen_cold.db", "gen_roadwarrior.db",
               "gen_demo.db", "gen_courier.db")
ALLOWED_INPUT_NAMES = {"indep_dev.tsv", "ir_gold.tsv", "diag_v1.tsv"}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(text: str) -> str | None:
    if text == "op=none":
        return text
    try:
        return Query.from_text(text).to_text()
    except (IRError, ValueError):
        return None


def full_key(text: str) -> str | None:
    if text == "op=none":
        return text
    try:
        return parts(Query.from_text(text))["full"]
    except (IRError, ValueError):
        return None


def strict_skeleton(text: str):
    return safe_skeleton(text)


def valid_program(text: str, profile) -> bool:
    if text == "op=none":
        return True
    try:
        return not validate(Query.from_text(text), profile).codes()
    except (IRError, ValueError):
        return False


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(p * (len(ordered) - 1))]


@dataclass
class Candidate:
    ir: str
    sources: set[str] = field(default_factory=set)
    ranks: dict[str, int] = field(default_factory=dict)
    generation_ms: float = 0.0
    frequency: int = 1

    def as_dict(self):
        return {"ir": self.ir, "sources": sorted(self.sources), "ranks": self.ranks,
                "generation_ms": self.generation_ms, "frequency": self.frequency}


class Pool:
    def __init__(self, profile):
        self.profile = profile
        self._items: dict[str, Candidate] = {}

    def add(self, text, source, rank=0, ms=0.0):
        normalized = canonical(text)
        if normalized is None or not valid_program(normalized, self.profile):
            return False
        key = full_key(normalized)
        item = self._items.get(key)
        if item is None:
            item = self._items[key] = Candidate(normalized, {source}, {}, ms, 1)
        else:
            item.sources.add(source); item.frequency += 1; item.generation_ms += ms
        if rank:
            item.ranks[source] = min(rank, item.ranks.get(source, rank))
        return True

    def ordered(self):
        priority = {"direct_greedy": 0, "direct_context": 1, "direct_low": 2,
                    "direct_diverse": 3, "rules": 4, "lexical": 5,
                    "embedding": 6, "typed_mutation": 7, "structural_projection": 8,
                    "semantic_repair_1": 9, "semantic_repair_2": 10}
        def key(item):
            best_source = min((priority.get(source, 99) for source in item.sources), default=99)
            best_rank = min(item.ranks.values(), default=9999)
            return (best_source, best_rank, -item.frequency, item.ir)
        return sorted(self._items.values(), key=key)


def reproduce_baseline(output_dir):
    """Re-score frozen rows and verify their immutable published summary."""
    with open(BASELINE_PATH, encoding="utf-8") as handle:
        frozen = json.load(handle)
    rows = load(DEV_PATH)
    if len(rows) != len(frozen["rows"]):
        raise RuntimeError("baseline row count changed")
    conn = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', WORLD_NAMES[0])}?mode=ro", uri=True)
    compiler = QueryCompiler(conn, now=DEFAULT_NOW)

    class Frozen:
        name = "frozen-qwen0.5b+lora"
        def __init__(self, predictions): self.predictions = iter(predictions)
        def plan(self, row):
            prediction = next(self.predictions)
            if prediction["id"] != row["id"]: raise RuntimeError("baseline row order changed")
            return prediction["pred"], {"ms": prediction["ms"], "out_tokens": prediction.get("tok", 0)}

    rescored_rows = evaluate_rows(Frozen(frozen["rows"]), rows, compiler)
    rescored = summarize(rescored_rows)
    conn.close()
    expected = frozen["summary"]
    checks = {
        "n": rescored["n"] == expected["n"],
        "ir_exact": rescored["ir_exact"]["k"] == expected["ir_exact"]["k"],
        "outcome_ok": rescored["outcome_ok"]["k"] == expected["outcome_ok"]["k"],
        "p95_latency": abs(rescored["latency_ms"]["p95"] - expected["latency_ms"]["p95"]) < .1,
    }
    structural = sum(strict_skeleton(row["pred"]) == strict_skeleton(row["gold"])
                     for row in frozen["rows"]) / len(rows)
    result = {"checks": checks, "passed": all(checks.values()), "rescored": rescored,
              "strict_skeleton_exact": structural,
              "source_sha256": sha256_file(BASELINE_PATH)}
    with open(os.path.join(output_dir, "phase0_reproduction.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if not result["passed"]:
        raise RuntimeError(f"baseline reproduction failed: {checks}")
    return result


def _mutations(text):
    """Generic typed transformations. They never consult question text or gold."""
    if text == "op=none":
        return []
    try:
        original = Query.from_text(text)
    except IRError:
        return []
    out = []
    signatures = {
        "current_value": (None,), "aggregate": ("mean", "median", "std", "sum", "p95"),
        "extreme": ("min", "max"), "trend": (None,), "compare": ("mean", "min", "max"),
        "count": (None,), "list_events": (None,), "event_context": (None,),
        "rank": (None,), "correlation": (None,), "percent_time": (None,),
        "explain_event": (None,), "capability": (None,), "explain_change": (None,),
        "detect_change": (None,),
    }
    for op, aggregations in signatures.items():
        for aggregation in aggregations:
            q = original.clone(); q.op = op; q.agg = aggregation
            out.append(q.to_text())
    for edit in ("drop_conditions", "drop_selection", "drop_event", "drop_scope_b", "drop_target2"):
        q = original.clone()
        if edit == "drop_conditions": q.scope.conditions = []
        elif edit == "drop_selection": q.scope.select = None
        elif edit == "drop_event": q.event = None
        elif edit == "drop_scope_b": q.scope_b = None
        elif edit == "drop_target2": q.target2 = None
        out.append(q.to_text())
    return out


def _project_structure(base_text, structure):
    """Project only payload-preserving structural decisions onto an executable IR."""
    if base_text == "op=none":
        return []
    try:
        original = Query.from_text(base_text)
    except IRError:
        return []
    candidates = []
    q = original.clone()
    q.op = structure["op"]
    q.agg = None if structure["agg"] == "none" else structure["agg"]
    if structure["condition_shape"] == "none": q.scope.conditions = []
    if structure["select_entity"] == "none": q.scope.select = None
    elif not q.scope.select or q.scope.select.entity != structure["select_entity"]: return []
    if structure["event"] == "none": q.event = None
    elif q.event is None: return []
    if structure["target_count"] == "none": q.target = None; q.target2 = None
    elif structure["target_count"] == "one": q.target2 = None
    elif structure["target_count"] == "two" and not q.target2: return []
    if structure["time_type"] != q.scope.time.type: return []
    candidates.append(q.to_text())
    return candidates


def _frozen_structures():
    path = os.path.join(BAKEOFF_DIR, "predictions.jsonl")
    by_index = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for method in ("factorized_global", "bottom_up", "reranked"):
                value = row[method]
                if value not in by_index[row["index"]]: by_index[row["index"]].append(value)
    return by_index


def _retrieve(train, dev):
    train_texts = [row["text"] for row in train]
    dev_texts = [row["text"] for row in dev]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = word.fit_transform(train_texts + dev_texts)
    lexical = cosine_similarity(matrix[len(train):], matrix[:len(train)])
    # Character retrieval is deliberately a separate, generic representation.
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    cmatrix = char.fit_transform(train_texts + dev_texts)
    character = cosine_similarity(cmatrix[len(train):], cmatrix[:len(train)])
    return lexical, character


class DirectSampler:
    def __init__(self, threads=2):
        from vgpt.ir_slm import IrSlm
        self.backend = IrSlm(MODEL_PATH, lora_path=LORA_PATH, threads=threads, mode="tuned")
        self.backend.warmup()

    def generate(self, text, temperature, seed, top_p=.95, top_k=40):
        started = time.perf_counter()
        response = self.backend.llm.create_chat_completion(
            messages=G.messages(text, "tuned"), grammar=self.backend.grammar,
            max_tokens=96, temperature=temperature, top_p=top_p, top_k=top_k, seed=seed)
        value = response["choices"][0]["message"]["content"].strip()
        return value, (time.perf_counter() - started) * 1000

    def close(self): self.backend.close()


def _dialogue_text(row):
    if not row.get("prev_text"):
        return None
    return (f"Previous request: {row['prev_text']}\nPrevious typed query: {row['prev_ir']}\n"
            f"Current follow-up: {row['text']}")


def generate_direct_candidates(output_dir, threads=2, samples=4):
    """Generate resumable direct candidates; gold IR is never loaded in this phase."""
    rows = load(DEV_PATH)
    path = os.path.join(output_dir, "direct_candidates.jsonl")
    complete = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line); complete[item["id"]] = item
    sampler = DirectSampler(threads)
    try:
        with open(path, "a", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                if row["id"] in complete: continue
                candidates = []
                configs = [("greedy", 0.0, 10000 + index)]
                configs += [("low", .25, 20000 + index * 31 + i) for i in range(samples)]
                configs += [("diverse", .70, 30000 + index * 31 + i) for i in range(samples)]
                for kind, temperature, seed in configs:
                    value, ms = sampler.generate(row["text"], temperature, seed)
                    candidates.append({"kind": kind, "ir": value, "ms": ms, "seed": seed,
                                       "temperature": temperature})
                contextual = _dialogue_text(row)
                if contextual:
                    value, ms = sampler.generate(contextual, .15, 40000 + index)
                    candidates.append({"kind": "context", "ir": value, "ms": ms,
                                       "seed": 40000 + index, "temperature": .15})
                item = {"id": row["id"], "index": index, "text": row["text"],
                        "candidates": candidates}
                handle.write(json.dumps(item, sort_keys=True) + "\n"); handle.flush()
                if (index + 1) % 10 == 0: print(f"generated {index + 1}/{len(rows)}", flush=True)
    finally:
        sampler.close()
    return path


def build_pools(output_dir, candidate_path):
    """Freeze candidate pools before looking at development gold for scoring."""
    dev_raw = load(DEV_PATH)
    train = diverse_prefix(training_pool())[:100]
    lexical, character = _retrieve(train, evaluation_rows())
    structures = _frozen_structures()
    direct = {item["id"]: item for item in
              (json.loads(line) for line in open(candidate_path, encoding="utf-8"))}
    conn = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', WORLD_NAMES[0])}?mode=ro", uri=True)
    profile = QueryCompiler(conn, now=DEFAULT_NOW).profile
    frozen_baseline = {row["id"]: row for row in json.load(open(BASELINE_PATH, encoding="utf-8"))["rows"]}
    out_path = os.path.join(output_dir, "candidate_pools.jsonl")
    summaries = []
    with open(out_path, "w", encoding="utf-8") as handle:
        for index, row in enumerate(dev_raw):
            pool = Pool(profile)
            frozen = frozen_baseline[row["id"]]
            pool.add(frozen["pred"], "direct_greedy", 1, frozen["ms"])
            for rank, generated in enumerate(direct[row["id"]]["candidates"], 1):
                source = {"greedy": "direct_greedy", "low": "direct_low",
                          "diverse": "direct_diverse", "context": "direct_context"}[generated["kind"]]
                pool.add(generated["ir"], source, rank, generated["ms"])
            state = nl2ir.ConvState()
            if row.get("prev_ir"):
                try: state.last = Query.from_text(row["prev_ir"])
                except IRError: pass
            parsed = nl2ir.parse(row["text"], now=DEFAULT_NOW, tz=3600, state=state)
            if parsed.ok: pool.add(parsed.query.to_text(), "rules", 1, getattr(parsed, "ms", 0.0))
            for source, scores in (("lexical", lexical[index]), ("embedding", character[index])):
                for rank, train_index in enumerate(np.argsort(scores)[::-1][:30], 1):
                    pool.add(train[int(train_index)]["ir"], source, rank)
            bases = [item.ir for item in pool.ordered()[:15]]
            for base in bases:
                for candidate in _mutations(base): pool.add(candidate, "typed_mutation")
                for structure in structures[index]:
                    for candidate in _project_structure(base, structure):
                        pool.add(candidate, "structural_projection")
            items = pool.ordered()[:500]
            record = {"id": row["id"], "index": index, "text": row["text"],
                      "candidates": [item.as_dict() for item in items]}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            summaries.append(len(items))
    conn.close()
    return out_path, {"mean": statistics.mean(summaries), "min": min(summaries),
                      "max": max(summaries)}


def evaluate_candidate_pools(output_dir, pool_path):
    """Gold becomes visible only here, after the candidate file is closed and hashed."""
    rows = load(DEV_PATH)
    pools = [json.loads(line) for line in open(pool_path, encoding="utf-8")]
    cutoffs = (1, 5, 10, 50, 100, 500)
    exact_hits = Counter(); skeleton_hits = Counter(); source_hits = Counter()
    per_question = []
    for row, pool in zip(rows, pools):
        gold_key = full_key(row["ir"]); gold_skeleton = strict_skeleton(row["ir"])
        candidates = pool["candidates"]
        exact_rank = next((i + 1 for i, candidate in enumerate(candidates)
                           if full_key(candidate["ir"]) == gold_key), None)
        skeleton_rank = next((i + 1 for i, candidate in enumerate(candidates)
                              if strict_skeleton(candidate["ir"]) == gold_skeleton), None)
        for cutoff in cutoffs:
            exact_hits[cutoff] += exact_rank is not None and exact_rank <= cutoff
            skeleton_hits[cutoff] += skeleton_rank is not None and skeleton_rank <= cutoff
        if exact_rank is not None:
            for source in candidates[exact_rank - 1]["sources"]: source_hits[source] += 1
        per_question.append({"id": row["id"], "family": row["family"],
                             "exact_rank": exact_rank, "skeleton_rank": skeleton_rank,
                             "candidate_count": len(candidates)})
    n = len(rows)
    metrics = {
        "n": n,
        "pool_sha256_before_evaluation": sha256_file(pool_path),
        "exact_full_ir_candidate_recall": {str(k): exact_hits[k] / n for k in cutoffs},
        "skeleton_candidate_recall": {str(k): skeleton_hits[k] / n for k in cutoffs},
        "exact_hit_sources": dict(source_hits),
        "candidate_generation_failures_at_500": n - exact_hits[500],
        "per_question": per_question,
    }
    with open(os.path.join(output_dir, "candidate_recall.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def multiworld(predictions):
    results = {}
    for world in WORLD_NAMES:
        path = os.path.join(ROOT, "data", world)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        compiler = QueryCompiler(connection, now=DEFAULT_NOW); cache = {}
        correct = 0
        for gold, prediction in predictions:
            if gold == "op=none": correct += prediction == "op=none"; continue
            gs, gv = outcome(compiler, gold, cache); ps, pv = outcome(compiler, prediction, cache)
            correct += (full_key(gold) == full_key(prediction)) if gs in ("invalid", "unparsed") else (
                ps == gs and (gv is None or same_gold(pv, gv)))
        connection.close(); results[world] = correct / len(predictions)
    return results


def write_manifest(output_dir, args, files):
    protected = [DEV_PATH, os.path.join(ROOT, "bench", "ir_gold.tsv"),
                 os.path.join(ROOT, "bench", "diag_v1.tsv"), MODEL_PATH, LORA_PATH,
                 os.path.join(BAKEOFF_DIR, "MANIFEST.json")]
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "protected_inputs_sha256": {os.path.relpath(path, ROOT): sha256_file(path) for path in protected},
        "artifacts_sha256": {os.path.basename(path): sha256_file(path) for path in files if os.path.exists(path)},
        "acceptance_or_final_opened": False,
        "development_set_status": "iteratively used; not an untouched estimate",
    }
    with open(os.path.join(output_dir, "MANIFEST.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output)
    if os.path.basename(output_dir).lower() in {"acceptance", "final"}:
        raise RuntimeError("refusing an ambiguous sealed-set output directory")
    os.makedirs(output_dir, exist_ok=True)
    phase0 = reproduce_baseline(output_dir)
    candidate_path = os.path.join(output_dir, "direct_candidates.jsonl")
    if not args.skip_generation or not os.path.exists(candidate_path):
        candidate_path = generate_direct_candidates(output_dir, args.threads, args.samples)
    pool_path, pool_sizes = build_pools(output_dir, candidate_path)
    recall = evaluate_candidate_pools(output_dir, pool_path)
    summary = {"phase0_passed": phase0["passed"], "pool_sizes": pool_sizes,
               "candidate_recall": recall["exact_full_ir_candidate_recall"],
               "skeleton_recall": recall["skeleton_candidate_recall"],
               "next_gate": "train selectors only if exact full-IR recall materially exceeds the frozen 48.3% baseline",
               "acceptance_or_final_opened": False}
    summary_path = os.path.join(output_dir, "PHASES_0_2.json")
    with open(summary_path, "w", encoding="utf-8") as handle: json.dump(summary, handle, indent=2)
    write_manifest(output_dir, args, [candidate_path, pool_path,
                                      os.path.join(output_dir, "phase0_reproduction.json"),
                                      os.path.join(output_dir, "candidate_recall.json"), summary_path])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

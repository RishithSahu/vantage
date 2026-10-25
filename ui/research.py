"""Read-only research dashboard data.

Only the iterative development campaign is exposed.  Acceptance and final files
are deliberately not referenced here, so opening the dashboard cannot unseal an
evaluation set.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGN = os.path.join(ROOT, "results", "final_campaign_20261012_001")
RECALL_FILE = "candidate_pools_semantic_v6_recall.json"
POOL_FILE = "candidate_pools_semantic_v6.jsonl"


def _json(name, default):
    path = os.path.join(CAMPAIGN, name)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _dev_rows():
    rows = []
    path = os.path.join(ROOT, "bench", "indep_dev.tsv")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    rows.append({"id": parts[0], "family": parts[1],
                                 "question": parts[2], "gold": parts[3]})
    except OSError:
        pass
    return rows


def _parts(ir):
    return {piece.split("=", 1)[0]: piece.split("=", 1)[1]
            for piece in str(ir).split() if "=" in piece}


def _distance(left, right):
    a, b = _parts(left), _parts(right)
    keys = set(a) | set(b)
    return sum(a.get(key) != b.get(key) for key in keys)


def _changed(left, right):
    a, b = _parts(left), _parts(right)
    names = {"op": "operation", "tgt": "signal", "tgt2": "second signal",
             "t": "time", "tb": "comparison time", "agg": "aggregation",
             "cond": "condition", "sel": "selection", "ev": "event",
             "evs": "event choice", "unit": "unit", "grp": "grouping"}
    return [names.get(key, key) for key in sorted(set(a) | set(b))
            if a.get(key) != b.get(key)]


@lru_cache(maxsize=1)
def _missing_cases():
    recall = _json(RECALL_FILE, {})
    missing_ids = {item["id"] for item in recall.get("per_question", [])
                   if item.get("exact_rank") is None}
    rows = {row["id"]: row for row in _dev_rows() if row["id"] in missing_ids}
    pool_path = os.path.join(CAMPAIGN, POOL_FILE)
    try:
        with open(pool_path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                row = rows.get(record.get("id"))
                if not row:
                    continue
                choices = record.get("candidates", [])[:2000]
                nearest = min(choices, key=lambda item: _distance(row["gold"], item.get("ir", "")),
                              default={"ir": "No candidate"})
                row["nearest"] = nearest.get("ir", "No candidate")
                row["missing_parts"] = _changed(row["gold"], row["nearest"])
    except (OSError, ValueError):
        pass
    return list(rows.values())


@lru_cache(maxsize=1)
def snapshot():
    metrics = _json("FINAL_METRICS.json", {})
    direct = metrics.get("frozen_direct", {})
    selector = metrics.get("selectors", {}).get("unanimous_edit_gate", {})
    recall_report = _json(RECALL_FILE, {})
    recall = recall_report.get("exact_full_ir_candidate_recall",
                               metrics.get("candidate_recall", {}).get("semantic_first", {}))
    deployment = metrics.get("deployment", {})
    missing = _missing_cases()
    families = {}
    parts = {}
    for item in missing:
        families[item["family"]] = families.get(item["family"], 0) + 1
        for name in item.get("missing_parts", []):
            parts[name] = parts.get(name, 0) + 1
    return {
        "ok": True,
        "scope": "iterative development set",
        "sealed": {"acceptance": True, "final": True},
        "headline": {
            "achieved_full_ir": selector.get("full_ir_exact", 0),
            "achieved_skeleton": selector.get("strict_skeleton_exact", 0),
            "candidate_recall_k1000": recall.get("1000", 0),
            "worst_world_execution": selector.get("multiworld_min_execution", 0),
            "direct_p95_ms": direct.get("p95_ms", 0),
            "hybrid_p95_ms": deployment.get("sequential_full_hybrid_p95_sum_ms", 0),
        },
        "candidate_curve": [{"k": int(k), "value": value} for k, value in recall.items()],
        "methods": [
            {"name": "Direct 0.5B generator", "exact": direct.get("full_ir_exact", 0),
             "skeleton": direct.get("strict_skeleton_exact", 0), "status": "Deployment baseline"},
            {"name": "Cross selector (K=10 mean)",
             "exact": metrics.get("selectors", {}).get("cross_listwise_k10", {}).get("full_ir_exact_mean", 0),
             "skeleton": None, "status": "Research component"},
            {"name": "Unanimous edit gate", "exact": selector.get("full_ir_exact", 0),
             "skeleton": selector.get("strict_skeleton_exact", 0), "status": "Best measured"},
        ],
        "missing": missing,
        "missing_summary": {"count": len(missing), "families": families, "parts": parts},
        "workstreams": [
            {"id": "A", "title": "Candidate completeness", "state": "active",
             "detail": "Repair reusable semantic atoms and composition paths behind the missing programs.",
             "gate": "Passed: at least 97% exact candidate recall without reading sealed sets."},
            {"id": "B", "title": "Outcome verifier", "state": "queued",
             "detail": "Train on candidates produced by the real inference pipeline and semantic counterfactuals.",
             "gate": "Complete-program accuracy improves across all three seeds."},
            {"id": "C", "title": "Selective expansion", "state": "queued",
             "detail": "Run the expensive path only when the fast path is uncertain.",
             "gate": "Accuracy gain survives with p95 latency below two seconds."},
        ],
        "provenance": {"campaign": os.path.basename(CAMPAIGN),
                       "acceptance_or_final_opened": False},
    }

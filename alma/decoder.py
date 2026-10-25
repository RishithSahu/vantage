"""Small exact-decoding and counterexample probes for the ALMA hypothesis."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from vgpt.query_ir import Query, Scope, TimeSpec
from vgpt.query_validator import validate


TASK_FORM = {
    "task:current": ("current_value", None),
    "task:mean": ("aggregate", "mean"),
    "task:max": ("extreme", "max"),
    "task:min": ("extreme", "min"),
    "task:sum": ("aggregate", "sum"),
    "task:median": ("aggregate", "median"),
    "task:delta": ("aggregate", "delta"),
    "task:trend": ("trend", None),
    "task:detect_change": ("detect_change", None),
    "task:capability": ("capability", None),
}


@dataclass(frozen=True)
class Scored:
    value: object
    score: float


def exact_decode(tasks: Iterable[Scored], targets: Iterable[Scored], times: Iterable[Scored]) -> list[tuple[float, Query]]:
    """Exhaustively enumerate the bounded single-target grammar.

    No beam or top-k-per-cell pruning occurs here.  Returned candidates are all
    validator-accepted programs, sorted by the additive model score.
    """
    out = []
    for task, target, tm in product(tasks, targets, times):
        if task.value not in TASK_FORM:
            continue
        op, agg = TASK_FORM[task.value]
        q = Query(op=op, target=str(target.value), agg=agg, scope=Scope(time=tm.value), source="alma")
        if validate(q).valid:
            out.append((task.score + target.score + tm.score, q))
    return sorted(out, key=lambda x: (-x[0], x[1].to_text()))


@dataclass(frozen=True)
class ScopeProgram:
    common_exclude: bool = False
    left_exclude: bool = False
    right_exclude: bool = False


def execute_scope(program: ScopeProgram, rows: list[dict]) -> tuple[float | None, float | None]:
    def avg(group: str, exclude: bool):
        xs = [float(r["value"]) for r in rows if r["group"] == group and not (exclude and r["charging"])]
        return None if not xs else sum(xs) / len(xs)
    left_ex = program.common_exclude or program.left_exclude
    right_ex = program.common_exclude or program.right_exclude
    return avg("left", left_ex), avg("right", right_ex)


def minimal_scope_counterexample(a: ScopeProgram, b: ScopeProgram) -> list[dict] | None:
    """Find the first tiny typed trace whose denotations distinguish two scopes."""
    candidate_rows = (
        {"group": "left", "charging": False, "value": 10.0},
        {"group": "left", "charging": True, "value": 100.0},
        {"group": "right", "charging": False, "value": 20.0},
        {"group": "right", "charging": True, "value": 200.0},
    )
    for n in range(1, len(candidate_rows) + 1):
        from itertools import combinations
        for subset in combinations(candidate_rows, n):
            rows = list(subset)
            if execute_scope(a, rows) != execute_scope(b, rows):
                return rows
    return None

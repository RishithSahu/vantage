"""Dynamic semantic cards and question-free microsemantic data generation."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import random
from typing import Iterable

from vgpt.query_ir import Query
from vgpt import signal_registry as R


@dataclass(frozen=True)
class Card:
    key: str
    kind: str
    description: str
    train_fragments: tuple[str, ...]
    held_fragments: tuple[str, ...] = ()


_TASKS = (
    Card("task:current", "task", "the current or latest observed value of a signal",
         ("current value", "latest observed value", "present reading", "value at the present time"),
         ("most recent reading", "value right now")),
    Card("task:mean", "task", "the arithmetic mean or average of numeric observations",
         ("average value", "arithmetic mean", "mean over the interval", "typical value represented by the mean"),
         ("average across the period", "mean reading")),
    Card("task:max", "task", "the greatest, highest, hottest, fastest, or peak observed value",
         ("maximum value", "highest observation", "greatest recorded value", "peak measurement"),
         ("upper extreme", "largest reading")),
    Card("task:min", "task", "the smallest, lowest, coldest, or minimum observed value",
         ("minimum value", "lowest observation", "smallest recorded value", "lower extreme"),
         ("least recorded value", "bottom reading")),
    Card("task:sum", "task", "the total accumulated amount over the selected observations",
         ("total amount", "sum over the interval", "accumulated quantity", "combined amount"),
         ("period total", "overall accumulated value")),
    Card("task:median", "task", "the median or middle numeric observation",
         ("median value", "middle observation after sorting", "fiftieth percentile", "central ranked value"),
         ("middle reading", "central observation")),
    Card("task:delta", "task", "the difference between the ending and starting value",
         ("end to start difference", "net change over the interval", "final value minus initial value", "period delta"),
         ("overall change", "difference from beginning to end")),
    Card("task:trend", "task", "the direction or rate of change over time",
         ("trend over time", "direction of movement", "whether the series rises or falls", "time based tendency"),
         ("longitudinal direction", "upward or downward movement")),
    Card("task:detect_change", "task", "the time or interval containing the largest change or change point",
         ("largest jump location", "time of the greatest change", "strongest change point", "interval with the biggest shift"),
         ("moment of the sharpest change", "largest movement location")),
    Card("task:compare", "task", "a comparison between two periods, groups, selections, or quantities",
         ("comparison between two groups", "contrast of two periods", "difference between two selections", "side by side comparison"),
         ("compare the alternatives", "contrast between the two sets")),
    Card("task:count", "task", "the number of matching events, episodes, or selected items",
         ("event count", "number of occurrences", "quantity of matching episodes", "numeric occurrence count"),
         ("occurrence total", "number of matching records")),
    Card("task:list", "task", "the individual matching events or episodes as a list",
         ("event listing", "individual occurrences", "enumerated matching episodes", "all matching event records"),
         ("occurrence list", "matching records individually")),
    Card("task:event_context", "task", "signal observations in a time window before or after a referenced event",
         ("measurements around the event", "context before and after the warning", "observations near the event time", "event centered signal window"),
         ("readings surrounding the incident", "values near the warning")),
    Card("task:explain_change", "task", "measured factors associated with a change in a target quantity",
         ("factors associated with the change", "measured contributors to the increase", "recorded contributors to the decrease", "evidence related to the change"),
         ("variables accompanying the change", "recorded factors behind the shift")),
    Card("task:capability", "task", "whether a signal or property is known and available in this vehicle history",
         ("data availability", "recording capability", "whether the signal is recorded", "presence of the measurement"),
         ("availability of the reading", "whether the vehicle stores the value")),
    Card("task:percent_time", "task", "the fraction or percentage of time for which a condition is true",
         ("percentage of time satisfying the condition", "share of the interval in a state", "fraction of time above the threshold", "time proportion"),
         ("portion of time matching the condition", "time share")),
    Card("task:rank", "task", "items ordered by a metric with the top or bottom items selected",
         ("ranked items", "ordered selection by a metric", "top items after sorting", "bottom items after sorting"),
         ("selection by rank", "ordered list of items")),
    Card("task:correlation", "task", "the statistical association or correlation between two numeric signals",
         ("correlation between two signals", "statistical association of two series", "co movement of two measurements", "relationship between paired values"),
         ("association strength", "paired signal correlation")),
)

_SCOPES = (
    Card("scope:common", "scope", "a restriction or exclusion applies to both branches of a comparison",
         ("the restriction applies to both groups", "both branches exclude the condition", "the filter has common scope", "each side uses the same exclusion"),
         ("the condition covers the entire comparison", "neither branch includes the excluded state")),
    Card("scope:left", "scope", "a restriction or exclusion applies only to the first branch of a comparison",
         ("only the first group has the restriction", "the left branch alone excludes the condition", "the filter has first branch scope", "the second side remains unrestricted"),
         ("the condition modifies the first alternative only", "only the first branch omits the state")),
    Card("scope:right", "scope", "a restriction or exclusion applies only to the second branch of a comparison",
         ("only the second group has the restriction", "the right branch alone excludes the condition", "the filter has second branch scope", "the first side remains unrestricted"),
         ("the condition modifies the second alternative only", "only the second branch omits the state")),
)


def task_cards() -> list[Card]:
    return list(_TASKS)


def scope_cards() -> list[Card]:
    return list(_SCOPES)


def all_cards() -> list[Card]:
    return task_cards() + scope_cards()


def signal_cards() -> list[Card]:
    cards = []
    for c in R.CONCEPTS.values():
        aliases = ", ".join(list(dict.fromkeys((c.label,) + tuple(c.synonyms)))[:8])
        detail = c.description.strip() or f"a {c.category} vehicle measurement"
        description = f"{c.label}; {detail}; unit {c.unit}; related names: {aliases}"
        cards.append(Card(f"signal:{c.name}", "signal", description, ()))
    return cards


def gold_task_key(q: Query) -> str | None:
    if q.op == "current_value":
        return "task:current"
    if q.op == "aggregate":
        return {
            "mean": "task:mean", "sum": "task:sum", "median": "task:median",
            "delta": "task:delta", "min": "task:min", "max": "task:max",
        }.get(q.agg)
    if q.op == "extreme":
        return {"max": "task:max", "min": "task:min"}.get(q.agg)
    return {
        "trend": "task:trend", "detect_change": "task:detect_change",
        "compare": "task:compare", "count": "task:count",
        "list_events": "task:list", "event_context": "task:event_context",
        "explain_event": "task:event_context", "explain_change": "task:explain_change",
        "capability": "task:capability", "percent_time": "task:percent_time",
        "rank": "task:rank", "correlation": "task:correlation",
    }.get(q.op)


def _signal_terms() -> tuple[str, ...]:
    preferred = (
        "battery temperature", "estimated range", "energy consumption",
        "vehicle speed", "motor torque", "state of charge", "tyre pressure",
        "coolant flow", "steering torque", "cabin temperature",
    )
    return preferred


def generate_micro_examples(seed: int = 13) -> list[dict]:
    """Create declarative fragment/card pairs without complete interrogatives.

    Context is deliberately crossed with every task so no context phrase predicts
    the label.  The held natural questions used by the experiment never enter here.
    """
    rng = random.Random(seed)
    periods = (
        "during the previous seven days", "over the selected interval",
        "within yesterday's observations", "across the recorded trip",
    )
    modifiers = (
        "", " while the vehicle is moving", " during highway operation",
        " excluding charging intervals", " before the warning",
    )
    records: list[dict] = []
    for card in _TASKS:
        combos = list(itertools.product(card.train_fragments, _signal_terms(), periods, modifiers))
        rng.shuffle(combos)
        for frag, signal, period, modifier in combos[:56]:
            text = f"{frag} of {signal} {period}{modifier}".strip()
            records.append({"text": text, "label": card.key, "kind": card.kind, "source": "algebra_micro"})

    predicates = ("charging", "rain", "hard braking", "cold operation")
    group_pairs = (("city", "highway"), ("weekday", "weekend"), ("first period", "second period"))
    for pred, (left, right) in itertools.product(predicates, group_pairs):
        common = (
            f"{pred} observations are excluded from both {left} and {right}",
            f"the comparison of {left} with {right} applies the {pred} restriction to each side",
            f"neither the {left} branch nor the {right} branch includes {pred}",
        )
        left_only = (
            f"{pred} is excluded from {left} only and {right} remains unrestricted",
            f"the {pred} restriction modifies the first branch {left} but not {right}",
            f"{left} omits {pred} while {right} includes all observations",
        )
        right_only = (
            f"{pred} is excluded from {right} only and {left} remains unrestricted",
            f"the {pred} restriction modifies the second branch {right} but not {left}",
            f"{right} omits {pred} while {left} includes all observations",
        )
        for key, texts in (("scope:common", common), ("scope:left", left_only), ("scope:right", right_only)):
            records.extend({"text": t, "label": key, "kind": "scope", "source": "algebra_scope"} for t in texts)

    for r in records:
        low = r["text"].strip().lower()
        if "?" in low or low.startswith(("what ", "which ", "when ", "why ", "how ", "did ", "does ", "is ", "are ")):
            raise AssertionError(f"question-shaped training item: {r['text']}")
    rng.shuffle(records)
    return records


def held_micro_examples() -> list[dict]:
    rows = []
    for card in all_cards():
        for text in card.held_fragments:
            rows.append({"text": text, "label": card.key, "kind": card.kind, "source": "held_micro"})
    return rows


def generate_roundtrip_scope_examples(seed: int = 41) -> list[dict]:
    """Generate complete declarative structures, never questions.

    This is a separate condition from strict microsemantic training.  It tests
    whether complete compositional exposure can transfer to unseen questions.
    """
    rng = random.Random(seed)
    pairs = (
        ("urban trips", "motorway trips"),
        ("morning journeys", "evening journeys"),
        ("warm days", "cool days"),
        ("outbound travel", "return travel"),
        ("workday driving", "holiday driving"),
        ("short journeys", "long journeys"),
    )
    metrics = ("energy efficiency", "battery heat", "average speed", "power usage")
    predicates = ("plugged-in samples", "wet-road samples", "idle intervals", "rapid acceleration")
    rows = []
    for (left, right), metric, pred in itertools.product(pairs, metrics, predicates):
        texts = {
            "scope:common": (
                f"{left} and {right} {metric} are contrasted after removing {pred} from both groups",
                f"a comparison of {left} with {right} {metric} excludes {pred} throughout",
                f"neither {left} nor {right} includes {pred} in the {metric} comparison",
            ),
            "scope:left": (
                f"{left} {metric} without {pred} is contrasted with unrestricted {right} {metric}",
                f"the restricted {left} group omits {pred}, whereas the complete {right} group does not",
                f"{pred} is removed from {left} alone before comparison with {right}",
            ),
            "scope:right": (
                f"unrestricted {left} {metric} is contrasted with {right} {metric} without {pred}",
                f"the complete {left} group is compared with the restricted {right} group that omits {pred}",
                f"{pred} is removed from {right} alone before comparison with {left}",
            ),
        }
        for label, variants in texts.items():
            rows.extend({"text": text, "label": label, "kind": "scope",
                         "source": "program_roundtrip_declarative"} for text in variants)
    rng.shuffle(rows)
    for row in rows:
        low = row["text"].strip().lower()
        if "?" in low or low.startswith(("what ", "which ", "when ", "why ", "how ", "did ", "does ")):
            raise AssertionError(f"question-shaped round-trip item: {row['text']}")
    return rows

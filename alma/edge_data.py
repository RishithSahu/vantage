"""Construction-separated data for the frozen-encoder ALMA-v2 edge probe."""
from __future__ import annotations

import itertools
import random

EDGE_LABELS = ("comparison", "left", "right")


def _row(text, label, left, right, modifier, source, family):
    def span(value):
        start = text.index(value)
        return [start, start + len(value)]
    return {
        "text": text, "label": label, "left": span(left), "right": span(right),
        "modifier": span(modifier), "source": source, "family": family,
    }


def training_rows(seed=101):
    """Question-free declaratives; no evaluation construction occurs here."""
    pairs = (("urban journeys", "motorway journeys"), ("morning runs", "evening runs"),
             ("outbound legs", "return legs"), ("short trips", "long trips"),
             ("workday travel", "holiday travel"))
    metrics = ("energy usage", "battery heat", "mean velocity", "motor load")
    modifiers = ("plugged-in intervals", "wet-road samples", "idle periods",
                 "rapid acceleration", "cold starts")
    templates = {
        "comparison": (
            "The {left} and {right} {metric} comparison removes {modifier} from each group.",
            "Neither {left} nor {right} includes {modifier} in the {metric} contrast.",
            "After omitting {modifier} throughout, {left} {metric} is contrasted with {right}.",
        ),
        "left": (
            "The restricted {left} {metric} omits {modifier}, while {right} remains complete.",
            "{left} without {modifier} is contrasted against unrestricted {right} for {metric}.",
            "Removal of {modifier} from {left} alone precedes its {metric} contrast with {right}.",
        ),
        "right": (
            "The complete {left} {metric} is contrasted with {right}, which omits {modifier}.",
            "Unrestricted {left} is compared against {right} without {modifier} for {metric}.",
            "Removal of {modifier} from {right} alone precedes its {metric} contrast with {left}.",
        ),
    }
    rows = []
    for (left, right), metric, modifier, label in itertools.product(pairs, metrics, modifiers, EDGE_LABELS):
        for i, template in enumerate(templates[label]):
            text = template.format(left=left, right=right, metric=metric, modifier=modifier)
            rows.append(_row(text, label, left, right, modifier, "question_free_declarative", f"train_{label}_{i}"))
    random.Random(seed).shuffle(rows)
    return rows


def challenge_rows(seed=211):
    """Balanced natural questions using held-out lexical and syntactic families."""
    pairs = (("city driving", "highway driving"), ("weekday trips", "weekend trips"),
             ("first period", "second period"), ("daytime travel", "night travel"),
             ("charging sessions", "ordinary journeys"))
    metrics = ("consumption", "efficiency", "temperature", "speed")
    modifiers = ("charging", "rain", "hard braking", "traffic jams", "overheating")
    templates = {
        "comparison": (
            "Compare {left} and {right} {metric}, excluding {modifier} from both.",
            "How does {left} {metric} compare with {right} when {modifier} is removed throughout?",
            "What is the {metric} difference between {left} and {right} if neither includes {modifier}?",
            "Contrast {left} with {right} for {metric}, leaving out {modifier} on either side.",
            "Between {left} and {right}, compare {metric} after filtering {modifier} everywhere.",
        ),
        "left": (
            "Compare {left} {metric} excluding {modifier} with unrestricted {right}.",
            "How does {left} without {modifier} compare with complete {right} for {metric}?",
            "Contrast the {metric} of {left}, leaving out {modifier}, against all {right}.",
            "For {metric}, compare non-{modifier} {left} with {right} overall.",
            "What is the {metric} difference when {modifier} is removed only from {left}, not {right}?",
        ),
        "right": (
            "Compare unrestricted {left} {metric} with {right} excluding {modifier}.",
            "How does complete {left} compare with {right} without {modifier} for {metric}?",
            "Contrast the {metric} of all {left} against {right}, leaving out {modifier} there.",
            "For {metric}, compare {left} overall with non-{modifier} {right}.",
            "What is the {metric} difference when {modifier} is removed only from {right}, not {left}?",
        ),
    }
    rows = []
    combos = list(itertools.product(pairs, metrics, modifiers))
    for label in EDGE_LABELS:
        for i, ((left, right), metric, modifier) in enumerate(combos):
            template = templates[label][i % len(templates[label])]
            text = template.format(left=left, right=right, metric=metric, modifier=modifier)
            rows.append(_row(text, label, left, right, modifier, "heldout_natural_construction", f"held_{label}_{i % 5}"))
    random.Random(seed).shuffle(rows)
    return rows


def existing_natural_rows():
    """The earlier 12-item sanity set with spans recovered from known entities."""
    specs = (
        ("Compare city and highway consumption, excluding charging.", "comparison", "city", "highway", "charging"),
        ("How does city consumption compare with highway consumption when charging periods are removed from both?", "comparison", "city", "highway", "charging"),
        ("Compare the two periods, ignoring rain in either period.", "comparison", "two", "periods", "rain"),
        ("City versus highway efficiency, with hard-braking samples left out throughout.", "comparison", "City", "highway", "hard-braking"),
        ("Compare city consumption excluding charging with highway consumption.", "left", "city", "highway", "charging"),
        ("How does non-charging city use compare with unrestricted highway use?", "left", "city", "highway", "charging"),
        ("Contrast the first period without rain against the complete second period.", "left", "first", "second", "rain"),
        ("Compare cold-weather weekday efficiency with weekend efficiency overall.", "left", "weekday", "weekend", "cold-weather"),
        ("Compare city consumption with highway consumption excluding charging.", "right", "city", "highway", "charging"),
        ("How does unrestricted city use compare with non-charging highway use?", "right", "city", "highway", "charging"),
        ("Contrast the complete first period against the second period without rain.", "right", "first", "second", "rain"),
        ("Compare weekday efficiency overall with cold-weather weekend efficiency.", "right", "weekday", "weekend", "cold-weather"),
    )
    return [_row(*spec, "existing_natural", "existing") for spec in specs]

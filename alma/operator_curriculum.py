"""Formally verified operator/aggregation contrast curriculum.

This module generates data only.  It deliberately does not train a model: the
oracle ablation must determine the curriculum target before a learned component
is introduced, and the generated examples must pass structural and denotational
checks first.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from vgpt.query_ir import Query
from vgpt.query_validator import validate
from .supervised_parser import ROOT, skeleton


TRAIN_WORDING = {
    "aggregate_extreme": ("Give the maximum {name} over the past week.", "Locate the peak {name} over the past week."),
    "count_list": ("How many {event} occurred this week?", "List the {event} from this week."),
    "context_explain_event": ("Show what happened around the latest {event_one}.", "Explain the latest {event_one}."),
    "trend_explain_change": ("Show the two-week trend in {name}.", "Explain the two-week change in {name}."),
    "explain_detect": ("Explain the recent change in {name}.", "Detect when {name} changed."),
    "rank_count": ("Rank {selection}.", "Count {selection}."),
    "share_count": ("What share of this week was speed above {threshold}?", "Count speed records above {threshold} this week."),
    "current_capability": ("What is the current {name}?", "Can this vehicle report {name}?"),
    "aggregate_compare": ("Give average {name} this week.", "Compare average {name} this week with last week."),
    "mean_median": ("Give mean {name} over the past week.", "Give median {name} over the past week."),
    "std_p95": ("Give the variability of {name} over the past week.", "Give the 95th percentile of {name} over the past week."),
    "min_max": ("Find the lowest {name} over the past week.", "Find the highest {name} over the past week."),
    "first_last": ("Give the first recorded {name} this week.", "Give the last recorded {name} this week."),
    "sum_delta": ("Give total {energy_name} {period}.", "Give the change in {energy_name} {period}."),
}

HELDOUT_WORDING = {
    "aggregate_extreme": ("What was the largest weekly {name} reading?", "At which extreme did {name} top out during the week?"),
    "count_list": ("What is the number of {event} for this week?", "Which {event} happened this week?"),
    "context_explain_event": ("Display the window surrounding the newest {event_one}.", "Why did the newest {event_one} happen?"),
    "trend_explain_change": ("How has {name} moved across two weeks?", "What accounts for the two-week shift in {name}?"),
    "explain_detect": ("What accounts for the latest {name} shift?", "When did the {name} shift begin?"),
    "rank_count": ("Order {selection}.", "How many of {selection} are selected?"),
    "share_count": ("For what fraction of this week did speed exceed {threshold}?", "How many speed samples exceeded {threshold} this week?"),
    "current_capability": ("Read {name} right now.", "Is {name} available from this car?"),
    "aggregate_compare": ("What did {name} average this week?", "How does this week's mean {name} differ from last week's?"),
    "mean_median": ("What is the arithmetic average {name} for the week?", "What is the middle weekly {name} reading?"),
    "std_p95": ("How dispersed were the weekly {name} readings?", "What weekly {name} value marks percentile 95?"),
    "min_max": ("Where did weekly {name} bottom out?", "Where did weekly {name} top out?"),
    "first_last": ("What {name} reading opened the week?", "What {name} reading closed the week?"),
    "sum_delta": ("How much {energy_name} accumulated {period}?", "By how much did {energy_name} change {period}?"),
}

TARGETS = (("speed", "speed"), ("battery_temperature", "battery temperature"),
           ("motor_torque", "motor torque"), ("battery_soc", "battery level"))
EVENTS = (("ANY_WARNING", "warnings", "warning"),
          ("HARSH_BRAKE", "harsh-braking events", "harsh-braking event"),
          ("TPMS_LOW", "low-tyre-pressure alerts", "low-tyre-pressure alert"),
          ("CHARGE_START", "charge starts", "charge start"))
SELECTIONS = (("trip:wh_per_km:desc:3:highway", "the three least efficient highway trips"),
              ("trip:distance_driven:desc:2:city", "the two longest city trips"),
              ("trip:avg_speed:desc:4:highway", "the four fastest highway trips"),
              ("charge_session:charging_power:desc:3:-", "the three highest-power charge sessions"))
ENERGY_CASES = (("energy_used", "energy used", "t=cal:this_week", "this week"),
                ("regen_energy", "regenerated energy", "t=cal:last_week", "last week"),
                ("distance_driven", "distance travelled", "t=cal:this_month", "this month"),
                ("energy_used", "energy used", "t=cal:yesterday", "yesterday"))


def _row(text, ir, split, family):
    query = Query.from_text(ir)
    return {"text": text, "ir": ir, "labels": skeleton(query), "split": split,
            "family": family, "source": "formal_operator_contrast"}


def _specs(target, name, variant):
    """Return (family, text args, IR A, IR B, allowed changed labels)."""
    week = "t=last:7:day"
    event_id, event_plural, event_one = EVENTS[variant]
    selection_ir, selection_text = SELECTIONS[variant]
    threshold = (40, 60, 80, 100)[variant]
    energy_target, energy_name, energy_time, energy_period = ENERGY_CASES[variant]
    common = {"name": name, "event": event_plural, "event_one": event_one,
              "selection": selection_text, "threshold": threshold,
              "energy_name": energy_name, "period": energy_period}
    return [
        ("aggregate_extreme", {"name": name},
         f"op=aggregate tgt={target} agg=max {week}", f"op=extreme tgt={target} agg=max {week}", {"op"}),
        ("count_list", common, f"op=count t=cal:this_week ev={event_id}",
         f"op=list_events t=cal:this_week ev={event_id}", {"op"}),
        ("context_explain_event", common, f"op=event_context t=all ev={event_id} bef=600 aft=600",
         f"op=explain_event t=all ev={event_id} bef=600 aft=600", {"op"}),
        ("trend_explain_change", {"name": name}, f"op=trend tgt={target} t=last:14:day",
         f"op=explain_change tgt={target} t=last:14:day", {"op"}),
        ("explain_detect", {"name": name}, f"op=explain_change tgt={target} t=last:14:day",
         f"op=detect_change tgt={target} t=last:14:day", {"op"}),
        ("rank_count", common, f"op=rank t=all sel={selection_ir}",
         f"op=count t=all sel={selection_ir}", {"op"}),
        ("share_count", common, f"op=percent_time tgt=speed t=cal:this_week c=sig:speed:gt:{threshold}",
         f"op=count tgt=speed t=cal:this_week c=sig:speed:gt:{threshold}", {"op"}),
        ("current_capability", {"name": name}, f"op=current_value tgt={target} t=now",
         f"op=capability tgt={target} t=now", {"op"}),
        ("aggregate_compare", {"name": name}, f"op=aggregate tgt={target} agg=mean t=cal:this_week",
         f"op=compare tgt={target} agg=mean t=cal:this_week t2=cal:last_week", {"op", "time2_type"}),
        ("mean_median", {"name": name}, f"op=aggregate tgt={target} agg=mean {week}",
         f"op=aggregate tgt={target} agg=median {week}", {"agg"}),
        ("std_p95", {"name": name}, f"op=aggregate tgt={target} agg=std {week}",
         f"op=aggregate tgt={target} agg=p95 {week}", {"agg"}),
        ("min_max", {"name": name}, f"op=extreme tgt={target} agg=min {week}",
         f"op=extreme tgt={target} agg=max {week}", {"agg"}),
        ("first_last", {"name": name}, f"op=aggregate tgt={target} agg=first t=cal:this_week",
         f"op=aggregate tgt={target} agg=last t=cal:this_week", {"agg"}),
        ("sum_delta", common, f"op=aggregate tgt={energy_target} agg=sum {energy_time}",
         f"op=aggregate tgt={energy_target} agg=delta {energy_time}", {"agg"}),
    ]


def contrast_pairs(split="train"):
    wording = TRAIN_WORDING if split == "train" else HELDOUT_WORDING
    repetitions = 4
    pairs = []
    for i in range(repetitions):
        target, name = TARGETS[i % len(TARGETS)]
        for family, args, ir_a, ir_b, allowed in _specs(target, name, i):
            ta, tb = wording[family]
            pairs.append((_row(ta.format(**args), ir_a, split, family),
                          _row(tb.format(**args), ir_b, split, family), family, allowed))
    return pairs


def deterministic_denotation(ir):
    """Execute the requested semantic reduction in a fixed miniature world.

    Values are deliberately heterogeneous (scalar, sequence and explanation)
    so operations that happen to share a numeric statistic still differ in the
    kind of answer they request.
    """
    q = Query.from_text(ir)
    values = [12.0, 27.0, 19.0, 44.0, 31.0]
    aggs = {"mean": sum(values) / len(values), "median": 27.0, "std": 10.8,
            "p95": 44.0, "min": min(values), "max": max(values),
            "first": values[0], "last": values[-1], "sum": sum(values),
            "delta": values[-1] - values[0]}
    if q.op == "aggregate": return ("statistic", q.agg, aggs[q.agg])
    if q.op == "extreme": return ("extreme_with_location", q.agg, aggs[q.agg], values.index(aggs[q.agg]))
    if q.op == "current_value": return ("current", values[-1])
    if q.op == "count": return ("count", 3)
    if q.op == "list_events": return ("events", ("W1", "W2", "W3"))
    if q.op == "event_context": return ("event_window", (27.0, 44.0, 31.0))
    if q.op == "explain_event": return ("event_explanation", "temperature_spike")
    if q.op == "trend": return ("trend_slope", 4.2)
    if q.op == "explain_change": return ("change_explanation", "driving_mix")
    if q.op == "detect_change": return ("change_point", 3)
    if q.op == "rank": return ("ranked_trips", ("T3", "T1", "T2"))
    if q.op == "percent_time": return ("fraction", 0.4)
    if q.op == "capability": return ("available", True)
    if q.op == "compare": return ("comparison", aggs[q.agg], 22.0, aggs[q.agg] - 22.0)
    raise AssertionError(f"No miniature-world evaluator for {q.op}")


def verify_pairs(pairs):
    failures, family_counts = [], Counter()
    for a, b, family, allowed in pairs:
        qa, qb = Query.from_text(a["ir"]), Query.from_text(b["ir"])
        va, vb = validate(qa), validate(qb)
        changed = {k for k in a["labels"] if a["labels"][k] != b["labels"][k]}
        problems = []
        if not va.valid: problems.append({"a_validation": va.codes()})
        if not vb.valid: problems.append({"b_validation": vb.codes()})
        if not changed: problems.append("no structural contrast")
        if not changed <= allowed: problems.append({"unexpected_changed_labels": sorted(changed - allowed)})
        if deterministic_denotation(a["ir"]) == deterministic_denotation(b["ir"]):
            problems.append("same denotation")
        if problems: failures.append({"family": family, "problems": problems, "a": a, "b": b})
        else: family_counts[family] += 1
    return {"pairs": len(pairs), "verified": len(pairs) - len(failures),
            "families": dict(sorted(family_counts.items())), "failures": failures}


def main():
    output = os.path.join(ROOT, "results", "operator_curriculum")
    os.makedirs(output, exist_ok=True)
    train, heldout = contrast_pairs("train"), contrast_pairs("heldout")
    verification = {"train": verify_pairs(train), "heldout": verify_pairs(heldout)}
    if verification["train"]["failures"] or verification["heldout"]["failures"]:
        raise RuntimeError("Operator curriculum failed formal verification")
    with open(os.path.join(output, "contrasts.jsonl"), "w", encoding="utf-8") as f:
        for a, b, family, allowed in train + heldout:
            f.write(json.dumps({"family": family, "allowed_changed_labels": sorted(allowed),
                                "a": a, "b": b,
                                "denotation_a": deterministic_denotation(a["ir"]),
                                "denotation_b": deterministic_denotation(b["ir"])}) + "\n")
    result = {"verification": verification, "natural_training_examples": 100,
              "natural_development_examples": 120, "synthetic_training_examples": 2 * len(train),
              "synthetic_heldout_examples": 2 * len(heldout), "trained_new_component": False,
              "opened_acceptance_or_final": False, "schema_linker_changed": False,
              "typed_ir_changed": False, "executor_changed": False,
              "design": "operator-priority contrasts with compatible aggregation contrasts"}
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lines = ["# Verified Operator/Aggregation Curriculum", "",
             "Generated after the operator/aggregation oracle ablation. No model was trained in this step.", "",
             f"- Training: {len(train)} pairs / {2*len(train)} examples, all verified.",
             f"- Held-out: {len(heldout)} pairs / {2*len(heldout)} examples, all verified.",
             f"- Contrast families: {len(verification['train']['families'])}.",
             "- Every IR passes the existing validator; every pair changes only its declared structural fields and has different output in the deterministic miniature vehicle world.",
             "- Natural and synthetic evaluation remain separate; acceptance and final benchmarks remain sealed.", "",
             "The next learned experiment should predict the compatible `(operator, aggregation)` signature jointly. The oracle's positive interaction and invalid combinations from isolated substitution make two unconstrained independent replacements a poor test design."]
    with open(os.path.join(output, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Structure-grouped curriculum for the shared 0.5B semantic front door.

Runtime imports only SYSTEM. Generated rows carry canonical frames plus their
compact model-token encoding. Splits hold out entire surface templates; a
separate composition split holds out operation/concept combinations.
"""
from __future__ import annotations
import hashlib, json, os, random
from vgpt.semantic_frame import SemanticFrame

SYSTEM = """Map a vehicle request to exactly four ontology words: task intent metric event.
task: data | define | estimate | capability | advice | other
intent: why | before | events | compare | largest | trend | now | usage | average | maximum | minimum | duration | other
metric: energy | battery | motor | coolant | speed | charge | chargepower | torque | outside | climate | range | tyre | weather | service | health | driving | place | vehicle | none
event: adas | overheat | fault | charging | camera | none
Output only the four words."""

CONCEPTS = {
    "speed": ["road velocity", "driving pace", "vehicle speed"],
    "charge": ["battery level", "state of charge", "remaining battery percentage"],
    "chargepower": ["electrical charging rate", "charger power", "power flowing into the pack", "charging speed"],
    "motor": ["traction motor temperature", "electric drive unit", "motor heat", "EV engine"],
    "battery": ["pack temperature", "battery heat", "traction battery temperature"],
    "coolant": ["coolant flow", "cooling circuit flow", "thermal-loop flow"],
    "energy": ["energy consumption", "efficiency", "electricity use"],
    "outside": ["ambient temperature", "outside temperature", "external air temperature"],
}
TELEMETRY = {
    "now": ["Report the current {x}", "Give me the live {x}", "What is {x} at the moment?"],
    "average": ["Calculate the mean {x} for last week", "Summarise the typical {x} yesterday", "Over the week, what did {x} average?"],
    "maximum": ["Find the peak {x} this month", "Return the greatest recorded {x}", "How high did {x} get?"],
    "minimum": ["Find the lowest {x} this month", "Return the smallest recorded {x}", "How low did {x} fall?"],
    "trend": ["Describe the trend in {x}", "Show how {x} changed over the week", "Has {x} been rising or falling?"],
}
COMPOSITION_HOLDOUT = {("trend", "chargepower"), ("minimum", "motor"), ("maximum", "charge"), ("average", "coolant")}

def _row(text, task, op, concept, event, template_id, structure_id):
    frame = SemanticFrame.from_tokens(task, op, concept, event)
    return {"text": text, "frame": frame.to_tokens(), "canonical_frame": frame.as_dict(),
            "template_id": template_id, "structure_id": structure_id,
            "composition_id": f"{task}:{op}:{concept}:{event}"}

def rows(seed=8128):
    out = []
    for concept, names in CONCEPTS.items():
        for name in names:
            for op, forms in TELEMETRY.items():
                for i, form in enumerate(forms):
                    out.append(_row(form.format(x=name), "data", op, concept, "none", f"telemetry.{op}.{i}", f"data:{op}"))
    families = [
        (("driver assistance", "adaptive cruise", "lane support"), "define", "other", "driving", "adas",
         ("Define {x}", "Explain the meaning of {x}", "In simple terms, what does {x} mean?"), "definition"),
        (("time remaining until full", "duration from the present level to full", "minutes required to reach the target level"), "estimate", "duration", "charge", "none",
         ("Estimate the {x}", "Predict the {x}", "Give a forecast for the {x}"), "estimate"),
        (("motor torque", "shaft torque", "wheel torque"), "capability", "other", "torque", "none",
         ("Does the data contain {x}?", "Why is measured {x} unavailable?", "Can this vehicle report {x}?"), "capability"),
        (("replacement tyres", "winter wheels", "higher-load tyres", "performance rubber"), "advice", "other", "tyre", "none",
         ("Recommend {x}", "Which {x} should the owner choose?", "Where might the owner source {x}?", "Help choose {x}"), "advice"),
        (("fault records", "diagnostic trouble codes", "ECU warnings", "saved diagnostics", "trouble records"), "data", "events", "none", "fault",
         ("List the {x}", "How many {x} occurred?", "Show every {x} from this month"), "event-list"),
    ]
    for names, task, op, concept, event, forms, structure in families:
        for name in names:
            for i, form in enumerate(forms): out.append(_row(form.format(x=name), task, op, concept, event, f"{structure}.{i}", structure))
    for i, text in enumerate(("Report the EV engine status", "Give the present condition of the drive unit", "Check the electric motor now")):
        out.append(_row(text, "data", "now", "motor", "none", f"motor-status.{i}", "data:now"))
    for i, text in enumerate(("Contrast city and highway operation", "Compare urban with motorway driving", "How do town and freeway use differ?")):
        out.append(_row(text, "data", "compare", "none", "none", f"mode-compare.{i}", "data:compare"))
    random.Random(seed).shuffle(out)
    return out

def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for row in data: f.write(json.dumps(row, sort_keys=True) + "\n")

def write(root):
    data = rows(); os.makedirs(root, exist_ok=True)
    # Entire constructions are held out. Index 2 is never present in training;
    # advice has a fourth construction, also held out.
    val_templates = {tid for tid in {r["template_id"] for r in data} if tid.endswith(".2") or tid == "advice.3"}
    comp, val, train = [], [], []
    for row in data:
        parts = row["composition_id"].split(":")
        if len(parts) == 4 and (parts[1], parts[2]) in COMPOSITION_HOLDOUT: comp.append(row)
        elif row["template_id"] in val_templates: val.append(row)
        else: train.append(row)
    for name, part in (("train", train), ("val", val), ("test_comp", comp)): _write(os.path.join(root, name + ".jsonl"), part)
    train_templates, val_templates_actual = {r["template_id"] for r in train}, {r["template_id"] for r in val}
    train_comp, held_comp = {r["composition_id"] for r in train}, {r["composition_id"] for r in comp}
    with open(__file__, "rb") as source: generator_hash = hashlib.sha256(source.read()).hexdigest()
    audit = {"counts": {"train": len(train), "val": len(val), "test_comp": len(comp)},
             "template_overlap_train_val": sorted(train_templates & val_templates_actual),
             "composition_overlap_train_test_comp": sorted(train_comp & held_comp),
             "held_templates": sorted(val_templates_actual), "held_compositions": sorted(held_comp),
             "generator_sha256": generator_hash}
    with open(os.path.join(root, "split_audit.json"), "w", encoding="utf-8") as f: json.dump(audit, f, indent=2)
    if audit["template_overlap_train_val"] or audit["composition_overlap_train_test_comp"]: raise AssertionError(audit)
    return audit

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "semantic_data_grouped")); a=p.parse_args()
    print(json.dumps(write(a.out), indent=2))

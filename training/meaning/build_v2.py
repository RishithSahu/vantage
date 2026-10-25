"""Build the training data for frame version 2 (docs/FRAME_V2.md): the full_v11 recipe, every frame row widened to eight words, plus rows for the new slots.

    python training/meaning/build_v2.py --out training/meaning/data/frame2 [--seed 7]

1. the base: build_meaning.py with the full_v11 sources (run once into <out>_base, version-1 frames of six words), every frame row of it gets `c2 none ref none` appended;
2. new rows, all with eight-word answers (labels fixed by construction or hand-written, never read from a model):
     v2_corr / v2_two / v2_either / v2_typical / v2_explain   single questions (templates x the pair lists below, and the hand-written natural_v2.txt)
     v2_period / v2_concept / v2_focus / v2_item / v2_answer   follow-ups: the row carries `prev`, the eight-word frame of the previously answered question, shown to the model as the line `Previous: ...`
     v2_fresh                                                   an unrelated question after an answer: ref stays none, so the model learns not to inherit when nothing is referred to
3. the leakage filter: a row whose wording overlaps (token-set Jaccard >= 0.8) the frozen sets of version 1 or of version 2 (slots_v2, followup_v2) is dropped and counted.
Output: train.jsonl, val.jsonl, test_comp.jsonl, manifest.json.  Training: `VGPT_FRAME=2 python training/meaning/train_meaning.py --data <out> ...` (the prompt then carries the version-2 system prompt and `Previous:`)."""
import argparse
import collections
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ["VGPT_FRAME"] = "2"
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from vgpt.meaning import glossary as GL, ontology as ON       # noqa: E402
from vgpt.meaning.frame import Frame                           # noqa: E402
import noise as NZ                                             # noqa: E402
import templates as T                                          # noqa: E402

O = ON.load(2); G = GL.load()
LEX = json.load(open(os.path.join(ROOT, "vgpt", "meaning", "lexicon.json"), encoding="utf-8"))
FULL_V11 = "aux,template,natural,abstain,feedback,honesty,hard1,hard2,hard3,platform,adv1,adv2,adv3,adv4"
norm_tokens = lambda s: re.findall(r"[a-z0-9]+", str(s).lower())


def F(act, concept, op="none", cond="none", vs="none", ent="none", c2="none", ref="none"): return Frame(act, concept, op, cond, vs, ent, c2, ref)


def phrases(cid):
    seen, out = set(), []
    c = O.concepts[cid]
    for p in LEX.get(cid, []) + G.aliases(cid) + list(c.aliases):
        p = " ".join(str(p).lower().split())
        if p and p not in seen and not re.search(r"^(how|when|whether|where|what|if|which|why|who)\b", p): seen.add(p); out.append(p)
    return out or [cid.replace("_", " ")]


# ----------------------------------------------------------------------------------------------------------------------------------------------- hand-written rows
def parse_spec(spec):
    toks = spec.split(); kw = dict(op="none", cond="none", vs="none", ent="none", c2="none", ref="none")
    for t in toks[2:]:
        k, v = t.split("="); kw[k] = v
    return F(toks[0], toks[1], **kw)


def parse_natural2(path):
    rows = []
    for ln, line in enumerate(io.open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if " | " not in line: raise ValueError(f"{path}:{ln}: missing ' | '")
        left, spec = line.rsplit(" | ", 1); prev = None; veh = None
        if left.startswith("@"): tag, left = left.split(" ", 1); veh = tag[1:] if tag[1:] in ("ev", "petrol", "diesel", "ice", "hybrid") else None
        if " >> " in left: pspec, text = left.split(" >> ", 1); prev = parse_spec(pspec.strip())
        else: text = left
        f = parse_spec(spec); issues = O.validate(f) + (O.validate(prev) if prev else [])
        if issues: raise ValueError(f"{path}:{ln}: {text!r} -> {f.to_text(2)}: {issues}")
        rows.append({"text": text.strip(), "frame": f, "prev": prev, "vehicle": veh})
    return rows


# ----------------------------------------------------------------------------------------------------------------------------------------------- templates
PAIRS = [("speed", "battery_power"), ("speed", "motor_power"), ("speed", "motor_temperature"), ("speed", "brake_pressure"), ("speed", "steering_angle"), ("speed", "yaw_rate"), ("speed", "lateral_acceleration"),
         ("accelerator_position", "motor_torque"), ("accelerator_position", "battery_power"), ("ambient_temperature", "hvac_power"), ("ambient_temperature", "tyre_pressure"), ("ambient_temperature", "battery_temperature"),
         ("battery_temperature", "motor_temperature"), ("battery_temperature", "coolant_flow"), ("battery_voltage", "battery_current"), ("battery_voltage", "battery_soc"), ("regen_power", "speed"), ("regen_power", "brake_pressure"),
         ("motor_rpm", "speed"), ("motor_rpm", "motor_torque"), ("motor_power", "battery_power"), ("hvac_power", "battery_power"), ("cell_temperature_max", "battery_temperature"), ("tyre_temperature", "speed"),
         ("tyre_temperature", "tyre_pressure"), ("camera_brightness", "light"), ("camera_sharpness", "speed"), ("lane_confidence", "camera_sharpness"), ("brake_pressure", "brake_pedal"), ("longitudinal_acceleration", "brake_pressure"),
         ("aux_battery_voltage", "battery_voltage"), ("coolant_flow", "motor_temperature"), ("battery_current", "motor_power"), ("steering_rate", "yaw_rate"), ("accelerator_position", "speed")]
EVENT_PAIRS = [("battery_overheat_warning", "overtemp_warning"), ("tyre_pressure_warning", "low_12v_warning"), ("hard_braking", "hard_acceleration"), ("hard_braking", "hard_cornering"), ("adas_disengagement", "aeb_activation"),
               ("camera_blocked", "camera_degraded"), ("ecu_reset", "power_derate"), ("voltage_dip", "low_12v_warning"), ("can_error", "can_anomaly"), ("unexpected_shutdown", "ecu_reset"), ("sensor_conflict", "sensor_invalid_reading"),
               ("source_dropout", "can_error"), ("coolant_flow_drop", "battery_overheat_warning"), ("tyre_inflation", "tyre_pressure_warning"), ("hard_cornering", "hard_acceleration"), ("camera_blocked", "tyre_pressure_warning")]
TYP_SIGNALS = ["speed", "battery_temperature", "motor_temperature", "battery_power", "hvac_power", "tyre_pressure", "battery_voltage", "aux_battery_voltage", "coolant_flow", "ambient_temperature", "regen_power", "wh_per_km", "energy_used",
               "distance_driven", "regen_energy", "motor_rpm", "accelerator_position", "brake_pressure", "cell_temperature_max", "camera_sharpness", "motor_torque", "battery_current"]
TYP_EVENTS = ["hard_braking", "hard_acceleration", "any_warning", "adas_disengagement", "tyre_pressure_warning", "camera_blocked", "charging_session", "ecu_reset", "diagnostic_code", "voltage_dip", "battery_overheat_warning", "hard_cornering"]
NOW_OK = {"speed", "battery_temperature", "motor_temperature", "battery_power", "hvac_power", "tyre_pressure", "battery_voltage", "aux_battery_voltage", "coolant_flow", "ambient_temperature", "motor_rpm", "accelerator_position", "brake_pressure", "cell_temperature_max", "battery_current"}

CORR_T = ["are {x} and {y} connected {t}", "does {x} rise when {y} rises {t}", "how much does {x} vary with {y} {t}", "{x} versus {y}: is there a pattern {t}", "does {x} change along with {y} {t}", "do you see {x} moving with {y}",
          "check whether {x} goes up with {y} {t}", "relationship of {x} to {y} {t}", "how tightly is {x} bound to {y}", "does a higher {y} mean a higher {x}", "what is the correlation of {x} with {y} {t}", "is {x} influenced by {y} {t}",
          "do {x} and {y} behave alike {t}", "does {x} respond to {y}", "{x} and {y}: do they correlate {t}", "is {x} driven by {y} {t}"]
TWO_T = {"mean": ["{x} and {y} together {t}", "show {x} and {y} side by side {t}", "{x} next to {y} {t}", "give me both {x} and {y} {t}", "compare {x} with {y} {t}", "{x} versus {y} {t}", "put {x} and {y} in one answer {t}",
                  "how do {x} and {y} stack up {t}", "{x} alongside {y} {t}", "average {x} and average {y} {t}"],
         "max": ["highest {x} and highest {y} {t}", "the peak of {x} and of {y} {t}", "max {x} next to max {y} {t}", "which got higher {t}, {x} or {y}", "top {x} together with top {y} {t}"],
         "min": ["lowest {x} and lowest {y} {t}", "the minimum of {x} and of {y} {t}", "min {x} next to min {y} {t}", "which got lower {t}, {x} or {y}", "bottom {x} together with bottom {y} {t}"]}
EITHER_T = {"exists": ["did {e} or {f} happen {t}", "any {e} or {f} {t}", "was there {e} or {f} {t}", "{e} and {f}, did either occur {t}", "has either {e} or {f} come up {t}", "either {e} or {f} {t}?"],
            "count": ["how many {e} or {f} {t}", "count {e} plus {f} {t}", "the number of {e} and {f} together {t}", "total of {e} and {f} {t}"]}
TYP_SIG_T = ["is {x} higher than usual {t}", "is {x} lower than normal {t}", "is {x} normal for my car {t}", "does {x} look unusual {t}", "how does {x} {t} compare with what is typical", "is {x} within its usual range {t}",
             "{x} {t}: normal or not", "has {x} been above its norm {t}", "compared with my usual {x}, how is it {t}", "is {x} out of the ordinary {t}", "is {x} typical for this vehicle {t}"]
TYP_EXT_T = {"max": ["is the highest {x} {t} normal", "was the peak {x} {t} unusual", "is the top {x} {t} typical for me"], "min": ["is the lowest {x} {t} normal", "was the minimum {x} {t} unusual", "is the bottom {x} {t} typical for me"]}
TYP_EVT_T = ["were there more {e} than usual {t}", "is the number of {e} {t} higher than normal", "are {e} unusually frequent {t}", "is the count of {e} {t} typical", "did {e} happen more often than usual {t}", "are {e} within what is normal {t}"]
TYP_NOW_T = ["is {x} normal right now", "is {x} at its usual level at the moment", "is the current {x} typical for me", "does {x} look normal at this moment", "is {x} what I would expect it to be now"]
EXPL = ["how did you work that out", "what did you base that on", "explain how you got this", "how reliable is that number", "what is the source of that figure", "can you justify that answer", "what data is behind that", "prove it",
        "how do you know that", "walk me through that answer", "where did you get that from", "what calculation was that", "how confident are you in that", "which records gave that result", "tell me your reasoning", "why that number",
        "how was this worked out", "give me the working", "what is that figure built on", "can you back that up", "how come you say so", "what is your evidence", "which data points did you use", "how accurate is that answer"]
PERIOD_U = ["and {t}", "what about {t}", "how about {t}", "same for {t}", "{t} instead", "now {t}", "and for {t}", "try {t}", "{t} then", "let us see {t}", "can you do {t}", "make it {t}"]
CONCEPT_U = ["and the {c}", "what about the {c}", "how about {c}", "same for the {c}", "and {c} too", "now the {c}", "do the {c}", "{c} as well", "what is it for {c}", "and for {c}"]
ITEM_U = {"max": ["which one was the {w}", "the {w} of them", "which was the {w}", "tell me the {w} one"], "min": ["which one was the {w}", "the {w} of those", "which was the {w}"], "first": ["the {w} one", "which was the {w}", "show me the {w}"],
          "last": ["the {w} one", "which was the {w}", "give me the {w}", "show me the {w} of them"]}
ITEM_W = {"max": ["strongest", "biggest", "largest", "worst", "most severe", "highest", "longest"], "min": ["weakest", "smallest", "mildest", "shortest", "lowest", "least severe"], "first": ["first", "earliest", "oldest"],
          "last": ["latest", "newest", "most recent", "last"]}
FOCUS_WHY = ["why did it happen", "what caused that", "what made that happen", "why was that", "what was the reason for it", "how come that happened", "what is behind that", "why did that occur"]
FOCUS_CTX = ["what was happening before that", "what happened leading up to it", "what came just before it", "show me the minutes before that", "what was going on around that time"]
FOCUS_AT = ["what was the {c} at that moment", "how was the {c} then", "and the {c} at that time", "what was the {c} when it happened", "the {c} back then"]
AT_SIGNALS = ["speed", "battery_temperature", "motor_temperature", "battery_power", "coolant_flow", "battery_soc", "ambient_temperature", "brake_pressure", "tyre_pressure", "hvac_power", "motor_rpm", "accelerator_position"]
FOCUS_EVENTS = ["battery_overheat_warning", "hard_braking", "adas_disengagement", "tyre_pressure_warning", "camera_blocked", "voltage_dip", "ecu_reset", "diagnostic_code", "unexpected_shutdown", "power_derate", "can_error", "low_12v_warning"]
ITEM_EVENTS = ["hard_braking", "adas_disengagement", "charging_session", "tyre_pressure_warning", "battery_overheat_warning", "camera_blocked", "diagnostic_code", "ecu_reset", "hard_acceleration", "voltage_dip"]
PERIOD_PREV = [("summary", "none", "none"), ("summary", "change", "none"), ("anomaly", "none", "none"), ("stat", "usual", "battery_temperature"), ("stat", "usual", "speed"), ("anomaly", "none", "battery_temperature"), ("summary", "none", "battery_temperature"), ("summary", "none", "any_warning")] + \
              [("events", "count", EV) for EV in ("hard_braking", "adas_disengagement", "tyre_pressure_warning", "camera_blocked", "ecu_reset", "hard_acceleration", "voltage_dip", "any_warning")] + \
              [("stat", "mean", S) for S in ("speed", "battery_temperature", "motor_temperature", "wh_per_km", "battery_power", "hvac_power", "tyre_pressure", "coolant_flow")] + \
              [("extreme", "max", S) for S in ("speed", "battery_temperature", "motor_temperature", "battery_power")] + [("extreme", "min", S) for S in ("battery_voltage", "tyre_pressure", "aux_battery_voltage")] + \
              [("trend", "change", S) for S in ("battery_temperature", "motor_temperature", "wh_per_km", "tyre_pressure")] + [("now", "none", S) for S in ("speed", "battery_temperature", "motor_temperature", "tyre_pressure")] + \
              [("events", "list", EV) for EV in ("charging_session", "hard_braking")]
CONCEPT_SWAP = {"signal": ["speed", "battery_temperature", "motor_temperature", "battery_power", "hvac_power", "tyre_pressure", "coolant_flow", "ambient_temperature", "regen_power", "battery_voltage", "aux_battery_voltage", "motor_rpm"],
                "event": ["hard_braking", "hard_acceleration", "adas_disengagement", "tyre_pressure_warning", "camera_blocked", "ecu_reset", "voltage_dip", "low_12v_warning", "any_warning", "diagnostic_code"],
                "derived": ["wh_per_km", "energy_used", "distance_driven", "regen_energy", "range_estimate"]}


# ----------------------------------------------------------------------------------------------------------------------------------------------- the digest families (summary / what changed / anomaly / usual / trip focus)
SUM_T = ["summarise {t}", "give me a summary of my driving {t}", "overview of the car {t}", "recap {t}", "how was {t} for the car", "what has the car done {t}", "brief me {t}", "a rundown of {t}", "highlights {t}", "sum up {t}",
         "report on {t}", "walk me through {t}", "tell me about my driving {t}", "how did the car do {t}", "what happened with my car {t}", "how has the vehicle been used {t}", "an account of {t}", "give me the picture {t}"]
CHG_T = ["what changed {t}", "what is different {t} compared with before", "anything new {t}", "what has moved {t}", "how are things different {t}", "list the changes {t}", "what shifted {t}", "what is not the same {t}",
         "differences {t} versus the period before", "has anything changed {t}", "which readings changed {t}", "what is trending differently {t}", "what has gone up or down {t}", "tell me what is different {t}"]
ANOM_T = ["anything strange {t}", "any anomalies {t}", "does anything look off {t}", "were there odd readings {t}", "spot oddities {t}", "any outliers {t}", "anything weird {t}", "any irregularities {t}", "did anything glitch {t}",
          "anything suspicious {t}", "were there unexpected spikes {t}", "find anything that looks wrong {t}", "scan for oddities {t}", "any red flags {t}", "what stood out as peculiar {t}"]
ANOM_C_T = ["any anomalies in the {c} {t}", "did the {c} show odd spikes {t}", "anything weird about the {c} {t}", "look for outliers in the {c} {t}", "irregular {c} readings {t}", "did the {c} glitch {t}",
            "anything strange with the {c} {t}", "any oddities in the {c} {t}", "did the {c} behave oddly {t}", "spot anything off in the {c} {t}"]
ANOM_TRIP_T = ["was my {p} trip odd", "anything weird on my {p} drive", "did anything strange happen on the {p} journey", "any anomalies in the {p} trip", "was the {p} drive out of the ordinary", "anything off about my {p} trip"]
ANOM_DAY_T = ["anything strange {d}", "was there anything odd {d}", "any anomalies {d}", "did anything weird happen {d}"]
USUAL_T = ["how does the {c} usually behave", "what is the normal range of the {c}", "what is a typical {c}", "what is the usual {c}", "how high does the {c} normally get", "what is normal for the {c}", "typical {c} for my car",
           "describe the usual {c}", "what does the {c} usually look like", "what values does the {c} normally take", "how does my {c} normally run"]
USUAL_EV_T = ["how often does {e} usually happen", "what is the usual number of {e}", "how many {e} do I normally get", "typical frequency of {e}", "how frequent are {e} normally", "what is a normal count of {e}"]
WHY_TRIP_T = {"rise": ["why was the {c} so high on my {p} trip", "what made the {c} so high on the {p} drive", "why did the {c} get so high on the {p} journey", "what pushed the {c} up on my {p} trip",
                       "why was the {c} up on the {p} drive"], "drop": ["why was the {c} so low on my {p} trip", "what made the {c} fall on the {p} drive", "why did the {c} drop on the {p} journey", "what pulled the {c} down on my {p} trip"],
              "none": ["what happened to the {c} on my {p} trip", "explain the {c} on the {p} drive", "what explains the {c} on my {p} journey", "tell me why the {c} looked like that on the {p} trip"]}
WHY_DAY_T = ["why was {d} so hard on the {c}", "what made the {c} so high {d}", "why did the {c} rise {d}", "what happened to the {c} {d}"]
SUM_TRIP_T = ["summarise my {p} trip", "tell me about the {p} drive", "how did my {p} journey go", "details of my {p} trip", "overview of the {p} drive", "walk me through my {p} trip", "recap the {p} drive"]
SUM_DAY_T = ["how was {d} for the car", "summarise {d}", "recap {d}'s driving", "what did the car do {d}", "tell me about {d} on the road"]
SUM_C_T = ["summarise the {c} {t}", "overview of the {c} {t}", "how has the {c} been {t}", "recap the {c} {t}", "give me a picture of the {c} {t}", "how did the {c} do {t}"]
PICKS = ["last", "latest", "longest", "shortest", "most recent", "first", "previous", "most efficient", "least efficient", "biggest"]
DAYS = ["yesterday", "today", "this morning", "last night", "yesterday evening"]
WHY_SIGNALS = ["battery_temperature", "motor_temperature", "wh_per_km", "fuel_rate", "engine_oil_temperature", "coolant_flow", "hvac_power", "battery_power", "speed", "tyre_pressure", "regen_power", "battery_soc", "range_estimate",
               "energy_used", "exhaust_gas_temperature", "engine_load", "boost_pressure", "brake_pressure", "battery_voltage", "cell_temperature_max"]
USUAL_SIGNALS = TYP_SIGNALS + ["engine_oil_temperature", "engine_load", "engine_rpm", "fuel_rate", "boost_pressure", "exhaust_gas_temperature", "throttle_position", "intake_manifold_pressure", "barometric_pressure", "engine_oil_pressure",
                               "battery_soc", "fuel_level", "charging_power", "tyre_temperature", "mass_air_flow", "intake_air_temperature", "lambda", "catalyst_temperature", "def_level", "dpf_soot_load"]
ANOM_SIGNALS = ["battery_temperature", "motor_temperature", "coolant_flow", "aux_battery_voltage", "tyre_pressure", "engine_oil_temperature", "fuel_rate", "battery_voltage", "charging_power", "speed", "battery_power", "hvac_power",
                "exhaust_gas_temperature", "boost_pressure", "engine_load", "regen_power", "ambient_temperature", "tyre_temperature", "engine_rpm", "intake_air_temperature"]
SUM_CONCEPTS = ["battery_temperature", "motor_temperature", "speed", "tyre_pressure", "battery_soc", "coolant_flow", "wh_per_km", "fuel_rate", "engine_oil_temperature", "any_warning", "hard_braking", "diagnostic_code", "charging_session",
                "adas_disengagement", "battery_condition", "cooling_condition", "tyre_condition", "motor_condition", "driving_style", "vehicle_health", "hard_acceleration", "tyre_pressure_warning", "energy_used", "distance_driven"]
TOPICS = ["battery_condition", "cooling_condition", "tyre_condition", "motor_condition", "brake_condition", "aux_battery_condition"]


def gen_digest(r, n_each):
    rows = []
    def add(src, text, f, prev=None): rows.append({"text": " ".join(text.split()), "frame": f, "prev": prev, "src": src, "group": src})
    t = lambda: r.choice([x for x in T.PERIODS if x and x not in ("the whole time", "since i got the car", "in the last hour", "this morning", "last night")])
    for _ in range(n_each["sum"]): add("v3_summary", r.choice(SUM_T).format(t=t()), F("summary", "none"))
    for _ in range(n_each["chg"]): add("v3_changes", r.choice(CHG_T).format(t=t()), F("summary", "none", "change"))
    for _ in range(n_each["anom"]): add("v3_anomaly", r.choice(ANOM_T).format(t=t()), F("anomaly", "none"))
    for _ in range(n_each["anomc"]):
        k = r.random()
        if k < 0.7: c = r.choice(ANOM_SIGNALS)
        elif k < 0.85: c = r.choice(TOPICS)
        else: c = r.choice(["any_warning", "diagnostic_code", "hard_braking", "tyre_pressure_warning"])
        add("v3_anomaly", r.choice(ANOM_C_T).format(c=r.choice(phrases(c)), t=t()), F("anomaly", c))
    for _ in range(n_each["anomt"]):
        if r.random() < 0.7: add("v3_anomaly", r.choice(ANOM_TRIP_T).format(p=r.choice(PICKS)), F("anomaly", "none", ent="trip"))
        else: add("v3_anomaly", r.choice(ANOM_DAY_T).format(d=r.choice(DAYS)), F("anomaly", "none", ent="day"))
    for _ in range(n_each["usual"]):
        if r.random() < 0.78: c = r.choice(USUAL_SIGNALS); add("v3_usual", r.choice(USUAL_T).format(c=r.choice(phrases(c))), F("stat", c, "usual"))
        else: e = r.choice(TYP_EVENTS); add("v3_usual", r.choice(USUAL_EV_T).format(e=r.choice(phrases(e))), F("stat", e, "usual"))
    for _ in range(n_each["why"]):
        c = r.choice(WHY_SIGNALS); d = r.choices(["rise", "drop", "none"], [6, 2, 2])[0]
        if r.random() < 0.85: add("v3_trip", r.choice(WHY_TRIP_T[d]).format(c=r.choice(phrases(c)), p=r.choice(PICKS)), F("why", c, d, ent="trip"))
        else: add("v3_trip", r.choice(WHY_DAY_T).format(c=r.choice(phrases(c)), d=r.choice(DAYS[:2])), F("why", c, "rise" if r.random() < 0.8 else "none", ent="day"))
    for _ in range(n_each["sumtrip"]):
        if r.random() < 0.7: add("v3_trip", r.choice(SUM_TRIP_T).format(p=r.choice(PICKS)), F("summary", "none", ent="trip"))
        else: add("v3_trip", r.choice(SUM_DAY_T).format(d=r.choice(DAYS)), F("summary", "none", ent="day"))
    for _ in range(n_each["sumc"]):
        c = r.choice(SUM_CONCEPTS); add("v3_summary", r.choice(SUM_C_T).format(c=r.choice(phrases(c)), t=t()), F("summary", c))
    # the universal catalogue: each quantity of the catalogue the registry did not have before, asked in every act it can be asked in
    cat = [d["name"] for d in json.load(open(os.path.join(ROOT, "vgpt", "ingest", "catalog.json"), encoding="utf-8"))["concepts"] if d["name"] in O.concepts]
    NOW_T = ["what is the {c}", "how is the {c}", "current {c}", "{c} right now", "what is my {c} at the moment", "tell me the {c}", "what does the {c} read"]
    for c in cat:
        for _ in range(n_each["cat"]):
            k = r.random(); ph = r.choice(phrases(c))
            if k < 0.34: add("v3_catalog", r.choice(NOW_T).format(c=ph), F("now", c))
            elif k < 0.50: add("v3_catalog", r.choice(["average {c} {t}", "mean {c} {t}", "what was the {c} on average {t}"]).format(c=ph, t=t()), F("stat", c, "mean"))
            elif k < 0.64: add("v3_catalog", r.choice(["highest {c} {t}", "peak {c} {t}", "max {c} {t}", "how high did the {c} get {t}"]).format(c=ph, t=t()), F("extreme", c, "max"))
            elif k < 0.74: add("v3_catalog", r.choice(["lowest {c} {t}", "minimum {c} {t}", "how low did the {c} get {t}"]).format(c=ph, t=t()), F("extreme", c, "min"))
            elif k < 0.84: add("v3_catalog", r.choice(["how has the {c} changed {t}", "trend of the {c} {t}", "is the {c} going up or down {t}"]).format(c=ph, t=t()), F("trend", c, "change"))
            elif k < 0.92: add("v3_catalog", r.choice(["what is {c}", "what does {c} mean", "explain {c}", "define {c}"]).format(c=ph), F("define", c))
            else: add("v3_catalog", r.choice(["what was the {c} {at}", "the {c} {at}"]).format(c=ph, at=r.choice(T.AT_POOL)), F("at", c))
    return rows


def gen_templates(r, n_each):
    rows = []
    def add(src, text, f, prev=None): rows.append({"text": " ".join(text.split()), "frame": f, "prev": prev, "src": src, "group": src})
    t = lambda: r.choice(T.PERIODS)
    for _ in range(n_each["corr"]):
        a, b = r.choice(PAIRS); a, b = (a, b) if r.random() < .5 else (b, a); add("v2_corr", r.choice(CORR_T).format(x=r.choice(phrases(a)), y=r.choice(phrases(b)), t=t()), F("stat", a, "corr", c2=b))
    for _ in range(n_each["two"]):
        a, b = r.choice(PAIRS); a, b = (a, b) if r.random() < .5 else (b, a); op = r.choices(["mean", "max", "min"], [5, 3, 2])[0]
        add("v2_two", r.choice(TWO_T[op]).format(x=r.choice(phrases(a)), y=r.choice(phrases(b)), t=t()), F("compare", a, op, c2=b))
    for _ in range(n_each["either"]):
        a, b = r.choice(EVENT_PAIRS); a, b = (a, b) if r.random() < .5 else (b, a); op = r.choices(["exists", "count"], [3, 2])[0]
        add("v2_either", r.choice(EITHER_T[op]).format(e=r.choice(phrases(a)), f=r.choice(phrases(b)), t=t()), F("events", a, op, c2=b))
    for _ in range(n_each["typical"]):
        k = r.random()
        if k < 0.50:
            x = r.choice(TYP_SIGNALS); add("v2_typical", r.choice(TYP_SIG_T).format(x=r.choice(phrases(x)), t=t()), F("compare", x, "mean", vs="typical"))
        elif k < 0.62:
            x = r.choice(TYP_SIGNALS); op = r.choice(["max", "min"]); add("v2_typical", r.choice(TYP_EXT_T[op]).format(x=r.choice(phrases(x)), t=t()), F("compare", x, op, vs="typical"))
        elif k < 0.85:
            e = r.choice(TYP_EVENTS); add("v2_typical", r.choice(TYP_EVT_T).format(e=r.choice(phrases(e)), t=t()), F("compare", e, "count", vs="typical"))
        else:
            x = r.choice(sorted(NOW_OK)); add("v2_typical", r.choice(TYP_NOW_T).format(x=r.choice(phrases(x))), F("now", x, vs="typical"))
    for _ in range(n_each["explain"]): add("v2_explain", r.choice(EXPL), F("explain", "none", ref="answer"))
    # follow-ups
    def prev_frame():
        act, op, cid = r.choice(PERIOD_PREV); return F(act, cid, op)
    for _ in range(n_each["period"]):
        p = prev_frame(); u = r.choice(PERIOD_U).format(t=r.choice([x for x in T.PERIODS if x]))
        act, op = (("stat", "mean" if O.concepts[p.concept].kind != "event" else "count") if p.act in ("now", "at") else (p.act, p.op))
        add("v2_period", u, F(act, p.concept, op, p.cond, p.vs, p.ent, ref="period"), p)
    for _ in range(n_each["concept"]):
        p = prev_frame(); kind = O.concepts[p.concept].kind; pool = [c for c in CONCEPT_SWAP.get("event" if kind == "event" else "derived" if kind == "derived" else "signal") if c != p.concept]
        if p.act == "events" and p.op == "list": pool = [c for c in CONCEPT_SWAP["event"] if c != p.concept]
        c = r.choice(pool); add("v2_concept", r.choice(CONCEPT_U).format(c=r.choice(phrases(c))), F(p.act, c, p.op, p.cond, p.vs, p.ent, ref="concept"), p)
    for _ in range(n_each["focus"]):
        e = r.choice(FOCUS_EVENTS); p = F("events", e, "last"); k = r.random()
        if k < 0.4: add("v2_focus", r.choice(FOCUS_WHY), F("why", e, ref="focus"), p)
        elif k < 0.6: add("v2_focus", r.choice(FOCUS_CTX), F("context", e, ref="focus"), p)
        else: c = r.choice(AT_SIGNALS); add("v2_focus", r.choice(FOCUS_AT).format(c=r.choice(phrases(c))), F("at", c, ref="focus"), p)
    for _ in range(n_each["item"]):
        e = r.choice(ITEM_EVENTS); o = r.choice(["max", "min", "first", "last", "last", "max"]); w = r.choice(ITEM_W[o]); p = F("events", e, "list")
        add("v2_item", r.choice(ITEM_U[o]).format(w=w), F("extreme" if o in ("max", "min") else "events", e, o, ref="item"), p)
    for _ in range(n_each["answer"]):
        p = prev_frame(); add("v2_answer", r.choice(EXPL), F("explain", "none", ref="answer"), p)
    return rows


def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


WRAPPERS = ["Quick one before I forget -- {q}. Cheers mate, you've been brilliant.", "{q}? Not urgent at all, I was just curious because a colleague mentioned it at lunch.",
            "Okay so last night we were talking about cars at dinner and now I'm wondering, {q}", "hi hi! {q} and also you're doing a great job today",
            "Sorry to bother you again but could you tell me {q} when you get a moment, thanks so much", "Hello there, I have a small request: {q}. That's all, take care!",
            "Namaste! {q} aur haan, bahut shukriya", "My neighbour was asking me about this earlier, so {q}, and thank you in advance", "{q} - I know you are busy so no rush at all",
            "Right, sorry, one more thing and then I'll leave you alone: {q}"]

ICE_ONLY = ("engine_", "fuel_", "dpf", "def_level", "egr_", "boost_", "throttle_", "gear", "intake_", "mass_air", "lambda", "catalyst", "exhaust_", "ignition_", "short_term", "long_term", "coolant_pressure")
DIESEL_ONLY = ("dpf", "def_level", "egr_", "boost_pressure")
PETROL_ONLY = ("ignition_timing", "lambda", "short_term_fuel_trim", "long_term_fuel_trim")
EV_ONLY = ("battery_soc", "battery_soh", "battery_capacity", "battery_condition", "battery_power", "battery_voltage", "battery_current", "battery_temperature", "battery_overheat", "battery_efficiency", "cell_", "regen_", "charging_", "charge_",
           "time_to_full", "inverter_", "motor_rpm", "motor_torque", "motor_power", "accelerator_position")
ICE_WORDS = {"petrol", "diesel", "fuel", "gearbox", "exhaust", "turbo", "dpf", "adblue", "refuel", "refuelling", "tank", "clutch", "gasoline", "litres", "liters", "mpg", "idling", "oil"}
EV_WORDS = {"kwh", "regen", "plug", "plugged", "charger", "charging", "charge", "soc", "soh", "wh", "ev"}


def pick_vehicle(row, rnd):
    """The car type a row is shown with.  A row about a concept only one kind of car has is shown with that kind; wording that names one kind decides; the rest is spread over all kinds, so that the model learns that most
    questions do not depend on it - and a few rows get no car type at all (an unknown vehicle)."""
    if row.get("vehicle"):
        v = row["vehicle"]
        return rnd.choice(["petrol", "petrol", "diesel"]) if v == "ice" else v
    f = Frame.from_text(row["answer"]); cs = [c for c in (f.concept, f.c2) if c not in ("none", "unlisted")]; words = set(norm_tokens(row["text"]))
    if any(c.startswith(DIESEL_ONLY) for c in cs): return "diesel"
    if any(c.startswith(PETROL_ONLY) for c in cs): return "petrol"
    if any(c.startswith(ICE_ONLY) for c in cs) and not any(c.startswith(EV_ONLY) for c in cs): return rnd.choice(["petrol", "petrol", "diesel"])
    if any(c.startswith(EV_ONLY) for c in cs) and not any(c.startswith(ICE_ONLY) for c in cs): return "ev"
    if words & ICE_WORDS and not words & EV_WORDS: return rnd.choice(["petrol", "petrol", "diesel"])
    if words & EV_WORDS and not words & ICE_WORDS: return "ev"
    x = rnd.random()
    return "ev" if x < 0.45 else "petrol" if x < 0.72 else "diesel" if x < 0.87 else "hybrid" if x < 0.90 else None


def assign_vehicles(part, rnd, hint_p):
    for x in part:
        if x.get("kind") != "frame": continue
        v = pick_vehicle(x, rnd) if (rnd.random() > 0.05 or x.get("vehicle")) else None            # 5 % of the rows are shown without a car type
        if v: x["vehicle"] = v
        else: x.pop("vehicle", None)
        if x.get("hint") or x.get("src", "").startswith("v10_"):
            h = G.hint_line(x["text"], vehicle=v)
            x["hint"] = h if (h and (x.get("hint") or rnd.random() < hint_p)) else ""


def token_sets(files):
    sets = []
    for f in files:
        p = os.path.join(ROOT, "bench", "meaning_eval", f)
        if not os.path.exists(p): continue
        for line in io.open(p, encoding="utf-8"):
            if line.startswith("#") or not line.strip(): continue
            parts = line.rstrip("\n").split("\t"); text = parts[2] if f.startswith("followup") else parts[1]; sets.append(frozenset(norm_tokens(text)))
    return sets


def leaks(text, sets):
    t = frozenset(norm_tokens(text))
    if not t: return False
    for e in sets:
        if t == e: return True
        inter = len(t & e)
        if inter and inter / len(t | e) >= 0.8: return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--noise", type=float, default=0.35); ap.add_argument("--hint-p", type=float, default=0.85)
    ap.add_argument("--natural-copies", type=int, default=3); ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of generated v2 rows")
    a = ap.parse_args(); r = random.Random(a.seed + 2); out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out); base = out + "_base"
    if not os.path.exists(os.path.join(base, "train.jsonl")):                                      # step 1: the full_v11 recipe, version 1 (six words)
        env = {k: v for k, v in os.environ.items() if k != "VGPT_FRAME"}
        cmd = [sys.executable, os.path.join(HERE, "build_meaning.py"), "--out", base, "--seed", str(a.seed), "--template-n", "11000", "--sources", FULL_V11, "--eval-leak", "v11", "--wrap-p", "0.25"]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
        if p.returncode: raise SystemExit("base build failed:\n" + p.stdout[-1500:] + p.stderr[-1500:])
    sets_all = token_sets(["slots_v2.tsv", "followup_v2.tsv", "digest_v1.tsv", "catalog_v1.tsv", "followup_v3.tsv", "basic_v2.tsv", "hist_v2.tsv"] + [f for f in ("novel_v1.tsv", "messy_v1.tsv", "user_v1.tsv", "sealed_v2.tsv", "sealed_v3.tsv", "hard_v1.tsv", "hard_v2.tsv", "hard_v3.tsv", "hard_v4.tsv", "hard_v5.tsv", "platform_v1.tsv", "platform_v2.tsv",
                                                                                         "adv_v1.tsv", "adv_v2.tsv", "adv_v3.tsv")] + [f for f in ("holdout_v10.tsv", "sealed_v4.tsv", "sealed_v5.tsv", "ice_v1.tsv", "trend_v1.tsv", "anomaly_v1.tsv", "anomaly_v2.tsv", "trend_v2.tsv", "sealed_v6.tsv", "sealed_v7.tsv", "wording_v1.tsv") if os.path.exists(os.path.join(ROOT, "bench", "meaning_eval", f))])
    new_sets = token_sets(["slots_v2.tsv", "followup_v2.tsv"])
    n = {k: int(v * a.scale) for k, v in dict(corr=170, two=110, either=90, typical=240, explain=60, period=240, concept=240, focus=130, item=90, answer=110).items()}
    raw = gen_templates(r, n)
    raw += gen_digest(r, {k: int(v * a.scale) for k, v in dict(sum=300, chg=200, anom=240, anomc=200, anomt=110, usual=260, why=260, sumtrip=120, sumc=190, cat=9).items()})
    for fname, src in (("natural_v2.txt", "v2_natural"), ("natural_v3.txt", "v3_natural"), ("natural_v4.txt", "v4_natural"), ("natural_v5.txt", "v5_natural"), ("natural_v6.txt", "v6_natural"), ("natural_v7.txt", "v7_natural"), ("natural_v8.txt", "v8_natural"), ("natural_v9.txt", "v9_natural"), ("natural_v10a.txt", "v10_label"), ("natural_v10b.txt", "v10_wording"), ("natural_v10_ice.txt", "v10_ice"), ("natural_v11a.txt", "v11_label"), ("natural_v11b.txt", "v11_more"), ("natural_v11c.txt", "v11_anom_trend"), ("natural_v11d.txt", "v11_misc"), ("natural_v12a.txt", "v12_fix"), ("natural_v12b.txt", "v12_gen"), ("natural_v13a.txt", "v13_fix")):
        if not os.path.exists(os.path.join(HERE, fname)): continue                                                   # natural_v4.txt: round 2 (2026-10-21), gaps found after the first frame-2 run began
        for x in parse_natural2(os.path.join(HERE, fname)):
            for i in range(a.natural_copies): raw.append(dict(x, src=src, group=src, copy=i))
    base_rows = {name: [json.loads(line) for line in io.open(os.path.join(base, name + ".jsonl"), encoding="utf-8")] for name in ("train", "val", "test_comp")}
    fresh_pool = [x for x in base_rows["train"] if x["kind"] == "frame" and x["src"] in ("template", "natural", "platform", "adv1", "adv2", "adv3", "adv4", "feedback") and Frame.from_text(x["answer"]).act not in ("ambiguous",)]
    for _ in range(int(220 * a.scale)):                                                          # an unrelated question after an answer: nothing is inherited
        b = r.choice(fresh_pool); pa, po, pc = r.choice(PERIOD_PREV); p = F(pa, pc, po)
        bf = Frame.from_text(b["answer"]); raw.append({"text": b["text"], "frame": F(bf.act, bf.concept, bf.op, bf.cond, bf.vs, bf.ent), "prev": p, "src": "v2_fresh", "group": "v2_fresh", "hint": b.get("hint", ""), "copy": 0, "keep_text": True})
    selfc = []                                                                                                      # a COMPLETE question about the digests (or a thank-you), asked after ANY answer, inherits nothing (round 2 inherited the topic of the previous turn)
    for fname in ("natural_v3.txt", "natural_v5.txt", "natural_v7.txt", "natural_v8.txt", "natural_c.txt"):
        if os.path.exists(os.path.join(HERE, fname)):
            try: selfc += [x for x in parse_natural2(os.path.join(HERE, fname)) if x["prev"] is None and x["frame"].ref == "none" and (x["frame"].act in ("summary", "anomaly", "social") or (x["frame"].act == "stat" and x["frame"].op == "usual"))]
            except ValueError: pass
    for x in selfc:
        for _ in range(3):
            pa, po, pc = r.choice(PERIOD_PREV); raw.append({"text": x["text"], "frame": x["frame"], "prev": F(pa, pc, po), "src": "v4_fresh", "group": "v4_fresh", "copy": 0, "keep_text": True})
    simple = [x for x in base_rows["train"] if x["kind"] == "frame" and x["src"] in ("template", "natural", "platform", "feedback") and len(x["text"].split()) <= 9 and "?" not in x["text"][:-1]
              and Frame.from_text(x["answer"]).act in ("now", "stat", "events", "estimate", "health", "extreme", "trend", "spec") and not x.get("prev")]
    for k in range(int(420 * a.scale)):                                       # a polite preamble or a thank-you around an ordinary question changes nothing about what is asked (the preambles differ from the ones of the frozen sets on purpose)
        b = r.choice(simple); bf = Frame.from_text(b["answer"]); q = b["text"].strip().rstrip("?.!").strip(); style = r.randrange(len(WRAPPERS))
        raw.append({"text": WRAPPERS[style].format(q=q), "frame": F(bf.act, bf.concept, bf.op, bf.cond, bf.vs, bf.ent), "prev": None, "src": "v10_wrap", "group": "v10_wrap", "hint": b.get("hint", ""), "copy": 0, "keep_text": True})
    rows, dropped, seen = [], collections.Counter(), set()
    for x in raw:
        noised = x.get("copy", 0) != 0 or x["src"] in ("v2_corr", "v2_two", "v2_either", "v2_typical", "v2_explain", "v2_period", "v2_concept", "v2_focus", "v2_item", "v2_answer", "v3_summary", "v3_changes", "v3_anomaly", "v3_usual", "v3_trip", "v3_catalog")
        text = x["text"] if x.get("keep_text") else NZ.apply(x["text"], r, a.noise) if noised else x["text"]
        if leaks(text, sets_all) or leaks(x["text"], sets_all): dropped["eval_overlap"] += 1; dropped["eval_overlap:" + x["src"]] += 1; continue
        pk = x["prev"].to_text(2) if x["prev"] else ""; key = (" ".join(norm_tokens(text)), x["frame"].to_text(2), pk)
        if key in seen: dropped["duplicate"] += 1; continue
        seen.add(key)
        issues = O.validate(x["frame"]) + (O.validate(x["prev"]) if x["prev"] else [])
        if issues: dropped["typing"] += 1; dropped["typing:" + x["src"]] += 1; continue
        hint = x["hint"] if "hint" in x else (G.hint_line(text) if r.random() < a.hint_p else "")
        row = {"kind": "frame", "text": text, "answer": x["frame"].to_text(2), "hint": hint, "src": x["src"], "group": x["group"]}
        if x.get("vehicle"): row["vehicle"] = x["vehicle"]
        if x["prev"]: row["prev"] = pk
        rows.append(row)
    train, val, comp = [], [], []
    for name, part in (("train", train), ("val", val), ("test_comp", comp)):
        for x in base_rows[name]:
            if x["kind"] == "frame": x = dict(x, answer=x["answer"] + " none none")
            part.append(x)
    for x in rows: (val if r.random() < 0.05 else train).append(x)
    tkey = lambda x: " ".join(x["text"].lower().split())                                      # train_lora.py refuses any wording that is in two splits, whatever its `prev` (a follow-up phrase seen with another previous frame counts)
    comp_keys = {tkey(x) for x in comp}; before = len(val); val = [x for x in val if tkey(x) not in comp_keys]; dropped["split_overlap_val"] += before - len(val); val_keys = {tkey(x) for x in val}
    before = len(train); train = [x for x in train if tkey(x) not in comp_keys and tkey(x) not in val_keys]; dropped["split_overlap"] += before - len(train)
    for part in (train, val, comp): assign_vehicles(part, r, a.hint_p)
    os.makedirs(out, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test_comp", comp)):
        with io.open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for i, x in enumerate(part): fh.write(json.dumps(dict(x, id=f"{name}{i:05d}"), ensure_ascii=False, sort_keys=True) + "\n")
    new = [x for x in train if str(x.get("src", "")).startswith(("v2_", "v3_"))]
    stats = {"seed": a.seed, "counts": {"train": len(train), "val": len(val), "test_comp": len(comp)}, "dropped": dict(dropped), "v2_rows_in_train": len(new), "v2_by_src": dict(collections.Counter(x["src"] for x in new)),
             "with_previous": sum(1 for x in train if x.get("prev")), "refs": dict(collections.Counter(x["answer"].split()[-1] for x in train if x["kind"] == "frame")), "c2_used": sum(1 for x in train if x["kind"] == "frame" and x["answer"].split()[-2] != "none"),
             "base_manifest_sha256": sha(os.path.join(base, "manifest.json")), "builder_sha256": sha(__file__), "natural_v2_sha256": sha(os.path.join(HERE, "natural_v2.txt")), "slots_v2_sha256": sha(os.path.join(ROOT, "bench", "meaning_eval", "slots_v2.tsv")),
             "followup_v2_sha256": sha(os.path.join(ROOT, "bench", "meaning_eval", "followup_v2.tsv")), "digest_v1_sha256": sha(os.path.join(ROOT, "bench", "meaning_eval", "digest_v1.tsv")), "catalog_v1_sha256": sha(os.path.join(ROOT, "bench", "meaning_eval", "catalog_v1.tsv")), "followup_v3_sha256": sha(os.path.join(ROOT, "bench", "meaning_eval", "followup_v3.tsv")), "natural_v3_sha256": sha(os.path.join(HERE, "natural_v3.txt")), "natural_v4_sha256": sha(os.path.join(HERE, "natural_v4.txt")) if os.path.exists(os.path.join(HERE, "natural_v4.txt")) else None, "ontology_sha256": sha(os.path.join(ROOT, "vgpt", "meaning", "ontology_data.json"))}
    json.dump(stats, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"), indent=1); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

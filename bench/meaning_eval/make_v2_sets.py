"""Write the two frame-version-2 evaluation sets and freeze them with a SHA-256.

    python bench/meaning_eval/make_v2_sets.py

slots_v2.tsv     single questions that need the new slots (a second quantity, "than usual", a question about the previous answer) + controls that must not use them:   id <TAB> text <TAB> gold (8 words) <TAB> tag
followup_v2.tsv  two-turn dialogues: the frame of the previous answered question, the utterance, the gold frame:                                                         id <TAB> previous (8 words) <TAB> text <TAB> gold (8 words) <TAB> tag

Authored 2026-10-20 after full_v11 was final, BEFORE any version-2 training data was written, so the training banks can be filtered against these sets (training/meaning/build_v2.py does that).
Never trained on.  The wording here is deliberately not the wording the training banks use."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ["VGPT_FRAME"] = "2"
from vgpt.meaning import ontology as ON                          # noqa: E402
from vgpt.meaning.frame import Frame                              # noqa: E402

O = ON.load(2)


def F(act, concept, op="none", cond="none", vs="none", ent="none", c2="none", ref="none"): return Frame(act, concept, op, cond, vs, ent, c2, ref)


# ----------------------------------------------------------------------------------------------------------------------------------------- single questions
SLOTS = []
def add(tag, text, f): SLOTS.append((tag, text, f))

# a correlation of two measured quantities: stat X corr, c2 = Y
for t, a, b in [("does my battery power depend on how fast I drive", "battery_power", "speed"), ("how closely does battery temperature follow the outside temperature", "battery_temperature", "ambient_temperature"),
                ("is there a link between motor temperature and speed", "motor_temperature", "speed"), ("do the heater power and the weather outside move together", "hvac_power", "ambient_temperature"),
                ("correlation between the accelerator and battery power", "battery_power", "accelerator_position"), ("is regeneration related to my speed", "regen_power", "speed"),
                ("does coolant flow follow the battery temperature", "coolant_flow", "battery_temperature"), ("do the tyre pressures change with the outside temperature", "tyre_pressure", "ambient_temperature"),
                ("how strongly are battery and motor temperature tied together", "battery_temperature", "motor_temperature"), ("does braking pressure scale with speed", "brake_pressure", "speed"),
                ("is cornering force connected to speed", "lateral_acceleration", "speed"), ("how well does motor power track battery power", "motor_power", "battery_power"),
                ("do voltage and current in the pack move in step", "battery_voltage", "battery_current"), ("is motor torque tied to how hard I press the pedal", "motor_torque", "accelerator_position"),
                ("do the hottest cell and the pack temperature agree", "cell_temperature_max", "battery_temperature"), ("is picture brightness related to the daylight", "camera_brightness", "light"),
                ("over the last month, does battery power correlate with speed", "battery_power", "speed"), ("any relationship between tyre temperature and speed this week", "tyre_temperature", "speed")]:
    add("corr", t, F("stat", a, "corr", c2=b))
# two quantities side by side
for t, a, b, op in [("put battery and motor temperature next to each other", "battery_temperature", "motor_temperature", "mean"), ("battery against motor temperature last week", "battery_temperature", "motor_temperature", "mean"),
                    ("show me speed and battery power together for the month", "speed", "battery_power", "mean"), ("which ran hotter this week, the pack or the motor", "battery_temperature", "motor_temperature", "max"),
                    ("tyre pressure and outside temperature side by side", "tyre_pressure", "ambient_temperature", "mean"), ("highest battery power and highest motor power this week", "battery_power", "motor_power", "max"),
                    ("lowest battery voltage next to the lowest 12 volt voltage", "battery_voltage", "aux_battery_voltage", "min"), ("heater power versus motor power on average", "hvac_power", "motor_power", "mean"),
                    ("how do coolant flow and battery temperature compare over the last day", "coolant_flow", "battery_temperature", "mean"), ("camera sharpness and camera brightness, last week", "camera_sharpness", "camera_brightness", "mean"),
                    ("set regen power beside battery power for the past three days", "regen_power", "battery_power", "mean"), ("which got lower this month, motor rpm or speed", "motor_rpm", "speed", "min")]:
    add("two", t, F("compare", a, op, c2=b))
# one of two events
for t, a, b, op in [("did the pack or the motor overheat last month", "battery_overheat_warning", "overtemp_warning", "exists"), ("any tyre warnings or camera problems this week", "tyre_pressure_warning", "camera_degraded", "exists"),
                    ("was there a hard brake or a hard acceleration yesterday", "hard_braking", "hard_acceleration", "exists"), ("how many ECU resets or unexpected shutdowns this month", "ecu_reset", "unexpected_shutdown", "count"),
                    ("did the camera get blocked or just degraded this week", "camera_blocked", "camera_degraded", "exists"), ("12 volt warnings or tyre warnings in the last week", "low_12v_warning", "tyre_pressure_warning", "exists"),
                    ("how often did a voltage dip or a power derate happen", "voltage_dip", "power_derate", "count"), ("did lane keeping drop out or did emergency braking trigger today", "adas_disengagement", "aeb_activation", "exists")]:
    add("either", t, F("events", a, op, c2=b))
# than usual: the vehicle's own history is the yardstick
for t, f in [("is my battery temperature higher than it usually is", F("compare", "battery_temperature", "mean", vs="typical")), ("has the motor been running hotter than normal this week", F("compare", "motor_temperature", "mean", vs="typical")),
             ("was my speed unusual last week", F("compare", "speed", "mean", vs="typical")), ("is this week's energy use per km typical", F("compare", "wh_per_km", "mean", vs="typical")),
             ("did I brake hard more often than normal this week", F("compare", "hard_braking", "count", vs="typical")), ("is my tyre pressure lower than what it normally is", F("compare", "tyre_pressure", "mean", vs="typical")),
             ("did I charge more than usual this month", F("compare", "charging_session", "count", vs="typical")), ("is the cabin heater drawing more than it typically does", F("compare", "hvac_power", "mean", vs="typical")),
             ("are there more warnings than I would expect for this car", F("compare", "any_warning", "count", vs="typical")), ("was yesterday's top speed out of the ordinary", F("compare", "speed", "max", vs="typical")),
             ("is the coolant flow lower than its usual level", F("compare", "coolant_flow", "mean", vs="typical")), ("is the battery voltage in its normal range lately", F("compare", "battery_voltage", "mean", vs="typical")),
             ("is the battery temperature normal right now", F("now", "battery_temperature", vs="typical")), ("is the motor temperature typical at the moment", F("now", "motor_temperature", vs="typical")),
             ("is the highest battery temperature this week normal for me", F("compare", "battery_temperature", "max", vs="typical")), ("did regeneration give less than it usually does last week", F("compare", "regen_energy", "sum", vs="typical")),
             ("have I driven further than I normally do this week", F("compare", "distance_driven", "sum", vs="typical")), ("are ADAS disengagements above their usual rate", F("compare", "adas_disengagement", "count", vs="typical"))]:
    add("typical", t, f)
# a question about the previous answer itself
for t in ["where did that number come from", "what is that figure based on", "walk me through how you worked that out", "how sure are you about that", "which data did you look at for that", "can you show how you got it", "explain what you just told me", "what was the evidence for that answer"]:
    add("explain", t, F("explain", "none", ref="answer"))
# controls: ordinary single questions that must keep using none of the new slots
for t, f in [("how fast am I going", F("now", "speed")), ("what is the battery temperature", F("now", "battery_temperature")), ("average motor temperature last week", F("stat", "motor_temperature", "mean")),
             ("how many hard brakes this week", F("events", "hard_braking", "count")), ("highest speed yesterday", F("extreme", "speed", "max")), ("compare this week with last week for energy per km", F("compare", "wh_per_km", "mean", vs="period")),
             ("why did the range drop", F("why", "range_estimate", "drop")), ("when did I last charge", F("events", "charging_session", "last")), ("what is a regen", F("define", "regen_power")), ("hello", F("social", "none")),
             ("what is the tyre pressure", F("now", "tyre_pressure")), ("is my car okay", F("health", "vehicle_health")), ("how much does the battery hold", F("spec", "battery_capacity")), ("what happened before the last warning", F("context", "any_warning")),
             ("average speed on the highway", F("stat", "speed", "mean", cond="highway")), ("turn on the heater", F("control", "none")), ("what is the weather tomorrow", F("unsupported", "none")),
             ("how does the pack temperature trend this month", F("trend", "battery_temperature", "change")), ("did the battery overheat", F("events", "battery_overheat_warning", "exists")), ("tyre pressure last night", F("at", "tyre_pressure"))]:
    add("control", t, f)

# ----------------------------------------------------------------------------------------------------------------------------------------- two-turn dialogues
FU = []
def fu(tag, prev, text, gold): FU.append((tag, prev, text, gold))

P_EVENTS = F("events", "hard_braking", "count"); P_SPEED = F("stat", "speed", "mean"); P_BATT = F("extreme", "battery_temperature", "max"); P_ENERGY = F("stat", "wh_per_km", "mean"); P_DISENG = F("events", "adas_disengagement", "count")
P_MOTOR = F("trend", "motor_temperature", "change"); P_NOW = F("now", "speed"); P_CHG = F("events", "charging_session", "list"); P_TYRE = F("stat", "tyre_pressure", "mean")
# the period changes
for prev, text in [(P_EVENTS, "same again for the week before"), (P_EVENTS, "now do the month before that"), (P_SPEED, "how about the day before yesterday"), (P_SPEED, "make it the past three days"), (P_BATT, "only for this morning"),
                   (P_BATT, "go back two weeks instead"), (P_ENERGY, "same question for last month"), (P_ENERGY, "and over the whole of September"), (P_DISENG, "since Monday"), (P_DISENG, "what about the weekend"),
                   (P_MOTOR, "this time over the last fortnight"), (P_MOTOR, "restrict that to today"), (P_TYRE, "and a week earlier"), (P_TYRE, "repeat it for the last hour"), (P_CHG, "and the charges before those"),
                   (P_CHG, "for the last two months instead"), (P_SPEED, "what was it on Tuesday"), (P_EVENTS, "just yesterday please")]:
    fu("period", prev, text, F(prev.act, prev.concept, prev.op, prev.cond, prev.vs, prev.ent, ref="period"))
# only the quantity changes
for prev, text, new in [(P_SPEED, "what about the motor temperature", "motor_temperature"), (P_SPEED, "and the battery power", "battery_power"), (P_BATT, "same for the motor", "motor_temperature"),
                        (P_BATT, "how about the cell temperature", "cell_temperature_max"), (P_ENERGY, "and the energy used", "energy_used"), (P_ENERGY, "same thing for regen energy", "regen_energy"),
                        (P_TYRE, "do the ambient temperature too", "ambient_temperature"), (P_TYRE, "and the 12 volt battery", "aux_battery_voltage"), (P_MOTOR, "let me see the battery instead", "battery_temperature"),
                        (P_MOTOR, "what about speed", "speed"), (P_EVENTS, "and the hard accelerations", "hard_acceleration"), (P_EVENTS, "how about tyre warnings", "tyre_pressure_warning"),
                        (P_DISENG, "and the camera blocked events", "camera_blocked"), (P_DISENG, "what about voltage dips", "voltage_dip"), (F("now", "battery_temperature"), "and the motor", "motor_temperature"),
                        (P_NOW, "and my battery power", "battery_power"), (P_CHG, "what about the hard brakes", "hard_braking"), (F("stat", "hvac_power", "mean"), "now the coolant flow", "coolant_flow")]:
    fu("concept", prev, text, F(prev.act, new, prev.op, prev.cond, prev.vs, prev.ent, ref="concept"))
# it / that moment: the instant or event the previous answer picked out
for prev, text, gold in [(F("events", "battery_overheat_warning", "last"), "why did it happen", F("why", "battery_overheat_warning", ref="focus")), (F("events", "battery_overheat_warning", "last"), "how fast was I going then", F("at", "speed", ref="focus")),
                         (F("events", "hard_braking", "last"), "what was the speed at that moment", F("at", "speed", ref="focus")), (F("events", "hard_braking", "last"), "what led up to it", F("context", "hard_braking", ref="focus")),
                         (F("events", "adas_disengagement", "last"), "what caused that", F("why", "adas_disengagement", ref="focus")), (F("events", "adas_disengagement", "last"), "and the battery level back then", F("at", "battery_soc", ref="focus")),
                         (F("events", "tyre_pressure_warning", "last"), "and why was that", F("why", "tyre_pressure_warning", ref="focus")), (F("events", "tyre_pressure_warning", "last"), "what was the outside temperature at the time", F("at", "ambient_temperature", ref="focus")),
                         (F("events", "diagnostic_code", "last"), "what was going on just before", F("context", "diagnostic_code", ref="focus")), (F("events", "camera_blocked", "last"), "what explains it", F("why", "camera_blocked", ref="focus")),
                         (F("events", "voltage_dip", "last"), "how hot was the motor then", F("at", "motor_temperature", ref="focus")), (F("events", "ecu_reset", "last"), "why did that occur", F("why", "ecu_reset", ref="focus"))]:
    fu("focus", prev, text, gold)
# one item out of the list just shown
for prev, text, gold in [(F("events", "hard_braking", "list"), "which of those was the strongest", F("extreme", "hard_braking", "max", ref="item")), (F("events", "hard_braking", "list"), "the latest one", F("events", "hard_braking", "last", ref="item")),
                         (F("events", "hard_braking", "list"), "and the first of them", F("events", "hard_braking", "first", ref="item")), (F("events", "adas_disengagement", "list"), "the most recent of these", F("events", "adas_disengagement", "last", ref="item")),
                         (F("events", "charging_session", "list"), "which one delivered the most", F("extreme", "charging_session", "max", ref="item")), (F("events", "charging_session", "list"), "the earliest", F("events", "charging_session", "first", ref="item")),
                         (F("events", "tyre_pressure_warning", "list"), "show me the last of them", F("events", "tyre_pressure_warning", "last", ref="item")), (F("events", "battery_overheat_warning", "list"), "which was the hottest", F("extreme", "battery_overheat_warning", "max", ref="item")),
                         (F("events", "camera_blocked", "list"), "the one that lasted longest", F("extreme", "camera_blocked", "max", ref="item")), (F("events", "diagnostic_code", "list"), "which came first", F("events", "diagnostic_code", "first", ref="item"))]:
    fu("item", prev, text, gold)
# a question about the answer just given
for prev, text in [(P_EVENTS, "how did you get that"), (P_SPEED, "what is that based on"), (P_BATT, "how sure are you of that number"), (F("events", "battery_overheat_warning", "last"), "show me how you worked that out"),
                   (P_ENERGY, "which records did that come from"), (P_CHG, "explain how you arrived at this")]:
    fu("answer", prev, text, F("explain", "none", ref="answer"))
# a new, unrelated question after an answer: nothing is inherited
for prev, text, gold in [(P_EVENTS, "how fast am I going", F("now", "speed")), (P_SPEED, "is my car okay", F("health", "vehicle_health")), (P_BATT, "what is the tyre pressure", F("now", "tyre_pressure")),
                         (P_ENERGY, "how many hard brakes this week", F("events", "hard_braking", "count")), (P_DISENG, "when did I last charge", F("events", "charging_session", "last")), (P_MOTOR, "thank you", F("social", "none")),
                         (P_TYRE, "what is regen", F("define", "regen_power")), (P_CHG, "average speed last week", F("stat", "speed", "mean"))]:
    fu("fresh", prev, text, gold)


def write():
    bad = []
    with open(os.path.join(HERE, "slots_v2.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# SLOTS_V2: single questions for frame version 2 (second quantity c2, vs=typical, act explain) + controls.  Authored 2026-10-20 after full_v11 was final, before any version-2 training data.  Never trained on.\n# Format: id <TAB> text <TAB> gold frame (act concept op cond vs ent c2 ref) <TAB> tag\n")
        for i, (tag, text, fr) in enumerate(SLOTS, 1):
            iss = O.validate(fr)
            if iss: bad.append((text, fr.to_text(2), iss))
            f.write(f"S{i:03d}\t{text}\t{fr.to_text(2)}\t{tag}\n")
    with open(os.path.join(HERE, "followup_v2.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# FOLLOWUP_V2: two-turn dialogues for frame version 2.  Authored 2026-10-20 after full_v11 was final, before any version-2 training data.  Never trained on.\n# Format: id <TAB> previous frame (8 words) <TAB> text <TAB> gold frame (8 words) <TAB> tag\n")
        for i, (tag, prev, text, gold) in enumerate(FU, 1):
            for nm, fr in (("prev", prev), ("gold", gold)):
                iss = O.validate(fr)
                if iss: bad.append((text + " [" + nm + "]", fr.to_text(2), iss))
            f.write(f"U{i:03d}\t{prev.to_text(2)}\t{text}\t{gold.to_text(2)}\t{tag}\n")
    for n in ("slots_v2", "followup_v2"):
        p = os.path.join(HERE, n + ".tsv"); h = hashlib.sha256(open(p, "rb").read()).hexdigest(); open(os.path.join(HERE, n + ".sha256"), "w").write(h + "  " + n + ".tsv\n"); print(n, h[:12])
    import collections
    print("slots:", len(SLOTS), dict(collections.Counter(t for t, _, _ in SLOTS))); print("followups:", len(FU), dict(collections.Counter(t for t, *_ in FU)))
    print("typing problems:", len(bad)); [print("  ", b) for b in bad]


if __name__ == "__main__":
    write()

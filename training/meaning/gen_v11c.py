"""Round 11, file c: generated anomaly / trend wording families, written to natural_v11c.txt (the builder reads it like the hand-written files).

    python training/meaning/gen_v11c.py

Why: the two sealed sets anomaly_v1 and trend_v1 (the owner's concern: "not up to basic standards") failed on WORDING, not on the executors - "is X unusual / behaving normally" was read as a plain statistic or a comparison with typical,
and the direction words of a trend (rarer, on the rise, more often, degrading) were read as "change".  Each pattern is taught over many different things, so the model learns the pattern and not the thing.
HELD OUT concepts (never appear in this file; bench/meaning_eval/anomaly_v2.tsv and trend_v2.tsv use them): see HELD_ANOMALY and HELD_TREND.  The sealed rows themselves are never copied (the builder's leak filter also checks)."""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
rnd = random.Random(1124)

# concept -> noun phrases a person would use (the thing itself, singular, "the" is added by the template)
SIGNALS = {
    "battery_temperature": ["pack temperature", "battery temperature", "battery heat"], "motor_temperature": ["motor temperature", "motor heat"], "tyre_pressure": ["tyre pressure", "tire pressure", "air in the tyres"],
    "battery_voltage": ["pack voltage", "battery voltage"], "battery_current": ["battery current", "current draw"], "battery_power": ["battery power", "power draw"], "motor_power": ["motor power"], "motor_torque": ["motor torque", "torque"],
    "regen_energy": ["regen", "regenerated energy"], "regen_power": ["regen power", "regenerative braking"], "energy_used": ["energy use", "energy consumption"], "wh_per_km": ["consumption per km", "efficiency", "energy per kilometre"],
    "range_estimate": ["range estimate", "range", "remaining range"], "battery_soc": ["charge level", "state of charge", "battery percentage"], "coolant_flow": ["coolant flow", "pump flow"], "ambient_temperature": ["outside temperature", "ambient temperature"],
    "aux_battery_voltage": ["12 volt battery", "12v supply voltage"], "wheel_speed": ["wheel speeds"], "brake_pressure": ["brake pressure"], "speed": ["vehicle speed", "speed"], "fuel_level": ["fuel level", "tank level"],
    "fuel_rate": ["fuel consumption", "fuel burn"], "engine_oil_temperature": ["oil temperature"], "engine_oil_pressure": ["oil pressure"], "engine_rpm": ["engine revs", "engine rpm"], "engine_load": ["engine load"],
    "intake_air_temperature": ["intake air temperature"], "exhaust_gas_temperature": ["exhaust temperature"], "charging_power": ["charging power", "charge rate"], "hvac_power": ["climate control power", "air conditioning draw"],
    "tyre_temperature": ["tyre temperature"], "cell_temperature_max": ["hottest cell"], "cell_voltage_min": ["weakest cell voltage"], "boost_pressure": ["boost pressure"], "def_level": ["adblue level"], "dpf_soot_load": ["soot load", "dpf soot level"],
    "steering_angle": ["steering angle"], "accelerator_position": ["throttle position", "pedal position"], "lambda": ["lambda"], "short_term_fuel_trim": ["fuel trim"], "inverter_temperature": ["inverter temperature"],
}
# components: the "<x>_condition" concepts used by "anything strange with X"
COMPONENTS = {"battery_condition": ["battery", "traction battery", "pack"], "motor_condition": ["motor", "drive unit"], "cooling_condition": ["cooling system", "cooling circuit", "cooling"], "brake_condition": ["brakes", "braking system"],
              "tyre_condition": ["tyres", "tires"], "aux_battery_condition": ["12 volt battery", "small battery"]}
# plural countable events for trends
EVENTS = {"hard_braking": ["hard brakes", "hard braking events", "sudden stops"], "hard_acceleration": ["hard accelerations", "flat-out acceleration events"], "diagnostic_code": ["fault codes", "diagnostic codes", "trouble codes"],
          "any_warning": ["warnings", "warning lights", "alerts"], "voltage_dip": ["voltage dips", "brown-outs"], "unexpected_shutdown": ["unexpected shutdowns", "random shutdowns"], "hard_cornering": ["harsh corners", "hard cornering events"],
          "adas_disengagement": ["lane assist drop-outs", "adas disengagements"]}
HELD_ANOMALY = {"brake_pressure", "wheel_speed", "engine_oil_temperature", "intake_air_temperature", "exhaust_gas_temperature", "charging_power", "tyre_temperature", "hvac_power", "battery_current", "engine_rpm", "tyre_condition"}
HELD_TREND = {"speed", "hvac_power", "tyre_temperature", "charging_power", "adas_disengagement", "hard_cornering", "aux_battery_voltage", "fuel_rate", "engine_oil_pressure", "def_level"}

ANOMALY_SIG = ["is the {n} unusual", "is the {n} behaving normally", "is the {n} acting normal", "does the {n} look abnormal", "is the {n} out of the ordinary", "anything weird about the {n}", "any irregularities in the {n}",
               "were there any anomalies in the {n}", "has the {n} been off lately", "is the {n} normal for this car", "did the {n} do anything strange", "is the {n} unusually high", "is the {n} unusually low", "does the {n} seem odd",
               "{n} looking strange", "is something wrong with how the {n} behaves", "is the {n} abnormal", "has the {n} been erratic", "anything unexpected in the {n}"]
ANOMALY_COMP = ["anything strange with the {n}", "is the {n} acting strange", "any odd behaviour from the {n}", "is the {n} behaving normally", "does the {n} seem off", "is the {n} unusual in any way", "any irregularities with the {n}",
                "anything abnormal about the {n}", "has the {n} been acting weird", "is the {n} doing anything unexpected"]
TREND_DROP_SIG = ["is the {n} going down", "is the {n} falling", "is the {n} declining", "has the {n} been dropping", "is the {n} shrinking", "is the {n} getting lower over time", "has the {n} decreased lately", "is the {n} trending down"]
TREND_RISE_SIG = ["is the {n} going up", "is the {n} climbing", "is the {n} creeping up", "has the {n} been rising", "is the {n} growing", "is the {n} getting higher over time", "has the {n} increased lately", "is the {n} trending up"]
TREND_CHG_SIG = ["how has the {n} changed", "how did the {n} evolve", "what is the trend in the {n}", "how has the {n} moved over time", "has the {n} changed lately", "show me how the {n} has been trending"]
TREND_DROP_EV = ["are {n} becoming rarer", "are {n} less frequent now", "are {n} dropping off", "do {n} happen less often than before", "are there fewer {n} lately", "are {n} decreasing"]
TREND_RISE_EV = ["are {n} on the rise", "are {n} showing up more often", "are {n} becoming more frequent", "do {n} happen more often than before", "are there more {n} lately", "are {n} increasing"]
TREND_CHG_EV = ["how have {n} changed over time", "what is the trend in {n}", "how did {n} evolve", "have {n} changed lately"]
# battery ageing: always trend battery_soh drop
SOH = ["is the battery degrading", "is the battery wearing out", "is my pack losing capacity", "is the battery getting weaker", "is the battery ageing badly", "is the battery capacity shrinking",
       "is the battery health going down", "has the battery deteriorated", "is my battery losing its health", "is the pack getting worse with age"]
# range in a season
RANGE = ["is my range shrinking in the {s} months", "does the range drop in the {s}", "is the range getting shorter in {s}", "is the range lower when it is {s}"]
EFF_IMPROVE = ["did my {n} improve after {e}", "is my {n} better since {e}", "has my {n} got better after {e}"]


def pick(lst, k): return rnd.sample(lst, min(k, len(lst)))


def main():
    out = ["# Round 11, file c (generated by gen_v11c.py, 2026-10-24): anomaly and trend wording taught over many different things (held-out concepts: see the generator).", ""]
    seen = set()
    def add(text, spec):
        t = " ".join(text.split())
        if t in seen: return
        seen.add(t); out.append(f"{t} | {spec}")
    out.append("# ---- anomaly over a signal")
    for c, nouns in SIGNALS.items():
        if c in HELD_ANOMALY: continue
        for tpl in pick(ANOMALY_SIG, 7): add(tpl.format(n=rnd.choice(nouns)), f"anomaly {c}")
    out.append("# ---- anomaly over a component")
    for c, nouns in COMPONENTS.items():
        if c in HELD_ANOMALY: continue
        for tpl in pick(ANOMALY_COMP, 7): add(tpl.format(n=rnd.choice(nouns)), f"anomaly {c}")
    out.append("# ---- trend: a signal going down / up / changing")
    for c, nouns in SIGNALS.items():
        if c in HELD_TREND or c in ("steering_angle", "lambda", "short_term_fuel_trim", "wheel_speed", "brake_pressure"): continue
        for tpl in pick(TREND_DROP_SIG, 2): add(tpl.format(n=rnd.choice(nouns)), f"trend {c} op=drop")
        for tpl in pick(TREND_RISE_SIG, 2): add(tpl.format(n=rnd.choice(nouns)), f"trend {c} op=rise")
        for tpl in pick(TREND_CHG_SIG, 2): add(tpl.format(n=rnd.choice(nouns)), f"trend {c} op=change")
    out.append("# ---- trend: events becoming rarer / more frequent")
    for c, nouns in EVENTS.items():
        if c in HELD_TREND: continue
        for tpl in pick(TREND_DROP_EV, 5): add(tpl.format(n=rnd.choice(nouns)), f"trend {c} op=drop")
        for tpl in pick(TREND_RISE_EV, 5): add(tpl.format(n=rnd.choice(nouns)), f"trend {c} op=rise")
        for tpl in pick(TREND_CHG_EV, 2): add(tpl.format(n=rnd.choice(nouns)), f"trend {c} op=change")
    out.append("# ---- battery ageing, range in a season, efficiency after a change")
    for t in SOH: add(t, "trend battery_soh op=drop")
    for s in ("cold", "winter", "hot weather", "summer"):
        for tpl in pick(RANGE, 2): add(tpl.format(s=s), f"trend range_estimate op=drop cond={'cold' if s in ('cold', 'winter') else 'hot_weather'}")
    for e in ("changing the tyres", "the service", "the software update", "I started driving slower"):
        for tpl in EFF_IMPROVE: add(tpl.format(n=rnd.choice(["energy use per km", "consumption", "efficiency"]), e=e), "trend wh_per_km op=drop")
    p = os.path.join(HERE, "natural_v11c.txt"); open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("rows", len(seen), "->", p)


if __name__ == "__main__":
    main()

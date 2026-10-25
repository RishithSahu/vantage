"""Writes the frozen evaluation sets of the extended frame version 2 (2026-10-21): digest_v1.tsv (summary / changes / anomaly / usual / trip focus), catalog_v1.tsv (the universal-catalogue vocabulary, the same gold
frame on every vehicle) and followup_v3.tsv (follow-ups to the new acts).  Written BEFORE the training data of these families was built; training rows that overlap them are dropped (training/meaning/build_v2.py).

    python bench/meaning_eval/make_v3_sets.py          # writes the three .tsv files and their .sha256 once; refuses to overwrite
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT); os.environ["VGPT_FRAME"] = "2"
from vgpt.meaning import ontology as ON  # noqa: E402
from vgpt.meaning.frame import Frame  # noqa: E402


def spec(s):
    toks = s.split(); kw = dict(op="none", cond="none", vs="none", ent="none", c2="none", ref="none")
    for t in toks[2:]: k, v = t.split("="); kw[k] = v
    return Frame(toks[0], toks[1], **kw)


DIGEST = [
    # summary of a period
    ("what has my car been through this week", "summary none", "summary"), ("walk me through the past seven days of driving", "summary none", "summary"), ("brief me on the month", "summary none", "summary"),
    ("how did I use the vehicle lately", "summary none", "summary"), ("a snapshot of the last ten days", "summary none", "summary"), ("tell me what the car has been up to", "summary none", "summary"),
    ("sum up my driving since the start of the month", "summary none", "summary"), ("quick recap of the past fortnight", "summary none", "summary"), ("give me the story of this week", "summary none", "summary"),
    ("how have my last few weeks looked", "summary none", "summary"), ("report on my recent journeys", "summary none", "summary"), ("fill me in on the week", "summary none", "summary"),
    ("what is the overall picture for the past month", "summary none", "summary"), ("describe how the car behaved over the weekend", "summary none", "summary"), ("so how was last week", "summary none", "summary"),
    ("give me an account of the last few days", "summary none", "summary"), ("headline numbers for my driving this month please", "summary none", "summary"), ("what should I take away from the last seven days", "summary none", "summary"),
    ("run through the recent period for me", "summary none", "summary"), ("an overview of everything that happened this week", "summary none", "summary"),
    # what changed
    ("what is different about the car versus last month", "summary none op=change", "changes"), ("did anything shift this week compared with last", "summary none op=change", "changes"),
    ("tell me what has moved lately", "summary none op=change", "changes"), ("what has gone up or down recently", "summary none op=change", "changes"), ("anything that has changed since the week before", "summary none op=change", "changes"),
    ("which readings are not what they were last month", "summary none op=change", "changes"), ("how is this fortnight different from the previous one", "summary none op=change", "changes"),
    ("compare the whole picture of this week with the last and tell me what changed", "summary none op=change", "changes"), ("what has the car started doing differently", "summary none op=change", "changes"),
    ("highlight the differences since last week", "summary none op=change", "changes"), ("what's moved in my numbers", "summary none op=change", "changes"), ("were there any notable changes this month", "summary none op=change", "changes"),
    # anomaly
    ("is there anything unusual going on with the car", "anomaly none", "anomaly"), ("did the vehicle do anything unexpected this week", "anomaly none", "anomaly"), ("check my data for anything fishy", "anomaly none", "anomaly"),
    ("any signals acting up lately", "anomaly none", "anomaly"), ("what stands out as strange in the last month", "anomaly none", "anomaly"), ("were there any peculiar readings", "anomaly none", "anomaly"),
    ("look through my recent data for anything abnormal", "anomaly none", "anomaly"), ("tell me about any unexpected spikes", "anomaly none", "anomaly"), ("anything that looks like a fault in the data lately", "anomaly none", "anomaly"),
    ("do you see any outliers this week", "anomaly none", "anomaly"), ("anything weird yesterday", "anomaly none ent=day", "anomaly"), ("did my longest drive have anything strange in it", "anomaly none ent=trip", "anomaly"),
    ("was the last journey normal or odd", "anomaly none ent=trip", "anomaly"), ("anything strange in the motor temperature this month", "anomaly motor_temperature", "anomaly"),
    ("did the battery temperature do anything odd this week", "anomaly battery_temperature", "anomaly"), ("look for unusual tyre pressure readings", "anomaly tyre_pressure", "anomaly"),
    ("has the coolant flow behaved strangely", "anomaly coolant_flow", "anomaly"), ("is the engine oil temperature showing anything odd", "anomaly engine_oil_temperature", "anomaly"),
    ("any oddities in the 12 volt battery readings", "anomaly aux_battery_voltage", "anomaly"), ("is anything wrong with the way the battery is behaving", "anomaly battery_condition", "anomaly"),
    ("anything off about the tyres lately", "anomaly tyre_condition", "anomaly"), ("did the warnings look unusual this week", "anomaly any_warning", "anomaly"),
    # usual
    ("what is the normal operating range of the pack temperature", "stat battery_temperature op=usual", "usual"), ("how does the motor temperature typically run", "stat motor_temperature op=usual", "usual"),
    ("what speed do I usually drive at", "stat speed op=usual", "usual"), ("what are normal values for the coolant flow", "stat coolant_flow op=usual", "usual"), ("typical tyre pressure for my car", "stat tyre_pressure op=usual", "usual"),
    ("how much does the battery voltage normally vary", "stat battery_voltage op=usual", "usual"), ("what is the usual engine oil temperature when driving", "stat engine_oil_temperature op=usual", "usual"),
    ("how much fuel does the engine normally burn per hour", "stat fuel_rate op=usual", "usual"), ("normal climate control power draw", "stat hvac_power op=usual", "usual"), ("what is typical for the engine load", "stat engine_load op=usual", "usual"),
    ("how often does hard braking normally happen to me", "stat hard_braking op=usual", "usual"), ("what is the typical number of warnings per week", "stat any_warning op=usual", "usual"),
    ("what range does my 12 volt battery usually sit in", "stat aux_battery_voltage op=usual", "usual"), ("normal outside temperature where I drive", "stat ambient_temperature op=usual", "usual"),
    # trip focus
    ("why was the pack so warm on the longest drive", "why battery_temperature op=rise ent=trip", "trip"), ("what made the last journey so energy hungry", "why wh_per_km op=rise ent=trip", "trip"),
    ("why did the motor get hot on my most recent trip", "why motor_temperature op=rise ent=trip", "trip"), ("what explains the high fuel burn on the latest drive", "why fuel_rate op=rise ent=trip", "trip"),
    ("why was the oil so hot on that big trip", "why engine_oil_temperature op=rise ent=trip", "trip"), ("why did the range collapse during the last trip", "why range_estimate op=drop ent=trip", "trip"),
    ("why was the coolant flow low on the previous drive", "why coolant_flow op=drop ent=trip", "trip"), ("why was yesterday so tough on the battery", "why battery_temperature op=rise ent=day", "trip"),
    ("tell me how the longest trip went", "summary none ent=trip", "trip"), ("go over my last drive", "summary none ent=trip", "trip"), ("what was special about my shortest journey", "summary none ent=trip", "trip"),
    ("describe the most efficient trip I made", "summary none ent=trip", "trip"), ("how did yesterday go for the car", "summary none ent=day", "trip"), ("recap today for me", "summary none ent=day", "trip"),
    # one quantity or event
    ("how has the battery temperature been this week", "summary battery_temperature", "concept"), ("give me a picture of the motor temperature this month", "summary motor_temperature", "concept"),
    ("overview of how my speed has been", "summary speed", "concept"), ("summarise the tyre pressure situation", "summary tyre_pressure", "concept"), ("how did the state of charge go over the week", "summary battery_soc", "concept"),
    ("rundown of my fuel use lately", "summary fuel_rate", "concept"), ("overview of the hard braking this month", "summary hard_braking", "concept"), ("how have the warnings been", "summary any_warning", "concept"),
    ("summarise how the battery is doing", "summary battery_condition", "concept"), ("how has the cooling system been lately", "summary cooling_condition", "concept"), ("how is my driving style looking", "summary driving_style", "concept"),
    ("overall health picture of the vehicle", "summary vehicle_health", "concept"),
]

CATALOG = [
    ("how hot is the oil in the engine", "now engine_oil_temperature"), ("what is the oil pressure at", "now engine_oil_pressure"), ("how far is the throttle open", "now throttle_position"),
    ("what is the turbo boost right now", "now boost_pressure"), ("current intake manifold pressure", "now intake_manifold_pressure"), ("how much air is going into the engine", "now mass_air_flow"),
    ("how warm is the air entering the engine", "now intake_air_temperature"), ("what is the atmospheric pressure", "now barometric_pressure"), ("what does the oxygen sensor say", "now lambda"),
    ("what is the short term trim at", "now short_term_fuel_trim"), ("current long term fuel trim", "now long_term_fuel_trim"), ("catalyst temperature now", "now catalyst_temperature"),
    ("what is the ignition advance", "now ignition_timing"), ("how hot are the exhaust gases", "now exhaust_gas_temperature"), ("how clogged is the diesel particulate filter", "now dpf_soot_load"),
    ("is the filter cleaning itself right now", "now dpf_regeneration"), ("how much AdBlue is in the tank", "now def_level"), ("what is the common rail pressure", "now fuel_rail_pressure"),
    ("what is the fuel temperature", "now fuel_temperature"), ("how far open is the EGR", "now egr_position"), ("how long since the engine started", "now engine_run_time"),
    ("total engine hours", "now engine_hours"), ("which gear is the car in", "now gear"), ("is the handbrake applied", "now parking_brake"), ("what is the altitude here", "now altitude"),
    ("is any door ajar", "now door_state"), ("am I wearing my seat belt", "now seatbelt_state"), ("are the lights on", "now headlight_state"), ("are the wipers going", "now wiper_state"),
    ("what is the inverter temperature", "now inverter_temperature"), ("highest individual cell voltage", "now cell_voltage_max"), ("how much current is the charger pushing", "now charging_current"),
    ("what is the charging voltage", "now charging_voltage"), ("what is the charge cap", "now charge_limit"), ("is the charging flap open", "now charge_port_state"), ("minutes until fully charged", "now time_to_full"),
    ("what range is the dashboard claiming", "now reported_range"), ("gearbox oil temperature", "now transmission_temperature"), ("what is the cooling system pressure", "now coolant_pressure"),
    ("what is the fuel delivery pressure", "now fuel_pressure"), ("peak oil temperature this week", "extreme engine_oil_temperature op=max"), ("minimum oil pressure last month", "extreme engine_oil_pressure op=min"),
    ("mean throttle opening yesterday", "stat throttle_position op=mean"), ("highest boost last week", "extreme boost_pressure op=max"), ("average manifold pressure over the week", "stat intake_manifold_pressure op=mean"),
    ("hottest intake air this month", "extreme intake_air_temperature op=max"), ("average lambda this week", "stat lambda op=mean"), ("how has long term trim moved this month", "trend long_term_fuel_trim op=change"),
    ("has the catalyst temperature fallen over the month", "trend catalyst_temperature op=drop"), ("hottest exhaust last week", "extreme exhaust_gas_temperature op=max"),
    ("how has the soot load moved this month", "trend dpf_soot_load op=change"), ("lowest AdBlue level this week", "extreme def_level op=min"), ("mean rail pressure last month", "stat fuel_rail_pressure op=mean"),
    ("hottest inverter this month", "extreme inverter_temperature op=max"), ("mean charging current this month", "stat charging_current op=mean"), ("what is a normal barometric pressure here", "stat barometric_pressure op=usual"),
    ("what does a lambda sensor measure", "define lambda"), ("what is a particulate filter", "define dpf_soot_load"), ("what is the egr for", "define egr_position"), ("what is fuel trim", "define short_term_fuel_trim"),
    ("what is manifold pressure", "define intake_manifold_pressure"), ("what is turbo boost", "define boost_pressure"), ("what is AdBlue for", "define def_level"), ("what does the throttle position sensor do", "define throttle_position"),
    ("is the engine oil temperature normal right now", "now engine_oil_temperature vs=typical"), ("was the boost pressure higher than usual this week", "compare boost_pressure op=mean vs=typical"),
    ("did the exhaust temperature spike this week", "extreme exhaust_gas_temperature op=max"), ("is the 12 volt battery low", "now aux_battery_voltage"), ("how is the dpf doing", "now dpf_soot_load"),
    ("does oil temperature follow coolant temperature", "stat engine_oil_temperature op=corr c2=motor_temperature"), ("is boost related to engine load", "stat boost_pressure op=corr c2=engine_load"),
    ("how do throttle position and speed relate", "stat throttle_position op=corr c2=speed"), ("fuel rate and engine rpm side by side", "compare fuel_rate op=mean c2=engine_rpm"),
]

FOLLOW = [
    ("summary none", "and for the previous week", "summary none ref=period", "period"), ("summary none", "what about the whole month", "summary none ref=period", "period"), ("summary none", "do it for yesterday", "summary none ref=period", "period"),
    ("summary none", "same thing for the last ten days", "summary none ref=period", "period"), ("summary none", "and what has changed", "summary none op=change ref=period", "period"), ("summary none", "how about the tyres", "summary tyre_condition ref=concept", "concept"),
    ("summary none", "and the battery", "summary battery_condition ref=concept", "concept"), ("summary none", "what about the warnings", "summary any_warning ref=concept", "concept"), ("summary none", "how did you get that", "explain none ref=answer", "answer"),
    ("summary none op=change", "and the month before", "summary none op=change ref=period", "period"), ("summary none op=change", "just the last three days", "summary none op=change ref=period", "period"),
    ("summary none op=change", "is any of that unusual", "anomaly none ref=concept", "concept"), ("anomaly none", "and last month", "anomaly none op=none ref=period", "period"), ("anomaly none", "just yesterday", "anomaly none ref=period", "period"),
    ("anomaly none", "check the battery temperature", "anomaly battery_temperature ref=concept", "concept"), ("anomaly none", "and the tyres", "anomaly tyre_condition ref=concept", "concept"),
    ("anomaly battery_temperature", "now the motor", "anomaly motor_temperature ref=concept", "concept"), ("anomaly battery_temperature", "over the last month", "anomaly battery_temperature ref=period", "period"),
    ("stat battery_temperature op=usual", "and the motor temperature", "stat motor_temperature op=usual ref=concept", "concept"), ("stat speed op=usual", "what about the tyre pressure", "stat tyre_pressure op=usual ref=concept", "concept"),
    ("summary none ent=trip", "why was it so hot", "why battery_temperature op=rise ent=trip ref=focus", "focus"), ("summary none ent=trip", "what made it use so much energy", "why wh_per_km op=rise ent=trip ref=focus", "focus"),
    ("summary none ent=trip", "was that one unusual", "anomaly none ent=trip ref=focus", "focus"), ("why battery_temperature op=rise ent=trip", "and the motor", "why motor_temperature op=rise ent=trip ref=concept", "concept"),
    ("events hard_braking op=count", "summarise the whole week", "summary none", "fresh"), ("stat speed op=mean", "give me an overview of last month", "summary none", "fresh"), ("now battery_soc", "anything unusual lately", "anomaly none", "fresh"),
    ("extreme motor_temperature op=max", "what changed this week", "summary none op=change", "fresh"), ("now speed", "what is the normal range of the pack temperature", "stat battery_temperature op=usual", "fresh"),
]


def main():
    O = ON.load(2); out = []
    for name, rows in (("digest_v1", [(i + 1, t, spec(s), tag) for i, (t, s, tag) in enumerate(DIGEST)]), ("catalog_v1", [(i + 1, t, spec(s), "catalog") for i, (t, s) in enumerate(CATALOG)])):
        path = os.path.join(HERE, name + ".tsv")
        if os.path.exists(path): print("exists, not overwritten:", path); continue
        bad = [(t, f.to_text(2), O.validate(f)) for _, t, f, _ in rows if O.validate(f)]
        if bad: raise SystemExit(f"{name}: gold frames that do not type-check: {bad}")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {name.upper()}: frame version 2 extended (summary / anomaly / usual / trip focus; universal-catalogue vocabulary).  Authored 2026-10-21 before the training data of these families was built.  Never trained on.\n# Format: id <TAB> text <TAB> gold frame (act concept op cond vs ent c2 ref) <TAB> tag\n")
            for i, t, f, tag in rows: fh.write(f"{name[0].upper()}{i:03d}\t{t}\t{f.to_text(2)}\t{tag}\n")
        open(os.path.join(HERE, name + ".sha256"), "w").write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "  " + name + ".tsv\n"); print("wrote", path, len(rows))
    path = os.path.join(HERE, "followup_v3.tsv")
    if os.path.exists(path): print("exists, not overwritten:", path); return
    rows = []
    for i, (p, t, g, tag) in enumerate(FOLLOW):
        pf, gf = spec(p), spec(g); bad = O.validate(pf) + O.validate(gf)
        if bad: raise SystemExit(f"followup_v3 {t!r}: {bad}")
        rows.append((i + 1, pf, t, gf, tag))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# FOLLOWUP_V3: two-turn dialogues for the new acts.  Authored 2026-10-21 before the training data was built.  Never trained on.\n# Format: id <TAB> previous frame (8 words) <TAB> text <TAB> gold frame (8 words) <TAB> tag\n")
        for i, pf, t, gf, tag in rows: fh.write(f"W{i:03d}\t{pf.to_text(2)}\t{t}\t{gf.to_text(2)}\t{tag}\n")
    open(os.path.join(HERE, "followup_v3.sha256"), "w").write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "  followup_v3.tsv\n"); print("wrote", path, len(rows))


if __name__ == "__main__":
    main()

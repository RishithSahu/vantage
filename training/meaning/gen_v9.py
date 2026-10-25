"""Writes training/meaning/natural_v9.txt: basic questions of every kind x period words (relative, calendar years and months, long windows) x driving conditions, with a frame that never carries the period (a period is a value, literals.py).
Round of 2026-10-23: the owner asked that every basic and time question be right.  The wording shells and quantity phrases here are written independently of the frozen sets (basic_v2, hist_v2); the builder drops any row that overlaps one.
    python training/meaning/gen_v9.py"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__)); R = random.Random(9)
P = ["today", "yesterday", "this week", "last week", "this month", "last month", "in the last 7 days", "over the past 3 days", "in the past 2 weeks", "in 2023", "in 2024", "in 2025", "in 2026", "last year", "this year",
     "in march 2025", "in july 2024", "in january 2026", "in october 2025", "in the last 12 months", "over the last 2 years", "in the past 6 months", "in the last 90 days", "since january", "on friday", "last weekend", "so far this year", "over the last three years", "during 2024", "throughout 2025"]
P_EVER = ["", " ever", " in total", " over the whole history", " since the beginning", " so far"]
rows = []


def add(text, frame): rows.append(f"{text} | {frame}")


def pick(xs, n): return R.sample(xs, min(n, len(xs)))


def period(): return R.choice(P)


# --------------------------------------------------------------------------------------------------- signals
SIG = {"speed": ["speed", "vehicle speed", "road speed"], "battery_soc": ["battery level", "state of charge", "battery charge"], "battery_temperature": ["battery temperature", "pack temperature"],
       "motor_temperature": ["motor temperature"], "ambient_temperature": ["outside temperature", "ambient temperature", "air temperature outside"], "battery_power": ["battery power", "power draw"],
       "hvac_power": ["climate control power", "air conditioning power"], "coolant_flow": ["coolant flow"], "tyre_pressure": ["tyre pressure", "tyre pressures"], "aux_battery_voltage": ["12 volt battery voltage", "auxiliary battery voltage"],
       "fuel_level": ["fuel level", "tank level"], "engine_rpm": ["engine speed", "engine rpm"], "engine_oil_temperature": ["engine oil temperature", "oil temperature"], "engine_load": ["engine load"], "throttle_position": ["throttle position"],
       "lane_confidence": ["lane keeping confidence", "lane detection confidence"], "odometer": ["odometer", "odometer reading"], "heading": ["heading", "compass direction"], "wheel_speed": ["wheel speeds"], "range_estimate": ["range", "remaining range"]}
NOW = ["what is the {q}", "how is the {q}", "{q} now", "current {q}", "tell me the {q} right now", "what is my {q} at the moment", "show me the {q}", "{q}?"]
for c, qs in SIG.items():
    for q in qs:
        for s in pick(NOW, 4): add(s.format(q=q), f"now {c}")
STAT = ["what was the average {q} {p}", "average {q} {p}", "mean {q} {p}", "what is the average {q} {p}", "how was the {q} {p} on average"]
MAXS = ["what was the highest {q} {p}", "peak {q} {p}", "maximum {q} {p}", "what was the max {q} {p}", "how high did the {q} get {p}"]
MINS = ["what was the lowest {q} {p}", "minimum {q} {p}", "what was the min {q} {p}", "how low did the {q} get {p}", "smallest {q} {p}"]
for c, qs in SIG.items():
    if c in ("odometer", "heading", "lane_confidence", "wheel_speed", "range_estimate", "tyre_pressure"): continue
    for q in qs:
        for s in pick(STAT, 3): add(s.format(q=q, p=period()), f"stat {c} op=mean")
        for s in pick(MAXS, 3): add(s.format(q=q, p=period()), f"extreme {c} op=max")
        for s in pick(MINS, 2): add(s.format(q=q, p=period()), f"extreme {c} op=min")
        add(f"has the {q} been rising {period()}", f"trend {c} op=rise"); add(f"is the {q} going down {period()}", f"trend {c} op=drop"); add(f"how has the {q} changed {period()}", f"trend {c} op=change")
        add(f"is the {q} getting higher", f"trend {c} op=rise"); add(f"is the {q} dropping", f"trend {c} op=drop")
for q in ["tyre pressure", "tyre pressures", "pressure in my tyres", "how are my tyre pressures", "check my tyre pressure", "are my tyres pumped up enough"]: add(q if q.startswith(("how", "check", "are")) else f"what is the {q}", "now tyre_pressure")
# --------------------------------------------------------------------------------------------------- accumulated quantities
ACC = {"energy_used": (["how much energy did i use {p}", "total energy used {p}", "energy consumed {p}", "how many kwh did i use {p}"], "sum"),
       "distance_driven": (["how far did i drive {p}", "total distance driven {p}", "how many kilometres did i drive {p}", "distance travelled {p}", "how many km did i do {p}"], "sum"),
       "wh_per_km": (["what was my consumption per km {p}", "average energy use per kilometre {p}", "how efficient was i {p}", "my efficiency {p}"], "mean")}
for c, (ss, op) in ACC.items():
    for s in ss:
        for _ in range(4): add(s.format(p=period()), f"stat {c} op={op}")
        add(s.format(p="").strip(), f"stat {c} op={op}")
    for p in pick(P, 6): add(f"how much {'energy' if c == 'energy_used' else 'distance'} {p}" if c != "wh_per_km" else f"efficiency {p}", f"stat {c} op={op}")
for e in P_EVER:
    add(f"total distance driven{e}", "stat distance_driven op=sum"); add(f"how far have i driven{e}", "stat distance_driven op=sum"); add(f"total energy used{e}", "stat energy_used op=sum")
add("what is the total distance since the start", "stat distance_driven op=sum"); add("how many kilometres in total", "stat distance_driven op=sum"); add("lifetime distance driven", "stat distance_driven op=sum")
# --------------------------------------------------------------------------------------------------- trips
for p in P + [""] * 3:
    pp = (" " + p) if p else ""
    for s in pick(["how many trips did i take{p}", "number of trips{p}", "how many journeys did i make{p}", "how many drives{p}", "how many times did i drive{p}"], 2): add(s.format(p=pp), "stat trip_record op=count ent=trip")
    for s in pick(["what was my longest trip{p}", "which was the longest drive{p}", "my longest journey{p}", "what is the longest distance in one trip{p}", "biggest trip{p}", "how far was my longest trip{p}"], 3): add(s.format(p=pp), "extreme distance_driven op=max ent=trip")
    for s in pick(["what was my shortest trip{p}", "which was the shortest drive{p}", "my smallest journey{p}", "shortest distance in one trip{p}"], 2): add(s.format(p=pp), "extreme distance_driven op=min ent=trip")
    for s in pick(["which trip used the most energy{p}", "most energy hungry trip{p}", "the trip that consumed the most{p}"], 1): add(s.format(p=pp), "extreme energy_used op=max ent=trip")
    for s in pick(["what was my most efficient trip{p}", "best economy trip{p}", "which trip was the most efficient{p}"], 1): add(s.format(p=pp), "extreme wh_per_km op=min ent=trip")
    for s in pick(["what was my least efficient trip{p}", "worst consumption trip{p}", "which trip was the least efficient{p}"], 1): add(s.format(p=pp), "extreme wh_per_km op=max ent=trip")
    for s in pick(["what was the average trip length{p}", "how long is my average trip{p}", "average distance per trip{p}"], 1): add(s.format(p=pp), "stat trip_record op=mean ent=trip")
    for s in pick(["how many days did i drive{p}", "on how many days did i use the car{p}"], 1): add(s.format(p=pp), "stat trip_record op=count ent=day")
for s in ["when was my last trip", "when did i last drive", "when did i last start the car", "when did i last switch on the car", "when did i last turn the car on", "when did i last use the car", "when was the car last driven", "last time i drove",
          "when was the last time i switched on my car", "when did the last journey start", "what time did my last trip begin", "when did i last take the car out"]: add(s, "events trip_record op=last")
for s in ["when was my first trip", "when did i first drive the car", "when was the first journey on record", "first trip ever"]: add(s, "events trip_record op=first")
for s in ["summarise my last trip", "tell me about my last trip", "how was my last drive", "details of my last journey"]: add(s, "summary none ent=trip")
# --------------------------------------------------------------------------------------------------- events
EV = {"hard_braking": (["hard braking event", "harsh braking event", "hard brake"], ["hard braking events", "harsh brakes", "hard brakes"]), "hard_acceleration": (["hard acceleration", "harsh acceleration"], ["hard accelerations", "harsh accelerations"]),
      "charging_session": (["charging session", "charge"], ["charging sessions", "charges"]), "any_warning": (["warning", "alert"], ["warnings", "alerts"]), "diagnostic_code": (["fault code", "diagnostic trouble code"], ["fault codes", "diagnostic codes"]),
      "tyre_pressure_warning": (["tyre pressure warning"], ["tyre pressure warnings"]), "battery_overheat_warning": (["battery overheating warning"], ["battery overheating warnings"]), "adas_disengagement": (["adas disengagement", "lane assist dropout"], ["adas disengagements", "lane assist dropouts"]),
      "ecu_reset": (["ecu reset"], ["ecu resets"]), "unexpected_shutdown": (["unexpected shutdown"], ["unexpected shutdowns"]), "camera_blocked": (["camera blockage"], ["camera blockages"]), "software_update": (["software update"], ["software updates"]),
      "voltage_dip": (["voltage dip"], ["voltage dips"]), "low_12v_warning": (["low 12 volt battery warning"], ["low 12 volt battery warnings"])}
for c, (sing, plu) in EV.items():
    for q in plu:
        for _ in range(3): add(R.choice(["how many {q} {p}", "number of {q} {p}", "count the {q} {p}"]).format(q=q, p=period()), f"events {c} op=count")
        add(f"how many {q} have there been", f"events {c} op=count"); add(f"how many {q} in total", f"events {c} op=count")
        for _ in range(2): add(R.choice(["were there any {q} {p}", "any {q} {p}", "did i get any {q} {p}"]).format(q=q, p=period()), f"events {c} op=exists")
        for _ in range(2): add(R.choice(["list the {q} {p}", "show me the {q} {p}", "what {q} did i have {p}"]).format(q=q, p=period()), f"events {c} op=list")
    for q in sing:
        for s in ["when was the last {q}", "when did the last {q} happen", "when was my most recent {q}", "last {q}"]: add(s.format(q=q), f"events {c} op=last")
        add(f"when was the first {q}", f"events {c} op=first"); add(f"when was the first {q} ever", f"events {c} op=first")
        add(f"when was the last {q} this year", f"events {c} op=last"); add(f"when was the last {q} in 2025", f"events {c} op=last")
# --------------------------------------------------------------------------------------------------- comparisons of two periods
PAIRS = [("this week", "last week"), ("this month", "last month"), ("2024", "2025"), ("this year", "last year"), ("march 2025", "march 2026"), ("the first half of 2025", "the second half of 2025"), ("last month", "the month before"), ("summer 2025", "winter 2025")]
CMP = {"energy_used": ("sum", ["energy use", "total energy used"]), "distance_driven": ("sum", ["distance driven", "kilometres driven"]), "wh_per_km": ("mean", ["consumption per km", "efficiency"]), "speed": ("mean", ["average speed"]),
       "battery_temperature": ("mean", ["battery temperature"]), "trip_record": ("count", ["number of trips", "trips taken"]), "hard_braking": ("count", ["hard braking", "hard braking events"]), "charging_session": ("count", ["charging sessions"])}
for c, (op, qs) in CMP.items():
    for q in qs:
        for a, b in pick(PAIRS, 5):
            ent = " ent=trip" if c == "trip_record" else ""
            add(R.choice(["compare {q} {a} and {b}", "{q} {a} versus {b}", "how does {q} {a} compare with {b}", "{q}: {a} against {b}"]).format(q=q, a=a, b=b), f"compare {c} op={op} vs=period{ent}")
add("how did 2024 compare with 2025", "summary none op=change"); add("how did this year compare with last year", "summary none op=change"); add("what changed since last year", "summary none op=change"); add("summarise 2025", "summary none")
add("how was my year", "summary none"); add("give me a summary of 2024", "summary none"); add("how was my week", "summary none"); add("summary of last month", "summary none")
# --------------------------------------------------------------------------------------------------- driving conditions mixed with a period
CND = {"highway": ["on the highway", "on the motorway"], "city": ["in the city", "in city driving"], "night": ["at night", "after dark"], "charging": ["while charging"], "raining": ["in the rain", "when it was raining"], "cold": ["in cold weather", "when it was cold"],
       "hot_weather": ["in hot weather", "on hot days"], "high_speed": ["at high speed"], "idling": ["while standing still"], "parked": ["while parked"]}
MQ = {"speed": ("mean", "average speed"), "battery_temperature": ("max", "highest battery temperature"), "wh_per_km": ("mean", "average consumption per km"), "energy_used": ("sum", "energy used"), "hvac_power": ("mean", "average climate control power"), "distance_driven": ("sum", "distance driven")}
for cond, ph in CND.items():
    for c, (op, q) in MQ.items():
        if cond in ("charging", "parked", "idling") and c in ("distance_driven", "speed"): continue
        act = "extreme" if op == "max" else "stat"
        for _ in range(2): add(f"{q} {R.choice(ph)} {period()}", f"{act} {c} op={op} cond={cond}")
    add(f"how many trips {R.choice(ph)} {period()}", f"stat trip_record op=count cond={cond} ent=trip")
    add(f"how many hard braking events {R.choice(ph)} {period()}", f"events hard_braking op=count cond={cond}")
for cond, ph in (("parked", "parked"), ("charging", "charging"), ("driving", "driving")):
    for _ in range(3):
        add(R.choice([f"how long was the car {ph} {{p}}", f"how many hours was it {ph} {{p}}", f"time spent {ph} {{p}}"]).format(p=period()), f"stat drive_mode op=duration cond={cond}")
        add(R.choice([f"what share of {{p}} was the car {ph}", f"what percentage of the time was it {ph} {{p}}"]).format(p=period()), f"stat drive_mode op=share cond={cond}")
for lim, c, ph in (("120 km/h", "speed", "above 120 km/h"), ("100 km/h", "speed", "faster than 100 km/h"), ("10 percent", "battery_soc", "below 10 percent"), ("40 degrees", "battery_temperature", "above 40 degrees")):
    for _ in range(3):
        add(f"how long was the {'speed' if c == 'speed' else 'battery ' + ('level' if c == 'battery_soc' else 'temperature')} {ph} {period()}", f"stat {c} op=duration cond=threshold")
        add(f"how many times did the {'speed go' if c == 'speed' else 'battery level go' if c == 'battery_soc' else 'battery temperature go'} {ph} {period()}", f"stat {c} op=count cond=threshold")
for n in ("100", "50", "200"):
    add(f"how many trips were longer than {n} km {period()}", "stat distance_driven op=count cond=threshold ent=trip"); add(f"trips over {n} km {period()}", "stat distance_driven op=count cond=threshold ent=trip")
# --------------------------------------------------------------------------------------------------- why, usual, ICE
for c, qs in (("range_estimate", ["range"]), ("wh_per_km", ["energy consumption", "consumption"]), ("battery_temperature", ["battery temperature"])):
    for q in qs:
        for a, op in (("drop", "drop"), ("go up", "rise"), ("fall", "drop"), ("increase", "rise")):
            add(f"why did my {q} {a} {period()}", f"why {c} op={op}"); add(f"why did the {q} {a}", f"why {c} op={op}")
for q, c in (("weekly distance", "distance_driven"), ("daily energy use", "energy_used"), ("trip length", "distance_driven")): add(f"what is my typical {q}", f"stat {c} op=usual")
add("how much fuel did i use last week", "stat fuel_rate op=sum"); add("how much fuel did i use in 2025", "stat fuel_rate op=sum"); add("total fuel used this month", "stat fuel_rate op=sum"); add("what is my average fuel consumption", "stat fuel_rate op=mean")
add("how many litres per 100 km", "stat fuel_rate op=mean"); add("how much fuel is left", "now fuel_level"); add("when did i last refuel", "events charging_session op=last")
# --------------------------------------------------------------------------------------------------- refusals and small talk the model must keep
for s in ["disable the brakes", "turn off the brakes", "switch off the airbags", "disable the abs", "override the speed limiter", "turn off traction control for good", "remove the speed limit", "disable the seatbelt warning"]: add(s, "unsafe none")
for s in ["open the windows", "turn on the air conditioning", "lock the doors", "set the cruise control to 100", "start the engine remotely", "turn the heating on", "play some music", "navigate home"]: add(s, "control none")
for s in ["hello", "hi there", "thanks a lot", "good morning", "good evening", "thank you", "bye", "hey vantage"]: add(s, "social none")
for s in ["tell me a joke", "what is the capital of france", "who won the match", "what is the meaning of life", "recommend a restaurant", "what is the weather in paris"]: add(s, "offtopic none")
random_out = os.path.join(HERE, "natural_v9.txt")
seen = set(); out = []
for r in rows:
    k = r.lower()
    if k in seen: continue
    seen.add(k); out.append(r)
with open(random_out, "w", encoding="utf-8", newline="\n") as f:
    f.write("# Frame version 2, round 9 (2026-10-23), written by gen_v9.py: basic questions x period words (relative, calendar years / months, long windows) x driving conditions.  Same format as natural_v3.txt.\n")
    f.write("\n".join(out) + "\n")
print(len(out), "rows")

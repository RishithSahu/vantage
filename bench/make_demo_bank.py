"""Generates the DEMONSTRATION BANK from the operation specification: for each requested result, several phrasings of that kind of request, with the slots (signal, event, period, condition, number, selection) filled from the vehicle REGISTRY's own labels and from period / condition vocabularies - varied independently of any benchmark wording. The bank is data used offline to build demonstrations; the runtime never reads a benchmark question.
Each entry: {"q": the message, "result": requested result, "kind": target kind (signal | event | entity | none), "target": id or None, "statistic": ..., "period": ..., "restriction": ...}.
Independence is checked, not assumed: every generated question is compared by embedding similarity with every question of the development, final, diagnostic and everyday-blind files (the sealed acceptance file is never read); an entry at or above SIM_MAX is dropped and counted.
    python bench/make_demo_bank.py [--per-result 14] [--seed 11] --out data/demo_bank.json"""
import argparse, glob, hashlib, json, os, random, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from vgpt import capreg, signal_registry as R, toolspec as T
from vgpt.retrieval import Embedder

SIM_MAX = 0.90
SIGNALS = {"battery_temperature": ["battery temperature", "pack temperature"], "motor_temperature": ["motor temperature"], "cabin_temperature": ["cabin temperature", "interior temperature"], "ambient_temperature": ["outside temperature"], "speed": ["speed", "vehicle speed"],
           "battery_soc": ["battery charge", "state of charge"], "energy_used": ["energy used", "energy consumption"], "wh_per_km": ["efficiency", "consumption per kilometre"], "hvac_power": ["climate control power", "heating and cooling power"], "regen_power": ["regeneration power"],
           "charging_power": ["charging power"], "coolant_flow": ["coolant flow"], "tyre_pressure": ["tyre pressure"], "aux_battery_voltage": ["12 volt battery voltage"], "range_estimate": ["estimated range"], "distance_driven": ["distance driven"],
           "battery_soh": ["battery health"], "brake_pad_life": ["brake pad life"], "lane_confidence": ["lane detection confidence"], "camera_sharpness": ["camera sharpness"], "motor_power": ["motor power"], "wheel_speed_fl": ["front left wheel speed"]}
EVENTS = {"HARSH_BRAKE": ["hard braking events", "harsh brakes"], "HARSH_ACCEL": ["hard accelerations"], "HARSH_CORNER": ["hard corners"], "TPMS_LOW": ["low tyre pressure warnings"], "LOW_12V": ["12 volt battery warnings"], "CAMERA_BLOCKED": ["camera blockages"],
          "CHARGE_START": ["charging sessions"], "ADAS_DISENGAGE": ["lane assist drop-outs", "driver assistance disengagements"], "DTC_SET": ["fault codes"], "WARN_BATT_OVERTEMP": ["battery overheating warnings"], "ANY_WARNING": ["warnings"]}
PERIODS = ["yesterday", "today", "last week", "this month", "in the last 3 days", "over the past two weeks", "last night", "this week", "during the last 10 days", "last month", "this morning", "in the past 48 hours", "over the last 5 days", "since last week"]
CONDS = [("while charging", "charging"), ("on the highway", "highway"), ("in the city", "city"), ("while it was raining", "raining"), ("at night", "night"), ("while parked", "parked"), ("in cold weather", "cold")]
NUM = [("above 40", "gt"), ("below 20", "lt"), ("over 60", "gt"), ("under 5", "lt"), ("above 30", "gt")]
PLACES = ["the nearest charger", "the office", "home", "the airport"]
THINGS = ["the air conditioning", "the heated seats", "the windows", "the radio", "the cruise control", "the sunroof", "the wipers"]

# result -> list of (template, kind, statistic). Slots: {s} signal, {s2} second signal, {e} event, {p} period, {p2} second period, {c} restriction phrase, {n} threshold phrase, {x} trip / session noun, {k} number word, {pl} place, {t} thing
TEMPLATES = {
    "latest_value": [("what is the {s} right now", "signal", None), ("current {s}?", "signal", None), ("how is the {s} at the moment", "signal", None), ("tell me the {s} now", "signal", None), ("can you tell me the {s}", "signal", None), ("what does the {s} read at the moment", "signal", None)],
    "summary": [("what was the average {s} {p}", "signal", "mean"), ("mean {s} {p}", "signal", "mean"), ("typical {s} {p}", "signal", "mean"), ("total {s} {p}", "signal", "sum"), ("how much {s} did I use in total {p}", "signal", "sum"),
                ("median {s} {p}", "signal", "median"), ("how variable was the {s} {p}", "signal", "std"), ("how much did the {s} vary {p}", "signal", "std"), ("how much did the {s} change {p}", "signal", "delta"), ("average {s} {c} {p}", "signal", "mean")],
    "extreme": [("what was the highest {s} {p}", "signal", "max"), ("peak {s} {p}", "signal", "max"), ("what was the lowest {s} {p}", "signal", "min"), ("minimum {s} {p}", "signal", "min"), ("how high did the {s} get {p}", "signal", "max"),
                ("how low did the {s} go {p}", "signal", "min"), ("max {s} {p}", "signal", "max"), ("the worst {s} {p}", "signal", "max")],
    "count": [("how many {e} {p}", "event", None), ("number of {e} {p}", "event", None), ("count the {e} {p}", "event", None), ("how many {e} did I have {p}", "event", None), ("{e} count {p}", "event", None), ("how many {x} did I do {p}", "entity", None)],
    "existence": [("were there any {e} {p}", "event", None), ("did I get any {e} {p}", "event", None), ("has there been a single case of {e} {p}", "event", None), ("any {e} {p}? yes or no", "event", None), ("did the {s} ever go {n} {p}", "signal", None)],
    "list": [("list the {e} {p}", "event", None), ("show me every one of the {e} {p}", "event", None), ("when was the last of the {e}", "event", None), ("what was the first of the {e} {p}", "event", None), ("which {e} happened {p}", "event", None), ("show the {e} from {p}", "event", None)],
    "time_share": [("what share of {p} was the car {c}", "signal", None), ("how long was the {s} {n} {p}", "signal", None), ("percentage of time {c} {p}", "signal", None), ("for how many minutes was the {s} {n} {p}", "signal", None), ("how much of the time {p} was the {s} {n}", "signal", None)],
    "comparison": [("compare the {s} {p} with {p2}", "signal", None), ("is the {s} higher {c} than otherwise", "signal", None), ("{s} {p} versus {p2}", "signal", None), ("how does the {s} differ between {c} and normal driving", "signal", None), ("did the {s} go up from {p2} to {p}", "signal", None)],
    "change_timing": [("when did the {s} change the most {p}", "signal", None), ("when did the {s} start going up", "signal", None), ("on which day did the {s} jump", "signal", None), ("when did the {s} begin to fall {p}", "signal", None), ("at what point did the {s} shift {p}", "signal", None)],
    "explanation": [("why did the {s} go up {p}", "signal", None), ("what caused the drop in {s} {p}", "signal", None), ("why is the {s} worse than before", "signal", None), ("what explains the change in {s} {p}", "signal", None), ("why did the {e} happen", "event", None), ("how come the {s} fell {p}", "signal", None)],
    "context": [("what was happening just before the {e}", "event", None), ("what led up to the {e}", "event", None), ("what happened right after the {e}", "event", None), ("what was the {s} when the {e} happened", "event", None), ("what was going on around the last of the {e}", "event", None)],
    "trend": [("is the {s} trending up", "signal", None), ("is the {s} getting worse over time", "signal", None), ("has the {s} been creeping down {p}", "signal", None), ("is there a long term drift in the {s}", "signal", None), ("is the {s} improving", "signal", None)],
    "ranking": [("which was my longest {x} {p}", "entity", None), ("my {k} most efficient {x}", "entity", None), ("rank my {x} by energy used {p}", "entity", None), ("the {k} slowest {x} {p}", "entity", None), ("top {k} {x} by distance", "entity", None)],
    "relationship": [("how does the {s} relate to the {s2}", "signal", None), ("is the {s} linked to the {s2}", "signal", None), ("correlation between {s} and {s2}", "signal", None), ("does the {s} depend on the {s2}", "signal", None)],
    "capability": [("do you record the {s}", "signal", None), ("is {s} something you keep track of", "signal", None), ("does the car measure the {s}", "signal", None), ("can you see the {s}", "signal", None), ("is there data on the {s}", "signal", None)],
    "domain_check": [("how are my tyres doing", "none", None), ("is any service due", "none", None), ("what is the weather like around the car", "none", None), ("is the car healthy overall", "none", None), ("how is my driving style", "none", None), ("how are my brakes", "none", None)],
    "refuse": [("turn on {t}", "none", None), ("please switch off {t}", "none", None), ("navigate to {pl}", "none", None), ("who won the football match", "none", None), ("what is the capital of France", "none", None), ("set the cabin to 21 degrees", "none", None), ("tell me a joke", "none", None)],
    "unsupported": [("what will the {s} be next week", "signal", None), ("predict the {s} for tomorrow", "signal", None), ("how will the {s} look in a month", "signal", None)],
}
NUMBERS = ["three", "five", "two", "four"]


def fill(tpl, rng, reg):
    sig = rng.choice(list(SIGNALS)); sig2 = rng.choice([x for x in SIGNALS if x != sig]); ev = rng.choice(list(EVENTS)); cond = rng.choice(CONDS); num = rng.choice(NUM); p, p2 = rng.sample(PERIODS, 2); sn = rng.choice(["trips", "charging sessions"])
    thr = f"{num[0]} {rng.choice(['km/h', 'degrees', 'percent', 'kW'])}"
    vals = {"s": rng.choice(SIGNALS[sig]), "s2": rng.choice(SIGNALS[sig2]), "e": rng.choice(EVENTS[ev]), "p": p, "p2": p2, "c": cond[0], "n": thr, "x": sn, "k": rng.choice(NUMBERS), "pl": rng.choice(PLACES), "t": rng.choice(THINGS)}
    text = tpl.format(**vals); used = {k for k in vals if "{" + k + "}" in tpl}
    return " ".join(text.split()), {"signal": sig if "s" in used else None, "event": ev if "e" in used else None, "period": p if "p" in used else None, "restriction": cond[1] if "c" in used else None, "number": thr if "n" in used else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-result", type=int, default=14); ap.add_argument("--seed", type=int, default=11); ap.add_argument("--out", default=os.path.join(HERE, "data", "demo_bank.json")); a = ap.parse_args()
    rng = random.Random(a.seed); emb = Embedder(os.path.join(HERE, "models", "bge-small-en-v1.5"), int8=True)
    eval_q = []
    for f in ("indep_dev.tsv", "indep_final.tsv", "diag_v1.tsv", "everyday_blind.tsv", "everyday_blind2.tsv", "everyday_blind3.tsv", "everyday_blind4.tsv", "heldout_dev.tsv", "heldout_final.tsv"):
        p = os.path.join(HERE, "bench", f)
        if os.path.exists(p):
            try: eval_q += [r["text"] for r in load(p)]
            except Exception: pass
    E = emb.encode(eval_q); out = []; dropped = 0; seen = set()
    for result, tpls in TEMPLATES.items():
        got = 0; tries = 0
        while got < a.per_result and tries < 400:
            tries += 1; tpl, kind, stat = tpls[(got + tries) % len(tpls)] if tries % 3 == 0 else rng.choice(tpls); q, slots = fill(tpl, rng, None)
            if q in seen: continue
            if float((emb.encode([q]) @ E.T).max()) >= SIM_MAX: dropped += 1; continue
            seen.add(q); got += 1; out.append({"q": q, "result": result, "kind": kind, "statistic": stat, **slots})
    json.dump({"version": 1, "seed": a.seed, "sim_max": SIM_MAX, "dropped_for_similarity": dropped, "checked_against": len(eval_q), "entries": out}, open(a.out, "w"), indent=1)
    print(f"{len(out)} demonstrations ({len(TEMPLATES)} results); {dropped} generated questions dropped for similarity >= {SIM_MAX} with one of {len(eval_q)} evaluation questions; sha256 {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()[:16]}")


if __name__ == "__main__": main()

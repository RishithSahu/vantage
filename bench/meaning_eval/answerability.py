"""Answerability across vehicles: does the assistant answer where the data exists and decline where it does not?  (End to end: the real model, the real resolver, every dataset; the truth comes from the database.)

    python bench/meaning_eval/answerability.py [--frame 2] [--adapter models/meaning-lora-frame2.gguf] [--name frame2] [--datasets world_aster.db ...] [--limit N] [--forms now,stat,extreme]

For every dataset and every quantity in a bank, three questions are asked - the value now, the average over the last seven days, the highest value last week.  The TRUTH is whether the dataset holds any reading of that
quantity: the `sig_catalog` of a world (every signal the vehicle's own profile declares, with rows) or the non-null columns of the 1-minute tier (the four recorded Case Study 5 files hold speed, plus
SOC and power in one).  An answer that presents a value counts as ANSWERED; a statement that the quantity is not recorded counts as DECLINED.
    correct          answered where recorded, declined where not
    false answer     answered where nothing is recorded (an invented or substituted value): must be 0
    false decline    declined where the reading exists
Both rates are printed per dataset and per vehicle class; every disagreement is listed so it can be read.  The classification of the answer text uses the refusal wording of the application itself (REFUSAL_RX) and
treats "the last recorded value was ..." as an answer.  Output: results/meaning/answerability_<name>.json"""
import argparse
import collections
import io
import json
import os
import re
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_NO_QA_CLF"] = "1"

DATASETS = {"world_aster.db": "EV", "world_meridian.db": "EV", "world_kestrel.db": "petrol", "world_rhino.db": "diesel", "cs5_diesel_ice.db": "sparse (speed only)", "cs5_petrol_hybrid.db": "sparse (speed only)",
            "cs5_battery_electric.db": "sparse (speed, SOC, power)"}
CORE = {"speed": "speed", "battery_soc": "soc", "battery_power": "power", "battery_temperature": "batt", "coolant_flow": "flow", "ambient_temperature": "amb", "hvac_power": "hvac", "motor_temperature": "motor"}
OTHER = {"camera_blockage": ("cam_1m", "block_avg"), "camera_sharpness": ("cam_1m", "sharp_avg"), "camera_brightness": ("cam_1m", "luma_avg"), "lane_confidence": ("cam_1m", "lane_avg"), "light": ("ext_1m", "lux"), "rain": ("ext_1m", "rain")}
EITHER = {"engine_torque": "motor_torque", "engine_rpm": "motor_rpm",       # the registry reads "engine torque" / "engine rpm" as the motor's on an electric car: answering with the motor's value is right, declining is right too
          "longitudinal_acceleration": "speed", "motor_rpm": "speed",         # computed from the speed signal and labelled as such ("Method: d(speed)/dt"): an answer is right when speed is recorded, a refusal is right too
          "reported_range": "speed"}                                          # "the dashboard range": an explicit estimate of ours is acceptable, a statement that the car reports none is too
FORMS = {"now": "what is the {label}", "stat": "what was the average {label} over the last 7 days", "extreme": "what was the highest {label} last week"}
REFUSAL_RX = re.compile(r"not something this vehicle records|isn't recorded|is not recorded|not recorded|no traction battery|nothing to answer this from|can't give a value|cannot answer it from its data|no (?:data|values?|readings?) (?:for|of|on|in)|"
                        r"not reported|does not (?:record|report|have|hold)|doesn't (?:record|report|have|hold)|outside what this vehicle|nothing was (?:estimated|substituted)|no such (?:signal|sensor)|has no |holds no |never recorded|"
                        r"couldn't find any data|nothing recorded|no record|\bno [^.\n]{0,80}(?:is|are) recorded|cannot be estimated|can't (?:determine|calculate|compute|estimate)|cannot (?:determine|calculate|compute)|\bno [^.\n]{0,40}measurement", re.I)
STALE_RX = re.compile(r"last recorded value was|last (?:known|recorded) (?:value|reading)", re.I)


def truth(path):
    """concept -> True/False for every concept the bank asks about, from the database itself."""
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True); have = set()
    try:
        cols = {r[1] for r in c.execute("PRAGMA table_info(sig_1m)")}
        for concept, col in CORE.items():
            if col in cols and c.execute(f"SELECT COUNT({col}) FROM sig_1m").fetchone()[0]: have.add(concept)
        try: have |= {r[0] for r in c.execute("SELECT concept FROM sig_catalog")}
        except sqlite3.Error: pass
        for concept, (table, col) in OTHER.items():
            try:
                if c.execute(f"SELECT COUNT({col}) FROM {table}").fetchone()[0]: have.add(concept)
            except sqlite3.Error: pass
    finally: c.close()
    return have


def end_of(path):
    """'Now' for a recorded history: just after its last minute, so "the last 7 days" means the last week OF THE DATA (the four Case Study 5 files end weeks before today)."""
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try: return int(c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]) + 120
    finally: c.close()


def bank():
    from vgpt import signal_registry as R
    names = set(CORE)
    for db in ("world_aster.db", "world_meridian.db", "world_kestrel.db", "world_rhino.db"):
        p = os.path.join(ROOT, "data", db)
        if os.path.exists(p): names |= truth(p)
    out = []
    for n in sorted(names):
        c = R.CONCEPTS.get(n)
        if c is None or c.dtype != "numeric" or c.kind != "measured": continue
        out.append((n, c.label))
    return out


GENERIC = {"temperature", "pressure", "level", "speed", "power", "rate", "current", "voltage", "state", "position", "load", "time", "total", "average", "highest", "lowest"}


def names_the_quantity(concept, answer):
    """An answer that presents a value must be ABOUT the quantity asked: it names it (a distinctive word of its label or of one of its synonyms).  "The highest battery voltage ..." for "boost pressure" does not."""
    from vgpt import signal_registry as R
    c = R.CONCEPTS.get(concept)
    if c is None: return True
    low = (answer or "").lower().replace("-", " "); words = set()
    for ph in (c.label, *c.synonyms): words |= {w for w in re.findall(r"[a-z]{3,}", ph.lower().replace("-", " "))}
    distinct = {w for w in words if w not in GENERIC and w not in ("the", "and", "for", "how", "what", "its")}
    pool = distinct or words
    return any(w in low for w in pool)


def classify(answer):
    a = answer or ""
    if STALE_RX.search(a): return "answered"
    return "declined" if REFUSAL_RX.search(a) else "answered"


def summarize(rows):
    by = collections.defaultdict(list)
    for x in rows: by[x["class"]].append(x)
    print()
    for k, v in by.items(): print(f"{k:28} correct {100 * sum(x['ok'] for x in v) / len(v):5.1f} %   false answers {sum(1 for x in v if x['want'] == 'declined' and x['got'] == 'answered'):3} of {sum(1 for x in v if x['want'] == 'declined')} absent   false declines {sum(1 for x in v if x['want'] == 'answered' and x['got'] == 'declined'):3} of {sum(1 for x in v if x['want'] == 'answered')} present")
    print(f"\nall: {100 * sum(x['ok'] for x in rows) / max(1, len(rows)):.1f} % of {len(rows)}")
    wrong = [x for x in rows if x["want"] == "answered" and x["got"] == "answered" and x.get("about") is False]
    print(f"answers about ANOTHER quantity than the one asked (counted as wrong): {len(wrong)}")


def rescore(name):
    """Score the stored answers of an earlier run again with the current truth rules and refusal wording (no model is run)."""
    rows = json.load(io.open(os.path.join(ROOT, "results", "meaning", f"answerability_{name}.json"), encoding="utf-8")); truths = {}
    for x in rows:
        if x["dataset"] not in truths: truths[x["dataset"]] = truth(os.path.join(ROOT, "data", x["dataset"]))
        have = truths[x["dataset"]]; x["got"] = classify(x["answer"]); x["want"] = "answered" if x["concept"] in have else ("either" if EITHER.get(x["concept"]) in have else "declined"); x["about"] = names_the_quantity(x["concept"], x["answer"]) if x["got"] == "answered" else None; x["ok"] = x["want"] == "either" or (x["got"] == x["want"] and not (x["got"] == "answered" and x["want"] == "answered" and not x["about"]))
    summarize(rows)
    for x in rows:
        if not x["ok"]: print(f"  [{x['dataset'][:14]}] want {x['want']:8} {x['q'][:60]:60} -> {x['answer'][:100]!r}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rescore", default=None, help="name of an earlier run (answerability_<name>.json): score its stored answers again, run nothing"); ap.add_argument("--frame", type=int, default=1); ap.add_argument("--adapter", default=None); ap.add_argument("--name", default="run"); ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N quantities of the bank"); ap.add_argument("--forms", default="now,stat,extreme"); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
    if a.rescore: return rescore(a.rescore)
    os.environ["VGPT_FRAME"] = str(a.frame)
    if a.adapter: os.environ["VGPT_MEANING_ADAPTER"] = a.adapter if os.path.isabs(a.adapter) else os.path.join(ROOT, a.adapter)
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    forms = [f for f in a.forms.split(",") if f in FORMS]; B = bank()[:a.limit or None]; rows = []; model = None; t0 = time.time()
    print(f"bank: {len(B)} quantities x {len(forms)} forms x {len(a.datasets)} datasets")
    for ds in a.datasets:
        path = os.path.join(ROOT, "data", ds)
        if not os.path.exists(path): print("missing", ds); continue
        have = truth(path); LK.adopt_path(path); brain = make_meaning_brain(path, slm=model, now=end_of(path), threads=a.threads); model = brain.slm
        for concept, label in B:
            for form in forms:
                q = FORMS[form].format(label=label)
                brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain._last_data = None
                getattr(brain, "dialogue", None) and brain.dialogue.reset(); brain._frames.clear()
                r = brain.ask(q); ans = str(r.get("answer")); f = r.get("facts") or {}; got = classify(ans); want = "answered" if concept in have else ("either" if EITHER.get(concept) in have else "declined")
                rows.append({"dataset": ds, "class": DATASETS.get(ds, "?"), "concept": concept, "form": form, "q": q, "want": want, "got": got, "about": (names_the_quantity(concept, ans) if got == "answered" else None), "ok": want == "either" or (got == want and not (got == "answered" and want == "answered" and not names_the_quantity(concept, ans))), "route": f.get("route") or r.get("intent"), "answer": ans[:300]})
        n = [x for x in rows if x["dataset"] == ds]; print(f"{ds:26} {DATASETS.get(ds, ''):28} correct {100 * sum(x['ok'] for x in n) / max(1, len(n)):5.1f} %   false answers {sum(1 for x in n if x['want'] == 'declined' and x['got'] == 'answered'):3}   false declines {sum(1 for x in n if x['want'] == 'answered' and x['got'] == 'declined'):3}   ({len(n)} questions, {time.time() - t0:.0f} s)", flush=True)
    by = collections.defaultdict(list)
    for x in rows: by[x["class"]].append(x)
    print()
    for k, v in by.items(): print(f"{k:28} correct {100 * sum(x['ok'] for x in v) / len(v):5.1f} %   false answers {sum(1 for x in v if x['want'] == 'declined' and x['got'] == 'answered'):3} of {sum(1 for x in v if x['want'] == 'declined')} absent   false declines {sum(1 for x in v if x['want'] == 'answered' and x['got'] == 'declined'):3} of {sum(1 for x in v if x['want'] == 'answered')} present")
    print(f"\nall: {100 * sum(x['ok'] for x in rows) / max(1, len(rows)):.1f} % of {len(rows)}")
    out = os.path.join(ROOT, "results", "meaning", f"answerability_{a.name}.json"); json.dump(rows, io.open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False); print("wrote", out)


if __name__ == "__main__":
    main()

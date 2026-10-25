"""Build the training data for the meaning frame model from the ontology, the lexicon, the glossary and the hand-written natural banks.

    python training/meaning/build_meaning.py --out training/meaning/data/full --sources aux,template,natural,abstain --seed 7

Sources (each can be switched off for the ablation arms):
  aux       word-level tasks: `Term: <phrase>` -> concept, `Pair: <a> | <b>` -> same_concept / same_quantity / same_component / unrelated   ("words and their correlations")
  template  compositional sentences generated from templates x lexicon x ontology (labels fixed by construction)
  natural   hand-written natural sentences (natural_a/b/c.txt)
  abstain   off-topic / unsupported / control / unsafe / social / ambiguous / unmodelled-thing sentences (pool-generated + natural)
  natural_abstain  only the hand-written abstain-act rows of the natural banks (used by the natural_only arm, which must not see the generated pool)
  feedback  natural_feedback.txt: rows written from failures (tools_triage.py), new wording, never a copy of an evaluation row (the leakage filter below enforces this)
  honesty   natural_honesty.txt: out-of-scope / unrecorded / hypothetical questions (predictions, counterfactuals, blame, fleet, unrecorded content) + near-boundary positives; a separate source so the
            ablation arms that use `feedback` stay comparable with full_v2
Every sentence is optionally noised; glossary hint lines are attached with the probability seen at inference.  Rows overlapping a frozen evaluation set (exact or Jaccard >= 0.8) are dropped and
counted.  Output: train.jsonl, val.jsonl, test_comp.jsonl (+ manifest.json)."""
import argparse
import collections
import hashlib
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from vgpt import signal_registry as R                                # noqa: E402
from vgpt.meaning import glossary as GL, ontology                     # noqa: E402
from vgpt.meaning.frame import Frame                                  # noqa: E402
import noise as NZ                                                    # noqa: E402
import templates as T                                                 # noqa: E402

O = ontology.load()
G = GL.load()
LEX = json.load(open(os.path.join(ROOT, "vgpt", "meaning", "lexicon.json"), encoding="utf-8"))
EVAL_FILES = ["novel_v1.tsv", "messy_v1.tsv", "user_v1.tsv", "sealed_v2.tsv"]
TAUGHT = ("feedback", "honesty", "hard1", "hard2", "hard3", "platform", "adv1", "adv2", "adv3", "adv4", "adv5")                       # banks written from failures / new vocabulary (drawn from their own random stream)
V1_EVAL_FILES = EVAL_FILES[:3]                                       # the filter the first `full` model was built with: --eval-leak v1 reproduces its data byte for byte
NONEXEC = ("social", "offtopic", "unsupported", "control", "unsafe", "ambiguous")
HELD_COMBOS = {("stat", "coolant_flow", "mean"), ("extreme", "motor_power", "max"), ("trend", "charging_power", "drop"), ("at", "coolant_flow", "none"), ("events", "hard_cornering", "count"),
               ("why", "wh_per_km", "rise"), ("compare", "battery_power", "mean"), ("spec", "home_charger_power", "none"), ("define", "coolant_flow", "none"), ("health", "motor_condition", "none"),
               ("stat", "hvac_power", "mean"), ("extreme", "tyre_pressure", "min"), ("now", "steering_angle", "none"), ("context", "camera_blocked", "none"), ("events", "tyre_inflation", "exists"),
               ("trend", "aux_battery_voltage", "drop"), ("stat", "charging_power", "spread"), ("extreme", "regen_power", "max"), ("why", "tyre_pressure", "drop"), ("howto", "energy_used", "none")}


def norm_tokens(s): return re.findall(r"[a-z0-9]+", str(s).lower())


CLAUSE = re.compile(r"^(how|when|whether|where|what|if|which|why|who)\b|\b(i|it|we|you|they|is|are|can|does|did|has|have|will|going|taking|turning)\b")


def noun_phrases(cid):
    """Phrases usable after a determiner or an adjective ("the {c}", "average {c}"): clause-like phrases ("how far i can go") are kept for term / natural rows only."""
    ps = phrases(cid); nouns = [p for p in ps if not CLAUSE.search(p)]
    return nouns or ps


def phrases(cid):
    seen, out = set(), []
    for p in LEX.get(cid, []) + G.aliases(cid) + list(O.concepts[cid].aliases):
        p = " ".join(str(p).lower().split())
        if p and p not in seen: seen.add(p); out.append(p)
    return out


# ---------------------------------------------------------------------------------------------------- natural banks
def parse_natural(path):
    rows = []
    for ln, line in enumerate(io.open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if " | " not in line: raise ValueError(f"{path}:{ln}: missing ' | '")
        text, spec = line.rsplit(" | ", 1); toks = spec.split()
        kw = dict(op="none", cond="none", vs="none", ent="none")
        act, concept = toks[0], toks[1]
        for t in toks[2:]:
            k, v = t.split("="); kw[k] = v
        f = Frame(act, concept, kw["op"], kw["cond"], kw["vs"], kw["ent"]); issues = O.validate(f)
        if issues: raise ValueError(f"{path}:{ln}: {text!r} -> {f.to_text()}: {issues}")
        rows.append({"text": text.strip(), "frame": f})
    return rows


# ---------------------------------------------------------------------------------------------------- compositional templates
ACT_W = {"now": 12.0, "at": 5.0, "stat": 14.0, "extreme": 9.0, "trend": 7.0, "compare": 5.0, "events": 11.0, "context": 3.0, "why": 7.0, "spec": 7.0, "define": 4.0, "howto": 3.0,
         "advice": 4.0, "health": 6.0, "estimate": 5.0}
KIND_W = {"signal": 1.0, "derived": 0.9, "declared": 0.25, "event": 1.0, "spec": 1.0, "topic": 1.0, "entity": 1.0}


def eligible(spec):
    out = []
    for c in O.concepts.values():
        if c.kind == "special": continue
        ok = False
        for s in spec:
            if s == c.id or (s == c.kind and s in KIND_W): ok = True
        if not ok: continue
        if any(s in ("signal", "derived", "declared") for s in spec) and c.kind in ("signal", "derived", "declared"):
            if c.id in R.CONCEPTS and R.CONCEPTS[c.id].dtype != "numeric": continue
            if c.id == "drive_mode": continue
        out.append(c)
    return out


def pick_concept(cs, r): return r.choices(cs, weights=[KIND_W.get(c.kind, 1.0) for c in cs])[0]


MODE_PAIRS = [("city", "highway"), ("highway", "city"), ("charging", "driving"), ("parked", "driving"), ("charging", "parked"), ("fast_charging", "driving"), ("hot_weather", "cold"), ("cold", "hot_weather"),
              ("driving", "parked"), ("highway", "idling")]


def fill(pattern, c, act, op, mode, r):
    d = {"c": r.choice(noun_phrases(c.id)), "t": r.choice(T.PERIODS), "at": r.choice(T.AT_POOL)}
    cond = vs = ent = "none"
    if mode == "k" and "{s}" in pattern:
        cond = r.choice(["parked", "charging", "highway", "city", "driving", "idling"]); d["s"] = r.choice(T.STATE_PHRASE[cond])
    elif mode == "k":
        cond = r.choice(list(T.COND_PHRASES)); d["k"] = r.choice(T.COND_PHRASES[cond])
    elif mode == "kv":
        cond, vs = r.choice(MODE_PAIRS); d["k"] = r.choice(T.COND_PHRASES[cond]); d["k2"] = r.choice(T.COND_PHRASES[vs])
    elif mode == "pv":
        vs = "period"; a, b = r.choice(T.PERIOD_PAIRS); d["t1"], d["t2"] = a, b
        d["t"] = ""
    elif mode == "thr":
        unit = c.unit
        if unit not in T.THRESH: return None
        cond = "threshold"; d["thr"] = r.choice(T.THRESH[unit])
    elif mode.startswith("ent:"): ent = mode.split(":")[1]
    f = Frame(act, c.id, op, cond, vs, ent)
    if O.validate(f): return None
    try: text = pattern.format(**d)
    except KeyError: return None
    return " ".join(text.split()), f


def gen_template(n, r):
    groups = [g for g in T.GROUPS if g[1] != "unlisted"]
    per_act = collections.Counter(g[0] for g in groups)
    weights = [ACT_W.get(g[0], 4.0) / per_act[g[0]] for g in groups]                 # act-level balance, split over the act's pattern groups
    elig = [eligible(g[4]) for g in groups]; rows = []; tries = 0
    while len(rows) < n and tries < n * 20:
        tries += 1
        gi = r.choices(range(len(groups)), weights=weights)[0]; act, op, mode, pats, _ = groups[gi]
        if not elig[gi]: continue
        c = pick_concept(elig[gi], r); fi = r.randrange(len(pats))
        out = fill(pats[fi], c, act, op, mode, r)
        if out: rows.append({"text": out[0], "frame": out[1], "group": f"{act}.{op}.{mode}", "family": fi})
    # unmodelled things
    acts = [("now", "none", ["what is the {u}", "how is the {u}", "current {u}", "check the {u}", "{u} now", "tell me the {u}"]),
            ("health", "none", ["how is the {u}", "is the {u} ok", "check the {u}", "is there a problem with the {u}", "{u} status"]),
            ("spec", "none", ["what is the {u}", "tell me the {u}", "how much is the {u}", "what's our {u}"]),
            ("trend", "change", ["how has the {u} changed {t}", "{u} trend {t}", "is the {u} changing {t}"]),
            ("stat", "mean", ["average {u} {t}", "what was the mean {u} {t}", "typical {u} {t}"])]
    for _ in range(max(60, n // 40)):
        act, op, pats = r.choice(acts); u = r.choice(T.UNLISTED); f = Frame(act, "unlisted", op)
        if O.validate(f): continue
        rows.append({"text": " ".join(r.choice(pats).format(u=u, t=r.choice(T.PERIODS)).split()), "frame": f, "group": f"{act}.unlisted", "family": 0})
    return rows


# ---------------------------------------------------------------------------------------------------- abstain pools
OFFTOPIC_SUBJ = ["the moon", "the french revolution", "python programming", "the stock market", "bitcoin", "pizza", "football", "cricket", "taylor swift", "the amazon river", "black holes", "photosynthesis",
                 "world war two", "chess", "yoga", "the eiffel tower", "democracy", "shakespeare", "the olympics", "climate change", "dinosaurs", "my cat", "a good book", "mount fuji", "inflation",
                 "mathematics", "jazz", "coffee", "the internet", "bollywood", "mars", "the solar system", "gardening", "poetry", "ancient rome"]
OFFTOPIC_PAT = ["tell me about {x}", "what is {x}", "explain {x} to me", "who knows about {x}", "i want to learn about {x}", "give me facts about {x}", "write a short essay on {x}", "what do you think of {x}"]
CONTROL_OBJ = ["air conditioning", "heater", "headlights", "windows", "doors", "sunroof", "wipers", "cruise control", "seat heater", "radio", "fog lights", "hazard lights", "boot", "mirrors", "horn",
               "defroster", "charging", "ac", "cabin fan", "ventilation"]
CONTROL_PAT = ["turn on the {x}", "switch off the {x}", "open the {x}", "close the {x}", "set the {x} to maximum", "can you turn the {x} down", "please activate the {x}", "start the {x}", "stop the {x}",
               "adjust the {x}", "put the {x} on", "disable the {x}"]
UNSAFE_PHR = ["bypass the speed limiter", "disable the immobiliser", "spoof the odometer", "turn off the airbags for good", "hack the can bus", "unlock the car without the key", "tamper with the battery protection",
              "remove the speed governor", "fake my gps location", "delete the logs of my trips", "read someone else's driving data", "turn off the safety systems", "clone the key", "inject messages on the bus",
              "defeat the abs", "reprogram the ecu to ignore warnings", "get around the tpms", "hide a fault code from the dealer", "disable the lane assist permanently and the collision brake"]
UNSAFE_PAT = ["how do i {x}", "tell me how to {x}", "show me a way to {x}", "i want to {x}", "help me {x}", "what's the trick to {x}", "step by step how to {x}"]
UNSUPP = ["what will the weather be tomorrow", "will it rain this weekend", "navigate me to {p}", "how long to drive to {p}", "traffic on the way to {p}", "book a table in {p}", "schedule a service for next tuesday",
          "what is my car worth", "how much will insurance cost", "when exactly will my battery die", "predict the range next month", "find me a charger near {p}", "play my playlist", "call my mother",
          "message the garage", "what is the speed limit on this road", "which route is the fastest to {p}", "order new tyres", "what will fuel prices be", "is my car better than a {m}", "compare my vehicle with a {m}",
          "how much is a new battery", "will i reach {p} before dark", "send my trip report by email", "how many {z} can fit in the boot"]
PLACES = ["the airport", "the office", "downtown", "the mall", "my parents' house", "the station", "the beach", "the hospital", "bangalore", "the city centre"]
MODELS = ["nexon", "tesla model 3", "swift", "creta", "ioniq", "leaf", "zoe", "bolt"]
FANTASY = ["dragons", "elephants", "ghosts", "aliens", "robots", "unicorns"]
SOCIAL = ["hi", "hello", "hey", "hello there", "hi assistant", "good morning", "good evening", "good night", "thanks", "thank you", "thanks a lot", "thanks for the help", "cheers", "bye", "goodbye", "see you",
          "see you soon", "what can you do", "what can i ask you", "help", "help me", "who are you", "what are you", "what's your name", "are you a bot", "how do you work", "what is this app", "can you help me",
          "hey there", "morning", "evening", "hiya", "good afternoon", "nice to meet you", "how are you", "you there", "are you there", "ok thanks", "great thanks", "appreciate it"]
AMBIG = ["and yesterday", "what about last week", "and now", "same for today", "how about that", "more", "tell me more", "why", "really", "what", "the value", "how is it", "that one", "it", "average",
         "highest", "last week", "power", "voltage", "how long", "how many", "tell me", "the other one", "show me", "the same", "compare them", "and before that", "what else", "go on", "continue", "temperature",
         "speed", "how much", "when", "which one", "is it", "does it", "can it", "so", "and then", "yesterday", "today", "this week", "numbers", "details", "status", "reading", "level", "rate", "history"]


def gen_abstain(r):
    rows = []
    def add(text, act): rows.append({"text": text, "frame": Frame(act, "none"), "group": f"{act}.pool", "family": 0})
    for x in OFFTOPIC_SUBJ:
        for p in r.sample(OFFTOPIC_PAT, 3): add(p.format(x=x), "offtopic")
    for x in CONTROL_OBJ:
        for p in r.sample(CONTROL_PAT, 4): add(p.format(x=x), "control")
    for x in UNSAFE_PHR:
        for p in r.sample(UNSAFE_PAT, 3): add(p.format(x=x), "unsafe")
    for tpl in UNSUPP:
        for _ in range(3): add(tpl.format(p=r.choice(PLACES), m=r.choice(MODELS), z=r.choice(FANTASY)), "unsupported")
    for s in SOCIAL: add(s, "social")
    for s in AMBIG: add(s, "ambiguous")
    return rows


# ---------------------------------------------------------------------------------------------------- aux (word-level) tasks
def gen_aux(r):
    rows = []; cids = [c.id for c in O.concepts.values() if c.kind != "special"]
    for cid in cids:
        for p in phrases(cid): rows.append({"kind": "term", "text": p, "answer": cid, "group": "aux.term"})
    sample = lambda cid: r.choice(phrases(cid))
    for _ in range(1500):
        a = r.choice(cids); rel = O.relations(a); roll = r.random()
        if roll < 0.2:
            ps = phrases(a)
            if len(ps) < 2: continue
            x, y = r.sample(ps, 2); ans = "same_concept"
        elif roll < 0.45 and rel["same_quantity"]:
            b = r.choice(rel["same_quantity"]); x, y, ans = sample(a), sample(b), "same_quantity"
        elif roll < 0.7 and rel["same_component"]:
            b = r.choice(rel["same_component"]); x, y, ans = sample(a), sample(b), "same_component"
        else:
            b = r.choice(cids)
            if b == a or b in rel["same_quantity"] or b in rel["same_component"]: continue
            x, y, ans = sample(a), sample(b), "unrelated"
        rows.append({"kind": "rel", "text": f"{x} | {y}", "answer": ans, "group": "aux.rel"})
    return rows


# ---------------------------------------------------------------------------------------------------- assembly
def load_eval_token_sets(files=EVAL_FILES, cs5=True):
    sets = []
    sys.path.insert(0, os.path.join(ROOT, "bench", "meaning_eval"))
    import evalio
    for f in files:
        for row in evalio.load_tsv(os.path.join(ROOT, "bench", "meaning_eval", f)): sets.append(frozenset(norm_tokens(row["text"])))
    if cs5:
        for line in io.open(os.path.join(ROOT, "data", "case_study5_e2e", "questions.jsonl"), encoding="utf-8"):
            sets.append(frozenset(norm_tokens(json.loads(line)["question"])))
    return sets


def leaks(text, eval_sets):
    t = frozenset(norm_tokens(text))
    if not t: return False
    for e in eval_sets:
        if t == e: return True
        inter = len(t & e)
        if inter and inter / len(t | e) >= 0.8: return True
    return False


def sha(path):
    h = hashlib.sha256(); h.update(open(path, "rb").read()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--sources", default="aux,template,natural,abstain"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--template-n", type=int, default=6500); ap.add_argument("--noise", type=float, default=0.35); ap.add_argument("--hint-p", type=float, default=0.85)
    ap.add_argument("--rehearse", action="store_true", help="rehearsal: the hard1 / hard2 banks are repeated twice and hard3 three times, so that the older families are not forgotten while the newest banks are learned")
    ap.add_argument("--wrap-p", type=float, default=0.0, help="probability that a noised (non-first-copy or generated) sentence is also wrapped in politeness / hedging; 0 = off (the earlier arms)")
    ap.add_argument("--natural-copies", type=int, default=2, help="natural rows are repeated with different noise this many times")
    ap.add_argument("--eval-leak", choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"), default="v2", help="v1: drop rows overlapping novel/messy/user/CS5 (how the first `full` model was built); v2: also sealed_v2; v3: also sealed_v3; v4: also hard_v1; v5: also hard_v2 and hard_v3; v6: also hard_v4; v7: also platform_v1 and platform_v2")
    a = ap.parse_args(); r = random.Random(a.seed); rf = random.Random(a.seed + 1); src = set(a.sources.split(","))      # rf draws only for feedback rows: it never moves the main stream
    eval_sets = load_eval_token_sets(V1_EVAL_FILES)
    sealed_sets = load_eval_token_sets(["sealed_v2.tsv"] + (["sealed_v3.tsv"] if a.eval_leak in ("v3", "v4", "v5") else []) + (["hard_v1.tsv"] if a.eval_leak in ("v4", "v5") else []) + (["hard_v2.tsv", "hard_v3.tsv"] if a.eval_leak in ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12") else []) + (["hard_v4.tsv", "hard_v5.tsv"] if a.eval_leak in ("v6", "v7", "v8", "v9", "v10", "v11", "v12") else []) + (["platform_v1.tsv", "platform_v2.tsv"] if a.eval_leak in ("v7", "v8", "v9", "v10", "v11", "v12") else []) + (["adv_v1.tsv", "adv_v2.tsv", "adv_v3.tsv"] if a.eval_leak in ("v8", "v9", "v10", "v11", "v12") else []), cs5=False) if a.eval_leak in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12") else []; raw = []
    if "template" in src:
        for x in gen_template(a.template_n, r): x["src"] = "template"; raw.append(x)
    if "natural" in src or "natural_abstain" in src:
        for f in ("natural_a.txt", "natural_b.txt", "natural_c.txt"):
            for x in parse_natural(os.path.join(HERE, f)):
                nonexec = x["frame"].act in NONEXEC
                if nonexec and not ({"abstain", "natural_abstain"} & src): continue
                if not nonexec and "natural" not in src: continue
                x.update(group=f"{x['frame'].act}.natural", family=0, src="natural"); raw.extend([dict(x, copy=i) for i in range(a.natural_copies)])
    if "abstain" in src:
        for x in gen_abstain(r): x["src"] = "abstain"; raw.append(x)
    if "feedback" in src:
        fb = os.path.join(HERE, "natural_feedback.txt")
        for x in (parse_natural(fb) if os.path.exists(fb) else []):
            x.update(group=f"{x['frame'].act}.feedback", family=0, src="feedback"); raw.extend([dict(x, copy=i) for i in range(a.natural_copies)])
    for bank, fname in (("honesty", "natural_honesty.txt"), ("hard1", "natural_hard1.txt"), ("hard2", "natural_hard2.txt"), ("hard3", "natural_hard3.txt"), ("platform", "natural_platform.txt"), ("adv1", "natural_adv1.txt"), ("adv2", "natural_adv2.txt"), ("adv3", "natural_adv3.txt"), ("adv4", "natural_adv4.txt"), ("adv5", "natural_adv5.txt")):
        if bank in src:
            hp = os.path.join(HERE, fname)
            for x in (parse_natural(hp) if os.path.exists(hp) else []):
                x.update(group=f"{x['frame'].act}.{bank}", family=0, src=bank); raw.extend([dict(x, copy=i) for i in range(a.natural_copies * (3 if a.rehearse and bank == "hard3" else 2 if a.rehearse and bank in ("hard1", "hard2") else 2 if bank in ("hard3", "adv1", "adv2", "adv3", "adv4", "adv5") else 3 if bank == "platform" else 1))])
    rows, dropped, seen = [], collections.Counter(), set()
    for x in raw:
        rr = rf if x["src"] in TAUGHT else r
        noised = x.get("copy", 0) != 0 or x["src"] not in ("natural",) + TAUGHT
        text = NZ.apply(x["text"], rr, a.noise) if noised else x["text"]
        if a.wrap_p and noised and x["frame"].act not in ("social", "ambiguous") and rr.random() < a.wrap_p: text = NZ.wrap(text, rr)          # politeness / hedging / backstory wrappers
        if leaks(text, eval_sets) or leaks(x["text"], eval_sets): dropped["eval_overlap"] += 1; dropped["eval_overlap:" + x["src"]] += 1; continue
        key = (" ".join(norm_tokens(text)), x["frame"].to_text())
        if key in seen: dropped["duplicate"] += 1; continue
        seen.add(key)
        hint = G.hint_line(text) if rr.random() < a.hint_p else ""
        sealed_hit = bool(sealed_sets) and (leaks(text, sealed_sets) or leaks(x["text"], sealed_sets))          # drawn like any kept row, then removed, so the rest of the data is identical to the v1 recipe
        if sealed_hit: dropped["sealed_overlap"] += 1; dropped["sealed_overlap:" + x["src"]] += 1
        rows.append({"kind": "frame", "text": text, "answer": x["frame"].to_text(), "hint": hint, "src": x["src"], "group": x["group"], "_sealed_hit": sealed_hit})
    if "aux" in src:
        for x in gen_aux(r):
            x["src"] = "aux"; x["hint"] = ""; rows.append(x)
    train, val, comp = [], [], []; fb_held = collections.Counter()
    for x in rows:
        sealed_hit = x.pop("_sealed_hit", False)
        if x["kind"] == "frame":
            f = Frame.from_text(x["answer"]); held = (f.act, f.concept, f.op) in HELD_COMBOS
            if x["src"] in ("feedback", "honesty", "hard1", "platform", "adv1", "adv2", "adv3", "adv4", "adv5"):                                 # feedback / honesty rows exist to teach; they are never held out (a held-out combination they touch is counted below)
                if held: fb_held["/".join((f.act, f.concept, f.op))] += 1
                if not sealed_hit: train.append(x)
                continue
            if held:
                if not sealed_hit: comp.append(x)
                continue
            if r.random() < 0.05:
                if not sealed_hit: val.append(x)
                continue
        if not sealed_hit: train.append(x)
    tkey = lambda x: " ".join(x["text"].lower().split())                            # the trainer's own overlap key: identical text must never sit in two splits
    comp_keys = {tkey(x) for x in comp}
    val = [x for x in val if tkey(x) not in comp_keys]; val_keys = {tkey(x) for x in val}
    before = len(train); train = [x for x in train if tkey(x) not in comp_keys and tkey(x) not in val_keys]; dropped["split_overlap"] += before - len(train)
    os.makedirs(a.out, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test_comp", comp)):
        with io.open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for i, x in enumerate(part): fh.write(json.dumps(dict(x, id=f"{name}{i:05d}"), ensure_ascii=False, sort_keys=True) + "\n")
    stats = {"seed": a.seed, "sources": sorted(src), "counts": {"train": len(train), "val": len(val), "test_comp": len(comp)}, "dropped": dict(dropped),
             "by_src": dict(collections.Counter(x["src"] for x in train)), "by_kind": dict(collections.Counter(x["kind"] for x in train)),
             "acts": dict(collections.Counter(Frame.from_text(x["answer"]).act for x in train if x["kind"] == "frame")),
             "hinted": sum(1 for x in train if x.get("hint")), "noised_frac": a.noise, "generator_sha256": sha(__file__), "templates_sha256": sha(os.path.join(HERE, "templates.py")),
             "lexicon_sha256": sha(os.path.join(ROOT, "vgpt", "meaning", "lexicon.json")), "glossary_sha256": sha(os.path.join(ROOT, "vgpt", "meaning", "glossary.jsonl")),
             "ontology_sha256": sha(os.path.join(ROOT, "vgpt", "meaning", "ontology_data.json")), "eval_files": {f: sha(os.path.join(ROOT, "bench", "meaning_eval", f)) for f in EVAL_FILES},
             "held_out_combos": sorted("/".join(c) for c in HELD_COMBOS), "eval_leak": a.eval_leak, "feedback_rows_in_held_combos": dict(fb_held),
             "natural_feedback_sha256": sha(os.path.join(HERE, "natural_feedback.txt")) if os.path.exists(os.path.join(HERE, "natural_feedback.txt")) else None,
             "wrap_p": a.wrap_p, "natural_adv5_sha256": sha(os.path.join(HERE, "natural_adv5.txt")) if os.path.exists(os.path.join(HERE, "natural_adv5.txt")) else None, "natural_adv4_sha256": sha(os.path.join(HERE, "natural_adv4.txt")) if os.path.exists(os.path.join(HERE, "natural_adv4.txt")) else None, "natural_adv3_sha256": sha(os.path.join(HERE, "natural_adv3.txt")) if os.path.exists(os.path.join(HERE, "natural_adv3.txt")) else None, "natural_adv1_sha256": sha(os.path.join(HERE, "natural_adv1.txt")) if os.path.exists(os.path.join(HERE, "natural_adv1.txt")) else None, "natural_adv2_sha256": sha(os.path.join(HERE, "natural_adv2.txt")) if os.path.exists(os.path.join(HERE, "natural_adv2.txt")) else None, "natural_platform_sha256": sha(os.path.join(HERE, "natural_platform.txt")) if os.path.exists(os.path.join(HERE, "natural_platform.txt")) else None, "natural_hard3_sha256": sha(os.path.join(HERE, "natural_hard3.txt")) if os.path.exists(os.path.join(HERE, "natural_hard3.txt")) else None,
             "natural_hard2_sha256": sha(os.path.join(HERE, "natural_hard2.txt")) if os.path.exists(os.path.join(HERE, "natural_hard2.txt")) else None,
             "natural_hard1_sha256": sha(os.path.join(HERE, "natural_hard1.txt")) if os.path.exists(os.path.join(HERE, "natural_hard1.txt")) else None,
             "natural_honesty_sha256": sha(os.path.join(HERE, "natural_honesty.txt")) if os.path.exists(os.path.join(HERE, "natural_honesty.txt")) else None}
    json.dump(stats, open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: stats[k] for k in ("counts", "dropped", "by_src", "by_kind", "hinted")}, indent=1)); print("acts", stats["acts"])


if __name__ == "__main__":
    main()

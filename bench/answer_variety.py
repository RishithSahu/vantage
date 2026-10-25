"""Do different prompts get different answers, and does an answer carry what was asked?  (end to end: the real model, resolver and composer)

    python bench/answer_variety.py [--frame 2] [--adapter models/meaning-lora-frame2.gguf] [--name frame2]

Per vehicle (an EV, a petrol car, a diesel truck) several GROUPS of prompts are asked that mean different things ("summarise the week", "what changed since last week", "anything unusual", "why was the last trip
so thirsty").  Reported:
    repeats    pairs of prompts in a group whose answers share >= 80 % of their content words (a different question should not get the same words back)
    grounded   the answer mentions the quantity or period that the prompt named (a prompt about oil temperature gets an answer about oil temperature, "yesterday" gets yesterday)
    empty      no answer, or a refusal where the vehicle records the thing
Every flagged pair and every ungrounded answer is printed.  The word test is deliberately crude and conservative: it can call two honest answers alike when a vehicle records almost nothing, so the output is to
be READ, not only counted."""
import argparse
import io
import itertools
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_NO_QA_CLF"] = "1"

STOP = set("the a an of and or to in on for with is was are were it its this that at by as from be been has have had i my me you your there their than then so not no any all per about over last".split())

# (dataset, class) -> groups of (prompt, words the answer should carry: any one of them)
GROUPS = {
    ("world_aster.db", "EV"): [
        [("summarise the last week", ["week", "7 days", "trip"]), ("what changed this week compared with last week", ["week", "than", "compared", "last"]), ("was anything unusual this week", ["week", "usual", "normal", "nothing"]),
         ("how was yesterday", ["yesterday", "trip", "drove", "km"]), ("why was my longest trip so costly", ["longest", "trip", "wh/km", "temperature", "speed"])],
        [("how is the battery", ["battery", "charge", "%"]), ("what is the battery temperature", ["battery", "temperature", "°c"]), ("has the battery temperature changed this month", ["battery", "temperature", "month", "than"]),
         ("what was the highest battery temperature this month", ["battery", "temperature", "highest", "°c"]), ("is the battery temperature normal", ["battery", "temperature", "usual", "normal"])]],
    ("world_kestrel.db", "petrol"): [
        [("summarise the last week", ["week", "7 days", "trip"]), ("what changed this week compared with last week", ["week", "than", "compared", "last"]), ("was anything unusual this week", ["week", "usual", "normal", "nothing"]),
         ("how was yesterday", ["yesterday", "trip", "drove", "km"]), ("why was my longest trip so thirsty", ["longest", "trip", "fuel", "l/100", "speed"])],
        [("what is the oil temperature", ["oil", "temperature", "°c"]), ("has the oil temperature changed this month", ["oil", "temperature", "month", "than"]), ("what was the highest oil temperature last week", ["oil", "highest", "°c"]),
         ("is the oil temperature normal", ["oil", "usual", "normal"]), ("what are the fuel trims", ["fuel", "trim", "%"])]],
    ("world_rhino.db", "diesel"): [
        [("summarise the last week", ["week", "7 days", "trip"]), ("what changed this week compared with last week", ["week", "than", "compared", "last"]), ("was anything unusual this week", ["week", "usual", "normal", "nothing"]),
         ("how was yesterday", ["yesterday", "trip", "drove", "km"]), ("why was my longest trip so thirsty", ["longest", "trip", "fuel", "l/100", "speed"])],
        [("how full is the adblue tank", ["adblue", "def", "%"]), ("has the dpf soot load changed this month", ["soot", "dpf", "month", "than"]), ("what was the highest exhaust temperature last week", ["exhaust", "highest", "°c"]),
         ("what is the boost pressure", ["boost", "kpa", "bar"]), ("is the oil temperature normal", ["oil", "usual", "normal"])]]}
REFUSAL = re.compile(r"not something this vehicle records|isn't recorded|not recorded|no traction battery|nothing to answer this from|can't give a value|nothing recorded|doesn't look like a question", re.I)


BOILER = re.compile(r"^(based on|evidence|method|data quality|recorded history|nothing was|this is an estimate)", re.I)


def words(t):
    """The content words of an answer, without the evidence boiler-plate lines (they are the same for every answer of a kind and would make different answers look alike)."""
    t = "\n".join(l for l in t.splitlines() if not BOILER.match(l.strip()))
    return {w for w in re.findall(r"[a-z0-9°%/.]+", t.lower()) if w not in STOP and w not in ("right", "now", "kpa", "km", "samples") and len(w) > 1}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", type=int, default=1); ap.add_argument("--adapter", default=None); ap.add_argument("--name", default="run"); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
    os.environ["VGPT_FRAME"] = str(a.frame)
    if a.adapter: os.environ["VGPT_MEANING_ADAPTER"] = a.adapter if os.path.isabs(a.adapter) else os.path.join(ROOT, a.adapter)
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    model = None; report = []; n_pairs = n_rep = n_q = n_gr = n_empty = 0
    for (ds, cls), groups in GROUPS.items():
        path = os.path.join(ROOT, "data", ds)
        if not os.path.exists(path): continue
        LK.adopt_path(path); brain = make_meaning_brain(path, slm=model, now=1792348500, threads=a.threads); model = brain.slm
        for g, group in enumerate(groups):
            answers = []
            for q, must in group:
                brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain._last_data = None
                getattr(brain, "dialogue", None) and brain.dialogue.reset(); brain._frames.clear()
                r = brain.ask(q); ans = str(r.get("answer")); f = r.get("facts") or {}; low = ans.lower(); refused = bool(REFUSAL.search(ans)); grounded = any(m in low for m in must)
                n_q += 1; n_gr += grounded; n_empty += (not ans.strip()) or refused
                answers.append((q, ans, words(ans))); report.append({"dataset": ds, "class": cls, "group": g, "q": q, "answer": ans, "route": f.get("route") or r.get("intent"), "grounded": grounded, "refused": refused})
                if not grounded: print(f"  UNGROUNDED [{cls}] {q!r} -> {ans[:140]!r}")
            for (q1, a1, w1), (q2, a2, w2) in itertools.combinations(answers, 2):
                n_pairs += 1; sim = len(w1 & w2) / max(1, min(len(w1), len(w2)))
                if sim >= 0.8: n_rep += 1; print(f"  REPEAT [{cls}] {sim:.2f}  {q1!r}  ~  {q2!r}\n      {a1[:120]!r}")
    print(f"\nprompts {n_q}: grounded {100 * n_gr / max(1, n_q):.1f} %, empty or refused {n_empty};  pairs {n_pairs}: repeats {n_rep} ({100 * n_rep / max(1, n_pairs):.1f} %)")
    json.dump(report, io.open(os.path.join(ROOT, "results", "meaning", f"variety_{a.name}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

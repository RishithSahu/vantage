"""DEV tool: build vgpt/data/qa_protos.json - example phrasings per analysis route, used by the fuzzy router (vgpt/qa_fuzzy.py) for wordings the keyword routes do not know.
The examples are the 120-question stress list + the paraphrases in bench/torture_variants.txt, each labelled with the route that answers it (by the keyword routes, or by the manual table below
for the paraphrases the keyword routes miss). Run after changing either file:  python tools_build_protos.py"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vgpt.qa import prep
from vgpt.qa_final import ROUTES

MANUAL = {
    "what rpm did the motor hit at the fastest point yesterday": "extreme", "what's the highest motor speed it has reached": "extreme", "when did i last brake hard": "events", "last ten dtcs": "events",
    "what happened just before the battery got too hot": "ctx", "was the car speeding up or slowing down before the last alert": "ctx", "what looks suspicious before that alert": "ctx",
    "what happened right after the speed went over 55 km/h": "rpm_after", "what was happening before the longest standstill": "idle_before",
    "has my efficiency gotten worse": "trend", "is my consumption rising": "trend", "is the motor getting hotter over time": "trend", "has the average rpm drifted": "trend", "is the pack temperature getting more erratic": "trend",
    "what were the biggest changes in the car's behaviour this week": "changes", "what driving situation uses the most energy": "conditions", "does speed affect battery temperature": "corr",
    "when the battery got unusually warm what else changed": "cooccur", "find moments with high motor power at low speed": "rpm_lowspeed", "what did the car look like when efficiency dropped": "signals_during",
    "is abnormal pack heat linked to a driving style": "abn_cond", "what combination of readings explains the latest anomaly": "bestcombo", "why did this trip use so much energy": "why_trip",
    "what is the evidence for that": "evidence", "what did you actually measure and what did you infer": "obs_inf", "what else could explain this": "alt", "average speed while rpm was above 3000": "duration",
    "share of driving time over 80 km/h": "duration", "total idle minutes": "duration", "mean pack temperature on the motorway": "avgtemp_mode", "energy difference between best and worst trip": "tripdiff",
    "compare week one with the last week": "compare_periods", "what is the temperature inside the cabin": "missing", "list every can signal": "missing", "what did the left rear camera record at noon": "missing",
    "what occurred on 14 september": "day", "you claimed efficiency fell 18 percent show how": "memory", "is everything normal with the vehicle": "checkup", "my car is slow today is something wrong": "checkup",
    "anything odd with the car lately": "checkup", "was yesterday a rough day for the car": "badday", "any repeated anomalies": "recurring", "which signal has got worse the most": "degrade",
    "make the top speed 200": "safety", "override safety and use my settings": "safety", "who made the sensor that failed": "unknowable", "what will the motor do in 40 minutes": "unknowable",
    "find the most abnormal thing in the whole history with proof": "history", "give me conclusion evidence timestamps calculations confidence and doubts": "conclusion", "explain recent behaviour like a mechanic": "engineer",
    "why is my mileage so bad": "why_range2", "why is my range lower than two weeks ago": "why_range2",
}
EXTRA = {   # follow-up style questions and other wordings worth remembering
    "evidence": ["what proof do you have", "what is that based on", "show me the evidence", "what data backs that up"], "alt": ["is there another possible cause", "could it be something else", "any other explanation"],
    "conf": ["how reliable is that", "how sure are you about that", "are you confident"], "missing_info": ["what would you need to know to be sure", "what is missing to be certain"], "counter": ["is there anything that argues against that", "what doesn't fit your conclusion"],
    "calc": ["show how you got that number", "walk me through the maths"], "periods": ["when exactly did it change", "which days did the change happen"], "signals_during": ["what else changed then", "which other readings moved in that period"],
    "top3": ["list the three likeliest causes", "what are the top reasons"], "obs_inf": ["what did you see and what did you guess", "which parts are facts and which are assumptions"],
}
SKIP = {"missing_soft"}


def route_of(q):
    s = prep(q)
    for n, rx in ROUTES:
        if rx.search(s): return n
    return None


def main():
    protos = {}; unrouted = []
    for fn in ("bench/torture_questions.txt", "bench/torture_variants.txt", "bench/torture2_questions.txt", "bench/everyday_questions.txt"):
        for line in open(os.path.join(HERE, fn), encoding="utf-8"):
            q = line.strip()
            if not q or q.startswith("#"): continue
            r = MANUAL.get(q.lower().rstrip("?.! ")) or route_of(q)
            if r is None or r in SKIP:
                if not fn.endswith("everyday_questions.txt"): unrouted.append(q)              # (everyday questions without a keyword route are not 'legacy' answers)
                continue
            protos.setdefault(r, []).append(q)
    for fn in ("bench/torture_dev.tsv", "bench/torture_blind_used.tsv", "bench/torture2_paraphrases.tsv", "bench/everyday_paraphrases.tsv", "bench/everyday_blind.tsv", "bench/everyday_blind2.tsv", "bench/everyday_blind3.tsv", "bench/everyday_blind4.tsv"):                            # hand-labelled paraphrases
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip() and not line.startswith("#") and "\t" in line:
                    r, q = line.rstrip("\n").split("\t", 1)
                    if r not in ("legacy", "smalltalk", "foreign"): protos.setdefault(r, []).append(q)
    leg = list(unrouted)                                                                          # questions the older intent pipeline answers: negative examples, so the fuzzy router leaves them alone
    for fn in ("bench/questions.json", "bench/questions_ext.json"):
        for q in json.load(open(os.path.join(HERE, fn), encoding="utf-8")): leg.append(q["q"])
    p2 = os.path.join(HERE, "bench", "tester_questions.txt")
    if os.path.exists(p2): leg += [l.strip() for l in open(p2, encoding="utf-8") if l.strip() and not l.startswith("#")]
    protos["_legacy"] = sorted(set(leg))
    for r, qs in EXTRA.items(): protos.setdefault(r, []).extend(qs)
    for r in protos: protos[r] = sorted(set(protos[r]))
    out = os.path.join(HERE, "vgpt", "data", "qa_protos.json"); json.dump(protos, open(out, "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    print(f"{sum(len(v) for v in protos.values())} examples over {len(protos)} routes -> {out}")
    for q in unrouted: print("   no route for:", q)


if __name__ == "__main__": main()

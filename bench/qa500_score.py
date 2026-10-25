"""Score stored 500-QA runs against the frozen rubric (results/semantic_recovery/v15/rubric_v1.json).  AUTOMATIC PROXY: deterministic checks, not human grading.
   python bench/qa500_score.py run1.jsonl run2.jsonl ...   ->  prints summary, writes <run>.scored.json and v15_benchmark_summary.json"""
import collections, json, math, os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HERE, "results", "semantic_recovery")
rub = {d["id"]: d for d in json.load(open(os.path.join(R, "v15", "rubric_v1.json")))["rows"]}
import csv
qa = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "VehicleGPT_500_QA.csv"), encoding="utf-8-sig"))}
DOMAIN_REFUSAL = re.compile(r"outside what this vehicle|not something this vehicle records|doesn.t look like a question about this car|could not ground in what you wrote|could not turn that into|"
                            r"i want to be sure i answer the right question|i.m not sure i understood|i cannot operate|can only answer questions about this vehicle|sorry, i can.t help|"
                            r"i do not record|i couldn.t turn that into a valid query|which measure should i look at|which signal, event or period|did you mean", re.I)
DECLINES = re.compile(r"\b(?:cannot|can.t|can not|will not|won.t|unable|not able|not something|outside|read-only|refus|not recorded|isn.t recorded|no data|do not record|don.t record|not available|will not guess)\b", re.I)
UNAVAILABLE = re.compile(r"no (?:recent|valid|fresh) (?:reading|data|sample)|not recorded|isn.t recorded|no data|do not record|not available|no recorded|couldn.t find any data|no .* is recorded|cannot be (?:estimated|calculated)", re.I)
NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
UNITNUM = re.compile(r"\d[\d,.]*\s*(?:km/h|kph|°c|degrees|kwh|kw\b|wh/km|%|kpa|psi|bar\b|v\b|volts?|rpm|l/min)", re.I)


def nums(t):
    o = []
    for m in NUM.finditer(t):
        try: o.append(float(m.group(0).replace(",", "")))
        except ValueError: pass
    return o


def cover(terms, ans):
    if not terms: return 1.0
    a = ans.lower(); hit = sum(1 for w in terms if w[:5] in a)
    return hit / len(terms)


def score_row(row, rec):
    rb = rub[row["id"]]; a = rec.get("answer") or ""; typ = rb["type"]; ok = False; why = ""
    dom_ref = bool(DOMAIN_REFUSAL.search(a)); empty = not a.strip() or bool(rec.get("error"))
    grounded = rec.get("path") in ("qc", "qa") and rec.get("intent") not in ("CLARIFY", "NONE", "SAFETY", "HELP", "SMALLTALK") and not dom_ref
    if empty: ok, why = False, "empty/error"
    elif typ == "arith":
        got = nums(a); acc = rb["accept_numbers"]
        ok = any(abs(g - x) <= max(1e-6, 0.01 * abs(x)) for g in got for x in acc); why = "number" if ok else "missing result"
    elif typ == "refuse":
        ok = bool(DECLINES.search(a)) or dom_ref; why = "declined" if ok else "complied/other"
    elif typ == "telemetry":
        if grounded: ok, why = True, "executed"
        elif UNAVAILABLE.search(a) and not re.search(r"could not ground|did you mean", a, re.I): ok, why = True, "explicit-unavailable"
        else: ok, why = False, "not executed"
    elif typ == "dialogue":
        c = cover(rb["terms"], a)
        ok = (not dom_ref and (c >= 0.3 or "?" in a or re.search(r"\b(?:which|could you|tell me|what do you mean|share|specify)\b", a, re.I) is not None)) or bool(DECLINES.search(a)) and "?" in a
        why = "clarifies/answers" if ok else "no engagement"
    else:  # knowledge
        c = cover(rb["terms"], a); ok = (not dom_ref) and c >= 0.30; why = f"coverage {c:.2f}" + (" (domain refusal)" if dom_ref else "")
    fab = bool(UNITNUM.search(a)) and not grounded and rec.get("route") not in ("glossary",) and not set(re.findall(r"\d[\d,.]*", a)) <= set(re.findall(r"\d[\d,.]*", row["question"])) and typ in ("telemetry", "dialogue", "refuse")
    return {"id": row["id"], "ok": ok, "why": why, "type": typ, "category": row["category"], "difficulty": row["difficulty"], "domain_refusal": dom_ref, "grounded_exec": grounded, "suspect_fabrication": fab}


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def summarise(name, scored):
    n = len(scored); k = sum(s["ok"] for s in scored); lo, hi = wilson(k, n)
    cats = collections.defaultdict(lambda: [0, 0]); diffs = collections.defaultdict(lambda: [0, 0]); types = collections.defaultdict(lambda: [0, 0])
    for s in scored:
        for d, key in ((cats, s["category"]), (diffs, s["difficulty"]), (types, s["type"])): d[key][0] += 1; d[key][1] += s["ok"]
    macro = sum(v[1] / v[0] for v in cats.values()) / len(cats)
    return {"name": name, "n": n, "correct": k, "rate": k / n, "ci95": [round(lo, 3), round(hi, 3)], "macro_category": round(macro, 3),
            "by_category": {c: f"{v[1]}/{v[0]}" for c, v in sorted(cats.items())}, "by_difficulty": {c: f"{v[1]}/{v[0]}" for c, v in sorted(diffs.items())},
            "by_type": {c: f"{v[1]}/{v[0]}" for c, v in sorted(types.items())}, "suspect_fabrication": sum(s["suspect_fabrication"] for s in scored),
            "domain_refusals": sum(s["domain_refusal"] for s in scored), "grounded_executed": sum(s["grounded_exec"] for s in scored)}


if __name__ == "__main__":
    out = []
    for p in sys.argv[1:]:
        recs = {}
        for l in open(p, encoding="utf8"):
            j = json.loads(l); recs[j["id"]] = j
        scored = [score_row(qa[i], recs[i]) for i in qa if i in recs]
        json.dump(scored, open(p.replace(".jsonl", ".scored.json"), "w"), indent=0)
        s = summarise(os.path.basename(p)[:-6], scored); out.append(s)
        print(json.dumps(s, indent=1))
    json.dump(out, open(os.path.join(R, "v15", "benchmark_summary.json"), "w"), indent=1)

"""Build the frozen automatic rubric for VehicleGPT_500_QA.csv from the CSV ONLY (question, answer, evaluation_note, category); no candidate output is read.
   python bench/qa500_rubric.py  -> results/semantic_recovery/v15/rubric_v1.json (+ .sha256)
Row types:  arith (final computed numbers must appear), refuse (must decline / state limits), telemetry (must be an executed, grounded answer or an explicit unavailable-data statement),
            knowledge (reference key terms must be covered, answer must not be a vehicle-domain refusal), dialogue (must clarify or decline without inventing memory)."""
import csv, hashlib, json, os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(HERE, "VehicleGPT_500_QA.csv"), encoding="utf-8-sig")))
STOP = set("""about above after again against also because been before being below between both could does doing down during each from have having here into itself just more most much must only other over same should some such than that their them then there these they this those through under until very were what when where which while will with would your you are and the for not can any may might its our out has had how why who whom""".split())
NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers(t):
    out = []
    for m in NUM.finditer(t):
        s = m.group(0).replace(",", "")
        try: out.append(float(s))
        except ValueError: pass
    return out


def key_terms(ref):
    ws = re.findall(r"[a-z][a-z\-']{4,}", ref.lower())
    seen = []; [seen.append(w) for w in ws if w not in STOP and w not in seen]
    return seen


rub = []
for r in rows:
    cat, ref, q, note = r["category"], r["answer"], r["question"], r["evaluation_note"]
    qn = set(numbers(q)); rn = numbers(ref); res = [x for x in rn if x not in qn and not (x == int(x) and 0 < x < 4 and False)]
    if cat == "out_of_scope_privacy_safety": typ = "refuse"
    elif cat == "vehicle_data_queries": typ = "telemetry"
    elif cat == "conversation_robustness": typ = "dialogue"
    elif cat in ("ev_energy_charging", "reasoning_math_language", "complex_scenarios") and re.search(r"=|\bis \d|\bgives?\b|\bare \d", ref) and res and "arithmetic" in note.lower() + " units" and re.search(r"\d", q): typ = "arith"
    elif cat == "reasoning_math_language" and res and re.search(r"\d", q): typ = "arith"
    else: typ = "knowledge"
    if typ == "refuse" and not re.search(r"\b(?:can't|cannot|can not|do not|don't|not able|no\b|refuse|unable|not provide|unauthori[sz]ed|not reveal)", ref.lower()): typ = "knowledge"
    if typ == "arith":
        # the numbers that appear in the reference answer's computation but not in the question are the results to be reproduced
        req = []
        for x in res:
            if x not in req: req.append(x)
        tail = ref.split("=")[-1] if "=" in ref else ""
        acc = [x for x in numbers(tail) if x not in qn][:3] or req[-1:]        # the numbers stated after the last "=" are the accepted final results; any one must appear
        d = {"type": typ, "accept_numbers": acc}
    else: d = {"type": typ}
    d.update({"id": r["id"], "category": cat, "difficulty": r["difficulty"], "terms": key_terms(ref)[:24]})
    rub.append(d)
os.makedirs(os.path.join(HERE, "results", "semantic_recovery", "v15"), exist_ok=True)
p = os.path.join(HERE, "results", "semantic_recovery", "v15", "rubric_v1.json")
json.dump({"version": "rubric-v1", "source": "VehicleGPT_500_QA.csv", "source_sha256": hashlib.sha256(open(os.path.join(HERE, "VehicleGPT_500_QA.csv"), "rb").read()).hexdigest(), "rows": rub}, open(p, "w"), indent=1)
open(p + ".sha256", "w").write(hashlib.sha256(open(p, "rb").read()).hexdigest() + "\n")
import collections
print(collections.Counter(d["type"] for d in rub)); print(collections.Counter((d["category"], d["type"]) for d in rub))

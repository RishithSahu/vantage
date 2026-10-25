"""Answer audit: ask a mixed bank of questions of the real assistant on one vehicle and flag what looks wrong without needing a gold answer.

    python bench/audit_answers.py --db world_rhino.db [--n 400] [--seed 7] [--out results/audit/rhino.json]

The bank is a seeded sample of: the frozen question sets (everything a vehicle owner asks), the vehicle-related categories of VehicleGPT_500_QA.csv (written by someone else, free wording) and the everyday
banks.  Flags (each is a bug class the owner met or could meet):
  error        the answer says a routine could not run, or an exception text, or the route is an error route
  placeholder  a Python / JSON artefact in the answer (None, nan, {, }, Traceback, undefined)
  empty        no answer, or fewer than 6 characters
  slow         answered in more than 2.5 s (the case study asks for 2 s on common questions)
  class_leak   words of the wrong kind of vehicle: a traction battery, state of charge, kWh or charging session on a combustion vehicle; fuel, litres, refuelling, AdBlue, DPF on an electric one
  repeat       the same sentence twice in one answer
Nothing here knows the right answer; it finds the answers that are wrong on their face.  The output keeps every row for reading."""
import argparse
import collections
import csv
import io
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import evalio                                                     # noqa: E402

NOW = 1792348500
FROZEN = ["platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5"]
QA_CATS = {"vehicle_data_queries", "ev_energy_charging", "camera_gps_weather", "complex_scenarios", "driving_safety", "conversation_robustness", "out_of_scope_privacy_safety"}
EV_WORDS = re.compile(r"\b(traction battery|state of charge|soc\b|kwh\b|charging session|regen\w*|plug(ged)? in|fast charg\w+|charge limit|battery pack|pack voltage)", re.I)
ICE_WORDS = re.compile(r"\b(litres?|fuel (level|tank|rate|economy|consumption)|refuel\w*|adblue|diesel exhaust fluid|particulate filter|dpf|fill-?up)\b", re.I)
PLACE = re.compile(r"\b(None|nan|NaN|undefined|Traceback|KeyError|TypeError|ValueError|AttributeError)\b|[{}]|\[\]|\(\)")
ERR = re.compile(r"could not (compute|run)|EXCEPTION|routine that answers it could not|Traceback", re.I)


def bank(n, seed):
    rnd = random.Random(seed); pool = []
    for name in FROZEN:
        p = os.path.join(HERE, "meaning_eval", name + ".tsv")
        if os.path.exists(p): pool += [("frozen:" + name, r["text"]) for r in evalio.load_tsv(p)]
    qa = []
    with io.open(os.path.join(ROOT, "VehicleGPT_500_QA.csv"), encoding="utf-8-sig", newline="") as f:
        qa = [("qa500:" + r["category"], r["question"]) for r in csv.DictReader(f) if r["category"] in QA_CATS]
    ev = []
    p = os.path.join(HERE, "everyday_questions.txt")
    if os.path.exists(p): ev = [("everyday", l.strip()) for l in io.open(p, encoding="utf-8") if l.strip() and not l.startswith("#")]
    rnd.shuffle(pool); rnd.shuffle(qa); rnd.shuffle(ev)
    k_f, k_q, k_e = int(n * 0.6), int(n * 0.3), n - int(n * 0.6) - int(n * 0.3)
    out = pool[:k_f] + qa[:k_q] + ev[:k_e]; rnd.shuffle(out); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", default=None)
    ap.add_argument("--adapter", default="models/meaning-lora-full_v11.gguf"); a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    db = os.path.join(ROOT, "data", a.db); LK.adopt_path(db); brain = make_meaning_brain(db, now=NOW, threads=2, adapter=os.path.join(ROOT, a.adapter))
    from vgpt.vehicle import caps as CAPS
    cp = CAPS.detect(brain.c) if hasattr(CAPS, "detect") else {}
    decl = ((cp.get("declared") or {}).get("fuel_type") or "") if isinstance(cp, dict) else ""
    kind = "ice" if (cp.get("energy") == "ice" or decl in ("diesel", "petrol")) else (cp.get("energy") or "?")                  # a recording that only labels itself diesel is still not an electric car
    rows = []; t0 = time.time()
    for i, (src, q) in enumerate(bank(a.n, a.seed)):
        brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain.units = {"p": "kpa", "d": "km", "t": "c"}; brain._last_data = None
        getattr(brain, "dialogue", None) and brain.dialogue.reset(); brain._frames.clear()
        t = time.perf_counter()
        try: res = brain.ask(q)
        except Exception as e: res = {"answer": f"EXCEPTION {type(e).__name__}: {e}", "facts": {"route": "exception"}}
        ms = round((time.perf_counter() - t) * 1000); ans = str(res.get("answer") or ""); f = res.get("facts") or {}; fr = f.get("frame")
        flags = []
        if ERR.search(ans) or str(f.get("route") or "").startswith(("error", "exception")): flags.append("error")
        if PLACE.search(ans): flags.append("placeholder")
        if len(ans.strip()) < 6: flags.append("empty")
        if ms > 2500: flags.append("slow")
        said = re.sub(r"[^.\n]*no traction battery[^.\n]*[.\n]?", " ", ans, flags=re.I); said = re.sub(r"(has|have|with|and|as|is)? ?no (traction )?battery|no (fuel|engine)[^.]*|does not (have|report|record)[^.]*|not (recorded|reported)[^.]*", " ", said, flags=re.I)         # an answer that DENIES the thing is right
        general = isinstance(fr, dict) and fr.get("act") in ("define", "howto", "social", "offtopic", "unsupported", "unsafe", "control", "ambiguous")         # a definition or a refusal may name any kind of vehicle
        if not general and kind == "ice" and EV_WORDS.search(said): flags.append("class_leak")
        if not general and kind == "ev" and ICE_WORDS.search(said): flags.append("class_leak")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", ans) if len(s.strip()) > 25]
        if len(sents) != len(set(sents)): flags.append("repeat")
        rows.append({"src": src, "q": q, "answer": ans[:700], "route": str(f.get("route") or res.get("intent") or ""), "frame": " ".join(fr.values()) if isinstance(fr, dict) else None, "ms": ms, "flags": flags})
        if (i + 1) % 100 == 0: print(f"  {i + 1} questions, {time.time() - t0:.0f} s", flush=True)
    out = a.out or os.path.join(ROOT, "results", "audit", a.db.replace(".db", "") + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"db": a.db, "kind": kind, "rows": rows}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    c = collections.Counter(fl for r in rows for fl in r["flags"]); ms = sorted(r["ms"] for r in rows)
    print(f"\n{a.db} ({kind}): {len(rows)} questions; flagged {sum(bool(r['flags']) for r in rows)}; by flag {dict(c)}; latency p50 {ms[len(ms) // 2]} ms, p95 {ms[int(len(ms) * .95)]} ms")


if __name__ == "__main__":
    main()

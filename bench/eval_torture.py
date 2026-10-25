"""How well the analysis engine understands the stress-test questions and their paraphrases.
Usage: python bench/eval_torture.py [db] [file.tsv ...]      (files: route<TAB>question; default: bench/torture_dev.tsv, torture_blind_used.tsv and - if you add one - a fresh torture_blind.tsv)
Each question is asked in order, in one conversation (follow-ups need the earlier answers). 'correct' = answered by the intended analysis route (or the older pipeline for the few questions it answers correctly).
Also prints the slowest answer and every miss."""
import os, sys, time
sys.path.insert(0, ".")
from vgpt.brain import Brain
db = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".db") else "data/gen_demo.db"
files = [a for a in sys.argv[1:] if a.endswith(".tsv")] or [f for f in ("bench/torture_dev.tsv", "bench/torture_blind_used.tsv", "bench/torture_blind.tsv") if os.path.exists(f)]
LEGACY_OK = {"why_range2": ("WHY_RANGE",), "legacy": ("COMPARE", "WHY_RANGE", "USAGE", "TREND", "LARGEST", "STATUS", "SUMMARY"), "trend": ()}
for fn in files:
    rows = [l.rstrip("\n").split("\t", 1) for l in open(fn, encoding="utf-8") if l.strip() and not l.startswith("#")]
    b = Brain(db); ok = 0; miss = []; slow = (0, ""); via = {"qa": 0, "fuzzy": 0}
    for lab, q in rows:
        t = time.perf_counter(); r = b.ask(q, use_slm=False); ms = (time.perf_counter() - t) * 1000; slow = max(slow, (ms, q))
        f = r["facts"] if isinstance(r["facts"], dict) else {}; route = f.get("route") if r["path"] == "qa" else None
        good = route == lab or (route == "guard" and lab in ("day", "point")) or (lab in LEGACY_OK and r["intent"] in LEGACY_OK[lab] and r["path"] != "gate")
        if r["path"] == "qa": via["qa"] += 1; via["fuzzy"] += 1 if f.get("fuzzy") else 0
        if good: ok += 1
        else: miss.append((lab, q, route or f"{r['intent']}/{r['path']}"))
    print(f"{os.path.basename(fn)}: {ok}/{len(rows)} routed as intended ({100 * ok // len(rows)}%); {via['qa']} answered by the analysis engine ({via['fuzzy']} through the fuzzy router); slowest {slow[0]:.0f} ms")
    for m in miss: print("   MISS", m)

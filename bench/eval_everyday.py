"""Routing accuracy on everyday phrasings (route<TAB>question files) with the language model off. Prints the share answered as intended and every miss.
Usage: python bench/eval_everyday.py [db] [file.tsv ...]   (default data/gen_courier.db and the held-out dev set)
Labels are the QA route that should answer; 'help' / 'smalltalk' / 'foreign' are the conversation layer; 'gate' = must NOT be answered as if about the car. A few labels also accept the
older pipeline's answer when it is the same thing (listed in ALT)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt import brain

ALT = {"top_speed": {"legacy:SUMMARY"}, "style": {"legacy:DRIVING"}, "eff_day": {"legacy:USAGE"}, "trip_count": {"legacy:USAGE", "legacy:QC_COUNT"}, "last_trip": {"legacy:USAGE"}, "soh_rate": {"battery_age"}, "battery_age": {"soh_rate"},
       "charging_now": {"legacy:STATUS"}, "unusual": {"checkup", "badday"}, "overview": {"checkup", "vehicle_health"},
       # the typed engine now answers these with the same fact under its own route name (verified by hand against the legacy answer)
       "longest_trip": {"legacy:QC_RANK"}, "battery_age": {"soh_rate", "soh_change"}, "soh_rate": {"battery_age", "soh_change"}, "battery_now": {"coordinated_request"}, "improve_range": {"efficiency_advice"}}


def outcome(r):
    f = r["facts"] if isinstance(r["facts"], dict) else {}
    if "->" in str(r.get("understood") or ""): return "foreign"
    if r["path"] == "qa": return f.get("route")
    if r["intent"] in ("HELP", "ABOUT"): return "help"
    if r["intent"] == "SMALLTALK": return "smalltalk"
    if r["intent"] == "FOREIGN" or "->" in str(r.get("understood") or ""): return "foreign"
    if r["path"] in ("gate", "unsupported", "clarify"): return "gate"
    return "legacy:" + r["intent"]


def run(db, files, quiet=False):
    b = brain.Brain(db); tot = ok = 0; miss = []
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            if not line.strip() or line.startswith("#") or "\t" not in line: continue
            lab, q = line.rstrip("\n").split("\t", 1); r = b.ask(q, use_slm=False); got = outcome(r); tot += 1
            if got == lab or got in ALT.get(lab, ()) or (lab == "foreign" and got == "foreign"): ok += 1
            else: miss.append((lab, q, got, r["brief"][:60]))
    return ok, tot, miss


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv; args = [a for a in sys.argv[1:] if a != "--quiet"]
    db = args[0] if args else "data/gen_courier.db"; files = args[1:] or ["bench/heldout_dev.tsv"]
    ok, tot, miss = run(db, files)
    print(f"{', '.join(files)}: {ok}/{tot} answered as intended ({100 * ok // max(tot, 1)}%)")
    if quiet:
        from collections import Counter
        c = Counter(m[0] for m in miss); print("  misses by label:", dict(c))          # (--quiet: the questions are not printed, so the set stays untouched for tuning)
    else:
        for m in miss: print("  MISS", m)

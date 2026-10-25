"""Re-score the Case Study 5 bundle from stored predictions (results/meaning/<arm>/predictions.json) with a corrected scorer.

    python bench/meaning_eval/cs5_score.py full full_v2 ...

Two defects in the scorer first used by eval_frames.py are fixed here, and both are reported next to the original number:
  1. a comparison's `conditions` in the bundle can be PERIOD labels ("week_1", "week_2" for "this week versus the previous week"); they were wrongly required to equal the frame's two driving
     conditions.  Only conditions that are driving states (city, highway, charging, ...) are compared with cond / vs; period labels require vs = period.
  2. the bundle's entity `energy_consumption` covers both energy per distance and total energy used (the questions mix "energy consumption" and "which uses more energy"); both concepts are accepted.
Everything else (act sets, concept map) is unchanged.  The model's predictions are not touched."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_frames                                                    # noqa: E402

DRIVING_STATES = {"city", "highway", "charging", "parked", "driving", "fast_charging", "raining", "night", "cold", "hot_weather", "idling", "high_speed"}
CONCEPT_ALT = {"wh_per_km": {"wh_per_km", "energy_used"}}


def score(preds):
    rows = {r["id"]: r for r in eval_frames.cs5_rows()}; out = []
    for p in preds:
        if p["set"] != "cs5" or p["id"] not in rows: continue
        r = rows[p["id"]]; f = p["pred"].split() if p.get("pred") else None
        if not f: out.append((p["id"], False)); continue
        act, concept, cond, vs = f[0], f[1], f[3], f[4]
        ok = act in r["acts"] and (r["concept"] == "none" or concept in CONCEPT_ALT.get(r["concept"], {r["concept"]}))
        if ok and act == "compare":
            states = {c for c in r["conds"] if c in DRIVING_STATES}
            if states: ok = states <= {cond, vs}
            elif r["conds"]: ok = vs == "period"
        out.append((p["id"], ok))
    return out


def main():
    for arm in sys.argv[1:]:
        p = os.path.join(ROOT, "results", "meaning", arm, "predictions.json")
        if not os.path.exists(p): print(arm, "(no predictions)"); continue
        J = json.load(open(p, encoding="utf-8")); old = J["summary"].get("cs5", {}).get("act_and_concept"); res = score(J["predictions"]); n = len(res); k = sum(ok for _, ok in res)
        print(f"{arm}: original scorer {old}, corrected scorer {k}/{n} = {k / n:.3f}; still wrong: {[i for i, ok in res if not ok]}")


if __name__ == "__main__":
    main()

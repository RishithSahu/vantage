"""Parsing accuracy of the full hybrid (normalise -> rules -> SLM -> grounding gate), per split, with an abstain count.
correct = right intent and the slots that matter; abstain = asked a clarifying question / declined; wrong = confidently misrouted.
Out-of-scope questions (expected NONE) are correct when the assistant declines (gate / unsupported / clarify / other).
Usage: python bench/eval_parse.py <model.gguf|none> [splits e.g. dev,test,blind,field,typo]"""
import os, sys, json, random, statistics as st
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
sys.path.insert(0, ".")
from vgpt.brain import Brain, Slm
Q = json.load(open("bench/questions.json"))
RELEVANT = {"EVENTS": ("event", "w"), "BEFORE_EVENT": ("event",), "TREND": ("metric", "w"), "STATUS": ("metric",), "WHY_RANGE": (), "COMPARE": (), "LARGEST": (), "NONE": (), "USAGE": ("w",), "SUMMARY": ("metric", "w"), "TYRES": (), "WEATHER": (), "SERVICE": (), "HEALTH": (), "DRIVING": (), "PLACES": ()}
def full_ok(q, intent, p): return intent == q["intent"] and all(q[k] == "-" or {"event": p["event"], "metric": p["metric"], "w": p["w"]}[k] == q[k] for k in RELEVANT[q["intent"]])

def typo(s, rng):
    """Deterministic keyboard-style corruption of ~1 in 4 words of length >= 5 (swap / drop / double a letter)."""
    out = []
    for w in s.split(" "):
        if len(w) >= 5 and w.isalpha() and rng.random() < 0.25:
            i = rng.randrange(1, len(w) - 1); k = rng.randrange(3)
            w = (w[:i] + w[i + 1] + w[i] + w[i + 2:]) if k == 0 else (w[:i] + w[i + 1:]) if k == 1 else (w[:i] + w[i] + w[i:])
        out.append(w)
    return " ".join(out)

model = sys.argv[1]; splits = (sys.argv[2] if len(sys.argv) > 2 else "dev,test,blind,field").split(",")
slm = None if model == "none" else Slm("models/" + model, threads=2)
if slm: slm.warmup()
b = Brain("data/vehicle.db", slm=slm)
rng = random.Random(3)
for split in splits:
    if split == "typo": qs = [dict(q, q=typo(q["q"], rng)) for q in Q if q["split"] in ("test", "blind")]
    else: qs = [q for q in Q if q["split"] == split]
    ok = ab = wrong = 0; ms = []; miss = []
    for q in qs:
        r = b.ask(q["q"]); ms.append(r["ms"]["total"]); good = full_ok(q, r["intent"], r["params"])
        if good: ok += 1
        elif r["path"] == "clarify" and q["intent"] != "NONE": ab += 1; miss.append(("ABSTAIN", q["q"], q["intent"]))
        else: wrong += 1; miss.append(("WRONG", q["q"], q["intent"], r["intent"], r["path"]))
    print(f"{model[:22]:22s} {split:6s}: correct {ok}/{len(qs)} ({100 * ok // len(qs)}%)  abstain {ab}  wrong {wrong} | latency median {st.median(ms):.1f} ms max {max(ms):.0f}")
    for m in miss: print("     ", m)

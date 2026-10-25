"""Quick dev probe: SLM-only intent/slot accuracy + latency on a split. Usage: python bench/slm_probe.py <model> <fmt> <split>"""
import json, sys, statistics as st, time
sys.path.insert(0, ".")
from vgpt.brain import Slm
model, fmt, split = sys.argv[1], sys.argv[2], sys.argv[3]
Q = [q for q in json.load(open("bench/questions.json")) if q["split"] == split]
s = Slm("models/" + model, threads=2, fmt=fmt); s.warmup()
ok_i = ok_all = 0; ms = []; toks = []; bad = []
for q in Q:
    i, p, info = s.parse(q["q"]); ms.append(info["ms"]); toks.append(info["out_tokens"] or 0)
    good_i = i == q["intent"]; ok_i += good_i
    slots = [(q["metric"], p["metric"]), (q["event"], p["event"]), (q["w"], p["w"])]
    good = good_i and all(e == "-" or e == g for e, g in slots); ok_all += good
    if not good: bad.append((q["q"], q["intent"], i, p))
print(f"{model} fmt={fmt} split={split}: intent {ok_i}/{len(Q)}  intent+slots {ok_all}/{len(Q)} | parse ms median {st.median(ms):.0f} p95 {sorted(ms)[int(.95*(len(ms)-1))]:.0f} | out tokens {st.mean(toks):.1f}")
for b in bad: print("   MISS", b)

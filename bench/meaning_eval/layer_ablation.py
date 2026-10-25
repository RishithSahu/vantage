"""Is it the model, or the phrase-based layers around it?

    python bench/meaning_eval/layer_ablation.py --name v11 [--db world_aster.db] [--adapter models/meaning-lora-full_v11.gguf] [--sets ...] [--threads 2] [--oracle]

The assistant is run on every frozen question under several configurations of the layers that stand between the learned frame and the answer (vgpt/meaning/layers.py):

    model_only         every optional layer off: the model's frame goes straight to the typed resolver (only the safety guard and the literal parser remain)
    default            the product as shipped (arbitrate off, everything else on)
    default+arbitrate  the retired rare-word rules switched back on
    own_vocab_first_version   the vehicle-vocabulary layer as first written (any signal of the vehicle, matched by a generic word), retired 2026-10-20
    minus_<layer>      the product with one layer off (own_vocab, unsafe_floor, investigate, discourse, fuel_remap)

For each question and configuration it records the frame the resolver finally executed, the route and the answer.  The model is asked once per distinct question (the frame is cached), so the layers
are compared on the very same model output.  Reported:
  * frame exact / act + concept, per configuration (all questions, and per set);
  * per layer: how many questions it CHANGED (the frame differs from model_only), and of those how many it HELPED (model_only frame wrong, with the layer right), HURT (right -> wrong) or neither;
  * the investigation route: how many questions it took over, with their gold acts (a multi-part analysis request is not a single frame, so "right" is read from the questions listed in the output);
  * with --oracle: the same-answer rate against the answer a perfect reader gets with every layer off.
Output: results/meaning/layer_ablation_<name>.json (every row), the table on stdout."""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import evalio                                                     # noqa: E402

NOW = 1792348500
SETS = ["platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5"]
LAYERS = ("own_vocab", "unsafe_floor", "investigate", "discourse", "fuel_remap")
CONFIGS = [("model_only", {"VGPT_LAYERS_OFF": "all"}), ("default", {}), ("own_vocab_first_version", {"VGPT_OWNVOCAB_ALL": "1"}), ("default+arbitrate", {"VGPT_ARBITRATE": "1"})] + [("minus_" + l, {"VGPT_LAYERS_OFF": l}) for l in LAYERS]
norm = lambda s: " ".join(str(s).split())


class GoldReader:
    def __init__(self, real): self.real, self.frame = real, None
    def parse(self, text, hint=None, prev=None): return self.frame, {"raw": self.frame.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}
    def __getattr__(self, k): return getattr(self.real, k)


def setenv(d):
    for k in ("VGPT_LAYERS_OFF", "VGPT_ARBITRATE", "VGPT_MODEL_ONLY", "VGPT_OWNVOCAB_ALL"): os.environ.pop(k, None)
    os.environ.update(d)


def ac(f):
    return " ".join(str(f).split()[:2]) if f else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--db", default="world_aster.db"); ap.add_argument("--adapter", default="models/meaning-lora-full_v11.gguf")
    ap.add_argument("--sets", nargs="+", default=SETS); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--oracle", action="store_true"); ap.add_argument("--configs", nargs="+", default=None)
    a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    db = os.path.join(ROOT, "data", a.db); LK.adopt_path(db)
    brain = make_meaning_brain(db, now=NOW, threads=a.threads, adapter=os.path.join(ROOT, a.adapter)); real = brain.frame_model; gold = GoldReader(real)
    rows = []
    for name in a.sets:
        p = os.path.join(HERE, name + ".tsv")
        if os.path.exists(p): rows += [dict(r, set=name) for r in evalio.load_tsv(p)]

    def fresh():
        brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain.units = {"p": "kpa", "d": "km", "t": "c"}; brain._last_data = None
        getattr(brain, "dialogue", None) and brain.dialogue.reset()

    def run(r):
        fresh(); brain._text = r["text"]; t = time.perf_counter()
        try: res = brain.ask(r["text"])
        except Exception as e: res = {"answer": f"EXCEPTION {type(e).__name__}: {e}", "facts": {}}
        f = res.get("facts") or {}; fr = f.get("frame")
        return {"frame": " ".join(fr.values()) if isinstance(fr, dict) else None, "route": str(f.get("route") or res.get("intent") or ""), "answer": str(res.get("answer"))[:500], "arb": f.get("arbitrated"), "ms": round((time.perf_counter() - t) * 1000)}

    cfgs = [c for c in CONFIGS if not a.configs or c[0] in a.configs]; out = []; t0 = time.time()
    oracle_frames = brain._frames; brain._frames = {}                                  # the oracle never shares the model's frame cache
    for i, r in enumerate(rows):
        row = {"set": r["set"], "id": r["id"], "text": r["text"], "gold": r["gold"].to_text(), "cfg": {}}
        if a.oracle:
            setenv({"VGPT_LAYERS_OFF": "all"}); brain.frame_model = gold; gold.frame = r["gold"]; brain._frames = {}
            row["oracle"] = run(r)["answer"]; brain.frame_model = real
        brain._frames = oracle_frames
        for cname, env in cfgs:
            setenv(env); row["cfg"][cname] = run(r)
        out.append(row)
        if (i + 1) % 100 == 0: print(f"  {i + 1}/{len(rows)} questions, {time.time() - t0:.0f} s", flush=True)
    setenv({})
    os.makedirs(os.path.join(ROOT, "results", "meaning"), exist_ok=True)
    json.dump({"db": a.db, "adapter": a.adapter, "rows": out}, open(os.path.join(ROOT, "results", "meaning", f"layer_ablation_{a.name}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    report(out, [c[0] for c in cfgs])


def report(out, names):
    n = len(out); print(f"\n{n} questions")
    print(f"{'configuration':22} {'frame exact':>12} {'act+concept':>12}" + ("   same answer as a perfect reader" if "oracle" in out[0] else ""))
    for c in names:
        ex = sum(r["cfg"][c]["frame"] == r["gold"] for r in out); acn = sum(ac(r["cfg"][c]["frame"]) == ac(r["gold"]) for r in out)
        line = f"{c:22} {100 * ex / n:11.1f}% {100 * acn / n:11.1f}%"
        if "oracle" in out[0]: line += f"   {100 * sum(norm(r['cfg'][c]['answer']) == norm(r['oracle']) for r in out) / n:5.1f}%"
        print(line)
    if "model_only" not in names: return
    print("\nWhat each layer changed, against the model alone (frame level; 'changed' = the executed frame differs from the model's)")
    print(f"{'layer / configuration':26} {'changed':>8} {'helped':>7} {'hurt':>6} {'wrong->other wrong':>19} {'net':>5}")
    for c in [x for x in names if x != "model_only"]:
        ch = hp = ht = ww = 0
        for r in out:
            m, x, g = r["cfg"]["model_only"]["frame"], r["cfg"][c]["frame"], r["gold"]
            if x == m: continue
            ch += 1
            if m != g and x == g: hp += 1
            elif m == g and x != g: ht += 1
            else: ww += 1
        print(f"{c:26} {ch:8} {hp:7} {ht:6} {ww:19} {hp - ht:+5}")
    inv = [r for r in out if "INVESTIG" in r["cfg"]["default"]["route"].upper()]
    print(f"\nInvestigation route took over {len(inv)} of {n} questions; gold acts of those: {dict(collections.Counter(r['gold'].split()[0] for r in inv).most_common())}")
    for r in inv[:40]: print(f"   [{r['set']} {r['id']}] gold {r['gold']!r:40} {r['text'][:110]!r}")
    print("\nPer set, frame exact (model_only / default / default+arbitrate):")
    for s in dict.fromkeys(r["set"] for r in out):
        rs = [r for r in out if r["set"] == s]; f = lambda c: 100 * sum(r["cfg"][c]["frame"] == r["gold"] for r in rs) / len(rs) if c in names else float("nan")
        print(f"   {s:12} n={len(rs):4}   {f('model_only'):5.1f}  {f('default'):5.1f}  {f('default+arbitrate'):5.1f}")


if __name__ == "__main__":
    main()

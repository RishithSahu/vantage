"""Which code actually computes the answers?  A census of the executors behind the frozen questions, with a perfect reader (the gold frame) and every layer off.

    python bench/meaning_eval/executor_census.py [--db world_aster.db]

The meaning path decides WHAT is asked (the learned frame); an executor computes the answer.  Typed executors take the frame's fields as arguments.  The *legacy adapter* turns the frame into a canonical sentence
and hands it to the older keyword handlers (`bindings.json`, `legacy`), so inside that executor words are read again by older rules - the part of the system that docs/semantic_method.md section 8 lists as
open.  This script counts how many of the frozen questions are answered by which kind of executor, so the size of that remainder is a number and not an impression."""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_LAYERS_OFF"] = "all"
import evalio                                                     # noqa: E402

NOW = 1792348500
SETS = ["platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5"]


class GoldReader:
    def __init__(self, real): self.real, self.frame = real, None
    def parse(self, text, hint=None, prev=None): return self.frame, {"raw": self.frame.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}
    def __getattr__(self, k): return getattr(self.real, k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default="world_aster.db"); ap.add_argument("--adapter", default="models/meaning-lora-full_v11.gguf"); a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    db = os.path.join(ROOT, "data", a.db); LK.adopt_path(db); brain = make_meaning_brain(db, now=NOW, threads=2, adapter=os.path.join(ROOT, a.adapter)); brain.frame_model = GoldReader(brain.frame_model)
    kinds = collections.Counter(); by_act = collections.defaultdict(collections.Counter); n = 0; examples = collections.defaultdict(list)
    for name in SETS:
        p = os.path.join(HERE, name + ".tsv")
        if not os.path.exists(p): continue
        for r in evalio.load_tsv(p):
            brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain.units = {"p": "kpa", "d": "km", "t": "c"}; brain._last_data = None; brain.dialogue.reset(); brain._frames.clear()
            brain.frame_model.frame = r["gold"]; brain._text = r["text"]
            try: res = brain.ask(r["text"])
            except Exception as e: res = {"facts": {"executor": "exception"}}
            f = res.get("facts") or {}; ex = f.get("executor") or ("none (no executor: " + r["gold"].act + ")" if r["gold"].act in ("social", "offtopic", "unsupported", "control", "unsafe", "ambiguous") else "typed:" + str(f.get("route") or res.get("intent") or "?").split(":")[0])
            kind = "legacy keyword handlers" if "legacy" in ex else "no executor needed (a refusal or a greeting)" if ex.startswith("none") else "typed or purpose-built executor"
            kinds[kind] += 1; by_act[r["gold"].act][kind] += 1; n += 1
            if "legacy" in ex and len(examples[r["gold"].act]) < 3: examples[r["gold"].act].append(r["text"][:70])
    print(f"{n} frozen questions answered through the gold frame with every layer off\n")
    for k, v in kinds.most_common(): print(f"  {k:46} {v:5}  {100 * v / n:5.1f}%")
    print("\nlegacy keyword handlers by the act of the question:")
    for act, c in sorted(by_act.items(), key=lambda kv: -kv[1]["legacy keyword handlers"]):
        t = sum(c.values()); lg = c["legacy keyword handlers"]
        if lg: print(f"  {act:12} {lg:4} of {t:4}  {100 * lg / t:5.1f}%   e.g. {examples[act]}")


if __name__ == "__main__":
    main()

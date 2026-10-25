"""End-to-end comparison on the 105-day world: does the WHOLE assistant (frame model + arbitration + resolver) give the answer that a perfectly understood question gets?

    python bench/meaning_eval/e2e_world.py --adapter models/meaning-lora-full_v11.gguf --name v11 [--sets platform_v1 adv_v1 ...] [--threads 2]

For every question of the frozen sets the script computes
  * the ORACLE answer: the resolver's answer for the gold frame (what a perfect reader of the question would have been told), and
  * the MODEL answer: what the shipped assistant says for the question text with the given adapter,
on the same database at the same moment, and records whether the frame was exact and whether the two answers are the same text.
Answer equality is strict (whitespace-normalised text), so it under-counts equivalent answers; it is the same measure for every adapter, so differences between adapters are real.
Output: results/meaning/e2e_world_<name>.json (every row) and a per-set summary on stdout."""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import evalio                                                     # noqa: E402

DB = os.path.join(ROOT, "data", "world_aster.db")
NOW = 1792348500                                                  # 2026-10-19 00:05 local, the end of the recorded history
SETS = ["platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5"]
norm = lambda s: " ".join(str(s).split())


class GoldReader:
    """Stands in for the model: reads every question perfectly (returns the gold frame).  Run through the SAME pipeline as the real adapter, so the two answers are formatted alike."""
    def __init__(self, real): self.real, self.frame = real, None
    def parse(self, text, hint=None, prev=None): return self.frame, {"raw": self.frame.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}
    def __getattr__(self, k): return getattr(self.real, k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--name", required=True); ap.add_argument("--sets", nargs="+", default=SETS); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--no-arbitrate", action="store_true", help="switch the rare-word arbitration rules off (the model's frame is used as it is)"); a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    LK.adopt_path(DB)
    brain = make_meaning_brain(DB, now=NOW, threads=a.threads, adapter=a.adapter if os.path.isabs(a.adapter) else os.path.join(ROOT, a.adapter)); out = []
    import vgpt.meaning.pipeline as PL
    real, arb, gold = brain.frame_model, PL.arbitrate, None; gold = GoldReader(real)
    if a.no_arbitrate: arb = PL.arbitrate = lambda text, frame, onto, glossary: (frame, None)

    def fresh():
        brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain.units = {"p": "kpa", "d": "km", "t": "c"}; brain._last_data = None
        getattr(brain, "dialogue", None) and brain.dialogue.reset()

    for name in a.sets:
        path = os.path.join(HERE, name + ".tsv")
        if not os.path.exists(path): continue
        for r in evalio.load_tsv(path):
            fresh(); brain._text = r["text"]; gold.frame = r["gold"]; brain.frame_model = gold; brain._frames.clear(); PL.arbitrate = lambda text, frame, onto, glossary: (frame, None)
            try: want = brain.ask(r["text"])
            except Exception as e: want = {"answer": f"EXCEPTION {type(e).__name__}"}
            brain.frame_model = real; PL.arbitrate = arb; brain._frames.clear()
            fresh(); t = time.perf_counter()
            try: res = brain.ask(r["text"])
            except Exception as e: res = {"answer": f"EXCEPTION {type(e).__name__}: {e}", "facts": {}}
            ms = round((time.perf_counter() - t) * 1000); f = res.get("facts") or {}; pred = " ".join((f.get("frame") or {}).values()) if f.get("frame") else None
            out.append({"set": name, "id": r["id"], "text": r["text"], "gold": r["gold"].to_text(), "pred": pred, "frame_ok": pred == r["gold"].to_text(), "same_answer": norm(res.get("answer")) == norm(want.get("answer")),
                        "answer": str(res.get("answer"))[:400], "oracle": str(want.get("answer"))[:400], "ms": ms})
    os.makedirs(os.path.join(ROOT, "results", "meaning"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "meaning", f"e2e_world_{a.name}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    by = collections.defaultdict(lambda: [0, 0, 0])
    for o in out: b = by[o["set"]]; b[0] += 1; b[1] += o["frame_ok"]; b[2] += o["same_answer"]
    tot = [sum(v[i] for v in by.values()) for i in range(3)]
    for k, v in by.items(): print(f"{k:12} n={v[0]:4} frame exact {100 * v[1] / v[0]:5.1f}%  same answer as a perfect reader {100 * v[2] / v[0]:5.1f}%")
    ms = sorted(o["ms"] for o in out); print(f"ALL          n={tot[0]:4} frame exact {100 * tot[1] / tot[0]:5.1f}%  same answer {100 * tot[2] / tot[0]:5.1f}%  latency p50 {ms[len(ms) // 2]} ms  p95 {ms[int(len(ms) * .95)]} ms")


if __name__ == "__main__":
    main()

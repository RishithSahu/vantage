"""Run the GOLD frames of the evaluation sets through the resolver (no model): what the deterministic side does when understanding is perfect.

    python bench/meaning_eval/run_oracle.py novel_v1 messy_v1 user_v1 --out results/meaning/oracle.json

Separates two questions: (1) does the model understand (eval_frames.py), (2) can the executors answer a correctly understood question (this script).  Every answer is stored for judging."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import evalio                                                     # noqa: E402

DB = os.path.join(ROOT, "data", "gen_demo.db")
NOW = 1787625900


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sets", nargs="+"); ap.add_argument("--out", required=True); a = ap.parse_args()
    from vgpt.brain import Brain
    from vgpt.meaning.resolve import Resolver
    brain = Brain(DB, slm=None, slm_ir=None, policy="clarify", qc_mode="all", now=NOW); res = Resolver(brain); out = []
    for name in a.sets:
        for r in evalio.load_tsv(os.path.join(HERE, name + ".tsv")):
            brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain._text = r["text"]
            t = time.perf_counter()
            try: x = res.answer(r["text"], r["gold"])
            except Exception as e: x = {"answer": f"EXCEPTION {type(e).__name__}: {e}", "intent": "ERR", "facts": {"route": "exception"}}
            f = x.get("facts") or {}
            out.append({"set": name, "id": r["id"], "text": r["text"], "gold": r["gold"].to_text(), "tag": r["tag"], "answer": str(x.get("answer"))[:600], "intent": x.get("intent"),
                        "route": f.get("route"), "executor": f.get("executor"), "ms": round((time.perf_counter() - t) * 1000)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    import collections
    print(len(out), "answers;", dict(collections.Counter(o["executor"] for o in out)))


if __name__ == "__main__":
    main()

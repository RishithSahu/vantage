"""Run the CURRENT product configuration (Qwen2.5-0.5B semantic LoRA + the rule layers) over frozen question sets and store every answer for judging.

    python bench/meaning_eval/run_current.py novel_v1 messy_v1 --out results/meaning/current_product.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import evalio                                              # noqa: E402

DB = os.path.join(ROOT, "data", "gen_demo.db")
NOW = 1787625900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--rules-only", action="store_true")
    a = ap.parse_args()
    from vgpt.product import make_brain, make_legacy_brain
    brain = make_legacy_brain(DB, slm=None, now=NOW) if a.rules_only else make_brain(DB, now=NOW, meaning=False)
    out = []
    for name in a.sets:
        rows = evalio.load_tsv(os.path.join(HERE, name + ".tsv"))
        for r in rows:
            brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []
            t = time.perf_counter()
            try: res = brain.ask(r["text"], use_slm=not a.rules_only)
            except Exception as e: res = {"answer": f"EXCEPTION {type(e).__name__}: {e}", "intent": "ERR", "path": "err", "facts": {}}
            ms = (time.perf_counter() - t) * 1000
            f = res.get("facts") or {}
            out.append({"set": name, "id": r["id"], "text": r["text"], "gold": r["gold"].to_text(), "tag": r["tag"], "answer": str(res.get("answer"))[:600],
                        "intent": res.get("intent"), "path": res.get("path"), "route": f.get("route"), "ms": round(ms)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(len(out), "answers ->", a.out)


if __name__ == "__main__":
    main()

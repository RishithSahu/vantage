"""Run the full meaning pipeline (frame model + resolver) over frozen sets and store every answer for judging.

    python bench/meaning_eval/run_pipeline.py novel_v1 messy_v1 user_v1 --adapter models/meaning-lora-full.gguf --out results/meaning/pipeline_full.json [--tau 0.5]"""
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
    ap = argparse.ArgumentParser(); ap.add_argument("sets", nargs="+"); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.0); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    brain = make_meaning_brain(DB, now=NOW, threads=a.threads, adapter=os.path.join(ROOT, a.adapter) if not os.path.isabs(a.adapter) else a.adapter, tau=a.tau); out = []
    for name in a.sets:
        for r in evalio.load_tsv(os.path.join(HERE, name + ".tsv")):
            brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []
            brain.units = {"p": "kpa", "d": "km", "t": "c"}; brain._last_data = None                      # every question is asked fresh: no display-unit preference carries over
            t = time.perf_counter()
            try: res = brain.ask(r["text"])
            except Exception as e: res = {"answer": f"EXCEPTION {type(e).__name__}: {e}", "intent": "ERR", "facts": {}}
            f = res.get("facts") or {}
            out.append({"set": name, "id": r["id"], "text": r["text"], "gold": r["gold"].to_text(), "pred": " ".join((f.get("frame") or {}).values()) if f.get("frame") else None, "tag": r["tag"],
                        "answer": str(res.get("answer"))[:600], "intent": res.get("intent"), "route": f.get("route"), "executor": f.get("executor"), "pmin": f.get("pmin"), "ms": round((time.perf_counter() - t) * 1000)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    exact = sum(1 for o in out if o["pred"] == o["gold"]); print(len(out), "answers; exact frame", exact, "/", len(out))


if __name__ == "__main__":
    main()

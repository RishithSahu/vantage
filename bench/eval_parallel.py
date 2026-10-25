"""Evaluate one adapter on every frozen set and every new set, in three processes at once, and merge the result into results/meaning/<name>_r10/predictions.json.

    python bench/eval_parallel.py models/meaning-lora-frame2r11.gguf frame2r11 [--no-hint] [--default-vehicle ev|none] [--no-vehicle-tag] [--procs 3] [--threads 4]

(The evaluation itself is bench/meaning_eval/eval_r10.py with the car type told to the model unless --no-hint.)  The parts are named <name>_p0 ... and kept; read the merged table with
    python bench/meaning_eval/rescore.py <name> [other names]"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
import eval_r10 as E                                          # noqa: E402  (does not run anything on import)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("adapter"); ap.add_argument("name"); ap.add_argument("--no-hint", action="store_true"); ap.add_argument("--procs", type=int, default=3); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sets", nargs="*"); ap.add_argument("--default-vehicle", default="ev"); ap.add_argument("--no-vehicle-tag", action="store_true"); a = ap.parse_args()
    sets = a.sets or (E.FROZEN + E.NEW); sizes = {s: len(E.load(s)) for s in sets}; parts = [[] for _ in range(a.procs)]; load = [0] * a.procs
    for s in sorted(sets, key=lambda s: -sizes[s]):                   # biggest first onto the lightest process
        k = load.index(min(load)); parts[k].append(s); load[k] += sizes[s]
    procs = []
    for k, ps in enumerate(parts):
        if not ps: continue
        cmd = [sys.executable, os.path.join(HERE, "meaning_eval", "eval_r10.py"), "--adapter", a.adapter, "--name", f"{a.name}_p{k}", "--threads", str(a.threads), "--default-vehicle", a.default_vehicle, "--sets", *ps] + ([] if a.no_hint else ["--vehicle-hint"]) + (["--no-vehicle-tag"] if a.no_vehicle_tag else [])
        log = open(os.path.join(ROOT, f"logs_eval_{a.name}_p{k}.txt"), "w", encoding="utf-8")
        procs.append((k, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8="1", VGPT_FRAME="2"))))
        print(f"part {k}: {len(ps)} sets, {load[k]} questions", flush=True)
    for k, p in procs: p.wait()
    merged = []; summary = {}
    for k, _ in procs:
        d = json.load(open(os.path.join(ROOT, "results", "meaning", f"{a.name}_p{k}_r10", "predictions.json"), encoding="utf-8")); merged += d["predictions"]; summary.update(d["summary"])
    out = os.path.join(ROOT, "results", "meaning", f"{a.name}_r10"); os.makedirs(out, exist_ok=True)
    json.dump({"adapter": a.adapter, "vehicle_hint": not a.no_hint, "default_vehicle": a.default_vehicle, "summary": summary, "predictions": merged}, open(os.path.join(out, "predictions.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("merged", len(merged), "predictions ->", out)


if __name__ == "__main__":
    main()

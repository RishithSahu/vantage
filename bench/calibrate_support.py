"""Calibration of the support check (vgpt/support.py) on real model outputs: for every call in a results/m*_*.json whose quoted words exist in the message, was the chosen period (or condition set) the one the gold interpretation has, and what did the support check say?
A good check REJECTS (False) the wrong values whose quotation exists and ACCEPTS the right ones; it may abstain (None). Reported: for correct values how often it falsely rejects, for wrong values how often it catches them.
    python tools_pin.py 4,5 bench/calibrate_support.py results/m3_dev_lfm.json [--margin 0.0]"""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import capreg, plancompile as PC, signal_registry as R, toolspec as T
from vgpt.query_ir import Query
from vgpt.retrieval import Embedder
from vgpt.support import SupportCheck


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--margin", type=float, default=0.0); a = ap.parse_args()
    import sqlite3
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True)))
    sc = SupportCheck(Embedder(os.path.join(HERE, "models", "bge-small-en-v1.5"), int8=True), reg, margin=a.margin); d = json.load(open(a.path, encoding="utf-8")); rec = {(r["text"], r["prev_text"]): r for r in d["records"]}
    tally = collections.Counter(); misses = []
    for row in d["rows"]:
        r = rec.get((row["text"], row["prev_text"]))
        if not r or not r.get("call") or row["gold"].startswith("op=none"): continue
        try: fn, args = T.parse_call("[" + r["call"]); plan = T.call_to_plan(fn, args); gq = Query.from_text(row["gold"])
        except Exception: continue
        msg = PC._norm_words(row["text"]); src = plan.get("src") or {}
        for key, getter in (("per", lambda q: q.scope.time.to_text()), ("cond", lambda q: sorted(map(repr, map(PC._norm_cond, PC.query_fields(q)["cond"]))))):
            if plan.get(key) in (None, "", [], {}) or PC._norm_words(src.get(key)) not in msg or not PC._norm_words(src.get(key)): continue
            ok = (plan[key] == getter(gq)) if key == "per" else (sorted(map(repr, map(PC._norm_cond, plan[key]))) == getter(gq))
            res = sc(key, PC._norm_words(src[key]), plan[key]); tally[(key, "right" if ok else "wrong", {True: "accept", False: "reject", None: "abstain"}[res]) ] += 1
            if (ok and res is False) or (not ok and res is True and key == "per"): misses.append((key, ok, src[key], plan[key], row["text"][:60], row["gold"][:60]))
    for key in ("per", "cond"):
        right = {k: tally[(key, "right", k)] for k in ("accept", "reject", "abstain")}; wrong = {k: tally[(key, "wrong", k)] for k in ("accept", "reject", "abstain")}
        print(f"{key}: value RIGHT {right} (false rejections {right['reject']}/{sum(right.values())});  value WRONG {wrong} (caught {wrong['reject']}/{sum(wrong.values())})")
    for m in misses[:12]: print("  ", m)


if __name__ == "__main__": main()

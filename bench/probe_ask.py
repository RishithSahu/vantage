"""Ask the real assistant (shipped adapter, arbitration, resolver) a list of questions on one or more generated vehicles and print the answers.

    python bench/probe_ask.py QUESTIONS.txt [--db data/world_aster.db ...] [--out results/probe_x.json] [--adapter models/meaning-lora-frame2w.gguf]

QUESTIONS.txt: one question per line; `#` comments; text after ` || ` is the expectation, a note for the reader (for example `must contain 3 tyres`).  Every question starts a fresh conversation.
Output per question and vehicle: the frame the model wrote, its confidence, and the answer text."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt                                                      # noqa: E402
vehiclegpt._adopt_product_model()

NOWS = {"world_aster": 1792364340, "world_kestrel": 1792364340, "world_rhino": 1792360740}


def _last_ts(path):
    import sqlite3
    c = sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)
    try:
        for t in ("sig_1s", "sig_1m"):
            r = c.execute(f"SELECT MAX(ts) FROM {t}").fetchone()[0]
            if r: return int(r)
    finally: c.close()
    return 1792364340


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("questions"); ap.add_argument("--db", nargs="+", default=["data/world_aster.db", "data/world_kestrel.db", "data/world_rhino.db"]); ap.add_argument("--out", default=None)
    ap.add_argument("--adapter", default=None); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--width", type=int, default=230); a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    qs = []
    for ln in open(a.questions, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"): continue
        q, _, exp = ln.partition(" || "); qs.append((q.strip(), exp.strip()))
    res = []
    for d in a.db:
        path = os.path.join(ROOT, d) if not os.path.isabs(d) else d; LK.adopt_path(path); name = os.path.basename(path).rsplit(".", 1)[0]
        brain = make_meaning_brain(path, now=NOWS.get(name) or _last_ts(path), threads=a.threads, adapter=(a.adapter if a.adapter and os.path.isabs(a.adapter) else os.path.join(ROOT, a.adapter)) if a.adapter else None)
        for q, exp in qs:
            brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []; brain._last_data = None; brain._frames.clear()
            getattr(brain, "dialogue", None) and brain.dialogue.reset()
            t = time.perf_counter()
            try: r = brain.ask(q)
            except Exception as e: r = {"answer": f"EXCEPTION {type(e).__name__}: {e}"}
            fr = (r.get("facts") or {}).get("frame") if isinstance(r.get("facts"), dict) else None
            row = {"db": name, "q": q, "expect": exp, "answer": str(r.get("answer", "")), "intent": r.get("intent"), "frame": fr, "ms": round((time.perf_counter() - t) * 1000), "conf": r.get("confidence")}
            res.append(row)
            print(f"[{name[6:]:7s}] {q[:70]}\n    -> {row['answer'][:a.width]}" + (f"\n    ?? {exp}" if exp else ""), flush=True)
    if a.out: json.dump(res, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

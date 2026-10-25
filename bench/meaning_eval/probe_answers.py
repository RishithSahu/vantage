"""Bug-hunting probe: ask the meaning pipeline questions from plain question files and write `question | read as | answer` lines for a person to read.  This is NOT an accuracy measurement (there is no gold
frame); it exists to find hallucinated numbers, wrong refusals and executor crashes on wording nobody authored for this model.

    python bench/meaning_eval/probe_answers.py --adapter models/meaning-lora-full.gguf --out results/meaning/probe_legacy_full.txt bench/everyday_questions.txt bench/heldout_final.tsv ...

A file is one question per line; for .tsv files the LAST tab-separated column is the question (route<TAB>question).  Lines starting with # are skipped."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DB = os.path.join(ROOT, "data", "gen_demo.db")
NOW = 1787625900


def questions(path):
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"): continue
        yield line.split("\t")[-1].strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--chars", type=int, default=170)
    a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    brain = make_meaning_brain(DB, now=NOW, threads=a.threads, adapter=os.path.join(ROOT, a.adapter) if not os.path.isabs(a.adapter) else a.adapter)
    lines = []
    for f in a.files:
        lines.append(f"##### {os.path.basename(f)}")
        for q in questions(f):
            brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []
            t = time.perf_counter()
            try: r = brain.ask(q)
            except Exception as e: r = {"answer": f"EXCEPTION {type(e).__name__}: {e}", "facts": {}}
            fr = (r.get("facts") or {}).get("frame") or {}
            read = " ".join(fr.values()) if fr else "-"
            lines.append(f"{q}\n    [{read}] {' '.join(str(r.get('answer')).split())[:a.chars]}  ({round((time.perf_counter() - t) * 1000)} ms)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n"); print(len(lines), "lines ->", a.out)


if __name__ == "__main__":
    main()

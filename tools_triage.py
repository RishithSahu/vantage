"""Turn a failing question into DATA, never into a code patch.

    python tools_triage.py "how many ppl can fit in my car" --expect "spec seat_count" [--adapter models/meaning-lora-full.gguf] [--apply]
    python tools_triage.py --batch rows.txt [--apply]        # rows.txt: one "question | act concept [op=..] [cond=..] [vs=..] [ent=..]" per line; the model is loaded once

What it does
  1. runs the frame model on the question with and without glossary hints;
  2. decides which kind of gap this is and prints the ONE data change that closes it:
       word      a word the model likely does not know (not in the glossary or lexicon)      -> a glossary.jsonl line (term / forms / meaning / hint)
       ontology  the expected concept does not exist                                         -> an ontology_data.json skeleton (spec / topic) - needs a human to confirm its data source
       model     the words are known and the concept exists, the model still got the frame wrong -> a natural training row appended to training/meaning/natural_feedback.txt
       executor  the frame is right but the resolver has no executor / answered "gap"          -> a coverage-gap record (data/ coverage list), executor work is a separate task
  3. with --apply, writes the suggested glossary line / training row (never touches code, never edits a frozen evaluation file).

Every call is appended to training/meaning/triage_log.jsonl so the history of what was fixed by which data is auditable.  A question that appears in a frozen evaluation set is refused:
fixing it would invalidate the measurement.  To learn from a failing frozen question, write a NEW question in different words that has the same meaning and triage that."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vgpt.meaning import glossary as GL, ontology                      # noqa: E402
from vgpt.meaning.frame import Frame                                   # noqa: E402

FEEDBACK = os.path.join(HERE, "training", "meaning", "natural_feedback.txt")
LOG = os.path.join(HERE, "training", "meaning", "triage_log.jsonl")
EVAL_FILES = ["novel_v1.tsv", "messy_v1.tsv", "user_v1.tsv", "sealed_v2.tsv"]


def tokens(s):
    import re
    return re.findall(r"[a-z0-9]+", s.lower())


def frozen(text):
    key = " ".join(tokens(text))
    for f in EVAL_FILES:
        for line in open(os.path.join(HERE, "bench", "meaning_eval", f), encoding="utf-8"):
            if line.startswith("#") or not line.strip(): continue
            if " ".join(tokens(line.split("\t")[1])) == key: return f
    return None


def known_words():
    O = ontology.load(); G = GL.load(); known = set()
    lex = json.load(open(os.path.join(HERE, "vgpt", "meaning", "lexicon.json"), encoding="utf-8"))
    for ps in lex.values():
        if isinstance(ps, list):
            for p in ps: known.update(tokens(p))
    for e in G.entries:
        for f in [e["term"]] + list(e.get("forms", ())): known.update(tokens(f))
    for c in O.concepts.values():
        for a in c.aliases: known.update(tokens(a))
    return known


def triage(question, expect, O, fm, known, apply=False):
    """-> the record that is also written to the log.  fm is a loaded FrameModel, or None for --no-model."""
    toks = expect.split(); kw = dict(op="none", cond="none", vs="none", ent="none")
    for t in toks[2:]:
        k, v = t.split("="); kw[k] = v
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": question, "expected": " ".join([toks[0], toks[1], kw["op"], kw["cond"], kw["vs"], kw["ent"]])}
    if toks[1] not in O.concepts:
        out.update(kind="ontology", fix={"ontology_data.json": {"spec_concepts or topic_concepts": {toks[1]: {"label": "...", "gloss": "...", "meta": ["veh.<key>"], "unit": "...", "component": "...", "quantity": "..."}}}},
                   note="the concept does not exist: add it to vgpt/meaning/ontology_data.json (a human must confirm where its data comes from), then retrain")
    else:
        exp = Frame(toks[0], toks[1], kw["op"], kw["cond"], kw["vs"], kw["ent"]); issues = O.validate(exp)
        if issues: sys.exit(f"the expected frame is ill-typed: {issues}")
        pred = pred_nh = None
        if fm is not None:
            pred, info = fm.parse(question); pred_nh, _ = fm.parse(question, hint="")
            out["predicted"] = pred.to_text() if pred else None; out["predicted_without_hints"] = pred_nh.to_text() if pred_nh else None; out["hint"] = info["hint"]
        unknown = [w for w in dict.fromkeys(tokens(question)) if w not in known and not w.isdigit() and len(w) > 2]
        if pred == exp: out.update(kind="ok", note="the model already gets this frame right")
        elif unknown and (pred_nh is None or pred_nh != exp):
            out.update(kind="word", unknown_words=unknown, fix={"glossary.jsonl": {"term": unknown[0], "forms": [], "kind": "slang", "meaning": "<what it means>", "concept": toks[1], "hint": True}},
                       note="the question contains words the glossary and lexicon do not know: add what they MEAN (a hint line is shown to the model immediately, no retraining)")
        else:
            row = f"{question} | {exp.act} {exp.concept}" + "".join(f" {k}={getattr(exp, k)}" for k in ("op", "cond", "vs", "ent") if getattr(exp, k) != "none")
            out.update(kind="model", fix={"natural_feedback.txt": row}, note="all words are known and the concept exists: the model needs more varied examples of this meaning (added to the next training run)")
            if apply:
                with open(FEEDBACK, "a", encoding="utf-8") as fh: fh.write(row + "\n")
                out["applied"] = FEEDBACK
    if apply and out.get("kind") == "word":
        gl = os.path.join(HERE, "vgpt", "meaning", "glossary.jsonl"); line = json.dumps(out["fix"]["glossary.jsonl"], ensure_ascii=False)
        out["applied"] = "edit the meaning in " + gl + ": " + line + "   (not written automatically: a meaning must be written by a person)"
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh: fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?"); ap.add_argument("--expect", help="the frame the question should get: 'act concept [op=.. cond=.. vs=.. ent=..]'")
    ap.add_argument("--batch", help="a file of 'question | frame' lines instead of one question")
    ap.add_argument("--adapter", default=os.path.join("models", "meaning-lora-full.gguf")); ap.add_argument("--apply", action="store_true"); ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args(); O = ontology.load(); known = known_words(); fm = None
    if a.batch:
        rows = []
        for line in open(a.batch, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"): continue
            q, spec = line.rsplit(" | ", 1); rows.append((q.strip(), spec.strip()))
    elif a.question and a.expect: rows = [(a.question, a.expect)]
    else: ap.error("give a question and --expect, or --batch FILE")
    if not a.no_model:
        from vgpt.meaning import interface
        fm = interface.FrameModel(os.path.join(HERE, a.adapter) if not os.path.isabs(a.adapter) else a.adapter, use_hints=True)
    kinds = {}
    for q, spec in rows:
        hit = frozen(q)
        if hit:
            if not a.batch: sys.exit(f"refused: this question is in the frozen evaluation set {hit}; fixing it would invalidate the measurement")
            print(f"REFUSED (frozen {hit}): {q}"); kinds["refused"] = kinds.get("refused", 0) + 1; continue
        out = triage(q, spec, O, fm, known, a.apply); kinds[out["kind"]] = kinds.get(out["kind"], 0) + 1
        if a.batch: print(f"{out['kind']:8s} | {q} | want {out['expected']} | got {out.get('predicted')}")
        else: print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.batch: print("summary", json.dumps(kinds))


if __name__ == "__main__":
    main()

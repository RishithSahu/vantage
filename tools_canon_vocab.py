"""Build vgpt/meaning/canon_vocab.json, the word list the pre-parser's spelling correction relies on.

    python tools_canon_vocab.py [--holdout]       --holdout leaves out the evaluation sets named in HELD so that tests/test_canon.py can measure on questions the list has never seen

The list is the words of CLEAN vehicle questions: the training questions (not their typo variants), the clean evaluation sets, and the 500-question QA file.  A word that occurs in any of them is a real word and is never
touched.  A word may only be the *target* of a correction when it is frequent (it occurs at least TARGET_MIN times), so a rare spelling cannot become the "correct" form of another word."""
import collections
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_MIN = 8
MISSPELT_MAX = 60
RATIO = 8
MISSPELT_MAX_HI = 200
RATIO_HI = 25
HELD = ("sealed_v7", "sealed_v6", "user_v1", "hard_v5", "holdout_v10", "wording_v1", "novel_v1")          # kept out of the list for the held-out measurement
NOISY = re.compile(r"typo|noise|noisy|messy|voice|abbrev|mixing|fragment", re.I)


def words(text): return re.findall(r"[a-z]+", str(text).lower())


def corpus(holdout=False):
    cnt = collections.Counter()
    for f in glob.glob(os.path.join(ROOT, "training", "meaning", "data", "frame2_r1[0-9]", "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except ValueError: continue
            if NOISY.search(str(r.get("src", ""))) or NOISY.search(str(r.get("group", ""))): continue
            cnt.update(set(words(r.get("text", ""))))
    for f in glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv")):
        base = os.path.basename(f)[:-4]
        if base.endswith("_expected") or NOISY.search(base) or (holdout and base in HELD): continue
        for line in open(f, encoding="utf-8"):
            if line.startswith("#") or "\t" not in line: continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 3 and NOISY.search(cols[-1]): continue
            cnt.update(set(words(cols[1])))
    try:
        for r in csv.DictReader(open(os.path.join(ROOT, "VehicleGPT_500_QA.csv"), encoding="utf-8-sig")): cnt.update(set(words(r.get("question", ""))))
    except OSError: pass
    return cnt


def english():
    """Whole English words from the language model's own vocabulary (tokens that begin with a space): about 24 000 real words and no misspellings."""
    from llama_cpp import Llama
    m = Llama(model_path=os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), vocab_only=True, verbose=False); out = set()
    for i in range(m.n_vocab()):
        try: b = m.detokenize([i]).decode("utf-8")
        except Exception: continue
        if b.startswith(" ") and re.fullmatch(r"[a-z]{3,}", b[1:]): out.add(b[1:])
    return out


def build(holdout=False):
    """known = English words, the vehicle's own glossary words and every corpus word that is not a misspelling.  A corpus word that is not English, not in the glossary, occurs at most MISSPELT_MAX times and sits one or two edits
    from a word at least RATIO times as frequent is a misspelling of it (the training data holds typo variants of its questions: 'lowst' x8 against 'lowest' x1129)."""
    sys.path.insert(0, ROOT)
    from vgpt.meaning.canon import _within, _ENGLISH, _inflection_of_known
    from vgpt.meaning import glossary as G, ontology as ON
    cnt = corpus(holdout); eng = english() | set(_ENGLISH)
    for e in G.load().entries:
        for t in [e.get("term", "")] + list(e.get("forms") or []): eng.update(words(t))
    for c in ON.load().concepts.values(): eng.update(words(getattr(c, "label", "") or ""))
    by = collections.defaultdict(list)
    for w, n in cnt.items(): by[w[:1]].append((w, n))
    fix = {}
    for w, n in cnt.items():
        if len(w) < 5 or n > MISSPELT_MAX_HI or w in eng: continue
        k = 1 if len(w) < 8 else 2
        best = [(v, m) for v, m in by[w[:1]] if m >= (RATIO if n <= MISSPELT_MAX else RATIO_HI) * n and abs(len(v) - len(w)) <= k and len(v) >= 4 and v != w and _within(w, v, k) and not _inflection_of_known(w, {v})]
        near = [x for x in best if _within(w, x[0], 1)]
        pool = sorted(near or best, key=lambda x: -x[1])
        if pool and (len(pool) == 1 or pool[0][1] >= 30 * pool[1][1]): fix[w] = pool[0][0]                          # the nearest word first; when two are about equally likely nothing is decided
    for w in list(fix):                                                                                              # a correction that leads to another misspelling is followed to the real word
        seen = 0
        while fix[w] in fix and seen < 5: fix[w] = fix[fix[w]]; seen += 1
    known = sorted((set(w for w, n in cnt.items() if len(w) >= 2) | eng) - set(fix))
    cands = sorted((eng | set(w for w, n in cnt.items() if n >= TARGET_MIN)) - set(fix))
    return {"known": known, "cands": [w for w in cands if len(w) >= 4], "target": {w: n for w, n in cnt.items() if n >= TARGET_MIN and len(w) >= 4 and w not in fix}, "fix": fix, "holdout": bool(holdout)}


def main():
    d = build("--holdout" in sys.argv); out = os.path.join(ROOT, "vgpt", "meaning", "canon_vocab.json") if not d["holdout"] else os.path.join(ROOT, "results", "_canon_vocab_holdout.json")
    json.dump(d, open(out, "w", encoding="utf-8"), separators=(",", ":")); print(out, len(d["known"]), "known,", len(d["target"]), "targets,", len(d["fix"]), "corpus misspellings")


if __name__ == "__main__": main()

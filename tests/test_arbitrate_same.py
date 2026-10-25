"""arbitrate._same: the memoised, bound-checked version equals the plain SequenceMatcher definition on real vocabulary and on random strings."""
import difflib
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.meaning import arbitrate as A, literals as L


def plain(a, b):
    if a == b: return True
    return len(a) >= 4 and len(b) >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def test_identical_on_vocabulary_and_random_words():
    from vgpt.meaning import glossary as G
    words = sorted({w for k in G.load().index for w in k.split()})[:600]
    r = random.Random(3); rnd = ["".join(r.choice("abcdefgh") for _ in range(r.randint(3, 9))) for _ in range(800)]
    pool = words + rnd; pairs = [(r.choice(pool), r.choice(pool)) for _ in range(30000)] + [(w, w[:-1] + "x") for w in words if len(w) > 4]
    assert all(A._same(a, b) == plain(a, b) for a, b in pairs)


def test_arbitrate_unchanged_on_every_frozen_question():
    """The first-word index and the memoised comparison give the same frame and note as the reference arbitrate on the questions of the frozen sets."""
    import glob, importlib.util
    from vgpt.meaning import glossary as G, ontology as ON
    from vgpt.meaning.frame import Frame
    spec = importlib.util.spec_from_file_location("_arb_ref", os.path.join(HERE, "tests", "_arbitrate_reference.py")); REF = importlib.util.module_from_spec(spec); REF.__package__ = "vgpt.meaning"; spec.loader.exec_module(REF)
    onto = ON.load(ON.active_version()); gl = G.load(); texts = []
    for f in sorted(glob.glob(os.path.join(HERE, "bench", "meaning_eval", "*.tsv")))[:40]:
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if len(p) > 1: texts.append(p[1])
    concepts = [c for c in onto.concepts][:40]; n = 0
    for i, t in enumerate(texts[::3][:400]):
        for cid in (concepts[i % len(concepts)], "unlisted"):
            fr = Frame("stat", cid, "mean", "none", "none", "none")
            for rules in ("B", "ABC"):
                assert A.arbitrate(t, fr, onto, gl, rules=rules)[1] == REF.arbitrate(t, fr, onto, gl, rules=rules)[1]
                assert A.arbitrate(t, fr, onto, gl, rules=rules)[0] == REF.arbitrate(t, fr, onto, gl, rules=rules)[0]; n += 1
    assert n > 500

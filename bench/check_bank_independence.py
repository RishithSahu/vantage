"""How close is the generated demonstration bank to the evaluation questions? For every demonstration, the highest bge-small cosine similarity to any question of the development, final, diagnostic and everyday files (the sealed acceptance file is NOT read). Prints the distribution and the closest pairs so that a leak (a demonstration that is a paraphrase of an evaluation question) is visible and removable at the source, in vgpt/demos.py.
    python bench/check_bank_independence.py"""
import glob, os, sqlite3, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from vgpt import capreg, demos, signal_registry as R
from vgpt.retrieval import Embedder


def main():
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True))); emb = Embedder(os.path.join(HERE, "models", "bge-small-en-v1.5"), int8=True)
    bank = demos.generate(reg); q = []; src = []
    for f in sorted(glob.glob(os.path.join(HERE, "bench", "*.tsv"))):
        if "acceptance" in os.path.basename(f): continue
        try: rows = load(f)
        except Exception: continue
        q += [r["text"] for r in rows]; src += [os.path.basename(f)] * len(rows)
    E = emb.encode(q); B = emb.encode([b["q"] for b in bank]); S = B @ E.T; mx = S.max(1); am = S.argmax(1)
    print(f"{len(bank)} demonstrations vs {len(q)} evaluation questions from {len(set(src))} files; similarity: median {np.median(mx):.2f}, p90 {np.percentile(mx, 90):.2f}, max {mx.max():.2f}; >=0.90: {(mx >= .9).sum()}, >=0.85: {(mx >= .85).sum()}, >=0.80: {(mx >= .8).sum()}")
    for i in np.argsort(-mx)[:12]: print(f"  {mx[i]:.2f}  demo: {bank[i]['q']!r}   eval ({src[am[i]]}): {q[am[i]]!r}")


if __name__ == "__main__": main()

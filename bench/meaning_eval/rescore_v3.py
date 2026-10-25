"""Read the predictions of a frame-2 run, score them against full_v11, and apply the catalogue-equivalence rule WITHOUT touching any frozen file.

    python bench/meaning_eval/rescore_v3.py [--prefix frame2_] [--out results/meaning/frame2_report.md]

Why a rescoring rule is needed.  Some frozen sets (written before the universal signal catalogue existed) give the gold concept `unlisted` for things such as "what gear am i in", "oil temperature", "turbo boost pressure",
"what is the adblue level", "exhaust temperature": at the time no concept existed for them.  Since 2026-10-21 the catalogue (vgpt/ingest/catalog.json, docs/DATA_SOURCES.md) lists them, and the model is trained to
name them.  A prediction is therefore counted as EQUIVALENT to a gold `unlisted` when (1) its concept is one the catalogue added, (2) the catalogue's label or one of its synonyms occurs in the question text, and
(3) every other word of the frame equals the gold's.  Both numbers are always printed - `strict` (text equality, no rule) and `adjusted` (with the rule) - and the rows the rule moved are listed, so the rule can be
audited.  The frozen files and their sha256 are not touched."""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "meaning")

# set -> (where the frame-2 predictions are, where full_v11's are, the key inside full_v11's predictions)
SETS = {"novel_v1": ("novel_v1", "full_v11", "novel_v1"), "messy_v1": ("messy_v1", "full_v11", "messy_v1"), "user_v1": ("user_v1", "full_v11", "user_v1"), "sealed_v2": ("sealed_v2", "full_v11_sealed", "sealed_v2"),
        "sealed_v3": ("sealed_v3", "full_v11_sealed3", "sealed_v3"), "hard_v1": ("hard_v1", "full_v11_hard1", "hard_v1"), "hard_v2": ("hard_v2", "full_v11_hard2", "hard_v2"), "hard_v3": ("hard_v3", "full_v11_hard3", "hard_v3"),
        "hard_v4": ("hard_v4", "full_v11_hard4", "hard_v4"), "hard_v5": ("hard_v5", "full_v11_hard5", "hard_v5"), "platform_v1": ("platform_v1", "full_v11_platform", "platform_v1"),
        "platform_v2": ("platform_v2", "full_v11_platform2", "platform_v2"), "adv_v1": ("adv_v1", "full_v11_adv1", "adv_v1"), "adv_v2": ("adv_v2", "full_v11_adv2", "adv_v2"), "adv_v3": ("adv_v3", "full_v11_adv3", "adv_v3")}


def catalogue():
    c = json.load(io.open(os.path.join(ROOT, "vgpt", "ingest", "catalog.json"), encoding="utf-8")); out = {}
    for x in c["concepts"]: out[x["name"]] = sorted({x.get("label", "").lower(), *[s.lower() for s in x.get("synonyms", [])]} - {""}, key=len, reverse=True)
    return out


CAT = catalogue()


def equivalent(text, gold, pred):
    """The rule above, on the text forms of two frames (the words, space separated)."""
    g, p = pad(gold).split(), pad(pred).split()
    if g[1] != "unlisted" or p[1] not in CAT: return False
    if [w for i, w in enumerate(g) if i != 1] != [w for i, w in enumerate(p) if i != 1]: return False
    t = text.lower()
    return any(re.search(r"\b" + re.escape(s) + r"\b", t) for s in CAT[p[1]])


def load_preds(folder, key):
    p = os.path.join(RES, folder, "predictions.json")
    if not os.path.exists(p): return None
    d = json.load(io.open(p, encoding="utf-8")); return [x for x in d["predictions"] if x["set"] == key]


def pad(frame, n=8):
    w = (frame or "").split(); return " ".join(w + ["none"] * (n - len(w))) if w else ""


def score(rows):
    strict = sum(1 for r in rows if r["ok"] or pad(r["pred"]) == pad(r["gold"])); moved = [r for r in rows if not (r["ok"] or pad(r["pred"]) == pad(r["gold"])) and equivalent(r["text"], pad(r["gold"]), pad(r["pred"]))]
    return strict, len(moved), moved


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="frame2_"); ap.add_argument("--out", default=os.path.join(RES, "frame2_report.md")); a = ap.parse_args()
    lines = ["| set | n | full_v11 | frame2 strict | frame2 adjusted | change vs v11 (adjusted) | rows moved by the rule |", "|---|---|---|---|---|---|---|"]; tot = collections.Counter(); moved_all = []; worst = 0.0
    for name, (folder, v11, key) in SETS.items():
        mine = load_preds(a.prefix + folder, key); old = load_preds(v11, key)
        if mine is None: lines.append(f"| {name} | - | - | not run | | | |"); continue
        n = len(mine); s, m, rows = score(mine); o = sum(1 for r in old if r["ok"]) if old else None; adj = s + m
        d = (adj / n - o / len(old)) * 100 if old else None; worst = min(worst, d if d is not None else 0)
        tot["n"] += n; tot["strict"] += s; tot["adj"] += adj; tot["old"] += (o if o is not None else 0); tot["old_n"] += (len(old) if old else 0)
        lines.append(f"| {name} | {n} | {100 * o / len(old):.1f} % | {100 * s / n:.1f} % | {100 * adj / n:.1f} % | {d:+.1f} | {m} |"); moved_all += [(name, r) for r in rows]
    if tot["n"]:
        lines.append(f"| **all** | {tot['n']} | {100 * tot['old'] / tot['old_n']:.1f} % | {100 * tot['strict'] / tot['n']:.1f} % | {100 * tot['adj'] / tot['n']:.1f} % | {100 * (tot['adj'] / tot['n'] - tot['old'] / tot['old_n']):+.1f} | {len(moved_all)} |")
        lines.append(""); lines.append(f"Gate (a) of docs/FRAME_V2.md: no set loses more than 1.5 points against full_v11 (worst change here {worst:+.1f}) and the mean does not fall ({100 * (tot['adj'] / tot['n'] - tot['old'] / tot['old_n']):+.1f}).")
    if moved_all:
        lines += ["", "Rows counted as equivalent by the catalogue rule (gold `unlisted` -> a catalogue concept named in the question):", ""]
        for name, r in moved_all: lines.append(f"* {name} {r['id']}: \"{r['text'][:90]}\" gold `{r['gold']}` -> predicted `{r['pred']}`")
    text = "\n".join(lines); print(text); io.open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()

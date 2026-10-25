"""Frame version 2 evaluation: single questions that need the new slots, and two-turn dialogues.

    python bench/meaning_eval/eval_v2.py --adapter models/meaning-lora-full_v11.gguf --version 1 --name v11_baseline
    python bench/meaning_eval/eval_v2.py --adapter models/meaning-lora-full_v12.gguf --version 2 --name v12

Sets: slots_v2.tsv (84), followup_v2.tsv (72).  With `--version 1` (the shipped adapter) the model can only say six words and cannot see the previous turn, so what is scored is how much of the gold the six
words can carry: `six` = the first six words equal the gold's.  With `--version 2` the full eight words are scored (`exact`), and `six` too.
Per tag the report shows exact / six; `new slot` = c2 or ref (or vs=typical, op=corr, act=explain) right, which is what the version-1 frame cannot express.
Output: results/meaning/v2_<name>.json (every row)."""
import argparse
import collections
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load(name, prev):
    rows = []
    for line in io.open(os.path.join(HERE, name + ".tsv"), encoding="utf-8"):
        if line.startswith("#") or not line.strip(): continue
        p = line.rstrip("\n").split("\t")
        rows.append(dict(id=p[0], prev=p[1], text=p[2], gold=p[3], tag=p[4]) if prev else dict(id=p[0], prev=None, text=p[1], gold=p[2], tag=p[3]))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--version", type=int, default=1); ap.add_argument("--name", required=True); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--sets", default="slots_v2,followup_v2", help="comma separated; a name starting with followup is a two-turn set"); a = ap.parse_args(); SETS = [x.strip() for x in a.sets.split(",") if x.strip()]
    os.environ["VGPT_FRAME"] = str(a.version)
    from vgpt.meaning.interface import FrameModel
    m = FrameModel(os.path.join(ROOT, a.adapter) if not os.path.isabs(a.adapter) else a.adapter, threads=a.threads, version=a.version); out = []; t0 = time.time()
    for setname, prev in [(x, x.startswith("followup")) for x in SETS]:
        for r in load(setname, prev):
            f, info = m.parse(r["text"], prev=r["prev"] if (prev and a.version >= 2) else None); got = f.to_text(a.version) if f else None
            g = r["gold"].split(); gw = got.split() if got else []
            out.append(dict(r, set=setname, pred=got, exact=got == r["gold"] if a.version >= 2 else None, six=gw[:6] == g[:6], new=(gw[6:8] == g[6:8] and gw[4] == g[4] and gw[2] == g[2] and gw[0] == g[0]) if a.version >= 2 else False, pmin=info.get("pmin"), ms=info.get("ms")))
    print(f"{len(out)} questions in {time.time() - t0:.0f} s\n")
    for setname in SETS:
        rows = [o for o in out if o["set"] == setname]; by = collections.defaultdict(list)
        for o in rows: by[o["tag"]].append(o)
        print(f"{setname}: n={len(rows)}   exact {100 * sum(bool(o['exact']) for o in rows) / len(rows):5.1f}%   first six words {100 * sum(o['six'] for o in rows) / len(rows):5.1f}%")
        for tag, rs in by.items(): print(f"   {tag:10} n={len(rs):3}   exact {100 * sum(bool(o['exact']) for o in rs) / len(rs):5.1f}%   six {100 * sum(o['six'] for o in rs) / len(rs):5.1f}%")
    os.makedirs(os.path.join(ROOT, "results", "meaning"), exist_ok=True)
    json.dump({"adapter": a.adapter, "version": a.version, "rows": out}, open(os.path.join(ROOT, "results", "meaning", f"v2_{a.name}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

"""Run questions with a GIVEN frame through the resolver (no language model), to test executors on their own.

    python bench/ask_frame.py FILE.txt --db data/world_kestrel.db [data/world_rhino.db ...] [--width 300]

FILE.txt: one case per line:  question | act concept [op=.. cond=.. vs=.. ent=.. c2=.. ref=..]    (# comments).  The answer text is printed under each case."""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from vgpt.meaning import glossary as GL                      # noqa: E402
from vgpt.meaning.frame import Frame                         # noqa: E402
from vgpt.meaning.pipeline import MeaningBrain               # noqa: E402
from vgpt.vehicle import lake as LK                          # noqa: E402


class Null:
    glossary = GL.load()
    def parse(self, text, hint=None, prev=None): raise RuntimeError("not used")


def last_ts(path):
    import sqlite3
    c = sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)
    try:
        for t in ("sig_1s", "sig_1m"):
            r = c.execute(f"SELECT MAX(ts) FROM {t}").fetchone()[0]
            if r: return int(r)
    finally: c.close()


def frame_of(spec):
    t = spec.split(); kw = dict(op="none", cond="none", vs="none", ent="none", c2="none", ref="none"); rest = t[1:]
    concept = rest.pop(0) if rest and "=" not in rest[0] else "none"
    for x in rest:
        k, v = x.split("="); kw[k] = v
    return Frame(t[0], concept, **kw)


def main():
    os.environ["VGPT_FRAME"] = "2"
    ap = argparse.ArgumentParser(); ap.add_argument("cases"); ap.add_argument("--db", nargs="+", required=True); ap.add_argument("--width", type=int, default=300); a = ap.parse_args()
    cases = []
    for ln in io.open(a.cases, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"): continue
        q, _, spec = ln.rpartition(" | "); cases.append((q.strip(), spec.strip()))
    for d in a.db:
        path = d if os.path.isabs(d) else os.path.join(ROOT, d); LK.adopt_path(path); b = MeaningBrain(path, Null(), now=last_ts(path), tau=0.0)
        for q, spec in cases:
            b.qcs.last = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []; b._last_data = None
            try: r = b.resolver.answer(q, frame_of(spec)); ans = str(r.get("answer", ""))
            except Exception as e: ans = f"EXCEPTION {type(e).__name__}: {e}"
            print(f"[{os.path.basename(path)[6:-3]:8s}] {q}  <{spec}>\n    -> {ans[:a.width].replace(chr(10), ' / ')}")


if __name__ == "__main__":
    main()

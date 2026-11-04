"""Every question the project owns, timed through the whole assistant, split into model time and the rest (database, analysis, answer text).

    python bench/full_pass.py --db data/aster_ev_4yr.db --out results/full_pass_laptop.json [--limit N] [--start N] [--count N]

Limits (owner, 2026-10-27): a short prompt (under 37 words) answers in under 2 s, a long one (over 40 words) in under 2.5 s, and the work after the model (database access and answer building) takes at most 0.2 s.
Run with `taskset -c 0,1` on the Pi.  Each question is asked on its own (the dialogue is reset), as the first turn of a session."""
import argparse, csv, glob, io, json, os, statistics, sys, time
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def questions():
    import eval_r10 as E
    seen, out = set(), []
    def add(src, q):
        q = " ".join(str(q).split())
        if q and q.lower() not in seen: seen.add(q.lower()); out.append((src, q))
    for ln in io.open(os.path.join(HERE, "acceptance_v1.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if not ln.startswith("#") and len(p) > 2: add("acc", p[2])
    for s in E.FROZEN + E.NEW:
        for r in E.load(s): add(s, r["text"])
    f = os.path.join(ROOT, "VehicleGPT_500_QA.csv")
    if os.path.exists(f):
        with io.open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh): add("qa500", r["question"])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default="data/aster_ev_4yr.db"); ap.add_argument("--out", default="results/full_pass.json"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=0); ap.add_argument("--list", action="store_true"); ap.add_argument("--only-slow-in", help="a full-pass file: ask only the questions whose time after the model was above --min-rest there"); ap.add_argument("--min-rest", type=float, default=0.02); ap.add_argument("--freeze", action="store_true", help="stop the vehicle clock, so two runs ask the same questions at the same moment (answers can be compared)"); ap.add_argument("--freeze-at", type=int, default=0, help="the moment to stop it at (epoch seconds); without it, the moment of the start"); a = ap.parse_args()
    import vehiclegpt; vehiclegpt._adopt_product_model()                                             # first: the evaluation module below sets VGPT_FRAME, which would make the product skip models/product.json
    qs = questions()
    if a.list: print(len(qs)); return
    if a.only_slow_in: keep = {r["q"] for r in json.load(open(a.only_slow_in)) if r["rest"] > a.min_rest}; qs = [x for x in qs if x[1] in keep]
    qs = qs[a.start:(a.start + a.count) if a.count else None]
    from ui import server as S
    from vgpt.meaning import ort_backend as OB
    STAGE = {}; _p = OB.OrtFrameModel.parse
    def _parse(self, *x, **k):
        t = time.perf_counter(); r = _p(self, *x, **k); STAGE.clear(); STAGE.update(self.stats, parse_s=time.perf_counter() - t); return r
    OB.OrtFrameModel.parse = _parse
    if os.environ.get("NO_RIG", "1") == "1": S.CameraRig.start = lambda self: None
    app = S.App(SimpleNamespace(db=a.db, model="0.5b", camera="clip", webcam_index=0, live=None, linked=False)); w = app.ask("how fast am I going", "driver")
    if not app.brain.model_loaded: sys.exit("the language model did not load: %s" % app.brain.load_error)
    from vgpt import colstore; tw = time.monotonic()
    while time.monotonic() - tw < 600:                                                       # the column copy of the history is built in the background when the app starts: the pass starts when it is there (as it is for a car that has run for a while)
        with colstore._LOCK:
            if not colstore._BUILDING: break
        time.sleep(0.5)
    print("column copy ready after %.0fs" % (time.monotonic() - tw), flush=True)
    rows = []; t0 = time.monotonic()
    for i, (src, q) in enumerate(qs):
        d = getattr(app.brain, "dialogue", None)
        if d is not None and hasattr(d, "reset"): d.reset()
        STAGE.clear(); t = time.perf_counter()
        try: r = app.ask(q, "driver"); ans = str(r.get("answer"))
        except Exception as e: ans = "EXC " + repr(e)[:80]
        dt = time.perf_counter() - t; ps = STAGE.get("parse_s", 0.0)
        rows.append({"src": src, "q": q, "words": len(q.split()), "s": round(dt, 3), "model": round(ps, 3), "rest": round(dt - ps, 3), "tail": STAGE.get("tail_tokens"), "passes": STAGE.get("passes"), "a": ans})
        if (i + 1) % 200 == 0: print(i + 1, "of", len(qs), "%.0fs" % (time.monotonic() - t0), flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True); json.dump(rows, io.open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), ensure_ascii=False)
    print("done", len(rows), "questions in %.0fs" % (time.monotonic() - t0))


if __name__ == "__main__": main()

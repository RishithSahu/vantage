"""Which database statements cost the most across a whole set of questions?   python bench/db_hot.py results/full_pass_X.json [--min-rest 0.03] [--limit N]
Takes the questions whose time after the model was above --min-rest in a full-pass file, asks each one with every statement timed (as bench/db_spy.py), and totals the time per statement shape and per calling function."""
import argparse, collections, json, os, re, sqlite3, sys, time, traceback
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
ap = argparse.ArgumentParser(); ap.add_argument("pass_file"); ap.add_argument("--db", default="data/aster_ev_4yr.db"); ap.add_argument("--min-rest", type=float, default=0.03); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--min", type=float, default=0.002); a = ap.parse_args()
Q = [r["q"] for r in json.load(open(a.pass_file)) if r["rest"] > a.min_rest]
if a.limit: Q = Q[:a.limit]
import vehiclegpt; vehiclegpt._adopt_product_model()
from ui import server as S
S.CameraRig.start = lambda self: None
LOG = []
class Cur:
    def __init__(self, cur, sql, t): self.cur, self.sql, self.t0 = cur, sql, t
    def __getattr__(self, k): return getattr(self.cur, k)
    def __iter__(self):
        n = 0
        for r in self.cur: n += 1; yield r
        self._done(n)
    def fetchall(self): r = self.cur.fetchall(); self._done(len(r)); return r
    def fetchone(self): r = self.cur.fetchone(); self._done(1 if r else 0); return r
    def _done(self, n):
        dt = time.perf_counter() - self.t0
        if dt >= a.min:
            fr = [f for f in traceback.extract_stack()[:-2] if "vehiclegpt2" in f.filename and "db_hot" not in f.filename][-2:]
            LOG.append((dt, " ".join(re.sub(r"\d+(\.\d+)?", "N", self.sql).split())[:150], " <- ".join("%s %s" % (os.path.basename(f.filename), f.name) for f in reversed(fr))))
class Conn(sqlite3.Connection):
    def execute(self, sql, *args): t = time.perf_counter(); return Cur(super().execute(sql, *args), sql, t)
_orig = sqlite3.connect; sqlite3.connect = lambda *x, **k: _orig(*x, **{**k, "factory": Conn})
app = S.App(SimpleNamespace(db=a.db, model="0.5b", camera="clip", webcam_index=0, live=None, linked=False)); T = int(app.clock.now()); app.clock.now = lambda: T; app.ask("how fast am I going", "driver")
from vgpt import colstore
for _ in range(1200):                                                                      # the column copy is built at start; the measurement starts when it is there
    with colstore._LOCK:
        if not colstore._BUILDING: break
    time.sleep(0.5)
if os.environ.get("TWICE") == "1":                                                          # steady state: ask everything once so that every column copy the questions need is built, wait for the builds, then measure
    for q in Q:
        app.brain.dialogue.reset() if hasattr(app.brain, "dialogue") else None; app.ask(q, "driver")
    for _ in range(2400):
        with colstore._LOCK:
            if not colstore._BUILDING: break
        time.sleep(0.5)
tot = collections.defaultdict(lambda: [0.0, 0, set()]); PER = []
for i, q in enumerate(Q):
    app.brain.dialogue.reset() if hasattr(app.brain, "dialogue") else None; LOG.clear(); app.ask(q, "driver"); PER.append((sum(x[0] for x in LOG), q))
    for dt, sql, who in LOG: x = tot[(sql, who)]; x[0] += dt; x[1] += 1; x[2].add(q)
for (sql, who), (dt, n, qs) in sorted(tot.items(), key=lambda kv: -kv[1][0])[:30]: print("%.2fs  n=%-4d questions=%-3d %s\n        %s" % (dt, n, len(qs), sql, who))
PER.sort(reverse=True); lim = 0.2
print("\ndatabase time per question: %d questions, over %.1f s: %d, median %.3f s, worst %.2f s" % (len(PER), lim, sum(1 for t, _ in PER if t > lim), PER[len(PER) // 2][0], PER[0][0]))
for t, q in PER[:15]: print("  %.2fs  %s" % (t, q[:90]))

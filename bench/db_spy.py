"""Which database statements does a question run, and how long does each take?   python bench/db_spy.py [--min 0.02] "question" ...
Wraps sqlite3 through a connection factory trace: prints every statement slower than --min seconds with its row count and the Python function that issued it, then the question's time split.  Run with taskset -c 0,1 on the Pi."""
import argparse, os, sqlite3, sys, time, traceback
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
ap = argparse.ArgumentParser(); ap.add_argument("--db", default="data/aster_ev_4yr.db"); ap.add_argument("--min", type=float, default=0.02); ap.add_argument("q", nargs="+"); a = ap.parse_args()
import vehiclegpt; vehiclegpt._adopt_product_model()
from ui import server as S
S.CameraRig.start = lambda self: None
from vgpt import query_executor as QE
LOG = []
class Cur:                                                               # times execute + fetch of one statement
    def __init__(self, cur, sql, args, t): self.cur, self.sql, self.args, self.t0 = cur, sql, args, t
    def __getattr__(self, k): return getattr(self.cur, k)
    def __iter__(self): return self._iter()
    def _iter(self):
        n = 0
        for r in self.cur: n += 1; yield r
        self._done(n)
    def fetchall(self): r = self.cur.fetchall(); self._done(len(r)); return r
    def fetchone(self): r = self.cur.fetchone(); self._done(1 if r else 0); return r
    def _done(self, n):
        dt = time.perf_counter() - self.t0
        if dt >= a.min:
            fr = [f for f in traceback.extract_stack()[:-2] if "vehiclegpt2" in f.filename and "full_pass" not in f.filename and "db_spy" not in f.filename][-3:]
            LOG.append((dt, n, " ".join(self.sql.split())[:230], " <- ".join("%s:%d %s" % (os.path.basename(f.filename), f.lineno, f.name) for f in reversed(fr))))
class Conn(sqlite3.Connection):
    def execute(self, sql, *args): t = time.perf_counter(); return Cur(super().execute(sql, *args), sql, args[0] if args else (), t)
_orig = sqlite3.connect
sqlite3.connect = lambda *x, **k: _orig(*x, **{**k, "factory": Conn})
app = S.App(SimpleNamespace(db=a.db, model="0.5b", camera="clip", webcam_index=0, live=None, linked=False)); app.ask("how fast am I going", "driver")
for q in a.q:
    app.brain.dialogue.reset() if hasattr(app.brain, "dialogue") else None; LOG.clear(); t = time.perf_counter(); r = app.ask(q, "driver"); dt = time.perf_counter() - t
    print("\n=== %.2fs  %s\n    -> %s" % (dt, q, str(r.get("answer"))[:90].replace("\n", " | ")))
    tot = 0
    for d, n, sql, who in sorted(LOG, reverse=True)[:8]: print("   %.3fs rows=%-8d %s\n        %s" % (d, n, sql, who)); tot += d
    print("   statements over %.2fs: %d, together %.2fs" % (a.min, len(LOG), sum(x[0] for x in LOG)))

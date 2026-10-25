import os, sys, time, sqlite3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.cause.data import History
from vgpt.cause import engine as E
from vgpt.vehicle import lake
db = os.path.join(ROOT, "data", "world_aster.db"); lake.adopt_path(db)
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True); now = int(c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]) + 60
for k in range(3):
    t0 = time.perf_counter(); H = History(c, 84, now=now); t1 = time.perf_counter(); F = E.build(H); t2 = time.perf_counter(); d = E.diagnose(H, F=F, since_day=H.D - 14); t3 = time.perf_counter()
    print("History %.0f ms   build %.0f ms   diagnose %.0f ms   total %.0f ms" % ((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t3 - t0) * 1000))
import cProfile, pstats
pr = cProfile.Profile(); pr.enable(); H = History(c, 84, now=now); F = E.build(H); pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(22)

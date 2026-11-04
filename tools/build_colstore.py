"""Build the columnar copies of a database's per-minute tables (vgpt/colstore.py):  python tools/build_colstore.py data/aster_ev_4yr.db [table ...]
Run once after a database is created or restored (the assistant also builds a missing copy in the background the first time a question needs it)."""
import os, sqlite3, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import colstore
db = os.path.abspath(sys.argv[1]); tables = sys.argv[2:]
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
if not tables: tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name") if r[0].endswith("_1m") and (r[0] in ("sig_1m", "ext_1m", "cam_1m") or r[0].startswith("lkv_"))]
cs = colstore.ColStore(db)
for t in tables:
    t0 = time.time(); cs.build(t, [])
    print("%-14s %.1fs" % (t, time.time() - t0), flush=True)

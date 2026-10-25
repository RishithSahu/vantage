"""Print the frame the model writes for each question (no answer): python bench/frames_of.py DB "question" ...   (or -f file)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.product import make_meaning_brain
from vgpt.vehicle import lake as LK
db = sys.argv[1]; args = sys.argv[2:]
qs = [l.rstrip("\n") for l in open(args[1], encoding="utf-8") if l.strip() and not l.startswith("#")] if args and args[0] == "-f" else args
import sqlite3
c = sqlite3.connect("file:" + db.replace("\\", "/") + "?mode=ro", uri=True); now = max(x for x in (c.execute("select max(ts) from sig_1s").fetchone()[0], c.execute("select max(ts) from sig_1m").fetchone()[0]) if x)
now = int(os.environ.get("VGPT_NOW", now)); LK.adopt_path(db); b = make_meaning_brain(db, now=now, threads=4)
for q in qs:
    r = b.ask(q); print(f"{q}\n    frame: {(r.get('slm') or {}).get('raw')}   route={(r.get('facts') or {}).get('route')}")

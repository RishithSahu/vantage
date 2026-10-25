"""ask questions straight through MeaningBrain on a static db (no server): python ask_direct.py DB [-f file | q ...]"""
import os, sys, time
sys.path.insert(0, "C:/Users/rishi/Rishith/TATA Elexi Teliport/vehiclegpt2"); os.chdir("C:/Users/rishi/Rishith/TATA Elexi Teliport/vehiclegpt2")
sys.stdout.reconfigure(encoding="utf-8")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.product import make_meaning_brain
from vgpt.vehicle import lake as LK
db = sys.argv[1]; args = sys.argv[2:]
qs = [l.rstrip("\n") for l in open(args[1], encoding="utf-8") if l.strip() and not l.startswith("#")] if args and args[0] == "-f" else args
import sqlite3
c = sqlite3.connect("file:" + db.replace("\\", "/") + "?mode=ro", uri=True); now = max(x for x in (c.execute("select max(ts) from sig_1s").fetchone()[0], c.execute("select max(ts) from sig_1m").fetchone()[0]) if x); c.close()
now = int(os.environ.get("VGPT_NOW", now)); LK.adopt_path(db); b = make_meaning_brain(db, now=now, threads=4)
for q in qs:
    r = b.ask(q); print(f"* {q}\n   {str(r['answer'])[:380]!r}\n   [{r.get('understood')}] route={(r.get('facts') or {}).get('route')}")

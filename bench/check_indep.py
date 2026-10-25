"""Check an evaluation file: every gold IR parses, validates against the vehicle's data profile, and executes. Prints the problems (a gold query that cannot run is a mistake in the file, not in the system).
    python bench/check_indep.py bench/indep_dev.tsv [--db data/gen_courier_ext.db]"""
import argparse, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.indep_io import load
from vgpt.qc import QueryCompiler
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate
from training.build_dataset import DEFAULT_NOW

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--db", default="data/gen_courier_ext.db"); a = ap.parse_args()
conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); qc = QueryCompiler(conn, now=DEFAULT_NOW); rows = load(a.path); bad = 0; fam = {}; status = {}
for r in rows:
    fam[r["family"]] = fam.get(r["family"], 0) + 1
    for key in ("ir", "prev_ir"):
        if r[key] and r[key] != "op=none":
            try: q = Query.from_text(r[key])
            except IRError as e: print(r["id"], key, "PARSE", e, "|", r[key]); bad += 1; continue
            if key == "ir":
                v = validate(q, qc.profile)
                if v.codes(): print(r["id"], "VALIDATE", v.codes(), "|", r["text"], "|", r[key]); bad += 1; continue
                try: res = qc.execute(q); status[res.status] = status.get(res.status, 0) + 1
                except Exception as e: print(r["id"], "EXEC", type(e).__name__, e); bad += 1
print(len(rows), "rows;", bad, "problems; families", fam, "; execution status", status)

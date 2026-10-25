"""Table of the understanding-versus-format diagnostics (results/dm_*.json written by bench/run_diag_matrix.py).   python bench/show_matrix.py"""
import glob, json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "results", "dm_*_*.json"))):
    d = json.load(open(f, encoding="utf-8")); s = d["summary"]; n = s["n"]; c = s["categories"]
    wrong = {k: v for k, v in c.items() if k not in ("malformed",)}
    rows.append((s["model"][:14], s["format"] + "/" + s.get("evidence", "none") + ("/" + os.path.basename(f)[3:-5]), s["decode"], s.get("schema", "-"), s.get("shots", "-"), f'{s["plan_correct"]}/{n}', f'{s.get("strict_correct", "?")}', s["malformed"], s["prefix_tokens"], s["mean_ms"],
                 ", ".join(f"{k} {v}" for k, v in list(wrong.items())[:7])))
print("| model | format | decode | schema | shots | plan correct (query level) | strict (result + query) | malformed | prefix tok | ms/q | main differences |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(str(x) for x in r) + " |")

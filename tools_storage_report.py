"""How much room does each car's history take, and how much would five years take?

    python tools_storage_report.py [--dir data/world4] [--years 5] [--budget-gb 6]

For every <dir>/<car>/<car>_<n>yr.db: the database file, its archive folder, the span of history, and the size projected to `years` years at the same growth, against the budget.  The projection scales the whole
(database + archive): the database part is bounded (14 days of seconds and the warm minute tier do not grow with age), so the real five-year figure is lower than the one printed."""
import argparse
import glob
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from vgpt import storage_budget as SB


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(ROOT, "data", "world4")); ap.add_argument("--years", type=float, default=5.0); ap.add_argument("--budget-gb", type=float, default=6.0); a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.dir, "*", "*yr.db"))):
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True); u = SB.usage(c); r = SB.report(c, a.years)
        lo, hi = c.execute("SELECT MIN(ts), MAX(ts) FROM sig_1m").fetchone(); a_lo = c.execute("SELECT MIN(lo) FROM archive_idx").fetchone()[0]
        years = ((hi - (a_lo if a_lo else lo)) / 86400.0 / 365.0) if hi else 0.0
        n_sig = c.execute("SELECT COUNT(*) FROM sig_catalog").fetchone()[0]; events = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]; c.close()
        rows.append((os.path.basename(os.path.dirname(p)), u["database"], u["archive"], years, n_sig, events, r["projected_total"]))
    print(f"{'car':26s} {'signals':>7s} {'years':>5s} {'database':>9s} {'archive':>8s} {'total':>8s}  {'projected ' + str(a.years).rstrip('0').rstrip('.') + ' y':>14s}")
    tot = 0
    for name, db, ar, yrs, n, ev, proj in rows:
        tot += db + ar; print(f"{name:26s} {n:7d} {yrs:5.1f} {db / 1e6:8.0f}M {ar / 1e6:7.0f}M {(db + ar) / 1e6:7.0f}M  {proj / 1e9:11.2f} GB  {'OK' if proj / 1e9 <= a.budget_gb else 'OVER BUDGET'}")
    if rows: print(f"{len(rows)} cars, {tot / 1e9:.1f} GB in all; the largest is {max(r[1] + r[2] for r in rows) / 1e6:.0f} MB for {a.years:g}-year projection {max(r[6] for r in rows) / 1e9:.2f} GB of a {a.budget_gb:g} GB budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

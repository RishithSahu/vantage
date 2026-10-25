"""Write the daily tier (agg_1d) of one or more vehicle databases.   python tools_rollup.py data/world_aster.db [more.db ...]
For databases that were created before the tier existed (the ingest daemon keeps it up to date from then on, in vgpt.store.compact).  Idempotent and incremental."""
import os, sqlite3, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vgpt import store


def main(paths):
    for p in paths:
        c = store.open_rw(p); t = time.perf_counter()
        store.ensure_daily_schema(c)
        n = store.roll_daily(c); c.close(); print(f"{p}: {n} rows written in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__": main(sys.argv[1:])

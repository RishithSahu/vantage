"""Report which typed gold components are absent from a candidate atom set.

Development-only diagnostic: never accepts an acceptance/final path.
"""
import argparse
import json
import os
import sqlite3

from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt.qc import QueryCompiler
from vgpt.query_ir import Query

from .final_campaign import DEV_PATH, ROOT, WORLD_NAMES
from .full_ir_bottom_up import atom_key, atoms, decompose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    if "final" in os.path.basename(args.pool).lower() or "accept" in os.path.basename(args.pool).lower():
        raise SystemExit("sealed evaluation paths are not allowed")
    rows = load(DEV_PATH)
    records = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    conn = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', WORLD_NAMES[0])}?mode=ro", uri=True)
    profile = QueryCompiler(conn, now=DEFAULT_NOW).profile
    wanted = set(args.ids or [])
    for row, record in zip(rows, records):
        if wanted and row["id"] not in wanted:
            continue
        values, _ = atoms(row, record["candidates"], profile)
        gold = decompose(Query.from_text(row["ir"]))
        present = [any(atom_key(value) == atom_key(gold[axis]) for value in values[axis])
                   for axis in range(len(values))]
        print(json.dumps({"id": row["id"], "present": present,
                          "missing_axes": [i for i, ok in enumerate(present) if not ok],
                          "gold_atoms": [atom_key(value) for value in gold]}))
    conn.close()


if __name__ == "__main__":
    main()

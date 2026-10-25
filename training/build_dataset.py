"""Build the execution-verified query dataset.

    python -m training.build_dataset --n 24000 --seed 0 --db data/gen_courier_ext.db --out training/data

1. generate (question, IR) pairs from formal templates with compositional splits  (training/gen_queries.py)
2. round-trip every IR through text and JSON (the two representations must agree)
3. validate it (structure and vocabulary) and EXECUTE it against a real database through the whole pipeline
   (planner -> executor -> evidence -> composer -> verifier); record the outcome (answered / missing_data / no_data / unsupported) and a compact numeric gold result
4. drop examples that cannot be executed or whose answer fails verification - a label that does not run is a bug in the generator or the engine, never training data - and report how many and why
5. write one JSONL file per split plus report.json (counts per split / operation / outcome, the held-out combinations, the seed)

Execution verification is what makes this dataset trustworthy: every IR the model is trained to write is one the engine really answers (or really refuses, with a reason)."""
import argparse
import collections
import hashlib
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from training import gen_queries as G
from vgpt.qc import QueryCompiler
from vgpt.query_ir import Query
from vgpt.query_validator import validate

DEFAULT_NOW = 1790406000


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def gold_of(r):
    """Compact, comparable summary of a QCResult: what an equivalent query must also produce."""
    g = {"status": r.status}
    res = r.result
    if res is not None:
        if isinstance(res.value, (int, float)) and not isinstance(res.value, bool): g["value"] = round(float(res.value), 4)
        elif isinstance(res.value, str): g["value"] = res.value[:60]
        g["unit"] = res.unit or ""; g["n"] = len(res.items or [])
        if res.items:
            def canon(x):
                if isinstance(x, float): return round(x, 6)
                if isinstance(x, dict): return {k: canon(v) for k, v in sorted(x.items())}
                if isinstance(x, (list, tuple)): return [canon(v) for v in x]
                return x
            complete = canon(res.items)
            g["items_sha256"] = hashlib.sha256(json.dumps(complete, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            g["preview"] = complete[:3]
    return g


def execute_all(examples, db, now=DEFAULT_NOW, progress=True):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True); qc = QueryCompiler(conn, now=now)
    cache = {}; out = []; t0 = time.time()
    for i, ex in enumerate(examples):
        if ex.op == "none": out.append((ex, {"status": "refused"}, True, "")); continue
        if ex.ir in cache: out.append((ex, *cache[ex.ir])); continue
        try:
            q = Query.from_text(ex.ir)
            if Query.from_text(q.to_text()).to_text() != q.to_text() or Query.from_dict(q.to_dict()).to_text() != q.to_text(): cache[ex.ir] = ({}, False, "round_trip"); out.append((ex, *cache[ex.ir])); continue
            if validate(q).codes(): cache[ex.ir] = ({}, False, "invalid"); out.append((ex, *cache[ex.ir])); continue
            r = qc.execute(q); ok = r.status in ("answered", "missing_data", "no_data", "unsupported") and (r.verification is None or r.verification.ok)
            why = "" if ok else (f"unverified: {r.verification}" if r.status == "unverified" else r.status)
            cache[ex.ir] = (gold_of(r), ok, why)
        except Exception as e:                                                       # an engine exception on a valid query is a bug to look at, and the example is not usable
            cache[ex.ir] = ({}, False, f"exception: {type(e).__name__}: {e}")
        out.append((ex, *cache[ex.ir]))
        if progress and (i + 1) % 2000 == 0: print(f"  executed {i + 1}/{len(examples)} ({time.time() - t0:.0f}s)", flush=True)
    conn.close(); return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24000); ap.add_argument("--adv", type=int, default=800); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db"))
    ap.add_argument("--out", default=os.path.join(HERE, "training", "data")); ap.add_argument("--max-test", type=int, default=800, help="cap on the size of each test split (random subset)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print(f"generating {a.n} + {a.adv} examples (seed {a.seed}) ..."); t = time.time(); ex = G.generate(n=a.n, seed=a.seed, adv_n=a.adv); print(f"  {len(ex)} generated in {time.time() - t:.1f}s")
    print(f"executing against {a.db} ..."); res = execute_all(ex, a.db)
    kept = collections.defaultdict(list); dropped = collections.Counter(); outcome = collections.Counter(); per_op = collections.Counter()
    for e, gold, ok, why in res:
        if not ok: dropped[why.split(":")[0]] += 1; continue
        d = e.to_dict(); d["gold"] = gold; d["id"] = hashlib.sha1((e.text + "|" + e.ir).encode()).hexdigest()[:12]; kept[e.split].append(d); outcome[gold.get("status")] += 1; per_op[e.op] += 1
    import random
    rnd = random.Random(a.seed + 99)
    for sp in list(kept):
        if sp.startswith("test") and len(kept[sp]) > a.max_test: rnd.shuffle(kept[sp]); kept[sp] = kept[sp][:a.max_test]
    owners = {}
    for sp, rows in kept.items():
        for row in rows:
            key = (" ".join(row["text"].lower().split()), json.dumps(row.get("context"), sort_keys=True))
            if key in owners and owners[key] != sp: raise ValueError(f"dataset split leakage: {sp} overlaps {owners[key]} for {row['text']!r}")
            owners[key] = sp
    for sp, rows in kept.items():
        with open(os.path.join(a.out, f"{sp}.jsonl"), "w", encoding="utf-8") as f:
            for d in rows: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    rep = {"seed": a.seed, "database": os.path.basename(a.db), "database_sha256": file_sha256(a.db), "builder_sha256": file_sha256(__file__), "generated": len(ex), "kept": {k: len(v) for k, v in kept.items()}, "dropped": dict(dropped), "outcomes": dict(outcome), "ops": dict(per_op),
           "held_out": {"comp_targets_x_predicates": [sorted(G.COMP_TARGETS), sorted(G.COMP_PREDS)], "comp_group_pairs": [sorted(g) for g in G.COMP_GROUPS], "comp_rank": sorted(G.COMP_RANK), "comp_agg_x_time": sorted(G.COMP_AGG_TIME),
                        "template_ids": sorted(G.HELD_TEMPLATES), "unseen_phrase_rule": "training/phrases.py UNSEEN_TARGET_PHRASES are used only in test_phrasing"}}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("kept", "dropped", "outcomes")}, indent=1))


if __name__ == "__main__": main()

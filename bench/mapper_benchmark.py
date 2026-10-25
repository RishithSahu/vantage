"""How good is the automatic mapper?  Signal (name, unit, bits, message, value table) -> quantity, measured against a truth set read from real vehicle databases.

    python bench/mapper_benchmark.py [--dbc-dir DIR] [--engine old|new] [--show wrong|refused|unreviewed|all] [--export]

The databases are the open-source opendbc files in tests/fixtures/opendbc (MIT licence, see LICENSE there).  The truth set, tests/fixtures/mapper_truth.json, holds every signal a person looked at:
  * a quantity  - the mapper must find exactly that one;
  * null        - the signal is a counter, a request, a flag whose meaning the name does not give, an abbreviation nobody can read ... the mapper must REFUSE it.
Besides the labelled signals the benchmark scans EVERY signal of every database and counts the mappings that nobody has reviewed ("unreviewed positives"): a new mapping on an unreviewed signal is a failure until
it is looked at and added to the truth set.  The score that matters is mistakes = wrong + kept + unreviewed = 0; recall is the share of labelled quantities that were found.

`--export` rebuilds the truth set from the curated car models (tools_cars_data*.py, whose rows were read by hand against these databases) plus the mapper's own current answers on all other signals AFTER they were
reviewed.  Run it only after reading the list `--show unreviewed` prints."""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vgpt.protocols import dbc as DBC  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures")
TRUTH_FILE = os.path.join(FIX, "mapper_truth.json")
DBC_DIR = os.path.join(FIX, "opendbc")


def curated(dbc_dir):
    """[(dbc file, message, signal, concept, note)] from the car models' bind rows (the rows a person read against the databases)."""
    from tools_cars_data import CARS
    out = []
    for car in CARS:
        if car["id"] == "proprietary_demo": continue                                      # opaque names by design: nothing to read
        for row in car.get("bind", ()):
            concept, ref = row[0], row[1]
            if concept.startswith("_"): continue
            msg, _, sig = (ref.split(":", 1)[-1]).partition(".")
            out.append((car["can"][0][0], msg, sig, concept, car["id"]))
    return out


def load_truth():
    if not os.path.exists(TRUTH_FILE): return [], {}
    d = json.load(open(TRUTH_FILE, encoding="utf-8"))
    return [(it["dbc"], it["msg"], it["sig"], it.get("concept"), it.get("note", "")) for it in d["items"]], {(it["dbc"], it["msg"], it["sig"]): (it.get("concept"), it.get("note", "")) for it in d.get("overrides", [])}


def evidence(db, m, s, ice=None):
    return dict(name=s.name, unit=s.unit or None, bits=s.length, factor=s.factor, offset=s.offset, signed=s.signed, lo=s.minimum if s.minimum < s.maximum else None, hi=s.maximum if s.minimum < s.maximum else None,
                choices=list(s.choices.values()) if s.choices else [], message=m.name, siblings=[x.name for x in m.signals if x is not s], dlc=m.dlc, ice=ice)


def run(engine, items, dbc_dir):
    dbs = {}; res = []; ices = {}
    from vgpt.ingest import autoprofile as AP
    from vgpt.ingest import mapper as MP
    for path, msg, sig, truth, note in items:
        p = path if os.path.isabs(path) else os.path.join(dbc_dir, path)
        if p not in dbs: dbs[p] = DBC.Database.load(p, lenient=True)
        db = dbs[p]; m = db.by_name(msg)
        s = next((x for x in m.signals if x.name == sig), None) if m else None
        if s is None: res.append((os.path.basename(p), msg, sig, truth, "MISSING", note)); continue
        if p not in ices: ices[p] = MP.powertrain_is_ice(db) if engine == "new" else None
        if engine == "old":
            best, _ = AP._decide(s.name, s.unit or None, None, set()); got = best[0] if best else None
        else:
            d = MP.decide(MP.Evidence(**evidence(db, m, s, ices[p]))); got = d.concept
        res.append((os.path.basename(p), msg, sig, truth, got, note))
    return res


def unreviewed(dbc_dir, known):
    """Mappings the mapper makes on signals nobody has reviewed: [(dbc, message, signal, Decision)]."""
    from vgpt.ingest import mapper as MP
    out = []
    for f in sorted(glob.glob(os.path.join(dbc_dir, "*.dbc"))):
        name = os.path.basename(f); db = DBC.Database.load(f, lenient=True)
        for m, s, d in MP.map_database(db):
            if d.concept and (name, m, s) not in known: out.append((name, m, s, d))
    return out


def report(res, show):
    right = wrong = refused = kept = ok_refuse = missing = 0; lines = []
    for f, msg, sig, truth, got, note in res:
        if got == "MISSING": missing += 1; lines.append(("missing", f, msg, sig, truth, got, note)); continue
        if truth is None:
            if got is None: ok_refuse += 1
            else: kept += 1; lines.append(("KEPT", f, msg, sig, truth, got, note))
        elif got == truth: right += 1; lines.append(("right", f, msg, sig, truth, got, note))
        elif got is None: refused += 1; lines.append(("refused", f, msg, sig, truth, got, note))
        else: wrong += 1; lines.append(("WRONG", f, msg, sig, truth, got, note))
    pos = right + wrong + refused
    print(f"labelled signals {len(res)}: right {right}, WRONG {wrong}, refused {refused} (of {pos} with a quantity) | must-refuse: refused {ok_refuse}, KEPT {kept} | missing {missing}")
    print(f"recall {right / max(pos, 1):.3f}   mistakes (wrong + kept) {wrong + kept}")
    for kind, f, msg, sig, truth, got, note in lines:
        if show == "all" or (show == "wrong" and kind in ("WRONG", "KEPT", "missing")) or (show == "refused" and kind in ("refused", "WRONG", "KEPT", "missing")):
            print(f"  {kind:8s} {f[:26]:26s} {msg}.{sig}: truth {truth} got {got} {note}")
    return wrong + kept, right / max(pos, 1)


def export(dbc_dir):
    """The truth file from the curated models plus the reviewed current answers."""
    old = json.load(open(TRUTH_FILE, encoding="utf-8")) if os.path.exists(TRUTH_FILE) else {"overrides": [], "items": []}
    over = {(it["dbc"], it["msg"], it["sig"]): (it.get("concept"), it.get("note", "")) for it in old.get("overrides", [])}
    for it in old.get("items", []): over.setdefault((it["dbc"], it["msg"], it["sig"]), (it.get("concept"), it.get("note", "")))
    items = []; seen = set()
    for f, m, s, c, n in curated(dbc_dir):
        k = (f, m, s)
        if k in seen: continue
        seen.add(k)
        if k in over and over[k][1] != n: c, n = over[k]
        items.append({"dbc": f, "msg": m, "sig": s, "concept": c, "note": n})
    for k, (c, n) in over.items():
        if k not in seen: seen.add(k); items.append({"dbc": k[0], "msg": k[1], "sig": k[2], "concept": c, "note": n})
    for f, m, s, d in unreviewed(dbc_dir, seen): items.append({"dbc": f, "msg": m, "sig": s, "concept": d.concept, "note": "reviewed"})
    json.dump({"about": old.get("about", ""), "overrides": [], "items": sorted(items, key=lambda i: (i["dbc"], i["msg"], i["sig"]))}, open(TRUTH_FILE, "w", encoding="utf-8"), indent=1)
    print(f"wrote {len(items)} items to {TRUTH_FILE}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dbc-dir", default=DBC_DIR); ap.add_argument("--engine", default="new"); ap.add_argument("--show", default="refused"); ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.export: export(a.dbc_dir); return 0
    items, over = load_truth()
    for k, (c, n) in over.items(): items.append((k[0], k[1], k[2], c, n))
    res = run(a.engine, items, a.dbc_dir)
    mistakes, recall = report(res, a.show)
    if a.engine == "new":
        known = {(f, m, s) for f, m, s, c, n in items}; un = unreviewed(a.dbc_dir, known)
        print(f"unreviewed positives: {len(un)}")
        if a.show in ("unreviewed", "all"):
            for f, m, s, d in un: print(f"  {f[:26]:26s} {m}.{s} -> {d.concept} ({d.confidence}) {d.transform or ''}")
        mistakes += len(un)
    return 0 if mistakes == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit and optionally repair exact question/context leakage across JSONL splits.

Evaluation partitions take precedence over training.  Removed rows are kept in
``split_quarantine.jsonl`` with their source split, so repair is recoverable and
auditable rather than destructive.
"""
import argparse
import hashlib
import json
import os

PRIORITY = ("holdout", "challenge", "test_comp", "test", "test_iid", "test_template", "test_phrasing", "test_adv", "dev", "val", "train")


def key(row):
    text = " ".join(str(row.get("text", "")).lower().split())
    return text, json.dumps(row.get("context"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(root, apply=False):
    paths = {os.path.splitext(f)[0]: os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jsonl") and f != "split_quarantine.jsonl"}
    order = [n for n in PRIORITY if n in paths] + sorted(set(paths) - set(PRIORITY))
    owner = {}; kept = {}; removed = []
    for split in order:
        rows = [json.loads(line) for line in open(paths[split], encoding="utf-8") if line.strip()]
        out = []; local = set()
        for row in rows:
            k = key(row)
            if k in local or k in owner:
                removed.append({"removed_from": split, "kept_in": owner.get(k, split), "row": row}); continue
            local.add(k); owner[k] = split; out.append(row)
        kept[split] = out
    report = {"directory": os.path.abspath(root), "priority": order, "before": {n: sum(1 for l in open(p, encoding="utf-8") if l.strip()) for n, p in paths.items()},
              "after": {n: len(kept[n]) for n in paths}, "removed": len(removed), "removed_by_split": {n: sum(x["removed_from"] == n for x in removed) for n in paths}}
    if apply:
        for split, rows in kept.items():
            with open(paths[split], "w", encoding="utf-8", newline="\n") as fh:
                for row in rows: fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        with open(os.path.join(root, "split_quarantine.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for row in removed: fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        report["sha256"] = {n: sha(p) for n, p in paths.items()}
        json.dump(report, open(os.path.join(root, "split_integrity.json"), "w", encoding="utf-8"), indent=2)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("directory"); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    print(json.dumps(run(a.directory, a.apply), indent=2))

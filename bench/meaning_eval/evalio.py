"""Loading and validating the meaning-frame evaluation sets."""
import collections
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from vgpt.meaning import ontology
from vgpt.meaning.frame import Frame


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def load_tsv(path):
    """-> list of dicts(id, text, gold(Frame), tag).  Lines starting with # are comments."""
    rows = []
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip(): continue
        parts = line.rstrip("\n").split("\t")
        rows.append({"id": parts[0], "text": parts[1], "gold": Frame.from_text(parts[2]), "tag": parts[3] if len(parts) > 3 else ""})
    return rows


def audit(rows):
    O = ontology.load(); bad = []
    for r in rows:
        issues = O.validate(r["gold"])
        if issues: bad.append((r["id"], r["text"], r["gold"].to_text(), issues))
    return bad


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "novel_v1.tsv")
    rows = load_tsv(p); bad = audit(rows)
    print(len(rows), "rows;", len(bad), "gold frames fail ontology typing")
    for b in bad: print(" ", b)
    print("acts:", dict(collections.Counter(r["gold"].act for r in rows)))
    print("tags:", dict(collections.Counter(r["tag"] for r in rows)))
    print("sha256", sha256(p))

"""Reader for the independent evaluation files (indep_dev.tsv / indep_final.tsv)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip(): continue
        p = line.rstrip(chr(10)).split(chr(9))
        if len(p) < 4: continue
        p += ["-"] * (7 - len(p))                                                                  # optional 7th column: reviewer-approved alternative interpretations, separated by " || "
        rows.append({"id": p[0], "family": p[1], "text": p[2], "ir": p[3], "prev_text": None if p[4] == "-" else p[4], "prev_ir": None if p[5] == "-" else p[5], "alts": [] if p[6] == "-" else [x.strip() for x in p[6].split(" || ") if x.strip()]})
    return rows

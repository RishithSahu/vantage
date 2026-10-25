"""Compare two probe runs (bench/probe_ask.py --out) question by question.

    python bench/probe_compare.py results/probe_a.json results/probe_b.json [--only-different]

Prints, for every question asked in both: the frame each run wrote and whether the printed answers are the same.  The summary counts questions whose answer changed; reading the changed ones is the point."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--only-different", action="store_true"); ap.add_argument("--width", type=int, default=170); x = ap.parse_args()
    A = {(r["db"], r["q"]): r for r in json.load(open(x.a, encoding="utf-8"))}; B = {(r["db"], r["q"]): r for r in json.load(open(x.b, encoding="utf-8"))}
    keys = [k for k in A if k in B]; diff = [k for k in keys if A[k]["answer"].strip() != B[k]["answer"].strip()]
    fr = lambda r: " ".join(str((r.get("frame") or {}).get(k, "")) for k in ("act", "concept", "op", "cond", "ent")).strip()
    for k in (diff if x.only_different else keys):
        a, b = A[k], B[k]; same = a["answer"].strip() == b["answer"].strip()
        print(f"[{k[0][6:] or k[0]}] {k[1][:70]}  {'SAME' if same else 'CHANGED'}\n   A [{fr(a)}] {a['answer'][:x.width].replace(chr(10), ' / ')}" + ("" if same else f"\n   B [{fr(b)}] {b['answer'][:x.width].replace(chr(10), ' / ')}"))
    print(f"\n{len(keys)} questions in both, {len(diff)} with a different answer")


if __name__ == "__main__":
    main()

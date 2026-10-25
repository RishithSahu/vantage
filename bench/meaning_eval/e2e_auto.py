"""Automatic end-to-end check: does the pipeline give the SAME answer it gives for the gold frame?

    python bench/meaning_eval/e2e_auto.py results/meaning/pipeline_X.json results/meaning/oracle_all.json [--show]

A question counts as answer-equal when the answer text (whitespace-normalised) equals the oracle's answer for the gold frame, i.e. the model's frame led to exactly the answer a perfectly
understood question gets.  This is STRICTER than a human reading (equivalent frames such as `now range_estimate` / `estimate range_estimate` give different texts), so the mismatch list is
what a person re-reads: it is the input to the judgments in results/meaning/make_e2e_judgments.py.  Correctness of the oracle answer itself is a separate, human-judged number."""
import collections
import json
import sys


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def norm(s):
    """Whitespace, the degree sign and the date style are presentation only (the app's final formatter turns 2026-08-24 15:00 into '24 Aug, 15:00' and writes the degree sign); compare content."""
    import re
    s = str(s).replace("\u00b0", "").replace("\ufffd", "")
    s = re.sub(r"\b(\d{4})-(\d{2})-(\d{2}) (\d{2}:\d{2})(?::\d{2})?", lambda m: f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]}, {m.group(4)}", s)
    s = re.sub(r"\b(\d{1,2} [A-Z][a-z]{2}, \d{2}:\d{2}):\d{2}", r"\1", s)
    s = re.sub(r"(?<=\d) C\b", " C", s).replace("  ", " ")
    return " ".join(s.split()).lower()                                                                          # label capitalisation is presentation too


def main():
    pipe = json.load(open(sys.argv[1], encoding="utf-8")); oracle = {r["id"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))}
    show = "--show" in sys.argv; by = collections.defaultdict(lambda: [0, 0]); bad = []
    for r in pipe:
        o = oracle.get(r["id"])
        if o is None: continue
        same = norm(r["answer"]) == norm(o["answer"]); by[r["set"]][0] += same; by[r["set"]][1] += 1
        if not same: bad.append((r, o))
    tot = [sum(v[0] for v in by.values()), sum(v[1] for v in by.values())]
    print({k: f"{a}/{n} = {a / n:.3f}" for k, (a, n) in by.items()}, "| all", f"{tot[0]}/{tot[1]} = {tot[0] / tot[1]:.3f}")
    if show:
        for r, o in bad: print(f"{r['id']} | {r['text']} | gold {r['gold']} | pred {r['pred']} | {norm(r['answer'])[:110]}  ||  oracle: {norm(o['answer'])[:80]}")


if __name__ == "__main__":
    main()

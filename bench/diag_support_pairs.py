"""Evaluation of the embedding SUPPORT check (vgpt/support.py) on MINIMAL MEANING CHANGES: does it tell apart wordings that are closely related but require different execution, and does it keep equivalent expressions? A general capability test with hand-built pairs of (quoted words, chosen value, expected verdict); nothing here is production logic and no pair is a benchmark question.
    expected True  = the words support the value;  False = the words are real but mean something else;  verdicts: True / False / None (the check abstains)
Categories: temporal anchor (calendar vs relative), different number, equivalent expressions (must stay True), threshold direction (above / below), event window (before / after), inclusion / exclusion, minimum / maximum.
The check sees only (key, words, value); contrasts it has no access to (a comparison operator, an edit mode, a window direction, a statistic) can at best abstain - which is what the measurement shows. Use: decide whether the check may be anything more than a diagnostic.
    python tools_pin.py 4,5 bench/diag_support_pairs.py"""
import collections, os, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import capreg, signal_registry as R
from vgpt.retrieval import Embedder
from vgpt.support import SupportCheck

P = lambda w, v, e, cat: ("per", w, v, e, cat)
C = lambda w, v, e, cat: ("cond", w, v, e, cat)
T = lambda op, val: [{"signal": "speed", "op": op, "value": val}]
PAIRS = [
    # temporal anchors
    P("the last seven days", "last:7:day", True, "temporal anchor"), P("the last seven days", "cal:last_week", False, "temporal anchor"), P("the previous calendar week", "cal:last_week", True, "temporal anchor"), P("the previous calendar week", "last:7:day", False, "temporal anchor"),
    P("yesterday", "cal:yesterday", True, "temporal anchor"), P("yesterday", "cal:day_before_yesterday", False, "temporal anchor"), P("the day before yesterday", "cal:day_before_yesterday", True, "temporal anchor"), P("the day before yesterday", "cal:yesterday", False, "temporal anchor"),
    P("this week", "cal:this_week", True, "temporal anchor"), P("this week", "cal:last_week", False, "temporal anchor"), P("this month", "cal:this_month", True, "temporal anchor"), P("this month", "cal:last_month", False, "temporal anchor"),
    P("last night", "cal:last_night", True, "temporal anchor"), P("this morning", "cal:last_night", False, "temporal anchor"), P("last weekend", "cal:last_weekend", True, "temporal anchor"), P("last weekend", "cal:this_week", False, "temporal anchor"),
    P("right now", "now", True, "temporal anchor"), P("right now", "last:1:day", False, "temporal anchor"), P("since the start", "all", True, "temporal anchor"), P("since the start", "now", False, "temporal anchor"),
    # different numbers and units
    P("the past 3 days", "last:3:day", True, "different number"), P("the past 3 days", "last:5:day", False, "different number"), P("the last 2 hours", "last:2:hour", True, "different number"), P("the last 2 hours", "last:2:day", False, "different number"),
    P("the last 6 months", "last:6:month", True, "different number"), P("the last 6 months", "last:6:week", False, "different number"), P("last 10 minutes", "last:10:minute", True, "different number"), P("last 10 minutes", "last:10:hour", False, "different number"),
    # equivalent expressions (the check must NOT reject these)
    P("a fortnight", "last:2:week", True, "equivalent expression"), P("a fortnight", "last:14:day", True, "equivalent expression"), P("14 days", "last:14:day", True, "equivalent expression"), P("14 days", "last:2:week", True, "equivalent expression"),
    P("two weeks", "last:14:day", True, "equivalent expression"), P("48 hours", "last:2:day", True, "equivalent expression"), P("the past day", "last:1:day", True, "equivalent expression"), P("24 hours", "last:1:day", True, "equivalent expression"),
    P("a week", "last:7:day", True, "equivalent expression"), P("seven days", "last:1:week", True, "equivalent expression"), P("a month", "last:30:day", True, "equivalent expression"), P("30 days", "last:1:month", True, "equivalent expression"),
    # threshold direction (the check sees the value dict, but a comparison operator is not visible to it)
    C("above 60", T("gt", 60), True, "threshold direction"), C("above 60", T("lt", 60), False, "threshold direction"), C("below 60", T("lt", 60), True, "threshold direction"), C("below 60", T("gt", 60), False, "threshold direction"),
    C("over 100 km/h", T("gt", 100), True, "threshold direction"), C("over 100 km/h", T("lt", 100), False, "threshold direction"), C("under 5 degrees", T("lt", 5), True, "threshold direction"), C("under 5 degrees", T("gt", 5), False, "threshold direction"),
    C("faster than 80", T("gt", 80), True, "threshold direction"), C("faster than 80", T("lt", 80), False, "threshold direction"),
    C("above 60", T("gt", 40), False, "threshold number"), C("below 30", T("lt", 20), False, "threshold number"),
    # named conditions: inclusion / exclusion (the call carries the edit mode, which the check does not see)
    C("only highway driving", ["highway"], True, "inclusion / exclusion"), C("excluding highway driving", ["highway"], False, "inclusion / exclusion"), C("without the rain restriction", ["raining"], False, "inclusion / exclusion"),
    C("while it was raining", ["raining"], True, "inclusion / exclusion"), C("while it was raining", ["night"], False, "inclusion / exclusion"), C("in the city", ["city"], True, "inclusion / exclusion"), C("in the city", ["highway"], False, "inclusion / exclusion"),
    C("while charging", ["charging"], True, "inclusion / exclusion"), C("while charging", ["parked"], False, "inclusion / exclusion"),
]


def main():
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True))); sc = SupportCheck(Embedder(os.path.join(HERE, "models", "bge-small-en-v1.5"), int8=True), reg)
    tally = collections.defaultdict(collections.Counter); bad = []
    for key, words, value, expect, cat in PAIRS:
        v = sc(key, words, value); outcome = "abstain" if v is None else ("correct" if v == expect else ("wrong accept" if v else "wrong reject")); tally[cat][outcome] += 1
        if outcome in ("wrong accept", "wrong reject"): bad.append((cat, words, value if not isinstance(value, list) else str(value)[:60], expect, v))
    print("| contrast | pairs | correct | wrong accept (certifies a wrong value) | wrong reject (rejects a right value) | abstain |\n|---|---|---|---|---|---|")
    for cat, c in tally.items():
        n = sum(c.values()); print(f"| {cat} | {n} | {c['correct']} | {c['wrong accept']} | {c['wrong reject']} | {c['abstain']} |")
    tot = collections.Counter(); [tot.update(c) for c in tally.values()]; print(f"\nall: {sum(tot.values())} pairs, correct {tot['correct']}, wrong accept {tot['wrong accept']}, wrong reject {tot['wrong reject']}, abstain {tot['abstain']}")
    for b in bad: print("  ", b)


if __name__ == "__main__": main()

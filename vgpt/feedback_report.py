"""Turns the thumbs-up / thumbs-down feedback people leave in the app (feedback.jsonl, on their own computer) into a triage list for us: which routes disappoint most, the exact questions that got a thumbs-down with
the answer they got and what the person expected, and a ready-to-edit TSV of those questions (route to be filled in) to add to the training wordings. This is how complaints become fixes."""
import json, os
from collections import Counter

from . import paths


def load(path=None):
    path = path or paths.feedback_file(); out = []
    for fn in (path + ".1", path):
        if os.path.exists(fn):
            for line in open(fn, encoding="utf-8"):
                try:
                    record = json.loads(line)
                    if isinstance(record, dict): out.append(record)
                except ValueError: pass
    return out


def summarize(items):
    up = [i for i in items if i.get("rating") == "up"]; down = [i for i in items if i.get("rating") == "down"]
    by_route = Counter(i.get("route") or i.get("path") or "?" for i in down); good_route = Counter(i.get("route") or i.get("path") or "?" for i in up)
    worst = sorted(((r, n, n / max(n + good_route.get(r, 0), 1)) for r, n in by_route.items()), key=lambda x: (-x[1], -x[2]))
    return {"total": len(items), "up": len(up), "down": len(down), "satisfaction": round(len(up) / len(items), 3) if items else None, "worst_routes": worst[:10], "down_items": down}


def format_report(items):
    s = summarize(items)
    if not s["total"]: return "No feedback yet."
    L = [f"{s['total']} answers rated: {s['up']} up, {s['down']} down ({100 * s['satisfaction']:.0f}% positive)", "", "Routes with the most thumbs-down:"]
    L += [f"  {r:<18} {n} down  ({100 * share:.0f}% of that route's ratings)" for r, n, share in s["worst_routes"]]
    L += ["", "Questions that got a thumbs-down (newest last):"]
    for i in s["down_items"][-40:]:
        L.append(f"  - {i.get('q') or '(question not stored)'}   -> [{i.get('route') or i.get('path')}] {str(i.get('brief') or '')[:90]}" + (f"   EXPECTED: {i['note']}" if i.get("note") else ""))
    return "\n".join(L)


def write_triage_tsv(items, out):
    """route<TAB>question lines for every thumbs-down question; the route column starts as the route that answered (usually the wrong one) - edit it, then add the file to the training wordings."""
    seen = set()
    with open(out, "w", encoding="utf-8") as f:
        f.write("# route<TAB>question - from thumbs-down feedback. Replace the route with the one that SHOULD have answered, then append to bench/everyday_paraphrases.tsv and retrain (tools_train_clf.py --ship)\n")
        for i in items:
            # TSV is a line-oriented interchange format.  Collapse controls so
            # one hostile feedback record cannot manufacture extra rows/columns.
            q = " ".join(str(i.get("q") or "").split())
            route = " ".join(str(i.get("route") or "unknown").split())
            if i.get("rating") == "down" and q and q.lower() not in seen:
                seen.add(q.lower()); f.write(f"{route}\t{q}\n")
    return len(seen)

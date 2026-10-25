"""A long answer for a person who is driving: the verdict first, in one short sentence; the working behind it kept, but under "How I got this".

A "why did X change" analysis is correct and long (the observed change, the share each condition explains, what the model did not capture).  What a person who asked "has my range decreased?" needs is yes or no and by how much; the rest is the evidence, so it
moves behind the page's "How I got this" fold (the page folds every line that begins with `Method:`) instead of being deleted.  Nothing is removed and nothing is added: the same sentences, in the order of importance."""
import re

NL = chr(10)
_CHANGE = re.compile(r"^(?P<what>.+?) (?P<dir>rose|fell) from (?:about )?(?P<a>[^.]*?) to (?P<b>[^.]*?)(?: (?P<per>(?:over|in|during|the last|since)[^.]*?))?(?:\.|, compared with the period just before\.|,)")
_SAME = re.compile(r"^(?P<what>.+?) was about the same")


def change_answer(frame, r):
    """-> the reply with a short verdict first for a direction question that ran the change analysis, else the reply unchanged."""
    if not (isinstance(r, dict) and isinstance(r.get("answer"), str)): return r
    route = str((r.get("facts") or {}).get("route") or "")
    if route != "qc:explain_change" or frame.act != "trend": return r
    lines = r["answer"].split(NL); head = lines[0]
    sent = re.split(r"(?<=[a-z0-9%])\. (?=[A-Z])", head, maxsplit=1); first = sent[0].rstrip(".") + "."; rest = sent[1] if len(sent) > 1 else ""
    m = _CHANGE.match(first); asked = frame.op
    if m:
        up = m.group("dir") == "rose"; what = m.group("what"); per = (" " + m.group("per")) if m.group("per") else ""
        yes = (asked == "rise" and up) or (asked == "drop" and not up)
        lead = ("Yes - " if yes else "No - ") if asked in ("rise", "drop") else ""
        verdict = f"{lead}{what[:1].lower() + what[1:] if lead else what} {'went up' if up else 'went down'}, from about {m.group('a')} to {m.group('b')}{per}."
    elif _SAME.match(first): verdict = ("No - " if asked in ("rise", "drop") else "") + first[:1].lower() * bool(asked in ("rise", "drop")) + (first[1:] if asked in ("rise", "drop") else first)
    else: return r
    if re.search(r"effect size [\d.\-]+; not clear", r["answer"]) and "within normal" not in verdict: verdict += " That is within normal day-to-day variation, not a real change."
    body = [("Method: " + x if x.strip() and not x.startswith("Method:") else x) for x in ([rest] if rest else []) + lines[1:]]
    ans = verdict + NL + NL.join(x for x in body if x.strip())
    return dict(r, answer=ans, brief=verdict)

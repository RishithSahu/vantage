"""Context completion: the parts of a follow-up that the previous answer supplies are COPIED from the previous frame after a bare reading, instead of making the model read the previous frame (about 12 extra prompt tokens, 0.4 to 0.7 s on the Raspberry Pi).

The frame has eight slots (act concept op cond vs ent c2 ref).  What a follow-up takes from the previous turn is named by its `ref` slot, and the audit sets show the pattern is regular:
  ref=period   the question gives only a new time ("same again for the week before"): every other slot is the previous frame's
  ref=concept  the question names a new quantity ("and the motor"): the act, op, cond, vs and ent are the previous frame's
  (ref=focus, item and answer need the model to read the previous frame: "why did it happen", "which one", "how did you get that"; measured, rules for them were wrong too often)
The bare reading (no `Previous:` line) already gives what the question itself says; this fills in what it cannot know.  Closed-class words only (prevgate lists); it never decides what a quantity is.
`complete` returns the completed words, or None when no case applies with certainty: the caller then reads the question again with the previous line, as before.  No answer or frame is cached: the previous frame is the one the dialogue holds."""
from . import prevgate as PG

EMPTY_CONCEPTS = ("none", "unlisted")


def pad(text):
    w = str(text or "").split()
    return (w + ["none"] * 8)[:8]


def _starts(low, phrases): return any(low == p or low.startswith(p + " ") for p in phrases)


def complete(bare, prev, text, has_time):
    """-> (eight words, case name) or None."""
    w = pad(bare); p = pad(prev)
    low = " ".join(PG._L.tokens(text))
    empty = w[0] == "ambiguous" or (w[1] in EMPTY_CONCEPTS and w[0] not in PG.STANDS_ALONE)
    opener = _starts(low, list(PG.OPEN) + list(PG.EXTRA_OPEN))
    if empty and has_time and p[0] != "none": return p[:7] + ["period"], "period"
    deixis = any(" " + q + " " in " " + low + " " for q in PG.INSIDE) or bool(PG.PRONOUNS & set(low.split()))
    if opener and not deixis and not has_time and not empty and w[1] != p[1] and w[1] not in EMPTY_CONCEPTS and p[0] not in ("none", "ambiguous"): return [p[0], w[1], p[2], p[3], p[4], p[5], w[6], "concept"], "concept"
    return None


def eligible(text):
    """An ellipsis ("and the motor", "same again for last week"): an opener and no pointer word.  Decided from the wording alone, before the model is used."""
    low = " ".join(PG._L.tokens(text))
    deixis = any(" " + q + " " in " " + low + " " for q in PG.INSIDE) or bool(PG.PRONOUNS & set(low.split()))
    return _starts(low, [p for p in list(PG.OPEN) + list(PG.EXTRA_OPEN) if not p.startswith("which")]) and not deixis


def has_time(text):
    """Does the question name a period?  (the literal reader needs a clock only to place it; whether one is named does not depend on it)"""
    from . import literals as L
    try: return bool(L.extract(text, 1792360800, 0).times)
    except Exception: return False


def attempt(model, text, hint, prev):
    """The ellipsis case without the previous line in the prompt.  -> (Frame, info) when the previous frame completes the bare reading, else (None, ms spent): the caller reads the question with the previous line, starting from this reading's work."""
    from .frame import Frame
    f0, i0 = model._parse(text, hint, None, keep=True)
    if f0 is None: return None, i0.get("ms", 0.0)
    r = complete(i0["raw"], prev, text, has_time(text))
    if r is None: return None, i0.get("ms", 0.0)
    words, why = r; info = dict(i0); info.update(raw=" ".join(words), context=why, prev_used=False, context_completed=True)
    try: return Frame.from_text(info["raw"]), info
    except ValueError: return None, i0.get("ms", 0.0)

"""Glossary arbitration: where the learned frame and the project's own word list disagree, the word list is consulted - under three structural rules and nothing else.

The learned frame sometimes picks the concept of a SHORT phrase inside a longer, more specific term ("brake pad" inside "brake pad temperature"), misreads a rare word the model never saw in training
("ECU reset" -> diagnostic code), or says `unlisted` for a signal that THIS vehicle's profile declares ("engine oil temperature").  The glossary is the sanctioned place where a human says what a word
means; this module compares word spans of that data with the question and the model's concept Y.  A glossary concept X replaces Y when

  A  contained    every word Y has in the question lies INSIDE a longer glossary phrase of X ("power" in "power derate", "brake pad" in "brake pad temperature");
  B  unsupported  the glossary entry is a hint entry (a word the 0.5B model is known to misread), the question contains its phrase, and none of Y's own words occurs in the question;
  C  unlisted     the model said `unlisted` and the question contains a glossary phrase of a concept this vehicle has.
If Y has a word of its own outside the glossary phrase the model's choice stands.  The act and operator are kept when the new frame type-checks with them; otherwise the nearest valid ones are chosen
(an events question keeps "how many"; "did it happen" becomes `exists`).  No meaning is read here: only word spans of data the project already maintains.  Every case is listed in
docs/TRAINING_QUEUE.md as behaviour a retrained model should make unnecessary."""
import difflib
import functools

from . import literals as L
from .frame import Frame

_toks = L.tokens
OPS_PREF = ("exists", "count", "list", "last", "first", "mean", "max")


@functools.lru_cache(maxsize=1 << 17)
def _same_cached(a, b):
    """A pure function of two words (so memoising it is exact): the cheap upper bounds of SequenceMatcher first (length, then letter counts), the full ratio only when they cannot rule the pair out."""
    sm = difflib.SequenceMatcher(None, a, b)
    return sm.real_quick_ratio() >= 0.8 and sm.quick_ratio() >= 0.8 and sm.ratio() >= 0.8


def _same(a, b):
    if a == b: return True
    return len(a) >= 4 and len(b) >= 4 and _same_cached(a, b)


def _find(hay, needle, first=None):
    """-> list of (start, end) where the token sequence `needle` occurs in `hay` (spelling variants allowed).  `first` (optional dict, one per `hay`): the positions where each first word matches, shared by every
    needle that starts with the same word (the glossary has hundreds of phrases and a few dozen first words)."""
    n = len(needle); out = []
    if n == 0: return out
    last = len(hay) - n
    if last < 0: return out
    if first is None: starts = range(last + 1)
    else:
        w = needle[0]; st = first.get(w)
        if st is None: st = first[w] = [i for i, h in enumerate(hay) if _same(h, w)]
        starts = [i for i in st if i <= last]
    for i in starts:
        if n == 1 or all(_same(hay[i + j], needle[j]) for j in range(1, n)):
            if first is not None or _same(hay[i], needle[0]): out.append((i, i + n))
    return out


def own_forms(onto, glossary, concept):
    forms = set()
    c = onto.concepts.get(concept)
    if c is not None:
        for a in tuple(c.aliases) + (c.label,): forms.add(tuple(_toks(a)))
    for a in glossary.aliases(concept): forms.add(tuple(_toks(a)))
    return {f for f in forms if f}


def own_word_present(forms, phrase, toks):
    """Does the question carry a distinctive word of the model's own concept (a word of its forms that is neither in the competing phrase nor a common word)?  'cleared' in 'when was a fault code last cleared'."""
    mine = {w for f in forms for w in f if len(w) >= 4 and w not in L.LOG_STOP} - set(phrase)
    return any(w in toks or w.rstrip("s") in toks for w in mine)


def own_vocabulary(text, frame, onto, glossary, available):
    """The model said `unlisted`, but the question carries the whole name or a synonym of a signal THIS vehicle reports (its catalogue, adopted from its own database): that signal is meant.
    Unique longest phrase wins; a tie or no match leaves the model's answer alone.  `available`: the concept ids the vehicle has."""
    if frame.act in ("define", "howto", "why", "social", "offtopic", "unsupported", "control", "unsafe", "ambiguous", "events", "context"): return frame, None
    toks = _toks(text); best, tie = None, False; unl = frame.concept == "unlisted"
    mine = own_forms(onto, glossary, frame.concept) if not unl and frame.concept in onto.concepts else set()
    for cid in available:
        c = onto.concepts.get(cid)
        if c is None or c.kind not in ("signal", "derived", "declared") or cid == frame.concept: continue
        for f in own_forms(onto, glossary, cid):
            if not _find(toks, list(f)): continue
            if not unl and own_word_present(mine, set(f), toks): continue                       # the question also carries a word of the model's own concept: its choice stands
            score = (len(f), sum(len(w) for w in f))
            if best is None or score > best[0]: best, tie = (score, cid, f), False
            elif score == best[0] and cid != best[1]: tie = True
    if best is None or tie: return frame, None
    cand = fit(Frame("now" if frame.act == "now" else frame.act, best[1], frame.op, frame.cond, frame.vs, frame.ent), best[1], onto)
    return (cand, f"the vehicle's own signal '{best[1]}' ({' '.join(best[2])})") if cand is not None else (frame, None)


def fit(frame, concept, onto):
    """The nearest well-typed frame about `concept`: keep act / operator / condition when valid, otherwise the first valid combination (act of the question, then the concept's default act)."""
    c = onto.concepts[concept]; acts = [frame.act] + ([c.default_act] if c.default_act and c.default_act != frame.act else [])
    for act in acts:
        for op in dict.fromkeys(((frame.op,) + OPS_PREF + ("none",)) if act == "events" else ((frame.op, "none") + OPS_PREF)):        # an events question needs a real operator ("did it happen" -> exists), never "none"
            for tail in ((frame.cond, frame.vs, frame.ent), ("none", "none", "none")):
                cand = Frame(act, concept, op, *tail)
                if not onto.validate(cand): return cand
    return None


def arbitrate(text, frame, onto, glossary, rules="ABC"):
    """-> (frame, note | None).  `rules`: which of the three rules may act (the default product runs only "B": a word the model is known to misread, flagged in the glossary)."""
    chosen = frame.concept
    if chosen == "none" or (chosen not in onto.concepts and chosen != "unlisted"): return frame, None
    toks = _toks(text); first = {}; mine = own_forms(onto, glossary, chosen) if chosen in onto.concepts else set(); mine_hits = [(a, b) for f in mine for (a, b) in _find(toks, list(f), first)]
    best = None                                                                                    # (priority, words in the phrase, concept, form)
    for form, entry in glossary.index.items():
        x = entry.get("concept")
        if x is None or x == chosen or x not in onto.concepts: continue
        ft = form.split()
        if len(ft) < 1: continue
        for (s, e) in _find(toks, ft, first):
            inside = [(a, b) for f in mine for (a, b) in _find(toks, list(f), first) if a >= s and b <= e]; outside = [(a, b) for (a, b) in mine_hits if b <= s or a >= e]
            rule = None
            if len(ft) >= 2 and inside and not outside: rule = "A"
            elif chosen == "unlisted": rule = "C"
            elif entry.get("hint") and not mine_hits and not own_word_present(mine, ft, toks): rule = "B"
            if rule is None or rule not in rules: continue
            key = ({"A": 3, "C": 2, "B": 1}[rule], len(ft))
            if best is None or key > best[0]: best = (key, x, form, rule)
    if best is None: return frame, None
    cand = fit(frame, best[1], onto)
    if cand is None: return frame, None
    return cand, f"'{best[2]}' names {best[1]} (rule {best[3]}), not {chosen}"

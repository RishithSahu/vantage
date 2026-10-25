"""When does the frame model need to be shown the previous turn?

Frame version 2 reads `Previous: <frame of the last answered question>` after the question.  On the Raspberry Pi that line is about 11 of the ~25 tokens read after the cached prefix (32 ms each), and an unrelated
previous frame also pulls the model's reading towards it (measured: without the line the frame is right for 90.8 % of the frozen-set questions, with an unrelated one 88.2 %; bench/prev_cascade_eval.py).
So the question is read WITHOUT the line first, and the line is added only when

  * the question points back (an opener such as "and" / "what about", a pointer such as "which one", a pronoun such as "it" / "that", "too", or it is a single word or two), or
  * the model's own reading of the bare question is not a complete one (ambiguous, or it names no quantity, or it is unsure: p < 0.5).

Closed-class words from discourse.json only; nothing here decides what a question means - it decides whether reading the previous turn is worth its tokens.  VGPT_PREV=always restores the line for every question."""
import ast
import json
import os

from . import literals as _L

_D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "discourse.json"), encoding="utf-8"))


def _list(k):
    v = _D[k]; v = ast.literal_eval(v) if isinstance(v, str) else v        # a few keys are stored as the text of a list
    return list(v.keys()) if isinstance(v, dict) else list(v)


OPEN = _list("ellipsis_openers")
INSIDE = [p for p in _list("deixis_time") + _list("pointer") + _list("provenance") + _list("referent_why") if p != "why"]      # a bare "why" opens most causal questions
EXTRA_OPEN = ("now the", "now my", "same", "which came", "which is", "which one")
PRONOUNS = frozenset("it that those them they again instead previous too also".split())
STANDS_ALONE = frozenset(("summary", "anomaly", "social", "offtopic", "unsupported", "control", "unsafe", "explain", "define", "health"))
MIN_P = 0.5


def wants_prev(text, short=2):
    """True when the wording itself points back to the previous turn."""
    w = _L.tokens(text); low = " ".join(w)
    if len(w) <= short: return True
    if any(low == p or low.startswith(p + " ") for p in OPEN + list(EXTRA_OPEN)): return True
    if w[-1] in ("one", "ones"): return True                                             # "the latest one"
    if any(" " + p + " " in " " + low + " " for p in INSIDE): return True
    return bool(PRONOUNS & set(w))


def frame_needs_prev(frame, pmin):
    """True when the model's reading of the bare question is not a complete one."""
    if frame.act == "ambiguous": return True
    if frame.concept in ("none", "unlisted") and frame.act not in STANDS_ALONE: return True
    return (pmin or 0.0) < MIN_P

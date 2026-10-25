"""The confidence the assistant reports, and the line below which it withholds an answer.

The decoder's lowest token probability (`pmin`) is a score, not a probability.  `calibration.json` (made by `bench/calibrate_confidence.py` from the frozen DEVELOPMENT sets - never the sealed, hold-out or acceptance sets - with the
fit and the check on different halves of the questions) maps it to the share of questions that were really read right at that score.  That share is what is reported as the confidence of an answer, and an answer whose
calibrated confidence is below `refuse_below()` is not given: the assistant says what it understood and asks for another wording.  The same map serves both backends because both read `pmin` over the tokens the decoder may allow.

Default line 0.50: on the check half it withholds 2.7 % of the questions, which are 17 % of the wrong readings, and loses 1.2 % of the right ones (the table is in calibration.json).  VGPT_MEANING_REFUSE_BELOW=0.6 withholds 8 % and
catches 43 % of the wrong readings; 0 switches the refusal off.  The owner turned the "did you mean 1 or 2" questions off on 2026-10-24 (`vgpt/product.py CLARIFY_BELOW`); this is a different thing - a refusal to guess at a
reading the model itself was very unsure of - and it asks nothing of the person but a rephrasing."""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
DEFAULT_REFUSE_BELOW = 0.50
_MAP = {}


def _load():
    if not _MAP:
        try:
            j = json.load(open(PATH, encoding="utf-8")); _MAP.update(x=[float(v) for v in j["x"]], y=[float(v) for v in j["y"]], made=j.get("made"), adapter=j.get("adapter"), questions=(j.get("report") or {}).get("questions"))
        except (OSError, ValueError, KeyError): _MAP.update(x=[], y=[], made=None, adapter=None, questions=0)
    return _MAP


def calibrated(pmin):
    """-> the share of development questions read right at this score (0..1); the score itself when there is no calibration."""
    if pmin is None: return 0.0
    m = _load(); xs, ys = m["x"], m["y"]; p = float(pmin)
    if not xs: return max(0.0, min(1.0, p))
    if p <= xs[0]: return ys[0]
    if p >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if p <= xs[i]: return ys[i - 1] + (ys[i] - ys[i - 1]) * (p - xs[i - 1]) / ((xs[i] - xs[i - 1]) or 1.0)
    return ys[-1]


def refuse_below():
    """The calibrated confidence under which an answer is withheld (VGPT_MEANING_REFUSE_BELOW; 0 = never)."""
    try: v = float(os.environ.get("VGPT_MEANING_REFUSE_BELOW", DEFAULT_REFUSE_BELOW))
    except ValueError: v = DEFAULT_REFUSE_BELOW
    return max(0.0, min(1.0, v))


def info():
    m = _load(); return {"calibrated": bool(m["x"]), "made": m["made"], "adapter": m["adapter"], "questions": m["questions"], "refuse_below": refuse_below()}

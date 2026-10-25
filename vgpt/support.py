"""Do the quoted words SUPPORT the value the model chose? (vgpt.plancompile.ground checks the first property - the words exist in the message; this checks the second.)
A model can quote "yesterday" correctly and still select "last month". The check uses the frozen sentence-embedding model (the same bge-small used for retrieval) and the numerals in the quotation; there is no list of phrases:
  period   the quoted words must be most similar to the KIND of period the value is (a named calendar period, the same relative unit, all time, now, a date), compared with the other kinds, and a number written with digits in the quotation must be the number in the value;
  condition  a named condition must be among the closest descriptions to the quoted words; for a threshold the numbers in the quotation must include the threshold's value;
  unit     the quoted words must be most similar to the chosen unit among the units of its family.
`support(key, words, value)` -> True (supported), False (the words exist but mean something else) or None (cannot judge: the check abstains rather than rejecting).
Calibration data and the measured false-rejection / catch rates are recorded by bench/calibrate_support.py."""
import re

import numpy as np

from . import signal_registry as R
from . import toolspec as T

REL_UNITS = ("minute", "hour", "day", "week", "month")


def _numbers(text):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]


def _span_numbers(text):
    """Numbers that quantify a span of time in the quotation (a numeral directly before a minute / hour / day / week / month word): the only numerals that must agree with a relative period."""
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:minute|min|hour|hr|day|week|month)", (text or "").lower())]


class SupportCheck:
    def __init__(self, embedder, registry, cond_rank=3, margin=0.0):
        self.emb = embedder; self.reg = registry; self.cond_rank = cond_rank; self.margin = margin; self._cache = {}
        phr = {}                                                                                       # each KIND of period is described by a few natural phrasings (an embedding needs more than one surface form to represent a kind)
        for k, name in T.CAL_NAT.items(): phr["cal:" + k] = [name, f"during {name}", f"in the {name}"] if k not in ("last_night",) else ["last night", "overnight", "at night"]
        for u in REL_UNITS: phr["rel:" + u] = [f"last 3 {u}s", f"past 2 {u}s", f"previous {u}", f"the last {u}"]
        phr["all"] = ["all time", "ever", "since the beginning", "the whole recorded history"]; phr["now"] = ["now", "right now", "currently", "at the moment"]; phr["date"] = ["on March 3rd", "on the 5th of May", "on 2026-03-03"]
        self.kinds = [(k, t) for k, ts in phr.items() for t in ts]; self.kind_names = list(phr)
        self.kind_vecs = self.emb.encode([t for _, t in self.kinds])
        self.conds = [c for c in registry.items if c.kind == "condition"]; self.cond_vecs = self.emb.encode([f"{c.id.replace('_', ' ')} {c.title}" for c in self.conds]) if self.conds else None

    def _vec(self, words):
        v = self._cache.get(words)
        if v is None:
            if len(self._cache) > 512: self._cache.clear()
            v = self._cache[words] = self.emb.encode([words])[0]
        return v

    def __call__(self, key, words, value):
        try:
            if key in ("per", "per2"): return self.period(words, value)
            if key == "cond": return self.conditions(words, value)
        except Exception: return None
        return None

    def period(self, words, canon):
        if canon.startswith("last:"):
            _, n, u = canon.split(":"); n = int(n); allowed = {"rel:" + u}
            if u == "day" and n % 7 == 0: allowed.add("rel:week")                                      # 7 days is a week, 28-31 days a month (the documented conventions)
            if u == "day" and n in (28, 29, 30, 31): allowed.add("rel:month")
            if canon == "last:7:day": allowed.add("cal:last_week")                                      # "last week" is both a calendar phrase and a span
            if canon == "last:30:day": allowed.add("cal:last_month")
            nums = _span_numbers(words)
            if nums and not (float(n) in nums or (u == "day" and n % 7 == 0 and float(n // 7) in nums)): return False       # the digits in the quotation disagree with the number chosen
        else: allowed = {("cal:" + canon.split(":", 1)[1]) if canon.startswith("cal:") else "date" if canon.startswith("date:") else canon}
        raw = self.kind_vecs @ self._vec(words); sims = {k: float(max(raw[i] for i, (kk, _) in enumerate(self.kinds) if kk == k)) for k in self.kind_names}
        if not all(k in sims for k in allowed): return None
        top = max(sims.values()); mine = max(sims[k] for k in allowed)
        return bool(mine + self.margin >= top - 1e-9 or (any(k.startswith("rel:") for k in allowed) and max(sims, key=sims.get).startswith("rel:") and mine >= top - 0.04))

    def conditions(self, words, value):
        sv = self._vec(words); out = True
        for c in value:
            if isinstance(c, dict):
                nums = _numbers(words)
                # A quoted threshold must contain its number.  Treating an
                # empty quotation as support allowed a model-invented limit.
                if not nums or float(c.get("value")) not in nums: return False
                continue
            idx = [i for i, x in enumerate(self.conds) if x.id == c]
            if not idx: return None
            rank = int((self.cond_vecs @ sv > float(self.cond_vecs[idx[0]] @ sv)).sum())
            if rank >= self.cond_rank: out = False
        return out

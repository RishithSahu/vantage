"""Reading a long request: what analysis, about which anchor, over which signals, in which windows.

No regular expression lives here (the project's rule: text patterns belong to the literal parser, vgpt/meaning/literals.py).  Values - sentences, durations, counts, a stated clock time, a period -
come from that parser; WORDS are looked up in dictionaries: the registry's concept and event phrases, the glossary, and the planner's own lists (lexicon.json), longest phrase first."""
import json
import os
from dataclasses import dataclass, field

from .. import signal_registry as R
from ..meaning import glossary as GL, literals as L

HERE = os.path.dirname(os.path.abspath(__file__))
LEX = json.load(open(os.path.join(HERE, "lexicon.json"), encoding="utf-8"))
SKIP_SINGLE = {"power", "mode", "range", "light", "current", "state", "level", "energy", "distance", "flow", "air"}
_DICT = None


@dataclass
class Request:
    text: str
    sentences: list
    n_words: int
    concepts: list = field(default_factory=list)          # canonical concepts named, in order
    events: list = field(default_factory=list)            # event types named, in order ("ANY_WARNING" = the group)
    windows: dict = field(default_factory=dict)           # before / after / within, seconds
    counts: list = field(default_factory=list)            # [(n, noun)]
    anchor_kind: str = None                               # latest | first | critical | rare | focus
    clock_ts: int = None                                  # a clock time the request states ("at 14:32 yesterday")
    times: list = field(default_factory=list)             # TimeSpec values
    scores: dict = field(default_factory=dict)
    template: str = None

    def has(self, concept): return concept in self.concepts


def dictionary():
    """phrase tuple -> [("concept"|"event", name)]; built once from the registry, the glossary and the planner's phrase list."""
    global _DICT
    if _DICT is not None: return _DICT
    d = {}
    def add(phrase, kind, name):
        key = tuple(L.tokens(phrase))
        if not key or (len(key) == 1 and key[0] in SKIP_SINGLE): return
        lst = d.setdefault(key, [])
        if (kind, name) not in lst: lst.append((kind, name))
    for phrase, names in R.LEXICON.items():
        for n in names:
            if n in R.CONCEPTS: add(phrase, "concept", n)
    for key, (label, syns) in R.EVENTS.items():
        for p in tuple(syns) + (label,): add(p, "event", key)
    for p, v in LEX["phrases"].items():
        key = tuple(L.tokens(p)); d[key] = [("concept", c) for c in v["concepts"]] + [("event", e) for e in v["events"]]          # the planner's own phrases override
    _DICT = d; return d


def invalidate():
    global _DICT
    _DICT = None


def mentions(text):
    """Longest-phrase-first scan: [(kind, name)] in order of appearance, without repeats."""
    toks = L.tokens(text.replace("-", " ")); d = dictionary(); out = []; i = 0; mx = max(len(k) for k in d)                                  # "cooling-system" is the same phrase as "cooling system"
    while i < len(toks):
        for n in range(min(mx, len(toks) - i), 0, -1):
            hit = d.get(tuple(toks[i:i + n]))
            if hit:
                for h in hit:
                    if h not in out: out.append(h)
                i += n; break
        else: i += 1
    return out


def anchor_kind(norm):
    for phrase in sorted(LEX["anchors"], key=len, reverse=True):
        if (" " + phrase + " ") in norm: return LEX["anchors"][phrase]
    return None


def template_scores(norm, req):
    s = {}
    for name, cues in LEX["templates"].items():
        s[name] = sum(1 for c in cues if (" " + c + " ") in norm)
    ev_specific = [e for e in req.events if e != "ANY_WARNING"]
    if len(ev_specific) >= 2 and s["cooccur"]: s["cooccur"] += 1
    if ("adas_state" in req.concepts or any(e.startswith("ADAS") for e in req.events) or " adas " in norm) and (s["adas"] or s["cooccur"] == 0 and s["leadtime"] == 0 and s["incident"] <= 1 and s["conditions"] == 0 and s["health"] == 0): s["adas"] += 1
    if any(c in req.concepts for c in ("range_estimate", "reported_range")) and s["range_drop"]: s["range_drop"] += 1
    if s["range_drop"] and not any(c in req.concepts for c in ("range_estimate", "reported_range")): s["range_drop"] = 0
    if s["thermal_stop"] and "battery_temperature" not in req.concepts and "battery temperature" not in norm: s["thermal_stop"] = 0
    if s["occurrences"] and not any(w for n, w in req.counts if w in ("occurrence", "instance", "incident")): s["occurrences"] = 0
    if s["spike_or_trend"] and "temperature" not in norm: s["spike_or_trend"] = 0
    return s


def parse(text, now, tz):
    clean = " ".join(str(text).split()); norm = " " + L.norm_words(clean).replace("-", " ") + " "
    req = Request(clean, L.sentences(clean), len(clean.split()))
    for kind, name in mentions(clean):
        if kind == "concept" and name not in req.concepts: req.concepts.append(name)
        elif kind == "event" and name not in req.events: req.events.append(name)
    req.windows = {k: v for k, v in L.window_phrases(clean).items() if v <= 86400};                          # an event window is minutes or hours; 'the last two weeks' is a period, not a window
    req.counts = L.count_phrases(clean); req.anchor_kind = anchor_kind(norm)
    lit = L.extract(clean, now, tz); req.times = list(lit.times)
    try: req.clock_ts = L.instant(clean, now, tz) if ":" in clean else None
    except Exception: req.clock_ts = None
    req.scores = template_scores(norm, req); order = LEX["priority"]
    best = max(order, key=lambda t: (req.scores.get(t, 0), -order.index(t)))
    req.template = best if req.scores.get(best, 0) >= 1 else None
    return req


def is_investigation(text, req=None):
    """A request for an analysis, not a question: several sentences or a long single one, and a recognised analysis cue."""
    words = len(str(text).split()); sents = len(L.sentences(text))
    return (sents >= 2 or words >= 24) and words >= 14

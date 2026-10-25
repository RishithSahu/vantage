"""Draft cues for lossless speculative decoding of the frame.

The frame model writes eight words, one token at a time; every pass over the weights costs 0.2 s on the Raspberry Pi 4 (the 640 MB of int8 weights stream through a memory bus), so a frame written one
token per pass takes 3 to 5 passes.  Speculative decoding verifies SEVERAL proposed tokens in one pass.  This module only proposes: a concept word when the question contains one of that concept's names, an
operator word when the question contains a cue for it ("average", "highest", "how many" ...).  Every proposed token is accepted only if it IS the model's own masked argmax at that position
(vgpt/meaning/ort_backend.py), so the frame that comes out is exactly the frame the model writes without any proposal; a wrong proposal costs nothing but the (cheap) extra rows in the pass.
Nothing here decides what a question means: it supplies guesses that the learned model confirms or overrules token by token (tests/test_draft_lossless.py compares the frames with and without it)."""
from . import literals as L

_tok = L.tokens
OP_CUES = (            # (phrase, operator word of the ontology); the first phrase found in the question wins, longer phrases listed first
    ("how many", "count"), ("number of", "count"), ("how often", "count"), ("how long", "duration"), ("for how", "duration"), ("most recent", "last"), ("latest", "last"), ("the last", "last"),
    ("percentage", "share"), ("percent", "share"), ("fraction", "share"), ("share", "share"), ("median", "median"), ("average", "mean"), ("mean", "mean"), ("typical", "mean"), ("usual", "mean"),
    ("maximum", "max"), ("highest", "max"), ("peak", "max"), ("max", "max"), ("minimum", "min"), ("lowest", "min"), ("min", "min"), ("total", "sum"), ("sum", "sum"), ("spread", "spread"),
    ("first", "first"), ("list", "list"), ("any", "exists"), ("was there", "exists"), ("did", "exists"),
)


WINDOW_WORDS = frozenset("a an one two three four five six seven eight nine ten twelve few couple several week weeks month months day days year years hour hours minute minutes fortnight night nights weekend season quarter".split())


class Cues:
    def __init__(self, onto, glossary):
        self.words = onto.words(); self.concept_words = set(self.words["concept"]); self.op_words = set(self.words["op"]); self.index = {}
        forms = {}
        for c in onto.concepts.values():
            if c.id not in self.concept_words: continue
            for a in tuple(c.aliases) + (c.label,): forms.setdefault(tuple(_tok(a)), set()).add(c.id)
        for entry in glossary.index.values():
            cid = entry.get("concept")
            if cid in self.concept_words:
                for f in tuple(entry.get("forms") or ()) + (entry.get("term") or "",): forms.setdefault(tuple(_tok(f)), set()).add(cid)
        for f, ids in forms.items():
            if f and len(ids) == 1: self.index.setdefault(f[0], []).append((f, next(iter(ids))))
        for k in self.index: self.index[k].sort(key=lambda x: (-len(x[0]), -sum(len(w) for w in x[0])))
        self.ops = tuple((tuple(_tok(p)), o) for p, o in OP_CUES if o in self.op_words)

    def propose(self, text):
        """-> {slot index: word}: slot 1 = concept, slot 2 = operator (only the slots that have a cue)."""
        toks = _tok(text); out = {}; best = None; tie = False; n = len(toks)
        for i, t in enumerate(toks):
            for f, cid in self.index.get(t, ()):
                if toks[i:i + len(f)] == list(f):
                    score = (len(f), sum(len(w) for w in f))
                    if best is None or score > best[0]: best, tie = (score, cid), False
                    elif score == best[0] and cid != best[1]: tie = True
                    break
        if best is not None and not tie: out[1] = best[1]
        for phrase, op in self.ops:
            if any(toks[i:i + len(phrase)] == list(phrase) and not self._window(op, toks, i + len(phrase)) for i in range(n - len(phrase) + 1)): out[2] = op; break
        return out

    @staticmethod
    def _window(op, toks, after):
        """\"the last 14 days\" / \"the last week\" is a period, not the operation `last`."""
        return op == "last" and after < len(toks) and (toks[after].isdigit() or toks[after] in WINDOW_WORDS)


GROUND_STOP = frozenset("""the and for are was were what when why how does did have has had can you your this that with from into over than then there their about which while where whose would could should will
    been being being much many more most some any all not out off per via car vehicle show tell give list make made get got going goes gone last first next ever just also very really like need want""".split())


VEHICLE_WORDS = frozenset("car vehicle drive driving drives drove driven trip trips journey journeys ride road highway city park parked parking charge charging charger it its".split())


class Support:
    """Does a question mention ANYTHING of the vehicle's vocabulary?  (A floor against a low-confidence reading of a question that has nothing to do with a vehicle: it looks for a shared word stem, nothing more.)"""

    def __init__(self, onto, glossary):
        self.stems = set()
        for c in onto.concepts.values():
            for a in tuple(c.aliases) + (c.label, c.id.replace("_", " ")): self.stems |= {w[:5] for w in _tok(a) if len(w) >= 3 and w not in GROUND_STOP}
        for entry in glossary.index.values():
            for f in tuple(entry.get("forms") or ()) + (entry.get("term") or "",): self.stems |= {w[:5] for w in _tok(f) if len(w) >= 3 and w not in GROUND_STOP}
        self.stems |= {w[:5] for w in VEHICLE_WORDS}

    def supported(self, text, concept=None):
        return any(w[:5] in self.stems for w in _tok(text) if len(w) >= 2)

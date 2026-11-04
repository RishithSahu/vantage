"""Word meanings as data.  The ONLY sanctioned place for a human to help the model: "this word means that".

A glossary entry says what a word, abbreviation, unit or piece of slang MEANS.  It never says which operation or answer a question gets.  Two uses:
  * training: entries supply the aliases from which diverse sentences are generated, and the hint line the model learns to read;
  * inference: the question's words are looked up in a hash index (exact tokens / short phrases, no pattern matching) and, only when an entry is flagged `hint` (the word is
    one a 0.5B model is likely to misread), its one-line meaning is appended to the prompt:   Terms: ppl = people (passengers or occupants); mAh = ...
A missing entry costs one JSON line to fix; no code and no retraining is needed for the hint to take effect."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_TOKEN = re.compile(r"[a-z0-9]+(?:[/'.-][a-z0-9]+)*")


def _norm(s): return " ".join(_TOKEN.findall(str(s).lower().replace("’", "'")))


COMBUSTION = ("petrol", "diesel", "ice", "hybrid")


def ice_only(): return os.environ.get("VGPT_ICE_ONLY") == "1"                                                    # no hint line for electric / unknown cars


def ice_line_only(): return os.environ.get("VGPT_ICE_ONLY") in ("1", "line")                                        # the car-type line only for combustion cars ("line": the hints stay for every car)


def first_clause(meaning, words=6):
    """The first clause of a meaning (up to the first ; , or " - "), at most `words` words: the short form of a hint (VGPT_HINT_CLAUSE=1)."""
    return " ".join(meaning.split(";")[0].split(",")[0].split(" - ")[0].split()[:words])


class Glossary:
    def __init__(self, path=None):
        path = path or os.path.join(HERE, "glossary.jsonl")
        self.entries = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.index = {}                                   # normalised surface form -> entry
        for e in self.entries:
            for form in [e["term"]] + list(e.get("forms", ())):
                self.index.setdefault(_norm(form), e)
        self.max_words = max((len(k.split()) for k in self.index), default=1)

    def extend(self, entries):
        """Add entries at run time (a vehicle profile's own signal names); an entry already present for a form is kept."""
        for e in entries:
            if e.get("term") and e not in self.entries:
                self.entries.append(e)
                for form in [e["term"]] + list(e.get("forms", ())): self.index.setdefault(_norm(form), e)
        self.max_words = max((len(k.split()) for k in self.index), default=1)

    def lookup(self, text):
        """-> entries whose surface form occurs in text, in order of appearance (longest form first at each position).  Pure dictionary lookup over word n-grams."""
        toks = _norm(text).split(); out, seen, i = [], set(), 0
        while i < len(toks):
            for n in range(min(self.max_words, len(toks) - i), 0, -1):
                e = self.index.get(" ".join(toks[i:i + n]))
                if e is not None:
                    if e["term"] not in seen: out.append(e); seen.add(e["term"])
                    i += n; break
            else: i += 1
        return out

    def hint_line(self, text, limit=4, vehicle=None):
        """-> 'Terms: a = meaning; b = meaning' for the flagged entries found in text, or '' when none (the usual case: no extra prompt tokens).  For a combustion vehicle (petrol / diesel / ice / hybrid) an entry that has a
        `meaning_ice` gives that one: "rpm" is the engine's, "gas" is the fuel in the tank - the electric wording would push the model the wrong way."""
        ice = vehicle in COMBUSTION
        hit = [e for e in self.lookup(text) if e.get("hint")][:limit]
        if os.environ.get("VGPT_HINT_SHORT") == "1" and not ice: hit = [e for e in hit if e["term"] not in skipped_terms()]       # terms the model reads correctly without a hint (bench/hint_ablation.py: removing each loses nothing on the 3,842 audit questions); combustion cars keep every hint
        mean = lambda e: e["meaning_ice"] if ice and e.get("meaning_ice") else e["meaning"]
        if os.environ.get("VGPT_HINT_SHORT") == "1":                                                                       # compact hand-written meanings (glossary_short.json): fewer prompt tokens; combustion-car meanings stay as they are
            sh = short_meanings(); full = mean
            mean = lambda e, full=full: full(e) if (ice and e.get("meaning_ice")) else sh.get(e["term"], full(e))
        if os.environ.get("VGPT_HINT_CLAUSE") == "1": mean = lambda e, m=mean: first_clause(m(e))                        # fewer prompt tokens (about 2 per question on average); the 2026-10-26 audit: no loss (bench/speed_variants.py)
        return "Terms: " + "; ".join(f"{e['term']} = {mean(e)}" for e in hit) if hit else ""

    def aliases(self, concept):
        """-> every surface form the glossary attaches to a concept (used by the training-data generator)."""
        return sorted({f for e in self.entries if e.get("concept") == concept for f in [e["term"]] + list(e.get("forms", ()))})


_SKIP = None


def skipped_terms():
    global _SKIP
    if _SKIP is None: _SKIP = frozenset(json.load(open(os.path.join(HERE, "glossary_skip.json"), encoding="utf-8"))) | frozenset(t for t in os.environ.get("VGPT_SKIP_EXTRA", "").split(",") if t)        # VGPT_SKIP_EXTRA: more terms without a hint, to try them (bench/hint_delta.py)
    return _SKIP


_SHORT = None


def short_meanings():
    global _SHORT
    if _SHORT is None: _SHORT = json.load(open(os.path.join(HERE, "glossary_short.json"), encoding="utf-8"))
    return _SHORT


_G = None


def load():
    global _G
    if _G is None: _G = Glossary()
    return _G

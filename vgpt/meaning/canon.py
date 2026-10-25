"""The pre-parser: what is done to a question BEFORE the language model reads it, and the few questions that never need the model.

The owner's instruction (2026-10-24): do not expect a 0.5 B model to know everything; help it.  Three things, each of them visible in the answer ("I read it as: ...") and none of them silent:

  1. SPELLING AND SHORTHAND (`normalise`): "avg acc" -> "average acceleration", "chargning" -> "charging", "srevicing" -> "servicing", "wat" -> "what", "u" -> "you".  A shorthand with two meanings is read from the words
     around it ("acc" is the adaptive cruise control next to "cruise / on / engaged", acceleration otherwise).  A word is corrected only when it is not a word of the vehicle's vocabulary or of plain English, and only
     to the one nearest word of that vocabulary.
  2. KNOWN AWKWARD WORDINGS (`REWRITES`): a wording the model keeps missing is said the way it reads well - "which tyre is bad" -> "which tyre is low".  Each is a pair (pattern, plain question) and is tested in
     tests/test_canon.py against the whole assistant, not only against the text.
  3. QUESTIONS THE MODEL SHOULD NOT DECIDE (`handle`): who or what the assistant is, what it can do, the clock, what lies outside the vehicle's data (traffic law, where to buy the car), the service record, and "how much
     battery will 50 km use" (a sum from the car's own consumption).  Each is answered from the vehicle's own records or says plainly that it is out of scope.

This is a rule layer and is counted as such: tests/test_meaning.py allows `re` here and in askme.py only because the owner asked for it; `docs/CANON.md` lists every rule so the debt is visible.  What a question MEANS in
general stays with the model."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------------------------------------- shorthand
SHORT = {"avg": "average", "tmp": "temperature", "temp": "temperature", "temps": "temperatures", "pres": "pressure", "press": "pressure", "batt": "battery", "batry": "battery", "batery": "battery", "batty": "battery", "btry": "battery", "bttry": "battery", "batety": "battery", "battrey": "battery",
         "chg": "charge", "chrg": "charge", "chging": "charging", "chrging": "charging", "chrgng": "charging", "lvl": "level", "pct": "percent", "perc": "percent", "odo": "odometer", "spd": "speed", "trq": "torque",
         "kmph": "km/h", "kph": "km/h", "yday": "yesterday", "yest": "yesterday", "tmrw": "tomorrow", "rn": "right now", "wk": "week", "wks": "weeks", "mth": "month", "mths": "months", "hrs": "hours", "hr": "hour",
         "u": "you", "ur": "your", "r": "are", "pls": "", "plz": "", "thx": "", "wat": "what", "wut": "what", "wht": "what", "whts": "whats", "hw": "how", "wen": "when", "wich": "which", "abt": "about", "dis": "this",
         "thr": "the", "teh": "the", "th": "the", "cuz": "because", "coz": "because", "tho": "though", "gonna": "going to", "wanna": "want to", "gotta": "have to", "lemme": "let me", "kinda": "kind of",
         "veh": "vehicle", "mileage": "mileage", "motr": "motor", "tyr": "tyre", "tyres": "tyres", "tires": "tyres", "tire": "tyre", "ac": "ac", "regn": "regen", "diag": "diagnostic", "diags": "diagnostics"}
ACCEL_CUES = {"avg", "average", "mean", "max", "maximum", "peak", "top", "hard", "harsh", "my", "g", "force", "brake", "braking", "throttle", "pedal", "launch", "quick", "quickly", "sudden", "rate", "m/s", "0-100", "sec", "seconds", "fast", "slow", "lateral", "longitudinal", "cornering"}
AMBIGUOUS = {"acc": [("acceleration", "how quickly the speed changes"), ("adaptive cruise control", "the driver assist that holds speed and distance to the car ahead")]}
CRUISE_WORDS = {"cruise", "adaptive", "lane", "engaged", "engage", "active", "on", "off", "standby", "following", "distance", "radar", "setting", "enabled", "disabled"}

# --------------------------------------------------------------------------------------------------------------------- the words that are known
_ENGLISH = """a about above across after again against all also always am an and any anything are around as at away back be because been before being below between both but by can cannot could day days did do does doing done
down during each either else enough even ever every everything few first for from get getting give given go goes going gone good got had has have having he help her here him his hold how if in into is it its just keep kind know
last later least less let like little long look made make many may me mean might more most much must my near need never new next no not now of off often ok okay on once one only or other our out over own per please put quite
rather really right same say see seem set she should show since so some something soon still such take tell than that the their them then there these they thing think this those though through time to today tonight too tomorrow
under until up upon us use used using usually very want was way we well were what whats when where which while who whom why will with without won would yes yesterday yet you your yours
am pm hello hi hey thanks thank bye morning evening night afternoon week weeks month months year years hour hours minute minutes second seconds
how many much often long far fast slow high low hot cold warm cool full empty open closed fine ok bad best worst top highest lowest average maximum minimum total latest earliest recent recently previous
car vehicle drive driving driven drove trip trips journey journeys route road speed speeds km mile miles kilometre kilometres metre metres percent
battery charge charging charged charger charges plug plugged range energy power fuel tank petrol diesel engine motor regen regenerative brake brakes braking braked tyre tyres wheel wheels pressure temperature temperatures
coolant cooling flow pump fan heater heating air conditioning climate cabin window windscreen wipers lights light headlights lamp warning warnings alert alerts fault faults error errors code codes diagnostic service servicing
serviced maintenance odometer mileage consumption efficiency efficient economy acceleration accelerate accelerating accelerated deceleration steering gear gears torque rpm voltage current amp amps volt volts kw kwh wh
state health healthy condition status camera blocked lane assist cruise adaptive collision emergency sensor sensors signal signals data record recorded records history log logs event events happen happened happening
reset resets shutdown shutdowns dropout dropouts bus network unusual normal strange odd weird anomaly trend trending rising falling increase decreasing improve improving worse better compare compared comparison difference
weekend weekends weekday weekdays daily weekly monthly pattern patterns usually typical rate share list count counts number numbers average totals total sum per each every between since ago
who whose name called assistant model language run runs running work works working built made company vantage vehiclegpt
side law legal licence license insurance buy purchase price cost costs dealer dealership
owner owners seats seat people passengers fit capacity size weight tall length wide
summary summarise summarize explain why reason cause caused because happened wrong problem problems issue issues okay safe safety danger dangerous
""".split()


def _vocab():
    """-> (known, target, fix, cands).  `known`: every word of clean vehicle questions (training questions, clean evaluation sets; vgpt/meaning/canon_vocab.json, built by tools_canon_vocab.py), plain English and the vehicle's own glossary: such a word
    is real and is never touched.  `target`: {word: how often it occurs}; only these (frequent, so surely right) words may be the result of a correction."""
    global _V
    try: return _V
    except NameError: pass
    known = set(_ENGLISH); target = {}; fix = {}; cands = set(_ENGLISH)
    try:
        with open(os.environ.get("VGPT_CANON_VOCAB") or os.path.join(HERE, "canon_vocab.json"), encoding="utf-8") as f: d = json.load(f)
        known.update(d.get("known", ())); target.update(d.get("target", {})); fix.update(d.get("fix", {})); cands.update(d.get("cands", ()))
    except (OSError, ValueError): pass
    try:
        from . import glossary as G, ontology as ON
        for e in G.load().entries:
            for t in [e.get("term", "")] + list(e.get("forms") or []): known.update(re.findall(r"[a-z]+", str(t).lower()))
        for c in ON.load().concepts.values(): known.update(re.findall(r"[a-z]+", str(getattr(c, "label", "") or "").lower()))
    except Exception: pass
    known.difference_update(fix); _V = (known, target, fix, cands); return _V


def _within(a, b, k):
    """Damerau-Levenshtein distance of a and b is at most k (restricted: one adjacent swap counts as one edit)."""
    if abs(len(a) - len(b)) > k: return False
    prev2 = None; prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]: cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
        if min(cur) > k: return False
    return prev[len(b)] <= k


def _inflection_of_known(w, known):
    for suf, rep in (("s", ""), ("es", ""), ("ies", "y"), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""), ("ing", "e"), ("ly", ""), ("er", ""), ("est", ""), ("ers", ""), ("ions", "ion"), ("ments", "ment")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and (w[:len(w) - len(suf)] + rep) in known: return True
    return False


def _correct(word, vocab):
    """The one real word that `word` is a misspelling of, or the word itself.  A word is changed only when it is not a word at all, and only to the single real word one edit away (two edits for 8 letters or more) - when
    two words could be meant ('chaxge': change or charge) nothing is changed."""
    known, target, fix, cands = vocab; n = len(word)
    if n < 5 or any(ch.isdigit() for ch in word): return word
    if word in fix: return fix[word]                                                                             # a misspelling the training questions themselves contain
    if word in known: return word
    if _inflection_of_known(word, known): return word                                                            # sundays, regenerations, charged, braking: a known word with an ending
    k = 1 if n < 8 else 2
    for dist in range(1, k + 1):
        hit = [w for w in cands if w[0] == word[0] and abs(len(w) - n) <= dist and _within(word, w, dist)]
        if len(hit) == 1: return hit[0]
        if hit: return word
    return word


# --------------------------------------------------------------------------------------------------------------------- awkward wordings -> plain questions
REWRITES = [                                                                                                   # each pattern must match the WHOLE question (after greeting words are set aside)
    (r"which (?:tyre|tire)s? (?:is|are) (?:bad|worst|flat|weak|soft|deflated|losing air|going down|the problem)", "which tyre is low"),
    (r"(?:is|are) (?:my |the )?(?:tyre|tire)s? (?:bad|flat|ok|okay|fine|alright)", "how are my tyres"),
    (r"(?:is|are) (?:the |my |this )?(?:car|vehicle|it|ev) (?:currently |now |right now |still )?(?:charging|plugged in|on charge|on the charger)(?: now| right now| currently)?", "is the car charging"),
    (r"(?:charging|charge) (?:state|status)", "is the car charging"),
    (r"how (?:much|many) (?:km|kilometres|miles|range) (?:do i|can i|will i) (?:have|get|go|drive)", "how far can i go"),
    (r"what (?:is|are|s) (?:you|your|ur) name", "what is your name"),
    (r"(?:tell|show) me about (?:yourself|you)", "who are you"),
    (r"what (?:do|can) you (?:do|help (?:me )?with)", "what can you do"),
]
_RW = [(re.compile(p), r, False) for p, r in REWRITES]
_FILLER_HEAD = re.compile(r"^(?:(?:hey|hi|hello|so|ok|okay|um|uh|please|pls|just|can you|could you|would you|will you|tell me|do you know|i want to know|i would like to know|i wanna know|quick question|one question)[ ,]+)+")
_FILLER_TAIL = re.compile(r"(?:[ ,]+(?:please|pls|now|thanks|thank you|for me))+$")


def _strip_filler(core): return _FILLER_TAIL.sub("", _FILLER_HEAD.sub("", core))


# --------------------------------------------------------------------------------------------------------------------- the pre-parser
TOKEN = re.compile(r"[A-Za-z][A-Za-z']*|\d+(?:[.,]\d+)?|[^\sA-Za-z\d]")


WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def _recase(old, new):
    if old[:1].isupper() and new: return new[:1].upper() + new[1:]
    return new


def normalise(text):
    """-> (text for the model, [what was changed]).  Only the words that need help are replaced, in place: spacing, numbers, units and punctuation stay exactly as typed.  The text is returned unchanged (and the list empty) when
    nothing needed help."""
    orig = " ".join(str(text).split()); ms = list(WORD.finditer(orig)); low = [m.group().lower() for m in ms]; notes = []; vocab = _vocab(); ctx = set(w for w in low if "'" not in w)
    edits = []                                                                                                      # (start, end, replacement)
    for i, (m, w) in enumerate(zip(ms, low)):
        t = m.group()
        if "'" in w: continue
        if m.start() > 0 and orig[m.start() - 1].isdigit(): continue                                              # 11th, 3rd, 2nd, 5km: a suffix glued to a number is not a word                                                                                       # what's, week's, don't stay as written
        if w == "acc":
            if ctx & CRUISE_WORDS: new = "adaptive cruise control"
            elif ctx & ACCEL_CUES: new = "acceleration"
            else: continue                                                                                          # two meanings and nothing in the question says which: `ambiguous` asks
            edits.append((m.start(), m.end(), new)); notes.append((t, new)); continue
        if w == "y" and (i == 0 or low[i - 1] in ("and", "but", "so")) and i + 1 < len(low) and low[i + 1] in ("did", "is", "are", "does", "was", "do", "has", "have"): edits.append((m.start(), m.end(), "why")); notes.append((t, "why")); continue
        if w in SHORT and SHORT[w] != w:
            new = SHORT[w]; edits.append((m.start(), m.end(), _recase(t, new))); notes.append((t, new)); continue
        if t[:1].isupper() and i > 0 and w not in vocab[0]: continue                                                # a name in the middle of a sentence stays
        c = _correct(w, vocab)
        if c != w: edits.append((m.start(), m.end(), _recase(t, c))); notes.append((t, c))
    text2 = orig
    for st, en, new in sorted(edits, reverse=True):
        if new == "":                                                                                               # a dropped filler word takes one neighbouring space with it
            if st > 0 and text2[st - 1] == " ": st -= 1
            elif en < len(text2) and text2[en] == " ": en += 1
        text2 = text2[:st] + new + text2[en:]
    text2 = text2.strip()
    core = " ".join(text2.lower().strip(" ?!.").split())
    for rx, plain, anywhere in _RW:
        m = (rx.search(core) if anywhere else rx.fullmatch(_strip_filler(core)))
        if m:
            new = " ".join(m.expand(plain).split()) if chr(92) in plain else plain                                  # a plain question may reuse what the pattern caught (\1, \2)
            if new != core: notes.append((text2, new)); text2 = new
            break
    return (text2, notes) if notes else (orig, [])


def reading(notes):
    """The visible part: 'avg acc -> average acceleration'."""
    if not notes: return ""
    return "; ".join(f"{a} -> {b}" if b else f"{a} dropped" for a, b in notes[:6])


def ambiguous(text):
    """-> (word, [(meaning, explanation), ...]) when the question holds a shorthand with two meanings and no word beside it chooses; else None.  The only question the assistant asks back: and only here."""
    for w in re.findall(r"[a-z]+", str(text).lower()):
        if w in AMBIGUOUS: return w, AMBIGUOUS[w]
    return None


def choose(text, options):
    """The meaning picked by an answer to the question above ("1", "second", "the cruise one", "acceleration"), or None."""
    t = " ".join(str(text).lower().strip(" .!?").split())
    nums = {"1": 0, "one": 0, "first": 0, "the first": 0, "first one": 0, "2": 1, "two": 1, "second": 1, "the second": 1, "second one": 1}
    if t in nums: return options[nums[t]][0]
    for i, (m, _) in enumerate(options):
        if t == m or (len(t) >= 4 and (t in m or any(x and x in t for x in m.split() if len(x) >= 5))): return m
    return None

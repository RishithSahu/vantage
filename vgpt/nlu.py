"""Language pre-processing for the assistant (stdlib only, ~O(100) vocabulary words, microseconds per query).

  normalize(text)   lower-case, split on punctuation, map synonyms (petrol/diesel/gas -> fuel, milage -> mileage, refuel -> recharge ...)
                    and repair typos by Damerau-Levenshtein distance against the assistant's OWN vocabulary only (so ordinary words are never "corrected")
  parse_when(t)     time window codes: y today-1, t today, we last weekend, sw since last weekend, tw this week, sc since last charge, 7 / 14 / 28 days
  unsupported(t)    topics a car has no data for here (tyre pressure, driver score, navigation ...) -> an honest answer instead of a wrong one
"""
import os, re

SYNONYMS = {"petrol": "fuel", "gasoline": "fuel", "diesel": "fuel", "gas": "fuel", "mpg": "efficiency", "kmpl": "efficiency", "milage": "mileage", "mileages": "mileage",
            "refuel": "recharge", "refuels": "recharge", "refuelling": "recharging", "refueling": "recharging", "refueled": "recharged", "refuelled": "recharged",
            "refill": "recharge", "refilled": "recharged", "tank": "battery", "kilometres": "km", "kilometers": "km", "kilometre": "km", "kilometer": "km", "miles": "km", "mile": "km",
            "tyre": "tire", "tyres": "tire", "tires": "tire", "electricity": "energy", "juice": "energy", "wh/km": "efficiency", "kms": "km", "temp": "temperature", "grear": "gear", "lst": "last", "diagnostc": "diagnostic", "grears": "gears", "breaks": "brakes"}

# text-message / voice-transcript shorthand and fillers (generic, not tied to any question): mapped before typo repair so "wats my batery temprature rite now" reads like the plain sentence
SMS = ({"wats": "whats", "wat": "what", "rite": "right", "hw": "how", "wk": "week", "wks": "weeks", "yestrdy": "yesterday", "yestrday": "yesterday", "yesterdy": "yesterday", "yday": "yesterday", "spd": "speed", "ppl": "people", "pls": "please", "plz": "please",
                  "b4": "before", "dis": "this", "da": "the", "u": "you", "ur": "your", "r": "are", "wen": "when", "whn": "when", "gimme": "give me", "lemme": "let me", "tmrw": "tomorrow", "lowst": "lowest", "presure": "pressure", "psi": "psi",
                  "lol": "", "yo": "", "um": "", "umm": "", "uh": "", "kinda": "", "sorta": "", "juice": "energy", "burn": "use", "burnt": "used", "burned": "used"})
SYNONYMS.update({"what's": "what is", "whats": "what is", "how's": "how is", "hows": "how is", "where's": "where is", "wheres": "where is", "when's": "when is", "that's": "that is", "it's": "it is", "who's": "who is", "i'm": "i am", "i've": "i have", "don't": "do not", "doesn't": "does not", "can't": "cannot"})
SYNONYMS.update({"dan": "than", "den": "then", "nite": "night", "tonite": "tonight", "wuz": "was", "cuz": "because", "becuz": "because", "wen": "when", "wat": "what", "wich": "which", "thru": "through", "yday": "yesterday", "tmrw": "tomorrow", "abv": "above", "blw": "below", "avg": "average", "mx": "maximum", "mn": "minimum"})
SYNONYMS.update(SMS)
SYNONYMS.update({"whats": "what is", "hows": "how is"})


def shorthand(text):
    """Replace text-message shorthand and drop fillers, leaving every other word exactly as typed (a typo is repaired later, only when the first reading fails)."""
    out = " ".join(x for x in (SMS.get(w.lower(), w) for w in str(text).split()) if x != "")
    return re.sub(r"^why you do not\b", "why do you not", out, flags=re.I)

# words the assistant itself knows: only unknown tokens close to one of these are repaired
VOCAB = set("""yesterday today tomorrow weather temperature battery charge charging recharge recharged consumption efficiency mileage distance fuel energy range highway city compare drive driving drove driven
weekend week month fortnight since last used spent using speed average maximum minimum coolant cooling camera fault faults error codes warning overheating overheat adas lane assist disengagement
disengagements sessions session summary trip trips outside climate heating pack hottest coldest remaining expected estimated current voltage power kilometres hybrid engine
recently lately maintenance servicing service odometer tyre tyres tire pressure rain raining rainy cloudy sunny weather bright wipers brake brakes braking pads health healthy problem problems issues location parked parking state behind motorway running before hotter worse harsh smooth aggressive regenerative lens blocked blockage glare blurry foggy destination status working condition overheated hard accelerate acceleration cornering diagnostic diagnostics overheating warning warnings events event anomaly anomalies degradation sensor sensors correlation temperature coolant statistically significant prove explain explanation evidence reduced reduction reducing""".split())

VOCAB |= set("puncture punctured punctures flat deflated deflation tread torque farthest furthest nearest creeping climbing dropping drifting slipping shortest longest quickest slowest stationary continuous continuously consecutive episodes episode percentile percentiles quartile interquartile median deviation baseline preceding overnight daytime weekday weekdays recommended threshold thresholds uninterrupted unbroken separate occasions occasion crossed crossing exceeded exceeding fluctuating variability stable steady parked charging driving moving idle idling regenerative regeneration correlated correlation causation causal confirmed corroborated independent certain certainty provably proof prove proven degradation degraded recorded unrecorded".split())
try:                                                                                    # the registry is the domain vocabulary: its names and synonyms are never "typos" of common English words (torque -> true)
    from . import signal_registry as _R
    VOCAB |= {w for c in _R.CONCEPTS.values() for ph in (c.label, c.title if hasattr(c, "title") else "", *c.synonyms) for w in re.findall(r"[a-z]{4,}", str(ph).lower())}
except Exception: pass
VOCAB |= {w + e for w in list(VOCAB) for e in ("s", "es", "ed", "ing", "d") if not (w.endswith("e") and e in ("ed", "ing", "es"))} | {w[:-1] + "ing" for w in list(VOCAB) if w.endswith("e") and len(w) > 3}          # inflections are known words too (weeks, sessions, charged ...)

# ordinary English the user may type; never "corrected" (keeps repairs conservative)
COMMON = set("""a an the is are was were be been am do does did have has had can could will would should may might my me i you we your our it its this that these those there here what when where which who whom why how
much many more most less least few some any all each every no not nor and or but if then than so as of at by for from in into on onto to up down out off over under with without about after before during while
again once also just only very too quite rather really now then still even ever never always often usually long ago later soon since until till per get got give gave go going gone come came take took make made see saw
show tell find look need want like know think say said ask asked say please thanks thank hello hi hey ok okay yes yeah lot lots bit good bad best worst better worse high low big small new old own same different
first second third next previous recent recently lately days day hours hour minutes minute seconds second morning night evening afternoon tonight monday tuesday wednesday thursday friday saturday sunday january
february march april may june july august september october november december one two three four five six seven eight nine ten twenty thirty hundred expect expected right left away near far fast slow
car vehicle auto drive driver rate skill score tire tires brake brakes break breaks oil music song radio call text message change changed changes time times charge charged trend drop drops dropped rise rose increase decrease higher lower hotter colder average total""".split())


def _load_en():
    """~10,000 most frequent English words (Google Trillion Word Corpus list): ordinary words are never 'corrected' into assistant vocabulary."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "en_10k.txt"), encoding="utf-8") as f: return set(w.strip() for w in f if w.strip())
    except OSError: return set()


EN = _load_en()


def _load_rank():
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "en_10k.txt"), encoding="utf-8") as f: return {w.strip(): i for i, w in enumerate(f) if w.strip()}
    except OSError: return {}


_EN_RANK = _load_rank()


def _is_english(tok):
    if tok in EN: return True
    for suf in ("s", "es", "ed", "d", "ing", "er", "ers", "ly", "y"):                   # simple inflections of a known word
        if tok.endswith(suf):
            base = tok[: -len(suf)]
            if len(base) < 3: continue
            if suf == "es" and base.endswith("y"): continue
            if suf in ("ing", "ed", "er", "ers") and base.endswith("e") and not base.endswith(("ee", "ye", "oe")): continue          # "chargeing" is not an inflection of "charge": the e is dropped                                   # "cityes" is a misspelling of "cities"
            if base in EN or (suf in ("ing", "ed", "d", "er", "ers") and base + "e" in EN) or (len(base) > 2 and base[-1] == base[-2] and base[:-1] in EN): return True      # only -ing/-ed forms restore a dropped e
    return False


def _dl(a, b, limit):
    """Damerau-Levenshtein distance with early exit above `limit`."""
    if abs(len(a) - len(b)) > limit: return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]: cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit: return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


_DOUBLES = set("ll ss ee oo tt ff pp nn mm rr dd gg bb zz cc".split())


def _both_regular_inflections(tok, cand):
    """True when tok and its proposed correction are both regular -ing / -ed forms and tok's stem is a well-formed English stem (no impossible doubled letters):
    two different real words, so the token was not a slip of the keyboard."""
    for suf in ("ing", "ed"):
        if tok.endswith(suf) and cand.endswith(suf):
            stem = tok[: -len(suf)]
            if len(stem) < 4 or not re.search(r"[aeiouy]", stem): return False
            if stem.endswith("e") and not stem.endswith(("ee", "ye", "oe")): return False                  # "makeing": an e before -ing is a slip, not a word
            if any(stem[i] == stem[i + 1] and stem[i:i + 2] not in _DOUBLES for i in range(len(stem) - 1)): return False
            return True
    return False


try:
    _SW = __import__("collections").Counter(w for c in _R.CONCEPTS.values() for ph in (c.label, *c.synonyms) for w in re.findall(r"[a-z]{4,}", str(ph).lower())); _SIGNAL_WORDS = set(_SW)
except Exception:
    _SIGNAL_WORDS = set()


def _dict_candidate(tok, limit):
    cand, cd, cr = None, limit + 1, 10 ** 9
    for w, rank in _EN_RANK.items():
        if w[0] != tok[0] or len(w) < 4 or abs(len(w) - len(tok)) > limit: continue
        d = _dl(tok, w, limit)
        if d < cd or (d == cd and rank < cr): cand, cd, cr = w, d, rank
    return (cand, cd) if cand is not None and cd <= limit else (None, limit + 1)


def repair(tok):
    """Return (word, changed). Only tokens that are neither vocabulary nor common words are candidates."""
    if tok in VOCAB or tok in COMMON or tok in SYNONYMS or len(tok) < 4 or not tok.isalpha() or _is_english(tok): return tok, False
    compound = any(tok[:i] in EN and tok[i:] in EN for i in range(4, len(tok) - 3))                  # two ordinary words joined (headwind, sidewall): a word, unless a single slip explains it
    limit = 1 if len(tok) <= 5 else 2
    best, bd = None, limit + 1
    for w in VOCAB | set(SYNONYMS):
        if w[0] != tok[0] and not (len(tok) >= 7 and w[1:3] == tok[1:3]): continue     # real typos almost never change the first letter ("france" must not become "range")
        d = _dl(tok, w, limit)
        if d < bd or (d == bd and best is not None and (_SW.get(w, 0) > _SW.get(best, 0) or (_SW.get(w, 0) == _SW.get(best, 0) and len(w) < len(best)))): best, bd = w, d
    if best is not None and bd <= limit and not (compound and bd >= 2):
        if bd >= 2:                                                                                          # an ordinary word one slip away beats an assistant word two slips away ("makeing" -> making, not marking)
            c1, d1 = _dict_candidate(tok, 1)
            if c1 is not None and d1 < bd and not _both_regular_inflections(tok, c1): return c1, True
        return best, True
    if compound: return tok, False
    # standard frequency-dictionary spell correction: no assistant word is close, so take the nearest ordinary English word (same first letter), ties broken by corpus frequency
    cand, cd, cr = None, limit + 1, 10 ** 9
    for w, rank in _EN_RANK.items():
        if w[0] != tok[0] or len(w) < 4 or abs(len(w) - len(tok)) > limit: continue
        d = _dl(tok, w, limit)
        if d < cd or (d == cd and rank < cr): cand, cd, cr = w, d, rank
    if cand is not None and cd <= limit and _both_regular_inflections(tok, cand): return tok, False       # "creeping" is a word, not a misspelling of "creating"
    return (cand, True) if cand is not None and cd <= limit else (tok, False)


def normalize(text, repair_typos=True):
    """-> (clean_text, typo_fixes[(from, to)]). repair_typos=False only lower-cases, splits and maps synonyms (the text as typed)."""
    t = re.sub(r"\s+", " ", re.sub(r"[-_]", " ", text.lower())).strip().rstrip("?.! ")
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sa-z0-9]", t)
    # Joint schema-aware recovery: two nearby damaged words may each look like
    # an ordinary English word in isolation (``betry ... hoy``), but together
    # they form an unambiguous component + physical-state frame.  Correct the
    # pair from the ontology rather than maintaining sentence exceptions.
    joint = {}
    if repair_typos:
        components = {"battery", "motor", "engine", "coolant", "tyre", "tire", "brake", "charger"}
        states = {"hot", "cold", "high", "low", "flat", "full", "empty", "healthy"}
        def close(tok, vocab, limit):
            # Never reinterpret a valid common English word here.  This keeps
            # function words such as ``how`` from becoming physical states.
            if not tok.isalpha() or tok in COMMON or _is_english(tok): return None
            hit = [( _dl(tok, cand, limit), cand) for cand in vocab if cand[0] == tok[0] and abs(len(cand) - len(tok)) <= limit]
            hit = sorted(x for x in hit if x[0] <= limit)
            return hit[0][1] if hit and (len(hit) == 1 or hit[0][0] < hit[1][0]) else None
        # Component names are longer and voice/text errors often collapse an
        # unstressed syllable (battery -> betry), so allow three edits only in
        # this paired component+state context.
        comp = [(i, close(tok, components, 3)) for i, tok in enumerate(tokens)]
        state = [(i, close(tok, states, 1)) for i, tok in enumerate(tokens)]
        pair = next(((i, a, j, b) for i, a in comp if a for j, b in state if b and i != j and abs(i-j) <= 4), None)
        if pair:
            i, a, j, b = pair; joint[i], joint[j] = a, b
    out, fixes = [], []
    for i, tok in enumerate(tokens):
        w, ch = (joint[i], joint[i] != tok) if i in joint else (repair(tok) if repair_typos else (tok, False))
        if ch: fixes.append((tok, w))                                   # typo repairs only (synonym mapping is silent)
        out.append(SYNONYMS.get(w, w))
    clean = " ".join(out)
    clean = re.sub(r"\bbreak (?=(?:pads?|fluid|wear)\b)", "brake ", clean)
    clean = re.sub(r"\b(\d+(?:\.\d+)?)(secs?|mins?|hrs?|hours?|days?|weeks?|months?)\b", r"\1 \2", clean)
    clean = re.sub(r"^why you do not\b", "why do you not", clean)
    clean = re.sub(r" ([?.,!])", r"\1", clean)
    return clean, fixes


WEEK_DEFAULT = "7"


_MON = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
_MONRX = "|".join(sorted(_MON, key=len, reverse=True))


def explicit_date(t):
    """'on 14 september', 'september 14th 2026', '2026-09-14' (normalised text: hyphens are spaces, '14th' is '14 th') -> 'd:YYYY-MM-DD' or 'd:MM-DD', else None."""
    m = re.search(r"\b(20\d\d) (\d{1,2}) (\d{1,2})\b", t)
    if m and 1 <= int(m[2]) <= 12 and 1 <= int(m[3]) <= 31: return f"d:{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
    m = re.search(r"\b(\d{1,2}) (?:st |nd |rd |th )?(?:of )?(" + _MONRX + r")\b(?: (20\d\d))?", t)
    if m: d, mo, y = int(m[1]), _MON[m[2]], m[3]
    else:
        m = re.search(r"\b(" + _MONRX + r") (\d{1,2})(?: st| nd| rd| th)?\b(?: (20\d\d))?", t)
        if not m: return None
        mo, d, y = _MON[m[1]], int(m[2]), m[3]
    if not 1 <= d <= 31: return None
    return f"d:{int(y):04d}-{mo:02d}-{d:02d}" if y else f"d:{mo:02d}-{d:02d}"


def parse_when(t, default=None):
    """Time window code from normalised text, else `default`."""
    ed = explicit_date(t)
    if ed: return ed
    if re.search(r"since (the |my |i )?(last |previous )?(re)?charg|since (the |my )?last (fill|top)|after (the |my )?last (re)?charg", t): return "sc"
    if re.search(r"since (the )?(last |previous )?weekend", t): return "sw"
    if re.search(r"last weekend|over the weekend|this weekend|on the weekend", t): return "we"
    if "yesterday" in t: return "y"
    if re.search(r"\btoday\b|this morning|so far today", t): return "t"
    if re.search(r"two weeks|fortnight|14 days|last 2 weeks|past 2 weeks", t): return "14"
    if re.search(r"this week", t): return "tw"
    if re.search(r"last week|past week|7 days|last seven days|few days", t): return "7"
    if re.search(r"month|4 weeks|four weeks|28 days", t): return "28"
    m = re.search(r"(?:last|past)\s+(\d+)\s+days?", t)
    if m: return "7" if int(m[1]) <= 7 else ("14" if int(m[1]) <= 14 else "28")
    return default


UNSUPPORTED = [
    (r"\bmy (?:phone|iphone|android|laptop|tablet|smartwatch|watch|computer|kindle|headphones|earbuds)\b", "other devices", "I only know this vehicle's recorded data, not your phone or other devices"),
    (r"forecast|will it rain|tomorrow|next week|next month|predict|\bwill\b.{0,40}\b(?:in (?:a|an|one|two|three|\d+) (?:day|week|month|year)s?|next (?:day|year|charge|trip)|after the next|in the future)\b", "forecasts", "I only know past and current data from the car, not forecasts"),
    (r"\b(?:write|compose|draw|sing|make up)\b.{0,25}\b(?:poem|song|story|essay|haiku|lyrics|joke)|tell me a joke|\bpoem\b", "creative writing", "writing poems, stories or jokes is outside this assistant"),
    (r"navigate|navigation|directions|route to|traffic(?! lights?)|nearest (?!second|minute|hour|day|whole|integer|decimal|tenth|hundredth|percent|neighbo)|nearby|petrol station|fuel station|charging station", "navigation", "navigation and map data are not available to me"),
    (r"warranty|insurance", "warranty and insurance", "warranty and insurance details are not stored in the car"),
    (r"how (to|do i|can i|would i|do you) (attack|hack|exploit|inject|spoof|jam|bypass|disable|crack|break into)", "attack instructions", "I report attacks my monitors detected on the car's networks, but I don't explain how to carry them out"),
    (r"where (can|should|do|could) (i|we) park|where to park|find (a )?parking|parking (spot|space|lot|garage|near)|nearest parking", "finding parking", "I have no map or parking data - I can only tell you where the car is now and where you parked before"),
    (r"\bplay |music|radio|song|podcast|\bcall |text message|whatsapp|remind", "media and phone", "media and phone control are outside this assistant"),
    (r"\bhorn\b|decib\w+|\bvolume\b|loudness", "horn and audio levels", "I have no audio-level data from the car"),
    (r"(how many|who).{0,15}(people|ppl|passengers|persons|kids).{0,15}(in|sitting|inside|on board|riding)|occupan|seat ?belt", "seat occupancy", "this car does not report who is sitting in it to the assistant"),
    (r"how many\b.{1,50}\b(?:destroy|damage|wreck|crush|break)\b.{0,25}\b(?:car|vehicle)\b|how many\b.{1,25}\b(?:car|vehicle)\b.{0,25}\b(?:destroy|damage|wreck|crush|break)\b", "hypothetical damage quantities", "vehicle telemetry cannot calculate fictional or unspecified damage scenarios"),
    (r"how (to|do i|can i|should i) (start|turn on|switch on|turn off|unlock|lock|open|close|reset|operate|put)\b|when did i (buy|purchase|get)|purchase date|date of purchase", "owner's-manual questions", "I answer from the car's recorded data - how-to instructions and purchase details are in the owner's manual and your paperwork"),
]


def unsupported(t):
    for rx, label, why in UNSUPPORTED:
        if re.search(rx, t): return label, why
    return None

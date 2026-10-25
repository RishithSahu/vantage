"""Surface noise for training sentences: typing slips, abbreviations, fillers, casing, missing punctuation.  The label never changes; the model learns that spelling and register are not meaning
(this replaces a typo-repair rule layer at run time)."""
import random

ADJ = {"q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awd", "d": "sef", "f": "drg", "g": "fth", "h": "gyj",
       "j": "huk", "k": "jil", "l": "kop", "z": "asx", "x": "zsc", "c": "xdv", "v": "cfb", "b": "vgn", "n": "bhm", "m": "nj"}
ABBR = {"yesterday": "yday", "average": "avg", "temperature": "temp", "please": "pls", "week": "wk", "weeks": "wks", "month": "mnth", "minutes": "mins", "hours": "hrs", "between": "btw",
        "versus": "vs", "tomorrow": "tmrw", "because": "bc", "you": "u", "your": "ur", "what is": "whats", "what's": "whats", "do not": "dont", "kilometres": "kms", "maximum": "max", "minimum": "min",
        "current": "curr", "battery": "batt", "motor": "mtr", "highway": "hwy", "kilometers": "kms", "number": "no.", "approximately": "approx", "message": "msg", "information": "info"}
FILLERS = ["uh ", "um ", "hey ", "hey so ", "ok so ", "so ", "quick question ", "just wondering ", "btw ", "hi ", "yo ", "right so ", "sorry ", "umm ", "also ", "ok ", "listen "]
TAILS = [" pls", " please", " thanks", " quickly", " asap", " for me", " rn", " ?", "?", "??", "!"]


def typo(s, r):
    ws = s.split(" ")
    idx = [i for i, w in enumerate(ws) if len(w) >= 4 and w.isalpha()]
    if not idx: return s
    i = r.choice(idx); w = ws[i]; k = r.randrange(1, len(w) - 1); mode = r.choice(("swap", "drop", "dup", "adj", "vowel"))
    if mode == "swap": w = w[:k] + w[k + 1] + w[k] + w[k + 2:] if k + 1 < len(w) else w
    elif mode == "drop": w = w[:k] + w[k + 1:]
    elif mode == "dup": w = w[:k] + w[k] + w[k:]
    elif mode == "adj" and w[k] in ADJ: w = w[:k] + r.choice(ADJ[w[k]]) + w[k + 1:]
    elif mode == "vowel":
        v = [j for j, ch in enumerate(w) if ch in "aeiou" and 0 < j]
        if v: j = r.choice(v); w = w[:j] + w[j + 1:]
    ws[i] = w
    return " ".join(ws)


def abbreviate(s, r):
    out = s
    keys = [k for k in ABBR if k in out]
    for k in r.sample(keys, min(len(keys), r.randint(1, 3))): out = out.replace(k, ABBR[k], 1)
    return out


def apply(s, r, p=0.35):
    """-> possibly-noised sentence.  p = probability that anything is applied."""
    if r.random() > p: return s
    ops = r.sample(("typo", "abbr", "filler", "tail", "case", "punct", "typo2"), r.randint(1, 3))
    for op in ops:
        if op == "typo": s = typo(s, r)
        elif op == "typo2": s = typo(typo(s, r), r)
        elif op == "abbr": s = abbreviate(s, r)
        elif op == "filler": s = r.choice(FILLERS) + s
        elif op == "tail": s = s.rstrip("?.! ") + r.choice(TAILS)
        elif op == "case": s = s.upper() if r.random() < 0.15 else s.capitalize() if r.random() < 0.5 else s.lower()
        elif op == "punct": s = s.rstrip("?.! ")
    return " ".join(s.split())


PREFIXES = ["could you tell me ", "i would like to know ", "i was wondering ", "please tell me ", "can you let me know ", "i need to know ", "would you mind telling me ", "i am curious ", "i just want to know ",
            "not sure if this is possible but ", "this might be a silly question but ", "sorry to bother you but ", "quick question: ", "so basically ", "i guess ", "my friend asked me and i do not know so ",
            "for my records ", "for the log ", "out of curiosity ", "by the way ", "one more thing ", "hmm let me think ", "ok so ", "well ", "tell me ", "i want to know ", "just checking ", "can i ask ",
            "i have been wondering ", "my wife wants to know ", "before i forget ", "i keep thinking about this: "]
SUFFIXES = [" please", " if you can", " thanks", " when you get a chance", " for me", " if possible", " thank you", " ok", " i guess", " or something", " you know"]


def wrap(s, r):
    """A meaning-preserving wrapper: politeness, hedging, backstory fragments, trailing filler (no time words, so `now` / period never changes).  The label never changes."""
    s = s.strip()
    if r.random() < 0.75: s = r.choice(PREFIXES) + s
    if r.random() < 0.40: s = s.rstrip("?.! ") + r.choice(SUFFIXES)
    if r.random() < 0.20: s = s.rstrip("?.! ") + r.choice(("?", "??", "?!", "..."))
    return " ".join(s.split())

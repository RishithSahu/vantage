"""Per-vehicle schema overlay: how a vehicle whose storage or signal set differs from the built-in registry is described to the system at run time, without changing code.

An overlay (JSON-able dict) may contain
  "store":    {signal: {tier: [table, column, scale, offset?]}}   replace the storage mapping of a known signal (a different column, a different stored unit: value = raw / scale + offset)
  "concepts": [{name, label, unit, category, synonyms, dtype?, kind?, lo?, hi?, description?, store?}]   register NEW signals (they are then available to every generic operation: statistics, extremes, thresholds, trends, change detection)
  "drop":     [signal, ...]   signals that are not part of this vehicle at all (removed from the ontology and the storage mapping; unknown to the parser and to the model's grammar)
Availability is still decided from the data (Profile): a registered signal with no observations is known but unavailable.
`with overlay(spec):` applies and restores everything it touched (registry, lexicon, parser vocabulary), so tests and experiments cannot leak into each other."""
import contextlib
import copy
import re

from . import signal_registry as R, nl2ir, vocab_parts


def _snapshot(): return (dict(R.CONCEPTS), copy.deepcopy(R.STORE), dict(R.LEXICON), nl2ir._CONCEPT_PHRASES, nl2ir._CONCEPT_RX, {t: set(c) for t, c in R.TABLE_COLUMNS.items()})


def _rebuild():
    R.LEXICON.clear(); R.LEXICON.update(R._build_lexicon())
    nl2ir._CONCEPT_PHRASES = sorted(R.LEXICON, key=lambda p: (-len(p), p)); nl2ir._CONCEPT_RX = re.compile(r"(?<![\w/])(" + nl2ir._alt(nl2ir._CONCEPT_PHRASES) + r")(?:'s|s|es)?(?![\w/])")


def apply(spec):
    """Apply an overlay; returns a token for restore()."""
    token = _snapshot()
    for name in spec.get("drop", []): R.CONCEPTS.pop(name, None); R.STORE.pop(name, None)
    for c in spec.get("concepts", []):
        con = R.C(c["name"], c["label"], c.get("unit", "none"), c.get("category", "vehicle"), tuple(c.get("synonyms", ())), c.get("lo"), c.get("hi"), c.get("dtype", "numeric"), c.get("kind", "measured"), c.get("description", ""), tuple(c.get("levels", ())))
        R.CONCEPTS[con.name] = con
        if c.get("store"): R.STORE[con.name] = {tier: tuple(v) for tier, v in c["store"].items()}
    for sig, tiers in spec.get("store", {}).items(): R.STORE[sig] = {tier: tuple(v) for tier, v in tiers.items()}
    for tiers in R.STORE.values():                                                                            # the planner's SQL whitelist covers exactly the registered columns (the model never writes SQL)
        for entry in tiers.values(): R.TABLE_COLUMNS.setdefault(entry[0], set()).add(entry[1])
    _rebuild(); return token


def restore(token):
    concepts, store, lexicon, phrases, rx, columns = token
    R.TABLE_COLUMNS.clear(); R.TABLE_COLUMNS.update(columns)
    R.CONCEPTS.clear(); R.CONCEPTS.update(concepts); R.STORE.clear(); R.STORE.update(store); R.LEXICON.clear(); R.LEXICON.update(lexicon); nl2ir._CONCEPT_PHRASES = phrases; nl2ir._CONCEPT_RX = rx


@contextlib.contextmanager
def overlay(spec):
    token = apply(spec)
    try: yield
    finally: restore(token)

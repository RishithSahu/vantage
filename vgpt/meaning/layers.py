"""Switches for every piece of code that stands between the learned frame and the answer and is not a literal parser, a type check, a safety guard or a calculation.

The principle of docs/semantic_method.md: code parses values, enforces types and safety, and calculates; the model decides what a question means.  The pieces below are the places where code still
influences meaning (each is listed in docs/TRAINING_QUEUE.md as debt).  They can be switched off one by one, or all at once (`model_only`), so that what the *model* contributes can be measured
without them (`bench/meaning_eval/layer_ablation.py`):

    hint_terms    only rule B of arbitrate: a glossary word flagged as one the model misreads ("switched on my car" read as an unexpected shutdown) is in the question and none of the model's own concept's words are
    arbitrate     rare-word rules that replace the model's concept from word spans of the glossary (retired: OFF unless VGPT_ARBITRATE=1)
    own_vocab     a signal this vehicle's own profile declares, named in full in the question, replaces `unlisted` or a clearly different concept
    unsafe_floor  an `unsafe` reading with p < 0.70 is asked again instead of refused
    grounding_floor  a concept read with p < 0.50 of which no word at all is in the question ("why does the Moon have phases" read as time_to_full) is treated as `unlisted`: refused as not about this vehicle
    investigate   a long or multi-sentence request that matches an analysis template goes to the investigator instead of the single-frame path
    discourse     follow-up completion ("and yesterday?", "which one was the longest", "how did you get that") from closed-class cue words
    fuel_remap    for a combustion vehicle the model's electric reading of a power / energy question is executed in engine terms

Not switchable here, on purpose: the safety guard (a decision about driving on with a hazard) and the typed literal parser (numbers, units, dates).

    VGPT_LAYERS_OFF=own_vocab,investigate        switch those off          VGPT_LAYERS_OFF=all  (or VGPT_MODEL_ONLY=1)  switch every one off
    VGPT_ARBITRATE=1                             switch `arbitrate` on (it is off by default)"""
import os

ALL = ("arbitrate", "hint_terms", "own_vocab", "unsafe_floor", "grounding_floor", "investigate", "discourse", "fuel_remap")
DEFAULT_OFF = ("arbitrate",)


def off():
    """-> the set of layers that are switched off right now (read from the environment at every call, so a test can change it)."""
    if os.environ.get("VGPT_MODEL_ONLY") == "1": return set(ALL)
    raw = os.environ.get("VGPT_LAYERS_OFF", "").strip().lower()
    if raw in ("all", "model_only", "*"): return set(ALL)
    s = {x.strip() for x in raw.split(",") if x.strip()} | set(DEFAULT_OFF)
    if os.environ.get("VGPT_ARBITRATE") == "1" and "arbitrate" not in raw.split(","): s.discard("arbitrate")
    return s


def on(name):
    return name not in off()


def preparser_on():
    """The pre-parser (canon.py spelling / shorthand repair, askme.py answers that need no model) is part of the product (docs/CANON.md); VGPT_PREPARSER=0, VGPT_MODEL_ONLY=1 or VGPT_LAYERS_OFF=all switch it off, so that what the model alone
    reads can be measured."""
    return os.environ.get("VGPT_PREPARSER") != "0" and os.environ.get("VGPT_MODEL_ONLY") != "1" and os.environ.get("VGPT_LAYERS_OFF", "").strip().lower() not in ("all", "model_only", "*")


def model_only():
    return off() >= set(ALL)

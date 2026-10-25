"""The typed non-executing outcomes of a query plan (op=none) and how each is rendered - the single place where their meaning lives.

Categories (p.why):
    greeting     social turn: hello, thanks, bye, who are you, help.                                  -> greeting + examples
    offtopic     a request that is not about this vehicle (trivia, translation, recipes, ...).         -> polite decline, scope statement
    unsupported  about the vehicle but not something the product can compute / a premise it cannot represent -> will not guess
    unrecorded   a quantity this vehicle does not record                                              -> not recorded
    control      an instruction to operate the vehicle                                                -> read-only statement
    unsafe       a request to harm, bypass, impersonate, tamper, surveil or obtain another party's data -> refusal; nothing is executed or revealed
    ambiguous    an in-scope question missing its quantity or period                                  -> asks for it
    definition   a request for a term's meaning (rendered by the glossary)

Contract versions.  An adapter is trained against one outcome contract; the renderer must read its labels with the same meaning.
    1  (v12, v13, v14*, v15): `offtopic` was supervised mostly as the social/greeting label (v12-v14d) and as "not about the vehicle" (v15).  The legacy
       rendering is kept bit-for-bit so that frozen adapters behave exactly as when they were evaluated.
    2  (semantic-contract adapters): every category above has its own meaning; `offtopic` is never a greeting.
A sidecar `<adapter>.contract.json` ({"outcome_contract": 2}) or the env var VGPT_QP_CONTRACT selects the version; the default is 1."""
import json
import os

CATEGORIES = ("greeting", "offtopic", "unsupported", "unrecorded", "control", "unsafe", "ambiguous", "definition")

_V1 = {
    "unsupported": "That is outside what this vehicle's recorded data can tell me, so I will not guess.",
    "unrecorded": "That is not something this vehicle records, so I cannot answer it from its data.",
    "smalltalk": "I am here to answer questions about this vehicle's recorded data.",
    "control": "I can only read this vehicle's recorded data; I cannot operate vehicle functions.",
    "offtopic": "Hello - I answer questions about this vehicle's recorded data: battery, temperatures, trips, charging, warnings and more. Try \"how many hard brakes last week?\".",
    "ambiguous": "I am not sure which signal or period you mean. Tell me the quantity and the time span and I will compute it.",
}
_EXAMPLES = "battery, temperatures, trips, charging, warnings and more. Try \"how many hard brakes last week?\"."
_V2 = dict(_V1)
_V2.update({
    "greeting": "Hello - I answer questions about this vehicle's recorded data: " + _EXAMPLES,
    "smalltalk": "Hello - I answer questions about this vehicle's recorded data: " + _EXAMPLES,
    "offtopic": "That is not a question about this vehicle's recorded data, so I will leave it. I can help with " + _EXAMPLES,
    "unsafe": "I cannot help with that. I only read this vehicle's own recorded data, and I will not help bypass, tamper with, or get into a vehicle, account or data that is not yours.",
})

# the facts every non-executing outcome carries, so a consumer can tell the categories apart without parsing text
_KIND = {"greeting": "social", "smalltalk": "social", "offtopic": "out_of_scope", "unsupported": "unsupported", "unrecorded": "unrecorded", "control": "control",
         "unsafe": "unsafe", "ambiguous": "ambiguous", "definition": "definition"}


def contract_for(adapter_path=None):
    v = os.environ.get("VGPT_QP_CONTRACT")
    if v in ("1", "2"): return int(v)
    if adapter_path:
        for p in (str(adapter_path) + ".contract.json", os.path.splitext(str(adapter_path))[0] + ".contract.json"):
            try:
                with open(p, encoding="utf-8") as fh: return int(json.load(fh).get("outcome_contract", 1))
            except (OSError, ValueError, TypeError): pass
    return 1


def known(why, contract=1):
    if contract >= 2: return why in _V2
    return why in _V1


def render(why, contract=1):
    """-> (text, outcome_kind).  An unknown category is never silently turned into another one: the caller must treat None as a validator failure."""
    table = _V2 if contract >= 2 else _V1
    if why not in table: return None, None
    return table[why], _KIND.get(why, why)

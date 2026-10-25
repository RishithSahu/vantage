"""One question, either powertrain: what a person says about "fuel" in a car that has none, read as the quantity that does the same job in that car.

A driver of an electric car who asks "what is my fuel level" or "which trip used the most fuel" means the charge left and the energy used; the language model is trained on both kinds of cars and correctly reads "fuel level" as the fuel
level.  Refusing ("that is not recorded") is right about the signal and wrong about the person.  So a question about a fuel quantity, put to a vehicle that has a traction battery and no fuel, is answered about the matching
battery quantity, and the first line says so.  The mapping is a table of what each fuel quantity IS for (how much is left; how much is used per distance), and it works in the other direction in `fuelx` (the combustion executors)."""
FUEL_TO_BATTERY = {
    "fuel_level": ("battery_soc", "its state of charge is the equivalent of a fuel gauge"),
    "fuel_rate": ("wh_per_km", "its energy consumption (Wh per km) is the equivalent of fuel consumption"),
}


def maybe(res, frame):
    """-> (frame for the battery quantity, one-line note) or None when this is not a fuel question put to a battery-only vehicle."""
    to = FUEL_TO_BATTERY.get(frame.concept)
    if to is None or to[0] not in res.onto.concepts: return None
    try:
        from . import fuelx as FX
        cp = FX.caps_of(res)
    except Exception: return None
    if not cp.get("battery") or cp.get("fuel") or cp.get("engine"): return None
    from .frame import Frame
    return Frame(frame.act, to[0], frame.op, frame.cond, frame.vs, frame.ent, getattr(frame, "c2", "none"), getattr(frame, "ref", "none")), "This is an electric car, so it has no fuel; " + to[1] + "."


BATTERY_TO_FUEL = {"battery_soc": ("fuel_level", "its fuel level is the equivalent of a state of charge", ("at", "trend", "extreme", "compare", "stat")),
                   "battery_condition": ("aux_battery_condition", "the only battery it has is the 12 V one", ("health", "now", "stat", "advice")),
                   "battery_soh": ("aux_battery_condition", "the only battery it has is the 12 V one, so this is its condition", ("health", "advice"))}


def maybe_ice(res, frame):
    """The other direction: "state of charge" / "battery level" put to a petrol or diesel car is read as the fuel level (the energy left), and the first line says so.  -> (frame, note) or None."""
    to = BATTERY_TO_FUEL.get(frame.concept)
    if to is None or to[0] not in res.onto.concepts or frame.act not in to[2]: return None
    try:
        from . import fuelx as FX
        cp = FX.caps_of(res)
        if not (FX.CAPS.is_combustion(cp) and cp.get("fuel")) or cp.get("battery"): return None
    except Exception: return None
    from .frame import Frame
    return Frame(frame.act, to[0], frame.op, frame.cond, frame.vs, frame.ent, getattr(frame, "c2", "none"), getattr(frame, "ref", "none")), "This vehicle has no traction battery; " + to[1] + "."


import re

_CONSUME = re.compile(r"\b(?:consum\w*|usage|drain\w*|discharg\w*|used|uses|use)\b")
_EFFIC = re.compile(r"\beffici\w*|\bregen\w*|\bratio\b|\bmotor\b")
_START = re.compile(r"\b(?:start(?:ed|ing|s)?|turn(?:ed|ing)? on|switch(?:ed|ing)? on|fire[ds]? up)\b.{0,25}\b(?:engine|car|vehicle|motor|ev)\b|\b(?:engine|car|vehicle|motor|ev)\b.{0,15}\bstart(?:s|ed|ing)?\b|\bhow many times\b.{0,20}\bstart")


def guard(text, frame, onto):
    """Words and frame disagree on a pair the model confuses: a question that says consumption / usage / drain is about the energy used, not the battery-to-wheel efficiency ratio; the frame is corrected to the quantity the words name."""
    if frame.concept == "battery_efficiency" and frame.act in ("stat", "extreme", "compare", "trend") and "wh_per_km" in onto.concepts:
        t = " " + str(text).lower() + " "
        if _CONSUME.search(t) and not _EFFIC.search(t):
            from .frame import Frame
            return Frame(frame.act, "wh_per_km", frame.op, frame.cond, frame.vs, frame.ent, getattr(frame, "c2", "none"), getattr(frame, "ref", "none"))
    return frame


def restate(text, frame, r):
    """The answer in the words of the question: a count of trips asked as "how many times did I start the car" is a count of starts (every drive begins with one)."""
    if not (isinstance(r, dict) and isinstance(r.get("answer"), str)) or frame.concept != "trip_record" or frame.op != "count": return r
    if not _START.search(" " + str(text).lower() + " "): return r
    m = re.match(r"There (?:was|were) (\d+) trips? (.+?)\.(.*)$", r["answer"], re.S)
    if not m: return r
    n = int(m.group(1)); ans = f"You started the car {n} time{'' if n == 1 else 's'} {m.group(2)} (each drive begins with one start).{m.group(3)}"
    return dict(r, answer=ans, brief=re.sub(r"^There (?:was|were) \d+ trips? (.+?)\.?$", f"You started the car {n} time{'' if n == 1 else 's'} \1.", str(r.get("brief") or ""), flags=re.S) or ans)

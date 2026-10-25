"""The universal signal catalogue (catalog.json) as code: what the registry is told at start-up, and how an unknown signal name is recognised.

    install(CONCEPTS, C)       adds the quantities the shared registry lacked (engine oil temperature, DPF soot load, throttle position, charging current ...)
    match(name, unit, stats)   -> [(concept, score, why)] best first: the quantity a signal called 'EngCoolTmp_C', 'Vehicle.Powertrain.CombustionEngine.ECT', 'Coolant temp (degC)' or 'v_ego' most likely is
    by_obd_pid / by_j1939      the exact decoding of a standard parameter

The matcher is a transparent scorer, not a model: names are split (camelCase, underscores, digits), abbreviations expanded (tmp -> temperature, spd -> speed, trq -> torque ...), compared with every phrase the
catalogue knows for a quantity (label, synonyms, names seen in DBC / CSV / JSON files, COVESA VSS leaf names, Android VHAL property names) by weighted token overlap, and then decided by what the name cannot
lie about: the unit (a pressure is not a temperature), the wheel position, and the range the values actually take (12 V against 400 V).  Every answer carries the reasons, so a mapping can be reviewed."""
import json
import os
import re

from . import units as U

HERE = os.path.dirname(os.path.abspath(__file__))
_CAT = None
STOP = {"android", "device", "phone", "on", "from", "the", "of", "and", "at", "in", "is", "raw", "sig", "signal", "value", "val", "meas", "measured", "sensor", "actual", "real", "data", "info", "msg", "message", "can", "id", "no", "num", "number"}
ABBR = {"spd": ["speed"], "vel": ["velocity"], "temp": ["temperature"], "tmp": ["temperature"], "tmpr": ["temperature"], "temperatur": ["temperature"], "press": ["pressure"], "pres": ["pressure"], "prs": ["pressure"], "psi": ["pressure"],
        "volt": ["voltage"], "volts": ["voltage"], "vol": ["voltage"], "volage": ["voltage"], "curr": ["current"], "crnt": ["current"], "amp": ["current"], "amps": ["current"], "trq": ["torque"], "torq": ["torque"], "eng": ["engine"],
        "engn": ["engine"], "veh": ["vehicle"], "vehic": ["vehicle"], "batt": ["battery"], "bat": ["battery"], "btry": ["battery"], "accel": ["acceleration"], "acc": ["acceleration"], "lng": ["longitudinal"], "lon": ["longitudinal"],
        "long": ["longitudinal"], "lat": ["lateral"], "pos": ["position"], "posn": ["position"], "pct": ["percent"], "perc": ["percent"], "cltnt": ["coolant"], "clnt": ["coolant"], "clt": ["coolant"], "ect": ["engine", "coolant", "temperature"],
        "eot": ["engine", "oil", "temperature"], "eop": ["engine", "oil", "pressure"], "iat": ["intake", "air", "temperature"], "maf": ["mass", "air", "flow"], "map": ["manifold", "pressure"], "tps": ["throttle", "position"],
        "rpm": ["rpm"], "revs": ["rpm"], "fl": ["front", "left"], "fr": ["front", "right"], "rl": ["rear", "left"], "rr": ["rear", "right"], "lf": ["front", "left"], "rf": ["front", "right"], "lr": ["rear", "left"], "vl": ["front", "left"],
        "vr": ["front", "right"], "hl": ["rear", "left"], "hr": ["rear", "right"], "whl": ["wheel"], "tire": ["tyre"], "tires": ["tyre"], "tyres": ["tyre"], "stft": ["short", "term", "fuel", "trim"], "ltft": ["long", "term", "fuel", "trim"],
        "soc": ["state", "charge"], "soh": ["state", "health"], "hv": ["traction", "battery"], "bms": ["battery"], "ev": ["electric"], "dc": ["dc"], "ac": ["ac"], "alt": ["altitude"], "dist": ["distance"], "odo": ["odometer"],
        "lvl": ["level"], "lev": ["level"], "ang": ["angle"], "angl": ["angle"], "rate": ["rate"], "sw": ["steering", "wheel"], "swa": ["steering", "wheel", "angle"], "brk": ["brake"], "bpp": ["brake", "pedal", "position"], "apps": ["accelerator", "pedal"],
        "kmh": ["kmh"], "kph": ["kmh"], "mph": ["mph"], "ms": ["ms"], "degc": ["degc"], "egt": ["exhaust", "gas", "temperature"], "egr": ["egr"], "dpf": ["dpf"], "def": ["def"], "adblue": ["adblue"], "dte": ["distance", "empty"],
        "mil": ["mil"], "hvac": ["hvac"], "ptc": ["heater"], "trans": ["transmission"], "tcu": ["transmission"], "ign": ["ignition"], "inj": ["injection"], "fuelr": ["fuel", "rate"], "aux": ["auxiliary"], "ext": ["external"],
        "amb": ["ambient"], "tyr": ["tyre"], "cool": ["coolant"], "thr": ["throttle"], "thrtl": ["throttle"], "barometer": ["barometric"], "hour": ["hour"], "outside": ["outside"], "oat": ["outside", "air", "temperature"], "sog": ["speed", "over", "ground"], "cog": ["course", "over", "ground"], "gnss": ["gps"], "gps": ["gps"], "chg": ["charging"],
        "charger": ["charging"], "regen": ["regeneration"], "recuperation": ["regeneration"], "recup": ["regeneration"], "gen": ["generator"], "pwr": ["power"], "kw": ["power"], "freq": ["frequency"], "avg": ["average"], "mean": ["average"],
        "max": ["maximum"], "min": ["minimum"], "hi": ["high"], "lo": ["low"], "bl": ["lambda"], "afr": ["air", "fuel", "ratio"], "ego": ["ego"], "kl": ["terminal"], "b1": [], "b2": [], "s1": [], "s2": []}
GENERIC = {"temperature", "pressure", "speed", "voltage", "power", "current", "level", "rate", "state", "position", "angle", "percent", "torque", "flow", "time", "distance", "status", "engine", "vehicle", "battery"}
QUANTITY_WORDS = GENERIC - {"engine", "vehicle", "battery"}                                   # what is measured; the words that say WHAT it is measured on (engine, battery, motor) decide ties
POSITIONS = {"front": "f", "rear": "r", "left": "l", "right": "rt"}
CLASS_OF = {"speed": "speed", "temperature": "temperature", "pressure": "pressure", "voltage": "voltage", "current": "current", "power": "power", "energy": "energy", "distance": "distance", "rpm": "rpm", "torque": "torque", "angle": "angle",
            "angular_rate": "angular_rate", "acceleration": "acceleration", "flow": "flow", "fuel_flow": "fuel_flow", "mass_flow": "mass_flow", "percent": "percent", "time": "time", "illuminance": "illuminance"}


def load():
    global _CAT
    if _CAT is None: _CAT = json.load(open(os.path.join(HERE, "catalog.json"), encoding="utf-8"))
    return _CAT


def install(concepts, C):
    """Add the catalogue's new quantities to the registry's `concepts` dict (name -> Concept) with the registry's own constructor `C`; existing names are never replaced."""
    for d in load()["concepts"]:
        if d["name"] in concepts: continue
        concepts[d["name"]] = C(d["name"], d["label"], d["unit"], d.get("category", "vehicle"), d.get("synonyms", ()), d.get("lo"), d.get("hi"), "categorical" if d["unit"] == "category" else "numeric", "measured", d.get("description", ""))


# ------------------------------------------------------------------------------------------------------------------ tokens
def tokens(name):
    s = str(name)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s); s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s); s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s); s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower(); out = []
    for t in s.split():
        if t in STOP or (t.isdigit() and len(t) > 1 and t not in ("12", "48", "24")): continue
        out += ABBR.get(t, [t])
    return out


def _wset(toks):
    return {t: (0.55 if t in GENERIC else 1.0) for t in toks if t}


def _dice(a, b):
    if not a or not b: return 0.0
    inter = sum(min(a[t], b[t]) for t in a if t in b); return 2.0 * inter / (sum(a.values()) + sum(b.values()))


def _pos(toks): return {POSITIONS[t] for t in toks if t in POSITIONS}


# ------------------------------------------------------------------------------------------------------------------ the index
_IDX = None


def _vss_leaf(path): return re.sub(r"([a-z])([A-Z])", r"\1 \2", path.split(".")[-1]) if path else ""


def index():
    global _IDX
    if _IDX is not None: return _IDX
    from .. import signal_registry as R
    cat = load()["map"]; idx = {}
    for name, c in R.CONCEPTS.items():
        if c.kind == "derived": continue
        phrases = {c.label, name.replace("_", " ")} | set(c.synonyms); m = cat.get(name, {})
        phrases |= set(m.get("names", ())); phrases |= {" ".join(p.split(".")[-3:]) for p in m.get("vss", ())}; phrases |= {p.replace("_", " ") for p in m.get("vhal", ())}
        idx[name] = {"class": CLASS_OF.get(U.OF_REGISTRY.get(c.unit, ""), None), "unit": c.unit, "lo": c.lo, "hi": c.hi, "phrases": [(p, _wset(tokens(p))) for p in phrases if tokens(p)], "category": c.category,
                     "positional": any(x in name for x in ("_fl", "_fr", "_rl", "_rr")), "dtype": c.dtype}
    _IDX = idx; return idx


def reset():
    global _IDX, _CAT
    _IDX = None; _CAT = None


# ------------------------------------------------------------------------------------------------------------------ matching
def match(name, unit=None, stats=None, top=5, floor=0.45):
    """-> [(concept, score 0..1, why)].  `unit`: the unit the source declared (or None); `stats`: {'min','max','mean'} of the values in that unit; both optional, both decisive when given."""
    base, found = U.split_header(name); unit = unit or found; q_toks = tokens(base); q = _wset(q_toks); qpos = _pos(q_toks); qcls = (U.find(unit) or (None,))[0] if unit else None
    if not q: return []
    conv = None
    if stats is not None and unit and U.find(unit):
        k, o = U.find(unit)[1:]; conv = {kk: stats[kk] * k + o for kk in ("min", "max", "mean") if stats.get(kk) is not None}
    elif stats is not None: conv = {kk: stats[kk] for kk in ("min", "max", "mean") if stats.get(kk) is not None}
    out = []
    for cid, e in index().items():
        best, why = 0.0, ""
        for phrase, w in e["phrases"]:
            s = _dice(q, w); pp = _pos(phrase.split())
            if s >= 0.99: s = 1.0
            p_toks = set(tokens(phrase)); pp = {POSITIONS[t] for t in p_toks if t in POSITIONS}
            if qpos and pp and qpos != pp: s *= 0.15
            elif e["positional"] and not qpos: s *= 0.55
            elif qpos and not pp and not e["positional"] and e["category"] != "chassis": s *= 0.8
            if s > best: best, why = s, phrase
        if best < 0.2: continue
        reasons = [f"name ~ '{why}' ({best:.2f})"]
        if qcls is not None and e["class"] is not None:
            if qcls == e["class"]: best = min(1.3, best + 0.10); reasons.append(f"unit {unit} is a {qcls}")
            elif {qcls, e["class"]} <= {"flow", "fuel_flow", "mass_flow"}: pass
            else: best *= 0.25; reasons.append(f"unit {unit} is a {qcls}, not a {e['class']}")
        elif qcls is not None and e["dtype"] == "categorical": best *= 0.5
        if conv and e["lo"] is not None and e["hi"] is not None and e["class"] is not None and conv.get("min") is not None and conv.get("max") is not None:
            lo, hi = e["lo"], e["hi"]; span = max(hi - lo, 1e-9); mn, mx = conv["min"], conv["max"]
            if mn >= lo - 0.1 * span and mx <= hi + 0.1 * span:
                tight = (mx - mn) / span; pos = (conv["mean"] - lo) / span if conv.get("mean") is not None else 0.5
                if tight < 0.9: best = min(1.3, best + 0.04); reasons.append("values fit the usual range")
                if 0.08 <= pos <= 0.92: best = min(1.3, best + 0.06); reasons.append("values sit well inside it")
            elif mx < lo or mn > hi or (mx - mn) > 3 * span: best *= 0.35; reasons.append("values are outside the usual range")
            else: best *= 0.8; reasons.append("values partly outside the usual range")
        out.append((cid, round(best, 3), "; ".join(reasons)))
    out.sort(key=lambda x: -x[1])
    return [x for x in out[:top] if x[1] >= floor]


def label_overlap(name, concept):
    """How many of the specific words of a signal name appear in the concept's own label or id: the tie-breaker between 'engine speed' -> engine_rpm and motor_rpm, 'throttle position' -> throttle_position and accelerator_position."""
    from .. import signal_registry as R
    c = R.CONCEPTS.get(concept)
    if c is None: return 0
    have = set(tokens(c.label)) | set(tokens(concept.replace("_", " "))); base, _ = U.split_header(name)
    return sum(1 for t in tokens(base) if t not in QUANTITY_WORDS and t in have)


# ------------------------------------------------------------------------------------------------------------------ the exact standards
def obd_table():
    """pid (int) -> (concept, bytes, scale, offset) for every concept the catalogue carries over OBD-II service 01."""
    out = {}
    for cid, m in load()["map"].items():
        o = m.get("obd")
        if o: out[int(o["pid"], 16)] = (cid, int(o["bytes"]), float(o["scale"]), float(o["offset"]))
    return out


def j1939_table():
    """pgn -> [(concept, spn, start byte (1-based), length, scale, offset)]."""
    out = {}
    for cid, m in load()["map"].items():
        for j in m.get("j1939", ()): out.setdefault(int(j["pgn"]), []).append((cid, int(j["spn"]), int(j["start"]), int(j["len"]), float(j["scale"]), float(j["offset"])))
    return out


def vss_table():
    out = {}
    for cid, m in load()["map"].items():
        for p in m.get("vss", ()): out[p.lower()] = cid
    return out


def vhal_names():
    out = {}
    for cid, m in load()["map"].items():
        for p in m.get("vhal", ()): out[p] = cid
    return out

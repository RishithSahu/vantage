"""The signal mapper: which quantity is this signal?  One engine for every place a vehicle describes its own signals (a DBC file, a table header, a VHAL property, a SOME/IP field).

    decide(Evidence(name, unit, bits, factor, offset, lo, hi, choices, message, siblings, ice, stats)) -> Decision(concept or None, confidence, reasons, alternatives, transform)
    map_database(db, ice=None)  -> [(message, signal, Decision)] for every signal of a DBC, with duplicates resolved

It is built to refuse rather than to guess.  What it looks at, in order:
  1. the name is cut into words (camelCase, underscores, digits, glued words like WHEELSPEED), wheel positions are read as positions (FL, FrL, ReR, BL, front-left ...) before abbreviations can turn them into
     something else, and abbreviations are expanded (spd -> speed, trq -> torque ...);
  2. a signal that is not itself a measurement is refused: counters and checksums, requests / commands / targets / set-points, validity and fault flags, warning lamps, configuration;
  3. the SHAPE of the signal decides what kind of quantity it can be: a single bit or a short value table is a state (door open, belt fastened ...), never a speed or a temperature; a wide field is a measurement;
  4. the unit, when it is known, must be the quantity's unit class (a pressure is not a temperature); 'deg' is an angle or a temperature and the name decides;
  5. the range the bits can hold must be able to hold the quantity (a 5-bit field with a scale of 1 is not an engine speed);
  6. wheel positions must agree, and a position-less name does not become one wheel;
  7. the message name ('ENGINE_1', 'STEERING', 'Charger_stats') says which subsystem a plain name ('RPM', 'DRIVER_TORQUE', 'HV_Voltage') belongs to, and the powertrain says engine or motor and which battery;
  8. a bare name ('VS', 'N') is read through its unit and message only for the few classes where nothing else can be meant; an abbreviation nobody can read is refused;
  9. a state is mapped only when its polarity is explicit (open / closed, latched / unlatched, applied / released) in the name or in the value table.
Every answer carries its reasons, so a mapping can be reviewed and a refusal explained.  Measured by bench/mapper_benchmark.py."""
import re
from dataclasses import dataclass, field

from .. import signal_registry as R
from . import catalog as K, units as U

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------- vocabulary
COUNTER = {"counter", "cnt", "count", "cntr", "alive", "aliv", "rolling", "roll", "seq", "sequence", "checksum", "chksum", "cksm", "crc", "parity", "chk", "msgcnt", "bz", "heartbeat", "timestamp", "stamp", "toggle", "sync", "rc",
           "encoder", "accumulator", "accumulat", "pulse", "tick"}
COMMAND = {"req", "request", "requested", "cmd", "command", "tgt", "tar", "target", "targeted", "set", "setpoint", "setting", "setpt", "demand", "demanded", "dem", "desired", "wish", "wished", "limit", "limited", "lim", "threshold",
           "nominal", "allowed", "allow", "permitted", "avail", "available", "enable", "enabled", "enabling", "disable", "disabled", "override", "trigger", "init", "initial", "calib", "calibration", "bias", "reserved", "spare",
           "unused", "preset", "reference", "ref", "feedforward", "ctrl", "control", "controller", "inhibit", "suppress", "prevent", "stop", "release", "button", "btn", "push", "key", "lock", "unlock", "ratio", "next",
           "trgt", "trg", "rq", "reservd", "rsvd", "rsv", "resv", "lkas", "lka", "ldw", "pilot", "autopilot", "relative", "rel", "adaptive", "stnd", "standard", "buttn", "bttn", "handle", "stalk", "knob"}
SWITCHES = {"switch", "sw"}
FOREIGN = {"cruise", "radar", "fan", "pump", "blower", "compressor", "mirror", "window", "starter", "alternator", "generator", "heater", "defroster", "washer", "wash", "seat", "sunroof", "trunk", "hood"}         # another component's own quantity                                                  # a command for most things, the state itself for wipers and lamps
VALIDITY = {"valid", "validity", "vld", "invalid", "qual", "quality", "qbit", "qf", "fault", "flt", "err", "error", "errors", "diag", "diagnostic", "diagnosis", "plaus", "plausible", "implausible", "failure", "fail", "failed",
            "defect", "defective", "sna", "unavailable", "unavail", "missing", "mil", "ub", "q"}
WARN = {"warn", "warning", "wrn", "alert", "indicator", "indication", "ind", "msg", "message", "text", "display", "disp", "hmi", "icon", "symbol", "led", "buzzer", "chime", "reminder", "notification"}
CONFIG = {"unit", "units", "uom", "language", "lang", "variant", "config", "configuration", "option", "opt", "type", "version", "ver", "code", "id", "index", "idx", "format", "scaling", "scale", "gain", "resolution", "mapping",
          "coding", "country", "market", "region", "vin", "serial", "part", "software", "hardware", "revision", "no", "num", "number", "immo", "pats", "immobilizer"}
PEAK = {"max", "maximum", "min", "minimum", "peak", "sum", "total", "integral", "delta", "diff", "difference", "gradient", "grad", "filtered", "filter", "smoothed"}
STOP = (K.STOP - {"id", "no", "num", "number"}) | {"signal", "sig", "ecu", "pdu", "frame", "gway", "hsc", "lsc", "msc", "cf", "cr", "cl", "can", "clu", "ems", "ecm", "tcu", "esp", "abs", "dsc", "bcm", "gw", "fatc", "datc", "mdps"}
SIDE = {"l": "left", "left": "left", "lh": "left", "lt": "left", "r": "right", "right": "right", "rh": "right", "rt": "right"}
AXLE = {"front": "front", "frnt": "front", "frt": "front", "fr": "front", "re": "rear", "rear": "rear", "bk": "rear", "back": "rear", "rr": "rear"}
COMBINED = {"fl": ("front", "left"), "fr": ("front", "right"), "rl": ("rear", "left"), "rr": ("rear", "right"), "bl": ("rear", "left"), "br": ("rear", "right"), "lf": ("front", "left"), "rf": ("front", "right"), "lr": ("rear", "left"),
            "vl": ("front", "left"), "vr": ("front", "right"), "hl": ("rear", "left"), "hr": ("rear", "right")}
POSITION_ABBR = set(COMBINED)                                           # K.ABBR maps some of these to other words (bl -> lambda ...); positions are read first, so they are dropped from the expansion table
ABBR = {k: v for k, v in K.ABBR.items() if k not in POSITION_ABBR and k not in ("sw", "swa", "acc", "ext", "gen", "kl", "bl")}
ABBR.update({"veh": ["vehicle"], "int": ["intake"], "incar": ["cabin"], "drv": ["driver"], "drvr": ["driver"], "psgr": ["passenger"], "pass": ["passenger"], "ign": ["ignition"], "igk": ["ignition"], "tq": ["torque"], "eps": ["steering"],
             "sas": ["steering", "angle"], "mdps": ["steering"], "pdl": ["pedal"], "brk": ["brake"], "epb": ["parking", "brake"], "hbrake": ["hand", "brake"], "handbrake": ["hand", "brake"], "park": ["parking"], "odo": ["odometer"],
             "odom": ["odometer"], "lvl": ["level"], "soc": ["state", "charge"], "baro": ["barometric"], "cool": ["coolant"], "clnt": ["coolant"], "wtr": ["water"], "sbelt": ["seat", "belt"], "wpr": ["wiper"], "hdlp": ["head", "lamp"],
             "hlmp": ["head", "lamp"], "pbrake": ["parking", "brake"], "pkb": ["parking", "brake"], "pbrk": ["parking", "brake"], "appd": ["applied"], "prs": ["pressure"], "seatbelt": ["seat", "belt"], "estd": ["estimated"], "actl": ["actual"], "frz": ["freeze"], "frm": ["frame"],
             "actu": ["actuator"], "sts": ["status"], "stat": ["status"], "calc": ["calculated"], "vbat": ["battery", "voltage"], "vbatt": ["battery", "voltage"], "lv": ["auxiliary"], "wss": ["wheel", "speed"], "tpms": ["tyre"], "obj": ["object"], "dist": ["distance"], "vspd": ["vehicle", "speed"], "vehspd": ["vehicle", "speed"], "vel": ["velocity"], "tmp": ["temperature"], "temp": ["temperature"], "tire": ["tyre"], "tires": ["tyre"]})
ABBR.pop("hl", None)
for _k in ("rh", "lh", "alt", "clt"): ABBR.pop(_k, None)
ABBR.update({"steer": ["steering"], "strg": ["steering"], "clt": ["coolant", "temperature"], "tqi": ["torque"]})
BIGRAM = {("acc", "pedal"): ["accelerator", "pedal"], ("gas", "pedal"): ["accelerator", "pedal"], ("gas", "position"): ["accelerator", "position"], ("acc", "gas"): ["accelerator"], ("accel", "pedal"): ["accelerator", "pedal"], ("accel", "pos"): ["accelerator", "position"],
          ("accel", "actu"): ["accelerator", "position"], ("int", "air"): ["intake", "air"], ("out", "temperature"): ["outside", "temperature"], ("out", "temp"): ["outside", "temperature"], ("in", "car"): ["cabin"],
          ("eng", "speed"): ["engine", "rpm"], ("engine", "speed"): ["engine", "rpm"], ("motor", "speed"): ["motor", "rpm"], ("hv", "voltage"): ["traction", "battery", "voltage"], ("long", "a"): ["longitudinal", "acceleration"], ("yaw", "w"): ["yaw", "rate"], ("lat", "a"): ["lateral", "acceleration"], ("vert", "a"): ["vertical", "acceleration"], ("of", "change"): [], ("eng", "spd"): ["engine", "rpm"], ("engine", "spd"): ["engine", "rpm"], ("motor", "spd"): ["motor", "rpm"], ("gps", "alt"): ["gps", "altitude"], ("gps", "altitude"): ["gps", "altitude"]}
# phrases for quantities whose usual names the registry's own synonym lists do not cover (they were read off real databases, not off the benchmark's expected answers)
EXTRA = {"lead_vehicle_distance": ["lead distance", "lead long dist", "lead longitudinal distance", "object distance", "acc object distance", "front vehicle distance", "preceding vehicle distance", "distance to lead vehicle"],
         "steering_rate": ["steering angle speed", "steering angle rate", "steering speed", "steering wheel speed", "steering wheel rate", "steering rate"],
         "steering_angle": ["steering position", "steering wheel position", "steering wheel angle", "steer angle", "steering angle", "steering wheel angle sensor"],
         "steering_torque": ["driver torque", "steering driver torque", "eps torque", "steering column torque", "torsion bar torque", "steering torque", "driver steering torque"],
         "accelerator_position": ["accelerator pedal", "pedal position", "accelerator position", "accelerator actuator position", "accelerator pedal position"],
         "brake_pressure": ["brake pedal pressure", "driver brake pressure", "brake pressure", "master cylinder pressure", "brake pedal driver applied pressure"],
         "speed": ["vehicle speed", "wheel based speed", "speed", "velocity", "vehicle velocity", "ego speed"],
         "ambient_temperature": ["outside temperature", "outside air temperature", "external temperature", "ambient temperature", "outdoor temperature"],
         "cabin_temperature": ["cabin temperature", "interior temperature", "incar temperature", "passenger compartment temperature", "inside temperature"],
         "odometer": ["odometer", "total distance", "mileage"], "engine_oil_temperature": ["oil temperature", "engine oil temperature"], "intake_air_temperature": ["intake air temperature"], "fuel_temperature": ["fuel temperature"],
         "transmission_temperature": ["transmission temperature", "gearbox temperature", "gearbox oil temperature", "transmission oil temperature", "transmission sump temperature"],
         "motor_temperature": ["coolant temperature", "engine coolant temperature", "engine temperature", "water temperature", "engine water temperature"],
         "aux_battery_voltage": ["12v battery voltage", "supply voltage", "system voltage", "power supply voltage", "battery voltage 12 v", "auxiliary battery voltage"],
         "charging_voltage": ["charger voltage", "charging voltage", "charger hv voltage", "charge voltage"], "charging_current": ["charger current", "charging current", "charger hv current", "charge current"],
         "battery_voltage": ["pack voltage", "battery pack voltage", "hv battery voltage", "traction battery voltage"], "engine_load": ["engine load", "calculated load", "load value"],
         "engine_torque": ["engine torque", "estimated torque", "engine estimated torque"], "reported_range": ["remaining range", "range remaining", "range"],
         "wheel_speed": ["wheel speed", "wheel speed clean"], "altitude": ["altitude", "gps altitude", "elevation"]}
DOMAIN_WORDS = {"engine", "motor", "vehicle", "wheel", "speed", "steering", "steer", "brake", "pedal", "accelerator", "throttle", "gear", "transmission", "coolant", "oil", "fuel", "battery", "cell", "charger", "charging", "tyre",
                "tire", "tpms", "door", "belt", "seat", "wiper", "lamp", "light", "head", "park", "parking", "hand", "odometer", "ambient", "outside", "cabin", "intake", "air", "exhaust", "boost", "lambda", "torque", "angle", "rate",
                "pressure", "temperature", "voltage", "current", "power", "level", "position", "yaw", "lateral", "longitudinal", "acceleration", "distance", "range", "state", "charge", "mass", "flow", "manifold", "absolute", "relative",
                "driver", "passenger", "front", "rear", "left", "right", "water", "rpm", "ignition", "hybrid", "high", "low", "beam", "rain", "ac", "heater", "fan", "pump", "inverter", "stator", "rotor", "regeneration", "object", "lead"}
NEUTRAL = {"servo", "actual", "act", "real", "value", "meas", "measured", "raw", "data", "info", "sensor", "signal", "main", "primary", "left", "right", "front", "rear", "avg", "average", "mean", "appd", "applied", "driver", "estimated",
           "actuator", "passenger", "high", "low", "status", "state", "pos", "any"}
GENERIC = K.GENERIC | {"state", "charge", "position", "level"}
BAD_PHRASES = {"heading": [("yaw", "angle")]}                  # a phrase the catalog lists for a quantity but that, read in a vehicle database, means something else (a yaw angle is relative to the car)
PHRASE_DROP = {"actual", "act", "real", "value", "meas", "measured", "raw", "data", "info", "sensor", "signal", "main", "primary", "avg", "average", "mean"}       # words that say nothing, in a phrase
QTY = K.QUANTITY_WORDS | {"state", "charge", "position", "level"}                  # words that say WHAT is measured; engine / battery / vehicle say what it is measured ON


def _singular(t): return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and t not in ("lambda", "gps", "bus", "gas", "pass", "ems") else t


def pieces(name):
    s = str(name)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s); s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s); s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s); s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    return re.sub(r"[^A-Za-z0-9]+", " ", s).lower().split()


_VOCAB = _PHR = _IDENT = None


def vocab():
    global _VOCAB
    if _VOCAB is None:
        v = set(DOMAIN_WORDS) | COUNTER | COMMAND | SWITCHES | VALIDITY | WARN | CONFIG | PEAK | set(ABBR) | set(SIDE) | set(AXLE) | set(COMBINED)
        for e in K.index().values():
            for phrase, w in e["phrases"]: v |= set(w)
        for a in ABBR.values(): v |= set(a)
        v |= NEUTRAL
        v |= OPEN | CLOSED | FASTENED | UNFASTENED | APPLIED | RELEASED | {"pressed", "depressed", "actuated", "inverted", "inv", "not", "and", "or", "raw", "servo", "accumulator", "encoder", "adaptive", "relative"}
        for cid, c in R.CONCEPTS.items():
            for phrase in [c.label, cid.replace("_", " ")] + list(c.synonyms) + EXTRA.get(cid, []): v |= set(expand(pieces(phrase)))
        _VOCAB = {x for x in v if x}
    return _VOCAB


def reset():
    global _VOCAB, _PHR, _IDENT
    _VOCAB = _PHR = _IDENT = None


def segment(piece):
    """'wheelspeed' -> ['wheel', 'speed'] when every part is a word the mapper knows (parts of at least 3 letters); other pieces are returned as they are."""
    voc = vocab()
    if piece in voc or piece.isdigit() or len(piece) < 6: return [piece]
    best = {0: []}
    for i in range(1, len(piece) + 1):
        for j in range(max(0, i - 12), i):
            w = piece[j:i]
            if j in best and len(w) >= 3 and (w in voc or _singular(w) in voc):
                cand = best[j] + [w]
                if i not in best or len(cand) < len(best[i]): best[i] = cand
    return best[len(piece)] if len(piece) in best and len(best[len(piece)]) <= 3 else [piece]


def read_position(ps):
    """-> (axle, side, remaining pieces).  'fl' / 'fr' / 'rl' / 'rr' / 'bl' / 'br', or front / rear followed by a side, or a lone front / rear / left / right word."""
    out = []; axle = side = None; i = 0
    while i < len(ps):
        p = ps[i]; nxt = ps[i + 1] if i + 1 < len(ps) else None
        if p in AXLE and nxt in SIDE and axle is None: axle, side = AXLE[p], SIDE[nxt]; i += 2; continue
        if p in COMBINED and axle is None: axle, side = COMBINED[p]; i += 1; continue
        if p in ("front", "rear", "back") and axle is None: axle = AXLE[p]; i += 1; continue
        if p in ("left", "right", "lh", "rh") and side is None: side = SIDE[p]; i += 1; continue
        if p in ("l", "r") and side is None and i == len(ps) - 1 and i > 0: side = SIDE[p]; i += 1; continue                # SPEED_L: the left one
        out.append(p); i += 1
    return axle, side, out


SENSOR_ACRONYMS = {"ect", "eot", "eop", "iat", "maf", "map", "tps", "egt", "stft", "ltft"}               # a sensor's name, not a subsystem: left alone in a message name
CTX_BLOCK = {"request", "req", "rq", "cmd", "command", "setting", "config", "configuration", "calibration", "coding", "control", "frz", "freeze", "snapshot", "frozen"}


def expand(ps, sensor_acronyms=True):
    """Pieces -> content words: bigram rules, abbreviations, singular forms, stop words dropped."""
    out = []; i = 0
    while i < len(ps):
        p = ps[i]; nxt = ps[i + 1] if i + 1 < len(ps) else None
        if (p, nxt) in BIGRAM: out += BIGRAM[(p, nxt)]; i += 2; continue
        if p in STOP and p not in ABBR: i += 1; continue
        for t in (ABBR.get(p, [p]) if (sensor_acronyms or p not in SENSOR_ACRONYMS) else [p]): out.append(_singular(t))
        i += 1
    seen = set(); res = []
    for t in out:
        if t and t not in STOP and t not in seen: seen.add(t); res.append(t)
    return res


_UNIT_WORDS = {"percent": "%", "percen": "%", "degrecelsi": "C", "degreecelsius": "C", "kilometerperhour": "km/h", "kilometerperhou": "km/h", "kilometersperhour": "km/h", "meterpersecon": "m/s", "meterperseconsquar": "m/s2",
               "meterperseconsquare": "m/s2", "bar": "bar", "millibar": "mbar", "volt": "V", "amper": "A", "ampere": "A", "kilowatt": "kW", "watt": "W", "newtometer": "Nm", "minutinver": "rpm", "degreofarcpersecon": "deg/s",
               "degreofarc": "deg", "meter": "m", "kilometer": "km", "forceofgravi": "g", "second": "s", "secon": "s", "cel": "C", "grad/s": "deg/s", "%indication": "%", "khp": "kph", "kilopascal": "kPa", "kilopascals": "kPa", "pascal": "Pa", "hectopascal": "hPa"}
NOISE_UNITS = {"unitless", "none", "sed", "sticky", "encrypted", "yes/no", "bool", "boolean", "enum", "flag", "count", "counts", "bit", "bits", "number", "-", "n/a", "na", "state", "status", "raw", "sna", "nounit", "note"}   # a unit field that holds no unit


UNIT_SUFFIX = {"mph": "mph", "kph": "kph", "kmh": "km/h", "dps": "deg/s", "mps": "m/s", "kpa": "kPa", "psi": "psi", "bar": "bar", "hpa": "hPa", "kw": "kW", "kwh": "kWh", "nm": "Nm", "rpm": "rpm", "pct": "%", "percent": "%", "mi": "mi",
               "km": "km", "ms2": "m/s2", "degc": "C", "degf": "F", "deg": "deg", "lpm": "L/min", "lph": "L/h"}
ACCEL_WORDS = {"lat", "long", "vert", "accel", "acc", "lateral", "longitudinal", "acceleration"}


def _name_unit(name):
    """-> (unit, the name's piece that wrote it): a unit written at the end of the name (AmbientTemp_F, Speed_mph, YawRate_dps)."""
    ps = pieces(name)
    if len(ps) < 2: return None, None
    last, rest = ps[-1], set(ps[:-1])
    if last in ("f", "c") and rest & {"temp", "temperature", "tmp"}: return last.upper(), last
    if last == "g" and rest & ACCEL_WORDS: return "g", last
    if last == "w" and rest & {"pwr", "power"}: return "W", last
    if last == "v" and rest & {"volt", "voltage", "vbat"}: return "V", last
    if last == "a" and rest & {"cur", "curr", "current", "amp"}: return "A", last
    return (UNIT_SUFFIX[last], last) if last in UNIT_SUFFIX else (None, None)


def canon_unit(unit):
    """A unit spelling the unit table understands, or None: 'Unit_PerCent' -> '%', 'Unit_DegreCelsi' -> 'C', 'R/min' -> 'rpm'."""
    if not unit: return None
    if U.find(unit): return unit
    u = re.sub(r"[^a-z/%0-9]", "", str(unit).lower().replace("unit_", ""))
    if u in _UNIT_WORDS: return _UNIT_WORDS[u]
    return "rpm" if u in ("r/min", "1/min", "min-1", "rev/min", "revs/min") else None


@dataclass
class Evidence:
    name: str
    unit: str = None
    bits: int = None
    factor: float = 1.0
    offset: float = 0.0
    signed: bool = False
    lo: float = None                      # the range the database DECLARES (advisory)
    hi: float = None
    choices: list = field(default_factory=list)     # labels of the value table
    message: str = ""
    siblings: list = field(default_factory=list)    # names of the other signals in the same message
    dlc: int = None
    ice: bool = None                      # True: a combustion powertrain (no traction battery anywhere in the database); False: electric / hybrid; None: unknown
    stats: dict = None                    # {'min','max','mean'} of observed values in the signal's own unit, when a capture exists


@dataclass
class Decision:
    concept: str = None
    confidence: float = 0.0
    reasons: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    refused: str = ""
    transform: dict = field(default_factory=dict)


SENTINEL = re.compile(r"\b(sna|snv|invalid|unavailable|not available|n/?a|error|reserved|unknown|undefined|unused|spare|init|not used|fault|failure|no signal|not supported|signal not)\b", re.I)


class Parsed:
    pass


def strip_tags(name, siblings):
    """A tag every signal of a message carries (HSC2, EPAS3S_, EPT) is the database's own, not part of the meaning: a capitals-and-digits tag glued to the end is cut, and the shared leading words are peeled while they
    are words the mapper does not know (stopping at the first one it does)."""
    names = [name] + [x for x in siblings if x != name]
    if len(names) < 2: return name
    voc = vocab()
    def unknown(w): return not w.isdigit() and segment(w) == [w] and w not in voc
    cs = [x[::-1] for x in names]; n = 0
    while all(len(x) > n for x in cs) and len({x[n] for x in cs}) == 1: n += 1
    t = names[0][len(names[0]) - n:] if n else ""; base = name
    if len(t) >= 3 and re.fullmatch(r"[A-Z]{2,}\d*", t) and all(len(x) > len(t) for x in names) and all(unknown(w) for w in pieces(re.sub(r"\d+$", "", t))): base = name[:-len(t)]
    pl = [pieces(x) for x in names]; mine = pieces(base); j = 0
    while j < len(mine) - 1 and all(len(x) > j + 1 and x[j] == mine[j] for x in pl) and (unknown(mine[j]) or mine[j].isdigit()): j += 1
    while j > 0 and mine[j - 1].isdigit(): j -= 1
    if j and any(unknown(w) for w in mine[:j]): return _rebuild(base, mine[j:])
    return base or name


def _rebuild(original, words):
    """The name's own text for the words kept (spelling is only read through pieces(), so the words are joined)."""
    return "_".join(words) if words else original


def parse(ev):
    P = Parsed(); ps = []; suffix_unit, suffix = _name_unit(ev.name); pcs = pieces(strip_tags(ev.name, ev.siblings))
    if suffix and pcs and pcs[-1] == suffix and len(pcs) > 1: pcs = pcs[:-1]                                     # the unit is not a word of the name
    for p in pcs:
        if p.isdigit() and p not in ("12", "48", "24"): continue
        ps += segment(p)
    P.axle, P.side, rest = read_position(ps)
    P.raw = [_singular(x) for x in rest]
    P.toks = expand(rest)
    ctx = []
    for p in pieces(ev.message or ""):
        if p.isdigit(): continue
        ctx += segment(p)
    P.ctx_axle, P.ctx_side, ctx = read_position(ctx)
    P.ctx = expand(ctx, sensor_acronyms=False)
    real = [c for c in ev.choices if not SENTINEL.search(str(c))]
    if len(real) >= 2: P.shape = "enum" if len(real) > 2 else "flag"
    elif ev.bits is not None and ev.bits <= 1: P.shape = "flag"
    elif ev.bits is not None and ev.bits <= 2: P.shape = "enum"
    else: P.shape = "analog"
    if P.shape == "analog" and ev.bits is not None and ev.bits <= 4 and not ev.unit: P.shape = "enum"
    P.labels = [str(c) for c in real]
    P.cu = canon_unit(ev.unit); P.name_unit = None
    if not ev.unit and suffix_unit == "deg": P.ambig_deg_name = True
    elif not ev.unit: P.cu = P.name_unit = suffix_unit
    cls = U.find(P.cu) if P.cu else None; P.has_unit = bool(ev.unit)
    P.unit_unknown = bool(ev.unit) and P.cu is None and str(ev.unit).strip().lower() not in NOISE_UNITS
    full = pieces(ev.name); kept = pieces(strip_tags(ev.name, ev.siblings)); msgw = set(pieces(ev.message or ""))
    P.peeled = [w for w in full if w not in kept and not w.isdigit() and w not in msgw and w not in STOP] if full != kept else []              # prefix words that were taken for the database's own tag and are not in the message's name either
    P.sib_ident = any(set(pieces(x)) & {"id", "idx", "index", "num", "number"} for x in ev.siblings)
    P.unit_cls = cls[0] if cls else None; P.unit_raw = (ev.unit or "").strip().lower()
    P.indexed = bool(re.search(r"(wheel|whl|tyre|tire)[\W_]*\d", str(ev.name).lower()))
    P.ctx_block = bool(set(P.ctx) & CTX_BLOCK)
    P.ambig_deg = P.unit_raw in ("deg", "degs", "degree", "degrees") or getattr(P, "ambig_deg_name", False)
    if P.ambig_deg: P.unit_cls = None                                           # a °C written 'deg' (Hyundai) or an angle: the name decides
    P.span = None; P.pmin = P.pmax = None
    if ev.bits and ev.factor:
        raw_hi = (2 ** (ev.bits - 1) - 1 if ev.signed else 2 ** ev.bits - 1); raw_lo = (-(2 ** (ev.bits - 1)) if ev.signed else 0)
        a, b = raw_lo * ev.factor + ev.offset, raw_hi * ev.factor + ev.offset; P.pmin, P.pmax = min(a, b), max(a, b); P.span = P.pmax - P.pmin
    cw = CLASS_ADD.get(P.unit_cls)
    if (ev.unit or P.name_unit) and cw and not P.ambig_deg and not set(P.toks) & (CLASS_WORDS.get(P.unit_cls, {cw}) | QTY | {"rpm", "torque", "power", "current", "voltage"}) and all(t.isdigit() or t in vocab() for t in P.toks): P.toks.append(cw)         # 'Engine' in rpm, 'Coolant' in degC: the unit says which quantity
    P.S = set(P.toks) | {x for x in (P.axle, P.side) if x}
    stem = re.sub(r"[_ ]?(?:[A-Za-z]|\d+)(?:_\d+)*$", "", str(ev.name)); P.family = 0
    if stem and len(stem) >= 3:
        P.family = sum(1 for x in ev.siblings if x.startswith(stem) and re.fullmatch(r"[_ ]?(?:[A-Za-z]|\d+)(?:_\d+)*", x[len(stem):] or "x") is not None)
    return P


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------- the name score
def ident_words():
    """The specific words concepts are named by (engine, wheel, coolant, charging ...): an extra word of this kind in a signal's name points at another quantity."""
    global _IDENT
    if _IDENT is None:
        w = set()
        for cid, c in R.CONCEPTS.items():
            if c.kind == "derived": continue
            w |= {t for t in expand(pieces(c.label)) + expand(pieces(cid.replace("_", " "))) if t not in GENERIC and t not in NEUTRAL}
        _IDENT = w
    return _IDENT


def _phrases():
    global _PHR
    if _PHR is not None: return _PHR
    out = {}
    for cid, e in K.index().items():
        ps = []; seen = set()
        for phrase in [p for p, _w in e["phrases"]] + EXTRA.get(cid, []):
            toks = tuple(t for t in expand(pieces(phrase)) if t not in PHRASE_DROP)
            if toks in BAD_PHRASES.get(cid, ()): continue
            if toks and toks not in seen: seen.add(toks); ps.append((phrase, list(toks)))
        out[cid] = ps
    _PHR = out; return out


def name_score(S, ctx, phrases, unit_ok=False):
    """Best coverage of a concept's phrase by the signal's words, discounted for words the phrase does not have.  Returns (score, phrase)."""
    best, why, wt = 0.0, "", []; voc = vocab(); ident = ident_words()
    for phrase, toks in phrases:
        pw = {t: (0.55 if t in GENERIC else 1.0) for t in toks}
        tot = sum(pw.values()); have = sum(w for t, w in pw.items() if t in S); have_ctx = sum(w for t, w in pw.items() if t not in S and t in ctx)
        if have == 0 and have_ctx == 0: continue
        cov = (have + 0.6 * have_ctx) / tot
        spec = [t for t in pw if t not in QTY]
        if spec and not any(t in S for t in spec) and have_ctx == 0: cov *= 0.5                         # only the generic words of a specific phrase are present
        if not spec and have == 0: continue                                                           # a purely generic phrase needs its words in the signal's own name
        if not spec and not unit_ok and "speed" not in pw and "velocity" not in pw: continue            # ... and the unit must agree: a bare 'torque' with no unit could be any torque
        extra_ident = [t for t in S if t not in pw and t in ident and t not in NEUTRAL]
        extra_known = [t for t in S if t not in pw and t in voc and t not in ident and t not in NEUTRAL]
        extra_unknown = [t for t in S if t not in pw and t not in voc and t not in NEUTRAL]
        s = cov * (0.78 ** len(extra_ident)) * (0.92 ** len(extra_known)) * (0.5 ** len(extra_unknown))
        if s > best: best, why, wt = s, phrase, toks
    return best, why, wt


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------- gates
def _qualifier(P, allowed, numeric):
    """The first word of the name that says it is not the measurement itself (a counter, a request, a validity flag ...), unless the concept's own words include it."""
    for t in P.toks:
        if t in allowed: continue
        if t == "control" and "module" in P.raw: continue                                   # 'Voltage (Control Module)' is the ECU's supply, not a command
        if t in COUNTER or t in COMMAND or t in VALIDITY or t in CONFIG or t in SWITCHES: return t
        if numeric and (t in WARN or t in PEAK or t == "lamp" or t in FOREIGN): return t
    if numeric and "raw" in P.raw and not P.has_unit: return "raw"
    if numeric and set(P.raw) & {"inverted", "inv", "invert", "negated"}: return "inverted"
    return None


def _concept_tokens(cid):
    c = R.CONCEPTS[cid]
    return {t for t in expand(pieces(c.label)) + expand(pieces(cid.replace("_", " "))) + [x for s in list(c.synonyms) + EXTRA.get(cid, []) for x in expand(pieces(s))]}


def _is_positional(cid): return cid[-3:] in ("_fl", "_fr", "_rl", "_rr")


def _pos_of(cid): return {"fl": ("front", "left"), "fr": ("front", "right"), "rl": ("rear", "left"), "rr": ("rear", "right")}[cid[-2:]]


CLASS_ADD = {"rpm": "rpm", "temperature": "temperature", "pressure": "pressure", "voltage": "voltage", "current": "current", "torque": "torque", "power": "power"}
SPEED_BLOCK = {"engine", "motor", "fan", "pump", "blower", "compressor", "shaft", "turbine", "steering", "wind", "input", "output"}              # the speed of something else
MIN_SPAN = {"odometer": 1000, "heading": 300, "altitude": 500}                                                                  # a counter that wraps sooner than this cannot be the total distance
CONTROL_WORDS = {"stalk", "switch", "sw", "lever", "handle", "knob", "button", "btn", "buttn", "selector"}      # the driver's control, not the lamp or wiper itself
NO_SPAN_GATE = {"odometer", "distance_driven", "engine_run_time", "engine_hours", "energy_used", "regen_energy", "time_to_full", "range_estimate", "reported_range", "altitude", "heading", "wh_per_km"}
OPEN, CLOSED = {"open", "opened", "ajar"}, {"closed", "close", "shut"}
FASTENED, UNFASTENED = {"latched", "buckled", "fastened", "belted", "on"}, {"unlatched", "unbuckled", "unfastened", "unbelted", "off", "open", "released"}
APPLIED, RELEASED = {"active", "act", "applied", "engaged", "on", "set", "pulled", "clamped"}, {"released", "off", "inactive", "free"}
CLASS_WORDS = {"temperature": {"temperature"}, "pressure": {"pressure"}, "voltage": {"voltage"}, "current": {"current"}, "speed": {"speed", "velocity"}, "rpm": {"rpm"}, "torque": {"torque"}, "angle": {"angle", "position"},
               "angular_rate": {"rate", "yaw", "speed"}, "acceleration": {"acceleration"}, "power": {"power"}, "distance": {"distance", "odometer", "range", "mileage"}, "mass_flow": {"flow", "maf"}, "fuel_flow": {"rate", "flow"}}
SIGNED_CONCEPTS = {"steering_angle", "steering_rate", "steering_torque", "yaw_rate", "lateral_acceleration", "longitudinal_acceleration", "acceleration", "motor_torque", "battery_current", "motor_power", "battery_power"}
FOREIGN_SUB = {"steering", "brake", "wiper", "window", "wheel", "seat", "mirror", "hvac", "climate"}                     # a message about another subsystem
TWINS = {"motor_rpm": "engine_rpm", "motor_torque": "engine_torque", "engine_rpm": "motor_rpm", "engine_torque": "motor_torque"}
TRACTION_WORDS = {"bms", "hv", "traction", "pack", "cell", "highvoltage"}                      # the name itself says it is the drive battery
TRACTION_ONLY = {"battery_soc", "battery_current", "battery_temperature", "battery_power", "battery_soh", "cell_temperature_max", "cell_temperature_min", "cell_voltage_max", "cell_voltage_min"}


def _polarity(P, positive, negative):
    """+1 when the name says the state is the positive one, -1 when it says the opposite, 0 when it does not say."""
    words = set(P.raw) | set(P.toks); pos = words & positive; neg = words & negative
    if pos and not neg: return +1
    if neg and not pos: return -1
    return 0


def _label_polarity(labels, positive, negative, need_both=True):
    """{label: 0/1} for the labels that can be read (others are left out); None unless both polarities were found."""
    out = {}
    for l in labels:
        w = {_singular(x) for x in pieces(l)}
        if w & {"not", "no"}: out[l] = 0
        elif w & negative and not w & positive: out[l] = 0
        elif w & positive and not w & negative: out[l] = 1
    return out if out and (not need_both or set(out.values()) == {0, 1}) else None


def _flag(P, concept, positive, negative, label, d):
    pol = _polarity(P, positive, negative); m = _label_polarity(P.labels, positive, negative) if P.labels else None
    if m is not None: d.concept, d.confidence, d.transform, d.reasons = concept, 0.8, {"map": m}, [f"{label} value table"]; return d
    if pol:
        d.concept, d.confidence, d.reasons = concept, 0.8, [f"{label} flag, {'positive' if pol > 0 else 'inverted'} polarity read from the name"]
        if pol < 0: d.transform = {"invert01": True}
        return d
    d.refused = f"a {label} signal without a stated polarity"; return d


def _categorical(ev, P):
    """State concepts, each with its own rule (-> Decision or None).  Only the ones whose numeric meaning the registry defines."""
    T = set(P.toks)
    if P.shape not in ("flag", "enum"): return None
    d = Decision()
    if T & (COUNTER | VALIDITY | CONFIG) or T & {"handle", "stalk", "knob", "buttn", "bttn"} or T & {"warn", "warning", "wrn", "msg", "message", "text", "display", "disp", "indicator", "ind", "reminder", "request", "req", "cmd", "command", "button", "btn", "push", "lock", "unlock", "key"}: return None
    if "door" in T and not T & {"lock", "mirror", "window", "trunk", "boot", "hood", "tailgate", "sliding", "fuel", "charge", "flap"}:
        if _unread(P): d.refused = f"unreadable word(s) {_unread(P)} in a door signal"; return d
        if P.axle == "rear" or ("passenger" in T and "driver" not in T): d.refused = "a door that is not the driver's"; return d
        if (P.axle, P.side) not in ((None, None), ("front", "left")) and "driver" not in T: d.refused = "one door of several (only the driver's door or 'any door' is mapped)"; return d
        return _flag(P, "door_state", OPEN, CLOSED, "door", d)
    if "belt" in T:
        if _unread(P): d.refused = f"unreadable word(s) {_unread(P)} in a seat belt signal"; return d
        if P.axle == "rear" or ("passenger" in T and "driver" not in T): d.refused = "a belt of another seat"; return d
        if (P.axle, P.side) not in ((None, None), ("front", "left")) and "driver" not in T: d.refused = "a belt of another seat"; return d
        return _flag(P, "seatbelt_state", FASTENED, UNFASTENED, "seat belt", d)
    if ("parking" in T and "brake" in T) or {"hand", "brake"} <= T:
        if _unread(P): d.refused = f"unreadable word(s) {_unread(P)} in a parking brake signal"; return d
        if T & {"service", "mode", "auto", "automatic", "motor", "force", "fault", "release"}: d.refused = "a parking brake function that is not whether it is applied"; return d
        return _flag(P, "parking_brake", APPLIED, RELEASED, "parking brake", d)
    if (T & {"headlight", "headlamp"}) or ("head" in T and (T & {"lamp", "light"})):
        if _unread(P): d.refused = f"unreadable word(s) {_unread(P)} in a headlamp signal"; return d
        if (T | set(P.ctx)) & CONTROL_WORDS: d.refused = "the driver's lamp control, not the lamp"; return d
        if P.shape == "flag" and not T & {"high", "beam", "fog", "auto", "automatic", "sensor", "level", "leveling", "adaptive", "tail", "taillight", "parking", "position", "side", "marker", "drl", "daytime", "running", "turn", "switch"}: d.concept, d.confidence, d.reasons = "headlight_state", 0.75, ["headlamp on/off flag"]; return d
        d.refused = "headlamp mode is not a plain on/off flag"; return d
    if "wiper" in T:
        if _unread(P): d.refused = f"unreadable word(s) {_unread(P)} in a wiper signal"; return d
        if (T | set(P.ctx)) & CONTROL_WORDS: d.refused = "the driver's wiper control, not the wipers"; return d
        if (set(P.raw) | T) & {"park", "parking", "wash", "washer", "rear", "mist", "interval", "intermittent", "int", "intt", "auto", "automatic", "sensor", "psd", "low", "lo", "high", "hi", "fast", "slow", "speed", "pk"} or P.axle == "rear": d.refused = "a wiper function that is not the wiper state"; return d
        d.concept, d.confidence, d.reasons = "wiper_state", 0.75, ["wiper state"]; return d
    return None


def _unread(P):
    """Words of the name that are neither digits nor words the mapper knows: a state is only mapped when everything the name says has been read."""
    voc = vocab()
    return [t for t in P.toks if not t.isdigit() and t not in voc]


def _brake_pedal_flag(ev, P):
    """A brake pedal / switch that is pressed or not (a flag) reported as the pedal position: 0 and 100 %."""
    T = set(P.toks)
    if P.shape not in ("flag", "enum") or "brake" not in T or T & (COUNTER | VALIDITY | WARN | {"light", "fluid", "parking", "hand", "torque", "pressure", "request", "cmd", "lamp"}): return None
    if "pedal" not in T and not T & {"pressed", "applied"}: return None
    if _unread(P): return Decision(refused=f"unreadable word(s) {_unread(P)} in a brake pedal signal")
    if "and" in P.raw: return Decision(refused="a compound condition (pressed AND something else) is not the pedal")
    inv = bool((set(P.raw) | T) & {"inv", "inverted", "invert"})
    if inv: return Decision("brake_pedal", 0.7, ["inverted brake pedal flag"], transform={"map": {"0": 100, "1": 0}, "snap": True})
    m = _label_polarity(P.labels, {"applied", "pressed", "on", "active", "depressed"}, {"released", "off", "inactive", "unpressed"}) if P.labels else None
    if m: return Decision("brake_pedal", 0.75, ["brake pedal value table"], transform={"map": {l: 100 * v for l, v in m.items()}, "snap": True})
    R_ = set(P.raw) | T
    if R_ & {"pressed", "applied", "depressed", "actuated", "active"} and not R_ & {"not", "released"}: return Decision("brake_pedal", 0.75, ["brake pedal pressed flag"], transform={"map": {"0": 0, "1": 100}, "snap": True})
    return None


def _gear(ev, P):
    T = set(P.toks)
    if "gear" not in T or T & (COMMAND | COUNTER | VALIDITY | WARN | CONFIG | {"ratio"}): return None
    if P.shape == "enum" and P.labels:
        g = _gear_map(P.labels)
        if g: return Decision("gear", 0.8, ["gear value table"], transform={"map": g})
        return Decision(refused="a gear value table that cannot be read")
    return Decision(refused="a gear signal without a value table: what its numbers mean is not stated")


def decide(ev):
    P = parse(ev); alts = []
    if not P.toks and not P.ctx: return Decision(refused="the name carries no word")
    q = next((t for t in P.toks if t in COUNTER), None)
    if q: return Decision(refused=f"'{q}' marks a counter / checksum, not a measurement")
    if P.ctx_block: return Decision(refused="the signal sits in a request / settings message, which carries commands and configuration, not readings")
    for rule in (_gear, _categorical, _brake_pedal_flag):
        d = rule(ev, P)
        if d is not None: return d
    if P.shape in ("flag", "enum"): return Decision(refused=f"a {ev.bits or '?'}-bit state with no rule for it")
    if P.unit_unknown: return Decision(refused=f"the unit '{ev.unit}' is not one the mapper can convert")
    if P.peeled and P.toks and all(t in QTY or t in NEUTRAL or t in GENERIC or t == "range" for t in P.toks): return Decision(refused="a bare quantity word behind a prefix that was taken for a tag but could not be read (it may name what is measured)")
    S = P.S; C = set(P.ctx); scored = []; phr = _phrases(); ident = ident_words()
    for cid, plist in phr.items():
        cdef = R.CONCEPTS[cid]
        if cdef.kind == "derived" or cdef.dtype in ("categorical", "binary"): continue
        s, why, wtoks = name_score(S, C, plist, unit_ok=(P.unit_cls is not None and P.unit_cls == U.OF_REGISTRY.get(cdef.unit)))
        if s < 0.3: continue
        reason = None; ct = _concept_tokens(cid)
        qual = _qualifier(P, ct, True)
        if qual: reason = f"'{qual}' says the signal is not the quantity itself"
        ccls0 = U.OF_REGISTRY.get(cdef.unit)
        if reason is None and (cid.endswith("_max") or cid.endswith("_min")) and not (S | C) & ({"max", "maximum", "highest", "hottest"} if cid.endswith("_max") else {"min", "minimum", "lowest", "coldest"}): reason = "a maximum / minimum quantity needs the name to say so (an average or a plain reading is not it)"
        if reason is None and ccls0 in CLASS_WORDS and P.unit_cls != ccls0 and not (P.ambig_deg and ccls0 == "angle") and not (S & CLASS_WORDS[ccls0]): reason = f"nothing in the name or the unit says it is a {ccls0}"
        if reason is None and ccls0 == "angular_rate" and "angle" in S and not S & {"rate", "speed", "velocity"} and P.unit_cls != "angular_rate": reason = "an angle, not the rate at which it changes"
        if reason is None and ccls0 in ("temperature", "pressure"):
            spec_w = {t for t in wtoks if t not in QTY}
            if spec_w and not (S & spec_w):                                                                                      # the subsystem is named by the message only
                sub = spec_w & C; same = [k for k, c2 in R.CONCEPTS.items() if k != cid and U.OF_REGISTRY.get(c2.unit) == ccls0 and c2.kind != "derived" and sub & _concept_tokens(k)]
                if sub and same: reason = "the subsystem comes from the message only, and it has several such sensors"
        if reason is None and cid in SIGNED_CONCEPTS and not ev.signed and P.pmin is not None and P.pmin >= 0 and cdef.lo is not None and cdef.lo < 0:
            reason = "a signed quantity from a field that cannot be negative (its sign is somewhere else, or it is only a magnitude)"
        if reason is None and cid == "speed" and S & SPEED_BLOCK and "vehicle" not in S: reason = "that is the speed of something else"
        if reason is None and cid in TWINS and C & FOREIGN_SUB and not S & FOREIGN_SUB and "engine" not in S: reason = "the message belongs to another subsystem (steering, brakes ...), whose own motor is not the drive"
        if reason is None and cid in ("engine_hours", "engine_run_time") and "engine" not in S and "engine" not in C: reason = "an hour / run-time count that does not say it is the engine's"
        if reason is None and (ev.bits or 0) > 32 and cid not in NO_SPAN_GATE: reason = f"a {ev.bits}-bit field is a block of data, not a measurement"
        if reason is None and P.sib_ident and cid in ("wheel_speed", "tyre_pressure", "tyre_temperature"): reason = "which wheel it is comes from another signal of the message (an identifier)"
        if reason is None and (P.axle or P.side) and not _is_positional(cid): reason = "a positioned signal belongs to one wheel, door or corner"
        if reason is None and P.family >= 2 and not _is_positional(cid): reason = f"one of an array of {P.family + 1} similar sensors (a quantity of its own would need a name for the array)"
        if reason is None and P.indexed and (_is_positional(cid) or cid in ("wheel_speed", "tyre_pressure", "tyre_temperature")): reason = "a numbered wheel whose position is not stated"
        if reason is None and P.unit_cls is not None:
            ccls = U.OF_REGISTRY.get(cdef.unit)
            if ccls is not None and ccls != P.unit_cls and not ({ccls, P.unit_cls} <= {"flow", "fuel_flow", "mass_flow"}): reason = f"unit class {P.unit_cls} is not {ccls}"
        if reason is None and P.ambig_deg:
            ccls = U.OF_REGISTRY.get(cdef.unit)
            if ccls not in ("angle", "temperature", "angular_rate"): reason = "unit 'deg' is an angle or a temperature"
        if reason is None and P.span is not None and cdef.lo is not None and cdef.hi is not None and cid not in NO_SPAN_GATE:
            T = cdef.hi - cdef.lo; fb = U.factor_between(P.cu, cdef.unit) if P.cu else None; k = abs(fb[0]) if fb else 1.0
            if cdef.unit == "%" and _is_fraction(ev, P): pass                                                        # a 0..1 fraction of a percentage
            elif P.span * k < 0.12 * T: reason = f"{ev.bits} bits at a scale of {ev.factor:g} cannot cover a {cdef.label} (span {P.span * k:.3g} of {T:g})"
            elif cdef.unit == "%" and not P.has_unit and not P.name_unit and P.span * k > 12 * T: reason = "a percentage from a wide field with no unit: the scale to percent is not stated"
        if reason is None and cid in MIN_SPAN and P.span is not None:
            fb2 = U.factor_between(P.cu, cdef.unit) if P.cu else None; wrap = P.span * (abs(fb2[0]) if fb2 else 1.0)
            if wrap < MIN_SPAN[cid]: reason = f"a {ev.bits}-bit counter that wraps after {wrap:g} {cdef.unit} cannot be the {cdef.label}"
        if reason is None and _is_positional(cid):
            want = _pos_of(cid)
            if P.axle is None or P.side is None: reason = "the wheel position is not stated"
            elif (P.axle, P.side) != want: reason = f"the position is {P.axle} {P.side}, not {want[0]} {want[1]}"
        elif reason is None and (P.axle or P.side) and cid in ("wheel_speed", "tyre_pressure", "tyre_temperature"):
            reason = "a positioned signal is one wheel, not the aggregate"
        if reason: alts.append((cid, round(s, 2), reason)); continue
        if s >= 0.45:                                                                                  # the message names the subsystem: 'Charger_stats', 'STEERING', 'ENGINE_1'
            lab = {t for t in expand(pieces(cdef.label)) if t in ident}
            s += min(0.3, 0.18 * len((lab & C) - set(P.S)))
        scored.append((s, cid, why))
    scored.sort(reverse=True)
    out = Decision(alternatives=[(c, round(s, 2)) for s, c, _ in scored[:3]] + alts[:2])
    twin = TWINS.get(scored[0][1]) if scored else None                                                 # engine / motor are told apart later, by the powertrain
    gated = max((a for a in alts if a[1] >= 0.85 and a[1] > (scored[0][0] if scored else 0) and a[0] != twin), key=lambda a: a[1], default=None)
    if gated is not None: out.refused = f"the name reads as {gated[0]} ({gated[1]}) but a gate ruled that out ({gated[2]}); a weaker reading is not trusted"
    elif scored:
        scored = _tiebreak(scored, P, ev)
        top = scored[0]; second = scored[1][0] if len(scored) > 1 else 0.0
        if top[0] >= 0.62 and top[0] - second >= 0.075:
            out.concept, out.confidence, out.reasons = top[1], round(min(top[0], 1.0), 2), [f"name ~ '{top[2]}' ({top[0]:.2f})"]
            return _finish(out, ev, P)
        out.refused = f"the best candidate ({top[1]}, {top[0]:.2f}) is not clear enough" + (f" against {scored[1][1]} ({second:.2f})" if second else "")
    else: out.refused = "no quantity passes the gates"
    u = _unit_only(ev, P) if gated is None else None
    if u is not None: u.alternatives = out.alternatives; return _finish(u, ev, P)
    return out


def binding_fields(d, unit=None):
    """The profile-signal fields a decision implies, with the unit conversion folded in (the binding's own unit stays 'none'): {'scale', 'offset'} or {'map'}, and 'snap' for a state that carries a continuous quantity.
    `unit` is the database's unit string; a unit read from the end of the name wins."""
    t = d.transform; cu = R.CONCEPTS[d.concept].unit; src = t.get("unit") or canon_unit(unit)
    scale, offset = float(t.get("scale", 1.0)), 0.0; f = U.factor_between(src, cu) if src else None
    if f is not None: scale, offset = scale * f[0], f[1]
    out = {}
    if t.get("map"): out["map"] = dict(t["map"])
    elif t.get("invert01"): out["map"] = {"0": 1, "1": 0}
    else:
        if scale != 1.0: out["scale"] = scale
        if offset: out["offset"] = offset
    if t.get("snap"): out["snap"] = True
    return out


def _finish(out, ev, P):
    """Powertrain corrections and the unit transform of an accepted mapping."""
    c = out.concept; T = set(P.toks) | set(P.ctx); nm = set(P.toks)
    if c in TWINS:
        mine = c.split("_")[0]; other = TWINS[c].split("_")[0]
        if other in nm and mine not in nm:                                                       # the name itself says the other one
            if (other == "motor" and ev.ice is True) or (other == "engine" and ev.ice is False): return Decision(refused=f"the name says {other} but this vehicle has none that drives it", alternatives=out.alternatives)
            out.concept = TWINS[c]; out.reasons.append(f"the name says {other}")
        elif T & {"engine", "motor"} and mine in T: pass                                         # the name says which
        elif ev.ice is None: return Decision(refused="engine or motor cannot be told (no word in the name or message, powertrain unknown)", alternatives=out.alternatives)
        else:
            want = TWINS[c] if ((ev.ice and mine == "motor") or (not ev.ice and mine == "engine")) else c
            if want != c: out.concept = want; out.reasons.append(f"{'combustion' if ev.ice else 'electric'} powertrain: {want}")
    explicit = bool((set(P.toks) | set(P.raw)) & TRACTION_WORDS)
    if out.concept == "battery_voltage" and "traction" not in P.S and not explicit:
        hi = P.pmax * (abs(U.factor_between(P.cu, "V")[0]) if P.cu and U.factor_between(P.cu, "V") else 1.0) if P.pmax is not None else None
        if ev.stats and ev.stats.get("max") is not None: hi = ev.stats["max"] * (abs(U.factor_between(P.cu, "V")[0]) if P.cu and U.factor_between(P.cu, "V") else 1.0)
        if ev.ice is True: out.concept = "aux_battery_voltage"; out.reasons.append("no traction battery in this vehicle: the 12 V battery")
        elif ev.ice is None:
            if hi is not None and 5 <= hi <= 40: out.concept = "aux_battery_voltage"; out.reasons.append("a field that tops out at the voltage of a 12 V battery")
            elif hi is None or not 100 <= hi <= 1500: return Decision(refused="12 V battery or traction battery cannot be told", alternatives=out.alternatives)
    elif out.concept in TRACTION_ONLY and ev.ice is not False and not explicit: return Decision(refused="a battery quantity of a vehicle that is not known to be electric (it may be the 12 V battery)", alternatives=out.alternatives)
    cdef = R.CONCEPTS[out.concept]
    if P.name_unit: out.transform = dict(out.transform, unit=P.name_unit); out.reasons.append(f"unit '{P.name_unit}' read from the end of the name")
    if cdef.unit == "%" and _is_fraction(ev, P): out.transform = dict(out.transform, scale=100.0); out.reasons.append("a 0..1 fraction, scaled to percent")
    return out


def _is_fraction(ev, P):
    """A unitless field that counts in hundredths (or finer) of one: a fraction, not a percentage."""
    if ev.unit or P.span is None or P.pmin is None or abs(P.pmin) > 0.05: return False
    return (0.8 <= P.span <= 1.35) or (abs(ev.factor) in (0.01, 0.005, 0.004, 0.0025, 0.002, 0.001) and (ev.bits or 0) >= 7 and P.span <= 3.0)


def _tiebreak(scored, P, ev):
    """Candidates within the margin are separated by what the name says explicitly: the concept whose own label words are all in the signal's name wins."""
    top = scored[0][0]; near = [x for x in scored if top - x[0] < 0.075]
    if len(near) < 2: return scored
    T = set(P.toks) | set(P.ctx)
    def explicit(x):
        c = R.CONCEPTS[x[1]]; lab = {t for t in expand(pieces(c.label)) if t not in GENERIC}
        return sum(1 for t in lab if t in T) - 0.5 * sum(1 for t in lab if t not in T)
    if {x[1] for x in near} == {"speed", "wheel_speed"} and "vehicle" in T: return [(x[0] + 0.08, x[1], x[2]) for x in near if x[1] == "speed"] + [x for x in scored if x[1] != "speed"]      # J1939's wheel-based vehicle speed is the vehicle's speed
    near.sort(key=lambda x: (-explicit(x), -x[0])); a, b = explicit(near[0]), explicit(near[1])
    if a > b: return [(near[0][0] + 0.08, near[0][1], near[0][2])] + [x for x in scored if x is not near[0]]
    return scored


def _unit_only(ev, P):
    """A bare name read through its unit and message: only for classes where nothing else can be meant."""
    if P.unit_cls is None or P.span is None or P.axle or P.side: return None
    T = set(P.toks)
    if T & (COUNTER | COMMAND | VALIDITY | CONFIG | WARN | PEAK | SWITCHES): return None
    if any(t in vocab() and t not in NEUTRAL and t not in ("vehicle", "speed", "rpm") for t in T): return None            # the name says something else
    if any(len(t) >= 3 and not t.isdigit() and t not in vocab() for t in T): return None                                   # an abbreviation nobody can read may be anything
    ctxw = set(P.ctx)
    if P.unit_cls == "speed" and not ctxw & {"wheel", "steering", "cruise", "acc", "limit", "target", "set", "gps", "wind", "yaw"}:
        cdef = R.CONCEPTS["speed"]
        if P.span * U.find(P.cu)[1] >= 0.12 * (cdef.hi - cdef.lo): return Decision("speed", 0.66, ["a speed in speed units with no other meaning in its name or message"])
    if P.unit_cls == "rpm" and (ctxw & {"engine", "motor"} or T & {"rpm"}):
        if ctxw & {"engine"}: return Decision("engine_rpm", 0.66, ["rpm in an engine message"])
        if ctxw & {"motor"} and ev.ice is None: return None                                    # a hybrid has both, and in some languages 'motor' is the engine
        if ctxw & {"motor"}: return Decision("motor_rpm", 0.66, ["rpm in a motor message"])
        if ev.ice is not None: return Decision("engine_rpm" if ev.ice else "motor_rpm", 0.64, ["rpm of this powertrain"])
    return None


def _gear_map(labels):
    """{label: gear number} from a gear selector's value table (P/N -> 0, R -> -1, D -> 1, 'First Gear' -> 1 ...), or None when a label that matters cannot be read."""
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
    out = {}
    for l in labels:
        w = [x for x in re.split(r"[^a-z0-9]+", str(l).lower()) if x and x != "gear"]
        last = w[-1] if w else ""
        if any(x in ("invalid", "unknown", "sna", "supported", "not", "undefined", "reserved", "fault", "faulty", "notused", "unused") for x in w): continue
        if last in ("p", "park", "parking", "n", "neutral"): out[l] = 0
        elif last in ("r", "reverse"): out[l] = -1
        elif last in ("d", "drive", "b", "s", "l", "m"): out[l] = 1
        elif w and w[0] in words: out[l] = words[w[0]]
        elif last.isdigit() and 1 <= int(last) <= 20: out[l] = int(last)
        elif last.startswith("d") and last[1:].isdigit(): out[l] = int(last[1:])
        else: return None
    return out or None


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------- whole databases
_ENGINE_STRONG = {"engine", "combustion", "petrol", "diesel", "gasoline", "lambda", "dpf", "egr", "turbo", "throttle", "injection", "spark", "catalyst", "exhaust"}        # an electric vehicle's database says 'motor' or 'drive unit'
_ENGINE_WEAK = {"fuel", "intake", "manifold", "idle", "clutch", "starter", "cylinder", "knock", "misfire", "alternator", "kickdown", "oil"}                                   # two of these together


def powertrain_from_words(word_sets):
    """True: a combustion vehicle (an engine word, none about a traction battery); False: electric (traction-battery words, no engine words); None: both (a hybrid) or neither (names that say
    nothing either way).  'Chg' is not read as charging (oil change, lane change), 'traction control' is not a traction battery, a brake booster is not a turbo, and a 12 V battery's state of charge says nothing."""
    ev = False; en = set()
    for w in word_sets:
        if w & {"cell", "charging", "charger", "inverter", "regen", "regeneration", "hv", "stator", "bms", "pack"} or ("traction" in w and "battery" in w) or ("hybrid" in w and "battery" in w): ev = True
        en |= w & (_ENGINE_STRONG | _ENGINE_WEAK)
    has_engine = bool(en & _ENGINE_STRONG) or len(en & _ENGINE_WEAK) >= 2
    if ev and has_engine: return None
    if ev: return False
    return True if has_engine else None


def powertrain_from_names(names):
    """The same from a list of signal / column names (a table has no message names)."""
    return powertrain_from_words({_singular(x) for p in pieces(n) for x in segment(p)} for n in names)


def powertrain_is_ice(dbs):
    """From one DBC or several (the buses of one vehicle): the signal and message names."""
    dbs = [dbs] if hasattr(dbs, "messages") else list(dbs); sets = []
    for db in dbs:
        for d in (db.messages, db.extended_messages):
            for m in d.values():
                for s in m.signals: sets.append({_singular(x) for p in pieces(s.name) for x in segment(p)} | {_singular(x) for x in pieces(m.name)})
    return powertrain_from_words(sets)


def map_database(db, ice="auto"):
    """-> [(message, signal, Decision)] for every signal of a DBC.  Several signals of one quantity: the best keeps it, the others are listed as duplicates (a human or a capture decides which is the redundant source)."""
    if ice == "auto": ice = powertrain_is_ice(db)
    out = []
    for d in (db.messages, db.extended_messages):
        for m in d.values():
            for s in m.signals:
                if m.mux_signal is not None and s is m.mux_signal: continue
                ev = Evidence(name=s.name, unit=s.unit or None, bits=s.length, factor=s.factor, offset=s.offset, signed=s.signed, lo=s.minimum if s.minimum < s.maximum else None, hi=s.maximum if s.minimum < s.maximum else None,
                              choices=list(s.choices.values()) if s.choices else [], message=m.name, siblings=[x.name for x in m.signals if x is not s], dlc=m.dlc, ice=ice)
                out.append((m.name, s.name, decide(ev)))
    best = {}
    for i, (m, s, dec) in enumerate(out):
        if dec.concept and (dec.concept not in best or dec.confidence > out[best[dec.concept]][2].confidence): best[dec.concept] = i
    for i, (m, s, dec) in enumerate(out):
        if dec.concept and best[dec.concept] != i:
            o = out[best[dec.concept]]
            out[i] = (m, s, Decision(refused=f"another signal of this vehicle ({o[0]}.{o[1]}) already carries {dec.concept}", alternatives=[(dec.concept, dec.confidence)] + dec.alternatives))
    return out

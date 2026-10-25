"""Vehicle profile pack: the ONE vehicle-specific artifact of the system (plug and play).

Everything in the assistant is generic over canonical CONCEPTS (speed, battery_temperature, tyre_pressure_fl, ...; vgpt/signal_registry.py).  What differs between vehicles is only how those concepts
arrive: which buses exist, which DBC / SOME-IP catalogue / VHAL properties / log formats carry them, in which units, under which names, and what counts as a warning.  A profile pack describes
exactly that, as data:

    packs/<id>/profile.json   identity + specs, sources (buses), signal bindings (source signal -> concept, unit, scale, priority), event rules, thresholds, retention
    packs/<id>/*.dbc          CAN / CAN-FD databases (J1939 and OEM)         packs/<id>/services.json   SOME/IP service catalogue
    packs/<id>/dtc.json       diagnostic trouble code texts                  packs/<id>/vhal_props.json (optional)  extra VHAL properties

Installing the assistant in a new vehicle = write (or generate with `python -m vgpt.vehicle.onboard`) a pack, run `python -m vgpt.vehicle.provision`, start the collectors.  No code changes.
A signal the canonical registry does not know is declared under "concepts" and is then stored, searched and answered like any other (it lands in the signal lake, vgpt/vehicle/lake.py).

Validation is strict and reports every problem with its path (unknown concept, unit that cannot convert, DBC signal that does not exist, plausible-range contradiction, a retention below the
case study's 'several weeks' ...); a pack that does not validate cannot be provisioned."""
import json
import os
from dataclasses import dataclass, field

from .. import signal_registry as R

SCHEMA = "vehiclegpt.profile/1"
BUSES = ("can", "someip", "vhal", "log", "camera", "gnss", "uds", "signal", "table", "lin", "flexray", "canopen")           # table: rows of named values (CSV / JSON / OpenXC / VSS), see vgpt/ingest
UNIT_ALIAS = {"degc": "C", "°c": "C", "celsius": "C", "degf": "F", "°f": "F", "fahrenheit": "F", "kph": "km/h", "kmh": "km/h", "km/hr": "km/h", "mps": "m/s", "metre/s": "m/s", "pct": "%", "percent": "%",
              "kilopascal": "kPa", "kpa": "kPa", "psi": "psi", "bar": "bar", "mph": "mph", "volt": "V", "v": "V", "amp": "A", "a": "A", "kw": "kW", "w": "W", "wh": "Wh", "kwh": "kWh", "l/min": "L/min",
              "lpm": "L/min", "l/h": "L/h", "rpm": "rpm", "nm": "Nm", "deg": "deg", "degrees": "deg", "degree": "deg", "m/s^2": "m/s2", "r/min": "rpm", "1/min": "rpm", "degc ": "C", "": "none", "none": "none", "mm": "mm", "g": "g", "ratio": "ratio", "lux": "lux", "km": "km", "m": "m"}
INTERNAL = ("_ignition", "_charging_active")                                      # state channels the store uses (drive mode), not concepts a person asks about
MIN_RETENTION_DAYS = 70                                                          # the owner's reading of "several weeks": at least ten weeks of searchable history


class ProfileError(ValueError): pass


@dataclass
class Issue:
    level: str                       # error | warning
    path: str
    message: str

    def __str__(self): return f"{self.level}: {self.path}: {self.message}"


def norm_unit(u):
    u = "" if u is None else str(u).strip()
    return UNIT_ALIAS.get(u.lower(), u)


@dataclass
class Binding:
    concept: str
    bus: str                         # can | someip | vhal | camera | gnss | log
    source: str                      # source id from profile["sources"]
    key: str                         # CAN: "Message.Signal" | SOME/IP: "Service.Event.field" | VHAL: "PROP_NAME" | camera: field | log: rule id
    unit: str = "none"
    scale: float = 1.0               # applied to the decoded physical value BEFORE unit conversion (an adapter fix: a DBC that publishes tenths of a unit)
    offset: float = 0.0
    priority: int = 1                # 1 = primary; a lower-priority source of the same concept is used when the primary is stale, and compared with it for conflicts
    rate_hz: float = 1.0
    deadband: float = 0.0            # report-by-exception in the 1 Hz lake tier: a change smaller than this is not stored (a heartbeat still is)
    ttl: float = None                # seconds a value stays current; default 3 / rate
    store: str = "auto"              # auto | core | lake
    invert: bool = False
    map: dict = None                 # enum label (CAN value table) or raw VHAL value -> number, e.g. {"charging": 1, "idle": 0}
    when: tuple = None               # a threshold turns the value into 1/0 (engine speed > 300 rpm -> running): (">", 300.0)
    valid: tuple = None              # plausible range of THIS signal in the canonical unit, when the registry range is too narrow for this vehicle (a truck engine that brakes: negative torque)
    conflict_abs: float = None       # redundant sources of one concept may differ by this much before a conflict is raised (default: 5 % of the plausible range)
    cunit: str = None                # canonical unit of the concept (set by the profile; declared concepts have no registry entry)
    snap: bool = False               # an enum signal that carries a continuous quantity as states (brake pedal OFF / ON): the nearest state is sent when no state equals the value

    def canonical(self, value):
        """The bound value in the concept's canonical unit."""
        v = float(value) * self.scale + self.offset
        if self.invert: v = -v
        cu = self.cunit or (R.CONCEPTS[self.concept].unit if self.concept in R.CONCEPTS else None)
        if cu is None or self.unit in ("none", "", cu): return v
        return R.convert(v, self.unit, cu)


@dataclass
class Source:
    id: str
    type: str
    name: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class EventRule:
    id: str
    type: str                        # event type stored in the events table (ADAS_DISENGAGE, WARN_BATT_OVERTEMP, ...)
    severity: str = "info"
    origin: str = ""                 # someip:Service.Event | uds:dtc | log:<source> | signal:<concept>
    match: str = ""                  # log: case-insensitive substring; signal: unused
    when: str = ""                   # signal: ">52", "<8" ...
    hold_s: float = 0.0              # signal: the condition must hold this long
    detail: str = ""                 # template: {reason_text} {value} {code} ...
    raw: dict = field(default_factory=dict)


class Profile:
    def __init__(self, spec, base_dir="."):
        self.spec = spec; self.base = base_dir
        self.id = spec.get("id", ""); self.vehicle = spec.get("vehicle", {}); self.specs = dict(self.vehicle.get("specs", {}))
        self.sources = {s["id"]: Source(s["id"], s["type"], s.get("name", ""), s) for s in spec.get("sources", []) if isinstance(s, dict) and "id" in s and "type" in s}
        self.bindings = []
        for b in spec.get("signals", []):
            if not isinstance(b, dict): continue
            src = str(b.get("from", "")); bus, _, rest = src.partition(":"); source, _, key = rest.partition("/")
            self.bindings.append(Binding(b.get("concept", ""), bus, source, key, norm_unit(b.get("unit", "none")), float(b.get("scale", 1.0)), float(b.get("offset", 0.0)), int(b.get("priority", 1)), float(b.get("rate_hz", 1.0)),
                                         float(b.get("deadband", 0.0)), b.get("ttl"), b.get("store", "auto"), bool(b.get("invert", False)), b.get("map"), (b["when"][0], float(b["when"][1:])) if b.get("when") else None, tuple(b["valid"]) if b.get("valid") else None, b.get("conflict_abs"), snap=bool(b.get("snap", False))))
        self.events = [EventRule(e.get("id", f"e{i}"), e.get("type", ""), e.get("severity", "info"), e.get("from", ""), e.get("match", ""), e.get("when", ""), float(e.get("hold_s", 0.0)), e.get("detail", ""), e) for i, e in enumerate(spec.get("events", []))]
        self.did_fmt = {}
        for s in self.sources.values():                                                       # UDS ReadDataByIdentifier responses: {"0xF1A0": {"concept": "battery_soh", "fmt": ">H", "scale": 0.01, "unit": "%"}}
            if s.type != "uds": continue
            for did, d in (s.raw.get("dids") or {}).items():
                self.bindings.append(Binding(d["concept"], "uds", s.id, f"did:{did}", norm_unit(d.get("unit", "none")), float(d.get("scale", 1.0)), float(d.get("offset", 0.0)), int(d.get("priority", 1)), float(d.get("rate_hz", 0.02))))
                self.did_fmt[int(did, 16)] = (d.get("fmt", ">H"), self.bindings[-1])
        self.od_fmt = {}                                                                      # CANopen: SDO reads of one object: (node, index, sub) -> (struct format, binding)
        for s in self.sources.values():
            if s.type != "can": continue
            for key, d in (s.raw.get("od") or {}).items():
                idx, _, sub = key.partition(":"); node = int(str(d.get("node", "0x01")), 16)
                self.bindings.append(Binding(d["concept"], "canopen", s.id, f"sdo:{key}@{node}", norm_unit(d.get("unit", "none")), float(d.get("scale", 1.0)), float(d.get("offset", 0.0)), int(d.get("priority", 1)), float(d.get("rate_hz", 0.5))))
                self.od_fmt[(node, int(idx, 16), int(sub or 0))] = (d.get("fmt", "<H"), self.bindings[-1])
        self.detectors = list(spec.get("detectors", [])); self.transitions = list(spec.get("transitions", []))
        self.concepts = list(spec.get("concepts", []))
        for b in self.bindings: b.cunit = self.canonical_unit(b.concept) if not b.concept.startswith("_") else None
        self.thresholds = dict(spec.get("thresholds", {}))
        self.declared_retention = dict(spec.get("retention", {}))
        self.retention = {"fine_days": 7, "coarse_days": 365, "log_days": 1825, "context_days": 1825,
                          "archive_days": 365, "records_days": 1825}
        self.retention.update(spec.get("retention", {}))
        # Product guarantees, independent of how old a decoder/profile pack is.
        # Packs may ask for longer retention but may not silently weaken these
        # floors.  This keeps decoder metadata separate from storage policy.
        for key, floor in (("coarse_days", 365), ("archive_days", 365),
                           ("log_days", 1825), ("context_days", 1825),
                           ("records_days", 1825)):
            self.retention[key] = max(floor, int(self.retention.get(key, floor)))
        self.drop = list(spec.get("drop", []))
        self._dbc, self._cat, self._dtc = {}, {}, None; self._ldf = {}; self._fx = {}

    # ------------------------------------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path):
        """A pack directory, or a profile.json file, or a pack name bundled with the package."""
        if not os.path.exists(path):
            bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs", str(path))
            if os.path.isdir(bundled): path = bundled
        p = os.path.join(path, "profile.json") if os.path.isdir(path) else path
        if not os.path.isfile(p): raise ProfileError(f"no profile at {path}")
        try: spec = json.load(open(p, encoding="utf-8"))
        except ValueError as e: raise ProfileError(f"{p} is not valid JSON: {e}") from e
        return cls(spec, os.path.dirname(os.path.abspath(p)))

    def file(self, name): return name if os.path.isabs(name) else os.path.join(self.base, name)

    def dbc(self, source_id):
        if source_id not in self._dbc:
            from ..protocols import dbc
            self._dbc[source_id] = dbc.Database.load(self.file(self.sources[source_id].raw["dbc"]), lenient=True)
        return self._dbc[source_id]

    def ldf(self, source_id):
        """The LIN Description File of a LIN source."""
        if source_id not in self._ldf:
            from ..protocols import lin
            self._ldf[source_id] = lin.Ldf.load(self.file(self.sources[source_id].raw["ldf"]))
        return self._ldf[source_id]

    def flexray(self, source_id):
        """The FlexRay description (slots, cycles, signals) of a FlexRay source."""
        if source_id not in self._fx:
            from ..protocols import flexray
            self._fx[source_id] = flexray.Description.load(self.file(self.sources[source_id].raw["description"]))
        return self._fx[source_id]

    def catalog(self, source_id):
        if source_id not in self._cat:
            from ..protocols import someip
            self._cat[source_id] = someip.ServiceCatalog.load(self.file(self.sources[source_id].raw["catalog"]))
        return self._cat[source_id]

    def dtc_table(self):
        if self._dtc is None:
            self._dtc = {}
            for s in self.sources.values():
                if s.type == "uds" and s.raw.get("dtc"): self._dtc.update(json.load(open(self.file(s.raw["dtc"]), encoding="utf-8")))
        return self._dtc

    def concept_decl(self, name): return next((c for c in self.concepts if c.get("name") == name), None)

    def canopen_nodes(self, source_id):
        """The node numbers a CANopen source speaks for: its object dictionary entries and the PDO identifiers of its DBC."""
        src = self.sources[source_id]; nodes = {int(str(d.get("node", "0x01")), 16) for d in (src.raw.get("od") or {}).values()} | {int(str(n), 16) if isinstance(n, str) else int(n) for n in src.raw.get("nodes", [])}
        try:
            db = self.dbc(source_id)
            for m in list(db.messages.values()):
                if (m.frame_id & 0x780) in (0x180, 0x200, 0x280, 0x300, 0x380, 0x400, 0x480, 0x500) and m.frame_id & 0x7F: nodes.add(m.frame_id & 0x7F)
        except (OSError, ValueError, KeyError): pass
        return sorted(nodes)

    def canonical_unit(self, concept):
        c = R.CONCEPTS.get(concept)
        if c is not None: return c.unit
        d = self.concept_decl(concept); return norm_unit(d.get("unit", "none")) if d else None

    def concept_names(self):
        return sorted({b.concept for b in self.bindings})

    def has_concept(self, concept): return any(b.concept == concept for b in self.bindings)

    # ------------------------------------------------------------------------------------------------ validation
    def validate(self):
        """-> [Issue].  Every message carries the JSON path of the problem."""
        out = []; err = lambda p, m: out.append(Issue("error", p, m)); warn = lambda p, m: out.append(Issue("warning", p, m))
        if self.spec.get("schema") != SCHEMA: err("schema", f"must be '{SCHEMA}'")
        if not self.id or not str(self.id).replace("_", "").replace("-", "").isalnum(): err("id", "a short identifier of letters, digits, '-' or '_'")
        for i, s in enumerate(self.spec.get("sources", [])):
            if not isinstance(s, dict) or "id" not in s or "type" not in s: err(f"sources[{i}]", "needs id and type"); continue
            if s["type"] not in BUSES: err(f"sources[{i}].type", f"unknown type {s['type']!r}; one of {', '.join(BUSES)}")
        ids = [s.get("id") for s in self.spec.get("sources", []) if isinstance(s, dict)]
        if len(ids) != len(set(ids)): err("sources", "duplicate source id")
        for i, c in enumerate(self.concepts):
            if not c.get("name") or not c.get("label"): err(f"concepts[{i}]", "needs name and label")
            elif c["name"] in R.CONCEPTS: warn(f"concepts[{i}]", f"{c['name']} is already a canonical concept; the declaration is ignored")
            elif norm_unit(c.get("unit", "none")) not in R._UNIT_TO_FAMILY and norm_unit(c.get("unit", "none")) not in ("none", ""): warn(f"concepts[{i}].unit", f"unit {c.get('unit')!r} has no conversion family; values are stored as given")
        seen = {}
        for i, b in enumerate(self.bindings):
            p = f"signals[{i}]"
            if b.bus in ("uds", "canopen"): continue
            if b.bus not in BUSES or not b.source: err(p + ".from", f"'{self.spec['signals'][i].get('from')}' must look like '<bus>:<source id>/<key>'"); continue
            if b.source not in self.sources and b.bus not in ("signal",): err(p + ".from", f"unknown source '{b.source}'"); continue
            if b.source in self.sources and self.sources[b.source].type != b.bus and not (b.bus == "can" and self.sources[b.source].type == "can"): err(p + ".from", f"source '{b.source}' is a {self.sources[b.source].type} source, not {b.bus}")
            if b.concept.startswith("_"):
                if b.concept not in INTERNAL: err(p + ".concept", f"internal channel {b.concept!r} unknown; one of {', '.join(sorted(INTERNAL))}")
                continue
            cu = self.canonical_unit(b.concept)
            if cu is None: err(p + ".concept", f"'{b.concept}' is neither a canonical concept nor declared under 'concepts'"); continue
            if b.unit not in ("none", "", cu):
                try: R.convert(1.0, b.unit, cu)
                except ValueError: err(p + ".unit", f"cannot convert {b.unit!r} to the canonical {cu!r} of {b.concept}")
            k = (b.concept, b.priority)
            if k in seen: err(p, f"two bindings for {b.concept} with priority {b.priority} (also signals[{seen[k]}]); give the redundant source priority 2")
            seen[k] = i
            if b.bus == "can" and b.source in self.sources: self._check_can(b, p, err, warn)
            if b.bus == "someip" and b.source in self.sources: self._check_someip(b, p, err)
            if b.bus == "lin" and b.source in self.sources: self._check_lin(b, p, err)
            if b.bus == "flexray" and b.source in self.sources: self._check_flexray(b, p, err)
            if b.bus == "vhal": self._check_vhal(b, p, err)
        for i, e in enumerate(self.events):
            if not e.type and not e.origin.startswith("uds:"): err(f"events[{i}]", "needs a type")
            elif e.type and e.type not in R.EVENTS and e.type not in ("DTC_CLEAR", "SENSOR_CONFLICT", "SOURCE_LOST", "SOURCE_RESTORED", "ECU_RESET", "SHUTDOWN", "BUS_OFF", "UNEXPECTED_SHUTDOWN", "VOLTAGE_DIP", "CAN_ERROR", "SENSOR_INVALID", "ADAS_STATE_CHANGE", "WARN_OVERTEMP", "AEB_ACTIVATION"):
                warn(f"events[{i}].type", f"{e.type!r} is not a known event type; it is stored and listed but no question grammar names it")
            if e.origin.startswith("signal:") and (not e.when or e.when[0] not in "<>"): err(f"events[{i}].when", "a signal event needs a condition like '>52' or '<8'")
            if e.origin.startswith("signal:") and e.origin[7:] not in {b.concept for b in self.bindings}: err(f"events[{i}].from", f"no binding provides {e.origin[7:]}")
        # Diagnose the pack as written even though runtime policy safely raises
        # an old pack to the current product floor.
        days = self.declared_retention.get("coarse_days", self.retention.get("coarse_days", 0))
        if days < MIN_RETENTION_DAYS: warn("retention.coarse_days", f"{days} days is below the {MIN_RETENTION_DAYS} days (ten weeks) the case study's 'several weeks of searchable history' is read as")
        if self.declared_retention.get("log_days", self.retention.get("log_days", 0)) < MIN_RETENTION_DAYS: warn("retention.log_days", f"log retention below {MIN_RETENTION_DAYS} days")
        return out

    def _check_can(self, b, p, err, warn):
        s = self.sources[b.source]
        if "dbc" not in s.raw: err(f"sources[{b.source}]", "a CAN source needs a 'dbc' file"); return
        try: db = self.dbc(b.source)
        except (OSError, ValueError) as e: err(f"sources[{b.source}].dbc", f"cannot load: {e}"); return
        msg, _, sig = b.key.partition(".")
        m = next((m for d in (db.messages, db.extended_messages) for m in d.values() if m.name == msg), None)
        if m is None: err(p + ".from", f"message {msg!r} is not in {s.raw['dbc']}"); return
        sg = next((x for x in m.signals if x.name == sig), None)
        if sg is None: err(p + ".from", f"signal {sig!r} is not in message {msg}"); return
        if sg.unit and b.unit not in ("none", "") and norm_unit(sg.unit) not in (b.unit, "none") and R.unit_family(norm_unit(sg.unit)) != R.unit_family(b.unit): warn(p + ".unit", f"DBC declares {sg.unit!r}, the binding says {b.unit!r}")

    def _check_lin(self, b, p, err):
        s = self.sources[b.source]
        if "ldf" not in s.raw: err(f"sources[{b.source}]", "a LIN source needs an 'ldf' file"); return
        try: ldf = self.ldf(b.source)
        except (OSError, ValueError) as e: err(f"sources[{b.source}].ldf", f"cannot load: {e}"); return
        frame, _, sig = b.key.partition(".")
        fr = ldf.frames.get(frame)
        if fr is None: err(p + ".from", f"frame {frame!r} is not in {s.raw['ldf']}"); return
        if not any(sg.name == sig for sg, _ in fr.signals): err(p + ".from", f"signal {sig!r} is not in frame {frame}")

    def _check_flexray(self, b, p, err):
        s = self.sources[b.source]
        if "description" not in s.raw: err(f"sources[{b.source}]", "a FlexRay source needs a 'description' file"); return
        try: d = self.flexray(b.source)
        except (OSError, ValueError, KeyError) as e: err(f"sources[{b.source}].description", f"cannot load: {e}"); return
        frame, _, sig = b.key.partition("."); fr = d.frames.get(frame)
        if fr is None: err(p + ".from", f"frame {frame!r} is not in {s.raw['description']}"); return
        if sig not in fr.signals: err(p + ".from", f"signal {sig!r} is not in frame {frame}")

    def _check_someip(self, b, p, err):
        s = self.sources[b.source]
        if "catalog" not in s.raw: err(f"sources[{b.source}]", "a SOME/IP source needs a 'catalog' file"); return
        try: cat = self.catalog(b.source)
        except (OSError, ValueError) as e: err(f"sources[{b.source}].catalog", f"cannot load: {e}"); return
        parts = b.key.split(".")
        if len(parts) != 3: err(p + ".from", "SOME/IP keys look like Service.Event.field"); return
        ok = any(sv["name"] == parts[0] and any(ev["name"] == parts[1] and parts[2] in ev["fields"] for ev in sv["events"].values()) for sv in cat.spec.values())
        if not ok: err(p + ".from", f"{b.key} is not in the SOME/IP catalogue")

    def _check_vhal(self, b, p, err):
        from ..protocols import vhal
        try: props = vhal.Vhal().props
        except OSError: return
        if b.key.split("@")[0] not in props: err(p + ".from", f"{b.key.split(chr(64))[0]} is not an AOSP VehicleProperty")

    def check(self):
        issues = self.validate(); errors = [i for i in issues if i.level == "error"]
        if errors: raise ProfileError("profile does not validate:\n" + "\n".join(str(i) for i in errors))
        return issues

    # ------------------------------------------------------------------------------------------------ what the store needs to know
    def summary(self):
        by_bus = {}
        for b in self.bindings: by_bus.setdefault(b.bus, set()).add(b.concept)
        return {"id": self.id, "vehicle": self.vehicle.get("model"), "powertrain": self.vehicle.get("powertrain"), "sources": {s.id: s.type for s in self.sources.values()},
                "concepts": len(self.concept_names()), "by_bus": {k: len(v) for k, v in by_bus.items()}, "events": len(self.events), "retention": self.retention}

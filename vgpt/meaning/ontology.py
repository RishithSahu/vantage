"""The vehicle ontology as DATA: what a vehicle question can be about, how concepts relate, and what a well-formed meaning looks like.

Sources, in order of authority:
  * vgpt/signal_registry.py   - signals, derived metrics, events, named conditions (already the single description of what a vehicle could know)
  * vgpt/meaning/ontology_data.json - speech acts, operators, spec (profile) concepts, whole-subject topics
  * vgpt/meaning/glossary.jsonl - word meanings (aliases, abbreviations, slang, units); see glossary.py

Nothing in this module reads or matches user text.  It produces (a) the closed vocabulary the learned interface predicts into, (b) the typing used to validate a prediction,
(c) the relations (same quantity / same component) from which hard contrasts are generated, and (d) the grammar and prompt text, so that vocabulary, training data, decoder
and validator cannot drift apart.  Adding a concept = adding data here; no code and no text rule changes."""
import json
import os

from .. import signal_registry as R

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_FIELDS = ("act", "concept", "op", "cond", "vs", "ent")                  # version 1: the six words the shipped adapter (full_v11) was trained on
FRAME2_FIELDS = FRAME_FIELDS + ("c2", "ref")                                 # version 2: + a second concept and a reference to the previous turn (docs/FRAME_V2.md)


_ABSENT = set()                                                                  # concepts the adopted vehicle does not have (a combustion vehicle: its battery, charging, motor ...)


def set_absent(names):
    """The model's vocabulary is the vocabulary it was trained on, whatever vehicle is open: a question about the traction pack of a diesel must be READ as that, and then answered 'this vehicle has no traction battery' by the resolver - not forced onto the nearest word the diesel does have."""
    _ABSENT.clear(); _ABSENT.update(names)
    for o in _ONT.values(): o.absent = tuple(sorted(n for n in _ABSENT if n not in o.concepts))


def active_version():
    """Which frame the application speaks: 1 (the shipped adapter) unless VGPT_FRAME=2 selects a version-2 frame model."""
    return 2 if os.environ.get("VGPT_FRAME") == "2" else 1


def fields(version=None):
    return FRAME2_FIELDS if int(version or active_version()) >= 2 else FRAME_FIELDS

COMPONENT_PREFIX = ("battery", "motor", "tyre", "brake", "camera", "cell", "coolant", "hvac", "steering", "wheel", "charging", "regen", "adas", "lane", "aux", "engine", "transmission",
                    "accelerator", "gps", "cabin", "fuel", "lead", "blindspot", "acc", "aeb")
QUANTITY_OVERRIDE = {"battery_soc": "charge_level", "battery_soh": "health_level", "brake_pad_life": "health_level", "tyre_tread_depth": "depth", "aux_battery_voltage": "voltage",
                     "wh_per_km": "efficiency", "energy_used": "energy", "distance_driven": "distance", "range_estimate": "distance", "motor_rpm": "rotation_rate", "motor_torque": "torque",
                     "ambient_temperature": "temperature", "odometer": "distance", "camera_blockage": "obstruction", "lane_confidence": "confidence", "adas_state": "state",
                     "drive_mode": "state", "charging_state": "state", "rain": "state", "light": "illuminance", "heading": "direction", "regen_energy": "energy", "regen_power": "power"}


class Concept:
    __slots__ = ("id", "kind", "label", "gloss", "unit", "component", "quantity", "aliases", "registry_key", "meta", "default_act", "category")

    def __init__(self, id, kind, label, gloss="", unit="none", component="vehicle", quantity="", aliases=(), registry_key=None, meta=(), default_act=None, category=""):
        self.id, self.kind, self.label, self.gloss, self.unit = id, kind, label, gloss, unit
        self.component, self.quantity, self.aliases = component, quantity, tuple(aliases)
        self.registry_key, self.meta, self.default_act, self.category = registry_key, tuple(meta), default_act, category

    def __repr__(self): return f"Concept({self.id!r}, {self.kind})"


def _component_of(cid, category):
    for p in COMPONENT_PREFIX:
        if cid.startswith(p + "_") or cid == p: return {"cell": "battery", "charging": "battery", "regen": "battery", "aux": "battery", "accelerator": "motor", "lane": "adas", "lead": "adas", "blindspot": "adas", "acc": "adas", "aeb": "adas", "cabin": "vehicle", "gps": "vehicle", "fuel": "vehicle"}.get(p, p)
    return {"powertrain": "motor", "battery": "battery", "chassis": "vehicle", "adas": "adas", "environment": "environment", "vehicle": "vehicle", "camera": "camera", "trip": "trip"}.get(category, "vehicle")


class Ontology:
    def __init__(self, data=None, version=None):
        data = data or json.load(open(os.path.join(HERE, "ontology_data.json"), encoding="utf-8"))
        self.data = data; self.version = int(version or active_version()); self.fields = fields(self.version)
        self.acts = dict(data["acts"])
        self.ops = dict(data["ops"])
        self.ents = dict(data["ents"])
        preds = list(R.PREDICATES)
        self.conds = ["none"] + preds + ["not_" + p for p in preds] + list(data["cond_extra"])
        self.vs = ["none"] + preds + list(data["vs_extra"])
        self.refs = []; self.absent = ()
        if self.version >= 2:                                                       # frame version 2: more values in four slots, and the two new slots
            f2 = data["frame2"]; self.acts.update(f2["acts"]); self.ops.update(f2["ops"]); self.vs += list(f2["vs"]); self.refs = list(f2["ref"])
            for a, extra in f2["act_ops_extra"].items(): self.acts[a] = dict(self.acts[a], ops=list(self.acts[a]["ops"]) + [o for o in extra if o not in self.acts[a]["ops"]])
        self.concepts = {}
        skip = set(data.get("skip_registry", ()))
        stored = set(R.STORE)
        for name, c in R.CONCEPTS.items():
            if name in skip: continue
            agg = data.get("aggregate_signals", {}).get(name)
            kind = "derived" if c.kind == "derived" else "signal" if (name in stored or (agg and any(a in stored for a in agg))) else "declared"
            q = QUANTITY_OVERRIDE.get(name) or R.unit_family(c.unit) or c.unit
            self.concepts[name] = Concept(name, kind, c.label, c.description, c.unit, _component_of(name, c.category), q, tuple(c.synonyms) + (c.label,), category=c.category)
        for key, word in data["event_words"].items():
            label, syn = R.EVENTS[key]
            self.concepts[word] = Concept(word, "event", label, "", "count", {"ADAS_DISENGAGE": "adas", "WARN_BATT_OVERTEMP": "battery", "TPMS_LOW": "tyre", "TYRE_INFLATED": "tyre", "LOW_12V": "battery",
                                                                                "CAMERA_BLOCKED": "camera", "CAMERA_DEGRADED": "camera", "COOLANT_FLOW_DROP": "battery", "CHARGE_START": "battery", "AEB_ACTIVATION": "adas", "ADAS_STATE_CHANGE": "adas", "VOLTAGE_DIP": "battery", "WARN_OVERTEMP": "motor", "POWER_DERATE": "motor"}.get(key, "vehicle"),
                                          "event", tuple(syn) + (label,), registry_key=key)
        for name, s in data["spec_concepts"].items():
            self.concepts[name] = Concept(name, "spec", s["label"], s["gloss"], s["unit"], s["component"], s["quantity"], (s["label"],), meta=s["meta"], default_act="spec")
        for name, s in data["topic_concepts"].items():
            self.concepts[name] = Concept(name, s.get("kind", "topic"), s["label"], s["gloss"], "none", name.split("_")[0] if name.endswith("_condition") else "vehicle", "condition", (s["label"],), default_act=s["default_act"])
        for name, s in data["special_concepts"].items():
            self.concepts[name] = Concept(name, "special", s["label"], s["gloss"])
        self.absent = tuple(sorted(n for n in _ABSENT if n not in self.concepts))
        for c in self.concepts.values():
            if c.default_act is None: c.default_act = {"signal": "now", "derived": "stat", "declared": "now", "event": "events"}.get(c.kind)

    # ------------------------------------------------------------------ vocabulary
    def words(self):
        w = {"act": list(self.acts), "concept": list(self.concepts) + list(self.absent), "op": list(self.ops), "cond": list(self.conds), "vs": list(self.vs), "ent": list(self.ents)}
        if self.version >= 2: w["c2"] = ["none"] + list(self.concepts) + list(self.absent); w["ref"] = list(self.refs)
        return w

    def kind_of(self, concept):
        c = self.concepts.get(concept)
        return c.kind if c else None

    def validate(self, f):
        """-> list of typing issues for a frame (act / concept / op / cond / vs / ent from the closed vocabulary and mutually compatible).  Typing only: never looks at text."""
        w = self.words(); issues = []
        for field in self.fields:
            if getattr(f, field) not in w[field]: issues.append(f"unknown {field} '{getattr(f, field)}'")
        if issues: return issues
        if f.concept in self.absent: return [f"this vehicle does not have '{f.concept}'"]
        a = self.acts[f.act]; k = self.kind_of(f.concept)
        if "any" not in a["kinds"] and k not in a["kinds"] and f.concept not in ("unlisted", "none"): issues.append(f"act '{f.act}' does not apply to a {k} concept")
        if f.concept == "none" and "any" not in a["kinds"]: issues.append(f"act '{f.act}' needs a concept")
        if f.op not in a["ops"] and f.op != "none": issues.append(f"operator '{f.op}' does not apply to act '{f.act}'")
        if f.act in ("compare",) and f.vs == "none" and not (self.version >= 2 and f.c2 != "none"): issues.append("compare needs a second operand (vs)")
        if self.version >= 2: issues += self._validate2(f, k)
        return issues

    def _validate2(self, f, k):
        """Typing of the two version-2 slots: a second concept needs an act that can hold it; a reference to the previous turn needs an act that can be about one."""
        out = []
        if f.c2 != "none":
            k2 = self.kind_of(f.c2)
            if f.c2 == f.concept: out.append("the second concept is the same as the first")
            if f.act not in ("stat", "compare", "events", "trend", "extreme"): out.append(f"act '{f.act}' cannot hold a second concept")
            if k2 is not None and k not in (None, "special") and (k == "event") != (k2 == "event"): out.append("a second concept must be of the same kind as the first (both events, or both measured quantities)")
        if f.op == "corr" and f.c2 == "none": out.append("a correlation needs a second concept")
        if f.vs == "typical" and f.act not in ("compare", "stat", "extreme", "trend", "now", "at"): out.append(f"act '{f.act}' cannot be compared with the usual values")
        if (f.act == "explain") != (f.ref == "answer"): out.append("an explanation is exactly a question about the previous answer (act explain <-> ref answer)")
        if f.ref == "item" and f.act not in ("events", "extreme", "context", "stat", "summary", "anomaly"): out.append(f"act '{f.act}' cannot pick an item out of a list")
        return out

    # ------------------------------------------------------------------ relations (the "correlations" between words, derived from structure)
    def relations(self, cid):
        """-> {same_quantity: [...], same_component: [...]}: concepts a careless reader would confuse with `cid` (hard contrasts)."""
        c = self.concepts[cid]; sq, sc = [], []
        for o in self.concepts.values():
            if o.id == cid or o.kind in ("special", "topic"): continue
            if o.quantity and o.quantity == c.quantity and o.component != c.component: sq.append(o.id)
            elif o.component == c.component and o.quantity != c.quantity and o.kind in ("signal", "derived", "spec", "declared") and c.kind in ("signal", "derived", "spec", "declared"): sc.append(o.id)
        return {"same_quantity": sq, "same_component": sc}

    # ------------------------------------------------------------------ decoder + prompt generated from the same data
    def gbnf(self):
        w = self.words()
        alt = lambda xs: " | ".join('"%s"' % x for x in xs)
        rules = ["root ::= " + " \" \" ".join(self.fields)] + [f"{k} ::= {alt(w[k])}" for k in self.fields]
        return "\n".join(rules).replace("_", "-") if False else "\n".join(rules)

    def system_prompt(self):
        """The one static prompt shared by training and inference (compact: the meaning of every word is learned from data and, for rare words, from glossary hints)."""
        w = self.words()
        if self.version >= 2: return self._system_prompt2(w)
        return ("Vehicle assistant parser. Output exactly six words: act concept op cond vs ent.\n"
                "act: " + " ".join(w["act"]) + "\n"
                "op: " + " ".join(w["op"]) + "\n"
                "cond: none threshold, or a driving condition (" + " ".join(c for c in w["cond"] if c not in ("none", "threshold") and not c.startswith("not_")) + "), or not_ plus one of those\n"
                "vs: none, period, or a second condition\n"
                "ent: " + " ".join(w["ent"]) + "\n"
                "concept: the vehicle signal, event, spec or topic asked about; unlisted if this vehicle does not model it; none for social, offtopic, unsupported, control, unsafe, ambiguous.\n"
                "Other tasks: 'Term: phrase' -> concept. 'Pair: a | b' -> same_concept, same_quantity, same_component or unrelated.")

    def _system_prompt2(self, w):
        return ("Vehicle assistant parser. Output exactly eight words: act concept op cond vs ent c2 ref.\n"
                "act: " + " ".join(w["act"]) + "\n"
                "op: " + " ".join(w["op"]) + "\n"
                "cond: none threshold, or a driving condition (" + " ".join(c for c in w["cond"] if c not in ("none", "threshold") and not c.startswith("not_")) + "), or not_ plus one of those\n"
                "vs: none, period, typical (the vehicle's own usual values), or a second condition\n"
                "ent: " + " ".join(w["ent"]) + "\n"
                "concept: the vehicle signal, event, spec or topic asked about; unlisted if this vehicle does not model it; none for social, offtopic, unsupported, control, unsafe, ambiguous.\n"
                "c2: a second concept (corr, two quantities side by side, A or B over two events); none otherwise.\n"
                "ref: none, or what the question takes from the line 'Previous:' (the frame of the last answered question): " + " ".join(r for r in w["ref"] if r != "none") + ".\n"
                "Other tasks: 'Term: phrase' -> concept. 'Pair: a | b' -> same_concept, same_quantity, same_component or unrelated.")


_ONT = {}


def reload():
    """Rebuild the ontology from the registries (after a vehicle's schema overlay was applied or removed)."""
    _ONT.clear()
    return load()


def load(version=None):
    v = int(version or active_version())
    if v not in _ONT: _ONT[v] = Ontology(version=v)
    return _ONT[v]

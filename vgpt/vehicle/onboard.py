"""Onboarding: draft a vehicle profile pack from the vehicle's own descriptions (DBC files, SOME/IP catalogue, VHAL property list).

    python -m vgpt.vehicle.onboard --id my_car --dbc powertrain.dbc:can0 --dbc body.dbc:can1 --services services.json --vhal --out packs/my_car

For every signal the tool splits its name into words (camelCase, underscores, digits), expands the usual abbreviations (spd, tmp, soc, trq, fl ...), reads the unit and the plausible range declared in
the DBC, and scores it against every canonical concept: word overlap with the concept's label and synonyms, compatibility of the unit family, and whether the declared range fits the concept's physical
range.  A confident, unambiguous match becomes a binding; everything else is listed with its three best candidates for a human to decide (nothing is guessed silently), or - with --keep-unmapped -
declared as a vehicle-specific concept so it is still stored, searched and answered.  The draft is validated like any pack and a report says what was mapped, how sure the tool is, and what to review.
Measured: against the three hand-written packs the draft's precision and recall are reported by tests/test_onboard.py."""
import argparse
import json
import os
import re
import shutil

from .. import signal_registry as R
from ..ingest import mapper as MP
from ..protocols import dbc as _dbc, vhal as _vhal
from .profile import Profile, norm_unit

ABBR = {"spd": "speed", "vel": "speed", "velocity": "speed", "veh": "vehicle", "tmp": "temperature", "temp": "temperature", "volt": "voltage", "vlt": "voltage", "curr": "current", "cur": "current", "pres": "pressure",
        "press": "pressure", "prs": "pressure", "trq": "torque", "eng": "engine", "batt": "battery", "bat": "battery", "hv": "hv", "amb": "ambient", "ambair": "ambient", "brk": "brake",
        "str": "steering", "steer": "steering", "whl": "wheel", "chg": "charging", "chrg": "charging", "pwr": "power", "odo": "odometer", "dist": "distance", "cab": "cabin", "clnt": "coolant", "lvl": "level", "aux": "auxiliary",
        "lat": "lateral", "lon": "longitudinal", "long": "longitudinal", "lng": "longitudinal", "yr": "yaw", "regen": "regen", "baro": "barometric", "bar": "barometric", "tyr": "tyre", "tire": "tyre", "tyres": "tyre", "tires": "tyre",
        "outside": "ambient", "pedal": "pedal", "pos": "position", "accel": "accel", "ang": "angle", "hvac": "hvac", "mot": "motor", "mtr": "motor", "fuel": "fuel", "oil": "oil", "rate": "rate", "load": "load", "disc": "disc", "pad": "pad",
        "fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right", "soc": "soc", "tpms": "tyre pressure", "lux": "light", "potential": "potential",
        "hottest": "maximum", "highest": "maximum", "max": "maximum", "coolest": "minimum", "lowest": "minimum", "weakest": "minimum", "min": "minimum", "accelerator": "accel", "acceleration": "accel", "cell": "cell"}
NOISE = {"avg", "drvn", "disp", "raw", "signal", "status", "value", "val", "sensor", "pct", "perc", "act", "actual", "meas", "measured", "cmd", "req", "info", "data", "main", "total", "wheelbased", "based", "high", "res", "hires", "hi",
         "mph", "psi", "kpa", "deg", "dps", "lpm", "c", "f", "kw", "nm", "km", "mi", "m", "v", "a", "s", "g", "l", "h", "one", "two", "the", "of", "for", "and", "current_unit", "pressure_unit"}
UNIT_WORD = {"power": "power", "temperature": "temperature", "pressure": "pressure", "speed": "speed", "voltage": "voltage", "current": "current", "torque": "torque", "flow": "flow", "rpm": "rpm"}
POS = {"fl": ("front", "left"), "fr": ("front", "right"), "rl": ("rear", "left"), "rr": ("rear", "right")}
EXTRA = {"engine_rpm": ("engine speed", "rpm"), "motor_rpm": ("motor speed", "rpm", "motor revolutions"), "battery_soc": ("soc", "state charge", "charge level", "battery level"), "ambient_temperature": ("ambient temperature", "outside temperature", "air temperature"),
         "steering_angle": ("steering wheel angle", "steer angle"), "brake_pedal": ("brake pedal position", "brake pedal"), "accelerator_position": ("accelerator pedal", "accel pedal", "throttle position", "pedal position"),
         "odometer": ("odometer", "distance high resolution", "total distance"), "speed": ("vehicle speed", "wheel speed", "wheel based speed"), "aux_battery_voltage": ("12v battery", "battery potential", "battery voltage 12 v", "supply voltage", "aux battery"),
         "battery_voltage": ("pack voltage", "hv voltage", "high voltage battery voltage", "battery pack voltage", "hv battery voltage")}
PER_WHEEL = {"tyre_pressure": ("tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr")}


def words(name):
    """'HVBatt_Temp_F' -> ['battery', 'temperature'] (+ unit hint 'F' separate)."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name); s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s); s = re.sub(r"[_\.]+", " ", s); s = re.sub(r"(\d+)", r" \1 ", s)
    out = []
    for w in s.lower().split():
        if w.isdigit(): continue
        w = ABBR.get(w, w)
        out += w.split()
    if "lateral" in out or "longitudinal" in out: out = ["accel" if w in ("acc", "accel") else w for w in out]
    if re.search(r"(?<!\d)12[_ ]?v", name.lower()) and "auxiliary" not in out: out.append("auxiliary")                  # "Batt12V": the 12 V battery
    return out


def unit_hint(name):
    m = re.search(r"(?:_|\b)(mph|psi|kpa|f|c|dps|lpm|kw|nm|km|mi)$", name.lower().replace("deg", ""))
    return m.group(1) if m else None


class Scorer:
    def __init__(self):
        self.phrases = {}
        for name, c in R.CONCEPTS.items():
            if name.startswith("tyre_pressure_") or name in ("motor_power_estimate",) or c.kind == "derived": continue
            ps = [c.label] + list(c.synonyms) + list(R.EXTRA_SYNONYMS.get(name, ())) + list(EXTRA.get(name, ()))
            self.phrases[name] = [set(w for w in words_text(p) if w not in NOISE) for p in ps if p]
        for base, cs in PER_WHEEL.items():
            for c in cs:
                pos = POS[c[-2:]]; self.phrases[c] = [set(("tyre", "pressure") + pos)]
        self.phrases["tyre_temperature"] = [{"tyre", "temperature"}]
        self.vocab = set(w for pl in self.phrases.values() for P in pl for w in P) | {"potential", "light", "tyre", "front", "left", "right", "rear", "hv", "auxiliary", "pressure", "state", "charge"}

    def score(self, tokens, unit, lo, hi, ctx=(), ice=None):
        """-> [(score, concept)] best first.  `ctx` = words of the message name (context, used for coverage only)."""
        tokset = set(t for t in tokens if t not in NOISE); out = []
        if unit and R.unit_family(unit) in UNIT_WORD and UNIT_WORD[R.unit_family(unit)] not in tokset and R.unit_family(unit) != "speed": tokset.add(UNIT_WORD[R.unit_family(unit)])
        if not tokset: return out
        full = tokset | set(t for t in ctx if t not in NOISE)
        for c, plist in self.phrases.items():
            best = 0.0
            for P in plist:
                if not P: continue
                inter = len(P & full)
                if not inter: continue
                cov, prec = inter / len(P), len(P & tokset) / len(tokset)
                v = cov * (0.4 + 0.6 * prec) if cov >= 0.99 else 0.55 * cov * (0.5 + 0.5 * prec)
                if prec <= 0.5: v *= 0.8
                best = max(best, v)
            if best <= 0: continue
            best *= 0.65 ** len([t for t in tokset if t not in self.vocab])                                      # a content word no concept knows ("oil") is evidence against a match
            cd = R.CONCEPTS[c]
            if unit not in (None, "none", "") and cd.unit not in ("none", "category"):
                fu, fc = R.unit_family(unit), R.unit_family(cd.unit)
                if fu is not None and fc is not None and fu != fc: best *= 0.3
            if lo is not None and hi is not None and cd.lo is not None and cd.hi is not None and hi > lo:
                over = max(0.0, min(hi, cd.hi) - max(lo, cd.lo)) / min(hi - lo, cd.hi - cd.lo)
                if over < 0.5: best *= 0.85                                                       # a DBC range is the bit field's, not the physical one: a mild signal only
            if ice is not None:
                if ice and c.startswith("motor_") and c != "motor_temperature": best *= 0.9
                if not ice and c.startswith("engine_"): best *= 0.9
                if ice and c == "battery_voltage": best *= 0.9                                                  # no traction battery: a plain 'battery voltage' is the 12 V one
            out.append((best, c))
        return sorted(out, reverse=True)


def words_text(p): return words(p.replace(" ", "_"))


SCORER = None


def powertrain_is_ice(dbcs):
    """No battery / state-of-charge / HV word anywhere in the DBC signal names -> a combustion vehicle (engine_* rather than motor_* concepts).
    'regen' alone means regenerative braking, but with an exhaust word ('DPFRegenStatus', 'FilterRegen') it is the particulate filter cleaning itself - a diesel sign, not an EV one."""
    ev = 0
    for path, _, _ in dbcs:
        db = _dbc.Database.load(path)
        for m in list(db.messages.values()) + list(db.extended_messages.values()):
            for s in m.signals:
                w = set(words(s.name)); ev += bool(w & {"soc", "hv", "cell", "charging"} or ("regen" in w and not w & {"dpf", "particulate", "filter", "soot", "exhaust", "scr", "aftertreatment"}))
    return ev == 0


def draft(pid, dbcs=(), services=None, vhal=False, keep_unmapped=False, base_dir="."):
    """-> (profile dict, report dict).  dbcs: [(path, channel, fd)]; services: path of a SOME/IP catalogue json."""
    global SCORER
    SCORER = SCORER or Scorer(); sources, signals, report = [], [], {"mapped": [], "review": [], "unmapped": []}; used = {}; concepts = []
    dbs = []
    for path, _c, _f in dbcs:
        try: dbs.append(_dbc.Database.load(path))
        except ValueError: dbs.append(_dbc.Database.load(path, lenient=True))                          # a database from outside may carry things the strict reader refuses
    ice = MP.powertrain_is_ice(dbs) if dbs else None                                                    # the buses of one vehicle are judged together
    for i, ((path, chan, fd), db) in enumerate(zip(dbcs, dbs)):
        sid = os.path.splitext(os.path.basename(path))[0].lower().replace("-", "_"); sources.append({"id": sid, "type": "can", "name": f"CAN {os.path.basename(path)}", "dbc": os.path.basename(path), "channel": chan, **({"fd": True} if fd else {})})
        verdict = {(m, s): d for m, s, d in MP.map_database(db, ice=ice)}
        for m in list(db.messages.values()) + list(db.extended_messages.values()):
            for s in m.signals:
                if s.choices and not (s.maximum - s.minimum > 3): pass
                toks = words(s.name); unit = norm_unit(s.unit) if s.unit else (unit_hint(s.name) and norm_unit(unit_hint(s.name))); lo, hi = (s.minimum, s.maximum) if s.minimum < s.maximum else (None, None)
                if m.mux_signal is not None and s is m.mux_signal: continue                                  # the selector of a multiplexed message is not a measurement
                enum = next((l for l in s.choices.values() if "charg" in l.lower()), None) if s.choices else None
                if enum:                                                                                   # a charger state enum: 1 while charging, 0 otherwise
                    signals.append({"concept": "_charging_active", "from": f"can:{sid}/{m.name}.{s.name}", "map": {l: (1 if "charg" in l.lower() and "not" not in l.lower() else 0) for l in s.choices.values()}, "rate_hz": 10.0}); report["mapped"].append({"from": f"can:{sid}/{m.name}.{s.name}", "concept": "_charging_active", "score": 1.0, "alternatives": []}); continue
                _mapped(f"can:{sid}/{m.name}.{s.name}", s.name, toks, verdict.get((m.name, s.name)), s.unit, unit, lo, hi, signals, report, used, concepts, keep_unmapped, rate=10.0)
    if services:
        sources.append({"id": "eth", "type": "someip", "name": "SOME/IP", "catalog": os.path.basename(services)})
        for sk, sv in json.load(open(services, encoding="utf-8")).items():
            for ek, ev in sv["events"].items():
                for f in ev["fields"]:
                    if f.endswith("_text"): continue
                    _decide(f"someip:eth/{sv['name']}.{ev['name']}.{f}", f"{sv['name']} {ev['name']} {f}", words(f"{sv['name']}_{f}") + words(f), None, None, None, signals, report, used, concepts, keep_unmapped, rate=1.0)
    if vhal:
        sources.append({"id": "hal", "type": "vhal", "name": "Android Automotive VHAL"}); props = _vhal.Vhal().props
        for name, p in props.items():
            if p["type"] not in ("FLOAT", "INT32", "INT64") or name.startswith(("INFO_", "HVAC_", "SEAT", "WINDOW", "MIRROR", "DOOR", "LIGHT", "HEADLIGHT", "FOG", "HAZARD", "TURN", "PARKING")): continue
            unit = {"METER_PER_SEC": "m/s", "CELSIUS": "C", "KILOPASCAL": "kPa", "KILOMETER": "km", "WATT_HOUR": "Wh", "PERCENTAGE": "%", "VOLT": "V", "RPM": "rpm"}.get(p.get("unit"), None)
            if unit is None: continue
            _decide(f"vhal:hal/{name}", name, words(name.replace("PERF_", "").replace("ENV_", "")), unit, None, None, signals, report, used, concepts, False, rate=0.2)
    spec = {"schema": "vehiclegpt.profile/1", "id": pid, "vehicle": {"make": "", "model": pid, "powertrain": "bev", "specs": {}}, "sources": sources, "signals": signals, "events": [{"id": "dtc", "from": "uds:dtc"}] if False else [],
            "concepts": concepts, "retention": {"fine_days": 7, "coarse_days": 120, "log_days": 120, "context_days": 365}}
    return spec, report


def _mapped(frm, name, toks, d, db_unit, unit, lo, hi, signals, report, used, concepts, keep, rate):
    """A CAN signal read by the mapper (vgpt.ingest.mapper): a binding with the unit folded into scale / offset, or the reason it was refused."""
    if d is not None and d.concept:
        c = d.concept; prio = used.get(c, 0) + 1; used[c] = prio; b = {"concept": c, "from": frm, "unit": "none", "rate_hz": rate, **MP.binding_fields(d, db_unit)}
        if prio > 1: b["priority"] = prio
        signals.append(b); report["mapped" if d.confidence >= 0.8 else "review"].append({"from": frm, "concept": c, "score": round(d.confidence, 2), "alternatives": [list(a) for a in d.alternatives[:3]]})
        return
    report["unmapped"].append({"from": frm, "alternatives": [list(a) for a in (d.alternatives[:3] if d else [])], "why": d.refused if d else ""})
    if keep and unit not in (None,) and (lo is None or hi is not None):
        cn = "x_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        concepts.append({"name": cn, "label": " ".join(toks) or name, "unit": unit or "none", "category": "vehicle", "synonyms": [" ".join(toks)] if toks else [], **({"lo": lo, "hi": hi} if lo is not None else {})})
        signals.append({"concept": cn, "from": frm, "unit": unit or "none", "rate_hz": rate}); report["unmapped"][-1]["declared_as"] = cn


def _decide(frm, name, toks, unit, lo, hi, signals, report, used, concepts, keep, rate, ctx=(), ice=None):
    ranked = SCORER.score(toks, unit, lo, hi, ctx, ice)
    top = ranked[0] if ranked else (0.0, None); second = ranked[1][0] if len(ranked) > 1 else 0.0
    alts = [(round(s, 2), c) for s, c in ranked[:3]]
    if top[1] and top[0] >= 0.62 and top[0] - second >= 0.075:
        c = top[1]; prio = used.get(c, 0) + 1; used[c] = prio
        b = {"concept": c, "from": frm, "unit": unit or "none", "rate_hz": rate}
        if prio > 1: b["priority"] = prio
        signals.append(b); report["mapped" if top[0] >= 0.8 else "review"].append({"from": frm, "concept": c, "score": round(top[0], 2), "alternatives": alts})
    else:
        report["unmapped"].append({"from": frm, "alternatives": alts})
        if keep and unit not in (None,) and (lo is None or hi is not None):
            cn = "x_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            concepts.append({"name": cn, "label": " ".join(toks) or name, "unit": unit or "none", "category": "vehicle", "synonyms": [" ".join(toks)] if toks else [], **({"lo": lo, "hi": hi} if lo is not None else {})})
            signals.append({"concept": cn, "from": frm, "unit": unit or "none", "rate_hz": rate}); report["unmapped"][-1]["declared_as"] = cn


def write_pack(spec, report, out, dbcs=(), services=None):
    os.makedirs(out, exist_ok=True)
    for path, _, _ in dbcs: shutil.copy(path, os.path.join(out, os.path.basename(path)))
    if services: shutil.copy(services, os.path.join(out, os.path.basename(services)))
    json.dump(spec, open(os.path.join(out, "profile.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    lines = [f"# Onboarding report: {spec['id']}", "", f"{len(report['mapped'])} signals mapped with confidence, {len(report['review'])} to review, {len(report['unmapped'])} unmapped.", ""]
    for key, title in (("review", "Review (mapped, lower confidence)"), ("unmapped", "Unmapped (decide a concept, or leave them stored under their own name)")):
        lines += [f"## {title}", ""] + [f"- `{r['from']}` -> {r.get('concept', '?')}  candidates: {r['alternatives']}" + (f"  declared as `{r['declared_as']}`" if r.get("declared_as") else "") for r in report[key]] + [""]
    open(os.path.join(out, "onboarding_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    Profile.load(out).check()


def evaluate(spec, reference):
    """Precision / recall of a draft against a reference profile, on CAN signals: {from-key: concept}."""
    ref = {}; got = {}
    for b in reference.bindings:
        if b.bus == "can" and not b.when: ref[f"{b.key}"] = b.concept
    for b in spec["signals"]:
        if b["from"].startswith("can:"): got[b["from"].split("/", 1)[1]] = b["concept"]
    ok = sum(1 for k, c in got.items() if ref.get(k) == c); known = {k: c for k, c in ref.items() if c in R.CONCEPTS}                      # `known`: the signals whose meaning the shared registry defines (engine-specific ones a pack declares itself, e.g. particulate filter soot, are not)
    return {"precision": ok / max(1, len(got)), "recall": ok / max(1, len(ref)), "recall_registry": sum(1 for k, c in known.items() if got.get(k) == c) / max(1, len(known)), "mapped": len(got), "reference": len(ref), "declared_by_pack": sorted(set(ref.values()) - set(known.values())),
            "wrong": {k: (c, ref.get(k)) for k, c in got.items() if ref.get(k) != c}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--dbc", action="append", default=[], help="path[:channel[:fd]]"); ap.add_argument("--services"); ap.add_argument("--vhal", action="store_true"); ap.add_argument("--keep-unmapped", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dbcs = []
    for d in a.dbc: p = d.split(":"); dbcs.append((p[0], p[1] if len(p) > 1 else "can0", len(p) > 2 and p[2] == "fd"))
    spec, rep = draft(a.id, dbcs, a.services, a.vhal, a.keep_unmapped); write_pack(spec, rep, a.out, dbcs, a.services)
    print(f"{a.id}: {len(rep['mapped'])} mapped, {len(rep['review'])} to review, {len(rep['unmapped'])} unmapped -> {a.out}")


if __name__ == "__main__":
    main()

"""Read a capture, work out what the vehicle is saying, and write the profile pack that lets the application ingest it: no DBC, no mapping table, no per-vehicle setup.

    analyse(path, dbc=None)              -> Analysis   (what is in the capture, which quantity each signal is, why, with what confidence)
    autoprofile(path, outdir, dbc=None)  -> (Profile, Analysis)   writes outdir/profile.json (+ a synthesised j1939.dbc, or the DBC files given) and validates it

How each kind of traffic is understood:
  * SAE J1939 (29-bit identifiers with parameter groups the catalogue knows): decoded with the catalogue's SPN scales; "not available" / "error" fields are dropped.
  * OBD-II service 01 over ISO-TP (responses from 0x7E8..0x7EF or 0x18DAF1xx): decoded with the J1939-independent PID formulas of the catalogue.
  * CAN with a DBC (given): every signal of every message seen is read by the mapper (vgpt.ingest.mapper: name, unit, bit width, value table, message, the other signals); what it cannot decide is listed with the reason, not guessed.
  * Tables (CSV, JSON, OpenXC, COVESA VSS paths): every column goes through the same mapper, its unit read from the header ('Speed (OBD)(mph)'), converted exactly to the registry's unit.
  * Android VHAL values and NMEA GNSS sentences: by property name / sentence; their units come from the AOSP definitions.
Nothing is invented: a column that cannot be recognised is reported as unmapped and its data is not stored."""
import collections
import json
import os
import statistics

from . import catalog as K
from . import mapper as MP
from . import readers as RD
from . import standards as S
from . import units as U
from .. import signal_registry as R
from ..protocols import canframes, dbc as DBC

ACCEPT = 0.62                                                       # a name match below this is not used
MARGIN = 0.06                                                       # the runner-up must be this much worse, or the values must decide
MAX_SIGNALS = 400
VHAL_UNIT = {"METER_PER_SEC": ("m/s", 1.0), "CELSIUS": ("C", 1.0), "PERCENTAGE": ("%", 1.0), "KILOMETER": ("km", 1.0), "METER": ("m", 1.0), "MILLIWATTS": ("kW", 1e-6), "WATT_HOUR": ("kWh", 1e-3), "VOLT": ("V", 1.0),
             "MILLIAMPERE": ("A", 1e-3), "KILOPASCAL": ("kPa", 1.0), "RPM": ("rpm", 1.0), "DEGREES": ("deg", 1.0), "KILOWATT_HOUR": ("kWh", 1.0), "BAR": ("bar", 1.0)}
VHAL_GEAR = {"1": 0, "2": -1, "4": 0, "8": 1, "16": 1, "32": 2, "64": 3, "128": 4, "256": 5, "512": 6, "1024": 7, "2048": 8, "4096": 9}         # AOSP VehicleGear: neutral, reverse, park, drive, then gears 1..9
WHEEL_AREAS = {1: "fl", 2: "fr", 4: "rl", 8: "rr"}                        # AOSP VehicleAreaWheel: left front, right front, left rear, right rear
DEFAULT_EVENTS = [{"id": "overtemp", "from": "signal:motor_temperature", "when": ">105", "hold_s": 10, "type": "WARN_OVERTEMP", "severity": "critical", "detail": "Coolant above 105 °C ({value:.1f} °C)"},
                  {"id": "low12v", "from": "signal:aux_battery_voltage", "when": "<11.6", "hold_s": 20, "type": "LOW_12V", "severity": "warn", "detail": "12 V battery at {value:.2f} V"}]
LOG_EVENTS = [{"id": "log_reset", "from": "log:any", "match": "watchdog reset", "max_severity": 3, "type": "ECU_RESET", "severity": "warn"}, {"id": "log_brownout", "from": "log:any", "match": "brown-out", "max_severity": 3, "type": "VOLTAGE_DIP", "severity": "warn"}]


class Analysis:
    def __init__(self):
        self.format = None; self.records = 0; self.skipped = 0; self.sources = []; self.signals = []; self.unmapped = []; self.notes = []; self.dbc_files = {}; self.t0 = None; self.t1 = None; self.concepts_declared = []

    def mapped(self): return sorted({s["concept"] for s in self.signals})

    def report(self):
        lines = [f"format {self.format}: {self.records} records" + (f", {self.skipped} lines skipped" if self.skipped else ""), f"sources: " + ", ".join(f"{s['id']} ({s['type']})" for s in self.sources)]
        for s in sorted(self.signals, key=lambda x: (x["concept"], x["from"])):
            conv = "" if s.get("scale", 1.0) == 1.0 and not s.get("offset") else f", x{s.get('scale', 1.0):g}{'%+g' % s['offset'] if s.get('offset') else ''}"
            lines.append(f"  {s['from']:48} -> {s['concept']:28} [{s.get('why', '')}{conv}]")
        if self.unmapped: lines.append("unmapped: " + ", ".join(str(u) for u in self.unmapped[:40]) + (" ..." if len(self.unmapped) > 40 else ""))
        lines += self.notes
        return "\n".join(lines)


def _rate(times):
    d = [b - a for a, b in zip(times, times[1:]) if b > a]
    return round(1.0 / statistics.median(d), 3) if len(d) >= 3 else 1.0


def _stats(vals):
    v = [x for x in vals if x is not None]
    return {"min": min(v), "max": max(v), "mean": sum(v) / len(v)} if v else None


def _binding(concept, frm, unit_from, rate, why, stats=None, conv=None):
    """One profile signal entry: the unit conversion is folded into scale / offset, exactly, so the router needs no unit string."""
    cu = R.CONCEPTS[concept].unit; scale, offset = 1.0, 0.0
    if conv is not None: scale, offset = conv
    elif unit_from:
        f = U.factor_between(unit_from, cu)
        if f is not None: scale, offset = f
    return {"concept": concept, "from": frm, "unit": "none", "scale": scale, "offset": offset, "rate_hz": rate, "why": why, "stats": stats}


def _decide(name, unit, stats, taken, strict=True):
    """The quantity a column or DBC signal is, or None: the best match above ACCEPT whose runner-up is clearly behind (or is decided by the values)."""
    cands = K.match(name, unit, stats, top=4, floor=ACCEPT)
    if not cands: return None, []
    best = cands[0]
    if len(cands) > 1 and best[1] - cands[1][1] < MARGIN and strict:
        # a tie: prefer the quantity nobody has been assigned yet, then the one whose label is a stricter match
        near = [c for c in cands if c[1] >= best[1] - MARGIN]; free = [c for c in near if c[0] not in taken]
        if len(free) == 1: return free[0], cands
        pool = free or near; ov = sorted(((K.label_overlap(name, c[0]), c) for c in pool), key=lambda x: -x[0])
        if len(ov) >= 2 and ov[0][0] > ov[1][0]: return ov[0][1], cands                                          # the concept whose own words the name uses more
        if len(free) == 0 and all(c[0] in taken for c in cands[:2]): return best, cands
        return None, cands
    return best, cands


# ------------------------------------------------------------------------------------------------------------------ the analysis
def analyse(path, dbc=None, sample=60000, anchor=None, fmt=None, powertrain=None):
    from ..vehicle import lake as LK
    LK.release(); K.reset()                                                  # the canonical registry, whatever vehicle this process opened before (an overlay removes the concepts a combustion vehicle lacks)
    st = RD.Stats(); recs = list(RD.read(path, fmt=fmt, anchor=anchor, limit=sample, stats=st))
    return analyse_records(recs, dbc=dbc, powertrain=powertrain, format=st.format, skipped=st.skipped, _reset=False)


def analyse_records(recs, dbc=None, powertrain=None, format="records", skipped=0, _reset=True):
    """The same analysis for records already in memory (`Rec(ts, bus, payload)`): what a live link has sent so far."""
    from ..vehicle import lake as LK
    if _reset: LK.release(); K.reset()
    a = Analysis(); a.format = format; a.records = len(recs); a.skipped = skipped
    stamps = [r.ts for r in recs if r.ts is not None]
    if stamps: a.t0, a.t1 = min(stamps), max(stamps)
    by_bus = collections.defaultdict(list)
    for r in recs: by_bus[r.bus].append(r)
    taken = set(); ice = {"ice": True, "diesel": True, "petrol": True, "bev": False}.get(powertrain, "auto")
    if "can" in by_bus: _can(a, by_bus["can"], dbc, taken, ice)
    if "table" in by_bus: _table(a, by_bus["table"], taken, ice)
    if "vhal" in by_bus: _vhal(a, by_bus["vhal"], taken)
    if "gnss" in by_bus: a.sources.append({"id": "gnss", "type": "gnss", "name": "GNSS (NMEA 0183)"})
    if "log" in by_bus: a.sources.append({"id": "log", "type": "log", "name": "text / DLT logs", "format": "rfc3164"})
    if not a.signals and "gnss" not in by_bus and "log" not in by_bus: a.notes.append("nothing in this capture could be mapped to a vehicle quantity")
    a.powertrain = powertrain or _powertrain(a.mapped())
    return a


def _powertrain(concepts):
    c = set(concepts)
    if c & {"battery_soc", "battery_voltage", "battery_current", "motor_torque", "charging_power", "charging_state"} and c & {"engine_rpm", "fuel_rate", "fuel_level", "engine_load"}:
        return "phev" if c & {"charging_power", "charging_state"} else "hev"                                      # a traction battery AND an engine: a hybrid keeps both sets of questions (a combustion profile would hide the battery)
    if c & {"dpf_soot_load", "def_level", "egr_position", "boost_pressure", "exhaust_gas_temperature", "fuel_rail_pressure"}: return "diesel"
    if c & {"lambda", "ignition_timing", "short_term_fuel_trim", "catalyst_temperature", "intake_manifold_pressure", "mass_air_flow"}: return "petrol"
    if c & {"battery_soc", "battery_voltage", "battery_current", "motor_torque", "charging_power"} and not (c & {"engine_rpm", "fuel_rate", "fuel_level"}): return "bev"
    if c & {"engine_rpm", "fuel_rate", "fuel_level", "engine_load"}: return "ice"
    return "unknown"


def _can(a, recs, dbc_files, taken, ice="auto"):
    frames = [r.payload for r in recs]; n = len(frames); times = collections.defaultdict(list)
    obd_ids = {f.can_id for f in frames if (not f.extended and 0x7E8 <= f.can_id <= 0x7EF) or (f.extended and (f.can_id & 0x1FFFFF00) == 0x18DAF100)}
    diag = [f for f in frames if f.can_id in obd_ids]; rest = [f for f in frames if f.can_id not in obd_ids and not (0x7DF == f.can_id or 0x7E0 <= f.can_id <= 0x7E7)]
    # OBD-II service 01
    if diag:
        S.install_obd(); from ..protocols import obd as _obd, isotp as _isotp
        reasm = {}; seen = collections.defaultdict(list)
        for r in recs:
            f = r.payload
            if f.can_id not in obd_ids: continue
            msg = reasm.setdefault(f.can_id, _isotp.Reassembler()).feed(f.data, now=r.ts)
            if msg is not None:
                o = _obd.decode(msg)
                if o and "error" not in o and o.get("value") is not None: seen[o["pid"]].append((r.ts, o["value"]))
        tbl = K.obd_table(); names = S.install_obd()
        if seen:
            a.sources.append({"id": "diag", "type": "uds", "name": "OBD-II service 01 over ISO-TP", "response_ids": ["0x7E8..0x7EF"]})
            for pid, pts in sorted(seen.items()):
                if pid not in names: a.unmapped.append(f"OBD PID {pid:02X}"); continue
                name, concept = names[pid]
                if concept not in R.CONCEPTS: a.unmapped.append(f"OBD PID {pid:02X} ({concept})"); continue
                b = _binding(concept, f"uds:diag/obd_{name}", None, _rate([p[0] for p in pts if p[0] is not None]), f"OBD-II PID {pid:02X}", _stats([p[1] for p in pts]))
                b["unit_native"] = R.CONCEPTS[concept].unit; a.signals.append(b); taken.add(concept)
    # J1939 and DBC-described CAN
    ext = [f for f in rest if f.extended]
    if dbc_files:
        paths = list(dbc_files) if isinstance(dbc_files, (list, tuple)) else [dbc_files]; dbs = []
        for p in paths:
            try: dbs.append(DBC.Database.load(p))
            except ValueError: dbs.append(DBC.Database.load(p, lenient=True))                          # a database from outside may carry things the strict reader refuses
        if ice == "auto": ice = MP.powertrain_is_ice(dbs)                                              # the buses of one vehicle are judged together
        for i, (p, db) in enumerate(zip(paths, dbs)):
            sid = f"can{i}"; a.dbc_files[sid] = p; a.sources.append({"id": sid, "type": "can", "name": f"CAN ({os.path.basename(p)})", "dbc": os.path.basename(p)})
            _dbc_signals(a, sid, db, rest, taken, ice)
    elif rest and S.looks_like_j1939(rest):
        known = K.j1939_table(); pgns = {S.j1939_fields(f.can_id)[1] for f in ext} & set(known); text, names = S.j1939_dbc(pgns)
        a.dbc_files["j1939"] = text; a.sources.append({"id": "j1939", "type": "can", "name": "SAE J1939", "dbc": "j1939.dbc", "j1939": True})
        t = collections.defaultdict(list)
        for r in recs:
            f = r.payload
            if f.extended and f.can_id not in obd_ids:
                pgn = S.j1939_fields(f.can_id)[1]
                if pgn in known: t[pgn].append(r.ts)
        for pgn in sorted(pgns):
            for concept, spn, start, ln, scale, offset in known[pgn]:
                if concept not in R.CONCEPTS or concept in taken: continue
                a.signals.append(_binding(concept, f"can:j1939/J1939_{pgn}.SPN{spn}_{concept}", None, _rate([x for x in t[pgn] if x is not None]), f"J1939 PGN {pgn} SPN {spn}")); taken.add(concept)
        other = {S.j1939_fields(f.can_id)[1] for f in ext} - set(known)
        if other: a.notes.append(f"{len(other)} J1939 parameter groups are not in the catalogue and are not stored (e.g. " + ", ".join(str(x) for x in sorted(other)[:6]) + ")")
    elif rest:
        ids = collections.Counter((f.can_id, f.extended) for f in rest)
        a.unmapped.append(f"{len(ids)} CAN identifiers with no DBC (give one with --dbc)")


def _redundant(a, b):
    """A second source of a quantity already bound is the redundant one: it gets the next priority (the router compares the two and raises a conflict when they disagree)."""
    n = sum(1 for s in a.signals if s["concept"] == b["concept"])
    if n: b["priority"] = n + 1


def _mapped_binding(d, frm, unit, rate, stats):
    """A profile signal from the mapper's decision: the unit conversion (the DBC's unit, or one read from the name) and a fraction's scale folded into scale / offset, a value table or a polarity flip as a map."""
    f = MP.binding_fields(d, unit)
    b = _binding(d.concept, frm, None, rate, "; ".join(d.reasons) or "mapper", stats, conv=(f.get("scale", 1.0), f.get("offset", 0.0)))
    if "map" in f: b["map"] = f["map"]
    if f.get("snap"): b["snap"] = True
    return b


def _dbc_signals(a, sid, db, frames, taken, ice="auto"):
    verdict = {(m, s): d for m, s, d in MP.map_database(db, ice=ice)}
    msgs = {}; seen = collections.defaultdict(list)
    for d in (db.messages, db.extended_messages):
        for m in d.values(): msgs[(m.frame_id, m.extended)] = m
    for f in frames:
        m = msgs.get((f.can_id & 0x1FFFFFFF, f.extended))
        if m is None or len(f.data) != m.dlc: continue
        try: vals = m.decode_with_metadata(f.data)
        except ValueError: continue
        for name, d in vals.items(): seen[(m.name, name)].append((f.ts, d.physical, d.unit))
    for (mname, sname), pts in sorted(seen.items()):
        unit = pts[0][2]; st = _stats([p[1] for p in pts]); d = verdict.get((mname, sname))
        if d is None or d.concept is None:
            a.unmapped.append(f"{mname}.{sname}" + (f" ({d.refused})" if d is not None and d.refused else "")); continue
        b = _mapped_binding(d, f"can:{sid}/{mname}.{sname}", unit, _rate([p[0] for p in pts if p[0] is not None]), st)
        _redundant(a, b)
        a.signals.append(b); taken.add(d.concept)


def _table(a, recs, taken, ice="auto"):
    cols = collections.defaultdict(list); times = collections.defaultdict(list); vss = K.vss_table()
    for r in recs:
        for k, v in r.payload.items():
            x = RD.num(v)
            if x is not None: cols[k].append(x); times[k].append(r.ts)
    if not cols: return
    a.sources.append({"id": "tab", "type": "table", "name": "table (CSV / JSON)"})
    if ice == "auto": ice = MP.powertrain_from_names([U.split_header(n)[0] for n in cols])                  # the columns say whether there is an engine or a traction battery
    for name, vals in sorted(cols.items(), key=lambda kv: kv[0]):
        if len(a.signals) >= MAX_SIGNALS: a.notes.append("more columns than the limit; the rest are ignored"); break
        base, unit = U.split_header(name); st = _stats(vals); rate = _rate([t for t in times[name] if t is not None])
        concept = vss.get(name.lower()); why = "COVESA VSS path"; d = None
        if concept is None:
            flag = set(vals) <= {0, 1, 0.0, 1.0}
            d = MP.decide(MP.Evidence(name=base, unit=unit, bits=1 if flag else None, stats=st, ice=ice))
            if d.concept is None:
                a.unmapped.append(name + (f" ({d.refused})" if d.refused else "")); continue
            concept, why = d.concept, "; ".join(d.reasons)
        if concept not in R.CONCEPTS: continue
        if unit and U.factor_between(unit, R.CONCEPTS[concept].unit) is None: a.unmapped.append(f"{name} (unit {unit} does not fit {concept})"); continue
        if d is not None and (d.transform.get("map") or d.transform.get("invert01") or d.transform.get("scale") or d.transform.get("unit")):
            b = _mapped_binding(d, f"table:tab/{name}", unit, rate, st)
            _redundant(a, b); a.signals.append(b); taken.add(concept); continue
        b = _binding(concept, f"table:tab/{name}", unit, rate, why + (f"; unit {unit}" if unit else "; no unit in the name: taken as " + R.CONCEPTS[concept].unit), st, None)
        _redundant(a, b); a.signals.append(b); taken.add(concept)


def _vhal(a, recs, taken):
    from ..protocols import vhal as V
    vh = V.Vhal(); names = K.vhal_names(); a.sources.append({"id": "hal", "type": "vhal", "name": "Android Vehicle HAL"}); seen = collections.defaultdict(list); wheels = collections.defaultdict(list)
    cap = None
    for r in recs:
        n = vh.by_id.get(r.payload.get("prop"))
        if n in ("INFO_EV_BATTERY_CAPACITY", "EV_CURRENT_BATTERY_CAPACITY") and r.payload.get("float"): cap = float(r.payload["float"][0]) if n == "INFO_EV_BATTERY_CAPACITY" or cap is None else cap     # the pack's size in Wh
        if n == "TIRE_PRESSURE" and r.payload.get("area") in WHEEL_AREAS: wheels[r.payload["area"]].append(r.ts)            # one reading per wheel: the AOSP wheel areas
        elif n: seen[n].append(r.ts)
    for area, ts in sorted(wheels.items()):
        concept = f"tyre_pressure_{WHEEL_AREAS[area]}"; unit, k = VHAL_UNIT.get(vh.props["TIRE_PRESSURE"]["unit"], (None, 1.0))
        if concept in R.CONCEPTS and concept not in taken: a.signals.append(_binding(concept, f"vhal:hal/TIRE_PRESSURE@{area}", unit, _rate([x for x in ts if x is not None]), f"Android VHAL TIRE_PRESSURE, wheel area {area}")); taken.add(concept)
    for n, ts in sorted(seen.items()):
        concept = names.get(n)
        if concept is None or concept not in R.CONCEPTS or concept in taken: a.unmapped.append(f"VHAL {n}"); continue
        unit, k = VHAL_UNIT.get(vh.props[n]["unit"], (None, 1.0))
        if unit is None and vh.props[n]["unit"] not in ("", "OTHER", None): a.unmapped.append(f"VHAL {n} (unit {vh.props[n]['unit']})"); continue
        b = _binding(concept, f"vhal:hal/{n}", unit, _rate([x for x in ts if x is not None]), f"Android VHAL {n}")
        if vh.props[n]["unit"] == "WATT_HOUR" and R.CONCEPTS[concept].unit == "%":                                                  # an energy in Wh is a state of charge only relative to the pack's size
            if not cap: a.unmapped.append(f"VHAL {n} (an energy in Wh; the battery capacity has not been seen, so no percentage can be given)"); continue
            b["scale"], b["offset"] = 100.0 / cap, 0.0; b["why"] += f"; Wh of a {cap:g} Wh pack as a percentage"
        elif vh.props[n]["unit"] in ("MILLIWATTS", "WATT_HOUR", "MILLIAMPERE"): b["scale"], b["offset"] = k, 0.0                  # the native unit is a thousandth / millionth of the registry unit
        elif unit is not None:
            f = U.factor_between(unit, R.CONCEPTS[concept].unit)
            if f is not None: b["scale"], b["offset"] = f
        if n in ("CURRENT_GEAR", "GEAR_SELECTION") and concept == "gear": b["map"] = dict(VHAL_GEAR)                                       # the HAL's gear is a bit flag, not the gear number the registry holds
        a.signals.append(b); taken.add(concept)


# ------------------------------------------------------------------------------------------------------------------ the pack
def build_spec(a, vehicle_id, model=None):
    concepts = {s["concept"] for s in a.signals}; ptr = a.powertrain
    spec = {"schema": "vehiclegpt.profile/1", "id": vehicle_id, "vehicle": {"make": "", "model": model or vehicle_id, "year": 0, "powertrain": ptr, "specs": {"veh.model": model or vehicle_id}},
            "concepts": [], "sources": a.sources, "signals": [], "events": [], "thresholds": {"overtemp_c": 105, "tyre_low_kpa": 200, "low_12v": 11.6},
            "retention": {"fine_days": 7, "coarse_days": 120, "log_days": 120, "context_days": 365}}
    for s in a.signals:
        e = {"concept": s["concept"], "from": s["from"], "unit": s.get("unit", "none"), "rate_hz": s["rate_hz"]}
        if s.get("scale", 1.0) != 1.0: e["scale"] = s["scale"]
        if s.get("offset"): e["offset"] = s["offset"]
        for k in ("map", "snap", "priority"):
            if s.get(k): e[k] = s[k]
        spec["signals"].append(e)
    spec["events"] = [e for e in DEFAULT_EVENTS if e["from"].split(":", 1)[1] in concepts]
    if any(x["type"] == "log" for x in a.sources): spec["events"] += LOG_EVENTS
    for w in ("fl", "fr", "rl", "rr"):
        if f"tyre_pressure_{w}" in concepts: spec["events"].append({"id": f"tpms_{w}", "from": f"signal:tyre_pressure_{w}", "when": "<200", "hold_s": 60, "type": "TPMS_LOW", "severity": "warn", "detail": "tyre pressure low ({value:.0f} kPa)"})
    return spec


def write_pack(a, outdir, vehicle_id="captured_vehicle", model=None):
    from ..vehicle.profile import Profile
    os.makedirs(outdir, exist_ok=True); spec = build_spec(a, vehicle_id, model)
    for sid, content in a.dbc_files.items():
        name = "j1939.dbc" if sid == "j1939" else os.path.basename(content)
        if sid == "j1939": open(os.path.join(outdir, name), "w", encoding="utf-8", newline="\n").write(content)
        else: open(os.path.join(outdir, name), "wb").write(open(content, "rb").read())
    json.dump(spec, open(os.path.join(outdir, "profile.json"), "w", encoding="utf-8"), indent=1)
    p = Profile.load(outdir); p.check(); return p


def autoprofile(path, outdir, dbc=None, vehicle_id=None, model=None, **kw):
    a = analyse(path, dbc=dbc, **kw); vid = vehicle_id or ("capture_" + os.path.splitext(os.path.basename(path))[0].replace(" ", "_").replace("-", "_").lower()[:40])
    if not a.signals: raise ValueError("no vehicle quantity could be recognised in this capture:\n" + a.report())
    return write_pack(a, outdir, vid, model or os.path.basename(path)), a

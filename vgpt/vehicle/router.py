"""ProfileRouter: raw traffic of ANY vehicle -> canonical records, driven only by the vehicle's profile pack.

    (bus, payload, ts)  --CAN/CAN-FD (DBC per source) | SOME/IP (catalogue) | Android VHAL | UDS+OBD (ISO-TP) | logs (QNX slog2info / logcat / syslog) | camera | GNSS-->
    ("sig", concept, value in the concept's canonical unit) | ("event", type, severity, detail) | ("log", source, text, ts_text) | ("sig_ts", ...)  | ("thumb", png)

What it adds on top of the fixed-signal adapters of vgpt/adapters.py:
  * every mapping (message/signal/field/property -> concept, unit, scale, offset, enum map) comes from the profile, so a new vehicle is data;
  * redundant sources: two sources of one concept (CAN primary, VHAL secondary) - the secondary is used when the primary is stale, and when both are fresh and disagree for `hold` seconds a
    SENSOR_CONFLICT event is recorded with both values (the "two sources report conflicting values" question of the case study);
  * plausibility: a decoded value outside the concept's physical range is dropped and counted (SENSOR_INVALID event, rate-limited), never stored;
  * signal-derived events: warnings a vehicle does not publish natively (coolant above 105 C for 10 s) are declared in the profile and raised here;
  * per-source statistics (frames, decoded, unknown, rejected, malformed, last seen) for source-health monitoring.
Listen-only like the adapters: nothing here can transmit onto a bus."""
import collections
import re
import math
import struct

from .. import adapters, signal_registry as R
from ..protocols import canopen as _co, dbc as _dbc, doip as _doip, flexray as _fx, isotp as _isotp, j1939 as _jtp, lin as _lin, logs as _logs, obd as _obd, someip as _someip, uds as _uds, vhal as _vhal, canframes as _canframes
from .profile import Profile
from ..ingest import standards as _j1939

ALIAS = {"speed": "speed", "battery_soc": "soc", "battery_voltage": "pack_v", "battery_current": "pack_a", "battery_power": "pack_kw", "battery_temperature": "batt_temp", "coolant_flow": "coolant_flow",
         "ambient_temperature": "amb_temp", "hvac_power": "hvac_w", "motor_temperature": "motor_temp", "charging_power": "chg_kw", "_ignition": "ign", "_charging_active": "chg_state",
         "camera_blockage": "cam_block", "camera_sharpness": "cam_sharp", "camera_brightness": "cam_luma", "lane_confidence": "lane_conf"}
INVERSE = {v: k for k, v in ALIAS.items()}
LEGACY_TO_CONCEPT = dict(INVERSE, obd_speed="speed", obd_ambient_temp="ambient_temperature")


def _ttl(b): return float(b.ttl) if b.ttl else max(2.0, 3.0 / max(b.rate_hz, 0.01))


class ProfileRouter:
    wants_ts = True

    def __init__(self, profile, health=None, conflict_hold=5.0, conflict_gap=120.0):
        self.p = profile if isinstance(profile, Profile) else Profile.load(profile)
        self.health, self.hold, self.gap = health, conflict_hold, conflict_gap
        self.alias = dict(ALIAS)                                                          # concept -> the store's internal column name (the nine core signals); every other concept goes to the lake
        self.ttl = {}
        for b in self.p.bindings: self.ttl[b.concept] = max(self.ttl.get(b.concept, 0.0), _ttl(b))
        self.stats = collections.defaultdict(collections.Counter); self.last_seen = {}; self.dropped = 0
        self.slots = {}; self._conf = {}; self._invalid_at = {}; self._sigstate = {}; self.error_classes = collections.Counter(); self._last_err = -1e9
        self._can = {}; self._sip = {}; self._vhal = {}; self._cam = {}; self._rules = {"someip": {}, "log": [], "signal": collections.defaultdict(list), "uds": []}
        self._fxs = {}; self._lin = {}; self._co = {s.id: {"nodes": set(self.p.canopen_nodes(s.id)), "state": {}, "channel": s.raw.get("channel")} for s in self.p.sources.values() if s.type == "can" and s.raw.get("canopen")}
        for src in self.p.sources.values():
            if src.type == "can": self._index_can(src)
            elif src.type == "lin": self._index_lin(src)
            elif src.type == "flexray": self._index_flexray(src)
        for b in self.p.bindings:
            if b.bus == "someip": self._sip.setdefault(tuple(b.key.split(".")[:2]), {}).setdefault(b.key.split(".")[2], []).append(b)
            elif b.bus == "vhal": self._vhal.setdefault(b.key.split("@")[0], []).append(b)
            elif b.bus == "camera": self._cam.setdefault(b.key, []).append(b)
        for e in self.p.events:
            if e.origin.startswith("someip:"): self._rules["someip"].setdefault(tuple(e.origin[7:].split(".")[:2]), []).append(e)
            elif e.origin.startswith("log:"): self._rules["log"].append(e)
            elif e.origin.startswith("signal:"): self._rules["signal"][e.origin[7:]].append(e)
            elif e.origin.startswith("uds:"): self._rules["uds"].append(e)
        uds = [s for s in self.p.sources.values() if s.type == "uds"]
        self.diag_ids = set()
        for s in uds:
            for x in s.raw.get("response_ids", ["0x7E8..0x7EF"]):
                if isinstance(x, str) and ".." in x: lo, hi = (int(t, 16) for t in x.split("..")); self.diag_ids |= set(range(lo, hi + 1))
                else: self.diag_ids.add(int(x, 16) if isinstance(x, str) else int(x))
        self._tp = {}; self._dtc_seen = {}; self.has_diag = bool(uds)
        self._jtp = _jtp.Reassembler(); self._jfaults = {}; self.j1939_vin = {}; self.j1939_components = {}                    # J1939 transport (BAM / RTS-CTS / extended) rebuilt from the wire; active DM1 faults per source address
        self._episodes = {}; self._prev_speed = None; self._trans = {}
        self.has_long = any(b.concept == "longitudinal_acceleration" for b in self.p.bindings)
        self._someip_rem = b""; self._vhal_dec = _vhal.Vhal(); self._camera = None; self._nmea = adapters.NmeaAdapter()
        _j1939.install_obd()                                                                  # every service-01 parameter of the universal catalogue (vgpt/ingest/catalog.json)
        self._obd = {b.key: b for b in self.p.bindings if b.bus == "uds" and b.key.startswith("obd_")}
        self._j1939 = {s.id: (set(s.raw["sa"]) if s.raw.get("sa") else None) for s in self.p.sources.values() if s.type == "can" and s.raw.get("j1939")}          # SAE J1939 sources: matched by parameter group, any source address (or the listed ones)
        self._jbus = bool(self._j1939) or any(src.type == "can" and "j1939" in str(src.raw.get("dbc", "")).lower() for src in self.p.sources.values())
        self._tab = {}
        for b in self.p.bindings:
            if b.bus == "table": self._tab.setdefault(b.key, []).append(b)
        self._dids = {did: v for did, v in self.p.did_fmt.items()}
        self.offered = {}

    # ------------------------------------------------------------------------------------------------ indexes
    def _index_can(self, src):
        db = self.p.dbc(src.id); idx = {}
        for b in self.p.bindings:
            if b.bus != "can" or b.source != src.id: continue
            msg, _, sig = b.key.partition(".")
            for d in (db.messages, db.extended_messages):
                for m in d.values():
                    if m.name == msg: idx.setdefault((m.frame_id, m.extended), (m, {}))[1].setdefault(sig, []).append(b)
        self._can[src.id] = (db, idx, src.raw.get("channel"))

    def _index_flexray(self, src):
        desc = self.p.flexray(src.id); idx = {}
        for b in self.p.bindings:
            if b.bus != "flexray" or b.source != src.id: continue
            fname, _, sig = b.key.partition("."); fr = desc.frames.get(fname)
            if fr is not None: idx.setdefault(fr.slot, (fr, {}))[1].setdefault(sig, []).append(b)
        self._fxs[src.id] = (desc, idx)

    def _index_lin(self, src):
        ldf = self.p.ldf(src.id); idx = {}
        for b in self.p.bindings:
            if b.bus != "lin" or b.source != src.id: continue
            fname, _, sig = b.key.partition("."); fr = ldf.frames.get(fname)
            if fr is not None: idx.setdefault(fr.id, (fr, {}))[1].setdefault(sig, []).append(b)
        self._lin[src.id] = (ldf, idx)

    # ------------------------------------------------------------------------------------------------ entry point
    def decode(self, bus, payload, ts=None):
        try:
            if bus == "can": return self._decode_can(payload, ts)
            if bus == "table": return self._decode_table(payload, ts)
            if bus == "lin": return self._decode_lin(payload, ts)
            if bus == "doip": return self._decode_doip(payload, ts)
            if bus == "flexray": return self._decode_flexray(payload, ts)
            if bus == "someip": return self._decode_someip(payload, ts)
            if bus == "vhal": return self._decode_vhal(payload, ts)
            if bus == "log": return self._decode_log(payload, ts)
            if bus == "cam": return self._decode_cam(payload, ts)
            if bus == "gnss":
                self._seen("gnss", ts); return self._nmea.decode(payload) if isinstance(payload, str) else []
        except (struct.error, KeyError, ValueError, TypeError, IndexError, AttributeError, _someip.SomeIpError): self.dropped += 1; return []
        return []

    def _seen(self, source, ts):
        self.stats[source]["frames"] += 1
        if ts is not None: self.last_seen[source] = ts

    # ------------------------------------------------------------------------------------------------ emission with redundancy / plausibility / conflicts / signal events
    def _emit(self, b, raw, ts, out, label=None):
        if isinstance(raw, bool) or raw is None: return
        try: x = b.canonical(raw)
        except (ValueError, TypeError): self.stats[b.source]["rejected"] += 1; return
        if b.when and math.isfinite(x): x = 1.0 if (x > b.when[1] if b.when[0] == ">" else x < b.when[1]) else 0.0
        if not math.isfinite(x): self.stats[b.source]["rejected"] += 1; return
        c = b.concept; cd = R.CONCEPTS.get(c); lo, hi = b.valid if b.valid else ((cd.lo, cd.hi) if cd is not None else (None, None))
        if cd is not None and lo is not None and hi is not None and not (lo <= x <= hi):
            self.stats[b.source]["rejected"] += 1
            if ts is not None and ts - self._invalid_at.get(c, -1e9) > 300:
                self._invalid_at[c] = ts; out.append(("event", "SENSOR_INVALID", "warn", f"{cd.label} reported {x:g} {cd.unit} outside its plausible range ({lo:g} to {hi:g}) by {b.source}/{b.key}; value dropped"))
            return
        self.stats[b.source]["decoded"] += 1
        slot = self.slots.setdefault(c, {}); slot[b.priority] = (x, ts, b)
        self._conflict(c, b, x, ts, slot, out)
        if ts is not None and any(p < b.priority and slot[p][1] is not None and ts - slot[p][1] <= _ttl(slot[p][2]) for p in slot): return        # a fresher higher-priority source speaks for this concept
        out.append(("sig", c, x)); self._signal_events(c, x, ts, out); self._derived_events(c, x, ts, out)

    def _derived_events(self, c, x, ts, out):
        """Driving events the vehicle does not publish: hard braking / acceleration / cornering from the acceleration signals (or from the speed trace when the vehicle has none), and state transitions."""
        if ts is None: return
        G = 9.80665
        if c == "longitudinal_acceleration": self._episode("HARSH_BRAKE", -x / G, ts, out, 0.30); self._episode("HARSH_ACCEL", x / G, ts, out, 0.30)
        elif c == "lateral_acceleration": self._episode("HARSH_CORNER", abs(x) / G, ts, out, 0.35)
        elif c == "speed" and not self.has_long:
            if self._prev_speed is not None and 0.5 <= ts - self._prev_speed[1] <= 2.5:
                a = (x - self._prev_speed[0]) / 3.6 / (ts - self._prev_speed[1]); self._episode("HARSH_BRAKE", -a / G, ts, out, 0.30); self._episode("HARSH_ACCEL", a / G, ts, out, 0.30)
            self._prev_speed = (x, ts)
        if c in self.p.transitions:
            prev = self._trans.get(c); cur = 1 if x >= 0.5 else 0; self._trans[c] = cur
            if prev is not None and prev != cur: out.append(("event", "ADAS_STATE_CHANGE" if c == "adas_state" else "STATE_CHANGE", "info", f"{R.label_of(c)} {'on' if cur else 'off'}"))

    def _episode(self, ty, g, ts, out, thr):
        ep = self._episodes.setdefault(ty, {"on": False, "peak": 0.0, "t": 0.0, "v": 0.0, "last": 0.0})
        if g >= thr:
            ep["last"] = ts
            if not ep["on"]: ep.update(on=True, peak=g, t=ts, v=(self.slots.get("speed", {}).get(1) or (0.0,))[0])
            elif g > ep["peak"]: ep.update(peak=g, t=ts, v=(self.slots.get("speed", {}).get(1) or (0.0,))[0])
        elif ep["on"] and ts - ep["last"] >= 0.5:
            ep["on"] = False; out.append(("event", ty, "info", f"{ep['peak']:.2f} g at {ep['v']:.0f} km/h"))

    def _conflict(self, c, b, x, ts, slot, out):
        if ts is None or len(slot) < 2: return
        others = [(p, v) for p, v in slot.items() if p != b.priority and v[1] is not None and ts - v[1] <= _ttl(v[2])]
        if not others: return
        p, (y, ty, by) = min(others); cd = R.CONCEPTS.get(c)
        tol = b.conflict_abs or by.conflict_abs or (0.05 * (cd.hi - cd.lo) if cd and cd.lo is not None and cd.hi is not None else 0.05 * max(abs(x), abs(y), 1.0))
        st = self._conf.setdefault(c, {"since": None, "last": -1e9})
        if abs(x - y) <= tol: st["since"] = None; return
        if st["since"] is None: st["since"] = ts
        if ts - st["since"] >= self.hold and ts - st["last"] >= self.gap:
            st["last"] = ts; lab = cd.label if cd else c; u = cd.unit if cd else ""
            first, second = ((b, x), (by, y)) if b.priority < by.priority else ((by, y), (b, x))
            out.append(("event", "SENSOR_CONFLICT", "warn", f"{lab}: {first[0].bus}/{first[0].key} reads {first[1]:g} {u} but {second[0].bus}/{second[0].key} reads {second[1]:g} {u} (differ by {abs(x - y):.3g} for more than {self.hold:g} s)"))

    def _signal_events(self, c, x, ts, out):
        for e in self._rules["signal"].get(c, ()):
            thr = float(e.when[1:].replace("=", "")); hit = x > thr if e.when[0] == ">" else x < thr
            st = self._sigstate.setdefault(e.id, {"since": None, "fired": False})
            if not hit: st["since"] = None; st["fired"] = False; continue
            if ts is None: continue
            if st["since"] is None: st["since"] = ts
            if not st["fired"] and ts - st["since"] >= e.hold_s:
                st["fired"] = True; cd = R.CONCEPTS.get(c)
                out.append(("event", e.type, e.severity, (e.detail or "{label} {value:.1f} {unit}").format(label=cd.label if cd else c, value=x, unit=cd.unit if cd else "", thr=thr)))

    # ------------------------------------------------------------------------------------------------ CAN / CAN-FD / diagnostics
    def _decode_can(self, payload, ts):
        if isinstance(payload, _canframes.Frame):
            f = payload; can_id, data, fd, ext, chan = f.can_id, f.data, f.fd, f.extended, f.channel
            ts = f.ts if (ts is None and f.ts) else ts
            if f.error_frame:
                self.stats["can"]["error_frames"] += 1; self.error_classes[f.error_class] += 1
                return [("event", "CAN_ERROR", "warn", f"CAN error frame class 0x{f.error_class:X} on {f.channel or 'bus'}")] if self._err_ok(ts) else []
        else:
            can_id, data = payload[0], payload[1]; fd = payload[2] if len(payload) > 2 else False; ext = payload[3] if len(payload) > 3 else None; chan = ""
        ident = can_id & 0x1FFFFFFF; ext = (can_id > 0x7FF) if ext is None else bool(ext) or can_id > 0x7FF
        if not ext and (ident == 0x7DF or 0x7E0 <= ident <= 0x7E7): return []                      # a scan tool's request on the diagnostic port: the answer carries the data, the request carries none
        if not ext and (ident == 0x7DF or 0x7E0 <= ident <= 0x7E7): return []                      # a scan tool's request on the diagnostic port: the answer carries the data, the request carries none
        if self._co and not ext and not fd:
            r = self._decode_canopen(ident, data, chan, ts)
            if r is not None: return r
        if self.has_diag and ident in self.diag_ids and not ext:
            self._seen("diag", ts); return self._decode_diag(ident, data, ts)
        out = []; hit = False
        if ext and self._jbus:
            _, jpgn, _, jsa = _jtp.fields(ident)
            if jpgn in (_jtp.TP_CM, _jtp.TP_DT, _jtp.ETP_CM, _jtp.ETP_DT):                                                  # a piece of a longer message: rebuild it, then decode it like any other message of that parameter group
                before = sum(self._jtp.errors.values()); done = self._jtp.feed(ident, data, ts); self.stats["can"]["tp_errors"] += sum(self._jtp.errors.values()) - before
                for mpgn, msa, mda, mdata in done:
                    self.stats["can"]["j1939_long"] += 1; out += self._decode_can((_jtp.encode_id(6, mpgn, msa, mda), mdata, False, True), ts)
                return out
            if jpgn in (_jtp.PGN_DM1, _jtp.PGN_DM2, _jtp.PGN_VIN, _jtp.PGN_COMPONENT_ID): out += self._j1939_diag(jpgn, jsa, data, ts)
        for sid, (db, idx, channel) in self._can.items():
            if channel and chan and channel != chan: continue
            j = sid in self._j1939
            if j:
                if not ext or (self._j1939[sid] is not None and (ident & 0xFF) not in self._j1939[sid]): continue
                ent = idx.get((_j1939.canonical_id(ident), True))
            else: ent = idx.get((ident, ext))
            if ent is None: continue
            hit = True; self._seen(sid, ts); m, sigs = ent
            if len(data) != m.dlc: self.stats[sid]["malformed"] += 1; continue
            vals = m.decode_with_metadata(data)
            lens = m.__dict__.get("_lens") or m.__dict__.setdefault("_lens", {x.name: x.length for x in m.signals}) if j else None
            for name, binds in sigs.items():
                d = vals.get(name)
                if d is None: continue
                if j and _j1939.is_na(d.raw, lens[name]): self.stats[sid]["not_available"] += 1; continue          # J1939 sends 'not available' and 'error' as all-ones fields: never a reading
                for b in binds:
                    if b.map is not None:
                        v = b.map.get(d.enum) if d.enum is not None and d.enum in b.map else b.map.get(str(d.raw))
                        if v is not None: self._emit(b, v, ts, out)
                    else: self._emit(b, d.physical, ts, out)
        if not hit:
            self.stats["can"]["unknown"] += 1
            if self.health is not None: return [("canraw", can_id, len(data))]
        return out

    def _decode_canopen(self, ident, data, chan, ts):
        """The management traffic of a CANopen bus (heartbeat, emergency, SDO, NMT, SYNC, TIME).  None for a frame that is not one of those - a process data object is read through the DBC like any signal."""
        kind, node = _co.classify(ident)
        if kind in ("tpdo1", "tpdo2", "tpdo3", "tpdo4", "rpdo1", "rpdo2", "rpdo3", "rpdo4", "other"): return None
        for sid, c in self._co.items():
            if c["channel"] and chan and c["channel"] != chan: continue
            if node and node not in c["nodes"]: continue
            try: d = _co.decode(ident, data)
            except _co.CanopenError: self.stats[sid]["malformed"] += 1; return []
            self._seen(sid, ts); out = []
            if kind == "heartbeat":
                was = c["state"].get(node); c["state"][node] = d["code"]
                if d["code"] == 0x00: out.append(("event", "ECU_RESET", "warn", f"CANopen node 0x{node:02X} booted (boot-up message)"))
                elif was == 0x05 and d["code"] in (0x04, 0x7F): out.append(("event", "ECU_RESET", "warn", f"CANopen node 0x{node:02X} left the operational state ({d['state']})"))
            elif kind == "emcy":
                code = bytes(d["manufacturer"]).rstrip(b"\0").decode("ascii", "replace")
                if d["reset"]: out.append(("event", "DTC_CLEAR", "info", f"{code or 'fault'} cleared on CANopen node 0x{node:02X}"))
                else:
                    sev = "critical" if d["register"] & 0x01 and d["code"] >> 8 in (0x23, 0x33, 0x42) else "warn"
                    table = self.p.dtc_table(); info = table.get(code) or {}; text = (" " + info["text"]) if info.get("text") else ""
                    out.append(("event", "DTC_SET", info.get("severity", sev), f"{code or 'EMCY 0x%04X' % d['code']}{text} (CANopen node 0x{node:02X}: {d['text']}, error register 0x{d['register']:02X})"))
            elif kind == "sdo_tx" and d.get("type") == "upload" and d.get("data") is not None:
                ent = self.p.od_fmt.get((node, d["index"], d["sub"]))
                if ent is not None:
                    fmt, b = ent; size = struct.calcsize(fmt)
                    if len(d["data"]) >= size: self._emit(b, struct.unpack(fmt, bytes(d["data"][:size]))[0], ts, out)
            return out
        return None

    def _j1939_diag(self, pgn, sa, data, ts):
        """DM1 (active faults), DM2 (previously active), vehicle identification, component identification - one event per fault that appears or goes away, per source address."""
        out = []
        if pgn == _jtp.PGN_VIN:
            v = _jtp.decode_vin(data)
            if v: self.j1939_vin[sa] = v
            return out
        if pgn == _jtp.PGN_COMPONENT_ID:
            c = _jtp.decode_component_id(data)
            if c: self.j1939_components[sa] = c
            return out
        dm = _jtp.decode_dm1(data)
        if dm is None: self.stats["can"]["malformed"] += 1; return out
        if pgn == _jtp.PGN_DM2: return out                                                                                  # history list: not a present condition
        now = {(f["spn"], f["fmi"]): f for f in dm["faults"]}; was = self._jfaults.get(sa, {}); sev = "critical" if dm["lamps"]["red_stop"] == "on" else "warn"
        for k, f in now.items():
            if k not in was: out.append(("event", "DTC_SET", sev, f"J1939 SPN {f['spn']} FMI {f['fmi']} ({f['fmi_text']}) from ECU 0x{sa:02X}" + (f", occurred {f['count']} times" if f["count"] else "")))
        for k, f in was.items():
            if k not in now: out.append(("event", "DTC_CLEAR", "info", f"J1939 SPN {f['spn']} FMI {f['fmi']} from ECU 0x{sa:02X} cleared"))
        self._jfaults[sa] = now; return out

    def _decode_flexray(self, raw, ts):
        """A captured FlexRay frame (channel byte, header, payload, CRC): both CRCs must agree; a null frame carries no valid payload; the cluster description of the source that owns the slot turns the payload into
        physical values, which the bindings read like any other signal."""
        try: f = _fx.decode_frame(raw)
        except _fx.FlexrayError: self.stats["flexray"]["malformed"] += 1; return []
        if f["null"]: self.stats["flexray"]["null"] += 1; return []
        out = []; hit = False
        for sid, (desc, idx) in self._fxs.items():
            ent = idx.get(f["slot"])
            if ent is None: continue
            hit = True; self._seen(sid, ts); fr, sigs = ent
            if not fr.sent_in(f["cycle"]): self.stats[sid]["unexpected_cycle"] += 1
            try: vals = {n: s.decode(f["payload"]) for n, s in fr.signals.items()}
            except _fx.FlexrayError: self.stats[sid]["malformed"] += 1; continue
            for name, binds in sigs.items():
                v = vals.get(name)
                if v is None: continue
                for b in binds:
                    if b.map is not None:
                        m = b.map.get(v) if isinstance(v, str) else b.map.get(str(int(v)))
                        if m is not None: self._emit(b, m, ts, out)
                    elif not isinstance(v, str): self._emit(b, v, ts, out)
        if not hit: self.stats["flexray"]["unknown"] += 1
        return out

    def _decode_lin(self, raw, ts):
        """A captured LIN frame (sync, protected identifier, data, checksum): the LDF of the source that defines the identifier turns the data into physical values, which the bindings read like any other signal."""
        try: f = _lin.decode_frame(raw)
        except _lin.LinError: self.stats["lin"]["malformed"] += 1; return []
        out = []; hit = False
        for sid, (ldf, idx) in self._lin.items():
            ent = idx.get(f["id"])
            if ent is None: continue
            hit = True; self._seen(sid, ts); fr, sigs = ent
            if len(f["data"]) != fr.length: self.stats[sid]["malformed"] += 1; continue
            vals = ldf.decode(f["id"], f["data"])
            for name, binds in sigs.items():
                v = vals.get(name)
                if v is None: continue
                for b in binds:
                    if b.map is not None:
                        m = b.map.get(v) if isinstance(v, str) else b.map.get(str(int(v)))
                        if m is not None: self._emit(b, m, ts, out)
                    elif not isinstance(v, str): self._emit(b, v, ts, out)
        if not hit: self.stats["lin"]["unknown"] += 1
        return out

    def _decode_table(self, row, ts):
        """A row of named values (a CSV line, a JSON object): every column the profile binds is a reading; the others are counted and ignored."""
        out = []; self.stats["table"]["frames"] += 1
        if ts is not None: self.last_seen["tab"] = ts
        for key, v in row.items():
            bs = self._tab.get(key)
            if not bs: self.stats["table"]["unknown"] += 1; continue
            for b in bs: self._emit(b, v, ts, out)
        return out

    def _err_ok(self, ts):
        if ts is None or ts - self._last_err >= 30: self._last_err = ts if ts is not None else 0; return True
        return False

    def _decode_diag(self, ident, data, ts):
        """ISO-TP reassembly of a diagnostic response, then OBD-II mode 01, UDS DTC reports (0x59) and ReadDataByIdentifier (0x62)."""
        r = self._tp.setdefault(ident, _isotp.Reassembler()); before = r.errors
        msg = r.feed(data, now=ts); self.stats["diag"]["tp_errors"] += r.errors - before
        if msg is None: return []
        return self._diag_message(ident, msg, ts)

    def _decode_doip(self, raw, ts):
        """Diagnostics over IP: the diagnostic messages an entity sends are the same UDS answers the CAN path rebuilds with ISO-TP; the tester's requests, acknowledgements, routing and alive checks carry no reading."""
        try: d = _doip.decode(raw)
        except _doip.DoipError: self.stats["doip"]["malformed"] += 1; return []
        self._seen("diag", ts)
        if d["name"] != "diagnostic_message": return []
        testers = {int(str(s.raw.get("tester", "0x0E80")), 16) for s in self.p.sources.values() if s.type == "uds" and s.raw.get("transport") == "doip"} or {0x0E80}
        if d["source_address"] in testers: return []                                                                           # a request
        return self._diag_message(("doip", d["source_address"]), d["data"], ts)

    def _diag_message(self, ident, msg, ts):
        out = []; o = _obd.decode(msg)
        if o and "error" not in o and o.get("value") is not None:
            b = self._obd.get("obd_" + o["name"])
            if b is not None: self._emit(b, float(o["value"]), ts, out)
            return out
        try: u = _uds.decode(msg)
        except _uds.UdsError: self.stats["diag"]["tp_errors"] += 1; return []
        table = self.p.dtc_table()
        if u and u["type"] == "dtc":
            for d in u["dtcs"]:
                key = (ident, d["code"]); st = d["status"] & 0x09; info = table.get(d["code"]) or {}; text = (" " + info["text"]) if info.get("text") else ""
                if st and self._dtc_seen.get(key) != d["status"]:
                    out.append(("event", "DTC_SET", info.get("severity", "warn"), f"{d['code']}{text} (status 0x{d['status']:02X}: {', '.join(d['status_flags'][:2])})"))
                elif not st and self._dtc_seen.get(key, 0) & 0x09:
                    out.append(("event", "DTC_CLEAR", "info", f"{d['code']}{text} cleared (status 0x{d['status']:02X})"))
                self._dtc_seen[key] = d["status"]
        elif u and u["type"] == "data" and u["did"] in self._dids:
            fmt, b = self._dids[u["did"]]; size = struct.calcsize(fmt)
            if len(u["data"]) >= size: self._emit(b, struct.unpack(fmt, bytes(u["data"][:size]))[0], ts, out)
        return out

    # ------------------------------------------------------------------------------------------------ SOME/IP
    def _decode_someip(self, raw, ts):
        out = []
        msgs, rest = _someip.split(self._someip_rem + bytes(raw)); self._someip_rem = rest
        for m in msgs:
            if m["service"] == _someip.SD_SERVICE and m["method"] == _someip.SD_METHOD:
                for e in _someip.parse_sd(m["payload"])["entries"]:
                    if e["type_name"] == "OfferService": self.offered[(e["service"], e["instance"])] = ts
                    elif e["type_name"] == "StopOfferService": self.offered.pop((e["service"], e["instance"]), None)
                continue
            if m["type_name"] != "NOTIFICATION" or m["rc"] != 0: continue
            for sid, src in self.p.sources.items():
                if src.type != "someip" or "catalog" not in src.raw: continue
                try: d = self.p.catalog(sid).decode(m["service"], m["method"], m["payload"])
                except _someip.SomeIpError: self.stats[sid]["malformed"] += 1; continue
                if d is None: continue
                self._seen(sid, ts); v = d["values"]; key = (d["service"], d["event"])
                for field, binds in self._sip.get(key, {}).items():
                    if field in v:
                        for b in binds:
                            if b.source == sid:
                                x = b.map.get(v.get(field + "_text"), b.map.get(str(v[field]))) if b.map else v[field]
                                if x is not None: self._emit(b, x, ts, out)
                for e in self._rules["someip"].get(key, ()):
                    try: detail = e.detail.format(**v) if e.detail else f"{d['service']}.{d['event']}"
                    except (KeyError, IndexError, ValueError): detail = f"{d['service']}.{d['event']}"
                    out.append(("event", e.type, e.severity, detail))
                break
        return out

    # ------------------------------------------------------------------------------------------------ Android VHAL
    def _decode_vhal(self, v, ts):
        if not isinstance(v, dict) or v.get("status") != 0: return []
        pid = v.get("prop"); name = self._vhal_dec.by_id.get(pid)
        if name is None: self._vhal_dec.unknown += 1; self.stats["vhal"]["unknown"] += 1; return []
        binds = self._vhal.get(name)
        if not binds: return []
        t = self._vhal_dec.props[name]["type"]
        raw = (v.get("float") or [None])[0] if t == "FLOAT" else (v.get("int32") or [None])[0] if t in ("INT32", "BOOLEAN") else (v.get("int64") or [None])[0] if t == "INT64" else None
        if raw is None: return []
        self._seen("vhal", ts); out = []
        for b in binds:
            area = b.key.partition("@")[2]
            if area and int(area) != int(v.get("area", 0)): continue
            x = b.map.get(str(int(raw))) if b.map else raw
            if x is not None: self._emit(b, x, ts, out)
        return out

    # ------------------------------------------------------------------------------------------------ logs
    def _decode_log(self, line, ts):
        if not isinstance(line, str): return []
        r = _logs.parse(line); self._seen("log", ts)
        if r is None: self.stats["log"]["unparsed"] += 1; return []
        out = []; low = r["message"].lower()
        for e in self._rules["log"]:
            if r["severity"] <= int(e.raw.get("max_severity", 4)) and e.match.lower() in low and (not e.raw.get("source") or e.raw["source"].lower() in r["source"].lower()):
                out.append(("event", e.type, e.severity, e.detail or r["message"]))
        if r["severity"] <= 4: out.insert(0, ("log", r["source"][:24], (r["message"] if re.match(r"(?:CRIT|ERR|ERROR|WARN|WARNING|INFO|FATAL)\b", r["message"]) else f"{r['level'].upper()} {r['message']}"), r.get("ts_text")))
        return out

    # ------------------------------------------------------------------------------------------------ camera
    def _decode_cam(self, msg, ts):
        if not isinstance(msg, dict): return []
        self._seen("camera", ts)
        if "y" in msg:                                                                    # a luma plane: analysed here (vgpt.camera), only metadata leaves
            if self._camera is None: self._camera = adapters.CameraAdapter()
            out = []
            for r in self._camera.decode(msg):
                out.append(("sig", LEGACY_TO_CONCEPT.get(r[1], r[1]), r[2]) if r[0] == "sig" else r)
            return out
        out = []
        for field, binds in self._cam.items():
            if field in msg:
                for b in binds: self._emit(b, msg[field], ts, out)
        if "blockage" in msg:
            self._block_run = getattr(self, "_block_run", 0) + 1 if msg["blockage"] > 0.5 else 0
            if self._block_run == 3: out.append(("event", "CAMERA_BLOCKED", "info", f"front camera blockage {msg['blockage']:.2f}"))
        return out

    # ------------------------------------------------------------------------------------------------ health
    def health_report(self, now):
        return {sid: {"frames": c.get("frames", 0), "decoded": c.get("decoded", 0), "unknown": c.get("unknown", 0), "rejected": c.get("rejected", 0), "malformed": c.get("malformed", 0),
                      "age_s": None if sid not in self.last_seen else round(now - self.last_seen[sid], 1)} for sid, c in self.stats.items()}

"""Bus emitter: canonical vehicle state -> raw wire traffic of ANY vehicle profile (the inverse of vgpt/vehicle/router.py).

The virtual car only knows concepts (speed = 57 km/h, tyre_pressure_fl = 231 kPa).  The profile says how this particular vehicle publishes them, so the emitter walks the profile's bindings in
reverse: it converts to the binding's unit, undoes scale and offset, packs the DBC message (Intel or Motorola, CAN-FD length), the SOME/IP notification (catalogue layout), the Android VHAL
parcelable (AOSP property id, native unit, area), the camera metadata record, and renders discrete events as what the vehicle would really emit - an ISO-TP/UDS DTC report, a SOME/IP event, a
QNX slog2info line.  Per-source behaviour (rates, redundancy, a secondary sensor that disagrees) comes from the profile and from `state` overrides, so a new vehicle is simulated by data.

    em = Emitter(profile);  for ts, bus, payload in em.tick(sec, state): ...      em.event(ts, {"type": "ADAS_DISENGAGE", "reason": "lane marking lost", "speed_kmh": 97})"""
import datetime as dt
import math
import struct

from .. import signal_registry as R
from ..ingest import catalog as _catalog, standards as _standards
from ..protocols import canframes, canopen as _co, doip as _doip, flexray as _fx, isotp, lin as _lin, obd as _obd, someip, uds, vhal as _vhal
from ..vehicle.profile import Profile

SEV_SLOG2 = {"error": 2, "warning": 3, "critical": 1, "notice": 4, "info": 5}


def _slog_ts(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%b %d %H:%M:%S.") + f"{int((t % 1) * 1000):03d}"
def _syslog_ts(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%b %d %H:%M:%S")
def _logcat_ts(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%m-%d %H:%M:%S.") + f"{int((t % 1) * 1000):03d}"


def native(b, canon):
    """Canonical value -> the value in the binding's own unit, scale and offset (inverse of Binding.canonical)."""
    cu = b.cunit or (R.CONCEPTS[b.concept].unit if b.concept in R.CONCEPTS else b.unit)
    v = float(canon)
    if b.unit not in ("none", "", cu): v = R.convert(v, cu, b.unit)
    if b.invert: v = -v
    return (v - b.offset) / (b.scale or 1.0)


def inverse_map(m, value, snap=False):
    """The enum label / raw code whose mapped number is `value` (with `snap`, the nearest one when none is equal)."""
    for k, v in m.items():
        if abs(float(v) - float(value)) < 1e-9: return k
    if snap and m: return min(m, key=lambda k: abs(float(m[k]) - float(value)))
    return None


class Emitter:
    def __init__(self, profile, can_fd_pad=True):
        self.p = profile if isinstance(profile, Profile) else Profile.load(profile)
        self.sess = 0; self._sd_sent = False; self._doip_sent = False; self.cnt = 0
        self._can = []; self._vhal = []; self._sip = {}; self._cam = {}
        for sid, src in self.p.sources.items():
            if src.type != "can": continue
            db = self.p.dbc(sid); plan = {}
            for b in self.p.bindings:
                if b.bus != "can" or b.source != sid or b.when: continue
                msg, _, sig = b.key.partition("."); m = db.by_name(msg)
                plan.setdefault(msg, [m, [], 0.0])[1].append((sig, b)); plan[msg][2] = max(plan[msg][2], b.rate_hz)
            for msg, (m, sigs, rate) in plan.items(): self._can.append((sid, m, sigs, rate, src.raw.get("channel", "")))
        for b in self.p.bindings:
            if b.bus == "vhal": self._vhal.append(b)
            elif b.bus == "camera": self._cam.setdefault(b.source, []).append(b)
            elif b.bus == "someip":
                svc, ev, field = b.key.split(".")
                self._sip.setdefault((b.source, svc, ev), []).append((field, b))
        self._co = []                                                                        # CANopen sources: (source id, channel, nodes, [(binding, node, index, sub, fmt)])
        for sid, src in self.p.sources.items():
            if src.type == "can" and src.raw.get("canopen"):
                ods = [(b, k[0], k[1], k[2], fmt) for k, (fmt, b) in self.p.od_fmt.items() if b.source == sid]
                self._co.append((sid, src.raw.get("channel", ""), self.p.canopen_nodes(sid), ods))
        self._co_booted = False; self._has_uds_can = any(s.type == "uds" for s in self.p.sources.values())
        self._lin = []                                                                       # LIN: one frame per (source, frame name), its signals encoded through the LDF
        for sid, src in self.p.sources.items():
            if src.type != "lin": continue
            ldf = self.p.ldf(sid); plan = {}
            for b in self.p.bindings:
                if b.bus != "lin" or b.source != sid: continue
                fname, _, sig = b.key.partition("."); plan.setdefault(fname, [ldf.frames[fname], [], 0.0])[1].append((sig, b)); plan[fname][2] = max(plan[fname][2], b.rate_hz)
            for fname, (fr, sigs, rate) in plan.items(): self._lin.append((ldf, fr, sigs, rate, bool(src.raw.get("enhanced", True))))
        self._fx = []                                                                        # FlexRay: one frame per (source, frame name), its signals packed through the description
        for sid, src in self.p.sources.items():
            if src.type != "flexray": continue
            desc = self.p.flexray(sid); plan = {}
            for b in self.p.bindings:
                if b.bus != "flexray" or b.source != sid: continue
                fname, _, sig = b.key.partition("."); plan.setdefault(fname, [desc.frames[fname], [], 0.0])[1].append((sig, b)); plan[fname][2] = max(plan[fname][2], b.rate_hz)
            for fname, (fr, sigs, rate) in plan.items(): self._fx.append((desc, fr, sigs, rate))
        self._obd = []                                                                       # OBD-II service 01: the vehicle's diagnostic port answers a scan tool's requests (a telematics dongle polls the parameters it wants)
        obd_bindings = [b for b in self.p.bindings if b.bus == "uds" and b.key.startswith("obd_")]
        if obd_bindings:
            _standards.install_obd(); by_name = {name: pid for pid, (name, _u, _n, _f) in _obd.PIDS.items()}; table = _catalog.obd_table()
            for b in obd_bindings:
                pid = by_name.get(b.key[4:])
                if pid is None: continue
                if pid in table: _c, n, scale, off = table[pid]; self._obd.append((b, pid, n, scale, off))
                else: self._obd.append((b, pid, _obd.PIDS[pid][2], None, None))
        self._doip = next(({"tester": int(str(s.raw.get("tester", "0x0E80")), 16), "entity": int(str(s.raw.get("entity", "0x1001")), 16), "eid": s.raw.get("eid", "0050c2a1b2c3"), "gid": s.raw.get("gid", "0050c2a1b2c3")}
                           for s in self.p.sources.values() if s.type == "uds" and s.raw.get("transport") == "doip"), None)           # the diagnostic gateway is on automotive Ethernet (ISO 13400), not behind ISO-TP on CAN
        self._dids = [(b, did, fmt) for did, (fmt, b) in self.p.did_fmt.items()]                   # UDS ReadDataByIdentifier (0x22): a vendor data identifier a battery or telematics dongle polls
        self._vh = _vhal.Vhal()
        self._cat = {}
        for sid, src in self.p.sources.items():
            if src.type == "someip": self._cat[sid] = self.p.catalog(sid)

    # ------------------------------------------------------------------------------------------------ cyclic traffic
    def _due(self, sec, rate, phase, full=False):
        """Sub-second send times within second `sec` for a message of `rate` Hz (a slow message sends in some seconds only; `full` = the first second after a silence: every message goes out once)."""
        if rate >= 1.0:
            n = int(math.ceil(rate)); return [sec + (phase + k * (1.0 / n)) % 1.0 for k in range(n)]
        period = max(1, int(round(1.0 / rate)))
        return [sec + phase] if full or (sec + int(phase * 100)) % period == 0 else []

    def _value(self, state, b):
        key = f"{b.concept}@{b.priority}"
        if key in state: return state[key]
        return state.get(b.concept)

    def tick(self, sec, state, full=False):
        """All cyclic messages scheduled in second `sec` for the canonical `state` ({concept: value}); yields (ts, bus, payload).  `full`: a car that wakes after a silence broadcasts every slow message once at once."""
        out = []
        if self._doip is not None and (not self._doip_sent or int(sec) % 30 == 0): self._doip_sent = True; out += self.doip_housekeeping(float(sec))
        if not self._sd_sent and any(s.type == "someip" for s in self.p.sources.values()):
            self._sd_sent = True; self.sess += 1
            out.append((float(sec), "someip", someip.build_sd_offer([(sid, 1, 1, 0) for cat in self._cat.values() for sid in cat.spec], session=self.sess)))
        for i, (sid, m, sigs, rate, chan) in enumerate(self._can):
            vals = {}
            for sig, b in sigs:
                v = self._value(state, b)
                if v is None or (isinstance(v, float) and not math.isfinite(v)): continue
                if b.map is not None:
                    lab = inverse_map(b.map, v, b.snap)
                    if lab is None: continue
                    vals[sig] = lab if not lab.lstrip("-").isdigit() else int(lab)
                else: vals[sig] = native(b, v)
            if not vals: continue
            groups = [vals]
            if m.mux_signal is not None:                                                    # a multiplexed message (J1939 tyre report): one frame per mux value, carrying the shared signals too
                muxed = {sg.name: sg for sg in m.signals if sg.mux is not None}; by = {}
                for name, sg in muxed.items():
                    if name in vals: by.setdefault(sg.mux, {})[name] = vals[name]
                shared = {k: v for k, v in vals.items() if k not in muxed}
                if by: groups = [dict({m.mux_signal.name: mv}, **sv, **shared) for mv, sv in by.items()]
            for gi, gvals in enumerate(groups):
                data = m.encode(gvals)
                for t in self._due(sec, rate, ((i * 0.013) + gi * 0.004) % 0.9, full):
                    out.append((t, "can", canframes.Frame(t, m.frame_id, data, extended=m.extended, fd=m.dlc > 8, channel=chan)))
        for sid, chan, nodes, ods in self._co:
            fr = lambda t, cid, data: (t, "can", canframes.Frame(t, cid, bytes(data).ljust(len(data), b"\0"), extended=False, channel=chan))
            if not self._co_booted:
                for n in nodes: out.append(fr(float(sec) + 0.001 * n, *_co.heartbeat(n, 0x00)))                                    # boot-up message of every node
            for n in nodes: out.append(fr(sec + 0.2 + 0.003 * n, *_co.heartbeat(n, 0x05)))                                       # heartbeat once a second
            out.append(fr(sec + 0.1, *_co.sync()))
            for k, (b, node, index, sub, fmt) in enumerate(ods):
                v = self._value(state, b)
                if v is None or (isinstance(v, float) and not math.isfinite(v)): continue
                try: raw = struct.pack(fmt, int(round(native(b, v))))
                except struct.error: continue
                for t in self._due(sec, b.rate_hz, 0.33 + 0.009 * (k % 50), full):
                    out.append(fr(t - 0.002, *_co.sdo_read_request(node, index, sub))); out.append(fr(t, *_co.sdo_read_response(node, index, sub, raw)))
        if self._co: self._co_booted = True
        for i, (desc, fr, sigs, rate) in enumerate(self._fx):
            vals = {}
            for sig, b in sigs:
                v = self._value(state, b)
                if v is None or (isinstance(v, float) and not math.isfinite(v)): continue
                vals[sig] = native(b, v)
            if not vals: continue
            payload = desc.encode(fr.name, vals)
            for t in self._due(sec, rate, 0.07 + 0.013 * (i % 30), full):
                cyc = int(round(t * 1000.0 / desc.cycle_ms)); cyc += (fr.base - cyc) % fr.repetition                    # the next cycle in which this frame has its slot
                for ch in ((0, 1) if fr.channel == "AB" else (1,) if fr.channel == "B" else (0,)):
                    out.append((t, "flexray", _fx.encode_frame(fr.slot, payload, cyc % 64, ch, sync=fr.slot in (10, 12), startup=False)))
        for i, (ldf, fr, sigs, rate, enhanced) in enumerate(self._lin):
            vals = {}
            for sig, b in sigs:
                v = self._value(state, b)
                if v is None or (isinstance(v, float) and not math.isfinite(v)): continue
                if b.map is not None:
                    lab = inverse_map(b.map, v, b.snap)
                    if lab is None: continue
                    vals[sig] = lab if not lab.lstrip("-").isdigit() else int(lab)
                else: vals[sig] = native(b, v)
            if not vals: continue
            frame = _lin.encode_frame(fr.id, ldf.encode(fr.name, vals), enhanced)
            for t in self._due(sec, rate, 0.05 + 0.017 * (i % 40), full): out.append((t, "lin", frame))
        for k, (b, pid, n, scale, off) in enumerate(self._obd):
            v = self._value(state, b)
            if v is None or (isinstance(v, float) and not math.isfinite(v)): continue
            x = native(b, v)
            if scale is not None: raw = max(0, min(256 ** n - 1, int(round((x - off) / scale)))); data = bytes([0x41, pid]) + raw.to_bytes(n, "big")
            else: data = _obd.encode(pid, x)
            for t in self._due(sec, b.rate_hz, 0.21 + 0.007 * (k % 90), full):
                out.append((t - 0.003, "can", canframes.Frame(t - 0.003, 0x7DF, bytes([2, 1, pid]).ljust(8, b"\xAA"), channel="")))
                out.append((t, "can", canframes.Frame(t, 0x7E8, (bytes([len(data)]) + data).ljust(8, b"\xAA"), channel="")))
        for k, (b, did, fmt) in enumerate(self._dids):
            v = self._value(state, b)
            if v is None or (isinstance(v, float) and not math.isfinite(v)): continue
            try: data = struct.pack(fmt, int(round(native(b, v))))
            except struct.error: continue
            resp = bytes([0x62, did >> 8, did & 0xFF]) + data
            for t in self._due(sec, b.rate_hz, 0.37 + 0.011 * (k % 60), full):
                if self._doip is not None: out += self._doip_exchange(t, bytes([0x22, did >> 8, did & 0xFF]), resp); continue
                out.append((t - 0.004, "can", canframes.Frame(t - 0.004, 0x7E0, bytes([3, 0x22, did >> 8, did & 0xFF]).ljust(8, b"\xAA"), channel="")))
                for j, fr in enumerate(isotp.encode(resp)): out.append((t + 0.001 * j, "can", canframes.Frame(t + 0.001 * j, 0x7E8, fr.ljust(8, b"\xAA"), channel="")))
        for b in self._vhal:
            v = self._value(state, b)
            if v is None: continue
            name, _, area = b.key.partition("@")
            if b.map is not None:
                lab = inverse_map(b.map, v, b.snap)
                if lab is None: continue
                raw = int(lab)
            else: raw = native(b, v)
            for t in self._due(sec, b.rate_hz, 0.31 + 0.002 * (self._vhal.index(b) % 40), full):
                out.append((t, "vhal", self._vh.encode(name, raw, area=int(area) if area else 0, ts=int(sec * 1e9))))
        for (sid, svc, ev), fields in self._sip.items():
            spec = next(((cat, s_id, e_id, s, e) for cat in [self._cat[sid]] for s_id, s in cat.spec.items() if s["name"] == svc for e_id, e in s["events"].items() if e["name"] == ev), None)
            if spec is None: continue
            cat, s_id, e_id, s, e = spec; vals = {f: state.get(f"{svc}.{ev}.{f}", 0) for f in e["fields"]}; have = False
            for field, b in fields:
                v = self._value(state, b)
                if v is not None: vals[field] = native(b, v); have = True
            if not have: continue
            rate = max(b.rate_hz for _, b in fields)
            for t in self._due(sec, rate, 0.47, full):
                self.sess += 1; out.append((t, "someip", someip.build(int(s_id, 16) if isinstance(s_id, str) else s_id, int(e_id, 16), self._pack(e, vals), session=self.sess, mtype=0x02)))
        for sid, binds in self._cam.items():
            msg = {"cam": "front", "frame": int(sec)}; have = False
            for b in binds:
                v = self._value(state, b)
                if v is not None: msg[b.key] = float(native(b, v)); have = True
            if have: out.append((sec + 0.55, "cam", msg))
        out.sort(key=lambda r: r[0])
        return out

    @staticmethod
    def _pack(e, vals):
        fmt = e["fmt"]; typed = []
        for ch, f in zip([c for c in fmt if c.isalpha()], e["fields"]):
            v = vals[f]; typed.append(float(v) if ch in "fd" else int(round(float(v))))
        return struct.pack(fmt, *typed)

    # ------------------------------------------------------------------------------------------------ discrete events -> what the vehicle would emit
    def event(self, ts, ev):
        """ev = {"type": ..., plus the event's fields / detail}.  Returns [(ts, bus, payload)]."""
        ty = ev["type"]; out = []
        if ty in ("DTC_SET", "DTC_CLEAR") and ("codes" in ev or "code" in ev): return self._dtc(ts, ty, ev)                       # a stored code is a UDS report whatever else the profile derives DTC_SET from
        rule = next((r for r in self.p.events if r.type == ty and r.origin.startswith("someip:")), None)
        if rule is not None:
            svc, evn = rule.origin[7:].split(".")[:2]
            for sid, cat in self._cat.items():
                for s_id, s in cat.spec.items():
                    if s["name"] != svc: continue
                    for e_id, e in s["events"].items():
                        if e["name"] != evn: continue
                        vals = {}
                        for f in e["fields"]:
                            v = ev.get(f, ev.get(f + "_text", 0)); enum = (e.get("enums") or {}).get(f)
                            if enum and isinstance(v, str):
                                inv = {t: int(k) for k, t in enum.items()}; v = inv.get(v, 0)
                            vals[f] = v
                        self.sess += 1; out.append((ts, "someip", someip.build(int(s_id, 16) if isinstance(s_id, str) else s_id, int(e_id, 16), self._pack(e, vals), session=self.sess, mtype=0x02)))
            return out
        rule = next((r for r in self.p.events if r.type == ty and r.origin.startswith("log:")), None)
        if rule is not None:
            msg = ev.get("log") or ev.get("detail") or rule.match
            out.append((ts, "log", self.log_line(ts, ev.get("source", "thermal_mgr"), msg, ev.get("severity", "error"), ev.get("fmt", "slog2info")))); return out
        if ty in ("DTC_SET", "DTC_CLEAR"): return self._dtc(ts, ty, ev)
        return out

    def _doip_exchange(self, t, request, response):
        """A tester's UDS request and the entity's answer as DoIP diagnostic messages (with the entity's positive acknowledgement of the request in between)."""
        d = self._doip; T, X = d["tester"], d["entity"]
        return [(t - 0.004, "doip", _doip.encode("diagnostic_message", source_address=T, target_address=X, data=request)),
                (t - 0.003, "doip", _doip.encode("diagnostic_ack", source_address=X, target_address=T, code=0, previous=request[:3])),
                (t, "doip", _doip.encode("diagnostic_message", source_address=X, target_address=T, data=response))]

    def doip_housekeeping(self, ts):
        """What a DoIP gateway says besides diagnostics: the vehicle announcement (with the VIN is added by the car), routing activation, alive checks."""
        d = self._doip
        if d is None: return []
        return [(ts, "doip", _doip.encode("routing_activation_request", source_address=d["tester"], activation_type=0)),
                (ts + 0.002, "doip", _doip.encode("routing_activation_response", tester_address=d["tester"], entity_address=d["entity"], response_code=0x10)),
                (ts + 0.01, "doip", _doip.encode("alive_check_request")), (ts + 0.012, "doip", _doip.encode("alive_check_response", source_address=d["tester"]))]

    def _dtc(self, ts, ty, ev):
        codes = ev.get("codes") or [ev["code"]]; out = []
        for sid, chan, nodes, ods in self._co:                                                  # a CANopen node reports a fault as an emergency message (EMCY), the code in the manufacturer-specific bytes
            for c in codes:
                cid, data = _co.emcy(nodes[0] if nodes else 1, 0xFF00 if ty == "DTC_SET" else 0x0000, 0x81 if ty == "DTC_SET" else 0x00, str(c).encode("ascii")[:5])
                out.append((ts, "can", canframes.Frame(ts, cid, data, extended=False, channel=chan)))
        if self._co: return out                                                                  # a CANopen vehicle reports a fault once, as the emergency message of the node
        resp = uds.encode_dtc_response([(*uds.parse_dtc(c), 0x01, 0x09 if ty == "DTC_SET" else 0x00) for c in codes])
        if self._doip is not None: return self._doip_exchange(ts, bytes([0x19, 0x02, 0x09]), resp)
        for fr in isotp.encode(resp): out.append((ts, "can", canframes.Frame(ts, 0x7E8, fr.ljust(8, b"\xAA"), channel="")))
        return out

    def log_line(self, ts, source, message, severity="error", fmt="slog2info"):
        if fmt == "rfc3164":                                                                       # Linux syslog: <facility*8+severity>Mon DD HH:MM:SS host tag[pid]: message
            pri = 3 * 8 + {"critical": 2, "error": 3, "warning": 4, "notice": 5, "info": 6}.get(severity, 4)
            return f"<{pri}>{_syslog_ts(ts)} vehicle {source}[{100 + abs(hash(source)) % 900}]: {message}"
        if fmt == "logcat": return f"{_logcat_ts(ts)}    88    88 {'E' if severity in ('error', 'critical') else 'W'} {source}: {message}"
        return f"{_slog_ts(ts)} {source} {SEV_SLOG2.get(severity, 3)} 0 {message}"

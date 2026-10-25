def sub(path, old, new, count=1):
    s = open(path, encoding="utf8").read()
    assert old in s, (path, old[:70])
    open(path, "w", encoding="utf8").write(s.replace(old, new, count))


# ---------------------------------------------------------------- profile: the object dictionary entries a CANopen source lists
P = "vgpt/vehicle/profile.py"
sub(P, '"table", "lin", "flexray")', '"table", "lin", "flexray", "canopen")')
sub(P, "        self.detectors = list(spec.get(\"detectors\", []))", '''        self.od_fmt = {}                                                                      # CANopen: SDO reads of one object: (node, index, sub) -> (struct format, binding)
        for s in self.sources.values():
            if s.type != "can": continue
            for key, d in (s.raw.get("od") or {}).items():
                idx, _, sub = key.partition(":"); node = int(str(d.get("node", "0x01")), 16)
                self.bindings.append(Binding(d["concept"], "canopen", s.id, f"sdo:{key}@{node}", norm_unit(d.get("unit", "none")), float(d.get("scale", 1.0)), float(d.get("offset", 0.0)), int(d.get("priority", 1)), float(d.get("rate_hz", 0.5))))
                self.od_fmt[(node, int(idx, 16), int(sub or 0))] = (d.get("fmt", "<H"), self.bindings[-1])
        self.detectors = list(spec.get("detectors", []))''')
sub(P, "            if b.bus == \"uds\": continue\n", "            if b.bus in (\"uds\", \"canopen\"): continue\n")
sub(P, "    def canonical_unit(self, concept):", '''    def canopen_nodes(self, source_id):
        """The node numbers a CANopen source speaks for: its object dictionary entries and the PDO identifiers of its DBC."""
        src = self.sources[source_id]; nodes = {int(str(d.get("node", "0x01")), 16) for d in (src.raw.get("od") or {}).values()} | {int(str(n), 16) if isinstance(n, str) else int(n) for n in src.raw.get("nodes", [])}
        try:
            db = self.dbc(source_id)
            for m in list(db.messages.values()):
                if (m.frame_id & 0x780) in (0x180, 0x200, 0x280, 0x300, 0x380, 0x400, 0x480, 0x500) and m.frame_id & 0x7F: nodes.add(m.frame_id & 0x7F)
        except (OSError, ValueError, KeyError): pass
        return sorted(nodes)

    def canonical_unit(self, concept):''')

# ---------------------------------------------------------------- emitter
E = "vgpt/world/emit.py"
sub(E, "from ..protocols import canframes, doip as _doip,", "from ..protocols import canframes, canopen as _co, doip as _doip,")
sub(E, "        self._lin = []   ", '''        self._co = []                                                                        # CANopen sources: (source id, channel, nodes, [(binding, node, index, sub, fmt)])
        for sid, src in self.p.sources.items():
            if src.type == "can" and src.raw.get("canopen"):
                ods = [(b, k[0], k[1], k[2], fmt) for k, (fmt, b) in self.p.od_fmt.items() if b.source == sid]
                self._co.append((sid, src.raw.get("channel", ""), self.p.canopen_nodes(sid), ods))
        self._co_booted = False
        self._lin = []   ''')
sub(E, "        for i, (ldf, fr, sigs, rate, enhanced) in enumerate(self._lin):", '''        for sid, chan, nodes, ods in self._co:
            fr = lambda t, cid, data: (t, "can", canframes.Frame(t, cid, bytes(data).ljust(len(data), b"\\0"), extended=False, channel=chan))
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
        for i, (ldf, fr, sigs, rate, enhanced) in enumerate(self._lin):''')
sub(E, "    def _dtc(self, ts, ty, ev):\n        codes = ev.get(\"codes\") or [ev[\"code\"]]; out = []", '''    def _dtc(self, ts, ty, ev):
        codes = ev.get("codes") or [ev["code"]]; out = []
        for sid, chan, nodes, ods in self._co:                                                  # a CANopen node reports a fault as an emergency message (EMCY), the code in the manufacturer-specific bytes
            for c in codes:
                cid, data = _co.emcy(nodes[0] if nodes else 1, 0xFF00 if ty == "DTC_SET" else 0x0000, 0x81 if ty == "DTC_SET" else 0x00, str(c).encode("ascii")[:5])
                out.append((ts, "can", canframes.Frame(ts, cid, data, extended=False, channel=chan)))
        if self._co and self._doip is None and not self._has_uds_can: return out''')
sub(E, "        self._co_booted = False\n", "        self._co_booted = False; self._has_uds_can = any(s.type == \"uds\" for s in self.p.sources.values())\n")

# ---------------------------------------------------------------- router
RT = "vgpt/vehicle/router.py"
sub(RT, "from ..protocols import dbc as _dbc, doip as _doip,", "from ..protocols import canopen as _co, dbc as _dbc, doip as _doip,")
sub(RT, "        self._lin = {}\n", "        self._lin = {}; self._co = {s.id: {\"nodes\": set(self.p.canopen_nodes(s.id)), \"state\": {}, \"channel\": s.raw.get(\"channel\")} for s in self.p.sources.values() if s.type == \"can\" and s.raw.get(\"canopen\")}\n")
sub(RT, "        if self.has_diag and ident in self.diag_ids and not ext:\n            self._seen(\"diag\", ts); return self._decode_diag(ident, data, ts)", '''        if self._co and not ext and not fd:
            r = self._decode_canopen(ident, data, chan, ts)
            if r is not None: return r
        if self.has_diag and ident in self.diag_ids and not ext:
            self._seen("diag", ts); return self._decode_diag(ident, data, ts)''')
sub(RT, "    def _j1939_diag(self, pgn, sa, data, ts):", '''    def _decode_canopen(self, ident, data, chan, ts):
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
                code = bytes(d["manufacturer"]).rstrip(b"\\0").decode("ascii", "replace")
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

    def _j1939_diag(self, pgn, sa, data, ts):''')
print("canopen patched")

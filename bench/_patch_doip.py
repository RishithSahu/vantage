def sub(path, old, new, count=1):
    s = open(path, encoding="utf8").read()
    assert old in s, (path, old[:70])
    open(path, "w", encoding="utf8").write(s.replace(old, new, count))


# ---------------------------------------------------------------- emitter: diagnostics over DoIP instead of ISO-TP on CAN
E = "vgpt/world/emit.py"
sub(E, "from ..protocols import canframes, isotp, lin as _lin,", "from ..protocols import canframes, doip as _doip, isotp, lin as _lin,")
sub(E, "        self._dids = [(b, did, fmt) for did, (fmt, b) in self.p.did_fmt.items()]", '''        self._doip = next(({"tester": int(str(s.raw.get("tester", "0x0E80")), 16), "entity": int(str(s.raw.get("entity", "0x1001")), 16), "eid": s.raw.get("eid", "0050c2a1b2c3"), "gid": s.raw.get("gid", "0050c2a1b2c3")}
                           for s in self.p.sources.values() if s.type == "uds" and s.raw.get("transport") == "doip"), None)           # the diagnostic gateway is on automotive Ethernet (ISO 13400), not behind ISO-TP on CAN
        self._dids = [(b, did, fmt) for did, (fmt, b) in self.p.did_fmt.items()]''')
sub(E, '''            for t in self._due(sec, b.rate_hz, 0.37 + 0.011 * (k % 60), full):
                out.append((t - 0.004, "can", canframes.Frame(t - 0.004, 0x7E0, bytes([3, 0x22, did >> 8, did & 0xFF]).ljust(8, b"\\xAA"), channel="")))
                for j, fr in enumerate(isotp.encode(resp)): out.append((t + 0.001 * j, "can", canframes.Frame(t + 0.001 * j, 0x7E8, fr.ljust(8, b"\\xAA"), channel="")))''', '''            for t in self._due(sec, b.rate_hz, 0.37 + 0.011 * (k % 60), full):
                if self._doip is not None: out += self._doip_exchange(t, bytes([0x22, did >> 8, did & 0xFF]), resp); continue
                out.append((t - 0.004, "can", canframes.Frame(t - 0.004, 0x7E0, bytes([3, 0x22, did >> 8, did & 0xFF]).ljust(8, b"\\xAA"), channel="")))
                for j, fr in enumerate(isotp.encode(resp)): out.append((t + 0.001 * j, "can", canframes.Frame(t + 0.001 * j, 0x7E8, fr.ljust(8, b"\\xAA"), channel="")))''')
sub(E, '''    def _dtc(self, ts, ty, ev):
        codes = ev.get("codes") or [ev["code"]]; out = []
        resp = uds.encode_dtc_response([(*uds.parse_dtc(c), 0x01, 0x09 if ty == "DTC_SET" else 0x00) for c in codes])
''', '''    def _doip_exchange(self, t, request, response):
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
        resp = uds.encode_dtc_response([(*uds.parse_dtc(c), 0x01, 0x09 if ty == "DTC_SET" else 0x00) for c in codes])
        if self._doip is not None: return self._doip_exchange(ts, bytes([0x19, 0x02, 0x09]), resp)
''')
# the periodic housekeeping rides the first second and then every 30 s
sub(E, "        out = []\n        if not self._sd_sent and any(s.type == \"someip\" for s in self.p.sources.values()):", "        out = []\n        if self._doip is not None and (not self._doip_sent or int(sec) % 30 == 0): self._doip_sent = True; out += self.doip_housekeeping(float(sec))\n        if not self._sd_sent and any(s.type == \"someip\" for s in self.p.sources.values()):")
sub(E, "        self.sess = 0; self._sd_sent = False; self.cnt = 0", "        self.sess = 0; self._sd_sent = False; self._doip_sent = False; self.cnt = 0")

# ---------------------------------------------------------------- transmitter + VIN announcement
T = "vgpt/car/transmit.py"
sub(T, "from ..protocols import canframes as cf, isotp, vhalparcel", "from ..protocols import canframes as cf, doip as _doip, isotp, vhalparcel")
sub(T, '            elif bus == "lin": self._out(ts, ("lin", ""), bytes(payload), "lin")', '            elif bus == "lin": self._out(ts, ("lin", ""), bytes(payload), "lin")\n            elif bus == "doip": self._out(ts, ("eth", "doip"), bytes(payload), "doip")')
sub(T, "    def announce_vin(self, ts, vin, uds=True, vhal=False):", '''    def announce_vin_doip(self, ts, vin, tester=0x0E80, entity=0x1001, eid="0050c2a1b2c3", gid="0050c2a1b2c3"):
        """The VIN the way a DoIP entity says it: the vehicle announcement (UDP) and the answer to ReadDataByIdentifier F190 (TCP)."""
        self.send(ts, "doip", _doip.encode("vehicle_announcement", vin=vin, logical_address=entity, eid=eid, gid=gid, further_action=0))
        self.send(ts + 0.001, "doip", _doip.encode("diagnostic_message", source_address=tester, target_address=entity, data=b"\\x22\\xf1\\x90"))
        self.send(ts + 0.002, "doip", _doip.encode("diagnostic_message", source_address=entity, target_address=tester, data=b"\\x62\\xf1\\x90" + vin.encode("ascii")))

    def announce_vin(self, ts, vin, uds=True, vhal=False):''')
sub("vgpt/car/car.py", "        self.has_uds = any(s.type == \"uds\" for s in self.profile.sources.values());", "        self.doip = next((s.raw for s in self.profile.sources.values() if s.type == \"uds\" and s.raw.get(\"transport\") == \"doip\"), None)\n        self.has_uds = any(s.type == \"uds\" for s in self.profile.sources.values());")
sub("vgpt/car/car.py", "        if self.has_uds or self.has_vhal: self.tx.announce_vin(self.drive.t, self.vin, uds=self.has_uds, vhal=self.has_vhal)", '''        if self.doip is not None: self.tx.announce_vin_doip(self.drive.t, self.vin, int(str(self.doip.get("tester", "0x0E80")), 16), int(str(self.doip.get("entity", "0x1001")), 16), self.doip.get("eid", "0050c2a1b2c3"), self.doip.get("gid", "0050c2a1b2c3"))
        if self.has_uds and self.doip is None or self.has_vhal: self.tx.announce_vin(self.drive.t, self.vin, uds=self.has_uds and self.doip is None, vhal=self.has_vhal)''')

# ---------------------------------------------------------------- sniffer
S = "vgpt/pipe/sniff.py"
sub(S, "from ..protocols import canframes as cf, lin as _lin,", "from ..protocols import canframes as cf, doip as _doip, lin as _lin,")
sub(S, 'KINDS = ("can", "someip", "vhal", "lin",', 'KINDS = ("can", "someip", "vhal", "lin", "doip",')
sub(S, "    if len(b) in (16, 72) and _is_can(b):", "    if len(b) >= 8 and b[1] == (b[0] ^ 0xFF) and _doip.looks_like_frame(b): return \"doip\"                   # ISO 13400 header: version, its inverse, a known payload type, a length that agrees with the block\n    if len(b) in (16, 72) and _is_can(b):")
sub(S, '        if kind == "lin": return [("lin", b)]', '        if kind == "lin": return [("lin", b)]\n        if kind == "doip": return [("doip", b)]')

# ---------------------------------------------------------------- fingerprint
F = "vgpt/pipe/fingerprint.py"
sub(F, "        self.lin = collections.Counter(); self.someip", "        self.doip = collections.Counter(); self.lin = collections.Counter(); self.someip")
sub(F, "    def add_lin(self, ident):", "    def add_doip(self, address): self.frames[\"doip\"] += 1; self.doip[address] += 1\n    def add_lin(self, ident):")
sub(F, "        self.lin = set()\n", "        self.doip = {int(str(s.raw.get(\"entity\", \"0x1001\")), 16) for s in p.sources.values() if s.type == \"uds\" and s.raw.get(\"transport\") == \"doip\"}\n        self.lin = set()\n")
sub(F, "    if inv.lin:\n", '''    if inv.doip:
        n = sum(inv.doip.values()); hit = sum(c for a, c in inv.doip.items() if a in entry.doip)
        parts["doip"] = hit / n; weights["doip"] = (math.log1p(n), 0.8 * parts["doip"] + 0.2 * (1.0 if entry.doip else 0.0)); why.append(f"{len([a for a in inv.doip if a in entry.doip])} of {len(inv.doip)} DoIP logical addresses are its diagnostic entities")
    if inv.lin:
''')

# ---------------------------------------------------------------- receiver
R = "vgpt/pipe/receiver.py"
sub(R, "from ..protocols import canframes as cf, isotp as _isotp, lin as _lin,", "from ..protocols import canframes as cf, doip as _doip, isotp as _isotp, lin as _lin,")
sub(R, '        elif bus == "lin":\n            try:', '''        elif bus == "doip":
            try: d = _doip.decode(bytes(obj))
            except _doip.DoipError: inv.frames["doip"] += 1; return
            inv.add_doip(d.get("logical_address", d.get("source_address", d.get("entity_address", 0))))
            vin = d.get("vin") if d["name"] == "vehicle_announcement" else (bytes(d["data"][3:]).decode("ascii", "replace") if d.get("data", b"")[:3] == b"\\x62\\xf1\\x90" else None)
            if ses.vin is None and vin and VIN_RE.fullmatch(vin.encode()): ses.vin = vin
        elif bus == "lin":
            try:''')

# ---------------------------------------------------------------- router
RT = "vgpt/vehicle/router.py"
sub(RT, "from ..protocols import dbc as _dbc, isotp as _isotp,", "from ..protocols import dbc as _dbc, doip as _doip, isotp as _isotp,")
sub(RT, '            if bus == "lin": return self._decode_lin(payload, ts)', '            if bus == "lin": return self._decode_lin(payload, ts)\n            if bus == "doip": return self._decode_doip(payload, ts)')
sub(RT, '''        msg = r.feed(data, now=ts); self.stats["diag"]["tp_errors"] += r.errors - before
        if msg is None: return []
        out = []; o = _obd.decode(msg)''', '''        msg = r.feed(data, now=ts); self.stats["diag"]["tp_errors"] += r.errors - before
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
        out = []; o = _obd.decode(msg)''')

# ---------------------------------------------------------------- runtime ledger
RN = "vgpt/live/runtime.py"
sub(RN, '            if bus == "lin":\n                for sid, (ldf, idx)', '            if bus == "doip": return ("diag", "UDS over DoIP", f"0x{key[1]:04X}")\n            if bus == "lin":\n                for sid, (ldf, idx)')
sub(RN, '        if bus == "lin":\n            raw = bytes(payload);', '''        if bus == "doip":
            raw = bytes(payload); return ("doip", int.from_bytes(raw[8:10], "big") if len(raw) >= 10 else 0)
        if bus == "lin":
            raw = bytes(payload);''')

# ---------------------------------------------------------------- the protocol module: a sniffer predicate
sub("vgpt/protocols/doip.py", "def diagnostic(frame):", '''def looks_like_frame(block):
    try: decode(block); return True
    except DoipError: return False


def diagnostic(frame):''')
print("doip patched")

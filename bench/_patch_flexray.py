def sub(path, old, new, count=1):
    s = open(path, encoding="utf8").read()
    assert old in s, (path, old[:70])
    open(path, "w", encoding="utf8").write(s.replace(old, new, count))


# ---------------------------------------------------------------- profile
P = "vgpt/vehicle/profile.py"
sub(P, "self._ldf = {}; self._fx = {}", "self._ldf = {}; self._fx = {}")
sub(P, "    def catalog(self, source_id):", '''    def flexray(self, source_id):
        """The FlexRay description (slots, cycles, signals) of a FlexRay source."""
        if source_id not in self._fx:
            from ..protocols import flexray
            self._fx[source_id] = flexray.Description.load(self.file(self.sources[source_id].raw["description"]))
        return self._fx[source_id]

    def catalog(self, source_id):''')
sub(P, "            if b.bus == \"lin\" and b.source in self.sources: self._check_lin(b, p, err)", "            if b.bus == \"lin\" and b.source in self.sources: self._check_lin(b, p, err)\n            if b.bus == \"flexray\" and b.source in self.sources: self._check_flexray(b, p, err)")
sub(P, "    def _check_someip(self, b, p, err):", '''    def _check_flexray(self, b, p, err):
        s = self.sources[b.source]
        if "description" not in s.raw: err(f"sources[{b.source}]", "a FlexRay source needs a 'description' file"); return
        try: d = self.flexray(b.source)
        except (OSError, ValueError, KeyError) as e: err(f"sources[{b.source}].description", f"cannot load: {e}"); return
        frame, _, sig = b.key.partition("."); fr = d.frames.get(frame)
        if fr is None: err(p + ".from", f"frame {frame!r} is not in {s.raw['description']}"); return
        if sig not in fr.signals: err(p + ".from", f"signal {sig!r} is not in frame {frame}")

    def _check_someip(self, b, p, err):''')

# ---------------------------------------------------------------- emitter
E = "vgpt/world/emit.py"
sub(E, "from ..protocols import canframes, canopen as _co, doip as _doip,", "from ..protocols import canframes, canopen as _co, doip as _doip, flexray as _fx,")
sub(E, "        self._obd = []   ", '''        self._fx = []                                                                        # FlexRay: one frame per (source, frame name), its signals packed through the description
        for sid, src in self.p.sources.items():
            if src.type != "flexray": continue
            desc = self.p.flexray(sid); plan = {}
            for b in self.p.bindings:
                if b.bus != "flexray" or b.source != sid: continue
                fname, _, sig = b.key.partition("."); plan.setdefault(fname, [desc.frames[fname], [], 0.0])[1].append((sig, b)); plan[fname][2] = max(plan[fname][2], b.rate_hz)
            for fname, (fr, sigs, rate) in plan.items(): self._fx.append((desc, fr, sigs, rate))
        self._obd = []   ''', 1)
sub(E, "        for i, (ldf, fr, sigs, rate, enhanced) in enumerate(self._lin):", '''        for i, (desc, fr, sigs, rate) in enumerate(self._fx):
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
        for i, (ldf, fr, sigs, rate, enhanced) in enumerate(self._lin):''')

# ---------------------------------------------------------------- transmitter
sub("vgpt/car/transmit.py", '            elif bus == "lin": self._out(ts, ("lin", ""), bytes(payload), "lin")', '            elif bus == "lin": self._out(ts, ("lin", ""), bytes(payload), "lin")\n            elif bus == "flexray": self._out(ts, ("fr", ""), bytes(payload), "flexray")')

# ---------------------------------------------------------------- sniffer
S = "vgpt/pipe/sniff.py"
sub(S, "from ..protocols import canframes as cf, doip as _doip, lin as _lin,", "from ..protocols import canframes as cf, doip as _doip, flexray as _fx, lin as _lin,")
sub(S, 'KINDS = ("can", "someip", "vhal", "lin", "doip",', 'KINDS = ("can", "someip", "vhal", "lin", "doip", "flexray",')
sub(S, "    if 4 <= len(b) <= 11 and b[0] == _lin.SYNC", "    if len(b) >= 9 and b[0] in (0, 1) and (len(b) - 9) % 2 == 0 and _fx.looks_like_frame(b): return \"flexray\"          # channel byte, header whose CRC agrees, payload, frame CRC that agrees\n    if 4 <= len(b) <= 11 and b[0] == _lin.SYNC")
sub(S, '        if kind == "doip": return [("doip", b)]', '        if kind == "doip": return [("doip", b)]\n        if kind == "flexray": return [("flexray", b)]')

# ---------------------------------------------------------------- fingerprint
F = "vgpt/pipe/fingerprint.py"
sub(F, "        self.doip = collections.Counter(); self.lin", "        self.flexray = collections.Counter(); self.doip = collections.Counter(); self.lin")
sub(F, "    def add_doip(self, address):", "    def add_flexray(self, slot): self.frames[\"flexray\"] += 1; self.flexray[slot] += 1\n    def add_doip(self, address):")
sub(F, "        self.lin = set()\n", "        self.flexray = set()\n        for sid, src in p.sources.items():\n            if src.type == \"flexray\" and \"description\" in src.raw:\n                try: self.flexray |= set(p.flexray(sid).by_slot)\n                except (OSError, ValueError, KeyError): pass\n        self.lin = set()\n")
sub(F, '    for kind, have, expected in (("doip", inv.doip, entry.doip), ("lin", inv.lin, entry.lin)):', '    for kind, have, expected in (("doip", inv.doip, entry.doip), ("lin", inv.lin, entry.lin), ("flexray", inv.flexray, entry.flexray)):')
sub(F, "    if inv.lin:\n", '''    if inv.flexray:
        n = sum(inv.flexray.values()); hit = sum(c for i, c in inv.flexray.items() if i in entry.flexray)
        parts["flexray"] = hit / n; sp = len([i for i in inv.flexray if i in entry.flexray]) / max(len(entry.flexray), 1)
        weights["flexray"] = (math.log1p(n), 0.7 * parts["flexray"] + 0.3 * min(1.0, sp)); why.append(f"{len([i for i in inv.flexray if i in entry.flexray])} of {len(inv.flexray)} FlexRay slots are in its cluster description")
    if inv.lin:
''')

# ---------------------------------------------------------------- receiver
R = "vgpt/pipe/receiver.py"
sub(R, "from ..protocols import canframes as cf, doip as _doip,", "from ..protocols import canframes as cf, doip as _doip, flexray as _fx,")
sub(R, '        elif bus == "lin":\n            try:', '        elif bus == "flexray":\n            try: inv.add_flexray(_fx.decode_frame(bytes(obj))["slot"])\n            except _fx.FlexrayError: inv.frames["flexray"] += 1\n        elif bus == "lin":\n            try:')

# ---------------------------------------------------------------- router
RT = "vgpt/vehicle/router.py"
sub(RT, "from ..protocols import canopen as _co, dbc as _dbc, doip as _doip,", "from ..protocols import canopen as _co, dbc as _dbc, doip as _doip, flexray as _fx,")
sub(RT, "        self._lin = {}; self._co =", "        self._fxs = {}; self._lin = {}; self._co =")
sub(RT, "            elif src.type == \"lin\": self._index_lin(src)", "            elif src.type == \"lin\": self._index_lin(src)\n            elif src.type == \"flexray\": self._index_flexray(src)")
sub(RT, "    def _index_lin(self, src):", '''    def _index_flexray(self, src):
        desc = self.p.flexray(src.id); idx = {}
        for b in self.p.bindings:
            if b.bus != "flexray" or b.source != src.id: continue
            fname, _, sig = b.key.partition("."); fr = desc.frames.get(fname)
            if fr is not None: idx.setdefault(fr.slot, (fr, {}))[1].setdefault(sig, []).append(b)
        self._fxs[src.id] = (desc, idx)

    def _index_lin(self, src):''')
sub(RT, '            if bus == "doip": return self._decode_doip(payload, ts)', '            if bus == "doip": return self._decode_doip(payload, ts)\n            if bus == "flexray": return self._decode_flexray(payload, ts)')
sub(RT, "    def _decode_lin(self, raw, ts):", '''    def _decode_flexray(self, raw, ts):
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

    def _decode_lin(self, raw, ts):''')

# ---------------------------------------------------------------- runtime ledger
RN = "vgpt/live/runtime.py"
sub(RN, '            if bus == "doip": return ("diag", "UDS over DoIP", f"0x{key[1]:04X}")', '''            if bus == "flexray":
                for sid, (desc, idx) in self.r._fxs.items():
                    ent = idx.get(key[1])
                    if ent is not None: return (sid, ent[0].name, f"slot {key[1]}")
                return ("flexray", "unknown slot", f"slot {key[1]}")
            if bus == "doip": return ("diag", "UDS over DoIP", f"0x{key[1]:04X}")''')
sub(RN, '        if bus == "doip":\n            raw = bytes(payload);', '''        if bus == "flexray":
            raw = bytes(payload); return ("flexray", ((raw[1] << 3) | (raw[2] >> 5)) & 0x7FF if len(raw) > 2 else 0)
        if bus == "doip":
            raw = bytes(payload);''')
print("flexray patched")

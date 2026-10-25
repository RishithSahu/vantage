def sub(path, old, new, count=1):
    s = open(path, encoding="utf8").read()
    assert old in s, (path, old[:60])
    open(path, "w", encoding="utf8").write(s.replace(old, new, count))


# ---------------------------------------------------------------- profile
P = "vgpt/vehicle/profile.py"
sub(P, 'BUSES = ("can", "someip", "vhal", "log", "camera", "gnss", "uds", "signal", "table")', 'BUSES = ("can", "someip", "vhal", "log", "camera", "gnss", "uds", "signal", "table", "lin", "flexray")')
sub(P, "        self._dbc, self._cat, self._dtc = {}, {}, None", "        self._dbc, self._cat, self._dtc = {}, {}, None; self._ldf = {}; self._fx = {}")
sub(P, "    def catalog(self, source_id):", '''    def ldf(self, source_id):
        """The LIN Description File of a LIN source."""
        if source_id not in self._ldf:
            from ..protocols import lin
            self._ldf[source_id] = lin.Ldf.load(self.file(self.sources[source_id].raw["ldf"]))
        return self._ldf[source_id]

    def catalog(self, source_id):''')
sub(P, "            if b.bus == \"someip\" and b.source in self.sources: self._check_someip(b, p, err)", "            if b.bus == \"someip\" and b.source in self.sources: self._check_someip(b, p, err)\n            if b.bus == \"lin\" and b.source in self.sources: self._check_lin(b, p, err)")
sub(P, "    def _check_someip(self, b, p, err):", '''    def _check_lin(self, b, p, err):
        s = self.sources[b.source]
        if "ldf" not in s.raw: err(f"sources[{b.source}]", "a LIN source needs an 'ldf' file"); return
        try: ldf = self.ldf(b.source)
        except (OSError, ValueError) as e: err(f"sources[{b.source}].ldf", f"cannot load: {e}"); return
        frame, _, sig = b.key.partition(".")
        fr = ldf.frames.get(frame)
        if fr is None: err(p + ".from", f"frame {frame!r} is not in {s.raw['ldf']}"); return
        if not any(sg.name == sig for sg, _ in fr.signals): err(p + ".from", f"signal {sig!r} is not in frame {frame}")

    def _check_someip(self, b, p, err):''')

# ---------------------------------------------------------------- emitter
E = "vgpt/world/emit.py"
sub(E, "from ..protocols import canframes, isotp, obd as _obd, someip, uds, vhal as _vhal", "from ..protocols import canframes, isotp, lin as _lin, obd as _obd, someip, uds, vhal as _vhal")
sub(E, "        self._obd = []   ", '''        self._lin = []                                                                       # LIN: one frame per (source, frame name), its signals encoded through the LDF
        for sid, src in self.p.sources.items():
            if src.type != "lin": continue
            ldf = self.p.ldf(sid); plan = {}
            for b in self.p.bindings:
                if b.bus != "lin" or b.source != sid: continue
                fname, _, sig = b.key.partition("."); plan.setdefault(fname, [ldf.frames[fname], [], 0.0])[1].append((sig, b)); plan[fname][2] = max(plan[fname][2], b.rate_hz)
            for fname, (fr, sigs, rate) in plan.items(): self._lin.append((ldf, fr, sigs, rate, bool(src.raw.get("enhanced", True))))
        self._obd = []   ''')
sub(E, "        for k, (b, pid, n, scale, off) in enumerate(self._obd):", '''        for i, (ldf, fr, sigs, rate, enhanced) in enumerate(self._lin):
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
        for k, (b, pid, n, scale, off) in enumerate(self._obd):''')

# ---------------------------------------------------------------- transmitter
sub("vgpt/car/transmit.py", '            elif bus == "vhal": self._out(ts, ("hal", "vhal"), vhalparcel.pack(payload), "vhal")', '            elif bus == "vhal": self._out(ts, ("hal", "vhal"), vhalparcel.pack(payload), "vhal")\n            elif bus == "lin": self._out(ts, ("lin", ""), bytes(payload), "lin")')

# ---------------------------------------------------------------- sniffer
S = "vgpt/pipe/sniff.py"
sub(S, "from ..protocols import canframes as cf, logs as _logs, nmea as _nmea, someip as _someip, vhalparcel", "from ..protocols import canframes as cf, lin as _lin, logs as _logs, nmea as _nmea, someip as _someip, vhalparcel")
sub(S, 'KINDS = ("can", "someip", "vhal", "image", "nmea", "json", "log", "text", "unknown")', 'KINDS = ("can", "someip", "vhal", "lin", "image", "nmea", "json", "log", "text", "unknown")')
sub(S, "    if vhalparcel.looks_like(b): return \"vhal\"\n", "    if vhalparcel.looks_like(b): return \"vhal\"\n    if 4 <= len(b) <= 11 and b[0] == _lin.SYNC and _lin.looks_like_frame(b): return \"lin\"                    # sync byte, protected identifier with valid parity, checksum that agrees\n")
sub(S, '        if kind == "someip": return [("someip", b)]', '        if kind == "someip": return [("someip", b)]\n        if kind == "lin": return [("lin", b)]')

# ---------------------------------------------------------------- fingerprint
F = "vgpt/pipe/fingerprint.py"
sub(F, "        self.someip = collections.Counter(); self.vhal = collections.Counter();", "        self.lin = collections.Counter(); self.someip = collections.Counter(); self.vhal = collections.Counter();")
sub(F, "    def add_someip(self, service):", "    def add_lin(self, ident): self.frames[\"lin\"] += 1; self.lin[ident] += 1\n    def add_someip(self, service):")
sub(F, '        self.someip = set(); self.vhal = set(); self.has =', '''        self.lin = set()
        for sid, src in p.sources.items():
            if src.type == "lin" and "ldf" in src.raw:
                try: self.lin |= set(p.ldf(sid).by_id)
                except (OSError, ValueError): pass
        self.someip = set(); self.vhal = set(); self.has =''')
sub(F, "    services = {s: n for s, n in inv.someip.items() if s != SD_SERVICE}", '''    if inv.lin:
        n = sum(inv.lin.values()); hit = sum(c for i, c in inv.lin.items() if i in entry.lin)
        parts["lin"] = hit / n; sp = len([i for i in inv.lin if i in entry.lin]) / max(len(entry.lin), 1)
        weights["lin"] = (math.log1p(n), 0.7 * parts["lin"] + 0.3 * min(1.0, sp)); why.append(f"{len([i for i in inv.lin if i in entry.lin])} of {len(inv.lin)} LIN frame identifiers are in its LIN description")
    services = {s: n for s, n in inv.someip.items() if s != SD_SERVICE}''')

# ---------------------------------------------------------------- receiver
R = "vgpt/pipe/receiver.py"
sub(R, "from ..protocols import canframes as cf, isotp as _isotp, someip as _someip", "from ..protocols import canframes as cf, isotp as _isotp, lin as _lin, someip as _someip")
sub(R, '        elif bus == "vhal":\n            inv.add_vhal(', '        elif bus == "lin":\n            try: inv.add_lin(_lin.decode_frame(bytes(obj))["id"])\n            except _lin.LinError: inv.frames["lin"] += 1\n        elif bus == "vhal":\n            inv.add_vhal(')

# ---------------------------------------------------------------- router
RT = "vgpt/vehicle/router.py"
sub(RT, "from ..protocols import dbc as _dbc, isotp as _isotp, j1939 as _jtp, logs as _logs,", "from ..protocols import dbc as _dbc, isotp as _isotp, j1939 as _jtp, lin as _lin, logs as _logs,")
sub(RT, "        for src in self.p.sources.values():\n            if src.type == \"can\": self._index_can(src)", '''        self._lin = {}
        for src in self.p.sources.values():
            if src.type == "can": self._index_can(src)
            elif src.type == "lin": self._index_lin(src)''')
sub(RT, "    # ------------------------------------------------------------------------------------------------ entry point", '''    def _index_lin(self, src):
        ldf = self.p.ldf(src.id); idx = {}
        for b in self.p.bindings:
            if b.bus != "lin" or b.source != src.id: continue
            fname, _, sig = b.key.partition("."); fr = ldf.frames.get(fname)
            if fr is not None: idx.setdefault(fr.id, (fr, {}))[1].setdefault(sig, []).append(b)
        self._lin[src.id] = (ldf, idx)

    # ------------------------------------------------------------------------------------------------ entry point''')
sub(RT, '            if bus == "table": return self._decode_table(payload, ts)', '            if bus == "table": return self._decode_table(payload, ts)\n            if bus == "lin": return self._decode_lin(payload, ts)')
sub(RT, "    def _decode_table(self, row, ts):", '''    def _decode_lin(self, raw, ts):
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

    def _decode_table(self, row, ts):''')

# ---------------------------------------------------------------- runtime ledger
RN = "vgpt/live/runtime.py"
sub(RN, '            if bus == "vhal":\n                name = self.r._vhal_dec.by_id.get(key[1], f"property {key[1]}"); return ("vhal", name, f"0x{key[1]:X}")', '''            if bus == "lin":
                for sid, (ldf, idx) in self.r._lin.items():
                    ent = idx.get(key[1])
                    if ent is not None: return (sid, ent[0].name, f"0x{key[1]:02X}")
                return ("lin", "unknown frame", f"0x{key[1]:02X}")
            if bus == "vhal":
                name = self.r._vhal_dec.by_id.get(key[1], f"property {key[1]}"); return ("vhal", name, f"0x{key[1]:X}")''')
sub(RN, '        if bus == "vhal": return ("vhal", payload.get("prop") if isinstance(payload, dict) else 0)', '''        if bus == "lin":
            raw = bytes(payload); return ("lin", raw[1] & 0x3F if len(raw) > 1 else 0)
        if bus == "vhal": return ("vhal", payload.get("prop") if isinstance(payload, dict) else 0)''')
print("lin patched")

"""Vantage's end of the link: whatever a car sends, work out what it is and store what it means.

    pipe  --records-->  COLLECTOR (this thread: look at the bytes, count the identifiers, decode with the vehicle's decoder file)  --queue-->  STORE (one thread: the only writer of the vehicle's database)
                                                                                                                                        QUERY (the assistant: read-only handles, elsewhere)

What the receiver is NOT told: the car's make, model, buses, protocols or channel numbers.  What it does, in order:
  1. LISTEN     every block is looked at (vgpt/pipe/sniff.py), turned into what the decoders take, and counted into an inventory of identifiers.  Nothing is stored yet; the blocks wait in a buffer.
  2. IDENTIFY   after a few seconds of the car's time the inventory is compared with every installed decoder file (vgpt/pipe/fingerprint.py).  The best fit is used and says why.  If no decoder file fits, what the
                standards allow is learned from the traffic itself (OBD-II over ISO-TP, SAE J1939, Android VHAL property names, NMEA, named readings - vgpt/ingest/autoprofile.py) and written down as a new decoder file.
                If that finds nothing either, the link stays 'unidentified': the identifiers are listed (a decoder file for them is what is missing) and the receiver tries again whenever the library changes.
  3. RUN        the vehicle's own database is opened (one per vehicle, so a second car does not mix into the first one's history), the waiting blocks are decoded into it, and from then on every block is.
  4. ADAPT      the inventory keeps growing; if another decoder file fits clearly better (a different car, or a second bus that came alive later) the receiver switches, replaying the last minute into the new vehicle.
The pipe is one-way: this module can read it and nothing else.  There is no network anywhere in it."""
import collections
import hashlib
import json
import os
import queue
import re
import shutil
import sqlite3
import threading
import time

from . import fingerprint as FP, sniff, wire
from ..ingest import autoprofile as AP, readers as RD
from ..protocols import canframes as cf, doip as _doip, flexray as _fx, isotp as _isotp, lin as _lin, someip as _someip

LISTEN_S = 5.0                                                          # seconds of the car's time to listen before deciding
MIN_CAN_FRAMES = 150
MAX_LISTEN_S = 25.0
RING_S = 60.0                                                           # how much is kept for replaying into a vehicle that is switched to
RE_EVAL_S = 30.0
RE_EVAL_LEARNED_S = 15.0                                                # a vehicle read by the standards may have properties that report rarely (an odometer every five seconds): it is looked at again sooner
SWITCH_MARGIN = 0.15
VIN_RE = re.compile(rb"[A-HJ-NPR-Z0-9]{17}")
QUEUE_MAX = 40000


class Session:
    """One connection of a car to Vantage."""
    def __init__(self, n, now):
        self.n, self.opened = n, now; self.inv = FP.Inventory(); self.sniff = {}; self.hints = {}; self.buffer = []; self.ring = collections.deque(); self.state = "listening"
        self.entry = None; self.decision = None; self.rt = None; self.key = None; self.dir = None; self.how = ""; self.chan_label = {}; self.channel_map = {}
        self.offset = None; self.time_kind = None; self.first_ts = None; self.last_ts = None; self.records = 0; self.next_eval = None; self.vin = None; self.tp = {}
        self.import_state = None; self.learn_seen = None; self.unreadable = collections.Counter(); self.log_seen = 0; self.closed = False; self.listen_started = None; self.retry_at = 0.0; self.lib_stamp = None; self.lc = 0.0; self.picture = None; self.pictures = 0; self.analyse_images = None

    def align(self, ts, arrival):
        if ts is None: return arrival
        if self.offset is None: self.offset, self.time_kind = wire.time_base(ts, arrival)
        return ts + self.offset


class LinkService:
    def __init__(self, home, roots=None, queue_size=QUEUE_MAX, drop_when_full=False, clock=time.time, monitor=True, listen_s=LISTEN_S, tier=None, accept_history=None):
        self.accept_history = (os.environ.get("VGPT_ACCEPT_HISTORY", "1") != "0") if accept_history is None else bool(accept_history)           # the test bench's instant load of a finished history
        self.home = home; self.clock = clock; self.monitor = monitor; self.listen_s = listen_s; self.tier = tier; self.drop_when_full = drop_when_full
        os.makedirs(os.path.join(home, "vehicles"), exist_ok=True); os.makedirs(os.path.join(home, "decoders"), exist_ok=True); os.makedirs(os.path.join(home, "learned"), exist_ok=True)
        bundled = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vehicle", "packs")
        # None means the normal installed product library.  An explicit empty
        # iterable means exactly that: no pre-installed vehicle decoders.  The
        # latter is important both for bring-up and for proving that standards
        # traffic can be learned without accidentally matching a bundled pack.
        installed = [bundled] if roots is None else list(roots)
        if roots is not None and installed and bundled not in installed:
            installed.append(bundled)
        self.library = FP.DecoderLibrary([os.path.join(home, "decoders"), *installed, os.path.join(home, "learned")])
        self.mu = threading.RLock(); self.session = None; self.sessions = 0; self.reader = None; self.q = queue.Queue(queue_size); self.store_thread = None; self.luma = sniff.luma_decoder()
        self.vehicle_seq = 0; self.dropped = 0; self.last_vehicle = None; self.events = collections.deque(maxlen=50); self._stop = threading.Event()
        self.parser_stats = {}

    # ---------------------------------------------------------------------------------------------------------- the pipe
    def attach(self, fd):
        """Start reading a pipe.  A new connection is a new session; the previous one's vehicle database is closed cleanly (its file stays readable for questions)."""
        with self.mu:
            if self.reader is not None: self.reader.stop()
            self._close_runtime()
            self.sessions += 1; ses = self.session = Session(self.sessions, self.clock()); self._note(f"a car connected (session {self.sessions})")
            self.reader = wire.PipeReader(fd, lambda rec, ses=ses: self._on_record(rec, ses), lambda rd, ses=ses: self._on_close(rd, ses), name=f"vlink-{self.sessions}"); self.reader.start()          # a reader only ever feeds ITS session: a block still in flight from the previous car is dropped
        return self.reader

    def _note(self, text): self.events.append({"t": self.clock(), "text": text})

    def _on_close(self, reader, ses=None):
        with self.mu:
            ses = ses or self.session
            if ses is not self.session: return
            self.parser_stats = reader.parser.stats() | {"bytes": reader.bytes}
            if ses is not None and reader is self.reader:
                ses.closed = True
                if ses.state in ("listening", "unidentified") and ses.buffer: self._identify(ses, force=True)            # the link ended: one last look at everything that arrived (an earlier look may have been taken on the first few frames)
                self._close_runtime(ses)                                                       # everything handed over is stored and the vehicle's database is closed cleanly; it stays open for questions
                self._note("the car disconnected")

    def stop(self):
        self._stop.set()
        with self.mu:
            if self.reader is not None: self.reader.stop()
        self._close_runtime()

    def join(self, timeout=10.0):
        r = self.reader
        if r is not None: r.join(timeout)

    # ---------------------------------------------------------------------------------------------------------- collector
    def _on_record(self, rec, ses=None):
        arrival = self.clock()
        with self.mu:
            if ses is not None and ses is not self.session: return
            ses = self.session
            if ses is None: return
            ses.records += 1
            if rec.channel == wire.CONTROL:
                o = rec.control()
                if o:
                    ses.hints.update({k: v for k, v in o.items() if isinstance(k, str)})
                    v = o.get("vin")
                    if isinstance(v, str) and VIN_RE.fullmatch(v.encode()): ses.vin = ses.vin or v
                return
            sn = ses.sniff.get(rec.channel)
            if sn is None: sn = ses.sniff[rec.channel] = sniff.ChannelSniffer()
            one = sn.observe(rec.payload); kind, share, n = sn.verdict()
            use = kind if (sn.settled() and one in ("unknown", kind, "can", "someip") and share >= 0.8) else one
            ts = ses.align(rec.ts, arrival)
            if ses.first_ts is None: ses.first_ts = ts
            ses.last_ts = ts; ses.inv.seen(ts)
            if use == "image":
                ses.picture = (ts, bytes(rec.payload)); ses.pictures += 1                                  # the latest picture is kept for the screen
                if ses.analyse_images is False: ses.inv.frames["cam"] += 1; ses.inv.images += 1; return        # the vehicle reports its camera's measurements itself: the picture is only shown
            items = sniff.to_bus(use, rec.payload, ts, self.luma)
            if not items: ses.unreadable[use] += 1; ses.inv.unknown[(rec.channel, use)] += 1; return
            for bus, obj in items: self._take(ses, rec.channel, bus, obj, ts)

    def _take(self, ses, ch, bus, obj, ts):
        self._note_inventory(ses, ch, bus, obj, ts)
        if ses.state == "running":
            self._ingest(ses, ch, bus, obj, ts)
            if ses.next_eval is not None and ts >= ses.next_eval: ses.next_eval = ts + (RE_EVAL_LEARNED_S if ses.how.startswith("learned") else RE_EVAL_S); self._reconsider(ses)
            return
        ses.buffer.append((ts, ch, bus, obj))
        if len(ses.buffer) > 400000: del ses.buffer[:100000]
        if ses.state in ("listening", "unidentified") and self._ready(ses, ts): self._identify(ses)

    def _note_inventory(self, ses, ch, bus, obj, ts):
        inv = ses.inv
        if bus == "can":
            inv.add_can(ch, obj)
            if FP.is_diag(obj.can_id & 0x1FFFFFFF, bool(obj.extended)) and ses.vin is None: self._vin_from_isotp(ses, obj, ts)
        elif bus == "someip":
            raw = bytes(obj); pos = 0
            while len(raw) - pos >= 16:
                svc, _, ln = int.from_bytes(raw[pos:pos + 2], "big"), 0, int.from_bytes(raw[pos + 4:pos + 8], "big")
                inv.add_someip(svc); pos += 8 + max(ln, 8)
        elif bus == "doip":
            try: d = _doip.decode(bytes(obj))
            except _doip.DoipError: inv.frames["doip"] += 1; return
            inv.add_doip(d.get("logical_address", d.get("source_address", d.get("entity_address", 0))))
            vin = d.get("vin") if d["name"] == "vehicle_announcement" else (bytes(d["data"][3:]).decode("ascii", "replace") if d.get("data", b"")[:3] == b"\x62\xf1\x90" else None)
            if ses.vin is None and vin and VIN_RE.fullmatch(vin.encode()): ses.vin = vin
        elif bus == "flexray":
            try: inv.add_flexray(_fx.decode_frame(bytes(obj))["slot"])
            except _fx.FlexrayError: inv.frames["flexray"] += 1
        elif bus == "lin":
            try: inv.add_lin(_lin.decode_frame(bytes(obj))["id"])
            except _lin.LinError: inv.frames["lin"] += 1
        elif bus == "vhal":
            inv.add_vhal(int(obj.get("prop", 0)), int(obj.get("area", 0)))
            if int(obj.get("prop", 0)) == 0x11100100 and obj.get("string") and ses.vin is None and VIN_RE.fullmatch(str(obj["string"]).encode()): ses.vin = obj["string"]          # INFO_VIN
        elif bus == "log": inv.frames["log"] += 1; inv.log_sources[_log_tag(obj)] += 1
        elif bus == "gnss": inv.frames["gnss"] += 1; inv.nmea += 1
        elif bus == "cam": inv.frames["cam"] += 1; inv.images += 1
        elif bus == "table":
            inv.frames["table"] += 1
            for k in obj: inv.table_keys[k] += 1

    def _vin_from_isotp(self, ses, frame, ts):
        r = ses.tp.setdefault(frame.can_id, _isotp.Reassembler())
        try: msg = r.feed(frame.data, now=ts)
        except Exception: return
        if msg and (bytes(msg[:3]) == b"\x62\xf1\x90" or bytes(msg[:3]) == b"\x49\x02\x01"):
            m = VIN_RE.search(bytes(msg[3:]))
            if m: ses.vin = m.group().decode()

    def _ready(self, ses, ts):
        if ses.listen_started is None: ses.listen_started = ts
        waited = ts - ses.listen_started
        if ses.state == "unidentified":
            now = time.monotonic()
            if now - ses.lc > 1.0:
                ses.lc = now
                if self.library.stamp() != ses.lib_stamp: return True                      # a decoder file was added: look again at once
            return ts >= ses.retry_at
        if waited >= MAX_LISTEN_S: return True
        enough = ses.inv.total_can() >= MIN_CAN_FRAMES or ses.inv.frames["someip"] + ses.inv.frames["vhal"] >= 40 or sum(ses.inv.frames.values()) >= 120
        return waited >= self.listen_s and enough

    # ---------------------------------------------------------------------------------------------------------- identify
    def _identify(self, ses, force=False):
        self.library.refresh()
        best, runners, amb = self.library.identify(ses.inv)
        decision = None; entry = None; how = ""
        # VHAL is self-describing, standardised traffic.  A different vehicle
        # pack may coincidentally bind many of the same standard property IDs;
        # that is not evidence for its make/model.  Learn the VHAL contract
        # directly when it is the sole vehicle bus instead of misidentifying a
        # generic AAOS source as whichever installed EV shares those IDs.
        standards_first = ses.inv.frames["vhal"] > 0 and ses.inv.total_can() == 0 and ses.inv.frames["someip"] == 0
        if standards_first:
            learned = self._learn(ses)
            if learned is not None: entry, how, decision = learned
        if entry is None and best is not None:
            entry = best.entry
            learned_root = os.path.abspath(os.path.join(self.home, "learned"))
            try: learned = os.path.commonpath((learned_root, os.path.abspath(entry.path))) == learned_root
            except ValueError: learned = False
            how = "learned from the standards" if learned else "decoder file"
            decision = dict(best.to_json(), mode=how, ambiguous=amb, runners_up=[m.to_json() for m in runners])
            if learned: decision["unexplained_can_ids"] = self._unexplained(ses, entry)
        elif entry is None:
            learned = self._learn(ses)
            if learned is not None: entry, how, decision = learned
        if entry is None:
            ses.state = "unidentified"; ses.retry_at = (ses.last_ts or 0) + min(60.0, max(10.0, (ses.last_ts or 0) - (ses.listen_started or 0))); ses.lib_stamp = self.library._stamp
            ses.decision = {"mode": "unidentified", "best_candidates": [m.to_json() for m in runners], "why": ["no installed decoder file fits what this link carries, and the standards (OBD-II, J1939, Android VHAL, NMEA, named readings) found nothing to read"],
                            "unexplained_can_ids": self._unexplained(ses, None)}
            return False
        self._start(ses, entry, how, decision); return True

    def _learn(self, ses, more_than=0):
        """No decoder file fits: learn what the standards allow from the buffered traffic and write it down as a decoder file.  `more_than`: only when it reads more quantities than that (a second look)."""
        recs = [RD.Rec(ts, bus, obj) for ts, ch, bus, obj in ses.buffer if bus in ("can", "vhal", "gnss", "log", "table")]
        if not recs: return None
        try: a = AP.analyse_records(recs)
        except Exception as e: self._note(f"learning from the standards failed: {type(e).__name__}"); return None
        if len(a.signals) <= more_than: return None
        kinds = sorted({{"uds": "obd", "can": "j1939" if any(s.get("j1939") for s in a.sources) else "can", "vhal": "vhal", "table": "table"}.get(s["type"], s["type"]) for s in a.sources if s["type"] in ("uds", "can", "vhal", "table")})
        name = "learned_" + "_".join(kinds); out = os.path.join(self.home, "learned", name)
        try: AP.write_pack(a, out, vehicle_id=name, model="learned from " + "+".join(kinds))
        except Exception as e: self._note(f"the learned decoder file did not validate: {type(e).__name__}: {e}"); return None
        self.library.refresh(force=True); entry = self.library.entries.get(name)
        if entry is None: return None
        return entry, "learned from the standards", {"mode": "learned from the standards", "id": name, "model": entry.model, "score": None, "why": [f"no installed decoder file fits; read {len(a.signals)} quantities from {', '.join(kinds)} traffic by the standards' own scales",
                                                                                                                             *a.notes], "unmapped": list(a.unmapped)[:30], "unexplained_can_ids": self._unexplained(ses, entry), "report": a.report()[:4000]}

    MAX_HISTORY_BYTES = 16 * 1024 ** 3

    def _import_history(self, ses, entry, vdir, db):
        """INSTANT LOAD, for testing: the car says (a control record) that a finished history of this vehicle is on this computer; the vehicle's database is made from it in one copy instead of the history being sent
        second by second.  Refused, with the reason written to the link's events, unless every check passes: the switch is on, the path is an absolute path of a regular SQLite file of at most 16 GB, it has the
        tables of a vehicle history, every quantity in it is one the identified decoder file binds, and this vehicle has no stored history yet.  Nothing is executed from the file; it is opened read-only and copied."""
        h = ses.hints.get("history")
        if not isinstance(h, dict) or "path" not in h: return
        st = ses.import_state = {"phase": "checking", "path": str(h.get("path"))[:300], "pages": 0, "total": 0, "ok": False, "reason": ""}
        def refuse(why): st.update(phase="refused", reason=why); self._note("instant load refused: " + why)
        if not self.accept_history: return refuse("this Vantage does not accept an instant load (VGPT_ACCEPT_HISTORY=0)")
        path = str(h["path"])
        if not os.path.isabs(path) or not os.path.isfile(path) or not path.lower().endswith((".db", ".sqlite")): return refuse("the history is not an absolute path of a .db / .sqlite file")
        size = os.path.getsize(path)
        if size > self.MAX_HISTORY_BYTES: return refuse("the file is larger than 16 GB")
        if os.path.exists(db) and os.path.getsize(db) > 1 << 20: return refuse("this vehicle already has a stored history")
        try:
            src = sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)
            try:
                tabs = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"meta", "sig_1m", "sig_catalog"} <= tabs: return refuse("the file is not a vehicle history (no meta / sig_1m / sig_catalog)")
                have = {r[0] for r in src.execute("SELECT concept FROM sig_catalog")}; known = set(entry.profile.concept_names()) | {"_ignition", "_charging_active"}
                extra = sorted(have - known)
                if extra: return refuse("the history holds quantities this vehicle's decoder file does not bind: " + ", ".join(extra[:6]))
                lo, hi = src.execute("SELECT MIN(ts), MAX(ts) FROM sig_1m").fetchone()
                os.makedirs(vdir, exist_ok=True); tmp = db + ".loading"; dst = sqlite3.connect(tmp); st.update(phase="copying", total=1)
                def progress(status, remaining, total): st.update(pages=total - remaining, total=total)
                t0 = time.time(); src.backup(dst, pages=4096, progress=progress); dst.commit(); dst.close()
            finally: src.close()
            for e in ("", "-wal", "-shm"):
                try: os.remove(db + e)
                except OSError: pass
            os.replace(tmp, db)
            arch = os.path.join(os.path.dirname(path), "archive")
            if os.path.isdir(arch): shutil.copytree(arch, os.path.join(vdir, "archive"), dirs_exist_ok=True)
            st.update(phase="done", ok=True, seconds=round(time.time() - t0, 1), bytes=size, first=lo, last=hi)
            self._note(f"instant load: {size / 1e6:.0f} MB of history ({(hi - lo) / 86400 / 365:.1f} years) loaded in {st['seconds']} s")
        except (sqlite3.Error, OSError) as e:
            st.update(phase="failed", reason=f"{type(e).__name__}: {e}"); self._note("instant load failed: " + st["reason"])
            try: os.remove(db + ".loading")
            except OSError: pass

    def _unexplained(self, ses, entry):
        out = []
        for (ident, ext), n in ses.inv.all_can().most_common(40):
            if entry is None or not entry.defines((ident, ext)): out.append({"id": f"0x{ident:X}", "extended": ext, "frames": n, "dlc": sorted(ses.inv.dlc.get((ident, ext), {}))})
        return out[:20]

    # ---------------------------------------------------------------------------------------------------------- run
    def _start(self, ses, entry, how, decision):
        from ..live.runtime import Runtime
        key = entry.id + (("-" + ses.vin) if ses.vin else "")
        if self.session is not ses: return
        self._close_runtime(ses)
        d = os.path.join(self.home, "vehicles", re.sub(r"[^A-Za-z0-9_.-]", "_", key)); os.makedirs(d, exist_ok=True); db = os.path.join(d, "vantage.db")
        self._import_history(ses, entry, d, db)
        rt = Runtime(db, profile=entry.profile, monitor=self.monitor, tier=self.tier)
        ses.rt, ses.entry, ses.key, ses.dir, ses.how, ses.decision = rt, entry, key, d, how, decision; ses.state = "running"; ses.analyse_images = not any(b.bus == "camera" for b in entry.profile.bindings); ses.next_eval = (ses.last_ts or 0) + (RE_EVAL_LEARNED_S if how.startswith("learned") else RE_EVAL_S); ses.learn_seen = tuple(sorted(ses.inv.summary().items()))
        match = decision.get("channel_map") or {}; ses.channel_map = dict(match); ses.chan_label = {ch: entry.sources[sid]["channel"] for ch, sid in match.items() if sid in entry.sources and entry.sources[sid]["channel"]}
        with open(os.path.join(d, "identity.json"), "w", encoding="utf-8") as f:
            json.dump({"key": key, "decoder": entry.id, "model": entry.model, "how": how, "vin": ses.vin, "decision": {k: v for k, v in decision.items() if k != "report"}, "identified_at": self.clock()}, f, indent=1, default=str)
        self.store_thread = threading.Thread(target=self._store_loop, args=(rt, self.q), daemon=True, name="vlink-store"); self.store_thread.start()
        buffered, ses.buffer = ses.buffer, []
        for ts, ch, bus, obj in buffered: self._ingest(ses, ch, bus, obj, ts)
        self.vehicle_seq += 1; self.last_vehicle = {"key": key, "id": entry.id, "model": entry.model, "db": db, "dir": d, "how": how}
        self._note(f"recognised the vehicle: {entry.model} ({how}{', ambiguous between ' + ', '.join(m['id'] for m in decision.get('runners_up', [])[:1]) if decision.get('ambiguous') else ''})")

    def _ingest(self, ses, ch, bus, obj, ts):
        if bus == "can" and ses.chan_label: obj.channel = ses.chan_label.get(ch, "")
        if ses.rt is None: return
        recs = ses.rt.decode(ts, bus, obj)
        ses.ring.append((ts, ch, bus, obj))
        while ses.ring and ses.ring[0][0] < ts - RING_S: ses.ring.popleft()
        try:
            if self.drop_when_full: self.q.put_nowait((ts, bus, recs))
            else: self.q.put((ts, bus, recs))
        except queue.Full: self.dropped += 1

    def _store_loop(self, rt, q):
        last_pub = time.monotonic()
        while True:
            try: item = q.get(timeout=0.5)
            except queue.Empty: item = None
            if item is None:
                if self._stop.is_set(): break
                if time.monotonic() - last_pub > 1.5: rt.publish(); last_pub = time.monotonic()
                continue
            if item is _END: break
            ts, bus, recs = item; rt.store(ts, bus, recs)
            if time.monotonic() - last_pub > 1.5: rt.publish(); last_pub = time.monotonic()

    def _close_runtime(self, ses=None):
        t = self.store_thread
        if t is not None and t.is_alive():
            self.q.put(_END); t.join(timeout=60.0)
        self.store_thread = None
        s = ses or self.session
        rts = {x.rt for x in (s,) if x is not None and x.rt is not None}
        for rt in rts:
            try: rt.close()
            except Exception: pass
        if s is not None: s.rt = None

    def _relearn(self, ses):
        """A vehicle that is read by the standards: when the link has shown something new since it was learned (a property that reports every few seconds, a second bus), read the last minute again and keep the larger result."""
        now = tuple(sorted(ses.inv.summary().items()))
        if now == ses.learn_seen or not ses.ring: return False
        ses.learn_seen = now; items = list(ses.ring); saved, ses.buffer = ses.buffer, items
        try: learned = self._learn(ses, more_than=len(ses.entry.profile.bindings))
        finally: ses.buffer = saved
        if learned is None: return False
        entry, how, decision = learned; self._note(f"the link now shows more of the vehicle ({len(entry.profile.bindings)} readings instead of {len(ses.entry.profile.bindings)}): reading it again")
        self._close_runtime(ses); ses.state = "listening"; ses.buffer = items; self._start(ses, entry, how, decision); return True

    def _reconsider(self, ses):
        """The inventory has grown: does another decoder file fit clearly better than the one in use?"""
        if ses.how.startswith("learned") and self._relearn(ses): return
        self.library.refresh(); ranked = self.library.rank(ses.inv, 3)
        if not ranked: return
        cur = next((m for m in ranked if m.entry.id == ses.entry.id), None) or FP.score(ses.entry, ses.inv)
        top = ranked[0]
        if top.entry.id != ses.entry.id and top.score >= FP.ACCEPT and top.score - cur.score >= SWITCH_MARGIN:
            self._note(f"the traffic now fits {top.entry.model} ({top.score:.2f}) better than {ses.entry.model} ({cur.score:.2f}): switching")
            items = list(ses.ring); self._close_runtime(ses); ses.state = "listening"; ses.buffer = items
            self._start(ses, top.entry, "decoder file", dict(top.to_json(), mode="decoder file", switched_from=ses.entry.id, ambiguous=False, runners_up=[]))
        else:
            ses.decision["score"] = round(cur.score, 3); ses.decision["channel_map"] = cur.channel_map
            if cur.channel_map:
                ses.channel_map = dict(cur.channel_map); ses.chan_label = {ch: ses.entry.sources[sid]["channel"] for ch, sid in cur.channel_map.items() if sid in ses.entry.sources and ses.entry.sources[sid]["channel"]}

    # ---------------------------------------------------------------------------------------------------------- views
    def runtime(self):
        with self.mu:
            s = self.session
            return s.rt if s is not None else None

    def picture(self):
        """(time, JPEG bytes) of the latest picture the car's camera sent, or None."""
        with self.mu:
            s = self.session
            return s.picture if s is not None else None

    def drain(self, timeout=30.0):
        """Wait until everything the collector has handed over is stored (tests, shutdown)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.q.empty(): time.sleep(0.01)
        rt = self.runtime()
        if rt is not None: time.sleep(0.05); rt.publish()

    def status(self):
        with self.mu:
            s = self.session
            if s is None: return {"state": "waiting", "text": "waiting for a car", "events": list(self.events)[-8:], "decoders": sorted(self.library.entries), "decoder_errors": self.library.errors}
            sn = []
            for ch, c in sorted(s.sniff.items()):
                k, share, n = c.verdict(); sn.append({"channel": ch, "kind": k, "share": round(share, 2), "blocks": n, "source": s.channel_map.get(ch), "hint": s.hints.get("channels", {}).get(str(ch)) if isinstance(s.hints.get("channels"), dict) else None})
            state = s.state if not (s.closed and s.state == "running") else "disconnected"
            ent = s.entry
            return {"state": state, "session": s.n, "closed": s.closed, "records": s.records, "vehicle": None if ent is None else {"key": s.key, "id": ent.id, "model": ent.model, "powertrain": ent.powertrain, "db": os.path.join(s.dir, "vantage.db"), "how": s.how, "vin": s.vin},
                    "decision": s.decision, "channels": sn, "inventory": s.inv.summary(), "wire": self.parser_stats or (self.reader.parser.stats() if self.reader else {}), "time": {"kind": s.time_kind, "offset": s.offset, "first": s.first_ts, "last": s.last_ts},
                    "unreadable": dict(s.unreadable), "import": s.import_state, "pictures": s.pictures, "queue": self.q.qsize(), "dropped": self.dropped, "events": list(self.events)[-8:], "decoders": sorted(self.library.entries), "decoder_errors": self.library.errors, "hints": {k: v for k, v in s.hints.items() if k != "vin"}}


_END = object()


def _log_tag(line):
    parts = str(line).split(" ", 6)
    if len(parts) >= 5:
        return (parts[3] if parts[3] and not parts[3].isdigit() else parts[4])[:20]
    return "log"

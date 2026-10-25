"""The live runtime: the ONE writer of a vehicle database.  Raw traffic in (from any collector), the assistant's history out.

    collectors / simulator --(ts, bus, payload)--> Runtime.feed --> ProfileRouter (this vehicle's mappings) --> Ingestor (1 Hz rows, events, logs) --> Lake (every other signal) --> SQLite

Beyond decode-and-store the runtime does what a deployed service has to do:
  * context freezing    the channels around every important event are kept in a short RAM ring and frozen into the database 30 s after the event (event_ctx_x), so the cause survives the 1 Hz expiry
  * ledger              trips and charging sessions are derived from the stored rows every two minutes (the same derivation the bulk importer uses)
  * housekeeping        once a day the 1 Hz tier is rolled into minutes and old tiers are expired by the retention the vehicle's profile declares
  * source health       a bus that goes silent while the car is awake is recorded (source_log) and its recovery too, so "the body bus dropped out at 14:03" is a recorded fact, not an inference
  * bus monitor         every message is decoded to its message / service / property name for the live bus view: what arrived, how often, what it decoded to
  * statistics          rates, decode time per message, drops, row counts - the numbers a validation engineer needs to size the box

Thread-safe: collectors and the HTTP layer may call `feed` and `snapshot` concurrently.  No numpy; stdlib sqlite3 only."""
import collections
import json
import math
import os
import struct
import threading
import time
import zlib

from .. import signal_registry as R, store
from ..protocols import canframes as _cf
from ..vehicle import lake as LK, ledger
from ..vehicle.profile import Profile
from ..vehicle.router import ProfileRouter

LEDGER_EVERY_S = 120
SILENT_FACTOR = 5.0


class LiveIngestor(store.Ingestor):
    """The ingestor plus the freezing of event_ctx_x (the non-core channels around an event)."""
    def __init__(self, *a, concepts=LK.CTX_X_CONCEPTS, xring_s=420, **k):
        super().__init__(*a, **k); self.xconcepts = tuple(concepts); self.xring = collections.deque(maxlen=max(int(xring_s), 200)); self.xpend = []
        self._rolled = None                                                                                            # the first minute not yet rolled up into sig_1m
        self.drop_location = False                                                                                     # the driver erased the location history: new fixes are not stored either

    def _gps_write(self, sec):
        if self.drop_location: self.gps_dirty = False; return
        super()._gps_write(sec)

    def _flush(self, sec):
        super()._flush(sec); m = sec // 60 * 60
        if self._rolled is None: self._rolled = m
        elif m > self._rolled:                                                                                       # every minute before this one is complete: roll it into the minute tier now, not at midnight
            try: store.rollup(self.c, self._rolled, m)
            except Exception: pass
            self._rolled = m
        row = {}
        for c in self.xconcepts:
            it = self.latest.get(c)
            if it is not None and 0 <= sec - it[1] <= 3: row[c] = it[0]
        self.xring.append((sec, row)); self._freeze_x(sec)

    def close(self):
        super().close()
        if self.sec is not None and self._rolled is not None:                                                         # the minute that was still open when the link ended: a short session still has a minute row
            try: store.rollup(self.c, self._rolled, self.sec // 60 * 60 + 60)
            except Exception: pass

    def _event(self, ts, ty, sv, de):
        super()._event(ts, ty, sv, de)
        if ty in LK.CTX_TYPES_X: self.xpend.append((ts, ty))

    def _freeze_x(self, sec):
        keep = []
        for (ts, ty) in self.xpend:
            if sec is not None and sec < ts + 30: keep.append((ts, ty)); continue
            a0, a1 = ts - 120, ts + 30; rows = {s: r for s, r in self.xring if a0 <= s < a1}
            sig = {c: [rows.get(s, {}).get(c) for s in range(a0, a1)] for c in self.xconcepts if any(c in r for r in rows.values())}
            if sig: self.c.execute("INSERT OR REPLACE INTO event_ctx_x VALUES(?,?,?)", (ts, ty, zlib.compress(json.dumps({"t0": a0, "hz": 1, "sig": sig}, separators=(",", ":")).encode(), 6)))
        self.xpend = keep

    def close(self):
        super().close(); self._freeze_x(None)


class BusMonitor:
    """What the live bus view shows: one row per message (CAN id / SOME/IP event / VHAL property / log source ...), with count, rate, last raw bytes and what they decoded to; plus the latest messages."""
    def __init__(self, router, keep=240):
        self.r = router; self.rows = {}; self.tail = collections.deque(maxlen=keep); self.names = {}; self.dec_s = 0.0; self.n = 0; self.t_last = 0.0; self.feed = collections.deque(maxlen=60)       # feed: the events and log lines the buses reported

    def _cat(self, key):
        names = self.names.get(key)
        if names is None:
            names = self.names[key] = self._resolve(key)
        return names

    def _resolve(self, key):
        bus = key[0]
        try:
            if bus == "can":
                ident, ext = key[1], key[2]
                if self.r.has_diag and ident in self.r.diag_ids and not ext: return ("diag", "UDS / OBD-II response", f"0x{ident:03X}")
                for sid, (db, idx, chan) in self.r._can.items():
                    ent = idx.get((ident, ext))
                    if ent is not None: return (sid, ent[0].name, f"0x{ident:X}")
                return ("?", f"unknown frame", f"0x{ident:X}")
            if bus == "someip":
                svc, meth = key[1], key[2]
                for sid, src in self.r.p.sources.items():
                    if src.type != "someip" or "catalog" not in src.raw: continue
                    spec = self.r.p.catalog(sid).spec.get(f"0x{svc:04X}") or self.r.p.catalog(sid).spec.get(svc)
                    if spec is not None:
                        ev = (spec["events"].get(f"0x{meth:04X}") or spec["events"].get(meth) or {}).get("name", f"0x{meth:04X}")
                        return (sid, f"{spec['name']}.{ev}", f"0x{svc:04X}/0x{meth:04X}")
                return ("someip", "service discovery" if svc == 0xFFFF else "service", f"0x{svc:04X}/0x{meth:04X}")
            if bus == "flexray":
                for sid, (desc, idx) in self.r._fxs.items():
                    ent = idx.get(key[1])
                    if ent is not None: return (sid, ent[0].name, f"slot {key[1]}")
                return ("flexray", "unknown slot", f"slot {key[1]}")
            if bus == "doip": return ("diag", "UDS over DoIP", f"0x{key[1]:04X}")
            if bus == "lin":
                for sid, (ldf, idx) in self.r._lin.items():
                    ent = idx.get(key[1])
                    if ent is not None: return (sid, ent[0].name, f"0x{key[1]:02X}")
                return ("lin", "unknown frame", f"0x{key[1]:02X}")
            if bus == "vhal":
                name = self.r._vhal_dec.by_id.get(key[1], f"property {key[1]}"); return ("vhal", name, f"0x{key[1]:X}")
            if bus == "log": return ("log", key[1] or "log", "text")
            if bus == "cam": return ("camera", key[1] or "front camera", "features")
            if bus == "gnss": return ("gnss", key[1] or "NMEA", "text")
        except Exception:
            pass
        return (bus, str(key[1:]), "")

    def key(self, bus, payload):
        if bus == "can":
            if isinstance(payload, _cf.Frame): return ("can", payload.can_id & 0x1FFFFFFF, bool(payload.extended or payload.can_id > 0x7FF))
            return ("can", int(payload[0]) & 0x1FFFFFFF, int(payload[0]) > 0x7FF)
        if bus == "someip":
            raw = bytes(payload)
            return ("someip", *struct.unpack(">HH", raw[:4])) if len(raw) >= 4 else ("someip", 0, 0)
        if bus == "flexray":
            raw = bytes(payload); return ("flexray", ((raw[1] << 3) | (raw[2] >> 5)) & 0x7FF if len(raw) > 2 else 0)
        if bus == "doip":
            raw = bytes(payload); return ("doip", int.from_bytes(raw[8:10], "big") if len(raw) >= 10 else 0)
        if bus == "lin":
            raw = bytes(payload); return ("lin", raw[1] & 0x3F if len(raw) > 1 else 0)
        if bus == "vhal": return ("vhal", payload.get("prop") if isinstance(payload, dict) else 0)
        if bus == "log":
            parts = str(payload).split(" ", 6); src = ""
            if len(parts) >= 5 and parts[-1:]:
                src = parts[3] if parts[3] and not parts[3].isdigit() else parts[4]
            return ("log", src[:20])
        if bus == "cam": return ("cam", "front")
        if bus == "gnss": return ("gnss", str(payload)[3:6])
        return (bus, "")

    def record(self, bus, payload, ts, recs, dt):
        self.n += 1; self.dec_s += dt; self.t_last = ts if ts is not None else self.t_last
        try: k = self.key(bus, payload)
        except Exception: k = (bus, "")
        row = self.rows.get(k)
        if row is None:
            sid, name, ident = self._cat(k); row = self.rows[k] = {"bus": bus, "source": sid, "name": name, "id": ident, "n": 0, "t0": ts or 0.0, "n0": 0, "rate": 0.0, "payload": None, "recs": None, "ts": ts}
        row["n"] += 1; row["payload"] = payload; row["recs"] = recs; row["ts"] = ts
        if ts is not None and ts - row["t0"] >= 3.0: row["rate"] = (row["n"] - row["n0"]) / (ts - row["t0"]); row["t0"] = ts; row["n0"] = row["n"]
        self.tail.append((ts, k, payload, recs))
        for r in recs or ():
            if r[0] == "event": self.feed.append({"ts": ts, "kind": "event", "type": r[1], "sev": r[2], "text": r[3], "source": row["source"], "via": row["name"]})
            elif r[0] == "log": self.feed.append({"ts": ts, "kind": "log", "type": r[1], "sev": "info", "text": r[2], "source": row["source"], "via": row["name"]})

    @staticmethod
    def _hex(payload):
        if isinstance(payload, _cf.Frame): return payload.data.hex(" ").upper()
        if isinstance(payload, (bytes, bytearray)): return bytes(payload)[:24].hex(" ").upper() + (" ..." if len(payload) > 24 else "")
        if isinstance(payload, (tuple, list)) and len(payload) > 1 and isinstance(payload[1], (bytes, bytearray)): return bytes(payload[1]).hex(" ").upper()
        if isinstance(payload, dict): return json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in payload.items() if k != "y" and not isinstance(v, (bytes, bytearray))}, separators=(",", ":"), default=str)[:120]
        return str(payload)[:120]

    @staticmethod
    def _dec(recs):
        out = []
        for r in recs or ():
            if r[0] in ("sig", "sig_ts"): out.append(f"{r[1]}={r[2]:.4g}" if isinstance(r[2], (int, float)) else f"{r[1]}={r[2]}")
            elif r[0] == "event": out.append(f"EVENT {r[1]}: {r[3]}")
            elif r[0] == "log": out.append(f"LOG {r[1]}: {r[2]}")
        return out

    def table(self, limit=200):
        rows = sorted(self.rows.values(), key=lambda r: (r["source"], r["name"]))[:limit]
        return [{"source": r["source"], "bus": r["bus"], "name": r["name"], "id": r["id"], "n": r["n"], "rate_hz": round(r["rate"], 1), "ts": r["ts"], "raw": self._hex(r["payload"]), "decoded": self._dec(r["recs"])} for r in rows]

    def latest(self, n=60):
        out = []
        for ts, k, payload, recs in list(self.tail)[-n:]:
            sid, name, ident = self._cat(k); out.append({"ts": ts, "bus": k[0], "source": sid, "name": name, "id": ident, "raw": self._hex(payload), "decoded": self._dec(recs)})
        return out


class Tap:
    """Wraps the profile router: same interface, but every decode is timed and shown to the bus monitor."""
    wants_ts = True

    def __init__(self, router, monitor): self._r = router; self.mon = monitor
    def __getattr__(self, name): return getattr(self._r, name)

    def decode(self, bus, payload, ts=None):
        t0 = time.perf_counter(); recs = self._r.decode(bus, payload, ts); self.mon.record(bus, payload, ts, recs, time.perf_counter() - t0); return recs


class Runtime:
    def _profile_changed(self):
        """The database was provisioned from an older version of this vehicle's profile (a decoder file that learned more since): provisioning is idempotent and adds what is new."""
        r = self.conn.execute("SELECT v FROM meta WHERE k='vehicle.profile'").fetchone()
        try: return r is not None and json.loads(r[0]) != json.loads(json.dumps(self.p.spec))
        except (TypeError, ValueError): return False

    def __init__(self, db, profile=None, monitor=True, ledger_every=LEDGER_EVERY_S, tier=None):
        self.db = db; self.lock = threading.RLock(); self.dlock = threading.RLock(); self.conn = store.open_rw(db, threads=True)
        if profile is None:
            row = self.conn.execute("SELECT v FROM meta WHERE k='vehicle.id'").fetchone()
            if row is None: raise ValueError(f"{db} has no vehicle profile: provision it first (python -m vgpt.vehicle.provision) or pass profile=")
            profile = row[0]
        self.p = profile if isinstance(profile, Profile) else Profile.load(profile)
        if not self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='sig_catalog'").fetchone() or self.conn.execute("SELECT v FROM meta WHERE k='vehicle.id'").fetchone() is None or self._profile_changed(): LK.provision(self.conn, self.p)
        self.tier = None
        if tier or os.environ.get("VGPT_TIER"):                                                                                  # a bigger box keeps more history from now on (never less than the vehicle's profile asks)
            from .. import tiers
            self.tier = tiers.get(tier)
            for k, v in tiers.retention_for(self.p.retention, self.tier.name).items(): self.conn.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", ("ret." + k, str(v)))
        self.router = ProfileRouter(self.p); self.mon = BusMonitor(self.router) if monitor else None
        self.ing = LiveIngestor(self.conn, Tap(self.router, self.mon) if monitor else self.router, lake=LK.Lake(self.conn), track_meta=True, xring_s=self.tier.ring_s if self.tier else 420)
        self.msgs = 0; self.by_bus = collections.Counter(); self.t_start = time.time(); self.ts = 0.0; self.first_ts = None; self.ledger_every = ledger_every; self._ledger_at = 0.0; self._day = None
        self.expected = self._expected_periods(); self.silent = {}; self._episode = None; self._health_at = 0.0; self.errors = collections.Counter(); self.new_trips = self.new_charges = 0
        self._rate_t = time.time(); self._rate_n = 0; self.wall_rate = 0.0

    # ---------------------------------------------------------------------------------------------------------- ingest
    def decode(self, ts, bus, payload):
        """The collector's half of `feed`: raw traffic -> canonical records.  Touches decoder state only, never the database."""
        with self.dlock:
            try: return self.ing.decode(bus, payload, ts)
            except Exception as e:                                                                              # a corrupt message must never stop the service
                self.errors[type(e).__name__] += 1; return []

    def feed(self, ts, bus, payload):
        self.store(ts, bus, self.decode(ts, bus, payload))

    def store(self, ts, bus, recs):
        """The store's half of `feed`: canonical records -> the database (the single writer).  `bus` is only counted."""
        with self.lock:
            self.msgs += 1; self.by_bus[bus] += 1
            if self.first_ts is None: self.first_ts = ts
            try: self.ing.feed_recs(ts, recs)
            except Exception as e:                                                                              # a corrupt message must never stop the service
                self.errors[type(e).__name__] += 1
            if ts > self.ts:
                self.ts = ts
                if ts - self._health_at >= 10.0:
                    self._health_at = ts
                    if self._ledger_at: self._source_health(ts)
                if ts - self._ledger_at >= self.ledger_every or self._ledger_at == 0.0: self._housekeeping(ts)

    def feed_many(self, items):
        for ts, bus, payload in items: self.feed(ts, bus, payload)

    def _housekeeping(self, ts):
        first = self._ledger_at == 0.0; self._ledger_at = ts
        if first: return
        try:
            nt, nc = ledger.update(self.conn, lookback_s=3600, now=int(ts)); self.new_trips += nt; self.new_charges += nc
        except Exception as e: self.errors["ledger:" + type(e).__name__] += 1
        tz = self._tz(); day = int((ts + tz) // 86400)
        if self._day is None: self._day = day
        elif day != self._day:
            self._day = day
            try: store.compact(self.conn, int(ts))
            except Exception as e: self.errors["compact:" + type(e).__name__] += 1

    def _tz(self):
        try: return int(float(self.conn.execute("SELECT v FROM meta WHERE k='env.tz_h'").fetchone()[0]) * 3600)
        except (TypeError, ValueError): return 0

    # ---------------------------------------------------------------------------------------------------------- source health
    def _expected_periods(self):
        top = {}
        for b in self.p.bindings: top[b.source] = max(top.get(b.source, 0.0), b.rate_hz)
        out = {}
        for sid, src in self.p.sources.items():
            if sid in top: out[sid] = max(2.0, SILENT_FACTOR / max(top[sid], 0.05))
            elif src.type == "gnss": out[sid] = 6.0
            elif src.type == "camera": out[sid] = 10.0
        return out

    def _awake(self, ts):
        for k in ("_ignition", "_charging_active"):
            it = self.ing.latest.get(k)
            if it is not None and it[0] >= 0.5 and ts - it[1] <= 10: return True
        return False

    def _source_health(self, ts):
        if not self._awake(ts): return
        seen = self.router.last_seen
        for sid, period in self.expected.items():
            key = "gnss" if self.p.sources[sid].type == "gnss" else ("vhal" if self.p.sources[sid].type == "vhal" else ("camera" if self.p.sources[sid].type == "camera" else sid))
            last = seen.get(key)
            if last is None: continue
            age = ts - last; was = self.silent.get(sid)
            if age > period and was is None:
                detail = f"{self.p.sources[sid].name}: no frames for {age:.0f} s while the vehicle is awake"; self.silent[sid] = int(last)
                ep = self._episode
                if ep is None or int(last) - ep["t"] > 30: ep = self._episode = {"t": int(last) + 1, "sids": set(), "fired": False, "ign": (self.ing.latest.get("_ignition") or (None,))[0]}      # sources that fall silent together are one episode
                ep["sids"].add(sid)
                self.conn.execute("INSERT INTO source_log VALUES(?,?,?,?)", (int(last) + 1, sid, "silent", detail)); self.ing._event(int(last) + 1, "SOURCE_SILENT", "warn", detail)             # also an event: the assistant answers about it like any other
            elif age <= period and was is not None:
                ep = self._episode
                if ep is not None and not ep["fired"] and sid in ep["sids"] and len(ep["sids"]) >= max(2, (len(self.expected) + 1) // 2) and (ep["ign"] is None or ep["ign"] >= 0.5):
                    ep["fired"] = True                                                              # everything went quiet at once while the car was on, no switch-off was seen, and traffic came back: the supply was lost, not the key turned
                    self.ing._event(ep["t"], "UNEXPECTED_SHUTDOWN", "critical", f"the vehicle stopped sending on {len(ep['sids'])} sources at once for about {ts - ep['t']:.0f} s while it was on, with no ignition-off; traffic then resumed (power loss or reboot)")
                del self.silent[sid]; self.conn.execute("INSERT INTO source_log VALUES(?,?,?,?)", (int(ts), sid, "restored", f"{self.p.sources[sid].name}: traffic is back after {ts - was:.0f} s"))

    # ---------------------------------------------------------------------------------------------------------- views
    def signals(self):
        out = []; ts = self.ts
        with self.dlock, self.lock:
            for c, (v, src_ts) in sorted(self.ing.latest.items()):
                cd = R.CONCEPTS.get(c); slot = self.router.slots.get(c) or {}; best = min(slot) if slot else None; b = slot[best][2] if best is not None else None
                out.append({"concept": c, "label": cd.label if cd else c, "unit": cd.unit if cd else "", "category": getattr(cd, "category", ""), "value": v, "age_s": round(ts - src_ts, 1) if ts else None,
                            "source": b.source if b is not None else None, "key": b.key if b is not None else None})
        return out

    def sources(self):
        rep = self.router.health_report(self.ts); out = []
        for sid, src in self.p.sources.items():
            key = {"gnss": "gnss", "vhal": "vhal", "camera": "camera", "uds": "diag", "log": "log"}.get(src.type, sid)
            r = rep.get(key) or rep.get(sid) or {}
            state = "silent" if sid in self.silent else ("ok" if r.get("frames") else "waiting")
            out.append({"id": sid, "type": src.type, "name": src.name, "frames": r.get("frames", 0), "decoded": r.get("decoded", 0), "unknown": r.get("unknown", 0), "rejected": r.get("rejected", 0), "malformed": r.get("malformed", 0),
                        "age_s": r.get("age_s"), "state": state, "extra": {k: v for k, v in (src.raw.items() if hasattr(src, "raw") else []) if k in ("channel", "bitrate", "data_bitrate", "fd")}})
        return out

    def stats(self):
        now = time.time(); dt = now - self._rate_t
        if dt >= 2.0: self.wall_rate = (self.msgs - self._rate_n) / dt; self._rate_t, self._rate_n = now, self.msgs
        try: size = sum(os.path.getsize(self.db + e) for e in ("", "-wal") if os.path.exists(self.db + e)) / 2 ** 20
        except OSError: size = 0.0
        ing = self.ing
        return {"t": self.ts, "msgs": self.msgs, "msgs_per_s_wall": round(self.wall_rate), "by_bus": dict(self.by_bus), "decode_us": round(1e6 * self.mon.dec_s / max(self.mon.n, 1), 1) if self.mon else None,
                "rows_1hz": ing.rows, "lake_rows": ing.lake.rows if ing.lake else 0, "dropped_late": ing.dropped_late, "dropped_invalid": ing.dropped_invalid, "router_dropped": self.router.dropped, "errors": dict(self.errors),
                "new_trips": self.new_trips, "new_charges": self.new_charges, "db_mb": round(size, 1), "uptime_s": round(now - self.t_start, 1), "pending_context": len(ing.xpend) + len(ing.pending_events)}

    def snapshot(self, bus_rows=True):
        with self.dlock, self.lock:                                                                                  # the collector thread decodes while this reads: both halves are held
            snap = {"stats": self.stats(), "sources": self.sources(), "signals": self.signals(), "vehicle": {"id": self.p.id, "model": self.p.specs.get("veh.model", self.p.id), "powertrain": self.p.spec["vehicle"].get("powertrain", "")}}
            if bus_rows and self.mon: snap["bus"] = self.mon.table(); snap["tail"] = self.mon.latest(40); snap["feed"] = list(self.mon.feed)[-30:]
        return snap

    def publish(self):
        """Make the partial current minute and pending events visible to readers without closing the second."""
        with self.lock:
            self.ing.snapshot(); s = self.ing.sec
            if s is not None:                                                                                         # the open minute is kept in the minute tier too (replaced as it fills), so 'the last hour' never lacks the present minute
                try: store.rollup(self.ing.c, s // 60 * 60, s // 60 * 60 + 60)
                except Exception: pass

    def close(self):
        with self.lock:
            try: self.ing.close()
            finally:
                try: self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception: pass
                self.conn.close()

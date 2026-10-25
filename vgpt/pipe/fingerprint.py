"""Which vehicle is this?  Match what a link is actually sending against the decoder files that are installed.

A decoder file (a profile pack: profile.json + its CAN databases + SOME/IP catalogue ...) says which identifiers its vehicle uses.  `Inventory` counts which identifiers the link really carries - CAN
identifiers per channel, SOME/IP services, VHAL property ids.  `DecoderLibrary.identify()` scores every installed decoder file against that inventory and names the best one, with the reasons.  Nothing
here knows a vehicle's name: a decoder file dropped into the folder joins the library the next time it is scanned.

The score is the share of the traffic the decoder can interpret (`cover`), the share it will turn into stored readings (`bound`) and how tight the fit is (`spec`: a database that defines 140 messages
of which the car sends 20 is a looser fit than one that defines 25).  Diagnostic traffic (OBD-II / UDS on the standard identifiers) belongs to every vehicle and is left out of the comparison.

`channel_map` answers a second question: with several CAN buses on several wires, which wire is which database's bus?  It is decided by the same identifiers, and is what lets two buses that reuse the same
identifier for different things be told apart."""
import collections
import json
import math
import os

from ..ingest import standards as _std
from ..vehicle.profile import Profile

ACCEPT = 0.55
DIAG_STD = range(0x7E0, 0x7F0)
DIAG_FUNCTIONAL = 0x7DF
SD_SERVICE = 0xFFFF


def is_diag(ident, ext):
    """OBD-II / UDS traffic on the standard diagnostic identifiers (11-bit 0x7DF, 0x7E0..0x7EF; 29-bit 0x18DA.... and 0x18DB....): every vehicle has it, so it says nothing about which vehicle this is."""
    if not ext: return ident == DIAG_FUNCTIONAL or ident in DIAG_STD
    return (ident >> 24) == 0x18 and ((ident >> 16) & 0xFF) in (0xDA, 0xDB)


class Inventory:
    """What a link carries so far (counts only; the blocks themselves are not kept here)."""
    def __init__(self):
        self.can = collections.defaultdict(collections.Counter)             # wire channel -> Counter{(ident, ext): frames}
        self.dlc = collections.defaultdict(dict)                            # (ident, ext) -> {dlc: count}
        self.flexray = collections.Counter(); self.doip = collections.Counter(); self.lin = collections.Counter(); self.someip = collections.Counter(); self.vhal = collections.Counter(); self.vhal_area = collections.Counter(); self.frames = collections.Counter()
        self.diag = 0; self.log_sources = collections.Counter(); self.nmea = 0; self.images = 0; self.table_keys = collections.Counter(); self.vin = None; self.unknown = collections.Counter()
        self.first = None; self.last = None

    def seen(self, ts):
        if ts is None: return
        if self.first is None: self.first = ts
        self.last = ts

    def add_can(self, channel, frame):
        key = (frame.can_id & 0x1FFFFFFF, bool(frame.extended))
        self.frames["can"] += 1
        if is_diag(*key): self.diag += 1; return
        self.can[channel][key] += 1; d = self.dlc[key]; d[len(frame.data)] = d.get(len(frame.data), 0) + 1

    def add_flexray(self, slot): self.frames["flexray"] += 1; self.flexray[slot] += 1
    def add_doip(self, address): self.frames["doip"] += 1; self.doip[address] += 1
    def add_lin(self, ident): self.frames["lin"] += 1; self.lin[ident] += 1
    def add_someip(self, service): self.frames["someip"] += 1; self.someip[service] += 1
    def add_vhal(self, prop, area=0): self.frames["vhal"] += 1; self.vhal[prop] += 1; self.vhal_area[(prop, area)] += 1

    def total_can(self): return sum(sum(c.values()) for c in self.can.values())

    def all_can(self):
        t = collections.Counter()
        for c in self.can.values(): t.update(c)
        return t

    def summary(self):
        return {"can_ids": len(self.all_can()), "can_frames": self.total_can(), "diag_frames": self.diag, "someip_services": len([s for s in self.someip if s != SD_SERVICE]), "vhal_props": len(self.vhal), "vhal_areas": len(self.vhal_area), "log_sources": len(self.log_sources),
                "nmea": self.nmea, "images": self.images, "table_keys": len(self.table_keys), "channels_with_can": len(self.can)}


class Entry:
    """One installed decoder file, reduced to the identifiers it defines."""
    def __init__(self, path):
        self.path = path; self.profile = Profile.load(path); p = self.profile; self.id = p.id; self.model = p.specs.get("veh.model", p.id); self.powertrain = p.spec.get("vehicle", {}).get("powertrain", "")
        self.mtime = _mtime(path); self.sources = {}                         # source id -> {"defined": {(ident, ext): dlc}, "bound": set, "j1939": bool, "channel": str}
        bound_keys = collections.defaultdict(set)
        for b in p.bindings:
            if b.bus == "can": bound_keys[b.source].add(b.key.partition(".")[0])
        for sid, src in p.sources.items():
            if src.type != "can": continue
            db = p.dbc(sid); j = bool(src.raw.get("j1939")); defined = {}; bound = set()
            for d in (db.messages, db.extended_messages):
                for m in d.values():
                    key = (_std.canonical_id(m.frame_id) if j else m.frame_id, True if j else m.extended); defined[key] = m.dlc
                    if m.name in bound_keys[sid]: bound.add(key)
            self.sources[sid] = {"defined": defined, "bound": bound, "j1939": j, "channel": src.raw.get("channel", "")}
        self.doip = {int(str(s.raw.get("entity", "0x1001")), 16) for s in p.sources.values() if s.type == "uds" and s.raw.get("transport") == "doip"}
        self.flexray = set()
        for sid, src in p.sources.items():
            if src.type == "flexray" and "description" in src.raw:
                try: self.flexray |= set(p.flexray(sid).by_slot)
                except (OSError, ValueError, KeyError): pass
        self.lin = set()
        for sid, src in p.sources.items():
            if src.type == "lin" and "ldf" in src.raw:
                try: self.lin |= set(p.ldf(sid).by_id)
                except (OSError, ValueError): pass
        self.someip = set(); self.vhal = set(); self.has = {t: any(s.type == t for s in p.sources.values()) for t in ("uds", "gnss", "log", "camera", "someip", "vhal")}
        for sid, src in p.sources.items():
            if src.type == "someip" and "catalog" in src.raw:
                try: self.someip |= {int(k, 16) if isinstance(k, str) else int(k) for k in p.catalog(sid).spec}
                except (OSError, ValueError): pass
        from ..protocols import vhal as _vh
        ids = _vh.Vhal().props
        for b in p.bindings:
            if b.bus == "vhal" and b.key.split("@")[0] in ids: self.vhal.add(ids[b.key.split("@")[0]]["id"])
        self.plain_defined = set(); self.plain_bound = set(); self.j_defined = set(); self.j_bound = set()
        for sdef in self.sources.values():
            if sdef["j1939"]: self.j_defined |= set(sdef["defined"]); self.j_bound |= sdef["bound"]
            else: self.plain_defined |= set(sdef["defined"]); self.plain_bound |= sdef["bound"]
        self.n_can_bound = len(self.plain_bound) + len(self.j_bound)

    def defines(self, k):
        ident, ext = k
        return k in self.plain_defined or (ext and (_std.canonical_id(ident), True) in self.j_defined)

    def binds(self, k):
        ident, ext = k
        return k in self.plain_bound or (ext and (_std.canonical_id(ident), True) in self.j_bound)


def _mtime(path):
    p = os.path.join(path, "profile.json") if os.path.isdir(path) else path
    try: return os.path.getmtime(p)
    except OSError: return 0.0


class Match:
    def __init__(self, entry, score, parts, channel_map, why):
        self.entry, self.score, self.parts, self.channel_map, self.why = entry, score, parts, channel_map, why
    def to_json(self): return {"id": self.entry.id, "model": self.entry.model, "score": round(self.score, 3), "parts": {k: round(v, 3) for k, v in self.parts.items()}, "channel_map": self.channel_map, "why": self.why}


def score(entry, inv):
    """-> Match.  See the module text for what the parts mean."""
    parts = {}; weights = {}; why = []; cmap = {}
    allcan = inv.all_can(); total = sum(allcan.values())
    if total and entry.sources:
        cov = sum(n for k, n in allcan.items() if entry.defines(k)); bnd = sum(n for k, n in allcan.items() if entry.binds(k))
        seen_defined = len([k for k in allcan if entry.defines(k)]); seen_ids = len(allcan); n_defined = len(entry.plain_defined) + len(entry.j_defined)
        parts["cover"] = cov / total; parts["bound"] = bnd / total; parts["spec"] = min(1.0, seen_defined / max(n_defined, 1))
        # a database that explains every identifier seen is good; one that explains them with few unused definitions is better; the log weight keeps a large link from drowning the other evidence
        weights["can"] = (math.log1p(total), 0.5 * parts["cover"] + 0.3 * parts["bound"] + 0.2 * parts["spec"])
        why.append(f"{seen_defined} of the {seen_ids} CAN identifiers on the link are defined by its CAN databases ({parts['cover'] * 100:.0f} % of the frames, {parts['bound'] * 100:.0f} % of them become stored readings)")
        cmap = channel_map(entry, inv)
    seen_frames = sum(inv.frames.values())
    for kind, have, expected in (("doip", inv.doip, entry.doip), ("lin", inv.lin, entry.lin), ("flexray", inv.flexray, entry.flexray)):                  # a decoder file that expects a bus the link never carries fits less well than one that does not (a car without that bus is not this car)
        if expected and not have and seen_frames >= 40:
            parts[kind + "_absent"] = 0.0; weights[kind + "_absent"] = (math.log1p(min(seen_frames, 400) / 20.0), 0.0); why.append(f"its {kind.upper()} bus never appeared on the link")
    if inv.doip:
        n = sum(inv.doip.values()); hit = sum(c for a, c in inv.doip.items() if a in entry.doip)
        parts["doip"] = hit / n; weights["doip"] = (math.log1p(n), 0.8 * parts["doip"] + 0.2 * (1.0 if entry.doip else 0.0)); why.append(f"{len([a for a in inv.doip if a in entry.doip])} of {len(inv.doip)} DoIP logical addresses are its diagnostic entities")
    if inv.flexray:
        n = sum(inv.flexray.values()); hit = sum(c for i, c in inv.flexray.items() if i in entry.flexray)
        parts["flexray"] = hit / n; sp = len([i for i in inv.flexray if i in entry.flexray]) / max(len(entry.flexray), 1)
        weights["flexray"] = (math.log1p(n), 0.7 * parts["flexray"] + 0.3 * min(1.0, sp)); why.append(f"{len([i for i in inv.flexray if i in entry.flexray])} of {len(inv.flexray)} FlexRay slots are in its cluster description")
    if inv.lin:
        n = sum(inv.lin.values()); hit = sum(c for i, c in inv.lin.items() if i in entry.lin)
        parts["lin"] = hit / n; sp = len([i for i in inv.lin if i in entry.lin]) / max(len(entry.lin), 1)
        weights["lin"] = (math.log1p(n), 0.7 * parts["lin"] + 0.3 * min(1.0, sp)); why.append(f"{len([i for i in inv.lin if i in entry.lin])} of {len(inv.lin)} LIN frame identifiers are in its LIN description")
    services = {s: n for s, n in inv.someip.items() if s != SD_SERVICE}
    if services:
        n = sum(services.values()); hit = sum(c for s, c in services.items() if s in entry.someip)
        parts["someip"] = hit / n; sp = len([s for s in services if s in entry.someip]) / max(len(entry.someip), 1)
        weights["someip"] = (math.log1p(n), 0.7 * parts["someip"] + 0.3 * min(1.0, sp)); why.append(f"{len([s for s in services if s in entry.someip])} of {len(services)} SOME/IP services are in its service catalogue")
    if inv.vhal:
        n = sum(inv.vhal.values()); hit = sum(c for p, c in inv.vhal.items() if p in entry.vhal)
        parts["vhal"] = hit / n; sp = len([p for p in inv.vhal if p in entry.vhal]) / max(len(entry.vhal), 1)
        weights["vhal"] = (math.log1p(n), 0.7 * parts["vhal"] + 0.3 * min(1.0, sp)); why.append(f"{len([p for p in inv.vhal if p in entry.vhal])} of {len(inv.vhal)} vehicle properties are ones it binds")
    if not weights: return Match(entry, 0.0, parts, cmap, ["the link carries nothing this decoder file defines"])
    wsum = sum(w for w, _ in weights.values()); s = sum(w * v for w, v in weights.values()) / wsum
    return Match(entry, s, parts, cmap, why)


def channel_map(entry, inv):
    """{wire channel: source id}: for every wire carrying CAN, the CAN database whose identifiers explain it best (a wire is only mapped when one database clearly wins)."""
    out = {}
    for ch, ids in inv.can.items():
        tot = sum(ids.values()); best = None; best_s = 0.0; second = 0.0
        for sid, s in entry.sources.items():
            c = 0
            for (ident, ext), n in ids.items():
                k = (_std.canonical_id(ident), True) if s["j1939"] and ext else (ident, ext)
                if k in s["defined"]: c += n
            sc = c / tot
            if sc > best_s: second, best, best_s = best_s, sid, sc
            elif sc > second: second = sc
        if best is not None and best_s >= 0.6 and best_s - second >= 0.15: out[ch] = best
    return out


class DecoderLibrary:
    """The decoder files that are installed.  `roots` are folders each holding one folder per decoder file; they are rescanned when something changes."""
    def __init__(self, roots):
        self.roots = [r for r in roots if r]; self.entries = {}; self.errors = {}; self._stamp = None; self.refresh()

    def _scan(self):
        found = {}
        for root in self.roots:
            if not os.path.isdir(root): continue
            for name in sorted(os.listdir(root)):
                d = os.path.join(root, name)
                if name.startswith(("_", ".")) or not os.path.isfile(os.path.join(d, "profile.json")): continue
                # The car simulator keeps a profile for every vehicle so it can
                # generate authentic traffic.  Some of those profiles are
                # deliberately *not* installed decoders: standards traffic must
                # be learned by Vantage.  Honour the adjacent catalogue flag so
                # sharing a fixture root cannot leak the answer to identification.
                meta = os.path.join(root, name + ".json")
                if not os.path.isfile(meta):
                    # The legacy bundled decoder root is vehicle/packs, while
                    # the simulator's catalogue (which declares whether the
                    # decoder is actually installed) lives at car/models.
                    # Consult that declaration too; otherwise a standards-only
                    # AAOS/OBD/J1939 fixture leaks its identity through packs.
                    here = os.path.dirname(os.path.abspath(__file__))
                    meta = os.path.join(os.path.dirname(here), "car", "models", name + ".json")
                if os.path.isfile(meta):
                    try:
                        if json.load(open(meta, encoding="utf-8")).get("decoder_installed") is False: continue
                    except (OSError, ValueError):
                        pass
                found.setdefault(name, d)                                    # an earlier root wins: what the person installed beats what shipped
        return found

    def stamp(self):
        return tuple(sorted((n, _mtime(p), tuple(sorted((f, os.path.getmtime(os.path.join(p, f))) for f in os.listdir(p) if f.endswith((".dbc", ".json"))))) for n, p in self._scan().items()))

    def refresh(self, force=False):
        st = self.stamp()
        if st == self._stamp and not force: return False
        self._stamp = st; found = self._scan(); entries = {}; errors = {}
        for name, path in found.items():
            old = self.entries.get(name)
            if old is not None and old.path == path and old.mtime == _mtime(path) and not force:
                entries[name] = old; continue
            try: entries[name] = Entry(path)
            except Exception as e: errors[name] = f"{type(e).__name__}: {e}"
        self.entries, self.errors = entries, errors; return True

    def rank(self, inv, top=3):
        ms = sorted((score(e, inv) for e in self.entries.values()), key=lambda m: (-m.score, m.entry.n_can_bound))
        return ms[:top]

    def identify(self, inv, accept=ACCEPT):
        """-> (best Match or None, [the runners-up], ambiguous: bool)."""
        r = self.rank(inv, 3)
        if not r or r[0].score < accept: return None, r, False
        amb = len(r) > 1 and r[0].score - r[1].score < 0.03 and r[1].score >= accept
        return r[0], r[1:], amb

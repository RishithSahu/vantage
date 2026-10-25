"""Who is using the head unit, and what each role may see.

The case study asks for value for drivers, service engineers, fleet operators and validation teams, and for privacy and security: local-only processing and protection of vehicle and driver data.
A role here is therefore ACCESS CONTROL, not a presentation style:

    driver       the person in the car; no sign-in.  Sees their own location, places and camera pictures, and may erase them.
    engineer     service engineer; signs in.  Diagnostics, evidence, a service report, the signals - but not where the car has been and not the camera pictures.
    fleet        fleet operator; signs in.  Utilisation, efficiency, risk and alerts across vehicles - no driver, no places, no pictures.
    validation   validation team; signs in.  Signal coverage and quality, protocol sources, the injected-fault scorecard, signed evidence packs - de-identified.

Everything personal (location, saved places, trip routes, camera pictures) is a permission only the driver role holds; every other role sees the same vehicle with that removed (`vgpt/privacy.py`).
Sign-in is a local passcode (scrypt-hashed, locked out after repeated failures), a session is an unguessable token in a cookie that never leaves this computer and expires when idle, and every sign-in, refusal,
export and erase is written to an append-only, hash-chained audit log that records WHAT happened, never the question or the answer.  Files live in the private VehicleGPT folder (`vgpt/paths.py`)."""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

from . import paths

ROLES = (
    {"id": "driver", "title": "Driver", "auth": False, "short": "The person in the car",
     "value": ["Plain, short answers about your own car", "Your location, saved places and camera pictures are yours alone", "You can erase your location history at any time"],
     "perms": ("location", "camera", "places", "erase", "privacy")},
    {"id": "engineer", "title": "Service engineer", "auth": True, "short": "Workshop and diagnostics",
     "value": ["Every diagnostic code with its history, state and what to check", "A service report: component health, warnings, evidence and suggested checks", "Answers show their working: samples, sources, quality"],
     "perms": ("diagnostics", "evidence", "service", "signals", "export", "privacy")},
    {"id": "fleet", "title": "Fleet operator", "auth": True, "short": "Many vehicles, one view",
     "value": ["Utilisation, efficiency, risk events and open alerts for every vehicle", "Which vehicle needs attention, and why", "No driver identity, no places, no camera pictures"],
     "perms": ("fleet", "aggregates", "privacy")},
    {"id": "validation", "title": "Validation team", "auth": True, "short": "Is the data and the assistant right?",
     "value": ["Signal coverage, rates, gaps and quality flags per signal and per bus", "Injected faults scored against what the vehicle recorded", "Signed (and optionally encrypted) evidence packs; the audit trail"],
     "perms": ("validation", "evidence", "signals", "coverage", "export", "audit", "privacy")},
)
BY_ID = {r["id"]: r for r in ROLES}
PERSONAL = ("location", "camera", "places")                                          # the data classes that belong to the driver


def role(rid): return BY_ID.get(rid) or BY_ID["driver"]
def can(rid, perm): return perm in role(rid)["perms"]
def persona_of(rid): return "engineer" if rid in ("engineer", "validation") else "driver"                # the answer style the assistant already has: plain, or with its working shown


WEAK = {"123456", "1234567", "12345678", "654321", "password", "qwerty", "abcdef", "111111", "000000"}
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)
MAX_TRIES, BASE_LOCK, MAX_LOCK = 5, 30, 900


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: f.write(text)
    if os.name != "nt":
        try: os.chmod(tmp, 0o600)
        except OSError: pass
    os.replace(tmp, path)


class Auth:
    """Local passcodes for the three signed-in roles.  The passcode is never stored: only a salted scrypt hash, in a file only this user can read."""
    def __init__(self, path=None, clock=time.time):
        self.path = path or os.path.join(paths.home(), "auth.json"); self.clock = clock; self.lock = threading.Lock()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f: d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError): return {}

    def configured(self, rid): return bool(self._load().get(rid, {}).get("hash"))

    @staticmethod
    def _hash(pin, salt): return hashlib.scrypt(pin.encode("utf-8"), salt=salt, **SCRYPT).hex()

    @staticmethod
    def policy(pin):
        pin = str(pin or "")
        if len(pin) < 6: return "use at least 6 characters"
        if len(pin) > 64: return "use at most 64 characters"
        if len(set(pin)) == 1 or pin.lower() in WEAK: return "that passcode is too easy to guess"
        return None

    def set_pin(self, rid, pin, current=None):
        """-> (ok, message).  The first passcode of a role is set at commissioning; changing it needs the current one."""
        if not role(rid)["auth"] or rid not in BY_ID: return False, "this role has no passcode"
        why = self.policy(pin)
        if why: return False, why
        with self.lock:
            d = self._load()
            if d.get(rid, {}).get("hash"):
                ok, msg, _ = self._check(d, rid, current)
                if not ok: self._save(d); return False, "the current passcode is wrong" if msg == "wrong" else msg
            salt = secrets.token_bytes(16); d[rid] = {"salt": salt.hex(), "hash": self._hash(pin, salt), "set_at": int(self.clock()), "failed": 0, "locked_until": 0}; self._save(d)
        return True, "passcode set"

    def _save(self, d): _atomic_write(self.path, json.dumps(d))

    def _check(self, d, rid, pin):
        e = d.get(rid) or {}; now = self.clock()
        if not e.get("hash"): return False, "no passcode is set for this role yet", 0
        if e.get("locked_until", 0) > now: return False, "locked", int(e["locked_until"] - now) + 1
        ok = hmac.compare_digest(self._hash(str(pin or ""), bytes.fromhex(e["salt"])), e["hash"])
        if ok: e["failed"] = 0; e["locked_until"] = 0; return True, "ok", 0
        e["failed"] = int(e.get("failed", 0)) + 1
        if e["failed"] >= MAX_TRIES:
            wait = min(MAX_LOCK, BASE_LOCK * 2 ** (e["failed"] - MAX_TRIES)); e["locked_until"] = now + wait; return False, "locked", int(wait)
        return False, "wrong", 0

    def verify(self, rid, pin):
        """-> (ok, message, seconds_to_wait).  Five wrong passcodes in a row lock the role for 30 s, then twice as long each time (at most 15 minutes)."""
        if rid not in BY_ID or not BY_ID[rid]["auth"]: return False, "this role has no passcode", 0
        with self.lock:
            d = self._load(); ok, msg, wait = self._check(d, rid, pin); self._save(d) if d.get(rid) else None
        return ok, msg, wait


class Sessions:
    """A signed-in role is an unguessable token held here and in a cookie of this browser; it expires after ten idle minutes and after eight hours whatever happens."""
    IDLE, ABSOLUTE = 600, 8 * 3600

    def __init__(self, clock=time.time): self.clock = clock; self.d = {}; self.lock = threading.Lock()

    def open(self, rid):
        tok = secrets.token_urlsafe(32); now = self.clock()
        with self.lock: self.d[tok] = {"role": rid, "t0": now, "last": now}
        return tok

    def get(self, tok):
        if not tok: return None
        with self.lock:
            s = self.d.get(tok)
            if s is None: return None
            now = self.clock()
            if now - s["last"] > self.IDLE or now - s["t0"] > self.ABSOLUTE: del self.d[tok]; return None
            s["last"] = now; return s["role"]

    def close(self, tok):
        with self.lock: self.d.pop(tok, None)

    def close_all(self):
        with self.lock: n = len(self.d); self.d.clear(); return n

    def close_role(self, rid):
        with self.lock:
            for k in [k for k, v in self.d.items() if v["role"] == rid]: del self.d[k]


class Audit:
    """An append-only log of what happened (never the question or the answer), each line chained to the one before by a hash: changing or deleting an earlier line breaks every later one."""
    MAX_BYTES = 4_000_000

    def __init__(self, path=None, clock=time.time):
        self.path = path or os.path.join(paths.home(), "audit.jsonl"); self.clock = clock; self.lock = threading.Lock()

    @staticmethod
    def _h(prev, body): return hashlib.sha256((prev + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode("utf-8")).hexdigest()

    def _last(self):
        try:
            with open(self.path, "rb") as f:
                f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4096)); lines = f.read().decode("utf-8", "replace").strip().splitlines()
            return json.loads(lines[-1]) if lines else None
        except (OSError, ValueError, IndexError): return None

    def add(self, who, action, **detail):
        with self.lock:
            last = self._last(); prev = last["h"] if last else "0" * 64; i = (last["i"] + 1) if last else 1
            body = {"i": i, "t": int(self.clock()), "who": str(who)[:20], "action": str(action)[:40], "detail": {str(k)[:24]: str(v)[:120] for k, v in detail.items()}, "prev": prev}
            rec = dict(body, h=self._h(prev, body))
            try:
                if os.path.exists(self.path) and os.path.getsize(self.path) > self.MAX_BYTES: os.replace(self.path, self.path + ".1")
            except OSError: pass
            with open(self.path, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            if os.name != "nt":
                try: os.chmod(self.path, 0o600)
                except OSError: pass
        return rec

    def records(self):
        try:
            with open(self.path, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
        except (OSError, ValueError): return []

    def verify(self):
        """-> {"ok", "count", "first_bad"}: the chain is intact when every line's hash matches its content and its predecessor."""
        recs = self.records(); prev = None
        for k, r in enumerate(recs):
            body = {x: r[x] for x in r if x != "h"}
            if prev is not None and r.get("prev") != prev: return {"ok": False, "count": len(recs), "first_bad": r.get("i", k + 1)}
            if r.get("h") != self._h(r.get("prev", ""), body): return {"ok": False, "count": len(recs), "first_bad": r.get("i", k + 1)}
            prev = r["h"]
        return {"ok": True, "count": len(recs), "first_bad": None}

    def tail(self, n=40): return self.records()[-n:][::-1]

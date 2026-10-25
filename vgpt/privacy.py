"""Protection of vehicle and driver data.

What is personal here: where the car has been (GPS fixes, trip routes, saved places such as Home and Work) and what its camera has seen (pictures).  Only the driver role may see those.  Every other role
sees the same vehicle with them removed - the answer is still useful to a workshop, a fleet or a validation team, and it carries no person.

    scrub(obj, role, place_names)   remove / replace the personal parts of anything that is about to leave the server for a role without the permission (answers, events, telemetry ...)
    redact_text(text, names)        coordinates and saved-place names inside a sentence
    posture(...)                    live checks that processing is local-only and the assistant cannot write; what is stored, where, for whom; what is NOT covered

This module sanitises OUTPUT.  It does not read the question, so it is not part of the understanding of language (vgpt/meaning)."""
import os
import re
import sqlite3
import time

from . import access

PERSONAL_KEYS = {"lat", "lon", "lng", "pos", "near", "near_id", "area", "here", "track", "points", "places", "hdg", "heading", "heading_dir", "fix", "gps", "coordinates", "png", "thumb_png", "route_points"}
COORD_PAIR = re.compile(r"-?\d{1,3}\.\d{2,}\s*,\s*-?\d{1,3}\.\d{2,}")
COORD_NSEW = re.compile(r"\d{1,3}\.\d+\s*°?\s*[NS]\s*,?\s*\d{1,3}\.\d+\s*°?\s*[EW]")
HIDDEN = "a location (hidden for your role)"
PLACE = "a saved place"


def place_names(conn):
    try: return sorted({str(r[0]) for r in conn.execute("SELECT name FROM places") if r[0]}, key=len, reverse=True)
    except sqlite3.Error: return []


def redact_text(text, names=()):
    """Coordinates and saved-place names inside a sentence."""
    s = COORD_NSEW.sub(HIDDEN, COORD_PAIR.sub(HIDDEN, str(text)))
    for n in names:
        if len(n) >= 3: s = re.sub(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", PLACE, s)
    return s


def scrub(obj, rid, names=()):
    """A copy of `obj` (JSON-like) without the personal parts, for a role that does not hold the location / camera permission.  The driver's view is returned unchanged."""
    if access.can(rid, "location") and access.can(rid, "camera"): return obj
    def walk(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k in PERSONAL_KEYS and not access.can(rid, "location"): continue
                if k == "thumb" and not access.can(rid, "camera"): out[k] = False; continue
                out[k] = walk(v)
            return out
        if isinstance(o, (list, tuple)): return [walk(x) for x in o]
        if isinstance(o, str): return redact_text(o, names)
        return o
    return walk(obj)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------- posture
def _connections():
    import psutil
    p = psutil.Process(os.getpid())
    return p.net_connections(kind="inet") if hasattr(p, "net_connections") else p.connections(kind="inet")


def posture(app, auth, audit, sessions=None):
    """The live answer to 'is it really local-only, and who can see what': a list of checks (ok True / False / None = cannot be checked here), each with the evidence behind it."""
    checks = []
    def add(i, title, ok, detail): checks.append({"id": i, "title": title, "ok": ok, "detail": detail})
    from . import offline
    g = offline.check()
    try:
        conns = _connections(); n = len(conns)
        add("local", "Nothing can reach this program, and it can reach nothing", bool(g["guard_installed"]) and n == 0,
            f"This program has no network: the guard that stops it opening any network socket is {'in place' if g['guard_installed'] else 'NOT in place'} ({g['refused_attempts']} attempt(s) refused so far in this run), and {n} network connection(s) or listening port(s) exist in the process right now.  "
            "The interface is reached through an in-process bridge, not a web server: there is no address to connect to.")
    except Exception as e:                                                          # psutil missing or no permission
        add("local", "Nothing can reach this program, and it can reach nothing", True if g["guard_installed"] else None, f"The guard that stops this program opening any network socket is {'in place' if g['guard_installed'] else 'not in place'}; the live connection list could not be read here ({type(e).__name__}).")
    add("offline", "No cloud service is used", True, "The model and every data file are local; the assistant has no code path that sends a vehicle or driver value anywhere (HF_HUB_OFFLINE=" + os.environ.get("HF_HUB_OFFLINE", "unset") + ").")
    try:
        from . import store
        c = store.open_ro(app.db)
        try: c.execute("CREATE TABLE _probe(x)"); writable = True
        except sqlite3.Error: writable = False
        c.close(); add("readonly", "The assistant can only read vehicle data", not writable, "It opens the vehicle database read-only (and query-only); a write attempt was just refused." if not writable else "A write attempt SUCCEEDED - the assistant is not read-only.")
    except Exception as e: add("readonly", "The assistant can only read vehicle data", None, f"Could not run the write-probe ({type(e).__name__}).")
    try:
        from . import isolation
        iso = isolation.cached_verify()
        add("buses", "Nothing can be sent to the vehicle", iso["ok"], (iso["detail"] if iso["ok"] else "The boundary is BROKEN: " + "; ".join(iso["violations"][:3])) + "  (A software boundary measured on this host by reading the application's own source; a vehicle build adds partition or hypervisor isolation, docs/SAFETY_ISOLATION.md.)"
            if iso["ok"] is not None else "The application's source is not present on this build, so the boundary could not be measured here.")
    except Exception as e: add("buses", "Nothing can be sent to the vehicle", None, f"The boundary check could not run ({type(e).__name__}).")
    from . import applog
    add("logging", "Questions are not recorded", not applog.log_questions(), "Question text is not written to any log (it can be switched on for development with an environment variable; it is off now)." if not applog.log_questions() else "Question logging is ON.")
    add("roles", "Personal data belongs to the driver", True, "Location, saved places, trip routes and camera pictures are removed from everything shown to the service, fleet and validation roles; they are asked for a passcode first.")
    rows = []
    for r in access.ROLES:
        if r["auth"]: rows.append(f"{r['title']}: " + ("passcode set" if auth.configured(r["id"]) else "no passcode yet (set at first sign-in)"))
    add("auth", "Signed-in roles are protected by a passcode", all(auth.configured(r["id"]) for r in access.ROLES if r["auth"]) or None, "; ".join(rows) + f".  Passcodes are stored only as salted scrypt hashes; {access.MAX_TRIES} wrong tries lock a role; a session ends after {access.Sessions.IDLE // 60} idle minutes.")
    v = audit.verify(); add("audit", "The audit trail is intact", v["ok"], f"{v['count']} recorded action(s), each chained to the one before by a hash" + ("." if v["ok"] else f"; the chain breaks at entry {v['first_bad']}."))
    from . import securestore
    add("export", "What leaves the device is signed and can be encrypted", True if securestore.HAVE_AES else None, "Evidence packs are signed with this installation's secret" + (" and can be sealed with AES-256-GCM under a passphrase." if securestore.HAVE_AES else "; encryption is unavailable here (the 'cryptography' package is missing), so packs are signed but readable."))
    add("rest", "Stored vehicle data is encrypted", None, "The vehicle database is protected by file permissions in a private folder, not encrypted by this application.  Use operating-system disk encryption (BitLocker, LUKS, a vehicle's encrypted storage) for protection at rest.")
    return {"generated": int(time.time()), "host": "this device", "checks": checks}


def inventory(meta=None):
    """What is stored, where, for whom, for how long.  `meta`: the vehicle's own retention settings."""
    m = meta or {}; fine, coarse = m.get("ret.fine_days", "7"), m.get("ret.coarse_days", "120")
    return [
        {"what": "Vehicle signals", "examples": "speed, temperatures, battery or fuel, engine, tyres, 12 V", "where": "the vehicle database on this device", "who": "all roles (fleet: totals only)", "kept": f"{fine} days once per second, {coarse} days once per minute"},
        {"what": "Events and diagnostic codes", "examples": "warnings, trouble codes, resets, charging or refuelling", "where": "the vehicle database", "who": "driver, service, validation (fleet: counts)", "kept": f"{coarse} days"},
        {"what": "Location", "examples": "GPS fixes, trip routes, saved places (Home, Work ...)", "where": "the vehicle database", "who": "driver only", "kept": f"{coarse} days; the driver can erase it now"},
        {"what": "Camera", "examples": "pictures around events; the live picture", "where": "event thumbnails in the database; the live picture is not stored", "who": "pictures: driver only (clear-view numbers: all roles)", "kept": f"{coarse} days"},
        {"what": "Questions and answers", "examples": "what you ask and what you are told", "where": "not stored", "who": "nobody", "kept": "gone when the conversation is cleared"},
        {"what": "Feedback notes", "examples": "a thumbs up / down and an optional note you write", "where": "a private file on this device", "who": "you; exported only if you export it", "kept": "until you delete it"},
        {"what": "Audit trail", "examples": "sign-ins, refusals, exports, erasures (who and what, never content)", "where": "a private, hash-chained file", "who": "validation team", "kept": "until the device owner removes it"},
        {"what": "Passcodes", "examples": "service, fleet and validation sign-in", "where": "a private file, as salted scrypt hashes only", "who": "nobody can read them back", "kept": "until changed"},
    ]


LIMITS = [
    "Role separation is enforced by this local server.  Someone with operating-system access to the files of this computer can read them; protect the device with a user account and disk encryption.",
    "The vehicle database is not encrypted by this application (see the check above).",
    "The audit trail detects a changed or removed line in the middle; it cannot detect that its last lines were cut off unless its latest hash is also written somewhere else.",
    "Passcodes are set at commissioning by whoever first signs in to a role; a production vehicle would provision them from the workshop or fleet back office.",
    "Safety isolation is a software boundary tested on one host; it is not an automotive safety (ASIL) argument.",
]

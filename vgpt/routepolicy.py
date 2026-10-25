"""Which role may call which route of the interface: one table, deny by default.

Every request the interface answers (`ui/server.py make_dispatcher`) is first checked here.  A route that is not in the table does not exist (404).  A route that is in the table names its rule:

    open        any role (the driver needs no sign-in): the questions, the live state, the voice controls, the role and health pages
    perm        a permission of `vgpt/access.py` the role must hold (service, fleet, validation, audit, erase, privacy)
    roles       only the listed roles (the raw feedback export belongs to the service and validation roles, never to the driver page or the fleet view)
    sim         a control of the SIMULATED car and test track (dataset switch, time speed, fault injection, camera source, reset, the driving lock): never in a production build, never for the fleet role, and never
                while the vehicle is driving with the driving lock on
    research    the interpretation-review study (its pages, its endpoints): only when the study is switched on (not in a production build) and, for every call that records anything, only with the participant's consent in the request

`decide(...)` returns None when the request may go on, or a `Verdict` (status code and a plain sentence) when it may not.  Nothing here looks at what a question says."""
from collections import namedtuple

from . import access

Verdict = namedtuple("Verdict", "code message rule")

SIM_ROLES = ("driver", "engineer", "validation")
FEEDBACK_ROLES = ("engineer", "validation")

OPEN, SIM, RESEARCH = ("open",), ("sim",), ("research",)
VIEW = "this view belongs to another role"
def perm(p, msg="this belongs to another role"): return ("perm", p, msg)
def roles(*r): return ("roles", r)


GET = {
    "/api/health": OPEN, "/api/role": OPEN, "/api/privacy": perm("privacy"),
    "/api/service": perm("service", VIEW), "/api/fleet": perm("fleet", VIEW), "/api/validation": perm("validation", VIEW), "/api/audit": perm("audit", VIEW),
    "/api/voice/status": OPEN, "/api/feedback/export": roles(*FEEDBACK_ROLES),
    "/": OPEN, "/classic": OPEN, "/app": OPEN, "/app.html": OPEN, "/headunit": OPEN, "/index.html": OPEN, "/favicon.ico": OPEN,
    "/review": RESEARCH, "/research": RESEARCH, "/research.html": RESEARCH, "/api/research/status": RESEARCH, "/api/catalogue": RESEARCH,
    "/api/state": OPEN, "/api/live": OPEN, "/api/track": OPEN, "/api/datasets": SIM, "/api/evidence": OPEN, "/api/thumb": OPEN, "/api/frame": OPEN,
}
POST = {
    "/api/login": OPEN, "/api/logout": OPEN, "/api/pin": OPEN,
    "/api/voice/start": OPEN, "/api/voice/audio": OPEN, "/api/voice/hold": OPEN, "/api/voice/arm": OPEN, "/api/voice/stop": OPEN,
    "/api/ask": OPEN, "/api/units": OPEN, "/api/feedback": OPEN, "/api/export": OPEN, "/api/privacy/erase": perm("erase", "only the driver can erase personal data"),
    "/api/interpret": RESEARCH, "/api/execute": RESEARCH, "/api/judge": RESEARCH, "/api/abandon": RESEARCH,
    "/api/restrict": SIM, "/api/timescale": SIM, "/api/fault": SIM, "/api/camera": SIM, "/api/reset": SIM, "/api/dataset": SIM,
    "/api/live/control": SIM, "/api/live/inject": SIM, "/api/live/clear": SIM, "/api/live/speed": SIM, "/api/live/jump": SIM,
}
PREFIX = {"GET": {"/modern/": OPEN}, "POST": {}}


def rule_of(method, path):
    """-> the rule of a route, or None when the route is not in the table."""
    m = str(method).upper(); table = GET if m == "GET" else POST if m == "POST" else {}
    r = table.get(path)
    if r is not None: return r
    for pre, rule in PREFIX.get(m, {}).items():
        if path.startswith(pre): return rule
    return None


def decide(method, path, role, app=None, body=None):
    """-> None (go on) or a Verdict (refuse).  `app` supplies `production`, `research`, `driving` and `restrict`; without it the build is taken as a test build."""
    rule = rule_of(method, path)
    if rule is None: return Verdict(404, "not found", "unknown")
    kind = rule[0]
    if kind == "open": return None
    if kind == "perm": return None if access.can(role, rule[1]) else Verdict(403, rule[2], "perm")
    if kind == "roles": return None if role in rule[1] else Verdict(403, "this belongs to another role", "roles")
    production = bool(getattr(app, "production", False))
    if kind == "sim":
        if production: return Verdict(403, "the simulation and test controls are not part of this build", "production")
        if role not in SIM_ROLES: return Verdict(403, "this control belongs to another role", "sim_role")
        if str(method).upper() == "POST" and getattr(app, "driving", False) and getattr(app, "restrict", False): return Verdict(423, "The test controls are locked while the vehicle is driving.", "driving")
        return None
    if kind == "research":
        if production or not getattr(app, "research", False): return Verdict(404, "not found", "research_off")
        if str(method).upper() == "POST" and not (isinstance(body, dict) and body.get("consent") is True):
            return Verdict(403, "The participant must agree to the study first: this call records the question and what the person did with the answer.", "consent")
        return None
    return Verdict(404, "not found", "unknown")


def table():
    """The whole policy as rows (method, path, rule): what the documentation and the tests list."""
    rows = [("GET", p, r) for p, r in GET.items()] + [("POST", p, r) for p, r in POST.items()]
    return rows + [("GET", pre + "*", r) for pre, r in PREFIX["GET"].items()]

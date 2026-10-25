"""One answer, four readers.  The driver gets the plain answer.  The other roles get the same answer and, under it, what their work needs - every line computed from the vehicle's record or from how the answer was made, none written
by a model:

    service engineer   the working: how the question was read, the typed query and its steps, the signals and samples used, the quality flags, and the trouble codes that are open right now
    fleet operator     this vehicle as a fleet number: the last 30 days of distance, trips, utilisation, efficiency, harsh events, alerts and open codes - no place, no driver, no picture
    validation team    the evidence: the model's reading and confidence, whether the wording was checked against the numbers, the period, the coverage of each signal and the limits of the result

`tech_of(facts)` keeps the small part of an answer's facts that these views need; `apply(role, reply, conn, db_path)` returns the reply with `answer` extended (the brief, the speech and the numbers of the plain answer are untouched)."""
from . import rolepages

NL = chr(10)
_CACHE = {}


def tech_of(f):
    """The part of an answer's facts the role views use (small, JSON-safe, nothing personal)."""
    if not isinstance(f, dict): return None
    ev = f.get("evidence") if isinstance(f.get("evidence"), dict) else {}; con = f.get("contract") if isinstance(f.get("contract"), dict) else {}
    return {"route": f.get("route"), "executor": f.get("executor"), "query": f.get("query"), "plan": [str(x) for x in (f.get("plan") or [])][:8], "frame": f.get("frame"), "pmin": f.get("pmin"), "verified": f.get("verified"),
            "status": f.get("status"), "timings_ms": f.get("timings_ms"),
            "evidence": {"period": ev.get("period"), "period_ts": ev.get("period_ts"), "samples": ev.get("samples"), "sources": (ev.get("sources") or [])[:6], "tier": ev.get("tier"), "quality": ev.get("quality"),
                         "kind": ev.get("kind"), "flags": [{"code": x.get("code"), "detail": str(x.get("detail", ""))[:90]} for x in (ev.get("flags") or [])[:4] if isinstance(x, dict)], "assumptions": (ev.get("assumptions") or [])[:3]},
            "contract": {k: con.get(k) for k in ("result_class", "resolution", "completeness", "signal_days")}}


def _n(x): return f"{x:,}" if isinstance(x, int) else f"{x:,.1f}" if isinstance(x, float) else str(x)


def _date(ts, tz=0.0): return rolepages.stamp(ts, tz)


# ------------------------------------------------------------------------------------------------------------------------------------ service engineer
def service_lines(reply, conn):
    t = reply.get("tech") or {}; ev = t.get("evidence") or {}; out = ["For the service engineer:"]
    fr = t.get("frame")
    if isinstance(fr, dict): out.append("- read as: " + " ".join(f"{k}={v}" for k, v in fr.items() if v not in (None, "none")) + (f" (model confidence {round(100 * float(t['pmin']))} %)" if t.get("pmin") is not None else ""))
    if t.get("query"): out.append(f"- query: {t['query']}  [{t.get('route')}, {t.get('executor') or 'typed'}]")
    elif t.get("route"): out.append(f"- route: {t['route']}")
    if t.get("plan"): out.append("- steps: " + " > ".join(p.split("  [")[0].split(". ", 1)[-1] for p in t["plan"]))
    if ev.get("samples") is not None or ev.get("sources"):
        out.append(f"- data: {_n(ev.get('samples', 0))} samples, {ev.get('tier') or 'recorded'} tier" + (f", from {', '.join(ev['sources'])}" if ev.get("sources") else "") + (f", quality {ev['quality']}" if ev.get("quality") else ""))
    for fl in ev.get("flags") or []: out.append(f"- flag {fl['code']}: {fl['detail']}")
    try:
        codes = [g for g in rolepages.active_codes(conn).values() if g["state"] == "active"]
        out.append("- open trouble codes: " + ("; ".join(f"{g['code']} {g['text']} (set {g['set']}x, last {_date(g['last'])})".strip() for g in codes[:5]) if codes else "none") + ("" if len(codes) <= 5 else f" and {len(codes) - 5} more"))
    except Exception: pass
    if (t.get("timings_ms") or {}).get("total") is not None: out.append(f"- answered in {t['timings_ms']['total']:.0f} ms")
    return out if len(out) > 1 else []


# ------------------------------------------------------------------------------------------------------------------------------------ fleet operator
def _data_version(db_path):
    """What the stored history contains right now - the newest minute, trip and event - so that a summary is reused only while nothing was added to the history it was made from.  None when it cannot be read (then nothing is reused)."""
    import sqlite3
    try:
        c = sqlite3.connect("file:" + str(db_path).replace(chr(92), "/") + "?mode=ro", uri=True, timeout=2.0)
        try: return tuple(c.execute(q).fetchone()[0] for q in ("SELECT MAX(ts) FROM sig_1m", "SELECT MAX(id) FROM trips", "SELECT MAX(rowid) FROM events"))
        finally: c.close()
    except Exception: return None


def fleet_lines(reply, db_path):
    if not db_path: return []
    ver = _data_version(db_path); k = _CACHE.get(db_path)
    if not k or ver is None or k[0] != ver:                                                              # keyed by the data version of the history, never by the clock: a live vehicle changes it every minute, a finished one never
        try: k = (ver, rolepages.vehicle_kpis(db_path))
        except Exception: return []
        if ver is not None: _CACHE[db_path] = k
    v = k[1]; out = [f"For the fleet operator (this vehicle, last 30 days to {v['history_to']}):"]
    out.append(f"- use: {_n(v['km'])} km in {v['trips']} trips, {v['driving_h']:g} h driving ({v['utilisation_pct']:g} % of the time), average trip {v['avg_trip_km'] if v['avg_trip_km'] is not None else '-'} km")
    if v.get("efficiency"): out.append(f"- efficiency: {v['efficiency']:g} {v['efficiency_unit']}" + (f"; harsh events {v['harsh_per_100km']:g} per 100 km" if v.get("harsh_per_100km") is not None else ""))
    out.append(f"- risk: {v['warnings']} warnings ({v['critical']} critical), {v['active_codes']} open trouble codes" + (f", service overdue by {_n(v['service_overdue_km'])} km" if v.get("service_overdue_km") else "") + f"; {v['energy_stops']} charge or fuel stops")
    out.append("- no driver, place or picture is part of this view")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ validation team
def validation_lines(reply, conn):
    t = reply.get("tech") or {}; ev = t.get("evidence") or {}; con = t.get("contract") or {}; out = ["For the validation team:"]
    fr = t.get("frame")
    if isinstance(fr, dict) and t.get("pmin") is not None: out.append(f"- model reading: {' '.join(str(v) for v in fr.values() if v not in (None, 'none'))}; lowest word probability {float(t['pmin']):.3f}" + (" (low: treat the reading as uncertain)" if float(t["pmin"]) < 0.6 else ""))
    if t.get("verified") is None: out.append(f"- wording check: not applicable - this answer came from a dedicated routine ({t.get('route') or 'unnamed'}) that computes its own figures; the typed-query check of text against numbers does not run on it")
    else: out.append("- wording check: " + ("the answer text was checked against its numbers and passed" if t.get("verified") else "NOT confirmed - the numbers were not verified against the text") + (f"; status {t['status']}" if t.get("status") else ""))
    if ev.get("period"):
        pt = ev.get("period_ts")
        out.append("- period: " + str(ev["period"]) + (f" ({_date(pt[0])} to {_date(pt[1])})" if isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(x, (int, float)) for x in pt) else ""))
    if con: out.append(f"- result class: {con.get('result_class') or '-'}; resolution {con.get('resolution') or '-'}; completeness {con.get('completeness') or '-'}")
    sd = con.get("signal_days") or {}
    for name, d in list(sd.items())[:4]:
        if isinstance(d, dict): out.append(f"- coverage of {name}: {d.get('covered', 0)} of {d.get('expected', 0)} days full" + (f", {d['partial']} partial" if d.get("partial") else "") + (f", {d['missing']} missing" if d.get("missing") else ""))
    for fl in ev.get("flags") or []: out.append(f"- quality flag {fl['code']}: {fl['detail']}")
    for a in ev.get("assumptions") or []: out.append(f"- assumption: {a}")
    out.append("- limit: this is what the vehicle recorded; a gap in the record is a gap in the answer, and the wording was never written by the model")
    return out


def apply(role, reply, conn=None, db_path=None):
    """The reply for this role.  The driver's reply is returned unchanged; a refusal, an out-of-scope answer or an error gets no extra view."""
    if not isinstance(reply, dict): return reply
    if role in (None, "driver") or not reply.get("answer") or reply.get("path") == "error": return {k: v for k, v in reply.items() if k != "tech"}
    route = str(reply.get("route") or "")
    if route.startswith(("askme:", "privacy:")) and role != "engineer": return {k: v for k, v in reply.items() if k != "tech"}
    try:
        lines = service_lines(reply, conn) if role == "engineer" else fleet_lines(reply, db_path) if role == "fleet" else validation_lines(reply, conn) if role == "validation" else []
    except Exception: lines = []
    r = {k: v for k, v in reply.items() if k != "tech"}
    if not lines: return r
    r["answer"] = str(reply["answer"]).rstrip() + NL + NL + NL.join(lines); r["role_view"] = role; return r

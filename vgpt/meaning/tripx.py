"""Questions that are about the shape of trips, not about a period: "what was the charge at the start and end of my last trip", "compare battery discharge across my last five trips".

A trip is an interval the vehicle itself recorded (the trip ledger).  So the time of such a question is not a calendar period; it is the start or the end of a trip (or each of the last N), and the value is read from the vehicle's own
tables at exactly those moments.  The trip is chosen by the typed value `literals.trip_pick` ("last", "longest" ...), the count by the number typed.  Every number is read from the ledger or from the one-minute tier by plain SQL; nothing is
estimated.  The anchors and the count are values typed in the question (like a date), not meaning: the frame still says which quantity is asked."""
import re

from .. import signal_registry as R
from . import digest as DG
from . import fuelx as FX

NUMW = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
TRIP = r"(?:trips?|drives?|journeys?)"
LAST_N = re.compile(r"\b(?:last|latest|previous|recent|past|final)\s+(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\s+" + TRIP + r"\b")
EACH = re.compile(r"\b(?:each|every|all)\s+(?:of\s+)?(?:my\s+|the\s+)?(?:last|latest|recent)?\s*(?:\d+\s+|two |three |four |five )?" + TRIP + r"\b")
START = re.compile(r"\b(?:start|starts|started|beginning|begin|begins|outset|departure|leaving|set off)\b")
END = re.compile(r"\b(?:end|ends|ended|finish|finished|arrival|arriving|destination|stopped|parked)\b")
HAS_TRIP = re.compile(r"\b" + TRIP + r"\b")
# the trip ledger's own columns: concept -> (key, scale, unit, decimals)
LEDGER = {"wh_per_km": ("wh_km", 1.0, "Wh/km", 0), "energy_used": ("kwh", 1.0, "kWh", 1), "distance_driven": ("km", 1.0, "km", 1), "speed": ("avg_speed", 1.0, "km/h", 0), "hvac_power": ("avg_hvac_w", 0.001, "kW", 1),
          "ambient_temperature": ("avg_amb", 1.0, "°C", 0)}


def _n(v, nd): return f"{v:,.{nd}f}"


def _value_at(cx, concept, ts, side):
    """The stored value of a quantity at the first (side 'start') or last ('end') minute of an interval, or None."""
    ent = (R.STORE.get(concept) or {}).get("1m")
    if not ent or ent[0] not in cx.D.have: return None
    table, col, scale = ent; col = "vmax" if table.startswith("lkv_") else col
    q = f"SELECT {col} FROM {table} WHERE ts>=? AND ts<=? AND {col} IS NOT NULL ORDER BY ts {'ASC' if side == 'start' else 'DESC'} LIMIT 1"
    r = cx.b.c.execute(q, (int(ts) - 60, int(ts) + 300) if side == "start" else (int(ts) - 300, int(ts) + 60)).fetchone()
    return None if r is None else float(r[0]) / scale


FUEL_CONCEPTS = ("fuel_rate", "wh_per_km", "energy_used")


def _fuel_trip(cx, concept, tr):
    """A combustion car's fuel over one trip, from the fuel-rate signal over the trip's own minutes and the distance of the trip: litres per 100 km (consumption) or litres (energy used)."""
    e = FX.economy(cx.b, tr["start"], tr["end"] + 60)
    if e is None: return None
    return (e["litres"], "L", 1) if concept == "energy_used" else (e["l100"], "L/100 km", 1)


def _ice(cx):
    cp = FX.caps_of(cx.res); return bool(FX.CAPS.is_combustion(cp) and cp.get("fuel"))


def _trip_value(cx, concept, tr):
    """-> (value, unit, decimals) of a quantity over one trip, or None."""
    if concept in FUEL_CONCEPTS and _ice(cx): return _fuel_trip(cx, concept, tr)
    if concept in LEDGER:
        key, k, unit, nd = LEDGER[concept]; v = tr.get(key)
        return None if v is None else (float(v) * k, unit, nd)
    pk = DG._trip_peak(cx, concept, tr["start"], tr["end"])
    if pk is None: return None
    c = cx.res.onto.concepts.get(concept); u = DG._unit(c.unit).strip() if c is not None else ""
    return pk[1], u, 1


def maybe(res, text, frame, lit, c):
    """-> a result dict or None when the question is not about trip start / end or the last N trips."""
    t = " " + str(text).lower() + " "
    if c is None or c.kind not in ("signal", "derived") or frame.act in ("define", "howto", "spec", "social", "unsafe"): return None
    m_n = LAST_N.search(t); has_trip = HAS_TRIP.search(t)
    s_, e_ = bool(START.search(t)), bool(END.search(t))
    if not has_trip: return None
    anchor = (s_ or e_) and not m_n and (frame.act in ("at", "now", "stat", "extreme") or (frame.act == "none"))
    if not anchor and not (m_n and frame.act in ("compare", "stat", "trend", "extreme", "now", "at")): return None
    cx = DG.Ctx(res, c, frame, lit, text)
    try:
        if anchor: return _bounds(cx, c, s_, e_)
        n = int(NUMW.get(m_n.group(1), m_n.group(1))) if m_n else 5
        return _last_n(cx, c, max(2, min(n, 12)))
    except Exception as e:
        import logging; logging.getLogger("vehiclegpt").warning("trip executor failed: %s: %s", type(e).__name__, e); return None


def _bounds(cx, c, want_start, want_end):
    tr, which = DG._pick(cx)
    if tr is None: return DG._gap(cx, "There is no trip on record to read the start or the end of.", "trip:none", ok=False)
    cx.step(f"the {which} trip of the trip ledger ({cx.stamp(tr['start'])} to {cx.stamp(tr['end'])})")
    lab = cx.label_of(c.id); out = []; vals = {}
    for side, key in (("start", "soc_start"), ("end", "soc_end")):
        if (side == "start" and not want_start and want_end) or (side == "end" and not want_end and want_start): continue
        v = tr.get(key) if c.id == "battery_soc" else None
        if v is None: v = _value_at(cx, c.id, tr["start"] if side == "start" else tr["end"], side)
        if v is None: continue
        vals[side] = v; unit = DG._unit(c.unit); nd = 0 if c.id == "battery_soc" else 1
        out.append(f"at the {side} ({cx.stamp(tr['start'] if side == 'start' else tr['end']).split(', ')[-1]}) it was {_n(v, nd)}{unit}")
    if not out: return DG._gap(cx, f"This vehicle does not store {lab} at the start or the end of its trips, so I cannot say.", "trip:nodata", ok=False)
    txt = f"On your {which} trip ({cx.stamp(tr['start'])}), the {lab}: " + "; ".join(out) + "."
    if "start" in vals and "end" in vals:
        d = vals["end"] - vals["start"]; u = DG._unit(c.unit)
        txt += f"\nIt {'fell' if d < 0 else 'rose' if d > 0 else 'did not change'}" + (f" by {_n(abs(d), 0 if c.id == 'battery_soc' else 1)}{' points' if c.unit == '%' else u} during the trip" if d else "") + f" ({_n(tr['km'], 1)} km)."
    return DG._result(cx, txt, "trip:bounds", trip_start=int(tr["start"]), event_ts=int(tr["start"]))


def _last_n(cx, c, n):
    trips = cx.trips(cx.now - 90 * 86400, cx.hi)
    if not trips: return DG._gap(cx, "There are no trips on record.", "trip:none", ok=False)
    trips = trips[-n:]; cx.step(f"the last {len(trips)} trips of the trip ledger")
    rows = []
    for tr in trips:
        v = _trip_value(cx, c.id, tr)
        if v is not None: rows.append((tr, v))
    lab = ("fuel used" if c.id == "energy_used" else "fuel consumption") if c.id in FUEL_CONCEPTS and _ice(cx) else cx.label_of(c.id)
    if not rows: return DG._gap(cx, f"This vehicle does not store {lab} per trip, so I cannot compare it across trips.", "trip:nodata", ok=False)
    unit, nd = rows[0][1][1], rows[0][1][2]; u = (" " + unit) if unit else ""
    lines = [f" - {cx.stamp(tr['start'])}, {_n(tr['km'], 1)} km: {_n(v[0], nd)}{u}" for tr, v in rows]
    hi = max(rows, key=lambda x: x[1][0]); lo = min(rows, key=lambda x: x[1][0])
    txt = f"{lab[:1].upper() + lab[1:]} over your last {len(rows)} trips (oldest first):\n" + "\n".join(lines)
    if len(rows) >= 2 and hi is not lo: txt += f"\nHighest: {cx.stamp(hi[0]['start'])} ({_n(hi[1][0], nd)}{u}); lowest: {cx.stamp(lo[0]['start'])} ({_n(lo[1][0], nd)}{u})."
    return DG._result(cx, txt, "trip:last_n", trips=len(rows), values=[round(v[0], 3) for _, v in rows])

"""Combustion vehicles (diesel and petrol) in the meaning pipeline.

The model reads WHAT a person asks - "how much energy is left", "what is the consumption", "when was the last fill-up", "how is the motor" - without knowing what powers the vehicle.  This module
executes those meanings in FUEL terms for a vehicle whose own data shows a fuel level and no traction battery (`vehicle/caps.py` decides that from the database, never from a setting):

    energy left      fuel level (%, litres) and the range it gives at the recent fuel economy
    consumption      litres per 100 km over the period asked, litres used
    a fill-up        the refuel events (tank level jumps): latest, first, list, count, existence
    engine health    coolant, oil, load and the diesel / petrol specific signals the vehicle reports, with the overheating warnings of the last week
and, for ANY vehicle, "what kind of vehicle is this": the model name AND what its data says it is (`caps.summary`).

Every answer says where its numbers come from; a quantity the data does not hold is said to be missing, never estimated from something else.
"""
import sqlite3

from ..query_ir import TimeSpec
from ..vehicle import caps as CAPS
from .. import analytics as A
from ..nearest import nearest
from .eventx import DAY, _plural, _r, _window, _win

NL = chr(10)

ENERGY_LEFT = {"battery_soc", "range_estimate", "fuel_level", "battery_condition", "battery_capacity", "charge_time"}
CONSUMPTION = {"wh_per_km", "energy_used", "fuel_rate"}


def caps_of(res):
    b = res.brain
    if getattr(b, "_vcaps", None) is None: b._vcaps = CAPS.detect(b.c)
    return b._vcaps


def _sid(b, concept):
    rd = getattr(b, "_vreader", None)
    if rd is None: rd = b._vreader = CAPS.Reader(b.c)
    return rd.sid.get(concept), rd


def _latest(b, concept):
    """(value, timestamp) of the newest stored value of a lake signal at or before the vehicle's moment, from the 1 Hz rows if there are any, else the minute rows."""
    sid, rd = _sid(b, concept)
    if sid is None: return None
    best = None
    for tb in ("lk_1s", "lk_1m"):
        try: r = b.c.execute(f"SELECT ts, v FROM {tb} WHERE sid=? AND ts<=? ORDER BY ts DESC LIMIT 1", (sid, int(b.now))).fetchone()
        except sqlite3.Error: r = None
        if r and r[1] is not None and (best is None or r[0] > best[0]): best = r
    return (float(best[1]), int(best[0])) if best else None


def economy(b, lo, hi):
    """Litres per 100 km over [lo, hi): fuel burned (the fuel-rate signal, one value per recorded minute) over the distance (the speed signal)."""
    sid, rd = _sid(b, "fuel_rate")
    if sid is None: return None
    try:
        lit = b.c.execute("SELECT SUM(v) / 60.0, COUNT(*) FROM lk_1m WHERE sid=? AND ts>=? AND ts<?", (sid, int(lo), int(hi))).fetchone()
        km = b.c.execute("SELECT SUM(speed) / 600.0 FROM sig_1m WHERE ts>=? AND ts<? AND speed IS NOT NULL", (int(lo), int(hi))).fetchone()[0]
    except sqlite3.Error: return None
    if not lit or not lit[0] or not km or km < 1.0: return None
    return {"litres": float(lit[0]), "km": float(km), "l100": float(lit[0]) / float(km) * 100.0}


def recent_economy(b):
    for span in (7200, DAY, 30 * DAY):
        e = economy(b, b.now - span, b.now + 1)
        if e and e["km"] >= 5.0: return dict(e, window_s=span)
    return None


def _ok(res, text, route, frame, **facts): return _r(res, text, route, frame, **facts)


def fuel_left(res, frame):
    b = res.brain; cp = caps_of(res); lv = _latest(b, "fuel_level")
    if lv is None: return _r(res, "This vehicle reports a fuel level, but no value is stored yet, so I cannot say how much fuel is left.", "now:fuel_level", frame, ok=False)
    pct, ts = lv; tank = cp.get("tank_l"); lit = pct / 100.0 * tank if tank else None; A.set_tz(b.tz)
    eco = recent_economy(b); rng = CAPS.range_km(pct, tank, eco["l100"]) if eco and tank else None
    head = ("This vehicle has no traction battery; its energy is fuel. " if frame.concept.startswith("battery") or frame.concept == "charge_time" else "") + f"Fuel level is {pct:.0f} %" + (f" ({lit:.0f} L of {tank:.0f} L)" if lit is not None else "") + (" right now." if b.now - ts <= 900 else f" as of {A.stamp(ts)}.")
    lines = [head]
    if rng is not None: lines.append(f"At the recent {eco['l100']:.1f} L/100 km that is about {rng:.0f} km of range (an estimate from your own driving, not a guarantee).")
    elif frame.concept == "range_estimate": lines.append("The range cannot be estimated: there is not enough fuel-rate and speed history yet.")
    lines += ["Based on the fuel-level signal" + (f" and {eco['km']:.0f} km of recent driving." if eco else "."), "Evidence: measured fuel level; the range is an estimate."]
    return _r(res, "\n".join(lines), "now:fuel_level", frame, fuel_pct=pct, fuel_l=lit, range_km=rng, l100=eco["l100"] if eco else None)


def consumption(res, c, frame, lit):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    if frame.act not in ("stat", "now", "estimate", "health", "advice"): return None
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day")) if frame.act == "stat" else (b.now - 7200, b.now + 1, "recently")
    e = economy(b, lo, hi)
    if e is None: return _r(res, f"No driving with fuel-rate data is recorded {label}, so there is no fuel used or consumption to report.", "stat:fuel_economy", frame, ok=False)
    if c.id == "energy_used" and frame.op == "sum":
        return _r(res, f"{e['litres']:.0f} L of fuel were used {label} over {e['km']:.0f} km ({e['l100']:.1f} L/100 km).\nBased on the fuel-rate signal, one value per recorded minute.\nEvidence: measured, integrated over time.", "stat:fuel_used", frame, litres=e["litres"], km=e["km"])
    return _r(res, f"Fuel consumption {label} was {e['l100']:.1f} L/100 km ({e['litres']:.0f} L over {e['km']:.0f} km).\nBased on the fuel-rate signal (one value per recorded minute) and the distance driven.\nEvidence: measured, integrated over time.", "stat:fuel_economy", frame, l100=e["l100"], litres=e["litres"], km=e["km"])


def refuels(res, c, frame, lit):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    if frame.act not in ("events", "stat", "extreme"): return None
    lo, hi, label = _window(b, qc, lit, TimeSpec("all") if frame.op in ("last", "first", "exists") else TimeSpec("relative", 90, "day"))
    rows = [(ts, d) for ts, d in b.c.execute("SELECT ts, detail FROM events WHERE type='REFUEL' AND ts>=? AND ts<? ORDER BY ts", (int(lo), int(hi))).fetchall()]
    if not rows: return _r(res, f"No refuelling is recorded {label}. A refuel shows as a jump of the fuel level while the vehicle is parked.", "events:refuel", frame, ok=False, count=0)
    def parse(d):                                                                                       # "62 L added (tank 14% to 92%)": a value written by the data pipeline, split, not matched
        try: lit, rest = str(d).split(" L added (tank ", 1); a, b = rest.rstrip(")").split("% to ", 1); return float(lit), float(a), float(b.rstrip("%"))
        except ValueError: return None, None, None
    one = lambda ts, d: f"{A.stamp(ts)}: {d}"
    n = len(rows); litres = sum(parse(d)[0] or 0 for _, d in rows); wk = max((hi - lo) / (7 * DAY), 1e-9)
    if frame.op in ("last", "first"):
        ts, d = rows[-1] if frame.op == "last" else rows[0]
        return _r(res, f"The {'latest' if frame.op == 'last' else 'first'} refuel {label}: {one(ts, d)}.\n{_plural(n, 'refuel')} {label}.\nBased on the refuel events (tank level jumps while parked).", "events:refuel", frame, count=n, ts=ts)
    if frame.op == "list": return _r(res, f"{_plural(n, 'refuel')} {label}" + (" (the latest 5):" if n > 5 else ":") + "\n" + "\n".join(" - " + one(ts, d) for ts, d in rows[-5:][::-1]) + "\nBased on the refuel events.", "events:refuel", frame, count=n)
    return _r(res, (f"Yes - {_plural(n, 'refuel')} {label}." if frame.op == "exists" else f"{_plural(n, 'refuel')} {label} ({n / wk:.1f} per week, {litres:.0f} L added in all).") + "\nBased on the refuel events.", "events:refuel", frame, count=n, litres=litres)


def engine_health(res, frame):
    b = res.brain; cp = caps_of(res); out, have = [], False; A.set_tz(b.tz)
    parts = (("motor_temperature", "Coolant", "°C"), ("engine_oil_temperature", "oil", "°C"), ("engine_load", "load", " %"), ("engine_rpm", "speed", " rpm"))
    seen = []
    for concept, label, unit in parts:
        lv = _latest(b, concept) if concept != "motor_temperature" else None
        if concept == "motor_temperature":
            r = _motor_now(b); lv = (r[1], r[0]) if r else None
        if lv: seen.append(f"{label} {lv[0]:.0f}{unit}")
    x = {}
    for concept in ("dpf_soot_load", "def_level", "long_term_fuel_trim", "catalyst_temperature"):
        lv = _latest(b, concept)
        if lv: x[concept] = lv[0]
    if "dpf_soot_load" in x: seen.append(f"particulate filter {x['dpf_soot_load']:.0f} % full")
    if "def_level" in x: seen.append(f"AdBlue {x['def_level']:.0f} %")
    if "long_term_fuel_trim" in x: seen.append(f"long-term fuel trim {x['long_term_fuel_trim']:+.1f} %")
    if "catalyst_temperature" in x: seen.append(f"catalyst {x['catalyst_temperature']:.0f} °C")
    warn = b.c.execute("SELECT COUNT(*), MAX(ts) FROM events WHERE type IN ('WARN_OVERTEMP','DTC_SET') AND ts>? AND ts<=?", (int(b.now) - 7 * DAY, int(b.now))).fetchone()
    if not seen: return _r(res, "This vehicle sends no engine readings, so I cannot say how the engine is.", "health:engine", frame, ok=False)
    verdict = "Nothing alarming: " if not (x.get("dpf_soot_load", 0) >= 85 or x.get("def_level", 100) < 15 or abs(x.get("long_term_fuel_trim", 0)) >= 15) and not warn[0] else "Worth a look: "
    wtxt = f" {_plural(warn[0], 'warning or fault code')} in the last 7 days (latest {A.stamp(warn[1])})." if warn[0] else " No overheating warning or fault code in the last 7 days."
    return _r(res, verdict + "engine " + ", ".join(seen) + "." + wtxt + "\nBased on the latest stored value of each engine signal and the event log.", "health:engine", frame, ok=True, readings=seen)


def identity(res, base):
    cp = caps_of(res); r = dict(base); r["answer"] = r["answer"].rstrip("\n") + "\nFrom the data this vehicle sends: " + cp["summary"]; r["brief"] = r.get("brief") or r["answer"].split("\n", 1)[0]
    return r


def power_asked(res, text):
    """The question carries a word the glossary tags as POWER (power, kW, hp ...): data, so a missing word is one JSON line to add."""
    g = getattr(res, "glossary", None)
    return bool(g) and any(e.get("quantity") == "power" for e in g.lookup(text))


def engine_power(res, frame, cp):
    """Power at the crankshaft from the two signals every engine reports: torque and speed (P[kW] = torque[Nm] x rpm / 9549)."""
    b = res.brain; t = _latest(b, "engine_torque"); r = _latest(b, "engine_rpm"); A.set_tz(b.tz); seen = ", ".join(CAPS.NAMES.get(c, c) for c in cp["read"] if c in CAPS.ENGINE_SIGNALS)
    if t is None or r is None:
        miss = " and ".join(n for n, v in (("torque", t), ("speed (rpm)", r)) if v is None)
        return _r(res, f"Engine power is worked out from the engine's torque and speed, and this vehicle does not report its {miss}." + (f"\nIt does report: {seen}." if seen else "") + "\nNothing was estimated from something else.", "now:engine_power", frame, ok=False)
    unit = str(((cp.get("concepts") or {}).get("engine_torque") or {}).get("unit") or "Nm").strip().lower()
    if unit not in ("nm", "n.m", "n m"):
        return _r(res, f"This engine reports its torque as {unit}, a share of a reference torque, so its power cannot be worked out without the engine's reference torque (not reported).\nNothing was estimated.", "now:engine_power", frame, ok=False)
    (tq, ts_t), (rpm, ts_r) = t, r; ts = min(ts_t, ts_r); when = " right now." if b.now - ts <= 900 else f" as of {A.stamp(ts)}."
    if rpm < 100: head = "The engine is not running (about 0 rpm), so it is producing no power" + when
    else: kw = tq * rpm / 9549.0; head = f"Engine power is about {kw:.0f} kW ({kw * 1.34102:.0f} hp)" + when
    return _r(res, head + f"\nCalculated from engine torque {tq:.0f} Nm at {rpm:,.0f} rpm (power = torque x speed / 9549).\nEvidence: computed from two measurements; the engine's rated output is not reported.", "now:engine_power", frame, kw=None if rpm < 100 else kw, torque_nm=tq, rpm=rpm)


def power_stat(res, frame, lit, cp):
    """Highest / average engine power over a period, from the minute averages of torque and speed while the engine ran."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); st, sr = _sid(b, "engine_torque")[0], _sid(b, "engine_rpm")[0]
    if st is None or sr is None: return engine_power(res, frame, cp)                                  # says which of the two the vehicle does not report
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day"))
    rows = b.c.execute("SELECT t.ts, t.v * r.v / 9549.0 FROM lk_1m t JOIN lk_1m r ON r.ts = t.ts AND r.sid = ? WHERE t.sid = ? AND t.ts >= ? AND t.ts < ? AND r.v > 100", (sr, st, int(lo), int(hi))).fetchall()
    if not rows: return _r(res, f"The engine was not running {label}, or its torque and speed were not recorded, so there is no engine power to report.", "stat:engine_power", frame, ok=False)
    tail = f"\nBased on {len(rows):,} one-minute samples with the engine running (torque x speed / 9549); the averaging smooths short peaks.\nEvidence: computed from measurements."
    if frame.op in ("max", "min"):
        ts, kw = (max if frame.op == "max" else min)(rows, key=lambda x: x[1]); return _r(res, f"The {'highest' if frame.op == 'max' else 'lowest'} engine power {label} was about {kw:.0f} kW ({kw * 1.34102:.0f} hp), on {A.stamp(ts)}." + tail, "stat:engine_power", frame, kw=kw)
    kw = sum(x[1] for x in rows) / len(rows); return _r(res, f"The engine's average power while running {label} was about {kw:.0f} kW ({kw * 1.34102:.0f} hp)." + tail, "stat:engine_power", frame, kw=kw)


def silent(res, frame, cp):
    """A vehicle whose own file says it has an engine but whose data holds none of it: say so, plainly, instead of answering in electric-car terms."""
    d = cp.get("declared") or {}
    reported = [n for flag, n in ((cp["speed"], "speed"), (cp["gps"], "position"), (cp["ambient"], "outside temperature"), (cp["tyres"], "tyre pressures"), (cp["camera"], "camera events")) if flag]
    batt = frame.concept.startswith(("battery", "charg", "regen", "cell", "inverter")) or frame.concept in ("charge_time", "charge_cost", "vehicle_health")                  # the missing battery is only worth a sentence when the question is about the battery
    return _r(res, f"This is a {d.get('fuel_type') or 'combustion'} vehicle (as its file labels it)" + (", so it has no traction battery. " if batt else ". ") + f"Its data holds only {', '.join(reported) or 'a few basic signals'} - no engine, fuel or power signal - so there is nothing to answer this from.\nNothing was estimated.", "now:not_reported", frame, ok=False)


def _motor_now(b):
    """(timestamp, degrees C) of the newest stored engine (coolant) temperature: the 1 Hz rows first - they are what has just arrived - then the minute rows."""
    for tb in ("sig_1s", "sig_1m"):
        try: r = b.c.execute(f"SELECT ts, motor / 10.0 FROM {tb} WHERE ts<=? AND motor IS NOT NULL ORDER BY ts DESC LIMIT 1", (int(b.now),)).fetchone()
        except sqlite3.Error: r = None
        if r: return int(r[0]), float(r[1])
    return None


def engine_temperature(res, frame, lit, text=""):
    """'How hot is the motor / engine': on a combustion vehicle the stored temperature of the drivetrain is the engine's coolant temperature (the same column an electric car uses for its motor)."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    if frame.act == "at":                                                                              # "motor temp 4pm yesterday": the stored value nearest that instant
        from . import literals as L
        ts = L.instant(text, b.now, b.tz) if text else None
        if ts is None: return _r(res, "Which time do you mean? Give me a clock time or how long ago (for example '3 pm yesterday' or '2 hours ago').", "clarify:instant", frame, ok=False)
        r = nearest(b.c, "sig_1m", "ts, motor / 10.0", int(ts), "motor")
        if not r or abs(r[0] - ts) > 180: return _r(res, f"No engine temperature is recorded close enough to {A.stamp(ts)}.", "at:engine_temperature", frame, ok=False)
        return _r(res, f"Engine coolant temperature at {A.stamp(r[0])} was {r[1]:.0f} °C.\nBased on the nearest stored one-minute value.", "at:engine_temperature", frame, value=r[1], sample_ts=r[0])
    if frame.act == "now":
        r = _motor_now(b)
        if not r: return _r(res, "This vehicle reports no engine temperature yet, so I cannot say how hot the engine is.", "now:engine_temperature", frame, ok=False)
        return _r(res, f"Engine coolant temperature is {r[1]:.0f} °C" + (" right now." if b.now - r[0] <= 900 else f" as of {A.stamp(r[0])}.") + "\nBased on the latest stored value of the engine temperature signal (coolant).", "now:engine_temperature", frame, value=r[1])
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day")); hi = min(int(hi), int(b.now) + 1)
    try: rows = [(int(t), float(v)) for t, v in b.c.execute("SELECT ts, motor / 10.0 FROM sig_1m WHERE ts>=? AND ts<? AND motor IS NOT NULL", (int(lo), hi)).fetchall()]
    except sqlite3.Error: rows = []
    if not rows: return _r(res, f"No engine temperature is recorded {label}.", "stat:engine_temperature", frame, ok=False)
    vals = [v for _, v in rows]
    if frame.op in ("max", "min"):
        ts, v = (max if frame.op == "max" else min)(rows, key=lambda x: x[1]); return _r(res, f"The {'highest' if frame.op == 'max' else 'lowest'} engine coolant temperature {label} was {v:.0f} °C, on {A.stamp(ts)}.\nBased on {len(rows):,} one-minute samples.\nEvidence: measured.", "stat:engine_temperature", frame, value=v)
    import statistics
    v = statistics.median(vals) if frame.op == "median" else statistics.mean(vals)
    return _r(res, f"The {'median' if frame.op == 'median' else 'average'} engine coolant temperature {label} was {v:.0f} °C (highest {max(vals):.0f}, lowest {min(vals):.0f}).\nBased on {len(rows):,} one-minute samples.\nEvidence: measured.", "stat:engine_temperature", frame, value=v)


class _EngineTemp:
    id, kind, label, unit = "motor_temperature", "signal", "engine coolant temperature", "°C"        # what an electric car calls the motor's temperature is the engine's coolant on a combustion vehicle


def engine_temperature_series(res, frame, lit, text):
    """Trend / unusual-behaviour questions about the engine temperature: the generic digests (trendv, digest.anomaly) on the stored drivetrain temperature, worded as the engine's coolant."""
    from . import digest as DG, trendv as TV
    c = _EngineTemp()
    if frame.act == "trend" and frame.op in ("rise", "drop", "change"):
        v = TV.verdict(res, c, frame, lit, text)
        if v is None: return _r(res, "No engine temperature is recorded in that period, so I cannot say whether it changed.", "trend:engine_temperature", frame, ok=False)
        head, detail, d, f = v; w = lambda t: ("".join(t) if "°C in the first" in t else t.replace(" in the first half", " °C in the first half").replace(" in the second half", " °C in the second half").replace(" in the second.", " °C in the second.")).replace("motor_temperature", "engine coolant temperature").replace("motor temperature", "engine coolant temperature")
        return _r(res, w(head) + "\n" + w(detail), "trend:engine_temperature", frame, direction=d, first=f["first"], second=f["second"])
    r = DG.anomaly(DG.Ctx(res, c, frame, lit, text), c)
    if isinstance(r, dict) and isinstance(r.get("answer"), str): r = dict(r, answer=r["answer"].replace("motor_temperature", "engine coolant temperature").replace("motor temperature", "engine coolant temperature"), brief=str(r.get("brief", "")).replace("motor_temperature", "engine coolant temperature").replace("motor temperature", "engine coolant temperature"))
    return r


def no_battery(res, frame, cp):
    """A question about something only an electric vehicle has (charging, state of charge, battery capacity, regeneration ...) asked of a vehicle whose data shows an engine and no traction battery."""
    b = res.brain; kind = cp.get("fuel_type") or (cp.get("declared") or {}).get("fuel_type") or "combustion"; lv = _latest(b, "fuel_level") if cp.get("fuel") else None
    fuel = f" Its energy is fuel: the tank is at {lv[0]:.0f} %." if lv else ""
    return _r(res, f"This is a {kind} vehicle: it has no traction battery, so there is no charging, state of charge, battery capacity or regeneration to report.{fuel}\nNothing was estimated from another quantity.", "outcome:no_traction_battery", frame, ok=True)


def dpf_regen_events(res, frame, lit):
    """Particulate-filter regeneration episodes (runs of minutes with the regeneration signal on): when the last one was, how many, how long."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); sid, rd = _sid(b, "dpf_regeneration"); NL = chr(10)
    if sid is None: return _r(res, "This vehicle records no particulate filter regeneration (it may have no particulate filter), so there is nothing to date or count. Nothing was estimated.", "events:dpf_regeneration", frame, ok=False)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day")); hi = min(int(hi), int(b.now) + 1)
    rows = b.c.execute("SELECT ts FROM lk_1m WHERE sid=? AND v>0.5 AND ts>=? AND ts<? ORDER BY ts", (sid, int(lo), hi)).fetchall()
    eps = []
    for (ts,) in rows:
        if eps and ts - eps[-1][1] <= 180: eps[-1][1] = ts
        else: eps.append([ts, ts])
    if not eps:
        return _r(res, ("No" if frame.op in ("exists", "count") else "There was no") + f" particulate filter regeneration {label}.", "events:dpf_regeneration", frame, count=0)
    fmt = lambda e: f"{A.stamp(e[0])}, about {max(1, round((e[1] - e[0]) / 60) + 1)} min"
    plural = "s" if len(eps) != 1 else ""
    tail = NL + "Based on the regeneration signal (one value per recorded minute); a run of minutes counts as one regeneration."
    if frame.op == "last": return _r(res, f"The last particulate filter regeneration {label} was on {fmt(eps[-1])}. {len(eps)} in all {label}." + tail, "events:dpf_regeneration", frame, count=len(eps), start=eps[-1][0])
    if frame.op == "exists": return _r(res, f"Yes - {len(eps)} particulate filter regeneration{plural} {label}." + tail, "events:dpf_regeneration", frame, count=len(eps))
    if frame.op == "list": return _r(res, f"{len(eps)} particulate filter regeneration{plural} {label}" + (" (the latest 5):" if len(eps) > 5 else ":") + NL + NL.join(" - " + fmt(e) for e in eps[-5:][::-1]) + tail, "events:dpf_regeneration", frame, count=len(eps))
    return _r(res, f"{len(eps)} particulate filter regeneration{plural} {label}, the latest on {fmt(eps[-1])}." + tail, "events:dpf_regeneration", frame, count=len(eps))


def fill_tank(res, frame, cp):
    """"How long to fill the tank": the litres still needed (from the stored level and the tank size) and what a typical stop added; the time depends on the pump, which the vehicle does not record, so no time is invented."""
    b = res.brain; lv = _latest(b, "fuel_level"); tank = cp.get("tank_l")
    if lv is None or not tank: return None
    pct, ts = lv; need = (100.0 - pct) / 100.0 * tank; A.set_tz(b.tz)
    rows = [d for (d,) in b.c.execute("SELECT detail FROM events WHERE type='REFUEL' ORDER BY ts DESC LIMIT 20").fetchall()]
    adds = sorted(float(str(d).split(" L added", 1)[0]) for d in rows if " L added" in str(d))
    typical = f" Your last {len(adds)} refuels added a typical {adds[len(adds) // 2]:.0f} L (from {adds[0]:.0f} to {adds[-1]:.0f} L)." if len(adds) >= 3 else ""
    return _r(res, f"The tank is at {pct:.0f} %, so about {need:.0f} L of {tank:.0f} L is still needed to fill it.{typical}{NL}How long that takes depends on the pump's flow, which the vehicle does not record, so I cannot give a time (a refuel shows only as a jump of the fuel level while parked).{NL}Based on the fuel-level signal and the refuel events.", "now:fill_tank", frame, litres_needed=round(need, 1), level_pct=round(pct, 1))


def economy_by_hour(res, c, frame, lit, cp):
    """Which hour of the day burns the most (or least) fuel per 100 km: the fuel-rate and speed minutes of the window grouped by the local clock hour they were recorded in; hours with under 5 km of driving are left out."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); tz = int(b.tz)
    sid, rd = _sid(b, "fuel_rate")
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 90, "day")); hi = min(int(hi), int(b.now) + 1)
    if sid is None: return None
    try: rows = b.c.execute("SELECT CAST(((l.ts + ?) % 86400) / 3600 AS INTEGER) h, SUM(l.v) / 60.0, SUM(s.speed) / 600.0 FROM lk_1m l JOIN sig_1m s ON s.ts = l.ts WHERE l.sid=? AND l.ts>=? AND l.ts<? AND s.speed IS NOT NULL AND s.mode IN (1, 2) GROUP BY h", (tz, sid, int(lo), hi)).fetchall()
    except sqlite3.Error: return None
    per = {int(h): (float(l) / float(k) * 100.0, float(k)) for h, l, k in rows if l and k and k >= 5.0}
    if len(per) < 2: return _r(res, f"Fewer than two hours of the day have at least 5 km of driving with fuel-rate data {label}, so I cannot compare them.", "calx:by_hour", frame, ok=False)
    top, low = max(per, key=lambda h: per[h][0]), min(per, key=lambda h: per[h][0]); pick = low if frame.op == "min" else top; other = top if pick == low else low
    txt = (f"Fuel consumption by hour of the day, {label}: the {'lowest' if pick == low else 'highest'} is between {pick:02d}:00 and {pick + 1:02d}:00 ({per[pick][0]:.1f} L/100 km over {per[pick][1]:.0f} km). "
           f"The {'highest' if pick == low else 'lowest'} is between {other:02d}:00 and {other + 1:02d}:00 ({per[other][0]:.1f} L/100 km).{NL}By hour: " + "; ".join(f"{h:02d}h {per[h][0]:.1f}" for h in sorted(per)) +
           f".{NL}Based on the fuel-rate and speed minutes while driving; hours with under 5 km of driving are left out. An association with the time of day, not a cause.{NL}Evidence: measured, integrated over time.")
    return _r(res, txt, "calx:by_hour", frame, hours=len(per), best=int(pick), l100=round(per[pick][0], 2))


def economy_bins(res, c, frame, lit, cp):
    """Which day / week burned the most (or least) fuel per 100 km: the economy of each local day or week of the window, days and weeks with under 5 km are left out."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); week = frame.ent == "week"
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 90 if week else 30, "day")); hi = min(int(hi), int(b.now) + 1); step = 7 * DAY if week else DAY; tz = int(b.tz)
    start = ((int(lo) + tz) // step) * step - tz if not week else lo; rows = []
    t = start
    while t < hi:
        e = economy(b, max(t, lo), min(t + step, hi))
        if e and e["km"] >= 5.0: rows.append((t, e))
        t += step
    if len(rows) < 2: return _r(res, f"Fewer than two {'weeks' if week else 'days'} with at least 5 km of driving and fuel-rate data are recorded {label}, so I cannot rank them.", "extreme:fuel_economy", frame, ok=False)
    best = (max if frame.op == "max" else min)(rows, key=lambda r: r[1]["l100"]); order = sorted(rows, key=lambda r: r[1]["l100"], reverse=frame.op == "max")[1:3]
    nm = "week starting " + A.stamp(best[0])[:10] if week else "day " + A.stamp(best[0])[:10]
    fmt = lambda r: f"{A.stamp(r[0])[:10]}: {r[1]['l100']:.1f} L/100 km"
    return _r(res, f"The {'week' if week else 'day'} with the {'highest' if frame.op == 'max' else 'lowest'} fuel consumption {label} was the {nm}: {best[1]['l100']:.1f} L/100 km ({best[1]['litres']:.0f} L over {best[1]['km']:.0f} km).\nNext: " + "; ".join(fmt(r) for r in order) + f".\nBased on the fuel-rate signal and the speed signal, {len(rows)} {'weeks' if week else 'days'} with enough driving.\nEvidence: measured, integrated over time.", "extreme:fuel_economy", frame, l100=best[1]["l100"], start=best[0])


def economy_usual(res, c, frame, lit, cp):
    """Is this period's fuel consumption normal for this vehicle: against the same-length periods before it (at least six)."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day")); hi = min(int(hi), int(b.now) + 1); span = max(hi - lo, 3600); e1 = economy(b, lo, hi)
    if e1 is None: return _r(res, f"No driving with fuel-rate data is recorded {label}, so it cannot be compared with the usual.", "compare:fuel_economy_usual", frame, ok=False)
    hist = sorted(e["l100"] for e in (economy(b, lo - k * span, lo - (k - 1) * span) for k in range(1, 13)) if e and e["km"] >= 5.0)
    if len(hist) < 6: return _r(res, f"Only {len(hist)} earlier periods of the same length have fuel-rate data, and at least 6 are needed to say what is usual for this vehicle. I will not guess.", "compare:fuel_economy_usual", frame, ok=False)
    q = lambda p: hist[min(len(hist) - 1, max(0, int(round(p * (len(hist) - 1)))))]; v = e1["l100"]; where = "higher than usual" if v > q(0.9) else "lower than usual" if v < q(0.1) else "within the usual range"; below = sum(1 for x in hist if x < v) / len(hist)
    return _r(res, f"{label.capitalize()}, fuel consumption was {v:.1f} L/100 km, which is {where} for this vehicle: the usual is about {q(0.5):.1f} L/100 km (the middle 80 % of earlier periods lie between {q(0.1):.1f} and {q(0.9):.1f}); this period is higher than {below * 100:.0f} % of them.\nBased on {len(hist)} earlier periods of the same length from this vehicle's own history.\nEvidence: measured, integrated over time.", "compare:fuel_economy_usual", frame, l100=v, typical=q(0.5), windows=len(hist))


def economy_change(res, c, frame, lit, cp):
    """How fuel consumption (and so the range a tank gives) changed: the period asked (default the last 7 days) against the period of the same length just before it.  Associations only, never a cause."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    two = [t for t in (lit.times or []) if t.type != "current"]
    if len(two) >= 2:                                                                                  # two named periods ("before and after my last service", "this month and last month"): the older is the base
        (lo0, hi0, label0), (lo, hi, label) = sorted((_win(b, qc, t) for t in two[:2]), key=lambda w: w[0]); span = hi - lo; e1 = economy(b, lo, hi); e0 = economy(b, lo0, hi0); named = (label0, label)
    else:
        lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 28 if frame.act == "trend" else 7, "day")); span = hi - lo; e1 = economy(b, lo, hi); e0 = economy(b, lo - span, lo); named = None
    asked_range = frame.concept == "range_estimate"
    if e1 is None or e0 is None:
        have = "no driving with fuel-rate data" if e1 is None else "no earlier driving with fuel-rate data to compare with"
        return _r(res, f"There is {have} {label}, so I cannot say whether fuel consumption{' (and so the range)' if asked_range else ''} changed. This vehicle has no traction battery; its range comes from the fuel in the tank and its fuel economy.", "trend:fuel_economy", frame, ok=False)
    d = (e1["l100"] - e0["l100"]) / e0["l100"] * 100.0; tank = cp.get("tank_l"); word = "higher" if d > 0 else "lower"
    lines = [f"Fuel consumption {label} was {e1['l100']:.1f} L/100 km against {e0['l100']:.1f} L/100 km {named[0] if named else 'in the period of the same length before it'} ({abs(d):.0f} % {word})."]
    if frame.act == "trend" and frame.op in ("rise", "drop") and frame.concept != "fuel_level":                  # the asked direction answered first: yes / no
        what = "the range a tank gives" if asked_range else "fuel consumption"; went_up = (d < 0) if asked_range else (d > 0); asked_up = frame.op == "rise"
        if abs(d) < 3.0: lines.insert(0, f"No - {what} stayed about the same {label}.")
        else: lines.insert(0, ("Yes - " if went_up == asked_up else "No - ") + f"{what} {'rose' if went_up else 'fell'} {label}.")
    if asked_range and tank: lines.append(f"A full {tank:.0f} L tank therefore gives about {tank / e1['l100'] * 100:.0f} km now, against about {tank / e0['l100'] * 100:.0f} km before.")
    elif asked_range: lines.append("The range a tank gives moves the opposite way to consumption.")
    try:
        v = lambda a, z: b.c.execute("SELECT AVG(speed) / 10.0 FROM sig_1m WHERE ts>=? AND ts<? AND speed>50", (int(a), int(z))).fetchone()[0]
        s1, s0 = v(lo, hi), v(lo - span, lo)
        if s1 and s0: lines.append(f"Average speed while moving: {s1:.0f} km/h now, {s0:.0f} km/h before. These are recorded alongside the change; they do not prove what caused it.")
    except sqlite3.Error: pass
    lines += [f"Based on the fuel-rate signal over {e1['km']:.0f} km and {e0['km']:.0f} km of driving.", "Evidence: measured, integrated over time."]
    return _r(res, "\n".join(lines), "trend:fuel_economy", frame, l100=e1["l100"], l100_before=e0["l100"], change_pct=d)



MODE_OF = {"parked": (0,), "city": (1,), "highway": (2,), "driving": (1, 2)}                         # the driving-mode column of the minute table (0 parked, 1 city, 2 highway)
MODE_WORDS = {"city": "in the city", "highway": "on the highway", "driving": "while driving", "parked": "while parked"}


def economy_modes(b, lo, hi, modes):
    """Litres per 100 km over [lo, hi) counting only the minutes whose driving mode is in `modes`."""
    sid, rd = _sid(b, "fuel_rate")
    if sid is None: return None
    q = ",".join("?" * len(modes))
    try: r = b.c.execute(f"SELECT SUM(l.v) / 60.0, SUM(s.speed) / 600.0 FROM lk_1m l JOIN sig_1m s ON s.ts = l.ts WHERE l.sid=? AND l.ts>=? AND l.ts<? AND s.mode IN ({q}) AND s.speed IS NOT NULL", (sid, int(lo), int(hi), *modes)).fetchone()
    except sqlite3.Error: return None
    if not r or not r[0] or not r[1] or r[1] < 1.0: return None
    return {"litres": float(r[0]), "km": float(r[1]), "l100": float(r[0]) / float(r[1]) * 100.0}


def compare_modes(res, c, frame, lit):
    """Fuel consumption in one driving condition against another ("city and highway", "driving against parked")."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    a_, z_ = frame.cond, frame.vs
    if a_ not in MODE_OF or z_ not in MODE_OF: return None
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day"))
    e1, e0 = economy_modes(b, lo, hi, MODE_OF[a_]), economy_modes(b, lo, hi, MODE_OF[z_])
    if e1 is None or e0 is None:
        miss = a_ if e1 is None else z_
        return _r(res, f"There is not enough driving {MODE_WORDS[miss]} with fuel-rate data {label} to compare fuel consumption {MODE_WORDS[a_]} with {MODE_WORDS[z_]}.", "compare:fuel_economy", frame, ok=False)
    d = (e1["l100"] - e0["l100"]) / e0["l100"] * 100.0
    return _r(res, f"Fuel consumption {label}: {e1['l100']:.1f} L/100 km {MODE_WORDS[a_]} ({e1['litres']:.0f} L over {e1['km']:.0f} km) against {e0['l100']:.1f} L/100 km {MODE_WORDS[z_]} ({e0['litres']:.0f} L over {e0['km']:.0f} km): {abs(d):.0f} % {'higher' if d > 0 else 'lower'} {MODE_WORDS[a_]}.\nBased on the fuel-rate signal, one value per recorded minute, split by the driving mode of each minute.\nEvidence: measured, integrated over time.", "compare:fuel_economy", frame, a=e1["l100"], b=e0["l100"], change_pct=d)


def trip_fuel(res, c, frame, lit):
    """The trip that burned the most (or least) fuel, or used it at the highest (lowest) rate: litres from the fuel-rate signal over the trip's own minutes."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    sid, rd = _sid(b, "fuel_rate")
    if sid is None or frame.ent != "trip": return None
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day"))
    try: trips = b.c.execute("SELECT start, end, km FROM trips WHERE start>=? AND start<? AND km>=3 ORDER BY start", (int(lo), int(hi))).fetchall()
    except sqlite3.Error: return None
    rows = []
    for st, en, km in trips:
        try: l = b.c.execute("SELECT SUM(v) / 60.0 FROM lk_1m WHERE sid=? AND ts>=? AND ts<=?", (sid, int(st), int(en))).fetchone()[0]
        except sqlite3.Error: l = None
        if l and l > 0: rows.append((st, km, float(l), float(l) / km * 100.0))
    if not rows: return _r(res, f"No trip of 3 km or more with fuel-rate data is recorded {label}.", "extreme:fuel_trip", frame, ok=False)
    f = min if frame.op == "min" else max; word = "least" if frame.op == "min" else "most"; by_l, by_r = f(rows, key=lambda x: x[2]), f(rows, key=lambda x: x[3])
    txt = f"The trip that used the {word} fuel {label} started on {A.stamp(by_l[0])}: {by_l[2]:.1f} L over {by_l[1]:.0f} km ({by_l[3]:.1f} L/100 km)."
    if by_r[0] != by_l[0]: txt += f"\nThe {'lowest' if frame.op == 'min' else 'highest'} consumption per km was on the trip of {A.stamp(by_r[0])}: {by_r[3]:.1f} L/100 km ({by_r[2]:.1f} L over {by_r[1]:.0f} km)."
    return _r(res, txt + f"\nBased on {len(rows)} trips of 3 km or more from the trip ledger and the fuel-rate signal over each trip.\nEvidence: measured, integrated over time.", "extreme:fuel_trip", frame, l100=by_r[3], litres=by_l[2], km=by_l[1])


def engine_idling(res, frame, lit):
    """Idling of a combustion engine: the engine turning while the vehicle stands still - not the hours it was parked with the engine off."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    sid, rd = _sid(b, "engine_rpm")
    if sid is None: return None
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 7, "day"))
    try: r = b.c.execute("SELECT COUNT(*) FROM lk_1m l JOIN sig_1m s ON s.ts = l.ts WHERE l.sid=? AND l.ts>=? AND l.ts<? AND l.v>=300 AND s.speed IS NOT NULL AND s.speed < 10", (sid, int(lo), int(hi))).fetchone()
    except sqlite3.Error: return None
    run = b.c.execute("SELECT COUNT(*) FROM lk_1m WHERE sid=? AND ts>=? AND ts<? AND v>=300", (sid, int(lo), int(hi))).fetchone()[0]
    n = int(r[0] or 0); h, m = divmod(n, 60)
    return _r(res, f"The engine was idling (running while the vehicle stood still) for {h} h {m:02d} min {label}" + (f", {100.0 * n / run:.0f} % of the time it ran." if run else ".") + "\nBased on one-minute samples of the engine speed (300 rpm or more) and the vehicle speed (under 1 km/h).\nEvidence: measured, one-minute samples.", "stat:engine_idling", frame, minutes=n)


def tank_size(res, frame, cp):
    t = cp.get("tank_l")
    if not t: return _r(res, "The size of the fuel tank is not recorded for this vehicle, and I will not estimate it.", "spec:tank", frame, ok=False)
    return _r(res, f"The fuel tank holds {t:.0f} L (from the vehicle profile).", "spec:tank", frame, tank_l=t)


def fuel_advice(res, frame, lit):
    """What the vehicle's own recent driving says about its fuel use; no generic tips are made up."""
    b = res.brain; A.set_tz(b.tz); e = recent_economy(b)
    if e is None: return _r(res, "No driving with fuel-rate data is recorded recently, so there is nothing in this vehicle's own data to base fuel-saving advice on.", "advice:fuel", frame, ok=False)
    lo, hi = b.now - 30 * DAY, b.now + 1; city, hwy = economy_modes(b, lo, hi, (1,)), economy_modes(b, lo, hi, (2,)); lines = [f"Your fuel consumption was {e['l100']:.1f} L/100 km over {e['km']:.0f} km of recent driving."]
    if city and hwy: lines.append(f"In the city it was {city['l100']:.1f} L/100 km and on the highway {hwy['l100']:.1f} L/100 km, so " + ("short city trips cost the most per km: combine errands into one trip." if city["l100"] > hwy["l100"] else "high-speed driving costs the most per km: a steady, lower cruising speed saves fuel."))
    try:
        hb = b.c.execute("SELECT COUNT(*) FROM events WHERE type IN ('HARSH_ACCEL','HARSH_BRAKE') AND ts>=?", (int(lo),)).fetchone()[0]
        if hb: lines.append(f"There were {hb} hard accelerations or hard brakes in the last 30 days; smoother driving uses less fuel.")
    except sqlite3.Error: pass
    lines += ["Based on the fuel-rate signal, the speed signal and the recorded events.", "Evidence: measured, integrated over time."]
    return _r(res, "\n".join(lines), "advice:fuel", frame, l100=e["l100"])


POWER_CONCEPTS = {"battery_power", "motor_power", "motor_power_estimate", "horsepower"}              # what the model calls power: on a combustion vehicle it is the engine's
POWER_MISREAD = {"battery_capacity", "engine_load", "battery_condition", "motor_torque"}               # read as power only when the question carries a power word
SKIP_ACTS = ("define", "howto", "why", "context", "social", "offtopic", "unsupported", "control", "unsafe", "ambiguous", "spec")
NO_BATTERY = {"battery_voltage", "battery_current", "battery_temperature", "battery_soh", "cell_temperature_max", "cell_temperature_min", "cell_voltage_min", "charging_power", "charging_state", "regen_power", "regen_energy",
              "battery_efficiency", "thermal_rise_rate", "nominal_voltage", "charge_cycles", "home_charger_power", "export_power", "charge_cost", "charging_session", "battery_capacity"}     # things only a traction battery has


def maybe(res, text, frame, lit):
    """The combustion-vehicle executors; None when the question is not one of theirs (or the vehicle has a battery)."""
    from . import layers as LY
    if not LY.on("fuel_remap"): return None
    cp = caps_of(res); cid = frame.concept
    if cp["battery"] or not CAPS.is_combustion(cp): return None
    if (frame.act in ("summary", "anomaly") or (frame.act == "stat" and frame.op == "usual") or (frame.act == "why" and frame.ent in ("trip", "day"))) and cid in res.onto.concepts: return None          # digests (vgpt/meaning/digest.py) read a quantity this vehicle has in its own terms; only what it lacks (a battery) is answered here
    if frame.act == "spec" and cid == "fuel_level" and cp["fuel"]: return tank_size(res, frame, cp)
    if frame.act == "stat" and cid == "drive_mode" and frame.op == "duration" and frame.cond == "idling" and cp["engine"]:
        r_ = engine_idling(res, frame, lit)
        if r_ is not None: return r_
    battery_q = cid in NO_BATTERY or cid == "battery_soc"
    if frame.act in SKIP_ACTS and not (frame.act in ("spec", "why") and battery_q) and not (frame.act == "why" and (cid in CONSUMPTION or cid == "range_estimate")): return None
    O = res.onto; c = O.concepts.get(cid)
    if not cp["fuel"] and not cp["engine"]:                                                              # an engine by the file's own label, but no engine data at all
        from ..vehicle import lake as LK
        return silent(res, frame, cp) if cid in (ENERGY_LEFT | CONSUMPTION | POWER_CONCEPTS | POWER_MISREAD | set(CAPS.ENGINE_SIGNALS) | set(LK.EV_ONLY) | NO_BATTERY | {"motor_condition"}) else None
    if cp["engine"] and (cid in POWER_CONCEPTS or (cid in POWER_MISREAD and power_asked(res, text))):
        if frame.act in ("now", "estimate", "health", "advice"): return engine_power(res, frame, cp)
        if frame.act in ("stat", "extreme"): return power_stat(res, frame, lit, cp)
    if cid == "motor_temperature" and frame.act in ("now", "at", "stat", "extreme") and (cp["engine"] or cp["fuel"]): return engine_temperature(res, frame, lit, text)
    if cid == "motor_temperature" and frame.act in ("trend", "anomaly") and (cp["engine"] or cp["fuel"]): return engine_temperature_series(res, frame, lit, text)
    if cid == "battery_soc" and frame.act in ("trend", "extreme", "why", "compare", "at", "events"): return no_battery(res, frame, cp)
    if cid in NO_BATTERY - {"charging_session", "battery_capacity"} or (cid in ("battery_capacity", "charging_session") and frame.act in ("spec", "now", "at", "trend", "why", "compare", "health")):
        if not (cid == "battery_capacity" and frame.act in ("now", "estimate", "health", "advice", "stat")):          # 'how much energy is left' is answered in fuel terms below
            return no_battery(res, frame, cp)
    if not cp["fuel"]: return None
    if cid == "fuel_level" and frame.act == "stat" and frame.op in ("sum", "count"): return consumption(res, O.concepts.get("energy_used") or type("C", (), {"id": "energy_used"})(), type(frame)("stat", "energy_used", "sum", frame.cond, frame.vs, frame.ent), lit)       # "how much fuel did I use": the fuel burned, not the gauge
    if cid in CONSUMPTION and frame.act == "compare" and frame.cond in MODE_OF and frame.vs in MODE_OF:
        r_ = compare_modes(res, c, frame, lit)
        if r_ is not None: return r_
    if cid in CONSUMPTION and frame.act == "extreme" and frame.ent == "trip":
        r_ = trip_fuel(res, c, frame, lit)
        if r_ is not None: return r_
    if cid in CONSUMPTION and frame.act == "stat" and frame.op == "corr": return None                      # a relationship between two series is the generic executor's
    if cid in CONSUMPTION and frame.act == "advice": return fuel_advice(res, frame, lit)
    if cid == "dpf_regeneration" and frame.act in ("events", "stat") and frame.op in ("last", "count", "exists", "list"):
        r_ = dpf_regen_events(res, frame, lit)
        if r_ is not None: return r_
    if cid in CONSUMPTION and frame.act in ("extreme", "stat", "compare", "trend") and getattr(lit, "pattern", False):
        r_ = economy_by_hour(res, c, frame, lit, cp)
        if r_ is not None: return r_
    if cid in CONSUMPTION and frame.act == "extreme" and frame.ent in ("day", "week") and frame.op in ("max", "min"): return economy_bins(res, c, frame, lit, cp)
    if cid in CONSUMPTION and frame.act == "compare" and frame.vs == "typical": return economy_usual(res, c, frame, lit, cp)
    if cid == "fuel_level" and frame.act == "extreme" and frame.ent in ("none", "day", "week", "month"): return None                       # the lowest / highest tank level of a period is a plain stored-signal extreme
    if (cid in CONSUMPTION or cid in ("range_estimate", "fuel_level")) and frame.act in ("trend", "why", "compare", "extreme"): return economy_change(res, c, frame, lit, cp)
    if cid == "charge_time" and frame.act in ("now", "estimate", "stat", "advice", "extreme", "compare"):
        r_ = fill_tank(res, frame, cp)
        if r_ is not None: return r_
    if cid in ENERGY_LEFT and frame.act in ("now", "estimate", "health", "advice", "stat"): return fuel_left(res, frame)
    if cid in CONSUMPTION and not (cid == "fuel_rate" and frame.act == "now"): return consumption(res, c if c is not None else type("C", (), {"id": cid})(), frame, lit)
    if cid == "charging_session":
        r = refuels(res, c if c is not None else type("C", (), {"id": cid})(), frame, lit)
        return r if r is not None else no_battery(res, frame, cp)
    if cid in ("motor_condition", "vehicle_health", "motor_temperature") and frame.act in ("health", "advice") and cid != "vehicle_health": return engine_health(res, frame)
    return None

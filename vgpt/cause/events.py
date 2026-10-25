"""Why did that event happen?  The cause of a single recorded event (a warning, a reset, a shutdown, a disengagement), read from what the vehicle recorded around it.

Three sources are combined, each with its own kind of evidence:
  * the EVENTS themselves, ordered in time and linked by a small causal graph of what can cause what within how many seconds (a supply dip can cause a shutdown, a controller reset can cause the assistance system to
    disengage, bus errors can silence a source and set a lost-communication code);
  * the LOG LINES the controllers wrote, whose wording names the mechanism (a lost heartbeat, a lost supply rail, an error counter);
  * the SIGNALS before the event, for the causes that leave no event of their own (a battery that heated up because the coolant flow was low, or because the power was high, or because of a fast charge).
The answer is a chain from the root to the event, the evidence for each link, what would confirm it, and a confidence; when nothing recorded explains the event it says so.

The knowledge here (the graph, the wording of the log mechanisms) is data about vehicles, not language rules: the question's wording is read by the model, this module only reads the vehicle's records."""
import math
import re
import sqlite3

import numpy as np

# --- what can cause what: (cause event type, effect event type, max seconds from cause to effect, why)
GRAPH = [
    ("VOLTAGE_DIP", "UNEXPECTED_SHUTDOWN", 5, "a dip of the 12 V supply below what the control units need makes them lose power"),
    ("LOW_12V", "UNEXPECTED_SHUTDOWN", 30, "the 12 V supply was low shortly before"),
    ("VOLTAGE_DIP", "ECU_RESET", 10, "the supply dipped, so the controller restarted"),
    ("UNEXPECTED_SHUTDOWN", "ECU_RESET", 30, "the power-on restart after the loss of supply"),
    ("ECU_RESET", "ADAS_DISENGAGE", 6, "the assistance controller restarted, so the assistance function dropped out"),
    ("ECU_RESET", "ADAS_STATE_CHANGE", 6, "the assistance controller restarted"),
    ("CAN_ERROR", "SOURCE_SILENT", 15, "the error frames kept the bus from carrying data"),
    ("CAN_ERROR", "DTC_SET", 40, "the bus errors set a lost-communication code"),
    ("SOURCE_SILENT", "DTC_SET", 40, "the silent source set a lost-communication code"),
    ("COOLANT_FLOW_DROP", "WARN_BATT_OVERTEMP", 900, "less coolant flow, so the pack could not shed its heat"),
    ("COOLANT_FLOW_DROP", "WARN_OVERTEMP", 900, "less coolant flow, so the part could not shed its heat"),
    ("WARN_BATT_OVERTEMP", "POWER_DERATE", 60, "the pack was too hot, so power was limited"),
    ("WARN_OVERTEMP", "POWER_DERATE", 60, "a part was too hot, so power was limited"),
    ("CAMERA_BLOCKED", "ADAS_DISENGAGE", 120, "the camera could not see, so the assistance function dropped out"),
    ("CAMERA_DEGRADED", "ADAS_DISENGAGE", 120, "the camera view was degraded"),
    ("VOLTAGE_DIP", "DTC_SET", 60, "the supply dip set a system-voltage code"),
    ("UNEXPECTED_SHUTDOWN", "DTC_SET", 60, "the loss of power set stored codes"),
]
# --- wording of the mechanisms controllers log (regular expression, cause id)
LOG_MECHANISMS = [
    (re.compile(r"heartbeat .*lost|watchdog|restarting", re.I), "ecu_watchdog_reset"),
    (re.compile(r"12\s?V.*(lost|dip)|supply lost|rail dipped|contactors? opened", re.I), "supply_voltage_loss"),
    (re.compile(r"error counter|error frames|bus.*off|error-passive", re.I), "bus_disturbance"),
    (re.compile(r"closing speed|AEB triggered", re.I), "obstacle_ahead"),
    (re.compile(r"coolant (pump )?flow low", re.I), "cooling_flow_low"),
]
SENTINELS = (-40.0, 255.0, 3276.7, -273.0, 6553.5)

CAUSES = {
    "supply_voltage_loss": ("a loss of the 12 V supply", "The 12 V supply dipped or was lost, so control units lost power or restarted and the stored codes name the supply.", "Check the 12 V battery, its terminals, the DC/DC converter and the fuses; read the stored supply codes."),
    "ecu_watchdog_reset": ("a watchdog reset of a controller", "A controller stopped answering its supervisor (a lost heartbeat), was restarted by its watchdog, and whatever it provided dropped out for the seconds it took to boot.", "Read the controller's reset log and software version; check for a software update, supply dips or a memory fault."),
    "bus_disturbance": ("an electrical disturbance on a data bus", "The bus filled with error frames (an electrical disturbance: a loose connector, a noisy load, a damaged wire), nothing could be received for a while, and a lost-communication code was set.", "Inspect the bus wiring and connectors of the affected network; look for a load switching at the time."),
    "speed_scaling_mismatch": ("two speed sources that differ by a constant factor", "The speed from one source reads a constant percentage above the other at every speed: a wheel or tyre size, or a calibration constant, differs between the two paths (a changed tyre size, not a sensor failure).", "Compare the tyre size with the configured rolling circumference and recalibrate the source that is off."),
    "invalid_sensor_value": ("an invalid value passed on by one path", "One path reports a value that is an out-of-range placeholder (an 'invalid' marker) while the other path reports a normal reading: the sensor itself is fine, the value was lost or not available on that path.", "Check the gateway or the display path that forwards the value; no sensor replacement is indicated."),
    "obstacle_ahead": ("a closing speed to an obstacle", "Emergency braking was triggered by a high closing speed to something ahead.", "No fault implied; review the camera clip of the event if the braking seemed unnecessary."),
    "cooling_flow_low": ("low coolant flow", "The coolant flow was below its usual value, so the part could not shed its heat.", "Check the coolant pump, the level and the lines."),
    "high_power_load": ("a sustained high power draw", "A long demand for high power heated the pack faster than the cooling removed it.", "No fault implied; the driving demanded it."),
    "fast_charging_heat": ("heat from fast charging", "The pack was charging at high power shortly before; fast charging heats the pack, especially on a warm day.", "No fault implied; precondition the pack before fast charging on warm days."),
    "hot_ambient": ("a hot day", "The outside temperature was high, which leaves the cooling less margin.", "No fault implied."),
    "heat_soak": ("heat soak after stopping", "The vehicle had just stopped, the coolant stopped flowing and the temperature at the sensor rose for a few minutes before falling.", "No fault implied."),
    "lane_marking_lost": ("lane markings not visible", "The assistance function dropped out because the camera lost the lane markings.", "No fault implied; worn markings, glare or heavy rain are typical."),
    "camera_view_blocked": ("the camera's view was degraded", "The camera view was blocked or degraded just before.", "Clean the windscreen in front of the camera."),
    "supply_dip": ("a short dip of the 12 V supply", "The 12 V supply dipped briefly (typically during start-up or a high load).", "Load-test the 12 V battery if the dips repeat."),
    "driver_or_system_limit": ("the assistance system reached its operating limit", "The assistance function ended itself because the situation was outside what it handles.", "No fault implied."),
}


# which kinds of recorded event each mechanism can explain; a mechanism logged near an event it cannot explain is not evidence (coolant pump logs say nothing about why the assistance system dropped out)
EXPLAINS = {"ecu_watchdog_reset": {"ECU_RESET", "ADAS_DISENGAGE", "ADAS_STATE_CHANGE", "DTC_SET", "SOURCE_SILENT", "UNEXPECTED_SHUTDOWN"}, "supply_voltage_loss": {"UNEXPECTED_SHUTDOWN", "ECU_RESET", "DTC_SET", "VOLTAGE_DIP", "LOW_12V", "SOURCE_SILENT", "ADAS_DISENGAGE"},
            "bus_disturbance": {"CAN_ERROR", "SOURCE_SILENT", "DTC_SET", "SENSOR_CONFLICT", "SENSOR_INVALID"}, "obstacle_ahead": {"AEB_ACTIVATION", "HARSH_BRAKE"}, "cooling_flow_low": {"WARN_BATT_OVERTEMP", "WARN_OVERTEMP", "POWER_DERATE", "COOLANT_FLOW_DROP"}}


def _rows(conn, sql, args=()):
    try: return conn.execute(sql, args).fetchall()
    except sqlite3.Error: return []


def events_around(conn, ts, before=900, after=60):
    return _rows(conn, "SELECT ts, type, sev, detail FROM events WHERE ts BETWEEN ? AND ? ORDER BY ts", (ts - before, ts + after))


def logs_around(conn, ts, before=120, after=30):
    return _rows(conn, "SELECT ts, src, msg FROM logs WHERE ts BETWEEN ? AND ? ORDER BY ts", (ts - before, ts + after))


def anchor(conn, ts, event_type=None, tol=30):
    """The event to explain: of the events within `tol` seconds of ts (or of the given type), the one that is most severe, then the nearest."""
    rank = {"critical": 0, "warn": 1, "info": 2}
    ev = [e for e in events_around(conn, ts, tol, tol) if e[1] not in ("HARSH_BRAKE", "HARSH_ACCEL", "HARSH_CORNER", "CHARGE_START")]
    if event_type: ev = [e for e in ev if e[1] == event_type]
    if not ev: return None
    return min(ev, key=lambda e: (rank.get(e[2], 3), abs(e[0] - ts)))


def causal_chain(conn, a):
    """Walk back from the anchor along the graph: the observed events that could have caused it, and what caused those.  -> [(event, link text), ...] oldest first."""
    chain = [(a, None)]; cur = a; seen = {(a[0], a[1])}
    for _ in range(6):
        best = None
        for ev in events_around(conn, cur[0], 3600, 0):
            if (ev[0], ev[1]) in seen or ev[0] > cur[0]: continue
            for c, e, lag, why in GRAPH:
                if c == ev[1] and e == cur[1] and 0 <= cur[0] - ev[0] <= lag:
                    gap = cur[0] - ev[0]
                    if best is None or gap < best[0]: best = (gap, ev, why)
        if best is None: break
        gap, ev, why = best; chain.append((ev, f"{why} ({gap} s earlier)")); seen.add((ev[0], ev[1])); cur = ev
    chain.reverse(); return chain


def _signals(conn, ts, before=900, after=0):
    cols = "ts, speed, power, batt, flow, amb, motor"
    for table in ("sig_1s", "sig_1m"):
        rows = _rows(conn, f"SELECT {cols} FROM {table} WHERE ts BETWEEN ? AND ? ORDER BY ts", (ts - before, ts + after))
        if len(rows) >= (60 if table == "sig_1s" else 5): return table, np.array(rows, float)
    return None, np.zeros((0, 7))


def _usual(conn, col, ts, days=28):
    r = _rows(conn, f"SELECT AVG({col}), MIN({col}), MAX({col}) FROM sig_1m WHERE ts BETWEEN ? AND ? AND mode IN (1,2)", (ts - days * 86400, ts - 3600))
    return r[0] if r and r[0][0] is not None else None


def explain_event(conn, ts, event_type=None, use_logs=True):
    """-> dict(event, causes [ranked, with evidence], chain, timeline) or None when no event is near `ts`."""
    a = anchor(conn, ts, event_type)
    if a is None: return None
    t0 = a[0]; chain = causal_chain(conn, a); logs = logs_around(conn, t0) if use_logs else []; ev_all = events_around(conn, t0, 120, 60); cands = {}

    def add(cid, score, *evidence):
        if cid in EXPLAINS and a[1] not in EXPLAINS[cid] and not (cid == "supply_voltage_loss" and a[1] == "DTC_SET"): return
        c = cands.setdefault(cid, {"id": cid, "score": 0.0, "evidence": []}); c["score"] = max(c["score"], score); c["evidence"] += [e for e in evidence if e]
    # -- the causal chain: the root of what was recorded is the strongest explanation
    if len(chain) > 1:
        root = chain[0][0]; rt = root[1]
        mapping = {"VOLTAGE_DIP": "supply_voltage_loss", "LOW_12V": "supply_voltage_loss", "UNEXPECTED_SHUTDOWN": "supply_voltage_loss", "ECU_RESET": "ecu_watchdog_reset", "CAN_ERROR": "bus_disturbance", "COOLANT_FLOW_DROP": "cooling_flow_low",
                   "CAMERA_BLOCKED": "camera_view_blocked", "CAMERA_DEGRADED": "camera_view_blocked", "SOURCE_SILENT": "bus_disturbance"}
        cid = mapping.get(rt)
        if cid: add(cid, 0.9, *[f"{e[1].replace('_', ' ').lower()} at {e[0] - t0:+d} s: {e[3][:90]}" + (f" - {why}" if why else "") for e, why in chain])
    # -- the controllers' own wording
    for t, src, msg in logs:
        for rx, cid in LOG_MECHANISMS:
            if rx.search(msg):
                near = abs(t - t0) <= 30; sc = 0.85 if (near and cid != "cooling_flow_low") else 0.55
                add(cid, sc, f"{src} logged at {t - t0:+d} s: \"{msg[:110]}\"")
    # -- the wording of the events themselves (an ECU reset event says what reset it)
    for e in ev_all:
        if abs(e[0] - t0) <= 30:
            for rx, cid in LOG_MECHANISMS:
                if rx.search(e[3] or "") and cid not in ("cooling_flow_low",): add(cid, 0.8 if e[0] <= t0 else 0.6, f"{e[1].replace('_', ' ').lower()} at {e[0] - t0:+d} s says: \"{(e[3] or '')[:100]}\"")
    # -- the anchor's own detail
    typ, det = a[1], a[3] or ""
    if typ == "VOLTAGE_DIP": add("supply_dip", 0.85, f"{det[:100]}")
    if typ == "SENSOR_CONFLICT":
        low = det.lower(); nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", det)]
        if any(abs(x - s) < 1e-6 for x in nums for s in SENTINELS): add("invalid_sensor_value", 0.9, f"one source reads {min(nums, key=lambda x: abs(x - -40))} - an out-of-range placeholder - while the other reads a normal value: {det[:110]}")
        elif "speed" in low:
            ratio = _speed_ratio(conn, t0)
            if ratio is not None: add("speed_scaling_mismatch", 0.9, f"across all {ratio[2]} recorded speed conflicts the second source reads {ratio[0]:.1f} % above the first, each time within {ratio[1]:.1f} % of that: a constant factor, not a failing sensor")
            else: add("speed_scaling_mismatch", 0.5, det[:100])
    if typ == "AEB_ACTIVATION": add("obstacle_ahead", 0.8, det[:100])
    if typ == "ADAS_DISENGAGE":
        low = det.lower()
        if "lane" in low: add("lane_marking_lost", 0.7, f"the system reported: {det}")
        elif "camera" in low or "blocked" in low: add("camera_view_blocked", 0.8, f"the system reported: {det}" + (f"; camera events recorded in the minutes before: {', '.join(sorted({e[1] for e in ev_all if e[1].startswith('CAMERA')}))}" if any(e[1].startswith("CAMERA") for e in ev_all) else ""))
        elif "system limit" in low and not cands: add("driver_or_system_limit", 0.5, f"the system reported: {det}")
    if typ in ("WARN_BATT_OVERTEMP", "WARN_OVERTEMP", "POWER_DERATE"): _thermal(conn, t0, a, ev_all, add)
    if typ in ("CAN_ERROR", "SOURCE_SILENT") and "bus_disturbance" not in cands: add("bus_disturbance", 0.8, det[:120])
    ranked = sorted(cands.values(), key=lambda c: -c["score"])
    out = []
    for c in ranked:
        t, mech, chk = CAUSES.get(c["id"], (c["id"], "", ""))
        out.append({"id": c["id"], "title": t, "confidence": round(min(0.97, c["score"]), 2), "mechanism": mech, "check": chk, "evidence": c["evidence"][:5]})
    timeline = [{"t": e[0] - t0, "type": e[1], "detail": (e[3] or "")[:100]} for e in ev_all if e[1] not in ("HARSH_BRAKE", "HARSH_ACCEL", "CHARGE_START")]
    return {"event": {"ts": t0, "type": typ, "detail": det, "severity": a[2]}, "causes": out, "chain": [{"t": e[0] - t0, "type": e[1], "detail": (e[3] or "")[:100], "link": why} for e, why in chain], "timeline": timeline[:14], "explained": bool(out)}


def _speed_ratio(conn, ts):
    rows = _rows(conn, "SELECT detail FROM events WHERE type='SENSOR_CONFLICT' AND detail LIKE '%speed%'")
    ratios = []
    for (d,) in rows:
        m = re.search(r"reads (\d+(?:\.\d+)?) km/h but .* reads (\d+(?:\.\d+)?) km/h", d)
        if m and float(m.group(1)) > 20: ratios.append(float(m.group(2)) / float(m.group(1)))
    if len(ratios) < 3: return None
    r = np.array(ratios); cv = float(np.std(r) / np.mean(r)) * 100
    return (float(np.mean(r) - 1) * 100, cv, len(r)) if cv < 2.0 and abs(float(np.mean(r)) - 1) > 0.01 else None


def _thermal(conn, t0, a, ev_all, add):
    """Overheating and derating leave their cause in the signals before them."""
    tab, S = _signals(conn, t0, 900, 0)
    if tab is None: return
    ts, speed, power, batt, flow, amb, motor = S.T; fine = tab == "sig_1s"; scale = 10.0 if np.nanmax(flow) > 50 else 1.0
    run = flow[flow > 0.5 * scale] / scale                                                                  # the pump idles at zero when parked: judge it while it runs
    hist = _rows(conn, "SELECT flow FROM sig_1m WHERE ts BETWEEN ? AND ? AND flow > ?", (t0 - 28 * 86400, t0 - 3600, 0.5 * scale))
    if len(run) >= 3 and len(hist) >= 200:
        usual = float(np.percentile([h[0] for h in hist], 75)) / scale; now_f = float(np.percentile(run, 80))
        if usual > 0 and now_f < 0.7 * usual: add("cooling_flow_low", min(0.9, 0.5 + (1 - now_f / usual)), f"coolant flow {now_f:.1f} L/min while running against {usual:.1f} usually (-{(1 - now_f / usual) * 100:.0f} %)")
    p = float(np.nanmean(power[-(300 if fine else 5):])) / (1000.0 if abs(np.nanmean(power)) > 500 else 1.0)
    u = _usual(conn, "power", t0); up = (u[0] / (1000.0 if abs(u[0]) > 500 else 1.0)) if u else None
    if up is not None and p > max(3.0 * up, 25.0): add("high_power_load", min(0.85, 0.5 + p / 200.0), f"mean power {p:.0f} kW over the last minutes against {up:.0f} kW on average")
    ch = _rows(conn, "SELECT start, kw FROM charges WHERE start BETWEEN ? AND ? AND kw >= 40 ORDER BY start DESC LIMIT 1", (t0 - 3600, t0))
    if ch: add("fast_charging_heat", 0.85, f"a {ch[0][1]:.0f} kW charge began {(t0 - ch[0][0]) / 60:.0f} min before")
    amb_v = float(np.nanmedian(amb[-(60 if fine else 3):])); amb_v = amb_v / 10.0 if abs(amb_v) > 80 else amb_v
    if amb_v >= 33: add("hot_ambient", 0.5 + min(0.3, (amb_v - 33) / 20.0), f"outside temperature {amb_v:.0f} °C")
    sp = speed[-(120 if fine else 3):]
    if np.nanmax(speed[-(900 if fine else 15):-(120 if fine else 3)] if len(speed) > 10 else speed) > 10 and np.nanmax(sp) < 1 and a[1] != "POWER_DERATE": add("heat_soak", 0.7, "the vehicle had just stopped from driving and the temperature was still rising")

"""DEV tool (needs cantools + numpy arrays from sim): turns the simulated vehicle into REAL wire-format traffic.

  CAN / CAN-FD  encoded by `cantools` (an independent implementation) from vgpt/protocols/vehicle.dbc; BmsStatus is a 16-byte CAN-FD frame
  ISO-TP + UDS  DTC reports (0x59) on 0x7E8, segmented into first/consecutive frames when longer than 7 bytes; OBD-II mode 01 PIDs on 0x7E8
  SOME/IP       binary notifications (ADAS Disengaged, ChargeState, SessionStarted) + SOME/IP-SD OfferService at stream start
  Android VHAL  VehiclePropValue parcelables with the real AOSP property ids and native units
  Logs          QNX slog2info lines (thermal manager) and an Android logcat ERROR line for the HMI warning
Yields (ts_float, bus, payload) with bus in can | someip | vhal | log | cam, in the order a gateway would see them."""
import datetime as dt, os, struct

MODE_CHARGING = 3
REASON = {"driver override": 0, "lane marking lost": 1, "camera blocked": 2}


def _slog_ts(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%b %d %H:%M:%S.") + f"{int((t % 1) * 1000):03d}"
def _logcat_ts(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%m-%d %H:%M:%S.") + f"{int((t % 1) * 1000):03d}"


def live_messages(veh, start_s, seconds, can_hz=10):
    import cantools
    from .protocols import isotp, uds, obd, someip, vhal
    P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocols")
    db = cantools.database.load_file(os.path.join(P, "vehicle.dbc")); V = vhal.Vhal()
    msgs = {n: db.get_message_by_name(n) for n in ("VehDyn", "BmsStatus", "Thermal", "Climate", "Charger")}
    sess = 0
    pad = lambda f: f.ljust(8, b"\xAA")
    # Several independently recorded events may share a one-second simulation
    # tick.  Keep all of them; a dict comprehension silently discarded all but
    # the last one.
    events = {}
    for ts, ty, sv, de in veh.events:
        tick = int(ts - veh.t0)
        if start_s <= tick < start_s + seconds:
            events.setdefault(tick, []).append((ty, de))
    first = True
    for i in range(start_s, min(veh.n, start_s + seconds)):
        if not veh.ign[i] and veh.mode[i] != MODE_CHARGING and i % 60: continue
        t = float(veh.t0 + i)
        if first:                                                         # service discovery at the start of the stream
            first = False; sess += 1
            yield t, "someip", someip.build_sd_offer([(0x4A11, 1, 1, 0), (0x4B01, 1, 1, 0)], session=sess)
        chg_kw = float(max(0.0, -veh.power[i])); charging = veh.mode[i] == MODE_CHARGING
        sig = {"VehDyn": {"Speed": float(veh.speed[i]), "Gear": 3 if veh.speed[i] > 1 else 0},
               "BmsStatus": {"Soc": float(veh.soc[i]), "PackVolt": 380.0 + float(veh.soc[i]) * 0.4, "PackCurrent": float(veh.power[i]) / 0.38, "BattTemp": float(veh.batt[i]),
                             "CellTempMax": float(veh.batt[i]) + 1.5, "CellTempMin": float(veh.batt[i]) - 1.0, "CellVoltMin": 3.7},
               "Thermal": {"CoolantFlow": float(veh.flow[i]), "PumpRpm": 1800.0 + 40 * float(veh.flow[i]), "MotorTemp": float(veh.motor[i])},
               "Climate": {"AmbTemp": float(veh.amb[i]), "HvacPower": float(veh.hvac[i])},
               "Charger": {"ChgKw": min(chg_kw, 600.0), "ChgState": 1 if charging else 0}}
        for k in range(can_hz):
            for name in ("VehDyn", "BmsStatus", "Thermal", "Climate", "Charger"):
                m = msgs[name]; yield t + k / can_hz, "can", (m.frame_id, bytes(m.encode(sig[name], scaling=True, strict=False)), m.length > 8)
        if i % 5 == 0:                                                    # Android VHAL property changes (native AOSP units)
            yield t + 0.01, "vhal", V.encode("PERF_ODOMETER", float(veh.odo[i]), ts=int(i * 1e9))
            yield t + 0.02, "vhal", V.encode("IGNITION_STATE", 4 if veh.ign[i] else 2, ts=int(i * 1e9))
            yield t + 0.03, "vhal", V.encode("ENV_OUTSIDE_TEMPERATURE", float(veh.amb[i]), ts=int(i * 1e9))
            yield t + 0.035, "vhal", V.encode("PERF_VEHICLE_SPEED", float(veh.speed[i]) / 3.6, ts=int(i * 1e9))
            yield t + 0.036, "vhal", V.encode("EV_BATTERY_LEVEL", float(veh.soc[i]) / 100.0 * 60000.0, ts=int(i * 1e9))
            yield t + 0.037, "vhal", V.encode("EV_CHARGE_STATE", 1 if charging else 3, ts=int(i * 1e9))
            yield t + 0.038, "vhal", V.encode("EV_BATTERY_INSTANTANEOUS_CHARGE_RATE", chg_kw * 1e6 if charging else 0.0, ts=int(i * 1e9))
        if i % 10 == 0:                                                   # SOME/IP ChargingService.ChargeState notification
            sess += 1; yield t + 0.04, "someip", someip.build(0x4B01, 0x8001, struct.pack(">Bf", 1 if charging else 0, chg_kw), session=sess, mtype=0x02)
        if i % 5 == 0:                                                    # OBD-II mode 01 (speed, ambient) from the gateway ECU
            for pid, val in ((0x0D, float(veh.speed[i])), (0x46, float(veh.amb[i]))):
                for fr in isotp.encode(obd.encode(pid, val)): yield t + 0.05, "can", (0x7E8, pad(fr), False)
        if veh.flow[i] < 8 and i % 20 == 0:                               # QNX thermal manager (slog2info, severity 3 = warning)
            yield t + 0.05, "log", f"{_slog_ts(t + 0.05)} thermal_mgr 3 0 coolant pump flow low ({veh.flow[i]:.1f} L/min) rpm={1800 + 40 * veh.flow[i]:.0f}"
        if veh.speed[i] > 60 and i % 97 == 0:                             # ADAS camera pipeline metadata (kept minimal)
            yield t + 0.06, "cam", {"cam": "front", "frame": i, "lane_conf": 0.9, "blockage": 0.02}
        for ty, de in events.get(i, ()):
            if ty == "ADAS_DISENGAGE":
                sess += 1; yield t + 0.5, "someip", someip.build(0x4A11, 0x8002, struct.pack(">BH", REASON.get(de, 3), int(veh.speed[i])), session=sess, mtype=0x02)
            elif ty == "CHARGE_START":
                sess += 1; kind, kw = de.split()[0], float(de.split()[1]); yield t + 0.5, "someip", someip.build(0x4B01, 0x8002, struct.pack(">Bf", 1 if kind == "DC" else 0, kw), session=sess, mtype=0x02)
            elif ty == "DTC_SET":
                b0, b1 = uds.parse_dtc("P1A00"); resp = uds.encode_dtc_response([(b0, b1, 0x01, 0x09), (0x0A, 0x94, 0x00, 0x04)])      # confirmed P1A00 + a pending second DTC
                for fr in isotp.encode(resp): yield t + 0.5, "can", (0x7E8, pad(fr), False)
            elif ty == "WARN_BATT_OVERTEMP":
                yield t + 0.5, "log", f"{_logcat_ts(t + 0.5)}    88    88 E ThermalHmi: Battery temperature warning raised (>52C)"

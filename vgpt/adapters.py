"""In-car protocol adapters (stdlib only). Each decodes ONE native wire format (see vgpt/protocols/) into normalised records; the ingest daemon owns the
only write path to the store. The adapters are READ-ONLY consumers of the vehicle networks: nothing here can transmit onto a bus.

  CanAdapter    raw CAN / CAN-FD frames  -> DBC signals (vehicle.dbc); ISO-TP + UDS DTC responses and OBD-II mode 01 on 0x7E8..0x7EF; unknown ids -> bus-health monitor
  SomeIpAdapter SOME/IP notifications (service catalog) + SOME/IP-SD offers  -> signals + events (ADAS state, charging)
  VhalAdapter   Android VHAL VehiclePropValue (real AOSP property ids/units) -> signals
  LogAdapter    QNX slog2info / Android logcat / syslog lines  -> log rows (WARN and above) + events
  CameraAdapter ADAS camera pipeline metadata (NO pixels stored)   [kept minimal on purpose - see README]
  NmeaAdapter   GNSS receiver: continuous NMEA 0183 stream (RMC + GGA) -> latitude / longitude / course / speed / HDOP (1 Hz while the car is awake)
Output record kinds:  ("sig", name, value) | ("event", type, severity, detail) | ("log", src, text)
Production transports: CAN via SocketCAN (read-only filters) / PEAK; SOME/IP via a vsomeip subscriber or an Ethernet tap; VHAL via a Binder client; QNX via slog2info;
Android via logcat; UDS DTCs via a read-only snoop of the diagnostic CAN ids (or DoIP)."""
import os, struct
import collections

from .protocols import dbc as _dbc, isotp as _isotp, uds as _uds, obd as _obd, someip as _someip, vhal as _vhal, logs as _logs, nmea as _nmea, canframes as _canframes

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocols")
DBC = _dbc.Database.load(os.path.join(_P, "vehicle.dbc"))
SIGNAL_MAP = {"Speed": "speed", "Gear": "gear", "Soc": "soc", "PackVolt": "pack_v", "PackCurrent": "pack_a", "BattTemp": "batt_temp", "CoolantFlow": "coolant_flow", "PumpRpm": "pump_rpm",
              "MotorTemp": "motor_temp", "AmbTemp": "amb_temp", "HvacPower": "hvac_w", "ChgKw": "chg_kw"}
DIAG_IDS = range(0x7E8, 0x7F0)                                             # ECU response ids of the standard (11-bit) diagnostic addressing


class CanAdapter:
    """CAN / CAN-FD. payload = (can_id, data, fd). One ISO-TP reassembler per diagnostic response id."""
    def __init__(self, db=DBC, health=None):
        self.db, self.tp, self.health, self.dtc_seen = db, {}, health, {}
        self.frames = self.unknown = self.tp_errors = 0; self.error_frames = collections.Counter()

    def decode(self, can_id, data=None, fd=False, extended=None, rx_time=None, session=None):
        if isinstance(can_id, _canframes.Frame):
            frame = can_id
            if frame.error_frame:
                self.error_frames[frame.error_class] += 1
                if self.health is not None and hasattr(self.health, "record_can_error"): self.health.record_can_error(frame.error_class, bytes(frame.data))
                return []
            can_id, data, fd, extended, rx_time, session = frame.can_id, frame.data, frame.fd, frame.extended, frame.ts, frame.channel or None
        self.frames += 1
        if can_id in DIAG_IDS and extended is not True: return self._diag(can_id, data, rx_time, session)
        vals = self.db.decode(can_id, data, extended=extended)
        if vals is None:
            self.unknown += 1
            if self.health is not None: return [("canraw", can_id, len(data))]
            return []
        out = [("sig", SIGNAL_MAP[k], v) for k, v in vals.items() if k in SIGNAL_MAP]
        if "ChgState" in vals:
            meta = self.db.decode_with_metadata(can_id, data, extended=extended).get("ChgState")
            if meta.enum in ("charging", "idle", "full", "fault"): out.append(("sig", "chg_state", 1.0 if meta.enum == "charging" else 0.0))
        return out

    def _diag(self, can_id, data, rx_time=None, session=None):
        r = self.tp.setdefault(can_id, _isotp.Reassembler()); before = r.errors
        msg = r.feed(data, now=rx_time, session=session); self.tp_errors += r.errors - before
        if msg is None: return []
        out = []
        o = _obd.decode(msg)
        if o and "error" not in o and o.get("value") is not None: return [("sig", "obd_" + o["name"], float(o["value"]))]
        try: u = _uds.decode(msg)
        except _uds.UdsError: self.tp_errors += 1; return []
        if u and u["type"] == "dtc":
            for d in u["dtcs"]:
                key = (can_id, d["code"]); st = d["status"] & 0x09                                       # testFailed or confirmedDTC
                if st and self.dtc_seen.get(key) != d["status"]:
                    out.append(("event", "DTC_SET", "warn", f"{d['code']} (status 0x{d['status']:02X}: {', '.join(d['status_flags'][:2])})"))
                elif not st and self.dtc_seen.get(key, 0) & 0x09:
                    out.append(("event", "DTC_CLEAR", "info", f"{d['code']} cleared (status 0x{d['status']:02X})"))
                self.dtc_seen[key] = d["status"]
        return out


class SomeIpAdapter:
    def __init__(self, catalog=None):
        self.cat = catalog or _someip.ServiceCatalog.load(os.path.join(_P, "services.json"))
        self.offered, self.errors, self.remainder = {}, 0, b""

    def decode(self, raw):
        out = []
        try: msgs, rest = _someip.split(self.remainder + bytes(raw)); self.remainder = rest
        except _someip.SomeIpError: self.errors += 1; return []
        for m in msgs:
            try:
                if m["service"] == _someip.SD_SERVICE and m["method"] == _someip.SD_METHOD:
                    for e in _someip.parse_sd(m["payload"])["entries"]:
                        if e["type_name"] == "OfferService": self.offered[(e["service"], e["instance"])] = e["ttl"]
                        elif e["type_name"] == "StopOfferService": self.offered.pop((e["service"], e["instance"]), None)
                    continue
                if m["type_name"] != "NOTIFICATION" or m["rc"] != 0: continue
                d = self.cat.decode(m["service"], m["method"], m["payload"])
            except _someip.SomeIpError: self.errors += 1; continue
            if d is None: continue
            v = d["values"]
            if d["service"] == "AdasState" and d["event"] == "Disengaged": out.append(("event", "ADAS_DISENGAGE", "info", v["reason_text"]))
            elif d["service"] == "ChargingService" and d["event"] == "ChargeState": out += [("sig", "chg_state", 1.0 if v["state"] == 1 else 0.0), ("sig", "chg_kw", float(v["kw"]))]
            elif d["service"] == "ChargingService" and d["event"] == "SessionStarted": out.append(("event", "CHARGE_START", "info", f"{v['kind_text']} {v['kw']:.0f} kW"))
        return out


class VhalAdapter:
    def __init__(self): self.v = _vhal.Vhal()
    def decode(self, msg):
        return [r for r in self.v.decode(msg) if r[0] == "sig"]


class LogAdapter:
    """slog2info / logcat / syslog. Only WARN and above are stored (the volume would otherwise swamp a small flash budget)."""
    def decode(self, line):
        r = _logs.parse(line)
        if r is None or r["severity"] > 4: return []
        out = [("log", r["source"][:24], f"{r['level'].upper()} {r['message']}", r.get("ts_text"))]
        if r["severity"] <= 3 and "battery temperature warning" in r["message"].lower(): out.append(("event", "WARN_BATT_OVERTEMP", "critical", r["message"]))
        return out


class CameraAdapter:
    """Camera feed. payload either carries a luma plane {"cam", "t", "w", "h", "y": bytes} -> analysed by vgpt.camera (numpy), or already-computed ISP/NPU
    metadata {"lane_conf", "blockage"} (many ADAS modules publish that). Only metadata (+ a 64x48 thumbnail at fault events) leaves this adapter."""
    def __init__(self):
        self.monitors, self.block_run = {}, 0

    def decode(self, msg):
        if "y" in msg:
            import numpy as np
            from .camera import CameraMonitor
            mon = self.monitors.setdefault(msg.get("cam", "front"), CameraMonitor(name=msg.get("cam", "front")))
            y = np.frombuffer(msg["y"], np.uint8).reshape(msg["h"], msg["w"])
            self.last = mon
            return mon.process(float(msg.get("t", 0.0)), y)[1]
        out = [("sig", "lane_conf", msg["lane_conf"]), ("sig", "cam_block", msg["blockage"])]
        self.block_run = self.block_run + 1 if msg["blockage"] > 0.5 else 0
        if self.block_run == 3: out.append(("event", "CAMERA_BLOCKED", "info", f"front camera blockage {msg['blockage']:.2f}"))
        return out


class NmeaAdapter:
    """GNSS stream: RMC gives position, speed and course; GGA adds HDOP. Invalid fixes (status V / quality 0) are dropped, bad checksums counted."""
    def __init__(self): self.bad = 0; self.fixes = 0

    def decode(self, line):
        try: m = _nmea.decode(line)
        except _nmea.NmeaError: self.bad += 1; return []
        if not m: return []
        if m["type"] == "RMC":
            if not m["valid"]: return [("sig", "gps_valid", 0.0)]
            self.fixes += 1; out = [("sig_ts", "lat", m["lat"], m["ts"]), ("sig_ts", "lon", m["lon"], m["ts"]), ("sig_ts", "gspeed", m["speed_kmh"], m["ts"]), ("sig_ts", "gps_valid", 1.0, m["ts"])]
            if m.get("course") is not None: out.append(("sig_ts", "hdg", m["course"], m["ts"]))
            return out
        if m["type"] == "GGA":
            if m.get("quality", 0) <= 0: return [("sig", "gps_valid", 0.0)]
            if m.get("hdop") is not None: return [("sig", "hdop", m["hdop"])]
        return []


class Router:
    """Dispatch (bus, payload) from any source to its adapter. Malformed input is dropped and counted, never fatal."""
    def __init__(self, health=None):
        self.can, self.someip, self.vhal, self.log, self.cam, self.gnss = CanAdapter(health=health), SomeIpAdapter(), VhalAdapter(), LogAdapter(), CameraAdapter(), NmeaAdapter()
        self.dropped = 0

    def decode(self, bus, payload):
        try:
            if bus == "can":
                if isinstance(payload, _canframes.Frame): return self.can.decode(payload)
                if not isinstance(payload, (tuple, list)) or len(payload) not in (2, 3, 4, 5, 6): raise TypeError("CAN payload")
                return self.can.decode(*payload)
            if bus == "someip": return self.someip.decode(payload)
            if bus == "vhal":
                if not isinstance(payload, dict): raise TypeError("VHAL payload")
                return self.vhal.decode(payload)
            if bus == "log":
                if not isinstance(payload, str): raise TypeError("log payload")
                return self.log.decode(payload)
            if bus == "cam":
                if not isinstance(payload, dict): raise TypeError("camera payload")
                return self.cam.decode(payload)
            if bus == "gnss":
                if not isinstance(payload, str): raise TypeError("GNSS payload")
                return self.gnss.decode(payload)
        except (struct.error, KeyError, ValueError, TypeError, IndexError, AttributeError): self.dropped += 1; return []
        return []

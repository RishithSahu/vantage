"""Collectors: the transports that bring a vehicle's raw traffic to the runtime.  A vehicle integration is "point a collector at the thing that already carries the bytes".

    FileTail          follows a file as it grows (candump -L, Vector ASC, JSON lines, NMEA, logs) - the recorded-drive and "tail a log" integration; `follow=False` replays it once
    SocketCanCollector  a Linux SocketCAN interface (can0 / vcan0, CAN-FD aware): the vehicle's CAN bus through the kernel's CAN socket family (AF_CAN: a bus, not a network).  The ONLY socket this program can open
                      (vgpt/offline.py refuses every other); it needs a real or virtual CAN interface and is NOT exercised by this repository's tests against one
    CollectorSet      starts / stops / reports several collectors that feed one sink (Runtime.feed)

There is no network collector: no UDP, no TCP, no broker, no WebSocket (the product is fully offline, `docs/OFFLINE_DESIGN.md`).  A vehicle's data comes from a bus, a vehicle API in the same device (on Android: CarPropertyManager)
or a file.  The record below is the neutral shape a source hands to the runtime in this process:  {"ts": 1792364400.2, "bus": "can", "payload": {"id": 291, "data": "0011223344556677", "fd": false, "ext": false, "ch": "can0"}}
  bus "can"      payload {"id", "data" (hex), "fd", "ext", "ch"}            bus "someip"  payload {"hex"} or {"base64"}      bus "vhal"  payload {the VehiclePropValue fields}
  bus "log"      payload "<line>"                                           bus "cam"     payload {feature dict}             bus "gnss"  payload "$GPRMC,..."
  bus "table"    payload {"vehicle_speed": 55.2, "Vehicle.Powertrain.TractionBattery.StateOfCharge": 81}   named readings (a file of VSS / OpenXC rows, a vehicle API in this device): the profile maps each name
All collectors are listen-only: nothing here can transmit on a bus."""
import base64
import json
import os
import socket
import struct
import sys
import threading
import time

from ..protocols import canframes as cf

BUSES = ("can", "someip", "vhal", "log", "cam", "gnss", "table")


def encode_record(ts, bus, payload):
    if isinstance(payload, cf.Frame): payload = {"id": payload.can_id, "data": payload.data.hex(), "fd": payload.fd, "ext": payload.extended, "ch": payload.channel, **({"err": payload.error_class} if payload.error_frame else {})}
    elif isinstance(payload, (bytes, bytearray)): payload = {"hex": bytes(payload).hex()}
    elif isinstance(payload, dict): payload = {k: (v.hex() if isinstance(v, (bytes, bytearray)) else v) for k, v in payload.items()}           # a VHAL parcelable carries a byte field
    return {"ts": round(float(ts), 3), "bus": bus, "payload": payload}


def decode_record(obj):
    """JSON record -> (ts, bus, payload) ready for the router.  Raises ValueError for anything malformed."""
    if not isinstance(obj, dict) or "bus" not in obj or "payload" not in obj: raise ValueError("not a record")
    bus, p, ts = obj["bus"], obj["payload"], obj.get("ts")
    if bus not in BUSES: raise ValueError(f"unknown bus {bus!r}")
    if ts is None or isinstance(ts, bool) or not isinstance(ts, (int, float)) or ts != ts or ts in (float("inf"), float("-inf")): raise ValueError("bad timestamp")
    if bus == "can":
        if not isinstance(p, dict): raise ValueError("can payload must be an object")
        data = bytes.fromhex(p["data"]); ident = int(p["id"])
        return float(ts), bus, cf.Frame(float(ts), ident, data, extended=bool(p.get("ext")) or ident > 0x7FF, fd=bool(p.get("fd")) or len(data) > 8, channel=str(p.get("ch", "")), error_frame="err" in p, error_class=int(p.get("err", 0)))
    if bus == "someip":
        if isinstance(p, dict) and "hex" in p: return float(ts), bus, bytes.fromhex(p["hex"])
        if isinstance(p, dict) and "base64" in p: return float(ts), bus, base64.b64decode(p["base64"], validate=True)
        raise ValueError("someip payload must carry hex or base64")
    if bus in ("vhal", "cam", "table") and not isinstance(p, dict): raise ValueError(f"{bus} payload must be an object")
    if bus == "vhal" and isinstance(p.get("bytes"), str): p = dict(p, bytes=bytes.fromhex(p["bytes"]))
    if bus in ("log", "gnss") and not isinstance(p, str): raise ValueError(f"{bus} payload must be text")
    return float(ts), bus, p


class Collector(threading.Thread):
    kind = "collector"

    def __init__(self, sink, name=None, clock=time.time):
        super().__init__(daemon=True, name=name or self.kind); self.sink, self.clock = sink, clock; self.n = 0; self.errors = 0; self.last_ts = None; self.stop_flag = threading.Event(); self.error_text = ""

    def push(self, ts, bus, payload):
        self.sink(ts, bus, payload); self.n += 1; self.last_ts = ts

    def push_record(self, obj):
        try: ts, bus, payload = decode_record(obj)
        except (ValueError, KeyError, TypeError) as e: self.errors += 1; self.error_text = str(e)[:120]; return
        self.push(ts, bus, payload)

    def push_line(self, line):
        line = line.strip()
        if not line: return
        try: obj = json.loads(line)
        except ValueError: self.errors += 1; self.error_text = "not JSON"; return
        self.push_record(obj)

    def stop(self): self.stop_flag.set()
    def stats(self): return {"name": self.name, "kind": self.kind, "records": self.n, "errors": self.errors, "last_error": self.error_text, "last_ts": self.last_ts, "alive": self.is_alive()}


class FileTail(Collector):
    kind = "file"

    def __init__(self, sink, path, fmt="jsonl", bus=None, follow=True, clock=time.time, name=None, channel="", start_at_end=False):
        super().__init__(sink, name or f"file:{os.path.basename(path)}", clock); self.path, self.fmt, self.follow, self.channel = path, fmt, follow, channel; self.bus = bus or {"nmea": "gnss", "log": "log", "candump": "can", "asc": "can"}.get(fmt); self.start_at_end = start_at_end; self.done = threading.Event()

    def _line(self, line):
        if self.fmt == "jsonl": return self.push_line(line)
        line = line.rstrip("\r\n")
        if not line.strip(): return
        if self.fmt in ("candump", "asc"):
            f = cf.parse_candump(line) if self.fmt == "candump" else cf.parse_asc(line)
            if f is None: return
            if self.channel and not f.channel: f.channel = self.channel
            self.push(f.ts or self.clock(), "can", f)
        else: self.push(self.clock(), self.bus, line.strip())

    def run(self):
        while not self.stop_flag.is_set() and not os.path.exists(self.path):
            if not self.follow: self.errors += 1; self.error_text = "file not found"; self.done.set(); return
            time.sleep(0.2)
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            if self.start_at_end: fh.seek(0, os.SEEK_END)
            while not self.stop_flag.is_set():
                line = fh.readline()
                if line:
                    try: self._line(line)
                    except (ValueError, IndexError, KeyError, struct.error) as e: self.errors += 1; self.error_text = str(e)[:120]
                    continue
                if not self.follow: break
                time.sleep(0.1)
        self.done.set()


class ReceiveOnly:
    """What the SocketCAN collector holds instead of the raw socket: it can receive and be closed, nothing else.  There is no send, no setsockopt, no way back to the underlying socket from the collector's code
    (a Linux raw CAN socket cannot be shut down for writing, so the capability is withheld here and, in a vehicle, removed in the controller with `ip link set can0 type can listen-only on`)."""
    __slots__ = ("_s",)

    def __init__(self, sock): object.__setattr__(self, "_s", sock)
    def recv(self, n): return self._s.recv(n)
    def settimeout(self, t): self._s.settimeout(t)
    def close(self): self._s.close()
    def __setattr__(self, k, v): raise AttributeError("a receive-only socket cannot be changed")


def controller_listen_only(channel):
    """Whether the CAN controller behind `channel` is in listen-only mode (it then neither transmits nor acknowledges): True / False, or None where it cannot be told (not Linux, no `ip`, a virtual bus)."""
    import subprocess
    from .. import isolation
    try: out = subprocess.run(["ip", "-details", "link", "show", channel], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError): return None
    return isolation.listen_only(out)


class SocketCanCollector(Collector):
    """Linux SocketCAN (raw, CAN-FD enabled), RECEIVE-ONLY.  Not exercised by the tests in this repository against a real bus: it needs a CAN interface (real hardware, or `ip link add dev vcan0 type vcan`); the
    receive-only wrapper and the controller check are tested with a stand-in socket.  `require_listen_only=True` refuses to start unless the controller is in listen-only mode (the setting for a real vehicle)."""
    kind = "socketcan"

    def __init__(self, sink, channel="can0", clock=time.time, name=None, sock=None, require_listen_only=False):
        super().__init__(sink, name or f"socketcan:{channel}", clock); self.channel = channel; self.listen_only = controller_listen_only(channel) if sock is None else None
        if require_listen_only and self.listen_only is not True: raise OSError(f"{channel} is not in listen-only mode (ip link set {channel} type can listen-only on); refusing to attach to a vehicle bus")
        if sock is None:
            if not hasattr(socket, "AF_CAN"): raise OSError("SocketCAN needs Linux")
            sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW); sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1); sock.bind((channel,))
        self.sock = ReceiveOnly(sock); self.sock.settimeout(0.2)

    def run(self):
        while not self.stop_flag.is_set():
            try: data = self.sock.recv(72)
            except socket.timeout: continue
            except OSError: break
            try: f = cf.parse_socketcan_struct(data); f.channel = self.channel; self.push(self.clock(), "can", f)
            except (ValueError, struct.error) as e: self.errors += 1; self.error_text = str(e)[:120]
        self.sock.close()


class CollectorSet:
    def __init__(self, sink): self.sink, self.items = sink, []

    def add(self, c): self.items.append(c); return c
    def start(self):
        for c in self.items:
            if not c.is_alive(): c.start()
    def stop(self):
        for c in self.items: c.stop()
        for c in self.items: c.join(timeout=2.0)
    def stats(self): return [c.stats() for c in self.items]

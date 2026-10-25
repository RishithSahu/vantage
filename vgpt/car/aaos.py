"""An Android Automotive device as the car.

A real Android Automotive OS (a Cuttlefish virtual device, an emulator, a development board or a head unit reachable with `adb`) keeps its vehicle data in the Vehicle HAL.  This program reads every property
the device's car service can give (`cmd car_service get-property-value`), puts each value into the form the HAL really uses (a VehiclePropValue parcel, vgpt/protocols/vhalparcel.py) and writes it on the pipe, exactly
as the simulated cars do.  Vantage on the other end of the pipe is not told that this car is Android: it sees VHAL parcels, finds the standard properties and reads them (vgpt/pipe, vgpt/ingest).

    python vehiclegpt.py car-aaos --adb ~/cf/img/bin/adb --serial 0.0.0.0:6520 [--drive 90]

`--drive` also changes what the device holds, using the one thing a stock Cuttlefish Vehicle HAL lets a client set (writable properties such as the turn-signal switch): the changes travel the whole path.  The stock HAL keeps
speed, gear and the like at fixed values (they are read-only and the fake hardware does not move them), so a moving Android Automotive car needs a vehicle simulator behind the HAL (CARLA, or a replay) - not this program.
The car side only: nothing here reads the pipe, and the device is only ever asked (the injections are the stand-in for the vehicle's own ECUs)."""
import re
import subprocess
import time

from ..pipe.wire import LinkClosed
from .transmit import Transmitter
from .writer import Writer

LINE = re.compile(r"HalPropValue\{Property ID: (?P<name>[A-Z0-9_]+)\((?P<id>0x[0-9a-fA-F]+)\), Area ID: [A-Z0-9_]+\((?P<area>0x[0-9a-fA-F]+)\), ElapsedRealtimeNanos: (?P<ns>\d+), SystemStatus: (?P<status>[A-Z_]+)\(0x[0-9a-fA-F]+\).*?Value: (?P<value>.*)\}\s*$")
NUM = re.compile(r"^(?P<n>-?\d+(\.\d+)?(e[+-]?\d+)?)(\s+[A-Z_]+)?$", re.I)
ENUM = re.compile(r"^[A-Z0-9_]+\((?P<h>0x[0-9a-fA-F]+)\)$")
TAIL_AREA = re.compile(r"\(0x(?P<a>[0-9a-fA-F]+)\)")


def parse_value(text):
    """The textual value of a property -> (int32 list, float list, string): booleans and enums and integers are int32, decimals are float, anything else is a string."""
    t = text.strip()
    if t in ("TRUE", "FALSE"): return [1 if t == "TRUE" else 0], [], ""
    m = ENUM.match(t)
    if m: return [int(m["h"], 16)], [], ""
    m = NUM.match(t)
    if m:
        s = m["n"]
        return ([], [float(s)], "") if ("." in s or "e" in s.lower()) else ([int(s)], [], "")
    if "," in t and all(NUM.match(x.strip()) for x in t.split(",")):
        xs = [x.strip() for x in t.split(",")]
        return ([], [float(x.split()[0]) for x in xs], "") if any("." in x for x in xs) else ([int(x.split()[0]) for x in xs], [], "")
    return [], [], t


def parse_dump(text):
    """[{prop, area, timestamp, status, int32, float, int64, bytes, string}] from the output of `cmd car_service get-property-value`; lines that are not property values (the car service says which ones it could not read) are skipped."""
    out = []
    for line in text.splitlines():
        m = LINE.search(line)
        if not m or m["status"] != "AVAILABLE": continue
        i32, fl, s = parse_value(m["value"]); i64 = []
        if any(not -2 ** 31 <= x < 2 ** 31 for x in i32): i32, i64 = [], list(i32)                          # a 64-bit property (a timestamp, a large counter)
        out.append({"prop": int(m["id"], 16), "area": int(m["area"], 16), "timestamp": int(m["ns"]), "status": 0, "int32": i32, "float": fl, "int64": i64, "bytes": b"", "string": s, "name": m["name"]})
    return out


class Device:
    """One Android Automotive device reached with adb."""
    def __init__(self, adb="adb", serial=None, timeout=60):
        self.adb = adb; self.serial = serial; self.timeout = timeout

    def shell(self, *args, timeout=None):
        cmd = [self.adb] + (["-s", self.serial] if self.serial else []) + ["shell", *args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout or self.timeout, errors="replace")
        return r.stdout.replace("\r", "")

    def connect(self):
        if self.serial and ":" in self.serial: subprocess.run([self.adb, "connect", self.serial], capture_output=True, timeout=30)

    def booted(self): return self.shell("getprop", "sys.boot_completed").strip() == "1"
    def properties(self): return parse_dump(self.shell("cmd", "car_service", "get-property-value"))
    def set(self, prop, value, area=0): return self.shell("cmd", "car_service", "set-property-value", prop, str(area), str(value))


def drive_script(t):
    """What the device is asked to hold at second t: the turn signal cycles none -> right -> left every few seconds (the one writable vehicle property of a stock device)."""
    return {"TURN_SIGNAL_SWITCH": int(t // 6) % 3}


def run(fd, adb="adb", serial=None, seconds=None, period=1.0, drive=False, log=print):
    """Read the device and write what it holds to the pipe every `period` seconds, until `seconds` have passed (None: until the pipe closes)."""
    dev = Device(adb, serial); dev.connect(); w = Writer(fd); tx = Transmitter(w, None, seed=17); sent = 0; t0 = time.time()
    w.control(proto=1, kind="vehicle data link", clock="calendar")
    end = None if seconds is None else t0 + seconds
    try:
        while end is None or time.time() < end:
            t = time.time(); tick = time.time() - t0
            if drive:
                try:
                    for name, value in drive_script(tick).items(): dev.set(name, value)
                except (subprocess.TimeoutExpired, OSError): pass
            try: props = dev.properties()
            except (subprocess.TimeoutExpired, OSError) as e: log(f"the device did not answer: {e}"); time.sleep(period); continue
            now = time.time()
            for p in props:
                tx.send(now, "vhal", {k: v for k, v in p.items() if k != "name"}); sent += 1
            log(f"\r{sent} property values written ({len(props)} per pass)", end="") if log is print else log(f"{sent} property values written")
            time.sleep(max(0.0, period - (time.time() - t)))
    except LinkClosed: log("Vantage closed the link")
    finally: w.close()
    return sent

"""From what a vehicle's buses carry to bytes on the pipe.

The car's simulation produces traffic exactly as the vehicle would put it on its buses (vgpt/world/emit.py walks the profile in reverse: CAN / CAN-FD frames, SOME/IP messages, Android VHAL values, log lines, NMEA
sentences, camera data).  The transmitter gives each wire a channel number of the car's own choosing and writes each message in the form that medium really has:

    CAN / CAN-FD   a Linux SocketCAN frame (struct can_frame, 16 bytes / struct canfd_frame, 72 bytes) - what a program reading can0 gets
    SOME/IP        the message as it travels on automotive Ethernet (16-byte header + payload)
    Android VHAL   a VehiclePropValue (vgpt/protocols/vhalparcel.py)
    log, GNSS      one text line (QNX slog2info, logcat, syslog; an NMEA 0183 sentence)
    camera         a JPEG picture of the forward camera, or - for an ADAS camera that only reports - a JSON record of its measurements
    named values   a JSON object

Channel numbers are drawn at random from a seed: nothing about a channel's number means anything to Vantage, which is the point (it has to work out what each wire is)."""
import json
import random

from ..protocols import canframes as cf, doip as _doip, isotp, vhalparcel

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def make_vin(seed):
    """A 17-character vehicle identification number derived from a seed (a test car's, not a real vehicle's: the first characters say so)."""
    r = random.Random(str(seed)); return "VGP" + "".join(r.choice(VIN_CHARS) for _ in range(14))


class Transmitter:
    def __init__(self, writer, profile=None, seed=1, jpeg_quality=60):
        self.w = writer; self.p = profile; self.rng = random.Random(seed); self.pool = self.rng.sample(range(1, 4000), 64); self.chan = {}; self.counts = {}; self.errors = 0; self.quality = jpeg_quality
        self.primary_can = next((s.raw.get("channel", "") for s in profile.sources.values() if s.type == "can"), "") if profile is not None else ""

    def channel(self, key):
        c = self.chan.get(key)
        if c is None: c = self.chan[key] = self.pool[len(self.chan) % len(self.pool)]
        return c

    def describe(self):
        return {str(k): v for k, v in self.chan.items()}

    def send(self, ts, bus, payload):
        """One message of the vehicle's traffic -> one record on the pipe."""
        try:
            if bus == "can":
                chan = payload.channel or self.primary_can; self._out(ts, ("can", chan), cf.pack_socketcan(payload), "can")
            elif bus == "someip": self._out(ts, ("eth", "someip"), bytes(payload), "someip")
            elif bus == "vhal": self._out(ts, ("hal", "vhal"), vhalparcel.pack(payload), "vhal")
            elif bus == "lin": self._out(ts, ("lin", ""), bytes(payload), "lin")
            elif bus == "flexray": self._out(ts, ("fr", ""), bytes(payload), "flexray")
            elif bus == "doip": self._out(ts, ("eth", "doip"), bytes(payload), "doip")
            elif bus in ("log", "gnss"): self._out(ts, (bus, ""), (str(payload) + "\n").encode("utf-8"), bus)
            elif bus == "cam":
                if isinstance(payload, (bytes, bytearray)): self._out(ts, ("cam", "video"), bytes(payload), "camera")
                elif "y" in payload: self._out(ts, ("cam", "video"), self._jpeg(payload), "camera")
                else: self._out(ts, ("cam", "report"), json.dumps(payload, separators=(",", ":")).encode(), "camera report")
            elif bus == "table": self._out(ts, ("table", ""), json.dumps(payload, separators=(",", ":")).encode(), "values")
        except OSError: raise
        except (ValueError, TypeError, KeyError): self.errors += 1

    def jpeg(self, bgr_or_gray):
        try:
            import cv2
        except ImportError: return b""
        ok, jb = cv2.imencode(".jpg", bgr_or_gray, [cv2.IMWRITE_JPEG_QUALITY, self.quality]); return jb.tobytes() if ok else b""

    def _jpeg(self, msg):
        import numpy as np
        return self.jpeg(np.frombuffer(msg["y"], np.uint8).reshape(msg["h"], msg["w"]))

    def send_picture(self, ts, jpeg_bytes):
        if jpeg_bytes: self._out(ts, ("cam", "video"), jpeg_bytes, "camera")

    def _out(self, ts, key, data, label):
        self.w.send(ts, self.channel(key), data); self.counts[label] = self.counts.get(label, 0) + 1

    # ------------------------------------------------------------------------------------------------ identity the car announces about itself
    def announce_vin_doip(self, ts, vin, tester=0x0E80, entity=0x1001, eid="0050c2a1b2c3", gid="0050c2a1b2c3"):
        """The VIN the way a DoIP entity says it: the vehicle announcement (UDP) and the answer to ReadDataByIdentifier F190 (TCP)."""
        self.send(ts, "doip", _doip.encode("vehicle_announcement", vin=vin, logical_address=entity, eid=eid, gid=gid, further_action=0))
        self.send(ts + 0.001, "doip", _doip.encode("diagnostic_message", source_address=tester, target_address=entity, data=b"\x22\xf1\x90"))
        self.send(ts + 0.002, "doip", _doip.encode("diagnostic_message", source_address=entity, target_address=tester, data=b"\x62\xf1\x90" + vin.encode("ascii")))

    def announce_vin(self, ts, vin, uds=True, vhal=False):
        """The VIN the way the vehicle's own interfaces carry it: a UDS ReadDataByIdentifier answer (0x62 F190, ISO-TP over CAN) and/or the Android INFO_VIN property."""
        if uds:
            for fr in isotp.encode(b"\x62\xf1\x90" + vin.encode("ascii")): self.send(ts, "can", cf.Frame(ts, 0x7E8, fr.ljust(8, b"\xAA"), channel=""))
        if vhal: self.send(ts, "vhal", {"prop": 0x11100100, "area": 0, "timestamp": int(ts * 1e9), "status": 0, "int32": [], "float": [], "int64": [], "bytes": b"", "string": vin})

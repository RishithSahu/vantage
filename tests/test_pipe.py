"""The link between the car and Vantage: a one-way pipe that carries whatever the car's buses carry, and a receiver that works out what it is.

Layers: the wire format (records, resynchronisation, time bases); the content sniffer (what form is this block); the whole path with every car model - the car writes at the moment its buses would, Vantage
recognises the vehicle from the traffic alone (its decoder file, or the standards when it has none), stores it in a database of its own, and follows a change of car.  Vantage is told nothing about the car:
channel numbers are drawn at random, and nothing in the receiver knows a make or a protocol by name."""
import json
import os
import re
import shutil
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.car import models as CM, writer as CW  # noqa: E402
from vgpt.car.car import Car  # noqa: E402
from vgpt.pipe import receiver, sniff, wire  # noqa: E402
from vgpt.protocols import canframes as cf, nmea, vhalparcel  # noqa: E402
from vgpt.vehicle.profile import Profile  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- the wire
class Wire(unittest.TestCase):
    def stream(self, n=40):
        return b"".join(wire.encode(1_700_000_000.0 + i * 0.01, i % 5 + 1, bytes([i % 256]) * (i % 17 + 1)) for i in range(n))

    def test_records_survive_any_split(self):
        data = self.stream(); whole = wire.Parser().feed(data)
        for step in (1, 3, 7, 64, 1000):
            p = wire.Parser(); got = []
            for i in range(0, len(data), step): got += p.feed(data[i:i + step])
            self.assertEqual([(r.ts, r.channel, r.payload) for r in got], [(r.ts, r.channel, r.payload) for r in whole], step)
        self.assertEqual(len(whole), 40); self.assertAlmostEqual(whole[3].ts, 1_700_000_000.03, places=5)

    def test_garbage_and_damage_are_skipped_and_counted_never_guessed(self):
        recs = [wire.encode(1.0 + i, 9, b"payload %d" % i) for i in range(10)]
        bad = bytearray(recs[4]); bad[-6] ^= 0xFF                                                 # a flipped bit in the middle of a record
        data = b"".join(recs[:4]) + b"\x00\x13garbage\xa7" + bytes(bad) + b"".join(recs[5:])
        p = wire.Parser(); got = p.feed(data)
        self.assertEqual([r.payload for r in got], [b"payload %d" % i for i in (0, 1, 2, 3, 5, 6, 7, 8, 9)])
        self.assertEqual(p.bad_crc, 1); self.assertGreater(p.skipped, 0)

    def test_joining_in_the_middle_of_a_stream_finds_the_next_good_record(self):
        data = self.stream(60); cut = len(data) // 3 + 5; p = wire.Parser(); got = p.feed(data[cut:])
        self.assertGreaterEqual(len(got), 38); self.assertTrue(all(r.payload == bytes([i % 256]) * (i % 17 + 1) for r, i in zip(got, range(60 - len(got), 60))))

    def test_the_format_promises_nothing_about_the_content(self):
        for payload in (b"", b"\x00" * 100, os.urandom(500), "$GPGGA,x".encode(), struct.pack("<I", 0x123)):
            r = wire.Parser().feed(wire.encode(None, 0xFFFE, payload)); self.assertEqual(r[0].payload, payload); self.assertIsNone(r[0].ts)
        with self.assertRaises(ValueError): wire.encode(1.0, 1, b"x" * ((1 << 20) + 1))
        with self.assertRaises(ValueError): wire.encode(1.0, 70000, b"x")

    def test_a_control_record_is_a_hint_only(self):
        r = wire.Parser().feed(wire.encode(None, wire.CONTROL, json.dumps({"proto": 1, "vin": "X"}).encode()))[0]
        self.assertEqual(r.control(), {"proto": 1, "vin": "X"}); self.assertIsNone(wire.Record(1.0, 5, b"{}").control())
        self.assertIsNone(wire.Record(None, wire.CONTROL, b"not json").control())

    def test_the_senders_clock_is_read_as_a_calendar_or_as_relative(self):
        self.assertEqual(wire.time_base(1_700_000_000.0, 123.0), (0.0, "calendar"))
        off, kind = wire.time_base(5.0, 1_800_000_000.0); self.assertEqual(kind, "relative"); self.assertAlmostEqual(5.0 + off, 1_800_000_000.0)
        self.assertEqual(wire.time_base(None, 9.0), (0.0, "arrival"))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- the sniffer
class Sniff(unittest.TestCase):
    def kinds(self):
        can = cf.pack_socketcan(cf.Frame(1.0, 0x1A0, bytes(range(8)))); fd = cf.pack_socketcan(cf.Frame(1.0, 0x1A0, bytes(range(12)), fd=True, brs=True)); ext = cf.pack_socketcan(cf.Frame(1.0, 0x18FEF100, bytes(8), extended=True))
        someip = struct.pack(">HHIHHBBBB", 0x1234, 0x8001, 12, 0, 1, 1, 1, 0x02, 0) + b"abcd"
        vh = vhalparcel.pack({"prop": 0x11600207, "area": 0, "timestamp": 5, "status": 0, "float": [12.5]})
        gga = nmea.encode_gga(1_700_000_000.0, 48.1, 11.5)
        return {"can": [can, fd, ext], "someip": [someip], "vhal": [vh], "image": [b"\xff\xd8\xff\xe0" + b"\x00" * 30, b"\x89PNG\r\n\x1a\n" + b"\x00" * 30], "nmea": [gga.encode()],
                "json": [b'{"speed": 12.5, "rpm": 800}'], "log": [b"<14>Oct 21 10:00:00 tcu app[123]: engine started\n", b"10-21 10:00:00.123  1234  5678 I Vehicle: ignition on\n"], "text": [b"just some words here"], "unknown": [b"\x00\x01\x02\xff\xfe" * 7, b""]}

    def test_every_form_is_recognised_from_its_structure(self):
        for kind, blocks in self.kinds().items():
            for b in blocks: self.assertEqual(sniff.classify(b), kind, (kind, b[:20]))

    def test_a_block_the_sniffer_cannot_read_is_unknown_not_misread(self):
        for b in (os.urandom(16), os.urandom(72), os.urandom(33), b"\x00" * 16 + b"\x01"):
            k = sniff.classify(b); self.assertNotEqual(k, "someip", b); self.assertIn(k, ("unknown", "can"))                 # random 16/72 bytes may happen to be a legal frame; never SOME/IP, never text

    def test_a_channel_is_read_the_way_most_of_its_blocks_read(self):
        s = sniff.ChannelSniffer()
        for i in range(30): s.observe(cf.pack_socketcan(cf.Frame(float(i), 0x100 + i % 5, bytes([i]) * 8)))
        s.observe(b"\x00\x01\x02\xff\xfe" * 7); k, share, n = s.verdict()
        self.assertEqual(k, "can"); self.assertGreater(share, 0.9); self.assertTrue(s.settled())

    def test_blocks_become_what_the_decoders_take(self):
        f = sniff.to_bus("can", cf.pack_socketcan(cf.Frame(1.0, 0x123, bytes(8))), 5.0)[0]; self.assertEqual((f[0], f[1].can_id, f[1].ts), ("can", 0x123, 5.0))
        self.assertEqual(sniff.to_bus("log", b"line one\nline two\n", 1.0), [("log", "line one"), ("log", "line two")])
        self.assertEqual(sniff.to_bus("json", b'{"a": 1, "b": "x", "c": 2.5}', 1.0), [("table", {"a": 1, "c": 2.5})])
        self.assertEqual(sniff.to_bus("can", b"short", 1.0), [])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- the whole path
def model_dirs(): return {m["id"]: m for m in CM.catalogue()}


class Bench:
    """A Vantage service on a pipe and a car on the other end of it."""
    def __init__(self, test, roots=None, listen_s=5.0):
        self.home = tempfile.mkdtemp(prefix="vantage_home_"); test.addCleanup(shutil.rmtree, self.home, True)
        self.svc = receiver.LinkService(self.home, roots=roots if roots is not None else [CM.BUILTIN], listen_s=listen_s); test.addCleanup(self.svc.stop)

    def car(self, model_id, seconds=40.0, speed=60.0, seed=7, channel_seed=None, wait=True):
        meta = model_dirs()[model_id]; r, w = os.pipe(); self.svc.attach(r); c = Car(meta["dir"], CW.Writer(w), speed=speed, seed=seed, channel_seed=channel_seed); c.start()
        if wait: self.finish(c, seconds)
        return c

    def finish(self, car, seconds):
        end = time.monotonic() + 120
        while car.drive.t - car.drive.t0 < seconds and car.is_alive() and time.monotonic() < end: time.sleep(0.1)
        car.stop(); self.svc.join(20); self.svc.drain(30)

    def db(self):
        v = (self.svc.status().get("vehicle") or {}).get("db")
        return sqlite3.connect("file:" + v.replace("\\", "/") + "?mode=ro", uri=True) if v else None


class WholePath(unittest.TestCase):
    def test_every_car_model_is_recognised_from_its_traffic_alone(self):
        for mid, meta in sorted(model_dirs().items()):
            with self.subTest(model=mid):
                b = Bench(self); b.car(mid, seconds=40.0, channel_seed=hash(mid) & 0xFFFF); st = b.svc.status(); v = st.get("vehicle")
                self.assertIsNotNone(v, f"{mid}: {st.get('state')} {json.dumps((st.get('decision') or {}), default=str)[:400]}")
                if meta.get("decoder_installed", True):
                    self.assertEqual(v["id"], Profile.load(meta["dir"]).id, (mid, st["decision"].get("mode"), st["decision"].get("runners_up")))
                    self.assertEqual(st["decision"]["mode"], "decoder file")
                else: self.assertTrue(st["decision"]["mode"].startswith("learned"), (mid, st["decision"].get("mode")))
                c = b.db(); self.assertIsNotNone(c)
                try:
                    self.assertGreater(c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0], 10, mid); self.assertGreater(c.execute("SELECT COUNT(*) FROM sig_catalog").fetchone()[0], 3, mid)
                finally: c.close()
                self.assertEqual(st["wire"]["bad_crc"], 0); self.assertEqual(st["dropped"], 0)

    def test_the_same_car_with_other_channel_numbers_is_the_same_vehicle(self):
        ids = set()
        for seed in (1, 2, 3):
            b = Bench(self); b.car("hyundai_elantra_ccan", seconds=30.0, channel_seed=seed * 977); ids.add(b.svc.status()["vehicle"]["id"])
        self.assertEqual(ids, {"hyundai_elantra_ccan"})

    def test_a_car_with_no_decoder_file_is_read_by_the_standards(self):
        for mid, want in (("generic_obd_dongle", {"speed", "engine_rpm"}), ("heavy_truck_j1939", {"speed", "engine_rpm"}), ("aaos_reference_ev", {"speed", "odometer"})):
            with self.subTest(model=mid):
                b = Bench(self, roots=[]); b.car(mid, seconds=40.0); st = b.svc.status(); self.assertIsNotNone(st.get("vehicle"), (mid, st.get("decision")))
                self.assertTrue(st["decision"]["mode"].startswith("learned"), st["decision"]["mode"]); c = b.db()
                try: got = {r[0] for r in c.execute("SELECT concept FROM sig_catalog")}
                finally: c.close()
                self.assertTrue(want <= got, (mid, want - got, sorted(got)))
                self.assertTrue(os.path.isdir(os.path.join(b.home, "learned")) and os.listdir(os.path.join(b.home, "learned")), "what was learned is written down as a decoder file")

    def test_a_reading_that_first_appears_after_the_vehicle_was_learned_is_added(self):
        """The standards-learned decoder is built from the first seconds; a property that only shows up later (or reports very rarely) must still be read."""
        from vgpt.protocols import vhalparcel
        b = Bench(self, roots=[]); r, w = os.pipe(); b.svc.attach(r); wr = CW.Writer(w); t0 = 1_790_000_000.0
        def prop(pid, val, ts, area=0): wr.send(ts, 7, vhalparcel.pack({"prop": pid, "area": area, "timestamp": int(ts * 1e9), "status": 0, "float": [val]}))
        for i in range(0, 600):                                                            # 60 s of car time at 10 Hz: speed and steering at once, the odometer only from second 20
            ts = t0 + i * 0.1; prop(0x11600207, 12.0 + i * 0.01, ts); prop(0x11600209, 3.0, ts)
            if i >= 200 and i % 50 == 0: prop(0x11600204, 1000.0 + i, ts)
        wr.close(); b.svc.join(20); b.svc.drain(30); db = b.db()
        try: got = {x[0] for x in db.execute("SELECT concept FROM sig_catalog")}
        finally: db.close()
        self.assertIn("speed", got); self.assertIn("odometer", got, (got, b.svc.status()["events"]))

    def test_a_car_nobody_has_a_decoder_for_is_listed_then_recognised_when_the_file_arrives(self):
        b = Bench(self, roots=[]); meta = model_dirs()["proprietary_demo"]
        r, w = os.pipe(); b.svc.attach(r); c = Car(meta["dir"], CW.Writer(w), speed=40.0, seed=3); c.start()
        try:
            end = time.monotonic() + 60
            while time.monotonic() < end and b.svc.status().get("state") != "unidentified": time.sleep(0.2)
            st = b.svc.status(); self.assertEqual(st["state"], "unidentified", st.get("events")); ex = st["decision"]["unexplained_can_ids"]
            self.assertGreaterEqual(len(ex), 4); self.assertTrue(all(set(e) >= {"id", "frames", "dlc"} for e in ex))
            shutil.copytree(meta["dir"], os.path.join(b.home, "decoders", "proprietary_demo"))                    # the person adds the decoder file
            end = time.monotonic() + 60
            while time.monotonic() < end and (b.svc.status().get("vehicle") or {}).get("id") != "proprietary_demo": time.sleep(0.2)
            st = b.svc.status(); self.assertEqual(st["state"], "running", st.get("events")); self.assertEqual(st["vehicle"]["id"], "proprietary_demo")
        finally: c.stop()

    def test_a_second_car_gets_a_database_of_its_own(self):
        b = Bench(self); b.car("aster_ev", seconds=30.0); first = b.svc.status()["vehicle"]; b.car("kestrel_petrol", seconds=30.0); second = b.svc.status()["vehicle"]
        self.assertNotEqual(first["db"], second["db"]); self.assertEqual((first["id"], second["id"]), ("aster_ev", "kestrel_petrol"))
        self.assertEqual(len(os.listdir(os.path.join(b.home, "vehicles"))), 2); self.assertTrue(os.path.exists(first["db"]) and os.path.exists(second["db"]))

    def test_the_vin_the_car_announces_keeps_two_cars_of_one_model_apart(self):
        b = Bench(self); b.car("hyundai_elantra_ccan", seconds=30.0, seed=11); a = b.svc.status()["vehicle"]; b.car("hyundai_elantra_ccan", seconds=30.0, seed=12); c = b.svc.status()["vehicle"]
        self.assertTrue(a["vin"] and c["vin"] and a["vin"] != c["vin"]); self.assertNotEqual(a["db"], c["db"])

    def test_the_time_of_the_car_is_the_time_of_the_data(self):
        b = Bench(self); b.car("meridian_ev", seconds=30.0); db = b.db()
        try: lo, hi = db.execute("SELECT MIN(ts), MAX(ts) FROM sig_1s").fetchone()
        finally: db.close()
        self.assertGreater(hi - lo, 15); self.assertGreater(lo, 1_000_000_000)                                      # a calendar time, set by the car's own clock

    def test_a_closed_link_is_the_end_of_the_session_not_a_crash(self):
        b = Bench(self); c = b.car("kestrel_petrol", seconds=25.0, wait=False)
        b.finish(c, 25.0); st = b.svc.status(); self.assertIn(st["state"], ("disconnected", "running")); self.assertTrue(st["closed"])
        db = b.db(); self.assertIsNotNone(db); db.close()                                                              # the database is closed cleanly and stays readable for questions


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- the one-way rule
class OneWay(unittest.TestCase):
    def test_vantage_code_cannot_transmit(self):
        pipe_dir = os.path.join(ROOT, "vgpt", "pipe")
        for name in sorted(os.listdir(pipe_dir)):
            if not name.endswith(".py"): continue
            src = open(os.path.join(pipe_dir, name), encoding="utf-8").read()
            self.assertNotRegex(src, r"\bos\.write\b|\.send\(|\.sendall\(|vgpt\.car|from \.\.car|(?:^|\n)\s*(?:import|from)\s+socket\b|\bsocket\.", name)

    def test_vantage_does_not_know_a_make_or_a_protocol_by_name(self):
        src = open(os.path.join(ROOT, "vgpt", "pipe", "receiver.py"), encoding="utf-8").read()
        for brand in ("toyota", "tesla", "hyundai", "bmw", "volvo", "mercedes", "nissan", "ford", "honda", "kia"): self.assertNotIn(brand, src.lower(), brand)

    def test_no_network_object_is_made_by_the_link(self):
        for mod in ("transport.py", "receiver.py", "wire.py"):
            src = open(os.path.join(ROOT, "vgpt", "pipe", mod), encoding="utf-8").read()
            self.assertNotRegex(src, r"import socket|socket\.socket|http\.|urllib|requests", mod)


if __name__ == "__main__":
    unittest.main()

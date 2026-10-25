"""GNSS: the NMEA decoder against an independent library and REAL GPS fixes, the live ingest path, and the generator's continuous 1 Hz track."""
import io, math, os, sys, tempfile, unittest, zipfile, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.protocols import nmea
from vgpt import adapters, store, datasets as D

try:
    import pynmea2
except ImportError:
    pynmea2 = None
HAVE = bool(D.load_vif()) and os.path.exists(os.path.join(D.REF, "epa", "uddscol.txt"))


def hav(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0]); h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2; return 12742 * math.asin(math.sqrt(h))


class Nmea(unittest.TestCase):
    def test_roundtrip_in_all_four_hemispheres(self):
        for la, lo in ((12.97160, 77.59460), (-33.86880, 151.20930), (59.91390, -10.75220), (-34.60370, -58.38160), (0.00001, -0.00001)):
            m = nmea.decode(nmea.encode_rmc(1_790_000_000.25, la, lo, 63.7, 271.5)); self.assertTrue(m["valid"]); self.assertAlmostEqual(m["lat"], la, delta=2e-6); self.assertAlmostEqual(m["lon"], lo, delta=2e-6)
            self.assertAlmostEqual(m["speed_kmh"], 63.7, delta=0.1); self.assertAlmostEqual(m["course"], 271.5, delta=0.1); self.assertAlmostEqual(m["ts"], 1_790_000_000.25, delta=0.01)

    @unittest.skipUnless(pynmea2, "pynmea2 not installed (dev venv has it)")
    def test_agrees_with_the_independent_pynmea2_library(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            la, lo = float(rng.uniform(-80, 80)), float(rng.uniform(-179, 179)); ts = 1_700_000_000 + float(rng.integers(0, 10 ** 8)); v = float(rng.uniform(0, 160)); co = float(rng.uniform(0, 359))
            line = nmea.encode_rmc(ts, la, lo, v, co); ours = nmea.decode(line); ref = pynmea2.parse(line)
            self.assertAlmostEqual(ours["lat"], ref.latitude, delta=1e-9); self.assertAlmostEqual(ours["lon"], ref.longitude, delta=1e-9); self.assertAlmostEqual(ours["speed_kmh"], ref.spd_over_grnd * 1.852, delta=0.02)
            g = nmea.encode_gga(ts, la, lo, 1, 9, 1.3, 912.4); og = nmea.decode(g); rg = pynmea2.parse(g)
            self.assertAlmostEqual(og["lat"], rg.latitude, delta=1e-9); self.assertEqual(og["sats"], int(rg.num_sats)); self.assertAlmostEqual(og["hdop"], float(rg.horizontal_dil), delta=1e-9); self.assertAlmostEqual(og["alt"], float(rg.altitude), delta=1e-9)

    def test_bad_input_is_rejected_never_fatal(self):
        good = nmea.encode_rmc(1_790_000_000, 12.9, 77.5, 10, 90)
        for bad in (good[:-1] + ("0" if good[-1] != "0" else "1"), "garbage", "$GPRMC,1,2*00", ""):
            with self.assertRaises(nmea.NmeaError): nmea.decode(bad)
        a = adapters.NmeaAdapter(); self.assertEqual(a.decode("garbage"), []); self.assertEqual(a.bad, 1)
        self.assertEqual(a.decode(nmea.encode_rmc(1_790_000_000, 0, 0, 0, 0, valid=False)), [("sig", "gps_valid", 0.0)])       # no fix: receiver still talks

    @unittest.skipUnless(HAVE and os.path.exists(D.VIF_ZIP), "needs the VIF-AEGIS zip")
    def test_real_gps_fixes_from_a_real_drive_decode_exactly(self):
        z = zipfile.ZipFile(D.VIF_ZIP); n = bad = 0; prev = None; dist = 0.0
        with z.open("positions.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                if row["trip_id"] != "4": continue
                dm = lambda s: int(float(s) // 100) + (float(s) % 100) / 60.0                       # the file stores ddmm.mmmm
                la, lo = dm(row["latitude"]), dm(row["longitude"]); ts = 1_484_850_000 + n
                m = nmea.decode(nmea.encode_rmc(ts, la, lo, 50.0, 90.0)); n += 1
                if abs(m["lat"] - la) > 2e-6 or abs(m["lon"] - lo) > 2e-6: bad += 1
                if prev: dist += hav(prev, (m["lat"], m["lon"]))
                prev = (m["lat"], m["lon"])
        self.assertGreater(n, 300); self.assertEqual(bad, 0); self.assertTrue(2 < dist < 200, dist)                      # a real trip's decoded track has a sensible length

    def test_live_ingest_one_row_per_second_moving_one_per_minute_parked(self):
        path = os.path.join(tempfile.mkdtemp(), "g.db"); c = store.open_rw(path); ing = store.Ingestor(c, adapters.Router()); t0 = 1_790_000_000
        for i in range(120):                                                                             # 2 minutes of driving at 1 Hz
            ts = t0 + i; la = 12.97 + i * 1e-5
            for line in (nmea.encode_rmc(ts, la, 77.59, 54.0, 10.0), nmea.encode_gga(ts, la, 77.59, 1, 10, 0.9, 900)): ing.feed(ts + 0.01, "gnss", line)
        for i in range(120, 420):                                                                        # 5 minutes parked: the receiver keeps talking at 1 Hz
            ts = t0 + i
            for line in (nmea.encode_rmc(ts, 12.9712, 77.59, 0.0, 10.0), nmea.encode_gga(ts, 12.9712, 77.59, 1, 10, 0.9, 900)): ing.feed(ts + 0.01, "gnss", line)
        ing.feed(t0 + 500, "gnss", "$GPRMC,garbage*00"); ing.close()
        moving = c.execute("SELECT COUNT(*) FROM gps_1s WHERE ts<?", (t0 + 120,)).fetchone()[0]; parked = c.execute("SELECT COUNT(*) FROM gps_1s WHERE ts>=?", (t0 + 120,)).fetchone()[0]
        self.assertGreaterEqual(moving, 118); self.assertTrue(4 <= parked <= 7, parked)
        r = c.execute("SELECT lat, lon, hdg, spd, hdop FROM gps_1s WHERE ts=?", (t0 + 60,)).fetchone(); self.assertAlmostEqual(r[0] / 1e6, 12.97 + 60e-5, delta=3e-6); self.assertAlmostEqual(r[2] / 10, 10.0, delta=0.2); self.assertAlmostEqual(r[3] / 10, 54.0, delta=0.2); self.assertAlmostEqual(r[4] / 10, 0.9, delta=0.1)


@unittest.skipUnless(HAVE, "needs the VIF-AEGIS zip (Downloads) and data/ref/epa")
class GeneratedTrack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt import datagen_life as DL, datagen_events as DE, brain
        cls.DE, cls.brain = DE, brain
        cls.gv = DL.build(seed=11, days=14, city="graz", driver="commuter", vehicle="family", end="2026-08-20", offline=True, verbose=False, work_km=22, stories=[])
        cls.path = os.path.join(tempfile.mkdtemp(), "t.db"); DE.seed(cls.path, cls.gv)

    def test_trips_chain_and_the_gps_track_is_continuous(self):
        tr = self.gv.trips
        for a, b in zip(tr, tr[1:]):                                                                    # a trip starts where the previous one ended
            if a["b"] >= 0 and b["a"] >= 0: self.assertEqual(a["b"], b["a"])
        rows = self.gv.ex["gps_1s"]; ts = np.array([r[0] for r in rows]); t_last = self.gv.t0 + self.gv.n
        self.assertGreater(len(rows), 10_000); self.assertTrue(ts.min() >= t_last - 7 * 86400 - 60)         # 7 days of fixes
        last = tr[-1]; seg = [r for r in rows if last["start"] <= r[0] < last["end"]]
        self.assertGreater(len(seg), 0.9 * (last["end"] - last["start"]) - 120)                             # ~1 fix per second while driving (a few dropouts)
        km = sum(hav((a[1] / 1e6, a[2] / 1e6), (b[1] / 1e6, b[2] / 1e6)) for a, b in zip(seg, seg[1:]) if b[0] - a[0] <= 2)
        self.assertAlmostEqual(km, last["km"], delta=0.25 * last["km"] + 0.5)                               # the track's length agrees with the trip's odometer distance (GNSS noise inflates it a little)

    def test_first_and_last_fix_of_a_trip_are_at_the_places(self):
        pl = {i + 1: (p["lat"], p["lon"]) for i, p in enumerate(self.gv.plan.places)}; rows = self.gv.ex["gps_1s"]
        for t in [t for t in self.gv.trips if t["a"] >= 0 and t["b"] >= 0 and t["start"] > self.gv.t0 + self.gv.n - 6 * 86400][:6]:
            seg = [r for r in rows if t["start"] <= r[0] < t["end"]]
            self.assertLess(hav((seg[0][1] / 1e6, seg[0][2] / 1e6), pl[t["a"] + 1]), 0.5); self.assertLess(hav((seg[-1][1] / 1e6, seg[-1][2] / 1e6), pl[t["b"] + 1]), 0.5)

    def test_assistant_uses_the_live_fix_heading_and_speed(self):
        import sqlite3
        c = sqlite3.connect(self.path); r = c.execute("SELECT ts, lat, lon, hdg, spd FROM gps_1s WHERE spd>600 ORDER BY ts DESC LIMIT 1 OFFSET 400").fetchone(); b = self.brain.Brain(self.path); b.now = r[0]
        a = b.ask("where am i"); self.assertIn("live fix", a["answer"]); self.assertIn("heading", a["answer"]); f = a["facts"]["here"]
        self.assertAlmostEqual(f["lat"], r[1] / 1e6, delta=1e-4); self.assertTrue(f["moving"]); self.assertAlmostEqual(f["heading"], r[3] / 10, delta=1.0)
        a = b.ask("which direction am i heading"); self.assertIn("degrees", a["answer"]); self.assertEqual(a["intent"], "PLACES")
        a = b.ask("how long to get home"); self.assertEqual(a["intent"], "PLACES"); self.assertIn("by road", a["answer"])
        b.now = self.gv.t0 + self.gv.n + 8 * 3600; self.assertIn("parked", b.ask("where am i")["answer"])


if __name__ == "__main__":
    unittest.main()

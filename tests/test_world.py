"""The virtual vehicle and its dataset: ground truth against the database, the ledger against the generator, and the bulk writer against the LIVE path (raw frames -> router -> ingestor -> lake).

The history of a vehicle is written in bulk (105 days of 10 Hz bus traffic would be 90 million frames).  The claim that this is the same data the live collectors would have produced is tested here on a
real hour of the simulated drive: the canonical state is turned into the raw frames of the vehicle's buses, decoded again by the profile router, written by the ingestor and the lake, and compared with
what the bulk writer stored for that hour."""
import os as _os; _os.environ.setdefault("VGPT_DRIVING_FIXTURE", "1")                                  # the small derived driving fixture stands in for the real traces when they are absent (tools_testdata.py)
import json
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import store
from vgpt.vehicle import ledger, lake as LK
from vgpt.vehicle.router import ProfileRouter
from vgpt.world import build as WB
from vgpt.world.emit import Emitter
from vgpt.world.signals import Signals

DAYS = 21


class WorldBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.gettempdir(), "tw_world.db")
        cls.gv = WB.build_world("aster_ev", days=DAYS, seed=5, end="2026-10-18", verbose=False); WB.write_world(cls.gv, cls.path, verbose=False)
        cls.c = sqlite3.connect(cls.path); cls.truth = json.load(open(cls.path[:-3] + "_truth.json")); cls.t0 = cls.gv.t0

    @classmethod
    def tearDownClass(cls): cls.c.close()

    def events(self, ty): return self.c.execute("SELECT ts, detail FROM events WHERE type=? ORDER BY ts", (ty,)).fetchall()

    def test_every_injected_fault_is_in_the_event_table_at_its_timestamp(self):
        for f in self.truth["faults"].get("ecu_resets_adas", []):
            self.assertIn(f["ts"], [t for t, _ in self.events("ECU_RESET")]); self.assertIn(f["adas_disengage_ts"], [t for t, _ in self.events("ADAS_DISENGAGE")])
        for f in self.truth["faults"].get("voltage_dip", []): self.assertIn(f["ts"], [t for t, _ in self.events("VOLTAGE_DIP")])
        for f in self.truth["faults"].get("speed_conflict", []): self.assertIn(f["ts"], [t for t, _ in self.events("SENSOR_CONFLICT")])
        for f in self.truth["faults"].get("aeb", []): self.assertIn(f["ts"], [t for t, _ in self.events("AEB_ACTIVATION")])
        n_dc = sum(1 for f in self.truth["faults"].get("dcdc_condition", []) if f["fault"]); self.assertEqual(len([1 for t, d in self.events("DTC_SET") if d.startswith("P0A94")]), n_dc)

    def test_the_unexpected_shutdown_leaves_a_gap_and_the_codes_after_the_restart(self):
        sd = self.truth["faults"].get("unexpected_shutdown")
        if not sd: self.skipTest("this seed has no shutdown inside the window")
        f = sd[0]; self.assertEqual(self.events("UNEXPECTED_SHUTDOWN")[0][0], f["ts"])
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM sig_1s WHERE ts>=? AND ts<?", (f["ts"], f["resumed_ts"])).fetchone()[0], 0)               # no data while the vehicle was off
        codes = [d.split(" ")[0] for t, d in self.events("DTC_SET") if f["resumed_ts"] <= t <= f["resumed_ts"] + 10]; self.assertIn("P0562", codes); self.assertIn("U0155", codes)
        self.assertGreater(f["last_valid"]["speed"], 0)

    def test_a_can_burst_silences_the_body_bus_signals_only(self):
        b = self.truth["faults"].get("can_burst")
        if not b: self.skipTest("no burst in this window")
        f = b[0]; sid = self.c.execute("SELECT sid FROM sig_catalog WHERE concept='tyre_pressure_fl'").fetchone()[0]; sp = self.c.execute("SELECT sid FROM sig_catalog WHERE concept='steering_angle'").fetchone()[0]
        last7 = f["ts"] > self.c.execute("SELECT MAX(ts) FROM lk_1s").fetchone()[0] - 7 * 86400
        if last7:
            self.assertEqual(self.c.execute("SELECT COUNT(*) FROM lk_1s WHERE sid=? AND ts>=? AND ts<?", (sid, f["ts"], f["ts"] + f["duration_s"])).fetchone()[0], 0)
            self.assertGreater(self.c.execute("SELECT COUNT(*) FROM lk_1s WHERE sid=? AND ts>=? AND ts<?", (sp, f["ts"], f["ts"] + f["duration_s"])).fetchone()[0], 0)    # the powertrain bus kept talking

    def test_ten_weeks_and_more_are_searchable_in_every_tier_that_should_hold_them(self):
        span = lambda t, col="ts": self.c.execute(f"SELECT (MAX({col}) - MIN({col})) / 86400.0 FROM {t}").fetchone()[0]
        self.assertGreater(span("sig_1m"), DAYS - 1); self.assertGreater(span("lk_1m"), DAYS - 1); self.assertLess(span("sig_1s"), 14.1); self.assertLess(span("lk_1s"), 14.1)
        self.assertGreater(span("events"), DAYS - 3); self.assertEqual(self.c.execute("SELECT v FROM meta WHERE k='ret.coarse_days'").fetchone()[0], "1830")                                          # a year of per-minute history (the product floor)

    def test_context_around_important_events_is_frozen_for_every_channel(self):
        n = self.c.execute("SELECT COUNT(*) FROM event_ctx_x").fetchone()[0]; self.assertGreater(n, 5)
        ts, ty, blob = self.c.execute("SELECT ts, type, blob FROM event_ctx_x ORDER BY ts LIMIT 1").fetchone(); import zlib
        d = json.loads(zlib.decompress(blob)); self.assertEqual(d["hz"], 1); self.assertIn("speed", d["sig"]); self.assertGreater(len(d["sig"]["speed"]), 100)

    def test_the_ledger_reproduces_the_generators_trips_and_charges(self):
        t7 = self.c.execute("SELECT MAX(ts) FROM sig_1s").fetchone()[0] - 7 * 86400 + 3600
        gen = self.c.execute("SELECT start, km, kwh FROM trips WHERE start>=? ORDER BY start", (t7,)).fetchall()
        trips, charges = ledger.derive(ledger.rows_from_sig_1s(self.c, t7 - 3600)); trips = [t for t in trips if t["start"] >= t7]
        self.assertEqual([g[0] for g in gen], [t["start"] for t in trips])
        for g, t in zip(gen, trips): self.assertAlmostEqual(g[1], t["km"], delta=0.02 + 0.005 * g[1]); self.assertAlmostEqual(g[2], t["kwh"], delta=0.01 + 0.01 * abs(g[2]))
        self.assertEqual(len([1 for ch in self.c.execute("SELECT start FROM charges WHERE start>=?", (t7,)).fetchall()]), len([c for c in charges if c["start"] >= t7]))


class BulkEqualsLive(unittest.TestCase):
    """one real hour of the drive, through raw frames, against what the bulk writer stored"""
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.gettempdir(), "tw_world.db")
        if not os.path.exists(cls.path): WorldBuild.setUpClass()
        cls.gv = WB.build_world("aster_ev", days=DAYS, seed=5, end="2026-10-18", verbose=False); cls.p = cls.gv.profile

    def test_an_hour_of_raw_frames_gives_the_same_rows(self):
        gv = self.gv; S = Signals(gv, self.p, gv.fault_seed, faults=gv.faults)
        secs = [int(((gv.mode[d * 86400:(d + 1) * 86400] == 1) | (gv.mode[d * 86400:(d + 1) * 86400] == 2)).sum()) for d in range(DAYS)]; day = max(range(DAYS - 6, DAYS - 1), key=lambda d: secs[d])     # the busiest day inside the 1 Hz window
        for d in range(day): S.day(d)
        out = S.day(day); drv = np.where((gv.mode[day * 86400:(day + 1) * 86400] == 1) | (gv.mode[day * 86400:(day + 1) * 86400] == 2))[0]
        if not len(drv): self.skipTest("no driving that day")
        a = int(drv[0]) - 120; b = a + 3600; base = gv.t0 + day * 86400
        path = os.path.join(tempfile.gettempdir(), "tw_live.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(path + e)
            except OSError: pass
        c = store.open_rw(path); LK.provision(c, self.p); lk = LK.Lake(c); ing = store.Ingestor(c, ProfileRouter(self.p), lake=lk, track_meta=True); em = Emitter(self.p)
        for i in range(a, b):
            st = {k: float(v[i]) for k, v in out.items() if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.isnan(v[i])}
            st["_ignition"] = float(out["_ignition"][i]); st["_charging_active"] = float(out["_charging_active"][i])
            for ts, bus, payload in em.tick(base + i, st): ing.feed(ts, bus, payload)
        ing.close(); bulk = sqlite3.connect(os.path.join(tempfile.gettempdir(), "tw_world.db"))
        lo, hi = base + a + 30, base + b - 30                                                           # skip the first seconds (the ingestor needs a sample of every channel)
        live = {r[0]: r for r in c.execute("SELECT ts, speed, soc, power, batt, flow, amb, hvac, motor, mode FROM sig_1s WHERE ts>=? AND ts<?", (lo, hi))}
        ref = {r[0]: r for r in bulk.execute("SELECT ts, speed, soc, power, batt, flow, amb, hvac, motor, mode FROM sig_1s WHERE ts>=? AND ts<?", (lo, hi))}
        common = sorted(set(live) & set(ref)); self.assertGreater(len(common), 300)
        tol = {1: 3, 2: 6, 3: 600, 4: 4, 5: 4, 6: 3, 7: 60, 8: 6}                                      # speed 0.3 km/h, soc 0.6 %, power 0.6 kW, temperatures 0.4 C, flow 0.4 L/min, hvac 60 W ...
        for col, t in tol.items():
            bad = sum(1 for k in common if live[k][col] is not None and ref[k][col] is not None and abs(live[k][col] - ref[k][col]) > t) / len(common)
            self.assertLess(bad, 0.02, f"column {col}: {bad:.1%} of the seconds differ beyond the wire resolution")
        self.assertGreater(sum(1 for k in common if live[k][9] == ref[k][9]) / len(common), 0.97)       # the drive mode agrees
        # a lake channel: per-minute means from the live path equal the bulk ones
        for concept in ("steering_angle", "tyre_pressure_fl", "brake_pressure", "battery_voltage"):
            sid = c.execute("SELECT sid FROM sig_catalog WHERE concept=?", (concept,)).fetchone()[0]; sid2 = bulk.execute("SELECT sid FROM sig_catalog WHERE concept=?", (concept,)).fetchone()[0]
            m1 = {r[0]: r[1] for r in c.execute("SELECT ts, v FROM lk_1m WHERE sid=? AND ts>=? AND ts<?", (sid, lo + 60, hi - 60))}; m2 = {r[0]: r[1] for r in bulk.execute("SELECT ts, v FROM lk_1m WHERE sid=? AND ts>=? AND ts<?", (sid2, lo + 60, hi - 60))}
            both = sorted(set(m1) & set(m2)); self.assertGreater(len(both), 40, concept)
            scale = max(1.0, float(np.nanmax(np.abs([m2[k] for k in both])))); err = float(np.median([abs(m1[k] - m2[k]) for k in both])) / scale
            self.assertLess(err, 0.03, f"{concept}: median minute-mean difference {err:.3f} of its range")
        # harsh braking found by the router's detector equals what the bulk writer recorded (a one-second event may land on either side of the boundary)
        hb_live = c.execute("SELECT COUNT(*) FROM events WHERE type='HARSH_BRAKE' AND ts>=? AND ts<?", (lo, hi)).fetchone()[0]; hb_ref = bulk.execute("SELECT COUNT(*) FROM events WHERE type='HARSH_BRAKE' AND ts>=? AND ts<?", (lo, hi)).fetchone()[0]
        self.assertLessEqual(abs(hb_live - hb_ref), 2, f"hard braking: live {hb_live} vs bulk {hb_ref}")


if __name__ == "__main__":
    unittest.main()

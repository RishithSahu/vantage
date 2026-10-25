"""The application receives its data: a recorded vehicle is a SOURCE, the runtime ingests what arrives, the assistant answers from it - and from the live bus when the question is about the present.

Built on the shipped Aster world (a short window of its recorded week); skipped when it is absent."""
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "data", "world_aster.db")

from vgpt import live_snapshot as LS, livebus, store  # noqa: E402
from vgpt.live import replay as RP  # noqa: E402
from vgpt.vehicle.profile import Profile  # noqa: E402


class Sched:
    laps = 0


def first_second():
    c = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True); r = c.execute("SELECT MIN(ts) FROM sig_1s WHERE mode=1 AND ts>(SELECT MIN(ts)+86400 FROM sig_1s)").fetchone()[0]; c.close(); return int(r)


@unittest.skipUnless(os.path.exists(SRC), "the Aster world is not present")
class Wire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.begin = first_second() - 90; cls.work = os.path.join(tempfile.gettempdir(), "pipeline_test_aster.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(cls.work + e)
            except OSError: pass
        s = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True); d = sqlite3.connect(cls.work); s.backup(d); d.close(); s.close()
        c = sqlite3.connect(cls.work, isolation_level=None); RP.cut_after(c, cls.begin); cls.before = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("sig_1s", "events", "gps_1s", "lk_1s")}; c.close()
        cls.src = RP.WireSource(Sched(), SRC, cls.work, Profile.load("aster_ev"), cls.begin)

    @classmethod
    def tearDownClass(cls):
        LS.LIVE_BUS = None; cls.src.close()

    def test_a_the_application_starts_with_the_history_up_to_the_replay_and_nothing_after_it(self):
        c = sqlite3.connect(f"file:{self.work}?mode=ro", uri=True)
        for tb, col in (("sig_1s", "ts"), ("sig_1m", "ts"), ("lk_1s", "ts"), ("gps_1s", "ts"), ("events", "ts"), ("trips", "end"), ("logs", "ts")):
            self.assertIsNone(c.execute(f"SELECT 1 FROM {tb} WHERE {col}>? LIMIT 1", (self.begin,)).fetchone(), tb + " must hold nothing after the replay start")
        self.assertGreater(c.execute("SELECT COUNT(*) FROM sig_1m").fetchone()[0], 10000, "the weeks before it are the history the application already holds"); c.close()

    def test_b_what_arrives_is_decoded_and_stored_exactly_as_recorded(self):
        T = self.src.deliver(self.begin + 300); self.src.publish(); self.assertEqual(T, self.begin + 300)
        s = self.src.status(); self.assertGreater(s["msgs"], 10000); self.assertEqual(s["dropped"], 0); self.assertTrue({"can", "someip", "vhal", "gnss"} <= set(s["by_bus"]))
        o = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True); w = sqlite3.connect(f"file:{self.work}?mode=ro", uri=True)
        want = {r[0]: r[1:] for r in o.execute("SELECT ts, speed, soc, batt, motor, mode FROM sig_1s WHERE ts>? AND ts<?", (self.begin, T))}; got = {r[0]: r[1:] for r in w.execute("SELECT ts, speed, soc, batt, motor, mode FROM sig_1s WHERE ts>?", (self.begin,))}
        self.assertGreater(len(want), 100); self.assertTrue(set(want) <= set(got), "every recorded second arrives"); self.assertLessEqual(len(set(got) - set(want)), 6, "nothing is invented (a parked car wakes at its own seconds)")
        bad = [t for t in want for a, b in zip(want[t], got[t]) if a is not None and b is not None and abs(a - b) > 2]; self.assertLessEqual(len(bad), 2, "values arrive as sent (within the bus's own quantisation)")

    def test_c_the_minute_tier_is_kept_up_as_the_data_arrives(self):
        self.src.deliver(self.begin + 600); self.src.publish(); w = sqlite3.connect(f"file:{self.work}?mode=ro", uri=True)
        n = w.execute("SELECT COUNT(*) FROM sig_1m WHERE ts>?", (self.begin,)).fetchone()[0]; self.assertGreaterEqual(n, 8, "each finished minute is rolled up at once, not at midnight"); w.close()

    def test_d_events_the_vehicle_reports_and_events_the_ingest_derives_both_arrive(self):
        o = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
        for kind in ("CHARGE_START", "DTC_SET"): self.one_event(o, kind)
        o.close()

    def one_event(self, o, kind):
        ev = o.execute("SELECT ts, type FROM events WHERE type=? AND ts>? ORDER BY ts LIMIT 1", (kind, self.begin + 600)).fetchone()
        self.assertIsNotNone(ev); ty = ev[1]
        w2 = os.path.join(tempfile.gettempdir(), "pipeline_test_events.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(w2 + e)
            except OSError: pass
        s = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True); d = sqlite3.connect(w2); s.backup(d); d.close(); s.close()
        b = ev[0] - 90; c = sqlite3.connect(w2, isolation_level=None); RP.cut_after(c, b); c.close(); src = RP.WireSource(Sched(), SRC, w2, Profile.load("aster_ev"), b)
        try: src.deliver(ev[0] + 90); src.publish()
        finally: src.close()
        w = sqlite3.connect(f"file:{w2}?mode=ro", uri=True); self.assertIsNotNone(w.execute("SELECT 1 FROM events WHERE type=? AND ts>?", (ty, b)).fetchone(), f"{ty} was recorded at {ev[0]} and must arrive through the bus"); w.close()

    def test_e_a_question_about_the_present_is_answered_from_the_live_bus_and_an_older_one_from_the_dataset(self):
        self.src.deliver(self.begin + 200); bus = livebus.LiveBus(self.src.rt); LS.LIVE_BUS = bus; T = int(bus.moment())
        try:
            w = store.open_ro(self.work); live = LS.snapshot(w, T)
            self.assertTrue(live.get("live")); self.assertEqual(live["tier"], "live"); self.assertLess(live["age_s"], 3)
            self.assertIn("speed", live["live_concepts"]); self.assertAlmostEqual(live["values"]["speed"], bus.value("speed")["value"], places=6)
            old = LS.snapshot(w, T - 25)                                                      # more than 20 s before the live moment: the dataset only
            self.assertFalse(old.get("live")); self.assertNotEqual(old.get("tier"), "live"); self.assertFalse(bus.covers(T - 25)); self.assertTrue(bus.covers(T - 19)); self.assertEqual(bus.value("speed", T - 25), None)
            w.close()
        finally: LS.LIVE_BUS = None

    def test_f_a_live_reading_is_never_older_than_its_signal_allows(self):
        self.src.deliver(self.begin + 220); bus = livebus.LiveBus(self.src.rt)
        r = bus.value("speed"); self.assertIsNotNone(r); self.assertLessEqual(r["age_s"], 5.0); self.assertIsNone(bus.value("this_is_not_a_signal")); self.assertIsNone(bus.value("range_estimate"))        # a derived estimate is never offered live
        self.assertIsNotNone(bus.value("battery_power")); self.assertIn(bus.value("drive_mode")["value"], (0.0, 1.0, 2.0, 3.0))

    def test_g_the_schedule_restart_gives_the_history_back_as_it_was(self):
        self.src.restart(); w = sqlite3.connect(f"file:{self.work}?mode=ro", uri=True)
        self.assertIsNone(w.execute("SELECT 1 FROM sig_1s WHERE ts>? LIMIT 1", (self.begin,)).fetchone(), "the recording plays again from its start")
        self.assertEqual(self.src.t, self.begin); w.close(); self.assertEqual(self.src.deliver(self.begin + 60), self.begin + 60); self.assertEqual(self.src.t, self.begin + 60)


class Templates(unittest.TestCase):
    def test_an_event_text_is_turned_back_into_its_fields(self):
        f = RP.invert("12 V supply dipped to {min_v:.1f} V for {duration_ms} ms", "12 V supply dipped to 8.4 V for 180 ms"); self.assertEqual((f["min_v"], f["duration_ms"]), (8.4, 180))
        f = RP.invert("{ecu_text} reset ({cause_text})", "infotainment reset (brown-out)"); self.assertEqual((f["ecu_text"], f["cause_text"]), ("infotainment", "brown-out"))
        self.assertIsNone(RP.invert("{kind_text} {kw:.0f} kW", "nothing like it"))


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "cs5_petrol_ice.db")), "the Case Study 5 petrol file is not present")
class Records(unittest.TestCase):
    def test_a_recording_without_a_vehicle_profile_is_delivered_row_by_row_as_recorded(self):
        src = os.path.join(ROOT, "data", "cs5_petrol_ice.db"); work = os.path.join(tempfile.gettempdir(), "pipeline_test_cs5.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(work + e)
            except OSError: pass
        s = sqlite3.connect(f"file:{src}?mode=ro", uri=True); d = sqlite3.connect(work); s.backup(d); d.close()
        lo, hi = s.execute("SELECT MIN(ts), MAX(ts) FROM sig_1s").fetchone(); mid = lo + (hi - lo) // 2; c = sqlite3.connect(work, isolation_level=None); RP.cut_after(c, lo); c.close()
        r = RP.RecordSource(Sched(), src, work, lo)
        try:
            self.assertEqual(sqlite3.connect(f"file:{work}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM sig_1s WHERE ts>?", (lo,)).fetchone()[0], 0)
            r.deliver(mid); n = sqlite3.connect(f"file:{work}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM sig_1s WHERE ts>?", (lo,)).fetchone()[0]; want = s.execute("SELECT COUNT(*) FROM sig_1s WHERE ts>? AND ts<=?", (lo, mid)).fetchone()[0]
            self.assertEqual(n, want); self.assertEqual(r.status()["kind"], "records")
        finally: r.close(); s.close()


if __name__ == "__main__":
    unittest.main()

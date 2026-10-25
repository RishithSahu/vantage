"""VehicleGPT tests:  python -m unittest discover -s tests -v   (run from vehiclegpt/; needs data/vehicle.db from tools_seed.py)"""
import json, os, subprocess, sys, unittest
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import adapters, store, analytics, brain, canhealth
DB = os.path.join(ROOT, "data", "vehicle.db")
from tests import testdata
HAVE, WHY = testdata.gate("legacy_worlds")                                                                  # an integration test of the first generated vehicle: absent data skips it with the reason (a failure in a release build)
TRUTH = json.load(open(os.path.join(ROOT, "data", "truth.json"))) if HAVE else {}


try:
    import cantools
except ImportError:                                                  # dev-only cross-check library
    cantools = None


@unittest.skipUnless(HAVE, WHY)
class Adapters(unittest.TestCase):
    @unittest.skipUnless(cantools, "needs the dev virtualenv (cantools)")
    def test_can_and_canfd_roundtrip_every_dbc_signal(self):
        db = cantools.database.load_file(os.path.join(ROOT, "vgpt", "protocols", "vehicle.dbc")); ca = adapters.CanAdapter()
        vals = {"VehDyn": dict(Speed=87.25, Gear=3), "BmsStatus": dict(Soc=64.5, PackVolt=396.4, PackCurrent=-123.4, BattTemp=41.7, CellTempMax=43.0, CellTempMin=40.5, CellVoltMin=3.71),
                "Thermal": dict(CoolantFlow=9.8, PumpRpm=2100, MotorTemp=63.2), "Climate": dict(AmbTemp=-3.5, HvacPower=1240), "Charger": dict(ChgKw=118.0, ChgState=1)}
        got = {}
        for name, v in vals.items():
            m = db.get_message_by_name(name); data = bytes(m.encode(v, strict=False)); self.assertEqual(len(data), m.length)
            got.update({k: x for t, k, x in ca.decode(m.frame_id, data, m.length > 8) if t == "sig"})
        want = {"speed": 87.25, "gear": 3, "soc": 64.5, "pack_v": 396.4, "pack_a": -123.4, "batt_temp": 41.7, "coolant_flow": 9.8, "pump_rpm": 2100, "motor_temp": 63.2, "amb_temp": -3.5, "hvac_w": 1240, "chg_kw": 118.0, "chg_state": 1.0}
        for k, v in want.items(): self.assertAlmostEqual(got[k], v, delta=0.11 if k != "hvac_w" else 10.1, msg=k)

    def test_unknown_or_malformed_input_is_dropped_not_fatal(self):
        r = adapters.Router()
        self.assertEqual(r.decode("can", (0x7FF, bytes(8), False)), []); self.assertEqual(r.decode("can", (0x100, b"\x01", False)), [])
        self.assertEqual(r.decode("someip", b"\x00\x01"), []); self.assertEqual(r.decode("vhal", {"prop": 12345}), []); self.assertEqual(r.decode("log", "garbage line"), [])

    def test_log_adapter_keeps_warn_and_above_only(self):
        la = adapters.LogAdapter()
        self.assertEqual(la.decode("10-01 13:00:00.000    88    88 I ThermalHmi: all fine"), [])
        self.assertEqual(la.decode("Oct 01 13:11:02.345 thermal_mgr 3 0 coolant pump flow low (6.0 L/min)")[0][0], "log")
        self.assertEqual([x[0] for x in la.decode("10-01 13:14:00.500    88    88 E ThermalHmi: Battery temperature warning raised (>52C)")], ["log", "event"])

    def test_camera_adapter_stores_metadata_only_and_debounces_blockage(self):
        ca = adapters.CameraAdapter(); ev = []
        for _ in range(4): ev += [r for r in ca.decode({"lane_conf": .9, "blockage": .8}) if r[0] == "event"]
        self.assertEqual(len(ev), 1)


@unittest.skipUnless(HAVE, WHY)
class Store(unittest.TestCase):
    def test_assistant_handle_is_read_only(self):
        c = store.open_ro(DB)
        with self.assertRaises(Exception): c.execute("INSERT INTO events VALUES(1,'x','x','x')")
        with self.assertRaises(Exception): c.execute("DELETE FROM sig_1m")

    def test_database_is_small(self):
        self.assertLess(os.path.getsize(DB), 20 * 1024 * 1024)

    @unittest.skipUnless(cantools, "needs the dev virtualenv (cantools)")
    def test_live_ingest_matches_ground_truth(self):
        from vgpt import sim
        veh = sim.build(); path = os.path.join(ROOT, "data", "ut_live.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(path + e)
            except OSError: pass
        c = store.open_rw(path); ing = store.Ingestor(c, adapters.Router())
        for m in sim.live_messages(veh, 26 * 86400 + 13 * 3600 + 6 * 60, 600, can_hz=10): ing.feed(*m)
        ing.close(); r = c.execute("SELECT batt, flow FROM sig_1s WHERE ts=?", (TRUTH["overtemp_ts"],)).fetchone(); c.close()
        self.assertAlmostEqual(r[0] / 10, TRUTH["batt_at_overtemp"], delta=0.2); self.assertAlmostEqual(r[1] / 10, TRUTH["flow_at_overtemp"], delta=0.2)
        for e in ("", "-wal", "-shm"):
            try: os.remove(path + e)
            except OSError: pass


@unittest.skipUnless(HAVE, WHY)
class Analytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = brain.Brain(DB)

    def test_event_counts_match_truth_for_every_window(self):
        analytics.set_tz(self.b.tz)                                   # the module-level time zone is shared state: other tests' vehicles may have changed it
        for ty, wins in TRUTH["events_counts"].items():
            for w, n in wins.items(): self.assertEqual(analytics.events(self.b.c, ty, self.b.now, w)["count"], n, (ty, w))

    def test_why_range_numbers(self):
        f = analytics.why_range(self.b.c, self.b.now, "14"); self.assertAlmostEqual(f["first"], TRUTH["why14"]["first"], delta=1); self.assertAlmostEqual(f["last"], TRUTH["why14"]["last"], delta=1)

    def test_largest_change_day(self):
        self.assertEqual(analytics.largest_change(self.b.c, "energy_use", self.b.now, "a")["day"], TRUTH["largest_eff_drop_day"])

    def test_compare_modes_within_tolerance(self):
        f = analytics.compare_modes(self.b.c, self.b.now)
        self.assertLess(abs(f["city"] - TRUTH["city_whkm"]) / TRUTH["city_whkm"], 0.06); self.assertLess(abs(f["highway"] - TRUTH["highway_whkm"]) / TRUTH["highway_whkm"], 0.06)

    def test_before_event_finds_cooling_failure(self):
        f = analytics.before_event(self.b.c, "WARN_BATT_OVERTEMP", self.b.now, 60)
        self.assertEqual(f["ts"], TRUTH["overtemp_ts"]); d = {x["signal"]: x for x in f["deviations"]}
        self.assertLess(d["coolant_flow"]["z"], -3); self.assertGreater(d["batt_temp"]["z"], 2); self.assertTrue(f["dc_fast_charge"]); self.assertIn("pump", f["causes"][0].lower())


@unittest.skipUnless(HAVE, WHY)
class Brain(unittest.TestCase):
    def test_brief_questions_route_on_the_fast_path(self):
        for q, want in (("Why has my driving range reduced over the last two weeks?", "WHY_RANGE"), ("When did fuel efficiency show the largest drop?", "LARGEST"),
                        ("Show all ADAS disengagement events from yesterday.", "EVENTS"), ("What happened before the battery overheating warning?", "BEFORE_EVENT"),
                        ("Compare energy consumption between city and highway driving.", "COMPARE")):
            i, p, conf = brain.parse_fast(q); self.assertEqual(i, want, q); self.assertGreaterEqual(conf, brain.FAST_MIN, q)

    def test_out_of_domain_never_reaches_tools(self):
        for q in ("Play some music", "Tell me a joke", "Translate good morning into French"):
            r = brain.Brain(DB).ask(q); self.assertEqual(r["intent"], "NONE", q); self.assertIn(r["path"], ("gate", "unsupported", "schema"), q)          # "schema": the request names something this vehicle does not record, so no tool runs

    def test_numbers_in_answers_come_from_tools(self):
        r = brain.Brain(DB).ask("Show all ADAS disengagement events from yesterday."); self.assertEqual(r["facts"]["count"], TRUTH["events_counts"]["ADAS_DISENGAGE"]["y"])


@unittest.skipUnless(HAVE, WHY)
class CanHealth(unittest.TestCase):
    def test_flood_and_unknown_id_detected_no_false_alarm_on_clean(self):
        m = canhealth.CanHealth()
        for s in range(50):
            for k in range(10): m.learn(s + k / 10, "0316")
            m.learn(s + .5, "02c0")
        m.finish_learning(); out = []
        for s in range(20):                                              # clean traffic: no alert
            for k in range(10): out += m.feed(100 + s + k / 10, "0316")
        self.assertEqual(out, [])
        for k in range(200): out += m.feed(130 + k / 200, "0000")        # flood of an unknown ID
        for k in range(60): out += m.feed(131 + k / 60, "0316")          # 6x rate spike on a known ID
        out += m.feed(133, "0316")
        self.assertTrue(any(a[1] == "UNKNOWN_ID" for a in out)); self.assertTrue(any(a[1] == "RATE_SPIKE" for a in out))


@unittest.skipUnless(os.name == "nt", "Windows Job Objects")
@unittest.skipUnless(HAVE, WHY)
class Governor(unittest.TestCase):
    def test_job_memory_limit_is_really_enforced(self):
        code = ("import sys; sys.path.insert(0, r'%s')\nfrom vgpt import governor\ngovernor.apply(mem_mb=150)\n"
                "try:\n    b = bytearray(400 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); print('ALLOCATED')\nexcept MemoryError:\n    print('BLOCKED')\n") % ROOT
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
        self.assertIn("BLOCKED", out)


if __name__ == "__main__":
    unittest.main()

"""The live side: a virtual car drives, its profile's bus emitter produces wire-format traffic, collectors carry it, the runtime stores it, and the assistant's database answers from it.

  * the car's physics is sane (energy balance, speed limits, no violent driving) on an electric car and a diesel truck
  * every concept a vehicle's profile binds is produced by the car, travels as raw frames, and arrives in the database - for all three bundled vehicles (plug and play on live data)
  * every injectable fault leaves the evidence a real vehicle would leave, through the real decoders (events, codes, signal behaviour, bus silence)
  * collectors (a JSON-lines file, a candump file, an NMEA file, garbage) deliver what was sent and survive what was not; there is no network collector (the product is fully offline)
  * the runtime freezes event context, derives trips and charges, notes silent buses, exposes a bus monitor"""
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.live import collect as C
from vgpt.live.car import FAULTS, VirtualCar
from vgpt.live.runtime import Runtime
from vgpt.live.sim import Simulation
from vgpt.protocols import canframes as cf
from vgpt.vehicle import lake as LK, provision as PV
from vgpt.vehicle.profile import Profile

T0 = 1792364400
HOME, WORK = (51.51257, -0.04426), (51.70886, -0.2739)


def rig(name="aster_ev", init=None, seed=3, tag=""):
    db = os.path.join(tempfile.gettempdir(), f"tl_{name}{tag}.db")
    PV.provision(name, db, replace=True); p = Profile.load(name); rt = Runtime(db, profile=p)
    car = VirtualCar(p, seed=seed, t0=T0, tz_h=1.0, init=init or {"battery_soc": 70.0, "ambient_temperature": 18.0, "odometer": 52557.2}, home=HOME, work=WORK)
    return db, rt, car, Simulation(rt, car)


def run(sim, seconds, until=None):
    for i in range(seconds):
        sim.step()
        if until is not None and until(): break
    return i + 1


def events(db, *types):
    c = sqlite3.connect(db); q = "SELECT ts, type, detail FROM events" + (" WHERE type IN (%s)" % ",".join("?" * len(types)) if types else "") + " ORDER BY ts"
    try: return c.execute(q, types).fetchall()
    finally: c.close()


class Physics(unittest.TestCase):
    def drive(self, name):
        p = Profile.load(name); car = VirtualCar(p, seed=3, t0=T0, tz_h=1.0, home=HOME, work=WORK); car.start_trip(); worst_lat = worst_dec = maxv = 0.0; soc0, fuel0, s = car.soc, car.fuel, 0
        for s in range(4000):
            car.step(); worst_lat = max(worst_lat, abs(car.a_lat)); worst_dec = min(worst_dec, car.a_lon); maxv = max(maxv, car.v * 3.6)
            if car.arrived: break
        return car, s, worst_lat, worst_dec, maxv, soc0, fuel0

    def test_the_electric_car_drives_a_route_sanely(self):
        car, s, lat, dec, vmax, soc0, _ = self.drive("aster_ev")
        self.assertTrue(car.arrived); self.assertLess(s, 2400); self.assertGreater(car.trip_km, 25.0); self.assertLess(lat, 3.4, "no violent cornering"); self.assertGreater(dec, -4.0, "no violent braking beyond an occasional cut-in"); self.assertLess(vmax, 115.0)
        wh_km = (soc0 - car.soc) / 100.0 * car.cap * 1000.0 / car.trip_km; self.assertTrue(130 < wh_km < 260, f"{wh_km:.0f} Wh/km")

    def test_the_diesel_truck_burns_fuel_and_shifts(self):
        car, s, lat, dec, vmax, _, fuel0 = self.drive("rhino_diesel")
        self.assertTrue(car.arrived); used = (fuel0 - car.fuel) / 100.0 * car.tank; l100 = used / car.trip_km * 100.0; self.assertTrue(10.0 < l100 < 45.0, f"{l100:.1f} l/100 km"); self.assertGreater(car.t_motor, 80.0)

    def test_charging_tapers_and_stops_at_the_target(self):
        db, rt, car, sim = rig(init={"battery_soc": 20.0, "ambient_temperature": 18.0}, tag="_chg"); sim.control("plug", kind="dc", to=90); kws = []
        for i in range(3600):
            sim.step()
            if i % 60 == 0 and car.charge_state == 1: kws.append(car.charge_kw)
            if car.charge_state == 2: break
        self.assertGreaterEqual(car.soc, 89.9); self.assertGreater(max(kws), 90.0); self.assertLess(kws[-1], 0.5 * max(kws), "the charge power tapers near the top"); self.assertLess(car.t_batt, 45.0, "a healthy pump keeps the pack cool")
        rt.close()

    def test_a_parked_car_sleeps_and_wakes_once_a_minute(self):
        db, rt, car, sim = rig(tag="_park"); awake = sum(1 for _ in range(600) if (sim.step() or True) and car.awake()); self.assertTrue(20 <= awake <= 40, awake); rt.close()


class Plug(unittest.TestCase):
    """Any vehicle: the car produces every bound concept, the raw traffic carries it, the database holds it."""
    def check(self, name, init=None):
        db, rt, car, sim = rig(name, init=init, tag="_plug"); car.start_trip(); run(sim, 420); got = set(); p = car.p
        st = car.state()
        missing = sorted(c for c in {b.concept for b in p.bindings} if c not in st and not c.startswith("_")); self.assertEqual(missing, [], f"{name}: the car does not produce {missing}")
        rt.publish(); latest = {s["concept"] for s in rt.signals()}; bound = {b.concept for b in p.bindings}
        self.assertEqual(sorted(bound - latest), [], f"{name}: concepts bound but never decoded")
        c = sqlite3.connect(db); n = c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0]; self.assertGreater(n, 300)
        lake = {r[0] for r in c.execute("SELECT s.concept FROM sig_catalog s JOIN lk_1s l ON l.sid=s.sid GROUP BY s.sid")}
        self.assertGreater(len(lake), 0.6 * len(LK.lake_concepts(p))); self.assertEqual(rt.stats()["errors"], {}); self.assertEqual(rt.stats()["dropped_late"], 0)
        rt.close(); return db

    def test_the_electric_hatchback(self): self.check("aster_ev")
    def test_the_second_electric_car_with_other_buses(self): self.check("meridian_ev")
    def test_the_diesel_truck_on_j1939(self): self.check("rhino_diesel")
    def test_the_petrol_hatchback_on_obd_style_data(self): self.check("kestrel_petrol")

    def test_a_whole_trip_becomes_one_ledger_entry_with_the_right_energy(self):
        db, rt, car, sim = rig(tag="_trip"); soc0 = car.soc; car.start_trip(); run(sim, 3000, until=lambda: car.arrived); run(sim, 400); rt.publish(); rt.close()
        c = sqlite3.connect(db); trips = c.execute("SELECT start, end, km, kwh, wh_km FROM trips").fetchall(); self.assertEqual(len(trips), 1)
        km, kwh = trips[0][2], trips[0][3]; self.assertAlmostEqual(km, car.trip_km, delta=0.6); truth = (soc0 - car.soc) / 100.0 * car.cap * car.soh / 100.0 * 1.0
        self.assertAlmostEqual(kwh, truth, delta=0.15 * truth + 0.1)
        # the ledger's distance equals the odometer difference the bus reported
        odo = c.execute("SELECT MIN(v), MAX(v) FROM lk_1s WHERE sid=(SELECT sid FROM sig_catalog WHERE concept='odometer')").fetchone(); self.assertAlmostEqual(odo[1] - odo[0], km, delta=0.8)


class Faults(unittest.TestCase):
    def test_every_fault_in_the_catalogue_is_injectable_and_clearable(self):
        db, rt, car, sim = rig(tag="_cat"); car.start_trip(); run(sim, 200)
        for name in FAULTS: sim.inject(name); run(sim, 5); sim.clear(name)
        self.assertEqual(rt.stats()["errors"], {}); rt.close()

    def test_a_degraded_pump_heats_the_pack_and_stores_its_code(self):
        db, rt, car, sim = rig(init={"battery_soc": 15.0, "ambient_temperature": 30.0}, tag="_pump"); car.t_batt = 31.0; sim.inject("coolant_pump", factor=0.2); sim.control("plug", kind="dc", to=100)
        peak = 0.0
        for _ in range(2400):
            sim.step(); peak = max(peak, car.t_batt)
            if car.charge_state == 2: break
        run(sim, 60); rt.publish(); ev = {e[1] for e in events(db)}
        self.assertGreater(peak, 45.0); self.assertIn("WARN_BATT_OVERTEMP", ev); self.assertIn("DTC_SET", ev); self.assertTrue(any("P1A00" in e[2] for e in events(db, "DTC_SET")))
        c = sqlite3.connect(db); self.assertGreater(c.execute("SELECT COUNT(*) FROM event_ctx_x WHERE type='WARN_BATT_OVERTEMP'").fetchone()[0], 0, "context frozen"); rt.close()

    def test_a_slow_puncture_ends_in_a_tyre_warning(self):
        db, rt, car, sim = rig(tag="_tyre"); car.start_trip(); sim.inject("tyre_leak", wheel=2, kpa_min=6.0); run(sim, 1500, until=lambda: car.arrived); run(sim, 300); rt.publish(); rt.close()
        low = events(db, "TPMS_LOW"); self.assertTrue(low and "rear left" in low[0][2], low); self.assertLess(car.pc[2], car.pc[0] - 30.0)

    def test_the_watchdog_reset_and_the_disengagement_come_together(self):
        db, rt, car, sim = rig(tag="_ecu"); car.start_trip(); run(sim, 700, until=lambda: car.adas_on); self.assertTrue(car.adas_on, "lane keeping engaged on the motorway"); run(sim, 30)
        t = car.t; sim.inject("ecu_reset"); run(sim, 20); rt.publish(); rt.close()
        reset = events(db, "ECU_RESET"); dis = events(db, "ADAS_DISENGAGE"); self.assertTrue(reset and "ADAS domain controller" in reset[-1][2]); self.assertTrue(any(0 <= d[0] - reset[-1][0] <= 5 and "system limit" in d[2] for d in dis), (reset, dis))
        c = sqlite3.connect(db); self.assertTrue(any("heartbeat" in r[0] for r in c.execute("SELECT msg FROM logs")))

    def test_an_unexpected_shutdown_leaves_a_gap_and_its_codes(self):
        db, rt, car, sim = rig(tag="_off"); car.start_trip(); run(sim, 500); last_ts = int(car.t); sim.inject("shutdown"); run(sim, 40); rt.publish(); rt.close()
        ev = events(db); types = [e[1] for e in ev]; self.assertIn("UNEXPECTED_SHUTDOWN", types); self.assertIn("VOLTAGE_DIP", types); codes = " ".join(e[2] for e in ev if e[1] == "DTC_SET"); self.assertIn("P0562", codes); self.assertIn("U0155", codes)
        c = sqlite3.connect(db); gap = c.execute("SELECT MIN(ts), MAX(ts) FROM sig_1s WHERE ts>? AND ts<?", (last_ts + 1, last_ts + 40)).fetchone()
        rows = [r[0] for r in c.execute("SELECT ts FROM sig_1s WHERE ts>? ORDER BY ts", (last_ts,))]; holes = [b - a for a, b in zip(rows, rows[1:])]; self.assertGreaterEqual(max(holes), 9, "no samples while the supply was lost")
        self.assertTrue(any(e[1] == "ECU_RESET" and "gateway" in e[2] for e in ev))

    def test_two_sources_that_disagree_are_reported_with_both_values(self):
        db, rt, car, sim = rig(tag="_conf"); car.start_trip(); sim.inject("speed_conflict", gain=1.08); run(sim, 90); sim.inject("ambient_glitch"); run(sim, 60); rt.publish(); rt.close()
        det = [e[2] for e in events(db, "SENSOR_CONFLICT")]; self.assertTrue(any("vehicle speed" in d and "vhal/PERF_VEHICLE_SPEED" in d for d in det), det); self.assertTrue(any("outside temperature" in d and "-40" in d for d in det), det)

    def test_a_blocked_camera_is_noticed_and_lane_keeping_drops(self):
        db, rt, car, sim = rig(tag="_cam"); car.start_trip(); run(sim, 650, until=lambda: car.adas_on); self.assertTrue(car.adas_on); sim.inject("camera_blocked"); run(sim, 90); rt.publish(); rt.close()
        ty = {e[1] for e in events(db)}; self.assertIn("CAMERA_BLOCKED", ty); self.assertIn("CAMERA_DEGRADED", ty); self.assertTrue(any("camera blocked" in e[2] for e in events(db, "ADAS_DISENGAGE")))

    def test_a_body_bus_dropout_is_error_frames_a_silent_source_and_a_code(self):
        db, rt, car, sim = rig(tag="_bus"); car.start_trip(); run(sim, 120); sim.inject("can_burst"); run(sim, 70); rt.publish(); rt.close()
        ty = [e[1] for e in events(db)]; self.assertIn("CAN_ERROR", ty); self.assertTrue(any("U0140" in e[2] for e in events(db, "DTC_SET")))
        c = sqlite3.connect(db); log = c.execute("SELECT source, status, detail FROM source_log ORDER BY ts").fetchall(); self.assertTrue(any(r[0] == "body" and r[1] == "silent" for r in log), log); self.assertTrue(any(r[0] == "body" and r[1] == "restored" for r in log), log)
        gap = c.execute("SELECT COUNT(*) FROM lk_1s WHERE sid=(SELECT sid FROM sig_catalog WHERE concept='tyre_pressure_fl') AND ts BETWEEN ? AND ?", (T0 + 130, T0 + 145)).fetchone()[0]; self.assertEqual(gap, 0, "the body-bus signals vanished")

    def test_automatic_emergency_braking_is_a_hard_brake_too(self):
        db, rt, car, sim = rig(tag="_aeb"); car.start_trip(); run(sim, 300); sim.inject("aeb"); run(sim, 12); rt.publish(); rt.close()
        self.assertTrue(events(db, "AEB_ACTIVATION")); self.assertTrue(events(db, "HARSH_BRAKE"), "the router derives hard braking from the acceleration signal")

    def test_a_dashboard_range_revision_shows_in_the_reported_range_only(self):
        db, rt, car, sim = rig(tag="_rng"); car.start_trip(); run(sim, 500); before = car.reported_range(); soc = car.soc; sim.inject("range_glitch"); run(sim, 600); after = car.reported_range(); soc2 = car.soc
        self.assertLess(after, before * 0.93); self.assertLess(soc - soc2, 8.0, "the energy used is normal"); rt.close()

    def test_the_diesel_has_its_own_events(self):
        db, rt, car, sim = rig("rhino_diesel", tag="_dz"); car.start_trip(); run(sim, 300); sim.inject("ecu_reset"); sim.inject("brownout"); sim.inject("tyre_leak", wheel=0, kpa_min=8.0); run(sim, 1500, until=lambda: car.arrived); run(sim, 200); rt.publish(); rt.close()
        ty = {e[1] for e in events(db)}; self.assertIn("ECU_RESET", ty); self.assertIn("VOLTAGE_DIP", ty); self.assertIn("TPMS_LOW", ty)


class Collectors(unittest.TestCase):
    def traffic(self, seconds=30):
        """(ts, bus, payload) of a short drive, captured from the emitter."""
        db, rt, car, sim = rig(tag="_cap"); out = []; sim.sink = lambda ts, bus, payload: out.append((ts, bus, payload)); car.start_trip(); run(sim, seconds); rt.close(); return out

    def test_a_record_survives_the_json_round_trip(self):
        for ts, bus, payload in self.traffic(6)[:600:7]:
            ts2, bus2, p2 = C.decode_record(json.loads(json.dumps(C.encode_record(ts, bus, payload)))); self.assertEqual(bus, bus2); self.assertAlmostEqual(ts, ts2, places=3)
            if bus == "can": self.assertEqual((payload.can_id, payload.data, payload.fd), (p2.can_id, p2.data, p2.fd))
            elif bus == "someip": self.assertEqual(bytes(payload), p2)

    def test_malformed_records_are_rejected_not_fatal(self):
        for bad in ({}, {"bus": "can"}, {"bus": "warp", "payload": {}, "ts": 1}, {"bus": "can", "payload": "x", "ts": 1}, {"bus": "can", "payload": {"id": 1, "data": "zz"}, "ts": 1}, {"bus": "log", "payload": "x", "ts": float("nan")}, {"bus": "log", "payload": 5, "ts": 1}):
            with self.assertRaises((ValueError, KeyError, TypeError)): C.decode_record(bad)

    def pipeline(self, tag):
        db = os.path.join(tempfile.gettempdir(), f"tl_col_{tag}.db"); PV.provision("aster_ev", db, replace=True); rt = Runtime(db, profile=Profile.load("aster_ev")); return db, rt

    def test_a_jsonl_file_delivers_the_same_database_as_feeding_directly(self):
        traffic = self.traffic(60); db1, rt1 = self.pipeline("direct"); [rt1.feed(*m) for m in traffic]; rt1.close()
        path = os.path.join(tempfile.gettempdir(), "tl_jsonl_same.jsonl")
        with open(path, "w") as f:
            for m in traffic: f.write(json.dumps(C.encode_record(*m)) + "\n")
        db2, rt2 = self.pipeline("file"); col = C.FileTail(rt2.feed, path, fmt="jsonl", follow=False); col.start(); col.done.wait(60); col.join(2); rt2.close()
        self.assertEqual(col.n, len(traffic)); self.assertEqual(col.errors, 0)
        a = sqlite3.connect(db1).execute("SELECT COUNT(*), AVG(speed), AVG(soc) FROM sig_1s").fetchone(); b = sqlite3.connect(db2).execute("SELECT COUNT(*), AVG(speed), AVG(soc) FROM sig_1s").fetchone()
        self.assertLessEqual(abs(a[0] - b[0]), 2); self.assertAlmostEqual(a[1], b[1], delta=15.0)

    def test_a_candump_file_is_replayed_into_the_database(self):
        traffic = [m for m in self.traffic(40) if m[1] == "can"]; path = os.path.join(tempfile.gettempdir(), "tl_candump.log")
        with open(path, "w") as f:
            for ts, _, fr in traffic:
                ch = fr.channel or "can0"
                f.write(f"({fr.ts:.6f}) {ch} {fr.can_id:03X}{'#' if not fr.fd else '##0'}{fr.data.hex().upper()}\n" if not fr.extended else f"({fr.ts:.6f}) {ch} {fr.can_id:08X}#{fr.data.hex().upper()}\n")
            f.write("this line is garbage\n")
        db, rt = self.pipeline("dump"); col = C.FileTail(rt.feed, path, fmt="candump", follow=False); col.start(); col.done.wait(20); col.join(2); rt.close()
        self.assertGreater(col.n, len(traffic) * 0.9); c = sqlite3.connect(db); self.assertGreater(c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0], 20); self.assertGreater(c.execute("SELECT MAX(speed) FROM sig_1s").fetchone()[0], 100)

    def test_a_tailed_file_picks_up_lines_as_they_are_appended(self):
        traffic = self.traffic(10); path = os.path.join(tempfile.gettempdir(), "tl_tail.jsonl"); open(path, "w").close(); db, rt = self.pipeline("tail"); col = C.FileTail(rt.feed, path, fmt="jsonl", follow=True); col.start()
        with open(path, "a") as f:
            for m in traffic[:200]: f.write(json.dumps(C.encode_record(*m)) + "\n")
            f.flush()
        t = time.time()
        while col.n < 200 and time.time() - t < 10: time.sleep(0.05)
        col.stop(); col.join(2); rt.close(); self.assertEqual(col.n, 200)

    def test_an_nmea_file_gives_gps_rows(self):
        from vgpt.protocols import nmea
        path = os.path.join(tempfile.gettempdir(), "tl_nmea.txt")
        with open(path, "w") as f:
            for i in range(30): f.write(nmea.encode_rmc(T0 + i * 0.5, 51.5 + i * 1e-4, -0.1, 50.0, 90.0) + "\n")
            f.write("$GPRMC,garbage*00\n")
        db, rt = self.pipeline("nmea"); col = C.FileTail(rt.feed, path, fmt="nmea", follow=False, clock=iter(T0 + i * 0.5 for i in range(10 ** 6)).__next__); col.start(); col.done.wait(20); col.join(2); rt.close()
        self.assertGreater(col.n, 20); self.assertGreater(sqlite3.connect(db).execute("SELECT COUNT(*) FROM gps_1s").fetchone()[0], 3)

    def test_garbage_does_not_stop_the_service(self):
        path = os.path.join(tempfile.gettempdir(), "tl_junk.jsonl")
        with open(path, "wb") as f:
            for junk in (b"\x00\xff\xfe", b"{not json", b'{"bus":"can","payload":{"id":1,"data":"00"},"ts":1}', b"[]", b'{"bus":"can","ts":1,"payload":{"id":2047,"data":"0011","ch":"can0"}}'): f.write(junk + b"\n")
        db, rt = self.pipeline("junk"); col = C.FileTail(rt.feed, path, fmt="jsonl", follow=False); col.start(); col.done.wait(20); col.join(2); self.assertGreater(col.errors, 2); rt.close()


class RuntimeServices(unittest.TestCase):
    def test_the_bus_monitor_names_every_message_and_shows_what_it_decoded_to(self):
        db, rt, car, sim = rig(tag="_mon"); car.start_trip(); run(sim, 90); table = rt.mon.table(); names = {r["name"] for r in table}; rt.close()
        for want in ("VehDyn", "BmsStatus", "Tpms", "AdasState.SystemState", "PERF_VEHICLE_SPEED"): self.assertIn(want, names, sorted(names))
        row = next(r for r in table if r["name"] == "VehDyn"); self.assertEqual(row["source"], "pt"); self.assertGreater(row["rate_hz"], 8.0); self.assertTrue(any(d.startswith("speed=") for d in row["decoded"])); self.assertRegex(row["raw"], r"^[0-9A-F]{2}( [0-9A-F]{2})+$")

    def test_the_snapshot_lists_every_source_with_its_state(self):
        db, rt, car, sim = rig(tag="_snap"); car.start_trip(); run(sim, 120); snap = rt.snapshot(); rt.close()
        by = {s["id"]: s for s in snap["sources"]}; self.assertEqual(set(by), set(car.p.sources)); self.assertTrue(all(by[k]["state"] == "ok" for k in ("pt", "bat", "body", "eth", "hal", "cam", "gnss")), {k: v["state"] for k, v in by.items()})
        self.assertGreater(snap["stats"]["msgs"], 10000); self.assertLess(snap["stats"]["decode_us"], 400)

    def test_context_is_frozen_thirty_seconds_after_an_event(self):
        db, rt, car, sim = rig(tag="_ctx"); car.start_trip(); run(sim, 300); sim.inject("ecu_reset"); run(sim, 20); c = sqlite3.connect(db); n0 = c.execute("SELECT COUNT(*) FROM event_ctx_x").fetchone()[0]; run(sim, 25); rt.publish()
        rows = c.execute("SELECT ts, type, blob FROM event_ctx_x WHERE type='ECU_RESET'").fetchall(); self.assertTrue(rows, "frozen once the post-event window exists"); import zlib; blob = json.loads(zlib.decompress(rows[0][2]))
        self.assertEqual(blob["hz"], 1); self.assertEqual(blob["t0"], rows[0][0] - 120); self.assertIn("speed", blob["sig"]); self.assertEqual(len(blob["sig"]["speed"]), 150); rt.close()

    def test_a_day_boundary_runs_the_housekeeping(self):
        db, rt, car, sim = rig(tag="_day"); car.t = T0 + 82800 - 5; car.start_trip(); run(sim, 600); rt.publish()                                    # crosses local midnight (tz +1 h)
        self.assertEqual(rt.stats()["errors"], {}); rt.close()


WORLD = os.path.join(ROOT, "data", "world_aster.db")


@unittest.skipUnless(os.path.exists(WORLD), "needs data/world_aster.db (python -m vgpt.world.build --out data/world_aster.db)")
class Served(unittest.TestCase):
    """The head-unit server with a live car: the cluster, the Live page's endpoints and the camera, over HTTP, on the 105-day history."""
    @classmethod
    def setUpClass(cls):
        import importlib.util
        from types import SimpleNamespace
        from tests.localclient import LocalClient
        os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp()
        spec = importlib.util.spec_from_file_location("uiserver_live", os.path.join(ROOT, "ui", "server.py")); cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)
        from tests.stub_planner import StubPlanner
        cls.app = cls.m.App(SimpleNamespace(db="data/world_aster.db", model="none", camera="clip", webcam_index=0, live=os.path.join(tempfile.mkdtemp(), "live.db")), planner_factory=lambda conn: StubPlanner(conn, {}))
        cls.c = LocalClient(cls.m.make_dispatcher(cls.app))

    @classmethod
    def tearDownClass(cls):
        cls.app._stop_live()

    def get(self, p): return self.c.get_json(p)
    def post(self, p, b): return self.c.post_json(p, b)

    def test_the_whole_loop_from_the_buttons_to_the_cluster_and_the_database(self):
        s = self.get("/api/state"); self.assertTrue(s["live"]["on"]); self.assertTrue(s["tele"]["ok"]); self.assertEqual(s["tele"]["mode"], 0); self.assertEqual(s["scenarios"], {})
        self.assertTrue(self.post("/api/live/speed", {"x": 20})["ok"]); self.assertTrue(self.post("/api/live/control", {"cmd": "start_trip"})["ok"]); t0 = s["sim_t"]
        t = time.time()
        while time.time() - t < 30:
            s = self.get("/api/state")
            if s["tele"].get("speed", 0) > 30 and s["sim_t"] > t0 + 40: break
            time.sleep(0.5)
        self.assertGreater(s["tele"]["speed"], 30); self.assertIn(s["tele"]["mode"], (1, 2)); self.assertTrue(s["live"]["driving"]); self.assertGreater(s["sim_t"], t0 + 30, "the clock follows the car")
        L = self.get("/api/live"); self.assertTrue(L["on"]); ok = {x["id"]: x["state"] for x in L["sources"]}; self.assertEqual(ok["pt"], "ok"); self.assertEqual(ok["eth"], "ok"); self.assertGreater(len(L["bus"]), 15); self.assertGreater(len(L["signals"]), 40); self.assertEqual(L["stats"]["errors"], {})
        self.assertIn("coolant_pump", L["faults"]); self.assertTrue(self.post("/api/live/inject", {"fault": "ambient_glitch"})["ok"]); self.assertFalse(self.post("/api/live/inject", {"fault": "warp_core"})["ok"])
        self.assertFalse(self.post("/api/live/control", {"cmd": "self_destruct"})["ok"]); self.assertTrue(self.post("/api/live/clear", {})["ok"])
        self.post("/api/live/speed", {"x": 1})

    def test_the_virtual_camera_gives_jpeg_frames_one_at_a_time(self):
        st, h, data = self.c.request("GET", "/api/frame"); self.assertEqual(st, 200); self.assertEqual(h["content-type"], "image/jpeg"); self.assertEqual(data[:2], bytes([0xFF, 0xD8]))

    def test_a_dataset_without_a_vehicle_profile_has_no_live_car(self):
        if not os.path.exists(os.path.join(ROOT, "data", "vehicle.db")): self.skipTest("needs data/vehicle.db")
        self.post("/api/dataset", {"name": "vehicle.db"}); s = self.get("/api/state"); self.assertIsNone(s["live"]); self.assertFalse(self.get("/api/live")["on"]); self.assertFalse(self.post("/api/live/speed", {"x": 5})["ok"])
        self.post("/api/dataset", {"name": "world_aster.db"}); self.assertTrue(self.get("/api/state")["live"]["on"])


if __name__ == "__main__":
    unittest.main()

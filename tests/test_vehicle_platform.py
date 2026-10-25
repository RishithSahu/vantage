"""Plug-and-play vehicle platform: profile packs, DBC encoder, the generic router, the signal lake and the emitter, on three vehicles that differ in buses, names, units, byte order and powertrain.

The decisive test is the round trip: a canonical state is turned into the raw frames THAT vehicle would put on its buses (vgpt/world/emit.py), decoded again with nothing but its profile (vgpt/vehicle/router.py)
and compared.  If it passes on a Motorola / CAN-FD / imperial SUV and on a J1939 diesel as well as on the reference EV, a new vehicle is data, not code."""
import copy
import json
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import signal_registry as R, store
from vgpt.protocols import dbc, someip
from vgpt.vehicle import lake as LK
from vgpt.vehicle.profile import Profile, ProfileError
from vgpt.vehicle.router import ProfileRouter
from vgpt.world.emit import Emitter
from vgpt.world.typical import TYPICAL

PACKS = ("aster_ev", "meridian_ev", "rhino_diesel")
T0 = 1790000000


def state_for(profile):
    return {c: TYPICAL[c] for c in profile.concept_names() if c in TYPICAL} | {"_charging_active": 0, "_ignition": 1}


class DbcCodec(unittest.TestCase):
    def test_intel_known_vector(self):
        db = dbc.Database.load(os.path.join(ROOT, "vgpt", "protocols", "vehicle.dbc")); m = db.by_name("VehDyn")
        data = m.encode({"Speed": 57.30, "Gear": 3}); self.assertEqual(data[:3], bytes([0x62, 0x16, 3]))              # 5730 = 0x1662, little endian
        self.assertAlmostEqual(m.decode(data)["Speed"], 57.30, places=6)

    def test_motorola_known_vector(self):
        db = dbc.Database.load(os.path.join(ROOT, "vgpt", "protocols", "vehicle.dbc")); m = db.by_name("Charger")
        data = m.encode({"ChgKw": 11.0, "ChgState": "charging"}); self.assertEqual(data[:3], bytes([0x00, 0x6E, 0x01]))     # 110 = 0x006E big endian at bit 7
        d = m.decode(data); self.assertAlmostEqual(d["ChgKw"], 11.0, places=6); self.assertEqual(d["ChgState"], 1)

    def test_round_trip_signed_multiplexed_and_fd(self):
        db = dbc.Database.load(os.path.join(ROOT, "vgpt", "vehicle", "packs", "rhino_diesel", "j1939.dbc")); m = db.by_name("TIRE")
        for mux, name in ((0, "PressFL"), (3, "PressRR")):
            d = m.decode(m.encode({"TireLoc": mux, name: 248.0, "TireTemp": 31.0})); self.assertEqual(d[name], 248.0); self.assertAlmostEqual(d["TireTemp"], 31.0, places=1)
        db2 = dbc.Database.load(os.path.join(ROOT, "vgpt", "vehicle", "packs", "meridian_ev", "main.dbc")); m2 = db2.by_name("VehDynamics"); self.assertEqual(m2.dlc, 16)
        d = m2.decode(m2.encode({"SteerWhlAngle_deg": -123.4, "LatAcc_g": -0.5})); self.assertAlmostEqual(d["SteerWhlAngle_deg"], -123.4, places=1); self.assertAlmostEqual(d["LatAcc_g"], -0.5, places=3)


class ProfileValidation(unittest.TestCase):
    def test_bundled_packs_validate_without_errors(self):
        for name in PACKS:
            errors = [i for i in Profile.load(name).validate() if i.level == "error"]; self.assertEqual(errors, [], name)

    def mutate(self, fn):
        p = Profile.load("aster_ev"); spec = copy.deepcopy(p.spec); fn(spec)
        return Profile(spec, p.base).validate()

    def test_errors_carry_their_path(self):
        issues = self.mutate(lambda s: s["signals"].append({"concept": "warp_core", "from": "can:pt/VehDyn.Speed", "unit": "km/h"}))
        self.assertTrue(any(i.level == "error" and "warp_core" in i.message and i.path.startswith("signals[") for i in issues))
        issues = self.mutate(lambda s: s["signals"][0].update({"unit": "furlongs"}))
        self.assertTrue(any("cannot convert" in i.message for i in issues))
        issues = self.mutate(lambda s: s["signals"][0].update({"from": "can:pt/VehDyn.NoSuchSignal"}))
        self.assertTrue(any("NoSuchSignal" in i.message for i in issues))
        issues = self.mutate(lambda s: s["signals"][0].update({"from": "can:ghost/VehDyn.Speed"}))
        self.assertTrue(any("unknown source" in i.message for i in issues))
        issues = self.mutate(lambda s: s["signals"].append(dict(s["signals"][0])))
        self.assertTrue(any("priority" in i.message for i in issues))

    def test_retention_below_ten_weeks_is_flagged(self):
        issues = self.mutate(lambda s: s["retention"].update({"coarse_days": 30}))
        self.assertTrue(any(i.level == "warning" and "ten weeks" in i.message for i in issues))

    def test_an_invalid_pack_cannot_be_provisioned(self):
        p = Profile.load("aster_ev"); spec = copy.deepcopy(p.spec); spec["signals"][0]["unit"] = "furlongs"; bad = Profile(spec, p.base)
        c = store.open_rw(os.path.join(tempfile.gettempdir(), "vp_bad.db"))
        with self.assertRaises(ProfileError): LK.provision(c, bad)


class RoundTrip(unittest.TestCase):
    """canonical state -> raw frames of THAT vehicle -> profile router -> the same canonical values"""
    def run_pack(self, name, seconds=4):
        p = Profile.load(name); r = ProfileRouter(p); em = Emitter(p); st = state_for(p); last = {}; events = []
        for i in range(seconds):
            for ts, bus, payload in em.tick(T0 + i, st):
                for rec in r.decode(bus, payload, ts):
                    if rec[0] == "sig": last[rec[1]] = rec[2]
                    elif rec[0] == "event": events.append(rec)
        return p, st, last, events, r

    def check(self, name):
        p, st, last, events, r = self.run_pack(name); bad = []
        for b in p.bindings:
            c = b.concept
            if b.priority != 1 or b.when or c.startswith("_") or b.bus in ("uds",) or c not in st: continue
            if c not in last: bad.append((c, "never decoded")); continue
            tol = max(0.03 * abs(st[c]), 0.6 if abs(st[c]) > 5 else 0.06)
            if abs(last[c] - st[c]) > tol: bad.append((c, st[c], last[c]))
        self.assertEqual(bad, [], f"{name}: concepts that did not survive the wire")
        self.assertFalse([e for e in events if e[1] in ("SENSOR_CONFLICT", "SENSOR_INVALID")], f"{name}: spurious {events}")
        return p, st, last, r

    def test_reference_ev(self): self.check("aster_ev")
    def test_motorola_canfd_imperial_suv(self): self.check("meridian_ev")

    def test_j1939_diesel(self):
        p, st, last, r = self.check("rhino_diesel"); self.assertNotIn("battery_soc", last); self.assertIn("engine_rpm", last); self.assertEqual(last["_ignition"], 1.0)

    def test_the_ev_has_redundant_sources_and_the_diesel_has_none(self):
        r = self.run_pack("aster_ev")[4]; self.assertTrue(any(len(v) > 1 for v in r.slots.values()))
        r = self.run_pack("rhino_diesel")[4]; self.assertFalse(any(len(v) > 1 for v in r.slots.values()))


class Redundancy(unittest.TestCase):
    def setUp(self): self.p = Profile.load("aster_ev"); self.r = ProfileRouter(self.p, conflict_hold=3.0); self.em = Emitter(self.p)

    def feed(self, secs, st, only=None):
        out = []
        for i in secs:
            for ts, bus, payload in self.em.tick(T0 + i, st):
                if only and bus not in only: continue
                out += self.r.decode(bus, payload, ts)
        return out

    def test_disagreeing_sources_raise_one_conflict_with_both_values(self):
        st = state_for(self.p); st["speed@2"] = 20.0                                   # the VHAL speed disagrees with the CAN speed (57)
        out = self.feed(range(12), st)
        ev = [r for r in out if r[0] == "event" and r[1] == "SENSOR_CONFLICT"]; self.assertEqual(len(ev), 1)
        self.assertIn("57", ev[0][3]); self.assertIn("20", ev[0][3]); self.assertIn("vehicle speed", ev[0][3])

    def test_agreeing_sources_raise_nothing(self):
        out = self.feed(range(12), state_for(self.p)); self.assertFalse([r for r in out if r[0] == "event" and r[1] == "SENSOR_CONFLICT"])

    def test_the_secondary_source_speaks_when_the_primary_is_silent(self):
        st = state_for(self.p); self.feed(range(3), st)
        out = self.feed(range(10, 14), st, only=("vhal",)); speeds = [r[2] for r in out if r[0] == "sig" and r[1] == "speed"]
        self.assertTrue(speeds and abs(speeds[-1] - 57.3) < 1.0)


class Plausibility(unittest.TestCase):
    def test_an_impossible_value_is_dropped_and_reported_once(self):
        p = Profile.load("aster_ev"); r = ProfileRouter(p); em = Emitter(p); st = state_for(p); st["battery_temperature"] = 150.0           # the BMS reports 150 C (range -40..90)
        out = []
        for i in range(5):
            for ts, bus, payload in em.tick(T0 + i, st): out += r.decode(bus, payload, ts)
        self.assertFalse([x for x in out if x[0] == "sig" and x[1] == "battery_temperature"])
        self.assertEqual(len([x for x in out if x[0] == "event" and x[1] == "SENSOR_INVALID"]), 1); self.assertGreater(r.stats["bat"]["rejected"], 0)


class Events(unittest.TestCase):
    def test_discrete_events_come_out_as_the_vehicle_would_emit_them(self):
        p = Profile.load("aster_ev"); r = ProfileRouter(p); em = Emitter(p); got = {}
        for ev in ({"type": "ADAS_DISENGAGE", "reason": "lane marking lost", "speed_kmh": 97}, {"type": "ECU_RESET", "ecu": "ADAS domain controller", "cause": "watchdog timeout"},
                   {"type": "VOLTAGE_DIP", "duration_ms": 180, "min_v": 8.4}, {"type": "UNEXPECTED_SHUTDOWN", "cause": "12 V supply loss", "last_valid_ms": 250}, {"type": "DTC_SET", "code": "U0100"},
                   {"type": "WARN_BATT_OVERTEMP", "log": "Battery temperature warning raised (>45C)", "source": "thermal_mgr", "severity": "critical"}):
            for ts, bus, payload in em.event(T0, ev):
                for rec in r.decode(bus, payload, ts):
                    if rec[0] == "event": got[rec[1]] = rec
        self.assertEqual(got["ADAS_DISENGAGE"][3], "lane marking lost"); self.assertIn("ADAS domain controller reset (watchdog timeout)", got["ECU_RESET"][3])
        self.assertIn("8.4 V", got["VOLTAGE_DIP"][3]); self.assertIn("12 V supply loss", got["UNEXPECTED_SHUTDOWN"][3]); self.assertIn("Lost communication with powertrain control module", got["DTC_SET"][3])
        self.assertEqual(got["WARN_BATT_OVERTEMP"][2], "critical")

    def test_signal_rules_raise_tyre_and_overtemperature_warnings_after_the_hold_time(self):
        p = Profile.load("aster_ev"); r = ProfileRouter(p); em = Emitter(p); st = state_for(p); st["tyre_pressure_rl"] = 170.0; st["battery_temperature"] = 47.0; out = []
        for i in range(90):
            for ts, bus, payload in em.tick(T0 + i, st): out += r.decode(bus, payload, ts)
        ev = {x[1]: x for x in out if x[0] == "event"}; self.assertIn("TPMS_LOW", ev); self.assertIn("rear left", ev["TPMS_LOW"][3]); self.assertIn("WARN_BATT_OVERTEMP", ev)

    def test_hard_braking_is_detected_from_the_acceleration_signal_with_its_strength_and_speed(self):
        p = Profile.load("aster_ev"); r = ProfileRouter(p); em = Emitter(p); st = state_for(p); out = []
        for i in range(8):
            st["longitudinal_acceleration"] = -4.2 if 2 <= i <= 3 else 0.2                  # -0.43 g for two seconds
            for ts, bus, payload in em.tick(T0 + i, st): out += r.decode(bus, payload, ts)
        ev = [x for x in out if x[0] == "event" and x[1] == "HARSH_BRAKE"]; self.assertEqual(len(ev), 1); self.assertRegex(ev[0][3], r"0\.4\d g at 57 km/h")


class LakeAndAssistant(unittest.TestCase):
    """provision -> live ingest through the router -> the typed engine answers about signals nobody wrote a column for"""
    @classmethod
    def build(cls, name, seconds=240):
        p = Profile.load(name); path = os.path.join(tempfile.gettempdir(), f"vp_{name}.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(path + e)
            except OSError: pass
        c = store.open_rw(path); LK.provision(c, p); ing = store.Ingestor(c, ProfileRouter(p), lake=LK.Lake(c), track_meta=True); em = Emitter(p); st = state_for(p)
        for i in range(seconds):
            st["speed"] = 40 + 20 * (i % 60) / 60.0
            for ts, bus, payload in em.tick(T0 + i, st): ing.feed(ts, bus, payload)
        ing.close(); c.close(); return p, path

    def ask(self, path, text_frames):
        from vgpt.meaning.frame import Frame
        from vgpt.meaning import glossary as GL
        from vgpt.meaning.pipeline import MeaningBrain

        class Stub:
            glossary = GL.load()
            def parse(self, text, hint=None):
                f = text_frames.get(text.lower(), Frame("ambiguous", "none")); return f, {"raw": f.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}
        try:
            b = MeaningBrain(path, Stub(), now=T0 + 300); return [b.ask(q) for q in text_frames]
        finally: LK.release()

    def test_the_reference_ev_answers_from_core_and_lake_signals(self):
        from vgpt.meaning.frame import Frame
        p, path = self.build("aster_ev")
        r = self.ask(path, {"what is my steering angle": Frame("now", "steering_angle"), "what was my average speed": Frame("stat", "speed", "mean"), "what is my tyre pressure": Frame("now", "tyre_pressure"),
                            "what is my 12 volt battery": Frame("now", "aux_battery_voltage")})
        self.assertIn("own recording of this signal", r[0]["answer"]); self.assertNotIn("lkv_", r[0]["answer"]); self.assertIn("km/h", r[1]["answer"]); self.assertIn("kPa", r[2]["answer"]); self.assertIn("13.8", r[3]["answer"])

    def test_the_diesel_answers_about_engine_signals_and_says_it_has_no_traction_battery(self):
        from vgpt.meaning.frame import Frame
        p, path = self.build("rhino_diesel")
        r = self.ask(path, {"what is my engine rpm": Frame("now", "engine_rpm"), "what is the fuel rate": Frame("now", "fuel_rate"), "what is my oil temperature": Frame("now", "engine_oil_temperature"),
                            "what is my battery level": Frame("now", "battery_soc")})
        self.assertIn("1,750", r[0]["answer"].replace(" ", "")); self.assertIn("14.5", r[1]["answer"]); self.assertIn("96", r[2]["answer"]); self.assertIn("no traction battery", r[3]["answer"]); self.assertIn("Fuel level is", r[3]["answer"])          # asked for a battery level, a diesel says it has none and gives what it does have
        self.assertEqual(r[2]["facts"]["frame"]["concept"], "engine_oil_temperature")
        c = sqlite3.connect(path); self.assertEqual(c.execute("SELECT v FROM meta WHERE k='vehicle.id'").fetchone()[0], "rhino_diesel")

    def test_the_imperial_suv_stores_canonical_units(self):
        from vgpt.meaning.frame import Frame
        p, path = self.build("meridian_ev")
        r = self.ask(path, {"what is the brake temperature": Frame("now", "brake_temperature"), "what is my tyre pressure": Frame("now", "tyre_pressure")})
        self.assertIn("95", r[0]["answer"]); self.assertIn("kPa", r[1]["answer"])                  # psi on the wire, kPa in the store (the display unit is the person's choice)

    @classmethod
    def tearDownClass(cls):
        from vgpt.vehicle import lake
        from vgpt import schema_overlay
        if lake._ADOPTED["token"] is not None: schema_overlay.restore(lake._ADOPTED["token"]); lake._ADOPTED.update(id=None, token=None)
        from vgpt.meaning import ontology; ontology.reload()


class Retention(unittest.TestCase):
    def test_a_provisioned_database_keeps_ten_weeks_and_more(self):
        p = Profile.load("aster_ev"); path = os.path.join(tempfile.gettempdir(), "vp_ret.db")
        for e in ("", "-wal", "-shm"):
            try: os.remove(path + e)
            except OSError: pass
        c = store.open_rw(path); LK.provision(c, p); sid = c.execute("SELECT sid FROM sig_catalog WHERE concept='steering_angle'").fetchone()[0]; now = T0 + 130 * 86400
        for days in (3, 20, 71, 100, 119, 125): c.execute("INSERT INTO lk_1m VALUES(?,?,?,?,?,?)", (sid, now - days * 86400, 1.0, 1.0, 1.0, 60)); c.execute("INSERT INTO sig_1m(ts) VALUES(?)", (now - days * 86400,)); c.execute("INSERT INTO logs VALUES(?,?,?)", (now - days * 86400, "x", "WARN y"))
        store.compact(c, now)
        kept = sorted(round((now - r[0]) / 86400) for r in c.execute("SELECT ts FROM lk_1m"))
        self.assertEqual(kept, [3, 20, 71, 100, 119, 125]); self.assertEqual(sorted(round((now - r[0]) / 86400) for r in c.execute("SELECT ts FROM sig_1m")), [3, 20, 71, 100, 119, 125])
        self.assertEqual(c.execute("SELECT COUNT(*) FROM logs").fetchone()[0], 6)                  # diagnostic logs are retained for at least five years


if __name__ == "__main__":
    unittest.main()

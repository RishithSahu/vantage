"""The cars that carry DoIP, LIN, CANopen and FlexRay, end to end: the car puts each bus on the pipe in the form that bus has, Vantage recognises the car from its traffic alone (the decoder file whose
buses are all present beats the file of the car that shares its CAN but lacks them), and the quantities that only travel on the new buses are in the car's own database."""
import json, os, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_pipe import Bench, model_dirs

ONLY_ON_THE_NEW_BUS = {
    "nova_ev": {"rain", "light", "humidity", "seatbelt_state"},                                 # LIN body network
    "prestige_gt": {"wheel_speed_fl", "yaw_rate", "brake_pressure", "humidity"},               # FlexRay chassis cluster + LIN
    "volta_bus": {"battery_soc", "motor_torque", "tyre_pressure_fl", "odometer", "aux_battery_voltage"},   # CANopen PDOs and SDO reads
}
MEDIA = {"nova_ev": {"lin", "doip"}, "prestige_gt": {"flexray", "lin"}, "volta_bus": {"can"}}


class EndToEnd(unittest.TestCase):
    def test_each_car_is_recognised_and_its_new_bus_is_read(self):
        for mid, want in ONLY_ON_THE_NEW_BUS.items():
            with self.subTest(car=mid):
                if mid not in model_dirs(): self.skipTest(mid + " is not built")
                b = Bench(self); car = b.car(mid, seconds=60.0, channel_seed=991); st = b.svc.status(); v = st.get("vehicle")
                self.assertIsNotNone(v, (mid, st.get("state"), json.dumps(st.get("decision") or {}, default=str)[:300])); self.assertEqual(v["id"], mid, st["decision"].get("runners_up")); self.assertEqual(st["decision"]["mode"], "decoder file")
                for medium in MEDIA[mid]: self.assertGreater(car.tx.counts.get(medium, 0), 0, f"{mid} sent nothing on {medium}: {car.tx.counts}")
                c = b.db()
                try:
                    got = {r[0] for r in c.execute("SELECT concept FROM sig_catalog")}; self.assertTrue(want <= got, (mid, want - got))
                    self.assertGreater(c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0], 10)
                finally: c.close()
                self.assertEqual(st["wire"]["bad_crc"], 0); self.assertEqual(st["dropped"], 0)

    def test_a_fault_on_the_canopen_bus_is_an_emergency_message_and_an_event(self):
        from vgpt.vehicle.profile import Profile
        from vgpt.vehicle.router import ProfileRouter
        from vgpt.world.emit import Emitter
        if "volta_bus" not in model_dirs(): self.skipTest("volta_bus is not built")
        p = Profile.load(model_dirs()["volta_bus"]["dir"]); em = Emitter(p); r = ProfileRouter(p); out = []
        frames = em.event(100.0, {"type": "DTC_SET", "code": "P0A80"}) + em.event(160.0, {"type": "DTC_CLEAR", "code": "P0A80"})
        self.assertEqual([f[1] for f in frames], ["can", "can"])
        for ts, bus, payload in frames: out += r.decode(bus, payload, ts)
        ev = [x for x in out if x[0] == "event"]; self.assertEqual([x[1] for x in ev], ["DTC_SET", "DTC_CLEAR"]); self.assertIn("CANopen node 0x10", ev[0][3]); self.assertIn("P0A80", ev[0][3])

    def test_a_dtc_on_the_doip_car_is_a_diagnostic_message_over_ethernet(self):
        from vgpt.vehicle.profile import Profile
        from vgpt.vehicle.router import ProfileRouter
        from vgpt.world.emit import Emitter
        from vgpt.protocols import doip
        if "nova_ev" not in model_dirs(): self.skipTest("nova_ev is not built")
        p = Profile.load(model_dirs()["nova_ev"]["dir"]); em = Emitter(p); r = ProfileRouter(p); frames = em.event(100.0, {"type": "DTC_SET", "code": "P0A80"})
        self.assertTrue(frames and all(f[1] == "doip" for f in frames)); self.assertTrue(any(doip.decode(f[2])["name"] == "diagnostic_ack" for f in frames))
        out = []
        for ts, bus, payload in frames: out += r.decode(bus, payload, ts)
        self.assertTrue(any(x[0] == "event" and x[1] == "DTC_SET" and "P0A80" in x[3] for x in out), out)


if __name__ == "__main__":
    unittest.main()

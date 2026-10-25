"""Head-unit simulator smoke test: the application in this process (no network), real clip, real assistant (rules-only), live DB in a temp dir."""
import importlib.util, json, os, sys, tempfile, time, unittest
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tests.localclient import LocalClient  # noqa: E402
try:
    import cv2
except ImportError:
    cv2 = None
HAVE = cv2 is not None and os.path.exists(os.path.join(ROOT, "data", "ref", "camera_clip.hevc")) and os.path.exists(os.path.join(ROOT, "data", "vehicle.db"))


@unittest.skipUnless(HAVE, "needs opencv, the dashcam clip and data/vehicle.db")
class HeadUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("uiserver", os.path.join(ROOT, "ui", "server.py")); cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)
        from tests.stub_planner import StubPlanner      # the model is replaced by a scripted test double: this class tests the head unit (camera, events, HTTP), not language understanding
        script = {"Is the camera blocked right now?": ("current_value", {"turn": "new", "signal": "camera_blockage"}), "Was the camera blocked today?": ("did_event_happen", {"turn": "new", "event": "CAMERA_BLOCKED", "period": "today"})}
        cls.app = cls.m.App(SimpleNamespace(research=True, db="data/vehicle.db", model="none", camera="clip", webcam_index=0, live=os.path.join(tempfile.mkdtemp(), "live.db")), planner_factory=lambda conn: StubPlanner(conn, script))
        cls.c = LocalClient(cls.m.make_dispatcher(cls.app))

    def get(self, p): return self.c.get_json(p)
    def post(self, p, b): return self.c.post_json(p, b)
    def wait(self, pred, secs=12):
        t = time.time()
        while time.time() - t < secs:
            s = self.get("/api/state")
            if pred(s): return s
            time.sleep(0.3)
        self.fail("timeout: " + json.dumps(s["cam"])[:200])

    def test_full_loop_fault_to_banner_state_answer_and_thumbnail(self):
        self.wait(lambda s: s["cam"]["state"] == "OK", 15)
        st, _, page = self.c.request("GET", "/"); self.assertEqual(st, 200)
        self.assertIn(b"<title>VehicleGPT</title>", page); self.assertIn(b"Head Unit", self.c.get_bytes("/headunit"))
        self.assertTrue(self.get("/api/state")["tele"]["ok"])
        self.post("/api/fault", {"kind": "cover"}); s = self.wait(lambda s: s["cam"]["state"] == "BLOCKED")
        a = self.post("/api/ask", {"q": "Is the camera blocked right now?"}); self.assertIn("route", a); self.assertIn("camera blockage", a["answer"].lower()); self.assertLess(a["ms"]["total"], 2000)       # the model product answers with the recorded signal; the old rule route that worded the live camera state ("BLOCKED") is gone
        self.post("/api/fault", {"kind": "none"}); s = self.wait(lambda s: s["cam"]["state"] == "OK")
        a = self.post("/api/ask", {"q": "Was the camera blocked today?"}); self.assertIn("blockage", a["answer"])
        ev = [e for e in s["events"] if e["type"] == "CAMERA_BLOCKED"]; self.assertTrue(ev and ev[0]["thumb"])
        png = self.c.get_bytes(f"/api/thumb?ts={ev[0]['ts']}"); self.assertEqual(png[:4], b"\x89PNG")

    def test_tunnel_darkening_is_not_a_fault_and_bad_inputs_are_rejected(self):
        self.wait(lambda s: s["cam"]["state"] == "OK", 15); self.post("/api/fault", {"kind": "dark"}); time.sleep(4); s = self.get("/api/state")
        self.assertEqual(s["cam"]["state"], "OK"); self.post("/api/fault", {"kind": "none"})
        self.assertEqual(self.c.request("POST", "/api/camera", body={"source": "file", "path": "C:/does/not/exist.mp4"})[0], 400)
        self.assertEqual(self.post("/api/ask", {"q": ""})["path"], "empty")

    def test_research_dashboard_is_read_only_and_truthful(self):
        page = self.c.get_bytes("/research")
        self.assertIn(b"VehicleGPT Research Studio", page)
        data = self.get("/api/research/status")
        self.assertEqual(data["scope"], "iterative development set")
        self.assertTrue(data["sealed"]["acceptance"] and data["sealed"]["final"])
        self.assertEqual(len(data["missing"]), 2)
        self.assertLess(data["headline"]["achieved_full_ir"],
                        data["headline"]["candidate_recall_k1000"])
        self.assertFalse(data["provenance"]["acceptance_or_final_opened"])

if __name__ == "__main__":
    unittest.main()

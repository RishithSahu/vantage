"""Who may call what (vgpt/routepolicy.py), through the real dispatcher of the interface: deny by default, simulation controls never in a production build, the study pages only when switched on and with consent, the raw
feedback only for the service and validation roles, records sealed and deletable, the camera analysis idle until the picture is used, and /api/health that says which model answers."""
import ast
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests import testdata  # noqa: E402
from vgpt import routepolicy as RP  # noqa: E402

DB = "data/world_aster.db"
HAVE = os.path.exists(os.path.join(ROOT, DB))


class Table(unittest.TestCase):
    def test_every_route_the_dispatcher_answers_is_in_the_table(self):
        src = open(os.path.join(ROOT, "ui", "server.py"), encoding="utf-8").read(); tree = ast.parse(src)
        disp = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "make_dispatcher"); lits = set()
        for n in ast.walk(disp):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and ((n.value.startswith("/api/") and not n.value.endswith("/")) or n.value in ("/", "/classic", "/app", "/app.html", "/headunit", "/index.html", "/review", "/research", "/research.html", "/favicon.ico")): lits.add(n.value)
        live = set()                                                                                                 # LIVE_POST is a module constant: the Live page's controls
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "LIVE_POST" for t in n.targets): live = {e.value for e in n.value.elts}
        known = {p for _, p, _ in RP.table()}
        self.assertEqual(sorted(l for l in (lits | live) if l not in known and RP.rule_of("GET", l) is None and RP.rule_of("POST", l) is None), [])

    def test_a_route_not_in_the_table_does_not_exist(self):
        for m in ("GET", "POST"): self.assertEqual(RP.decide(m, "/api/nothing", "driver").code, 404)
        self.assertEqual(RP.decide("PUT", "/api/ask", "driver").code, 404); self.assertEqual(RP.decide("DELETE", "/", "driver").code, 404)

    def test_the_rules(self):
        app = SimpleNamespace(production=False, research=False, driving=False, restrict=False)
        self.assertIsNone(RP.decide("POST", "/api/ask", "driver", app)); self.assertIsNone(RP.decide("GET", "/modern/app.js", "driver", app))
        self.assertEqual(RP.decide("GET", "/api/fleet", "driver", app).code, 403); self.assertIsNone(RP.decide("GET", "/api/fleet", "fleet", app))
        self.assertEqual(RP.decide("GET", "/api/feedback/export", "driver", app).code, 403); self.assertEqual(RP.decide("GET", "/api/feedback/export", "fleet", app).code, 403)
        for r in ("engineer", "validation"): self.assertIsNone(RP.decide("GET", "/api/feedback/export", r, app))
        self.assertIsNone(RP.decide("POST", "/api/dataset", "driver", app)); self.assertEqual(RP.decide("POST", "/api/dataset", "fleet", app).code, 403)
        app.driving = app.restrict = True; self.assertEqual(RP.decide("POST", "/api/restrict", "driver", app).code, 423); self.assertIsNone(RP.decide("GET", "/api/datasets", "driver", app))
        app.production = True; app.driving = app.restrict = False
        for p in ("/api/dataset", "/api/fault", "/api/camera", "/api/timescale", "/api/reset", "/api/restrict", "/api/live/jump"): self.assertEqual(RP.decide("POST", p, "engineer", app).code, 403, p)
        self.assertEqual(RP.decide("GET", "/api/datasets", "driver", app).code, 403)
        self.assertEqual(RP.decide("GET", "/review", "driver", app).code, 404)

    def test_the_study_needs_to_be_switched_on_and_needs_consent_for_every_recording_call(self):
        app = SimpleNamespace(production=False, research=True, driving=False, restrict=False)
        self.assertIsNone(RP.decide("GET", "/review", "driver", app)); self.assertIsNone(RP.decide("GET", "/api/catalogue", "driver", app))
        for p in ("/api/interpret", "/api/execute", "/api/judge", "/api/abandon"):
            self.assertEqual(RP.decide("POST", p, "driver", app, {"q": "x"}).code, 403, p); self.assertEqual(RP.decide("POST", p, "driver", app, {"consent": "yes"}).code, 403, "only a real yes counts")
            self.assertIsNone(RP.decide("POST", p, "driver", app, {"consent": True}))
        app.production = True; self.assertEqual(RP.decide("POST", "/api/interpret", "driver", app, {"consent": True}).code, 404, "never in a production build")


@unittest.skipUnless(HAVE, "needs data/world_aster.db")
class Dispatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.localclient import LocalClient
        from tests.stub_planner import StubPlanner
        cls.home = tempfile.mkdtemp(); cls._home = os.environ.get("VEHICLEGPT_HOME"); os.environ["VEHICLEGPT_HOME"] = cls.home
        spec = importlib.util.spec_from_file_location("uiserver_rp", os.path.join(ROOT, "ui", "server.py")); cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)
        cls.app = cls.m.App(SimpleNamespace(db=DB, model="none", camera="clip", webcam_index=0, live=os.path.join(tempfile.mkdtemp(), "live.db")), planner_factory=lambda conn: StubPlanner(conn, {}))
        cls.c = LocalClient(cls.m.make_dispatcher(cls.app))

    @classmethod
    def tearDownClass(cls):
        cls.app._stop_live()
        if cls._home is None: os.environ.pop("VEHICLEGPT_HOME", None)
        else: os.environ["VEHICLEGPT_HOME"] = cls._home

    def setUp(self): self.app.production = False; self.app.research = False; self.app.driving = False; self.app.restrict = False

    def cookie(self, role): return {"Cookie": "vgpt_s=" + self.app.sessions.open(role)}

    def test_the_study_pages_are_off_unless_switched_on(self):
        self.assertEqual(self.c.request("GET", "/review")[0], 404); self.assertEqual(self.c.request("GET", "/api/catalogue")[0], 404); self.assertEqual(self.c.request("POST", "/api/interpret", body={"q": "x", "consent": True})[0], 404)
        self.app.research = True
        self.assertEqual(self.c.request("GET", "/review")[0], 200); self.assertEqual(self.c.request("POST", "/api/interpret", body={"q": "x"})[0], 403)
        self.assertNotIn(self.c.request("POST", "/api/interpret", body={"q": "x", "consent": True})[0], (403, 404))

    def test_simulation_controls_are_refused_in_a_production_build_for_every_role(self):
        for role in (None, "engineer", "validation"):
            h = self.cookie(role) if role else None
            self.assertEqual(self.c.request("POST", "/api/fault", h, {"kind": "cover"})[0], 200 if not self.app.production else 403)
        self.app.production = True
        for role in (None, "engineer", "validation", "fleet"):
            h = self.cookie(role) if role else None
            for p, b in (("/api/fault", {"kind": "cover"}), ("/api/timescale", {"mult": 5}), ("/api/dataset", {"name": "world_aster.db"}), ("/api/reset", {}), ("/api/restrict", {"on": False}), ("/api/camera", {"source": "clip"})):
                st, _, body = self.c.request("POST", p, h, b); self.assertEqual(st, 403, (role, p)); self.assertIn("not part of this build", json.loads(body)["error"])
        self.assertEqual(self.c.request("GET", "/api/datasets")[0], 403); self.assertEqual(self.c.request("POST", "/api/ask", None, {"q": ""})[0], 200)               # the questions themselves still work
        self.app.rig.fault = "none"

    def test_the_fleet_role_cannot_use_the_test_controls(self):
        self.assertEqual(self.c.request("POST", "/api/timescale", self.cookie("fleet"), {"mult": 2})[0], 403); self.assertEqual(self.c.request("POST", "/api/timescale", None, {"mult": 1})[0], 200)

    def test_the_driving_lock_cannot_be_switched_off_while_driving(self):
        self.app.driving = self.app.restrict = True; self.assertEqual(self.c.request("POST", "/api/restrict", None, {"on": False})[0], 423); self.assertTrue(self.app.restrict)

    def test_the_raw_feedback_belongs_to_the_service_and_validation_roles(self):
        self.assertEqual(self.c.post_json("/api/feedback", {"rating": "down", "q": "how hot is the battery" + " x" * 200, "brief": "b" * 900, "note": "n" * 900})["ok"], True)
        self.assertEqual(self.c.request("GET", "/api/feedback/export")[0], 403); self.assertEqual(self.c.request("GET", "/api/feedback/export", self.cookie("fleet"))[0], 403)
        st, _, body = self.c.request("GET", "/api/feedback/export", self.cookie("engineer")); self.assertEqual(st, 200); j = json.loads(body); self.assertGreaterEqual(j["count"], 1)
        rec = j["items"][-1]; self.assertLessEqual(len(rec["q"]), 200); self.assertLessEqual(len(rec["brief"]), 300); self.assertLessEqual(len(rec["note"]), 300)

    def test_feedback_is_sealed_on_the_disk_and_the_driver_can_erase_it(self):
        from vgpt import recordstore as RS, paths
        self.c.post_json("/api/feedback", {"rating": "up", "q": "secret question words", "route": "r"}); raw = open(paths.feedback_file(), encoding="utf-8").read()
        if RS.HAVE_AES: self.assertNotIn("secret question words", raw)
        st, _, body = self.c.request("POST", "/api/privacy/erase", None, {"what": "feedback", "confirm": "ERASE"}); self.assertEqual(st, 200); self.assertFalse(os.path.exists(paths.feedback_file()))
        self.assertEqual(self.c.request("POST", "/api/privacy/erase", self.cookie("engineer"), {"what": "feedback", "confirm": "ERASE"})[0], 403)

    def test_health_says_which_model_answers_and_what_kind_of_build_this_is(self):
        h = self.c.get_json("/api/health"); self.assertTrue(h["ok"]); self.assertIn("production", h["build"]); self.assertIn("research", h["build"])
        for k in ("max", "expired", "effects_undone"): self.assertIn(k, h["queue"])
        self.assertIn("loaded", h["model"]); self.assertIn("source", h["product"]); self.assertIn("analysing", h["camera"])

    def test_the_camera_analysis_is_idle_until_the_picture_is_used(self):
        rig = self.app.rig; rig.always = False; rig.fault = "none"; rig.watch_t = -1e9
        self.assertFalse(rig.active()); self.assertFalse(self.c.get_json("/api/health")["camera"]["analysing"])
        keep, self.app.livecar = self.app.livecar, None
        try: self.app.frame_jpeg()
        finally: self.app.livecar = keep
        self.assertTrue(rig.active())                                                                                # asking for the picture wakes it
        rig.watch_t = time.monotonic() - rig.ACTIVE_S - 1; self.assertFalse(rig.active())                             # and it goes quiet again
        rig.fault = "blur"; self.assertTrue(rig.active()); rig.fault = "none"
        rig.always = True; self.assertTrue(rig.active()); rig.always = False

    def test_a_question_is_answered_through_the_queue_and_a_busy_queue_is_a_503(self):
        r = self.c.request("POST", "/api/ask", None, {"q": "hello there", "req": "abc"}); self.assertEqual(r[0], 200)
        real = self.app.ask
        try:
            self.app.ask = lambda *a, **k: dict(self.m.BUSY)
            self.assertEqual(self.c.request("POST", "/api/ask", None, {"q": "hello"})[0], 503)
        finally: self.app.ask = real


if __name__ == "__main__": unittest.main()

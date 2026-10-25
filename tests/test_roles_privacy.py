"""Roles are access control, not a presentation style; what is personal is removed for every role that may not see it; what is signed in can be audited; what leaves the device is signed and can be encrypted."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import access, privacy, rolepages, securestore  # noqa: E402


class Clock:
    def __init__(self): self.t = 1_000_000.0
    def __call__(self): return self.t


class Passcodes(unittest.TestCase):
    def setUp(self): self.dir = tempfile.mkdtemp(); self.clk = Clock(); self.a = access.Auth(os.path.join(self.dir, "auth.json"), clock=self.clk)

    def test_a_passcode_is_stored_only_as_a_salted_hash(self):
        ok, _ = self.a.set_pin("engineer", "workshop-77"); self.assertTrue(ok)
        raw = open(self.a.path, encoding="utf-8").read(); self.assertNotIn("workshop-77", raw); d = json.loads(raw)["engineer"]; self.assertEqual(len(d["salt"]), 32); self.assertEqual(len(d["hash"]), 64)
        self.assertTrue(self.a.verify("engineer", "workshop-77")[0]); self.assertFalse(self.a.verify("engineer", "workshop-78")[0])

    def test_weak_and_short_passcodes_are_refused_and_the_driver_has_none(self):
        for bad in ("12345", "123456", "aaaaaaaa", "password", ""): self.assertFalse(self.a.set_pin("fleet", bad)[0], bad)
        self.assertFalse(self.a.set_pin("driver", "workshop-77")[0]); self.assertFalse(self.a.verify("driver", "x")[0])

    def test_five_wrong_tries_lock_the_role_and_the_wait_doubles(self):
        self.a.set_pin("validation", "valid-pass-1")
        for _ in range(4): self.assertEqual(self.a.verify("validation", "wrong-one")[1], "wrong")
        ok, msg, wait = self.a.verify("validation", "wrong-one"); self.assertEqual((ok, msg), (False, "locked")); self.assertEqual(wait, access.BASE_LOCK)
        self.assertEqual(self.a.verify("validation", "valid-pass-1")[1], "locked", "even the right passcode waits")
        self.clk.t += access.BASE_LOCK + 1; ok, msg, wait = self.a.verify("validation", "wrong-again"); self.assertEqual((ok, msg, wait), (False, "locked", access.BASE_LOCK * 2), "a further wrong try doubles the wait")
        self.clk.t += access.BASE_LOCK * 2 + 1; self.assertTrue(self.a.verify("validation", "valid-pass-1")[0], "and the right passcode works once the wait is over")

    def test_changing_a_passcode_needs_the_current_one(self):
        self.a.set_pin("fleet", "fleet-pass-1"); self.assertFalse(self.a.set_pin("fleet", "fleet-pass-2", current="nope-nope")[0]); self.assertTrue(self.a.set_pin("fleet", "fleet-pass-2", current="fleet-pass-1")[0])
        self.assertTrue(self.a.verify("fleet", "fleet-pass-2")[0]); self.assertFalse(self.a.verify("fleet", "fleet-pass-1")[0])


class SessionsAndAudit(unittest.TestCase):
    def test_a_session_ends_after_idle_time_and_when_the_vehicle_drives_away(self):
        clk = Clock(); s = access.Sessions(clock=clk); tok = s.open("engineer"); self.assertEqual(s.get(tok), "engineer"); clk.t += access.Sessions.IDLE - 1; self.assertEqual(s.get(tok), "engineer", "use keeps it alive")
        clk.t += access.Sessions.IDLE + 1; self.assertIsNone(s.get(tok)); t2 = s.open("fleet"); self.assertEqual(s.close_all(), 1); self.assertIsNone(s.get(t2)); self.assertIsNone(s.get("made-up")); self.assertIsNone(s.get(None))

    def test_the_audit_trail_is_chained_and_a_changed_line_is_found(self):
        d = tempfile.mkdtemp(); a = access.Audit(os.path.join(d, "audit.jsonl"), clock=Clock())
        for i in range(5): a.add("engineer", "view", page=f"p{i}")
        self.assertEqual(a.verify(), {"ok": True, "count": 5, "first_bad": None})
        lines = open(a.path, encoding="utf-8").read().splitlines(); rec = json.loads(lines[2]); rec["who"] = "validation"; lines[2] = json.dumps(rec)
        open(a.path, "w", encoding="utf-8").write("\n".join(lines) + "\n"); v = a.verify(); self.assertFalse(v["ok"]); self.assertEqual(v["first_bad"], 3)
        lines = [json.dumps(json.loads(l)) for l in open(a.path, encoding="utf-8").read().splitlines()]; del lines[1]; open(a.path, "w", encoding="utf-8").write("\n".join(lines) + "\n"); self.assertFalse(a.verify()["ok"], "a removed line breaks the chain")

    def test_the_trail_records_what_happened_never_the_content(self):
        d = tempfile.mkdtemp(); a = access.Audit(os.path.join(d, "audit.jsonl")); r = a.add("engineer", "ask", route="now:speed", path="meaning"); self.assertNotIn("q", r["detail"]); self.assertEqual(set(r["detail"]), {"route", "path"})


class Roles(unittest.TestCase):
    def test_only_the_driver_holds_the_personal_permissions(self):
        for r in ("engineer", "fleet", "validation"):
            for p in access.PERSONAL: self.assertFalse(access.can(r, p), (r, p))
        for p in access.PERSONAL: self.assertTrue(access.can("driver", p))
        self.assertTrue(access.can("engineer", "service") and not access.can("fleet", "service")); self.assertTrue(access.can("validation", "audit") and not access.can("engineer", "audit"))
        self.assertTrue(access.can("driver", "erase") and not access.can("fleet", "erase")); self.assertEqual(access.role("nonsense")["id"], "driver")

    def test_the_answer_style_follows_the_role(self):
        self.assertEqual(access.persona_of("driver"), "driver"); self.assertEqual(access.persona_of("engineer"), "engineer"); self.assertEqual(access.persona_of("validation"), "engineer")


class Scrubbing(unittest.TestCase):
    NAMES = ["Home", "Work"]

    def test_coordinates_and_saved_places_are_removed_from_text(self):
        t = privacy.redact_text("The car is at Home (GPS 51.5130 N, 0.0437 W) and 42.278, -83.759 is near Work.", self.NAMES)
        for leak in ("Home", "Work", "51.5130", "42.278", "83.759"): self.assertNotIn(leak, t)
        self.assertIn("a saved place", t); self.assertIn("hidden", t); self.assertEqual(privacy.redact_text("Network and homework are fine", self.NAMES), "Network and homework are fine")

    def test_a_role_without_the_permission_gets_a_copy_without_the_personal_keys(self):
        obj = {"answer": "Parked at Home", "tele": {"speed": 3, "pos": {"lat": 1.0, "lon": 2.0, "near": "Home"}, "soc": 70}, "events": [{"type": "CAMERA_BLOCKED", "thumb": True, "detail": "ok"}], "points": [[1, 2]]}
        for role in ("engineer", "fleet", "validation"):
            out = privacy.scrub(obj, role, self.NAMES); self.assertNotIn("pos", out["tele"]); self.assertNotIn("points", out); self.assertEqual(out["tele"]["soc"], 70); self.assertIs(out["events"][0]["thumb"], False); self.assertNotIn("Home", json.dumps(out))
        self.assertIs(privacy.scrub(obj, "driver", self.NAMES), obj, "the driver sees their own data")
        self.assertIn("pos", obj["tele"], "the original is not changed")


class Signing(unittest.TestCase):
    def setUp(self): os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp()

    def test_a_pack_is_signed_and_a_change_is_detected(self):
        pack = {"vehicle": "x", "answer": "74 km/h"}; sig = securestore.sign(pack); self.assertTrue(securestore.verify(pack, sig)); self.assertFalse(securestore.verify(dict(pack, answer="75 km/h"), sig))

    @unittest.skipUnless(securestore.HAVE_AES, "the cryptography package is not installed")
    def test_a_pack_is_encrypted_with_a_passphrase_and_only_that_passphrase_opens_it(self):
        blob = securestore.seal(b'{"secret": 1}', "correct horse"); self.assertNotIn(b"secret", blob); self.assertEqual(securestore.unseal(blob, "correct horse"), b'{"secret": 1}')
        with self.assertRaises(ValueError): securestore.unseal(blob, "wrong horse")
        with self.assertRaises(ValueError): securestore.unseal(blob[:-3] + b"xyz", "correct horse")
        with self.assertRaises(ValueError): securestore.seal(b"x", "short")


from tests import testdata
SRC = os.path.join(ROOT, "data", "world_aster.db")


@unittest.skipUnless(os.path.exists(SRC), "the Aster world is not present")
class Reports(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)

    @classmethod
    def tearDownClass(cls): cls.c.close()

    def test_the_service_report_gives_each_verdict_its_evidence_and_never_calls_missing_data_fine(self):
        r = rolepages.service_report(self.c); self.assertGreaterEqual(len(r["checks"]), 8)
        for ch in r["checks"]: self.assertTrue(ch["evidence"]); self.assertIn(ch["status"], ("ok", "watch", "attention", "unknown"))
        by = {c["id"]: c for c in r["checks"]}; self.assertEqual(by["service"]["status"], "attention", "the Aster is 5,150 km past its service interval"); self.assertIn("overdue", by["service"]["detail"])
        self.assertTrue(r["codes"]); self.assertTrue(any(s["step"] for s in r["steps"])); blob = json.dumps(r); self.assertNotIn('"lat"', blob); self.assertNotIn('"lon"', blob)

    def test_the_fleet_view_names_a_vehicle_by_model_and_an_anonymous_id_and_says_why_it_needs_attention(self):
        testdata.need_files("data/world_rhino.db")
        f = rolepages.fleet_overview([SRC, os.path.join(ROOT, "data", "world_rhino.db")]); self.assertEqual(f["totals"]["vehicles"], 2)
        for v in f["vehicles"]: self.assertRegex(v["id"], r"^V-[0-9A-F]{4}$"); self.assertIn(v["status"], ("ok", "watch", "attention")); self.assertTrue(v["attention"] or v["status"] == "ok")
        blob = json.dumps(f).lower(); self.assertNotIn("home", blob); self.assertNotIn("longitude", blob); self.assertIn("no driver identity", blob)

    def test_the_validation_report_scores_injected_faults_against_what_was_recorded(self):
        testdata.need_files("data/world_aster_truth.json")
        v = rolepages.validation_report(self.c, "world_aster.db", ROOT); self.assertGreater(len(v["signals"]), 30); sc = v["scorecard"]; self.assertIsNotNone(sc); self.assertEqual(sc["found"], sc["injected"]); self.assertEqual(sc["recall"], 1.0)
        self.assertTrue(any(b["bus"] == "can" for b in v["buses"])); self.assertTrue(v["vehicle"]["synthetic"])

    def test_an_evidence_pack_carries_the_question_the_answer_and_how_it_was_read(self):
        p = rolepages.evidence_pack("what is my speed", {"answer": "74 km/h", "path": "meaning", "route": "qc:current_value", "slm": {"raw": "now speed none none none none"}, "ms": {"total": 40}}, self.c, "validation")
        self.assertEqual(p["question"], "what is my speed"); self.assertEqual(p["frame"], "now speed none none none none"); self.assertEqual(p["route"], "qc:current_value"); self.assertEqual(len(p["question_sha256"]), 64)


class Shell(unittest.TestCase):
    def test_the_app_window_has_no_browser_chrome_its_own_profile_and_no_way_out(self):
        from vgpt.shell import pipe_shell
        cmd = pipe_shell.browser_flags("/tmp/work", kiosk=True, root=False); self.assertIn("--app=about:blank", cmd); self.assertIn("--kiosk", cmd); self.assertTrue(any(c.startswith("--user-data-dir=") for c in cmd))
        self.assertIn("--remote-debugging-pipe", cmd); self.assertFalse([c for c in cmd if c.startswith("--remote-debugging-port")], "no debugging port: the browser is driven over pipes only")
        self.assertIn("--disable-sync", cmd); self.assertIn("--disable-background-networking", cmd); self.assertIn("--host-resolver-rules=MAP * ~NOTFOUND", cmd)
        win = pipe_shell.browser_flags("/tmp/work", kiosk=False, root=False); self.assertNotIn("--kiosk", win); self.assertTrue(any(c.startswith("--window-size=") for c in win)); self.assertNotIn("--no-sandbox", win)
        self.assertIn("--no-sandbox", pipe_shell.browser_flags("/tmp/work", root=True))


if __name__ == "__main__":
    unittest.main()

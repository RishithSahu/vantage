"""The four people the case study names - drivers, service engineers, fleet operators, validation teams - each walk through the RUNNING application the way they would: the shipped application with the real language model, through
the same dispatcher the app window uses (no network), a session cookie for the roles that sign in.  What is checked is behaviour a person would notice (an answer that explains, a page that says why a vehicle needs attention, a page that refuses what is not theirs), not exact wording.

These are walkthroughs on simulated and recorded histories, by the author.  No driver, workshop, fleet or validation team has used the application: `docs/SCORECARD.md` says that is the largest gap in "value for the user"."""
import base64
import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import securestore  # noqa: E402

DB = os.path.join(ROOT, "data", "gen_courier.db")
HAVE = os.path.exists(DB) and os.path.exists(os.path.join(ROOT, "data", "ref", "camera_clip.hevc")) and os.path.exists(os.path.join(ROOT, "data", "world_aster.db"))
PIN = "walkthrough-pin-7"


@unittest.skipUnless(HAVE, "needs the shipped histories and the camera clip")
class Walkthroughs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp()
        spec = importlib.util.spec_from_file_location("uiserver_roles", os.path.join(ROOT, "ui", "server.py")); cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)
        cls.app = cls.m.App(SimpleNamespace(db="data/gen_courier.db", model="0.5b", camera="clip", webcam_index=0, port=0))                       # the shipped path, the real model
        cls.dispatch = staticmethod(cls.m.make_dispatcher(cls.app)); cls.cookie = {}
        for role in ("engineer", "fleet", "validation"):
            assert cls.app.auth.set_pin(role, PIN)[0]; st, _, h = cls.req("POST", "/api/login", None, {"role": role, "pin": PIN}); assert st == 200, (role, st); cls.cookie[role] = h["Set-Cookie"].split(";")[0]

    @classmethod
    def req(cls, method, path, cookie=None, body=None):
        h = {"Cookie": cookie} if cookie else {}
        if body is not None: h["Content-Type"] = "application/json"
        r = cls.dispatch(method, path, h, json.dumps(body) if body is not None else b""); hd = dict(r.headers)
        try: return r.status, json.loads(r.body), hd
        except ValueError: return r.status, r.body, hd

    def ask(self, q, role=None):
        st, j, _ = self.req("POST", "/api/ask", self.cookie.get(role) if role else None, {"q": q}); self.assertEqual(st, 200, q); return j

    # ---- the driver: no sign-in, plain answers about their own car ----------------------------------------------------------------------------------------------------------------
    def test_driver_asks_for_a_summary_and_gets_the_state_the_range_the_last_trip_and_what_needs_attention(self):
        a = self.ask("give me a summary of the week")["answer"]; self.assertRegex(a, r"\d+ ?km"); self.assertRegex(a, r"\d+ ?%"); self.assertIn("attention", a.lower()); self.assertNotIn("Traceback", a)

    def test_driver_asks_why_and_gets_a_hedged_explanation_not_a_verdict(self):
        a = self.ask("why did my coolant flow drop")["answer"]; self.assertIn("coolant", a.lower()); self.assertRegex(a, r"(?i)not a proven cause|evidence"); self.assertNotRegex(a, r"(?i)\bdefinitely\b|\bis broken\b")

    def test_driver_asks_what_happened_before_a_warning_and_gets_a_window_of_the_signals_around_it(self):
        a = self.ask("what happened before the last warning")["answer"]; self.assertRegex(a, r"(?i)\d+ minutes before"); self.assertIn("warning", a.lower()); self.assertRegex(a, r"\d")

    def test_driver_compares_city_and_highway_and_gets_both_figures_the_difference_and_whether_it_is_more_than_chance(self):
        a = self.ask("compare city and highway driving")["answer"]; self.assertIn("city", a.lower()); self.assertIn("highway", a.lower()); self.assertEqual(len(re.findall(r"\d+ ?Wh/km", a)) >= 2, True); self.assertIn("p-value", a)

    def test_driver_asks_about_unusual_behaviour_and_gets_counts_with_dates(self):
        a = self.ask("was anything unusual lately")["answer"]; self.assertRegex(a, r"\d+ warning"); self.assertRegex(a, r"\d{1,2} [A-Z][a-z]{2}")

    def test_only_the_driver_is_told_where_the_car_is(self):
        self.assertIn("GPS", self.ask("where am I")["answer"])
        for role in ("engineer", "fleet", "validation"):
            a = self.ask("where am I", role)["answer"]; self.assertIn("only the driver", a, role); self.assertNotRegex(a, r"GPS|\d+\.\d{3,} ?[NS]", role)
            st, j, _ = self.req("GET", "/api/track?s=600", self.cookie[role]); self.assertFalse(j.get("ok"), role); self.assertIn("hidden for this role", j.get("why", ""), role)

    def test_the_driver_cannot_open_another_roles_page(self):
        for path in ("/api/service", "/api/fleet", "/api/validation", "/api/audit"): self.assertEqual(self.req("GET", path)[0], 403, path)
        self.assertEqual(self.req("POST", "/api/export", None, {"what": "service"})[0], 403)

    # ---- the service engineer: every verdict with its evidence and the next step -----------------------------------------------------------------------------------------------
    def test_the_engineer_gets_a_service_report_where_every_verdict_has_evidence_and_trouble_has_a_next_step(self):
        st, j, _ = self.req("GET", "/api/service", self.cookie["engineer"]); self.assertEqual(st, 200)
        for k in ("codes", "warnings", "checks", "steps", "summary"): self.assertIn(k, j, k)
        self.assertTrue(j["checks"]); statuses = set()
        for ch in j["checks"]: self.assertTrue(ch["evidence"], ch); self.assertIn(ch["status"], ("ok", "watch", "attention", "unknown")); statuses.add(ch["status"])
        if statuses & {"watch", "attention"}: self.assertTrue(j["steps"], "a component needs attention but no next step is offered")

    def test_the_engineer_asks_about_faults_and_the_answer_says_which_records_are_illustrative(self):
        a = self.ask("what faults were recorded", "engineer")["answer"]; self.assertRegex(a, r"(?i)diagnostic trouble code|fault"); self.assertRegex(a, r"P[0-9A-F]{4}"); self.assertIn("illustrative", a.lower())

    def test_the_engineer_exports_a_signed_service_report_and_a_signed_evidence_pack_for_one_answer(self):
        st, j, _ = self.req("POST", "/api/export", self.cookie["engineer"], {"what": "service"}); self.assertEqual((st, j["ok"], j["signed"]), (200, True, True)); doc = json.loads(base64.b64decode(j["b64"]))
        sig = [k for k in doc if "sig" in k][0]; self.assertTrue(securestore.verify(doc["pack"], doc[sig])); self.assertIn("model", json.dumps(doc["pack"]["vehicle"]))
        r = self.ask("what faults were recorded", "engineer"); st, j, _ = self.req("POST", "/api/export", self.cookie["engineer"], {"what": "evidence", "rid": r["rid"]}); self.assertEqual((st, j["ok"], j["signed"]), (200, True, True))
        doc = json.loads(base64.b64decode(j["b64"])); self.assertTrue(securestore.verify(doc["pack"], doc[[k for k in doc if "sig" in k][0]])); self.assertIn("what faults were recorded", json.dumps(doc["pack"]).lower())
        doc["pack"]["role"] = "someone-else"; self.assertFalse(securestore.verify(doc["pack"], doc[[k for k in doc if "sig" in k][0]]), "a changed pack must not verify")

    @unittest.skipUnless(securestore.HAVE_AES, "the cryptography package is not installed")
    def test_an_evidence_pack_can_be_encrypted_with_a_passphrase_and_then_is_not_readable_without_it(self):
        r = self.ask("what faults were recorded", "engineer"); st, j, _ = self.req("POST", "/api/export", self.cookie["engineer"], {"what": "evidence", "rid": r["rid"], "passphrase": "a long enough passphrase"})
        self.assertEqual((st, j["ok"], j["encrypted"]), (200, True, True)); blob = base64.b64decode(j["b64"]); self.assertNotIn(b"faults", blob); self.assertNotIn(b"pack", blob)
        self.assertIn(b"what faults", securestore.unseal(blob, "a long enough passphrase").lower()); self.assertRaises(ValueError, securestore.unseal, blob, "another passphrase")

    # ---- the fleet operator: many vehicles, which needs attention and why, nobody's identity or places ----------------------------------------------------------------------------
    def test_the_fleet_operator_sees_every_vehicle_by_model_and_anonymous_id_and_why_one_needs_attention(self):
        st, j, _ = self.req("GET", "/api/fleet", self.cookie["fleet"]); self.assertEqual(st, 200); vs = j["vehicles"]; self.assertGreaterEqual(len(vs), 2)
        for v in vs:
            self.assertRegex(v["id"], r"^V-[0-9A-F]{4}$"); self.assertTrue(v["model"]); self.assertIn(v["status"], ("ok", "attention", "watch", "critical"))
            if v["status"] != "ok": self.assertTrue(v["attention"], f"{v['id']} needs attention but says no reason")
        self.assertTrue(any(v["status"] == "attention" for v in vs)); self.assertEqual(len({v["id"] for v in vs}), len(vs), "two vehicles share an id")
        self.assertIn("medians", j); self.assertIn("totals", j); self.assertTrue(j["privacy"])

    def test_the_fleet_view_carries_no_driver_no_place_no_route_and_no_coordinates(self):
        st, j, _ = self.req("GET", "/api/fleet", self.cookie["fleet"]); text = json.dumps(j).lower()
        for key in ("lat", "lon", "gps", "route", "place", "driver", "home", "address", "name\""): self.assertNotRegex(text, r"\"" + re.escape(key.rstrip('"')) + r"[a-z_]*\"\s*:", key)
        self.assertNotRegex(text, r"-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}")

    def test_the_fleet_operator_cannot_open_service_validation_audit_or_exports(self):
        for path in ("/api/service", "/api/validation", "/api/audit"): self.assertEqual(self.req("GET", path, self.cookie["fleet"])[0], 403, path)
        for what in ("service", "validation", "evidence"): self.assertEqual(self.req("POST", "/api/export", self.cookie["fleet"], {"what": what})[0], 403, what)

    # ---- the validation team: is the data right, is the assistant right, can it be shown -----------------------------------------------------------------------------------------
    def test_the_validation_team_sees_signal_coverage_buses_the_injected_fault_score_and_which_model_the_accuracy_belongs_to(self):
        st, j, _ = self.req("GET", "/api/validation", self.cookie["validation"]); self.assertEqual(st, 200)
        for k in ("signals", "buses", "silent_sources", "events", "scorecard", "assistant", "flags"): self.assertIn(k, j, k)
        self.assertTrue(j["signals"]); s = j["signals"][0]
        for k in ("signal", "unit", "rate", "samples", "from", "to"): self.assertIn(k, s, k)
        self.assertIsNone(j["scorecard"], "a history with no injected faults has no ground truth to score against")
        a = j["assistant"]; product = json.load(open(os.path.join(ROOT, "models", "product.json"), encoding="utf-8"))["adapter"] if os.path.exists(os.path.join(ROOT, "models", "product.json")) else None
        self.assertEqual(a["product_adapter"], product); self.assertIn("earlier adapters", a["note"]); self.assertIn("SCORECARD", a["note"])                      # figures of an old model are never shown as the current model's

    def test_on_a_history_with_injected_faults_the_validation_team_gets_each_fault_family_scored_against_what_was_recorded(self):
        self.assertEqual(self.req("POST", "/api/dataset", None, {"name": "world_aster.db"})[0], 200)
        try:
            st, j, _ = self.req("GET", "/api/validation", self.cookie["validation"]); self.assertEqual(st, 200); card = j["scorecard"]; self.assertTrue(card); self.assertGreaterEqual(len(card["rows"]), 5)
            for r in card["rows"]:
                for k in ("family", "event", "injected", "found", "recall", "recorded_events"): self.assertIn(k, r, k)
                self.assertLessEqual(r["found"], r["injected"]); self.assertTrue(0.0 <= r["recall"] <= 1.0)
            self.assertEqual(card["injected"], sum(r["injected"] for r in card["rows"])); self.assertEqual(card["found"], sum(r["found"] for r in card["rows"])); self.assertTrue(card["note"])
        finally: self.assertEqual(self.req("POST", "/api/dataset", None, {"name": "gen_courier.db"})[0], 200)

    def test_the_validation_team_reads_the_audit_trail_which_is_chained_and_never_holds_what_was_asked(self):
        q = "a very particular question about my coolant flow xyzzy"; self.ask(q, "engineer"); self.req("GET", "/api/service", self.cookie["engineer"]); self.req("GET", "/api/fleet", self.cookie["engineer"])        # the last one is refused and recorded
        st, j, _ = self.req("GET", "/api/audit", self.cookie["validation"]); self.assertEqual(st, 200); self.assertTrue(j["chain"]["ok"]); self.assertGreaterEqual(j["chain"]["count"], 6)
        acts = {(r["who"], r["action"]) for r in j["records"]}; self.assertIn(("engineer", "sign_in"), acts); self.assertIn(("engineer", "ask"), acts); self.assertIn(("engineer", "view"), acts); self.assertIn(("engineer", "denied"), acts)
        self.assertNotIn("xyzzy", json.dumps(j).lower()); self.assertNotIn(PIN, json.dumps(j))
        for role in ("engineer", "fleet"): self.assertEqual(self.req("GET", "/api/audit", self.cookie[role])[0], 403, role)

    def test_the_validation_team_exports_an_evidence_pack_for_an_answer_given_to_the_driver(self):
        r = self.ask("why did my coolant flow drop"); st, j, _ = self.req("POST", "/api/export", self.cookie["validation"], {"what": "evidence", "rid": r["rid"]}); self.assertEqual((st, j["ok"], j["signed"]), (200, True, True))
        doc = json.loads(base64.b64decode(j["b64"])); self.assertTrue(securestore.verify(doc["pack"], doc[[k for k in doc if "sig" in k][0]])); self.assertIn("coolant", json.dumps(doc["pack"]).lower())


if __name__ == "__main__":
    unittest.main()

"""Regression tests for shared bug layers found during UI transcript replay."""
import os
import importlib.util
import sqlite3
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace

from vgpt import live_snapshot
from vgpt.brain import Brain
from vgpt.qc import QueryCompiler
from vgpt.query_ir import Query
from vgpt.store import open_ro


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "gen_demo.db")
MOVING_TS = 1787109669


@unittest.skipUnless(os.path.exists(DB), "fixture database missing")
class LiveSnapshotContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("snapshot_ui_server", os.path.join(ROOT, "ui", "server.py"))
        cls.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.server)

    def dashboard(self, c, ts):
        holder = SimpleNamespace(ro=lambda: c, tz=19800, tele_cache=(0, None), tyre_cache=(0, None), pos_cache=(0, None), caps=self.server.CAPS.detect(c), eco_cache=(None, None), gen=0, tl=threading.local())   # what the server holds once a dataset is open: the vehicle type read from its data
        return self.server.App.telemetry(holder, ts)

    def assistant(self, db, ts):
        b = Brain(db, slm=None, slm_ir=None, qc_mode="all", now=ts)
        try:
            return b.ask("what is the vehicle speed right now", use_slm=False)
        finally:
            b.c.close()

    def test_dashboard_contract_and_current_query_use_the_same_speed(self):
        c = open_ro(DB)
        try:
            snap = live_snapshot.snapshot(c, MOVING_TS)
            out = QueryCompiler(c, now=MOVING_TS).execute(Query.from_text("op=current_value tgt=speed t=now"))
            self.assertTrue(snap["available"])
            self.assertEqual(out.status, "answered")
            self.assertEqual(out.result.value, snap["values"]["speed"])
            self.assertEqual(out.result.when, snap["sample_ts"])
        finally:
            c.close()

    def test_current_value_does_not_claim_day_level_coverage_for_an_instant(self):
        c = open_ro(DB)
        try:
            out = QueryCompiler(c, now=MOVING_TS).execute(Query.from_text("op=current_value tgt=speed t=now"))
            self.assertEqual(out.status, "answered")
            contract = out.result.extra["contract"]
            self.assertFalse(contract["signal_coverage"])
            self.assertFalse(any("active days" in item for item in contract.get("limits", [])))
        finally:
            c.close()

    def test_missing_live_sample_is_unknown_not_zero(self):
        c = open_ro(DB)
        try:
            last = c.execute("SELECT MAX(ts) FROM sig_1s").fetchone()[0]
            snap = live_snapshot.snapshot(c, last + 3600)
            out = QueryCompiler(c, now=last + 3600).execute(Query.from_text("op=current_value tgt=speed t=now"))
            self.assertFalse(snap["available"])
            self.assertEqual(out.status, "no_data")
            self.assertIsNone(out.result.value)
        finally:
            c.close()

    def test_absent_vehicle_measure_is_not_mapped_to_a_similar_recorded_signal(self):
        b = Brain(DB, slm=None, slm_ir=None, qc_mode="all", now=MOVING_TS)
        try:
            b.qcs.last = Query.from_text("op=current_value tgt=speed t=now")
            out = b.ask("how has washer fluid level changed over the last 14 days", use_slm=False)
        finally:
            b.c.close()
        self.assertEqual(out["path"], "schema")
        self.assertIn("does not include washer fluid", out["answer"])
        self.assertNotIn("vehicle speed", out["answer"].lower())

    def test_dashboard_and_assistant_agree_across_two_replay_snapshots(self):
        c = open_ro(DB)
        try:
            rows = c.execute("SELECT ts, speed FROM sig_1s WHERE ts>=? AND speed IS NOT NULL ORDER BY ts LIMIT 120", (MOVING_TS,)).fetchall()
            first = rows[0]
            second = next(r for r in rows[1:] if abs(r[1] - first[1]) > 0.05)
            for ts, _ in (first, second):
                dash = self.dashboard(c, ts)
                answer = self.assistant(DB, ts)
                facts = answer["facts"]
                speed = next(i["value"] for i in facts["evidence"]["items"] if i["metric"] == "speed")
                self.assertEqual(answer["path"], "qc")
                self.assertAlmostEqual(speed, dash["speed"], places=9)
                self.assertEqual(facts["evidence"]["period_ts"][0], dash["sample_ts"])
        finally:
            c.close()

    def test_dashboard_and_assistant_both_reject_a_stale_snapshot(self):
        c = open_ro(DB)
        try:
            last = c.execute("SELECT MAX(ts) FROM sig_1s").fetchone()[0]
            dash = self.dashboard(c, last + 3600)
            answer = self.assistant(DB, last + 3600)
            self.assertFalse(dash["ok"])
            self.assertIsNone(dash["speed"])
            self.assertEqual(answer["facts"]["status"], "no_data")
            self.assertNotRegex(answer["answer"], r"\b0(?:\.0+)?\s*km/h\b(?! at \d)")   # a zero must never stand in for the missing live value; the last recorded value may be quoted when it is dated and labelled "not a current reading"
        finally:
            c.close()

    def test_dashboard_and_assistant_both_preserve_a_missing_signal(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "missing_speed.db")
            shutil.copy2(DB, db)
            c = sqlite3.connect(db)
            c.execute("UPDATE sig_1s SET speed=NULL WHERE ts=?", (MOVING_TS,))
            c.commit(); c.close()
            c = open_ro(db)
            try:
                dash = self.dashboard(c, MOVING_TS)
                answer = self.assistant(db, MOVING_TS)
                self.assertTrue(dash["ok"])
                self.assertIsNone(dash["speed"])
                self.assertEqual(answer["facts"]["status"], "no_data")
            finally:
                c.close()


@unittest.skipUnless(os.path.exists(DB), "fixture database missing")
class DialogueIsolation(unittest.TestCase):
    def setUp(self):
        self.b = Brain(DB, slm=None, slm_ir=None, policy="clarify", qc_mode="all", now=MOVING_TS)

    def tearDown(self):
        self.b.c.close()

    def test_visible_duplicate_readings_are_collapsed(self):
        r = self.b.ask("how fast can we go", use_slm=False)
        self.assertEqual(r["answer"].lower().count("highest vehicle speed"), 1)

    def test_new_domain_request_expires_open_clarification(self):
        # Rated-speed capability is now answerable directly.  Start with a
        # genuinely underspecified measurement instead.
        self.b.ask("temperature", use_slm=False)
        self.assertTrue(self.b.qcs.choices)
        r = self.b.ask("where are we", use_slm=False)
        self.assertEqual(r["intent"], "PLACES")
        self.assertFalse(self.b.qcs.choices)
        self.assertNotIn("vehicle speed", r["answer"].lower())

    def test_identity_and_definition_are_not_hijacked_by_prior_query(self):
        self.b.ask("average acceleration", use_slm=False)
        car = self.b.ask("what car is this", use_slm=False)
        adas = self.b.ask("what is ADAS", use_slm=False)
        self.assertEqual(car["intent"], "SERVICE")
        self.assertIn("this is a", car["answer"].lower())
        self.assertEqual(adas["intent"], "GLOSSARY")
        self.assertIn("advanced driver-assistance", adas["answer"].lower())

    def test_ev_fuel_remaining_is_range_not_energy_already_used(self):
        r = self.b.ask("how much petrol left", use_slm=False)
        self.assertIn("range", r["answer"].lower())
        self.assertIn("no petrol", r["answer"].lower())
        self.assertNotIn("current value of energy used", r["answer"].lower())

    def test_battery_ageing_does_not_publish_a_linear_lifetime_forecast(self):
        r = self.b.ask("cant you calculate how much my battery life has reduced", use_slm=False)
        self.assertEqual((r.get("facts") or {}).get("route"), "soh_rate")
        self.assertIn("no remaining-life date", r["answer"].lower())
        self.assertNotIn("would reach 80%", r["answer"].lower())

    def test_diagnostics_defaults_to_the_diagnostic_overview(self):
        r = self.b.ask("diagnostics", use_slm=False)
        self.assertEqual((r.get("facts") or {}).get("route"), "qc:list_events")
        self.assertIn("diagnostic trouble code", r["answer"].lower())
        self.assertNotIn("do you mean", r["answer"].lower())

    def test_charge_session_timing_is_not_soc_or_an_aggregate(self):
        for text in ("When did I charge?", "How long ago was the car last plugged in?"):
            r = self.b.ask(text, use_slm=False)
            self.assertEqual((r.get("facts") or {}).get("route"), "last_charge")
            self.assertIn("last charge", r["answer"].lower())
            self.assertNotIn("average", r["answer"].lower())

    def test_drive_mode_configuration_is_not_operating_state_share(self):
        for text in ("How many driving modes does my car have?", "Which selectable drive modes are fitted?"):
            r = self.b.ask(text, use_slm=False)
            self.assertEqual((r.get("facts") or {}).get("route"), "drive_modes")
            self.assertIn("does not record", r["answer"].lower())
            self.assertIn("observed states", r["answer"].lower())
            self.assertNotIn("share of time", r["answer"].lower())

    def test_city_highway_general_comparison_is_unit_consistent(self):
        for text in ("What is the difference between city and highway?", "Compare city versus motorway driving"):
            r = self.b.ask(text, use_slm=False)
            self.assertEqual((r.get("facts") or {}).get("route"), "city_highway")
            self.assertIn("wh/km", r["answer"].lower())
            self.assertIn("descriptive", r["answer"].lower())
            self.assertNotIn("do you mean", r["answer"].lower())

    def test_mah_equivalence_requires_voltage_then_uses_energy(self):
        missing = self.b.ask("How many 7000 mAh batteries can I charge?", use_slm=False)
        self.assertIn("nominal voltage", missing["answer"].lower())
        self.assertNotIn("charge percentage", missing["answer"].lower())
        answer = self.b.ask("How many 7000 mAh batteries at 3.7 V can the car charge?", use_slm=False)
        self.assertEqual((answer.get("facts") or {}).get("route"), "battery_equiv")
        self.assertIn("25.9 wh", answer["answer"].lower())
        self.assertIn("85%", answer["answer"])

    def test_score_improvement_is_actionable_not_a_repeated_summary(self):
        r = self.b.ask("How can I improve my score?", use_slm=False)
        self.assertEqual((r.get("facts") or {}).get("route"), "improve_score")
        self.assertIn("largest measured opportunity", r["answer"].lower())
        self.assertIn("not an industry-standard", r["answer"].lower())

    def test_illustrative_diagnostics_are_never_presented_as_real_faults(self):
        r = self.b.ask("critical diagnostics", use_slm=False)
        self.assertIn("illustrative", r["answer"].lower())
        self.assertIn("not confirmed", r["answer"].lower())

    def test_normalization_preserves_decimal_values(self):
        from vgpt import nlu
        clean, _ = nlu.normalize("3.7 V and 0.85 kW", repair_typos=True)
        self.assertIn("3.7 v", clean)
        self.assertIn("0.85 kw", clean)
        self.assertNotIn("\x01", clean)

    def test_unspecified_damage_quantity_is_not_mapped_to_trip_analytics(self):
        for text in ("How many Hog Riders can destroy my car?", "How many imaginary robots would wreck the vehicle?"):
            r = self.b.ask(text, use_slm=False)
            self.assertIn("cannot calculate", r["answer"].lower())
            self.assertNotIn("trip", r["answer"].lower())
            self.assertNotEqual((r.get("facts") or {}).get("route"), "trip_count")


class RecordingWithoutHeading(unittest.TestCase):
    """The recorded Case Study 5 files carry GPS fixes without a heading.  That used to raise inside the shared state lookup, which swallowed the error and dropped the position (an empty map)."""
    def test_a_fix_without_heading_or_speed_still_gives_a_position(self):
        path = os.path.join(ROOT, "data", "cs5_petrol_ice.db")
        if not os.path.exists(path): self.skipTest("Case Study 5 petrol file not present")
        c = open_ro(path)
        try:
            ts = c.execute("SELECT ts FROM gps_1s WHERE hdg IS NULL ORDER BY ts LIMIT 1").fetchone()[0]; g = live_snapshot.snapshot(c, ts + 1)["gps"]
            self.assertIsNotNone(g); self.assertIsNone(g["hdg"]); self.assertAlmostEqual(g["lat"], 42.3, delta=0.1)
        finally:
            c.close()


if __name__ == "__main__":
    unittest.main()

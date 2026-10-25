"""Focused regressions for Round 13 robustness and integrity findings."""
import json
from tests import testdata
import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from vgpt import analytics, feedback_report, safety_policy, store
from vgpt.camera import CameraMonitor
from vgpt.protocols import dbc, isotp, logs, someip
from vgpt.query_ir import IRError, Query
from vgpt.query_planner import compile_entities


class SafetyPolicyRound13(unittest.TestCase):
    def test_historical_evidence_does_not_cancel_current_safety_decision(self):
        intent = safety_policy.classify("Should I keep driving after yesterday's red brake warning?")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.hazard, "brake")

    def test_warning_hazard_beats_unrelated_normal_measurement(self):
        intent = safety_policy.classify("My tyre warning is on but the battery temperature is normal; can I continue driving?")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.hazard, "tyre")


class ProtocolRound13(unittest.TestCase):
    def test_someip_sd_rejects_option_that_runs_past_options_block(self):
        # SD header, zero entries, three-byte options block; the option claims
        # four bytes after its length field so it cannot fit.
        payload = b"\0\0\0\0" + (0).to_bytes(4, "big") + (3).to_bytes(4, "big") + b"\0\x04\x04"
        with self.assertRaises(someip.SomeIpError):
            someip.parse_sd(payload)

    def test_dbc_rejects_zero_width_nonfinite_and_invalid_float_metadata(self):
        base = "BO_ 1 X: 8 ECU\n SG_ x : 0|{width}@1+ ({factor},0) [0|1] \"\" ECU\n"
        with self.assertRaises(ValueError): dbc.Database.load(base.format(width=0, factor=1))
        with self.assertRaises(ValueError): dbc.Database.load(base.format(width=8, factor="nan"))
        float_dbc = "BO_ 1 X: 8 ECU\n SG_ x : 0|32@1+ (1,0) [0|1] \"\" ECU\nSIG_VALTYPE_ 1 x : 7;\n"
        with self.assertRaises(ValueError): dbc.Database.load(float_dbc)

    def test_syslog_rejects_out_of_range_pri(self):
        self.assertIsNotNone(logs.parse("<191>1 2026-10-01T22:14:15Z ecu app 1 ID - ok"))
        self.assertIsNone(logs.parse("<192>1 2026-10-01T22:14:15Z ecu app 1 ID - bad"))
        self.assertIsNone(logs.parse("<999>Oct 11 22:14:15 host app: bad"))

    def test_isotp_rejects_first_frame_with_extra_or_single_frame_sized_payload(self):
        r = isotp.Reassembler()
        self.assertIsNone(r.feed(bytes([0x10, 0x03]) + b"abcdef"))
        self.assertGreaterEqual(r.error_counts["malformed"], 1)
        self.assertIsNone(r.feed(bytes([0x10, 0x07]) + b"abcde"))


class FeedbackRound13(unittest.TestCase):
    def test_load_discards_valid_json_of_the_wrong_type(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "feedback.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(["not", "a record"]) + "\n")
                f.write(json.dumps({"q": "valid", "rating": "up"}) + "\n")
            self.assertEqual(feedback_report.load(path), [{"q": "valid", "rating": "up"}])

    def test_tsv_feedback_cannot_inject_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "triage.tsv")
            feedback_report.write_triage_tsv([{"rating": "down", "route": "bad\troute", "q": "first\nsecond"}], path)
            rows = open(path, encoding="utf-8").read().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1], "bad route\tfirst second")


class EventEvidenceRound13(unittest.TestCase):
    def test_same_timestamp_contexts_are_retained_by_event_type(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.open_rw(os.path.join(d, "events.db"))
            ing = store.Ingestor(conn, type("D", (), {"decode": lambda *_: []})())
            ing.ring.extend([(100 + n, 0, 0, 0, 250, 0, 200, 0, 0, 1) for n in range(10)])
            ing._freeze(105, "WARN_BATT_OVERTEMP")
            ing._freeze(105, "ADAS_DISENGAGE")
            rows = conn.execute("SELECT type FROM event_ctx_v2 WHERE ts=? ORDER BY type", (105,)).fetchall()
            self.assertEqual(rows, [("ADAS_DISENGAGE",), ("WARN_BATT_OVERTEMP",)])
            conn.close()


class SelectorAndCameraRound13(unittest.TestCase):
    def test_event_selector_does_not_normalize_malformed_selector(self):
        self.assertEqual(Query.from_text("op=list_events t=cal:yesterday ev=ADAS_DISENGAGE evs=all").event.selector, "all")
        with self.assertRaises(IRError):
            Query.from_text("op=list_events t=cal:yesterday ev=ADAS_DISENGAGE evs=all2")

    def test_camera_does_not_accept_a_blurred_boot_frame_as_healthy_baseline(self):
        # A low-texture frame at startup is evidence of neither a healthy
        # camera nor a calibrated baseline.  It must remain unavailable until
        # a usable scene is observed.
        monitor = CameraMonitor()
        frame = __import__("numpy").full((240, 320), 100, dtype=__import__("numpy").uint8)
        status, _records = monitor.process(0.0, frame)
        self.assertEqual(status["state"], "CALIBRATING")
        self.assertIsNone(monitor.base_tex)


class ResolutionRound13(unittest.TestCase):
    def test_speed_extrema_use_raw_samples_not_minute_averages(self):
        with tempfile.TemporaryDirectory() as d:
            conn = __import__("sqlite3").connect(os.path.join(d, "speed.db"))
            conn.execute("CREATE TABLE sig_1s(ts INTEGER, speed REAL, mode INTEGER)")
            now = 1_800_000_000
            lo, hi = analytics.window(now, "t", conn)
            # The minute aggregate would hide this 90 km/h one-second peak.
            conn.executemany("INSERT INTO sig_1s VALUES(?,?,?)", [(lo, 100.0, 1), (lo + 1, 900.0, 1), (hi - 1, 200.0, 1)])
            result = analytics.summary(conn, "speed", now, "t")
            self.assertTrue(result["ok"])
            self.assertEqual(result["max"], 90.0)
            conn.close()

    def test_ranked_trip_selection_excludes_unknown_metric_values(self):
        sql, _args = compile_entities("trip", None, "wh_per_km", "asc", 3, 0, 100)
        self.assertIn("t.wh_km IS NOT NULL", sql)


@testdata.needs("legacy_worlds")
class ProductionModelRound13(unittest.TestCase):
    def test_qwen_product_route_loads_only_the_semantic_05b_classifier(self):
        """The production factory must not revive a second learned planner."""
        from vgpt import product
        semantic = mock.Mock()
        with mock.patch.object(product.os.path, "exists", return_value=True), \
             mock.patch("vgpt.brain.Slm", return_value=semantic) as slm, \
             mock.patch.object(product, "Brain", return_value=type("B", (), {})()) as brain_ctor:
            brain = product.make_qwen05_brain("unused.db", meaning=False)                       # the previous front end (VGPT_MEANING=0)
        self.assertIs(brain, brain_ctor.return_value)
        self.assertEqual(slm.call_count, 1)
        self.assertEqual(slm.call_args.args[0].rsplit(os.sep, 1)[-1], "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        self.assertEqual(slm.call_args.kwargs["lora_path"].rsplit(os.sep, 1)[-1], "semantic-lora-qwen25-05b-v2.gguf")
        self.assertEqual(brain_ctor.call_args.kwargs["slm"], semantic)
        self.assertIsNone(brain_ctor.call_args.kwargs["slm_ir"])

    def test_complete_typed_program_skips_the_classifier(self):
        """Known complete programs must not spend a 0.5B inference budget."""
        from vgpt.brain import Brain
        from tests.test_product import DBF, NOW
        class MustNotRun:
            def parse(self, _text):
                raise AssertionError("classifier ran for a complete typed query")
        with mock.patch.dict(os.environ, {"VGPT_SEMANTIC_FRONTDOOR": "1"}):
            brain = Brain(DBF, slm=MustNotRun(), now=NOW, policy="clarify", qc_mode="all")
            response = brain.ask("What is the highest recorded speed yesterday?")
        self.assertEqual((response["path"], response["intent"]), ("qc", "QC_EXTREME"))


@testdata.needs("legacy_worlds")
class AnswerAndStateFollowupRound13(unittest.TestCase):
    def _result(self, **kw):
        base = dict(items=[], samples=0, value=None, extra={}, period=(1_790_000_000, 1_790_003_600), unit="", when=None, reason="", period_label="")
        base.update(kw)
        return SimpleNamespace(**base)

    def test_timestamp_never_supplies_numeric_evidence(self):
        from vgpt.answer_composer import Answer
        from vgpt.answer_verifier import verify
        from vgpt.evidence import Evidence
        report = verify(Answer("The result is 1.", kind="value"), Evidence(), res=self._result())
        self.assertFalse(report.ok)
        self.assertTrue(any(code == "NUMBER" for code, _detail in report.problems))

    def test_contract_resolution_number_is_not_treated_as_a_measurement(self):
        """"1-second samples" is provenance added by the contract, not a claim."""
        from vgpt.answer_composer import Answer
        from vgpt.answer_verifier import verify
        from vgpt.evidence import Evidence
        result = self._result(extra={"contract": {"resolution": "1-second samples"}})
        report = verify(Answer("Evidence: measured, 1-second samples.", kind="analysis"), Evidence(), res=result)
        self.assertTrue(report.ok, report.problems)

    def test_speed_value_never_supplies_percentage_evidence(self):
        from vgpt.answer_composer import Answer
        from vgpt.answer_verifier import verify
        from vgpt.evidence import Evidence
        evidence = Evidence().add("speed", 50, "km/h")
        report = verify(Answer("Speed increased by 50%.", kind="value"), evidence, res=self._result(value=50, unit="km/h"))
        self.assertFalse(report.ok)
        self.assertTrue(any(code == "UNIT_VALUE" for code, _detail in report.problems))

    def test_unverified_result_is_not_answered_or_high_confidence(self):
        from vgpt.modelbrain import ModelBrain, STATUS_OUTCOME
        self.assertEqual(STATUS_OUTCOME["unverified"], "unverified")
        from tests.test_model_route import DBF, NOW
        b = ModelBrain(DBF, now=NOW)
        b._text = "question"
        response = b._finish(0, "not verified", "NONE", {}, "qc", "unverified", {"ok": False}, {})
        self.assertEqual(response["facts"]["outcome"], "unverified")
        self.assertLess(response["conf"], 0.9)

    def test_failed_plan_preserves_open_clarification(self):
        from vgpt.modelbrain import ModelBrain
        from vgpt.modelplan import PlanResult
        from tests.test_model_route import DBF, FakePlanner, NOW
        class FirstThenFailure(FakePlanner):
            def __init__(self): super().__init__(["extreme(turn=\"new\", signal=\"speed\")"]); self.calls_seen = 0
            def plan(self, *args, **kwargs):
                self.calls_seen += 1
                if self.calls_seen == 1: return super().plan(*args, **kwargs)
                return PlanResult(error="model failure", timing={})
        b = ModelBrain(DBF, planner=FirstThenFailure(), now=NOW)
        b.ask("speed")
        pending = dict(b.state.pending)
        b.ask("the highest")
        self.assertEqual(b.outcome, "failed_plan")
        self.assertEqual(b.state.pending, pending)

    def test_planner_initialization_failure_is_an_explicit_result(self):
        from vgpt.modelbrain import ModelBrain
        from tests.test_model_route import DBF, NOW
        b = ModelBrain(DBF, now=NOW)
        with mock.patch("vgpt.modelbrain.Planner", side_effect=RuntimeError("model unavailable")):
            response = b.ask("battery charge")
        self.assertEqual(b.outcome, "failed_plan")
        self.assertTrue(response["facts"]["failed"])


if __name__ == "__main__":
    unittest.main()

"""Confirmed Round 14 parser, IR, and execution-boundary regressions."""
import importlib.util
import os
import unittest

from vgpt.protocols import obd, someip, uds, vhal
from vgpt.query_ir import Condition, EntitySelect, Query, Scope, TimeSpec
from vgpt.query_validator import validate


class ProtocolRound14(unittest.TestCase):
    def test_vhal_requires_explicit_available_status_and_finite_value(self):
        decoder = vhal.Vhal()
        pid = decoder.props["PERF_VEHICLE_SPEED"]["id"]
        for status, value in ((None, 3.0), (0, float("nan")), (0, float("inf")), (0, -float("inf"))):
            packet = {"prop": pid, "float": [value]}
            if status is not None: packet["status"] = status
            self.assertEqual(decoder.decode(packet), [])
        self.assertEqual(decoder.decode({"prop": pid, "status": 0, "float": [10.0]}), [("sig", "speed", 36.0)])

    def test_catalog_rejects_mismatched_invalid_and_overlong_payloads(self):
        with self.assertRaises(someip.SomeIpError): someip.ServiceCatalog({"0x1": {"name": "x", "events": {"0x8001": {"name": "e", "fmt": ">B", "fields": ["x", "y"]}}}})
        with self.assertRaises(someip.SomeIpError): someip.ServiceCatalog({"0x1": {"name": "x", "events": {"0x8001": {"name": "e", "fmt": ">Z", "fields": ["x"]}}}})
        catalog = someip.ServiceCatalog({"0x1": {"name": "x", "events": {"0x8001": {"name": "e", "fmt": ">B", "fields": ["x"]}}}})
        self.assertEqual(catalog.decode(1, 0x8001, b"\x05")["values"], {"x": 5})
        with self.assertRaises(someip.SomeIpError): catalog.decode(1, 0x8001, b"\x05\x06")

    def test_obd_and_uds_reject_overlong_single_responses(self):
        self.assertIsNone(obd.decode(bytes([0x41, 0x0D, 50, 99])))
        self.assertEqual(obd.decode(bytes([0x41, 0x0D, 50]))["value"], 50)
        with self.assertRaises(uds.UdsError): uds.decode(bytes([0x59, 0x01, 0xFF, 1, 0, 2, 0xAA]))
        self.assertEqual(uds.decode(bytes([0x59, 0x01, 0xFF, 1, 0, 2]))["count"], 2)


class IRRound14(unittest.TestCase):
    def test_thresholds_round_trip_without_rounding(self):
        c = Condition(signal="speed", op="gt", value=123.456789, unit="km/h")
        self.assertEqual(Condition.from_text(c.to_text()).value, c.value)
        q = Query(op="aggregate", target="speed", agg="mean", scope=Scope(TimeSpec("relative", 7, "day"), conditions=[c]))
        self.assertEqual(Query.from_text(q.to_text()).scope.conditions[0].value, c.value)

    def test_compare_is_explicitly_pairwise(self):
        q = Query(op="compare", target="speed", agg="mean", groups=["city", "highway", "charging"])
        self.assertFalse(validate(q).valid)

    def test_second_scope_can_explicitly_clear_conditions_and_selection(self):
        q = Query(op="compare", target="speed", agg="mean", scope=Scope(TimeSpec("relative", 7, "day"), EntitySelect("trip", "city", "wh_per_km", "desc", 3), [Condition(predicate="charging")]), scope_b=Scope(TimeSpec("previous", 7, "day"), None, []))
        wire = q.to_text(); again = Query.from_text(wire)
        self.assertIn("c2=-", wire); self.assertIn("sel2=-", wire)
        self.assertEqual(again.scope_b.conditions, [])
        self.assertIsNone(again.scope_b.select)


class VerifierRound14(unittest.TestCase):
    def test_measurement_with_unlicensed_unit_fails_closed(self):
        from types import SimpleNamespace
        from vgpt.answer_composer import Answer
        from vgpt.answer_verifier import verify
        from vgpt.evidence import Evidence
        result = SimpleNamespace(items=[], samples=0, value=0.8, extra={}, period=(1, 2), unit="r", when=None, reason="", period_label="")
        report = verify(Answer("The correlation is 0.8 km/h.", kind="analysis"), Evidence().add("pearson_r", 0.8, "r"), res=result)
        self.assertFalse(report.ok)
        self.assertTrue(any(code == "UNIT_VALUE" for code, _ in report.problems))


class UiModelFailureRound14(unittest.TestCase):
    def test_missing_product_model_fails_closed_without_an_alternate_answer_path(self):
        """A failed 0.5B load must be visible to the UI, not silently bypassed."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("ui_server_round14", os.path.join(root, "ui", "server.py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        db = next((p for p in (os.path.join(root, "data", n) for n in ("gen_courier_ext.db", "world_aster.db")) if os.path.exists(p)), None)
        if db is None: self.skipTest("no vehicle history present to open (tools_testdata.py check)")
        clock = type("Clock", (), {"now": lambda _self: 1_790_406_000})()
        def broken_factory(*_args, **_kwargs):
            raise FileNotFoundError("qwen2.5-0.5b-instruct-q4_k_m.gguf")
        worker = module.BrainWorker(db, None, clock, brain_factory=broken_factory)
        worker.start(); self.assertTrue(worker.ready.wait(5))
        response = worker.ask("what is speed", "driver")
        self.assertEqual(response["path"], "error")
        self.assertIn("Qwen 0.5B model could not be loaded", response["answer"])


if __name__ == "__main__":
    unittest.main()

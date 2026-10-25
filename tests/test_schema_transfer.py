"""Schema transfer and missing capability, END TO END on fixture histories whose STORAGE was edited (bench/schema_fixtures.py): storage -> profile -> capability -> parser -> planner -> executor -> contract -> composer.
Numbers are checked against independent SQL / numpy on the fixture's own tables. Specificity is checked explicitly: a missing front-left pressure must never be answered with another wheel or an overall figure.
The fixture databases are built once (a missing source database is a skip with the reason)."""
import json, os, shutil, sqlite3, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
SRC = os.path.join(ROOT, "data", "gen_courier_ext.db"); NOW = 1790406000; DAY = 86400


@unittest.skipUnless(os.path.exists(SRC), f"source history missing: {SRC}")
class SchemaTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bench import schema_fixtures
        cls._old_qc = os.environ.get("VGPT_QC"); os.environ["VGPT_QC"] = "all"                     # the query-compiler path answers every question in these tests (restored afterwards)
        cls.dir = tempfile.mkdtemp(); cls.made = schema_fixtures.build(cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)
        if cls._old_qc is None: os.environ.pop("VGPT_QC", None)
        else: os.environ["VGPT_QC"] = cls._old_qc

    def vehicle(self, name):
        from vgpt import schema_overlay as SO
        from vgpt.brain import Brain
        spec = json.load(open(self.made[name][:-3] + ".overlay.json")); token = SO.apply(spec); self.addCleanup(SO.restore, token)
        b = Brain(self.made[name], slm_ir=None, now=NOW); return b, b.qc(), sqlite3.connect(f"file:{self.made[name]}?mode=ro", uri=True)

    def run_ir(self, qc, ir):
        from vgpt.query_ir import Query
        return qc.execute(Query.from_text(ir))

    def window(self, n=7): return NOW - n * DAY, NOW + 1

    # ---------------------------------------------------------------------------------------------------------- missing individual wheel
    def test_a_missing_front_left_pressure_is_never_replaced_by_another_wheel_or_an_overall_figure(self):
        b, qc, c = self.vehicle("fl_missing")
        for q in ("what is the tyre pressure on the front left", "average front left tyre pressure last week"):
            a = b.ask(q, use_slm=False)["answer"]; self.assertRegex(a, r"(?i)front-left"); self.assertRegex(a, r"(?i)not recorded|isn't recorded|no values"); self.assertNotRegex(a, r"\d+ kPa", q)
        r = self.run_ir(qc, "op=aggregate tgt=tyre_pressure_fl agg=mean t=last:7:day"); self.assertIn(r.status, ("missing_data", "no_data"))
        overall = b.ask("what is the tyre pressure", use_slm=False)["answer"]; self.assertRegex(overall, r"(?i)front-left"); self.assertNotRegex(overall, r"\d+ kPa")
        self.assertIn("rear-right", overall)                                                                  # the recorded wheels are named, the missing one is named: the explanation is specific

    def test_the_other_wheels_still_answer_and_match_sql(self):
        b, qc, c = self.vehicle("fl_missing"); lo, hi = self.window()
        x = np.array([r[0] for r in c.execute("SELECT tp_fr FROM ext_1m WHERE ts >= ? AND ts < ? AND tp_fr IS NOT NULL", (lo, hi))], float) / 10.0
        r = self.run_ir(qc, "op=aggregate tgt=tyre_pressure_fr agg=mean t=last:7:day"); self.assertEqual(r.status, "answered"); self.assertAlmostEqual(float(r.result.value), float(x.mean()), delta=0.01)

    # ---------------------------------------------------------------------------------------------------------- aggregate only
    def test_an_aggregate_only_vehicle_answers_the_overall_figure_but_not_a_single_wheel(self):
        b, qc, c = self.vehicle("aggregate_only"); lo, hi = self.window()
        x = np.array([r[0] for r in c.execute("SELECT tp_avg FROM ext_1m WHERE ts >= ? AND ts < ? AND tp_avg IS NOT NULL", (lo, hi))], float) / 10.0
        r = self.run_ir(qc, "op=aggregate tgt=tyre_pressure agg=mean t=last:7:day"); self.assertEqual(r.status, "answered"); self.assertAlmostEqual(float(r.result.value), float(x.mean()), delta=0.01)       # secondary-table columns are aligned to the one-minute base timeline: a handful of edge samples may differ
        a = b.ask("average tyre pressure last week", use_slm=False)["answer"]; self.assertNotIn("lowest", a)                       # the stored aggregate is not described as the minimum of four wheels
        a = b.ask("what is the tyre pressure on the front left", use_slm=False)["answer"]; self.assertRegex(a, r"(?i)front-left"); self.assertNotRegex(a, r"\d+ kPa")

    # ---------------------------------------------------------------------------------------------------------- different stored units and scales
    def test_values_stored_in_other_units_give_the_same_answers_as_the_original_storage(self):
        from vgpt.brain import Brain
        from vgpt.query_ir import Query
        cases = (("op=aggregate tgt=speed agg=mean t=last:7:day", 0.3), ("op=extreme tgt=speed agg=max t=last:7:day", 0.5), ("op=aggregate tgt=battery_temperature agg=mean t=last:7:day", 0.2), ("op=extreme tgt=battery_temperature agg=max t=last:7:day", 0.3), ("op=aggregate tgt=distance_driven agg=sum t=last:7:day", 0.5))
        base = Brain(SRC, slm_ir=None, now=NOW).qc(); ref = {ir: float(base.execute(Query.from_text(ir)).result.value) for ir, _ in cases}                    # the original storage, answered BEFORE the overlay is applied
        b, qc, c = self.vehicle("units")
        for ir, tol in cases:
            u = qc.execute(Query.from_text(ir)); self.assertEqual(u.status, "answered", ir); self.assertAlmostEqual(float(u.result.value), ref[ir], delta=tol, msg=ir)

    # ---------------------------------------------------------------------------------------------------------- registered but empty / intermittent
    def test_a_registered_signal_without_observations_is_declined_not_invented(self):
        b, qc, c = self.vehicle("empty_registered")
        r = self.run_ir(qc, "op=aggregate tgt=coolant_flow agg=mean t=last:7:day"); self.assertIn(r.status, ("missing_data", "no_data")); a = b.ask("average coolant flow last week", use_slm=False)["answer"]; self.assertNotRegex(a, r"\d+(\.\d+)? L/min")
        self.assertRegex(b.ask("do you record the coolant flow", use_slm=False)["answer"], r"(?i)known|no values|isn't recorded|not recorded")

    def test_an_intermittent_signal_states_its_day_coverage(self):
        b, qc, c = self.vehicle("intermittent"); r = self.run_ir(qc, "op=aggregate tgt=battery_temperature agg=mean t=last:14:day")
        self.assertEqual(r.status, "answered"); ct = r.result.extra["contract"]; self.assertEqual(ct["completeness"], "partial"); self.assertTrue(ct["signal_coverage"], ct); self.assertRegex(r.answer.text, r"readings covering \d+ of \d+ active days")

    # ---------------------------------------------------------------------------------------------------------- confusable signals
    def test_motor_torque_is_never_offered_for_steering_torque(self):
        b, qc, c = self.vehicle("steering_absent")
        a = b.ask("what was the peak steering torque yesterday", use_slm=False)["answer"]; self.assertRegex(a, r"(?i)steering torque"); self.assertNotRegex(a, r"\d+ Nm")
        r = self.run_ir(qc, "op=extreme tgt=motor_torque agg=max t=cal:yesterday"); self.assertEqual(r.status, "answered")

    # ---------------------------------------------------------------------------------------------------------- a signal the built-in registry does not know
    def test_a_new_signal_supports_the_generic_operations_and_matches_numpy(self):
        b, qc, c = self.vehicle("new_signal"); lo, hi = self.window()
        x = np.array([r[0] for r in c.execute("SELECT wfl FROM sig_1m WHERE ts >= ? AND ts < ? AND wfl IS NOT NULL", (lo, hi))], float) / 10.0
        for ir, ref in (("op=aggregate tgt=washer_fluid_level agg=mean t=last:7:day", x.mean()), ("op=extreme tgt=washer_fluid_level agg=min t=last:7:day", x.min()), ("op=extreme tgt=washer_fluid_level agg=max t=last:7:day", x.max()), ("op=aggregate tgt=washer_fluid_level agg=median t=last:7:day", np.median(x))):
            r = self.run_ir(qc, ir); self.assertEqual(r.status, "answered", ir); self.assertAlmostEqual(float(r.result.value), float(ref), delta=1e-6, msg=ir)
        r = self.run_ir(qc, "op=trend tgt=washer_fluid_level t=last:14:day"); self.assertEqual(r.status, "answered"); self.assertLess(r.result.value, 0)                      # the level empties
        r = self.run_ir(qc, "op=percent_time t=last:14:day c=sig:washer_fluid_level:lt:50:%"); self.assertEqual(r.status, "answered")

    def test_a_new_signal_is_understood_by_the_parser_from_its_synonyms_and_the_overlay_is_reversible(self):
        from vgpt import nl2ir, signal_registry as R
        n0 = len(R.CONCEPTS); b, qc, c = self.vehicle("new_signal")
        p = nl2ir.parse("average screen wash level last week", now=NOW, tz=3600, profile=qc.profile); self.assertTrue(p.ok); self.assertEqual(p.query.target, "washer_fluid_level")
        for q in ("lowest washer fluid last week", "is the windscreen washer level going down"): self.assertEqual(nl2ir.parse(q, now=NOW, tz=3600, profile=qc.profile).query.target, "washer_fluid_level", q)
        self.doCleanups(); self.assertEqual(len(R.CONCEPTS), n0); self.assertNotIn("washer fluid level", R.LEXICON)

    def test_an_opaque_identifier_is_understood_through_its_description_and_answers_in_words(self):
        from vgpt import nl2ir
        b, qc, c = self.vehicle("opaque_signal"); lo, hi = self.window()
        x = np.array([r[0] for r in c.execute("SELECT c0417 FROM sig_1m WHERE ts >= ? AND ts < ? AND c0417 IS NOT NULL", (lo, hi))], float) / 10.0
        for text, ref in (("average inverter coolant temperature last week", x.mean()), ("how hot did the inverter coolant outlet temperature get last week", x.max())):
            p = nl2ir.parse(text, now=NOW, tz=3600, profile=qc.profile); self.assertTrue(p.ok, text); self.assertEqual(p.query.target, "CH_0417")
            r = qc.execute(p.query); self.assertEqual(r.status, "answered"); self.assertAlmostEqual(float(r.result.value), float(ref), delta=1e-6); self.assertNotIn("CH_0417", r.answer.text); self.assertRegex(r.answer.text, r"(?i)inverter coolant")
        a = b.ask("what is the power electronics coolant temperature now", use_slm=False)["answer"]; self.assertNotIn("CH_0417", a)


if __name__ == "__main__": unittest.main()

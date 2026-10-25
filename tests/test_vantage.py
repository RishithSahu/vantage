"""VANTAGE tests: evidence contracts, the change operators (checked against the generator's own ground truth, not against the engine), the schema view and its round trip, the catalog-generated exemplar bank, the rules parser's
new operators, the prefix-state model backend (only when the small model file is present). Run under the target's limits:  python tools_pin.py 0,1 tools_run_tests.py tests.test_vantage"""
import json, os, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tests.perfgate import PERF
from tests import testdata
from vgpt import contracts as CT, nl2ir, signal_registry as R
from vgpt.qc import QueryCompiler
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate
from vgpt.schema_view import SchemaView, renamed_view
from vgpt.exemplar_bank import ExemplarBank

DATA = os.path.join(ROOT, "data"); NOW = 1790406000


def db(name):
    p = os.path.join(DATA, name + ".db"); return sqlite3.connect(f"file:{p}?mode=ro", uri=True) if os.path.exists(p) else None


@testdata.needs("legacy_worlds")
class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.conn = db("gen_courier_ext"); cls.qc = QueryCompiler(cls.conn, now=NOW)

    def run_q(self, text): return self.qc.execute(Query.from_text(text))

    def test_history_extremes_are_narrowed_to_the_resolution_that_is_stored(self):
        r = self.run_q("op=extreme tgt=speed agg=max t=last:30:day"); self.assertEqual(r.status, "answered")
        self.assertIn("one-minute averages", r.answer.text); c = r.result.extra["contract"]; self.assertEqual(c["result_class"], "observed"); self.assertIn("1-minute", c["resolution"])

    def test_a_list_that_was_cut_says_how_many_exist(self):
        r = self.run_q("op=list_events t=last:30:day ev=HARSH_BRAKE evs=all"); c = r.result.extra["contract"]
        self.assertEqual(c["completeness"], "truncated"); self.assertEqual(c["total"], 240); self.assertIn("of 240", r.answer.text)

    def test_a_period_longer_than_the_history_reports_the_coverage(self):
        r = self.run_q("op=aggregate tgt=speed agg=mean t=last:60:day"); c = r.result.extra["contract"]
        self.assertLess(c["coverage"], 0.6); self.assertIn("on record", r.answer.text)

    def test_estimates_and_associations_are_labelled_and_verified(self):
        r = self.run_q("op=current_value tgt=range_estimate t=now"); self.assertEqual(r.result.extra["contract"]["result_class"], "estimated"); self.assertEqual(r.status, "answered")
        r = self.run_q("op=correlation tgt=speed tgt2=battery_temperature t=last:7:day"); self.assertEqual(r.result.extra["contract"]["result_class"], "associated"); self.assertEqual(r.status, "answered")

    def test_measured_values_over_the_fine_tier_are_not_narrowed(self):
        r = self.run_q("op=aggregate tgt=wh_per_km agg=mean t=last:30:day"); self.assertNotIn("one-minute", r.answer.text)

    def test_the_verifier_rejects_a_completeness_claim_on_a_cut_list(self):
        c = CT.Contract(completeness="truncated", total=10, shown=3); q = Query.from_text("op=list_events t=last:7:day ev=HARSH_BRAKE evs=all")
        self.assertTrue(CT.check("These are all the events.", q, c)); self.assertFalse(CT.check("Showing the latest 3 of 10.", q, c))


@testdata.needs("legacy_worlds")
class ChangeOperators(unittest.TestCase):
    TARGETS = ("range_estimate", "wh_per_km", "battery_temperature", "coolant_flow", "tyre_pressure", "hvac_power", "motor_temperature", "speed")

    def test_every_answer_verifies_on_every_vehicle(self):
        for name in ("gen_courier_ext", "gen_demo", "gen_cold", "gen_roadwarrior"):
            conn = db(name)
            if conn is None: continue
            qc = QueryCompiler(conn, now=NOW if name == "gen_courier_ext" else None)
            for tg in self.TARGETS:
                for op, t in (("explain_change", "last:14:day"), ("explain_change", "last:7:day"), ("detect_change", "all")):
                    r = qc.execute(Query.from_text(f"op={op} tgt={tg} t={t}")); self.assertNotEqual(r.status, "unverified", (name, tg, op, str(r.verification)))
                    self.assertNotEqual(r.status, "invalid", (name, tg, op))

    def test_the_unexplained_part_is_stated_and_the_parts_add_up(self):
        qc = QueryCompiler(db("gen_roadwarrior")); r = qc.execute(Query.from_text("op=explain_change tgt=wh_per_km t=last:14:day")); cb = r.result.extra["contrib"]
        self.assertAlmostEqual(sum(f["contribution"] for f in cb["factors"]) + cb["residual"], cb["gap"], delta=0.35 + 0.1 * len(cb["factors"]))   # factors below 0.05 Wh/km are not listed
        self.assertIn("not captured", r.answer.text); self.assertIn("remaining", r.answer.text); self.assertEqual(r.result.extra["contract"]["result_class"], "estimated"); self.assertNotRegex(r.answer.text.lower(), r"caused|due to|because of")

    def test_no_change_is_reported_as_no_change(self):
        qc = QueryCompiler(db("gen_cold")); r = qc.execute(Query.from_text("op=explain_change tgt=wh_per_km t=last:14:day")); self.assertIn("about the same", r.answer.text)

    def test_the_largest_move_matches_the_generators_own_daily_truth(self):
        import datetime as dt
        for name in ("gen_courier", "gen_demo", "gen_cold", "gen_roadwarrior"):
            conn = db(name); tp = os.path.join(DATA, name + "_truth.json")
            if conn is None or not os.path.exists(tp): continue
            daily = [d for d in json.load(open(tp))["daily"] if d["km"] > 5]; eff = [1000 * d["kwh"] / d["km"] for d in daily]; top = max(eff[i] - eff[i - 1] for i in range(1, len(eff)))
            r = QueryCompiler(conn).execute(Query.from_text("op=detect_change tgt=wh_per_km t=all")); self.assertEqual(r.status, "answered"); self.assertAlmostEqual(r.result.extra["jump"]["change"], top, delta=1.0, msg=name)

    def test_a_planted_cooling_fault_shows_up_as_a_sustained_fall_in_flow(self):
        r = QueryCompiler(db("gen_courier_ext"), now=NOW).execute(Query.from_text("op=detect_change tgt=coolant_flow t=all")); sh = r.result.extra["shift"]
        self.assertTrue(sh["clear"], sh); self.assertLess(sh["change"], -2.0)

    def test_insufficient_history_is_an_explained_non_answer(self):
        r = QueryCompiler(db("gen_courier_ext"), now=NOW).execute(Query.from_text("op=detect_change tgt=wh_per_km t=last:3:day")); self.assertEqual(r.status, "no_data"); self.assertIn("too few", r.answer.text)


class RulesNewOperators(unittest.TestCase):
    def parse(self, q): return nl2ir.parse(q, now=NOW, tz=3600)

    def test_change_questions(self):
        self.assertEqual(self.parse("why has my range dropped over the past two weeks").query.to_text(), "op=explain_change tgt=range_estimate t=last:2:week")
        self.assertEqual(self.parse("when did the coolant flow start falling").query.to_text(), "op=detect_change tgt=coolant_flow t=all")

    def test_state_explanations_stay_with_the_older_engines(self):
        self.assertIsNone(self.parse("why is my battery power negative").query); self.assertIsNone(self.parse("what causes the coolant flow to drop to zero").query)

    def test_listing_everything_selects_all_events(self):
        self.assertEqual(self.parse("show all adas disengagements from yesterday").query.to_text(), "op=list_events t=cal:yesterday ev=ADAS_DISENGAGE evs=all")
        self.assertEqual(self.parse("when was the last time a fault code appeared").query.event.selector, "last")

    def test_range_and_efficiency_why_questions_are_not_taken_over(self):
        q = Query.from_text("op=explain_change tgt=range_estimate t=last:14:day"); self.assertFalse(nl2ir.takes_over(q)); self.assertTrue(nl2ir.takes_over(Query.from_text("op=explain_change tgt=battery_temperature t=last:14:day")))


class SchemaViewTests(unittest.TestCase):
    def test_display_names_round_trip(self):
        v = renamed_view(5)
        for ir in ("op=aggregate tgt=speed agg=mean t=last:7:day c=highway", "op=compare tgt=wh_per_km agg=mean t=last:7:day grp=city,highway", "op=percent_time t=last:7:day c=sig:battery_temperature:gt:35:C", "op=list_events t=all ev=DTC_SET evs=all"):
            self.assertEqual(v.to_registry(v.to_display(ir)), ir); self.assertNotEqual(v.to_display(ir), ir)

    def test_every_item_has_a_unique_display_name(self):
        v = renamed_view(1); names = [v.show(n) for n in v.signals + v.events + v.preds]; self.assertEqual(len(names), len(set(names)))

    def test_a_withheld_signal_cannot_be_named(self):
        v = SchemaView(drop=["tyre_temperature"]); self.assertNotIn("tyre_temperature", v.signals); self.assertNotIn('"tyre_temperature"', v.grammar())

    def test_the_grammar_is_valid_gbnf(self):
        try: from llama_cpp import LlamaGrammar
        except Exception: self.skipTest("llama_cpp not installed")
        for v in (SchemaView(), renamed_view(3), SchemaView(signals={"speed", "battery_soc"})): LlamaGrammar.from_string(v.grammar(), verbose=False)


class ExemplarBankTests(unittest.TestCase):
    def test_every_example_is_a_valid_query_under_its_own_catalog(self):
        for v in (SchemaView(), renamed_view(2)):
            for q, ir, prev in ExemplarBank(v).rows:
                if ir == "op=none": continue
                reg = v.to_registry(ir); qq = Query.from_text(reg); self.assertEqual(validate(qq).codes(), [], (q, ir))

    def test_nearest_returns_a_mix_of_operations_and_respects_the_previous_turn(self):
        b = ExemplarBank(SchemaView()); got = b.nearest("how many hard brakes last week", 4); self.assertTrue(got); self.assertTrue(all(p is None for _, _, p in got))
        fu = b.nearest("and yesterday", 6, prev_ir="op=aggregate tgt=speed agg=mean t=last:7:day"); self.assertIsNotNone(fu[0][2]); self.assertEqual(fu[0][1], "op=aggregate tgt=speed agg=mean t=cal:yesterday")            # the closest example is the follow-up that changes only the period of a speed query


MODEL = os.path.join(ROOT, "models", "LFM2.5-350M-Q4_K_M.gguf")


@unittest.skipUnless(os.path.exists(MODEL), "the small model file is not present")
class PrefixBackend(unittest.TestCase):
    def test_restoring_the_prefix_state_is_fast_and_the_output_obeys_the_grammar(self):
        import tempfile, time
        from vgpt.slm_backend import PrefixLlama
        m = PrefixLlama(MODEL, n_ctx=2048, cache_dir=tempfile.mkdtemp()); m.set_prefix("Write one query line. " + "Signals: speed, battery_soc, battery_temperature. " * 40)
        g = 'root ::= "op=current_value tgt=" ("speed" | "battery_soc") " t=now"'
        t = time.perf_counter(); out, info = m.generate("Q: how charged is the battery", grammar=g); dt_ = time.perf_counter() - t; m.close()
        self.assertTrue(out.startswith("op=current_value tgt=")); self.assertTrue(info["prefill_ms"] >= 0); (PERF and (self.assertLess(dt_, 3.0), self.assertLess(info["prefill_ms"], 1500)))


if __name__ == "__main__": unittest.main()


class StubBackend:
    """Records what the planner would send to the model, so prompt wiring is tested without a model."""
    prefix_tokens = 0; prefix_from_disk = False

    def __init__(self): self.prefix = None; self.users = []; self.grammars = []

    def set_prefix(self, system): self.prefix = system; return 0.0

    def generate(self, user, grammar=None, max_tokens=64, stop_text=("\n",)):
        self.users.append(user); self.grammars.append(grammar); return "op=none", {"ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0, "prompt_tokens": 0, "out_tokens": 1, "prefix_tokens": 0}


class PlannerWiring(unittest.TestCase):
    """Regression guards for an earlier silent failure: options that were configured but never reached the prompt."""
    def planner(self, **kw):
        from vgpt.schema_planner import SchemaPlanner
        be = StubBackend(); return SchemaPlanner(None, backend=be, **kw), be

    def test_retrieved_examples_reach_the_prompt(self):
        p, be = self.planner(bank_k=4); p.compile("how many hard brakes last week"); self.assertIn("SOLVED EXAMPLES", be.users[-1]); self.assertGreaterEqual(be.users[-1].count("\nQ: "), 2)
        p0, be0 = self.planner(bank_k=0); p0.compile("how many hard brakes last week"); self.assertNotIn("SOLVED EXAMPLES", be0.users[-1])

    def test_the_parser_draft_reaches_the_prompt(self):
        p, be = self.planner(); p.compile("why is it hotter", draft="op=explain_change tgt=battery_temperature t=last:14:day"); self.assertIn("Draft query from a simple parser", be.users[-1]); self.assertIn("explain_change", be.users[-1])
        p.compile("why is it hotter"); self.assertNotIn("Draft query", be.users[-1])

    def test_the_previous_turn_reaches_the_prompt(self):
        p, be = self.planner(); p.compile("and yesterday", prev=("average speed last week", "op=aggregate tgt=speed agg=mean t=last:7:day")); self.assertIn("Previous question: average speed last week", be.users[-1])

    def test_no_example_substitutes_a_different_signal_for_the_one_asked(self):
        from vgpt.schema_planner import make_examples
        for v in (SchemaView(), SchemaView(drop=["tyre_temperature"]), SchemaView(drop=["battery_soh", "cabin_temperature"])):
            for q, ir, prev in make_examples(v):
                if ir.startswith("op=capability"):
                    tgt = ir.split("tgt=")[1].split()[0]; words = q.lower()
                    self.assertTrue(any(w in words for w in R.CONCEPTS[v.back[tgt]].label.lower().split()), (q, ir))                         # the signal named in the query is the one named in the question


@testdata.needs("legacy_worlds")
class ChangeOperatorsKeepTheScope(unittest.TestCase):
    def test_conditions_restrict_the_analysis(self):
        qc = QueryCompiler(db("gen_courier_ext"), now=NOW)
        a = qc.execute(Query.from_text("op=detect_change tgt=wh_per_km t=all")); b = qc.execute(Query.from_text("op=detect_change tgt=wh_per_km t=all c=highway"))
        self.assertEqual(b.status, "answered"); self.assertLess(b.result.samples, a.result.samples)                                 # fewer samples: the restriction was applied
        e = qc.execute(Query.from_text("op=explain_change tgt=battery_temperature t=last:14:day c=highway")); u = qc.execute(Query.from_text("op=explain_change tgt=battery_temperature t=last:14:day"))
        self.assertNotEqual(e.result.samples, u.result.samples)

    def test_the_trip_level_model_is_not_claimed_for_a_restricted_scope(self):
        qc = QueryCompiler(db("gen_roadwarrior")); r = qc.execute(Query.from_text("op=explain_change tgt=wh_per_km t=last:14:day c=highway"))
        if r.status == "answered": self.assertIsNone(r.result.extra["contrib"]); self.assertEqual(r.result.extra["contract"]["result_class"], "associated")

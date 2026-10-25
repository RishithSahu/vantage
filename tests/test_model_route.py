"""The model-based product route, tested WITHOUT the language model (deterministic parts): capability cards, the prompt the model is shown, the grammar, call parsing, explicit state edits, faithful compilation, fail-closed validation, existence rendering, the absence of any rule-parser call on the product path, the cache key, and the independence of the format examples from the evaluation sets.
Semantic expectations (does the model understand a question) are NOT asserted here: they live in the accuracy evaluation (bench/eval_model.py). Opt-in tests that load the model run with VGPT_SLM_TESTS=1."""
import json, os, sqlite3, sys, time, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import capreg, plancompile as PC, plangrammar, prompt, signal_registry as R, toolspec as T
from vgpt.retrieval import Retriever
from vgpt.modelplan import PlanResult, Planner, State

DBF = os.path.join(ROOT, "data", "gen_courier_ext.db"); NOW = 1790406000
needs_db = unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
needs_model = unittest.skipUnless(os.environ.get("VGPT_SLM_TESTS") == "1", "opt-in: set VGPT_SLM_TESTS=1 (loads the frozen model)")


def reg():
    return capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True)))


def plan_of(call): fn, a = T.parse_call("[" + call + "]"); return T.call_to_plan(fn, a)


def compile_call(call, prev=None, pending=None, registry=None):
    return PC.compile_plan(plan_of(call), prev, pending, registry or REG, NOW, 0)


@needs_db
class Base(unittest.TestCase):
    pass


REG = reg() if os.path.exists(DBF) else None


class Cards(Base):
    def test_every_capability_has_a_card_with_a_meaning(self):
        for c in REG.items:
            parts = [x.strip() for x in c.card.split("|")]
            self.assertGreaterEqual(len(parts), 4, c.id); self.assertTrue(parts[3], f"{c.id} has no meaning on its card")

    def test_a_card_says_what_the_quantity_means_not_just_its_name(self):
        card = REG.by_id["wh_per_km"].card
        self.assertIn("per kilometre", card); self.assertIn("LESS efficient", card)

    def test_a_known_but_unrecorded_signal_is_marked_on_its_card(self):
        c = [x for x in REG.items if x.kind == "signal" and not x.available]
        self.assertTrue(c, "the reference vehicle should have known-but-unrecorded signals (fuel)")
        for x in c: self.assertIn("NOT RECORDED", x.card)

    def test_registry_meaning_words_survive_the_generic_stop_word_filter(self):
        r = Retriever(REG, None)
        hits, _ = r.search("where are we?")
        self.assertIn("tool:places", [h.id for h in hits])

    def test_a_signal_and_its_event_stream_are_related_on_the_card(self):
        self.assertIn("CAMERA_BLOCKED", REG.by_id["camera_blockage"].card); self.assertIn("camera_blockage", REG.by_id["CAMERA_BLOCKED"].card)

    def test_a_tool_card_shows_the_name_the_call_uses(self):
        card = REG.by_id["tool:tyres"].card
        self.assertTrue(card.startswith("tyres | tool"), card)


class PromptInspection(Base):
    """What the model is actually shown: descriptions, availability and the conversation state are in the rendered input."""
    @classmethod
    def setUpClass(cls):
        cls.P = Planner(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True), load_model=False)

    def test_the_message_carries_retrieved_descriptions(self):
        msg = self.P.build_message("was the camera blocked yesterday?", State(), NOW)[0]
        self.assertIn("Today:", msg); self.assertIn("CAMERA_BLOCKED | event", msg); self.assertIn("fraction of the front camera view that is blocked", msg); self.assertIn("Message: was the camera blocked yesterday?", msg)

    def test_the_message_marks_a_signal_the_vehicle_does_not_record(self):
        msg = self.P.build_message("what is the fuel level", State(), NOW)[0]
        self.assertIn("fuel_level", msg); self.assertIn("NOT RECORDED", msg)

    def test_only_identifiers_that_can_appear_in_a_call_are_shown_as_cards(self):
        ids = self.P.build_message("show my three longest trips", State(), NOW)[2]
        for i in ids: self.assertIn(self.P.reg.by_id[i].kind, ("signal", "event", "tool"), i)

    def test_a_clarification_reply_is_retrieved_with_the_question_it_answers(self):
        st = State(pending={"ask": "Which signal?", "original": "how hot was the battery", "plan": None})
        msg, hits, ids, *_ = self.P.build_message("yesterday", st, NOW)
        self.assertIn("battery_temperature", ids)

    def test_the_plan_so_far_of_an_open_clarification_is_in_the_message(self):
        st = State(pending={"ask": "Which condition?", "original": "did it", "plan": {"fn": "did_condition_hold", "tgt": "speed"}})
        msg = self.P.build_message("when charging", st, NOW)[0]; self.assertIn("Plan so far: did the condition hold | signal=speed", msg); self.assertIn("Awaiting your answer to: \u201cWhich condition?\u201d (about: \u201cdid it\u201d)", msg)

    def test_the_previous_state_ids_are_offered_as_cards_for_a_follow_up(self):
        st = State(prev_plan={"fn": "statistic", "tgt": "motor_temperature", "stat": "mean", "per": "last:5:day"})
        self.assertIn("motor_temperature", self.P.build_message("and for last week?", st, NOW)[2])

    def test_the_instruction_no_longer_tells_the_model_to_name_the_nearest_signal(self):
        low = prompt.instruction().lower()
        self.assertNotIn("nearest known signal", low); self.assertIn("unresolved", low); self.assertIn("merely related", low)

    def test_the_static_prefix_names_the_conditions_and_the_tools_are_declared(self):
        self.assertIn("highway = ", self.P.system); self.assertEqual(len(self.P.tools), len(T.FUNCS))

    def test_format_examples_are_few_and_are_one_per_kind_of_call(self):
        sh = prompt.shots(self.P.reg); self.assertLessEqual(len(sh), 30)
        kinds = {s[1].split("[")[1].split("(")[0] for s in sh}; self.assertGreaterEqual(len(kinds), 12)

    def test_format_examples_are_independent_of_the_evaluation_sets(self):
        from bench.indep_io import load
        from vgpt.retrieval import Embedder
        emb = self.P.emb; ex = [q for q, _, _ in prompt._EX]; texts = []
        for f in ("indep_dev.tsv", "indep_final.tsv", "diag_v1.tsv"):
            p = os.path.join(ROOT, "bench", f)
            if os.path.exists(p): texts += [r["text"] for r in load(p)]
        if not texts: self.skipTest("no evaluation files")
        A = emb.encode(ex); B = emb.encode(texts); sim = (A @ B.T).max(axis=1)
        close = [(q, round(float(x), 2)) for q, x in zip(ex, sim) if float(x) >= 0.93]
        self.assertEqual(close, [], "examples too close to an evaluation question")

    @needs_model
    def test_the_rendered_input_contains_descriptions_availability_and_state(self):
        P = Planner(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True))
        st = State(prev_plan={"fn": "statistic", "tgt": "wh_per_km", "stat": "mean", "per": "last:7:day"}, history=[("Message: how efficient was I", prompt.call_turn('statistic(turn="new", signal="wh_per_km", stat="mean", period="last 1 week")'))])
        text = P.rendered_prompt("what is the fuel level", st, NOW)
        for must in ("List of tools:", "unresolved", "NOT RECORDED", "fuel_level", "Message: how efficient was I", "wh_per_km", "Today:"): self.assertIn(must, text)
        self.assertNotIn("nearest known signal", text)


class Grammar(Base):
    def test_it_compiles_with_and_without_state(self):
        from llama_cpp import LlamaGrammar
        for ws in (True, False): LlamaGrammar.from_string(plangrammar.build(REG, with_state=ws), verbose=False)

    def test_without_a_conversation_state_a_call_cannot_refer_to_one(self):
        def rules(g): return " ".join(l for l in g.splitlines() if l.startswith("f-"))
        no, yes = rules(plangrammar.build(REG, with_state=False)), rules(plangrammar.build(REG, with_state=True))
        for ref in ("follow_up", "correction", "clarification_reply", "conditions_edit", "val-dropl"): self.assertNotIn(ref, no)
        for ref in ("follow_up", "correction", "clarification_reply", "conditions_edit", "val-dropl"): self.assertIn(ref, yes)

    def test_every_registered_identifier_is_admitted_including_known_but_unrecorded(self):
        g = plangrammar.build(REG)
        for i in REG.signal_ids() + REG.event_ids() + REG.condition_ids(): self.assertIn(f'"{i}"', g, i)

    def test_there_is_no_question_word_gate_and_no_function_shortlist(self):
        g = plangrammar.build(REG)
        for fn in T.FUNCS: self.assertIn("f-" + fn.replace("_", "-") + " ::=", g)                           # every function is always available
        self.assertIn("val-period", g); self.assertIn("thr ::=", g); self.assertIn("val-dur", g)             # a period, a threshold and a duration are always representable

    def test_the_thresholds_the_grammar_admits_have_a_signal_an_op_a_value_and_a_unit(self):
        g = plangrammar.build(REG)
        thr = [l for l in g.splitlines() if l.startswith("thr ::=")][0]
        for part in ("signal", "op", "value", "unit"): self.assertIn(part, thr)
        self.assertIn('"km/h"', g); self.assertIn('"%"', g)

    def test_the_event_selector_admits_only_supported_selectors_and_the_period_forms_are_structural(self):
        g = plangrammar.build(REG)
        for form in ('"now"', '"all time"', '"previous week"', '"last weekend"'): self.assertIn(form, g)


class ToolSpec(Base):
    def test_parse_call_accepts_one_wrapped_call_and_rejects_prose_and_truncation(self):
        fn, a = T.parse_call('<|tool_call_start|>[statistic(turn="new", signal="speed", stat="mean")]<|tool_call_end|>'); self.assertEqual(fn, "statistic"); self.assertEqual(a["signal"], "speed")
        for bad in ("I think the average speed.", 'statistic(turn="new", signal="speed"', "[statistic(turn=", "[nope(x=1)]", '[statistic(turn="new", frobnicate="x")]', '[statistic("a")]', "[a(x=1), b(y=2)]"):
            with self.assertRaises(T.CallError, msg=bad): T.parse_call(bad)

    def test_period_forms_are_structural(self):
        cases = {"last 2 weeks": "last:2:week", "last 1 month": "last:1:month", "yesterday": "cal:yesterday", "previous week": "cal:last_week", "this month": "cal:this_month", "all time": "all", "now": "now", "on 2026-03-03": "date:2026-03-03", "last 30 minutes": "last:30:minute"}
        for text, canon in cases.items(): self.assertEqual(T.period_canonical(text), canon); self.assertEqual(T.period_canonical(T.period_natural(canon)), canon)
        self.assertIsNone(T.period_canonical("last fortnight"))
        self.assertEqual(T.period_canonical("last week"), "last:7:day"); self.assertEqual(T.period_canonical("last month"), "last:30:day")           # the product's documented convention
        g = plangrammar.build(REG); self.assertIn('"last week"', g); self.assertIn('"last month"', g)

    def test_durations_are_amount_and_unit(self):
        self.assertEqual(T.duration_seconds("10 minutes"), 600); self.assertEqual(T.duration_seconds("2 hours"), 7200); self.assertIsNone(T.duration_seconds("soon"))
        self.assertEqual(T.duration_natural(600), "10 minutes")

    def test_a_threshold_is_signal_comparison_value_unit(self):
        p = plan_of('statistic(turn="new", signal="wh_per_km", conditions=[{"signal":"ambient_temperature","op":"gt","value":20,"unit":"C"}])')
        self.assertEqual(p["cond"], [{"signal": "ambient_temperature", "op": "gt", "value": 20, "unit": "C"}])

    def test_the_plan_and_the_call_it_came_from_agree(self):
        for call in ('statistic(turn="new", signal="speed", stat="mean", period="last 3 days")', 'count_events(turn="new", event="HARSH_BRAKE", period="yesterday")', 'rank_trips(turn="new", entity="trip", metric="distance_driven", order="desc", k=3)',
                     'event_context(turn="new", event="TPMS_LOW", which="last", before="10 minutes")', 'compare(turn="new", signal="motor_temperature", stat="mean", groups=["highway","city"])'):
            p = plan_of(call); self.assertEqual(plan_of(T.plan_to_call(p)), p, call)


class StateEdits(Base):
    BASE = plan_of('statistic(turn="new", signal="ambient_temperature", stat="mean", period="last 7 days", conditions=[{"signal":"ambient_temperature","op":"gt","value":20,"unit":"C"}])')

    def resolved(self, call, prev=None, pending=None):
        o = compile_call(call, prev, pending); self.assertEqual(o.kind, "query", o.reason); return o.plan

    def test_adding_the_other_bound_keeps_the_first(self):
        base = self.resolved('statistic(turn="new", signal="wh_per_km", conditions=[{"signal":"ambient_temperature","op":"gt","value":20,"unit":"C"}])')
        p = self.resolved('statistic(turn="follow_up", conditions=[{"signal":"ambient_temperature","op":"lt","value":30,"unit":"C"}], conditions_edit="add")', base)
        self.assertEqual({(c["op"], c["value"]) for c in p["cond"]}, {("gt", 20), ("lt", 30)})            # "above 20 and below 30" needs both bounds

    def test_replace_swaps_the_whole_set_remove_removes_only_the_named(self):
        base = self.resolved('statistic(turn="new", signal="speed", conditions=["highway", {"signal":"ambient_temperature","op":"gt","value":20,"unit":"C"}])')
        p = self.resolved('statistic(turn="correction", conditions=["city"], conditions_edit="replace")', base); self.assertEqual(p["cond"], ["city"])
        p = self.resolved('statistic(turn="follow_up", conditions=["highway"], conditions_edit="remove")', base); self.assertEqual(len(p["cond"]), 1); self.assertIsInstance(p["cond"][0], dict)

    def test_a_follow_up_keeps_what_it_does_not_change(self):
        base = self.resolved('statistic(turn="new", signal="motor_temperature", stat="mean", period="last 5 days", conditions=["driving"])')
        p = self.resolved('statistic(turn="follow_up", period="yesterday")', base)
        self.assertEqual((p["tgt"], p["stat"], p["cond"], p["per"]), ("motor_temperature", "mean", ["driving"], "cal:yesterday"))

    def test_drop_removes_a_part_of_the_previous_plan(self):
        base = self.resolved('statistic(turn="new", signal="speed", stat="median", period="last 5 days", conditions=["driving"])')
        p = self.resolved('statistic(turn="follow_up", drop=["period", "conditions"])', base); self.assertNotIn("per", p); self.assertNotIn("cond", p); self.assertEqual(p["tgt"], "speed")

    def test_a_new_turn_stands_alone(self):
        base = self.resolved('statistic(turn="new", signal="speed", stat="median", period="last 5 days", conditions=["driving"])')
        p = self.resolved('current_value(turn="new", signal="battery_soc")', base); self.assertNotIn("cond", p); self.assertNotIn("per", p)

    def test_a_clarification_reply_continues_the_plan_so_far_not_the_previous_answer(self):
        prev = self.resolved('current_value(turn="new", signal="battery_soc")')
        partial = {"fn": "did_condition_hold", "op": "exists", "tgt": "speed", "exists_condition": True}
        o = compile_call('did_condition_hold(turn="clarification_reply", conditions=["highway"])', prev, {"plan": partial}); self.assertEqual(o.kind, "query", o.reason); self.assertEqual(o.plan["tgt"], "speed")

    def test_two_driving_modes_at_once_is_a_structural_conflict(self):
        base = self.resolved('statistic(turn="new", signal="speed", conditions=["highway"])')
        self.assertEqual(compile_call('statistic(turn="follow_up", conditions=["city"], conditions_edit="add")', base).kind, "invalid")


class Fidelity(Base):
    """The compiled query carries exactly what the plan states."""
    def q(self, call, prev=None): o = compile_call(call, prev); self.assertEqual(o.kind, "query", o.reason); return o.query

    def test_an_explicit_statistic_is_never_overridden_even_for_an_accumulating_quantity(self):
        self.assertEqual(self.q('statistic(turn="new", signal="energy_used", stat="mean", period="last 7 days")').agg, "mean")

    def test_an_omitted_statistic_gets_a_documented_default_that_is_disclosed(self):
        q = self.q('statistic(turn="new", signal="speed", period="last 7 days")'); self.assertEqual(q.agg, "mean"); self.assertTrue(any("No statistic" in n for n in q.notes))
        q = self.q('statistic(turn="new", signal="energy_used", period="last 7 days")'); self.assertEqual(q.agg, "sum"); self.assertTrue(any("total" in n for n in q.notes))

    def test_energy_used_is_not_turned_into_energy_per_distance(self):
        self.assertEqual(self.q('explain_change(turn="new", signal="energy_used", period="last 14 days")').target, "energy_used")
        self.assertEqual(self.q('trend(turn="new", signal="energy_used")').target, "energy_used")

    def test_a_signal_condition_is_not_replaced_by_an_event_stream(self):
        o = compile_call('did_condition_hold(turn="new", signal="camera_blockage")'); self.assertEqual(o.kind, "invalid"); self.assertIn("condition", o.say.lower())     # a question, not a substituted event
        q = self.q('did_condition_hold(turn="new", signal="camera_blockage", conditions=[{"signal":"camera_blockage","op":"gt","value":0.5}], period="yesterday")')
        self.assertIsNone(q.event); self.assertEqual(q.target, "camera_blockage")

    def test_the_stated_period_is_kept_and_nothing_is_substituted(self):
        self.assertEqual(self.q('statistic(turn="new", signal="speed", stat="mean", period="last 2 weeks")').scope.time.to_text(), "last:2:week")
        self.assertEqual(self.q('statistic(turn="new", signal="speed", stat="mean", period="previous week")').scope.time.to_text(), "cal:last_week")

    def test_a_domain_tool_keeps_the_requested_period_or_reports_the_limitation(self):
        o = compile_call('use_tool(turn="new", tool="tyres", period="last 2 weeks")'); self.assertEqual((o.kind, o.window), ("tool", "14"))
        o = compile_call('use_tool(turn="new", tool="health", period="last 3 days")'); self.assertEqual(o.kind, "unsupported"); self.assertIn("have not used another", o.reason)
        o = compile_call('use_tool(turn="new", tool="places", period="yesterday")'); self.assertEqual(o.window, "y")

    def test_no_period_for_a_tool_uses_its_standard_window_and_says_so(self):
        o = compile_call('use_tool(turn="new", tool="tyres")'); self.assertEqual(o.kind, "tool"); self.assertTrue(o.notes)

    def test_every_explicit_field_survives_into_the_query(self):
        o = compile_call('statistic(turn="new", signal="wh_per_km", stat="p95", period="last 2 weeks", conditions=["city", {"signal":"ambient_temperature","op":"gt","value":20,"unit":"C"}], unit="kWh/100km")')
        self.assertEqual(o.kind, "query"); self.assertEqual(PC.fidelity(o.plan, o.query), [])

    def test_the_fidelity_check_catches_a_tampered_query(self):
        o = compile_call('statistic(turn="new", signal="speed", stat="mean", period="last 2 weeks")')
        o.query.agg = "max"; self.assertIn("stat", PC.fidelity(o.plan, o.query))
        o = compile_call('statistic(turn="new", signal="speed", stat="mean", period="last 2 weeks")'); o.query.scope.time = PC.TimeSpec("relative", 7, "day"); self.assertIn("per", PC.fidelity(o.plan, o.query))

    def test_a_calculation_the_engine_does_not_define_is_reported_not_changed(self):
        o = compile_call('statistic(turn="new", signal="drive_mode", stat="mean", period="last 7 days")')
        self.assertEqual(o.kind, "unsupported"); self.assertIn("mean", o.reason)

    def test_the_resolved_plan_round_trips_through_the_query(self):
        o = compile_call('extreme(turn="new", signal="speed", which="max", period="this week")'); back = PC.query_to_plan(o.query)
        self.assertEqual((back["fn"], back["tgt"], back["stat"], back["per"]), ("extreme", "speed", "max", "cal:this_week"))


class Grounding(Base):
    """Optional arguments a small model can invent must cite words, and the quote is checked against the source it claims. A failed check makes the interpretation UNRESOLVED: nothing is dropped, nothing is defaulted in its place, no broader query is executed."""
    def go(self, call, message, **kw): return PC.compile_plan(plan_of(call), kw.get("prev"), kw.get("pending"), REG, NOW, 0, message=message, original=kw.get("original"), support=kw.get("support"))

    def test_a_period_whose_words_occur_in_the_message_is_kept_with_current_provenance(self):
        o = self.go('statistic(turn="new", signal="speed", stat="mean", period_words="past 3 days", period="last 3 days")', "average speed over the past 3 days")
        self.assertEqual(o.query.scope.time.to_text(), "last:3:day"); self.assertEqual(o.provenance["per"], "current")

    def test_a_fabricated_quotation_makes_the_interpretation_unresolved_and_executes_nothing(self):
        o = self.go('statistic(turn="new", signal="speed", stat="mean", period_words="in recent weeks", period="last 2 weeks")', "average speed")
        self.assertEqual(o.kind, "ungrounded"); self.assertIsNone(o.query); self.assertEqual(o.failures[0]["key"], "per"); self.assertEqual(o.proposed["per"], "last:2:week")       # the proposed field stays in the record
        self.assertTrue(o.say)

    def test_a_fabricated_restriction_is_never_answered_for_all_driving(self):
        o = self.go('statistic(turn="new", signal="speed", stat="mean", period_words="yesterday", period="yesterday", conditions_words="highway only", conditions=["highway"])', "average speed yesterday")
        self.assertEqual(o.kind, "ungrounded"); self.assertIsNone(o.query); self.assertEqual({f["key"] for f in o.failures}, {"cond"}); self.assertEqual(o.pending_plan["per"], "cal:yesterday"); self.assertNotIn("cond", o.pending_plan)

    def test_no_default_replaces_a_failed_explicit_field(self):
        o = self.go('statistic(turn="new", signal="speed", stat="mean", period_words="in recent weeks", period="last 2 weeks")', "average speed")
        self.assertNotEqual(o.kind, "query")

    def test_a_missing_quote_is_ungrounded(self):
        self.assertEqual(self.go('statistic(turn="new", signal="speed", stat="mean", conditions=["driving"])', "average speed").kind, "ungrounded")

    def test_the_check_is_of_copying_and_a_paraphrase_is_not_a_quote(self):
        o = self.go('statistic(turn="new", signal="speed", stat="mean", conditions_words="Highway", conditions=["highway"])', "Average speed on the Highway?"); self.assertEqual([c.predicate for c in o.query.scope.conditions], ["highway"])
        self.assertEqual(self.go('statistic(turn="new", signal="speed", stat="mean", conditions_words="motorway", conditions=["highway"])', "Average speed on the Highway?").kind, "ungrounded")

    def test_a_required_argument_that_is_ungrounded_is_a_question_not_a_guess(self):
        o = self.go('time_share(turn="new", conditions_words="while it rained", conditions=["raining"])', "share of time"); self.assertEqual(o.kind, "ungrounded"); self.assertTrue(o.say)

    def test_a_quotation_is_checked_against_the_source_it_claims(self):
        o = self.go('statistic(turn="clarification_reply", signal="speed", period_words="last 3 days", period="last 3 days")', "the speed", original="how was it over the last 3 days")           # a clarification reply may quote the ORIGINAL question it answers
        self.assertEqual(o.kind, "query"); self.assertEqual(o.provenance["per"], "original")
        self.assertEqual(self.go('statistic(turn="clarification_reply", signal="speed", period_words="last 3 days", period="last 3 days")', "the speed").kind, "ungrounded")

    def test_a_follow_up_may_inherit_a_period_without_repeating_its_words(self):
        base = self.go('statistic(turn="new", signal="speed", stat="mean", period_words="last 5 days", period="last 5 days")', "average speed in the last 5 days")
        o = self.go('statistic(turn="follow_up", conditions_words="on the highway", conditions=["highway"], conditions_edit="add")', "only on the highway", prev=base.plan)
        self.assertEqual(o.kind, "query"); self.assertEqual(o.query.scope.time.to_text(), "last:5:day"); self.assertEqual(o.provenance["per"], "state"); self.assertEqual(o.provenance["cond"], "current"); self.assertEqual(o.plan["src"]["per"], "last 5 days")

    def test_the_support_check_catches_correct_words_for_the_wrong_value(self):
        sup = lambda key, words, value: not (words == "yesterday" and value != "cal:yesterday")                      # stands in for a semantic judge: the words exist but do not mean the chosen value
        o = self.go('statistic(turn="new", signal="speed", period_words="yesterday", period="last month")', "average speed yesterday", support=sup)
        self.assertEqual(o.kind, "ungrounded"); self.assertIn("do not support", o.failures[0]["why"])
        self.assertEqual(self.go('statistic(turn="new", signal="speed", period_words="yesterday", period="yesterday")', "average speed yesterday", support=sup).kind, "query")

    def test_the_resolved_plan_is_shown_back_with_its_quoted_words(self):
        base = self.go('statistic(turn="new", signal="speed", period_words="last 5 days", period="last 5 days")', "speed last 5 days")
        self.assertIn('period_words="last 5 days"', T.plan_to_call(base.plan)); self.assertIn("period=last 5 days", T.plan_to_frame(base.plan))

    def test_the_grammar_makes_the_quote_and_the_value_an_inseparable_pair(self):
        g = plangrammar.build(REG); rule = [l for l in g.splitlines() if l.startswith("f-statistic")][0]
        self.assertIn('"period_words=" val-span ", " "period=" val-period', rule); self.assertNotIn('( ", " "period=" val-period )?', rule)

    def test_schema_operation_grammar_can_restrict_without_language_rules(self):
        g = plangrammar.build(REG, style="json", only="use_tool")
        root = g.splitlines()[0]
        self.assertIn("f-use-tool", root)
        self.assertNotIn("f-current-value", root)
        no_period = plangrammar.build(REG, style="json", only="use_tool", exclude={"period"})
        tool_rule = next(line for line in no_period.splitlines() if line.startswith("f-use-tool"))
        self.assertNotIn('"period"', tool_rule)

    def test_without_the_repair_option_there_is_no_second_generation(self):
        from vgpt.modelbrain import ModelBrain
        fp = FakePlanner(['statistic(turn="new", signal="speed", stat="mean", period_words="in recent weeks", period="last 2 weeks")', 'statistic(turn="new", signal="speed", stat="mean", period_words="last 3 days", period="last 3 days")'])
        b = ModelBrain(DBF, planner=fp, now=NOW); b.ask("average speed"); self.assertEqual(b.outcome, "ambiguous"); self.assertEqual(len(fp.calls), 1); self.assertIsNone(b.repair)

    def test_the_embedding_support_check_never_acts_on_a_plan_by_default(self):
        from vgpt.modelbrain import ModelBrain
        class Verdict(FakePlanner):
            support = staticmethod(lambda key, words, value: False)                                      # a judge that rejects everything
        b = ModelBrain(DBF, planner=Verdict(['statistic(turn="new", signal="speed", stat="mean", period_words="yesterday", period="yesterday")']), now=NOW); b.ask("average speed yesterday")
        self.assertEqual(b.outcome, "answered"); self.assertEqual(b.support_diag, {"per": False})                # recorded as a diagnostic, not enforced

    def test_the_brain_repairs_once_and_otherwise_asks_a_specific_question(self):
        from vgpt.modelbrain import ModelBrain
        b = ModelBrain(DBF, planner=FakePlanner(['statistic(turn="new", signal="speed", stat="mean", period_words="in recent weeks", period="last 2 weeks")', 'statistic(turn="new", signal="speed", stat="mean", period_words="last 3 days", period="last 3 days")']), now=NOW, repair=True)
        b.ask("average speed over the last 3 days"); self.assertEqual(b.outcome, "answered"); self.assertEqual(b.last_qc.query.scope.time.to_text(), "last:3:day"); self.assertEqual(b.repair["failures"][0]["key"], "per")
        b = ModelBrain(DBF, planner=FakePlanner(['statistic(turn="new", signal="speed", stat="mean", period_words="in recent weeks", period="last 2 weeks")'] * 2), now=NOW, repair=True)
        r = b.ask("average speed"); self.assertEqual(b.outcome, "ambiguous"); self.assertIsNone(b.last_qc); self.assertIn("could not find where you asked", r["answer"]); self.assertEqual(r["facts"]["ungrounded"][0]["value"], "last:2:week")
        self.assertEqual(b.state.pending["plan"]["tgt"], "speed"); self.assertNotIn("per", b.state.pending["plan"])


class Validation(Base):
    def test_an_unmentioned_native_output_unit_is_a_schema_default_not_a_clarification(self):
        plan = plan_of('current_value(turn="new", signal="speed", unit_words="km/h", unit="km/h")')
        o = PC.compile_plan(plan, None, None, REG, NOW, 0, message="what is speed")
        self.assertEqual(o.kind, "query", o.reason)
        self.assertEqual(o.query.target, "speed")
        self.assertIsNone(o.query.unit)

    def test_unregistered_identifiers_are_rejected(self):
        self.assertEqual(compile_call('statistic(turn="new", signal="warp_core", stat="mean")').kind, "invalid")
        self.assertEqual(compile_call('count_events(turn="new", event="NOPE")').kind, "invalid")

    def test_a_threshold_unit_must_match_the_signal(self):
        o = compile_call('statistic(turn="new", signal="speed", conditions=[{"signal":"ambient_temperature","op":"gt","value":20,"unit":"km/h"}])'); self.assertEqual(o.kind, "invalid"); self.assertIn("unit", o.reason)

    def test_a_numeric_threshold_on_a_categorical_signal_is_rejected(self):
        self.assertEqual(compile_call('statistic(turn="new", signal="speed", conditions=[{"signal":"drive_mode","op":"gt","value":1}])').kind, "invalid")

    def test_a_future_date_is_rejected(self):
        self.assertEqual(compile_call('statistic(turn="new", signal="speed", period="on 2099-01-01")').kind, "invalid")

    def test_required_arguments_missing_ask_a_specific_question_and_keep_the_plan_so_far(self):
        o = compile_call('extreme(turn="new", signal="speed")'); self.assertEqual(o.kind, "invalid"); self.assertTrue(o.say); self.assertEqual(o.pending_plan["tgt"], "speed")
        o = compile_call('rank_trips(turn="new", entity="trip")'); self.assertTrue(o.say)

    def test_a_follow_up_with_nothing_to_continue_is_an_incomplete_plan_not_a_guess(self):
        o = compile_call('statistic(turn="follow_up", period="yesterday")'); self.assertEqual(o.kind, "invalid"); self.assertTrue(o.say)

    def test_an_exception_while_validating_is_a_validation_failure_not_a_pass(self):
        orig = PC.validate
        try:
            PC.validate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
            o = compile_call('statistic(turn="new", signal="speed", stat="mean")'); self.assertEqual(o.kind, "invalid"); self.assertIsNone(o.query)
        finally: PC.validate = orig

    def test_unresolved_declined_and_chat_never_carry_a_query(self):
        for call, kind in (('unresolved(what="oil pressure")', "unresolved"), ('decline(reason="control")', "refuse"), ('say(text="thanks")', "chat"), ('ask_user(question="Which one?")', "ambiguous")):
            o = compile_call(call); self.assertEqual(o.kind, kind); self.assertIsNone(o.query)


class FakePlanner:
    """Stands in for the language model: returns the call a test gives it."""
    def __init__(self, calls):
        from vgpt.dialects import Pythonic
        self.reg = REG; self.calls = list(calls); self.seen = []; self.dialect = Pythonic()

    def plan(self, text, state, now, tz=0, **k):
        call = self.calls.pop(0); self.seen.append((text, state.prev_plan, state.pending)); fn, a = T.parse_call("[" + call + "]")
        return PlanResult(raw=call + "]", fn=fn, args=a, plan=T.call_to_plan(fn, a), card_ids=["speed"], message="Message: " + text, timing={"retrieve_ms": 1.0, "generate_ms": 2.0})


class Brain(Base):
    def make(self, calls):
        from vgpt.modelbrain import ModelBrain
        return ModelBrain(DBF, planner=FakePlanner(calls), now=NOW)

    def test_a_valid_plan_is_executed_and_becomes_the_conversation_state(self):
        b = self.make(['statistic(turn="new", signal="speed", stat="p95", period_words="last week", period="last week")', 'statistic(turn="follow_up", period_words="yesterday", period="yesterday")'])
        r = b.ask("highest speed last week"); self.assertEqual(b.outcome, "answered"); self.assertEqual(r["path"], "qc"); self.assertIn("I read this as", r["answer"])
        b.ask("and yesterday?"); self.assertEqual(b.last_qc.query.scope.time.to_text(), "cal:yesterday"); self.assertEqual(b.last_qc.query.target, "speed")
        self.assertEqual(b.planner().seen[1][1]["tgt"], "speed")                                           # the second turn saw the first turn's resolved plan

    def test_an_invalid_plan_is_never_executed(self):
        b = self.make(['statistic(turn="new", signal="warp_core", stat="mean")'])
        calls = []; orig = b.qc().execute
        b.qc().execute = lambda q: calls.append(q) or orig(q)
        r = b.ask("what about the warp core"); self.assertEqual(b.outcome, "invalid_plan"); self.assertEqual(calls, []); self.assertIsNone(b.last_qc)

    def test_a_question_about_an_incomplete_plan_does_not_make_it_executable(self):
        b = self.make(['extreme(turn="new", signal="speed")', 'extreme(turn="clarification_reply", which="max")'])
        r = b.ask("speed"); self.assertEqual(b.outcome, "ambiguous"); self.assertIsNone(b.last_qc); self.assertIsNone(b.state.prev_plan); self.assertEqual(b.state.pending["plan"]["tgt"], "speed")
        b.ask("the highest"); self.assertEqual(b.outcome, "answered"); self.assertEqual(b.last_qc.query.target, "speed"); self.assertIsNone(b.state.pending)

    def test_an_echoed_clarification_gets_one_clean_semantic_retry(self):
        b = self.make(['ask_user(question="What is speed?")', 'current_value(turn="new", signal="speed")'])
        r = b.ask("what is speed")
        self.assertIn(b.outcome, ("answered", "no_data"))
        self.assertEqual(b.last_qc.query.target, "speed")
        self.assertEqual(b.recovery["first_call"], 'ask_user(question="What is speed?")]')
        self.assertFalse(b.recovery["ignored_pending"])

    def test_a_complete_new_request_recovers_from_poisoned_pending_state(self):
        b = self.make(['extreme(turn="new", signal="speed")',
                       'current_value(turn="clarification_reply", signal="speed", unit_words="C", unit="C")',
                       'current_value(turn="new", signal="speed")'])
        b.ask("speed")
        self.assertEqual(b.outcome, "ambiguous")
        b.ask("how fast are we driving")
        self.assertIn(b.outcome, ("answered", "no_data"))
        self.assertEqual(b.last_qc.query.target, "speed")
        self.assertIsNone(b.state.pending)
        self.assertTrue(b.recovery["ignored_pending"])

    def test_a_refusal_or_a_failure_never_erases_the_conversation_state(self):
        b = self.make(['current_value(turn="new", signal="battery_soc")', 'decline(reason="control")', 'unresolved(what="oil pressure")'])
        b.ask("charge"); b.ask("open the window"); self.assertEqual(b.outcome, "refused"); b.ask("oil pressure"); self.assertEqual(b.outcome, "unresolved_target")
        self.assertEqual(b.state.prev_plan["tgt"], "battery_soc")

    def test_a_model_that_cannot_form_a_plan_is_an_explicit_failure_not_a_route_change(self):
        class Broken(FakePlanner):
            def plan(self, text, state, now, tz=0, **k): return PlanResult(error="plan truncated at the token limit", timing={})
        from vgpt.modelbrain import ModelBrain
        b = ModelBrain(DBF, planner=Broken([]), now=NOW); r = b.ask("anything"); self.assertEqual(b.outcome, "failed_plan"); self.assertTrue(r["facts"]["failed"]); self.assertNotEqual(r["path"], "qc")

    def test_a_planner_exception_is_a_failure_not_a_crash(self):
        class Boom(FakePlanner):
            def plan(self, *a, **k): raise RuntimeError("model gone")
        from vgpt.modelbrain import ModelBrain
        b = ModelBrain(DBF, planner=Boom([]), now=NOW); b.ask("anything"); self.assertEqual(b.outcome, "failed_plan")

    def test_unsupported_calculation_and_missing_data_have_their_own_outcomes(self):
        b = self.make(['statistic(turn="new", signal="drive_mode", stat="mean", period_words="last 7 days", period="last 7 days")', 'current_value(turn="new", signal="fuel_level")'])
        b.ask("average drive mode over the last 7 days"); self.assertEqual(b.outcome, "unsupported_calculation")
        b.ask("fuel level"); self.assertEqual(b.outcome, "missing_data")

    def test_the_unresolved_target_names_the_gap_and_does_not_substitute(self):
        b = self.make(['unresolved(what="engine oil pressure")']); r = b.ask("oil pressure"); self.assertEqual(b.outcome, "unresolved_target"); self.assertIn("engine oil pressure", r["answer"]); self.assertIsNone(b.last_qc)

    def test_a_domain_tool_runs_with_an_evidence_statement(self):
        b = self.make(['use_tool(turn="new", tool="tyres", period_words="last 2 weeks", period="last 2 weeks")']); r = b.ask("tyres over the last 2 weeks"); self.assertEqual(b.outcome, "tool"); self.assertIn("Evidence: read-only domain tool", r["answer"])

    def test_timing_is_reported_in_separate_stages(self):
        b = self.make(['current_value(turn="new", signal="battery_soc")']); b.ask("charge")
        for k in ("retrieve_ms", "generate_ms", "compile_ms", "execute_ms", "total_ms"): self.assertIn(k, b.timing)

    def test_no_rule_parser_or_lexical_check_is_called_on_the_product_path(self):
        from vgpt import interpretation, nl2ir, nlu, policy
        trips = []
        def trap(name): return lambda *a, **k: trips.append(name) or (_ for _ in ()).throw(AssertionError(f"{name} called on the product path"))
        saved = [(nl2ir, "parse", nl2ir.parse), (policy, "decide", policy.decide), (interpretation, "interpret", interpretation.interpret)]
        try:
            for m, n, _ in saved: setattr(m, n, trap(n))
            b = self.make(['count_events(turn="new", event="HARSH_BRAKE", period_words="yesterday", period="yesterday")', 'ask_user(question="Which one?")', 'say(text="help")', 'decline(reason="offtopic")'])
            for q in ("harsh brakes yesterday", "it", "hi", "tell me a joke"): b.ask(q)
        finally:
            for m, n, f in saved: setattr(m, n, f)
        self.assertEqual(trips, [])

    def test_the_product_factory_returns_the_semantic_front_door_and_the_legacy_baseline_is_separate(self):
        """The autoregressive planner route is retired (vgpt/product.py): the product is the compact-classifier Brain, and the baseline has no model on the answer path."""
        from vgpt.brain import Brain as Legacy
        from vgpt.product import make_brain, make_legacy_brain

        class Classifier:                                               # stands in for the loaded Qwen2.5-0.5B classifier; the factory only needs it to exist
            pass
        b = make_brain(DBF, now=NOW, slm=Classifier()); self.assertIs(type(b), Legacy); self.assertIsNotNone(b.slm)
        base = make_legacy_brain(DBF, now=NOW); self.assertIs(type(base), Legacy); self.assertIsNone(base.slm)

    def test_model_written_text_is_cleaned_before_it_is_shown(self):
        b = self.make(['ask_user(question="Which <script>x</script> one?\\u0007")']); r = b.ask("it"); self.assertNotIn("<script>", r["answer"]); self.assertNotIn("\x07", r["answer"])


class Existence(Base):
    class R:
        def __init__(self, value, **contract):
            self.result = type("X", (), {"value": value, "extra": {"contract": contract}})()

    def render(self, value, **contract):
        from vgpt.modelbrain import render_existence
        return render_existence("There were N.", self.R(value, **contract), "camera blocked")

    def test_yes_only_when_there_is_a_positive_observation(self): self.assertTrue(self.render(2, completeness="partial").startswith("Yes."))

    def test_zero_with_complete_records_is_no_recorded_occurrence_never_a_bare_no(self):
        t = self.render(0, completeness="complete", recorded_share=1.0); self.assertTrue(t.startswith("No recorded occurrence"), t); self.assertFalse(t.startswith("No."))

    def test_zero_with_incomplete_or_unknown_coverage_cannot_establish_absence(self):
        for k in (dict(completeness="partial", recorded_share=0.97), dict(completeness="complete", recorded_share=None), dict(completeness="complete", recorded_share=1.0, coverage=0.4) and dict(completeness="partial", recorded_share=1.0, coverage=0.4)):
            t = self.render(0, **k); self.assertTrue(t.startswith("Cannot establish absence"), t)

    def test_there_is_no_ninety_percent_threshold(self):
        t = self.render(0, completeness="partial", recorded_share=0.95); self.assertTrue(t.startswith("Cannot establish absence"), t)                  # 95% recorded is still not complete


class CacheKey(unittest.TestCase):
    def key(self, **over):
        from vgpt.slm_backend import PrefixLlama
        pl = object.__new__(PrefixLlama); pl.identity = "abc"; pl.n_ctx = 6144; pl.n_batch = 64; pl.template = "tpl"
        for k, v in over.items(): setattr(pl, k, v)
        return pl.cache_key("prefix text")

    def test_the_persisted_state_is_keyed_by_model_context_batch_template_and_prefix(self):
        base = self.key(); self.assertEqual(base, self.key())
        for over in (dict(identity="other"), dict(n_ctx=4096), dict(n_batch=32), dict(template="tpl2")): self.assertNotEqual(base, self.key(**over), over)
        from vgpt.slm_backend import PrefixLlama
        pl = object.__new__(PrefixLlama); pl.identity = "abc"; pl.n_ctx = 6144; pl.n_batch = 64; pl.template = "tpl"; self.assertNotEqual(base, pl.cache_key("another prefix"))

    def test_the_runtime_version_is_part_of_the_key(self):
        import llama_cpp
        from vgpt.slm_backend import PrefixLlama
        pl = object.__new__(PrefixLlama); pl.identity = "abc"; pl.n_ctx = 6144; pl.n_batch = 64; pl.template = "tpl"; k1 = pl.cache_key("p"); old = llama_cpp.__version__
        try: llama_cpp.__version__ = "0.0.0-test"; k2 = pl.cache_key("p")
        finally: llama_cpp.__version__ = old
        self.assertNotEqual(k1, k2)


@needs_model
class WithModel(Base):
    @classmethod
    def setUpClass(cls): cls.P = Planner(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True))

    def test_cached_and_ordinary_inference_produce_the_same_call(self):
        P = self.P
        for q in ("how many hard brakes did I have this month", "was the camera blocked yesterday?", "average speed over the last 3 days"):
            msg = P.build_message(q, State(), NOW)[0]; cached, _ = P.lm.generate(msg, grammar=P.grammar_new, max_tokens=130, stop_text=None, prefix="plan", assistant_prefix=P.dialect.open_text)
            plain = P.lm.generate_uncached(P.system, msg, P.shots, grammar=P.grammar_new, max_tokens=130, stop_text=None, assistant_prefix=P.dialect.open_text, tools=P.tools)
            self.assertEqual(cached, plain, q)

    def test_the_planner_reports_the_separated_timings(self):
        r = self.P.plan("how many hard brakes did I have this month", State(), NOW)
        for k in ("retrieve_ms", "render_ms", "state_ms", "prefill_ms", "decode_ms", "generate_ms", "plan_ms"): self.assertIn(k, r.timing)
        self.assertIsNotNone(r.plan, r.error)

    def test_an_end_to_end_turn_is_structurally_safe(self):
        from vgpt.modelbrain import ModelBrain
        b = ModelBrain(DBF, planner=self.P, now=NOW)
        for q in ("open the trunk", "average speed over the last 3 days", "and yesterday?"):
            r = b.ask(q); self.assertIn(b.outcome, ("answered", "no_data", "missing_data", "unsupported_calculation", "invalid_plan", "ambiguous", "unresolved_target", "refused", "chat", "failed_plan", "tool", "execution_error")); self.assertTrue(r["answer"])


if __name__ == "__main__": unittest.main()

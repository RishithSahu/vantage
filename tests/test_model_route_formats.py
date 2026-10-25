"""Output interfaces (tool-call dialects and the frame), the one authoritative schema, retrieval budgets, the bounded expansion and the rendered prompts, tested without running a language model.
The tokenizer / chat-template checks read only the vocabulary and metadata of the local GGUF files (vocab_only: no weights are loaded)."""
import json, os, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import capreg, dialects, plancompile as PC, plangrammar, prompt, schema, signal_registry as R, toolspec as T
from vgpt.gbnf import Grammar
from vgpt.modelplan import PlanResult, Planner, State

DBF = os.path.join(ROOT, "data", "gen_courier_ext.db"); NOW = 1790406000
needs_db = unittest.skipUnless(os.path.exists(DBF), "fixture database missing")
REG = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True))) if os.path.exists(DBF) else None
STYLE = {"lfm": "py", "granite": "json", "frame": "frame"}


def strip_wrapper(d, text):
    """What the grammar covers: the call itself, after the assistant turn has been opened with the template's own marker."""
    if d.name == "lfm": return text[len(d.open_text):-len(T.CALL_CLOSE)]
    if d.name == "granite": return text[len(d.open_text):-len("\n</tool_call>")]
    return text


@needs_db
class Interfaces(unittest.TestCase):
    def test_every_example_is_accepted_by_the_grammar_of_its_own_dialect(self):
        for name in ("lfm", "granite", "frame"):
            d = dialects.by_name(name); g = Grammar(plangrammar.build(REG, style=STYLE[name], with_state=True)); bad = []
            for user, asst in prompt.shots(REG, d):
                if not g.accepts(strip_wrapper(d, asst)): bad.append(asst[:140])
            self.assertEqual(bad, [], name)

    def test_the_new_conversation_grammar_rejects_follow_up_calls(self):
        for name in ("lfm", "granite", "frame"):
            d = dialects.by_name(name); gn = Grammar(plangrammar.build(REG, style=STYLE[name], with_state=False)); gs = Grammar(plangrammar.build(REG, style=STYLE[name], with_state=True))
            fu = strip_wrapper(d, d.assistant_text("statistic", {"turn": "follow_up", "period_words": "yesterday", "period": "yesterday"} if name != "frame" else {"turn": "follow_up", "period": "yesterday"}))
            self.assertTrue(gs.accepts(fu), name); self.assertFalse(gn.accepts(fu), name)

    def test_a_call_survives_the_round_trip_through_every_dialect(self):
        args = {"turn": "new", "signal": "wh_per_km", "stat": "mean", "period_words": "last 2 weeks", "period": "last 2 weeks", "conditions_words": "above 20", "conditions": [{"signal": "ambient_temperature", "op": "gt", "value": 20, "unit": "C"}],
                "selection": {"entity": "trip", "metric": "wh_per_km", "order": "desc", "k": 2}}
        for name in ("lfm", "granite"):
            d = dialects.by_name(name); fn, back = d.parse(strip_wrapper(d, d.assistant_text("statistic", args))); self.assertEqual((fn, back), ("statistic", args), name)
        d = dialects.Frame(); noq = {k: v for k, v in args.items() if not k.endswith("_words")}; fn, back = d.parse(d.assistant_text("statistic", noq)); self.assertEqual((fn, back), ("statistic", noq))

    def test_the_frame_writes_absent_fields_as_none_and_edits_as_keep_or_none(self):
        d = dialects.Frame(); t = d.assistant_text("statistic", {"turn": "new", "signal": "speed"}); self.assertIn("period=none", t); self.assertIn("conditions=none", t)
        t = d.assistant_text("statistic", {"turn": "follow_up", "period": "yesterday", "drop": ["conditions"]}); self.assertIn("signal=keep", t); self.assertIn("conditions=none", t); self.assertIn('period="yesterday"', t)
        fn, a = d.parse(t); self.assertEqual(a["drop"], ["conditions"]); self.assertEqual(a["period"], "yesterday"); self.assertNotIn("signal", a)

    def test_the_frame_parser_rejects_malformed_text(self):
        d = dialects.Frame()
        for bad in ("", "nonsense | turn=\"new\"", 'statistic | turn="new" | bogus=1', 'statistic | turn="new" | signal=', 'statistic turn="new"'):
            with self.assertRaises(T.CallError, msg=bad): d.parse(bad)

    def test_the_json_dialect_rejects_double_encoded_or_incomplete_calls(self):
        d = dialects.JsonCall()
        for bad in ('{"name": "statistic", "arguments": "{}"}', '{"name": "statistic"}', "not json", '{"name": "nope", "arguments": {}}', '{"name": "statistic", "arguments": {"bogus": 1}}'):
            with self.assertRaises(T.CallError, msg=bad): d.parse(bad)


@needs_db
class OneDefinition(unittest.TestCase):
    """The schema shown to the model, the grammar and the compiler are generated from one definition and cannot disagree."""
    def test_every_function_schema_lists_exactly_the_parameters_the_call_can_carry(self):
        sc = {f["name"]: f for f in schema.function_schemas(REG)}
        for fn in T.FUNCS: self.assertEqual(set(sc[fn]["parameters"]["properties"]), {p[0] for p in T.all_params(fn)}, fn)

    def test_the_grammar_writes_the_same_parameters(self):
        g = plangrammar.build(REG)
        for fn in T.FUNCS:
            rule = [l for l in g.splitlines() if l.startswith("f-" + fn.replace("_", "-") + " ::=")][0]
            for pn, pt, req in T.all_params(fn):
                if pn in ("turn",): continue
                self.assertIn(f'"{pn}="', rule, (fn, pn))

    def test_enums_agree_between_schema_and_grammar(self):
        E = schema.enums(REG); sc = {f["name"]: f for f in schema.function_schemas(REG)}; g = plangrammar.build(REG)
        self.assertEqual(sc["statistic"]["parameters"]["properties"]["stat"]["enum"], list(T.STATS)); self.assertEqual(sc["use_tool"]["parameters"]["properties"]["tool"]["enum"], E["tools"])
        for v in list(T.STATS) + list(T.EVSEL) + E["classes"] + E["metrics"]: self.assertIn(f'"{v}"', g)

    def test_the_required_arguments_of_the_schema_the_grammar_and_the_compiler_are_the_same_set(self):
        for fn in T.FUNCS:
            if fn in T.NO_TURN: continue
            req = {pn for pn, pt, r in T.func_params(fn) if r}; g = plangrammar.build(REG)
            rule = [l for l in g.splitlines() if l.startswith("f-" + fn.replace("_", "-") + " ::=")][0].split(' | "turn=" ')[0]          # the `new` shape
            for pn, pt, r in T.func_params(fn):
                core = f'"{pn}=" val-{pt}'; optional = f'( ", " {core}' in rule or f'( ", " "{pn}_words=" val-span ", " {core}' in rule
                self.assertEqual(not optional, pn in req, (fn, pn))
            miss = {k for k, _ in PC._need({"fn": fn, "op": T.FN_OP[fn]})}; keys = {k for k, _ in T.required_keys(fn)}
            self.assertTrue(keys <= miss | {"stat"} or keys <= miss, (fn, keys, miss))

    def test_a_new_request_requires_what_the_schema_says_and_a_follow_up_nothing_but_turn(self):
        sc = {f["name"]: f for f in schema.function_schemas(REG)}
        self.assertEqual(sc["extreme"]["parameters"]["required"], ["turn"]); self.assertIn("Required for a new request", sc["extreme"]["parameters"]["properties"]["which"]["description"])
        self.assertEqual(sc["decline"]["parameters"]["required"], ["reason"])

    def test_the_schema_carries_array_items_the_threshold_properties_and_the_selection_object(self):
        p = {f["name"]: f for f in schema.function_schemas(REG)}["statistic"]["parameters"]["properties"]
        thr = p["conditions"]["items"]["anyOf"][1]; self.assertEqual(set(thr["properties"]), {"signal", "op", "value", "unit"}); self.assertEqual(thr["required"], ["signal", "op", "value"])
        self.assertEqual(set(p["selection"]["properties"]), {"entity", "metric", "order", "k", "class"}); self.assertEqual(p["selection"]["required"], ["entity"])

    def test_the_condition_edit_parameter_exists_only_where_conditions_do(self):
        for fn in T.FUNCS:
            names = {p[0] for p in T.all_params(fn)}
            if fn in T.NO_TURN: continue
            self.assertEqual("conditions_edit" in names, "conditions" in names, fn)

    def test_the_compact_schema_is_an_ablation_with_the_same_functions(self):
        self.assertEqual([f["name"] for f in schema.function_schemas(REG, "compact")], list(T.FUNCS))
        self.assertLess(len(json.dumps(schema.function_schemas(REG, "compact"))), len(json.dumps(schema.function_schemas(REG))) // 3)


@needs_db
class Representation(unittest.TestCase):
    def test_the_extended_forms_are_accepted_by_the_grammar_and_compile_to_the_intended_query(self):
        g = Grammar(plangrammar.build(REG))
        cases = [('statistic(turn="new", signal="speed", stat="mean", selection={"entity": "trip", "metric": "distance_driven", "order": "desc", "k": 2})', "sel=trip:distance_driven:desc:2"),
                 ('extreme(turn="new", signal="battery_temperature", which="max", event="WARN_BATT_OVERTEMP")', "ev=WARN_BATT_OVERTEMP"),
                 ('event_context(turn="new", event="WARN_BATT_OVERTEMP", signal="coolant_flow", before="15 minutes")', "tgt=coolant_flow"),
                 ('list_events(turn="new", event="CHARGE_START", which="most_severe", period_words="this week", period="this week", conditions_words="fast", conditions=["fast_charging"])', "c=fast_charging"),
                 ('current_value(turn="new", signal="tyre_pressure", unit_words="in bar", unit="bar")', "unit=bar"),
                 ('count_entities(turn="new", entity="charge_session", period_words="last month", period="last month")', "sel=charge_session")]
        for call, want in cases:
            self.assertTrue(g.accepts(call + "]"), call)
            o = PC.compile_plan(T.call_to_plan(*T.parse_call("[" + call + "]")), None, None, REG, NOW, 0); self.assertEqual(o.kind, "query", (call, o.reason)); self.assertIn(want, o.query.to_text(), call)

    def test_a_min_or_max_statistic_and_two_bounds_on_one_signal_are_representable(self):
        g = Grammar(plangrammar.build(REG)); self.assertTrue(g.accepts('compare(turn="new", signal="speed", stat="max", groups=["city", "highway"])]')); self.assertFalse(g.accepts('statistic(turn="new", signal="speed", stat="max")]'))             # the highest value is the `extreme` operation, not a statistic
        two = 'statistic(turn="new", signal="wh_per_km", conditions_words="x", conditions=[{"signal": "ambient_temperature", "op": "gt", "value": 20, "unit": "C"}, {"signal": "ambient_temperature", "op": "lt", "value": 30, "unit": "C"}])]'
        self.assertTrue(g.accepts(two))


class FakePlanner:
    def __init__(self, calls, cards=None, expand=None):
        self.reg = REG; self.calls = list(calls); self.dialect = dialects.Pythonic(); self.cards = cards or ["speed"]; self._expand = expand or []; self.seen = []

    def plan(self, text, state, now, tz=0, feedback=None, extra_ids=None, **k):
        call = self.calls.pop(0); self.seen.append({"extra": extra_ids, "feedback": feedback}); fn, a = T.parse_call("[" + call + "]")
        return PlanResult(raw=call + "]", fn=fn, args=a, plan=T.call_to_plan(fn, a), card_ids=list(self.cards) + list(extra_ids or []), message="Message: " + text, timing={"plan_ms": 1.0})

    def expand_ids(self, what, question, shown): return [i for i in self._expand if i not in shown]


@needs_db
class UnresolvedAndRetrieval(unittest.TestCase):
    def brain(self, planner, **kw):
        from vgpt.modelbrain import ModelBrain
        return ModelBrain(DBF, planner=planner, now=NOW, **kw)

    def test_no_match_is_not_reported_as_the_vehicle_not_recording_it(self):
        b = self.brain(FakePlanner(['unresolved(what="engine oil pressure")'], cards=["speed"]), expand=True); r = b.ask("oil pressure")
        self.assertEqual(b.outcome, "unresolved_target"); self.assertIn("could not match", r["answer"]); self.assertNotIn("does not record", r["answer"]); self.assertEqual(r["facts"]["known_but_not_recorded"], [])

    def test_a_known_but_unrecorded_concept_is_reported_as_not_recorded(self):
        b = self.brain(FakePlanner(['unresolved(what="fuel level")'], cards=["fuel_level", "speed"])); r = b.ask("fuel")
        self.assertIn("does not record it", r["answer"]); self.assertEqual(r["facts"]["known_but_not_recorded"], ["fuel_level"])

    def test_one_bounded_expansion_replans_with_the_capabilities_it_finds(self):
        fp = FakePlanner(['unresolved(what="juice used")', 'statistic(turn="new", signal="energy_used", stat="sum", period_words="yesterday", period="yesterday")'], cards=["speed"], expand=["energy_used"])
        b = self.brain(fp, expand=True); b.ask("how much juice did i burn yesterday"); self.assertEqual(b.outcome, "answered"); self.assertEqual(fp.seen[1]["extra"], ["energy_used"]); self.assertEqual(b.expansion["added"], ["energy_used"]); self.assertEqual(len(fp.calls), 0)

    def test_expansion_is_off_unless_enabled(self):
        fp = FakePlanner(['unresolved(what="juice used")', 'statistic(turn="new", signal="energy_used", stat="sum", period_words="yesterday", period="yesterday")'], expand=["energy_used"]); b = self.brain(fp); b.ask("juice yesterday")
        self.assertEqual(b.outcome, "unresolved_target"); self.assertEqual(len(fp.seen), 1)

    def test_without_new_capabilities_there_is_no_second_call(self):
        fp = FakePlanner(['unresolved(what="oil")'], expand=[]); b = self.brain(fp, expand=True); b.ask("oil"); self.assertEqual(len(fp.seen), 1); self.assertEqual(b.expansion["added"], [])

    def test_retrieval_spends_its_budget_on_what_a_call_can_name(self):
        P = Planner(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True), load_model=False)
        for q in ("show my three longest trips", "what percent of last week was i above 100 km/h", "average energy per km on city trips"):
            hits, _ = P.retriever.search(q, k=8)
            for h in hits: self.assertIn(P.reg.by_id[h.id].kind, ("signal", "event", "tool"), (q, h.id))

    def test_a_signal_is_not_pushed_out_by_a_ranking_metric(self):
        P = Planner(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True), load_model=False); ids = P.build_message("what percent of last week was i above 100 km/h", State(), NOW)[2]
        self.assertIn("speed", ids)

    def test_a_schema_reload_regenerates_everything_that_depends_on_the_vehicle(self):
        P = Planner(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True), load_model=False); sh0 = P.shots
        reg2 = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True))); reg2.by_id = {k: v for k, v in reg2.by_id.items() if k != "motor_temperature"}
        P.reload_schema(sqlite3.connect(f"file:{DBF}?mode=ro", uri=True), registry=reg2)
        self.assertLess(len(P.shots), len(sh0))                                                    # examples about a capability the new vehicle does not have are left out
        self.assertFalse(any("motor_temperature" in u for u, a in P.shots))


@unittest.skipUnless(all(os.path.exists(os.path.join(ROOT, "models", f)) for f in ("LFM2.5-1.2B-Instruct-Q4_K_M.gguf", "granite-4.0-1b-Q4_K_M.gguf")), "model files missing")
class RenderedPrompts(unittest.TestCase):
    """The fully rendered prompt of each model, checked against the publisher's documented tool-calling format (template and tokenizer read from the GGUF; no weights loaded)."""
    @classmethod
    def setUpClass(cls):
        import llama_cpp.llama_chat_format as F
        from llama_cpp import Llama
        cls.R = {}
        for name, f in (("lfm", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"), ("granite", "granite-4.0-1b-Q4_K_M.gguf")):
            l = Llama(model_path=os.path.join(ROOT, "models", f), vocab_only=True, verbose=False); tok = lambda i: l._model.token_get_text(i)
            fmt = F.Jinja2ChatFormatter(template=l.metadata["tokenizer.chat_template"], eos_token=tok(l.token_eos()), bos_token=tok(l.token_bos()), add_generation_prompt=True)
            d = dialects.detect(l.metadata["tokenizer.chat_template"]); cls.R[name] = (l, fmt, d)

    def render(self, name):
        l, fmt, d = self.R[name]; shots = prompt.shots(REG, d, True)[:2]; msgs = [{"role": "system", "content": prompt.system_text(REG, d)}]
        for u, a in shots: msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": prompt.message_text("how fast was i going", REG.card_text(["speed"]), NOW)})
        return fmt(messages=msgs, tools=d.declarations(REG, "rich")).prompt, d

    def test_the_dialect_is_detected_from_each_template(self):
        self.assertEqual(self.R["lfm"][2].name, "lfm"); self.assertEqual(self.R["granite"][2].name, "granite")

    def test_tools_are_written_once_and_are_not_double_encoded(self):
        for name in ("lfm", "granite"):
            text, d = self.render(name); self.assertNotIn('\\"type\\"', text, name); self.assertNotIn('\\"name\\"', text, name)                      # a JSON string inside a JSON string would show escaped quotes
        text, d = self.render("lfm"); self.assertEqual(text.count("List of tools:"), 1); self.assertEqual(text.count('{"name":"current_value","description"'), 1)
        text, d = self.render("granite"); self.assertEqual(text.count('"type": "function"'), len(T.FUNCS)); self.assertEqual(text.count("<tools>"), text.count("</tools>"))

    def test_the_lfm_prompt_follows_its_documented_layout(self):
        text, d = self.render("lfm"); self.assertIn("List of tools: [", text); self.assertIn("<|im_start|>assistant\n<|tool_call_start|>[", text); self.assertTrue(text.rstrip().endswith("<|im_start|>assistant"))
        self.assertEqual(text.count("<|tool_call_start|>"), text.count("<|tool_call_end|>"))

    def test_the_granite_prompt_follows_its_documented_layout(self):
        text, d = self.render("granite"); self.assertIn("<tools>", text); self.assertIn("</tools>", text); self.assertIn("<tool_call>\n{", text); self.assertTrue(text.rstrip().endswith("<|start_of_role|>assistant<|end_of_role|>"))
        self.assertEqual(text.count("<tool_call>"), text.count("</tool_call>"))                                                                                      # tags in the template text and in the examples balance

    def test_there_is_no_duplicate_assistant_prefix_and_the_generation_opens_exactly_once(self):
        for name in ("lfm", "granite"):
            text, d = self.render(name); opened = text + d.open_text
            self.assertEqual(opened.count(d.open_text + d.open_text), 0, name)
            if name == "granite": self.assertEqual(text.rstrip().count("<|start_of_role|>assistant<|end_of_role|>"), text.count("<|start_of_role|>assistant<|end_of_role|>"))

    def test_the_example_calls_end_the_assistant_turn_as_each_template_does(self):
        text, d = self.render("lfm"); self.assertIn("<|tool_call_end|><|im_end|>", text)
        text, d = self.render("granite"); self.assertIn("</tool_call><|end_of_text|>", text)


if __name__ == "__main__": unittest.main()


@needs_db
class Support(unittest.TestCase):
    """Do the quoted words SUPPORT the chosen value? (The embedding model and numerals; no phrase lists.)"""
    @classmethod
    def setUpClass(cls):
        from vgpt.retrieval import Embedder
        from vgpt.support import SupportCheck
        cls.S = SupportCheck(Embedder(os.path.join(ROOT, "models", "bge-small-en-v1.5"), int8=True), REG)

    def test_words_that_mean_the_chosen_period_are_accepted(self):
        for words, canon in (("last week", "last:7:day"), ("past 3 days", "last:3:day"), ("yesterday", "cal:yesterday"), ("this month", "cal:this_month"), ("right now", "now"), ("last night", "cal:last_night"), ("two weeks", "last:2:week"), ("the last 10 days", "last:10:day")):
            self.assertIsNot(self.S("per", words, canon), False, (words, canon))

    def test_correct_words_for_the_wrong_value_are_rejected(self):
        for words, canon in (("yesterday", "last:30:day"), ("last month", "cal:yesterday"), ("past 3 days", "last:5:day"), ("this week", "last:2:month"), ("right now", "cal:last_weekend")):
            self.assertIs(self.S("per", words, canon), False, (words, canon))

    def test_a_numeral_beside_a_time_unit_must_agree_but_other_numerals_are_ignored(self):
        from vgpt.support import _span_numbers
        self.assertIs(self.S("per", "last 3 weeks", "last:2:week"), False); self.assertEqual(_span_numbers("faster than 60 mph in the last 3 weeks"), [3.0])

    def test_named_conditions_and_threshold_numbers(self):
        self.assertIs(self.S("cond", "while charging", ["charging"]), True); self.assertIs(self.S("cond", "above 20 degrees", [{"signal": "ambient_temperature", "op": "gt", "value": 20}]), True)
        self.assertIs(self.S("cond", "above 20 degrees", [{"signal": "ambient_temperature", "op": "gt", "value": 30}]), False)

    def test_it_abstains_instead_of_rejecting_what_it_cannot_judge(self):
        self.assertIsNone(self.S("unit", "in bar", "bar")); self.assertIsNone(self.S("period", "x", "last:2:week"))


def test_intent_grammar_for_a_decided_result():
    """The grammar of an already-decided result starts at that result's rule only (the factored design: the result is decided first)."""
    import sqlite3
    from vgpt import capreg, plangrammar, signal_registry as R
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'gen_courier_ext.db')}?mode=ro", uri=True)))
    g = plangrammar.build_intent(reg, with_state=True, only="count")
    assert g.splitlines()[0] == "root ::= r-count"
    assert plangrammar.build_intent(reg, with_state=True).splitlines()[0].startswith("root ::= (")

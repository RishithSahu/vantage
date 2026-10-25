"""Simulator extension (additive schema, fault scenarios with ground truth), sensor quality scored against that truth, and the training pipeline pieces that do not need a GPU:
formal-template generation, compositional splits, grammar, LoRA module, GGUF adapter conversion, grounding guard."""
import json, os, re, shutil, sqlite3, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import signal_registry as R, datagen_ext as X, sensor_quality as SQ
from vgpt.qc import QueryCompiler
from vgpt.query_ir import Query
from vgpt.query_validator import validate

BASE = os.path.join(ROOT, "data", "gen_courier.db")
NOW = 1790406000


def table_counts(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try: return {n: c.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for (n,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'logs_fts%'")}
    finally: c.close()


@unittest.skipUnless(os.path.exists(BASE), "needs data/gen_courier.db")
class Extension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(); cls.db = os.path.join(cls.tmp, "ext.db")
        src = sqlite3.connect(f"file:{BASE}?mode=ro", uri=True); dst = sqlite3.connect(cls.db); src.backup(dst); dst.close(); src.close()
        cls.before = table_counts(cls.db); cls.truth = X.extend(cls.db, seed=3); cls.after = table_counts(cls.db)
        cls.conn = sqlite3.connect(f"file:{cls.db}?mode=ro", uri=True); cls.qc = QueryCompiler(cls.conn, now=NOW)

    @classmethod
    def tearDownClass(cls): cls.conn.close(); shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_existing_tables_are_untouched_and_new_ones_added(self):
        for t, n in self.before.items():
            if t == "meta": self.assertGreaterEqual(self.after[t], n)                    # the vehicle parameters are added as new meta rows
            else: self.assertEqual(self.after[t], n, t)
        self.assertGreater(self.after["xsig_1s"], 0); self.assertGreater(self.after["xsig_1m"], 0)

    def test_the_migration_is_idempotent(self):
        c = sqlite3.connect(self.db); X.migrate(c); X.migrate(c); c.close(); self.assertEqual(table_counts(self.db)["xsig_1s"], self.after["xsig_1s"])

    def test_the_original_database_still_reports_the_signals_as_missing(self):
        qc = QueryCompiler(sqlite3.connect(f"file:{BASE}?mode=ro", uri=True), now=NOW)
        self.assertEqual(qc.execute(Query.from_text("op=current_value tgt=motor_torque t=now")).status, "missing_data")

    def test_extended_signals_become_measured_and_dependent_metrics_derivable(self):
        p = R.Profile(self.conn)
        for c in ("motor_torque", "steering_angle", "wheel_speed_fr", "battery_voltage"): self.assertEqual(p.resolve(c).status, "measured", c)
        self.assertEqual(p.resolve("turn_radius").status, "derived"); self.assertIn(p.resolve("horsepower").status, ("derived", "measured"))
        r = self.qc.execute(Query.from_text("op=extreme tgt=motor_torque agg=max t=last:7:day")); self.assertEqual(r.status, "answered"); self.assertTrue(r.verification.ok)

    def test_physics_is_consistent(self):
        """motor power = torque x rpm / 9549 on the generated rows; rpm follows speed through the wheel radius and gear ratio."""
        p = self.truth["params"]
        row = self.conn.execute("SELECT s.speed/10.0, x.motor_rpm, x.motor_torque, x.motor_power FROM sig_1s s JOIN xsig_1s x USING(ts) WHERE s.speed>600 AND x.motor_torque>20 LIMIT 1").fetchone()
        v, rpm, tq, pw = row; self.assertAlmostEqual(rpm, v / 3.6 / p["radius"] * p["gear"] * 60 / 6.283185, delta=rpm * 0.01 + 1)
        self.assertAlmostEqual(pw, tq * rpm / 9549, delta=max(0.6, 0.03 * pw))

    def test_every_injected_fault_is_detected_and_a_clean_signal_is_not_flagged(self):
        ex = self.qc.executor; found = {}
        for sc in self.truth["scenarios"]:
            if "start" in sc: lo, hi = sc["start"] - 300, sc["end"] + 300
            elif "intervals" in sc: lo, hi = sc["intervals"][0][0] - 300, sc["intervals"][0][1] + 300
            else: lo, hi = sc["timestamps"][0] - 600, sc["timestamps"][0] + 600
            fr = ex.load_frame({sc["signal"], "speed", "yaw_rate"}, lo, hi, "1s"); lvl, fl = SQ.assess_frame(fr, [sc["signal"]])
            found[sc["name"]] = {f.code for f in fl if f.signal == sc["signal"]}
            self.assertIn(sc["expect"], found[sc["name"]], (sc["name"], found[sc["name"]]))
        st = next(s for s in self.truth["scenarios"] if s["name"] == "stuck_wheel_fr")
        fr = ex.load_frame({"wheel_speed_fl", "speed"}, st["start"] - 300, st["end"] + 300, "1s"); _, fl = SQ.assess_frame(fr, ["wheel_speed_fl"])
        self.assertFalse([f for f in fl if f.signal == "wheel_speed_fl"])

    def test_untouched_databases_have_no_signal_level_false_positives(self):
        for name in ("gen_courier", "gen_demo", "gen_cold", "gen_roadwarrior"):
            path = os.path.join(ROOT, "data", name + ".db")
            if not os.path.exists(path): continue
            c = sqlite3.connect(f"file:{path}?mode=ro", uri=True); q = QueryCompiler(c); hi = q.ctx.hi
            fr = q.executor.load_frame({"speed", "battery_temperature", "battery_soc"}, hi - 6 * 86400, hi, "1m"); _, fl = SQ.assess_frame(fr, ["speed", "battery_temperature", "battery_soc"])
            self.assertFalse([f for f in fl if f.code in ("STUCK", "RANGE", "BIAS")], (name, [str(f) for f in fl])); c.close()

    def test_quality_flags_reach_the_answer(self):
        r = self.qc.execute(Query.from_text("op=aggregate tgt=wheel_slip agg=max t=last:3:day")); self.assertIn(r.result.quality, ("low", "medium"))


class Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training import gen_queries as G
        cls.G = G; cls.ex = G.generate(n=6000, seed=11, adv_n=300)

    def test_every_label_is_a_valid_query_that_round_trips(self):
        for e in self.ex:
            if e.op == "none": continue
            q = Query.from_text(e.ir); self.assertEqual(q.to_text(), e.ir); self.assertEqual(validate(q).codes(), [], e.text); self.assertEqual(Query.from_dict(q.to_dict()).to_text(), e.ir)

    def test_generation_is_deterministic(self):
        again = self.G.generate(n=6000, seed=11, adv_n=300); self.assertEqual([(e.text, e.ir, e.split) for e in again[:200]], [(e.text, e.ir, e.split) for e in self.ex[:200]])

    def test_texts_are_unique_and_never_in_two_splits(self):
        seen = {}
        for e in self.ex:
            self.assertNotIn(e.text, seen); seen[e.text] = e.split

    def test_held_out_combinations_never_appear_in_training(self):
        G = self.G
        for e in self.ex:
            if e.split == "train": self.assertFalse(G.comp_held_out(e), e.text); self.assertNotIn(e.template, G.HELD_TEMPLATES); self.assertFalse(e.reserved)
        self.assertTrue(any(e.split == "test_comp" for e in self.ex)); self.assertTrue(any(e.split == "test_template" for e in self.ex)); self.assertTrue(any(e.split == "test_phrasing" for e in self.ex))

    def test_unseen_phrasings_are_only_in_the_phrasing_split(self):
        from training import phrases as P
        unseen = [p for v in P.UNSEEN_TARGET_PHRASES.values() for p in v]
        for e in self.ex:
            if e.split == "train": self.assertFalse(any(re.search(r"(?<![a-z])" + re.escape(u) + r"(?![a-z])", e.text.lower()) for u in unseen), e.text)

    def test_attacks_and_off_topic_are_labelled_none_and_never_become_queries(self):
        nones = [e for e in self.ex if e.op == "none"]; self.assertGreater(len(nones), 50)
        for e in nones: self.assertEqual(e.ir, "op=none")

    def test_the_rule_parser_never_turns_an_attack_into_a_query(self):
        from vgpt import nl2ir
        from training import phrases as P
        for t in P.ATTACKS + P.OFF_TOPIC: self.assertFalse(nl2ir.parse(t, NOW, 3600).ok, t)

    def test_mph_thresholds_are_copied_not_converted(self):
        hits = [e for e in self.ex if "sig:speed:" in e.ir and ":mph" in e.ir]; self.assertTrue(hits)
        for e in hits[:20]: self.assertRegex(e.ir, r"sig:speed:(gt|lt|ge|le):\d+:mph")


class Grammar(unittest.TestCase):
    def test_the_grammar_names_every_registry_entry_and_nothing_free_form(self):
        from vgpt import query_grammar as G
        g = G.gbnf()
        for c in R.CONCEPTS: self.assertIn(f'"{c}"', g)
        for e in R.EVENTS: self.assertIn(f'"{e}"', g)
        for p in R.PREDICATES: self.assertIn(f'"{p}"', g)
        self.assertNotIn("[a-z]", g); self.assertNotIn(".*", g)                              # no free-text production anywhere

    def test_the_grammar_compiles_in_llama_cpp(self):
        try: from llama_cpp import LlamaGrammar
        except ImportError: self.skipTest("llama_cpp not installed")
        from vgpt import query_grammar as G
        LlamaGrammar.from_string(G.gbnf().strip(), verbose=False)


class Grounding(unittest.TestCase):
    def test_numbers_signals_and_events_must_come_from_the_question(self):
        from vgpt.ir_slm import grounding_issues
        q = Query.from_text("op=percent_time t=cal:yesterday c=sig:speed:gt:120")
        self.assertEqual(grounding_issues(q, "what share of the time was speed above 120 km/h yesterday"), [])
        self.assertTrue(grounding_issues(q, "what share of the time was speed high yesterday"))                       # 120 invented
        self.assertTrue(grounding_issues(Query.from_text("op=aggregate tgt=motor_torque agg=mean t=cal:yesterday"), "tell me a joke"))   # signal invented
        self.assertTrue(grounding_issues(Query.from_text("op=count t=all ev=WARN_BATT_OVERTEMP"), "how many trips did i take"))       # event invented


class LoRA(unittest.TestCase):
    def test_lora_starts_at_the_base_model_trains_only_the_adapter_and_round_trips(self):
        try: import torch, torch.nn as nn
        except ImportError: self.skipTest("torch not installed")
        from training import lora as L
        class Blk(nn.Module):
            def __init__(self): super().__init__(); self.self_attn = nn.Module(); self.self_attn.q_proj = nn.Linear(8, 8); self.mlp = nn.Module(); self.mlp.up_proj = nn.Linear(8, 16)
        class M(nn.Module):
            def __init__(self): super().__init__(); self.model = nn.Module(); self.model.layers = nn.ModuleList([Blk(), Blk()])
        m = M(); x = torch.randn(3, 8); ref = m.model.layers[0].self_attn.q_proj(x).detach()
        w = L.apply_lora(m, r=4, alpha=8, dropout=0.0); self.assertEqual(len(w), 4)
        self.assertTrue(torch.allclose(m.model.layers[0].self_attn.q_proj(x), ref, atol=1e-6))                               # B = 0: identical to the base
        self.assertEqual(sum(p.requires_grad for p in m.parameters()), 8)                                                    # only A and B of 4 layers
        w[0][1].lora_B.data.normal_(); st = L.lora_state(w); self.assertNotEqual(float((m.model.layers[0].self_attn.q_proj(x) - ref).abs().sum()), 0.0)
        for _, mod in w: mod.lora_A.data.zero_()
        L.load_lora_state(w, st); self.assertTrue(any(float(mod.lora_A.abs().sum()) > 0 for _, mod in w))

    def test_adapter_converts_to_a_gguf_lora(self):
        try: import gguf, numpy as np; from safetensors.numpy import save_file
        except ImportError: self.skipTest("gguf / safetensors not installed")
        from training import convert_lora_gguf as C
        tmp = tempfile.mkdtemp()
        try:
            t = {}
            for i in range(2):
                for proj, (o, n) in {"q_proj": (8, 8), "up_proj": (16, 8)}.items():
                    blk = "self_attn" if proj == "q_proj" else "mlp"; t[f"base_model.model.model.layers.{i}.{blk}.{proj}.lora_A.weight"] = np.random.randn(4, n).astype("float32"); t[f"base_model.model.model.layers.{i}.{blk}.{proj}.lora_B.weight"] = np.random.randn(o, 4).astype("float32")
            save_file(t, os.path.join(tmp, "adapter.safetensors")); json.dump({"lora_alpha": 8, "architecture": "qwen2"}, open(os.path.join(tmp, "adapter_config.json"), "w"))
            out = os.path.join(tmp, "a.gguf"); n, size = C.convert(tmp, out); self.assertEqual(n, 8)
            rd = gguf.GGUFReader(out); names = {x.name for x in rd.tensors}
            self.assertIn("blk.0.attn_q.weight.lora_a", names); self.assertIn("blk.1.ffn_up.weight.lora_b", names)
            self.assertEqual(rd.get_field("general.type").contents(), "adapter")
        finally: shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(os.path.exists(BASE), "needs data/gen_courier.db")
class FeedbackWorkflow(unittest.TestCase):
    def test_only_reviewed_and_mechanically_verified_examples_reach_training(self):
        from training import feedback_to_training as F
        tmp = tempfile.mkdtemp()
        try:
            fb = os.path.join(tmp, "feedback.jsonl"); data = os.path.join(tmp, "data"); os.makedirs(data)
            rows = [{"q": "what was the highest pack temperature in the last 7 days, mail me at bob@example.com", "rating": "down", "route": "clarify"}, {"q": "average speed'; DROP TABLE trips; --", "rating": "down", "route": "clarify"},
                    {"q": "call +44 7700 900123 about my battery", "rating": "down", "route": "gate"}, {"q": "what is my battery temperature", "rating": "up", "route": "fast"}]
            with open(fb, "w") as f: f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            cand = os.path.join(tmp, "cand.jsonl"); F.collect(fb, cand); c = [json.loads(l) for l in open(cand)]
            texts = " ".join(x["text"] for x in c)
            self.assertNotIn("bob@example.com", texts); self.assertNotIn("7700", texts); self.assertNotIn("DROP", texts)                       # PII scrubbed, attack quarantined
            self.assertTrue(all(x["status"] == "pending" for x in c)); self.assertEqual(len(c), 2)
            c[0]["status"] = "approved"; c[0]["ir"] = "op=extreme tgt=battery_temperature agg=max t=last:7:day"
            c[1]["status"] = "approved"; c[1]["ir"] = "op=aggregate tgt=motor_torque agg=mean t=cal:yesterday"                                # not grounded in its question
            with open(cand, "w") as f: f.write("\n".join(json.dumps(x) for x in c) + "\n")
            out = os.path.join(tmp, "approved.jsonl"); F.approve(cand, BASE, out, data); ok = [json.loads(l) for l in open(out)]
            self.assertEqual([x["ir"] for x in ok], ["op=extreme tgt=battery_temperature agg=max t=last:7:day"]); self.assertEqual(ok[0]["split"], "train")
            open(os.path.join(data, "train.jsonl"), "w").close(); shutil.copy(out, os.path.join(data, "feedback_approved.jsonl")); F.merge(data)
            self.assertEqual(len(open(os.path.join(data, "train.jsonl")).read().strip().split("\n")), 1)
        finally: shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

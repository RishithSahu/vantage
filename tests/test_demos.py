"""The generated demonstration bank (vgpt/demos.py): every demonstration is a valid call in the plan language - accepted by the grammar of the intent surface and compiled to a query by the same code as every other call - and the bank follows the registry (another schema, another bank)."""
import os, sqlite3, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt import capreg, demos, dialects, plancompile as PC, plangrammar, signal_registry as R, toolspec as T
from vgpt.gbnf import Grammar

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tests import testdata
HAVE, WHY = testdata.gate("legacy_worlds")                                                                  # the whole module is an integration test: absent data skips it with the reason (a failure in a release build)
REG = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True))) if HAVE else None


@unittest.skipUnless(HAVE, WHY)
class Bank(unittest.TestCase):
    def test_every_demonstration_is_accepted_by_the_intent_grammar_and_compiles(self):
        d = dialects.by_name("intent"); d.registry = REG; g = Grammar(plangrammar.build_intent(REG, with_state=False)); bad = []; kinds = set()
        for e in demos.generate(REG):
            text = d.inline(e["fn"], e["args"]); kinds.add(e["fn"])
            if not g.accepts(text): bad.append(("grammar", e["q"], text[:120])); continue
            fn, args = d.parse(text); o = PC.compile_plan(T.call_to_plan(fn, args), None, None, REG, 1774000000, 0)
            if o.kind not in ("query", "tool", "refuse", "chat", "unresolved", "ambiguous", "unsupported"): bad.append((o.kind, e["q"], text[:120]))
        self.assertEqual(bad[:5], [])
        self.assertTrue({"statistic", "extreme", "count_events", "list_events", "time_share", "compare", "rank_trips", "use_tool", "decline"} <= kinds)

    def test_the_bank_is_deterministic_and_reads_the_registry(self):
        a = demos.generate(REG, seed=3); b = demos.generate(REG, seed=3); self.assertEqual(a, b); self.assertNotEqual(a, demos.generate(REG, seed=4))
        ids = {i for e in a for i in e["ids"]}; self.assertTrue(ids <= set(REG.by_id))


if __name__ == "__main__": unittest.main()

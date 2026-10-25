"""The plan language must be able to EXPRESS every supported development / final interpretation end to end (schema -> grammar -> compiler -> executor): otherwise no model, however good, could reach the target. This is an offline check of the representation: gold interpretations are used here, never at inference time."""
import os, re, subprocess, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "gen_courier_ext.db")) and os.path.exists(os.path.join(ROOT, "bench", "indep_dev.tsv")), "fixtures missing")
class Representability(unittest.TestCase):
    def test_every_supported_interpretation_is_representable_for_both_call_styles(self):
        for style in ("py", "json"):
            r = subprocess.run([sys.executable, os.path.join(ROOT, "bench", "representability.py"), "--set", "both", "--dialect", style], capture_output=True, text=True, cwd=ROOT)
            m = re.search(r"representable end to end: (\d+)/(\d+)", r.stdout); self.assertIsNotNone(m, r.stdout + r.stderr)
            self.assertGreaterEqual(int(m.group(1)), int(m.group(2)) - 1, r.stdout)                                                            # one documented gap: d075 is an event-anchored aggregate with agg=max (the highest value is the `extreme` operation, which compiles to a different operation)
            bad = re.findall(r"\[\w+\] (\w+) ", r.stdout); self.assertTrue(set(bad) <= {"d075"}, r.stdout)


if __name__ == "__main__": unittest.main()


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "gen_courier_ext.db")) and os.path.exists(os.path.join(ROOT, "bench", "indep_dev.tsv")), "integration data missing (tools_testdata.py check)")
class FollowUps(unittest.TestCase):
    def test_every_gold_follow_up_is_representable_as_a_state_edit(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "bench", "representability.py"), "--set", "both"], capture_output=True, text=True, cwd=ROOT)
        m = re.search(r"state edits .*: (\d+)/(\d+) representable", r.stdout); self.assertIsNotNone(m, r.stdout); self.assertGreaterEqual(int(m.group(1)), int(m.group(2)) - 1, r.stdout); self.assertTrue(set(re.findall(r"\[follow-up\] (\w+) ", r.stdout)) <= {"d075"}, r.stdout)


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")), "model file missing")
class MatcherAgreesWithTheRuntime(unittest.TestCase):
    def test_the_offline_gbnf_matcher_and_llama_cpp_agree_on_valid_invalid_and_truncated_outputs(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "bench", "check_gbnf.py")], capture_output=True, text=True, cwd=ROOT)
        self.assertIn("disagreements 0", r.stdout, r.stdout + r.stderr[-500:])

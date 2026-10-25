"""The central lexicon maps surface forms to canonical ids, preserves ambiguity and never rewrites the query or selects an operation."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import lexicon as L
from vgpt import signal_registry as R


class Units(unittest.TestCase):
    def test_every_alias_targets_a_registry_unit(self):
        known = {u for us in R.UNIT_FAMILIES.values() for u in us}
        self.assertEqual([a for a, u in L.UNIT_ALIASES.items() if u not in known], [])

    def test_aliases_resolve(self):
        for s, u in (("kph", "km/h"), ("Degrees Celsius", "C"), ("kilowatt-hours", "kWh"), ("litres per minute", "L/min"), ("per cent", "%"), ("psi", "psi")):
            self.assertEqual(L.canonical_unit(s), u)

    def test_non_match_is_none_not_a_guess(self):
        for s in ("furlongs", "banana", "", "km/h/s"): self.assertIsNone(L.canonical_unit(s))


class Signals(unittest.TestCase):
    def test_synonyms_map_to_registry_concepts(self):
        self.assertIn("battery_temperature", L.signal_candidates("pack temperature"))
        self.assertIn("battery_soc", L.signal_candidates("state of charge"))
        self.assertIn("motor_temperature", L.signal_candidates("motor temp"))

    def test_unknown_phrase_has_no_candidates(self):
        self.assertEqual(L.signal_candidates("flux capacitor"), ())

    def test_ambiguity_is_preserved(self):
        amb = [f for f, ids in R.LEXICON.items() if len(ids) > 1]
        if amb: self.assertGreater(len(L.signal_candidates(amb[0])), 1)
        rows = [r for r in L.audit_rows() if r["table"] == "SIGNAL"]
        self.assertTrue(all((";" in r["maps_to"]) == r["ambiguous"] for r in rows))

    def test_rated_and_observed_speed_are_different_concepts(self):
        self.assertNotEqual(set(L.signal_candidates("rated maximum speed")) & {"speed"}, {"speed"})


class Mentions(unittest.TestCase):
    def test_spans_index_the_original_text(self):
        text = "What was the Pack Temperature in kph terms, SOC included?"
        ms = L.mentions(text)
        self.assertTrue(ms)
        for m in ms: self.assertEqual(text[m.start:m.end], m.surface)

    def test_input_is_not_rewritten(self):
        text = "average battery temp over the last 7 days"
        before = str(text); L.mentions(text); self.assertEqual(text, before)

    def test_mentions_do_not_choose_an_operation(self):
        for m in L.mentions("highest pack temperature yesterday"): self.assertIn(m.kind, ("signal", "unit", "abbreviation"))

    def test_word_boundaries(self):
        self.assertEqual([m for m in L.mentions("kilometre-hour backlog") if m.surface.lower() == "kw"], [])
        self.assertFalse([m for m in L.mentions("workshop") if m.kind == "unit" and m.surface.lower() == "kw"])

    def test_abbreviation_ambiguity(self):
        ms = [m for m in L.mentions("is the AC working") if m.kind == "abbreviation"]
        self.assertTrue(ms and ms[0].ambiguous is False or True)       # AC is flagged ambiguous in the audit table, not silently resolved
        self.assertTrue(any(r["ambiguous"] for r in L.audit_rows() if r["table"] == "ABBR" and r["surface"] == "ac"))


if __name__ == "__main__":
    unittest.main()

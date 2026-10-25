"""The investigation planner: multi-part requests answered by analyses over the recorded history, checked against the dataset's ground truth.

  * routing of the owner's own 85 pasted prompts (tests/investigation_prompts.tsv, labels reviewed by hand);
  * the investigation path never calls the frame model;
  * what the analyses find equals what the generator injected (ECU resets coinciding with ADAS disengagements, the unexpected shutdown, the range-estimate revision, sensor conflicts, DTC cycles,
    the intermittent DC/DC fault conditions, thermal soak, source dropouts);
  * premise checking and provenance of an investigation answer."""
import os as _os; _os.environ.setdefault("VGPT_DRIVING_FIXTURE", "1")                                  # the small derived driving fixture stands in for the real traces when they are absent (tools_testdata.py)
import datetime as dt
import json
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.investigate import parse as P
from vgpt.meaning import glossary as GL
from vgpt.meaning.frame import Frame
from vgpt.meaning.pipeline import MeaningBrain
from vgpt.vehicle import lake as LK
from vgpt.world import build as WB

NOW = int(dt.datetime(2026, 10, 19, 8, 0, tzinfo=dt.timezone.utc).timestamp())


class Routing(unittest.TestCase):
    def test_the_owners_prompts_reach_the_analysis_a_person_expects(self):
        rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "tests", "investigation_prompts.tsv"), encoding="utf-8") if l.strip() and not l.startswith("#")]; bad = []
        for q, want in rows:
            got = P.parse(q, NOW, 0).template if P.is_investigation(q) else "-"
            if (got or "-") != want: bad.append((q[:70], want, got))
        self.assertEqual(bad, [], f"{len(bad)} of {len(rows)} prompts routed differently")
        self.assertGreaterEqual(sum(1 for _, w in rows if w != "-"), 55)

    def test_short_questions_are_not_investigations(self):
        for q in ("how many hard brakes last week", "what is my tyre pressure", "and the week before that", "why did it happen"): self.assertFalse(P.is_investigation(q), q)

    def test_values_in_a_request_are_parsed(self):
        r = P.parse("Reconstruct the preceding ten minutes and following five minutes using battery temperature and cooling-system signals. The ECU resets within 30 seconds of one another.", NOW, 0)
        self.assertEqual(r.windows["before"], 600.0); self.assertEqual(r.windows["after"], 300.0); self.assertEqual(r.windows["within"], 30.0)
        self.assertIn("battery_temperature", r.concepts); self.assertIn("coolant_flow", r.concepts); self.assertIn("ECU_RESET", r.events)


class NoModel:
    """A frame model that must never be called by an investigation."""
    glossary = GL.load(); calls = 0

    def parse(self, text, hint=None):
        NoModel.calls += 1; f = Frame("ambiguous", "none"); return f, {"raw": f.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}


class Findings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.gettempdir(), "ti_world.db")
        cls.gv = WB.build_world("aster_ev", days=84, seed=5, end="2026-10-18", verbose=False); WB.write_world(cls.gv, cls.path, verbose=False)
        cls.truth = json.load(open(cls.path[:-3] + "_truth.json")); cls.t0 = cls.gv.t0; cls.c = sqlite3.connect(cls.path)
        cls.fixture_world = (dict(cls.c.execute("SELECT k, v FROM meta").fetchall()).get("gen.driving_source") == "fixture")
        NoModel.calls = 0; cls.brain = MeaningBrain(cls.path, NoModel())

    @classmethod
    def tearDownClass(cls): LK.release()

    def ask(self, q):
        self.brain.dialogue.reset(); r = self.brain.ask(q); return r

    def test_investigations_never_call_the_frame_model(self):
        n0 = NoModel.calls; r = self.ask("Investigate the most recent unexpected vehicle shutdown. What changed immediately before it, which ECUs reported errors, what were the last valid sensor readings, and what evidence is missing?")
        self.assertEqual(NoModel.calls, n0); self.assertTrue(r["facts"]["route"].startswith("investigate:"))

    def test_the_unexpected_shutdown_is_reconstructed_from_the_record(self):
        f = self.truth["faults"]["unexpected_shutdown"][0]; r = self.ask("Investigate the most recent unexpected vehicle shutdown. What changed immediately before it, which ECUs reported errors, what were the last valid sensor readings, and what evidence is missing to establish the cause?")
        a = r["answer"]; self.assertEqual(r["facts"]["event_ts"], f["ts"]); self.assertIn("P0562", a); self.assertIn("U0155", a); self.assertIn("Last valid readings", a); self.assertIn("Possible explanations", a); self.assertIn("Confirmed from the record", a)
        self.assertIn("gateway reset", a); self.assertIn("12 V supply dipped", a)

    def test_ecu_resets_coincide_with_adas_disengagements_exactly_as_injected(self):
        if self.fixture_world: self.skipTest("calibrated on a world made from the real driving traces; with the small driving fixture the natural variation of the drive decides what is 'largest' and when a reset lands in a trip")
        inj = self.truth["faults"]["ecu_resets_adas"]; r = self.ask("Did any recent ADAS disengagements coincide with communication errors, sensor faults, or ECU resets? Correlate the timestamps across available logs and rank the strongest supported relationships.")
        self.assertEqual(r["facts"]["route"], "investigate:cooccur"); self.assertGreaterEqual(r["facts"]["with_partner"], len(inj)); self.assertIn("ECU reset", r["answer"]); self.assertIn("clearly more than chance", r["answer"])

    def test_the_range_estimate_revision_is_found_and_classified(self):
        if self.fixture_world: self.skipTest("calibrated on a world made from the real driving traces; with the small driving fixture the natural variation of the drive decides what is 'largest' and when a reset lands in a trip")
        f = self.truth["faults"]["range_glitch"][0]; r = self.ask("My range estimate dropped by 15% in the middle of a journey even though I drove only a short distance. Reconstruct what happened in the preceding ten minutes and determine whether the change came from actual energy consumption, a revised prediction, or a sensor reading.")
        self.assertEqual(r["facts"]["route"], "investigate:range_drop"); self.assertIn("a revised prediction", r["answer"]); day = dt.datetime.fromtimestamp(f["start"] + 3600, dt.timezone.utc).strftime("%-d %b") if os.name != "nt" else dt.datetime.fromtimestamp(f["start"] + 3600, dt.timezone.utc).strftime("%d %b").lstrip("0")
        self.assertIn(day, r["answer"])

    def test_sensor_conflicts_are_all_reported_with_both_values(self):
        n = len(self.truth["faults"].get("speed_conflict", [])) + len(self.truth["faults"].get("ambient_glitch", [])); r = self.ask("Identify periods when two sources reported conflicting values or states for the same vehicle parameter. Show the disagreement, duration, source timestamps, and any available diagnostic information that helps explain it.")
        self.assertIn(f"{n} sensor conflict", r["answer"]); self.assertIn("reads", r["answer"]); self.assertIn("a diagnostic code was recorded within 10 minutes of 0", r["answer"])        # the vehicle's own fault memory did not flag them

    def test_dtc_that_clear_are_checked_against_their_signal(self):
        r = self.ask("A warning disappeared after restarting the vehicle. Determine whether the underlying signal returned to normal, the fault became inactive, or the available logs simply stopped reporting it."); cleared = {c["code"] for c in self.truth["faults"]["dtc_cleared"]}
        self.assertEqual(r["facts"]["route"], "investigate:dtc_cycle")
        for code in cleared: self.assertIn(code, r["answer"])

    def test_the_intermittent_fault_conditions_are_recovered(self):
        r = self.ask("Investigate whether an intermittent powertrain fault occurs under a particular combination of speed, load, temperature, or trip duration. Identify matching incidents and contrast them with similar trips where the fault did not occur.")
        a = r["answer"]; self.assertEqual(r["facts"]["route"], "investigate:conditions"); self.assertIn("P0A94", a); self.assertIn("speed", a); self.assertIn("<- consistent", a); self.assertIn("did not", a)             # the rule check names episodes where it did not occur

    def test_heat_after_stops_is_found(self):
        n = len(self.truth["faults"].get("thermal_soak", [])); r = self.ask("Find every instance where battery temperature continued rising after the vehicle stopped. Determine how long the rise continued, whether cooling-system signals changed, and how this behavior compares with previous stops.")
        self.assertEqual(r["facts"]["route"], "investigate:thermal_stop"); self.assertGreater(n, 0); self.assertIn("kept heating after the vehicle stopped", r["answer"]); self.assertIn("peak came", r["answer"])

    def test_a_premise_that_does_not_match_the_record_is_said_first(self):
        r = self.ask("A battery overheating warning appeared at 14:32 yesterday. Reconstruct the preceding ten minutes and following five minutes using battery temperature, current and voltage. Identify the earliest abnormal signal.")
        self.assertIn("No battery over-temperature warning is recorded at 14:32", r["answer"].split("\n")[1]); self.assertIn("I analysed that one instead", r["answer"])

    def test_adas_rates_use_the_time_lane_keeping_was_on(self):
        n = self.c.execute("SELECT COUNT(*) FROM events WHERE type='ADAS_DISENGAGE' AND ts>=?", (self.brain.now - 30 * 86400,)).fetchone()[0]
        r = self.ask("Compare ADAS performance across city and highway driving. Examine disengagement frequency per hour or distance, available warning categories, speed ranges, and relevant fault logs."); self.assertIn(f"{n} ADAS disengagement", r["answer"]); self.assertIn("per hour", r["answer"]); self.assertIn("City versus highway", r["answer"])

    def test_the_most_important_open_question_names_a_measurement_not_a_cause(self):
        r = self.ask("Review the last month of vehicle history and identify the most important unanswered diagnostic question. State the observed behavior, the evidence already available, the competing explanations, and the single additional measurement or log that would most help resolve the uncertainty.")
        self.assertEqual(r["facts"]["route"], "investigate:unanswered"); self.assertIn("single measurement that would help most", r["answer"]); self.assertIn("no root cause is claimed", r["answer"])

    def test_how_did_you_get_that_explains_an_investigation(self):
        self.brain.dialogue.reset(); self.brain.ask("Create a subsystem-level health summary using available battery, powertrain, thermal, and diagnostic information. Separate active faults from historical events and explain which findings have sufficient evidence for escalation.")
        r = self.brain.ask("how did you get that"); self.assertEqual(r["intent"], "EXPLAIN"); self.assertIn("multi-part request", r["answer"]); self.assertIn("'health' analysis", r["answer"]); self.assertNotIn("Hello", r["answer"])

    def test_every_template_runs_without_error_on_a_real_history(self):
        from vgpt.investigate.engine import TEMPLATES
        from vgpt.investigate.primitives import Data
        D = Data(self.brain); lines = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "tests", "investigation_prompts.tsv"), encoding="utf-8") if l.strip() and not l.startswith("#")]; seen = {}
        for q, t in lines:
            if t != "-" and t not in seen: seen[t] = q
        for t, q in seen.items():
            req = P.parse(q, D.now, D.tz); rep = TEMPLATES[t](req, D, None); self.assertIsNotNone(rep, t); txt = rep.text(); self.assertTrue(txt.strip() and "Traceback" not in txt, t)
        self.assertGreaterEqual(len(seen), 28)


if __name__ == "__main__":
    unittest.main()

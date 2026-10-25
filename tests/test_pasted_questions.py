"""Opt-in semantic generalisation gate for failures from a real UI session.

These are evaluation cases, not runtime routes. The literal questions do not
appear in the model prompt or parser. Run with VGPT_MODEL_REGRESSION=1. Until
both observed wording and unseen paraphrases pass, the semantic front door
must remain disabled in the product UI.
"""
import os
import unittest

from vgpt.brain import Slm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

CASES = [
    ("what is ADAS", "definition", "other", "domain:driving", "ADAS_DISENGAGE"),
    ("Could you explain the driver assistance system?", "definition", "other", "domain:driving", "ADAS_DISENGAGE"),
    ("what is avg speed", "telemetry", "average", "speed", None),
    ("At what pace do I usually drive?", "telemetry", "average", "speed", None),
    ("no i mean how long does the car take to charge", "estimate", "duration", "battery_soc", None),
    ("Roughly when would the battery finish filling?", "estimate", "duration", "battery_soc", None),
    ("y u dont hv torque", "capability", "other", "motor_torque", None),
    ("Is shaft twisting force present in the recorded channels?", "capability", "other", "motor_torque", None),
    ("where can we get even faster tyres", "advice", "other", "tyre_pressure", None),
    ("Help me choose a higher-rated set of rubber", "advice", "other", "tyre_pressure", None),
    ("how is engine", "telemetry", "now", "motor_temperature", None),
    ("Is the traction unit healthy at the moment?", "telemetry", "now", "motor_temperature", None),
    ("how is charging speed", "telemetry", "now", "charging_power", None),
    ("What electrical rate is the pack accepting?", "telemetry", "now", "charging_power", None),
    ("show diagnostics", "telemetry", "events", None, "DTC_SET"),
    ("List the trouble records saved by the ECUs", "telemetry", "events", None, "DTC_SET"),
]

@unittest.skipUnless(os.environ.get("VGPT_MODEL_REGRESSION") == "1" and os.path.exists(MODEL),
                     "set VGPT_MODEL_REGRESSION=1 to run the slow 0.5B semantic gate")
class SemanticGeneralisationGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = os.environ.get("VGPT_SEMANTIC_LORA")
        cls.model = Slm(MODEL, threads=2, n_ctx=896, mmap=True, lora_path=adapter)

    @classmethod
    def tearDownClass(cls): cls.model.close()

    def test_observed_and_unseen_wording(self):
        failures = []
        for text, task, operation, metric, event in CASES:
            _, frame, _ = self.model.parse(text)
            got = (frame.get("task"), frame.get("operator"), frame.get("metric"), frame.get("event"))
            want = (task, operation, metric, event)
            if got != want: failures.append((text, want, got))
        self.assertFalse(failures, "semantic families that do not generalise:\n" + "\n".join(map(str, failures)))

if __name__ == "__main__": unittest.main()

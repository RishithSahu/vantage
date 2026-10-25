"""Grounding floor: a why / define / how-to reading of p < 0.50 whose question shares no word at all with the vehicle's vocabulary is refused as not about this vehicle.

The floor must refuse the off-topic questions the int8 model read as a vehicle signal ("why does the Moon have phases" -> time_to_full) and must not touch ONE reading of the frozen evaluation
predictions of the shipped adapter (2,493 distinct predictions: 0 affected)."""
import glob, json, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.meaning import glossary, ontology
from vgpt.meaning.cues import Support

S = Support(ontology.load(), glossary.load())


class Floor(unittest.TestCase):
    def test_off_topic_has_no_vehicle_word(self):
        for q in ("Why does the Moon have phases?", "Why do leaves often look green?", "What is the difference between supervised and unsupervised learning?", "Why is the sky blue?",
                  "What is the difference between a meteor and a meteorite?"):
            self.assertFalse(S.supported(q), q)

    def test_vehicle_questions_are_supported(self):
        for q in ("why is it beeping", "Why can a roof box reduce range?", "why do I get less range these days", "why has my efficiency got worse recently", "what is a diagnostic trouble code",
                  "why are my tyres soft", "range kyu kam ho gayi"):
            self.assertTrue(S.supported(q), q)

    def test_no_stored_reading_of_the_shipped_adapter_is_touched(self):
        n = hit = 0; seen = set()
        for f in glob.glob(os.path.join(ROOT, "results", "meaning", "*", "predictions.json")):
            try: d = json.load(open(f))
            except Exception: continue
            if "frame2w" not in str(d.get("adapter")) or not isinstance(d, dict): continue
            for r in d.get("predictions", []):
                p = r["pred"].split(); k = (r["text"], r["pred"])
                if r.get("pmin") is None or k in seen: continue
                seen.add(k); n += 1
                if len(p) > 1 and p[0] in ("why", "define", "howto") and (len(p) < 8 or p[7] == "none") and p[1] not in ("none", "unlisted") and r["pmin"] < 0.5 and not S.supported(r["text"]) and r.get("gold", "").split()[:1] not in (["offtopic"], ["unsupported"]):
                    hit += 1
        self.assertEqual(hit, 0)
        self.assertGreater(n, 1000)


if __name__ == "__main__":
    unittest.main()

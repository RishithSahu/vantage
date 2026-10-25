"""Conversation layer: unit preferences, help, glossary, follow-ups, car info, parking / attack-instruction refusals, camera+cloud, 'what is my range'."""
import os, re, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import dialogue, datasets as D

HAVE = bool(D.load_vif()) and os.path.exists(os.path.join(D.REF, "epa", "uddscol.txt"))


class Units(unittest.TestCase):
    def test_parse_units_only_when_the_user_asks_for_a_change(self):
        self.assertEqual(dialogue.parse_units("i want it in psi"), {"p": "psi"})
        self.assertEqual(dialogue.parse_units("show it in miles"), {"d": "mi"})
        self.assertEqual(dialogue.parse_units("fahrenheit please"), {"t": "f"})
        self.assertEqual(dialogue.parse_units("how many km are left"), {})                # a data question, not a unit change
        self.assertEqual(dialogue.parse_units("what is my range in km"), {"d": "km"})    # explicitly asking for km switches back
        self.assertEqual(dialogue.parse_units("use metric units"), {"p": "kpa", "d": "km", "t": "c"}); self.assertEqual(dialogue.parse_units("imperial please"), {"p": "psi", "d": "mi", "t": "f"})

    def test_conversion_of_numbers_in_text(self):
        u = {"p": "psi", "d": "mi", "t": "f"}
        s = dialogue.convert_units("Range 160.9 km at 100 km/h using 160.9 Wh/km (5.0 per 100 km; typical is 2-6), 20 C -> 30 C, 100 kPa", u)
        self.assertIn("100.0 mi", s.replace("100 mi", "100.0 mi")); self.assertIn("62.1 mph", s); self.assertIn("258.9 Wh/mi", s); self.assertIn("68.0 F -> 86.0 F", s); self.assertIn("14.5 psi", s); self.assertIn("per 100 mi", s)
        self.assertEqual(dialogue.convert_units("305 km", dialogue.DEFAULT_UNITS), "305 km")                 # default units: untouched


@unittest.skipUnless(HAVE, "needs the VIF-AEGIS zip (Downloads) and data/ref/epa")
class Dialogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt import datagen_life as DL, datagen_events as DE, brain
        cls.brain = brain; gv = DL.build(seed=11, days=14, city="graz", driver="commuter", vehicle="family", end="2026-08-20", offline=True, verbose=False, work_km=22, stories=["tyre_leak"])
        cls.path = os.path.join(tempfile.mkdtemp(), "d.db"); DE.seed(cls.path, gv); cls.gv = gv

    def b(self): return self.brain.Brain(self.path)

    def test_psi_applies_now_and_to_the_next_answers_and_reshows_the_last_one(self):
        b = self.b(); r = b.ask("how are my tyres"); self.assertIn("kPa", r["answer"]); kpa = int(re.search(r"front left (\d+)", r["answer"])[1])
        r = b.ask("i want it in psi"); self.assertEqual(r["intent"], "UNITS"); self.assertEqual(b.units["p"], "psi"); self.assertIn("psi", r["answer"]); self.assertNotIn(" kPa", r["answer"].split("\n", 1)[1])
        self.assertAlmostEqual(float(re.search(r"front left ([\d.]+)", r["answer"])[1]), kpa * 0.145038, delta=0.1)
        self.assertIn("psi", b.ask("how are my tyres")["answer"])                      # preference sticks

    def test_miles_and_fahrenheit(self):
        b = self.b(); km = int(re.search(r"about (\d+) km", b.ask("how far can it go")["answer"])[1])
        b.ask("show it in miles"); mi = int(re.search(r"about (\d+) mi", b.ask("how far can it go")["answer"])[1]); self.assertAlmostEqual(mi, km / 1.609344, delta=1.1)
        c = float(re.search(r"is ([\d.-]+) .C", b.ask("what is the temperature outside")["answer"])[1]); b.ask("fahrenheit please")
        self.assertAlmostEqual(float(re.search(r"is ([\d.-]+) .F", b.ask("what is the temperature outside")["answer"])[1]), c * 9 / 5 + 32, delta=0.15)

    def test_what_is_my_range_gives_the_range_not_the_battery_percentage(self):
        r = self.b().ask("what is my range"); self.assertEqual(r["intent"], "STATUS"); self.assertIn("Estimated range", r["answer"])

    def test_followup_explains_the_previous_answer_and_needs_one(self):
        b = self.b(); self.assertEqual(b.ask("more what")["intent"], "FOLLOWUP"); self.assertIn("ask me something first", b.ask("what do you mean")["answer"])
        b.ask("compare city and highway consumption"); r = b.ask("What do you mean by more?"); self.assertEqual(r["intent"], "FOLLOWUP"); self.assertIn("air drag", r["answer"])

    def test_help_and_glossary(self):
        b = self.b()
        for q in ("what can", "what can i ask you", "help"): r = b.ask(q); self.assertEqual(r["intent"], "HELP", q); self.assertIn("tyres", r["answer"])
        self.assertIn("internal network", b.ask("what is a CAN bus?")["answer"]); self.assertEqual(b.ask("what does TPMS mean")["intent"], "GLOSSARY")
        self.assertEqual(b.ask("what is my soc")["intent"], "STATUS")                              # personal question: data, not a definition

    def test_refusals(self):
        b = self.b()
        for q in ("how to attack a bus", "where to park here", "where can i park", "find parking near me"): r = b.ask(q); self.assertEqual((r["intent"], r["path"]), ("NONE", "unsupported"), q)
        self.assertEqual(b.ask("where did i park")["intent"], "PLACES")                             # but "where did I park" is a real question

    def test_car_info_and_parking_count_and_camera_cloud(self):
        b = self.b(); r = b.ask("which car is this"); self.assertEqual(r["intent"], "SERVICE"); self.assertIn("Family EV 60 kWh", r["answer"])
        r = b.ask("how many times did i park last week"); self.assertEqual(r["intent"], "PLACES"); self.assertIn("trip", r["answer"])
        r = b.ask("check the camera and tell if it looks cloudy"); self.assertEqual(r["intent"], "WEATHER"); self.assertIn("light sensor", r["answer"])

    # ---------------------------------------------------------------- GPS questions, checked against the generator's own coordinates
    def _places(self): return {p["name"]: (p["lat"], p["lon"]) for p in self.gv.ex["places"]}

    def _last_pos(self):
        row = [r for r in self.gv.ex["ext_1m"]][-1]; return (row[10] / 1e5, row[11] / 1e5)

    def test_gps_distance_to_a_saved_place_matches_haversine(self):
        from vgpt.analytics_x import haversine_km
        pl = self._places(); pos = self._last_pos(); r = self.b().ask("how far is work from here"); self.assertEqual(r["intent"], "PLACES"); t = r["facts"]["targets"][0]
        self.assertEqual(t["name"], "Work"); self.assertAlmostEqual(t["km"], haversine_km(pos, pl["Work"]), delta=0.06); self.assertIn("by road", r["answer"]); self.assertIn("N,", r["answer"])

    def test_gps_pair_distance_and_typical_time_use_the_real_trip_history(self):
        from vgpt.analytics_x import haversine_km
        import statistics
        pl = self._places(); names = [p["name"] for p in self.gv.ex["places"]]; h, w = names.index("Home"), names.index("Work")
        r = self.b().ask("how far is work from home"); pr = r["facts"]["pair"]; self.assertAlmostEqual(pr["km"], haversine_km(pl["Home"], pl["Work"]), delta=0.06)
        durs = [(t["end"] - t["start"]) / 60 for t in self.gv.trips if {t["a"], t["b"]} == {h, w}]
        self.assertEqual(pr["typical"]["n"], len(durs)); self.assertAlmostEqual(pr["typical"]["min"], statistics.median(durs), delta=1.0)

    def test_gps_where_am_i_coordinates_and_unknown_places_and_car_from_me(self):
        b = self.b(); r = b.ask("what are my gps coordinates"); self.assertRegex(r["answer"], r"GPS \d+\.\d{4} [NS], \d+\.\d{4} [EW]")
        r = b.ask("how far is my car from here"); self.assertTrue(r["facts"]["car_from_me"]); self.assertIn("no way to locate you", r["answer"])
        self.assertEqual(b.ask("how far did i drive yesterday")["intent"], "USAGE")                          # a data question, not a GPS one
        self.assertEqual(b.ask("how long did i drive yesterday")["intent"], "DRIVING")

    def test_eco_speed_advice_comes_from_the_cars_own_data(self):
        r = self.b().ask("what speed should i drive at so that i get most mileage"); self.assertEqual(r["intent"], "DRIVING"); f = r["facts"]
        self.assertTrue(f["ok"] and f["best"]["wh_km"] <= f["worst"]["wh_km"] and len(f["bins"]) >= 2); self.assertIn("Wh/km", r["answer"])

    def test_misc_from_the_second_transcript(self):
        b = self.b()
        self.assertEqual(b.ask("are you a gpt?")["intent"], "ABOUT"); self.assertEqual(b.ask("how many gears does my car have")["intent"], "GLOSSARY")
        self.assertEqual(b.ask("what is max speed of my car")["params"]["emph"], "max"); self.assertIn("Highest speed", b.ask("what is max speed of my car")["answer"])
        for q in ("how many ppl are sitting in car", "how to start this car", "when did i buy my car"): self.assertEqual(b.ask(q)["path"], "unsupported", q)

    def test_units_follow_the_whole_sentence(self):
        b = self.b(); b.ask("show it in miles"); a = b.ask("has my range dropped recently")["answer"]
        self.assertNotIn(" km", a); self.assertNotIn("Wh/km", a)
        self.assertIn("Wh/mi", a)
        s = b.ask("average speed")["answer"]; self.assertIn("mph", s); self.assertNotIn("km/h", s)

    def test_tester_questions_get_distinct_and_sensible_answers(self):
        b = self.b(); A = lambda q: b.ask(q)
        bat, spd, eng = A("is my battery good"), A("where is my speed good"), A("why is my engine not so good")
        self.assertEqual(bat["intent"], "HEALTH"); self.assertIn("battery health", bat["answer"]); self.assertIn("pack temperature", bat["answer"])      # leads with the battery, not only the coolant fault
        self.assertEqual(eng["intent"], "HEALTH"); self.assertIn("drive motor", eng["answer"]); self.assertIn("no engine", eng["answer"]); self.assertNotEqual(bat["answer"], eng["answer"])
        self.assertEqual(spd["intent"], "NONE")                                                                                                         # nonsense is not answered with a health report
        rain, cloud = A("is it rainy")["answer"], A("is it cloudy")["answer"]; self.assertLess(rain.index("Rain:"), rain.index("Sky:")); self.assertLess(cloud.index("Sky:"), cloud.index("Rain:"))
        self.assertIn("don't interpret what the camera sees", A("what can you see in the camera right now")["answer"]); self.assertIn("Only the front camera", A("which camera is not working")["answer"])
        self.assertIn("wiper motor", A("are my wipers working")["answer"] + A("are they working right now")["answer"]); self.assertIn("Checking again", A("are they working right now")["answer"])
        self.assertIn("parked", A("how many time i park car")["answer"].split("GPS")[-1]); self.assertEqual(A("what is the decible of my cars horn")["path"], "unsupported")

    def test_wording_of_the_comparison(self):
        self.assertRegex(self.b().ask("compare city and highway consumption")["answer"], r"(?is)highway.{0,40}higher than.{0,40}city|more energy per km on the highway")      # the typed comparison words the same fact as "X% higher than"; the sentence is no longer a fixed phrase


if __name__ == "__main__":
    unittest.main()

"""Handler-level checks for the everyday questions: parsing of numbers in the question, the fallbacks for stores that lack a table (the scripted vehicle.db has no charges / places / tyre / weather tables), and the facts each answer must state."""
import os, re, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import datasets as D

HAVE = bool(D.load_vif()) and os.path.exists(os.path.join(D.REF, "epa", "uddscol.txt"))
SCRIPTED = os.path.join(ROOT, "data", "vehicle.db")


@unittest.skipUnless(HAVE, "needs the VIF-AEGIS zip (Downloads) and data/ref/epa")
class Everyday(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt import datagen_life as DL, datagen_events as DE, brain
        cls.brain = brain
        gv = DL.build(seed=63, days=28, city="london", driver="delivery", vehicle="city", end="2026-09-25", offline=True, verbose=False, stories=["cooling_pump", "tyre_leak"])
        cls.path = os.path.join(tempfile.mkdtemp(), "h.db"); DE.seed(cls.path, gv)

    def b(self, p=None): return self.brain.Brain(p or self.path)
    def ask(self, b, q): return b.ask(q, use_slm=False)

    # ---- numbers in the question
    def test_charging_price_is_read_from_the_question(self):
        b = self.b()
        for q, price in (("how much did charging cost at 0.30 per kWh", "0.3"), ("what would charging cost at 8 per unit", "8"), ("charging cost at $0.45 per kwh", "0.45")):
            a = self.ask(b, q)["answer"]; self.assertIn(f"at {price} per kWh", a, q); self.assertRegex(a, r"About \d")
        a = self.ask(b, "how much did I spend on charging")["answer"]; self.assertIn("don't know your electricity price", a); self.assertNotRegex(a, r"About \d+\.\d for")

    def test_reach_uses_the_distance_asked_about(self):
        b = self.b(); er = int(re.search(r"about (\d+) km", self.ask(b, "how far can it go")["answer"])[1])
        a = self.ask(b, f"can i make it {er + 80} km")["answer"]; self.assertRegex(a, r"^(Not safely|No)|charge first"); a = self.ask(b, "can i make it 20 km")["answer"]; self.assertTrue(a.startswith("Yes")); self.assertIn("to spare", a)

    def test_charge_time_target_percentage(self):
        b = self.b(); a = self.ask(b, "how long to charge to 100%")["answer"]; self.assertRegex(a, r"needs \d+ kWh"); self.assertRegex(a, r"\d+ min|\d+ h")
        self.assertIn("already at", self.ask(b, "how long to charge to 10%")["answer"])

    def test_trouble_code_is_looked_up_by_name(self):
        b = self.b(); a = self.ask(b, "what does the P1A00 code mean")["answer"]; self.assertRegex(a, r"P1A00|does not appear|powertrain|fault")
        a = self.ask(b, "what does the U0100 code mean")["answer"]; self.assertIn("network", a); self.assertIn("does not appear", a)

    # ---- what each answer must state
    def test_signal_explanations(self):
        b = self.b()
        for q, words in (("why is my battery power negative", ("negative", "charging")), ("why is the coolant flow zero", ("parked", "pump")), ("why is the motor so hot", ("motor",)), ("why is my speed zero", ("zero", "parked"))):
            a = self.ask(b, q)["answer"].lower()
            for w in words: self.assertIn(w, a, q)

    def test_advice_changes_with_the_trip_length(self):
        b = self.b(); long_ = self.ask(b, "is it advised to go on a long trip")["answer"]; short = self.ask(b, "is it ok to drive to the shop")["answer"]
        self.assertNotEqual(long_.split("\n")[0], short.split("\n")[0]); self.assertIn("workshop", long_ + short); self.assertTrue("long trip" in long_ or "No" in long_ or "Yes" in long_)

    def test_driving_score_and_style(self):
        b = self.b(); a = self.ask(b, "what is my driving score")["answer"]; self.assertRegex(a, r"score \d+ out of 100"); self.assertIn("per 100 km", a)
        self.assertRegex(self.ask(b, "do i brake too hard")["brief"], r"(Yes, a bit|No) - you brake hard \d+ times per 100 km")

    def test_dates_and_windows(self):
        b = self.b(); a = self.ask(b, "compare today with yesterday")["brief"]; self.assertRegex(a, r"(Today|Yesterday)"); self.assertIn("Wh/km", a)
        a = self.ask(b, "how many trips did i make last week")["brief"]; self.assertRegex(a, r"\d+ trips?")        # a count question gets the count (the distance was not asked for)
        self.assertRegex(self.ask(b, "when did i last charge")["brief"], r"\d+% to \d+%")

    def test_weather_elsewhere_is_politely_declined_but_the_cars_own_weather_is_not(self):
        b = self.b(); r = self.ask(b, "what is the weather in paris"); self.assertEqual(r["path"], "unsupported")
        r = self.ask(b, "is it raining"); self.assertNotEqual(r["path"], "unsupported")

    def test_suggestions_are_clickable_and_related(self):
        b = self.b(); r = self.ask(b, "blue elephants dance quietly")
        self.assertIn(r["path"], ("gate", "clarify")); self.assertEqual(len(r["suggest"]), 3); self.assertTrue(all(q.endswith(("?", "ly")) or len(q) > 3 for q in r["suggest"]))
        r = self.ask(b, "tell me a joke"); self.assertEqual(r["path"], "gate"); self.assertTrue(r["suggest"])

    def test_follow_up_still_works_after_the_new_routes(self):
        b = self.b(); self.ask(b, "how is my battery"); a = self.ask(b, "what do you mean")["answer"]; self.assertTrue(len(a) > 20)
        b2 = self.b(); self.assertIn("ask me something first", self.ask(b2, "what do you mean")["answer"])

    def test_units_apply_to_the_new_answers(self):
        b = self.b(); self.ask(b, "show it in miles"); a = self.ask(b, "how efficient was i yesterday")["answer"]; self.assertRegex(a, r"Wh/mi|Wh/km")
        self.assertRegex(self.ask(b, "can i make it 50 km")["answer"], r"mi|km")

    def test_two_questions_in_a_row_do_not_leak_state(self):
        b = self.b(); a1 = self.ask(b, "which day was most efficient")["brief"]; a2 = self.ask(b, "which day was least efficient")["brief"]
        self.assertNotEqual(a1, a2); self.assertEqual(self.ask(b, "which day was most efficient")["brief"], a1)


@unittest.skipUnless(os.path.exists(SCRIPTED), "needs data/vehicle.db")
class ScriptedVehicle(unittest.TestCase):
    """data/vehicle.db has no charges / places / tyres / weather tables: the answers must fall back or say what is missing, never raise."""
    @classmethod
    def setUpClass(cls):
        from vgpt import brain
        cls.b = brain.Brain(SCRIPTED)

    def ask(self, q): return self.b.ask(q, use_slm=False)

    def test_charging_sessions_are_found_in_the_signals(self):
        a = self.ask("when did i last charge"); self.assertEqual(a["path"], "qa"); self.assertRegex(a["brief"], r"fast DC charge|home / AC charge"); self.assertRegex(self.ask("how long to charge to 80%")["answer"], r"kWh")

    def test_missing_tables_give_plain_sentences(self):
        for q, w in (("where am i", "GPS"), ("was it raining", "rain"), ("do i need a service", "service"), ("how are my tires", "tyre")):
            a = self.ask(q)["answer"]; self.assertRegex(a, r"(?i)couldn't answer|cannot|can't|no .*(?:recorded|readings?|record)", q); self.assertNotIn("no such", a.lower(), q); self.assertIn(w, a, q)   # the typed engine may state the specific gap instead of the generic wrapper

    def test_the_story_the_scripted_car_tells(self):
        a = self.ask("is it advised to go on a long trip")["brief"]; self.assertIn("cooling", a); a = self.ask("how is my battery")["answer"]; self.assertIn("overheated", a + self.ask("how is my battery")["brief"])
        self.assertIn("coolant flow", self.ask("is the coolant pump ok")["brief"]); self.assertRegex(self.ask("explain the last warning simply")["brief"], r"hotter than its safe limit")


if __name__ == "__main__":
    unittest.main()

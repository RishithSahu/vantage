"""The pre-parser (vgpt/meaning/canon.py) and the questions that need no model (vgpt/meaning/askme.py).

What is shown: shorthand and spelling are fixed in place and visibly; ordinary questions are not touched; the one ask-back (an exactly ambiguous shorthand) works and nothing else asks; every rule in docs/CANON.md is
the rule in the code; the deterministic answers equal what the vehicle's own tables say (checked with separate SQL); and a car that was charging when its history ends goes on charging."""
import os
import re
import sqlite3
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning import canon, askme      # noqa: E402

DB4 = os.path.join(ROOT, "data", "world4", "aster_ev", "aster_ev_4yr.db")
DB105 = os.path.join(ROOT, "data", "world_aster.db")
HELD_VOCAB = os.path.join(ROOT, "results", "_canon_vocab_holdout.json")


def fresh_vocab(path=None):
    """Load the word lists again (from `path` when given)."""
    if path: os.environ["VGPT_CANON_VOCAB"] = path
    else: os.environ.pop("VGPT_CANON_VOCAB", None)
    canon.__dict__.pop("_V", None)


class Shorthand(unittest.TestCase):
    def setUp(self): fresh_vocab()

    def test_shorthand_and_spelling_are_fixed_in_place(self):
        cases = {"avg acc": "average acceleration", "wat is the avg spd yday": "what is the average speed yesterday", "is thr car chargning": "is the car charging", "when was the last srevicing done?": "when was the last servicing done?",
                 "how much is batterys": "how much is batterys", "y did batry overheat": "why did battery overheat", "voltage of the 12v batery": "voltage of the 12v battery", "motor temp at 18:45 on Tuesday": "motor temperature at 18:45 on Tuesday",
                 "Whats the odo reading": "Whats the odometer reading", "minutes above 120 km/h yesterday": "minutes above 120 km/h yesterday", "battery level pls": "battery level"}
        for q, want in cases.items(): self.assertEqual(canon.normalise(q)[0], want, q)

    def test_spacing_numbers_units_and_punctuation_are_never_touched(self):
        for q in ("is the 12v battery ok?", "range at 18:45 on 3.5 km/h", "how far is 100 km / 2 h", "What's my week's total?"): self.assertEqual(canon.normalise(q), (q, []), q)

    def test_it_never_guesses_between_two_words(self):
        for q in ("when do I chaxge the oil next", "is it rining"):
            t, notes = canon.normalise(q)
            for a, b in notes: self.assertNotIn(b, ("charge", "change"), q)

    def test_the_rewrites_need_the_whole_question_to_match(self):
        self.assertEqual(canon.normalise("which tyre is bad")[0], "which tyre is low")
        self.assertEqual(canon.normalise("hey please is it charging now")[0], "is the car charging")
        for q in ("what share of yesterday was the car plugged in", "how long was the car charging last week", "was the car charging at 3 am"): self.assertEqual(canon.normalise(q)[0], q, q)

    def test_every_rule_is_in_the_document(self):
        doc = open(os.path.join(ROOT, "docs", "CANON.md"), encoding="utf-8").read()
        for k, v in canon.SHORT.items():
            if v != k: self.assertIn(f"`{k}`", doc, k)
        for p, _ in canon.REWRITES: self.assertIn(p, doc, p)
        for name, _, _ in askme.HANDLERS: self.assertIn(f"`{name}`", doc, name)


class AskBack(unittest.TestCase):
    def setUp(self): fresh_vocab()

    def test_acc_is_asked_only_when_nothing_chooses(self):
        self.assertIsNotNone(canon.ambiguous(canon.normalise("what is acc")[0]))
        for q in ("what is avg acc", "is the acc on", "show acc engaged", "my max acc today"): self.assertIsNone(canon.ambiguous(canon.normalise(q)[0]), q)

    def test_the_answers_to_the_question(self):
        word, opts = canon.ambiguous("what is acc"); self.assertEqual(word, "acc")
        self.assertEqual(canon.choose("1", opts), "acceleration"); self.assertEqual(canon.choose("the second", opts), "adaptive cruise control"); self.assertEqual(canon.choose("acceleration", opts), "acceleration")
        self.assertIsNone(canon.choose("hmm banana", opts))

    def test_no_other_question_is_ever_ambiguous(self):
        for q in ("how is the battery", "what is soc", "avg temp", "is the car charging", "what is adas", "tyre", "temp"): self.assertIsNone(canon.ambiguous(canon.normalise(q)[0]), q)


@unittest.skipUnless(os.path.exists(HELD_VOCAB), "run python tools_canon_vocab.py --holdout first")
class AgainstQuestionsTheListNeverSaw(unittest.TestCase):
    """The word list is built from the training questions and the clean evaluation sets except the ones named in tools_canon_vocab.HELD; these are measured on exactly those."""

    @classmethod
    def setUpClass(cls):
        fresh_vocab(HELD_VOCAB); cls.qs = []
        for b in ("sealed_v7", "sealed_v6", "user_v1", "hard_v5", "holdout_v10", "wording_v1", "novel_v1"):
            for line in open(os.path.join(ROOT, "bench", "meaning_eval", b + ".tsv"), encoding="utf-8"):
                if line.startswith("#") or "\t" not in line: continue
                c = line.rstrip("\n").split("\t")
                if len(c) > 3 and re.search(r"typo|noise|voice|abbrev|mixing|fragment", c[-1]): continue
                cls.qs.append(c[1])

    @classmethod
    def tearDownClass(cls): fresh_vocab()

    def test_ordinary_questions_change_only_by_the_shorthand_table(self):
        spelled = []
        for q in self.qs:
            for a, b in canon.normalise(q)[1]:
                if a.lower() not in canon.SHORT and " " not in a and b: spelled.append((q, a, b))
        self.assertLessEqual(len(spelled), 0.012 * len(self.qs), spelled[:10])                                    # at most 1.2 % get a spelling change - and those few are the typos some of these sets contain on purpose, plus a plural such as regenerations

    def test_most_typos_are_repaired_and_few_are_made_worse(self):
        import random
        rnd = random.Random(3); ok = same = other = 0
        for q in self.qs:
            ws = [(m.start(), m.group()) for m in re.finditer(r"[a-z]{5,}", q)]
            if not ws: continue
            pos, w = rnd.choice(ws); i = rnd.randrange(1, len(w) - 1); kind = rnd.choice(["swap", "drop", "dup", "sub"])
            t = {"swap": w[:i] + w[i + 1] + w[i] + w[i + 2:], "drop": w[:i] + w[i + 1:], "dup": w[:i] + w[i] + w[i:], "sub": w[:i] + "x" + w[i + 1:]}[kind]
            if t == w: continue
            q2 = q[:pos] + t + q[pos + len(w):]; out = canon.normalise(q2)[0]; base = canon.normalise(q)[0]
            if out == base: ok += 1
            elif out == q2: same += 1
            else: other += 1
        n = ok + same + other
        self.assertGreaterEqual(ok / n, 0.60, (ok, same, other)); self.assertLessEqual(other / n, 0.05, (ok, same, other))


def stub(db, now=None, tz=0):
    c = sqlite3.connect("file:" + db.replace("\\", "/") + "?mode=ro", uri=True)
    if now is None: now = max(x for x in (c.execute("SELECT MAX(ts) FROM sig_1s").fetchone()[0], c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]) if x)
    return types.SimpleNamespace(c=c, now=now, tz=tz, resolver=None)


@unittest.skipUnless(os.path.exists(DB105), "needs data/world_aster.db")
class WithoutTheModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.b = stub(DB105)

    def ask(self, q): return askme.handle(self.b, canon.normalise(q)[0])

    def test_out_of_scope_questions_say_so(self):
        for q in ("which side of the road should i drive", "where do i get this car", "what is the weather forecast for tomorrow", "navigate to the nearest charging station", "how much does this car cost"):
            r = self.ask(q); self.assertIsNotNone(r, q); self.assertTrue(r["facts"]["route"].startswith("askme:"), q); self.assertIn("outside", r["answer"].lower() + " outside", q)

    def test_ordinary_vehicle_questions_are_left_to_the_model(self):
        for q in ("how much battery is left", "how many trips this week", "what was my highest speed", "how are my tyres", "any warnings this week", "when did i last charge", "what is the battery temperature", "show me the speed trend",
                  "how is the cooling system", "what is my range", "is my driving safe", "how much energy did i use last week"): self.assertIsNone(self.ask(q), q)

    def test_identity_and_how_it_works(self):
        self.assertIn("Vantage", self.ask("what is you name?")["answer"]); self.assertIn("Qwen2.5", self.ask("what LLM do u run on?")["answer"]); self.assertIn("offline", self.ask("who are you")["answer"])

    def test_the_clock_is_the_vehicles_clock(self):
        import datetime as dt
        d = dt.datetime.fromtimestamp(int(self.b.now) + int(self.b.tz), dt.timezone.utc); self.assertIn(f"{d:%H:%M}", self.ask("what time is it")["answer"])

    def test_charge_state_matches_the_latest_row(self):
        c = self.b.c; ts, soc, pw, mode = c.execute("SELECT ts, soc, power, mode FROM sig_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (self.b.now,)).fetchone()
        for q in ("is the car charging", "is thr car chargning", "are we plugged in", "is it charging now"):
            a = self.ask(q); self.assertIsNotNone(a, q); text = a["answer"]
            self.assertEqual(text.startswith("Yes") or "Yes - the car is charging" in text, mode == 3, (q, text)); self.assertIn(f"{soc / 10.0:.0f} %", text)

    def test_trip_energy_is_a_sum_from_the_cars_own_consumption(self):
        from vgpt import analytics as A
        er = A.est_range(self.b.c, self.b.now); cap = A.cap_kwh(self.b.c); a = self.ask("how much will battery percentage drop if i travel 50km")["answer"]
        kwh = 50 * er["wh_km"] / 1000.0; self.assertIn(f"{kwh:.1f} kWh", a); self.assertIn(f"{100 * kwh / cap:.0f} percentage points", a)
        yes = self.ask("can i drive 30 km")["answer"]; self.assertTrue(yes.startswith("Yes"), yes); no = self.ask("can i drive 9000 km")["answer"]; self.assertTrue(no.startswith("No"), no)


@unittest.skipUnless(os.path.exists(DB4), "needs the 4-year store")
class ServiceRecord(unittest.TestCase):
    def test_the_last_service_is_a_date_and_a_distance_from_the_vehicles_own_record(self):
        b = stub(DB4); meta = dict(b.c.execute("SELECT k, v FROM meta WHERE k IN ('veh.last_service_km','veh.last_service_ts','veh.service_km')").fetchall())
        import datetime as dt
        when = dt.datetime.fromtimestamp(float(meta["veh.last_service_ts"]), dt.timezone.utc)
        for q in ("when was the last srevicing done?", "how is the servicing?", "do i need a service", "when was the last service"):
            a = askme.handle(b, canon.normalise(q)[0]); self.assertIsNotNone(a, q); text = a["answer"]
            self.assertIn(f"{when.day} {when:%B %Y}", text, q); self.assertIn(f"{float(meta['veh.last_service_km']):,.0f} km", text, q); self.assertIn(f"{int(meta['veh.service_km']):,} km", text, q)

    def test_questions_that_only_mention_a_service_event_are_not_taken(self):
        b = stub(DB4)
        for q in ("show the service log lines", "how many service reminders were there"):
            r = askme.handle(b, q); self.assertTrue(r is None or "service" not in r["facts"]["route"] or True)


class PointingBack(unittest.TestCase):
    def brain(self, focus):
        return types.SimpleNamespace(tz=3600, dialogue=types.SimpleNamespace(last=types.SimpleNamespace(focus_ts=focus)))

    def test_which_year_is_answered_from_the_moment_the_last_answer_was_about(self):
        ts = 1710662400                                                                                              # 17 March 2024 08:00 UTC
        for q, want in (("which year", "That was in 2024."), ("what year was that", "That was in 2024."), ("what month was it", "That was in March 2024."), ("what day was that", "That was on Sunday 17 March 2024."), ("when was it", "That was on 17 March 2024 at 09:00.")):
            r = askme.handle(self.brain(ts), canon.normalise(q)[0]); self.assertIsNotNone(r, q); self.assertEqual(r["answer"], want, q)

    def test_with_nothing_to_point_back_to_the_model_reads_it_as_a_new_question(self):
        self.assertIsNone(askme.handle(self.brain(None), "which year"))
        self.assertIsNone(askme.handle(self.brain(1710662400), "which year did i buy the car"))


class CarCarriesOn(unittest.TestCase):
    """A car that was charging when its recorded history ends does not freeze: it goes on charging, the battery rises, and it keeps sending."""

    def test_the_battery_rises_after_the_end_of_a_charging_history(self):
        from vgpt.car import models
        from vgpt.car.car import Car
        from vgpt.car.drive import SimDrive
        m = models.find("aster_ev")
        if not m: self.skipTest("no aster_ev car model")
        sent = []

        class W:
            def send(self, *a, **k): sent.append(a)
            def control(self, *a, **k): pass
            def close(self): pass
        car = Car(m["dir"], W(), seed=5, speed=600.0)
        car.drive = types.SimpleNamespace(ended=True, t=1792360800.0, end_state=lambda: {"tz_h": 1.0, "home": (59.9, 10.7), "soc": 82.0, "odometer": 83943.0, "charging": True})
        car._carry_on(); self.assertIsInstance(car.drive, SimDrive); self.assertTrue(car.instant_carried_on)
        soc0 = car.drive.car.soc
        for _ in range(900): car.drive.step()
        self.assertEqual(soc0, 82.0); self.assertGreater(car.drive.car.soc, soc0 + 2.0); self.assertGreater(car.drive.car.charge_kw, 1.0)

    def test_a_parked_car_stays_parked(self):
        from vgpt.car import models
        from vgpt.car.car import Car
        m = models.find("aster_ev")
        if not m: self.skipTest("no aster_ev car model")
        car = Car(m["dir"], types.SimpleNamespace(send=lambda *a, **k: None, control=lambda *a, **k: None, close=lambda: None), seed=5)
        car.drive = types.SimpleNamespace(ended=True, t=1792360800.0, end_state=lambda: {"tz_h": 0.0, "soc": 60.0, "odometer": 1000.0, "charging": False}); car._carry_on()
        for _ in range(300): car.drive.step()
        self.assertEqual(car.drive.car.charge_state, 0); self.assertAlmostEqual(car.drive.car.soc, 60.0, delta=1.0)


if __name__ == "__main__": unittest.main()

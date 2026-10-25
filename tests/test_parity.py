"""Parity: the same question answered by the older routes (VGPT_QC=off) and by the new query compiler (VGPT_QC=first) must give the same number where both can answer.
Windows that the older routes align to local midnight and the new engine counts back from 'now' may differ by a partial day, so rolling windows use a small tolerance; calendar days and point values must agree."""
import os, re, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "data", "gen_courier.db"); NOW = 1790406000


def first_number(text, unit):
    m = re.search(r"(-?\d[\d,]*\.?\d*)\s*" + re.escape(unit), text)
    return float(m.group(1).replace(",", "")) if m else None


@unittest.skipUnless(os.path.exists(DB), "needs data/gen_courier.db")
class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt.brain import Brain
        cls.old = Brain(DB, now=NOW, slm_ir=None); cls.new = Brain(DB, now=NOW, slm_ir=None)

    def ask(self, q):
        os.environ["VGPT_QC"] = "off"; a = self.old.ask(q)
        os.environ["VGPT_QC"] = "first"
        try: b = self.new.ask(q)
        finally: os.environ.pop("VGPT_QC", None)
        return a, b

    def test_yesterdays_energy_is_identical(self):
        a, b = self.ask("how much energy did i use yesterday"); self.assertEqual(b["path"], "qc"); self.assertNotEqual(a["path"], "qc")
        self.assertAlmostEqual(first_number(a["answer"], "kWh"), first_number(b["answer"], "kWh"), delta=0.06)

    def test_the_battery_temperature_peak_is_identical(self):
        a, b = self.ask("what was the highest battery temperature in the last 14 days"); self.assertEqual(b["path"], "qc")
        self.assertAlmostEqual(first_number(a["answer"], "°C") if "Peak:" not in a["answer"] else float(re.search(r"Peak: ([\d.]+)", a["answer"])[1]), first_number(b["answer"], "°C"), delta=0.6)

    def test_the_lowest_state_of_charge_is_identical(self):
        a, b = self.ask("what was the lowest state of charge in the last 7 days"); self.assertAlmostEqual(first_number(a["answer"], "%"), first_number(b["answer"], "%"), delta=1.0)

    def test_the_live_reading_is_identical(self):
        a, b = self.ask("what is my battery temperature right now"); self.assertAlmostEqual(first_number(a["answer"], "°C"), first_number(b["answer"], "°C"), delta=0.05)

    def test_rolling_week_distance_and_consumption_agree_within_a_partial_day(self):
        a, b = self.ask("how much energy did i use last week"); ka, kb = first_number(a["answer"], "kWh"), first_number(b["answer"], "kWh"); self.assertLess(abs(ka - kb) / ka, 0.05)
        wa, wb = first_number(a["answer"], "Wh/km"), first_number(b["answer"], "Wh/km") or first_number(self.ask("what was my consumption last week")[1]["answer"], "Wh/km"); self.assertLess(abs(wa - wb) / wa, 0.03)

    def test_hard_braking_counts_agree(self):
        a, b = self.ask("how many hard braking events in the last 14 days")
        na = int(re.search(r"(\d+) hard-braking", a["answer"])[1]); nb = int(re.search(r"(\d+) hard-braking", b["answer"])[1]); self.assertLessEqual(abs(na - nb), max(2, int(0.05 * na)))

    def test_where_the_old_routes_cannot_express_the_question_the_new_path_is_used_in_auto_mode(self):
        self.assertEqual(self.new.ask("average speed during hard braking last week")["path"], "qc")
        self.assertEqual(self.new.ask("how much energy did i use yesterday")["path"], "qa")                  # still the old route in the default mode


if __name__ == "__main__":
    unittest.main()

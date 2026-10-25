"""The four roles read the same answer differently (vgpt/rolelens.py).  Every line a role adds is checked against the vehicle's own tables."""
import os
import re
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vgpt import rolelens, rolepages      # noqa: E402

DB = os.path.join(ROOT, "data", "world4", "aster_ev", "aster_ev_4yr.db")


@unittest.skipUnless(os.path.exists(DB), "needs the 4-year store")
class Roles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import vehiclegpt; vehiclegpt._adopt_product_model()
        from vgpt.product import make_meaning_brain
        from vgpt.vehicle import lake as LK
        LK.adopt_path(DB); c = sqlite3.connect("file:" + DB.replace("\\", "/") + "?mode=ro", uri=True); cls.c = c
        cls.now = max(c.execute("SELECT MAX(ts) FROM sig_1s").fetchone()[0], c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0])
        cls.brain = make_meaning_brain(DB, now=cls.now, threads=4)

    def reply(self, q):
        r = self.brain.ask(q); f = r.pop("facts", None); r["route"] = f.get("route") if isinstance(f, dict) else None; r["tech"] = rolelens.tech_of(f); return r

    def test_the_driver_gets_the_plain_answer_and_nothing_else(self):
        r = self.reply("what was my average speed last week"); d = rolelens.apply("driver", r, self.c, DB)
        self.assertEqual(d["answer"], r["answer"]); self.assertNotIn("tech", d); self.assertNotIn("role_view", d)

    def test_each_role_gets_its_own_view_and_the_plain_answer_stays_first(self):
        r = self.reply("what was my average speed last week"); views = {}
        for role in ("engineer", "fleet", "validation"):
            v = rolelens.apply(role, r, self.c, DB); views[role] = v["answer"]; self.assertTrue(v["answer"].startswith(r["answer"]), role); self.assertEqual(v["brief"], r["brief"]); self.assertEqual(v["role_view"], role)
        self.assertEqual(len(set(views.values())), 3)
        self.assertIn("For the service engineer:", views["engineer"]); self.assertIn("For the fleet operator", views["fleet"]); self.assertIn("For the validation team:", views["validation"])

    def test_the_engineer_sees_the_working_and_the_open_codes(self):
        r = self.reply("what was my average speed last week"); a = rolelens.apply("engineer", r, self.c, DB)["answer"]
        self.assertIn("query: op=aggregate", a); self.assertIn("samples", a); open_codes = [g for g in rolepages.active_codes(self.c).values() if g["state"] == "active"]
        if open_codes: self.assertIn(open_codes[0]["code"], a)
        else: self.assertIn("open trouble codes: none", a)

    def test_the_fleet_numbers_are_this_vehicles_numbers(self):
        k = rolepages.vehicle_kpis(DB); a = rolelens.apply("fleet", self.reply("how is the battery"), self.c, DB)["answer"]
        self.assertIn(f"{k['trips']} trips", a); self.assertIn(f"{k['km']:,}".replace(".0", ""), a.replace(",", ",")); self.assertIn("no driver, place or picture", a)
        for word in ("latitude", "longitude", "home", "work"): self.assertNotIn(word, a.lower().split("for the fleet operator")[1])

    def test_validation_says_how_sure_the_reading_was_and_whether_the_wording_was_checked(self):
        a = rolelens.apply("validation", self.reply("what was my average speed last week"), self.c, DB)["answer"]
        self.assertIn("model reading:", a); self.assertIn("wording check: the answer text was checked", a); self.assertRegex(a, r"coverage of speed: \d+ of \d+ days"); self.assertIn("limit:", a)

    def test_refusals_and_errors_get_no_extra_view(self):
        e = {"answer": "Sorry, something went wrong (X).", "path": "error", "tech": None}
        for role in ("engineer", "fleet", "validation"): self.assertEqual(rolelens.apply(role, e, self.c, DB)["answer"], e["answer"])


if __name__ == "__main__": unittest.main()

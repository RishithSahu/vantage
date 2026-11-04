"""How often things happened, and the text logs: every number the new executors give is recomputed here with plain SQL over the same history (a 60-day generated world).

The frame model is replaced by a table of frames (this file tests what happens AFTER a question has been read); every frame used here also type-checks against the ontology."""
import os as _os; _os.environ.setdefault("VGPT_DRIVING_FIXTURE", "1")                                  # the small derived driving fixture stands in for the real traces when they are absent (tools_testdata.py)
import datetime as dt
import math
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning import glossary as GL, ontology
from vgpt.meaning.arbitrate import arbitrate
from vgpt.meaning.frame import Frame
from vgpt.meaning.pipeline import MeaningBrain
from vgpt.vehicle import lake as LK
from vgpt.world import build as WB

DAY = 86400


class Scripted:
    glossary = GL.load()

    def __init__(self): self.t = {}
    def parse(self, text, hint=None, prev=None):
        f = self.t.get(text)
        return (f, {"raw": f.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}) if f is not None else (None, {"raw": "", "pmin": 0.0, "ms": 0.0})


_STATE = {}


def world():
    """One 60-day world and one brain for the whole module (building it takes ~15 s)."""
    if not _STATE:
        path = os.path.join(tempfile.gettempdir(), f"tx_world_{os.getpid()}_{id(_STATE)}.db")                                  # one file per copy of this module (the records tests import it under another name)
        for e in ("", "-wal", "-shm"):
            try: os.remove(path + e)
            except OSError: pass
        gv = WB.build_world("aster_ev", days=60, seed=5, end="2026-10-18", verbose=False); WB.write_world(gv, path, verbose=False); LK.release(); LK.adopt_path(path)
        m = Scripted(); b = MeaningBrain(path, m); b.now = int(gv.t0 + gv.n) - 3600; _STATE.update(path=path, m=m, b=b, now=b.now, tz=int(b.tz), c=sqlite3.connect(path), onto=ontology.load())
    return _STATE


def tearDownModule():
    LK.release()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for k, v in world().items(): setattr(cls, k, v)
        LK.adopt_path(cls.path)                                         # another module's tearDownModule released the lake: adopt the shared world's vehicle again

    def ask(self, text, frame):
        self.assertEqual(self.onto.validate(frame), [], frame.to_text()); self.m.t[text] = frame; self.b.dialogue.reset(); r = self.b.ask(text); return r, (r.get("facts") or {})

    def ev(self, types, lo, hi): return [r[0] for r in self.c.execute(f"SELECT ts FROM events WHERE type IN ({','.join('?' * len(types))}) AND ts>=? AND ts<? ORDER BY ts", (*types, lo, hi))]


class Frames(Base):
    def test_the_new_vocabulary_is_in_the_ontology_and_types_check(self):
        for cid in ("diagnostic_code_cleared", "software_update", "source_dropout", "service_log", "ecu_reset", "unexpected_shutdown"): self.assertIn(cid, self.onto.concepts)
        for f in (Frame("trend", "ecu_reset", "change"), Frame("trend", "hard_braking", "rise"), Frame("compare", "ecu_reset", "count", "none", "period"), Frame("compare", "adas_disengagement", "count", "highway", "city"),
                  Frame("extreme", "hard_braking", "max", "none", "none", "week"), Frame("events", "adas_disengagement", "count", "none", "none", "day"), Frame("events", "service_log", "list")):
            self.assertEqual(self.onto.validate(f), [], f.to_text())
        self.assertIn("week", self.onto.ents)

    def test_a_distinctive_word_of_the_models_own_concept_stops_arbitration_from_overruling_it(self):
        g = GL.load(); f, note = arbitrate("when was a fault code last cleared", Frame("events", "diagnostic_code_cleared", "last"), self.onto, g); self.assertEqual(f.concept, "diagnostic_code_cleared", note)
        f, note = arbitrate("did the ecu reset last week", Frame("events", "diagnostic_code", "exists"), self.onto, g); self.assertEqual(f.concept, "ecu_reset", "a misread rare word is still corrected")


class Trends(Base):
    def test_the_halves_of_a_window_are_counted_exactly(self):
        r, f = self.ask("is hard braking more frequent over the last 40 days", Frame("trend", "hard_braking", "change")); lo, hi = self.now - 40 * DAY, self.now + 1; mid = lo + (hi - lo) // 2
        n1 = len(self.ev(["HARSH_BRAKE"], lo, mid)); n2 = len(self.ev(["HARSH_BRAKE"], mid, hi)); self.assertEqual((f["first_half"], f["second_half"]), (n1, n2)); self.assertEqual(f["count"], n1 + n2)
        k = min(n1, n2); n = n1 + n2; pm = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]; self.assertAlmostEqual(f["p_value"], min(1.0, sum(x for x in pm if x <= pm[n2] * (1 + 1e-9))), places=9)
        self.assertIn("exploratory", r["answer"]); self.assertIn(f"{n1} in the first half", r["answer"])

    def test_the_week_with_the_largest_rise_is_found(self):
        r, f = self.ask("in which week did hard braking rise most over the last 10 weeks", Frame("trend", "hard_braking", "rise")); lo = self.now - 70 * DAY; hi = self.now + 1
        lo7 = ((lo + self.tz + DAY - 1) // DAY) * DAY - self.tz; weeks = (hi - lo7) // (7 * DAY); cnt = [len(self.ev(["HARSH_BRAKE"], lo7 + i * 7 * DAY, lo7 + (i + 1) * 7 * DAY)) for i in range(weeks)]
        self.assertEqual(f["counts"], cnt); best = max(range(1, weeks), key=lambda i: cnt[i] - cnt[i - 1]); self.assertEqual(f["stretch_start"], lo7 + best * 7 * DAY)

    def test_now_is_not_a_period_of_events(self):
        r, f = self.ask("is hard braking more frequent now than before", Frame("trend", "hard_braking", "change")); lo, hi = self.now - 28 * DAY, self.now + 1
        self.assertEqual(f["count"], len(self.ev(["HARSH_BRAKE"], lo, hi))); self.assertGreater(f["count"], 0)

    def test_no_events_is_said_plainly(self):
        lo, hi = self.now - 20 * DAY, self.now + 1
        for key, cid in (("AEB_ACTIVATION", "aeb_activation"), ("UNEXPECTED_SHUTDOWN", "unexpected_shutdown"), ("TYRE_INFLATED", "tyre_inflation"), ("SERVICE_DUE", "service_reminder")):
            if not self.ev([key], lo, hi): break
        else: self.skipTest("every candidate event happened in the window")
        r, f = self.ask(f"are {cid} events getting more frequent over the last 20 days", Frame("trend", cid, "change")); self.assertEqual(f.get("count"), 0); self.assertIn("No ", r["answer"]); self.assertFalse(f["ok"])


class Comparisons(Base):
    def test_two_calendar_months(self):
        r, f = self.ask("hard braking this month versus last month", Frame("compare", "hard_braking", "count", "none", "period"))
        local = dt.datetime.fromtimestamp(self.now + self.tz, dt.timezone.utc); m0 = dt.datetime(local.year, local.month, 1, tzinfo=dt.timezone.utc); pm = dt.datetime(m0.year - (m0.month == 1), 12 if m0.month == 1 else m0.month - 1, 1, tzinfo=dt.timezone.utc)
        a0, a1 = int(m0.timestamp()) - self.tz, self.now + 1; b0, b1 = int(pm.timestamp()) - self.tz, int(m0.timestamp()) - self.tz
        self.assertEqual(f["counts"], [len(self.ev(["HARSH_BRAKE"], a0, a1)), len(self.ev(["HARSH_BRAKE"], b0, b1))]); self.assertIn("per day", r["answer"])

    def test_highway_against_city_counts_events_and_hours_in_each_condition(self):
        r, f = self.ask("compare adas disengagements on the highway with the city over the last 60 days", Frame("compare", "adas_disengagement", "count", "highway", "city")); lo = self.now - 60 * DAY; hi = self.now + 1
        mode = dict(self.c.execute("SELECT ts, mode FROM sig_1m WHERE ts>=? AND ts<?", (lo, hi)).fetchall()); ts = self.ev(["ADAS_DISENGAGE"], lo, hi)
        na = sum(1 for t in ts if mode.get(t // 60 * 60) == 2); nb = sum(1 for t in ts if mode.get(t // 60 * 60) == 1); ha = sum(1 for v in mode.values() if v == 2) / 60.0; hb = sum(1 for v in mode.values() if v == 1) / 60.0
        self.assertEqual(f["counts"], [na, nb]); self.assertAlmostEqual(f["hours"][0], ha, delta=0.5); self.assertAlmostEqual(f["hours"][1], hb, delta=0.5); self.assertGreater(na, 0)
        self.assertAlmostEqual(f["rates_per_hour"][0], na / f["hours"][0], places=6); self.assertIn("per hour", r["answer"])

    def test_a_comparison_without_two_periods_asks_for_them(self):
        r, f = self.ask("compare hard braking with before", Frame("compare", "hard_braking", "count", "none", "period")); self.assertFalse(f["ok"]); self.assertIn("two periods", r["answer"])


class ByDay(Base):
    def days(self, types, lo, hi):
        d = {}
        for t in self.ev(types, lo, hi): k = (t + self.tz) // DAY; d[k] = d.get(k, 0) + 1
        return d

    def test_the_day_with_the_most_events(self):
        r, f = self.ask("which day had the most hard braking over the last 60 days", Frame("extreme", "hard_braking", "max", "none", "none", "day")); d = self.days(["HARSH_BRAKE"], self.now - 60 * DAY, self.now + 1)
        top = max(d.values()); self.assertEqual(f["count"], top); self.assertEqual(f["bins"], len(d)); day = dt.datetime.fromtimestamp(min(k for k, v in d.items() if v == top) * DAY, dt.timezone.utc); self.assertIn(f"{day.day} {day.strftime('%b')}", r["answer"])

    def test_the_week_with_the_fewest_only_counts_weeks_the_car_was_driven(self):
        r, f = self.ask("which week had the fewest hard braking events over the last 10 weeks", Frame("extreme", "hard_braking", "min", "none", "none", "week")); lo, hi = self.now - 70 * DAY, self.now + 1
        wk = lambda t: ((t + self.tz) // DAY + 3) // 7; driven = {wk(r[0]) for r in self.c.execute("SELECT start FROM trips WHERE start>=? AND start<? AND km>0", (lo, hi))}
        cnt = {}
        for t in self.ev(["HARSH_BRAKE"], lo, hi): cnt[wk(t)] = cnt.get(wk(t), 0) + 1
        self.assertEqual(f["count"], min(cnt.get(k, 0) for k in driven)); self.assertEqual(f["bins"], len(driven))

    def test_on_how_many_days(self):
        r, f = self.ask("on how many days did the adas disengage over the last 60 days", Frame("events", "adas_disengagement", "count", "none", "none", "day")); d = self.days(["ADAS_DISENGAGE"], self.now - 60 * DAY, self.now + 1)
        self.assertEqual(f["bins"], len(d)); self.assertEqual(f["count"], sum(d.values()))


class Logs(Base):
    def like(self, lo, hi, *stems, src=None):
        q = "SELECT COUNT(*) FROM logs WHERE ts>=? AND ts<?" + "".join(" AND msg LIKE ?" for _ in stems) + (" AND src=?" if src else "")
        return self.c.execute(q, [lo, hi] + ["%" + s + "%" for s in stems] + ([src] if src else [])).fetchone()[0]

    def test_a_word_search_matches_a_plain_sql_search(self):
        r, f = self.ask("show me the log messages about the cooling pump over the last 30 days", Frame("events", "service_log", "list")); lo, hi = self.now - 30 * DAY, self.now + 1
        self.assertEqual(f["count"], self.like(lo, hi, "cool", "pump")); self.assertGreater(f["count"], 0); self.assertIn("pump", r["answer"].lower()); self.assertIn("never treated as an instruction", r["answer"])
        self.assertLess(f["distinct"], f["count"], "repeated lines are folded into one entry with a count")

    def test_a_named_source_filters_the_search(self):
        r, f = self.ask("what did the thermal manager log over the last 30 days", Frame("events", "service_log", "count")); lo, hi = self.now - 30 * DAY, self.now + 1
        self.assertEqual(f["sources"], ["qnx-thermal"]); self.assertEqual(f["count"], self.like(lo, hi, src="qnx-thermal")); self.assertEqual(f["terms"], [])

    def test_no_match_says_what_was_searched_and_what_the_log_keeps(self):
        r, f = self.ask("how many log lines mention teleportation over the last 30 days", Frame("events", "service_log", "count")); self.assertFalse(f["ok"]); self.assertIn("teleportation", r["answer"]); self.assertIn("warning and error lines", r["answer"])

    def test_exists_and_last(self):
        r, f = self.ask("any logs about the gateway over the last 60 days", Frame("events", "service_log", "exists")); self.assertGreater(f["count"], 0); self.assertTrue(r["answer"].startswith("Yes"))
        r, f = self.ask("the latest log line from the gateway over the last 60 days", Frame("events", "service_log", "last")); self.assertIn("[can_gw]", r["answer"])

    def test_the_word_that_names_a_source_is_not_also_searched_for(self):
        r, f = self.ask("the latest thing the thermal manager logged over the last 30 days", Frame("events", "service_log", "last")); self.assertEqual(f["sources"], ["qnx-thermal"]); self.assertEqual(f["terms"], []); self.assertGreater(f["count"], 0)

    def test_a_log_line_is_not_a_measurement(self):
        r, f = self.ask("why did the log lines happen", Frame("why", "service_log")); self.assertFalse(f["ok"]); self.assertIn("searched, listed or counted", r["answer"])


class Refusals(Base):
    def test_a_refusal_the_model_is_not_sure_of_is_asked_again_not_accused(self):
        class Unsure:
            def parse(self, text, hint=None, prev=None): return Frame("unsafe", "none"), {"raw": "unsafe none none none none none", "pmin": 0.5, "ms": 0.0, "hint": ""}
        class Sure(Unsure):
            def parse(self, text, hint=None, prev=None): return Frame("unsafe", "none"), {"raw": "unsafe none none none none none", "pmin": 0.97, "ms": 0.0, "hint": ""}
        keep = self.b.frame_model
        try:
            self.b.frame_model = Unsure(); self.b.dialogue.reset(); r = self.b.ask("zxcv qwer unsure refusal"); self.assertNotIn("bypass", r["answer"]); self.assertRegex((r.get("facts") or {}).get("route", "") + str(r.get("intent")) + r["answer"].lower(), "ambiguous|low_confidence")                                            # asked again, either through the unsafe floor or through the calibrated refusal
            self.b.frame_model = Sure(); self.b.dialogue.reset(); r = self.b.ask("zxcv qwer sure refusal"); self.assertIn("bypass", r["answer"])
        finally: self.b.frame_model = keep


class NewEvents(Base):
    def test_a_silent_source_is_an_event_the_assistant_can_count_and_date(self):
        r, f = self.ask("how many times did a source go silent over the last 60 days", Frame("events", "source_dropout", "count")); lo, hi = self.now - 60 * DAY, self.now + 1
        self.assertEqual(f.get("count", None) if "count" in f else int(r["answer"].split(" ")[2]), len(self.ev(["SOURCE_SILENT"], lo, hi))); self.assertGreater(len(self.ev(["SOURCE_SILENT"], 0, hi)), 0)
        r, f = self.ask("when did a source last go silent", Frame("events", "source_dropout", "last")); self.assertIn("no frames for", r["answer"])

    def test_cleared_codes_and_software_updates_are_their_own_events(self):
        r, f = self.ask("when was a fault code last cleared", Frame("events", "diagnostic_code_cleared", "last")); self.assertIn("cleared", r["answer"]); self.assertNotIn("0x09", r["answer"], "a cleared code is not a set code")
        r, f = self.ask("when was the software last updated", Frame("events", "software_update", "last")); self.assertIn("software", r["answer"].lower())

    def test_the_bulk_world_and_the_live_runtime_write_the_same_kind_of_record(self):
        c = sqlite3.connect(self.path); n = c.execute("SELECT COUNT(*) FROM source_log WHERE status='silent'").fetchone()[0]; m = c.execute("SELECT COUNT(*) FROM events WHERE type='SOURCE_SILENT'").fetchone()[0]; self.assertEqual(n, m); self.assertGreater(n, 0)


class States(Base):
    def test_the_time_lane_keeping_was_on_is_the_minutes_with_the_state_on(self):
        lo, hi = self.now - 30 * DAY, self.now + 1
        n = self.c.execute("SELECT COUNT(*) FROM ext_1m e JOIN sig_1m s ON s.ts=e.ts WHERE e.ts>=? AND e.ts<? AND e.adas>50 AND s.mode IN (1,2)", (lo, hi)).fetchone()[0]; self.assertGreater(n, 0)
        import re
        for text, op in (("what share of the time was lane keeping on over the last 30 days", "share"), ("how long was lane keeping engaged over the last 30 days", "duration")):
            r, f = self.ask(text, Frame("stat", "adas_state", op)); got = int(re.search(r"(\d[\d,]*) of", r["answer"]).group(1).replace(",", ""))
            self.assertLessEqual(abs(got - n), max(8, n // 25), f"the engine counted {got} minutes, plain SQL {n}"); self.assertIn("engaged", r["answer"]); self.assertNotIn("above 0.5", r["answer"])


if __name__ == "__main__":
    unittest.main()

"""Answers from the vehicle's own records - the reporting unit of an event, the state of the diagnostic codes, how long a source was silent, where a measurement comes from, the charging ledger - each recomputed
with plain SQL over the same 60-day generated world as tests/test_eventx.py (one build shared by both modules)."""
import os as _os; _os.environ.setdefault("VGPT_DRIVING_FIXTURE", "1")                                  # the small derived driving fixture stands in for the real traces when they are absent (tools_testdata.py)
import json
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_eventx import Base, DAY, Frame, tearDownModule  # noqa: E402,F401
from vgpt.meaning import literals as L  # noqa: E402


class Reporters(Base):
    def dtc(self): return json.loads(self.c.execute("SELECT v FROM meta WHERE k='vehicle.dtc'").fetchone()[0])

    def test_fault_codes_by_subsystem(self):
        r, f = self.ask("how many fault codes were set in each subsystem over the last 60 days", Frame("events", "diagnostic_code", "count", "none", "none", "source")); lo, hi = self.now - 60 * DAY, self.now + 1; t = self.dtc(); want = {}
        for (det,) in self.c.execute("SELECT detail FROM events WHERE type='DTC_SET' AND ts>=? AND ts<?", (lo, hi)): k = t.get(det.split(" ")[0], {}).get("subsystem", "unlisted subsystem"); want[k] = want.get(k, 0) + 1
        self.assertEqual(f["sources"], want); self.assertEqual(f["count"], sum(want.values()))

    def test_the_subsystem_with_the_most_codes(self):
        r, f = self.ask("which subsystem has the most diagnostic codes over the last 60 days", Frame("extreme", "diagnostic_code", "max", "none", "none", "source")); top = max(f["sources"].values()); self.assertEqual(f["count"], top); self.assertIn(f["source"], r["answer"])

    def test_which_unit_reset(self):
        r, f = self.ask("which ecu reset the most over the last 60 days", Frame("extreme", "ecu_reset", "max", "none", "none", "source")); lo, hi = self.now - 60 * DAY, self.now + 1; want = {}
        for (det,) in self.c.execute("SELECT detail FROM events WHERE type='ECU_RESET' AND ts>=? AND ts<?", (lo, hi)): k = det.split(" reset (")[0]; want[k] = want.get(k, 0) + 1
        self.assertEqual(f["sources"], want); self.assertIn("ADAS domain controller", f["sources"])

    def test_which_bus_went_silent(self):
        r, f = self.ask("which bus went silent most over the last 60 days", Frame("extreme", "source_dropout", "max", "none", "none", "source")); self.assertEqual(len(f["sources"]), 1); self.assertIn("Body CAN", f["source"])


class Codes(Base):
    def events(self):
        out = {}
        for ts, ty, det in self.c.execute("SELECT ts, type, detail FROM events WHERE type IN ('DTC_SET','DTC_CLEAR') ORDER BY ts"): out.setdefault(det.split(" ")[0].upper(), []).append((ts, ty))
        return out

    def test_active_cleared_and_recurring_codes(self):
        r, f = self.ask("which diagnostic codes are active and which came back", Frame("health", "diagnostic_status")); ev = self.events()
        active = sorted(k for k, s in ev.items() if s[-1][1] == "DTC_SET"); back = sorted(k for k, s in ev.items() if any(s[i][1] == "DTC_SET" and any(x[1] == "DTC_CLEAR" for x in s[:i]) for i in range(len(s))))
        never = sorted(k for k, s in ev.items() if not any(x[1] == "DTC_CLEAR" for x in s)); self.assertEqual(sorted(f["active"]), active); self.assertEqual(sorted(f["recurring"]), back); self.assertEqual(sorted(f["never_cleared"]), never); self.assertEqual(f["codes"], len(ev))
        self.assertIn("not proof that the fault is gone", r["answer"])

    def test_at_the_moment_means_the_state_reached_so_far(self):
        r, f = self.ask("which diagnostic codes are open at the moment", Frame("health", "diagnostic_status")); ev = self.events()
        self.assertEqual(sorted(f["active"]), sorted(k for k, s in ev.items() if s[-1][1] == "DTC_SET")); self.assertGreater(len(ev), 0); self.assertNotIn("was set or cleared right now", r["answer"])

    def test_a_code_class_letter_filters_the_codes(self):
        ev = self.events(); lo = self.now - 60 * 86400
        r, f = self.ask("any p codes stored over the last 60 days", Frame("events", "diagnostic_code", "exists")); self.assertTrue(f["codes"])
        self.assertTrue(all(k.startswith("P") for k in f["codes"])); self.assertEqual(sorted(f["codes"]), sorted(k for k, s in ev.items() if k.startswith("P") and any(t >= lo for t, _ in s)))
        r, f = self.ask("were there any u codes over the last 60 days", Frame("events", "diagnostic_code", "exists")); self.assertFalse(any(k.startswith("U") for k in ev) and f.get("codes")); self.assertTrue(all(k.startswith("U") for k in f.get("codes", [])) or not f["ok"])               # a world with no U code at all answers without a list of codes

    def test_a_typed_code_has_its_own_history(self):
        ev = self.events(); code = max(ev, key=lambda k: sum(1 for x in ev[k] if x[1] == "DTC_SET"))
        r, f = self.ask(f"how often does {code} occur", Frame("events", "diagnostic_code", "count")); self.assertEqual(f["codes"], [code]); self.assertEqual(f["count"], sum(1 for x in ev[code] if x[1] == "DTC_SET")); self.assertIn(code, r["answer"])
        r, f = self.ask("show the P9999 history", Frame("events", "diagnostic_code", "list")); self.assertEqual(f["codes"], ["P9999"]); self.assertFalse(f["ok"])


class Durations(Base):
    def test_stat_count_is_the_events_count(self):
        r, f = self.ask("how many times did an ecu reset over the last 60 days", Frame("stat", "ecu_reset", "count")); lo, hi = self.now - 60 * DAY, self.now + 1
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM events WHERE type='ECU_RESET' AND ts>=? AND ts<?", (lo, hi)).fetchone()[0], int(r["answer"].split(" ")[2]))

    def test_how_long_a_source_was_silent_in_total(self):
        r, f = self.ask("how long was the body bus down in total over the last 60 days", Frame("stat", "source_dropout", "duration")); lo, hi = self.now - 60 * DAY, self.now + 1; tot = 0; open_at = None
        for ts, st in self.c.execute("SELECT ts, status FROM source_log WHERE source='body' AND ts>=? AND ts<? ORDER BY ts", (lo, hi)):
            if st == "silent": open_at = ts
            elif open_at is not None: tot += ts - open_at; open_at = None
        self.assertEqual(f["seconds"], tot); self.assertGreater(tot, 0); self.assertIn("cause is not recorded", r["answer"])

    def test_a_duration_that_is_not_recorded_is_not_invented(self):
        r, f = self.ask("how long did the ecu resets last", Frame("stat", "ecu_reset", "duration")); self.assertFalse(f["ok"]); self.assertIn("not recorded", r["answer"])


class Provenance(Base):
    def test_where_vehicle_speed_comes_from(self):
        r, f = self.ask("which sources provide vehicle speed", Frame("spec", "speed")); prof = json.loads(self.c.execute("SELECT v FROM meta WHERE k='vehicle.profile'").fetchone()[0]); want = [s["from"] for s in prof["signals"] if s["concept"] == "speed"]
        self.assertEqual(sorted(f["sources"]), sorted(want)); self.assertGreaterEqual(len(want), 2); self.assertIn("primary", r["answer"]); self.assertIn("secondary", r["answer"]); self.assertIn("sensor conflict", r["answer"])

    def test_a_signal_the_vehicle_does_not_publish_is_said_not_to_be_provided(self):
        r, f = self.ask("which sources provide the fuel level", Frame("spec", "fuel_level")) if "fuel_level" in self.onto.concepts else (None, None)
        if r is None: self.skipTest("no fuel_level concept in this vehicle's ontology")
        self.assertIn("no fuel", r["answer"]); self.assertIn("electric car", r["answer"])                                       # an electric car has no fuel signal: the question is answered about the state of charge, and says so (vgpt/meaning/evmap.py)


class Charging(Base):
    def rows(self, where=""):
        return self.c.execute(f"SELECT start, end, kind, kw, kwh FROM charges WHERE 1=1 {where} ORDER BY start").fetchall()

    def test_the_last_fast_charge(self):
        r, f = self.ask("how many kwh did the last dc fast charge add", Frame("events", "charging_session", "last", "fast_charging")); dc = self.rows("AND kind='DC'"); last = dc[-1]
        self.assertEqual(f["start"], last[0]); self.assertAlmostEqual(f["kwh"], last[4]); self.assertIn("DC charge", r["answer"]); self.assertIn(f"{last[4]:.1f} kWh", r["answer"])

    def test_how_often_do_i_fast_charge(self):
        r, f = self.ask("how often do i fast charge over the last 60 days", Frame("events", "charging_session", "count", "fast_charging")); lo, hi = self.now - 60 * DAY, self.now + 1
        n = self.c.execute("SELECT COUNT(*) FROM charges WHERE kind='DC' AND start>=? AND start<?", (lo, hi)).fetchone()[0]; self.assertEqual(f["count"], n); self.assertAlmostEqual(f["per_week"], n / ((hi - lo) / (7 * DAY)), places=6)

    def test_where_i_charge_most(self):
        r, f = self.ask("which charger location did i use most over the last 60 days", Frame("extreme", "charging_session", "max", "none", "none", "place")); self.assertEqual(f["count"], max(f["places"].values())); self.assertIn(f["place"], r["answer"])
        self.assertEqual(sum(f["places"].values()), self.c.execute("SELECT COUNT(*) FROM charges WHERE start>=? AND start<?", (self.now - 60 * DAY, self.now + 1)).fetchone()[0])


class Tyres(Base):
    def test_the_tyre_pressure_now_names_every_wheel(self):
        r, f = self.ask("what is the tyre pressure right now", Frame("now", "tyre_pressure"))
        self.assertEqual(len(f["wheels"]), 4); self.assertIn("front left", r["answer"]); self.assertIn("rear right", r["answer"]); self.assertIn("Lowest:", r["answer"])
        r, f = self.ask("what is the pressure of the front left tyre right now", Frame("now", "tyre_pressure")); self.assertNotIn("wheels", f)


class Words(unittest.TestCase):
    def test_comparator_words_for_distances_and_sizes(self):
        for text, cmp in (("how often was the lead vehicle closer than 10 metres", "lt"), ("how often was the gap farther than 50 metres", "gt"), ("trips shorter than 5 km", "lt"), ("days hotter than 30 degrees", "gt")):
            self.assertEqual(L.extract(text, 1_790_000_000, 0).cmp, cmp, text)


if __name__ == "__main__":
    unittest.main()

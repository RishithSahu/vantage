"""The fleet summary under an answer is reused only while the history it was made from has not changed (keyed by what the history contains), never by the clock."""
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vgpt import rolelens, rolepages


KPI = {"history_to": "x", "km": 1, "trips": 1, "driving_h": 1, "utilisation_pct": 1, "avg_trip_km": 1, "warnings": 0, "critical": 0, "active_codes": 0, "energy_stops": 0}


class FleetCache(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "v.db"); c = sqlite3.connect(self.path)
        c.executescript("CREATE TABLE sig_1m(ts INTEGER); CREATE TABLE trips(id INTEGER PRIMARY KEY, start INTEGER); CREATE TABLE events(ts INTEGER, type TEXT); INSERT INTO sig_1m VALUES(100); INSERT INTO trips(start) VALUES(100); INSERT INTO events VALUES(100, 'A');"); c.commit(); c.close(); rolelens._CACHE.clear()

    def add(self, sql):
        c = sqlite3.connect(self.path); c.execute(sql); c.commit(); c.close()

    def test_the_version_is_what_the_history_contains(self):
        v0 = rolelens._data_version(self.path); self.assertEqual(v0, (100, 1, 1)); self.add("INSERT INTO sig_1m VALUES(160)"); self.assertEqual(rolelens._data_version(self.path)[0], 160)
        self.assertIsNone(rolelens._data_version(os.path.join(os.path.dirname(self.path), "none.db")))

    def test_the_summary_is_recomputed_when_the_history_grew_and_only_then(self):
        calls = []
        with mock.patch.object(rolepages, "vehicle_kpis", side_effect=lambda p: calls.append(p) or dict(KPI)):
            rolelens.fleet_lines({}, self.path); rolelens.fleet_lines({}, self.path); self.assertEqual(len(calls), 1, "unchanged history: one computation, however long it takes")
            self.add("INSERT INTO events VALUES(200, 'B')"); rolelens.fleet_lines({}, self.path); self.assertEqual(len(calls), 2)
            self.add("INSERT INTO trips(start) VALUES(300)"); rolelens.fleet_lines({}, self.path); self.assertEqual(len(calls), 3)
            self.add("INSERT INTO sig_1m VALUES(220)"); rolelens.fleet_lines({}, self.path); self.assertEqual(len(calls), 4)

    def test_a_history_whose_version_cannot_be_read_is_never_reused(self):
        calls = []
        with mock.patch.object(rolepages, "vehicle_kpis", side_effect=lambda p: calls.append(p) or dict(KPI)), mock.patch.object(rolelens, "_data_version", return_value=None):
            rolelens.fleet_lines({}, self.path); rolelens.fleet_lines({}, self.path); self.assertEqual(len(calls), 2)

    def test_the_cache_has_no_time_to_live(self):
        self.assertNotIn("monotonic", open(os.path.join(ROOT, "vgpt", "rolelens.py"), encoding="utf-8").read())


if __name__ == "__main__": unittest.main()

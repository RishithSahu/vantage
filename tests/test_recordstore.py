"""Records a person chose to leave (feedback, a study line): sealed with the installation's key where the platform can, removed after the retention period, deletable at once, and tolerant of damaged lines."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import recordstore as RS


class Clock:
    def __init__(self, t=1_800_000_000): self.t = t
    def __call__(self): return self.t


class Store(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(); self._home = os.environ.get("VEHICLEGPT_HOME"); os.environ["VEHICLEGPT_HOME"] = self.dir; self.path = os.path.join(self.dir, "fb.jsonl"); self.clock = Clock()

    def tearDown(self):
        if self._home is None: os.environ.pop("VEHICLEGPT_HOME", None)
        else: os.environ["VEHICLEGPT_HOME"] = self._home

    def store(self, **k): return RS.RecordStore(self.path, clock=self.clock, **k)

    @unittest.skipUnless(RS.HAVE_AES, "needs the cryptography package")
    def test_a_record_is_sealed_on_the_disk_and_read_back(self):
        s = self.store(); self.assertTrue(s.append({"rating": "down", "q": "how hot is the battery", "note": "wrong unit"}))
        raw = open(self.path, encoding="utf-8").read()
        self.assertNotIn("battery", raw); self.assertNotIn("wrong unit", raw); self.assertIn('"c"', raw)
        got = s.read_all(); self.assertEqual(len(got), 1); self.assertEqual(got[0]["q"], "how hot is the battery"); self.assertEqual(got[0]["t"], self.clock.t)
        self.assertTrue(s.status()["encrypted"])

    def test_the_plain_mode_writes_readable_lines_and_says_so(self):
        s = self.store(plain=True); s.append({"rating": "up", "q": "hello"}); self.assertIn("hello", open(self.path, encoding="utf-8").read()); self.assertFalse(s.status()["encrypted"])
        self.assertEqual(s.read_all()[0]["q"], "hello")

    def test_a_plain_line_of_an_earlier_version_is_still_read(self):
        with open(self.path, "w", encoding="utf-8") as f: f.write(json.dumps({"t": self.clock.t, "rating": "up", "q": "old"}) + "\n")
        self.assertEqual(self.store().read_all()[0]["q"], "old")

    def test_records_older_than_the_retention_are_removed(self):
        s = self.store(retention_days=30); s.append({"q": "old", "t": self.clock.t - 31 * 86400}); s.append({"q": "fresh", "t": self.clock.t - 86400})
        self.assertEqual([r["q"] for r in s.read_all()], ["fresh"]); self.assertEqual(len(open(self.path, encoding="utf-8").read().strip().split("\n")), 1, "the old line is gone from the file, not only hidden")

    def test_the_purge_runs_while_appending_at_most_once_an_hour(self):
        s = self.store(retention_days=1); s.append({"q": "a", "t": self.clock.t})
        with open(self.path, "a", encoding="utf-8") as f: f.write(s._seal({"q": "stale", "t": self.clock.t - 10 * 86400}) + chr(10))
        on_disk = lambda: [(s._open(l.strip()) or {}).get("q") for l in open(self.path, encoding="utf-8") if l.strip()]
        s.append({"q": "b"}); self.assertIn("stale", on_disk())                                                      # within the hour of the last purge: not purged yet
        self.clock.t += 3601; s.append({"q": "c"}); self.assertNotIn("stale", on_disk())

    def test_delete_all_removes_every_file(self):
        s = self.store(max_bytes=50); [s.append({"q": "x" * 40, "i": i}) for i in range(6)]; self.assertTrue(os.path.exists(self.path + ".1"))
        n = s.delete_all(); self.assertGreaterEqual(n, 2); self.assertFalse(os.path.exists(self.path)); self.assertFalse(os.path.exists(self.path + ".1")); self.assertEqual(s.read_all(), [])

    def test_a_damaged_line_is_not_a_record_and_does_not_stop_the_others(self):
        s = self.store(); s.append({"q": "good"})
        with open(self.path, "a", encoding="utf-8") as f: f.write("not json at all\n{\"v\":1,\"n\":\"AAAA\",\"c\":\"AAAA\"}\n")
        self.assertEqual([r["q"] for r in s.read_all()], ["good"])

    def test_the_two_stores_of_the_app_live_in_the_private_folder_with_their_own_retention(self):
        fb, st = RS.feedback_store(), RS.study_store(); self.assertEqual(os.path.dirname(fb.path), self.dir); self.assertEqual(os.path.dirname(st.path), self.dir)
        self.assertEqual(fb.retention_days, 30.0); self.assertLessEqual(st.retention_days, 14.0)

    def test_the_reader_for_tools_decrypts_and_applies_no_retention(self):
        s = self.store(retention_days=1e9); s.append({"q": "ancient", "t": 1}); self.assertEqual(RS.read_file(self.path)[0]["q"], "ancient")


if __name__ == "__main__": unittest.main()

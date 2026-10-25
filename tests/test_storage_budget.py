"""The history keeps within its size budget: the oldest archive files go first, seconds before minutes, and the database itself is never touched."""
import os, shutil, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from vgpt import archive, storage_budget as SB, store
from test_archive import DAY, T0, make_db


class Budget(unittest.TestCase):
    def test_usage_adds_the_database_and_the_archive(self):
        path, rows, c = make_db(days=10, fine_days=3); store.compact(c, T0 + 10 * DAY); u = SB.usage(c)
        self.assertGreater(u["archive"], 0); self.assertEqual(u["total"], u["database"] + u["archive"]); self.assertGreater(u["archive_files"], 0); c.close(); shutil.rmtree(os.path.dirname(path), True)

    def test_a_budget_that_is_too_small_removes_the_oldest_files_first_and_leaves_the_database_alone(self):
        path, rows, c = make_db(days=10, fine_days=3); store.compact(c, T0 + 10 * DAY); c.commit()
        before = SB.usage(c); live = c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0]; first = c.execute("SELECT MIN(lo) FROM archive_idx").fetchone()[0]
        c.execute("INSERT OR REPLACE INTO meta VALUES('ret.budget_gb',?)", (str((before["database"] + before["archive"] // 2) / 1e9),)); c.commit()
        out = SB.enforce(c, T0 + 10 * DAY); c.commit(); after = SB.usage(c)
        self.assertGreater(out["removed"], 0); self.assertLessEqual(after["archive"] + before["database"], out["budget"])          # the database file itself is not part of what is removed; self.assertGreater(c.execute("SELECT COALESCE(MIN(lo), 0) FROM archive_idx").fetchone()[0], first)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0], live); ok, bad = archive.verify(c)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM archive_idx").fetchone()[0], before["archive_files"] - out["removed"]); c.close(); shutil.rmtree(os.path.dirname(path), True)

    def test_a_history_within_its_budget_is_left_alone(self):
        path, rows, c = make_db(days=6, fine_days=2); store.compact(c, T0 + 6 * DAY); self.assertEqual(SB.enforce(c, T0 + 6 * DAY)["removed"], 0); c.close(); shutil.rmtree(os.path.dirname(path), True)

    def test_the_report_projects_five_years(self):
        path, rows, c = make_db(days=10, fine_days=3); store.compact(c, T0 + 10 * DAY); r = SB.report(c); self.assertGreater(r["projected_total"], r["total"]); self.assertEqual(r["years"], 5); c.close(); shutil.rmtree(os.path.dirname(path), True)


if __name__ == "__main__":
    unittest.main()

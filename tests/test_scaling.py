"""Scalability seams: resource tiers, the model pool, per-person sessions.  Timing-based assertions use generous margins (they check that work overlaps, not how fast a machine is)."""
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import sessions as S, store, tiers


class FakeModel:
    glossary = object()

    def __init__(self, delay=0.15): self.delay = delay; self.thread = None

    def parse(self, text, hint=None):
        time.sleep(self.delay); self.thread = threading.current_thread().name; return ("frame:" + text, {"raw": text})


class FakeBrain:
    """What a session needs from a brain: a model-backed ask() with its own dialogue state."""
    def __init__(self, db, slm=None): self.slm = slm; self.history = []; self.now = 0; self.persona = "driver"; self.c = sqlite3.connect(":memory:")

    def ask(self, text):
        frame, info = self.slm.parse(text); self.history.append(text); return {"answer": f"{frame} (turn {len(self.history)})", "turns": list(self.history)}


class Tiers(unittest.TestCase):
    def test_every_tier_declares_the_same_knobs(self):
        names = tiers.names(); self.assertEqual(set(names), {"edge2g", "pi5", "vm8", "hpc"})
        for n in names:
            t = tiers.get(n); self.assertGreater(t.cores, 0); self.assertGreater(t.ram_gb, 0)
            for k in ("base", "threads", "workers", "n_ctx"): self.assertIn(k, t.model, n)
            for k in ("fine_days", "coarse_days", "log_days", "context_days"): self.assertIn(k, t.retention, n)
            self.assertIn("cache_kb", t.sqlite); self.assertGreaterEqual(t.sessions, 1)

    def test_a_bigger_tier_never_has_less_of_anything(self):
        order = ["edge2g", "pi5", "vm8", "hpc"]
        for a, b in zip(order, order[1:]):
            ta, tb = tiers.get(a), tiers.get(b)
            self.assertLessEqual(ta.sessions, tb.sessions); self.assertLessEqual(ta.sqlite["cache_kb"], tb.sqlite["cache_kb"]); self.assertLessEqual(ta.model["workers"], tb.model["workers"])
            for k in ta.retention: self.assertLessEqual(ta.retention[k], tb.retention[k], (a, b, k))

    def test_the_reference_board_keeps_at_least_ten_weeks_and_fits_the_stated_budget(self):
        t = tiers.get("edge2g"); self.assertGreaterEqual(t.retention["coarse_days"], 70); self.assertEqual((t.cores, t.ram_gb), (2, 2)); self.assertEqual(t.model["workers"], 1)

    def test_retention_is_raised_never_lowered(self):
        own = {"fine_days": 7, "coarse_days": 365, "log_days": 1825, "context_days": 1825, "archive_days": 365, "records_days": 1825}
        self.assertEqual(tiers.retention_for(own, "edge2g"), own); hpc = tiers.retention_for(own, "hpc"); self.assertEqual(hpc["coarse_days"], 730)
        long = dict(own, coarse_days=1000); self.assertEqual(tiers.retention_for(long, "hpc")["coarse_days"], 1000)

    def test_unknown_tiers_are_rejected(self):
        with self.assertRaises(KeyError): tiers.get("quantum")

    def test_the_tier_sets_the_page_cache_of_every_database_handle(self):
        path = os.path.join(tempfile.gettempdir(), "ts_tier.db"); old = os.environ.get("VGPT_TIER")
        try:
            os.environ.pop("VGPT_TIER", None); c = store.open_rw(path); self.assertEqual(c.execute("PRAGMA cache_size").fetchone()[0], -4096); c.close()
            os.environ["VGPT_TIER"] = "vm8"; c = store.open_rw(path); self.assertEqual(c.execute("PRAGMA cache_size").fetchone()[0], -65536); self.assertEqual(c.execute("PRAGMA mmap_size").fetchone()[0], 512 * 2 ** 20); c.close()
            r = store.open_ro(path); self.assertEqual(r.execute("PRAGMA cache_size").fetchone()[0], -65536); r.close()
        finally:
            if old is None: os.environ.pop("VGPT_TIER", None)
            else: os.environ["VGPT_TIER"] = old

    def test_a_runtime_on_a_bigger_tier_writes_its_retention_into_the_database(self):
        from vgpt.live.runtime import Runtime
        from vgpt.vehicle import provision as PV
        db = os.path.join(tempfile.gettempdir(), "ts_rt.db"); PV.provision("aster_ev", db, replace=True); rt = Runtime(db, profile="aster_ev", tier="hpc"); rt.close()
        meta = dict(sqlite3.connect(db).execute("SELECT k, v FROM meta WHERE k LIKE 'ret.%'").fetchall()); self.assertEqual(meta["ret.coarse_days"], "1830"); self.assertEqual(meta["ret.fine_days"], "60")                 # the car's own five years stand; the bigger tier adds seconds
        rt = Runtime(db, profile="aster_ev", tier="edge2g"); rt.close(); meta = dict(sqlite3.connect(db).execute("SELECT k, v FROM meta WHERE k LIKE 'ret.%'").fetchall()); self.assertEqual(meta["ret.coarse_days"], "1830", "a small tier never shortens the car's own five years")


class Pool(unittest.TestCase):
    def test_several_model_contexts_serve_questions_at_the_same_time(self):
        for workers, expect_max in ((1, 10.0), (2, 0.75), (4, 0.45)):
            pool = S.ModelPool(lambda: FakeModel(0.2), workers=workers); out = []; t0 = time.perf_counter()
            ths = [threading.Thread(target=lambda i=i: out.append(pool.parse(f"q{i}"))) for i in range(4)]
            for t in ths: t.start()
            for t in ths: t.join()
            wall = time.perf_counter() - t0; self.assertEqual(len(out), 4); self.assertLess(wall, expect_max + 0.2, f"{workers} workers took {wall:.2f}s")
            if workers == 1: self.assertGreater(wall, 0.75, "one context serves the four questions one after the other")
            self.assertEqual(pool.stats()["parses"], 4); pool.close()

    def test_the_pool_looks_like_one_model(self):
        pool = S.ModelPool(lambda: FakeModel(0.0), workers=2); frame, info = pool.parse("hello"); self.assertEqual(frame, "frame:hello"); self.assertIsNotNone(pool.glossary); pool.close()

    def test_a_model_that_fails_to_load_raises_at_start(self):
        def bad(): raise FileNotFoundError("no model file")
        with self.assertRaises(FileNotFoundError): S.ModelPool(bad, workers=2)

    def test_a_failing_parse_reaches_the_caller_and_the_worker_survives(self):
        class Flaky(FakeModel):
            def parse(self, text, hint=None):
                if text == "boom": raise RuntimeError("bad input")
                return super().parse(text, hint)
        pool = S.ModelPool(lambda: Flaky(0.0), workers=1)
        with self.assertRaises(RuntimeError): pool.parse("boom")
        self.assertEqual(pool.parse("fine")[0], "frame:fine"); pool.close()


class Hub(unittest.TestCase):
    def hub(self, workers=2, max_sessions=3):
        pool = S.ModelPool(lambda: FakeModel(0.05), workers=workers); return S.SessionHub("unused.db", pool, lambda db, slm: FakeBrain(db, slm), max_sessions=max_sessions), pool

    def test_each_person_has_their_own_dialogue(self):
        hub, pool = self.hub(); a1 = hub.ask("alice", "one"); b1 = hub.ask("bob", "uno"); a2 = hub.ask("alice", "two")
        self.assertEqual(a1["turns"], ["one"]); self.assertEqual(b1["turns"], ["uno"]); self.assertEqual(a2["turns"], ["one", "two"], "alice's second question follows HER first one"); hub.close(); pool.close()

    def test_sessions_overlap_and_share_the_model_pool(self):
        hub, pool = self.hub(workers=3, max_sessions=6); [hub.ask(f"s{i}", "warm") for i in range(6)]; t0 = time.perf_counter(); out = []
        ths = [threading.Thread(target=lambda i=i: out.append(hub.ask(f"s{i}", "q"))) for i in range(6)]
        for t in ths: t.start()
        for t in ths: t.join()
        wall = time.perf_counter() - t0; self.assertEqual(len(out), 6); self.assertLess(wall, 0.6, "six questions on three contexts overlap; serial would be 0.30 s plus overhead - allow slack")
        self.assertEqual(sum(pool.stats()["busy_s"]) > 0, True); hub.close(); pool.close()

    def test_a_full_hub_reuses_an_idle_slot_and_refuses_when_everyone_is_busy(self):
        hub, pool = self.hub(workers=1, max_sessions=2); hub.ask("a", "x"); hub.ask("b", "x"); hub.ask("c", "x")
        self.assertEqual(hub.stats()["sessions"], 2, "the least recently used idle session gave up its slot")
        class Slow(FakeModel): pass
        pool2 = S.ModelPool(lambda: FakeModel(0.5), workers=1); hub2 = S.SessionHub("unused.db", pool2, lambda db, slm: FakeBrain(db, slm), max_sessions=1); hub2.ask("a", "warm")
        t = threading.Thread(target=lambda: hub2.ask("a", "long")); t.start(); time.sleep(0.15)
        with self.assertRaises(S.Busy): hub2.ask("b", "x")
        t.join(); hub.close(); hub2.close(); pool.close(); pool2.close()

    def test_a_brain_that_cannot_be_built_does_not_leave_a_dead_session_behind(self):
        pool = S.ModelPool(lambda: FakeModel(0.0), workers=1)
        def broken(db, slm): raise OSError("database missing")
        hub = S.SessionHub("unused.db", pool, broken, max_sessions=2)
        with self.assertRaises(OSError): hub.ask("a", "x")
        self.assertEqual(hub.stats()["sessions"], 0); hub.close(); pool.close()


if __name__ == "__main__":
    unittest.main()

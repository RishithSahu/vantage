"""The process pool: each person stays on one worker, the driver goes first, a dead worker costs one retry and nothing else, and the controller can grow and shrink it."""
import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.scale import Hardware, make_plan                              # noqa: E402
from vgpt.scale.autoscale import Autoscaler, Refused                    # noqa: E402
from vgpt.scale.procpool import ProcessPool                             # noqa: E402


def cfg(delay=0.1): return {"fake": delay, "root": ROOT, "start_timeout": 60}


class Pool(unittest.TestCase):
    def test_a_person_stays_on_one_worker_and_keeps_their_dialogue(self):
        pool = ProcessPool(cfg(0.02), workers=3, max_workers=3)
        try:
            a = [pool.ask("alice", f"q{i}") for i in range(5)]; b = [pool.ask("bob", f"q{i}") for i in range(5)]
            self.assertEqual({x["pid"] for x in a}, {a[0]["pid"]}); self.assertEqual([x["turn"] for x in a], [1, 2, 3, 4, 5]); self.assertEqual([x["turn"] for x in b], [1, 2, 3, 4, 5])
            self.assertNotEqual(a[0]["pid"], b[0]["pid"], "a new person goes to the idle worker with the fewest people")
        finally: pool.close()

    def test_workers_are_separate_processes_that_answer_at_the_same_time(self):
        pool = ProcessPool(cfg(0.3), workers=4, max_workers=4)
        try:
            out = []; t0 = time.perf_counter()
            ths = [threading.Thread(target=lambda i=i: out.append(pool.ask(f"u{i}", "x"))) for i in range(4)]; [t.start() for t in ths]; [t.join() for t in ths]; wall = time.perf_counter() - t0
            self.assertEqual(len({o["pid"] for o in out}), 4); self.assertLess(wall, 0.3 * 2.2, f"four questions on four workers took {wall:.2f}s (one worker would need 1.2 s)")
            self.assertNotIn(os.getpid(), {o["pid"] for o in out})
        finally: pool.close()

    def test_the_driver_is_served_before_a_batch_job_that_asked_earlier(self):
        pool = ProcessPool(cfg(0.25), workers=1, max_workers=1); order = []
        try:
            def ask(s, persona): pool.ask(s, s, persona=persona); order.append(s)
            first = threading.Thread(target=ask, args=("first", "driver")); first.start(); time.sleep(0.1); ths = []
            for s, persona in (("batch", "validation"), ("fleet", "fleet"), ("service", "service"), ("driver", "driver")):
                t = threading.Thread(target=ask, args=(s, persona)); t.start(); ths.append(t); time.sleep(0.03)
            first.join(); [t.join() for t in ths]; self.assertEqual(order[:3], ["first", "driver", "service"]); self.assertEqual(set(order[3:]), {"fleet", "batch"})
        finally: pool.close()

    def test_a_worker_that_dies_is_replaced_and_the_other_people_never_notice(self):
        pool = ProcessPool(cfg(0.05), workers=2, max_workers=2)
        try:
            for p in ("alice", "bob"): pool.ask(p, "hi")
            pool.kill_worker(0); time.sleep(0.5)
            ok = [pool.ask(f"p{i}", "after") for i in range(4)]; self.assertEqual(len(ok), 4)
            deadline = time.monotonic() + 30
            while pool.contexts() < 2 and time.monotonic() < deadline: time.sleep(0.2)
            self.assertEqual(pool.contexts(), 2, "a replacement was started"); self.assertGreaterEqual(pool.restarts, 1)
        finally: pool.close()

    def test_a_question_that_kills_its_worker_is_retried_once_and_then_reported_never_hangs(self):
        pool = ProcessPool(cfg(0.05), workers=2, max_workers=2)
        try:
            pool.ask("alice", "warm"); box = []
            def go():
                try: box.append(pool.ask("alice", "die", timeout=60))
                except Exception as e: box.append(e)
            t = threading.Thread(target=go); t.start(); t.join(60)
            self.assertFalse(t.is_alive(), "the caller gets an answer or an error, never a hang"); self.assertIsInstance(box[0], RuntimeError)
            self.assertEqual(pool.ask("bob", "still alive")["answer"], "still alive", "and the pool still works")
        finally: pool.close()

    def test_the_controller_can_grow_and_shrink_the_pool(self):
        pool = ProcessPool(cfg(0.05), workers=1, min_workers=1, max_workers=3)
        try:
            self.assertTrue(pool.add_context()); deadline = time.monotonic() + 30
            while pool.metrics()["ready"] < 2 and time.monotonic() < deadline: time.sleep(0.2)
            self.assertEqual(pool.metrics()["ready"], 2); self.assertTrue(pool.add_context())
            while pool.metrics()["ready"] < 3 and time.monotonic() < deadline: time.sleep(0.2)
            self.assertEqual(pool.contexts(), 3); self.assertFalse(pool.add_context(), "never above the maximum")
            self.assertTrue(pool.remove_context()); self.assertTrue(pool.remove_context()); time.sleep(0.5); self.assertEqual(pool.contexts(), 1); self.assertFalse(pool.remove_context(), "never below the minimum")
            self.assertEqual(pool.ask("z", "still works")["turn"], 1)
        finally: pool.close()

    def test_a_busy_worker_can_be_retired_it_finishes_the_question_in_hand_first(self):
        pool = ProcessPool(cfg(0.6), workers=2, min_workers=1, max_workers=2); res = []
        try:
            ts = [threading.Thread(target=lambda k=k: res.append(pool.ask(f"p{k}", "q")["answer"])) for k in range(4)]; [t.start() for t in ts]; time.sleep(0.25)
            self.assertEqual(pool.metrics()["in_flight"], 2, "both workers are answering: nobody is idle")
            self.assertTrue(pool.remove_context(), "a busy worker is retired anyway"); self.assertEqual(pool.contexts(), 1)
            [t.join(30) for t in ts]; self.assertEqual(len(res), 4, "no question is lost"); time.sleep(1.0); self.assertEqual(len(pool.workers), 1)
        finally: pool.close()

    def test_closed_loop_the_controller_adds_workers_under_load_and_refuses_batch_at_the_limit(self):
        pool = ProcessPool(cfg(0.2), workers=1, min_workers=1, max_workers=3); plan = make_plan(Hardware(cores=2, mem_mb=2048)); ctl = Autoscaler(pool, plan, target_s=0.8, interval_s=0.2, calm_steps=5, cooldown_s=0.3).start()
        refused = []; driver = []; stop = time.monotonic() + 9.0

        def bulk(k):
            while time.monotonic() < stop:
                try: pool.ask(f"b{k}", "batch", persona="validation")
                except Refused: refused.append(1); time.sleep(0.05)

        def drv():
            while time.monotonic() < stop: t = time.perf_counter(); pool.ask("driver", "drive", persona="driver"); driver.append(time.perf_counter() - t); time.sleep(0.15)
        try:
            ths = [threading.Thread(target=bulk, args=(k,)) for k in range(10)] + [threading.Thread(target=drv)]; [t.start() for t in ths]; [t.join() for t in ths]
            self.assertEqual(pool.contexts(), 3, ctl.log); self.assertTrue(refused, "the ladder reached refusing non-driver questions"); late = sorted(driver[len(driver) // 2:]); self.assertLess(late[int(len(late) * 0.9)], 0.9)
        finally: ctl.stop(); pool.close()


if __name__ == "__main__":
    unittest.main()

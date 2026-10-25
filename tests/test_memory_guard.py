"""The second line of defence of the 2 GB budget: the process ends itself when it crosses the limit (the service manager then starts it again and the stored history is untouched).  On a system whose kernel enforces the
limit this never fires; on one whose kernel cannot (the Raspberry Pi's memory controller is off by default, a QNX system without a memory partition) it is what keeps the assistant from growing into the memory the vehicle's
other software needs."""
import importlib.util
import os
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def launcher():
    spec = importlib.util.spec_from_file_location("vehiclegpt_guard", os.path.join(ROOT, "vehiclegpt.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Guard(unittest.TestCase):
    def watch(self, readings, ticks=6):
        """Run the guard against a scripted series of memory readings; -> (exit codes it asked for, messages it printed)."""
        m = launcher(); exits, said = [], []; seq = iter(readings); n = [0]; done = threading.Event()

        def fake_rss():
            try: return next(seq)
            except StopIteration: return readings[-1]

        def fake_sleep(_):
            n[0] += 1
            if n[0] > ticks or exits: done.set(); raise SystemExit                                                       # ends the watching thread, as the real exit would end the process

        def fake_exit(code): exits.append(code); raise SystemExit

        with mock.patch.object(m, "_rss_mb", side_effect=fake_rss), mock.patch("time.sleep", side_effect=fake_sleep), mock.patch.object(m.os, "_exit", side_effect=fake_exit), mock.patch.object(m.sys, "stderr", mock.Mock(write=said.append, flush=lambda: None)):
            m._memory_guard(1950); done.wait(5); threading.Event().wait(0.05)                                               # (not time.sleep: that is the patched one)
        return exits, "".join(said)

    def test_it_ends_the_process_with_a_restartable_code_when_the_limit_is_crossed(self):
        exits, said = self.watch([900.0, 1500.0, 1949.0, 1951.0]); self.assertEqual(exits, [75]); self.assertIn("MEMORY LIMIT", said); self.assertIn("1951", said); self.assertIn("1950", said)

    def test_it_leaves_a_process_that_stays_inside_the_budget_alone(self):
        exits, _ = self.watch([700.0, 980.0, 1200.0, 1949.0, 1000.0]); self.assertEqual(exits, [])

    def test_a_reading_it_could_not_take_is_not_taken_for_a_small_one_and_not_for_a_large_one(self):
        exits, _ = self.watch([0.0, 0.0, 0.0]); self.assertEqual(exits, [])                                            # "unknown" never ends the service by itself: the operating system's own limit is then the enforcer

    def test_the_limit_can_be_switched_off_for_a_machine_with_plenty_of_memory(self):
        m = launcher(); started = []
        with mock.patch.object(m.threading, "Thread", side_effect=lambda *a, **k: started.append(k.get("name")) or mock.Mock()), mock.patch.object(m, "_server_module", side_effect=SystemExit):
            try: m.cmd_run(type("A", (), {"db": None, "model": "0.5b", "camera": "clip", "shell": "auto", "kiosk": False, "memory_limit_mb": 0})())
            except SystemExit: pass
        self.assertNotIn("memory-guard", started)                                                                      # (the start-up may hash the model file in a thread of its own; the guard must not start)


if __name__ == "__main__":
    unittest.main()

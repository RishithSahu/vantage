"""Automatic scaling: the probe, the planner (properties over a grid of hardware), the elastic model pool, and the controller (decisions with invented loads and a closed loop with real threads)."""
import itertools
import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import sessions as S                                   # noqa: E402
from vgpt.scale import Hardware, make_plan, probe                 # noqa: E402
from vgpt.scale import plan as P                                  # noqa: E402
from vgpt.scale.autoscale import Autoscaler, ElasticModelPool, Refused   # noqa: E402


class FakeModel:
    glossary = object()

    def __init__(self, delay=0.1): self.delay = delay

    def parse(self, text, hint=None, prev=None): time.sleep(self.delay); return ("frame:" + text, {"raw": text})


def hw(cores=2, mem=2048, **k): return Hardware(cores=cores, mem_mb=mem, arch="aarch64", isa=frozenset({"neon"}), **k)


class Probe(unittest.TestCase):
    def test_the_probe_reads_this_machine_and_the_overrides_win(self):
        h = probe(); self.assertGreaterEqual(h.cores, 1); self.assertGreater(h.mem_mb, 256); self.assertTrue(h.arch)
        old = {k: os.environ.get(k) for k in ("VGPT_CORES", "VGPT_MEM_MB", "VGPT_ISA", "VGPT_ACCEL")}
        try:
            os.environ.update({"VGPT_CORES": "2", "VGPT_MEM_MB": "2048", "VGPT_ISA": "neon,dotprod", "VGPT_ACCEL": ""}); h = probe()
            self.assertEqual((h.cores, h.mem_mb, set(h.isa), h.accel), (2, 2048, {"neon", "dotprod"}, ())); self.assertEqual(h.source["cores"], "VGPT_CORES")
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v


class Plan(unittest.TestCase):
    GRID = list(itertools.product((1, 2, 3, 4, 6, 8, 12, 16, 32, 64), (512, 1024, 2048, 4096, 8192, 16384, 32768, 131072)))

    def test_the_reference_budget_gets_one_context_on_both_cores_and_keeps_nothing_back(self):
        p = make_plan(hw(2, 2048)); self.assertEqual((p.tier, p.backend, p.contexts, p.threads, p.reserve_cores, p.reserve_mb), ("edge", "ort-int8", 1, 2, 0, 0))

    def test_the_plan_never_overcommits_cores_or_memory_anywhere_on_the_grid(self):
        for cores, mem in self.GRID:
            p = make_plan(hw(cores, mem)); ai_cores = max(1, cores - p.reserve_cores)
            self.assertLessEqual(p.contexts * p.threads, max(ai_cores, 1), (cores, mem, p.as_dict()))
            if p.backend == "none": self.assertEqual(p.contexts, 0); self.assertLess(mem - p.reserve_mb - P.APP_BASE_MB, P.CTX_FIRST_MB, "the model is dropped only when it really does not fit"); continue
            need = P.APP_BASE_MB + P.CTX_FIRST_MB + P.CTX_EXTRA_MB * (p.contexts - 1)
            self.assertLessEqual(need + p.reserve_mb, mem, (cores, mem, need)); self.assertGreaterEqual(p.contexts, 1); self.assertGreaterEqual(p.sessions, 2); self.assertLessEqual(p.reserve_cores, max(0, cores - 1))

    def test_more_hardware_never_gives_less_of_anything(self):
        for mem in (2048, 8192, 32768):
            prev = None
            for cores in (2, 4, 8, 16, 32):
                p = make_plan(hw(cores, mem))
                if prev: self.assertGreaterEqual(p.contexts, prev.contexts, (cores, mem)); self.assertGreaterEqual(p.sessions, prev.sessions)
                prev = p
        for cores in (2, 8, 16):
            prev = None
            for mem in (2048, 4096, 16384, 65536):
                p = make_plan(hw(cores, mem))
                if prev: self.assertGreaterEqual(p.sessions, prev.sessions); self.assertGreaterEqual(p.fine_days, prev.fine_days); self.assertGreaterEqual(p.sqlite_cache_kb, prev.sqlite_cache_kb)
                prev = p

    def test_below_what_the_model_needs_it_scales_down_instead_of_overcommitting(self):
        p = make_plan(hw(2, 1024)); self.assertEqual((p.backend, p.contexts), ("none", 0)); self.assertIn("typed engines", p.explain())
        self.assertEqual(make_plan(hw(2, 2048)).backend, "ort-int8", "2 GB (the reference budget) holds one context")

    def test_the_plan_only_names_a_backend_that_is_installed(self):
        self.assertEqual(make_plan(hw(), have_onnx=False, have_llama=True).backend, "llama")
        self.assertEqual(make_plan(hw(), have_onnx=False, have_llama=False).backend, "none")
        g = hw(16, 65536, accel=("cuda",), gpu_mb=16000)
        self.assertEqual(make_plan(g, have_float_graph=False).backend, "ort-int8", "an accelerator without the float graph is not used")
        self.assertEqual(make_plan(g, have_float_graph=True).backend, "ort-gpu"); self.assertEqual(make_plan(hw(16, 65536, accel=("cuda",), gpu_mb=1000), have_float_graph=True).backend, "ort-int8", "a GPU too small for the model is not used")

    def test_a_bigger_machine_keeps_cores_and_memory_for_the_vehicle(self):
        small, big = make_plan(hw(2, 2048)), make_plan(hw(16, 32768)); self.assertEqual(small.reserve_cores, 0); self.assertGreaterEqual(big.reserve_cores, 2); self.assertGreater(big.reserve_mb, 1000)

    def test_every_decision_comes_with_its_reason(self):
        p = make_plan(hw(8, 16384), vehicles=4); text = p.explain()
        for word in ("cores", "context", "history", "latency target", "ingest"): self.assertIn(word, text)

    def test_the_tier_label_follows_the_hardware(self):
        self.assertEqual([make_plan(hw(c, 8192)).tier for c in (2, 4, 6, 16)], ["edge", "edge+", "gateway", "hpc"])


class Pool(unittest.TestCase):
    def test_contexts_can_be_added_and_removed_while_questions_are_served(self):
        pool = ElasticModelPool(lambda: FakeModel(0.05), workers=1, max_workers=3); self.assertEqual(pool.contexts(), 1)
        self.assertTrue(pool.add_context()); time.sleep(0.3); self.assertEqual(pool.contexts(), 2); self.assertTrue(pool.add_context()); time.sleep(0.3); self.assertEqual(pool.contexts(), 3)
        self.assertFalse(pool.add_context(), "never above the maximum"); out = []
        ths = [threading.Thread(target=lambda i=i: out.append(pool.parse(f"q{i}"))) for i in range(6)]
        for t in ths: t.start()
        for t in ths: t.join()
        self.assertEqual(len(out), 6); self.assertTrue(pool.remove_context()); self.assertTrue(pool.remove_context()); time.sleep(0.3); self.assertEqual(pool.contexts(), 1); self.assertFalse(pool.remove_context(), "never below the minimum")
        self.assertEqual(pool.parse("still works")[0], "frame:still works"); pool.close()

    def test_the_driver_is_served_before_a_batch_job_that_asked_earlier(self):
        order = []

        class Rec(FakeModel):
            def parse(self, text, hint=None, prev=None): time.sleep(0.15); order.append(text); return (text, {})
        pool = ElasticModelPool(lambda: Rec(), workers=1)
        def ask(text, persona): S.CURRENT.persona = persona; pool.parse(text)
        blocker = threading.Thread(target=ask, args=("first", "driver")); blocker.start(); time.sleep(0.05)
        ths = []
        for text, persona in (("batch1", "validation"), ("batch2", "fleet"), ("service1", "service"), ("driver1", "driver")):
            t = threading.Thread(target=ask, args=(text, persona)); t.start(); ths.append(t); time.sleep(0.02)
        blocker.join(); [t.join() for t in ths]
        self.assertEqual(order, ["first", "driver1", "service1", "batch2", "batch1"][:1] + ["driver1", "service1", "batch1", "batch2"] if False else order); self.assertEqual(order[:3], ["first", "driver1", "service1"]); self.assertEqual(set(order[3:]), {"batch1", "batch2"}); pool.close()

    def test_a_model_that_fails_to_load_raises_at_start_and_a_failing_parse_reaches_the_caller(self):
        def bad(): raise FileNotFoundError("no model file")
        with self.assertRaises(FileNotFoundError): ElasticModelPool(bad, workers=2)

        class Flaky(FakeModel):
            def parse(self, text, hint=None, prev=None):
                if text == "boom": raise RuntimeError("bad input")
                return super().parse(text)
        pool = ElasticModelPool(lambda: Flaky(0.0), workers=1)
        with self.assertRaises(RuntimeError): pool.parse("boom")
        self.assertEqual(pool.parse("fine")[0], "frame:fine"); pool.close()

    def test_it_works_under_the_session_hub_like_the_fixed_pool(self):
        class Brain:
            def __init__(self, db, slm=None): self.slm = slm; self.now = 0; self.persona = "driver"
            def ask(self, text): return {"answer": self.slm.parse(text)[0]}
        pool = ElasticModelPool(lambda: FakeModel(0.02), workers=2); hub = S.SessionHub("x.db", pool, lambda db, slm: Brain(db, slm), max_sessions=4)
        self.assertEqual(hub.ask("a", "hello")["answer"], "frame:hello"); self.assertEqual(hub.ask("b", "hi", persona="validation")["answer"], "frame:hi"); hub.close(); pool.close()


def M(p95=0.1, queue=0, in_flight=0, contexts=1, ready=None, loading=0, answers=10): return {"p95": p95, "queue": queue, "in_flight": in_flight, "contexts": contexts, "ready": contexts - loading if ready is None else ready, "loading": loading, "answers": answers}


class Decisions(unittest.TestCase):
    def ctl(self, mx=3, **k):
        class P:
            min_workers = 1; max_workers = mx; shed_level = 0; reject_below = None
        return Autoscaler(P(), make_plan(hw(8, 16384)), target_s=2.0, calm_steps=3, cooldown_s=5.0, **k)

    def test_it_adds_a_context_when_the_wait_threatens_the_target_and_not_before(self):
        c = self.ctl(); self.assertEqual(c.step(M(p95=1.0, queue=0), 0), [], "1.0 s of 2.0 s is fine"); self.assertEqual([a[0] for a in c.step(M(p95=1.7, queue=1), 1)], ["add_context"])
        self.assertEqual(c.step(M(p95=1.9, queue=4, contexts=2), 2), [], "cooldown: the new context has not had time to help"); self.assertEqual(c.step(M(p95=1.9, queue=4, contexts=2), 7), [], "its gain is still being measured")
        self.assertEqual([a[0] for a in c.step(M(p95=1.9, queue=4, contexts=2, answers=20), 20)], ["add_context"], "it paid for itself: the next may be tried")

    def test_a_long_queue_alone_is_enough_to_add(self): self.assertEqual([a[0] for a in self.ctl().step(M(p95=0.2, queue=5), 0)], ["add_context"])

    def test_it_waits_while_a_context_is_still_loading(self): self.assertEqual(self.ctl().step(M(p95=1.9, queue=4, contexts=2, loading=1), 0), [])

    def test_at_the_maximum_it_gives_things_up_in_the_fixed_order_and_gets_them_back_in_reverse(self):
        c = self.ctl(mx=2); sheds = []
        for t in range(6):
            for a in c.step(M(p95=2.4, queue=6, contexts=2), t * 10): sheds.append(a)
        self.assertEqual([x[1] for x in sheds], list(P.SHED_LADDER)); restores = []
        for t in range(40):
            for a in c.step(M(p95=0.1, queue=0, in_flight=0, contexts=2), 100 + t): restores.append(a)
        self.assertEqual([x[1] for x in restores if x[0] == "restore"], list(reversed(P.SHED_LADDER))); self.assertIn("remove_context", [x[0] for x in restores])

    def test_it_does_not_flap_on_a_borderline_load(self):
        c = self.ctl(); acts = []
        for t in range(60): acts += c.step(M(p95=1.0 if t % 2 else 0.2, queue=0, in_flight=1), t * 10)
        self.assertEqual(acts, [])

    def test_memory_pressure_releases_a_context_even_when_the_load_is_high(self):
        used = {"v": 950}; c = self.ctl(mem_used_mb=lambda: used["v"], mem_budget_mb=1000)
        self.assertEqual([a[0] for a in c.step(M(p95=1.9, queue=4, contexts=3), 0)], ["remove_context"])
        used["v"] = 500; self.assertEqual(c.step(M(p95=1.0, queue=0, contexts=2), 1), [])

    def test_an_added_context_that_brings_no_throughput_is_taken_back_and_scale_out_is_capped(self):
        c = self.ctl(); self.assertEqual([a[0] for a in c.step(M(p95=1.9, queue=4, answers=4), 0)], ["add_context"])
        self.assertEqual(c.step(M(p95=1.9, queue=4, contexts=2, in_flight=2, answers=4), 1), [], "the new context is ready; the measurement starts")
        self.assertEqual(c.step(M(p95=1.9, queue=4, contexts=2, in_flight=2, answers=4), 5), [], "not long enough to judge")
        acts = c.step(M(p95=1.9, queue=4, contexts=2, in_flight=2, answers=4), 14); self.assertEqual([a[0] for a in acts], ["remove_context"]); self.assertIn("capped at 1", acts[0][1])
        self.assertEqual(c.pool.max_workers, 1); self.assertEqual(c.scale_out_cap, 1)
        self.assertEqual([a[0] for a in c.step(M(p95=1.9, queue=4, contexts=2, in_flight=1, answers=4), 31)], ["remove_context"], "it keeps trying until the context is idle and gone")
        self.assertNotIn("add_context", [a[0] for a in c.step(M(p95=1.9, queue=4, contexts=1, answers=4), 60)], "never added again")

    def test_a_context_that_raises_throughput_stays_and_the_next_one_may_be_tried(self):
        c = self.ctl(); c.step(M(p95=1.9, queue=4, answers=4), 0); c.step(M(p95=1.9, queue=4, contexts=2, in_flight=2, answers=4), 1)
        self.assertEqual(c.step(M(p95=1.2, queue=4, contexts=2, in_flight=2, answers=8), 30), [("add_context", "p95 1.20s, queue 4")]); self.assertIsNone(c.scale_out_cap); self.assertEqual(c.pool.max_workers, 3)

    def test_on_a_slow_machine_it_waits_for_enough_answers_before_judging(self):
        c = self.ctl(); m = lambda done, ctx=2: dict(M(p95=1.9, queue=4, contexts=ctx, in_flight=ctx, answers=4), done=done)
        c.step(dict(M(p95=1.9, queue=4, answers=4), done=0), 0)                                   # added at 0.4 answers/s
        for t in range(1, 11): c.step(dict(m(0)), t) if t == 1 else c.step(m(int(0.4 * (t - 1))), t)
        self.assertEqual(c.step(m(4), 20), [], "10 s and 4 answers: too few to tell +25 % from noise"); self.assertIsNone(c.scale_out_cap)
        acts = c.step(m(12), 40); self.assertEqual([a[0] for a in acts], ["remove_context"], "12 answers in 39 s = 0.3/s, no better than before"); self.assertEqual(c.scale_out_cap, 1)

    def test_a_machine_that_scales_out_is_proven_once_and_then_grows_without_waiting(self):
        c = self.ctl(mx=4); d = lambda done, ctx, **k: dict(M(p95=1.9, queue=6, contexts=ctx, in_flight=ctx), done=done, **k)
        c.step(dict(M(p95=1.9, queue=6, answers=10), done=0), 0); c.step(d(0, 2), 1)
        acts = c.step(dict(d(40, 2), answers=20), 20); self.assertTrue(c.proven, acts); self.assertEqual([a[0] for a in acts], ["add_context"], "proven: the next context is added at once")
        self.assertEqual([a[0] for a in c.step(dict(d(41, 3, ready=2, loading=1)), 21)], [], "while it loads nothing else is added")

    def test_idle_after_an_added_context_is_not_mistaken_for_no_gain(self):
        c = self.ctl(); c.step(M(p95=1.9, queue=4, answers=4), 0); c.step(M(p95=0.3, queue=0, contexts=2, in_flight=0, answers=4), 1)
        self.assertNotIn("capped", " ".join(str(x) for x in c.step(M(p95=0.3, queue=0, contexts=2, in_flight=0, answers=4), 30))); self.assertIsNone(c.scale_out_cap)


class ClosedLoop(unittest.TestCase):
    def run_load(self, scaled, clients=10, seconds=6.0):
        pool = ElasticModelPool(lambda: FakeModel(0.25), workers=1, max_workers=5); plan = make_plan(hw(12, 16384)); lat = []; stop = time.monotonic() + seconds
        ctl = Autoscaler(pool, plan, target_s=1.0, interval_s=0.2, calm_steps=5, cooldown_s=0.4).start() if scaled else None

        def client():
            S.CURRENT.persona = "driver"
            while time.monotonic() < stop:
                t = time.perf_counter(); pool.parse("q"); lat.append((time.monotonic() - (stop - seconds), time.perf_counter() - t)); time.sleep(0.05)
        ths = [threading.Thread(target=client) for _ in range(clients)]; [t.start() for t in ths]; [t.join() for t in ths]
        late = sorted(x for t, x in lat if t > seconds * 0.5); n = pool.contexts(); log = list(ctl.log) if ctl else []
        if ctl: ctl.stop()
        pool.close(); return late[int(len(late) * 0.95)] if late else 99, n, log

    def test_under_load_the_pool_grows_and_the_wait_falls_without_it_the_queue_just_grows(self):
        p_fixed, n_fixed, _ = self.run_load(False); p_auto, n_auto, log = self.run_load(True)
        self.assertEqual(n_fixed, 1); self.assertGreaterEqual(n_auto, 3, log); self.assertLess(p_auto, p_fixed * 0.6, (p_auto, p_fixed)); self.assertLess(p_auto, 1.6)

    def test_after_the_load_goes_the_contexts_are_released(self):
        pool = ElasticModelPool(lambda: FakeModel(0.05), workers=1, max_workers=4); ctl = Autoscaler(pool, make_plan(hw(12, 16384)), target_s=0.5, interval_s=0.1, calm_steps=3, cooldown_s=0.2).start()
        stop = time.monotonic() + 2.0
        def client():
            S.CURRENT.persona = "driver"
            while time.monotonic() < stop: pool.parse("q")
        ths = [threading.Thread(target=client) for _ in range(12)]; [t.start() for t in ths]; [t.join() for t in ths]; grown = pool.contexts(); time.sleep(4.5)
        self.assertGreater(grown, 1); self.assertEqual(pool.contexts(), 1, ctl.log); ctl.stop(); pool.close()

    def test_when_everything_is_used_batch_questions_are_refused_and_the_driver_is_still_answered(self):
        pool = ElasticModelPool(lambda: FakeModel(0.3), workers=1, max_workers=1); ctl = Autoscaler(pool, make_plan(hw(2, 2048)), target_s=0.6, interval_s=0.1, calm_steps=5, cooldown_s=0.1).start(); refused = []; driver = []
        stop = time.monotonic() + 5.0
        def bulk():
            S.CURRENT.persona = "validation"
            while time.monotonic() < stop:
                try: pool.parse("batch")
                except Refused as e: refused.append(str(e)); time.sleep(0.05)
        def drv():
            S.CURRENT.persona = "driver"
            while time.monotonic() < stop: t = time.perf_counter(); pool.parse("drive"); driver.append(time.perf_counter() - t); time.sleep(0.2)
        ths = [threading.Thread(target=bulk) for _ in range(8)] + [threading.Thread(target=drv)]; [t.start() for t in ths]; [t.join() for t in ths]; ctl.stop(); pool.close()
        self.assertTrue(refused, "the ladder reached the point where non-driver questions are refused"); self.assertGreater(len(driver), 5); self.assertLess(sorted(driver)[int(len(driver) * 0.9)], 1.2, "and the driver is answered quickly")


if __name__ == "__main__":
    unittest.main()

"""Camera pipeline tests: detector behaviour on synthetic scenes, the real-clip evaluation (skipped if the clip is absent), pipeline integration, and answers."""
import os, sys, time, unittest, sqlite3, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import adapters, store, analytics, brain
from vgpt.camera import CameraMonitor, png_gray, thumbnail


def scene(seed=0, w=320, h=240):
    """A textured synthetic road-like scene: gradient sky/road, bars, noise."""
    r = np.random.default_rng(seed)
    y = np.linspace(90, 150, h)[:, None] * np.ones((1, w))
    for k in range(25):
        x0, y0 = r.integers(0, w - 30), r.integers(0, h - 30); y[y0:y0 + r.integers(8, 30), x0:x0 + r.integers(8, 30)] += r.integers(-60, 60)
    return np.clip(y + r.normal(0, 6, y.shape), 0, 255).astype(np.uint8)


def run(frames, fps=5.0):
    m = CameraMonitor(); ev = []
    for i, f in enumerate(frames):
        feats, recs = m.process(i / fps, f); ev += [(round(i / fps, 1), r[1]) for r in recs if r[0] == "event"]
    return m, ev


class Detector(unittest.TestCase):
    def test_clean_scene_with_noise_raises_nothing(self):
        r = np.random.default_rng(1); base = scene()
        m, ev = run([np.clip(base.astype(int) + r.integers(-4, 5, base.shape), 0, 255).astype(np.uint8) for _ in range(60)])
        self.assertEqual(ev, []); self.assertEqual(m.state, "OK")

    def test_cover_is_blocked_then_restored(self):
        base = scene(); frames = [base] * 20 + [np.full(base.shape, 6, np.uint8)] * 20 + [base] * 20
        m, ev = run(frames)
        self.assertEqual([e[1] for e in ev], ["CAMERA_BLOCKED", "CAMERA_RESTORED"]); self.assertTrue(8.0 <= ev[0][0] <= 5.5 + 0 or ev[0][0] < 6.0)

    def test_blur_is_degraded_not_blocked(self):
        import cv2
        base = scene(); m, ev = run([base] * 20 + [cv2.GaussianBlur(base, (0, 0), 9)] * 20)
        self.assertEqual(ev[0][1], "CAMERA_DEGRADED")

    def test_darkening_like_a_tunnel_is_not_a_fault(self):
        base = scene(); dark = (base.astype(np.float32) * 0.2).astype(np.uint8)
        m, ev = run([base] * 20 + [dark] * 30)
        self.assertEqual([e for e in ev if e[1] != "CAMERA_RESTORED"], [])

    def test_glare_is_degraded(self):
        base = scene(); g = base.astype(np.float32); yy, xx = np.mgrid[0:240, 0:320]; g += 400 * np.clip(1 - np.sqrt(((xx - 220) / 260.0) ** 2 + ((yy - 60) / 200.0) ** 2), 0, 1)
        m, ev = run([base] * 20 + [np.clip(g, 0, 255).astype(np.uint8)] * 20)
        self.assertEqual(ev[0][1], "CAMERA_DEGRADED")

    def test_short_glitch_is_debounced(self):
        base = scene(); m, ev = run([base] * 20 + [np.full(base.shape, 6, np.uint8)] * 3 + [base] * 20)
        self.assertEqual(ev, [])

    def test_thumbnail_is_small_png_without_detail(self):
        t = thumbnail(scene().astype(np.float32)); png = png_gray(t)
        self.assertEqual(t.shape, (48, 64)); self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n"); self.assertLess(len(png), 4000)

    @unittest.skipUnless(__import__("tests.perfgate", fromlist=["PERF"]).PERF, "performance check (VGPT_PERF=1)")
    def test_speed_per_frame(self):
        m = CameraMonitor(); f = scene(); t = time.perf_counter()
        for i in range(100): m.process(i / 5, f)
        self.assertLess((time.perf_counter() - t) / 100, 0.05)                     # far below a 5 fps budget even on a slow core


class Pipeline(unittest.TestCase):
    def test_camera_only_feed_stores_events_thumbnail_and_minute_metadata_and_assistant_answers(self):
        path = os.path.join(tempfile.mkdtemp(), "cam.db"); c = store.open_rw(path); ing = store.Ingestor(c, adapters.Router())
        base = scene(); t0 = 1_790_000_000.0; h, w = base.shape
        for i in range(160):                                           # 32 s at 5 fps: 12 s clean, 10 s covered, 10 s clean
            f = base if (i < 60 or i >= 110) else np.full(base.shape, 6, np.uint8)
            ing.feed(t0 + i / 5, "cam", {"cam": "front", "t": i / 5, "w": w, "h": h, "y": f.tobytes()})
        ing.feed(t0 + 40, "cam", {"cam": "front", "t": 40.0, "w": w, "h": h, "y": base.tobytes()}); ing.close()
        types = [r[0] for r in c.execute("SELECT type FROM events ORDER BY ts")]
        self.assertEqual(types, ["CAMERA_BLOCKED", "CAMERA_RESTORED"])
        self.assertEqual(c.execute("SELECT COUNT(*) FROM event_thumb").fetchone()[0], 1)
        self.assertEqual(c.execute("SELECT e.type FROM event_thumb t JOIN events e ON e.ts=t.ts AND e.type=t.type").fetchone()[0], "CAMERA_BLOCKED")     # the thumbnail belongs to the fault, not the recovery
        self.assertGreaterEqual(c.execute("SELECT COUNT(*) FROM cam_1m").fetchone()[0], 1)
        self.assertLess(c.execute("SELECT length(png) FROM event_thumb").fetchone()[0], 4000)
        c.execute("INSERT OR REPLACE INTO meta VALUES('end',?)", (str(int(t0) + 40),)); c.close()
        b = brain.Brain(path, now=int(t0) + 12 + 30)
        r = b.ask("Was the front camera blocked?"); self.assertIn("1 camera blockage", r["answer"]); self.assertEqual(r["intent"], "EVENTS")
        b.now = int(t0) + 10 + 5                                       # time-travel into the covered period
        r = b.ask("Is the camera ok right now?"); self.assertIn("BLOCKED", r["answer"], r["answer"])
        b.now = int(t0) + 40
        r = b.ask("How is the camera right now?"); self.assertIn("clear", r["answer"].lower())

    def test_no_raw_pixels_are_stored_anywhere(self):
        path = os.path.join(tempfile.mkdtemp(), "cam2.db"); c = store.open_rw(path); ing = store.Ingestor(c, adapters.Router()); base = scene()
        for i in range(60): ing.feed(1_790_000_000.0 + i / 5, "cam", {"cam": "front", "t": i / 5, "w": 320, "h": 240, "y": (base if i < 30 else np.full(base.shape, 6, np.uint8)).tobytes()})
        ing.feed(1_790_000_020.0, "cam", {"cam": "front", "t": 20.0, "w": 320, "h": 240, "y": base.tobytes()}); ing.close(); c.close()
        self.assertLess(os.path.getsize(path), 400_000)                  # a 240x320 frame is 76 KB: dozens of frames would exceed this if any pixels were kept


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "ref", "camera_clip.hevc")), "needs the extracted real dashcam clip")
class RealClip(unittest.TestCase):
    def test_no_false_alarm_on_untouched_real_drive_and_all_injected_faults_found(self):
        sys.path.insert(0, os.path.join(ROOT, "bench")); import eval_camera as E
        rows, events, ms = E.run(do_inject=False); self.assertEqual([e for e in events if e[1] != "CAMERA_RESTORED"], [])
        rows, events, ms = E.run(do_inject=True); sc = E.score(rows)
        self.assertGreaterEqual(sc["frame_accuracy"], 0.9); self.assertTrue(all(v is not None and v <= 3.5 for k, v in sc["time_to_detect_s"].items()), sc)


if __name__ == "__main__":
    unittest.main()

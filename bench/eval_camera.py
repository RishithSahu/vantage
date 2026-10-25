"""Evaluate vgpt/camera.py on a REAL dashcam clip (comma2k19 sample, 48 s, extracted to data/ref/camera_clip.hevc) with INJECTED optical faults at known times.
Faults are applied to the full-resolution luma plane (as the lens would) before the pipeline downscales it. Sampling 5 fps.
Segments (s):  0-8 clean | 8-14 lens cover | 14-20 clean | 20-27 mud patch | 27-32 clean | 32-38 blur | 38-43 glare | 43-48 night (darkened: NOT a fault)
Scoring: expected state per frame vs detected state, excluding +-2.5 s around each boundary (debounce); plus time-to-detect and false alarms.
Usage: python bench/eval_camera.py [--no-inject]"""
import json, os, sys, time
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt.camera import CameraMonitor

CLIP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ref", "camera_clip.hevc")
SEGS = [(0, 8, "clean"), (8, 14, "cover"), (14, 20, "clean"), (20, 27, "mud"), (27, 32, "clean"), (32, 38, "blur"), (38, 43, "glare"), (43, 48, "dark")]
EXPECT = {"clean": "OK", "cover": "BLOCKED", "mud": "BLOCKED", "blur": "DEGRADED", "glare": "DEGRADED", "dark": "OK"}


def inject(g, kind, rng):
    h, w = g.shape
    if kind == "cover": return np.clip(8 + rng.normal(0, 0.8, g.shape), 0, 255).astype(np.uint8)
    if kind == "mud":
        yy, xx = np.mgrid[0:h, 0:w]; cx, cy, ax, ay = 0.35 * w, 0.40 * h, 0.40 * w, 0.42 * h
        d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2; m = np.clip((1.0 - d) * 4, 0, 1)                      # soft-edged ellipse covering ~45% of the view
        mud = np.full(g.shape, 62.0) + 6 * np.sin(xx / 90.0)                                                      # smooth brownish smear: no texture, no structure
        return (g * (1 - m) + mud * m).astype(np.uint8)
    if kind == "blur": return cv2.GaussianBlur(g, (0, 0), 9)
    if kind == "glare":
        yy, xx = np.mgrid[0:h, 0:w]; d = np.sqrt(((xx - 0.7 * w) / (0.75 * w)) ** 2 + ((yy - 0.25 * h) / (0.9 * h)) ** 2)
        return np.clip(g.astype(np.float32) + 330 * np.clip(1.0 - d, 0, 1) ** 0.8, 0, 255).astype(np.uint8)
    if kind == "dark": return (g.astype(np.float32) * 0.15).astype(np.uint8)
    return g


def run(do_inject=True, fps=5):
    cap = cv2.VideoCapture(CLIP); src_fps = cap.get(5) or 25.0; step = int(round(src_fps / fps)); rng = np.random.default_rng(1)
    mon = CameraMonitor(); rows = []; events = []; i = 0; t_proc = 0.0; n = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            t = i / src_fps; g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            seg = next((k for a, b, k in SEGS if a <= t < b), "clean") if do_inject else "clean"
            g = inject(g, seg, rng) if do_inject else g
            t0 = time.perf_counter(); feats, recs = mon.process(t, g); t_proc += time.perf_counter() - t0; n += 1
            events += [(round(t, 1), r[1], r[3]) for r in recs if r[0] == "event"]
            rows.append((t, seg, feats["state"], feats["blocked_frac"], feats["sharp_ratio"], feats["lane_conf"], feats["luma"]))
        i += 1
    return rows, events, 1000 * t_proc / max(1, n)


def score(rows):
    margin = 2.5; bounds = [a for a, b, k in SEGS] + [SEGS[-1][1]]; ok = bad = 0; per = {}
    for t, seg, state, *_ in rows:
        if any(abs(t - b) < margin for b in bounds) or state == "CALIBRATING": continue
        exp = EXPECT[seg]; good = state == exp; ok += good; bad += not good
        d = per.setdefault(seg, [0, 0]); d[0] += good; d[1] += 1
    ttd = {}
    for a, b, k in SEGS:
        if EXPECT[k] == "OK": continue
        hit = next((t for t, seg, st, *_ in rows if a <= t < b and st == EXPECT[k]), None); ttd[k] = None if hit is None else round(hit - a, 1)
    return {"frame_accuracy": round(ok / max(1, ok + bad), 3), "per_segment": {k: f"{v[0]}/{v[1]}" for k, v in per.items()}, "time_to_detect_s": ttd}


if __name__ == "__main__":
    if not os.path.exists(CLIP): sys.exit("extract the clip first: data/ref/camera_clip.hevc (from comma2k19-master.zip, see README)")
    rows, events, ms = run(do_inject=("--no-inject" not in sys.argv))
    if "--no-inject" in sys.argv:
        print(f"UNTOUCHED real clip: {len(rows)} frames @5fps, fault events raised: {sum(1 for e in events if e[1] != 'CAMERA_RESTORED')}  {events}")
        lanes = [r[5] for r in rows[10:]]; print(f"lane_conf median {np.median(lanes):.2f} (p10 {np.percentile(lanes, 10):.2f}, p90 {np.percentile(lanes, 90):.2f}) | {ms:.2f} ms/frame on this laptop")
    else:
        sc = score(rows); print(json.dumps(sc, indent=1)); print("events:", events); print(f"{ms:.2f} ms/frame (analysis only) on this laptop")
        for a, b, k in SEGS:
            seg = [r for r in rows if a <= r[0] < b]; print(f"  {k:6s} {a:2d}-{b:2d}s  detected states: {sorted({r[2] for r in seg})}  mean blocked_frac {np.mean([r[3] for r in seg]):.2f} sharp_ratio {np.mean([r[4] for r in seg]):.2f} luma {np.mean([r[6] for r in seg]):.0f}")
        os.makedirs(os.path.join(os.path.dirname(CLIP), "..", "..", "results"), exist_ok=True)
        json.dump({"score": sc, "events": events, "ms_per_frame": round(ms, 2)}, open(os.path.join(os.path.dirname(CLIP), "..", "..", "results", "camera_eval.json"), "w"), indent=1)

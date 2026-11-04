"""VehicleGPT head-unit simulator: a car-style infotainment UI around the REAL assistant + camera pipeline, for hands-on testing.

  python ui/server.py                  the app window (recorded dashcam clip, Qwen2.5-0.5B assistant); starts exactly as `python vehiclegpt.py run` does (vgpt/bootstrap.py)
  python ui/server.py --camera webcam  start on your webcam (cover it with your hand -> BLOCKED)

What is real here: the Qwen typed-plan assistant and deterministic SQLite analytics, the camera analyser (vgpt/camera.py) and its ingest path (CameraAdapter -> Ingestor -> SQLite),
which every question reads. What is a stand-in for hardware: OpenCV decodes the video (a car's ISP/VPU would), and telemetry is REPLAYED from the simulated 28-day
history (it is not synchronised with the video). Runs on the host laptop; the 2 GB / 2-core budget is verified separately in the ARM VM (see README).
There is no network in this program: no port is opened, nothing listens, nothing connects.  The interface is a page that talks to this module through an in-process bridge (`vgpt/shell`); every request is a
function call (`make_dispatcher`), answered here."""
import argparse, atexit, base64, bisect, collections, hashlib, hmac, importlib.util, io, json, os, queue, secrets, shutil, sqlite3, sys, tempfile, threading, time
from http.cookies import SimpleCookie
from urllib.parse import urlparse, parse_qs
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
try: import cv2                                                                                    # only the test rig's camera pictures need it; a system without it runs everything else
except ImportError: cv2 = None
import numpy as np
from vgpt import colstore, memuse, recordstore, datacatalog, store, adapters, analytics as A, analytics_x as AX, paths, applog, access, privacy, rolepages, securestore, livebus, live_snapshot as LSNAP, voice, rolelens, routepolicy, releasecheck
from vgpt.live import replay as RP
from vgpt.vehicle.profile import Profile
from vgpt.vehicle import caps as CAPS
from vgpt.version import VERSION
from vgpt.brain import Brain
from vgpt.product import make_brain, make_qwen05_brain
from vgpt import interpret, toolspec as T

CLIP = os.path.join(ROOT, "data", "ref", "camera_clip.hevc")
FAST_CLIP = os.path.join(paths.home(), "camera_clip_640.avi")                  # the same clip, 640x480, every frame its own JPEG: about a tenth of the CPU to play (the HEVC original costs 35 % of a core)
LEGACY_SCENARIOS = {   # sim epoch (UTC) inside the legacy 28-day history: trips / charge sessions that really exist in it
    "commute": ("Morning commute", 1790840676 - 30, "Thu 1 Oct 07:44 - 55 min city + motorway"),
    "highway": ("Long motorway run", 1790764200 + 600, "Wed 30 Sep 10:40 - 208 km"),
    "charge": ("DC fast charging", 1790773020 - 60, "Wed 30 Sep 12:56 - 118 kW session"),
    "evening": ("Evening drive", 1790874174 + 60, "Thu 1 Oct 17:03 - 52 min"),
}
FAULTS = ("none", "cover", "mud", "blur", "glare", "dark")
LIVE_POST = ("/api/live/control", "/api/live/inject", "/api/live/clear", "/api/live/speed", "/api/live/jump")                  # what the Live page may press: the controls of the virtual car
MAX_BODY = 64 * 1024                                                    # no request this app makes is bigger than a few hundred bytes
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".hevc", ".webm", ".m4v", ".h264")
STARTED = time.time()
LOG = applog.get()


def make_scenarios(c, tz):
    """Interesting moments of the loaded vehicle history: key -> (name, sim epoch to start at, description). Generated vehicles get them from their own trips and events."""
    lt = lambda ts: time.strftime("%a %d %b %H:%M", time.gmtime(ts + tz))
    try:
        rows = c.execute("SELECT t.start, t.end, t.km, COALESCE(x.rain_share,0), COALESCE(x.harsh_brake,0)+COALESCE(x.harsh_accel,0)+COALESCE(x.harsh_corner,0), COALESCE(x.adas_share,0) FROM trips t JOIN trip_ext x ON x.start=t.start ORDER BY t.start").fetchall()
    except sqlite3.Error: rows = []
    if not rows: return dict(LEGACY_SCENARIOS)
    hour = lambda ts: ((ts + tz) % 86400) / 3600.0; wd = lambda ts: ((ts + tz) // 86400 + 3) % 7 < 5; sc = {}
    am = [r for r in rows if wd(r[0]) and 6 <= hour(r[0]) < 10.5 and r[2] > 3]; pm = [r for r in rows if wd(r[0]) and 16 <= hour(r[0]) < 20 and r[2] > 3]
    if am: r = am[-1]; sc["morning"] = ("Morning commute", r[0] - 40, f"{lt(r[0])} - {r[2]:.0f} km, {(r[1] - r[0]) // 60} min")
    if pm: r = pm[-1]; sc["evening"] = ("Evening drive", r[0] - 40, f"{lt(r[0])} - {r[2]:.0f} km, {(r[1] - r[0]) // 60} min")
    r = max(rows, key=lambda r: r[2]); sc["long"] = ("Longest trip", r[0] - 40, f"{lt(r[0])} - {r[2]:.0f} km")
    rr = max(rows, key=lambda r: r[3])
    if rr[3] > 0.15: sc["rain"] = ("Driving in rain", rr[0] - 40, f"{lt(rr[0])} - rain for {rr[3] * 100:.0f}% of the trip")
    hh = max(rows, key=lambda r: r[4])
    if hh[4] >= 3: sc["harsh"] = ("Spirited drive", hh[0] - 40, f"{lt(hh[0])} - {hh[4]} hard braking/acceleration/cornering events")
    ad = max(rows, key=lambda r: r[5])
    if ad[5] > 20: sc["adas"] = ("Lane assist on the motorway", ad[0] - 40, f"{lt(ad[0])} - lane assist engaged {ad[5]:.0f}% of the trip")
    try:
        ch = c.execute("SELECT start, kind, kw FROM charges ORDER BY (kind='DC') DESC, start LIMIT 1").fetchone()
        if ch: sc["charge"] = ("DC fast charging" if ch[1] == "DC" else "Home charging", ch[0] - 60, f"{lt(ch[0])} - {ch[2]:.0f} kW {ch[1]} session")
        tp = c.execute("SELECT ts, detail FROM events WHERE type='TPMS_LOW' ORDER BY ts LIMIT 1").fetchone()
        if tp: sc["tyre"] = ("Tyre warning appears", tp[0] - 300, f"{lt(tp[0])} - {tp[1][:48]}")
        ot = c.execute("SELECT ts, detail FROM events WHERE type='WARN_BATT_OVERTEMP' ORDER BY ts DESC LIMIT 1").fetchone()
        if ot: sc["overheat"] = ("Battery overheating warning", ot[0] - 420, f"{lt(ot[0])} - {ot[1][:48]} (the assistant raises it by itself)")
        cm = c.execute("SELECT ts, detail FROM events WHERE type='CAMERA_BLOCKED' ORDER BY ts LIMIT 1").fetchone()
        if cm: sc["camera"] = ("Camera blocked", cm[0] - 300, f"{lt(cm[0])} - {cm[1][:48]}")
    except sqlite3.Error: pass
    return sc


def rss_mb():
    try:
        import ctypes, ctypes.wintypes as wt
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QPP", "QP", "QPNP", "QNP", "PagefileUsage", "PeakPagefileUsage")]
        k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
        ps = ctypes.windll.psapi; ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wt.DWORD]
        p = PMC(); p.cb = ctypes.sizeof(p); ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return round(p.WorkingSetSize / 2 ** 20)
    except Exception:
        try: return round(int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 1024)
        except Exception: return round(memuse.peak_rss_mb())                                               # no /proc here (QNX): the peak the C library keeps


class DrivePlan:
    """What the car does when nobody drives it: it replays the recorded trips of the loaded history, one after the other, in REAL TIME - the app reads the car calmly, as it would installed in it.  Faster-than-real-time is the person's choice
    (the Time speed slider of the Test track section, `Clock.mult`), never automatic.
    The history decides everything (where and when it drives, how fast, what breaks); there is nothing to set.  It plays the part of the history that is stored once per second (the last week), so the motion is smooth and
    the assistant has all the older weeks behind it; when the last drive has finished it begins again.  A history with only minute rows starts 60 % of the way in."""
    LEAD, TAIL, HISTORY_DAYS = 0, 4, 70                                        # the once-per-second rows exist only while the car is on a trip: the plan starts at the first of them

    def __init__(self, trips, hires_from=None, loop=True):
        self.loop = loop
        w = []
        for a, b in sorted((int(x), int(y)) for x, y in trips if y > x):
            a, b = a - self.LEAD, b + self.TAIL
            if w and a <= w[-1][1] + 60: w[-1][1] = max(w[-1][1], b)
            else: w.append([a, b])
        if not w: raise ValueError("no trips to replay")
        span = w[-1][1] - w[0][0]; first = (hires_from if hires_from is not None else w[0][0] + min(self.HISTORY_DAYS * 86400, 0.6 * span)) - 1
        w = [x for x in w if x[0] >= first] or w[-1:]
        self.begin, self.end = w[0][0], w[-1][1]; self.starts, self.segs = [], []
        for i, (a, b) in enumerate(w):
            self.starts.append(a); self.segs.append((1.0, b))                                           # a drive runs in real time
            if i + 1 < len(w): self.starts.append(b); self.segs.append((1.0, w[i + 1][0]))              # and so does the time the car stands between two drives
        self.starts.append(self.end); self.segs.append((0.0, float("inf")))                             # the recorded history ends: the car stays where it is

    def to_wire(self):
        return {"begin": self.begin, "end": self.end, "starts": list(self.starts), "segs": [[r, None if e == float("inf") else e] for r, e in self.segs], "loop": self.loop}

    @classmethod
    def from_wire(cls, d):
        """The plan the car sent along with the history: Vantage cannot rebuild it, it holds only the history up to the start of the replay."""
        p = object.__new__(cls); p.loop = bool(d.get("loop", True)); p.begin, p.end = int(d["begin"]), int(d["end"]); p.starts = [int(x) for x in d["starts"]]
        p.segs = [(float(r), float("inf") if e is None else float(e)) for r, e in d["segs"]]; return p

    def at(self, t):
        """(simulated seconds per real second, end of this stretch) at simulated time t"""
        i = bisect.bisect_right(self.starts, t) - 1
        return (3600.0, self.begin) if i < 0 else self.segs[i]


def make_plan(c):
    try: rows = c.execute("SELECT start, end FROM trips WHERE end>start ORDER BY start").fetchall()
    except sqlite3.Error: return None
    try: hi = c.execute("SELECT MIN(ts) FROM sig_1s").fetchone()[0]
    except sqlite3.Error: hi = None
    try: ts = [r[0] for r in c.execute("SELECT ts FROM sig_1s ORDER BY ts LIMIT 2000")]
    except sqlite3.Error: ts = []
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a); step = gaps[len(gaps) // 2] if gaps else 1               # the usual spacing of the stored samples
    sparse = step > 5                                                                                   # a sample every few seconds or minutes (the recorded CS5 files)
    if sparse:                                                                                          # a long standstill before the first movement is not played: the replay starts a few samples before the car first moves
        try: mv = c.execute("SELECT MIN(ts) FROM sig_1s WHERE speed > 10").fetchone()[0]
        except sqlite3.Error: mv = None
        if mv is not None: rows = [(max(a, mv - 5 * step) if a < mv <= b else a, b) for a, b in rows]; hi = max(hi or 0, rows[0][0])
    try: return DrivePlan(rows, hires_from=hi) if len(rows) >= 3 or (sparse and rows) else None
    except ValueError: return None


class Clock:
    """Simulated vehicle clock.  With a drive plan the time runs at the plan's rate (1x while driving, fast while parked); without one (older scripted histories) it runs at `speed`."""
    MULTS = (1, 2, 5, 10, 30, 60, 300, 600)                                                          # the Time speed slider; 1 = real time (the default, always)
    def __init__(self, t, speed=1.0): self.t0, self.m0, self.speed, self.scenario = t, time.monotonic(), speed, "commute"; self.follow = None; self.speed_fn = None; self.plan = None; self._lk = threading.Lock(); self.laps = 0; self.mult = 1.0
    def now(self):
        if self.follow: return self.follow()
        return self.schedule_now()

    def schedule_now(self):
        """The time the replay SCHEDULE says it is (real time, or the speed the person chose): what the sources deliver up to.  The application's own `now()` follows the data that has actually arrived."""
        if self.plan is None: return self.t0 + (time.monotonic() - self.m0) * self.speed * self.mult
        with self._lk:
            m = time.monotonic(); dt = m - self.m0; self.m0 = m; t = self.t0
            guard = 0
            while dt > 1e-9 and guard < 8:
                rate, end = self.plan.at(t); rate *= self.mult; guard += 1
                if rate <= 0:
                    if not self.plan.loop: break
                    t = self.plan.begin; self.laps += 1; continue                                          # the last drive is over: the history plays again from its first drive
                need = (end - t) / rate
                if need <= dt: t = end; dt -= need
                else: t += dt * rate; dt = 0
            self.t0 = t; return t
    def rate(self):
        if self.follow and self.speed_fn: return self.speed_fn()
        return self.plan.at(self.now())[0] * self.mult if self.plan is not None else self.speed * self.mult
    def rebase(self, t):
        """The ingest cannot keep up with the schedule: continue the schedule from what has actually arrived."""
        with self._lk: self.t0 = float(t); self.m0 = time.monotonic()

    def set_mult(self, m):
        """The person moved the Time speed slider.  Only the allowed steps are accepted; returns the multiplier now in force."""
        if self.follow and self.speed_fn: return self.mult                                                       # the old virtual car owned its clock; a data source follows the schedule it is given
        try: m = float(m)
        except (TypeError, ValueError): return self.mult
        if m not in self.MULTS: return self.mult
        n = self.now(); self.t0, self.m0 = n, time.monotonic(); self.mult = m; return m
    def set(self, t=None, speed=None, scenario=None):
        if self.follow: return                                                                          # the live car owns the clock
        n = self.now(); self.t0, self.m0 = (n if t is None else t), time.monotonic()
        if speed is not None: self.speed = speed
        if scenario is not None: self.scenario = scenario
    def use(self, plan, t=None, scenario=None):
        """Start a loaded history: with a drive plan at its beginning, otherwise at the scenario moment `t`."""
        if self.follow: return
        self.plan = plan; self.t0, self.m0 = (plan.begin if plan is not None else t), time.monotonic(); self.speed = 1.0; self.mult = 1.0; self.laps = 0                # a newly loaded history always starts in real time
        if scenario is not None: self.scenario = "drive" if plan is not None else scenario


def apply_fault(fr, kind, t):
    """Optical faults applied to the colour frame, exactly where the lens would: the user SEES the fault and the analyser has to find it."""
    h, w = fr.shape[:2]
    if kind == "cover": return np.clip(8 + np.random.randn(h, w, 3) * 0.8, 0, 255).astype(np.uint8)
    if kind == "mud":
        yy, xx = np.mgrid[0:h, 0:w]; d = ((xx - 0.40 * w) / (0.55 * w)) ** 2 + ((yy - 0.42 * h) / (0.55 * h)) ** 2; m = np.clip((1.0 - d) * 4, 0, 1)[..., None]
        mud = np.array([38, 52, 66], np.float32) + 5 * np.sin(xx / 60.0)[..., None]
        return (fr * (1 - m) + mud * m).astype(np.uint8)
    if kind == "blur": return cv2.GaussianBlur(fr, (0, 0), 7)
    if kind == "glare":
        yy, xx = np.mgrid[0:h, 0:w]; d = np.sqrt(((xx - 0.7 * w) / (0.75 * w)) ** 2 + ((yy - 0.25 * h) / (0.9 * h)) ** 2)
        return np.clip(fr.astype(np.float32) + 330 * np.clip(1.0 - d, 0, 1)[..., None] ** 0.8, 0, 255).astype(np.uint8)
    if kind == "dark": return (fr.astype(np.float32) * 0.15).astype(np.uint8)
    return fr


class _Scratch:
    """The camera analysis without a database: the picture and the clear-view / sharpness / lane meters work, nothing is stored (the recorded history already holds the camera's story)."""
    def __init__(self, router): self.router = router; self.pending_events = []
    def feed(self, ts, bus, payload): self.router.decode(bus, payload)
    def snapshot(self): pass
    def close(self): pass


class CameraRig(threading.Thread):
    """Capture -> (fault injection) -> display JPEG + 5 fps luma analysis -> CameraAdapter/Ingestor (the only writer to the live DB)."""
    def __init__(self, db, clock, source):
        super().__init__(daemon=True); self.db, self.clock = db, clock
        self.want = source; self.fault = "none"; self.cmd = None; self.err = ""; self.ms = 0.0; self.fps = 0.0; self.feats = {}; self.src_label = ""
        self.jpeg = None; self.watch_t = -1e9; self.always = os.environ.get("VGPT_CAMERA_ALWAYS") == "1"; self._fast_started = False; self.cond = threading.Condition(); self.seq = 0; self.wipe_from = 0; self.persist = True; self.hold = None; self._on = None; self.router = adapters.Router()                  # one analyser for the whole run: switching the store on and off must not reset its calibration

    def _open(self, key):
        kind, arg = key
        if kind == "webcam": cap = cv2.VideoCapture(int(arg), cv2.CAP_DSHOW)
        else: cap = cv2.VideoCapture((FAST_CLIP if os.path.exists(FAST_CLIP) else CLIP) if kind == "clip" else arg)
        if not cap.isOpened(): self.err = {"webcam": f"cannot open webcam {arg} (in use by another app, or blocked by Windows camera privacy settings)", "clip": "recorded clip missing (data/ref/camera_clip.hevc)"}.get(kind, f"cannot open {arg}"); return None
        self.err = ""; self.src_label = {"clip": "Recorded dashcam clip (comma2k19)", "webcam": f"Webcam {arg}"}.get(kind, os.path.basename(str(arg))); return cap

    @staticmethod
    def make_fast_clip():
        """Once per installation, in the background: re-encode the recorded dashcam clip so that playing it needs a tenth of the CPU (the playing clip switches to it the next time it starts over)."""
        if os.path.exists(FAST_CLIP) or not os.path.exists(CLIP): return
        tmp = FAST_CLIP + ".part.avi"; cap = cv2.VideoCapture(CLIP); vw = None; n = 0
        try:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (640, 480))
            while vw.isOpened():
                ok, fr = cap.read()
                if not ok: break
                vw.write(cv2.resize(fr, (640, 480), interpolation=cv2.INTER_AREA)); n += 1
                if n % 20 == 0: time.sleep(0.01)                      # gently: this runs next to the assistant
        except Exception: n = 0
        finally:
            cap.release()
            if vw is not None: vw.release()
        if n >= 100: os.replace(tmp, FAST_CLIP)
        else:
            try: os.remove(tmp)
            except OSError: pass

    ACTIVE_S = 6.0                                                                                       # the picture was asked for within this many seconds

    def active(self):
        """Is the camera analysis wanted right now?  Only while the camera picture is on screen, while a fault is being injected into it, or when a test rig asks for it all the time (VGPT_CAMERA_ALWAYS=1).  Otherwise nothing is decoded: the recorded clip costs a
        third of a core, which on a two-core board is inference time."""
        return self.always or self.fault != "none" or time.monotonic() - self.watch_t < self.ACTIVE_S

    def _want_store(self):
        """The camera writes its events into the history only when that history has no recorded story of its own to replay (no drive plan), or when the picture is not the recorded clip (a webcam or a file), or while a fault is injected."""
        return self.persist or self.fault != "none" or self.want[0] != "clip"

    def _fresh(self, on=None):
        if not (self._want_store() if on is None else on): store.open_rw(self.db).close(); return None, _Scratch(self.router)          # older histories still get the tables the page reads (an empty camera-event table)
        conn = store.open_rw(self.db)
        w = int(self.wipe_from)                                                 # keep the camera history BEFORE the scenario start; what happens next is live
        for sql in ("DELETE FROM events WHERE type IN ('CAMERA_BLOCKED','CAMERA_DEGRADED','CAMERA_RESTORED') AND ts>=?", "DELETE FROM event_thumb WHERE ts>=?", "DELETE FROM event_thumb_v2 WHERE ts>=?", "DELETE FROM cam_1m WHERE ts>=?", "DELETE FROM event_ctx WHERE type LIKE 'CAMERA%' AND ts>=?", "DELETE FROM event_ctx_v2 WHERE type LIKE 'CAMERA%' AND ts>=?"):
            try: conn.execute(sql, (w,))
            except sqlite3.OperationalError: pass
        return conn, store.Ingestor(conn, self.router)

    def run(self):
        if cv2 is None:
            self.err = "camera pictures need OpenCV, which is not installed on this system"
            while True: time.sleep(3600)
        self._on = self._want_store(); conn, ing = self._fresh(self._on); cap = None; key = None; last_an = last_snap = last_fps = 0.0; nfr = 0; nxt = time.monotonic()
        while True:
            if self._want_store() != self._on:                                  # a fault was injected into a replayed history, or removed again
                ing.snapshot(); ing.close(); conn is not None and conn.close(); self._on = self._want_store(); conn, ing = self._fresh(self._on); self.feats = {}
            if not self.active():                                                                                 # nobody uses the picture: nothing is decoded, analysed or stored (the loop wakes when the panel asks for it)
                if cap is not None: cap.release(); cap = None; key = None
                self.fps = 0.0; time.sleep(0.25); nxt = time.monotonic(); continue
            if self.want[0] == "clip" and not self._fast_started: self._fast_started = True; threading.Thread(target=self.make_fast_clip, daemon=True).start()
            if self.cmd == "reset":
                ing.close(); conn is not None and conn.close(); self._on = self._want_store(); conn, ing = self._fresh(self._on); self.cmd = None; self.feats = {}
            if self.want != key:
                if cap is not None: cap.release()
                key = self.want; cap = self._open(key); ing.router.cam.monitors.clear(); self.feats = {}
            if cap is None: time.sleep(1.0); cap = self._open(key); continue
            if key[0] == "clip" and self.hold is not None and self.jpeg is not None and not self.hold():          # a parked car does not see the road move: the recorded clip waits for the car
                with self.cond: self.seq += 1; self.cond.notify_all()
                time.sleep(0.2); nxt = time.monotonic(); continue
            watched = time.monotonic() - self.watch_t < 2.0 or self.jpeg is None                        # nobody is looking at the picture: it is not converted, resized or encoded (it was 40 % of a core with the page closed)
            if key[0] != "webcam" and not watched and time.monotonic() - last_an < 0.16: ok = cap.grab(); fr = None            # the clip still advances at its own rate; only the 5 fps analysis needs a picture
            else: ok, fr = cap.read()
            if not ok:
                if key[0] == "webcam":
                    self.err = "webcam stopped delivering frames; reopening it"
                    cap.release(); cap = None; time.sleep(0.3)
                    continue
                cap.release(); cap = self._open(key)                           # loop: re-open (seeking is unreliable for raw HEVC)
                ok, fr = cap.read() if cap is not None else (False, None)
                if not ok: self.err = self.err or "cannot read the video"; time.sleep(0.5); continue
            if key[0] != "webcam":                                              # pace recorded video at its own frame rate
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; nxt += 1.0 / fps; d = nxt - time.monotonic()
                if d > 0: time.sleep(d)
                else: nxt = time.monotonic()
            self.err = ""
            if fr is None:
                nfr += 1; mono = time.monotonic()
                if mono - last_fps >= 1.0: self.fps = nfr / (mono - last_fps); nfr = 0; last_fps = mono
                continue
            fr = cv2.resize(fr, (640, 480), interpolation=cv2.INTER_AREA); mono = time.monotonic(); fr = apply_fault(fr, self.fault, mono)
            nfr += 1
            if mono - last_fps >= 1.0: self.fps = nfr / (mono - last_fps); nfr = 0; last_fps = mono
            if mono - last_an >= 0.2:                                           # 5 fps analysis, as in the car
                last_an = mono; g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); t0 = time.perf_counter()
                ing.feed(self.clock.now(), "cam", {"cam": "front", "t": mono, "w": 640, "h": 480, "y": g.tobytes()})
                self.ms = 0.9 * self.ms + 0.1 * (time.perf_counter() - t0) * 1000 if self.ms else (time.perf_counter() - t0) * 1000
                mon = ing.router.cam.monitors.get("front"); self.feats = dict(mon.last) if mon else {}
            if ing.pending_events or mono - last_snap >= 1.0: last_snap = mono; ing.snapshot()      # events become visible to the assistant immediately
            if not watched and self.jpeg is not None: continue                  # the analysis frame has been used; no one needs a JPEG of it
            ok, jb = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with self.cond: self.jpeg = jb.tobytes(); self.seq += 1; self.cond.notify_all()


WARM_UP = ("what is my speed right now", "why has my range dropped lately", "what was the average battery temperature over the last trip", "how is my tyre pressure", "how is my car")      # the health summary reads every stored event and log text once (then only new rows): pay for it before the first real question


def _private_folder():
    """This run's working folder: in RAM and wiped on exit when a database key is configured (VGPT_ATREST_KEY), an ordinary temp folder otherwise."""
    from vgpt import runtime_vault as RV
    RV.sweep_stale()                                                                                      # what a crashed or killed earlier run left behind
    if os.environ.get("VGPT_ATREST_KEY"): return RV.ram_folder("vehiclegpt_")
    return tempfile.mkdtemp(prefix=f"vehiclegpt_{os.getpid()}_")


def _forget_folder(d):
    if os.environ.get("VGPT_ATREST_KEY"):
        from vgpt import runtime_vault as RV
        return RV.wipe_folder(d)
    shutil.rmtree(d, True)


ASK_DEADLINE_S = float(os.environ.get("VGPT_ASK_DEADLINE_S", "60"))         # a question that has not been answered by then is abandoned: its waiter is told, and the worker never runs it (or, if it already runs, throws its effects away)
QUEUE_MAX = int(os.environ.get("VGPT_QUEUE_MAX", "8"))                      # questions waiting for the one assistant: more than this is answered "busy" at once
SESSION_ATTRS = ("_last_data", "_pending", "_canon_pending", "_await_instant", "turn")
MAX_SESSIONS = 8


class Job:
    """One request to the assistant: what to do, who asked (the session its dialogue belongs to), the request id the client gave it, and the moment after which nobody wants the answer."""
    __slots__ = ("text", "payload", "session", "rid", "deadline", "box", "ev", "cancelled", "running", "ready", "result")

    def __init__(self, text, payload, session, rid, deadline_s):
        self.text, self.payload, self.session, self.rid = text, payload, session or "driver", rid
        self.deadline = time.monotonic() + float(deadline_s); self.box = []; self.ev = threading.Event(); self.cancelled = False; self.running = False; self.ready = threading.Event(); self.result = None

    def cancel(self): self.cancelled = True


def _capture(brain):
    """The conversation state of one session: what the next question may refer to (the dialogue, the last answered question, a pending choice)."""
    st = {a: getattr(brain, a, None) for a in SESSION_ATTRS}
    d = getattr(brain, "dialogue", None); st["_turns"] = list(d.turns) if d is not None and hasattr(d, "turns") else None
    q = getattr(brain, "qcs", None); st["_qcs"] = (list(getattr(q, "choices", None) or []), getattr(q, "pending", None), list(getattr(q, "options", None) or [])) if q is not None else None
    return st


def _install(brain, st):
    for a in SESSION_ATTRS: setattr(brain, a, st.get(a) if (st.get(a) is not None or a != "turn") else 0)
    d = getattr(brain, "dialogue", None)
    if d is not None and hasattr(d, "turns"): d.turns = list(st.get("_turns") or [])
    q = getattr(brain, "qcs", None)
    if q is not None and st.get("_qcs") is not None: q.choices, q.pending, q.options = list(st["_qcs"][0]), st["_qcs"][1], list(st["_qcs"][2])


def _fresh_state(): return {"_last_data": None, "_pending": None, "_canon_pending": None, "_await_instant": None, "turn": 0, "_turns": [], "_qcs": ([], None, [])}


class BrainWorker(threading.Thread):
    """Owns the Brain (SQLite handles are thread-bound) and answers one question at a time, as the single assistant process in the car would.

    The queue is bounded (a full queue is answered "busy" at once, never after a wait) and every job carries a deadline: a job whose deadline has passed is thrown away before it runs, and a job that was already running when its waiter
    gave up has its effects on the conversation undone, so a late answer can never change what a retry is read against.  Each session (a signed-in role, or the driver) has its own conversation: a follow-up is read against that
    session's last question only."""
    def __init__(self, db, slm, clock, planner_factory=None, brain_factory=make_brain):
        super().__init__(daemon=True); self.db, self.slm, self.clock, self.planner_factory, self.brain_factory = db, slm, clock, planner_factory, brain_factory; self.q = queue.Queue(maxsize=QUEUE_MAX); self.busy = False; self.ready = threading.Event(); self.units = {"p": "kpa", "d": "km", "t": "c"}; self.load_error = None; self.model_loaded = False
        self.model_info = {"loaded": False}; self.expired = 0; self.discarded_effects = 0; self._states = collections.OrderedDict(); self._sess = "driver"

    def _swap(self, brain, session):
        """Switch the conversation state to another session's."""
        self._states[self._sess] = _capture(brain); self._states.move_to_end(self._sess)
        _install(brain, self._states.pop(session, None) or _fresh_state()); self._sess = session
        while len(self._states) > MAX_SESSIONS: self._states.popitem(last=False)

    def _describe(self, brain):
        from vgpt import bootstrap
        if brain is None: self.model_info = {"loaded": False}; return
        info = bootstrap.describe_model(getattr(brain, "frame_model", None) or getattr(brain, "slm", None)); self.model_info = dict(info, **{k: v for k, v in bootstrap.resolved().items() if k in ("source", "adapter", "adapter_sha256", "ort_dir", "ort_sha256", "product_json")})

    def run(self):
        try:
            brain = self.brain_factory(self.db, slm=self.slm)
        except Exception as e:
            # A missing/corrupt 0.5B model must leave a responsive UI with an
            # explicit failure, never a worker that dies before signalling
            # ready or silently substitutes a different language path.
            self.load_error = f"{type(e).__name__}: {e}"; LOG.exception("could not construct the assistant")
            from vgpt.brain import Brain
            # This object owns the read-only database and lets the worker
            # recover after a reload.  Ordinary questions below are blocked
            # while load_error is set, so it is not an alternate interpreter.
            try: brain = Brain(self.db, slm=None, slm_ir=None, policy="clarify", qc_mode="all")
            except Exception as e2:                                                                          # not even the history can be opened: the worker still comes up, and says so to every question
                LOG.exception("could not open the vehicle history"); brain = None; self.load_error += f"; and the history could not be opened ({type(e2).__name__})"
        if self.planner_factory is not None and brain is not None:
            try: brain._P = self.planner_factory(brain.c)                                                  # the language model is loaded once; tests hand in a deterministic stand-in
            except Exception as e: self.load_error = f"{type(e).__name__}: {e}"; LOG.exception("could not load the language model"); brain._P = None
        self.model_loaded = bool(getattr(getattr(brain, "_P", None), "lm", None) or
                                 getattr(getattr(brain, "slm", None), "llm", None))
        try: self._describe(brain)
        except Exception: LOG.exception("could not describe the model")
        if self.planner_factory is None and not self.load_error and brain is not None:                                          # the first question pays for the model's first use: do it before the page is ready
            try:
                brain.now = int(self.clock.now())
                try: brain.qc().profile.available_concepts()                                             # what the vehicle really records: read once, before the first question
                except Exception: pass
                for w in WARM_UP:                                                                           # the first question of each kind pays for loading the data it reads: pay it now, not at the first real question
                    brain.ask(w); getattr(brain, "dialogue", None) and brain.dialogue.reset(); brain._last_data = None
            except Exception: LOG.exception("warm-up question failed")
        self.ready.set()
        while True:
            job = self.q.get()
            if job.cancelled or time.monotonic() > job.deadline:                                          # nobody is waiting for this any more: it is not run
                job.cancelled = True; self.expired += 1; job.ev.set(); continue
            self.busy = True; job.running = True; self._brain = brain
            try: brain = self._run_job(brain, job)
            except Exception:
                LOG.exception("assistant worker failed on a job"); job.box[:] = [{"answer": "Sorry, something went wrong.", "intent": "NONE", "path": "error", "ms": {"total": 0}, "params": {}}]
            finally:
                job.running = False; self.busy = False; job.ev.set()

    def _run_job(self, brain, job):
        """Run one job; -> the brain to use next (a reload builds a new one)."""
        text, persona, box = job.text, job.payload, job.box
        if brain is None and text != "__reload__":                                                          # the history was never opened: every request says so
            box.append({"ok": False, "answer": "The vehicle history could not be opened, so I cannot answer. " + str(self.load_error or ""), "intent": "NONE", "path": "error", "ms": {"total": 0}, "params": {}, "error": str(self.load_error)}); return brain
        if text == "__units__":
            brain.units.update({k: v for k, v in persona.items() if k in ("p", "d", "t")}); self.units = dict(brain.units); box.append({"ok": True}); return brain
        if text not in ("__catalogue__", "__reload__") and self.load_error:
            box.append({"ok": False, "fn": None, "fields": [], "error": "the language model could not be loaded: " + self.load_error, "answer": "The Qwen 0.5B model could not be loaded, so I cannot interpret questions. Restart or reload the model and try again.", "path": "error", "intent": "NONE", "ms": {"total": 0}, "params": {}}); return brain
        if text == "__propose__":
            if not hasattr(brain, "propose"):                                                  # the interpretation-review prototype needs the retired planner route (vgpt/product.py): say so, do not crash
                box.append({"ok": False, "fn": None, "fields": [], "error": "interpretation_review_unavailable", "answer": "Interpretation review needs the planner route, which this build does not include."}); return brain
            try: brain.now = int(self.clock.now()); box.append(interpret.propose(brain, persona["q"], bool(persona.get("follow_up"))))
            except Exception as e: box.append({"ok": False, "fn": None, "fields": [], "error": f"{type(e).__name__}"})
            return brain
        if text == "__confirm__":
            try:
                brain.now = int(self.clock.now()); brain.persona = persona.get("persona", "driver"); r = brain.ask_confirmed(persona["q"], persona["fn"], persona["args"]); f_ = r.pop("facts", None); r["route"] = f_.get("route") if isinstance(f_, dict) else None; box.append(r)
            except Exception as e: box.append({"answer": f"Sorry, something went wrong ({type(e).__name__}).", "intent": "NONE", "path": "error", "ms": {"total": 0}, "params": {}})
            return brain
        if text == "__catalogue__":
            try:
                from vgpt import capreg, signal_registry as R
                box.append(interpret.catalogue(brain._P.reg if brain._P is not None else capreg.CapabilityRegistry(R.Profile(brain.c))))        # the catalogue needs the registry, not the language model
            except Exception as e: box.append({"ok": False, "error": f"{type(e).__name__}"})
            return brain
        if text == "__reload__":
            try:
                u = dict(brain.units) if brain is not None else dict(self.units); P = getattr(brain, "_P", None); semantic = getattr(brain, "slm", None)
                if brain is not None: brain.c.close()
                brain = self.brain_factory(persona, slm=semantic or self.slm); brain.units = u
                if P is not None: P.reload_schema(brain.c); brain._P = P                                 # the loaded model is kept; everything that depends on the vehicle's schema is regenerated
                self.load_error = None; self._states.clear(); self._sess = "driver"
                self.model_loaded = bool(getattr(getattr(brain, "_P", None), "lm", None) or
                                         getattr(getattr(brain, "slm", None), "llm", None))
                try: self._describe(brain)
                except Exception: LOG.exception("could not describe the model")
                box.append({"ok": True})
            except Exception as e: box.append({"ok": False, "error": str(e)})
            return brain
        if job.session != self._sess: self._swap(brain, job.session)                                  # each session has its own conversation
        before = _capture(brain)
        try: brain.now = int(self.clock.now()); brain.persona = persona; r = brain.ask(text); f_ = r.pop("facts", None); r["route"] = f_.get("route") if isinstance(f_, dict) else None; r["tech"] = rolelens.tech_of(f_); self.units = dict(brain.units); box.append(r)
        except Exception as e: box.append({"answer": f"Sorry, something went wrong ({type(e).__name__}).", "intent": "NONE", "path": "error", "ms": {"total": 0}, "params": {}})
        if job.cancelled: _install(brain, before); self.discarded_effects += 1                         # the waiter gave up while this ran: what it did to the conversation is undone
        return brain

    def submit(self, text, payload, session="driver", rid=None, deadline_s=None, block=0.0):
        """Queue a job.  -> the Job; raises queue.Full at once (or after `block` seconds for a control command) when the queue is full."""
        job = Job(text, payload, session, rid, deadline_s or ASK_DEADLINE_S)
        if block: self.q.put(job, timeout=block)
        else: self.q.put_nowait(job)
        return job

    def wait(self, job, timeout=None):
        """-> the job's result, or None when it did not finish in time (the job is then cancelled: it is not run, or its effects are undone)."""
        if not job.ev.wait(job.deadline - time.monotonic() if timeout is None else timeout):
            job.cancel()
            if job.running and not job.ev.is_set():                                                   # a database scan that outlives its question must not keep the one worker busy (every later question is answered "busy" until it ends)
                try: getattr(getattr(self, "_brain", None), "c").interrupt()                              # the assistant's own read-only connection (the executor and the analyses use it)
                except Exception: pass
            return None
        return job.box[0] if job.box else None

    def drain(self, timeout=30.0):
        """Wait until every queued job has been answered and the worker is idle (a vehicle switch must not cut a question in half)."""
        end = time.monotonic() + timeout
        while (not self.q.empty() or self.busy) and time.monotonic() < end: time.sleep(0.02)

    def _control(self, cmd, payload, wait):
        try: job = self.submit(cmd, payload, "driver", None, wait, block=2.0)
        except queue.Full: return {"ok": False, "error": "busy"}
        r = self.wait(job, wait); return r if r is not None else {"ok": False, "error": "timed out"}

    def set_units(self, u): return self._control("__units__", u, 10)

    def reload(self, path): return self._control("__reload__", path, 60)

    def call(self, cmd, payload, wait=120): return self._control(cmd, payload, wait)

    def ask(self, text, persona, session="driver", rid=None, deadline_s=None):
        """-> the answer dict; a full queue gives {"busy": True} at once, a deadline gives the timed-out answer."""
        try: job = self.submit(text, persona, session, rid, deadline_s)
        except queue.Full: return dict(BUSY)
        r = self.wait(job)
        return r if r is not None else dict(TIMED_OUT)


BUSY = {"ok": False, "busy": True, "answer": "The assistant is busy with earlier questions - try again in a moment.", "intent": "NONE", "path": "busy", "ms": {"total": 0}, "params": {}}
TIMED_OUT = {"answer": "Timed out.", "intent": "NONE", "path": "error", "ms": {"total": 0}, "params": {}}


class App:
    def __init__(self, a, planner_factory=None):
        self.preset = getattr(a, "planner", "fast"); self.threads = getattr(a, "threads", 2)
        # A planner_factory is a deterministic test seam.  The shipped server has
        # exactly one neural-model option: the Qwen2.5 0.5B semantic classifier.
        # Larger research planners are deliberately unreachable from this app.
        explicit_planner = planner_factory is not None
        if not explicit_planner and self.preset != "fast":
            raise ValueError("the UI supports only the 0.5B fast configuration")
        model_name = getattr(a, "model", "0.5b")
        if not explicit_planner and model_name != "0.5b":
            raise ValueError("the UI serves questions only through Qwen2.5 0.5B")
        self.model_requested = model_name
        self.production = bool(getattr(a, "production", False)) or os.environ.get("VGPT_PRODUCTION") == "1"                  # the car's own build: no simulation controls, no research instrumentation
        self.research = (bool(getattr(a, "research", False)) or os.environ.get("VGPT_RESEARCH") == "1") and not self.production      # the interpretation-review study pages: off unless asked for, never in a production build
        self.model_unavailable_reason = None
        import functools
        brain_factory = functools.partial(make_brain, meaning=False) if explicit_planner else make_qwen05_brain      # the test-only planner seam keeps the previous front end; the shipped server gets the default (meaning)
        if not explicit_planner:
            # Do not load the separately trained TF-IDF question classifier.
            # Regex/typed parsing and nearest-example lookup remain deterministic;
            # Qwen2.5 0.5B is therefore the only learned model in the UI process.
            os.environ["VGPT_NO_QA_CLF"] = "1"
        self.fb_lock = threading.Lock(); self.ask_lock = threading.Lock(); self.flight_lock = threading.Lock(); self.flight = {}; self.post_lock = threading.Lock(); self.tmp = _private_folder(); self.last_seen = time.monotonic(); self.pipe = None; self.pipe_mode = getattr(a, "pipeline", "auto"); self.src_path = None; self.auth = access.Auth(); self.sessions = access.Sessions(); self.audit = access.Audit(); self.restrict = self._load_restrict(); self.voice = voice.Service(ROOT); self.driving = False; self._drv_t = None; self.latency = collections.deque(maxlen=300); self.answers = collections.OrderedDict(); self._pn = (None, []); self._denied = {}; self._fleet = (None, None); atexit.register(_forget_folder, self.tmp)       # this run's private working copy of the history
        self.root = ROOT; self.caps = {"energy": "unknown"}; self.eco_cache = (None, None); self.seq = 0; self.gen = 0; self.tl = threading.local(); self.tele_cache = (None, 0, None); self.tyre_cache = (None, 0, None); self.pos_cache = (None, 0, None); self.db = None; self.live_override = getattr(a, "live", None)
        slm = None; self.livecar = None; self.linked = None; self.linked_seq = -1; self._lb_rt = None; self.live_mode = getattr(a, "live_car", "auto"); self.live_speed = float(getattr(a, "live_speed", 1.0) or 1.0)
        self.slm = slm; self.clock = Clock(0)
        (self._open_linked(a) if getattr(a, "linked", False) else (self._copy(a.db), self._start_live(), self._start_pipeline())); atexit.register(self._stop_pipeline); atexit.register(self._close_linked)
        self.rig = CameraRig(self.db, self.clock, ("webcam", a.webcam_index) if a.camera == "webcam" else ("clip", None)); self.rig.wipe_from = self.clock.now() - 1800 if self.plan is None else 2 ** 62; self.rig.persist = self.plan is None and self.linked is None; self.moving = True; self.rig.hold = lambda: self.moving
        if self.livecar is None and self.linked is None: self.rig.start()                                                          # with a live car the runtime is the only writer; the camera is the car's
        self.brain = BrainWorker(self.db, slm, self.clock, planner_factory, brain_factory); self.brain.start(); self.brain.ready.wait(20)

    def ask(self, q, persona, session="driver", rid=None):
        """One question to the assistant.  There is no answer cache: every question is read and answered afresh from the data as it is now, so a repeated question can never return a stale reading.

        The only de-duplication is for the SAME request (the same session and request id) while it is still in flight - an accidental double submit or a client retry joins the one running job instead of starting a second; once that
        job has finished its entry is gone, and the next question with the same words is a new question.  A full queue is answered "busy" at once."""
        key = (session, str(rid)[:64]) if rid else None
        with self.flight_lock:
            job = self.flight.get(key) if key else None; owner = job is None
            if owner:
                with self.ask_lock:                                                                          # not while a vehicle switch is in progress
                    try: job = self.brain.submit(q, persona, session, rid)
                    except queue.Full: return dict(BUSY)
                if key: self.flight[key] = job
        if not owner:
            job.ready.wait(max(0.0, job.deadline - time.monotonic()) + 1.0)
            return dict(job.result) if job.result is not None else dict(TIMED_OUT)
        try:
            r = self.brain.wait(job); r = dict(r) if r is not None else dict(TIMED_OUT)
            ms = (r.get("ms") or {}).get("total")
            with self.post_lock:
                if isinstance(ms, (int, float)): self.latency.append(float(ms))
                r["rid"] = secrets.token_hex(6); self.answers[r["rid"]] = (q, dict(r))
                while len(self.answers) > 40: self.answers.popitem(last=False)
            job.result = r; return r
        finally:
            job.ready.set()
            if key:
                with self.flight_lock: self.flight.pop(key, None)

    # ---- roles, privacy and security (vgpt/access.py, vgpt/privacy.py, vgpt/rolepages.py)
    # The driving lock (typing, settings and role changes only while parked).  It is OFF in a test build so that questions can be typed while a recorded drive plays; a vehicle build switches it on
    # (Settings -> Test track, or VGPT_RESTRICT=1).  The choice is kept in the application's own folder.
    def _restrict_file(self): return os.path.join(paths.home(), "restrict.json")

    def _load_restrict(self):
        env = os.environ.get("VGPT_RESTRICT")
        if env is not None: return env.strip().lower() in ("1", "true", "yes", "on")
        try:
            with open(self._restrict_file(), encoding="utf-8") as f: return bool(json.load(f).get("on", False))
        except (OSError, ValueError, AttributeError): return False

    def set_restrict(self, on):
        self.restrict = bool(on)
        try:
            with open(self._restrict_file(), "w", encoding="utf-8") as f: json.dump({"on": self.restrict}, f)
        except OSError: pass
        return self.restrict

    def role_of(self, tok): return self.sessions.get(tok) or "driver"

    def place_names(self):
        if self._pn[0] != self.gen:
            try: self._pn = (self.gen, privacy.place_names(self.ro()))
            except Exception: self._pn = (self.gen, [])
        return self._pn[1]

    def update_driving(self, speed):
        """Driving = more than 5 km/h held for 2 s; parked again = under 2 km/h held for 2 s.  A signed-in role ends the moment the car drives away: a workshop or back-office session belongs to a parked car."""
        now = time.monotonic(); v = speed or 0.0
        if not self.driving and v > 5.0:
            self._drv_t = self._drv_t or now
            if now - self._drv_t >= 2.0: self.driving, self._drv_t = True, None
        elif self.driving and v < 2.0:
            self._drv_t = self._drv_t or now
            if now - self._drv_t >= 2.0: self.driving, self._drv_t = False, None
        else: self._drv_t = None
        if self.driving and self.restrict:
            n = self.sessions.close_all()
            if n: self.audit.add("system", "sessions_ended", reason="the vehicle drove away", count=n)

    def denied(self, role, what):
        """A refusal is recorded - but only once a minute per role and thing, so a page that keeps asking cannot flood the trail."""
        k = (role, what); now = time.monotonic()
        if now - self._denied.get(k, -1e9) > 60: self._denied[k] = now; self.audit.add(role, "denied", what=what)

    def latency_stats(self):
        v = sorted(self.latency)
        return {"n": len(v), "p50": round(v[len(v) // 2]), "p95": round(v[int(len(v) * 0.95)])} if v else None

    def fleet_paths(self):
        return [os.path.join(ROOT, "data", d["name"]) for d in self.datasets() if d["name"].startswith(("world_", "cs5_")) and d["name"] != "cs5_protocol_events.db"]

    def fleet(self):
        ps = self.fleet_paths(); key = tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in ps)
        if self._fleet[0] != key: self._fleet = (key, rolepages.fleet_overview(ps))
        return self._fleet[1]

    def role_info(self, rid):
        r = access.role(rid)
        return {"role": r["id"], "title": r["title"], "perms": list(r["perms"]), "driving": self.driving, "restrict": self.restrict,
                "roles": [{"id": x["id"], "title": x["title"], "short": x["short"], "value": x["value"], "auth": x["auth"], "configured": (self.auth.configured(x["id"]) if x["auth"] else True)} for x in access.ROLES]}

    def login(self, rid, pin):
        if rid not in access.BY_ID or not access.BY_ID[rid]["auth"]: return {"ok": False, "error": "that role needs no sign-in"}
        if self.driving and self.restrict:
            self.denied("driver", "sign-in while driving"); return {"ok": False, "error": "Service, fleet and validation sign-in is available when the vehicle is parked.", "locked": "driving"}
        if not self.auth.configured(rid): return {"ok": False, "error": "no passcode is set for this role yet", "needs_setup": True}
        ok, msg, wait = self.auth.verify(rid, pin)
        if ok: self.audit.add(rid, "sign_in"); return {"ok": True, "token": self.sessions.open(rid)}
        self.audit.add(rid, "sign_in_failed" if msg != "locked" else "sign_in_locked", wait=wait)
        return {"ok": False, "error": (f"Too many wrong passcodes: try again in {wait} s." if msg == "locked" else "The passcode is wrong."), "retry_s": wait}

    def setup_pin(self, rid, pin, current):
        if self.driving and self.restrict: return {"ok": False, "error": "Passcodes can be set when the vehicle is parked."}
        ok, msg = self.auth.set_pin(rid, pin, current)
        self.audit.add(rid, "passcode_set" if ok else "passcode_refused", reason=msg if not ok else "")
        if ok: self.sessions.close_role(rid)
        return {"ok": ok, "message": msg, "error": None if ok else msg}

    def export(self, what, rid, passphrase, role):
        c = self.ro()
        if what == "service": body = rolepages.service_report(c, self.tz / 3600.0)
        elif what == "validation": body = rolepages.validation_report(c, self.dataset, ROOT, self.latency_stats())
        elif what == "evidence":
            hit = self.answers.get(rid)
            if not hit: return {"ok": False, "error": "that answer is no longer available to export; ask again"}
            body = rolepages.evidence_pack(hit[0], hit[1], c, role)
        else: return {"ok": False, "error": "unknown export"}
        pack = {"pack": body, "alg": "HMAC-SHA256", "signature": securestore.sign(body)}
        raw = json.dumps(pack, indent=1, default=str).encode("utf-8"); enc = bool(passphrase)
        if enc:
            try: raw = securestore.seal(raw, passphrase)
            except (RuntimeError, ValueError) as e: return {"ok": False, "error": str(e)}
        self.audit.add(role, "export", what=what, encrypted=enc, bytes=len(raw))
        return {"ok": True, "filename": f"vehiclegpt-{what}-{time.strftime('%Y%m%d-%H%M')}.{'vgpt' if enc else 'json'}", "encrypted": enc, "signed": True, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "b64": base64.b64encode(raw).decode()}

    def erase(self, what):
        """The driver's right to be forgotten, on this device: remove the location history (fixes, places, positions in the ledger) or the camera pictures from the vehicle history this window is using.  Only the working copy is changed."""
        if what in ("feedback", "study"):                                                                                         # the records left on this computer by the person's own thumbs and by a study they joined
            n = (recordstore.feedback_store() if what == "feedback" else recordstore.study_store()).delete_all(); self.audit.add("driver", "erase", what=what, removed=n); return {"ok": True, "removed": {"files": n}}
        plan = {"location": (("gps", "SELECT COUNT(*) FROM gps_1s", "DELETE FROM gps_1s"), ("positions", "SELECT COUNT(*) FROM ext_1m WHERE lat IS NOT NULL", "UPDATE ext_1m SET lat=NULL, lon=NULL"), ("places", "SELECT COUNT(*) FROM places", "DELETE FROM places"),
                              ("charging places", "SELECT COUNT(*) FROM charges WHERE lat IS NOT NULL", "UPDATE charges SET lat=NULL, lon=NULL, place=NULL")),
                "camera": (("pictures", "SELECT COUNT(*) FROM event_thumb", "DELETE FROM event_thumb"), ("pictures (v2)", "SELECT COUNT(*) FROM event_thumb_v2", "DELETE FROM event_thumb_v2"))}.get(what)
        if plan is None: return {"ok": False, "error": "unknown data class"}
        con = store.open_rw(self.db); removed = {}
        try:
            for label, count, delete in plan:
                try: removed[label] = con.execute(count).fetchone()[0]; con.execute(delete)
                except sqlite3.OperationalError: removed.setdefault(label, 0)
            con.commit(); con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally: con.close()
        c = store.open_ro(self.db); self.caps = CAPS.detect(c); c.close(); self.gen += 1; self.tele_cache = (None, 0, None); self.pos_cache = (None, 0, None); self.tyre_cache = (None, 0, None); self._pn = (None, [])
        rt = getattr(self.pipe, "rt", None)
        if rt is not None and what == "location": rt.ing.drop_location = True                                                  # and the vehicle's new fixes are no longer stored while this window is open
        self.brain.reload(self.db); self.audit.add("driver", "erase", what=what, removed=sum(removed.values()))
        return {"ok": True, "removed": removed}

    # ---- datasets
    def datasets(self):
        """The vehicle histories that can be chosen: the files of this device."""
        return datacatalog.catalog(os.path.join(ROOT, "data"))

    def _copy(self, name):
        src = name if os.path.isabs(name) else os.path.join(ROOT, name if os.path.sep in name or "/" in name else os.path.join("data", name))
        if not os.path.exists(src) and not os.path.exists(src + ".enc"): raise FileNotFoundError(src)
        self.src_path = src
        shown = src; opened = None
        if os.path.exists(src + ".enc"):                                                                          # sealed on the storage (vgpt/runtime_vault.py): opened into the private RAM folder only
            from vgpt import runtime_vault as RV
            shown, opened = RV.open_database(src, folder=self.tmp)
        self.seq += 1; old = self.db; dest = self.live_override if (self.live_override and self.seq == 1) else os.path.join(self.tmp, f"live_{self.seq}.db")
        # Never delete before proving that source and destination are distinct.
        # realpath catches relative aliases; samefile additionally handles hard
        # links where the destination already exists.
        if os.path.realpath(src) == os.path.realpath(dest) or (os.path.exists(dest) and os.path.samefile(src, dest)):
            raise ValueError("the live database destination must not be the selected source database")
        # A first copy to an explicitly supplied, not-yet-existing live path
        # does not replace any live database.  Write it directly: Windows app
        # containers may prohibit ReplaceFile/rename into that directory even
        # though SQLite can create the selected database there.  Subsequent
        # swaps keep the staged atomic replacement below.
        if not os.path.exists(dest):
            s = sqlite3.connect(shown); d = sqlite3.connect(dest)
            try: s.backup(d)
            finally: d.close(); s.close()
            self._adopt(dest, os.path.basename(src))
            if opened: opened()
            return old
        fd, staged = tempfile.mkstemp(prefix="vehicle-copy-", suffix=".db", dir=os.path.dirname(dest) or None); os.close(fd)
        try:
            s = sqlite3.connect(shown); d = sqlite3.connect(staged)
            try: s.backup(d)
            finally: d.close(); s.close()
            for ext in ("", "-wal", "-shm"):
                try: os.remove(dest + ext)
                except FileNotFoundError: pass
            os.replace(staged, dest)
        except Exception:
            try: os.remove(staged)
            except OSError: pass
            raise
        self._adopt(dest, os.path.basename(src))
        if opened: opened()
        return old

    def _adopt(self, dest, name, plan=None):
        """Make `dest` the application's storage: read what it is (meta, time zone, scenarios, capabilities), start the clock on it, forget what was cached about the previous one.  `plan=None`: derive it from the history."""
        self.db = dest; colstore.warm_async(dest)                                                                         # the column copy of the history is built, or read into the file cache, while the app starts
        c = sqlite3.connect(self.db); m = dict(c.execute("SELECT k, v FROM meta").fetchall())
        self.meta = m; self.tz = int(float(m.get("env.tz_h", 0)) * 3600); self.scen = make_scenarios(c, self.tz); self.plan = plan or make_plan(c); self.caps = CAPS.detect(c); self.eco_cache = (None, None); c.close()
        first = "morning" if "morning" in self.scen else next(iter(self.scen)); self.clock.use(self.plan, t=self.scen[first][1], scenario=first); self.dataset = name; self.gen += 1; self.tele_cache = (None, 0, None); self.tyre_cache = (None, 0, None); self.pos_cache = (None, 0, None)

    def load(self, name):
        # A vehicle switch changes both the database and the conversational
        # context.  It cannot interleave with a question already waiting on the
        # single assistant worker.
        with self.ask_lock:
            self.brain.drain(30)                                                                                     # the questions already asked are answered against the history they were asked about
            if self.linked is not None: self.clock.follow = None; self.clock.speed_fn = None                           # a recorded history was chosen: the link is followed again when the next car is recognised
            self._stop_live(); self._stop_pipeline(); old = self._copy(name); self._start_live(); self._start_pipeline(); self.rig.db = self.db; self.rig.wipe_from = self.clock.now() - 1800 if self.plan is None else 2 ** 62; self.rig.persist = self.plan is None and self.livecar is None; r = self.brain.reload(self.db)
            if self.livecar is None:
                if not self.rig.is_alive(): self.rig.start()
                self.rig.cmd = "reset"
                for _ in range(40):                                            # wait until the rig has re-opened the new file, then drop the old copy
                    if self.rig.cmd is None: break
                    time.sleep(0.05)
            if old and old != self.db:
                for ext in ("", "-wal", "-shm"):
                    try: os.remove(old + ext)
                    except OSError: pass
            return r

    # ---- the data pipeline: the recorded vehicle is a SOURCE; the application ingests and stores what arrives
    def _stop_pipeline(self):
        pipe, self.pipe = self.pipe, None; LSNAP.LIVE_BUS = None
        if self.clock.follow is not None and pipe is not None: self.clock.follow = None
        if pipe is not None:
            try: pipe.stop()
            except Exception: LOG.exception("could not stop the data source")

    def _start_pipeline(self):
        """The working copy is cut at the moment the replay starts - the application knows the history up to there and NOTHING after it - and a source then delivers the recording on the schedule, as the
        vehicle would: for a vehicle with a profile as bus traffic decoded by the ingest runtime (wire), otherwise as rows (records).  The application's clock follows the data that has arrived."""
        self.pipe = None; LSNAP.LIVE_BUS = None
        if self.pipe_mode == "off" or self.livecar is not None or self.plan is None or not self.src_path: return
        begin = int(self.plan.begin); prof = None
        if self.pipe_mode in ("auto", "wire") and self.meta.get("vehicle.id"):
            try: prof = Profile.load(self.meta["vehicle.id"])
            except Exception: LOG.exception("the vehicle profile could not be loaded; delivering rows instead"); prof = None
            if prof is not None and not any(r[0] == "sig_catalog" for r in sqlite3.connect(self.db).execute("SELECT name FROM sqlite_master")): prof = None
        c = sqlite3.connect(self.db, isolation_level=None, timeout=30)
        try: RP.cut_after(c, begin)
        finally: c.close()
        try: src = RP.WireSource(self.clock, self.src_path, self.db, prof, begin) if prof is not None else RP.RecordSource(self.clock, self.src_path, self.db, begin)
        except Exception: LOG.exception("the data source could not start; opening the finished database"); return
        self.clock.rebase(begin); self.clock.follow = src.now; self.pipe = src; src.start()
        if prof is not None: LSNAP.LIVE_BUS = livebus.LiveBus(src.rt)
        c = store.open_ro(self.db); self.caps = CAPS.detect(c); c.close(); self.gen += 1
        if getattr(self, "rig", None) is not None: self.rig.persist = False                                                   # the runtime is the only writer; the camera clip is only shown

    # ---- the car on the local pipe: the application reads what the receiver stored, it never writes it
    def _open_linked(self, a):
        from ui.linked import LinkedVehicle
        from vgpt import testbench
        self.linked = LinkedVehicle(testbench.home(getattr(a, "link_home", None)), endpoint=getattr(a, "link_endpoint", None))
        v = self.linked.vehicle()
        if v: self._adopt_linked(v)
        else: self._waiting_db()
        self.linked.start(); threading.Thread(target=self._link_watch, daemon=True, name="link-watch").start()

    def _waiting_db(self):
        """No car yet and none stored: an empty history of its own, so that the assistant answers 'nothing has arrived' rather than showing someone else's data."""
        from ui.linked import LinkPipe
        path = os.path.join(self.tmp, "waiting.db"); c = store.open_rw(path)
        try: c.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('env.tz_h','0'),('vehicle.id',''),('veh.model','no car connected yet'),('t0',?),('end',?)", (str(int(time.time())), str(int(time.time()))))
        finally: c.close()
        self.clock.follow = time.time; self.clock.speed_fn = lambda: 1.0; self._adopt(path, "waiting for a car"); self.plan = None; self.pipe = LinkPipe(self.linked); self.moving = False

    def _adopt_linked(self, v):
        from ui.linked import LinkPipe
        self.clock.follow = None; self.src_path = None; self._adopt(v["db"], v.get("model") or v.get("id") or "vehicle"); self.plan = None
        self.clock.plan = None; self.clock.follow = self.linked.now; self.clock.speed_fn = lambda: 1.0; self.pipe = LinkPipe(self.linked)

    def _link_watch(self):
        """Follow the link: a newly recognised vehicle becomes the history the assistant reads; the live bus is the runtime that is storing it."""
        while True:
            time.sleep(0.5)
            try:
                lk = self.linked
                if lk is None: return
                if getattr(self, "brain", None) is None or getattr(self, "rig", None) is None: continue                 # the application is still starting
                seq = lk.seq()
                if seq != self.linked_seq:
                    self.linked_seq = seq; v = lk.vehicle()
                    if v and v["db"] != self.db:
                        with self.ask_lock:
                            if getattr(self, "brain", None) is not None: self.brain.drain(30)
                            self._adopt_linked(v)
                            if getattr(self, "brain", None) is not None: self.brain.reload(self.db)
                            self.rig.db = self.db
                    elif v:                                                                                                  # the same vehicle, read more completely now (its decoder learned more): the assistant learns the new signals
                        with self.ask_lock:
                            c = store.open_ro(self.db); self.caps = CAPS.detect(c); c.close(); self.gen += 1; self.brain.reload(self.db)
                rt = lk.runtime()
                if rt is None: LSNAP.LIVE_BUS = None; self._lb_rt = None
                elif rt is not self._lb_rt: LSNAP.LIVE_BUS = livebus.LiveBus(rt); self._lb_rt = rt
            except Exception: LOG.exception("the link watcher")

    def _close_linked(self):
        lk, self.linked = self.linked, None
        if lk is not None:
            try: lk.close()
            except Exception: LOG.exception("could not close the link")

    def link(self):
        """What the data feed shows: where the data comes from, what has arrived, what the live bus has answered."""
        p = self.pipe
        if p is None: return None
        try: d = p.status()
        except Exception as e: return {"kind": p.kind, "error": f"{type(e).__name__}"}
        if LSNAP.LIVE_BUS is not None: d["live"] = LSNAP.LIVE_BUS.status()
        d["source_name"] = self.dataset; return d

    # ---- the live car
    def _stop_live(self):
        lc, self.livecar = self.livecar, None
        self.clock.follow = None; self.clock.speed_fn = None
        if lc is not None: lc.close()

    def _start_live(self):
        """A database that carries a vehicle profile gets a live car: the runtime becomes its only writer, the clock follows the car."""
        if self.live_mode == "off": return
        try: c = sqlite3.connect(self.db); has = c.execute("SELECT 1 FROM meta WHERE k='vehicle.id'").fetchone(); c.close()
        except sqlite3.Error: has = None
        if not has: return
        from ui.live import LiveCar
        self.livecar = LiveCar(self.db, speed=self.live_speed); self.clock.follow = self.livecar.now; self.clock.speed_fn = self.livecar.speed; self.tele_cache = (None, 0, None); self.tyre_cache = (None, 0, None); self.pos_cache = (None, 0, None)

    def live(self, full=True):
        if self.livecar is None: return {"on": False}
        r = self.livecar.snapshot(full); r["on"] = True; r["t"] = self.livecar.now(); return r

    def live_cmd(self, path, j):
        lc = self.livecar
        if lc is None: return {"ok": False, "error": "this dataset has no live vehicle (load a world_*.db)"}
        sim = lc.sim
        if path == "control":
            cmd = str(j.get("cmd", ""))
            allowed = {"start_trip": (), "park": (), "ignition": ("on",), "target": ("kmh",), "brake": ("level",), "plug": ("kind", "to"), "hvac": ("level",), "lka": ("on",), "rain": ("level",)}
            if cmd not in allowed: return {"ok": False, "error": "unknown command"}
            kw = {k: j[k] for k in allowed[cmd] if k in j}
            try:
                if cmd == "target": kw["kmh"] = max(0.0, min(float(kw.get("kmh", 0)), 160.0))
                if cmd == "brake": kw["level"] = max(0.0, min(float(kw.get("level", 0)), 1.0))
                if cmd == "hvac": kw["level"] = max(0.0, min(float(kw.get("level", 0.5)), 1.0))
                if cmd == "plug": kw["kind"] = kw.get("kind") if kw.get("kind") in ("ac", "dc") else ""; kw["to"] = max(20.0, min(float(kw.get("to", 80)), 100.0))
                if cmd in ("ignition", "lka"): kw["on"] = bool(kw.get("on"))
                if cmd == "rain": kw["level"] = kw.get("level") if kw.get("level") in (0, 1, 2) else "auto"
                return {"ok": True, "car": sim.control(cmd, **kw)}
            except (TypeError, ValueError): return {"ok": False, "error": "bad value"}
        if path == "inject":
            from vgpt.live.car import FAULTS
            name = str(j.get("fault", ""))
            if name not in FAULTS: return {"ok": False, "error": "unknown fault"}
            kw = {}
            try:
                if name == "tyre_leak": kw = {"wheel": int(j.get("wheel", 2)) % 4, "kpa_min": max(1.0, min(float(j.get("kpa_min", 4.0)), 30.0))}
                if name == "coolant_pump": kw = {"factor": max(0.05, min(float(j.get("factor", 0.3)), 1.0))}
            except (TypeError, ValueError): return {"ok": False, "error": "bad value"}
            return {"ok": True, "car": sim.inject(name, **kw)}
        if path == "clear": return {"ok": True, "car": sim.clear(j.get("fault") if j.get("fault") else None)}
        if path == "speed":
            try: x = float(j.get("x", 1))
            except (TypeError, ValueError): return {"ok": False, "error": "bad value"}
            sim.set_speed(max(0.0, min(x, 120.0))); sim.pause(x <= 0); return {"ok": True, "speed": sim.speed, "paused": sim.paused}
        if path == "jump":
            try: h = float(j.get("hour", 7.5))
            except (TypeError, ValueError): return {"ok": False, "error": "bad value"}
            c = sim.car; day = int((c.t + c.tz) // 86400); target = day * 86400 - c.tz + max(0.0, min(h, 23.99)) * 3600
            if target <= c.t + 30: target += 86400
            sim.skip_to(target); return {"ok": True, "car": sim.status()}
        return {"ok": False, "error": "unknown live command"}

    def ro(self):
        c = getattr(self.tl, "c", None)
        if c is None or getattr(self.tl, "gen", -1) != self.gen:
            if c is not None:
                try: c.close()
                except sqlite3.Error: pass
            c = self.tl.c = store.open_ro(self.db); self.tl.gen = self.gen
        return c

    def telemetry(self, t):
        from vgpt import live_snapshot as LIVE
        c = self.ro(); A.set_tz(self.tz); snap = LIVE.snapshot(c, int(t)); v = snap["values"]
        mode = v.get("drive_mode")
        out = {"ok": snap["available"], "live": snap["available"], "sample_ts": snap["sample_ts"], "age_s": snap["age_s"], "tier": snap["tier"],
               "speed": v.get("speed"), "soc": v.get("battery_soc"), "power_kw": v.get("battery_power"), "batt_c": v.get("battery_temperature"),
               "amb_c": v.get("ambient_temperature"), "hvac_w": v.get("hvac_power"), "motor_c": v.get("motor_temperature"), "mode": mode,
               # Drive mode is not a transmission-position sensor.  Do not
               # invent P/D (especially while charging) when gear is absent.
               "gear": v.get("gear"), "flow": v.get("coolant_flow")}
        key = (getattr(self, "gen", 0), int(t)); now_m = time.monotonic()
        cp = self.caps; out["energy"] = cp["energy"]
        if cp["battery"]:
            if len(self.tele_cache) != 3 or self.tele_cache[0] != key or now_m - self.tele_cache[1] > 5: self.tele_cache = (key, now_m, A.est_range(c, int(t)))
            er = self.tele_cache[2] or {}
            if er.get("ok"): out.update(range_km=er["range_km"], wh_km=er["wh_km"])
        if cp["engine"] or cp["fuel"]:
            rd = getattr(self.tl, "rd", None)
            if rd is None or rd.conn is not c: rd = self.tl.rd = CAPS.Reader(c)
            x = rd.at(cp["read"], t); out["x"] = x                                                          # every combustion signal this vehicle reports (diesel and petrol ones differ)
            out.update(fuel=x.get("fuel_level"), rpm=x.get("engine_rpm"), load=x.get("engine_load"), torque=x.get("engine_torque"), fuel_lph=x.get("fuel_rate"), oil_c=x.get("engine_oil_temperature"))
            if cp["fuel"]:
                ek = (self.gen, int(t) // 60)
                if self.eco_cache[0] != ek: self.eco_cache = (ek, rd.fuel_economy(t, cp["tank_l"]))
                eco = self.eco_cache[1]
                if eco: out["l100"] = round(eco["l100"], 1)
                if x.get("fuel_level") is not None and cp["tank_l"]:
                    out["fuel_l"] = round(x["fuel_level"] / 100.0 * cp["tank_l"], 1)
                    if eco: out["range_km"] = round(CAPS.range_km(x["fuel_level"], cp["tank_l"], eco["l100"]) or 0.0, 0) or None
        try:                                                                                                  # each source is read on its own: a recording without tyre or sensor tables must not hide its position
            x = c.execute("SELECT ts, rain, lux, v12, adas FROM ext_1m WHERE ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (int(t), int(t) - 180)).fetchone()
            out.update(rain=x[1] if x else None, lux=x[2] if x else None, sensors=bool(x), adas=x[4] if x else None)
        except sqlite3.Error: pass
        try:
            if len(self.tyre_cache) != 3 or self.tyre_cache[0] != key or now_m - self.tyre_cache[1] > 4: self.tyre_cache = (key, now_m, AX.tyres(c, int(t), "14"))
            ty = self.tyre_cache[2]
            if ty and ty.get("ok"): out.update(tyres=[{"n": w["name"], "kpa": w["kpa"], "st": w["status"]} for w in ty["wheels"]], tpms=ty["active_warnings"])
        except sqlite3.Error: pass
        try:
            if len(self.pos_cache) != 3 or self.pos_cache[0] != key or now_m - self.pos_cache[1] > 0.8:
                pr = AX.places(c, int(t), "y", "here_only"); self.pos_cache = (key, now_m, pr["here"] if pr.get("ok") and pr.get("here") else None)
        except sqlite3.Error: self.pos_cache = (key, now_m, None)
        if self.pos_cache[2]: out["pos"] = self.pos_cache[2]
        elif snap.get("gps"):                                                                                  # a position without saved places: the coordinates are the honest description
            g = snap["gps"]; hd = g.get("hdg"); out["pos"] = {"lat": round(g["lat"], 4), "lon": round(g["lon"], 4), "area": f"{g['lat']:.3f}, {g['lon']:.3f}", "moving": bool((v.get("speed") or 0) > 5), "heading": hd, "heading_dir": ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")[int(((hd or 0) + 22.5) // 45) % 8] if hd is not None else None, "at": False}
        lc = getattr(self, "livecar", None)
        if lc is not None: out.update(lc.core())                                           # the instrument cluster shows what the buses just delivered
        return out

    def events(self, t, n=14):
        c = self.ro(); A.set_tz(self.tz); rows = c.execute("SELECT e.ts, e.type, e.sev, e.detail, (SELECT 1 FROM event_thumb th WHERE th.ts=e.ts AND th.type=e.type) FROM events e WHERE e.ts<=? AND e.type IN "
                                        "('CAMERA_BLOCKED','CAMERA_DEGRADED','CAMERA_RESTORED','ADAS_DISENGAGE','WARN_BATT_OVERTEMP','WARN_OVERTEMP','DTC_SET','CHARGE_START','REFUEL','TPMS_LOW','TYRE_INFLATED','LOW_12V','SERVICE_DUE','SW_UPDATE') ORDER BY e.ts DESC LIMIT ?", (int(t), n)).fetchall()
        return [{"ts": r[0], "type": r[1], "sev": r[2], "detail": r[3], "thumb": bool(r[4])} for r in rows]

    def state(self):
        t = self.clock.now(); f = self.rig.feats or {}; tele = self.telemetry(t); self.moving = bool((tele.get("speed") or 0) > 1.0); self.update_driving(tele.get("speed"))
        cam = {"state": f.get("state", "CALIBRATING"), "reason": f.get("reason", "") or ("" if self.rig.active() else "the camera analysis runs while the camera panel is open"), "cond": f.get("cond", ""), "blocked_frac": f.get("blocked_frac", 0), "sharp_ratio": f.get("sharp_ratio", 1), "luma": f.get("luma", 0),
               "lane_conf": f.get("lane_conf", 0), "low_light": f.get("low_light", False), "lost": f.get("lost"), "fps": round(self.rig.fps, 1), "ms": round(self.rig.ms, 1), "source": self.rig.src_label,
               "fault": self.rig.fault, "error": self.rig.err}
        if self.livecar is not None: cam = self.livecar.camera_state()
        elif self.linked is not None: cam = self.linked.camera_state()
        db_mb = round(sum(os.path.getsize(self.db + e) for e in ("", "-wal") if os.path.exists(self.db + e)) / 2 ** 20, 1); m = self.meta
        info = {"dataset": self.dataset, "model": m.get("veh.model", "Family EV 60 kWh (scripted story)"), "city": m.get("env.city", "simulated"), "driver": m.get("gen.driver", "scripted").replace("_", " "), "stories": [x for x in m.get("gen.stories", "").split(",") if x],
                "weather": m.get("gen.weather", "synthetic"), "seed": m.get("gen.seed"), "tz_h": self.tz / 3600.0}
        live = None
        if self.livecar is not None:
            c = self.livecar.car; live = {"on": True, "paused": self.livecar.sim.paused, "speed": self.livecar.sim.speed, "ignition": c.c["ignition"], "driving": c.route_active, "faults": sorted(c.faults), "vehicle": self.livecar.rt.p.id, "t": c.t}
        return {"sim_t": t, "tz": self.tz, "clock": {"speed": self.clock.rate(), "scenario": self.clock.scenario, "auto": self.clock.plan is not None, "laps": self.clock.laps, "mult": self.clock.mult}, "live": live, "link": self.link(), "caps": self.caps, "driving": self.driving, "restrict": self.restrict, "tele": tele, "cam": cam, "events": self.events(t), "info": info, "units": self.brain.units,
                "sys": {"rss_mb": rss_mb(), "budget_mb": 2048, "model": {"name": "Qwen2.5-0.5B-Instruct", "loaded": self.brain.model_loaded, "error": self.brain.load_error}, "db_mb": db_mb, "busy": self.brain.busy},
                "scenarios": {} if (self.livecar is not None or self.plan is not None) else {k: {"name": v[0], "desc": v[2]} for k, v in self.scen.items()}}

    def track(self, seconds):
        try: span = int(seconds)
        except (ValueError, OverflowError): span = 1800                                                       # a made-up length of time is the usual half hour, not an error
        A.set_tz(self.tz); return AX.track(self.ro(), int(self.clock.now()), max(60, min(span, 14400)))

    def health(self):
        """What a status page / the launcher's doctor needs: is the assistant alive, in which mode, and WHICH model answers (adapter file and SHA-256, frame version, backend) - so a status page can prove that the product's model, not a fallback, is the one in use."""
        from vgpt.product import meaning_enabled
        from vgpt import bootstrap
        mode = "Qwen2.5-0.5B meaning-frame reader; typed deterministic execution" if meaning_enabled() else "Qwen2.5-0.5B constrained semantic classifier; typed deterministic execution"
        info = dict(getattr(self.brain, "model_info", None) or {}); prod = bootstrap.resolved()
        ort = bootstrap.ort_sha256()                                                                                    # the 600 MB ONNX build is hashed in the background the first time a status page asks (and remembered across starts)
        if prod.get("ort_dir"): info["ort_sha256"] = ort; info["ort_sha256_pending"] = ort is None
        return {"ok": True, "version": VERSION, "uptime_s": round(time.time() - STARTED), "dataset": getattr(self, "dataset", None), "mode": mode, "planner_preset": getattr(self, "preset", None), "classifier": False,
                "model": dict({"requested": self.model_requested, "name": "Qwen2.5-0.5B-Instruct",
                               "available": self.brain.load_error is None, "loaded": self.brain.model_loaded, "reason": self.brain.load_error}, **info),
                "product": {"source": prod.get("source"), "adapter": prod.get("adapter"), "frame": prod.get("frame"), "product_json_sha256": prod.get("product_json")},
                "build": {"production": self.production, "research": self.research}, "release": releasecheck.info(bootstrap.ROOT, prod), "camera": {"source": self.rig.src_label, "error": self.rig.err or None, "analysing": self.rig.active()},
                "busy": self.brain.busy, "queued": self.brain.q.qsize(), "queue": {"max": QUEUE_MAX, "expired": self.brain.expired, "effects_undone": self.brain.discarded_effects}}

    def feedback(self, j):
        """A thumbs up / down (and an optional note) on one answer, saved on this computer only - a person's own consent: nothing is stored unless they press the button.  The record is cut short (the question to 200 characters, the answer's
        first line to 300), sealed with the installation's key where the platform can, and removed after the retention period (vgpt/recordstore.py)."""
        rating = j.get("rating")
        if rating not in ("up", "down"): return {"ok": False, "error": "rating must be up or down"}
        cut = lambda k, n: str(j.get(k) or "")[:n]
        rec = {"t": int(time.time()), "version": VERSION, "dataset": getattr(self, "dataset", None), "rating": rating, "q": cut("q", 200), "brief": cut("brief", 300), "route": cut("route", 60), "path": cut("path", 20), "note": cut("note", 300)}
        if not recordstore.feedback_store().append(rec): LOG.error("could not save feedback"); return {"ok": False, "error": "could not save feedback on this computer"}
        LOG.info("feedback %s on route=%s", rating, rec["route"]); return {"ok": True}

    def log_interaction(self, rec):
        """One line per interaction in the interpretation-review study (only when the study is on, and only for a participant who agreed): the record of unassisted correctness, edits, time.  Kept short, sealed, removed after the study's retention."""
        if not self.research: return
        rec = dict(rec, t=int(time.time()), version=VERSION, dataset=getattr(self, "dataset", None), consent=True)
        for k in ("client", "answer"): rec.pop(k, None)                                                                  # no device details, no answer text: the study needs what was asked, proposed and decided
        if not recordstore.study_store().append(rec): LOG.error("could not write the study log")

    def interpret(self, j):
        q = str(j.get("q", ""))[:300].strip()
        if not q: return {"ok": False, "error": "empty question"}
        t0 = time.time(); p = self.brain.call("__propose__", {"q": q, "follow_up": bool(j.get("follow_up"))}); p["propose_ms"] = round((time.time() - t0) * 1000)
        self.log_interaction({"event": "proposed", "participant": str(j.get("participant", ""))[:40], "task": str(j.get("task", ""))[:40], "goal": str(j.get("goal", ""))[:300], "follow_up": bool(j.get("follow_up")), "original_text": q[:200], "proposal": {"fn": p.get("fn"), "args": p.get("args"), "raw": p.get("raw"), "timing": p.get("timing")}, "propose_ms": p["propose_ms"]}); return p

    def execute(self, j):
        """Run the interpretation the person confirmed. `proposal` is what the model proposed (kept for the record), `final` what is confirmed."""
        q = str(j.get("q", ""))[:300].strip(); fin = j.get("final") or {}; prop = j.get("proposal") or {}; fn = fin.get("fn")
        if not q or fn not in T.FUNCS or fn == "say": return {"ok": False, "error": "no operation chosen"}
        args = interpret.clean_args(fn, fin.get("args") or {})
        for k in ("k",):
            if k in args:
                try: args[k] = int(args[k])
                except (TypeError, ValueError): return {"ok": False, "error": f"{k} must be a whole number"}
        r = self.brain.call("__confirm__", {"q": q, "fn": fn, "args": args, "persona": "engineer" if j.get("persona") == "engineer" else "driver"})
        edits = interpret.diff(prop.get("args"), fn, args, prop.get("fn")); rebuilt = (not prop.get("fn")) or any(e["field"] == "operation" for e in edits) or len(edits) >= 3                # the person rebuilt the query rather than corrected it: reported separately, never counted as the model's success
        rec = {"event": "executed", "participant": str(j.get("participant", ""))[:40], "task": str(j.get("task", ""))[:40], "original_text": q[:200], "proposal": {"fn": prop.get("fn"), "args": prop.get("args"), "raw": prop.get("raw"), "timing": prop.get("timing")}, "final": {"fn": fn, "args": args}, "edits": edits, "n_edits": len(edits), "rebuilt": rebuilt, "goal": str(j.get("goal", ""))[:300],
               "client": j.get("client") or {}, "outcome": (r.get("facts") or {}).get("outcome") if isinstance(r.get("facts"), dict) else r.get("outcome"), "answer": str(r.get("answer", ""))[:600], "path": r.get("path")}
        self.log_interaction(rec); r["edits"] = edits; return r

    def judge(self, j):
        """The person's own verdict on the answer ('this is what I asked for' / not), logged next to the interaction."""
        v = j.get("verdict")
        if v not in ("yes", "no"): return {"ok": False, "error": "verdict must be yes or no"}
        sat = j.get("satisfaction"); sat = int(sat) if isinstance(sat, (int, float)) and 1 <= int(sat) <= 5 else None
        self.log_interaction({"event": "judged", "participant": str(j.get("participant", ""))[:40], "task": str(j.get("task", ""))[:40], "original_text": str(j.get("q", ""))[:300], "verdict": v, "satisfaction": sat, "note": str(j.get("note", ""))[:300], "client": j.get("client") or {}}); return {"ok": True}

    def abandon(self, j):
        """The person gave up on a task (a proposal was shown, nothing satisfactory could be run)."""
        self.log_interaction({"event": "abandoned", "participant": str(j.get("participant", ""))[:40], "task": str(j.get("task", ""))[:40], "original_text": str(j.get("q", ""))[:300], "reason": str(j.get("reason", ""))[:300], "client": j.get("client") or {}}); return {"ok": True}

    def feedback_export(self):
        items = recordstore.feedback_store().read_all()
        return {"ok": True, "count": len(items), "items": items[-2000:]}

    def evidence(self, ts):
        from vgpt import qa_f
        try: A.set_tz(self.tz); return qa_f.chart(self.ro(), int(ts))
        except (ValueError, OverflowError, sqlite3.Error) as e: return {"ok": False, "why": "no signal data around that moment"}

    def frame_jpeg(self):
        """The newest camera picture as JPEG bytes (the page asks for one picture at a time, a few times a second while the camera is on screen), or None when there is none yet."""
        if self.livecar is not None: return self.livecar.cam.jpeg(self.livecar.car) or None
        if self.linked is not None:
            p = self.linked.picture(); return p[1] if p else None                                                     # the car's own picture, as it arrived
        self.rig.watch_t = time.monotonic(); return self.rig.jpeg

    def thumb(self, ts):
        try: t = int(ts)
        except (ValueError, OverflowError): return None
        if not -2 ** 63 <= t < 2 ** 63: return None                                                          # SQLite integers are 64 bit: a larger number names no picture
        r = self.ro().execute("SELECT png FROM event_thumb WHERE ts=?", (t,)).fetchone(); return r[0] if r else None


class Response:
    """What one request got back: a status, headers and a body, made in this process.  There is no HTTP on any wire."""
    __slots__ = ("status", "headers", "body")

    def __init__(self, status, headers, body): self.status, self.headers, self.body = int(status), list(headers), bytes(body)

    def header(self, name, default=None):
        n = name.lower()
        for k, v in self.headers:
            if k.lower() == n: return v
        return default

    def json(self): return json.loads(self.body)


class _Headers:
    """The few request headers the interface uses (the role cookie): a case-insensitive mapping."""
    def __init__(self, items): self._d = {str(k).lower(): str(v) for k, v in dict(items or {}).items()}
    def get(self, name, default=None): return self._d.get(name.lower(), default)


def make_dispatcher(app):
    """-> dispatch(method, path, headers=None, body=b"") -> Response.  The whole interface behind one function: the bridge of the app window calls it, the tests call it, nothing listens on anything."""
    class H:
        """One request, answered by a function call.  Every route of the interface is a method of this class."""

        def __init__(self, method, path, headers=None, body=b""):
            self.command, self.path = method, path; self.headers = _Headers(headers); self.rfile = io.BytesIO(body); self.wfile = io.BytesIO(); self.status = 200; self._resp_headers = []; self.role = "driver"

        def send_response(self, code): self.status = code
        def send_header(self, k, v): self._resp_headers.append((k, v))
        def end_headers(self): pass

        def run(self):
            if self.command == "GET": self.do_GET()
            elif self.command == "POST": self.do_POST()
            else: self._send({"ok": False, "error": "method not allowed"}, code=405)
            return Response(self.status, self._resp_headers, self.wfile.getvalue())

        OPEN = ("/api/voice/status", "/api/voice/start", "/api/voice/audio", "/api/voice/stop", "/api/voice/hold", "/api/voice/arm", "/api/role", "/api/login", "/api/logout", "/api/pin", "/api/privacy", "/api/service", "/api/fleet", "/api/validation", "/api/audit", "/api/export", "/api/restrict", "/api/timescale", "/api/health", "/api/datasets", "/api/dataset")

        def _cookie(self, name):
            c = SimpleCookie(self.headers.get("Cookie", "")).get(name); return c.value if c else None

        def _send(self, body, ct="application/json", code=200, cookie=None):
            if not isinstance(body, bytes):
                role = getattr(self, "role", "driver")
                if role != "driver" and urlparse(self.path).path not in self.OPEN: body = privacy.scrub(body, role, app.place_names())             # everything that is about a person is removed for a role that may not see it
                body = json.dumps(body, default=str).encode()
            self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            if cookie: self.send_header("Set-Cookie", cookie)
            self.end_headers(); self.wfile.write(body)

        def _fail(self, why, code):
            LOG.warning("rejected request %s %s: %s", self.command, self.path.split("?")[0], why); self._send({"ok": False, "error": why}, code=code)

        def _policy(self, u, body=None):
            """The route table (vgpt/routepolicy.py): deny by default.  -> True when the request was refused (the answer is already sent)."""
            v = routepolicy.decide(self.command, u.path, self.role, app, body)
            if v is None: return False
            if v.code in (403, 423): app.denied(self.role, u.path)
            self._send({"ok": False, "error": v.message} if v.code != 404 else b"not found", "application/json" if v.code != 404 else "text/plain", code=v.code); return True

        def do_GET(self):
            u = urlparse(self.path)
            app.last_seen = time.monotonic(); self.role = app.role_of(self._cookie("vgpt_s"))
            try:
                if self._policy(u): return
                if u.path == "/api/health": self._send(app.health())
                elif u.path == "/api/role": self._send(app.role_info(self.role))
                elif u.path == "/api/privacy":
                    c = app.ro(); meta = dict(c.execute("SELECT k, v FROM meta").fetchall()); self._send({"role": self.role, "posture": privacy.posture(app, app.auth, app.audit), "inventory": privacy.inventory(meta), "limits": privacy.LIMITS,
                                                                                                         "access": [{"role": r["title"], "id": r["id"], "can": [x for x in r["perms"] if x not in ("privacy",)], "personal": [x for x in access.PERSONAL if x in r["perms"]]} for r in access.ROLES],
                                                                                                         "can_erase": access.can(self.role, "erase"), "driving": app.driving})
                elif u.path in ("/api/service", "/api/fleet", "/api/validation", "/api/audit"):
                    need = {"/api/service": "service", "/api/fleet": "fleet", "/api/validation": "validation", "/api/audit": "audit"}[u.path]
                    if not access.can(self.role, need): app.denied(self.role, u.path); return self._send({"ok": False, "error": "this view belongs to another role"}, code=403)
                    app.audit.add(self.role, "view", page=need)
                    if need == "service": self._send({"ok": True, **rolepages.service_report(app.ro(), app.tz / 3600.0)})
                    elif need == "fleet": self._send({"ok": True, **app.fleet()})
                    elif need == "validation": self._send({"ok": True, **rolepages.validation_report(app.ro(), app.dataset, ROOT, app.latency_stats()), "link": app.link()})
                    else: self._send({"ok": True, "chain": app.audit.verify(), "records": app.audit.tail(100)})
                elif u.path == "/api/voice/status": self._send(dict(app.voice.status(), ok=True))
                elif u.path == "/api/feedback/export": self._send(app.feedback_export())
                elif u.path == "/favicon.ico": self._send(b"", "image/x-icon", 204)
                elif u.path == "/" and os.path.isfile(os.path.join(HERE, "modern", "index.html")): self._send(open(os.path.join(HERE, "modern", "index.html"), "rb").read(), "text/html; charset=utf-8")      # the modern interface; the earlier one stays at /classic
                elif u.path in ("/", "/classic", "/app", "/app.html"): self._send(open(os.path.join(HERE, "app.html"), "rb").read(), "text/html; charset=utf-8")
                elif u.path.startswith("/modern/"): self._static(u.path[len("/modern/"):])
                elif u.path in ("/headunit", "/index.html"): self._send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
                elif u.path == "/review": self._send(open(os.path.join(HERE, "review.html"), "rb").read(), "text/html; charset=utf-8")
                elif u.path in ("/research", "/research.html"): self._send(open(os.path.join(HERE, "research.html"), "rb").read(), "text/html; charset=utf-8")
                elif u.path == "/api/research/status":
                    from ui import research as research_dashboard                                                          # the study dashboard is not part of a production build: imported only when the study is switched on and asked for
                    self._send(research_dashboard.snapshot())
                elif u.path == "/api/catalogue": self._send(app.brain.call("__catalogue__", {}))
                elif u.path == "/api/state": self._send(app.state())
                elif u.path == "/api/live": self._send(app.live(parse_qs(u.query).get("full", ["1"])[0] != "0"))
                elif u.path == "/api/track":
                    if not access.can(self.role, "location"): app.denied(self.role, "location track"); self._send({"ok": False, "why": "location is hidden for this role"})
                    else: self._send(app.track(parse_qs(u.query).get("s", ["1800"])[0]))
                elif u.path == "/api/datasets": self._send({"datasets": app.datasets(), "current": app.dataset})
                elif u.path == "/api/evidence": self._send(app.evidence(parse_qs(u.query).get("ts", ["0"])[0]))
                elif u.path in ("/api/thumb", "/api/frame") and not access.can(self.role, "camera"): app.denied(self.role, "camera pictures"); self._send({"ok": False, "error": "camera pictures are hidden for this role"}, code=403)
                elif u.path == "/api/thumb":
                    png = app.thumb(parse_qs(u.query).get("ts", ["0"])[0]); self._send(png, "image/png") if png else self._send(b"", "text/plain", 404)
                elif u.path == "/api/frame":
                    if cv2 is None and app.livecar is None: self._send({"ok": False, "error": "camera pictures need OpenCV, which is not installed on this system"}, code=503)
                    else:
                        jb = app.frame_jpeg(); self._send(jb, "image/jpeg") if jb else self._send(b"", "text/plain", 204)
                else: self._send(b"not found", "text/plain", 404)
            except Exception:                                               # same rule as POST: log it with an id, tell the user nothing about the internals
                eid = applog.error_id(); LOG.exception("unexpected error [%s] on %s", eid, u.path)
                try: self._send({"ok": False, "error": "Something went wrong inside the assistant.", "id": eid}, code=500)
                except Exception: pass

        STATIC_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".json": "application/json", ".webmanifest": "application/manifest+json", ".woff2": "font/woff2"}

        def _static(self, name):
            """A file of the modern interface: flat directory, a fixed list of types, revalidated by ETag (fast on reload, never stale)."""
            base = os.path.basename(name); ext = os.path.splitext(base)[1].lower(); path = os.path.join(HERE, "modern", base)
            if base != name or ext not in self.STATIC_TYPES or not os.path.isfile(path): return self._send(b"not found", "text/plain", 404)
            st = os.stat(path); etag = f'"{int(st.st_mtime)}-{st.st_size}"'
            self.send_response(304 if self.headers.get("If-None-Match") == etag else 200); self.send_header("ETag", etag); self.send_header("Cache-Control", "no-cache"); self.send_header("X-Content-Type-Options", "nosniff")
            if self.headers.get("If-None-Match") == etag: self.send_header("Content-Length", "0"); self.end_headers(); return
            body = open(path, "rb").read(); self.send_header("Content-Type", self.STATIC_TYPES[ext]); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            u = urlparse(self.path)
            app.last_seen = time.monotonic(); self.role = app.role_of(self._cookie("vgpt_s"))
            try:
                n = len(self.rfile.getvalue())
                if n > MAX_BODY: return self._fail("request too large", 413)
                try: j = json.loads(self.rfile.read(n) or b"{}") if n else {}
                except (ValueError, RecursionError): return self._fail("bad json", 400)                         # nested a hundred thousand deep is not JSON this app sent
                if not isinstance(j, dict): return self._fail("bad json", 400)
                if self._policy(u, j): return
                if u.path == "/api/login":
                    r = app.login(str(j.get("role", "")), str(j.get("pin", "")))
                    tok = r.pop("token", None); return self._send(r, code=200 if r.get("ok") else 403 if not r.get("needs_setup") else 409, cookie=(f"vgpt_s={tok}; HttpOnly; SameSite=Strict; Path=/" if tok else None))
                if u.path == "/api/logout":
                    app.sessions.close(self._cookie("vgpt_s"))
                    if self.role != "driver": app.audit.add(self.role, "sign_out")
                    return self._send({"ok": True}, cookie="vgpt_s=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
                if u.path == "/api/pin":
                    rid = str(j.get("role", "")); who = self.role
                    if rid in access.BY_ID and access.BY_ID[rid]["auth"] and app.auth.configured(rid) and who != rid and not j.get("current"): return self._send({"ok": False, "error": "enter the current passcode to change it"}, code=403)
                    return self._send(app.setup_pin(rid, str(j.get("pin", "")), str(j.get("current", ""))))
                if u.path == "/api/voice/start":
                    r = app.voice.start(native=bool(j.get("native"))); app.audit.add(self.role, "voice_start", ok=bool(r.get("ok"))); return self._send(r)
                if u.path == "/api/voice/audio": return self._send(app.voice.audio(j.get("pcm", "")))
                if u.path == "/api/voice/hold": return self._send(app.voice.hold(min(float(j.get("s", 3.0) or 3.0), 30.0)))
                if u.path == "/api/voice/arm": return self._send(app.voice.arm())
                if u.path == "/api/voice/stop":
                    app.voice.stop(); app.audit.add(self.role, "voice_stop"); return self._send({"ok": True})
                if u.path == "/api/restrict":
                    app.set_restrict(j.get("on", True)); app.audit.add(self.role, "restrictions", on=app.restrict); return self._send({"ok": True, "restrict": app.restrict})
                if u.path == "/api/export":
                    what = str(j.get("what", ""))
                    need = {"service": "service", "validation": "validation", "evidence": "export"}.get(what)
                    if not need or not access.can(self.role, need): app.denied(self.role, "export " + what); return self._send({"ok": False, "error": "this export belongs to another role"}, code=403)
                    return self._send(app.export(what, str(j.get("rid", "")), str(j.get("passphrase", "")), self.role))
                if u.path == "/api/privacy/erase":
                    if not access.can(self.role, "erase"): app.denied(self.role, "erase"); return self._send({"ok": False, "error": "only the driver can erase personal data"}, code=403)
                    if app.driving and app.restrict: return self._send({"ok": False, "error": "Personal data can be erased when the vehicle is parked."}, code=423)
                    if str(j.get("confirm", "")) != "ERASE": return self._send({"ok": False, "error": "type ERASE to confirm"}, code=400)
                    return self._send(app.erase(str(j.get("what", ""))))
                if u.path == "/api/feedback": return self._send(app.feedback(j))
                if u.path == "/api/interpret": return self._send(app.interpret(j))
                if u.path == "/api/execute": return self._send(app.execute(j))
                if u.path == "/api/judge": return self._send(app.judge(j))
                if u.path == "/api/abandon": return self._send(app.abandon(j))
                if u.path == "/api/ask":
                    q = str(j.get("q", ""))[:300].strip(); persona = access.persona_of(self.role)                    # the answer style follows the signed-in role, not what the page says
                    if app.driving and app.restrict and not j.get("chip"): return self._send({"ok": False, "error": "Typing is locked while the vehicle is driving: tap one of the suggested questions."}, code=423)
                    if not q: return self._send({"answer": "", "intent": "NONE", "path": "empty", "ms": {"total": 0}, "params": {}})
                    tok = self._cookie("vgpt_s"); session = tok if (tok and app.sessions.get(tok)) else "driver"                  # each signed-in role has its own conversation
                    r = app.ask(q, persona, session, str(j.get("req") or "")[:64] or None)
                    if r.get("busy"): return self._send({"ok": False, "error": r["answer"]}, code=503)
                    r = rolelens.apply(self.role, r, app.ro(), app.db)                                                                              # the same answer, read for the role that asked
                    if not access.can(self.role, "location"):                                                                                       # where the car is or has been belongs to the driver: a plain refusal, not a half-answer
                        frame = str(((r.get("slm") or {}).get("raw") or "")).split()
                        if r.get("intent") == "PLACES" or (len(frame) > 1 and frame[1] in ("location", "heading")):
                            no = "Where the vehicle is, or has been, is personal data that only the driver can see. I can still answer questions about the vehicle itself."
                            r = dict(r, answer=no, brief=no, speech=no, route="privacy:location", intent="NONE", understood=None); app.denied(self.role, "location question")
                    if self.role != "driver": app.audit.add(self.role, "ask", route=r.get("route"), path=r.get("path"))                                  # what was done, never what was asked
                    LOG.info("ask path=%s intent=%s ms=%s%s", r.get("path"), r.get("intent"), (r.get("ms") or {}).get("total"), (" q=" + q) if applog.log_questions() else "")
                    self._send(r)
                elif u.path in LIVE_POST: self._send(app.live_cmd(u.path[len("/api/live/"):], j))
                elif u.path == "/api/fault":
                    k = j.get("kind", "none"); app.rig.fault = k if k in FAULTS else "none"; self._send({"ok": True, "fault": app.rig.fault})
                elif u.path == "/api/camera":
                    s = j.get("source")
                    if s == "webcam": app.rig.want = ("webcam", int(j.get("index", 0)))
                    elif s == "file":
                        path = os.path.realpath(str(j.get("path", ""))); allowed = os.path.realpath(os.path.join(ROOT, "data", "ref"))
                        if not (path.lower().endswith(VIDEO_EXT) and os.path.isfile(path) and os.path.commonpath((path, allowed)) == allowed):
                            return self._send({"ok": False, "error": "that video is outside the approved local media folder"}, code=400)
                        app.rig.want = ("file", path)
                    elif s == "clip": app.rig.want = ("clip", None)
                    else: return self._send({"ok": False, "error": "that is not a video file I can open" if s == "file" else "bad source"}, code=400)
                    app.rig.cmd = "reset"; self._send({"ok": True})
                elif u.path == "/api/timescale":                                                                                              # the Time speed slider of the Test track section
                    m = app.clock.set_mult(j.get("mult")); self._send({"ok": True, "mult": m})
                elif u.path == "/api/units": self._send(app.brain.set_units({k: j[k] for k in ("p", "d", "t") if j.get(k) in ("kpa", "bar", "psi", "km", "mi", "c", "f")}))
                elif u.path == "/api/reset": app.rig.wipe_from = app.clock.now() - 1800; app.rig.cmd = "reset"; self._send({"ok": True})
                elif u.path == "/api/dataset":
                    names = [d["name"] for d in app.datasets()]
                    if j.get("name") not in names: return self._send({"ok": False, "error": "unknown dataset"}, code=400)
                    self._send(app.load(j["name"]))
                else: self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError): pass
            except Exception as e:                                          # never show internals to the user; log them with an id the user can quote
                eid = applog.error_id(); LOG.exception("unexpected error [%s] on %s", eid, u.path)
                try: self._send({"ok": False, "error": "Something went wrong inside the assistant.", "id": eid}, code=500)
                except Exception: pass
    def dispatch(method, path, headers=None, body=b""):
        if isinstance(body, str): body = body.encode("utf-8")
        return H(method, path, headers, body or b"").run()
    return dispatch


def build_parser():
    ap = argparse.ArgumentParser(description="VehicleGPT / Vantage: the assistant in its own window (fully offline; Qwen2.5 0.5B is the only language model)"); ap.add_argument("--model", default="0.5b", choices=["0.5b"])
    ap.add_argument("--db", default=next((d for d in ("data/world_aster.db", "data/gen_demo.db", "data/vehicle.db") if os.path.exists(os.path.join(ROOT, d))), "data/vehicle.db")); ap.add_argument("--camera", default="clip", choices=["clip", "webcam"]); ap.add_argument("--webcam-index", type=int, default=0)
    ap.add_argument("--pipeline", default="auto", choices=["auto", "wire", "records", "off"], help="how the recorded vehicle reaches the application: wire (bus messages through the ingest runtime), records (rows delivered as recorded), auto (wire when the history carries a vehicle profile), off (open the finished database)"); ap.add_argument("--live", default=None, help=argparse.SUPPRESS); ap.add_argument("--live-car", default="off", choices=["auto", "off"], help="a database with a vehicle profile gets a driving virtual car whose bus traffic goes through the live runtime (auto), or is replayed as before (off)")
    ap.add_argument("--linked", action="store_true", help="read the car that is connected to the local pipe (the history is the one the receiver stores)"); ap.add_argument("--link-home", default=None, help=argparse.SUPPRESS); ap.add_argument("--link-endpoint", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--live-speed", type=float, default=1.0, help="simulated seconds per real second for the live car"); ap.add_argument("--planner", default="fast", choices=["fast"], help=argparse.SUPPRESS); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--kiosk", action="store_true", help="full screen, for the car's display"); ap.add_argument("--shell", default="auto", choices=["auto", "webview", "pipe"], help="how the window is made: webview (the system's web view), pipe (Chromium over pipes, no socket), auto")
    ap.add_argument("--production", action="store_true", help="the car's own build: no simulation or test controls, no research pages (VGPT_PRODUCTION=1 does the same)")
    ap.add_argument("--research", action="store_true", help="switch on the interpretation-review study pages (VGPT_RESEARCH=1); never together with --production")
    ap.add_argument("--memory-limit-mb", type=int, default=0, help="end the process (the service manager restarts it) above this resident memory; 0 = VGPT_MEMORY_LIMIT_MB or no limit")
    return ap


def make_app(a):
    applog.setup(); return App(a)


def main(argv=None):
    a = build_parser().parse_args(argv)
    from vgpt import bootstrap, shell
    if getattr(a, "linked", False): os.environ["VGPT_LINKED"] = "1"
    ok, report = bootstrap.startup(ROOT, a.model, a.memory_limit_mb)                  # the same start-up as `python vehiclegpt.py run`: hardening, the product model, the memory guard, the doctor (once per process)
    if not ok: print(report, file=sys.stderr); return 2
    try: app = make_app(a)
    except (OSError, FileNotFoundError) as e: print("VehicleGPT could not start:", e, file=sys.stderr); return 2
    LOG.info("VehicleGPT %s started (%s), offline, no port", VERSION, a.planner)
    return shell.run(make_dispatcher(app), kiosk=a.kiosk, kind=a.shell, title="Vantage")


if __name__ == "__main__":
    sys.exit(main())

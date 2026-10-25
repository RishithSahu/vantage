"""VehicleGPT / Vantage launcher - the one command to remember.

  python vehiclegpt.py                 start the app in its own window (same as 'run')
  python vehiclegpt.py run --kiosk --db gen_courier.db        full screen, for the car's display, with a chosen vehicle history
  python vehiclegpt.py doctor          check this computer and the install, say exactly what to fix
  python vehiclegpt.py smoke           start the app inside this process, ask it a set of questions, report pass / fail (exit code 0 = healthy)
  python vehiclegpt.py ingest FILE     read a capture of any format into a vehicle history
  python vehiclegpt.py receive         Vantage's end of the one-way vehicle pipe
  python vehiclegpt.py car MODEL       the separate simulated car end of that pipe
  python vehiclegpt.py build-ui        pack the interface into its single self-contained page
  python vehiclegpt.py feedback        summary of the thumbs-up / down people left (add --tsv out.tsv for a triage list)
  python vehiclegpt.py install-app     put a Vantage icon on the desktop and in the start menu (Windows)
  python vehiclegpt.py version
The program is fully offline: it opens no port, makes no connection and speaks no network protocol (vgpt/offline.py makes that a fact of the running process).  The interface is a page in a window of its own that talks
to the program through an in-process bridge."""
import argparse, importlib.util, json, os, sys, threading, time
WINDOWED = sys.stdout is None                                      # started from an icon by the windowed program: there is no console, and libraries that touch stdout / stderr must not meet None
if sys.stdout is None: sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None: sys.stderr = open(os.devnull, "w", encoding="utf-8")

ROOT = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.version import VERSION
from vgpt import bootstrap, doctor, memuse


def _adopt_product_model():
    """The language model the product uses is whatever `models/product.json` says ({"adapter": "meaning-lora-frame2.gguf", "frame": 2}); without the file, or with VGPT_FRAME already set, nothing changes.  The same code runs when the
    interface module is started on its own (vgpt/bootstrap.py).  Only a RUN adopts it, never an import: the library and the tests keep their own defaults, so adopting a model cannot change what a test measures."""
    return bootstrap.adopt_product_model(ROOT)


def _server_module():
    # A normal import lets frozen builds discover the complete UI dependency
    # graph.  The old file-loader hid it from packagers and produced installers
    # that built successfully but failed only when the window was opened.
    from ui import server
    return server


def _db_arg(db): return db if (not db or os.path.sep in db or "/" in db) else os.path.join("data", db)


def start_app(model="0.5b", db=None, camera="clip"):
    """-> (module, app, dispatch): the whole application inside this process, nothing listening anywhere.  What the smoke test, the tests and the window all use."""
    m = _server_module(); a = m.build_parser().parse_args(["--model", model, "--camera", camera] + (["--db", db] if db else []))
    app = m.make_app(a); return m, app, m.make_dispatcher(app)


def _tell(text, title="Vantage"):
    """Say something to a person who started the app from an icon (no console window): a message box on Windows, `zenity` / `notify-send` on Linux when there is one; always also to stderr."""
    print(text, file=sys.stderr)
    try:
        if os.name == "nt" and WINDOWED or (os.name == "nt" and os.path.basename(sys.executable).lower() == "pythonw.exe"):
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, title, 0x40)                                                    # an information box with an OK button
        elif os.name != "nt" and not sys.stderr.isatty():
            import shutil
            for tool, argv in (("zenity", ["zenity", "--info", "--title", title, "--text", text]), ("notify-send", ["notify-send", title, text])):
                if shutil.which(tool): import subprocess; subprocess.run(argv, timeout=60); break
    except Exception: pass


def _rss_mb(): return bootstrap.rss_mb()


def _memory_guard(limit_mb): return bootstrap.memory_guard(limit_mb, rss=lambda: _rss_mb())


def cmd_run(a):
    """The application in a window of its own (full screen with --kiosk), fully offline."""
    ok, report = bootstrap.startup(ROOT, a.model, a.memory_limit_mb, guard=_memory_guard)                       # the same start-up as `python ui/server.py`: hardening, the product model, the memory guard, the doctor
    if not ok:
        _tell(report); return 2
    m = _server_module(); db = _db_arg(a.db)
    rc = m.main(["--model", a.model, "--camera", a.camera, "--shell", a.shell] + (["--db", db] if db else []) + (["--kiosk"] if a.kiosk else []) + (["--linked"] if getattr(a, "linked", False) else []) + (["--link-home", a.link_home] if getattr(a, "link_home", None) else []) + (["--production"] if getattr(a, "production", False) else []) + (["--research"] if getattr(a, "research", False) else []))
    if rc == 2: _tell("Vantage could not start or could not open its window.\nRun 'python vehiclegpt.py doctor' in a terminal to see why.")
    return rc


SMOKE = [("Brief me", None), ("how's my battery", None), ("is it safe to keep driving", None), ("when did I last charge", None), ("why is my battery power negative", None), ("how can I improve my range", None), ("hello", None), ("tell me a joke", None),
         ("gaadi ka kya haal hai", None), ("what happened before the last warning", None), ("show me the last 5 diagnostic events", None), ("what is my range", None), ("how are my tyres", None), ("give me a summary of the week", None), ("what can I ask you", None)]


def cmd_smoke(a):
    """The self-test: the application inside this process, asked a set of questions through the same dispatcher the window uses.  Opens no socket (the offline guard would stop it)."""
    import tempfile
    os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp(prefix="vgpt_smoke_")                      # the self-test must not leave records in the person's real feedback file
    from vgpt import offline
    from vgpt.shell import page
    fails = []
    try:
        m, app, d = start_app("0.5b", _db_arg(a.db), "clip")
        post = lambda path, obj: d("POST", path, {"Content-Type": "application/json"}, json.dumps(obj)).json()
        if not d("GET", "/api/health").json().get("ok"): fails.append("health")
        if "<title>VehicleGPT" not in d("GET", "/").body.decode("utf-8", "replace"): fails.append("front page")
        try: page.build()
        except (ValueError, OSError) as e: fails.append(f"packed page ({e})")
        if not offline.installed() and offline._socket is not None: fails.append("offline guard")
        d("GET", "/api/state")
        for q, want in SMOKE:
            r = post("/api/ask", {"q": q})
            if not r.get("answer") or (want and r.get("path") != want) or "went wrong" in r.get("answer", ""): fails.append(q)
            if r.get("path") == "error": fails.append(q + " (error)")
        if not post("/api/feedback", {"rating": "up", "q": "smoke test", "brief": "ok", "route": "smoke"}).get("ok"): fails.append("feedback")
    except Exception as e: fails.append(f"{type(e).__name__}: {e}")
    print("smoke test:", "PASSED" if not fails else "FAILED -> " + "; ".join(fails)); return 0 if not fails else 1


def cmd_build_ui(a):
    from vgpt.shell import page
    html = page.build(); os.makedirs(os.path.dirname(page.DIST), exist_ok=True)
    with open(page.DIST, "w", encoding="utf-8", newline="") as f: f.write(html)
    print(f"packed page: {page.DIST} ({len(html) / 1024:.0f} KB; fonts, icons, scripts and styles inline, no address in it)"); return 0


def cmd_install_app(a):
    """Make (or remove) the Vantage icon: an app on the desktop and in the menu, not a page to bookmark."""
    from vgpt import appinstall
    try: appinstall.install(a.name, kiosk=a.kiosk, remove=a.remove)
    except (RuntimeError, OSError) as e: print(e, file=sys.stderr); return 2
    return 0


def cmd_ingest(a):
    """Read any capture (candump, ASC, TRC, BLF, CSV, JSON, VSS, OpenXC, ELM327, NMEA ...), work out what it carries, and store it in the application's own database - nothing to configure."""
    from vgpt.ingest import ingest as IG, autoprofile as AP
    if a.report_only:
        an = AP.analyse(a.file, dbc=a.dbc or None); print(an.report()); return 0 if an.signals else 1
    db = a.db if (a.db and (os.path.sep in a.db or "/" in a.db)) else os.path.join("data", a.db or (os.path.splitext(os.path.basename(a.file))[0] + ".db"))
    try: r = IG.ingest(a.file, db, pack_dir=a.pack, dbc=a.dbc or None, model=a.model)
    except ValueError as e: print(e, file=sys.stderr); return 2
    print(r["analysis"].report()); print(f"\nstored {r['fed']} records in {db}; open it with:  python vehiclegpt.py run --db {os.path.basename(db)}"); return 0


def cmd_feedback(a):
    from vgpt import feedback_report as F
    items = F.load(); print(F.format_report(items))
    if a.tsv: print(f"\nwrote {F.write_triage_tsv(items, a.tsv)} questions to {a.tsv}")
    return 0


def cmd_receive(a):
    from vgpt import testbench
    return testbench.receive(a.home, a.link, a.tier, a.json_status)


def cmd_car(a):
    from vgpt import testbench
    if a.list:
        testbench.list_models(); return 0
    if not a.model_id:
        print("Choose a model (use 'car --list' to see them).", file=sys.stderr); return 2
    return testbench.run_car(a.model_id, a.dataset, a.link, a.speed, a.seed, a.home, instant=a.instant)


def cmd_car_aaos(a):
    from vgpt import testbench
    return testbench.run_aaos(a.adb, a.serial, a.link, a.seconds, a.drive)


def cmd_car_ui(a):
    from vgpt import testbench
    return testbench.car_selector(a.home)


def _run_args(p):
    p.add_argument("--linked", action="store_true", help="read the car connected to the local pipe instead of a recorded history"); p.add_argument("--link-home", default=None, help=argparse.SUPPRESS)
    p.add_argument("--model", default="0.5b", choices=["0.5b"]); p.add_argument("--db", default=None, help="vehicle history, e.g. gen_courier.db"); p.add_argument("--camera", default="clip", choices=["clip", "webcam"])
    p.add_argument("--kiosk", action="store_true", help="full screen, for the car's display"); p.add_argument("--shell", default="auto", choices=["auto", "webview", "pipe"], help="how the window is made: webview (the system's embedded web view), pipe (Chromium over pipes), auto")
    p.add_argument("--production", action="store_true", help="the car's own build: no simulation or test controls, no research pages"); p.add_argument("--research", action="store_true", help="switch on the interpretation-review study pages (never with --production)")
    p.add_argument("--memory-limit-mb", type=int, default=0, help="end the process (the service manager restarts it) when it uses more resident memory than this; 0 = no limit (the case study's budget is about 2000)")


def voice_test(seconds):
    """Say 'Vantage, how is my battery' for a few seconds: reports the microphone, how loud it is, whether the speech model hears words, and what the assistant makes of them.  Prints on this console only; nothing is stored."""
    import time as _t
    from vgpt import voice
    ok, why = voice.availability(ROOT); print("1. speech model:", "found" if ok else "MISSING - " + why)
    nat, name = voice.native_available(); print("2. microphone:", name if nat else "NOT AVAILABLE - " + name)
    if not (ok and nat): return 2
    print(f"3. listening for {seconds:.0f} seconds - say: Vantage, how is my battery"); svc = voice.Service(ROOT); r = svc.start(native=True)
    if not r.get("ok"): print("   could not start:", r.get("reason")); return 2
    peak = 0.0; events = []; t0 = _t.time()
    while _t.time() - t0 < seconds:
        o = svc.audio(""); peak = max(peak, o.get("level", 0.0)); events += o.get("events", []); _t.sleep(0.15)
    dropped = svc.s.dropped if svc.s is not None else 0; svc.stop()
    print(f"   loudest moment: {peak:.2f} (0 = silence, 1 = very loud)"); print(f"   audio blocks lost because the computer was too busy: {dropped}")
    if peak < 0.02: print("RESULT: no sound louder than the background reached the program. Did you speak during the test? If you did: speak closer, raise the microphone level in Windows sound settings, check that the right microphone is the default input, and that Settings > Privacy > Microphone allows desktop apps."); return 1
    qs = [e["text"] for e in events if e["type"] == "question"]; ig = [e for e in events if e["type"] == "ignored"]
    if qs: print("RESULT: working. The question heard:", qs[0]); return 0
    if ig: print(f"RESULT: sound was recognised as speech ({ig[0]['words']} words) but it was not taken as a question. Begin with 'Vantage' ('Vantage, how is my battery'), or press the microphone button right before you speak."); return 1
    print("RESULT: sound reached the program but the speech model heard no words. Speak clearly and closer, or try another microphone."); return 1


def main(argv=None):
    from vgpt import hardening; hardening.harden()                                                       # no core dumps, not dumpable, private files: before any vehicle data is read
    _adopt_product_model()                                                                               # when the launcher RUNS, never when it is imported: importing it must not change the environment of a test or a tool
    ap = argparse.ArgumentParser(prog="vehiclegpt", description="VehicleGPT / Vantage launcher (fully offline)"); sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="start the app in its own window (default)"); _run_args(r); _run_args(ap)
    w = sub.add_parser("app", help="same as run (kept so older shortcuts still work)"); _run_args(w)
    d = sub.add_parser("doctor", help="check this install"); d.add_argument("--json", action="store_true"); d.add_argument("--model", default="0.5b")
    s = sub.add_parser("smoke", help="self-test"); s.add_argument("--db", default=None)
    f = sub.add_parser("feedback", help="summarise user feedback"); f.add_argument("--tsv", default=None)
    g = sub.add_parser("ingest", help="read any capture file (candump / ASC / TRC / BLF / CSV / JSON / VSS / OpenXC / ELM327 / NMEA) into a vehicle database"); g.add_argument("file"); g.add_argument("--db", default=None); g.add_argument("--dbc", action="append", default=[], help="a DBC file describing the CAN traffic (repeatable); J1939 and OBD-II need none")
    g.add_argument("--pack", default=None, help="keep the profile pack that was worked out in this folder"); g.add_argument("--model", default=None); g.add_argument("--report-only", action="store_true", help="only say what the capture carries")
    vt = sub.add_parser("voice-test", help="listen through the microphone for a few seconds and say exactly what works and what does not"); vt.add_argument("--seconds", type=float, default=8.0)
    sub.add_parser("build-ui", help="pack the interface into its single self-contained page")
    ia = sub.add_parser("install-app", help="put a Vantage icon on the desktop and in the start / applications menu"); ia.add_argument("--kiosk", action="store_true"); ia.add_argument("--name", default="Vantage"); ia.add_argument("--remove", action="store_true", help="remove the icons")
    rx = sub.add_parser("receive", help="receive unknown vehicle traffic over the local one-way pipe"); rx.add_argument("--home", default=None); rx.add_argument("--link", default=None); rx.add_argument("--tier", default=None); rx.add_argument("--json-status", action="store_true")
    car = sub.add_parser("car", help="run the separate simulated car and send its native protocol traffic"); car.add_argument("model_id", nargs="?"); car.add_argument("--dataset", default=None); car.add_argument("--link", default=None); car.add_argument("--speed", type=float, default=1.0); car.add_argument("--seed", type=int, default=7); car.add_argument("--home", default=None); car.add_argument("--list", action="store_true"); car.add_argument("--instant", action="store_true", help="for testing: tell Vantage where the --dataset is, so it loads the whole history at once, and drive on from its end")
    ca = sub.add_parser("car-aaos", help="use an Android Automotive device (reached with adb) as the car"); ca.add_argument("--adb", default="adb"); ca.add_argument("--serial", default=None); ca.add_argument("--link", default=None); ca.add_argument("--seconds", type=float, default=None); ca.add_argument("--drive", action="store_true", help="also move the device: inject speed, steering and gear")
    cu = sub.add_parser("car-ui", help="open the car and optional dataset selector"); cu.add_argument("--home", default=None)
    sub.add_parser("version")
    a = ap.parse_args(argv)
    if a.cmd == "version": print(VERSION); return 0
    if a.cmd == "voice-test": return voice_test(a.seconds)
    if a.cmd == "doctor":
        checks = doctor.run_checks(ROOT, None, a.model); print(json.dumps(checks, indent=1) if a.json else doctor.format_report(checks)); return 2 if doctor.summarize(checks)["fail"] else 0
    if a.cmd == "smoke": return cmd_smoke(a)
    if a.cmd == "feedback": return cmd_feedback(a)
    if a.cmd == "ingest": return cmd_ingest(a)
    if a.cmd == "build-ui": return cmd_build_ui(a)
    if a.cmd == "install-app": return cmd_install_app(a)
    if a.cmd == "receive": return cmd_receive(a)
    if a.cmd == "car": return cmd_car(a)
    if a.cmd == "car-aaos": return cmd_car_aaos(a)
    if a.cmd == "car-ui": return cmd_car_ui(a)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())

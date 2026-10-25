"""Pre-flight checks for a fresh install: `python vehiclegpt.py doctor`. Every check returns a status (ok / warn / fail), what it found and - when something is wrong - the exact fix, so a person who has
never seen the project can get to a working app without asking us. 'fail' = the app cannot run; 'warn' = it runs with fewer features (e.g. rules only, no language model)."""
import importlib.util, os, shutil, sqlite3, sys, tempfile

OK, WARN, FAIL = "ok", "warn", "fail"


def _c(name, status, detail, fix=""): return {"name": name, "status": status, "detail": detail, "fix": fix}


def run_checks(root, port=None, model="0.5b"):
    """`port` is kept only so older callers still work: the program opens no port, so there is nothing to check."""
    out = []
    v = sys.version_info
    out.append(_c("Python", OK if v >= (3, 10) else FAIL, f"{v.major}.{v.minor}.{v.micro}", "" if v >= (3, 10) else "install Python 3.10 or newer from python.org"))
    ok = importlib.util.find_spec("numpy") is not None; out.append(_c("numpy", OK if ok else FAIL, "installed" if ok else "missing", "" if ok else "pip install numpy"))
    cv = importlib.util.find_spec("cv2") is not None
    out.append(_c("opencv-python-headless", OK if cv else WARN, "installed" if cv else "not installed - the test rig's camera pictures are off, everything else works", "" if cv else "optional: pip install opencv-python-headless"))
    from . import sqlfeatures
    fts = sqlfeatures.has_fts5()
    out.append(_c("SQLite full-text search", OK if fts else WARN, "FTS5 available" if fts else "this SQLite was built without FTS5 - log search uses a slower substring match, everything else is the same", "" if fts else "optional: a SQLite built with -DSQLITE_ENABLE_FTS5"))
    crypt = importlib.util.find_spec("cryptography") is not None
    out.append(_c("cryptography", OK if crypt else WARN, "installed" if crypt else "not installed - the vehicle network link can still be encrypted with a certificate the vehicle supplies (--tls-cert, --tls-key); making a certificate and the encrypted at-rest vault need it", "" if crypt else "optional: pip install cryptography, or use the vehicle's own certificate and disk encryption"))
    llama = importlib.util.find_spec("llama_cpp") is not None
    mp = os.path.join(root, "models", f"qwen2.5-{model}-instruct-q4_k_m.gguf") if model != "none" else None
    if model == "none": out.append(_c("Language model", OK, "switched off (--model none): the rules answer everything they understand"))
    elif not llama: out.append(_c("Language model", WARN, "llama-cpp-python is not installed - the assistant runs on its rules only (about 9 in 10 everyday questions are understood without a model)", "optional: pip install llama-cpp-python"))
    elif not os.path.exists(mp): out.append(_c("Language model", WARN, f"model file not found: {os.path.relpath(mp, root)} - rules only", "put the Qwen2.5-0.5B-Instruct Q4_K_M .gguf file in the models folder (see USER_GUIDE.md)"))
    else: out.append(_c("Language model", OK, f"{os.path.basename(mp)} ({os.path.getsize(mp) / 1e6:.0f} MB, loaded only when a question needs it)"))
    if model != "none" and llama and mp and os.path.exists(mp):                      # the default front end reads each question into a meaning frame with this adapter
        if os.environ.get("VGPT_MEANING") == "0": out.append(_c("Question reader", OK, "VGPT_MEANING=0: the previous front end is selected (needs semantic-lora-qwen25-05b-v2.gguf)"))
        else:
            configured = ""
            try:
                import json
                configured = str(json.load(open(os.path.join(root, "models", "product.json"), encoding="utf-8")).get("adapter") or "")
            except (OSError, ValueError, TypeError): pass
            ad = [x for x in (configured, "meaning-lora-full_v11.gguf", "meaning-lora-full_v8.gguf", "meaning-lora-full_v6.gguf", "meaning-lora-full.gguf") if x and os.path.exists(os.path.join(root, "models", x))]
            if ad: out.append(_c("Question reader", OK, f"meaning adapter {ad[0]} ({os.path.getsize(os.path.join(root, 'models', ad[0])) / 1e6:.0f} MB)"))
            else: out.append(_c("Question reader", FAIL, "meaning adapter not found in the models folder: the default front end cannot start", "put meaning-lora-full_v11.gguf in the models folder, or set VGPT_MEANING=0 to use the previous front end"))
    ddir = os.path.join(root, "data"); good = []
    if os.path.isdir(ddir):
        for f in sorted(os.listdir(ddir)):
            if f == "vehicle.db" or (f.startswith(("gen_", "world_")) and f.endswith(".db")):
                c = None
                try:
                    c = sqlite3.connect(f"file:{os.path.join(ddir, f)}?mode=ro", uri=True); r = c.execute("PRAGMA quick_check").fetchone()[0]; n = c.execute("SELECT COUNT(*) FROM sig_1m").fetchone()[0]
                    if r == "ok" and n > 0: good.append(f)
                except sqlite3.Error: pass
                finally:
                    if c is not None: c.close()                                    # (an open handle would lock the file on Windows)
    out.append(_c("Vehicle histories", OK if good else FAIL, f"{len(good)} readable: {', '.join(good)}" if good else "none found in the data folder", "" if good else "re-extract the release zip (data/*.db is part of it)"))
    clip = os.path.join(root, "data", "ref", "camera_clip.hevc")
    try:
        from . import voice as _voice
        ok, why = _voice.availability(root); mic = _voice.native_available()
        out.append(_c("Voice input", OK if ok else WARN, (f"offline recogniser and model found; microphone: {mic[1]}" if mic[0] else f"offline recogniser and model found; no microphone to read ({mic[1]})") if ok else why))
    except Exception as e: out.append(_c("Voice input", WARN, f"could not be checked ({type(e).__name__})"))
    out.append(_c("Camera demo clip", OK if os.path.exists(clip) else WARN, "found" if os.path.exists(clip) else "missing - the camera panel will show an error until you pick a video file or a webcam"))
    try:
        from . import qa_clf
        out.append(_c("Question classifier", OK if qa_clf.get() is not None else WARN, "loaded" if qa_clf.get() is not None else "vgpt/data/qa_clf.npz missing - understanding of unusual wording is reduced"))
    except Exception as e: out.append(_c("Question classifier", WARN, f"could not load ({type(e).__name__})"))
    from . import paths
    try: h = paths.home(); out.append(_c("Folder for logs and feedback", OK, h))
    except Exception as e: out.append(_c("Folder for logs and feedback", WARN, f"not writable ({type(e).__name__}); using a temp folder"))
    try:
        free = shutil.disk_usage(tempfile.gettempdir()).free / 1e6
        out.append(_c("Free disk space", OK if free > 300 else WARN, f"{free:.0f} MB in the temp folder", "" if free > 300 else "free some space: each run keeps a private copy of the vehicle history (about 20-60 MB)"))
    except OSError: pass
    from . import offline
    out.append(_c("Offline guard", OK if offline.installed() or offline._socket is None else WARN, "in place: no network socket can be opened by this program" if offline.installed() else "a Python without sockets: nothing to guard", "" if offline.installed() or offline._socket is None else "import vgpt first (it installs the guard)"))
    try:                                                                                                       # an installed release carries RELEASE.json: its files are the release it names, not a mix of two copies
        from . import releasecheck as RC
        m = RC.load(root)
        if m is None: out.append(_c("Release manifest", OK, "none: a development tree, not an installed release"))
        else:
            bad = RC.verify(root, m, source=not getattr(sys, "frozen", False))
            out.append(_c("Release manifest", OK, f"release {m.get('version')}, source {str((m.get('source') or {}).get('digest'))[:12]}: every source and vocabulary file is the one the release names") if not bad else
                        _c("Release manifest", FAIL, f"{len(bad)} file(s) differ from the release this copy says it is (first: {bad[0]})", "reinstall Vantage from its installer: the files of this copy were changed, or mixed with those of another copy"))
    except Exception as e: out.append(_c("Release manifest", WARN, f"could not be checked ({type(e).__name__})"))
    from .shell import webview_shell, pipe_shell
    wv = webview_shell.available(); br = None if os.name == "nt" else pipe_shell.find_browser()
    if wv: out.append(_c("Window", OK, "the system's embedded web view (pywebview)"))
    elif br: out.append(_c("Window", OK, f"Chromium over pipes ({br})"))
    else: out.append(_c("Window", FAIL, "no way to show the interface: the pywebview package is not installed and no Chromium was found", "Windows / macOS / a Linux desktop: pip install pywebview; Linux without a desktop toolkit: install chromium (or set VGPT_BROWSER)"))
    return out


def summarize(checks): return {"fail": sum(c["status"] == FAIL for c in checks), "warn": sum(c["status"] == WARN for c in checks), "ok": sum(c["status"] == OK for c in checks)}


def format_report(checks):
    mark = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}; lines = []
    for c in checks:
        lines.append(f"{mark[c['status']]} {c['name']}: {c['detail']}")
        if c["fix"]: lines.append(f"         fix: {c['fix']}")
    s = summarize(checks); lines.append("")
    lines.append("READY" if not s["fail"] else "NOT READY - fix the [FAIL] lines above, then run the check again.")
    if not s["fail"] and s["warn"]: lines.append(f"({s['warn']} warning{'s' if s['warn'] != 1 else ''}: the app runs, with fewer features)")
    return "\n".join(lines)

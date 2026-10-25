"""DEV tool: builds the distributable package.   python tools_release.py [--with-model] [--out dist]

Creates dist/VehicleGPT-<version>/ and dist/VehicleGPT-<version>.zip (+ .sha256) from an ALLOW-LIST - only what the app needs at run time:
the launcher, the vgpt package, the UI, the demo vehicle histories (compacted copies, no write-ahead files, no ground-truth answer keys), the camera demo clip, the documents and the start scripts.
Never included: tests, benchmarks, training tools, the VM harness, the .devenv folder, caches, logs, other people's feedback, absolute paths from this machine.
--with-model also copies the 0.5B language model (Apache-2.0, ~490 MB); without it the app runs on its rules (see USER_GUIDE.md for where to put the model).
After building, `python vehiclegpt.py smoke` is run from the extracted copy with a clean environment: a release that cannot start is not released."""
import argparse, hashlib, json, os, re, shutil, sqlite3, subprocess, sys, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from vgpt.version import VERSION

TOP = ["vehiclegpt.py", "requirements.txt", "Start-VehicleGPT.bat", "Install-VehicleGPT.bat", "start.sh", "install.sh"]
DOCS = ["README.md", "USER_GUIDE.md", "TROUBLESHOOTING.md", "KNOWN_LIMITATIONS.md", "PRIVACY.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "DESIGN.md", "LICENSE.txt"]
UI = ["app.html", "index.html", "server.py"]      # the study pages (research.html, review.html) and its dashboard module (research.py) are not part of a release: the study is off unless switched on (vgpt/routepolicy.py)
DATA_DBS = ["gen_demo.db", "gen_courier.db", "gen_cold.db", "gen_roadwarrior.db", "vehicle.db"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def compact_db(src, dst):
    """A clean single-file copy of a SQLite history: no -wal / -shm, vacuumed, journal mode default."""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True); d = sqlite3.connect(dst); s.backup(d); d.execute("PRAGMA journal_mode=DELETE"); d.execute("VACUUM"); d.commit(); d.close(); s.close()


def build(out, with_model=False, verbose=True):
    name = f"VehicleGPT-{VERSION}"; dst = os.path.join(out, name)
    if os.path.exists(dst): shutil.rmtree(dst)
    os.makedirs(dst)
    def cp(rel, to=None):
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src): raise FileNotFoundError(f"release needs {rel}")
        t = os.path.join(dst, to or rel); os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(src, t)
    for f in TOP + DOCS:
        if os.path.exists(os.path.join(ROOT, f)): cp(f)
        elif f not in ("LICENSE.txt",): raise FileNotFoundError(f"release needs {f}")
    for f in UI: cp(os.path.join("ui", f))
    for f in sorted(os.listdir(os.path.join(ROOT, "ui", "modern"))): cp(os.path.join("ui", "modern", f))      # the modern interface: page, script, stylesheet, fonts, icons and their licences
    for base, dirs, files in os.walk(os.path.join(ROOT, "vgpt")):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith((".pyc", ".pyo")): continue
            cp(os.path.relpath(os.path.join(base, f), ROOT))
    os.makedirs(os.path.join(dst, "data", "ref"), exist_ok=True)
    for db in DATA_DBS:
        if os.path.exists(os.path.join(ROOT, "data", db)): compact_db(os.path.join(ROOT, "data", db), os.path.join(dst, "data", db))
    cp(os.path.join("data", "ref", "camera_clip.hevc"))
    os.makedirs(os.path.join(dst, "models"), exist_ok=True)
    open(os.path.join(dst, "models", "README.txt"), "w").write("Optional: put qwen2.5-0.5b-instruct-q4_k_m.gguf here (Qwen2.5-0.5B-Instruct, Q4_K_M, Apache-2.0) and run: pip install llama-cpp-python\nWithout it VehicleGPT answers with its rules only.\n")
    if with_model:                                                                   # base model + the default meaning adapter + the previous front end's adapter (VGPT_MEANING=0)
        cp(os.path.join("models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")); cp(os.path.join("models", "meaning-lora-full_v11.gguf"))
        if os.path.exists(os.path.join(ROOT, "models", "semantic-lora-qwen25-05b-v2.gguf")): cp(os.path.join("models", "semantic-lora-qwen25-05b-v2.gguf"))
        pj = os.path.join(ROOT, "models", "product.json")                           # the adapter the launcher adopts (frame 2), if one was adopted
        if os.path.exists(pj):
            cp(os.path.join("models", "product.json")); ad = json.load(open(pj, encoding="utf-8")).get("adapter")
            if ad and os.path.exists(os.path.join(ROOT, "models", ad)): cp(os.path.join("models", ad))
    for f in ("deploy/raspberrypi/install-vantage.sh", "deploy/raspberrypi/vehiclegpt-vantage.service.in", "deploy/raspberrypi/README.md", "deploy/update-release.sh", "docs/DEPLOYMENT.md", "docs/SCORECARD.md", "docs/SAFETY_ISOLATION.md", "docs/DATA_SOURCES.md"):
        if os.path.exists(os.path.join(ROOT, f)): cp(f)                              # what installs and updates a device, and what the application claims
    # manifest
    files = []
    for base, dirs, fs in os.walk(dst):
        for f in sorted(fs):
            p = os.path.join(base, f); files.append({"path": os.path.relpath(p, dst).replace(os.sep, "/"), "bytes": os.path.getsize(p), "sha256": sha256(p)})
    json.dump({"name": "VehicleGPT", "version": VERSION, "files": files}, open(os.path.join(dst, "MANIFEST.json"), "w"), indent=1)
    zp = os.path.join(out, name + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, dirs, fs in os.walk(dst):
            for f in sorted(fs): z.write(os.path.join(base, f), os.path.join(name, os.path.relpath(os.path.join(base, f), dst)))
    open(zp + ".sha256", "w").write(f"{sha256(zp)}  {os.path.basename(zp)}\n")
    if verbose: print(f"built {zp}  ({os.path.getsize(zp) / 1e6:.0f} MB, {len(files)} files)")
    return dst, zp


def clean_env():
    keep = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "HOME", "HOMEDRIVE", "HOMEPATH", "USERNAME"}             # what a normal user session has; no PYTHON* variables, no dev settings
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    env["PYTHONDONTWRITEBYTECODE"] = "1"; return env


def verify(dst):
    """Run the shipped self-test from the extracted copy, from a different working directory, with a clean environment and a throw-away home folder."""
    import tempfile
    env = clean_env(); env["VEHICLEGPT_HOME"] = tempfile.mkdtemp(prefix="vgpt_rel_")
    r = subprocess.run([sys.executable, os.path.join(dst, "vehiclegpt.py"), "smoke"], cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, timeout=600)
    return r.returncode == 0, (r.stdout + r.stderr)[-600:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--with-model", action="store_true"); ap.add_argument("--out", default="dist"); ap.add_argument("--no-verify", action="store_true"); a = ap.parse_args()
    dst, zp = build(os.path.join(ROOT, a.out), a.with_model)
    if not a.no_verify:
        ok, tail = verify(dst); print("self-test of the built package:", "PASSED" if ok else "FAILED\n" + tail)
        sys.exit(0 if ok else 1)

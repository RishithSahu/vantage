"""Install the Vantage APP: an icon on the desktop and in the start / applications menu that opens the app window - no terminal, no browser, no address bar, no network.

    python vehiclegpt.py install-app [--kiosk] [--name Vantage] [--remove]

The icon starts `vehiclegpt.py run` (the assistant in its own window, fully offline; `--kiosk` makes it full screen).
Windows: two `.lnk` shortcuts (Desktop, Start menu) that start `pythonw.exe` (no console window), made through PowerShell's WScript.Shell.  Linux: two `.desktop` files (applications menu, Desktop).  macOS: an application
bundle in ~/Applications (Info.plist, an `.icns` icon, a launcher script) - written and its layout tested on Windows, **not run on a Mac**.  Cars run Android or an embedded Linux: they get the Android project (`android/`)
or the Yocto layer (`yocto/`), not these icons (`docs/INSTALLERS.md`).
The icon is drawn by `make_icon` (no image library beyond OpenCV, which the application already needs); the files are written to `ui/modern/vantage.png` and `vantage.ico`."""
import os
import shutil
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_PNG = os.path.join(ROOT, "ui", "modern", "vantage.png"); ICON_ICO = os.path.join(ROOT, "ui", "modern", "vantage.ico")


def make_icon(png=ICON_PNG, ico=ICON_ICO, size=256):
    """A rounded blue square with a white V and a small light on top, as PNG and as a one-image ICO (the PNG inside the ICO is what Windows reads from Vista on)."""
    import cv2
    import numpy as np
    img = np.zeros((size, size, 4), np.uint8); top, bot = np.array([184, 111, 11]), np.array([110, 62, 6])                                   # BGR of #0b6fb8 and a darker blue
    for y in range(size): img[y, :, :3] = (top + (bot - top) * (y / size)).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8); r = size // 5; cv2.rectangle(mask, (r, 0), (size - r, size), 255, -1); cv2.rectangle(mask, (0, r), (size, size - r), 255, -1)
    for cx, cy in ((r, r), (size - r, r), (r, size - r), (size - r, size - r)): cv2.circle(mask, (cx, cy), r, 255, -1)
    img[:, :, 3] = mask; w = max(2, size // 11)
    cv2.polylines(img, [np.array([[size * 0.27, size * 0.30], [size * 0.5, size * 0.74], [size * 0.73, size * 0.30]], np.int32)], False, (255, 255, 255, 255), w, cv2.LINE_AA)
    cv2.circle(img, (int(size * 0.5), int(size * 0.2)), max(3, size // 24), (255, 255, 255, 255), -1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if not ok: raise RuntimeError("could not encode the icon")
    data = buf.tobytes(); os.makedirs(os.path.dirname(png), exist_ok=True); open(png, "wb").write(data)
    open(ico, "wb").write(struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", 0 if size >= 256 else size, 0 if size >= 256 else size, 0, 0, 1, 32, len(data), 22) + data)
    return png, ico


def make_pwa_icons(directory=None):
    """The icons an installable web app asks for: 192 and 512 (any), a 512 maskable one (the picture inside the central 80 %, full bleed, because systems crop it into a circle or a squircle), and the 180 px touch icon of iOS."""
    import cv2
    import numpy as np
    directory = directory or os.path.join(ROOT, "ui", "modern"); out = {}
    for name, size in (("vantage-192.png", 192), ("vantage-512.png", 512)):
        png = os.path.join(directory, name); make_icon(png, os.path.join(directory, "_tmp.ico"), size); out[name] = png
    try: os.remove(os.path.join(directory, "_tmp.ico"))
    except OSError: pass
    size = 512; img = np.zeros((size, size, 3), np.uint8); top, bot = np.array([184, 111, 11]), np.array([110, 62, 6])
    for y in range(size): img[y, :] = (top + (bot - top) * (y / size)).astype(np.uint8)
    w = size // 11; cv2.polylines(img, [np.array([[size * 0.36, size * 0.36], [size * 0.5, size * 0.66], [size * 0.64, size * 0.36]], np.int32)], False, (255, 255, 255), w, cv2.LINE_AA); cv2.circle(img, (size // 2, int(size * 0.27)), size // 30, (255, 255, 255), -1, cv2.LINE_AA)
    p = os.path.join(directory, "vantage-maskable-512.png"); ok, buf = cv2.imencode(".png", img); open(p, "wb").write(buf.tobytes()); out["vantage-maskable-512.png"] = p
    ap = os.path.join(directory, "apple-touch-icon.png"); ok, buf = cv2.imencode(".png", cv2.resize(img, (180, 180), interpolation=cv2.INTER_AREA)); open(ap, "wb").write(buf.tobytes()); out["apple-touch-icon.png"] = ap     # iOS wants an OPAQUE square and rounds it itself
    return out


def make_icns(png=ICON_PNG, icns=None):
    """An Apple icon file with one 256 x 256 PNG image (type `ic08`), which macOS reads."""
    data = open(png, "rb").read(); icns = icns or os.path.splitext(png)[0] + ".icns"
    body = b"ic08" + struct.pack(">I", 8 + len(data)) + data; open(icns, "wb").write(b"icns" + struct.pack(">I", 8 + len(body)) + body); return icns


def macos_bundle(app_dir, name, program, args, icon_png=ICON_PNG):
    """Write `<app_dir>/<name>.app`: Contents/Info.plist, Contents/MacOS/<name> (a shell script that starts the app), Contents/Resources/<name>.icns.  -> the bundle path."""
    import plistlib
    bundle = os.path.join(app_dir, name + ".app"); mac = os.path.join(bundle, "Contents", "MacOS"); res = os.path.join(bundle, "Contents", "Resources"); os.makedirs(mac, exist_ok=True); os.makedirs(res, exist_ok=True)
    make_icns(icon_png, os.path.join(res, name + ".icns"))
    plist = {"CFBundleName": name, "CFBundleDisplayName": name, "CFBundleIdentifier": "local.vantage." + name.lower().replace(" ", "-"), "CFBundleExecutable": name, "CFBundleIconFile": name, "CFBundlePackageType": "APPL",
             "CFBundleVersion": "1", "CFBundleShortVersionString": "1.0", "LSMinimumSystemVersion": "10.13", "NSHighResolutionCapable": True, "LSApplicationCategoryType": "public.app-category.utilities"}
    with open(os.path.join(bundle, "Contents", "Info.plist"), "wb") as f: plistlib.dump(plist, f)
    sh = "#!/bin/sh\ncd " + _shq(ROOT) + " || exit 1\nexec " + " ".join(_shq(a) for a in [program] + args) + "\n"
    launcher = os.path.join(mac, name); open(launcher, "w", encoding="utf-8", newline="\n").write(sh); os.chmod(launcher, 0o755); return bundle


def _shq(t):
    import shlex
    return shlex.quote(t)


def _command(kiosk):
    """The command line the icon runs: (program, arguments list)."""
    py = sys.executable
    if os.name == "nt":
        pw = os.path.join(os.path.dirname(py), "pythonw.exe"); py = pw if os.path.exists(pw) else py                                         # pythonw: no console window behind the app
    script = os.path.join(ROOT, "vehiclegpt.py")
    return py, [script, "run"] + (["--kiosk"] if kiosk else [])


def _windows_shortcut(path, target, args, workdir, icon, desc):
    quote = lambda t: '"' + t + '"' if " " in t else t
    ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:VGPT_LNK);$s.TargetPath=$env:VGPT_TARGET;$s.Arguments=$env:VGPT_ARGS;$s.WorkingDirectory=$env:VGPT_DIR;"
          "$s.IconLocation=$env:VGPT_ICON;$s.Description=$env:VGPT_DESC;$s.WindowStyle=7;$s.Save()")
    env = dict(os.environ, VGPT_LNK=path, VGPT_TARGET=target, VGPT_ARGS=" ".join(quote(a) for a in args), VGPT_DIR=workdir, VGPT_ICON=icon, VGPT_DESC=desc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], env=env, capture_output=True, text=True, timeout=60)
    if p.returncode or not os.path.exists(path): raise RuntimeError("could not create the shortcut: " + (p.stderr or p.stdout).strip()[:200])


def desktop_entry(name, comment, program, args, icon):
    q = lambda t: '"' + t.replace('"', '\\"') + '"' if (" " in t or '"' in t) else t
    return ("[Desktop Entry]\nType=Application\nVersion=1.0\nName=%s\nComment=%s\nExec=%s\nIcon=%s\nTerminal=false\nCategories=Utility;Vehicle;\nStartupWMClass=%s\n" % (name, comment, " ".join(q(x) for x in [program] + args), icon, name))


def locations(name, home=None):
    """-> [paths] of the icons this machine gets (and `kind`: 'windows' or 'linux'), under `home` if given (a test uses a temporary folder)."""
    if sys.platform == "darwin":
        h = home or os.path.expanduser("~"); return "macos", [os.path.join(h, "Applications", name + ".app")]
    if os.name == "nt":
        appdata = os.path.join(home, "AppData", "Roaming") if home else os.environ.get("APPDATA", os.path.expanduser("~\\AppData\\Roaming")); desk = os.path.join(home, "Desktop") if home else os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "Desktop")
        return "windows", [os.path.join(desk, name + ".lnk"), os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", name + ".lnk")]
    h = home or os.path.expanduser("~")
    return "linux", [os.path.join(h, "Desktop", name + ".desktop"), os.path.join(h, ".local", "share", "applications", name.lower().replace(" ", "-") + ".desktop")]


def install(name="Vantage", kiosk=False, remove=False, home=None, say=print):
    """Create (or remove) the icons.  -> the list of paths written (or removed)."""
    kind, paths = locations(name, home)
    if remove:
        gone = []
        for p in paths:
            if os.path.isdir(p): shutil.rmtree(p); gone.append(p)
            elif os.path.exists(p): os.remove(p); gone.append(p)
        say("removed: " + (", ".join(gone) or "nothing was installed")); return gone
    if not (os.path.exists(ICON_PNG) and os.path.exists(ICON_ICO)): make_icon()
    program, args = _command(kiosk); comment = "Vantage: ask your car about its data (fully offline)"
    for p in paths:
        if kind == "macos": macos_bundle(os.path.dirname(p), name, program, args)
        elif kind == "windows": _windows_shortcut(p, program, args, ROOT, ICON_ICO, comment)
        else:
            os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w", encoding="utf-8").write(desktop_entry(name, comment, program, args, ICON_PNG)); os.chmod(p, 0o755)
    say("installed: " + ", ".join(paths) + "\nIt starts: " + " ".join([program] + args)); return paths

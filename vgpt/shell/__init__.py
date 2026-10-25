"""How the interface gets onto a screen with no network.

`run(dispatch, ...)` packs the page (`page.py`), connects it to the program through the in-process bridge (`bridge.py`) and shows it with the first shell that works on this system:

    webview   the operating system's own embedded web view (the `pywebview` package)           Windows, macOS, Linux desktops
    pipe      Chromium driven over two pipes                                                    Linux, Yocto, Raspberry Pi OS

Android has its own shell (a WebView in the Android project, `android/`), which talks to the same bridge protocol."""
import sys

from . import bridge as _bridge, page


def run(dispatch, kiosk=False, kind="auto", title="Vantage"):
    """Show the interface and block until its window is closed.  -> exit code."""
    html = page.load(); b = _bridge.Bridge(dispatch); problems = []
    order = [k for k in ("webview", "pipe") if kind in ("auto", k)]
    try:
        for k in order:
            if k == "webview":
                from . import webview_shell as w
                if not w.available(): problems.append("webview: the pywebview package is not installed"); continue
                return w.run(b, html, title=title, kiosk=kiosk)
            from . import pipe_shell as p
            try: return p.run(b, html, title=title, kiosk=kiosk)
            except RuntimeError as e: problems.append(f"pipe: {e}")
    finally: b.close()
    print("VehicleGPT could not open its window:\n  " + "\n  ".join(problems or ["no window system is available"]), file=sys.stderr)
    return 2

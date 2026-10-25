"""A window made by Chromium driven over two PIPES - no socket, no debugging port, no extra package.  For Linux, Yocto and Raspberry Pi OS, where a Chromium is the usual way an embedded display shows an interface.

Chromium's DevTools protocol can run over a pair of pipes (`--remote-debugging-pipe`: the browser reads file descriptor 3 and writes file descriptor 4; every message is a JSON text ended by a NUL byte).  This module
starts the browser with its pipes, attaches to the page, adds a binding the page calls (`__vgpt_post`) and delivers replies by evaluating `window.__vgpt_reply(...)` in the page.  The page itself is a file of the packed
interface (`vgpt/shell/page.py`).  Everything but `file:`, `data:`, `blob:` and `about:` is failed by the browser's `Fetch` domain, and name lookups are disabled, so even a script that tried could not leave.

`CdpPipe` is plain Python over two descriptors and is tested here with a stand-in browser; `run` needs a POSIX system and a Chromium and has NOT been run in this work (the development PC has neither a Linux nor a
Chromium that accepts pipes: Edge on Windows hands the pipes to a launcher process that exits).  It is the first thing to try on the Pi under Raspberry Pi OS and on a Yocto image with `chromium`."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome", "microsoft-edge")
ALLOWED_SCHEMES = ("file:", "data:", "blob:", "about:")
FLAGS = ["--no-first-run", "--no-default-browser-check", "--noerrdialogs", "--disable-infobars", "--disable-session-crashed-bubble", "--disable-sync", "--disable-background-networking", "--disable-component-update",
         "--disable-domain-reliability", "--disable-client-side-phishing-detection", "--disable-default-apps", "--no-pings", "--disable-breakpad", "--password-store=basic", "--overscroll-history-navigation=0",
         "--disable-pinch", "--use-fake-ui-for-media-stream", "--disable-features=Translate,TranslateUI,MediaRouter,OptimizationHints,AutofillServerCommunication,InterestFeedContentSuggestions",
         "--autoplay-policy=no-user-gesture-required", "--host-resolver-rules=MAP * ~NOTFOUND", "--disable-quic"]


def find_browser():
    env = os.environ.get("VGPT_BROWSER")
    if env: return env if os.path.exists(env) else shutil.which(env)
    for name in BROWSERS:
        p = shutil.which(name)
        if p: return p
    return None


class CdpPipe:
    """The DevTools protocol over two descriptors: `rfd` (messages from the browser) and `wfd` (messages to it)."""
    def __init__(self, rfd, wfd):
        self.rfd, self.wfd = rfd, wfd; self._id = 0; self._lock = threading.Lock(); self._wait = {}; self._handlers = {}; self.closed = threading.Event(); self._buf = b""
        self._t = threading.Thread(target=self._read, daemon=True, name="cdp-pipe"); self._t.start()

    def on(self, method, fn): self._handlers[method] = fn

    def send(self, method, params=None, session=None, _waiter=None):
        """Send a command and return its id.  The waiter, when there is one, is registered under the same lock that hands out the id, so an answer can never be matched to the wrong command."""
        with self._lock:
            self._id += 1; i = self._id; msg = {"id": i, "method": method}
            if params: msg["params"] = params
            if session: msg["sessionId"] = session
            if _waiter is not None: self._wait[i] = _waiter
            try: os.write(self.wfd, json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\0")
            except OSError: self.closed.set()
        return i

    def request(self, method, params=None, session=None, timeout=15):
        ev = threading.Event(); box = []
        self.send(method, params, session, _waiter=(ev, box))
        if not ev.wait(timeout): raise TimeoutError(f"{method}: the browser did not answer")
        r = box[0]
        if "error" in r: raise RuntimeError(f"{method}: {r['error'].get('message')}")
        return r.get("result", {})

    def _read(self):
        while True:
            try: chunk = os.read(self.rfd, 65536)
            except OSError: break
            if not chunk: break
            self._buf += chunk
            while b"\0" in self._buf:
                raw, self._buf = self._buf.split(b"\0", 1)
                try: m = json.loads(raw)
                except ValueError: continue
                if "id" in m:
                    with self._lock: w = self._wait.pop(m["id"], None)
                    if w: w[1].append(m); w[0].set()
                elif m.get("method") in self._handlers:
                    try: self._handlers[m["method"]](m.get("params") or {}, m.get("sessionId"))
                    except Exception: pass
        self.closed.set()
        with self._lock: waiting = list(self._wait.values()); self._wait.clear()
        for ev, box in waiting: box.append({"error": {"message": "the browser closed"}}); ev.set()


def attach(cdp, bridge, page_url, timeout=20):
    """Attach to the browser's page, add the binding, fence off the network, load the page.  Returns the session id."""
    end = time.monotonic() + timeout; target = None
    while time.monotonic() < end and target is None:
        for t in cdp.request("Target.getTargets").get("targetInfos", []):
            if t.get("type") == "page": target = t["targetId"]; break
        else: time.sleep(0.1)
    if target is None: raise RuntimeError("the browser opened no window")
    sid = cdp.request("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]

    def binding(params, session):
        if params.get("name") == "__vgpt_post": bridge.post(params.get("payload", ""))

    def paused(params, session):                                                          # every request the page tries: only our own kinds of address may pass
        url = (params.get("request") or {}).get("url", "")
        if url.startswith(ALLOWED_SCHEMES): cdp.send("Fetch.continueRequest", {"requestId": params["requestId"]}, session)
        else: cdp.send("Fetch.failRequest", {"requestId": params["requestId"], "errorReason": "BlockedByClient"}, session)
    cdp.on("Runtime.bindingCalled", binding); cdp.on("Fetch.requestPaused", paused)
    cdp.request("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}, sid)
    cdp.request("Runtime.enable", None, sid); cdp.request("Runtime.addBinding", {"name": "__vgpt_post"}, sid); cdp.request("Page.enable", None, sid)
    bridge.attach(lambda text: cdp.send("Runtime.evaluate", {"expression": "window.__vgpt_reply(" + text + ")"}, sid))
    cdp.request("Page.navigate", {"url": page_url}, sid)
    return sid


def browser_flags(work, kiosk=False, root=None):
    """The command-line flags of the window: an application window (no address bar, no tabs) with its own private profile, the DevTools protocol over pipes, every outside address unreachable."""
    flags = ["--app=about:blank", f"--user-data-dir={os.path.join(work, 'profile')}", "--remote-debugging-pipe", *FLAGS]
    flags += ["--kiosk", "--start-fullscreen"] if kiosk else ["--window-size=1280,780"]
    if (hasattr(os, "geteuid") and os.geteuid() == 0) if root is None else root: flags.append("--no-sandbox")         # a kiosk that runs as root has no user namespace to sandbox in
    extra = os.environ.get("VGPT_BROWSER_FLAGS")
    if extra: flags += extra.split()
    return flags


def run(bridge, html, title="Vantage", kiosk=False, browser=None):
    if os.name == "nt": raise RuntimeError("the pipe shell is for Linux and other POSIX systems (on Windows use the web view shell)")
    browser = browser or find_browser()
    if not browser: raise RuntimeError("no Chromium found: install chromium (sudo apt install chromium, or the chromium package of the image) or set VGPT_BROWSER")
    work = tempfile.mkdtemp(prefix="vantage_ui_"); page = os.path.join(work, "vantage.html")
    with open(page, "w", encoding="utf-8", newline="") as f: f.write(html)
    r_in, w_in = os.pipe(); r_out, w_out = os.pipe()                                      # we write to w_in (the browser reads r_in as fd 3), the browser writes w_out (as fd 4) and we read r_out
    flags = browser_flags(work, kiosk)
    cmd = ["/bin/sh", "-c", 'exec "$@" 3<&%d 4>&%d' % (r_in, w_out), "vantage-shell", browser, *flags]
    proc = subprocess.Popen(cmd, pass_fds=(r_in, w_out), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(r_in); os.close(w_out)
    cdp = CdpPipe(r_out, w_in)
    try:
        attach(cdp, bridge, "file://" + page)
        while proc.poll() is None and not cdp.closed.is_set(): time.sleep(0.5)
    finally:
        try: proc.terminate()
        except OSError: pass
        shutil.rmtree(work, ignore_errors=True)
    return 0

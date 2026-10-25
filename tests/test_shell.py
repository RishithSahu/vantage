"""How the interface reaches a screen with no network: the packed page, the in-process bridge (both ends), the Chromium-over-pipes protocol (with a stand-in browser) and the embedded-web-view shell (with a stand-in package).

What is NOT shown here: a real Chromium on Linux driven over pipes, and the real pywebview on Windows (`docs/INSTALLERS.md` says which of those were run)."""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.shell import bridge as B, page, pipe_shell, webview_shell  # noqa: E402
import vgpt.shell as shell  # noqa: E402


def resp(status=200, body=b"{}", headers=None): return SimpleNamespace(status=status, headers=headers or [("Content-Type", "application/json")], body=body)


class Collect:
    """What a shell delivers to the page, gathered."""
    def __init__(self): self.items = []; self.ev = threading.Event()
    def __call__(self, text): self.items.append(json.loads(text)); self.ev.set()
    def wait(self, n=1, seconds=5):
        end = time.monotonic() + seconds
        while len(self.items) < n and time.monotonic() < end: time.sleep(0.01)
        return self.items


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ThePackedPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.html = page.build()

    def test_it_asks_for_nothing_by_address(self):
        import re
        self.assertEqual(sorted(set(re.findall(r"https?://[^\s\"'<>)]+", self.html))), ["http://www.w3.org/2000/svg"], "the only address left is an XML namespace name, which is never requested")
        self.assertNotIn("/modern/", self.html); self.assertEqual(re.findall(r"""(?:src|href)\s*=\s*["'](?!#|data:)[^"']""", self.html), [])

    def test_a_policy_forbids_every_connection_and_the_bridge_comes_before_any_script_of_the_page(self):
        self.assertIn("connect-src 'none'", self.html); self.assertIn("default-src 'none'", self.html); self.assertIn("frame-src 'none'", self.html); self.assertIn("object-src 'none'", self.html)
        self.assertLess(self.html.index("window.__vgpt_bridge"), self.html.index("const ic = ")); self.assertLess(self.html.index("Content-Security-Policy"), self.html.index("window.__vgpt_bridge"))
        self.assertLess(self.html.index('<meta charset'), 1024, "the character set must be declared in the first kilobyte")

    def test_fonts_icons_scripts_and_the_worklet_are_inside(self):
        self.assertEqual(self.html.count("url(data:font/woff2;base64,"), 5); self.assertIn('<svg style="display:none" aria-hidden="true"', self.html); self.assertIn("window.__VGPT_WORKLET = ", self.html)
        self.assertGreater(self.html.count('href="#i-'), 20); self.assertIn("registerProcessor", self.html)

    def test_it_is_small_enough_for_a_web_view_to_take_as_text(self): self.assertLess(len(self.html.encode("utf-8")), 1_500_000, "WebView2 accepts about 2 MB of page text")

    def test_the_page_script_has_no_dynamic_code(self):
        for bad in ("eval(", "new Function(", "document.write(", "importScripts("): self.assertNotIn(bad, self.html.replace("new Function()", ""), bad)

    def test_it_is_rebuilt_when_a_source_is_newer_and_reused_when_not(self):
        d = tempfile.mkdtemp(); mod = os.path.join(d, "modern"); shutil.copytree(os.path.join(ROOT, "ui", "modern"), mod, ignore=shutil.ignore_patterns("dist"))
        with mock.patch.object(page, "MODERN", mod), mock.patch.object(page, "DIST", os.path.join(mod, "dist", "vantage.html")):
            first = page.load(); self.assertTrue(os.path.exists(page.DIST))
            with open(page.DIST, "a", encoding="utf-8") as f: f.write("<!-- marker -->")
            self.assertIn("<!-- marker -->", page.load(), "a prebuilt page that is current is used as it is")
            time.sleep(0.05); os.utime(os.path.join(mod, "app.js"), None); self.assertNotIn("<!-- marker -->", page.load(), "a newer source makes it rebuild")
            self.assertEqual(page.load(), first)
        shutil.rmtree(d, ignore_errors=True)

    def test_a_page_that_would_still_load_something_by_address_is_refused_at_build_time(self):
        d = tempfile.mkdtemp(); mod = os.path.join(d, "modern"); shutil.copytree(os.path.join(ROOT, "ui", "modern"), mod, ignore=shutil.ignore_patterns("dist"))
        with open(os.path.join(mod, "index.html"), "a", encoding="utf-8") as f: f.write('<img src="/modern/leak.png"><script src="/modern/other.js"></script>')
        with self.assertRaises(ValueError) as e: page.build(mod)
        self.assertIn("/modern/", str(e.exception)); shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TheBridgeOfTheProgram(unittest.TestCase):
    def make(self, dispatch):
        b = B.Bridge(dispatch); c = Collect(); b.attach(c); return b, c

    def test_a_request_is_answered_through_the_dispatcher_and_the_body_travels_as_base64(self):
        seen = []
        def d(m, p, h, body): seen.append((m, p, h, body)); return resp(200, b'{"ok": true, "\xc3\xa9": 1}', [("Content-Type", "application/json"), ("Set-Cookie", "vgpt_s=abc")])
        b, c = self.make(d); self.assertTrue(b.post(json.dumps({"id": 7, "m": "POST", "p": "/api/ask?x=1", "h": {"Cookie": "vgpt_s=q"}, "b": '{"q":"hi"}'}))); r = c.wait()[0]
        self.assertEqual(seen, [("POST", "/api/ask?x=1", {"Cookie": "vgpt_s=q"}, '{"q":"hi"}')]); self.assertEqual((r["id"], r["s"]), (7, 200)); self.assertEqual(base64.b64decode(r["b"]), b'{"ok": true, "\xc3\xa9": 1}'); self.assertIn(["Set-Cookie", "vgpt_s=abc"], r["h"]); b.close()

    def test_nothing_that_is_not_well_formed_reaches_the_dispatcher(self):
        called = []; b, c = self.make(lambda *a: called.append(a) or resp())
        for bad in ("not json", "[]", "7", "null", json.dumps({"m": "GET", "p": "/x"}), json.dumps({"id": True, "m": "GET", "p": "/x"}), json.dumps({"id": "7", "m": "GET", "p": "/x"}), "x" * (B.MAX_MESSAGE + 1), "[" * 100000): self.assertFalse(b.post(bad), bad[:30])
        self.assertFalse(b.post(None)); self.assertEqual(c.items, [], "without an id there is nobody to answer")
        for bad in ({"id": 1, "m": "DELETE", "p": "/x"}, {"id": 2, "m": "GET", "p": "http://evil/"}, {"id": 3, "m": "GET", "p": "no-slash"}, {"id": 4, "m": "GET", "p": "/x\r\nHost: evil"}, {"id": 5, "m": "GET", "p": "/x", "h": {"A": 1}},
                    {"id": 6, "m": "GET", "p": "/x", "h": {str(i): "v" for i in range(40)}}, {"id": 7, "m": "POST", "p": "/x", "b": 5}, {"id": 8, "m": "GET", "p": "/" + "a" * 3000}): self.assertFalse(b.post(json.dumps(bad)), bad)
        rep = c.wait(8); self.assertEqual(sorted(r["id"] for r in rep), [1, 2, 3, 4, 5, 6, 7, 8]); self.assertTrue(all(r["s"] == 400 for r in rep)); self.assertEqual(called, []); self.assertEqual(b.refused, 18); b.close()

    def test_a_failing_dispatcher_gives_a_500_that_says_nothing_about_the_inside(self):
        def d(*a): raise RuntimeError("secret internals: C:\\Users\\x\\vault.key")
        b, c = self.make(d); b.post(json.dumps({"id": 1, "m": "GET", "p": "/api/x"})); r = c.wait()[0]; self.assertEqual(r["s"], 500); body = base64.b64decode(r["b"]).decode(); self.assertNotIn("secret", body); self.assertNotIn("vault", body); b.close()

    def test_many_requests_at_once_each_get_their_own_answer(self):
        def d(m, p, h, body): time.sleep(0.01 * (int(p[1:]) % 3)); return resp(200, p.encode())
        b, c = self.make(d)
        for i in range(60): b.post(json.dumps({"id": i, "m": "GET", "p": f"/{i}"}))
        rep = c.wait(60, 10); self.assertEqual(len(rep), 60)
        for r in rep: self.assertEqual(base64.b64decode(r["b"]).decode(), f"/{r['id']}")
        b.close()

    def test_a_window_that_has_gone_does_not_break_the_answer_thread(self):
        def boom(text): raise RuntimeError("the window is closed")
        b = B.Bridge(lambda *a: resp()); b.attach(boom); b.post(json.dumps({"id": 1, "m": "GET", "p": "/api/x"})); time.sleep(0.2); self.assertEqual(b.answered, 1); b.close()

    def test_the_real_interface_answers_through_the_bridge(self):
        """The whole chain with the real dispatcher: bridge -> dispatcher -> the application's routes (no model needed for the unknown route and the method check)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("uiserver_bridge", os.path.join(ROOT, "ui", "server.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        d = m.make_dispatcher(SimpleNamespace(role_of=lambda tok: "driver", last_seen=0)); b, c = self.make(d)
        b.post(json.dumps({"id": 1, "m": "GET", "p": "/nothing/here"})); b.post(json.dumps({"id": 2, "m": "POST", "p": "/api/ask", "b": "{" * 70000})); rep = {r["id"]: r for r in c.wait(2)}
        self.assertEqual(rep[1]["s"], 404); self.assertEqual(rep[2]["s"], 413); b.close()


@unittest.skipUnless(shutil.which("node"), "needs node (a JavaScript engine) to run the page's bridge script")
class ThePagesBridgeScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = subprocess.run(["node", os.path.join(ROOT, "tests", "js", "bridge_driver.js")], capture_output=True, text=True, timeout=60, encoding="utf-8")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]; cls.out = json.loads(line[0][7:]) if line else {"crashed": p.stderr[-800:]}

    def test_it_ran(self): self.assertNotIn("crashed", self.out, self.out.get("crashed"))

    def test_fetch_of_the_programs_own_routes_goes_through_the_bridge_and_comes_back_as_a_response(self):
        o = self.out; self.assertEqual((o["get"]["status"], o["get"]["ok"], o["get"]["body"]["echo"]), (200, True, "/api/health")); self.assertEqual((o["get"]["sent"]["m"], o["get"]["sent"]["b"]), ("GET", ""))
        self.assertEqual((o["post"]["sent"]["m"], o["post"]["sent"]["b"]), ("POST", '{"q":"hi"}')); self.assertEqual(o["refused"], {"status": 403, "ok": False})

    def test_the_role_session_lives_in_the_pages_memory_and_goes_with_each_request(self):
        self.assertEqual(self.out["cookie_after_login"], "vgpt_s=tok123"); self.assertIsNone(self.out["cookie_after_logout"])

    def test_empty_and_binary_replies_are_real_responses(self):
        self.assertEqual(self.out["no_content"], {"status": 204, "body_is_null": True}); self.assertEqual(self.out["binary"], {"status": 200, "type": "image/jpeg", "first": [255, 216], "length": 7})

    def test_every_other_address_is_refused_and_nothing_is_sent_for_it(self):
        self.assertEqual(set(self.out["rejected"].values()), {"OfflineError"}); self.assertEqual(self.out["no_extra_traffic_for_rejected"], 9, "only the nine legitimate requests were passed on")

    def test_the_other_ways_a_page_could_reach_out_are_switched_off(self):
        self.assertEqual(self.out["classes"], {"WebSocket": "OfflineError", "XMLHttpRequest": "OfflineError", "EventSource": "OfflineError"}); self.assertEqual(self.out["beacon"], {"own": True, "other": False}); self.assertEqual(self.out["beacon_sent"], 1)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class FakeBrowser(threading.Thread):
    """Speaks the DevTools protocol over two pipes the way a Chromium would: answers the commands the shell sends, and (when told) the page calls its binding and the network wants out."""
    def __init__(self, rfd, wfd):
        super().__init__(daemon=True); self.rfd, self.wfd = rfd, wfd; self.methods = []; self.sent = []; self.evals = []; self.navigated = None; self.verdicts = {}; self.buf = b""; self.stop = False

    def emit(self, obj):
        try: os.write(self.wfd, json.dumps(obj).encode() + b"\0")
        except OSError: pass                                                                       # the test closed this end on purpose

    def run(self):
        while not self.stop:
            try: chunk = os.read(self.rfd, 65536)
            except OSError: return
            if not chunk: return
            self.buf += chunk
            while b"\0" in self.buf:
                raw, self.buf = self.buf.split(b"\0", 1); m = json.loads(raw); self.methods.append(m["method"]); self.sent.append(m); res = {}
                if m["method"] == "Target.getTargets": res = {"targetInfos": [{"targetId": "T1", "type": "browser"}, {"targetId": "T2", "type": "page"}]}
                elif m["method"] == "Target.attachToTarget": res = {"sessionId": "S1"}
                elif m["method"] == "Page.navigate": self.navigated = m["params"]["url"]
                elif m["method"] == "Runtime.evaluate": self.evals.append(m["params"]["expression"])
                elif m["method"] in ("Fetch.continueRequest", "Fetch.failRequest"): self.verdicts[m["params"]["requestId"]] = (m["method"], m["params"].get("errorReason"))
                if "id" in m and m["method"] not in ("Fetch.continueRequest", "Fetch.failRequest"): self.emit({"id": m["id"], "result": res})
                elif "id" in m: self.emit({"id": m["id"], "result": {}})


class TheChromiumPipeProtocol(unittest.TestCase):
    def setUp(self):
        r1, w1 = os.pipe(); r2, w2 = os.pipe(); self.fb = FakeBrowser(r1, w2); self.fb.start(); self.cdp = pipe_shell.CdpPipe(r2, w1); self.fds = {"r1": r1, "w1": w1, "r2": r2, "w2": w2}

    def close(self, name):
        fd = self.fds.pop(name, None)
        if fd is not None:
            try: os.close(fd)
            except OSError: pass

    def tearDown(self):
        # writers first, so the two reading threads see end-of-file and leave; closing a pipe end that a thread is blocked reading can deadlock on Windows
        self.fb.stop = True; self.close("w1"); self.close("w2"); self.fb.join(3); self.cdp._t.join(3); self.close("r1"); self.close("r2")

    def test_the_shell_attaches_adds_the_binding_fences_the_network_and_loads_the_page(self):
        b = B.Bridge(lambda *a: resp(200, b"pong")); sid = pipe_shell.attach(self.cdp, b, "file:///tmp/vantage.html"); self.assertEqual(sid, "S1")
        self.assertEqual(self.fb.methods, ["Target.getTargets", "Target.attachToTarget", "Fetch.enable", "Runtime.enable", "Runtime.addBinding", "Page.enable", "Page.navigate"])
        attach = next(m for m in self.fb.sent if m["method"] == "Target.attachToTarget"); self.assertEqual(attach["params"], {"targetId": "T2", "flatten": True}, "the page target, not the browser target")
        self.assertEqual(next(m for m in self.fb.sent if m["method"] == "Fetch.enable")["params"], {"patterns": [{"urlPattern": "*"}]}); self.assertEqual(next(m for m in self.fb.sent if m["method"] == "Runtime.addBinding")["params"], {"name": "__vgpt_post"})
        self.assertTrue(all(m.get("sessionId") == "S1" for m in self.fb.sent if m["method"] in ("Fetch.enable", "Runtime.addBinding", "Page.navigate"))); self.assertEqual(self.fb.navigated, "file:///tmp/vantage.html"); b.close()

    def test_a_call_of_the_page_reaches_the_bridge_and_the_reply_is_evaluated_in_the_page(self):
        b = B.Bridge(lambda m, p, h, body: resp(200, ("echo " + p).encode())); pipe_shell.attach(self.cdp, b, "file:///x.html")
        self.fb.emit({"method": "Runtime.bindingCalled", "sessionId": "S1", "params": {"name": "__vgpt_post", "executionContextId": 1, "payload": json.dumps({"id": 41, "m": "GET", "p": "/api/health"})}})
        end = time.monotonic() + 5
        while not self.fb.evals and time.monotonic() < end: time.sleep(0.01)
        self.assertTrue(self.fb.evals); expr = self.fb.evals[0]; self.assertTrue(expr.startswith("window.__vgpt_reply(") and expr.endswith(")")); r = json.loads(expr[len("window.__vgpt_reply("):-1])
        self.assertEqual((r["id"], r["s"]), (41, 200)); self.assertEqual(base64.b64decode(r["b"]), b"echo /api/health")
        self.fb.emit({"method": "Runtime.bindingCalled", "sessionId": "S1", "params": {"name": "some_other_binding", "payload": json.dumps({"id": 42, "m": "GET", "p": "/api/x"})}}); time.sleep(0.2); self.assertEqual(len(self.fb.evals), 1, "only our own binding is answered"); b.close()

    def test_every_request_of_the_page_is_failed_unless_it_is_a_file_data_blob_or_about_address(self):
        b = B.Bridge(lambda *a: resp()); pipe_shell.attach(self.cdp, b, "file:///x.html")
        for i, url in enumerate(("file:///tmp/vantage.html", "data:text/plain,hi", "blob:null/abc", "about:blank", "https://example.com/x", "http://127.0.0.1:8765/api/health", "ws://evil/", "ftp://evil/f", "chrome://settings")):
            self.fb.emit({"method": "Fetch.requestPaused", "sessionId": "S1", "params": {"requestId": f"r{i}", "request": {"url": url}}})
        end = time.monotonic() + 5
        while len(self.fb.verdicts) < 9 and time.monotonic() < end: time.sleep(0.01)
        kinds = [self.fb.verdicts[f"r{i}"] for i in range(9)]; self.assertEqual([k[0] for k in kinds[:4]], ["Fetch.continueRequest"] * 4); self.assertEqual(kinds[4:], [("Fetch.failRequest", "BlockedByClient")] * 5); b.close()

    def test_a_browser_that_closes_ends_the_waiting_calls_instead_of_hanging(self):
        self.close("w2")                                                                             # the browser's write end closes: end of file
        with self.assertRaises(Exception): self.cdp.request("Target.getTargets", timeout=3)
        self.assertTrue(self.cdp.closed.wait(3))

    def test_no_browser_is_a_clear_message_and_windows_is_refused(self):
        with mock.patch.object(pipe_shell, "find_browser", return_value=None), mock.patch.object(pipe_shell.os, "name", "posix"), self.assertRaises(RuntimeError) as e: pipe_shell.run(B.Bridge(lambda *a: resp()), "<html>")
        self.assertIn("no Chromium found", str(e.exception))
        with mock.patch.object(pipe_shell.os, "name", "nt"), self.assertRaises(RuntimeError) as e: pipe_shell.run(B.Bridge(lambda *a: resp()), "<html>")
        self.assertIn("web view", str(e.exception))

    def test_the_browser_is_found_by_its_usual_names_or_by_the_variable(self):
        with mock.patch.dict(os.environ, {"VGPT_BROWSER": ""}), mock.patch.object(pipe_shell.shutil, "which", side_effect=lambda n: "/usr/bin/" + n if n == "chromium-browser" else None): self.assertEqual(pipe_shell.find_browser(), "/usr/bin/chromium-browser")
        with mock.patch.dict(os.environ, {"VGPT_BROWSER": sys.executable}): self.assertEqual(pipe_shell.find_browser(), sys.executable)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TheEmbeddedWebViewShell(unittest.TestCase):
    def fake_webview(self, script):
        windows = []
        class Win:
            def __init__(self, **kw): self.kw = kw; self.js = []
            def evaluate_js(self, js): self.js.append(js)
        def create_window(title, html=None, js_api=None, fullscreen=False, width=0, height=0, min_size=None, text_select=False):
            w = Win(title=title, html=html, js_api=js_api, fullscreen=fullscreen); windows.append(w); return w
        def start(debug=False, private_mode=True): script(windows[0])
        return SimpleNamespace(create_window=create_window, start=start), windows

    def test_the_page_is_given_as_text_the_page_may_call_only_post_and_the_reply_comes_back_by_evaluating_script(self):
        def script(w):
            w.kw["js_api"].post(json.dumps({"id": 5, "m": "GET", "p": "/api/health"})); end = time.monotonic() + 5
            while not w.js and time.monotonic() < end: time.sleep(0.01)
        fake, windows = self.fake_webview(script); b = B.Bridge(lambda *a: resp(200, b"{}"))
        with mock.patch.dict(sys.modules, {"webview": fake}): rc = webview_shell.run(b, "<!doctype html><html></html>", title="Vantage", kiosk=True)
        w = windows[0]; self.assertEqual(rc, 0); self.assertTrue(w.kw["html"].startswith("<!doctype html>"), "text, not an address: no file server is started"); self.assertTrue(w.kw["fullscreen"])
        self.assertEqual([n for n in dir(w.kw["js_api"]) if not n.startswith("_")], ["post"], "everything public on this object is callable from the page"); self.assertTrue(w.js[0].startswith("window.__vgpt_reply(")); b.close()

    def test_the_package_is_optional(self):
        with mock.patch("importlib.util.find_spec", return_value=None): self.assertFalse(webview_shell.available())


class ChoosingTheShell(unittest.TestCase):
    def test_with_neither_shell_available_it_says_what_is_missing_and_gives_exit_code_2(self):
        import io
        buf = io.StringIO()
        with mock.patch.object(webview_shell, "available", return_value=False), mock.patch.object(pipe_shell, "run", side_effect=RuntimeError("no Chromium found: install chromium")), mock.patch.object(page, "load", return_value="<html>"), \
             mock.patch.object(sys, "stderr", buf): rc = shell.run(lambda *a: resp())
        said = buf.getvalue(); self.assertEqual(rc, 2); self.assertIn("pywebview", said); self.assertIn("no Chromium found", said)

    def test_the_web_view_is_tried_first_and_a_requested_kind_is_the_only_one_tried(self):
        order = []
        with mock.patch.object(webview_shell, "available", return_value=True), mock.patch.object(webview_shell, "run", side_effect=lambda *a, **k: order.append("webview") or 0), \
             mock.patch.object(pipe_shell, "run", side_effect=lambda *a, **k: order.append("pipe") or 0), mock.patch.object(page, "load", return_value="<html>"):
            self.assertEqual(shell.run(lambda *a: resp()), 0); self.assertEqual(order, ["webview"]); order.clear(); self.assertEqual(shell.run(lambda *a: resp(), kind="pipe"), 0); self.assertEqual(order, ["pipe"])


if __name__ == "__main__":
    unittest.main()

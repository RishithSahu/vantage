"""A DEVELOPER harness: show the product's page in an ordinary browser, to test every button by hand or by script.

    python bench/dev_web.py [--port 8794] [the product's own options: --db data/world_aster.db --live-car auto ...]

The product itself has no HTTP server and opens no socket (it is fully offline).  This script is NOT part of the product and is never packed: it starts a loopback-only server that gives the browser the packed page
(the same text the app window shows, minus the content-security policy that forbids connections) and carries the page's bridge messages to the very same dispatcher the window uses (`ui.server.make_dispatcher`)."""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.addaudithook = lambda hook: None                          # this harness needs a loopback socket (the product never does): the offline guard is an audit hook, so it is not installed here
from vgpt import applog                                    # noqa: E402
from vgpt.shell import bridge as _bridge, page            # noqa: E402
from ui import server as S                                 # noqa: E402

SHIM = """<script>(()=>{const f=window.fetch.bind(window);window.__vgpt_post=s=>{f('/__post',{method:'POST',body:s}).then(r=>r.text()).then(t=>window.__vgpt_reply(JSON.parse(t)));};})();</script>"""


def main():
    ap = argparse.ArgumentParser(add_help=False); ap.add_argument("--port", type=int, default=8794); known, rest = ap.parse_known_args()
    a = S.build_parser().parse_args(rest); applog.setup(); app = S.App(a); dispatch = S.make_dispatcher(app); lock = threading.Lock()
    html = page.load(); i = html.index('<meta http-equiv="Content-Security-Policy"'); j = html.index(">", i) + 1; html = html[:i] + SHIM + html[j:]

    def answer(text):
        b = _bridge.Bridge(dispatch, workers=1); out = []; b.attach(out.append); msg = b._parse(text)
        if msg is None: return json.dumps({"id": 0, "s": 400, "h": [], "b": ""})
        if "error" in msg: b.post(text); b.close(); return out[0] if out else "{}"
        b._answer(msg); b.close(); return out[0]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def do_GET(self):
            body = html.encode("utf-8") if self.path in ("/", "/index.html") else b""
            self.send_response(200 if body else 404); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0); text = self.rfile.read(n).decode("utf-8", "replace"); body = answer(text).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", known.port), H); print(f"dev page on http://127.0.0.1:{known.port}/  (db {a.db})", flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()

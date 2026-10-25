"""Layout check of the interpretation-review page WITHOUT a language model: the web app with the scripted test double of tests/stub_planner.py (it maps a question to a plan by a stable hash - it understands nothing). For looking at the page only; the real prototype is `python ui/server.py` and then /review.
    python bench/serve_review_stub.py [--port 8777]"""
import argparse, importlib.util, os, sys, tempfile, threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("VEHICLEGPT_HOME", tempfile.mkdtemp())
from tests.stub_planner import StubPlanner

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8777); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("uiserver_stub", os.path.join(ROOT, "ui", "server.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
script = {"how many hard brakes did i have yesterday": ("count_events", {"turn": "new", "event": "HARSH_BRAKE", "period": "yesterday"}), "highest battery temperature last week while charging": ("extreme", {"turn": "new", "signal": "battery_temperature", "which": "max", "period": "last week", "conditions": ["charging"]})}
app = m.App(SimpleNamespace(db="data/gen_courier.db", model="none", camera="clip", webcam_index=0, port=a.port), planner_factory=lambda conn: StubPlanner(conn, script))
srv = ThreadingHTTPServer(("127.0.0.1", a.port), m.make_handler(app)); print("review page: http://127.0.0.1:%d/review" % a.port, flush=True); srv.serve_forever()

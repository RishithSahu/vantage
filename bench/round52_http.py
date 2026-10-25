"""Ask every Round-52 question through the running application's HTTP interface (/api/ask), exactly as the head-unit browser does, against the replay database.
    python bench/round52_http.py [--db data/vehicle.db] [--out results/round52/http_answers.json]
It starts the real server (ui/server.py) on a random local port with the rules-only assistant, so the whole stack - snapshot lease, brain, composed/temporal layer - is exercised."""
import argparse, importlib.util, json, os, sys, tempfile, threading, time, urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--db", default="data/vehicle.db"); ap.add_argument("--out", default=os.path.join(HERE, "results", "round52", "http_answers.json")); a = ap.parse_args()
os.chdir(HERE)
spec = importlib.util.spec_from_file_location("uiserver", os.path.join(HERE, "ui", "server.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
app = m.App(SimpleNamespace(db=a.db, model="none", camera="clip", webcam_index=0, live=os.path.join(tempfile.mkdtemp(), "live.db")))
srv = ThreadingHTTPServer(("127.0.0.1", 0), m.make_handler(app)); port = srv.server_address[1]; app.port = port
threading.Thread(target=srv.serve_forever, daemon=True).start()
def get(p): return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{p}", timeout=20).read())
def post(p, b): return json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{p}", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=120).read())
t0 = time.time()
while time.time() - t0 < 30:
    try:
        if get("/api/state")["tele"]["ok"]: break
    except Exception: pass
    time.sleep(0.5)
rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(HERE, "bench", "round52_questions.tsv"), encoding="utf8") if l.strip()]
out = []
for qid, q in rows:
    r = post("/api/ask", {"q": q})
    out.append({"id": qid, "q": q, "route": r.get("route"), "path": r.get("path"), "intent": r.get("intent"), "answer": r.get("answer"), "ms": (r.get("ms") or {}).get("total")})
    print(f"{qid:4s} [{r.get('path')}/{r.get('route')}] {r.get('ms', {}).get('total') if isinstance(r.get('ms'), dict) else ''} ms :: {(r.get('answer') or '')[:200].replace(chr(10), ' / ')}")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
print(f"{len(out)} questions answered over HTTP; latency max {max((o['ms'] or 0) for o in out)} ms")

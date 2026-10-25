"""The program's end of the page's bridge.

The page (`ui/modern/bridge.js`) passes each request as one JSON text; a shell (a native web view, a Chromium window driven over pipes) hands that text to `Bridge.post`, which answers on a worker thread by calling the
interface's dispatcher (`ui/server.py: make_dispatcher`) and gives the reply text back to the shell to deliver.  No socket, no address, no port: a function call and a queue.

Everything that arrives is untrusted input from a page (which could be showing anything): it is checked before it is used, and a request that is not well formed gets a 400 or is dropped, never an exception."""
import base64
import json
from concurrent.futures import ThreadPoolExecutor

MAX_MESSAGE = 1_200_000                                                                   # characters of one request (the interface's own limit on a request body is 64 KB; this leaves room for headers)
METHODS = ("GET", "POST")


class Bridge:
    def __init__(self, dispatch, workers=6):
        self.dispatch = dispatch; self._send = None; self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bridge"); self.answered = 0; self.refused = 0

    def attach(self, send_reply):
        """`send_reply(text)`: how the shell delivers a reply text to the page (any thread)."""
        self._send = send_reply

    def post(self, text):
        """The text the page sent.  Returns at once; the reply is delivered later through the shell."""
        msg = self._parse(text)
        if msg is None: self.refused += 1; return False
        if "error" in msg:
            self.refused += 1; self._deliver({"id": msg["id"], "s": 400, "h": [["Content-Type", "application/json"]], "b": base64.b64encode(json.dumps({"ok": False, "error": msg["error"]}).encode("utf-8")).decode("ascii")}); return False
        self._pool.submit(self._answer, msg); return True

    @staticmethod
    def _parse(text):
        if not isinstance(text, str) or len(text) > MAX_MESSAGE: return None
        try: m = json.loads(text)
        except (ValueError, RecursionError): return None
        if not isinstance(m, dict) or not isinstance(m.get("id"), int) or isinstance(m.get("id"), bool): return None                     # without an id there is nobody to answer
        out = {"id": m["id"]}
        if m.get("m") not in METHODS: return {"id": m["id"], "error": "bad method"}
        p = m.get("p")
        if not isinstance(p, str) or not p.startswith("/") or len(p) > 2048 or "\x00" in p or "\r" in p or "\n" in p: return {"id": m["id"], "error": "bad path"}
        h = m.get("h", {})
        if not isinstance(h, dict) or len(h) > 20 or not all(isinstance(k, str) and isinstance(v, str) and len(k) < 100 and len(v) < 4096 for k, v in h.items()): return {"id": m["id"], "error": "bad headers"}
        b = m.get("b", "")
        if not isinstance(b, str): return {"id": m["id"], "error": "bad body"}
        out.update(m=m["m"], p=p, h=h, b=b); return out

    def _answer(self, msg):
        try: r = self.dispatch(msg["m"], msg["p"], msg["h"], msg["b"])
        except Exception:                                                                 # the dispatcher answers its own errors; this is the last net: nothing about the inside is said
            self._deliver({"id": msg["id"], "s": 500, "h": [["Content-Type", "application/json"]], "b": base64.b64encode(b'{"ok": false, "error": "Something went wrong inside the assistant."}').decode("ascii")}); return
        self.answered += 1
        self._deliver({"id": msg["id"], "s": r.status, "h": [list(x) for x in r.headers], "b": base64.b64encode(r.body).decode("ascii")})

    def _deliver(self, obj):
        send = self._send
        if send is None: return
        try: send(json.dumps(obj, separators=(",", ":")))
        except Exception: pass                                                            # the window is gone: nobody to tell

    def close(self): self._pool.shutdown(wait=False, cancel_futures=True)

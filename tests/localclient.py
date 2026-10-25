"""A client for the interface that needs no network: it calls the same dispatcher the app window's bridge calls.

The product has no HTTP server and opens no socket (vgpt/offline.py would refuse), so tests that used to start a `ThreadingHTTPServer` and talk to it with `http.client` use this instead:

    c = LocalClient(server_module.make_dispatcher(app))
    status, headers, body = c.request("POST", "/api/ask", body={"q": "hello"})          # headers: lower-case names
    c.get_json("/api/health");  c.post_json("/api/ask", {"q": "hello"});  c.get_bytes("/api/thumb?ts=1")
"""
import json


class LocalClient:
    def __init__(self, dispatch): self.dispatch = dispatch

    def request(self, method, path, headers=None, body=None):
        h = dict(headers or {})
        if isinstance(body, (dict, list)): body = json.dumps(body); h.setdefault("Content-Type", "application/json")
        r = self.dispatch(method, path, h, body if body is not None else b"")
        return r.status, {k.lower(): v for k, v in r.headers}, r.body

    def get_bytes(self, path, headers=None): return self.request("GET", path, headers)[2]

    def get_json(self, path, headers=None): return json.loads(self.request("GET", path, headers)[2])

    def post_json(self, path, obj, headers=None): return json.loads(self.request("POST", path, headers, obj)[2])

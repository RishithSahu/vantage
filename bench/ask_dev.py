"""Ask the running dev page (bench/dev_web.py) questions through the same bridge the page uses and print what the page would show.

    python bench/ask_dev.py [--port 8794] [--role driver] "question one" "question two" ... | -f questions.txt [--json]"""
import argparse
import base64
import json
import sys
import urllib.request

import os
sys.stdout.reconfigure(encoding="utf-8")


def post(port, method, path, body=None, headers=None, sid=[0], cookie=[""]):
    sid[0] += 1; h = dict(headers or {})
    if body is not None: h["Content-Type"] = "application/json"
    if cookie[0]: h["Cookie"] = cookie[0]
    msg = json.dumps({"id": sid[0], "m": method, "p": path, "h": h, "b": json.dumps(body) if body is not None else ""})
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/__post", data=msg.encode("utf-8"), method="POST"), timeout=120); j = json.loads(r.read())
    for k, v in j.get("h", []):
        if k.lower() == "set-cookie" and v.startswith("vgpt_s="): cookie[0] = v.split(";")[0]
    raw = base64.b64decode(j.get("b", ""))
    try: return j["s"], json.loads(raw)
    except ValueError: return j["s"], raw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("q", nargs="*"); ap.add_argument("--port", type=int, default=8794); ap.add_argument("-f"); ap.add_argument("--json", action="store_true"); ap.add_argument("--role", default=None); a = ap.parse_args()
    qs = list(a.q)
    if a.f: qs += [l.rstrip("\n") for l in open(a.f, encoding="utf-8") if l.strip() and not l.startswith("#")]
    if a.role:
        s, j = post(a.port, "POST", "/api/role", {"role": a.role}); print("role ->", s, str(j)[:120])
    for q in qs:
        s, j = post(a.port, "POST", "/api/ask", {"q": q})
        if a.json: print(json.dumps(j, ensure_ascii=False)[:3000]); continue
        if not isinstance(j, dict): print(f"* {q}\n   [{s}] {j!r:.200}"); continue
        hero = j.get("hero") or j.get("tiles") or j.get("numbers")
        print(f"* {q}\n   {str(j.get('answer') or j.get('brief') or '')[:420]!r}" + (f"\n   hero: {json.dumps(hero, ensure_ascii=False)[:200]}" if hero else "") + f"\n   read as: {j.get('read_as') or j.get('reading')} | route {j.get('route')}")


if __name__ == "__main__":
    main()

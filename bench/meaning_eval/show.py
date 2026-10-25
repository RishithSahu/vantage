import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8")); a, b = int(sys.argv[2]), int(sys.argv[3]); w = int(sys.argv[4]) if len(sys.argv) > 4 else 170
for r in rows[a:b]:
    g = " ".join(r["gold"].split()[:2]) + (" " + r["gold"].split()[2] if r["gold"].split()[2] != "none" else "")
    ans = r["answer"].replace("\n", " / ")
    print(f"{r['id']} | {r['text']} | G:{g} | {r.get('route') or r.get('intent')} | {ans[:w]}")

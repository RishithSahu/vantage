import json, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.meaning import ontology, glossary
from vgpt.meaning.cues import Support
onto = ontology.load() if hasattr(ontology, "load") else ontology.Ontology()
gl = glossary.load() if hasattr(glossary, "load") else glossary.Glossary()
S = Support(onto, gl); NON = {"social", "offtopic", "unsupported", "control", "unsafe", "ambiguous"}
seen = {}; hit = []
for f in glob.glob(os.path.join(ROOT, "results", "meaning", "*", "predictions.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if 'frame2w' not in str(d.get('adapter')): continue
    for r in d.get("predictions", []) if isinstance(d, dict) else []:
        if r.get("pmin") is None: continue
        k = (r["text"], r["pred"])
        if k in seen: continue
        seen[k] = 1; p = r["pred"].split()
        if len(p) < 2 or p[0] not in ("why","define","howto") or (len(p) >= 8 and p[7] != "none") or p[1] in ("none", "unlisted") or r["pmin"] >= 0.5: continue
        if not S.supported(r["text"], p[1]): hit.append((r["text"], r.get("gold", "?? ??"), r["pred"], round(r["pmin"], 2)))
print(len(seen), "distinct predictions;", len(hit), "affected")
good = [h for h in hit if h[1].split()[0] in NON or h[1].split()[1] in ("none", "unlisted")]
print("gold is non-vehicle / unlisted (floor helps):", len(good))
for h in hit:
    if h not in good: print("NOT-GOLD-NONVEHICLE", h, "| pred correct" if h[1]==h[2] else "| pred already wrong")

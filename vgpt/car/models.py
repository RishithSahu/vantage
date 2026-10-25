"""The car models the test bench can run: a description (vgpt/car/models/<id>.json) and the pack that says how that car's buses carry its data (vgpt/car/models/<id>/ or, for the four original vehicles, the
pack of the same name that Vantage's decoder library also holds).  A model installed by the person (a folder with a profile.json and a car.json in <home>/models) joins the list.

    catalogue()  -> [ {id, name, year, region, powertrain, class, interfaces, notes, source, decoder_installed, dir, ...} ]
    find(id)     -> one entry or None"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUILTIN = os.path.join(HERE, "models")


def roots(home=None):
    out = []
    if home: out.append(os.path.join(home, "models"))
    out.append(BUILTIN)
    return out


def catalogue(home=None):
    seen = {}; out = []
    for root in roots(home):
        if not os.path.isdir(root): continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if name.endswith(".json") and os.path.isfile(path):
                try: meta = json.load(open(path, encoding="utf-8"))
                except (OSError, ValueError): continue
                mid = meta.get("id") or name[:-5]
                if mid in seen: continue
                d = os.path.join(root, mid)
                if not os.path.isfile(os.path.join(d, "profile.json")):
                    rel = meta.get("profile_dir")
                    d = os.path.normpath(os.path.join(root, rel)) if rel else d
                if not os.path.isfile(os.path.join(d, "profile.json")): continue
                meta = dict(meta, id=mid, dir=d); seen[mid] = meta; out.append(meta)
    return sorted(out, key=lambda m: (m.get("name") or m["id"]).lower())


def find(model_id, home=None):
    return next((m for m in catalogue(home) if m["id"] == model_id), None)

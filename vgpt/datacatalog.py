"""The vehicle histories this device carries: the database files in its data folder and any capture dropped in `data/captures` (its vehicle profile is derived from its content).

This used to live in the car-to-Vantage link module; with the link gone (the product has no network) it is its own small module."""
import os
import sqlite3
from pathlib import Path


def catalog(data_dir):
    """The vehicle histories a car carries: [{name, label, mb}]; the 105-day worlds first."""
    out = []; order = lambda f: (0 if f.startswith("world_") else 1 if f.startswith("gen_") else 2 if f == "vehicle.db" else 3, f)
    try: names = sorted(os.listdir(data_dir), key=order)
    except OSError: return out
    for f in names:
        if f == "vehicle.db" or (f.endswith(".db") and f.startswith(("gen_", "cs5_", "world_"))):
            path = os.path.join(data_dir, f)
            try:
                c = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True); m = dict(c.execute("SELECT k, v FROM meta").fetchall()); c.close()
                lab = f"{m.get('veh.model', '60 kWh EV')} - {m.get('env.city', 'simulated')} - {m.get('gen.driver', 'scripted story').replace('_', ' ')}"
            except sqlite3.Error: lab = f
            out.append({"name": f, "label": lab, "mb": round(os.path.getsize(path) / 2 ** 20, 1)})
    cap = os.path.join(data_dir, "captures")                                                                  # any log file dropped in data/captures is a vehicle too: its profile is derived from its content
    if os.path.isdir(cap):
        from .ingest import readers as RD
        for f in sorted(os.listdir(cap)):
            p = os.path.join(cap, f)
            if os.path.isfile(p) and not f.startswith(".") and not f.lower().endswith((".dbc", ".md", ".txt.bak")):
                try: fmt = RD.sniff(p)
                except OSError: continue
                if fmt not in ("unknown", "pcap"): out.append({"name": "capture:" + f, "label": f"Capture {f} ({fmt})", "mb": round(os.path.getsize(p) / 2 ** 20, 1)})
    return out

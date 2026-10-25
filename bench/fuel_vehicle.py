"""A clearly labelled SIMULATED fuel-capable vehicle history, so that the case study's own wording ("When did fuel efficiency show the largest drop?") can be answered literally by the same registry, interpreter and generic operations as every other question.

It is NOT real data. It is a copy of a generated 28-day EV history (data/gen_courier_ext.db) to which three simulated signals are ADDED in the extended table xsig_1m, from the vehicle's own recorded speed:
    fuel_rate        L/h      = 0.9 + 0.050 v + 0.00035 v^2   (v in km/h, +-3% noise)   x 1.15 from the 14th day before the end of the history (a simulated efficiency loss with no recorded cause)
    fuel_level       %        a 50 L tank starting at 85%, falling with the integrated fuel rate; refilled to 95% when it falls below 12%
    fuel_efficiency  km/L     = v / fuel_rate while v >= 10 km/h (a new signal registered by the schema overlay)
The history keeps its EV signals (so this is best read as a simulated range-extender / plug-in hybrid). The overlay (data/fixtures/sim_fuel_demo.overlay.json) maps the storage and registers fuel_efficiency; availability is decided from the data. Nothing in the assistant knows this question: it is answered through the general retrieval -> plan -> generic change-detection path.
    python bench/fuel_vehicle.py [--out data/fixtures]"""
import argparse, json, os, random, shutil, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SRC = os.path.join(HERE, "data", "gen_courier_ext.db")
OVERLAY = {"store": {"fuel_level": {"1m": ["xsig_1m", "fuel_level", 1.0]}, "fuel_rate": {"1m": ["xsig_1m", "fuel_rate", 1.0]}},
           "concepts": [{"name": "fuel_efficiency", "label": "fuel efficiency", "unit": "km/L", "category": "powertrain", "synonyms": ["fuel efficiency", "fuel economy", "fuel mileage", "km per litre", "kilometres per litre", "kilometers per liter", "mpg", "distance per litre of fuel"],
                             "lo": 0, "hi": 60, "description": "distance travelled per litre of fuel while moving; higher is better (simulated)", "store": {"1m": ["xsig_1m", "fuel_efficiency", 1.0]}}]}


def build(out):
    os.makedirs(out, exist_ok=True); dst = os.path.join(out, "sim_fuel_demo.db")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(dst + ext): os.remove(dst + ext)
    con = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True); d = sqlite3.connect(dst); con.backup(d); con.close()
    for col in ("fuel_level", "fuel_rate", "fuel_efficiency"): d.execute(f"ALTER TABLE xsig_1m ADD COLUMN {col} REAL")
    end = int(float(dict(d.execute("SELECT k, v FROM meta").fetchall())["end"])); loss_from = end - 14 * 86400; rnd = random.Random(2026)
    speed = {ts: sp / 10.0 for ts, sp in d.execute("SELECT ts, speed FROM sig_1m WHERE speed IS NOT NULL")}; level = 85.0; rows = []
    for (ts,) in d.execute("SELECT ts FROM xsig_1m ORDER BY ts").fetchall():
        v = speed.get(ts)
        if v is None: continue
        rate = (0.9 + 0.050 * v + 0.00035 * v * v) * (1.0 + rnd.uniform(-0.03, 0.03)) * (1.15 if ts >= loss_from else 1.0)
        level -= rate / 60.0 / 50.0 * 100.0
        if level < 12.0: level = 95.0
        rows.append((round(level, 2), round(rate, 3), round(v / rate, 3) if v >= 10 else None, ts))
    d.executemany("UPDATE xsig_1m SET fuel_level = ?, fuel_rate = ?, fuel_efficiency = ? WHERE ts = ?", rows)
    d.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('veh.model', 'SIMULATED fuel demonstration vehicle (generated EV history + simulated fuel signals)')"); d.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('veh.simulated_fuel', '1')")
    d.commit(); d.close(); json.dump(OVERLAY, open(dst[:-3] + ".overlay.json", "w"), indent=1); return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "data", "fixtures")); a = ap.parse_args(); print(build(a.out))

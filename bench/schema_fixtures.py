"""Schema-transfer and missing-capability FIXTURES, built from a real generated history (gen_courier_ext) by editing its STORAGE, so that every test is end to end (storage -> profile -> capability -> planner -> executor -> contract -> composer):
  fl_missing            front-left pressure column empty; the other three wheels remain
  aggregate_only        no individual wheel columns; one aggregate pressure column (overlay maps the aggregate signal to it)
  units                 speed stored in mph x10, battery temperature stored in Fahrenheit x10 (the overlay gives scale and offset)
  empty_registered      coolant flow is registered (the column exists) but has no observations at all
  intermittent          battery temperature missing on a seeded 40% of days
  opaque_signal         a signal with an opaque identifier (CH_0417) known only through its label, description and synonyms
  new_signal            a signal the built-in registry does not know (washer fluid level, %), registered by overlay with a description and synonyms, with its own column
  steering_absent       motor torque present (own column, by overlay), steering torque absent (confusable pair)
Missing signals are removed from STORAGE (the data), and the capability metadata is derived from the data (Profile), so a grammar-only omission is not what is being tested.
Build:  python bench/schema_fixtures.py [--out data/fixtures]"""
import argparse, json, os, random, shutil, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data", "gen_courier_ext.db")


def _copy(out, name):
    os.makedirs(out, exist_ok=True); dst = os.path.join(out, name + ".db")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(dst + ext): os.remove(dst + ext)
    con = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True); d = sqlite3.connect(dst); con.backup(d); con.close(); return d, dst


def _save(d, dst, overlay):
    d.commit(); d.close(); json.dump(overlay, open(dst[:-3] + ".overlay.json", "w"), indent=1)


def build(out):
    made = {}
    d, dst = _copy(out, "fl_missing"); d.execute("UPDATE ext_1m SET tp_fl = NULL"); _save(d, dst, {}); made["fl_missing"] = dst
    d, dst = _copy(out, "aggregate_only")
    d.execute("ALTER TABLE ext_1m ADD COLUMN tp_avg INTEGER"); d.execute("UPDATE ext_1m SET tp_avg = CAST(ROUND((tp_fl + tp_fr + tp_rl + tp_rr) / 4.0) AS INTEGER)")
    for c in ("tp_fl", "tp_fr", "tp_rl", "tp_rr"): d.execute(f"UPDATE ext_1m SET {c} = NULL")
    _save(d, dst, {"store": {"tyre_pressure": {"1m": ["ext_1m", "tp_avg", 10.0]}}}); made["aggregate_only"] = dst
    d, dst = _copy(out, "units")
    for t in ("sig_1m", "sig_1s"):
        d.execute(f"UPDATE {t} SET speed = CAST(ROUND(speed / 1.609344) AS INTEGER) WHERE speed IS NOT NULL"); d.execute(f"UPDATE {t} SET batt = CAST(ROUND(batt * 1.8 + 320) AS INTEGER) WHERE batt IS NOT NULL")
    _save(d, dst, {"store": {"speed": {"1m": ["sig_1m", "speed", 6.2137119, 0.0], "1s": ["sig_1s", "speed", 6.2137119, 0.0]}, "battery_temperature": {"1m": ["sig_1m", "batt", 18.0, -17.777778], "1s": ["sig_1s", "batt", 18.0, -17.777778]}}}); made["units"] = dst
    d, dst = _copy(out, "empty_registered")
    for t in ("sig_1m", "sig_1s"): d.execute(f"UPDATE {t} SET flow = NULL")
    _save(d, dst, {}); made["empty_registered"] = dst
    d, dst = _copy(out, "intermittent"); rng = random.Random(7)
    days = sorted({r[0] // 86400 for r in d.execute("SELECT ts FROM sig_1m")}); gone = set(rng.sample(days, int(0.4 * len(days))))
    d.executemany("UPDATE sig_1m SET batt = NULL WHERE ts >= ? AND ts < ?", [(x * 86400, (x + 1) * 86400) for x in gone]); _save(d, dst, {}); made["intermittent"] = dst
    d, dst = _copy(out, "new_signal")
    d.execute("ALTER TABLE sig_1m ADD COLUMN wfl INTEGER"); rows = d.execute("SELECT ts FROM sig_1m ORDER BY ts").fetchall(); t0, t1 = rows[0][0], rows[-1][0]; rng = random.Random(11)
    d.executemany("UPDATE sig_1m SET wfl = ? WHERE ts = ?", [(int(max(0.0, 900 - 700 * (ts - t0) / max(1, t1 - t0) + rng.gauss(0, 8))), ts) for (ts,) in rows])           # washer fluid level, % x10, slowly emptying
    _save(d, dst, {"concepts": [{"name": "washer_fluid_level", "label": "washer fluid level", "unit": "%", "category": "vehicle", "synonyms": ["washer fluid level", "washer fluid", "screen wash level", "windscreen washer level"], "lo": 0, "hi": 100,
                                 "description": "level of the windscreen washer fluid reservoir", "store": {"1m": ["sig_1m", "wfl", 10.0]}}]}); made["new_signal"] = dst
    d, dst = _copy(out, "opaque_signal")                                                                   # an OPAQUE identifier whose meaning is carried only by its label, description and synonyms
    d.execute("ALTER TABLE sig_1m ADD COLUMN c0417 INTEGER"); rows = d.execute("SELECT ts FROM sig_1m ORDER BY ts").fetchall(); rng = random.Random(23)
    d.executemany("UPDATE sig_1m SET c0417 = ? WHERE ts = ?", [(int(250 + 60 * ((ts // 3600) % 24) / 24 + rng.gauss(0, 5)), ts) for (ts,) in rows])                      # a slowly varying reading, x10
    _save(d, dst, {"concepts": [{"name": "CH_0417", "label": "inverter coolant outlet temperature", "unit": "C", "category": "powertrain", "synonyms": ["inverter coolant temperature", "inverter coolant outlet temperature", "power electronics coolant temperature"], "lo": -40, "hi": 150,
                                 "description": "temperature of the coolant leaving the traction inverter", "store": {"1m": ["sig_1m", "c0417", 10.0]}}]}); made["opaque_signal"] = dst
    d, dst = _copy(out, "steering_absent")                                                               # motor torque IS recorded (own column, by overlay); steering torque is not
    d.execute("ALTER TABLE sig_1m ADD COLUMN trq INTEGER"); rows = d.execute("SELECT ts, power FROM sig_1m ORDER BY ts").fetchall(); rng = random.Random(5)
    d.executemany("UPDATE sig_1m SET trq = ? WHERE ts = ?", [(int((p or 0) * 3.0 + rng.gauss(0, 20)), ts) for ts, p in rows])
    _save(d, dst, {"store": {"motor_torque": {"1m": ["sig_1m", "trq", 10.0]}}}); made["steering_absent"] = dst
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "data", "fixtures")); a = ap.parse_args()
    for k, v in build(a.out).items(): print(k, os.path.getsize(v) // 1024, "KB")

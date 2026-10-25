"""A small deterministic vehicle history with hand-checkable values, for the Round-52 semantic tests.

14 UTC days (2026-09-01 .. 2026-09-14), one-minute samples, every value a closed-form function of (day, minute of day), so a test can compute its expected answer from the
arrays below without going through the assistant.  The schedule is deliberately lumpy: distinct speeds per state, a 12-minute and a 5-minute hot episode, an isolated hot
minute, a fully parked day, a parked-21-hour day and a day with only six recorded hours.

    python bench/fixture_r52.py            # writes data/fixture_r52.db
"""
import os, sqlite3, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = 1788220800                      # 2026-09-01 00:00:00 UTC
DAYS = 14
NOW = T0 + DAYS * 86400 - 3600       # one hour before the end of the 14th day: 2026-09-14 23:00 UTC
MODE_PARKED, MODE_CITY, MODE_HWY, MODE_CHARGE = 0, 1, 2, 3


def build_arrays():
    """-> dict of equally long numpy arrays over the recorded minutes plus the event / trip / charge lists."""
    ts = []; mode = []; speed = []; batt = []; motor = []; flow = []; power = []; soc = []; amb = []
    s = 80.0
    for d in range(DAYS):
        for m in range(1440):
            if d == 10 and not (6 * 60 <= m < 12 * 60): continue                    # day 10: only 06:00-12:00 is recorded
            if d == 3: md = MODE_PARKED                                             # day 3: parked all day, fully recorded
            elif d == 7: md = MODE_CITY if 12 * 60 <= m < 13 * 60 + 30 else MODE_PARKED   # day 7: 1.5 h of city driving, 22.5 h parked
            else:
                md = MODE_PARKED
                if 7 * 60 <= m < 8 * 60: md = MODE_CITY
                elif 8 * 60 <= m < 8 * 60 + 30: md = MODE_HWY
                elif 17 * 60 <= m < 18 * 60: md = MODE_CITY
                elif 21 * 60 <= m < 22 * 60 + 30: md = MODE_CHARGE
            t = T0 + d * 86400 + m * 60
            if md == MODE_CITY: v = 25.0 + (m % 7) * 2.0 + d * 0.5                 # 25..37 (+ day offset)
            elif md == MODE_HWY: v = 90.0 + (m % 5) * 3.0                          # 90..102
            else: v = 0.0
            b = 24.0 + 0.5 * (d % 4) + (0.8 if md in (MODE_CITY, MODE_HWY) else 0.0) + 0.002 * m
            if d == 5 and 14 * 60 <= m < 14 * 60 + 12: b = 52.0                    # 12 continuous hot minutes
            if d == 6 and 14 * 60 <= m < 14 * 60 + 5: b = 52.5                     # 5 continuous hot minutes
            if d == 6 and m == 14 * 60 + 10: b = 51.0                              # one isolated hot minute
            mt = b + 8.0 + (m % 3) + d * 0.1
            f = {MODE_CITY: 8.0 + (m % 4) * 0.5, MODE_HWY: 10.0 + (m % 3) * 0.5, MODE_CHARGE: 3.0, MODE_PARKED: 0.0}[md]
            if d == 9 and md == MODE_CITY: f = 1.5                                  # day 9: low flow while driving (every driving minute)
            p = {MODE_CITY: 12.0, MODE_HWY: 30.0, MODE_CHARGE: -7.0, MODE_PARKED: 0.3}[md]
            ts.append(t); mode.append(md); speed.append(v); batt.append(b); motor.append(mt); flow.append(f); power.append(p); soc.append(60.0); amb.append(20.0 + 0.1 * d)
    arr = {k: np.array(v, dtype=float) for k, v in dict(speed=speed, batt=batt, motor=motor, flow=flow, power=power, soc=soc, amb=amb, mode=mode).items()}
    for k in ("speed", "batt", "motor", "flow", "soc", "amb"): arr[k] = np.round(arr[k], 1)            # stored in tenths
    arr["power"] = np.round(arr["power"], 3)
    arr["ts"] = np.array(ts, dtype=np.int64)
    return arr


def events():
    """(ts, type, severity, detail): each placed in a known drive mode."""
    out = []
    def at(day, hh, mm): return T0 + day * 86400 + hh * 3600 + mm * 60
    for d in (1, 2, 4):
        out.append((at(d, 7, 20), "HARSH_BRAKE", "info", "0.30 g at 40 km/h"))        # city
        out.append((at(d, 8, 10), "HARSH_BRAKE", "info", "0.32 g at 95 km/h"))        # highway
    out.append((at(2, 21, 30), "HARSH_BRAKE", "info", "0.29 g at 0 km/h"))            # while charging (impossible physically; a fixture value)
    out.append((at(5, 8, 5), "HARSH_ACCEL", "info", "0.30 g at 92 km/h"))             # highway
    out.append((at(5, 7, 40), "HARSH_ACCEL", "info", "0.31 g at 35 km/h"))            # city
    out.append((at(5, 3, 0), "HARSH_ACCEL", "info", "0.28 g at 0 km/h"))              # parked
    out.append((at(5, 14, 6), "WARN_BATT_OVERTEMP", "warn", "Battery temperature above 50 C"))   # parked, hot episode
    out.append((at(5, 21, 40), "WARN_BATT_OVERTEMP", "warn", "Battery temperature above 40 C"))  # charging
    out.append((at(6, 17, 30), "WARN_BATT_OVERTEMP", "warn", "Battery temperature above 40 C"))  # city
    out.append((at(8, 7, 30), "TPMS_LOW", "warn", "rear left tyre pressure low"))     # city
    out.append((at(8, 8, 15), "TPMS_LOW", "warn", "front right tyre pressure low"))   # highway
    return sorted(out)


def trips():
    """(start, end, km, kwh, wh_km, avg_speed, hwy_share): city trips 07:00-08:00 and 17:00-18:00 on days with that schedule, a highway trip 08:00-08:30."""
    rows = []
    for d in range(DAYS):
        if d in (3, 7, 10): continue
        base = T0 + d * 86400
        rows.append((base + 7 * 3600, base + 8 * 3600, 16.0, 2.4 + 0.01 * d, 150.0 + d, 16.0, 0.0))
        rows.append((base + 8 * 3600, base + 8 * 3600 + 1800, 22.0, 4.6, 209.0, 44.0, 1.0))
        rows.append((base + 17 * 3600, base + 18 * 3600, 16.0, 2.4 + 0.01 * d, 150.0 + d, 16.0, 0.0))
    rows.append((T0 + 7 * 86400 + 12 * 3600, T0 + 7 * 86400 + 13 * 3600 + 1800, 24.0, 3.7, 154.0, 16.0, 0.0))
    return sorted(rows)


def charges():
    """(start, end, kind, kw, kwh, soc0, soc1): the 21:00-22:30 session of each driven day, with a distinct energy per day."""
    rows = []
    for d in range(DAYS):
        if d in (3, 7, 10): continue
        s = T0 + d * 86400 + 21 * 3600
        rows.append((s, s + 5400, "AC", 7.0, 6.0 + d * 0.5, 40.0, 60.0))
    return rows


def build(path):
    if os.path.exists(path): os.remove(path)
    src = sqlite3.connect(os.path.join(HERE, "data", "gen_demo.db"))
    c = sqlite3.connect(path)
    for name, sql in src.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'logs_fts%' AND sql IS NOT NULL"):
        c.execute(sql)
    for name, sql in src.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"): c.execute(sql)
    A = build_arrays()
    rows_m = []; rows_s = []
    for i in range(len(A["ts"])):
        rows_m.append((int(A["ts"][i]), int(round(A["speed"][i] * 10)), int(round(A["soc"][i] * 10)), int(round(A["power"][i] * 1000)), int(round(A["batt"][i] * 10)), int(round(A["batt"][i] * 10)),
                       int(round(A["flow"][i] * 10)), int(round(A["flow"][i] * 10)), int(round(A["amb"][i] * 10)), 0, int(round(A["motor"][i] * 10)), int(A["mode"][i])))
    c.executemany("INSERT INTO sig_1m VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows_m)
    c.executemany("INSERT INTO events VALUES (?,?,?,?)", events())
    c.executemany("INSERT INTO trips(start,end,km,kwh,wh_km,avg_speed,hwy_share) VALUES (?,?,?,?,?,?,?)", trips())
    c.executemany("INSERT INTO charges(start,end,kind,kw,kwh,soc0,soc1) VALUES (?,?,?,?,?,?,?)", charges())
    meta = {"t0": str(T0), "end": str(T0 + DAYS * 86400), "env.city": "Fixtureville", "env.lat": "0", "env.lon": "0", "env.tz_h": "0", "veh.model": "Fixture EV", "veh.cap_kwh": "60.0", "veh.soh": "98.0",
            "veh.tyre_spec_front": "240", "veh.tyre_spec_rear": "260", "veh.home_charger_kw": "7.0"}
    c.executemany("INSERT INTO meta VALUES (?,?)", list(meta.items()))
    c.commit(); c.close()
    return path


if __name__ == "__main__":
    out = os.path.join(HERE, "data", "fixture_r52.db"); build(out); print("wrote", out)

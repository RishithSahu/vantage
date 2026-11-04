"""The history as the cause engine sees it: one value per local day for each canonical quantity, per-trip arrays, per-day event counts.

Everything is read from the vehicle's own stored tiers through the signal registry's mapping (concept -> table, column, scale), so a vehicle that stores other quantities simply has other series; a quantity the vehicle does not store
is absent, and the hypotheses that need it are reported as not assessable (never guessed).  Rows whose key starts with `gen.` in the meta table (how a test history was made) are never read."""
import sqlite3

import numpy as np

from .. import signal_registry as R
from ..tablenames import table_names

DAY = 86400
ANY_TIME = {"ambient_temperature", "aux_battery_voltage", "barometric_pressure", "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr", "tyre_pressure", "battery_soh"}


class History:
    def __init__(self, conn, horizon_days=70, now=None):
        self.c = conn; self.tables = table_names(conn)
        meta = {k: v for k, v in conn.execute("SELECT k, v FROM meta").fetchall() if not str(k).startswith("gen.")} if "meta" in self.tables else {}
        self.meta = meta
        try: self.tz = int(float(meta.get("env.tz_h", 0)) * 3600)
        except ValueError: self.tz = 0
        hi = conn.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0] if "sig_1m" in self.tables else None
        if hi is None and "trips" in self.tables: hi = conn.execute("SELECT MAX(end) FROM trips").fetchone()[0]
        self.hi = int(min(hi, now) if (now and hi) else (hi or 0)); self.d_hi = (self.hi + self.tz) // DAY; self.D = int(horizon_days); self.d0 = self.d_hi - self.D + 1; self.lo = self.d0 * DAY - self.tz
        self._cache = {}; self._cols = {}

    # ---------------------------------------------------------------- time
    def day_of(self, ts): return int((int(ts) + self.tz) // DAY - self.d0)

    def day_start(self, d): return int((self.d0 + d) * DAY - self.tz)

    def date(self, d):
        import datetime as dt
        return (dt.date(1970, 1, 1) + dt.timedelta(days=int(self.d0 + d))).strftime("%d %b")

    # ---------------------------------------------------------------- quantities
    def has_col(self, table, col):
        if table not in self.tables: return False
        if table not in self._cols: self._cols[table] = {r[1] for r in self.c.execute(f"PRAGMA table_info({table})")}
        return col in self._cols[table]

    def available(self, concept):
        ent = (R.STORE.get(concept) or {}).get("1m"); return bool(ent and self.has_col(ent[0], ent[1]))

    def daily(self, concept, agg="AVG", drive=None):
        """-> float array of length D (NaN where the vehicle has no data that day).  `drive`: only minutes while driving (default: for everything but the any-time quantities).
        One scan of a table yields the daily mean, maximum and minimum of EVERY column the registry maps there (twelve series used to be twelve scans: 0.8 s of the 1.3 s a "why" took on the Raspberry Pi 4)."""
        key = (concept, agg, drive)
        if key in self._cache: return self._cache[key]
        ent = (R.STORE.get(concept) or {}).get("1m"); out = np.full(self.D, np.nan)
        if ent and self.has_col(ent[0], ent[1]):
            table = ent[0]; masked = (concept not in ANY_TIME) if drive is None else drive
            arr = self._prefetch(table, bool(masked)).get((ent[1], agg))
            if arr is not None: scale = ent[2]; off = ent[3] if len(ent) > 3 else 0.0; out = np.where(np.isfinite(arr), arr / scale + off, np.nan)
        self._cache[key] = out; return out

    def _prefetch(self, table, masked):
        """{(column, 'AVG'|'MAX'|'MIN'): raw daily array} for every column of `table` that some concept maps to.  Finished days come from the DAILY TIER (agg_1d, written by the ingest daemon: about 100
        rows per quantity per year); only the days after the last rolled one - at most today - are aggregated from the minute rows.  Days with fewer than 5 samples are NaN, as before."""
        key = ("pre", table, masked)
        if key in self._cache: return self._cache[key]
        cols = sorted({e[1] for e in (v.get("1m") for v in R.STORE.values()) if e and e[0] == table and self.has_col(table, e[1])})
        res = {}
        if cols:
            for c in cols:
                for a in ("AVG", "MAX", "MIN"): res[(c, a)] = np.full(self.D, np.nan)
            live_from = self.d0
            if "agg_1d" in self.tables and "agg_1d_state" in self.tables:
                try:
                    st = self.c.execute("SELECT through, sig_max_ts FROM agg_1d_state WHERE tbl=? AND masked=? AND tz=?", (table, int(masked), self.tz)).fetchone()
                    cur_hi = self.c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0] if "sig_1m" in self.tables else None
                    if st and cur_hi is not None and int(cur_hi) >= int(st[1]):                                         # the tier belongs to this data (it is only ever extended)
                        through = min(int(st[0]), self.d0 + self.D - 1)
                        for c, d, n, av, mx, mn in self.c.execute("SELECT col, d, n, avg, mx, mn FROM agg_1d WHERE tbl=? AND masked=? AND tz=? AND d>=? AND d<=?", (table, int(masked), self.tz, self.d0, through)):
                            i = int(d) - self.d0
                            if c in cols and n >= 5 and 0 <= i < self.D and av is not None: res[(c, "AVG")][i] = float(av); res[(c, "MAX")][i] = float(mx); res[(c, "MIN")][i] = float(mn)
                        live_from = max(self.d0, through + 1)
                except sqlite3.Error: live_from = self.d0
                    
            if live_from <= self.d0 + self.D - 1:
                join = " JOIN sig_1m s ON s.ts = t.ts" if masked and table != "sig_1m" else ""
                where = ((" AND s.mode IN (1,2)" if table != "sig_1m" else " AND t.mode IN (1,2)") if masked and self.has_col("sig_1m", "mode") else "")
                sel = ", ".join(f"AVG(t.{c}), MAX(t.{c}), MIN(t.{c}), COUNT(t.{c})" for c in cols)
                try:
                    for row in self.c.execute(f"SELECT CAST((t.ts + ?) / 86400 AS INTEGER) d, {sel} FROM {table} t{join} WHERE t.ts>=? AND t.ts<?{where} GROUP BY d", (self.tz, max(self.lo, live_from * 86400 - self.tz), self.hi + 1)):
                        i = int(row[0]) - self.d0
                        if not (0 <= i < self.D): continue
                        for ci, c in enumerate(cols):
                            av, mx, mn, n = row[1 + 4 * ci: 5 + 4 * ci]
                            if n is not None and n >= 5 and av is not None:
                                res[(c, "AVG")][i] = float(av); res[(c, "MAX")][i] = float(mx); res[(c, "MIN")][i] = float(mn)
                except sqlite3.Error: pass
        self._cache[key] = res; return res

    def raw(self, table, col, agg="AVG", drive=False):
        """A column that has no canonical concept (a vehicle's own extra)."""
        key = (table, col, agg, drive)
        if key in self._cache: return self._cache[key]
        out = np.full(self.D, np.nan)
        if self.has_col(table, col):
            fn = {"AVG": "AVG", "MAX": "MAX", "MIN": "MIN"}[agg]; tscol = "ts" if self.has_col(table, "ts") else "start"
            try:
                for d, v in self.c.execute(f"SELECT CAST(({tscol} + ?) / 86400 AS INTEGER) d, {fn}({col}) FROM {table} WHERE {tscol}>=? AND {tscol}<? AND {col} IS NOT NULL GROUP BY d", (self.tz, self.lo, self.hi + 1)):
                    i = int(d) - self.d0
                    if 0 <= i < self.D and v is not None: out[i] = float(v)
            except sqlite3.Error: pass
        self._cache[key] = out; return out

    def event_counts(self, *types):
        key = ("ev",) + tuple(types); out = np.zeros(self.D)
        if key in self._cache: return self._cache[key]
        if "events" in self.tables:
            q = ",".join("?" * len(types))
            for d, n in self.c.execute(f"SELECT CAST((ts + ?) / 86400 AS INTEGER) d, COUNT(*) FROM events WHERE type IN ({q}) AND ts>=? AND ts<? GROUP BY d", (self.tz, *types, self.lo, self.hi + 1)):
                i = int(d) - self.d0
                if 0 <= i < self.D: out[i] = n
        self._cache[key] = out; return out

    def event_times(self, *types, lo=None, hi=None):
        if "events" not in self.tables: return []
        q = ",".join("?" * len(types)); return [(int(t), ty, det) for t, ty, det in self.c.execute(f"SELECT ts, type, detail FROM events WHERE type IN ({q}) AND ts>=? AND ts<? ORDER BY ts", (*types, lo if lo is not None else self.lo, hi if hi is not None else self.hi + 1))]

    def trips(self):
        """Per-trip arrays: day, km, kwh, wh_km, avg_speed, hwy_share, amb, hvac_w, harsh counts (when the vehicle stores them)."""
        if "trips" in self._cache: return self._cache["trips"]
        t = {k: [] for k in ("start", "day", "km", "kwh", "wh", "speed", "hwy", "amb", "hvac", "hb", "ha", "hc")}
        if "trips" in self.tables:
            cols = {r[1] for r in self.c.execute("PRAGMA table_info(trips)")}
            ext = "trip_ext" in self.tables
            sel = "t.start, t.km, t.kwh, t.wh_km, t.avg_speed, t.hwy_share, t.avg_amb, t.avg_hvac_w" if {"avg_speed", "hwy_share", "avg_amb", "avg_hvac_w"} <= cols else "t.start, t.km, t.kwh, t.wh_km, NULL, NULL, NULL, NULL"
            q = f"SELECT {sel}, " + ("e.harsh_brake, e.harsh_accel, e.harsh_corner" if ext else "NULL, NULL, NULL") + " FROM trips t " + ("LEFT JOIN trip_ext e ON e.start = t.start " if ext else "") + "WHERE t.start>=? AND t.start<? AND t.km > 0.3 ORDER BY t.start"
            for r in self.c.execute(q, (self.lo, self.hi + 1)):
                t["start"].append(r[0]); t["day"].append(self.day_of(r[0])); t["km"].append(r[1]); t["kwh"].append(r[2]); t["wh"].append(r[3]); t["speed"].append(r[4]); t["hwy"].append(r[5]); t["amb"].append(r[6]); t["hvac"].append(r[7]); t["hb"].append(r[8]); t["ha"].append(r[9]); t["hc"].append(r[10])
        out = {k: np.array([np.nan if x is None else x for x in v], float) for k, v in t.items()}; self._cache["trips"] = out; return out

    def by_day(self, trip_vals, trip_days, how="mean", weights=None):
        out = np.full(self.D, np.nan)
        for d in range(self.D):
            m = (trip_days == d) & np.isfinite(trip_vals)
            if m.any(): out[d] = float(np.average(trip_vals[m], weights=weights[m]) if weights is not None else (np.mean(trip_vals[m]) if how == "mean" else np.sum(trip_vals[m])))
        return out

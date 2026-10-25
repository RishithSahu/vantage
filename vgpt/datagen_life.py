"""DEV tool - the timeline simulation of the generator (see datagen.py): drives, charging sessions and parked periods second by second, with physics driven by REAL weather and REAL speed traces."""
import datetime as dt, time
import numpy as np
from .datagen import (movavg, CITIES, VEHICLES, DRIVERS, STORIES, G, World, Segments, Plan, build_speed, hvac_w, dc_curve, temp_derate, tz_offset_h)
from .sim import Vehicle


def tyre_timeline(rng, spec, days, st):
    """Cold tyre pressure (kPa) per wheel per hour, with natural diffusion, an optional slow leak, the TPMS warning and the owner's top-up."""
    H = days * 24 + 2; th = np.arange(H) / 24.0; base = spec + rng.uniform(-6, 8, 4); cold = np.tile(base[:, None], (1, H)) - 1.2 * th[None, :] / 30.0; events = []
    lk = st.get("tyre_leak")
    if lk:
        w = lk["wheel"]; cold[w] -= lk["rate"] * np.clip(th - lk["start"], 0, None)
        below = np.where(cold[w] < 0.78 * spec[w])[0]
        if len(below):
            hw_ = int(below[0]); events.append((hw_ * 3600 + int(rng.integers(0, 3600)), "TPMS_LOW", "warn", w))
            if rng.random() < 0.85:
                hr = hw_ + int(rng.uniform(1, 6) * 24)
                if hr < H - 2:
                    lk["refill"] = hr; cold[w, hr:] = spec[w] + rng.uniform(0, 10) - 1.2 * (th[hr:] - th[hr]) / 30.0; events.append((hr * 3600 + int(rng.integers(8 * 3600, 20 * 3600)) % 3600 + 3600, "TYRE_INFLATED", "info", w))
    return cold, events


def cond(fl):
    """Heat-exchange conductance factor vs coolant flow (1.0 at the nominal 12 L/min); with a weak pump the heat transfer collapses faster than the flow does."""
    return fl / 12.0 if fl >= 9.0 else (fl / 12.0) * (max(fl, 0.0) / 9.0) ** 0.5


class Life:
    def __init__(self, rng, V, C, DR, world, seg, plan, days, t0, tzh, st):
        self.rng, self.V, self.C, self.DR, self.world, self.seg, self.plan, self.days, self.t0, self.tzh, self.st = rng, V, C, DR, world, seg, plan, days, t0, tzh, st
        n = self.n = days * 86400; self.m = n // 60
        self.speed = np.zeros(n, np.float32); self.power = np.full(n, 0.012, np.float32); self.soc = np.zeros(n, np.float32); self.batt = np.zeros(n, np.float32); self.flow = np.zeros(n, np.float32)
        self.hvac = np.zeros(n, np.float32); self.motor = np.zeros(n, np.float32); self.tyre_t = np.zeros(n, np.float32); self.mode = np.zeros(n, np.int8); self.ign = np.zeros(n, bool)
        self.chg = np.zeros(n, np.float32); self.odo = np.zeros(n, np.float32); self.amb = world.amb
        self.soh0 = float(rng.uniform(0.90, 1.0)); self.cap = V["cap"] * self.soh0; self.e = self.cap * float(rng.uniform(0.55, 0.9)); self.tb = float(self.amb[0]) + 2; self.tt = float(self.amb[0])
        self.odo0 = float(rng.uniform(4000, 60000)); self.odo_km = self.odo0; self.est_city = 0.17; self.est_hwy = 0.23; self.fric_kwh = 0.0; self.regen_kwh = 0.0
        self.spec = np.array([V["tyre"][0], V["tyre"][0], V["tyre"][1], V["tyre"][1]], np.float64)
        self.cold, self.tyre_events = tyre_timeline(rng, self.spec, days, st)
        self.pump_t = st["cooling_pump"]["start"] * 86400 if "cooling_pump" in st else 10 ** 12
        self.roof_t = st["roof_box"]["start"] * 86400 + 13 * 3600 if "roof_box" in st else 10 ** 12
        self.drift_t = st["alignment_drift"]["start"] * 86400 if "alignment_drift" in st else 10 ** 12; self.drift_rate = st["alignment_drift"].get("rate", 0.0) if "alignment_drift" in st else 0.0
        self.home_kw = float(min(V["ac"], rng.choice([3.7, 7.4, 11.0], p=[0.15, 0.55, 0.30]))); self.work_ok = rng.random() < 0.7
        self.sessions, self.trips = [], []; self.at = plan.pid("Home"); self.jid = 0

    # ------------------------------------------------------------------ helpers
    def local_hour(self, s): return ((self.t0 + s) % 86400 / 3600.0 + self.tzh) % 24

    def rain_at(self, s): return int(self.world.rain[min(s // 60, self.m - 1)])

    def cold_at(self, s): return self.cold[:, min(s // 3600, self.cold.shape[1] - 1)]

    def fill_idle(self, a, b):
        if b <= a: return
        b = min(b, self.n); L = b - a; k = np.arange(L, dtype=np.float64); am = self.amb[a:b].astype(np.float64)
        tbv = am + (self.tb - am[0]) * np.exp(-k / 21600.0); ttv = am + (self.tt - am[0]) * np.exp(-k / 2400.0)
        self.batt[a:b] = tbv; self.tyre_t[a:b] = ttv; self.tb, self.tt = float(tbv[-1]), float(ttv[-1])
        e1 = max(0.0, self.e - 0.012 * L / 3600.0); self.soc[a:b] = 100 * np.linspace(self.e, e1, L) / self.cap; self.e = e1
        self.flow[a:b] = 0.0; self.motor[a:b] = am + 1.5; self.odo[a:b] = self.odo_km

    def charge(self, a, kind, kw_max, target, max_len):
        n, V = self.n, self.V; L = 0; e0 = self.e; soc0 = 100 * self.e / self.cap; pk = 0.0; dc = kind == "DC"; ua0 = 280.0 * (self.cap / 60.0); cth = 150e3 * (self.cap / 60.0); tgt = target * self.cap
        tb, tt, e = self.tb, self.tt, self.e
        while a + L < n and L < max_len and e < tgt - 0.01:
            s = a + L; f = e / self.cap
            p = kw_max * float(dc_curve(f)) * float(temp_derate(tb)) if dc else kw_max * (1.0 if f < 0.9 else max(0.1, 1 - (f - 0.9) * 8))
            p = min(p, (tgt - e) * 3600.0)
            base = 10.0 + min(4.0, p * 0.04 + 0.2)
            if s >= self.pump_t: base *= max(0.30, 1 - 0.70 * (s - self.pump_t) / (8 * 86400.0))
            fl = base; ua = ua0 * cond(fl) * (1.0 + 0.8 * min(1.0, max(0.0, (fl - 5.0) / 5.0)) if tb > 33 else 1.0); a_ = abs(p); hk = (0.0004 * a_ * a_ + 0.010 * a_ + (0.0001 * a_ * a_ if dc else 0.0)) * (1.0 + 0.03 * max(0.0, 20.0 - tb))
            tb += (hk * 1000.0 - ua * (tb - float(self.amb[s]))) / cth; e += p / 3600.0
            self.flow[s] = fl; self.batt[s] = tb; self.soc[s] = 100 * e / self.cap; self.chg[s] = p; self.power[s] = -p; self.mode[s] = 3; self.speed[s] = 0; self.odo[s] = self.odo_km
            self.hvac[s] = 0; self.motor[s] = self.amb[s] + 2; tt += (float(self.amb[s]) - tt) / 2400.0; self.tyre_t[s] = tt; pk = max(pk, p); L += 1
        self.tb, self.tt, self.e = tb, tt, e
        if L > 0: self.sessions.append(dict(start=a, end=a + L, kind=kind, kw=pk, kwh=e - e0, soc0=soc0, soc1=100 * e / self.cap, place=self.at))
        return a + L

    def idle_with_charge(self, a, b, place):
        r = self.rng; kind = self.plan.places[place]["kind"] if place >= 0 else "x"
        kw = self.home_kw if kind == "home" else (min(self.V["ac"], 7.4) if kind == "work" and self.DR["work_charge"] and self.work_ok else 0.0)
        need = self.DR["target"] * self.cap - self.e
        if b - a > 900 and kw > 0 and need > 1.5 and (r.random() < self.DR["plug"] if kind == "home" else True):
            lh = self.local_hour(a); delay = int(r.uniform(120, 2400))
            if kind == "home" and r.random() < 0.45: delay = int(((r.uniform(22.5, 25.5) - lh) % 24) * 3600)                # scheduled off-peak charging
            s0 = min(b - 300, a + delay)
            if s0 > a:
                self.fill_idle(a, s0); fin = self.charge(s0, "AC", kw, self.DR["target"], b - s0)
                self.fill_idle(fin, b); return
        self.fill_idle(a, b)

    def fast_stop(self, cursor, target):
        self.at = -1; self.fill_idle(cursor, cursor + 180); fin = self.charge(cursor + 180, "DC", min(self.V["dc"], 150), target, 3600); self.fill_idle(fin, fin + 120); return fin + 120

    # ------------------------------------------------------------------ driving
    def estimate(self, hw): return self.est_city * (1 - hw) + self.est_hwy * hw

    def drive(self, s, leg_km, hw, ag, ids, jmeta):
        """Drive up to leg_km from second s. Returns (end_second, km_driven). Stops early (battery nearly empty) so the caller can insert a fast-charge stop."""
        V, C, rng, world, m, n = self.V, self.C, self.rng, self.world, self.m, self.n
        mh = self.local_hour(s); rush = (7.0 <= mh < 9.8) or (16.8 <= mh < 19.8); traffic = min(1.05, (0.78 if rush else 1.0) / C["cong"] * float(rng.uniform(0.93, 1.07)))
        ag = ag + (1.0 if ("harsh_week" in self.st and self.st["harsh_week"]["start"] <= s // 86400 < self.st["harsh_week"]["start"] + 6) else 0.0)
        v, lo, la = build_speed(rng, self.seg, leg_km, hw if leg_km > 3 else 0.0, ag, traffic, self.rain_at(s) > 0, C["cap"])
        v = np.minimum(v, V["top"]); L = min(len(v), n - s)
        if L < 60: return s, 0.0
        v, lo, la = v[:L], lo[:L], la[:L]
        low_p = float(np.mean(self.spec - self.cold_at(s))); payload = 80 + float(rng.choice([0, 0, 75, 150, 300])); cda = V["cda"] * (1.0 + (0.22 if s >= self.roof_t else 0.0)); mass = V["mass"] * 1.04 + payload
        wind = float(world.wind[min(s // 60, m - 1)]) / 3.6 * 0.35 * float(rng.choice([-1, 1])); reg_cap = min(V["regen"], max(8.0, V["regen"] * float(temp_derate(self.tb)))) * 1000.0
        for attempt in range(3):
            L = len(v); idx = np.arange(s, s + L); mi = np.minimum(idx // 60, m - 1); T = self.amb[s:s + L].astype(np.float64); sun = world.sun[mi]; ts = np.arange(L, dtype=np.float64); rain_l = world.rain[mi] > 0
            vm = v / 3.6; acc = np.gradient(vm); acc = np.convolve(np.r_[acc[0], acc, acc[-1]], np.ones(3) / 3, "valid")
            drift = 1.0 + (self.drift_rate * max(0.0, (s - self.drift_t) / 86400.0) if s >= self.drift_t else 0.0)             # story: rolling resistance creeps up from a given day on
            crr = V["crr"] * drift * (1 + 0.0008 * max(0.0, low_p) + 0.004 * np.clip(15 - T, 0, 35) / 10 + 0.10 * rain_l)
            rho = 1.225 * 288.15 / (T + 273.15); vair = vm + wind; p_w = (mass * acc + 0.5 * rho * cda * vair * np.abs(vair) + crr * mass * G) * vm; hv = hvac_w(V, T, sun, ts + 120)
            elec = np.where(p_w >= 0, p_w / V["eta"], 0.0); rg = np.where(p_w < 0, np.maximum(p_w * 0.72, -reg_cap), 0.0); fric = np.where(p_w < 0, p_w - rg, 0.0)
            pk = (elec + rg + hv + V["aux"]) / 1000.0; pk = np.where(pk > 0, pk * (1.0 + 0.004 * max(0.0, 20.0 - self.tb)), pk); cum = np.cumsum(pk) / 3600.0
            lim = self.e - 0.05 * self.cap; over = cum > lim
            if over.any() and attempt < 2:                                                         # not enough energy for the whole leg: stop early, ramp down, charge, continue
                k = max(120, int(np.argmax(over)) - 20)
                if k < L - 30: v, lo, la = v[:k].copy(), lo[:k].copy(), la[:k].copy(); r = min(14, k // 4); v[-r:] *= np.linspace(1, 0, r); v[-1] = 0.0; continue
            break
        e_trip = min(float(cum[-1]), max(self.e - 0.2, 0.0)); soc_arr = 100 * np.maximum(self.e - cum, 0.0) / self.cap
        f_ = s + L; tbs = np.empty(L); fls = np.empty(L); tts = np.empty(L); ua0 = 280.0 * (self.cap / 60.0); cth = 150e3 * (self.cap / 60.0); tb, tt = self.tb, self.tt; nz = rng.normal(0, 0.012, L)
        pl, Tl, vl = pk.tolist(), T.tolist(), v.tolist(); pump = self.pump_t
        for i in range(L):
            p = pl[i]; a_ = abs(p); base = 10.0 + min(4.0, a_ * 0.04 + 0.2); si = s + i
            if si >= pump: base *= max(0.30, 1 - 0.70 * (si - pump) / (8 * 86400.0))
            fl = base * (1 + nz[i]); ua = ua0 * cond(fl) * (1.0 + 0.8 * min(1.0, max(0.0, (fl - 5.0) / 5.0)) if tb > 33 else 1.0)
            tb += (((0.0004 * a_ * a_ + 0.010 * a_) * (1.0 + 0.03 * max(0.0, 20.0 - tb))) * 1000.0 - ua * (tb - Tl[i])) / cth; tbs[i] = tb; fls[i] = fl
            tt += ((Tl[i] + 3 + 0.16 * vl[i]) - tt) / 600.0; tts[i] = tt
        self.tb, self.tt = tb, tt; sl = slice(s, f_)
        self.speed[sl] = v; self.power[sl] = pk; self.soc[sl] = soc_arr; self.batt[sl] = tbs; self.flow[sl] = fls; self.hvac[sl] = hv; self.tyre_t[sl] = tts; self.chg[sl] = 0
        self.motor[sl] = T + 10 + 1.3 * movavg(np.abs(pk), 240); self.ign[sl] = True; self.mode[sl] = np.where(v > 72, 2, 1).astype(np.int8)
        self.odo[sl] = self.odo_km + np.cumsum(v) / 3600.0; kmt = float(v.sum() / 3600.0); self.odo_km += kmt
        self.e = max(0.0, self.e - e_trip); self.fric_kwh += float(-fric.sum() / 3.6e6); self.regen_kwh += float(-rg.sum() / 3.6e6)
        if kmt > 3:
            w = e_trip / kmt; hh = float((v > 72).mean())
            if hh > 0.6: self.est_hwy = 0.7 * self.est_hwy + 0.3 * w
            elif hh < 0.2: self.est_city = 0.8 * self.est_city + 0.2 * w
        self.trips.append(dict(start=self.t0 + s, end=self.t0 + f_, km=round(kmt, 2), kwh=round(e_trip, 3), wh_km=round(1000 * e_trip / max(kmt, 0.1), 1), avg_speed=round(float(v.mean()), 1), hwy_share=round(float((v > 72).mean()), 3),
                               avg_amb=round(float(T.mean()), 1), avg_hvac_w=round(float(hv.mean()), 0), soc_start=float(100 * (self.e + e_trip) / self.cap), soc_end=float(100 * self.e / self.cap), day=int(s // 86400),
                               a=ids[0], b=ids[1], jid=jmeta[0], j_off=jmeta[1], j_km=jmeta[2], oa=jmeta[3], ob=jmeta[4], regen_kwh=round(float(-rg.sum() / 3.6e6), 3), fric_kwh=float(-fric.sum() / 3.6e6), rain_share=round(float(rain_l.mean()), 3),
                               lon=lo, lat=la, s=s, v=v))
        return f_, kmt

    # ------------------------------------------------------------------ the whole timeline
    def run(self, trips_plan):
        n, plan, rng = self.n, self.plan, self.rng; cur = 0
        for tp in trips_plan:
            a, b = tp["a"], tp["b"]; km = plan.km_between(a, b); hw = plan.hwy_share(a, b, km); start = max(int(tp["t"]), cur + 120)
            if start >= n - 3600: break
            if a != self.at: continue                                                      # the car is not where this trip starts (an earlier trip was skipped): skip it too, so places chain correctly
            if self.rain_at(start) >= 2 and plan.places[b]["kind"] == "other" and rng.random() < 0.3: continue          # discretionary trips are skipped in heavy rain
            self.idle_with_charge(cur, start, self.at); cursor = start; self.jid += 1; off = 0.0; rem = km; guard = 0; tol = min(max(1.6, 0.04 * km), 0.25 * km)                                  # short hops (1-2 km) must not be skipped by the tolerance
            while rem > tol and cursor < n - 1800 and guard < 8:
                guard += 1
                afford = max(0.0, (self.e - 0.10 * self.cap) / (self.estimate(hw) * 1.15))
                if afford < min(rem, 12.0): cursor = self.fast_stop(cursor, 0.8); continue                                   # cannot even make the next stretch: charge first
                step = rem if afford >= rem * 1.05 else afford * 0.92
                end, got = self.drive(cursor, step, hw, self.DR["aggr"], (a if off == 0 else -1, -2), (self.jid, off, km, a, b))
                if got <= 0: break
                off += got; rem -= got; cursor = end
                if rem > tol and cursor < n - 3600: cursor = self.fast_stop(cursor, 0.8)
                elif rem <= tol: self.trips[-1]["b"] = b
            cur = cursor; self.at = b if cursor < n else self.at
            if "cooling_pump" in self.st and not getattr(self, "forced_dc", False) and cur > self.pump_t + 7 * 86400 and cur < n - 7200 and self.e < 0.40 * self.cap:       # story: a week after the pump fails the driver fast-charges - the weakened cooling cannot cope
                at0 = self.at; cur = self.fast_stop(cur, 0.95); self.at = at0; self.forced_dc = True
        if cur < n: self.idle_with_charge(cur, n, self.at)


def build(seed=1, days=28, city="bengaluru", driver="commuter", vehicle="family", end=None, stories=None, offline=False, verbose=True, work_km=None):
    t_all = time.time(); rng = np.random.default_rng(seed); V = VEHICLES[vehicle]; C = CITIES[city]; DR = DRIVERS[driver]
    end_d = dt.date.fromisoformat(end) if end else (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=8))
    d0 = end_d - dt.timedelta(days=days - 1); tzh = tz_offset_h(C["tz"], d0)
    t0 = int(dt.datetime(d0.year, d0.month, d0.day, tzinfo=dt.timezone.utc).timestamp() - tzh * 3600)             # local midnight of day 0 as a UTC epoch
    n = days * 86400
    if stories is None: stories = [s for s in STORIES if rng.random() < {"cooling_pump": 0.3, "tyre_leak": 0.5, "weak_12v": 0.25, "dirty_windscreen": 0.35, "roadworks": 0.3, "roof_box": 0.25, "harsh_week": 0.3, "alignment_drift": 0.0}[s]]
    st = {s: {} for s in stories}
    if "tyre_leak" in st: st["tyre_leak"] = dict(start=float(rng.uniform(1, max(2, days - 14))), wheel=int(rng.integers(0, 4)), rate=float(rng.uniform(3.0, 8.0)))
    if "cooling_pump" in st: st["cooling_pump"] = dict(start=int(rng.integers(max(2, days // 3), max(3, days - 8))))
    if "weak_12v" in st: st["weak_12v"] = dict(start=int(rng.integers(2, max(3, days - 8))))
    if "dirty_windscreen" in st: st["dirty_windscreen"] = dict(start=int(rng.integers(3, max(4, days - 8))), days=int(rng.integers(3, 8)))
    if "roadworks" in st: st["roadworks"] = dict(days=sorted(set(int(x) for x in rng.choice(days, size=min(4, days), replace=False))))
    if "roof_box" in st: st["roof_box"] = dict(start=int(rng.integers(days // 4, max(days // 4 + 1, days - 5))))
    if "harsh_week" in st: st["harsh_week"] = dict(start=int(rng.integers(0, max(1, days - 8))))
    if "alignment_drift" in st: st["alignment_drift"] = dict(start=int(days * float(rng.uniform(0.45, 0.6))), rate=float(rng.uniform(0.006, 0.010)))
    world = World(rng, city, t0, n, offline); seg = Segments(); plan = Plan(rng, city, driver, days, d0, tzh, work_km); tplan = plan.schedule()
    if verbose: print(f"  {C['label']} | {DR['label']} | {V['label']} | {d0}..{end_d} | stories: {', '.join(stories) or 'none'} | weather: {world.source}")
    life = Life(rng, V, C, DR, world, seg, plan, days, t0, tzh, st); life.run(tplan)
    gv = Vehicle(t0, n, life.ign, life.speed, life.mode, life.soc, life.power, life.batt, life.flow, life.amb, life.hvac, life.motor, life.odo, life.chg)
    gv.trips = life.trips; gv.life, gv.world, gv.plan, gv.V, gv.C, gv.DR, gv.st, gv.tzh, gv.d0, gv.end_d, gv.seed, gv.city, gv.driver, gv.vehicle = life, world, plan, V, C, DR, st, tzh, d0, end_d, seed, city, driver, vehicle
    if verbose: print(f"  simulated {len(life.trips)} trips, {len(life.sessions)} charging sessions in {time.time() - t_all:.1f}s")
    from . import datagen_events as ev
    ev.finish(gv, rng, verbose)
    return gv

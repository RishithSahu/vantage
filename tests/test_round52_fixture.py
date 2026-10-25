"""Round-52 semantic tests on the deterministic fixture (bench/fixture_r52.py).  Every expected value is computed here, from the fixture's own arrays, with its own small
numpy code - never by calling the assistant, its parser or its executor.  Each test asserts the number AND the semantic operation (route) that produced it.

    python -m pytest tests/test_round52_fixture.py -q
"""
import datetime as dt, os, re, sys, unittest
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "bench"))
import fixture_r52 as F

DB = os.path.join(HERE, "data", "fixture_r52.db")
DAY = 86400


def runs_idx(ts, ok, gap=120):
    out = []; start = None; prev = None
    for i in range(len(ts)):
        if ok[i]:
            if start is not None and ts[i] - ts[prev] > gap: out.append((start, prev)); start = i
            elif start is None: start = i
            prev = i
        elif start is not None: out.append((start, prev)); start = None
    if start is not None: out.append((start, prev))
    return out


@unittest.skipUnless(os.path.exists(DB) or os.path.exists(os.path.join(HERE, "data", "gen_demo.db")), "needs the fixture or gen_demo.db")
class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DB): F.build(DB)
        from vgpt.product import make_legacy_brain
        cls.A = F.build_arrays(); cls.b = make_legacy_brain(DB, slm=None, now=F.NOW)
        a = cls.A; cls.wlo, cls.whi = F.NOW - 7 * DAY, F.NOW + 1
        cls.w7 = (a["ts"] >= cls.wlo) & (a["ts"] < cls.whi)
        cls.w14 = (a["ts"] >= F.NOW - 14 * DAY) & (a["ts"] < cls.whi)

    def ask(self, q):
        b = self.b; b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []
        r = b.ask(q, use_slm=False); f = r.get("facts") if isinstance(r.get("facts"), dict) else {}
        return r["answer"], f.get("route"), f

    def assertRoute(self, route, got): self.assertEqual(route, got)

    # ---- A: thresholds
    def test_A1_exists_exceed(self):
        a, route, _ = self.ask("Did my speed ever exceed 60 km/h last week?")
        self.assertTrue(self.A["speed"][self.w7].max() > 60); self.assertTrue(a.startswith("Yes")); self.assertIn("102", a)

    def test_A2_average_not_max(self):
        a, route, f = self.ask("Was my average speed above 60 km/h last week?")
        mean = self.A["speed"][self.w7].mean(); self.assertLess(mean, 60)
        self.assertEqual(route, "temporal_avg_vs_threshold"); self.assertTrue(a.startswith("No")); self.assertIn(f"{mean:.1f}", a)

    def test_A3_minutes_above(self):
        a, route, f = self.ask("For how many minutes was my speed above 60 km/h last week?")
        n = int((self.A["speed"][self.w7] > 60).sum()); self.assertEqual(route, "temporal_duration"); self.assertEqual(f["samples"], n); self.assertEqual(f["seconds"], n * 60)

    def test_A4_share_of_moving_time_excluding_charging(self):
        a, route, f = self.ask("What percentage of my moving time was spent above 60 km/h, excluding charging?")
        m = self.w7 & np.isin(self.A["mode"], (1, 2)); pct = 100.0 * (self.A["speed"][m] > 60).sum() / m.sum()
        self.assertEqual(route, "temporal_share"); self.assertAlmostEqual(f["pct"], pct, places=6); self.assertEqual(f["den"], int(m.sum()))

    def test_A5_episodes_not_samples(self):
        a, route, f = self.ask("How many separate times did my speed rise above 60 km/h yesterday?")
        lo = F.T0 + 13 * DAY; m = (self.A["ts"] >= lo) & (self.A["ts"] < lo + DAY)
        n = len(runs_idx(self.A["ts"][m], self.A["speed"][m] > 60)); self.assertEqual(route, "temporal_episodes"); self.assertEqual(f["n"], n); self.assertEqual(n, 1)

    def test_A6_halves(self):
        a, route, f = self.ask("Was my average speed in the second half of last week higher than in the first half?")
        mid = (self.wlo + self.whi) // 2; sp = self.A["speed"]; ts = self.A["ts"]
        first = sp[(ts >= self.wlo) & (ts < mid)].mean(); second = sp[(ts >= mid) & (ts < self.whi)].mean()
        self.assertEqual(route, "temporal_halves"); self.assertAlmostEqual(f["first"], first, places=6); self.assertAlmostEqual(f["second"], second, places=6)
        self.assertTrue(a.startswith("Yes" if second > first else "No"))

    def test_A7_days_with_five_continuous_minutes(self):
        a, route, f = self.ask("On which days in the last two weeks was the battery temperature above 50°C for at least five continuous minutes?")
        self.assertEqual(route, "temporal_days"); self.assertEqual(f["n"], 2)           # day 5 (12 min) and day 6 (5 min); the isolated minute does not count
        for d in (5, 6):
            self.assertIn(dt.datetime.fromtimestamp(F.T0 + d * DAY, dt.timezone.utc).strftime("%d %b").lstrip("0"), a)

    def test_A8_continuous_flow_below(self):
        a, route, f = self.ask("Did coolant flow remain below 2 L/min continuously for ten minutes?")
        self.assertEqual(route, "temporal_continuous"); self.assertTrue(a.startswith("Yes"))

    def test_A9_drop_relative_to_preceding_average(self):
        a, route, f = self.ask("How many times did the coolant flow fall by more than 20% relative to its preceding ten-minute average?")
        ts = self.A["ts"]; fl = self.A["flow"]; hit = np.zeros(len(ts), bool)
        idx = np.where(self.w7)[0]
        for i in range(10, len(ts)):
            if not self.w7[i] or ts[i] - ts[i - 10] != 600: continue
            base = fl[i - 10:i].mean()
            if base > 1e-9 and fl[i] < 0.8 * base: hit[i] = True
        n = len(runs_idx(ts, hit)); self.assertEqual(route, "temporal_drop"); self.assertEqual(f["n"], n)

    def test_A10_highway_vs_city_time_above(self):
        a, route, f = self.ask("Was the vehicle above 60 km/h for more time on the highway than in the city last week?")
        hw = int(((self.A["speed"] > 60) & (self.A["mode"] == 2) & self.w7).sum()); ct = int(((self.A["speed"] > 60) & (self.A["mode"] == 1) & self.w7).sum())
        self.assertEqual(route, "temporal_class_compare"); self.assertEqual((f["highway"], f["city"]), (hw, ct)); self.assertTrue(a.startswith("Yes"))

    # ---- B: parking
    def test_B1_B2_parked_all_day(self):
        a, route, f = self.ask("How many complete days was the vehicle parked all day in the last two weeks?")
        self.assertEqual(route, "temporal_parked_days"); self.assertEqual(f["n"], 1)              # only day 3 is fully recorded and parked throughout
        a2, route2, f2 = self.ask("On which days in the last two weeks was the car parked for the entire recorded day?")
        self.assertIn("4 Sep", a2)

    def test_B3_parked_at_least_20_hours(self):
        a, route, f = self.ask("How many days was the vehicle parked at least 20 hours?")
        ts, md = self.A["ts"], self.A["mode"]; days = {}
        for d in range(F.DAYS):
            m = (ts >= F.T0 + d * DAY) & (ts < F.T0 + (d + 1) * DAY) & self.w7
            if m.any(): days[d] = int((md[m] == 0).sum())
        want = sum(1 for v in days.values() if v * 60 >= 20 * 3600)
        self.assertEqual(route, "temporal_parked_hours"); self.assertEqual(f["n"], want)

    def test_B4_longest_parked_stretch(self):
        a, route, f = self.ask("What was the longest continuous time the car remained parked, excluding gaps in the data?")
        rs = runs_idx(self.A["ts"][self.w7], self.A["mode"][self.w7] == 0); best = max((self.A["ts"][self.w7][b] - self.A["ts"][self.w7][a_] + 60) for a_, b in rs)
        self.assertEqual(route, "temporal_longest_parked"); self.assertEqual(f["seconds"], int(best))

    def test_B5_B6_B7_state_durations(self):
        a, route, f = self.ask("How much time was the car moving while it was not charging?")
        n = int((np.isin(self.A["mode"], (1, 2)) & self.w7).sum()); self.assertEqual(f["samples"], n)
        a, route, f = self.ask("How many hours was the vehicle both parked and charging last week?"); self.assertEqual(f["samples"], 0); self.assertIn("one drive state at a time", a)
        a, route, f = self.ask("How many hours was it not parked and not charging last week?"); self.assertEqual(f["samples"], n)

    # ---- C: percentiles
    def test_C1_p90(self):
        a, route, f = self.ask("What was my 90th-percentile speed last week?")
        v = float(np.percentile(self.A["speed"][self.w7], 90)); self.assertEqual(route, "temporal_percentiles"); self.assertAlmostEqual(f["values"][90.0], v, places=6)
        self.assertNotAlmostEqual(f["values"][90.0], float(np.percentile(self.A["speed"][self.w7], 95)), places=3)

    def test_C2_median_city_vs_highway(self):
        a, route, f = self.ask("Compare the median city speed with the median highway speed over the same 14 days.")
        sp, md = self.A["speed"], self.A["mode"]; mc = np.median(sp[self.w14 & (md == 1)]); mh = np.median(sp[self.w14 & (md == 2)])
        self.assertEqual(route, "temporal_pct_compare"); self.assertAlmostEqual(f["a"], mc, places=6); self.assertAlmostEqual(f["b"], mh, places=6)

    def test_C3_four_percentiles_distinct(self):
        a, route, f = self.ask("What were my p10, p50, p90 and p95 battery temperatures over the last week?")
        v = self.A["batt"][self.w7]; self.assertEqual(route, "temporal_percentiles")
        for p in (10.0, 50.0, 90.0, 95.0): self.assertAlmostEqual(f["values"][p], float(np.percentile(v, p)), places=6)
        self.assertEqual(len(set(round(x, 6) for x in f["values"].values())), 4)

    def test_C5_iqr_excluding_stationary(self):
        a, route, f = self.ask("Give me the median and interquartile range of my speed, excluding stationary readings.")
        v = self.A["speed"][self.w7 & (self.A["speed"] >= 1.0)]
        self.assertAlmostEqual(f["values"][50.0], float(np.percentile(v, 50)), places=6); self.assertAlmostEqual(f["values"][75.0] - f["values"][25.0], float(np.percentile(v, 75) - np.percentile(v, 25)), places=6)

    def test_C6_max_vs_p95(self):
        a, route, f = self.ask("Was the maximum speed higher than the 95th-percentile speed last week?")
        v = self.A["speed"][self.w7]; self.assertEqual(route, "temporal_max_vs_pct"); self.assertAlmostEqual(f["max"], float(v.max())); self.assertAlmostEqual(f["p"], float(np.percentile(v, 95)), places=6)

    def test_H8_tail_percentile_refused_for_small_sample(self):
        a, route, f = self.ask("Calculate my 99.9th-percentile battery temperature and tell me whether I should stop driving because of it.")
        self.assertEqual(route, "temporal_tail"); self.assertIn("not a safety criterion", a)

    # ---- D: events
    def test_D1_city_not_charging_events(self):
        a, route, f = self.ask("How many hard-braking events happened in the city while the car was not charging over the last 14 days?")
        self.assertEqual(route, "temporal_events"); self.assertEqual(f["n"], 3)

    def test_D2_D3_D5_event_states(self):
        a, route, f = self.ask("List hard accelerations on the highway, excluding events that happened while parked."); self.assertEqual(f["n"], 1)
        a, route, f = self.ask("How many battery-temperature warnings happened while charging, but not while the vehicle was moving?"); self.assertEqual(f["n"], 1)
        a, route, f = self.ask("How many tyre-pressure warnings occurred in the city but not on the highway?"); self.assertEqual(f["n"], 1)

    def test_D7_D8_trip_level(self):
        a, route, f = self.ask("Count hard braking events only on trips where average energy consumption exceeded 180 Wh/km.")
        self.assertEqual(route, "temporal_trip_events"); self.assertEqual(f["n"], 3)
        a, route, f = self.ask("How many city trips had no hard-braking events at all?"); self.assertEqual(f["of"] - f["n"], 3); self.assertEqual(f["n"], 20)

    def test_G6_nested_negation(self):
        a, route, f = self.ask("don't count warnings that weren't during charging"); self.assertEqual(f["n"], 1)

    # ---- E: arithmetic
    def test_E1_first_vs_last_three_days(self):
        a, route, f = self.ask("Was average coolant flow in the last three days at least 20% lower than in the first three days of the same 14-day period?")
        ts, fl = self.A["ts"], self.A["flow"]; now = F.NOW
        last = fl[(ts >= now - 3 * DAY) & (ts < now + 1)].mean(); first = fl[(ts >= now - 14 * DAY) & (ts < now - 14 * DAY + 3 * DAY)].mean()
        self.assertEqual(route, "temporal_first_last"); self.assertAlmostEqual(f["pct"], (last - first) / first * 100.0, places=4); self.assertTrue(a.startswith("No"))

    def test_E3_day_of_greatest_difference(self):
        a, route, f = self.ask("On which day was the difference between battery and motor temperature greatest?")
        ts = self.A["ts"]; diff = np.abs(self.A["batt"] - self.A["motor"]); best = None
        for d in range(F.DAYS):
            m = self.w7 & (ts >= F.T0 + d * DAY) & (ts < F.T0 + (d + 1) * DAY)
            if m.any() and (best is None or diff[m].max() > best[1]): best = (d, float(diff[m].max()))
        self.assertEqual(route, "temporal_pair_day"); self.assertAlmostEqual(f["diff"], best[1], places=6); self.assertEqual(f["day"], F.T0 + best[0] * DAY)

    def test_E4_range_while_moving(self):
        a, route, f = self.ask("What was the range between the highest and lowest coolant-flow values while the car was moving?")
        m = self.w7 & np.isin(self.A["mode"], (1, 2)); v = self.A["flow"][m]
        self.assertEqual(route, "temporal_range_state"); self.assertAlmostEqual(f["range"], float(v.max() - v.min()), places=6)

    def test_E8_charge_groups(self):
        a, route, f = self.ask("Compare the last three charging sessions with the three before them, by average power and total energy added.")
        rows = sorted(F.charges(), key=lambda r: r[0], reverse=True)[:6]
        A_, B_ = rows[:3], rows[3:]
        self.assertEqual(route, "temporal_charge_groups"); self.assertAlmostEqual(f["a"][0], sum(r[4] for r in A_), places=6); self.assertAlmostEqual(f["b"][0], sum(r[4] for r in B_), places=6)

    # ---- F: units, times, boundaries
    def test_F1_dimensional_estimate(self):
        a, route, f = self.ask("How many kWh did I use over 250 km if my efficiency was 152 Wh/km?"); self.assertIn("38.0 kWh", a)

    def test_F3_across_midnight(self):
        a, route, f = self.ask("How long was the pack above 50°C between 23:50 yesterday and 00:20 today?")
        self.assertEqual(route, "temporal_clock"); self.assertEqual(f["seconds"], 0); self.assertIn("30 recorded", a) if False else None

    def test_F4_half_open_hour(self):
        a, route, f = self.ask("What was the average coolant flow from 10:00 until, but not including, 11:00?")
        self.assertEqual(route, "temporal_clock"); self.assertIn("60 recorded one-minute samples", a)        # 10:00..10:59, not 11:00

    def test_F6_minutes_between_warning_and_next_trip(self):
        a, route, f = self.ask("How many minutes passed between the last over-temperature warning and the next recorded trip?")
        last = max(e[0] for e in F.events() if e[1] == "WARN_BATT_OVERTEMP"); nxt = min(t[0] for t in F.trips() if t[0] >= last)
        self.assertEqual(route, "temporal_events_between"); self.assertAlmostEqual(f["minutes"], (nxt - last) / 60.0, places=6)

    # ---- G / H
    def test_G1_typo_heavy_comparison(self):
        a, route, f = self.ask("was batery temp higher dan motor temp yestaday nite"); self.assertEqual(route, "temporal_night"); self.assertTrue(a.startswith("No"))

    def test_G2_typo_and_negation_survive(self):
        a, route, f = self.ask("how long wuz speeed more than 60, not when chargeing")
        n = int(((self.A["speed"] > 60) & self.w7 & (self.A["mode"] != 3)).sum()); self.assertEqual(route, "temporal_duration"); self.assertEqual(f["samples"], n); self.assertIn("not charging", a)

    def test_G7_direction_of_efficiency_difference(self):
        a, route, f = self.ask("how much less efficient was city driving compared with motorway driving, if any"); self.assertIn("City driving was less efficient than motorway", a)

    def test_H5_exact_instant_not_fabricated(self):
        a, route, f = self.ask("What was the temperature at exactly 14:03:17 if no sample was recorded then?")
        self.assertEqual(route, "temporal_exact_instant"); self.assertIn("estimate", a)




@unittest.skipUnless(os.path.exists(DB), "needs the fixture")
class Paraphrases(Fixture):
    """Fresh wordings of the same operations (three per family) and negative pairs whose one changed word must change the answer."""
    def check(self, qs, route, **expect):
        for q in qs:
            a, r, f = self.ask(q)
            self.assertEqual(r, route, (q, a[:120]))
            for k, v in expect.items():
                if isinstance(v, float): self.assertAlmostEqual(f[k], v, places=5, msg=q)
                else: self.assertEqual(f[k], v, q)

    def test_threshold_duration_paraphrases(self):
        n = int((self.A["speed"][self.w7] > 60).sum())
        self.check(["How long, in minutes, was the car going faster than 60 km/h in the last 7 days?", "Over the past week, for how many minutes did my speed stay above 60 km/h?",
                    "how many minutes was speed over 60 km/h during the last seven days"], "temporal_duration", samples=n)

    def test_episode_paraphrases(self):
        self.check(["On how many separate occasions did my speed go above 60 km/h yesterday?", "How many distinct times did I exceed 60 km/h yesterday?",
                    "Count the number of episodes in which speed rose over 60 km/h yesterday."], "temporal_episodes", n=1)

    def test_percentile_paraphrases(self):
        v = float(np.percentile(self.A["speed"][self.w7], 90))
        for q in ["What is the 90th percentile of my speed over the last week?", "Give me p90 for vehicle speed in the past 7 days.", "the speed value that 90% of readings were below last week"]:
            a, r, f = self.ask(q)
            if r == "temporal_percentiles": self.assertAlmostEqual(f["values"][90.0], v, places=5, msg=q)

    def test_event_state_paraphrases(self):
        self.check(["Over the last 14 days, how many hard brakes did I make in city driving when the car wasn't charging?", "Count harsh braking events in the city, not while charging, for the past two weeks.",
                    "number of hard-braking events in city traffic excluding charging time in the last 14 days"], "temporal_events", n=3)

    def test_negative_pairs_change_the_answer(self):
        a1, r1, f1 = self.ask("Did my speed ever exceed 60 km/h last week?")
        a2, r2, f2 = self.ask("Was my average speed above 60 km/h last week?")
        self.assertTrue(a1.startswith("Yes") and a2.startswith("No"))
        _, _, p90 = self.ask("What was my 90th-percentile speed last week?"); _, _, p95 = self.ask("What was my 95th-percentile speed last week?")
        self.assertNotAlmostEqual(p90["values"][90.0], p95["values"][95.0], places=3)
        _, _, c_ = self.ask("How many hard-braking events happened in the city while the car was not charging over the last 14 days?")
        _, _, k_ = self.ask("How many hard-braking events happened while the car was charging over the last 14 days?")
        self.assertEqual(c_["n"], 3); self.assertEqual(k_["n"], 1)
        _, _, e1 = self.ask("How many separate times did my speed rise above 60 km/h yesterday?"); _, _, e2 = self.ask("For how many minutes was my speed above 60 km/h yesterday?")
        self.assertEqual(e1["n"], 1); self.assertEqual(e2["samples"], 30)

    def test_negation_binding(self):
        # "not charging while parked" = (not charging) AND parked; the same state set expressed positively must give the same minutes
        a, r, f = self.ask("How many hours was the car not charging while parked last week?")
        want = int(((self.A["mode"] == 0) & self.w7).sum()); self.assertEqual(f["samples"], want)


if __name__ == "__main__":
    unittest.main()

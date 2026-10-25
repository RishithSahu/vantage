"""Fourth generator for the VANTAGE campaign: compositional questions (arg-extremum over days, day counts, change between two windows, difference between two signals,
range between highest and lowest, negated conditions, events under conditions, share of time in a state).  The expected value of every question is computed directly
from the recorded one-minute samples with plain SQL, independently of the typed engine.
    python bench/vantage_stress4.py [--n 300] [--seed 4] [--now 1787625900]"""
import argparse, collections, datetime as dt, json, os, random, re, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)

TZ = 19800
DAY = 86400
SIGCOL = {"speed": ("speed", 10.0, "km/h", "speed"), "battery temperature": ("batt", 10.0, "°C", "battery temperature"), "motor temperature": ("motor", 10.0, "°C", "motor temperature")}


def nums_of(a): return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", a)]
def near(nums, v, rel=0.015, ab=0.12): return any(abs(n - v) <= max(ab, abs(v) * rel) for n in nums)
CLAR = re.compile(r"i want to be sure|do you mean|reply with a number|could not turn that|i'm not sure i understood|doesn't look like a question|I do not record|can't compare", re.I)


def day_name(ts):
    d = dt.datetime.fromtimestamp(ts + TZ, dt.timezone.utc); return f"{d.strftime('%A')} {d.day} {d.strftime('%b')}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=4); ap.add_argument("--now", type=int, default=1787625900)
    ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_demo.db")); ap.add_argument("--out", default=os.path.join(HERE, "results", "vantage_stress4.json")); a = ap.parse_args(); rng = random.Random(a.seed)
    from vgpt.product import make_legacy_brain
    from vgpt.brain import LazySlm
    slm = LazySlm(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), idle_s=3600, threads=2, mmap=True, fmt="words")
    b = make_legacy_brain(a.db, slm=slm, now=a.now); c = b.c; res = []; t0 = time.perf_counter(); now = a.now
    today0 = ((now + TZ) // DAY) * DAY - TZ
    def fresh(): b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []

    def rows(col, lo, hi, extra=""):
        return c.execute(f"SELECT ts,{col},mode FROM sig_1m WHERE ts>=? AND ts<? AND {col} IS NOT NULL {extra}", (lo, hi)).fetchall()

    for i in range(a.n):
        fam = rng.choice(["argday", "countdays", "change", "gap", "extent", "negate", "evcond", "share", "implicit", "whichdays", "groupcount", "polar", "clock", "exists", "session", "wheel", "tripcount", "lasttrip", "rain"]); fl = []; ans = ""; asked = ""
        try:
            fresh()
            if fam == "argday":
                sig = rng.choice(list(SIGCOL)); col, sc, unit, nm = SIGCOL[sig]; n = rng.choice([5, 7, 10]); hi = rng.choice([True, False])
                lo_t = now - n * DAY; data = rows(col, lo_t, now + 1)
                byday = collections.defaultdict(list)
                for ts, v, m in data: byday[((ts + TZ) // DAY) * DAY - TZ].append(v / sc)
                means = {d: sum(v) / len(v) for d, v in byday.items()}
                if len(means) < 3: continue
                best = (max if hi else min)(means.items(), key=lambda x: x[1])
                word = rng.choice(["highest", "most"]) if hi else rng.choice(["lowest", "least"])
                asked = rng.choice([f"which day in the last {n} days had the {word} average {nm}", f"on which day over the last {n} days was the average {nm} the {word}"])
                ans = b.ask(asked).get("answer") or ""; head = ans.split("\n")[0]
                if CLAR.search(ans): fl.append("clarification")
                elif day_name(best[0]) not in head: fl.append(f"wrong_day(expected {day_name(best[0])} {best[1]:.1f})")
                elif not near(nums_of(head), best[1], 0.03, 0.3): fl.append(f"value_mismatch(expected {best[1]:.1f})")
            elif fam == "countdays":
                n = rng.choice([7, 10, 14]); th = rng.choice([60, 80, 90, 100]); lo_t = now - n * DAY
                data = rows("speed", lo_t, now + 1); byday = collections.defaultdict(list)
                for ts, v, m in data: byday[((ts + TZ) // DAY) * DAY - TZ].append(v / 10.0)
                hit = sum(1 for d, v in byday.items() if max(v) > th); total = len(byday)
                asked = rng.choice([f"how many days in the last {n} days did my speed exceed {th} km/h", f"on how many days over the last {n} days was my speed above {th} km/h"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not re.search(rf"\b{hit} of {total}\b", ans): fl.append(f"count_mismatch(expected {hit} of {total})")
            elif fam == "change":
                sig = rng.choice(list(SIGCOL)); col, sc, unit, nm = SIGCOL[sig]; n = 7
                A = rows(col, now - n * DAY, now + 1); B = rows(col, now - 2 * n * DAY, now - n * DAY)
                if not A or not B: continue
                va = sum(v for _, v, _ in A) / len(A) / sc; vb = sum(v for _, v, _ in B) / len(B) / sc
                asked = rng.choice([f"by how much did my average {nm} change between last week and the week before", f"how does my average {nm} last week compare with the week before"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not (near(nums_of(ans), va, 0.02, 0.15) and near(nums_of(ans), vb, 0.02, 0.15)): fl.append(f"values_mismatch(expected {va:.1f} and {vb:.1f})")
            elif fam == "gap":
                pair = rng.choice([("motor", "battery", "motor", "batt"), ("battery", "motor", "batt", "motor")]); lo_t = today0 - DAY; hi_t = today0
                A = rows(pair[2], lo_t, hi_t); B = rows(pair[3], lo_t, hi_t)
                if not A or not B: continue
                va = sum(v for _, v, _ in A) / len(A) / 10.0; vb = sum(v for _, v, _ in B) / len(B) / 10.0
                adj = "hotter" if va >= vb else "colder"
                asked = f"how much {adj} was the {pair[0]} than the {pair[1]} on average yesterday"
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not near(nums_of(ans), abs(va - vb), 0.05, 0.2): fl.append(f"gap_mismatch(expected {abs(va - vb):.1f})")
            elif fam == "extent":
                sig = rng.choice(["battery temperature", "motor temperature"]); col, sc, unit, nm = SIGCOL[sig]; n = rng.choice([5, 7, 10])
                D = rows(col, now - n * DAY, now + 1)
                if not D: continue
                hi_v = max(v for _, v, _ in D) / sc; lo_v = min(v for _, v, _ in D) / sc
                asked = rng.choice([f"what is the difference between the highest and lowest {nm} over the last {n} days", f"what was the range of the {nm} over the last {n} days between its highest and lowest"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not near(nums_of(ans), hi_v - lo_v, 0.05, 0.4): fl.append(f"range_mismatch(expected {hi_v - lo_v:.1f})")
            elif fam == "negate":
                n = rng.choice([3, 5, 7]); state = rng.choice([("charging", 3, "charging"), ("parked", 0, "parked")])
                D = rows("speed", now - n * DAY, now + 1); D = [(t, v, m) for t, v, m in D if m != state[1]]
                if not D: continue
                want = sum(v for _, v, _ in D) / len(D) / 10.0
                phrase = rng.choice([f"excluding the time I was {state[2]}", f"when not {state[2]}", f"except when {state[2]}"])
                asked = f"what was my average speed {phrase} over the last {n} days"
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not near(nums_of(ans), want, 0.03, 0.3): fl.append(f"negation_mismatch(expected {want:.1f})")
                if not re.search(r"excluding|not|except", ans, re.I): fl.append("exclusion_not_stated")
            elif fam == "evcond":
                n = rng.choice([7, 14]); cls = rng.choice([("city driving", 1), ("highway driving", 2)])
                ev = c.execute("SELECT ts FROM events WHERE type='HARSH_BRAKE' AND ts>=? AND ts<=?", (now - n * DAY, now)).fetchall()
                if not ev: continue
                modes = {t: m for t, m in c.execute("SELECT ts,mode FROM sig_1m WHERE ts>=? AND ts<=?", (now - n * DAY - 60, now + 60)).fetchall()}
                def mode_at(t):
                    k = (t // 60) * 60
                    return modes.get(k)
                inn = sum(1 for (t,) in ev if mode_at(t) == cls[1]); out = sum(1 for (t,) in ev if mode_at(t) is not None and mode_at(t) != cls[1])
                if rng.random() < 0.5: asked = f"how many hard brakes were there in {cls[0]} over the last {n} days"; want = inn
                else: asked = f"how many hard brakes were there excluding {cls[0]} over the last {n} days"; want = out
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not re.search(rf"\b{want}\b", ans.split("\n")[0]): fl.append(f"count_mismatch(expected {want})")
            elif fam == "share":
                n = rng.choice([3, 7, 10]); st = rng.choice([("parked", 0), ("charging", 3)])
                D = rows("speed", now - n * DAY, now + 1)
                if not D: continue
                want = 100.0 * sum(1 for _, _, m in D if m == st[1]) / len(D)
                asked = rng.choice([f"what fraction of the last {n} days was the car {st[0]}", f"what percentage of the last {n} days was the car {st[0]}"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not near(nums_of(ans), want, 0.04, 0.6): fl.append(f"share_mismatch(expected {want:.1f}%)")
            elif fam == "implicit":
                n = rng.choice([5, 7, 10]); kind = rng.choice([("fastest", "speed", True), ("slowest", "speed", False), ("coolest", "batt", False), ("hottest", "batt", True)])
                data = rows(kind[1], now - n * DAY, now + 1); byday = collections.defaultdict(list)
                for ts, v, m in data: byday[((ts + TZ) // DAY) * DAY - TZ].append(v / 10.0)
                means = {d: sum(v) / len(v) for d, v in byday.items()}
                if len(means) < 3: continue
                best = (max if kind[2] else min)(means.items(), key=lambda x: x[1])
                if kind[1] == "speed": asked = rng.choice([f"which was my {kind[0]} day in the last {n} days", f"what was my {kind[0]} day over the last {n} days"])
                else: asked = rng.choice([f"which day was the battery {kind[0]} in the last {n} days", f"on which day in the last {n} days was the battery {kind[0]}"])
                ans = b.ask(asked).get("answer") or ""; head = ans.split(chr(10))[0]
                if CLAR.search(ans): fl.append("clarification")
                elif day_name(best[0]) not in head: fl.append(f"wrong_day(expected {day_name(best[0])} {best[1]:.1f})")
            elif fam == "whichdays":
                n = rng.choice([7, 10, 14]); st = rng.choice([("charge", 3), ("park", 0)])
                data = rows("speed", now - n * DAY, now + 1); days = sorted({((ts + TZ) // DAY) * DAY - TZ for ts, v, m in data if m == st[1]})
                asked = f"on which days over the last {n} days did i {st[0]}"
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                else:
                    got = [d for d in sorted({((ts + TZ) // DAY) * DAY - TZ for ts, v, m in data}) if day_name(d) in ans]
                    if got != days: fl.append(f"days_mismatch(expected {[day_name(d) for d in days]})")
            elif fam == "groupcount":
                n = rng.choice([7, 14]); ev = c.execute("SELECT ts FROM events WHERE type='HARSH_BRAKE' AND ts>=? AND ts<=?", (now - n * DAY, now)).fetchall()
                if not ev: continue
                modes = {t: m for t, m in c.execute("SELECT ts,mode FROM sig_1m WHERE ts>=? AND ts<=?", (now - n * DAY - 60, now + 60)).fetchall()}
                city = sum(1 for (t,) in ev if modes.get((t // 60) * 60) == 1); hwy = sum(1 for (t,) in ev if modes.get((t // 60) * 60) == 2)
                asked = rng.choice([f"how many hard brakes did I have on the highway versus the city over the last {n} days", f"how many hard brakes in city driving and on the highway over the last {n} days"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans) or "What would you like" in ans: fl.append("clarification")
                else:
                    nums = [int(x) for x in re.findall(r"\b(\d+) hard", ans)]
                    if sorted(nums) != sorted([city, hwy]): fl.append(f"counts_mismatch(expected city {city}, highway {hwy})")
            elif fam == "polar":
                sig = rng.choice(list(SIGCOL)); col, sc, unit, nm = SIGCOL[sig]; n = 7
                A = rows(col, now - n * DAY, now + 1); B = rows(col, now - 2 * n * DAY, now - n * DAY)
                if not A or not B: continue
                va = sum(v for _, v, _ in A) / len(A); vb = sum(v for _, v, _ in B) / len(B)
                adj = rng.choice(["higher", "lower"]); truth = (va > vb) if adj == "higher" else (va < vb)
                asked = f"was my average {nm} {adj} in the last {n} days than in the week before"
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not ans.lower().startswith("yes" if truth else "no"): fl.append(f"verdict_wrong(expected {'yes' if truth else 'no'})")
            elif fam == "clock":
                h1 = rng.choice([6, 7, 8, 9, 12, 14, 17]); h2 = h1 + rng.choice([1, 2, 3]); sig = rng.choice(["speed", "battery temperature", "motor temperature"]); col, sc, unit, nm = SIGCOL[sig]
                lo_t = today0 - DAY + h1 * 3600; hi_t = today0 - DAY + h2 * 3600
                D = c.execute(f"SELECT {col} FROM sig_1s WHERE ts>=? AND ts<? AND {col} IS NOT NULL", (lo_t, hi_t)).fetchall()
                if len(D) < 30: continue
                want = sum(v for (v,) in D) / len(D) / sc
                f12 = lambda h: f"{h if h <= 12 else h - 12} {'am' if h < 12 else 'pm'}"
                asked = rng.choice([f"what was my average {nm} between {f12(h1)} and {f12(h2)} yesterday", f"average {nm} from {f12(h1)} to {f12(h2)} yesterday"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not near(nums_of(ans), want, 0.04, 0.4): fl.append(f"clock_mismatch(expected {want:.1f})")
            elif fam == "exists":
                th = rng.choice([40, 70, 90, 100, 120]); n = rng.choice([1, 3, 7])
                lo_t = today0 - DAY if n == 1 else now - n * DAY; hi_t = today0 if n == 1 else now + 1
                mx = c.execute("SELECT MAX(speed) FROM sig_1m WHERE ts>=? AND ts<?", (lo_t, hi_t)).fetchone()[0]
                if mx is None: continue
                yes = mx / 10.0 > th
                win = "yesterday" if n == 1 else f"in the last {n} days"
                asked = rng.choice([f"did my speed ever exceed {th} km/h {win}", f"did i go over {th} km/h {win}", f"was my speed above {th} km/h at any point {win}"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not ans.lower().startswith("yes" if yes else "no"): fl.append(f"verdict_wrong(expected {'yes' if yes else 'no'}; max {mx / 10.0:.1f})")
            elif fam == "session":
                rows_ = c.execute("SELECT start,end,kw,kwh,soc0,soc1 FROM charges WHERE end<=? ORDER BY start DESC LIMIT 2", (now,)).fetchall()
                if len(rows_) < 2: continue
                kind = rng.choice(["duration", "energy", "soc"]); two = rng.random() < 0.5
                if kind == "duration": vals = [(r[1] - r[0]) / 60.0 for r in rows_]; asked = "how long did my last charge take"
                elif kind == "energy": vals = [r[3] for r in rows_]; asked = "how much energy did my last charge add"
                else: vals = [r[5] - r[4] for r in rows_]; asked = "how much did the state of charge go up during my last charge"
                if two: asked += rng.choice([" compared to the one before", " versus the previous charge"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                else:
                    if kind == "duration":
                        mm = [int(x) * 60 + int(y) for x, y in re.findall(r"(\d+) h (\d+) min", ans)] + [int(x) for x in re.findall(r"(?<![\d:])(\d+) min\b", ans)]
                        ok_ = any(abs(m - vals[0]) <= 1.5 for m in mm) and (not two or any(abs(m - vals[1]) <= 1.5 for m in mm))
                    else:
                        nn = nums_of(ans); ok_ = near(nn, vals[0], 0.02, 0.15) and (not two or near(nn, vals[1], 0.02, 0.15))
                    if not ok_: fl.append(f"session_mismatch(expected {vals[:2 if two else 1]})")
            elif fam == "wheel":
                lo_t = today0 - DAY; hi_t = today0; low = rng.random() < 0.5
                per = {}
                for pos in ("fl", "fr", "rl", "rr"):
                    r_ = c.execute(f"SELECT {'MIN' if low else 'MAX'}(tp_{pos}) FROM ext_1m WHERE ts>=? AND ts<? AND tp_{pos} IS NOT NULL", (lo_t, hi_t)).fetchone()[0]
                    if r_ is None: per = {}; break
                    per[pos] = r_
                if not per: continue
                best = (min if low else max)(per, key=per.get); name = {"fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right"}[best]
                asked = f"which tyre had the {'lowest' if low else 'highest'} pressure yesterday"
                ans = b.ask(asked).get("answer") or ""; head = ans.split(chr(10))[0]
                if CLAR.search(ans): fl.append("clarification")
                elif name not in head: fl.append(f"wheel_wrong(expected {name})")
            elif fam == "tripcount":
                which = rng.choice(["yesterday", "in the last 7 days", "in the last 3 days"])
                lo_t, hi_t = (today0 - DAY, today0) if which == "yesterday" else (now - (7 if "7" in which else 3) * DAY, now + 1)
                n_ = c.execute("SELECT COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.2", (lo_t, hi_t)).fetchone()[0]
                asked = rng.choice([f"how many trips did i take {which}", f"how many trips did i make {which}"])
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not re.search(rf"\b{n_} (?:matching )?trips?\b", ans): fl.append(f"tripcount_mismatch(expected {n_})")
            elif fam == "lasttrip":
                r_ = c.execute("SELECT start,end,km FROM trips WHERE end<=? ORDER BY end DESC LIMIT 1", (now,)).fetchone()
                if not r_: continue
                kind = rng.choice(["long", "far"])
                asked = "how long was my last trip" if kind == "long" else "how far was my last trip"
                ans = b.ask(asked).get("answer") or ""
                want = (r_[1] - r_[0]) / 60.0 if kind == "long" else r_[2]
                if CLAR.search(ans): fl.append("clarification")
                elif not near(nums_of(ans), want, 0.03, 1.2): fl.append(f"lasttrip_mismatch(expected {want:.1f})")
            elif fam == "rain":
                n = rng.choice([7, 10]); lo_t = now - n * DAY
                D = c.execute("SELECT s.batt FROM sig_1m s JOIN ext_1m e ON e.ts = s.ts WHERE s.ts>=? AND s.ts<=? AND e.rain > 0", (lo_t, now)).fetchall()
                if len(D) < 5: continue
                lowv = min(v for (v,) in D) / 10.0
                asked = f"what was my lowest battery temperature when it was raining over the last {n} days"
                ans = b.ask(asked).get("answer") or ""
                if CLAR.search(ans): fl.append("clarification")
                elif not near(nums_of(ans), lowv, 0.03, 0.3): fl.append(f"rain_mismatch(expected {lowv:.1f})")
        except Exception as e: fl.append("EXCEPTION " + type(e).__name__ + ": " + str(e)[:80]); ans = ans or str(e)
        res.append({"family": fam, "asked": asked, "answer": ans[:500], "flags": fl, "status": "PASS" if not fl else "FAIL"})
    p = sum(r["status"] == "PASS" for r in res); print(f"{p}/{len(res)} pass in {time.perf_counter() - t0:.0f}s")
    cnt = collections.Counter((r["family"], r["status"]) for r in res)
    for f in sorted({r["family"] for r in res}): print(f"  {f:12s} {cnt[(f, 'PASS')]}/{cnt[(f, 'PASS')] + cnt[(f, 'FAIL')]}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()

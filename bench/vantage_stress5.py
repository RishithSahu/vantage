"""Fifth generator: compositions of several kinds of reasoning in one question (threshold x duration/share/episodes/percentile x state negation x window x typo), asked of
the deterministic fixture (bench/fixture_r52.py).  The expected value is computed here with numpy from the fixture's own arrays - not through the assistant.

    python bench/vantage_stress5.py [--n 300] [--seed 5] [--out results/vantage_stress5.json]
"""
import argparse, collections, json, os, random, re, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "bench"))
import fixture_r52 as F

DAY = 86400
SIG = {"speed": ("vehicle speed", "km/h", [30, 60, 90]), "batt": ("battery temperature", "°C", [25, 26, 30]), "motor": ("motor temperature", "°C", [34, 38, 42]), "flow": ("coolant flow", "L/min", [2, 9])}
STATES = [("", None), (" while not charging", ("not", 3)), (" excluding the time it was charging", ("not", 3)), (" while parked", ("is", 0)), (" while driving", ("in", (1, 2))),
          (" in city driving", ("is", 1)), (" on the highway", ("is", 2)), (" when it was not parked and not charging", ("in", (1, 2)))]
TYPOS = {"battery": "batery", "temperature": "temprature", "percentage": "percentge", "minutes": "minuts", "average": "averge", "separate": "seperate", "charging": "chargeing", "speed": "speeed"}


def window(name):
    if name == "7": return F.NOW - 7 * DAY, F.NOW + 1, "over the last 7 days"
    if name == "14": return F.NOW - 14 * DAY, F.NOW + 1, "over the last 14 days"
    d = ((F.NOW // DAY) * DAY) - DAY
    return d, d + DAY, "yesterday"


def mask_state(A, st):
    if st is None: return np.ones(len(A["ts"]), bool)
    kind, v = st; m = A["mode"]
    if kind == "is": return m == v
    if kind == "not": return m != v
    return np.isin(m, v)


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=5); ap.add_argument("--out", default=os.path.join(HERE, "results", "vantage_stress5.json"))
    a = ap.parse_args(); rng = random.Random(a.seed)
    db = os.path.join(HERE, "data", "fixture_r52.db")
    if not os.path.exists(db): F.build(db)
    from vgpt.product import make_legacy_brain
    b = make_legacy_brain(db, slm=None, now=F.NOW); A = F.build_arrays(); t0 = time.time(); res = []
    for i in range(a.n):
        fam = rng.choice(["duration", "share", "episodes", "percentile", "average"]); fl = []; ans = ""
        key = rng.choice(list(SIG)); label, unit, ths = SIG[key]; th = rng.choice(ths); above = rng.random() < 0.7
        sp, st = rng.choice(STATES); wn = rng.choice(["7", "14", "yd"]); lo, hi, wtxt = window(wn)
        w = (A["ts"] >= lo) & (A["ts"] < hi); sm = mask_state(A, st) & w; arr = A[key]
        cmp_w = "above" if above else "below"; cond = (arr > th) if above else (arr < th)
        tpl = rng.randrange(3)
        try:
            b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []
            if fam == "duration":
                q = [f"For how many minutes was the {label} {cmp_w} {th} {unit}{sp} {wtxt}?", f"How long was the {label} {cmp_w} {th} {unit}{sp} {wtxt}, in minutes?", f"how many minutes was {label} {cmp_w} {th} {unit}{sp} {wtxt}"][tpl]
                want = int((cond & sm).sum())
            elif fam == "share":
                q = [f"What percentage of the time was the {label} {cmp_w} {th} {unit}{sp} {wtxt}?", f"What share of recorded time had {label} {cmp_w} {th} {unit}{sp} {wtxt}?", f"what percent of the time was {label} {cmp_w} {th} {unit}{sp} {wtxt}"][tpl]
                dm = sm & np.isin(A["mode"], (1, 2)) if (st is None or st[0] == "not") and tpl != 1 else sm       # default denominator: driving time unless a state or "recorded time" is named
                want = (int((cond & dm).sum()), int(dm.sum())); sm_den = dm
            elif fam == "episodes":
                q = [f"How many separate times did the {label} go {cmp_w} {th} {unit}{sp} {wtxt}?", f"On how many separate occasions was {label} {cmp_w} {th} {unit}{sp} {wtxt}?", f"how many distinct times was the {label} {cmp_w} {th} {unit}{sp} {wtxt}"][tpl]
                want = len(runs_idx(A["ts"][sm], cond[sm]))
            elif fam == "percentile":
                p = rng.choice([10, 50, 90, 95]); name = {50: "median"}.get(p, f"{p}th percentile")
                q = [f"What was the {name} {label}{sp} {wtxt}?", f"Give me the {name} of the {label}{sp} {wtxt}.", f"what is the {name} {label}{sp} {wtxt}"][tpl]
                if p == 50: q = q.replace("the median", "the median")
                want = float(np.percentile(arr[sm], p)) if sm.sum() >= 20 else None
            else:
                q = [f"What was the average {label}{sp} {wtxt}?", f"What was the mean {label}{sp} {wtxt}?", f"average {label}{sp} {wtxt}"][tpl]
                want = float(arr[sm].mean()) if sm.any() else None
            if rng.random() < 0.3:
                for k, v in TYPOS.items():
                    if re.search(r"\b" + k + r"\b", q): q = re.sub(r"\b" + k + r"\b", v, q, count=1); break
            ans, route, f = (lambda r: (r["answer"], (r["facts"] or {}).get("route") if isinstance(r["facts"], dict) else None, r["facts"] if isinstance(r["facts"], dict) else {}))(b.ask(q, use_slm=False))
            if re.search(r"I want to be sure|do you mean|reply with a number|could not turn that", ans, re.I): fl.append("clarification")
            elif fam == "duration":
                if f.get("samples") != want: fl.append(f"duration_mismatch(expected {want} min, got {f.get('samples')}; route {route})")
            elif fam == "share":
                if sm_den.sum() == 0: pass
                elif (f.get("num"), f.get("den")) != want: fl.append(f"share_mismatch(expected {want}, got {(f.get('num'), f.get('den'))}; route {route})")
            elif fam == "episodes":
                if f.get("n") != want: fl.append(f"episodes_mismatch(expected {want}, got {f.get('n')}; route {route})")
            elif fam == "percentile":
                if want is not None:
                    got = (f.get("values") or {}).get(float(p))
                    need = 20 if p == 50 else int(np.ceil(10.0 / (min(p, 100 - p) / 100.0)))
                    if sm.sum() < need:
                        if not re.search(r"cannot be estimated|too few", ans): fl.append("expected a refusal for a small sample")
                    elif got is None or abs(got - want) > 0.06: fl.append(f"percentile_mismatch(expected {want:.2f}, got {got}; route {route})")
            else:
                if want is not None:
                    nums = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", ans)]
                    if not any(abs(n - want) <= max(0.06, abs(want) * 0.004, 0.55 if re.search(r'\b\d+ km/h', ans) else 0.0) for n in nums): fl.append(f"average_mismatch(expected {want:.2f}; route {route})")
        except Exception as e:
            fl.append("EXCEPTION " + type(e).__name__ + ": " + str(e)[:80])
        res.append({"family": fam, "asked": q, "answer": ans[:300], "flags": fl, "status": "PASS" if not fl else "FAIL"})
    p_ = sum(r["status"] == "PASS" for r in res); print(f"{p_}/{len(res)} pass in {time.time() - t0:.0f}s")
    cnt = collections.Counter((r["family"], r["status"]) for r in res)
    for f_ in sorted({r["family"] for r in res}): print(f"  {f_:11s} {cnt[(f_, 'PASS')]}/{cnt[(f_, 'PASS')] + cnt[(f_, 'FAIL')]}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

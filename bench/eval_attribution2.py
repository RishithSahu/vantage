"""Attribution against the two-cause counterfactual families (bench/counterfactual2.py): an UNRECORDED step cause (roof box) and a RECORDED gradual driver (tyre pressure, whose true effect on consumption is tiny in this generator).
Per seed the window is the last N days (N = days after the roof-box onset) against the N days before.
  roof_true / tyre_true   gap(all) - gap(no_roof) / gap(all) - gap(no_tyre): the observed effect of each cause on this window (engine's own gap measure, so definitions agree)
  engine: the tyre factor's contribution, the residual, the explained total
  tyre false attribution = engine tyre contribution - tyre_true (a recorded but nearly irrelevant driver must not take the credit for the roof box)
  residual error        = residual - (roof_true + tyre_true if tyre sensors are removed else roof_true)
Missing-driver variant: the same 'all' history with the tyre pressure sensors removed; the explanation must not invent another cause and must list the unmeasured conditions.
    python tools_pin.py 4,5 bench/eval_attribution2.py"""
import glob, json, os, shutil, sqlite3, statistics as st, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.eval_attribution import explain


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "cf2_s*_all.db"))):
        base = f[:-7]; tr = json.load(open(f[:-3] + "_truth.json")); roof = tr["meta"]["stories"]["roof_box"]["start"]; n = tr["meta"]["days"] - roof
        if n < 4: continue
        gaps = {}
        for v in ("all", "no_roof", "no_tyre", "none"):
            r, cb = explain(f"{base}_{v}.db", n); gaps[v] = cb
        if not all(gaps.values()): rows.append((os.path.basename(base), "no decomposition")); continue
        a = gaps["all"]; roof_true = a["gap"] - gaps["no_roof"]["gap"]; tyre_true = a["gap"] - gaps["no_tyre"]["gap"]
        tyre = next((x["contribution"] for x in a["factors"] if "tyre" in x["name"]), 0.0)
        tmp = tempfile.mkdtemp(); g = os.path.join(tmp, "x.db"); shutil.copy(f, g); c = sqlite3.connect(g)
        for col in ("tp_fl", "tp_fr", "tp_rl", "tp_rr"): c.execute(f"UPDATE ext_1m SET {col} = NULL")
        c.commit(); c.close(); r2, m = explain(g, n); shutil.rmtree(tmp, ignore_errors=True)
        rows.append((os.path.basename(base), roof, n, a["gap"], roof_true, tyre_true, tyre, a["residual"], m["residual"] if m else None, [x["name"] for x in (m or {}).get("factors", [])], "tyre" in (r2.answer.text.lower() if r2 else "")))
    print(f"{'family':12s} {'roofOn':>6s} {'N':>3s} {'gap':>6s} {'roof_true':>9s} {'tyre_true':>9s} | {'tyre_factor':>11s} {'residual':>8s} | missing-driver: residual  other factors")
    ok = [r for r in rows if len(r) > 3]
    for r in rows:
        if len(r) <= 3: print(r); continue
        print(f"{r[0]:12s} {r[1]:6d} {r[2]:3d} {r[3]:6.1f} {r[4]:9.1f} {r[5]:9.1f} | {r[6]:11.1f} {r[7]:8.1f} | {r[8] if r[8] is None else round(r[8], 1)}  {[x[:22] for x in r[9]]}")
    if not ok: return
    fa = [abs(r[6] - r[5]) for r in ok]; print(f"\nfamilies {len(ok)}. Recorded-but-irrelevant driver (tyre): |engine tyre contribution - true tyre effect| median {st.median(fa):.1f} Wh/km, max {max(fa):.1f}; true roof effect median {st.median(r[4] for r in ok):.1f}")
    cov = [r[7] / r[4] for r in ok if r[4] > 3]; print(f"unrecorded roof effect left in the unexplained part: median {st.median(cov):.2f} (min {min(cov):.2f}, max {max(cov):.2f})")
    md = [r[8] / (r[4] + r[5]) for r in ok if r[8] is not None and (r[4] + r[5]) > 3]; print(f"missing tyre driver: share of the total true effect left unexplained: median {st.median(md):.2f}")


if __name__ == "__main__": main()

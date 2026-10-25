"""Scores 'why did it change' explanations against counterfactual twins (see bench/counterfactual.py).

For each pair (same routes and weather; the roof box exists only in `with`), the window straddles the roof-box onset: the last N days, N = days after the onset, against the N days before (truncated at the start of the history).
  gap_with / gap_without   the change in consumption per trip, as the engine measures it on each history
  box                      gap_with - gap_without: the true size of the injected cause over this window
  F_with / R_with          what the engine assigns to the recorded conditions / leaves unexplained on the `with` history (gap_with = F_with + R_with)
Good attribution: F_with ~ gap_without (the recorded conditions explain what they explain without the box) and R_with ~ box (the unrecorded cause stays in the residual).
  leak     = F_with - gap_without   how much of the box effect the recorded conditions absorbed (positive = false attribution to recorded factors)
  residual coverage = R_with / box  how much of the true box effect the engine left in the unexplained part
Negative control: on the `without` history nothing unrecorded changed, so R_without should be small relative to its gap.
    python tools_pin.py 4,5 bench/eval_attribution.py"""
import glob, json, os, re, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.qc import QueryCompiler
from vgpt.query_ir import Query


def explain(db, n_days, target="wh_per_km"):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True); r = QueryCompiler(conn).execute(Query.from_text(f"op=explain_change tgt={target} t=last:{n_days}:day")); conn.close()
    cb = (r.result.extra or {}).get("contrib") if r.result is not None else None
    return r, cb


def score():
    """-> rows: (pair, onset, N, gap_with, gap_without, box, F_with, R_with, leak, R_without, gap_without, r2_without, r2_with) or (pair, onset, reason)."""
    rows = []
    for w in sorted(glob.glob(os.path.join(HERE, "data", "cf_*_with.db"))):
        wo = w.replace("_with.db", "_without.db"); tr = json.load(open(w[:-3] + "_truth.json")); start = tr["meta"]["stories"]["roof_box"]["start"]; n = tr["meta"]["days"] - start
        if n < 4 or start < 4: rows.append((os.path.basename(w), start, "skipped: the window is too short to compare")); continue
        r1, a = explain(w, n); r0, b = explain(wo, n)
        if not a or not b: rows.append((os.path.basename(w), start, f"no decomposition: with={r1.status} without={r0.status}")); continue
        gap_w, gap_0 = a["gap"], b["gap"]; box = gap_w - gap_0; F = gap_w - a["residual"]; leak = F - gap_0
        rows.append((os.path.basename(w).replace("_with.db", ""), start, n, gap_w, gap_0, box, F, a["residual"], leak, b["residual"], gap_0, b.get("r2"), a.get("r2")))
    return rows


def main():
    rows = score()
    print(f"{'pair':24s} {'onset':>5s} {'N':>3s} {'gap(with)':>9s} {'gap(w/o)':>9s} {'box':>7s} {'F_with':>7s} {'R_with':>7s} {'leak':>7s} {'R_w/o':>7s}")
    ok = [r for r in rows if len(r) > 3]
    for r in rows:
        if len(r) <= 3: print(f"{r[0]:24s} {r[1]:5d}  {r[2]}"); continue
        print(f"{r[0]:24s} {r[1]:5d} {r[2]:3d} {r[3]:9.1f} {r[4]:9.1f} {r[5]:7.1f} {r[6]:7.1f} {r[7]:7.1f} {r[8]:7.1f} {r[9]:7.1f}")
    if not ok: return
    import statistics as st
    def stratum(rows, name):
        if not rows: print(f"  {name}: no pairs"); return
        cover = [r[7] / r[5] for r in rows]; leak = [r[8] / r[5] for r in rows]
        print(f"  {name}: n={len(rows)}  residual coverage of the true effect: median {st.median(cover):.2f} (min {min(cover):.2f}, max {max(cover):.2f})  leak into recorded factors: median {st.median(leak):.2f} (min {min(leak):.2f}, max {max(leak):.2f}); more than half leaked in {sum(1 for x in leak if x > 0.5)}")
    print(f"ALL {len(ok)} pairs are listed above; none is dropped. Strata by the true size of the injected effect on this window:")
    stratum([r for r in ok if abs(r[5]) < 5.0], "effect < 5 Wh/km (small: ratios are unstable)"); stratum([r for r in ok if 5.0 <= abs(r[5]) < 15.0], "effect 5-15 Wh/km"); stratum([r for r in ok if abs(r[5]) >= 15.0], "effect >= 15 Wh/km"); stratum(ok, "all pairs (effect-weighted ratios are unstable for small effects)")
    absr = [abs(r[7] - r[5]) for r in ok]; print(f"  absolute error of the unexplained part vs the true effect (Wh/km): median {st.median(absr):.1f}, max {max(absr):.1f}")
    neg = [abs(r[9]) for r in ok]; print(f"negative control (history without the box): |unexplained| median {st.median(neg):.1f} Wh/km, max {max(neg):.1f}; |gap| median {st.median(abs(r[10]) for r in ok):.1f}")

if __name__ == "__main__": main()

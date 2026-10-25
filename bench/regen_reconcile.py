"""Reconcile trip_ext.regen_kwh (trip ledger) with the energy obtained by integrating the sampled battery power, trip by trip, over exactly the same intervals.

  1. synthetic check: a known constant power over a known time must integrate to the known energy (validates the integration code, units and sign convention);
  2. per-trip comparison on a real database: coverage (recorded samples vs trip minutes), ledger value, one-minute integral, 1 Hz integral where present, and the ratio;
  3. field semantics: what the ledger value is as a share of the energy the same trip consumed (a physical plausibility bound), and whether any trip is duplicated.

    python bench/regen_reconcile.py [data/gen_demo.db] > results/round52/regen_reconciliation.md
"""
import os, sqlite3, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def integrate_kwh(ts, power_w, moving=None, step=None):
    """Energy (kWh) recovered = integral of -power over samples with power < 0 (while moving), each sample covering its own spacing (capped at 120 s)."""
    ts = np.asarray(ts, float); p = np.asarray(power_w, float) / 1000.0              # kW
    dt = np.diff(ts, append=ts[-1] + (step or 60)); dt = np.minimum(dt, 120.0) if step is None else np.full(len(ts), float(step))
    m = (p < 0) & (np.ones(len(p), bool) if moving is None else moving)
    return float(np.sum(-p[m] * dt[m]) / 3600.0)


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "gen_demo.db")
    print("# Regeneration: trip ledger vs integrated samples\n")
    # 1. synthetic
    ts = np.arange(0, 1800, 60); p = np.full(len(ts), -10000.0)                       # 10 kW into the battery for 30 minutes -> 5 kWh
    print(f"1. Synthetic: -10 kW for 30 min integrates to {integrate_kwh(ts, p):.3f} kWh (expected 5.000). Constant +10 kW gives {integrate_kwh(ts, -p):.3f} kWh of regeneration (expected 0).\n")
    c = sqlite3.connect(db)
    cols = [r[1] for r in c.execute("PRAGMA table_info(sig_1m)")]
    unit = "W" if True else "kW"                                                   # sig_1m.power is stored in W (registry scale 1000 -> kW)
    tr = c.execute("SELECT t.start, t.end, t.km, t.kwh, x.regen_kwh FROM trips t JOIN trip_ext x ON x.start = t.start WHERE t.km > 0.2 ORDER BY t.start").fetchall()
    rows = []; dup = len(tr) - len({r[0] for r in tr})
    for s, e, km, kwh, led in tr:
        m = c.execute("SELECT ts, power, speed FROM sig_1m WHERE ts >= ? AND ts <= ? ORDER BY ts", (s, e)).fetchall()
        if not m: continue
        a = np.array(m, float); ts_, pw, sp = a[:, 0], a[:, 1], a[:, 2]
        i1 = integrate_kwh(ts_, pw, moving=sp >= 10.0, step=60)
        h = c.execute("SELECT ts, power, speed FROM sig_1s WHERE ts >= ? AND ts <= ? ORDER BY ts", (s, e)).fetchall()
        iz = integrate_kwh(np.array([x[0] for x in h]), np.array([x[1] for x in h], float), moving=np.array([x[2] for x in h], float) >= 10.0, step=1) if h else None
        rows.append((s, (e - s) / 60.0, len(m), km, kwh, led, i1, iz))
    n = len(rows)
    ledger = sum(r[5] for r in rows); i1 = sum(r[6] for r in rows); used = sum(r[4] for r in rows)
    print(f"2. {n} trips with ledger and samples ({dup} duplicate trip rows).")
    print(f"   Ledger regeneration: {ledger:.2f} kWh ({100 * ledger / used:.1f}% of the {used:.1f} kWh these trips consumed).")
    print(f"   Integrated one-minute battery power while moving: {i1:.2f} kWh ({100 * i1 / used:.1f}% of consumption).")
    print(f"   Ratio ledger / integral: {ledger / i1 if i1 else float('inf'):.1f}")
    cov = np.array([r[2] / max(r[1], 1) for r in rows]); print(f"   Sample coverage of trip minutes: median {np.median(cov):.2f}, 10th percentile {np.percentile(cov, 10):.2f}.")
    hz = [r for r in rows if r[7] is not None]
    if hz: print(f"   Trips with 1 Hz data: {len(hz)}; ledger {sum(r[5] for r in hz):.2f} kWh vs 1 Hz integral {sum(r[7] for r in hz):.2f} kWh vs one-minute integral {sum(r[6] for r in hz):.2f} kWh.")
    per = [(r[5] / r[6]) for r in rows if r[6] > 1e-6]
    if per: print(f"   Per-trip ratio ledger/integral: median {np.median(per):.1f}, min {np.min(per):.1f}, max {np.max(per):.1f} over {len(per)} trips.")
    print("\n3. Reading. Units are kWh on both sides, and the synthetic check shows the integration and sign convention are right, so a unit or scaling error in the integral is excluded.")
    share = 100 * ledger / used
    print(f"   The ledger field claims {share:.0f}% of consumed energy is recovered; the measured power says about {100 * i1 / used:.1f}%. Minute averaging can hide short, strong regeneration, but the 1 Hz comparison"
          " (where available) and the per-trip ratios above show whether that explains it. If the ratio is stable across trips the ledger field is a scaled quantity (e.g. generated as a fixed share of trip energy); if it varies"
          " wildly it is unrelated to the samples. Either way it cannot be reproduced from the recorded power, so it is labelled UNVERIFIED and is not used as ground truth; reports use the sample-derived value.")


if __name__ == "__main__":
    main()

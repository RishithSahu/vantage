"""CausalBench: does the assistant name the real cause?

Histories are generated with a known cause planted in them (the generator knows the truth; nothing in the assistant reads it).  For every scenario the assistant is asked the symptom question a person would ask
and its ranked diagnosis is compared with the planted cause.  Scenarios carry the situations that make causes hard to tell apart: other cities and seasons (cold weather lowers range with no fault), drivers that
differ, a distractor (road works) and two causes at once.  Negative controls have no cause at all: the right answer is "nothing to explain".

    python bench/causal_bench.py gen  --set dev  [--n 6]         build the dev scenarios under data/causal/dev (seeds 1-)
    python bench/causal_bench.py gen  --set test [--n 10]        the sealed test scenarios (seeds 5000-); built once, used once per release
    python bench/causal_bench.py run  --set dev [--show]         diagnose each scenario and score it
Metrics: top-1 and top-3 cause accuracy per planted cause, onset error in days, false-cause rate on the negative controls, how often the stated confidence was right (calibration)."""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)

# planted story -> the cause id the diagnosis must name
CAUSE_OF = {"cooling_pump": "cooling_pump_degradation", "tyre_leak": "tyre_slow_leak", "weak_12v": "weak_12v_battery", "dirty_windscreen": "camera_obstruction", "roof_box": "added_drag", "harsh_week": "aggressive_driving",
            "alignment_drift": "rolling_resistance_increase"}
# scenario kind -> (stories, city choices, driver choices, season) ; season: 'winter' | 'summer'
KINDS = {
    "cooling_pump": (["cooling_pump"], ["london", "graz", "sydney", "singapore"], ["commuter", "rideshare", "long_distance"], "summer"),
    "tyre_leak": (["tyre_leak"], ["london", "oslo", "bengaluru", "chicago"], ["commuter", "delivery", "retiree"], "summer"),
    "weak_12v": (["weak_12v"], ["london", "graz", "phoenix", "oslo"], ["commuter", "rideshare"], "winter"),
    "dirty_windscreen": (["dirty_windscreen"], ["london", "sydney", "bengaluru"], ["commuter", "delivery"], "summer"),
    "roof_box": (["roof_box"], ["london", "graz", "sydney", "chicago"], ["commuter", "long_distance"], "summer"),
    "harsh_week": (["harsh_week"], ["london", "bengaluru", "chicago"], ["commuter", "rideshare", "delivery"], "summer"),
    "alignment_drift": (["alignment_drift"], ["london", "graz", "sydney"], ["commuter", "long_distance"], "summer"),
    "cold_season": ([], ["oslo", "graz", "chicago"], ["commuter", "retiree", "delivery"], "autumn"),
    "none_summer": ([], ["london", "sydney", "bengaluru", "singapore"], ["commuter", "retiree", "rideshare"], "summer"),
    "roof_box+cold": (["roof_box"], ["oslo", "graz", "chicago"], ["commuter", "retiree"], "autumn"),
    "tyre_leak+harsh": (["tyre_leak", "harsh_week"], ["london", "chicago", "sydney"], ["commuter", "delivery"], "summer"),
    "pump+roadworks": (["cooling_pump", "roadworks"], ["london", "graz"], ["commuter", "rideshare"], "summer"),
}
VEHICLES = ["city", "compact", "family", "suv"]
DAYS = 70


def ends(season): return {"summer": "2026-08-10", "winter": "2026-01-25", "autumn": "2025-12-20"}[season]


def realised_effects(gv, stories):
    """How large was each planted cause in the data that was actually generated?  Computed from the generator's own daily truth, never given to the assistant.  Used only to say which scenarios a cause could be seen in at all
    (a harsh-driving week that the simulated driver barely changed is not a fair test of anything)."""
    import numpy as np
    d = gv.truth["daily"]; km = np.array([x["km"] for x in d], float); kwh = np.array([x["kwh"] for x in d], float); hr = np.array([(x["harsh_brake"] + x["harsh_accel"] + x["harsh_corner"]) for x in d], float)
    wh = np.where(km > 5, kwh / np.maximum(km, 1) * 1000, np.nan); rate = np.where(km > 5, hr / np.maximum(km, 1) * 100, np.nan); out = {}; st = gv.st
    if "harsh_week" in stories:
        a = st["harsh_week"]["start"]; w = slice(a, a + 6); m = np.ones(len(km), bool); m[w] = False; out["harsh_week"] = round(float(np.nanmean(rate[w]) / max(np.nanmean(rate[m]), 1e-6)), 2)
        e_w = float(np.nansum(hr[w])); expd = float(np.nansum(hr[m]) / max(np.nansum(km[m]), 1) * np.nansum(km[w])); out["harsh_week_z"] = round((e_w - expd) / max(expd, 1.0) ** 0.5, 1)
    if "roof_box" in stories:
        a = st["roof_box"]["start"]; out["roof_box"] = round(float(np.nanmean(wh[a:]) / np.nanmean(wh[max(0, a - 28):a]) - 1), 3) if a >= 6 else 0.0
    if "alignment_drift" in stories:
        a = st["alignment_drift"]["start"]; out["alignment_drift"] = round(float(np.nanmean(wh[-7:]) / np.nanmean(wh[max(0, a - 28):a]) - 1), 3)
    return out


def cold_effect(db):
    """From the stored history (bench side only): how much colder did it get, and how much more did the car use per kilometre?"""
    import sqlite3
    c = sqlite3.connect("file:" + db.replace("\\", "/") + "?mode=ro", uri=True); t = c.execute("SELECT MIN(start), MAX(start) FROM trips").fetchone(); span = t[1] - t[0]
    a = c.execute("SELECT AVG(avg_amb), SUM(kwh)/SUM(km)*1000 FROM trips WHERE start < ? AND km > 0.3", (t[0] + span * 0.4,)).fetchone(); b = c.execute("SELECT AVG(avg_amb), SUM(kwh)/SUM(km)*1000 FROM trips WHERE start > ? AND km > 0.3", (t[1] - span * 0.25,)).fetchone(); c.close()
    return {"cold_drop": round(float(a[0] - b[0]), 1), "cold_wh_rise": round(float(b[1] / a[1] - 1), 3)}


def detectable(sc):
    e = sc.get("effect", {})
    if "harsh_week" in e and e.get("harsh_week_z", 0) < 4.0: return False
    if "roof_box" in e and e["roof_box"] < 0.05: return False
    if "alignment_drift" in e and e["alignment_drift"] < 0.05: return False
    if "cold_drop" in e and (e["cold_drop"] < 4.0 or e["cold_wh_rise"] < 0.03): return False
    return True


def gen(set_name, n):
    from vgpt import datagen_events as DE, datagen_life as DL
    base = 1 if set_name == "dev" else 5000; out_dir = os.path.join(ROOT, "data", "causal", set_name); os.makedirs(out_dir, exist_ok=True); meta = []
    for kind, (stories, cities, drivers, season) in KINDS.items():
        for k in range(n):
            seed = base + len(meta) * 7 + k; city = cities[k % len(cities)]; driver = drivers[(k // 2) % len(drivers)]; veh = VEHICLES[(k + len(kind)) % len(VEHICLES)]
            sid = f"{kind}_{k}"; db = os.path.join(out_dir, sid + ".db"); t = time.time()
            gv = DL.build(days=DAYS, offline=True, verbose=False, seed=seed, city=city, driver=driver, vehicle=veh, stories=list(stories), end=ends(season))
            for ext in ("", "-wal", "-shm"):
                if os.path.exists(db + ext): os.remove(db + ext)
            DE.seed(db, gv)
            planted = {s: gv.st[s] for s in stories}; causes = [CAUSE_OF[s] for s in stories if s in CAUSE_OF]; effect = realised_effects(gv, stories)
            if kind in ("cold_season", "roof_box+cold"):
                causes = causes + ["cold_weather"] if kind == "roof_box+cold" else ["cold_weather"]; effect.update(cold_effect(db))
            row = {"id": sid, "kind": kind, "db": os.path.relpath(db, ROOT), "seed": seed, "city": city, "driver": driver, "vehicle": veh, "season": season, "days": DAYS, "stories": planted, "causes": causes, "end": ends(season), "effect": effect}
            meta.append(row); print(f"{sid:22s} {city:10s} {driver:13s} {veh:8s} stories={list(stories)} -> {causes}  ({time.time() - t:.0f}s)", flush=True)
    json.dump(meta, open(os.path.join(out_dir, "scenarios.json"), "w"), indent=1)


def run(set_name, show=False, only=None):
    from vgpt.cause.engine import diagnose_db
    meta = json.load(open(os.path.join(ROOT, "data", "causal", set_name, "scenarios.json"))); rows = []
    for sc in meta:
        if only and not sc["kind"].startswith(only): continue
        t = time.time(); res = diagnose_db(os.path.join(ROOT, sc["db"]), symptom=None); dt = time.time() - t
        ranked = [h["id"] for h in res["hypotheses"]]; truth = sc["causes"]; top = ranked[0] if ranked else None
        rows.append({"id": sc["id"], "kind": sc["kind"], "detectable": detectable(sc), "truth": truth, "ranked": ranked[:5], "top": top, "conf": res["hypotheses"][0]["confidence"] if ranked else 0.0, "nothing": res.get("nothing_to_explain", False), "s": round(dt, 2), "res": res})
        if show: print(f"{sc['id']:20s} truth={truth} -> {ranked[:3]} conf={rows[-1]['conf']:.2f} nothing={rows[-1]['nothing']} ({dt:.1f}s)", flush=True)
    return score(rows)


def score(rows):
    hidden = [r for r in rows if r["truth"] and not r["detectable"]]
    if hidden: print("\nnot scored (the planted effect is too small to be seen in the data that was generated): " + ", ".join(r["id"] for r in hidden))
    rows = [r for r in rows if r["detectable"] or not r["truth"]]
    pos = [r for r in rows if r["truth"]]; neg = [r for r in rows if not r["truth"]]
    top1 = [bool(r["top"] in r["truth"]) for r in pos]; top3 = [bool(set(r["ranked"][:3]) & set(r["truth"])) for r in pos]
    allc = [all(t in r["ranked"][:len(r["truth"]) + 1] for t in r["truth"]) for r in pos]
    by = {}
    for r, a, b in zip(pos, top1, top3): by.setdefault(r["kind"], []).append((a, b))
    print("\nper scenario kind (top-1 / top-3):")
    for k, v in by.items(): print("  %-18s %2d/%2d  %2d/%2d" % (k, sum(a for a, _ in v), len(v), sum(b for _, b in v), len(v)))
    fc = [not r["nothing"] and r["conf"] >= 0.5 for r in neg]
    print("\npositives: %d  top-1 %.1f%%  top-3 %.1f%%  all planted causes in top-(n+1) %.1f%%" % (len(pos), 100 * sum(top1) / max(1, len(pos)), 100 * sum(top3) / max(1, len(pos)), 100 * sum(allc) / max(1, len(pos))))
    print("negative controls: %d  confident false cause %.1f%%  said 'nothing to explain' %.1f%%" % (len(neg), 100 * sum(fc) / max(1, len(neg)), 100 * sum(r["nothing"] for r in neg) / max(1, len(neg))))
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]; print("calibration (stated confidence -> how often the top cause was right):")
    for lo, hi in bins:
        sel = [(a) for r, a in zip(pos, top1) if lo <= r["conf"] < hi]
        if sel: print("  %.2f-%.2f: %d cases, right %.0f%%" % (lo, min(hi, 1.0), len(sel), 100 * sum(sel) / len(sel)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "run"]); ap.add_argument("--set", default="dev"); ap.add_argument("--n", type=int, default=6); ap.add_argument("--show", action="store_true"); ap.add_argument("--only")
    a = ap.parse_args()
    if a.cmd == "gen": gen(a.set, a.n)
    else:
        rows = run(a.set, a.show, a.only)
        os.makedirs(os.path.join(ROOT, "results", "causal"), exist_ok=True); json.dump([{k: v for k, v in r.items() if k != "res"} for r in rows], open(os.path.join(ROOT, "results", "causal", f"{a.set}.json"), "w"), indent=1)


if __name__ == "__main__": main()

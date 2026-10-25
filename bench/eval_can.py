"""Score the CAN bus-health monitor on REAL recorded vehicle CAN traffic (HCRL Car-Hacking dataset), streamed straight from the zip.
Baseline = clean 'normal_run_data'; test = DoS / Fuzzy / RPM-spoof / gear-spoof recordings with per-message attack labels (flag T).
Window-level scoring (1 s windows): a window is 'attack' if it contains >=1 injected (T) message; 'flagged' if the monitor raised any alert.
Usage: python bench/eval_can.py [path-to-zip]   (the data stays in Downloads; nothing is copied or redistributed)"""
import io, json, os, re, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt.canhealth import CanHealth

ZIP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Downloads", "9) Car-Hacking Dataset.zip")
LINE = re.compile(r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)")
z = zipfile.ZipFile(ZIP)

mon = CanHealth(spike_factor=2.0, spike_floor=3)
t = time.time(); n = 0
with z.open("normal_run_data/normal_run_data.txt") as f:
    for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
        m = LINE.search(raw)
        if m: mon.learn(float(m[1]), m[2].lower()); n += 1
mon.finish_learning()
res = {"baseline_msgs": n, "baseline_seconds": mon.base_n, "known_ids": len(mon.base_max), "baseline_s": round(time.time() - t, 1)}
print(f"baseline: {n:,} real frames, {mon.base_n} s, {len(mon.base_max)} IDs learned in {res['baseline_s']} s")

out = {}
for name in ("DoS_dataset.csv", "Fuzzy_dataset.csv", "RPM_dataset.csv", "gear_dataset.csv"):
    m2 = CanHealth(spike_factor=2.0, spike_floor=3); m2.base_max = dict(mon.base_max); m2.learning = False
    attack_sec, flagged_sec, lat = set(), set(), []; msgs = 0; first_attack = None; first_flag = None; t0 = time.time()
    with z.open(name) as f:
        for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
            p = raw.strip().split(",")
            if len(p) < 4: continue
            try: ts = float(p[0])
            except ValueError: continue
            msgs += 1; cid = p[1].lower(); is_atk = p[-1] == "T"
            if is_atk:
                attack_sec.add(int(ts))
                if first_attack is None: first_attack = ts
            for (sec, kind, i, c) in m2.feed(ts, cid):
                flagged_sec.add(sec)
                if first_flag is None and first_attack is not None and sec >= int(first_attack): first_flag = sec
    for (sec, kind, i, c) in m2.flush(): flagged_sec.add(sec)
    tp = len(attack_sec & flagged_sec); fp = len(flagged_sec - attack_sec); fn = len(attack_sec - flagged_sec)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn); f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out[name] = {"msgs": msgs, "attack_windows": len(attack_sec), "flagged_windows": len(flagged_sec), "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                 "detect_delay_s": (first_flag - int(first_attack)) if first_flag is not None and first_attack is not None else None, "msgs_per_s": round(msgs / (time.time() - t0))}
    print(f"{name:20s} {msgs:>9,} frames | attack windows {len(attack_sec):4d} flagged {len(flagged_sec):4d} | precision {prec:.3f} recall {rec:.3f} F1 {f1:.3f} | first-detect delay {out[name]['detect_delay_s']} s | {out[name]['msgs_per_s']:,} frames/s (laptop, 1 core)")
res["attacks"] = out
json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "can_real_eval.json"), "w"), indent=1)

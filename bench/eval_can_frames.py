"""Frame-level CAN bus-health monitor (vgpt/canhealth.py) on two more REAL datasets, read in place from Downloads:
  Autohack2025  per-message labels (Normal vs attack classes): learn on Normal rows of the Train file, score 1-second windows on the Test file
  OTIDS         attack-free recording split in time (learn 60% / evaluate false alarms on the unseen 40%), then DoS: the injected frames are ID 0000 (known from the
                dataset paper), so DoS gets true labels; Fuzzy / Impersonation have NO per-message labels, so only the fraction of flagged windows is reported.
Usage: python bench/eval_can_frames.py"""
import csv, io, json, os, re, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt.canhealth import CanHealth

DL = os.path.join(os.path.expanduser("~"), "Downloads"); OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LINE = re.compile(r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)")


def window_scores(msgs, mon, label_of=None):
    attack, flagged = set(), set(); n = 0; kinds = {}
    for ts, cid, atk in msgs:
        n += 1
        if atk: attack.add(int(ts))
        for (sec, kind, i, c) in mon.feed(ts, cid): flagged.add(sec); kinds[kind] = kinds.get(kind, 0) + 1
    for (sec, kind, i, c) in mon.flush(): flagged.add(sec)
    tp = len(attack & flagged); fp = len(flagged - attack); fn = len(attack - flagged)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"messages": n, "attack_windows": len(attack), "flagged_windows": len(flagged), "tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3), "alerts_by_kind": kinds}


def autohack():
    z = zipfile.ZipFile(os.path.join(DL, "Autohack2025_Dataset.zip")); mon = CanHealth(); n = 0; t0 = time.time()
    r = csv.DictReader(io.TextIOWrapper(z.open("Autohack2025_Dataset/Train/autohack_train_both.csv"), encoding="utf-8", errors="replace"))
    for row in r:
        if row["Label"] == "Normal": mon.learn(float(row["Timestamp"]), row["Arbitration_ID"]); n += 1
    mon.finish_learning(); print(f"Autohack2025: baseline from {n:,} clean frames, {len(mon.base_max)} ids ({time.time() - t0:.0f}s)")
    labels = {}; msgs = []
    for row in csv.DictReader(io.TextIOWrapper(z.open("Autohack2025_Dataset/Test/autohack_test_both.csv"), encoding="utf-8", errors="replace")):
        labels[row["Label"]] = labels.get(row["Label"], 0) + 1; msgs.append((float(row["Timestamp"]), row["Arbitration_ID"], row["Label"] != "Normal"))
    res = window_scores(msgs, mon); res["label_counts"] = labels
    print(f"  test: {res['messages']:,} frames, labels {labels} | windows attack {res['attack_windows']} flagged {res['flagged_windows']} | P {res['precision']} R {res['recall']} F1 {res['f1']} | {res['alerts_by_kind']}")
    return res


def otids():
    z = zipfile.ZipFile(os.path.join(DL, "10) CAN-Intrusion Dataset.zip")); frames = []
    for raw in io.TextIOWrapper(z.open("Attack_free_dataset.txt"), encoding="utf-8", errors="replace"):
        m = LINE.search(raw)
        if m: frames.append((float(m[1]), m[2].lower()))
    cut = int(len(frames) * 0.6); mon = CanHealth()
    for ts, cid in frames[:cut]: mon.learn(ts, cid)
    mon.finish_learning(); print(f"OTIDS: baseline from first 60% of attack-free ({cut:,} frames, {len(mon.base_max)} ids); evaluating the unseen 40% + attack files")
    out = {"attack_free_holdout": window_scores(((ts, cid, False) for ts, cid in frames[cut:]), CanHealth.__new__(CanHealth).__class__() if False else _clone(mon))}
    print("  attack-free hold-out:", {k: out["attack_free_holdout"][k] for k in ("messages", "flagged_windows", "alerts_by_kind")})
    for name, atk_id in (("DoS_attack_dataset.txt", "0000"), ("Fuzzy_attack_dataset.txt", None), ("Impersonation_attack_dataset.txt", None)):
        msgs = []
        for raw in io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace"):
            m = LINE.search(raw)
            if m: msgs.append((float(m[1]), m[2].lower(), atk_id is not None and m[2].lower() == atk_id))
        res = window_scores(msgs, _clone(mon)); secs = len({int(t) for t, _, _ in msgs}); res["windows_total"] = secs; out[name] = res
        lab = "labelled (ID 0000)" if atk_id else "UNLABELLED: share of windows flagged only"
        print(f"  {name:36s} {res['messages']:>9,} frames {secs:>5} windows | flagged {res['flagged_windows']:>5} ({100 * res['flagged_windows'] // max(1, secs)}%) | {lab} | P {res['precision']} R {res['recall']} | {res['alerts_by_kind']}")
    return out


def _clone(mon):
    c = CanHealth(); c.base_max = dict(mon.base_max); c.learning = False; return c


if __name__ == "__main__":
    out = {"autohack2025": autohack(), "otids": otids()}
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "can_frame_security.json"), "w"), indent=1)

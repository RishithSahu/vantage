"""Signal-level CAN intrusion detection on REAL / labelled datasets read in place from Downloads (never copied):
  SynCAN  (ETAS/CANet) - train on clean traffic, test on 6 labelled attack types (continuous, flooding, plateau, playback, suppress, normal)
  ROAD    (ORNL, real vehicle on a dynamometer) - train on ambient signal extractions, test on the labelled attack captures
Metrics are per MESSAGE (alert raised on a message vs its Label). Honest reporting: playback/correlated attacks are expected to be hard for a range/step/period monitor.
Usage: python bench/eval_signal.py [syncan|road|all]"""
import csv, io, json, os, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt.signalhealth import SignalHealth

DL = os.path.join(os.path.expanduser("~"), "Downloads")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def rows(f, time_col, id_col, label_col="Label", scale=1.0, limit=None):
    r = csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")); hdr = next(r)
    ti, ii, li = hdr.index(time_col), hdr.index(id_col), hdr.index(label_col) if label_col in hdr else None
    sig_idx = [k for k, h in enumerate(hdr) if h.startswith("Signal")]
    for n, row in enumerate(r):
        if limit and n >= limit: break
        try: t = float(row[ti]) * scale
        except ValueError: continue
        vals = []
        for k in sig_idx:
            x = row[k] if k < len(row) else ""
            vals.append(float(x) if x not in ("", "nan", "NaN") else None)
        while vals and vals[-1] is None: vals.pop()                      # unused trailing signal columns
        yield t, row[ii], vals, (int(float(row[li])) if li is not None and row[li] != "" else 0)


def score(h, it, name):
    tp = fp = fn = tn = 0; by_kind = {}; t0 = time.time(); n = 0
    for t, cid, vals, lab in it:
        n += 1; al = h.check(t, cid, vals); flag = bool(al)
        if flag and lab: tp += 1
        elif flag: fp += 1
        elif lab: fn += 1
        else: tn += 1
        for k, _ in al[:1]: by_kind[k] = by_kind.get(k, 0) + 1
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); f = 2 * p * r / max(1e-9, p + r)
    return {"name": name, "messages": n, "attack_msgs": tp + fn, "tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "false_alarm_rate": round(fp / max(1, fp + tn), 5), "alerts_by_kind": by_kind, "msgs_per_s": round(n / max(1e-9, time.time() - t0))}


def syncan():
    z = zipfile.ZipFile(os.path.join(DL, "SynCAN-master.zip")); h = SignalHealth(); n = 0; t0 = time.time()
    tz = zipfile.ZipFile(io.BytesIO(z.read("SynCAN-master/train_1.zip"))); name = [i.filename for i in tz.infolist() if i.filename.endswith(".csv")][0]
    for t, cid, vals, lab in rows(tz.open(name), "Time", "ID", scale=0.001): h.learn(t, cid, vals); n += 1
    h.finish(); print(f"SynCAN: learned {len(h.ids)} ids from {n:,} clean messages in {time.time() - t0:.0f}s")
    res = []
    for kind in ("normal", "continuous", "flooding", "plateau", "playback", "suppress"):
        tz = zipfile.ZipFile(io.BytesIO(z.read(f"SynCAN-master/test_{kind}.zip"))); nm = [i.filename for i in tz.infolist() if i.filename.endswith(".csv")][0]
        h.reset_stream(); r = score(h, rows(tz.open(nm), "Time", "ID", scale=0.001), "syncan_" + kind); res.append(r)
        print(f"  {kind:11s} msgs {r['messages']:>8,} attack {r['attack_msgs']:>7,} | precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f} | false-alarm rate {r['false_alarm_rate']:.4f} | {r['alerts_by_kind']}")
    return res


def road(max_attacks=40):
    z = zipfile.ZipFile(os.path.join(DL, "road.zip")); h = SignalHealth(); n = 0; t0 = time.time()
    names = sorted(i.filename for i in z.infolist() if "signal_extractions/ambient/" in i.filename and i.filename.endswith(".csv") and "__MACOSX" not in i.filename
                   and i.filename.split("/")[-1].startswith("ambient_"))                   # only the genuinely clean captures (the folder also holds attack files)
    for nm in names:
        for t, cid, vals, lab in rows(z.open(nm), "Time", "ID", label_col="Label", limit=2_000_000): h.learn(t, cid, vals); n += 1
    train = f"{len(names)} clean captures"
    h.finish(); print(f"ROAD: learned {len(h.ids)} ids from {n:,} real clean messages ({train}) in {time.time() - t0:.0f}s")
    atk = sorted((i for i in z.infolist() if "signal_extractions/attacks/" in i.filename and i.filename.endswith(".csv") and "__MACOSX" not in i.filename and i.file_size < 70e6), key=lambda i: i.filename)
    res = []
    for info in atk[:max_attacks]:
        h.reset_stream(); r = score(h, rows(z.open(info.filename), "Time", "ID"), info.filename.split("/")[-1][:-4]); res.append(r)
        print(f"  {r['name'][:46]:46s} msgs {r['messages']:>8,} attack {r['attack_msgs']:>6,} | P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} | FAR {r['false_alarm_rate']:.4f} | {r['alerts_by_kind']}")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"; out = {}
    if which in ("syncan", "all"): out["syncan"] = syncan()
    if which in ("road", "all"): out["road"] = road()
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, "signal_security.json")
    try: old = json.load(open(path))
    except (OSError, ValueError): old = {}
    old.update(out); json.dump(old, open(path, "w"), indent=1)                                  # merge: running one dataset must not erase the other

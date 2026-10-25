"""Anomaly recall against the ground truth of a generated world.

    python bench/anomaly_truth_eval.py [--db data/world_aster.db] [--per-kind 8] [--out results/anomaly_truth.json]

The world builder writes every injected fault with its time (<db>_truth.json, key `faults`).  For each fault (up to --per-kind of each kind) the assistant is asked, as of three hours after the fault, "is anything
unusual" (frame `anomaly`, the last 7 days) and "are there any warnings" (`events any_warning`, list); the answer counts as finding the fault when it names the day of the fault or a word that names its kind.
Reported per kind: how often the first question found it, how often either did.  No language model: the frames are given."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ["VGPT_FRAME"] = "2"
import ask_frame as AF                                  # noqa: E402
from vgpt.meaning.pipeline import MeaningBrain          # noqa: E402
from vgpt.vehicle import lake as LK                     # noqa: E402

WORDS = {"ecu_resets_adas": ["reset", "adas", "controller", "watchdog"], "voltage_dip": ["voltage", "12 v", "dip", "supply"], "unexpected_shutdown": ["shutdown", "shut down", "power", "supply"], "can_burst": ["bus", "can ", "dropout", "silent", "communication"],
         "speed_conflict": ["speed", "sensor", "conflict", "source"], "ambient_glitch": ["outside", "ambient", "temperature"], "aeb": ["emergency", "aeb", "brak"], "range_glitch": ["range"], "adas_flapping": ["adas", "driver assist", "lane"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default="data/world_aster.db"); ap.add_argument("--per-kind", type=int, default=8); ap.add_argument("--out", default="results/anomaly_truth.json"); a = ap.parse_args()
    path = a.db if os.path.isabs(a.db) else os.path.join(ROOT, a.db); truth = json.load(open(path[:-3] + "_truth.json", encoding="utf-8")); faults = truth["faults"]
    faults = dict(faults) if isinstance(faults, dict) else {k: v for k, v in faults}
    LK.adopt_path(path); b = MeaningBrain(path, AF.Null(), now=AF.last_ts(path), tau=0.0); rows = []
    for kind, items in faults.items():
        if kind not in WORDS: continue
        for it in items[:a.per_kind]:
            ts = int(it.get("ts") or it.get("start") or 0)
            if not ts: continue
            b.now = ts + 3 * 3600; day = time.strftime("%d %b", time.gmtime(ts + int(b.tz))).lstrip("0"); res = {}
            for q, spec in (("is anything unusual", "anomaly none"), ("are there any warnings", "events any_warning op=list")):
                b.qcs.last = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []; b._last_data = None
                try: ans = str(b.resolver.answer(q, AF.frame_of(spec)).get("answer", ""))
                except Exception as e: ans = "EXC " + type(e).__name__
                low = ans.lower(); res[q] = bool(any(w in low for w in WORDS[kind]) or day.lower() in low or day.lower().replace(" 0", " ") in low)
            rows.append({"kind": kind, "ts": ts, "day": day, **{("unusual" if i == 0 else "warnings"): v for i, v in enumerate(res.values())}})
    by = {}
    for r in rows: by.setdefault(r["kind"], []).append(r)
    print(f"{'fault kind':22s} {'n':>3s} {'unusual':>9s} {'warnings':>9s} {'either':>7s}")
    for k, v in by.items(): print(f"{k:22s} {len(v):3d} {sum(r['unusual'] for r in v):6d}/{len(v):<2d} {sum(r['warnings'] for r in v):6d}/{len(v):<2d} {sum(r['unusual'] or r['warnings'] for r in v):4d}/{len(v):<2d}")
    n = len(rows); print(f"\nall: unusual found {sum(r['unusual'] for r in rows)}/{n}; either {sum(r['unusual'] or r['warnings'] for r in rows)}/{n}")
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

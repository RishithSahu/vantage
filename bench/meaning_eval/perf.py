"""Latency and memory of the meaning pipeline on the dev machine, restricted to two logical cores (the closest local stand-in for the PDF's ~2 A5x cores).

    python tools_pin.py 8,9 bench/meaning_eval/perf.py --adapter models/meaning-lora-full.gguf --n 80

Reports: model load time, resident set size after load / peak during questions, per-question frame-parse time and total answer time (p50 / p95 / max), tokens decoded per frame.
x86 numbers: an ARM A55 is slower; this does not measure the PDF target hardware and must not be quoted as if it did."""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import evalio                                                     # noqa: E402

DB = os.path.join(ROOT, "data", "gen_demo.db")
NOW = 1787625900


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--n", type=int, default=80); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    r0 = rss_mb(); t = time.perf_counter()
    brain = make_meaning_brain(DB, now=NOW, threads=a.threads, adapter=os.path.join(ROOT, a.adapter) if not os.path.isabs(a.adapter) else a.adapter)
    load_s = time.perf_counter() - t; r1 = rss_mb()
    rows = evalio.load_tsv(os.path.join(HERE, "novel_v1.tsv"))[:a.n]; parse, total, peak, toks = [], [], r1, []
    brain.ask("warm up question about the car speed")                                         # first call pays the prompt prefill; steady-state is what a driver experiences
    for r in rows:
        brain.qcs.last = None; t = time.perf_counter(); res = brain.ask(r["text"]); total.append((time.perf_counter() - t) * 1000)
        info = res.get("slm") or {}; parse.append(info.get("ms") or 0.0); toks.append(info.get("tokens") or 0); peak = max(peak, rss_mb())
    print(f"threads={a.threads} load={load_s:.1f}s rss_before={r0:.0f}MB rss_after_load={r1:.0f}MB rss_peak={peak:.0f}MB")
    print(f"frame parse ms  p50={pct(parse, 50):.0f} p95={pct(parse, 95):.0f} max={max(parse):.0f} mean_tokens={statistics.mean(toks):.1f}")
    print(f"total answer ms p50={pct(total, 50):.0f} p95={pct(total, 95):.0f} max={max(total):.0f}  (n={len(total)})")


if __name__ == "__main__":
    main()

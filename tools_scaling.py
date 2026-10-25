"""Scaling measurements: what happens to ingest, memory, query latency and concurrency when the data, the traffic or the number of people grows - on THIS machine.

  python tools_scaling.py ingest [--seconds 600]                  messages/s of the live runtime (with / without the bus monitor), and N independent runtimes on N processes
  python tools_scaling.py history --days 14 35 70 105             query latency against history length (model replaced by its recorded answers, so only the data side is timed), DB size, build time
  python tools_scaling.py sessions --workers 1 2 3 --sessions 1 2 4 8 [--cores 2] [--db data/world_aster.db]
                                                                  concurrent assistants: answers per second and latency percentiles, real model, with the model contexts and session counts of the tiers
  python tools_scaling.py all --out results/scaling/latest.json   all of the above

Everything is measured on the host that runs it; ARM boards are NOT measured here (no QEMU / board run has been approved).  docs/ARCHITECTURE.md quotes these numbers with that caveat."""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)

QUESTIONS = [
    "what is my speed right now", "what is the tyre pressure", "how much charge is left", "how many hard brakes last week", "what was the average energy consumption over the last 30 days",
    "how many fast charges did I do in the last 10 weeks", "what was my longest trip", "which trip used the most energy in the last two months", "what was the highest battery temperature in the last 10 weeks",
    "how many trips did I make in the last 10 weeks", "compare energy consumption between city and highway driving", "when did energy efficiency show the largest drop", "show all ADAS disengagement events from last week",
    "what happened before the battery overheating warning", "why has my driving range reduced over the last two weeks", "how many times did the tyre pressure warning appear in the last 10 weeks",
    "compare my energy use this week with last week", "how many diagnostic codes were set this month", "what was my average speed yesterday", "how long was I charging last week",
    "Investigate the most recent unexpected vehicle shutdown. What changed immediately before it, which ECUs reported errors, what were the last valid sensor readings, and what evidence is missing?",
    "Did any recent ADAS disengagements coincide with communication errors, sensor faults, or ECU resets? Correlate the timestamps across available logs and rank the strongest supported relationships.",
    "Identify periods when two sources reported conflicting values or states for the same vehicle parameter. Show the disagreement, duration, source timestamps, and any available diagnostic information that helps explain it.",
    "Review the last month of vehicle history and identify the most important unanswered diagnostic question. State the observed behavior, the evidence already available, the competing explanations, and the single additional measurement or log that would most help resolve the uncertainty.",
]
INVESTIGATIONS = 4


def rss_mb():
    import psutil
    return round(psutil.Process().memory_info().rss / 2 ** 20)


def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


# ------------------------------------------------------------------------------------------------ ingest
def _capture(seconds, profile="aster_ev"):
    from vgpt.live.car import VirtualCar
    from vgpt.live.sim import Simulation
    from vgpt.vehicle.profile import Profile
    p = Profile.load(profile); car = VirtualCar(p, seed=3, t0=1792364400, tz_h=1.0, home=(51.51257, -0.04426), work=(51.70886, -0.2739)); out = []
    sim = Simulation(None, car, sink=lambda ts, bus, pl: out.append((ts, bus, pl))); car.start_trip()
    for _ in range(seconds): sim.step()
    return p, out


def _replay_proc(args):
    profile, seconds, monitor, ready, go = args
    from vgpt.live.runtime import Runtime
    from vgpt.vehicle import provision as PV
    p, traffic = _capture(seconds, profile); db = os.path.join(tempfile.gettempdir(), f"sc_ing_{os.getpid()}.db"); PV.provision(profile, db, replace=True); rt = Runtime(db, profile=p, monitor=monitor)
    span = traffic[-1][0] - traffic[0][0] + 1
    if ready is not None: ready.release(); go.wait()                                                       # every process starts its timed loop together, or the totals would not overlap
    n = 0; t0 = time.perf_counter(); cpu0 = time.process_time(); loops = 0
    while time.perf_counter() - t0 < 10.0:
        for ts, bus, pl in traffic: rt.feed(ts + loops * span, bus, pl)
        n += len(traffic); loops += 1
    wall = time.perf_counter() - t0; cpu = time.process_time() - cpu0; st = rt.stats(); rt.close(); sim_s = loops * span
    return {"msgs": n, "wall_s": wall, "cpu_s": cpu, "sim_s": sim_s, "rss_mb": rss_mb(), "decode_us": st["decode_us"], "rows": st["rows_1hz"], "lake_rows": st["lake_rows"]}


def bench_ingest(seconds=300, procs=(1, 2, 4, 8)):
    """One process, with and without the bus monitor; then k processes (one vehicle each) started together."""
    res = {}
    for monitor in (True, False):
        r = _replay_proc(("aster_ev", seconds, monitor, None, None)); key = "monitor_on" if monitor else "monitor_off"
        res[key] = {"msgs_per_s": round(r["msgs"] / r["wall_s"]), "realtime_factor": round(r["sim_s"] / r["wall_s"]), "cpu_s_per_sim_hour": round(r["cpu_s"] / r["sim_s"] * 3600, 1), "rss_mb": r["rss_mb"], "decode_us": r["decode_us"]}
        res["traffic_msgs_per_sim_s"] = round(r["msgs"] / r["sim_s"], 1)
    ctx = mp.get_context("spawn"); res["processes"] = {}; base = None
    for k in procs:
        m = ctx.Manager(); ready = m.Semaphore(0); go = m.Event()
        with ctx.Pool(k) as pool:
            fut = pool.map_async(_replay_proc, [("aster_ev", seconds, False, ready, go)] * k)
            for _ in range(k): ready.acquire()
            go.set(); rs = fut.get()
        agg = sum(r["msgs"] / r["wall_s"] for r in rs); base = base or agg
        res["processes"][str(k)] = {"aggregate_msgs_per_s": round(agg), "vehicles_realtime": round(sum(r["sim_s"] / r["wall_s"] for r in rs)), "speedup_vs_1": round(agg / base, 2), "max_rss_mb": max(r["rss_mb"] for r in rs)}
        print(f"  {k} process(es): {round(agg)} msgs/s ({round(agg / base, 2)}x)", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ history
class ReplayModel:
    """The frame model's recorded answers: the data side of a question can then be timed on its own."""
    def __init__(self, table, glossary): self.t, self.glossary = table, glossary

    def parse(self, text, hint=None, prev=None):
        from vgpt.meaning.frame import Frame
        hit = self.t.get(text)
        if hit is None: return None, {"raw": "", "pmin": 0.0, "ms": 0.0}
        return Frame.from_text(hit["raw"]), dict(hit, ms=0.0)


def capture_frames(db, threads=2):
    from vgpt.meaning.interface import FrameModel
    from vgpt.vehicle import lake as LK
    from vgpt import product
    LK.adopt_path(db); adapter = os.path.join(ROOT, "models", "meaning-lora-full_v11.gguf"); m = FrameModel(adapter, threads=threads); table = {}
    for q in QUESTIONS[:-INVESTIGATIONS]:
        f, info = m.parse(q)
        if f is not None: table[q] = {"raw": info["raw"], "pmin": info.get("pmin")}
    return table, m.glossary


def bench_history(days_list, capture_db=None):
    from vgpt.meaning.pipeline import MeaningBrain
    from vgpt.vehicle import lake as LK
    from vgpt.world import build as WB
    out = []; table = glossary = None
    for d in days_list:
        db = os.path.join(tempfile.gettempdir(), f"sc_hist_{d}.db"); t0 = time.time(); gv = WB.build_world("aster_ev", days=d, seed=5, end="2026-10-18", verbose=False); WB.write_world(gv, db, verbose=False); build_s = time.time() - t0; LK.release()
        if table is None: table, glossary = capture_frames(capture_db or db)
        LK.adopt_path(db); b = MeaningBrain(db, ReplayModel(table, glossary)); b.now = int(gv.t0 + gv.n) - 3600; lat = {}
        for rep in range(3):
            for q in QUESTIONS:
                if q not in table and QUESTIONS.index(q) < len(QUESTIONS) - INVESTIGATIONS: continue
                b.dialogue.reset(); t = time.perf_counter(); r = b.ask(q); dt = time.perf_counter() - t; lat.setdefault(q, []).append((dt, (r.get("facts") or {}).get("route") if isinstance(r.get("facts"), dict) else None))
        warm = [min(v[0] for v in vs[1:]) * 1000 for vs in lat.values()]; inv = [statistics.median(v[0] for v in vs[1:]) * 1000 for q, vs in lat.items() if QUESTIONS.index(q) >= len(QUESTIONS) - INVESTIGATIONS]; std = [statistics.median(v[0] for v in vs[1:]) * 1000 for q, vs in lat.items() if QUESTIONS.index(q) < len(QUESTIONS) - INVESTIGATIONS]
        c = sqlite3.connect(db); rows = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("sig_1s", "sig_1m", "lk_1s", "lk_1m", "events", "trips")}; c.close(); b.c.close(); LK.release()
        out.append({"days": d, "db_mb": round(os.path.getsize(db) / 2 ** 20, 1), "rows": rows, "build_s": round(build_s, 1), "answered": len(lat), "standard_p50_ms": round(statistics.median(std), 1), "standard_p95_ms": round(pct(std, 0.95), 1), "standard_max_ms": round(max(std), 1),
                    "investigation_p50_ms": round(statistics.median(inv), 1) if inv else None, "investigation_max_ms": round(max(inv), 1) if inv else None, "rss_mb": rss_mb(), "slowest": sorted(((round(statistics.median(v[0] for v in vs[1:]) * 1000), q[:60]) for q, vs in lat.items()), reverse=True)[:3]})
        print(f"  {d:4d} days: {out[-1]['db_mb']} MB, standard p50 {out[-1]['standard_p50_ms']} ms (p95 {out[-1]['standard_p95_ms']}, max {out[-1]['standard_max_ms']}), investigations p50 {out[-1]['investigation_p50_ms']} ms", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ sessions
def bench_sessions(db, workers_list, sessions_list, cores=0, threads=2, replay=False, questions=6):
    from vgpt import product, sessions as S
    from vgpt.meaning.interface import FrameModel
    from vgpt.vehicle import lake as LK
    if cores:
        import psutil
        psutil.Process().cpu_affinity(list(range(cores)))
    LK.adopt_path(db); adapter = os.path.join(ROOT, "models", "meaning-lora-full_v11.gguf"); qs = QUESTIONS[:-INVESTIGATIONS][:questions]; out = []
    table = glossary = None
    if replay: table, glossary = capture_frames(db, threads)
    for W in workers_list:
        factory = (lambda: ReplayModel(table, glossary)) if replay else (lambda: FrameModel(adapter, threads=threads)); t0 = time.perf_counter(); pool = S.ModelPool(factory, W); load_s = time.perf_counter() - t0; rss0 = rss_mb()
        for N in sessions_list:
            hub = S.SessionHub(db, pool, lambda d, slm: product.make_meaning_brain(d, slm=slm), max_sessions=max(N, 1)); lats = []; lock = threading.Lock(); now = int(time.time())
            for i in range(N): hub.ask(f"s{i}", "what is the tyre pressure")                                  # sessions are created and warm before the clock starts
            def person(i):
                for q in qs:
                    t = time.perf_counter(); hub.ask(f"s{i}", q)
                    with lock: lats.append(time.perf_counter() - t)
            ths = [threading.Thread(target=person, args=(i,)) for i in range(N)]; t0 = time.perf_counter()
            for t in ths: t.start()
            for t in ths: t.join()
            wall = time.perf_counter() - t0; rss1 = rss_mb()
            out.append({"model_workers": W, "sessions": N, "answers": len(lats), "wall_s": round(wall, 2), "answers_per_s": round(len(lats) / wall, 2), "p50_ms": round(1000 * statistics.median(lats)), "p95_ms": round(1000 * pct(lats, 0.95)), "max_ms": round(1000 * max(lats)),
                        "model_wait_p50_ms": hub.stats()["model"]["wait_p50_ms"], "rss_mb": rss1, "rss_after_pool_mb": rss0, "model_load_s": round(load_s, 1), "cores": cores or os.cpu_count(), "replay": replay})
            print(f"  workers={W} sessions={N}: {out[-1]['answers_per_s']} answers/s, p50 {out[-1]['p50_ms']} ms, p95 {out[-1]['p95_ms']} ms, RSS {rss1} MB", flush=True); hub.close()
        pool.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--seconds", type=int, default=300)
    p = sub.add_parser("history"); p.add_argument("--days", type=int, nargs="+", default=[14, 35, 70, 105]); p.add_argument("--capture-db")
    p = sub.add_parser("sessions"); p.add_argument("--db", default=os.path.join(ROOT, "data", "world_aster.db")); p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3]); p.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 4, 8]); p.add_argument("--cores", type=int, default=0)
    p.add_argument("--threads", type=int, default=2); p.add_argument("--replay", action="store_true")
    p = sub.add_parser("all"); p.add_argument("--out", default=os.path.join(ROOT, "results", "scaling", "latest.json"))
    a = ap.parse_args(); res = {"host": {"cpus": os.cpu_count(), "python": sys.version.split()[0], "when": time.strftime("%Y-%m-%d %H:%M")}}
    if a.cmd in ("ingest", "all"): print("ingest"); res["ingest"] = bench_ingest(getattr(a, "seconds", 300))
    if a.cmd in ("history", "all"): print("history"); res["history"] = bench_history(getattr(a, "days", [14, 35, 70, 105]), getattr(a, "capture_db", None))
    if a.cmd == "sessions": print("sessions"); res["sessions"] = bench_sessions(a.db, a.workers, a.sessions, a.cores, a.threads, a.replay)
    if a.cmd == "all":
        db = os.path.join(ROOT, "data", "world_aster.db"); print("sessions (all cores)"); res["sessions"] = bench_sessions(db, [1, 2, 3], [1, 2, 4, 8])
    out = getattr(a, "out", None) or os.path.join(ROOT, "results", "scaling", time.strftime("%Y%m%d_%H%M%S") + f"_{a.cmd}.json"); os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1); print("wrote", out)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()

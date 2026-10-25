"""Scaling curves of the real assistant on the machine it runs on (the same files on the 2-core board, the Pi, a laptop and a vehicle-computer-class machine).

    python bench/scale_curve.py threads  [--cores 1,2,4,6,8,12]      question latency of ONE model context against the cores it is given (process affinity, physical cores first)
    python bench/scale_curve.py contexts [--cores 12] [--clients 1,2,4,8,16]   throughput and latency of the elastic service against the number of people asking, with the autoscaler on and off
    python bench/scale_curve.py history  [--db data/world_3yr.db]    data side of the answers against how much history is asked about (7 days to 3 years)
    python bench/scale_curve.py out results/scaling/X.json           (any mode: --out)

Everything runs the real model and the real answer pipeline; nothing is simulated."""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("VGPT_NO_QA_CLF", "1")

QUESTIONS = ["how fast am I going", "what is the average battery temperature over the last trip", "highest speed last week", "is my tyre pressure fine", "how many trips did I take this week", "what was my average speed yesterday",
             "how much energy did I use last week", "when was my last hard braking", "how long did I drive this week", "what is the motor temperature", "how many warnings this month", "what was my longest trip",
             "how much charge is left", "what is my odometer", "how many times did I charge this month", "what was the coldest temperature last week"]


SIGNALS = ["speed", "battery temperature", "motor temperature", "tyre pressure", "energy consumption", "state of charge", "motor power", "cabin temperature", "ambient temperature", "regen power", "12 V battery voltage", "coolant flow"]
TEMPLATES = ["what was the average {s} over the last {n} days", "what was the highest {s} in the past {n} days", "what was the lowest {s} during the last {n} days", "how did {s} change over the last {n} days",
             "how often was {s} abnormal in the last {n} days", "give me the {s} trend for the past {n} days"]


def unique_question(k):
    """A question nobody has asked this run (so no repeated-wording shortcut can help): template x signal x number of days."""
    t = TEMPLATES[k % len(TEMPLATES)]; s = SIGNALS[(k // len(TEMPLATES)) % len(SIGNALS)]; n = 2 + (k // (len(TEMPLATES) * len(SIGNALS))) % 28 + (k // 2000); return t.format(s=s, n=n)


def set_affinity(cpus):
    if os.name == "nt":
        import ctypes
        mask = sum(1 << c for c in cpus); ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), mask)
    elif hasattr(os, "sched_setaffinity"): os.sched_setaffinity(0, set(cpus))


def cpu_list(n, logical=None):
    """The first n logical CPUs, one per physical core first (SMT siblings are numbered 2k, 2k+1 on this kind of machine)."""
    logical = logical or (os.cpu_count() or n)
    if logical <= 4: return list(range(n))                                                                      # a four-core board: no SMT siblings
    evens = list(range(0, logical, 2)); odds = list(range(1, logical, 2)); return (evens + odds)[:n]


def rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"): return int(line.split()[1]) // 1024
    except OSError: pass
    from vgpt.scale.service import _rss_mb
    return _rss_mb()


def child_threads(a):
    cpus = [int(x) for x in a.cpus.split(",")]; set_affinity(cpus)
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.meaning.ort_backend import OrtFrameModel
    from vgpt.vehicle import lake
    lake.adopt_path(os.path.join(ROOT, a.db))
    m = OrtFrameModel(os.environ["VGPT_ORT_DIR"], threads=a.threads, version=2); m.parse("how fast am I going"); m.parse("how fast am I going"); ts = []; pf = []; dc = []
    for q in QUESTIONS * a.reps:
        t = time.perf_counter(); m.parse(q, prev="now speed none none none none none none"); ts.append(time.perf_counter() - t); pf.append(m.stats["prefill_ms"] / max(1, m.stats["tail_tokens"])); dc.append(m.stats["decode_ms"] / max(1, m.stats["passes"]))
    print(json.dumps({"threads": a.threads, "cpus": cpus, "parse_median_s": round(st.median(ts), 3), "parse_p95_s": round(sorted(ts)[int(len(ts) * .95)], 3), "ms_per_prompt_token": round(st.median(pf), 1), "ms_per_decode_pass": round(st.median(dc), 1), "rss_mb": rss_mb(), "provider": m.provider}))


def run_threads(a):
    out = []
    for n in [int(x) for x in a.cores.split(",")]:
        cpus = cpu_list(n)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "_child_threads", "--threads", str(n), "--cpus", ",".join(map(str, cpus)), "--db", a.db, "--reps", str(a.reps)], capture_output=True, text=True, cwd=ROOT, env=dict(os.environ))
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        if not line: print("failed for", n, r.stderr[-400:]); continue
        row = json.loads(line[-1]); out.append(row); print("%2d core(s): parse median %.2fs p95 %.2fs | %.1f ms per prompt token | %.0f ms per decode pass | %d MB" % (n, row["parse_median_s"], row["parse_p95_s"], row["ms_per_prompt_token"], row["ms_per_decode_pass"], row["rss_mb"]), flush=True)
    return out


def run_contexts(a):
    import vehiclegpt; vehiclegpt._adopt_product_model()
    cpus = cpu_list(a.total_cores); set_affinity(cpus)
    import threading
    from vgpt.scale import Hardware
    from vgpt.scale.service import AssistantService
    from vgpt.scale.plan import make_plan
    db = os.path.join(ROOT, a.db); res = []
    for auto in (False, True):
        for clients in [int(x) for x in a.clients.split(",")]:
            import dataclasses; from vgpt.scale.probe import probe; hw = dataclasses.replace(probe(), cores=a.total_cores, mem_mb=a.mem_mb)
            plan = make_plan(hw, vehicles=1, knee=a.knee or None)
            if a.force_max: plan.max_contexts = a.force_max; plan.threads = min(plan.threads, 2)
            mx = plan.max_contexts if auto else 1
            svc = AssistantService(db, hw=hw, plan=plan, autoscale=auto, start_contexts=1); svc.pool.max_workers = mx if auto else 1
            threads_ctx = plan.threads
            for i in range(3): svc.ask("warm", unique_question(i))
            lat = []; stop = time.monotonic() + a.seconds; lock = threading.Lock(); counter = [1000 * (1 + len(res))]

            def client(k):
                while time.monotonic() < stop:
                    with lock: counter[0] += 1; i = counter[0]
                    t = time.perf_counter(); svc.ask(f"u{k}", unique_question(i), persona="driver"); d = time.perf_counter() - t
                    with lock: lat.append((time.monotonic(), d))
            ths = [threading.Thread(target=client, args=(k,)) for k in range(clients)]; t0 = time.monotonic(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.monotonic() - t0
            second = sorted(d for t, d in lat if t - t0 > a.seconds * 0.4) or sorted(d for t, d in lat)
            row = {"autoscale": auto, "clients": clients, "contexts_end": svc.pool.contexts(), "plan_max_contexts": plan.max_contexts, "threads_per_context": threads_ctx, "answers": len(lat), "answers_per_s": round(len(lat) / wall, 2), "p50_s": round(second[len(second) // 2], 2), "p95_s": round(second[int(len(second) * .95)], 2), "rss_mb": rss_mb()}
            row["decisions"] = [list(x) for x in (svc.ctl.log if svc.ctl is not None else [])][:40]; res.append(row); print("auto=%-5s clients=%2d: %.2f answers/s, p50 %.2fs p95 %.2fs, contexts %d (max %d), %d MB" % (auto, clients, row["answers_per_s"], row["p50_s"], row["p95_s"], row["contexts_end"], plan.max_contexts, row["rss_mb"]), flush=True)
            svc.close()
    return res


def run_procs(a):
    """Throughput and latency of the process pool (the real model in each worker) against workers and people asking."""
    import threading
    import vehiclegpt; vehiclegpt._adopt_product_model(); set_affinity(cpu_list(a.total_cores))
    from vgpt.scale.procpool import ProcessPool
    from vgpt.scale.autoscale import Autoscaler
    from vgpt.scale import Hardware, make_plan
    env = {k: os.environ[k] for k in ("VGPT_FRAME", "VGPT_MEANING_ADAPTER", "VGPT_ORT_DIR") if k in os.environ}; env.update({"HF_HUB_OFFLINE": "1", "VGPT_NO_QA_CLF": "1", "VGPT_ORT_SPIN": os.environ.get("VGPT_ORT_SPIN", "0")})
    cfg = {"db": os.path.join(ROOT, a.db), "backend": "ort-int8", "ort_dir": env["VGPT_ORT_DIR"], "threads": a.threads, "adapter": env.get("VGPT_MEANING_ADAPTER"), "root": ROOT, "env": env, "start_timeout": 600}
    res = []; counter = [0]; lock = threading.Lock()
    for workers in [int(x) for x in a.workers.split(",")]:
        for clients in [int(x) for x in a.clients.split(",")]:
            pool = ProcessPool(cfg, workers=workers, min_workers=workers, max_workers=workers)
            for i in range(workers): pool.ask(f"warm{i}", unique_question(i), now=1792348500)
            lat = []; stop = time.monotonic() + a.seconds

            def client(k):
                while time.monotonic() < stop:
                    with lock: counter[0] += 1; i = counter[0]
                    t = time.perf_counter(); pool.ask(f"u{k}", unique_question(i), persona="driver", now=1792348500); d = time.perf_counter() - t
                    with lock: lat.append((time.monotonic(), d))
            ths = [threading.Thread(target=client, args=(k,)) for k in range(clients)]; t0 = time.monotonic(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.monotonic() - t0
            late = sorted(d for t, d in lat if t - t0 > a.seconds * 0.4) or sorted(d for t, d in lat)
            row = {"workers": workers, "clients": clients, "threads": a.threads, "answers": len(lat), "answers_per_s": round(len(lat) / wall, 2), "p50_s": round(late[len(late) // 2], 2), "p95_s": round(late[int(len(late) * .95)], 2)}
            res.append(row); print("workers=%d clients=%2d: %.2f answers/s, p50 %.2fs p95 %.2fs" % (workers, clients, row["answers_per_s"], row["p50_s"], row["p95_s"]), flush=True); pool.close()
    return res


def run_history(a):
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.product import make_qwen05_brain
    b = make_qwen05_brain(os.path.join(ROOT, a.db)); import sqlite3
    c = sqlite3.connect("file:" + os.path.join(ROOT, a.db).replace("\\", "/") + "?mode=ro", uri=True); hi = c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]; lo = c.execute("SELECT MIN(ts) FROM sig_1m").fetchone()[0]; c.close()
    b.now = int(hi); b.ask("how fast am I going"); fm = b.frame_model; res = []
    orig = fm.parse; T = {}
    def parse(*x, **k): t = time.perf_counter(); r = orig(*x, **k); T["parse"] = time.perf_counter() - t; return r
    fm.parse = parse
    qs = ["what was my highest speed in the last {n} days", "what was the average energy consumption over the last {n} days", "how many hard brakes in the last {n} days", "how many trips in the last {n} days", "why did my efficiency change over the last {n} days"]
    for days in [7, 30, 90, 180, 365, 730, 1095]:
        if (hi - lo) / 86400 < days - 1: continue
        rest = []
        for q in qs:
            b.dialogue.reset(); t = time.perf_counter(); b.ask(q.format(n=days)); tot = time.perf_counter() - t; rest.append(tot - T.get("parse", 0))
        row = {"days": days, "data_side_median_s": round(st.median(rest), 3), "data_side_max_s": round(max(rest), 3)}; res.append(row); print("%5d days asked about: data side median %.2fs max %.2fs" % (days, row["data_side_median_s"], row["data_side_max_s"]), flush=True)
    return {"db": a.db, "days_stored": round((hi - lo) / 86400), "db_mb": round(os.path.getsize(os.path.join(ROOT, a.db)) / 1e6), "rows": res}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--cores", default="1,2,4,6,8,12"); ap.add_argument("--clients", default="1,2,4,8,16"); ap.add_argument("--total-cores", type=int, default=12); ap.add_argument("--mem-mb", type=int, default=16384)
    ap.add_argument("--db", default="data/world_aster.db"); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--seconds", type=float, default=30.0); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--cpus", default="0,1"); ap.add_argument("--out"); ap.add_argument("--knee", type=int, default=0); ap.add_argument("--force-max", type=int, default=0, help="let the autoscaler try this many contexts even where the plan allows fewer (to watch it discover that a machine does not scale out)"); ap.add_argument("--workers", default="1,2,4,6")
    a = ap.parse_args()
    if a.mode == "_child_threads": return child_threads(a)
    res = {"threads": run_threads, "contexts": run_contexts, "procs": run_procs, "history": run_history}[a.mode](a)
    if a.out: os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True); json.dump({"mode": a.mode, "host": os.environ.get("COMPUTERNAME") or os.uname().nodename, "result": res, "time": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(ROOT, a.out), "w"), indent=1)


if __name__ == "__main__": main()

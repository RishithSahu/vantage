"""VehicleGPT interactive assistant (minimal UI).
  python app.py                         chat in the terminal (driver persona), SLM = Qwen2.5-0.5B
  python app.py --persona engineer      evidence + timeline included in answers
  python app.py --constrain             run INSIDE the simulated 2 GB / 2-core target (Windows Job Object + 700 MB 'OS' ballast + live ingest)
Everything is local: no network access is used. The database is opened READ-ONLY (the ingest daemon is the only writer)."""
import argparse, os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="0.5b", choices=["0.5b"]); ap.add_argument("--mmap", type=int, default=1, choices=[1]); ap.add_argument("--persona", default="driver", choices=["driver", "engineer"])
ap.add_argument("--constrain", action="store_true"); ap.add_argument("--slowdown", type=float, default=1.0); ap.add_argument("--os-mb", type=int, default=700)
ap.add_argument("--db", default=os.path.join(HERE, "data", "vehicle.db")); ap.add_argument("--ask", default="")
a = ap.parse_args()

gov = ballast = None
if a.constrain:
    from vgpt import governor as gov
    print("limits:", gov.apply(mem_mb=2048, cores=(0, 2), slowdown=a.slowdown)); ballast = gov.start_os_ballast(a.os_mb)
    print(f"vehicle OS ballast committed: {a.os_mb} MB (of 2048 MB shared)")

from vgpt import bootstrap
ok, report = bootstrap.startup(HERE, a.model, None, doctor_checks=False)                  # the same hardening and the same product model as `vehiclegpt.py run` and `ui/server.py`
from vgpt.product import make_qwen05_brain
brain = make_qwen05_brain(a.db, persona=a.persona, threads=2)


def show(r):
    print(f"\n{r['answer']}\n  [{r['intent']} via {r['path']} | parse {r['ms']['parse']} ms, tools {r['ms']['tools']} ms, total {r['ms']['total']} ms]")
    if gov: print(f"  [job memory peak {gov.peak_job_mb():.0f} MB of 2048 MB]")


if a.ask: show(brain.ask(a.ask)); sys.exit()
print("\nAsk about your car (blank line to quit). Try: 'Why has my driving range reduced over the last two weeks?'")
while True:
    try: q = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt): break
    if not q: break
    show(brain.ask(q))

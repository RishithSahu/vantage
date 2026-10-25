"""End to end, with the real model: a car model drives and sends its buses (CAN / SOME/IP / Android VHAL / UDS / logs / GNSS / camera) through the local pipe, Vantage learns what it is receiving, and the assistant answers questions about that car.

    python bench/linked_e2e.py CAR_ID [CAR_ID ...] [--drive 90] [--speed 4] [--out results/linked_e2e.json]

For every car: the time until Vantage has recognised it, the number of signals learned, the answer and the time of each question (the model reads it, the data is read from what the pipe delivered).
Run it where the claim is to be made (the Raspberry Pi; an emulated ARM guest)."""
import argparse
import json
import os
import secrets
import shutil
import statistics as st
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VGPT_NO_QA_CLF", "1")

QUESTIONS = ["how fast am I going", "what is my battery level", "how is my tyre pressure", "what was my highest speed in this drive", "is anything wrong with the car",
             "how far have I driven", "what is the motor temperature", "how much energy did I use per kilometre"]


def wait(cond, seconds, step=0.5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond(): return True
        time.sleep(step)
    return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cars", nargs="+"); ap.add_argument("--drive", type=float, default=90.0, help="seconds of the car's own clock to drive before asking"); ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--out", default="results/linked_e2e.json"); a = ap.parse_args()
    os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp(prefix="vgpt_e2e_home_")
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from ui import server as S
    from vgpt.car import models, transport as CT, writer as CW
    from vgpt.car.car import Car
    from vgpt.pipe import transport as PT
    home = tempfile.mkdtemp(prefix="vgpt_e2e_link_"); endpoint = PT.endpoint("e" + secrets.token_hex(4))
    ns = SimpleNamespace(db="data/vehicle.db", model="0.5b", camera="clip", webcam_index=0, live=None, linked=True, link_home=home, link_endpoint=endpoint, planner="fast", threads=2)
    t = time.monotonic(); app = S.App(ns); print("assistant ready in %.1fs (%s)" % (time.monotonic() - t, type(getattr(app.brain, "slm", None)).__name__), flush=True)
    app.ask("how fast am I going", "driver"); results = []
    for cid in a.cars:
        m = models.find(cid)
        if not m: print("no such car", cid); continue
        before = app.db; t0 = time.monotonic(); fd = CT.connect(endpoint, timeout=30); car = Car(m["dir"], CW.Writer(fd), speed=a.speed, seed=7); car.start()
        got = wait(lambda: app.db != before and "waiting" not in str(app.dataset), 180); t_adopt = time.monotonic() - t0
        wait(lambda: car.drive.t - car.drive.t0 >= a.drive or not car.is_alive(), 600)
        st_ = app.linked.runtime() if app.linked else None; link = app.pipe and getattr(app, "linked", None)
        info = {"car": cid, "buses": sorted({s.type for s in car.profile.sources.values()}), "recognised": bool(got), "how": (app.linked.view() or {}).get("decision") if app.linked else None, "vehicle": (app.linked.view() or {}).get("model") if app.linked else None, "seconds_to_recognise": round(t_adopt, 1), "frames_sent": car.sent, "drive_s": round(car.drive.t - car.drive.t0)}
        try:
            import sqlite3
            c = sqlite3.connect("file:" + app.db.replace("\\", "/") + "?mode=ro", uri=True)
            info["signals"] = c.execute("SELECT COUNT(*) FROM sig_catalog").fetchone()[0]; info["rows_1s"] = c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0]; c.close()
        except Exception as e: info["db_error"] = str(e)[:80]
        qa = []
        for q in QUESTIONS:
            t = time.monotonic(); r = app.ask(q, "driver"); qa.append({"q": q, "s": round(time.monotonic() - t, 2), "a": str(r.get("answer"))[:140]})
        info["answers"] = qa; ts = [x["s"] for x in qa]; info["median_s"] = round(st.median(ts), 2); info["max_s"] = max(ts); results.append(info)
        print("%-22s buses=%s recognised=%s (%s as %s) in %.0fs signals=%s | answers median %.2fs max %.2fs" % (cid, ",".join(info["buses"]), got, info["how"], info["vehicle"], t_adopt, info.get("signals"), info["median_s"], info["max_s"]), flush=True)
        for x in qa: print("     %5.2fs  %-45s -> %s" % (x["s"], x["q"], x["a"][:90]), flush=True)
        car.stop(); time.sleep(2)
    try: print("memory:", " ".join(l.split(":")[0] + l.split(":")[1].strip() for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM"))))
    except OSError: pass
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True); json.dump(results, open(os.path.join(ROOT, a.out), "w"), indent=1)
    app._close_linked(); shutil.rmtree(home, True); os._exit(0)


if __name__ == "__main__": main()

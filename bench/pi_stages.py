"""Stage times of a few questions on the board (model read vs the rest).   python bench/pi_stages.py"""
import os, sys, time
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from ui import server as S
if os.environ.get("NO_RIG"): S.CameraRig.start = lambda self: None                      # the linked product does not decode a camera clip in the application: the car sends its own
app = S.App(SimpleNamespace(db="data/world_aster.db", model="0.5b", camera="clip", webcam_index=0, live=None, linked=False)); app.ask("how fast am I going", "driver")
for q in ["why do I get less range these days", "what is the average battery temperature over the last trip", "is my tyre pressure fine", "highest speed last week"]:
    t = time.monotonic(); r = app.ask(q, "driver"); dt = time.monotonic() - t; time.sleep(2.2); t = time.monotonic(); app.ask(q, "driver"); print("   repeat (frame cached, no model run): %.2fs" % (time.monotonic() - t))
    print("%.2fs total" % dt, {k: v for k, v in r.items() if k not in ("answer", "facts", "evidence") and isinstance(v, (int, float, str)) and ("ms" in k or "time" in k or "lat" in k or k in ("route", "mode"))}, "|", q, flush=True)

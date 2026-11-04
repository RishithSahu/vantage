"""Where does the time of one slow question go?  python bench/pi_profile_ask.py "question" ["question" ...]   (cProfile of App.ask after a warm-up question; run with taskset -c 0,1 on the Pi)"""
import cProfile, os, pstats, sys, time
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from ui import server as S
app = S.App(SimpleNamespace(db=os.environ.get("DB", "data/aster_ev_4yr.db"), model="0.5b", camera="clip", webcam_index=0, live=None, linked=False))
app.ask("how fast am I going", "driver")
for q in sys.argv[1:]:
    pr = cProfile.Profile(); t = time.monotonic(); pr.enable(); app.ask(q, "driver"); pr.disable(); print("\n=== %.2fs  %s" % (time.monotonic() - t, q))
    pstats.Stats(pr).sort_stats("cumulative").print_stats(22)

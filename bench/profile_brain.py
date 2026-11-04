"""cProfile of the assistant's own work on a question (no UI thread).   python bench/profile_brain.py [--db data/aster_ev_4yr.db] [--top 25] "question" ...   (taskset -c 0,1 on the Pi)"""
import argparse, cProfile, os, pstats, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
ap = argparse.ArgumentParser(); ap.add_argument("--db", default="data/aster_ev_4yr.db"); ap.add_argument("--top", type=int, default=25); ap.add_argument("--sort", default="tottime"); ap.add_argument("q", nargs="+"); a = ap.parse_args()
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt import product
brain = product.make_qwen05_brain(a.db); brain.now = int(time.time())
brain.ask("how fast am I going")
for q in a.q:
    getattr(brain, "dialogue", None) and brain.dialogue.reset(); brain._last_data = None
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); r = brain.ask(q); pr.disable(); print("\n=== %.2fs  %s" % (time.perf_counter() - t, q))
    pstats.Stats(pr).sort_stats(a.sort).print_stats(a.top)

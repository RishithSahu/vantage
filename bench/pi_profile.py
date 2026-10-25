"""Where the time of one question goes on the board, without the model: the frame is cached, then cProfile of the rest.   python bench/pi_profile.py "question" ..."""
import cProfile, os, pstats, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_NO_QA_CLF"] = "1"
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.product import make_qwen05_brain
b = make_qwen05_brain(os.path.join(ROOT, "data", "world_aster.db")); b.now = 1792348500; b.ask("what is my speed right now")
qs = sys.argv[1:] or ["why do I get less range these days", "what is the average battery temperature over the last trip"]
for q in qs:
    for i in range(4):
        t = time.perf_counter(); b.ask(q); print("%.2fs  ask %d%s  %s" % (time.perf_counter() - t, i + 1, " (frame cached)" if i else " (model reads it)", q), flush=True); b.dialogue.reset()
pr = cProfile.Profile(); pr.enable()
for q in qs: b.ask(q); b.dialogue.reset()
pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_stats(35)

import os, sys, time, cProfile, pstats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from ui.server import make_brain
brain = make_brain("data/world_aster.db")
from vgpt.meaning import ort_backend as OB
_p = OB.OrtFrameModel.parse; cache = {}
def P(self, text, *a, **k):
    if text in cache: return cache[text]
    r = _p(self, text, *a, **k); cache[text] = r; return r
OB.OrtFrameModel.parse = P
qs = sys.argv[1:]
for w in ("how fast am I going",): brain.ask(w)
for q in qs: brain.dialogue.reset(); brain.ask(q)
for q in qs:
    brain.dialogue.reset(); t = time.perf_counter(); brain.ask(q); print("%.3fs post-parse" % (time.perf_counter() - t), q)
pr = cProfile.Profile(); pr.enable()
for q in qs: brain.dialogue.reset(); brain.ask(q)
pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_callers("re.__init__.py:178|execute.of .sqlite3.Connection")

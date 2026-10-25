import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from types import SimpleNamespace
import vehiclegpt; vehiclegpt._adopt_product_model()
from ui import server as S
S.CameraRig.start = lambda self: None
app = S.App(SimpleNamespace(db="data/world_aster.db", model="0.5b", camera="clip", webcam_index=0, live=None, linked=False))
for q in sys.argv[1:]:
    t = time.time(); r = app.ask(q, "driver"); print("%.2fs" % (time.time() - t), q, "->", str(r.get("answer"))[:110], flush=True)

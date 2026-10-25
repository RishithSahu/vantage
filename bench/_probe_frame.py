import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from types import SimpleNamespace
import vehiclegpt; vehiclegpt._adopt_product_model()
from ui import server as S
S.CameraRig.start = lambda self: None
app = S.App(SimpleNamespace(db="data/world_aster.db", model="0.5b", camera="clip", webcam_index=0, live=None, linked=False))
from vgpt.meaning import ort_backend as OB
_p = OB.OrtFrameModel.parse
def P(self, *a, **k):
    r = _p(self, *a, **k); print("FRAME", a[:1], "->", r, flush=True); return r
OB.OrtFrameModel.parse = P
for q in sys.argv[1:]:
    r = app.ask(q, "driver"); print(q, "->", str(r.get("answer"))[:80], r.get("facts", {}).get("route") if isinstance(r.get("facts"), dict) else "", flush=True)

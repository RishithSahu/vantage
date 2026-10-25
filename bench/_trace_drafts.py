import os, sys, io, contextlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["DEBUG_DRAFT"] = "1"
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.meaning.ort_backend import OrtFrameModel
m = OrtFrameModel(os.path.join(ROOT, "models", "ort_frame2w"), threads=4); m.parse("how fast am I going")
qs = [l.rstrip("\n").split("\t")[2] for l in open(os.path.join(ROOT, "bench", "acceptance_v1.tsv"), encoding="utf-8") if not l.startswith("#") and l.count("\t") > 1][::6][:30]
for q in qs:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): f, i = m.parse(q)
    lines = [l.strip() for l in buf.getvalue().splitlines() if "gen=" in l]
    print("Q:", q, "| frame:", i["raw"], "| passes", m.stats["passes"])
    for l in lines: print("    ", l[:150])

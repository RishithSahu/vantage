"""Does a different weight format of the SAME model give the same frames, and how fast is it?   python bench/pi_formats.py <base.gguf> [more.gguf ...]   (first file = reference)"""
import os, sys, time, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.vehicle import lake
from vgpt.meaning.interface import FrameModel
lake.adopt_path(os.path.join(ROOT, "data", "world_aster.db"))
qs = []
for ln in open(os.path.join(ROOT, "bench", "acceptance_v1.tsv"), encoding="utf-8"):
    p = ln.rstrip("\n").split("\t")
    if not ln.startswith("#") and len(p) > 2: qs.append(p[2])
qs = qs[:int(os.environ.get("N", 40))]; ref = None
for path in sys.argv[1:]:
    m = FrameModel(os.environ["VGPT_MEANING_ADAPTER"], base=path, threads=2, version=int(os.environ.get("VGPT_FRAME", 2))); m.parse("how fast am I going"); out = []; ts = []
    for q in qs:
        t = time.perf_counter(); f, info = m.parse(q); ts.append(time.perf_counter() - t); out.append(info["raw"])
    if ref is None: ref = out
    same = sum(a == b for a, b in zip(ref, out)); print("%-34s median %.2fs  mean %.2fs  max %.2fs  same frame as reference: %d/%d" % (os.path.basename(path), st.median(ts), st.mean(ts), max(ts), same, len(qs)), flush=True)
    del m

"""Time the frame model alone on the board: prompt length in tokens, total parse time, tokens produced.   python bench/pi_parse.py"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.vehicle import lake
from vgpt.meaning.interface import FrameModel, prompt_text
lake.adopt_path(os.path.join(ROOT, "data", "world_aster.db"))
m = FrameModel(os.environ["VGPT_MEANING_ADAPTER"], **({"base": os.environ["BASE"]} if os.environ.get("BASE") else {}), threads=int(os.environ.get("T", 2)), version=int(os.environ.get("VGPT_FRAME", 2)))
for q in ["how fast am I going", "current odometer", "share of time above 100 km/h", "current odometer"]:
    p = prompt_text("frame", q, m.glossary.hint_line(q), None, m.version); n = len(m.llm.tokenize(p.encode()))
    t = time.perf_counter(); f, info = m.parse(q); print("%.2fs prompt=%d tok out=%d  %s -> %s" % (time.perf_counter() - t, n, info["tokens"], q, info["raw"]), flush=True)
# per-step time: the gaps between the model's calls to the logits processor (step 0 = whatever is evaluated before the first token)
from llama_cpp import LogitsProcessorList
q = "current odometer"; p = prompt_text("frame", q, m.glossary.hint_line(q), None, m.version); marks = []
def tick(ids, sc): marks.append((time.perf_counter(), len(ids))); return sc
for rep in range(2):
    marks.clear(); t0 = time.perf_counter(); m.llm.create_completion(p, max_tokens=12, temperature=0.0, logits_processor=LogitsProcessorList([tick, m.mask.processor()]))
    ts = [t0] + [x[0] for x in marks]; print("steps ms:", [round((b - a) * 1000) for a, b in zip(ts, ts[1:])], "ctx tokens:", marks[0][1], "n_tokens", m.llm.n_tokens, flush=True)

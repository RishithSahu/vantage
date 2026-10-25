import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.vehicle import lake
from vgpt.meaning import glossary as G
from vgpt.meaning.interface import prompt_text
from llama_cpp import Llama
lake.adopt_path(os.path.join(ROOT, "data", "world_aster.db")); g = G.load(); l = Llama(model_path=os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), vocab_only=True, verbose=False)
qs = ["current odometer", "share of time above 100 km/h", "how fast am I going", "was my tyre pressure low last week", "what is the average battery temperature over the last trip"]
base = None
for q in qs:
    h = g.hint_line(q); p = prompt_text("frame", q, h, None, 2); ids = l.tokenize(p.encode(), add_bos=False, special=True)
    if base is None: base = ids
    n = 0
    while n < len(ids) and n < len(base) and ids[n] == base[n]: n += 1
    print(len(ids), "tail", len(ids) - n, "hint=%r" % h, "|", q)

"""Raw model speed on the board with and without the adapter: one-token evals and a 24-token batch."""
import os, sys, time
from llama_cpp import Llama
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.environ.get("BASE", os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"))
for name, lora in (("base only", None), ("with adapter", os.path.join(ROOT, "models", "meaning-lora-frame2d.gguf"))):
    kw = dict(model_path=base, n_ctx=1024, n_threads=int(os.environ.get("T", 2)), n_threads_batch=int(os.environ.get("T", 2)), n_batch=128, n_ubatch=128, use_mmap=True, verbose=False)
    if lora: kw["lora_path"] = lora
    m = Llama(**kw); toks = m.tokenize(b"hello world this is a test of the speed of the model on the board " * 30)[:300]
    m.eval(toks[:40]); m.reset(); m.eval(toks[:280])
    t = time.perf_counter(); m.eval(toks[280:300]); b = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(10): m.eval([toks[i]])
    one = (time.perf_counter() - t) / 10
    print("%-13s 20-token batch %.0f ms (%.0f ms/token)   single token %.0f ms" % (name, b * 1000, b * 50, one * 1000), flush=True)

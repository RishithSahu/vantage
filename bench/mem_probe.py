"""Where does the committed memory go? Usage: python bench/mem_probe.py <model> <mmap 0/1> [n_ctx] [n_batch]  (no job limit; reads this process' commit charge)"""
import os, sys, psutil
sys.path.insert(0, ".")
p = psutil.Process(); mb = lambda x: x / 1048576
def snap(tag): m = p.memory_info(); print(f"  {tag:34s} commit(pagefile) {mb(m.pagefile):7.0f} MB | working set {mb(m.rss):7.0f} MB")
model, mmap = sys.argv[1], int(sys.argv[2]); nctx = int(sys.argv[3]) if len(sys.argv) > 3 else 640; nb = int(sys.argv[4]) if len(sys.argv) > 4 else 32
snap("python baseline")
from llama_cpp import Llama; snap("after import llama_cpp")
llm = Llama(model_path=model, n_ctx=nctx, n_threads=2, n_threads_batch=2, n_batch=nb, n_ubatch=nb, use_mmap=bool(mmap), verbose=False); snap(f"after load (mmap={mmap}, ctx={nctx}, batch={nb})")
llm.create_completion("Hello", max_tokens=2); snap("after first generation")

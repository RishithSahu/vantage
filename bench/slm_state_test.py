"""Measure the persisted-state speed-up locally: cold build vs warm load of the shared prompt, and lazy unload/reload."""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(_v, "1")
sys.path.insert(0, ".")
from vgpt.brain import Slm, LazySlm
path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"; st = "data/slm_state_test.pkl"
if os.path.exists(st): os.remove(st)
for label in ("first boot (no state file)", "second boot (state file present)"):
    t = time.perf_counter(); s = Slm(path, threads=2, state_path=st); t_load = time.perf_counter() - t
    t = time.perf_counter(); s.warmup(); t_warm = time.perf_counter() - t
    t = time.perf_counter(); r = s.parse("Why did the pack temperature alert trigger?"); t_q = time.perf_counter() - t
    print(f"{label:34s} model load {t_load:5.2f}s | warmup {t_warm:5.2f}s ({s.state_source}) | first question {t_q*1000:6.0f} ms -> {r[0]} {r[1]}")
    s.close()
print("state file:", os.path.getsize(st) // 1024, "KB")
lz = LazySlm(path, idle_s=0.2, threads=2, state_path=st)
t = time.perf_counter(); lz.parse("hello car"); print(f"lazy: first use (load+warm+query) {time.perf_counter()-t:.2f}s loaded={lz.loaded}")
time.sleep(0.4); lz.tick(); print("after idle tick loaded =", lz.loaded)
t = time.perf_counter(); lz.parse("hello car"); print(f"lazy: reload after unload {time.perf_counter()-t:.2f}s loads={lz.loads}")
os.remove(st)

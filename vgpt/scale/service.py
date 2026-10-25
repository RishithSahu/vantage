"""The assistant service that sizes itself: probe -> plan -> elastic model pool + sessions + autoscaler, on any machine.

    svc = AssistantService("data/world_aster.db")            # reads the hardware, builds what fits
    answer = svc.ask("alice", "how is my tyre pressure?", persona="driver")
    svc.status()                                              # the plan, the pool, the controller's decisions
    svc.close()

The same call works on a 2-core board (one context, two threads, no spare cores kept back), on a 12-core laptop (several contexts, cores and memory left for the rest of the machine) and on a machine with an
accelerator (the float graph on the GPU).  Nothing in the caller changes; `status()["plan"]` says what was chosen and why."""
import os
import time

from .. import sessions as S
from .autoscale import Autoscaler, ElasticModelPool
from .procpool import ProcessPool
from .plan import make_plan
from .probe import probe

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def installed(models_dir=None):
    """Which model builds can be used on this machine."""
    md = models_dir or os.path.join(ROOT, "models"); out = {"onnx": False, "llama": False, "float": False, "ort_dir": None, "float_dir": None}
    try:
        import onnxruntime  # noqa: F401
        for name in sorted(os.listdir(md)) if os.path.isdir(md) else []:
            d = os.path.join(md, name)
            if name.startswith("ort_") and os.path.exists(os.path.join(d, "model.onnx")):
                if name.endswith("_float"): out["float"] = True; out["float_dir"] = d
                elif not out["ort_dir"] or name == "ort_frame2d_c": out["onnx"] = True; out["ort_dir"] = d
    except ImportError: pass
    try:
        import llama_cpp  # noqa: F401
        out["llama"] = os.path.exists(os.path.join(md, "qwen2.5-0.5b-instruct-q4_k_m.gguf"))
    except ImportError: pass
    return out


class AssistantService:
    def __init__(self, db, hw=None, plan=None, models_dir=None, model_factory=None, brain_factory=None, autoscale=True, vehicles=1, min_contexts=1, start_contexts=None, latency_target_s=2.0, now_fn=None, mode=None, worker_cfg=None, knee=None):
        self.hw = hw or probe(); inst = installed(models_dir) if model_factory is None else {"onnx": True, "llama": True, "float": False, "ort_dir": None, "float_dir": None}
        self.plan = plan or make_plan(self.hw, have_onnx=inst["onnx"], have_llama=inst["llama"], have_float_graph=inst["float"], vehicles=vehicles, latency_target_s=latency_target_s, knee=knee)
        self.mode = mode or ("process" if (self.plan.max_contexts > 1 and model_factory is None) or worker_cfg else "thread")
        if self.mode == "process":                                                                                # one process per worker: the part of an answer that is not the model is Python, and threads of one process share one interpreter lock
            env = {k: os.environ[k] for k in ("VGPT_FRAME", "VGPT_MEANING_ADAPTER", "VGPT_ORT_DIR", "VEHICLEGPT_HOME") if k in os.environ}; env.update({"HF_HUB_OFFLINE": "1", "VGPT_NO_QA_CLF": "1", "VGPT_ORT_SPIN": "0" if self.plan.max_contexts > 2 else "1"})
            cfg = {"db": db, "backend": self.plan.backend if self.plan.backend != "none" else "llama", "ort_dir": inst.get("float_dir") if self.plan.backend == "ort-gpu" else inst.get("ort_dir"), "threads": max(1, self.plan.threads), "adapter": os.environ.get("VGPT_MEANING_ADAPTER") or os.path.join(ROOT, "models", "meaning-lora-frame2d.gguf"),
                   "root": ROOT, "env": env, "start_timeout": 900}
            cfg.update(worker_cfg or {})
            start = int(start_contexts or self.plan.contexts or 1); self.pool = ProcessPool(cfg, workers=start, min_workers=min_contexts, max_workers=max(self.plan.max_contexts, start)); self.hub = None; self.factory = None
        else:
            self.factory = model_factory or self._model_factory(inst)
            start = int(start_contexts or min_contexts); mx = max(self.plan.max_contexts, self.plan.contexts, start, 1)
            self.pool = ElasticModelPool(self.factory, workers=start, min_workers=min_contexts, max_workers=mx)
            if brain_factory is None:
                from ..product import make_meaning_brain
                brain_factory = lambda d, slm: make_meaning_brain(d, slm=slm, threads=self.plan.threads)
            self.hub = S.SessionHub(db, self.pool, brain_factory, max_sessions=self.plan.sessions, now_fn=now_fn)
        self.now = now_fn
        self.ctl = Autoscaler(self.pool, self.plan, mem_used_mb=_rss_mb, mem_budget_mb=max(256, self.hw.mem_mb - self.plan.reserve_mb)).start() if autoscale and self.pool.max_workers > self.pool.min_workers else None
        self.started = time.time()

    def _model_factory(self, inst):
        threads = max(1, self.plan.threads)
        if self.plan.backend == "ort-gpu" and inst["float_dir"]:
            from ..meaning.ort_backend import OrtFrameModel
            return lambda: OrtFrameModel(inst["float_dir"], threads=threads, provider="gpu")
        if self.plan.backend == "ort-int8" and inst["ort_dir"]:
            from ..meaning.ort_backend import OrtFrameModel
            return lambda: OrtFrameModel(inst["ort_dir"], threads=threads)
        from ..meaning.interface import FrameModel
        adapter = os.environ.get("VGPT_MEANING_ADAPTER") or os.path.join(ROOT, "models", "meaning-lora-frame2d.gguf")
        return lambda: FrameModel(adapter, threads=threads)

    def ask(self, session, text, persona="driver", timeout=180.0):
        if self.hub is not None: return self.hub.ask(session, text, persona, timeout)
        return self.pool.ask(session, text, persona=persona, timeout=timeout, now=self.now() if self.now else None)

    def status(self):
        return {"hardware": self.hw.describe(), "plan": self.plan.as_dict(), "pool": self.pool.metrics(), "pool_totals": self.pool.stats(), "sessions": self.hub.stats() if self.hub else {}, "autoscaler": list(self.ctl.log[-20:]) if self.ctl else []}

    def close(self):
        if self.ctl: self.ctl.stop()
        if self.hub: self.hub.close()
        self.pool.close()


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"): return int(line.split()[1]) // 1024
    except OSError: pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure): _fields_ = [("cb", wintypes.DWORD), ("pf", wintypes.DWORD), ("peak", ctypes.c_size_t), ("ws", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("pr", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True); ps = ctypes.WinDLL("psapi", use_last_error=True); k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]; ps.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        return int(pmc.ws) // (1 << 20) if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb) else 0
    except Exception: return 0

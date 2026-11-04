"""ctypes binding of the native engine (engine/vg_engine.c): the whole transformer forward pass in C, called with the same arguments and results as the ONNX graph.

    eng = Engine("models/vg_frame2r12.vgw", threads=2)
    logits, kv = eng.forward1(ids, start, kv, nkeep, cand)          # one sequence; kv = the 48 [1,2,P,64] float32 arrays of the ONNX graph
    outs = eng.forward([(ids, start, kv, nkeep), ...], cand)        # several sequences (cars) in one call: their rows share every weight read

The library is built from engine/vg_engine.c (see its header); `build()` compiles it with the local gcc."""
import ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); ENG = os.path.join(ROOT, "engine")
L, NKV, HD = 24, 2, 64


class Seq(ctypes.Structure):
    _fields_ = [("ids", ctypes.c_void_p), ("T", ctypes.c_int), ("pos0", ctypes.c_int), ("P", ctypes.c_int), ("kv_in", ctypes.c_void_p), ("kv_out", ctypes.c_void_p), ("nkeep", ctypes.c_int), ("logits", ctypes.c_void_p)]


def libname(): return os.path.join(ENG, "libvg.dll" if os.name == "nt" else "libvg.so")


def build(force=False, out=None):
    """Compile the engine for this machine (engine/build.py holds the compiler call, so this module starts no process).  -> the library path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("vg_engine_build", os.path.join(ENG, "build.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.build(force, out or libname())


class Engine:
    def __init__(self, vgw, threads=2, lib=None):
        self.lib = ctypes.CDLL(lib or build()); self.lib.vg_load.restype = ctypes.c_void_p; self.lib.vg_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.vg_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]; self.lib.vg_kernel.restype = ctypes.c_char_p
        self.lib.vg_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.h = self.lib.vg_load(os.fsencode(vgw), threads)
        if not self.h: raise RuntimeError("cannot load " + str(vgw))
        self.kernel = self.lib.vg_kernel().decode()

    def forward(self, seqs, cand):
        """seqs: [(ids, start, kv, nkeep)] -> [(logits [nkeep, C] float32, kv_out list of 48 arrays [1,2,P+T,64])]"""
        cand = np.ascontiguousarray(cand, np.int64); n = len(seqs); arr = (Seq * n)(); keep = []
        for i, (ids, start, kv, nkeep) in enumerate(seqs):
            ids = np.ascontiguousarray(ids, np.int64); T = len(ids); P = kv[0].shape[2]
            kin = np.array([np.ascontiguousarray(a, np.float32).ctypes.data for a in kv], np.uintp); kin_keep = [np.ascontiguousarray(a, np.float32) for a in kv]; kin = np.array([a.ctypes.data for a in kin_keep], np.uintp)
            outs = [np.empty((1, NKV, P + T, HD), np.float32) for _ in range(2 * L)]; kout = np.array([a.ctypes.data for a in outs], np.uintp); lg = np.empty((nkeep, len(cand)), np.float32)
            arr[i].ids = ids.ctypes.data; arr[i].T = T; arr[i].pos0 = int(start); arr[i].P = P; arr[i].kv_in = kin.ctypes.data; arr[i].kv_out = kout.ctypes.data; arr[i].nkeep = int(nkeep); arr[i].logits = lg.ctypes.data
            keep.append((ids, kin, kin_keep, kout, outs, lg))
        rc = self.lib.vg_forward(self.h, n, ctypes.addressof(arr), cand.ctypes.data, len(cand))
        if rc != 0: raise ValueError("the engine refused the request: a token id, a position or a buffer is out of range")
        return [(k[5], k[4]) for k in keep]

    def forward1(self, ids, start, kv, nkeep, cand):
        (lg, outs), = self.forward([(ids, start, kv, nkeep)], cand); return lg, outs

    def profile(self, reset=True):
        a = (ctypes.c_double * 10)(); self.lib.vg_profile(self.h, a, int(reset)); return list(a)

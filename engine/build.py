"""Compiles engine/vg_engine.c for this machine: python engine/build.py   (needs gcc).  Called by vgpt/meaning/native.py when the library is missing or older than the source."""
import os, subprocess, sys
ENG = os.path.dirname(os.path.abspath(__file__))


def build(force, out):
    """Compile the engine for this machine (aarch64: Cortex-A72 tuning and the assembly kernels; x86-64: AVX2).  -> the library path."""
    if os.path.exists(out) and not force and os.path.getmtime(out) >= max(os.path.getmtime(os.path.join(ENG, f)) for f in ("vg_engine.c", "k_aarch64.h")): return out
    import platform
    arch = ["-mcpu=cortex-a72"] if platform.machine() in ("aarch64", "arm64") else ["-mavx2"]
    cmd = ["gcc", "-O3", *arch, "-ffp-contract=off", "-shared", "-fPIC", "-pthread", "-o", out, os.path.join(ENG, "vg_engine.c"), "-lm"] + (["-static-libgcc", "-Wl,-Bstatic", "-lwinpthread", "-Wl,-Bdynamic"] if os.name == "nt" else [])
    subprocess.check_call(cmd); return out


if __name__ == "__main__":
    print(build(True, os.path.join(ENG, "libvg.dll" if os.name == "nt" else "libvg.so")))

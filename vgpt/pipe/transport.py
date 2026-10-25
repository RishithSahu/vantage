r"""Where a car and Vantage meet when they were started separately: a local pipe with a well-known name.

    Windows   a named pipe  \\.\pipe\VehicleGPT-<user>-link   (created by Vantage, refuses remote clients: it cannot be reached from another machine)
    Linux     a FIFO        <runtime dir>/vehiclegpt-<user>/link.fifo   (mode 0600, created by Vantage)

Neither is a network socket: both are objects of the local operating system, and vgpt/offline.py would refuse a socket anyway.  Vantage LISTENS (it holds the read end and waits for a car to open the pipe); the car
CONNECTS (vgpt/car/transport.py, which holds the write end).  When the car closes its end the reader sees the end of the stream and Vantage listens again, so a car can be switched off, changed for another one and
switched on, and Vantage follows.

This module is the reading side.  It is in the collector domain because it makes an operating-system object (isolation.py lists it with the reason).
Android: the same two ends are a pair of ParcelFileDescriptors handed over by Binder (docs/ANDROID_READY.md); nothing above this layer changes."""
import getpass
import os
import stat
import sys
import tempfile
import threading


def user_tag():
    try: return getpass.getuser()
    except Exception: return str(os.environ.get("USERNAME") or os.environ.get("USER") or "user")


def endpoint(name=None):
    """The well-known name of the link on this machine (VGPT_LINK overrides it, which is how two test benches run side by side)."""
    name = name or os.environ.get("VGPT_LINK") or "link"
    if sys.platform == "win32": return r"\\.\pipe\VehicleGPT-%s-%s" % (user_tag(), name)
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(base, "vehiclegpt-%s" % (os.getuid() if hasattr(os, "getuid") else user_tag()), name + ".fifo")


class Listener:
    """Vantage's side.  `accept()` waits for a car and returns a file descriptor to read; it returns None after `close()`."""
    def __init__(self, path=None):
        self.path = path or endpoint(); self._stop = threading.Event(); self._lock = threading.Lock()
        if sys.platform == "win32": self._k32 = _kernel32()
        else:
            d = os.path.dirname(self.path); os.makedirs(d, mode=0o700, exist_ok=True)
            if os.path.exists(self.path) and not stat.S_ISFIFO(os.stat(self.path).st_mode): raise OSError(f"{self.path} exists and is not a pipe")
            if not os.path.exists(self.path): os.mkfifo(self.path, 0o600)

    def accept(self):
        if self._stop.is_set(): return None
        fd = self._accept_windows() if sys.platform == "win32" else os.open(self.path, os.O_RDONLY)
        if self._stop.is_set():
            if fd is not None:
                try: os.close(fd)
                except OSError: pass
            return None
        return fd

    def _accept_windows(self):
        try:
            import ctypes
            import msvcrt
        except ImportError as e: raise OSError("the Windows named pipe needs Windows") from e                    # only reached when sys.platform is win32
        k = self._k32; INVALID = ctypes.c_void_p(-1).value
        h = k.CreateNamedPipeW(self.path, 0x1, 0x0 | 0x0 | 0x0 | 0x8, 255, 65536, 65536, 0, None)             # INBOUND, byte type, byte read mode, wait, reject remote clients, many instances
        if h in (None, INVALID): raise OSError(ctypes.get_last_error(), "could not create the link pipe")
        ok = k.ConnectNamedPipe(h, None)
        if not ok and ctypes.get_last_error() != 535:                                                               # 535: the car connected between create and connect, which is fine
            k.CloseHandle(h); raise OSError(ctypes.get_last_error(), "waiting for a car failed")
        return msvcrt.open_osfhandle(h, os.O_RDONLY)

    def close(self):
        """Release a blocked `accept()` and stop listening."""
        self._stop.set()
        try:
            if sys.platform == "win32":
                with open(self.path, "wb", buffering=0): pass                                                       # a throw-away client lets ConnectNamedPipe return
            else:
                fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK); os.close(fd)
        except OSError: pass


def _kernel32():
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
    except (ImportError, AttributeError) as e: raise OSError("the Windows named pipe needs Windows") from e
    k.CreateNamedPipeW.restype = wintypes.HANDLE
    k.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    k.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p]; k.ConnectNamedPipe.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    return k

"""The car's side of the rendezvous: open the write end of the local pipe Vantage listens on (vgpt/pipe/transport.py names it).  Retries until Vantage is there, so the car and Vantage may be started in either order."""
import os
import sys
import time

from ..pipe import transport as T


def connect(path=None, timeout=None, stop=None, poll=0.4):
    """-> a file descriptor to write to, or None when `timeout` ran out / `stop` (a threading.Event) was set."""
    path = path or T.endpoint(); end = None if timeout is None else time.monotonic() + timeout
    while True:
        if stop is not None and stop.is_set(): return None
        try:
            if sys.platform == "win32": return os.open(path, os.O_WRONLY | os.O_BINARY)
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)                         # fails while no reader has the FIFO open
            os.set_blocking(fd, True); return fd
        except OSError: pass
        if end is not None and time.monotonic() >= end: return None
        time.sleep(poll)

"""The car's end of the pipe: records onto a file descriptor (an anonymous pipe, a FIFO, a Windows named pipe).  It can only write; Vantage never imports this module."""
import json
import os
import threading

from ..pipe import wire
from ..pipe.wire import LinkClosed


class Writer:
    """The car's end: records onto a file descriptor (a pipe, a FIFO, a named pipe).  Thread-safe.  It can only write."""
    def __init__(self, fd, hello=None):
        self.fd = fd; self._lock = threading.Lock(); self.records = 0; self.bytes = 0; self.closed = False; self._first = True
        if hello is not None: self.control(**hello)

    def send(self, ts, channel, payload):
        data = wire.encode(ts, channel, payload, wire.FLAG_FIRST if self._first else 0)
        with self._lock:
            if self.closed: raise LinkClosed("the link is closed")
            view = memoryview(data)
            try:
                while view:
                    n = os.write(self.fd, view); view = view[n:]
            except (BrokenPipeError, OSError) as e:
                self.closed = True; raise LinkClosed(str(e)) from e
            self._first = False; self.records += 1; self.bytes += len(data)

    def control(self, **obj): self.send(None, wire.CONTROL, json.dumps(obj, separators=(",", ":")).encode("utf-8"))

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                try: os.close(self.fd)
                except OSError: pass



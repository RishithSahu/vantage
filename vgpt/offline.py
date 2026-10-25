"""The product is fully offline, and this module makes that a fact about the running process, not a promise in a document.

Nothing in the product speaks HTTP or any other network protocol: the interface is reached through an in-process bridge (`vgpt/shell`), the data arrives from a bus, a vehicle API or a file.  `install()` adds a Python *audit hook*
that refuses, for every caller in the process (the product, its libraries, `ctypes`-free native modules going through Python's `socket`), the creation of any socket that is not a **CAN bus** socket, and refuses every name
lookup.  An audit hook cannot be removed once it is installed.  It is installed when `vgpt` is imported.

What this is not: a defence against native code that bypasses Python (the source scan in `vgpt/isolation.py` forbids `ctypes` and foreign-function modules in the assistant, and the operating system layer - no INTERNET permission on
Android, `PrivateNetwork=yes` and an address-family filter in the systemd unit - closes the same door from outside).  Three layers, each tested on its own."""
import sys

try:
    import socket as _socket
    _CAN = getattr(_socket, "AF_CAN", None)                                           # Linux only; on Windows and Android's Python there is none, and then no socket at all is allowed
except ImportError:                                                                   # a Python built without sockets: nothing to guard
    _socket = None; _CAN = None

BLOCKED_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo", "socket.getservbyname", "socket.getservbyport", "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}
_installed = False
refused = []                                                                           # what was refused in this process: [(event, detail)], for the Privacy page and for the tests


class OfflineError(OSError):
    """Raised when something tries to reach a network.  An OSError, so code that already copes with a failed connection copes with this."""


def _hook(event, args):
    if event == "socket.__new__":
        family = args[1] if len(args) > 1 else None
        if _CAN is not None and family == _CAN: return                              # a CAN bus socket: a vehicle bus, not a network
        refused.append((event, f"family={family!r}")); raise OfflineError("this product is fully offline: it cannot create a network socket (only a CAN bus socket is allowed)")
    if event in BLOCKED_EVENTS:
        if event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg") and _CAN is not None and args and _is_can(args[0]): return
        refused.append((event, "")); raise OfflineError(f"this product is fully offline: {event} is refused")


def _is_can(sock):
    try: return sock.family == _CAN
    except Exception: return False


def install():
    """Add the guard (once).  Returns True the first time, False when it was already in place."""
    global _installed
    if _installed or _socket is None: return False
    sys.addaudithook(_hook); _installed = True; return True


def installed(): return _installed


def check():
    """-> a plain statement of what the guard sees right now, for the Privacy page: nothing can be opened, and what was refused."""
    return {"guard_installed": _installed, "network_sockets_allowed": False, "can_sockets_allowed": _CAN is not None, "refused_attempts": len(refused), "last_refusals": [f"{e} {d}".strip() for e, d in refused[-5:]]}

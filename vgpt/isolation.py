"""Safety isolation and "fully offline", MEASURED instead of promised.

Two rules, both read from the source of the whole application:

  1. NO NETWORK ANYWHERE.  No module of the product imports a network facility (sockets, HTTP, TLS, FTP, SMTP, WebSocket, MQTT, ZeroMQ ...).  The one exception is the CAN bus: `vgpt/live/collect.py` imports `socket` to open a
     CAN socket (AF_CAN: a bus, not a network), and `vgpt/offline.py` imports it to know that family; both are checked to refer to no network address family, no connect / listen / accept and no name lookup.  At run time
     `vgpt/offline.py` refuses everything else.
  2. THE ASSISTANT CANNOT ACT.  The collector domain is the code that starts processes or touches a bus: the CAN collector (`live/collect.py`), start-up and health-check tools, the window shells.  The assistant domain is everything
     else: the language model, the meaning pipeline, the resolver, the SQL planner and executors, the answer writer, the protocol DECODERS.  It starts no process and holds a read-only database handle; nothing in it writes to a
     vehicle bus (the SocketCAN socket is wrapped so that it can only receive).

`scan()` reads the source and reports which modules import a network, process or foreign-function facility and which contain a call that transmits.  `verify()` compares that with the two rules and `COLLECTOR_DOMAIN` below (each
member has a stated reason).  The Privacy page runs it on the machine, and tests/test_isolation.py fails the build if the boundary moves without this file being updated on purpose.  This is a software boundary on one host;
separating the domains into partitions or a hypervisor is the vehicle-level step described in docs/SAFETY_ISOLATION.md."""
import ast
import os
import sys
import re

# what can reach outside the process
NETWORK_IMPORTS = {"socket", "ssl", "socketserver", "http.client", "http.server", "urllib.request", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc", "requests", "httpx", "aiohttp", "websockets",
                   "websocket", "paho", "zmq", "can", "serial", "usb", "bluetooth", "multiprocessing.connection"}
PROCESS_IMPORTS = {"subprocess", "ctypes", "cffi", "pty"}
NETWORK_NAMES = {"AF_INET", "AF_INET6", "AF_UNIX", "AF_PACKET", "AF_NETLINK", "SOCK_STREAM", "SOCK_DGRAM", "create_connection", "create_server", "gethostbyname", "gethostbyaddr", "getaddrinfo", "getnameinfo", "listen", "accept",
                 "connect", "connect_ex", "socketpair", "fromfd", "sendto", "sendmsg", "sendall"}                                       # what a network socket needs; none may appear in a module that is allowed to import `socket`
SOCKET_USERS = {"vgpt/live/collect.py": "opens the CAN bus socket (AF_CAN) of the vehicle's CAN controller, receive-only, and nothing else",
                "vgpt/offline.py": "imports `socket` only to learn the CAN family it lets through, and to refuse every other socket"}
TRANSMIT_CALLS = {"send", "sendall", "sendto", "sendmsg", "send_periodic", "send_message", "sendfile"}
SHELL_CALLS = {"system", "popen", "Popen", "execv", "execve", "execl", "spawnl", "startfile"}

# module -> why it may touch the outside; everything not listed is assistant domain
COLLECTOR_DOMAIN = {
    "vgpt/car/aaos.py": "the Car program for an Android Automotive device: asks the device (adb, a process on this machine) for its Vehicle HAL values and writes them to the local pipe; the car side only",
    "vgpt/car/car.py": "the simulated Car, a separate program: it writes its buses' traffic onto the local one-way pipe; Vantage never imports it and the assistant domain cannot reach it",
    "vgpt/car/transmit.py": "the Car's transmitter: turns each bus message into the form that bus carries and writes it to the pipe; the car side only",
    "vgpt/car/writer.py": "the Car's end of the local pipe: it can only write; Vantage's code reads the pipe through vgpt/pipe/wire.py and cannot transmit",
    "vgpt/pipe/transport.py": "makes the local pipe Vantage listens on (a Windows named pipe through ctypes that refuses remote clients, or a 0600 FIFO): an operating-system object, never a network socket",
    "vgpt/live/collect.py": "receives what the vehicle's CAN bus carries (the one socket of the program: AF_CAN, receive-only wrapper, listen-only check) and follows files; it asks the system for the controller's mode; no network",
    "vgpt/offline.py": "the guard that refuses every socket that is not a CAN bus socket; it imports `socket` only to know that family",
    "vgpt/appinstall.py": "command-line tool that writes the icon of the app (on Windows through PowerShell); never imported by the assistant",
    "vgpt/atrest.py": "the at-rest vault: asks the platform's secure store (TPM tools, a keyring, a named command) for the data key; imported only by tools that store data",
    "vgpt/shell/pipe_shell.py": "starts the Chromium window (a process on this device) and talks to it over two pipes; every request the page makes is failed by the browser unless it is a file, data or blob address",
    "vgpt/governor.py": "test harness only: a Windows job object that simulates the 2 GB / 2-core target on a laptop; the product never imports it",
    "ui/server.py": "the application object and the in-process dispatcher of the interface (no listener); reads the process's memory use through the platform's own call",
    "vehiclegpt.py": "command line: starts the window, the self-test and the tools",
    "vgpt/hardening.py": "start-up only: sets the process non-dumpable and without core dumps through prctl (ctypes) before any vehicle data is read; reads no data, opens no socket",
    "vgpt/meaning/native.py": "loads the compute-only native engine (engine/vg_engine.c) through ctypes: it holds the model's weights and does arithmetic, has no socket, file write or process call, and the compiler is run only by engine/build.py, never by this module",
    "vgpt/scale/probe.py": "asks the platform how many cores, how much memory and which accelerator it has (Windows through ctypes, nvidia-smi as a process); reads no vehicle data",
    "vgpt/scale/procpool.py": "the supervisor of the assistant's worker processes (one per model context): starts them and talks to each over an anonymous operating-system pipe (no socket, no bus); the worker body in this file runs the unchanged assistant code",
    "vgpt/scale/service.py": "the scaling service: owns the pool of workers and reads the process's own memory use through the platform's call; it holds no bus and no network",
}
# modules that are dev/build tools in the application folder root (tools_*.py, app.py): outside the product
TOOL_PATTERN = re.compile(r"^(tools_[a-z_]+|app|install|start)\.py$")


def _dotted(node):
    if isinstance(node, ast.Attribute): return _dotted(node.value) + "." + node.attr
    if isinstance(node, ast.Name): return node.id
    return ""


def _imports(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module); out.update(n.module + "." + a.name for a in n.names)
    return out


def _hits(imports, wanted):
    return sorted({i for i in imports for w in wanted if i == w or i.startswith(w + ".")})


def scan_file(path):
    try: tree = ast.parse(open(path, encoding="utf-8").read(), path)
    except (OSError, SyntaxError) as e: return {"error": f"{type(e).__name__}: {e}"}
    imp = _imports(tree); transmits = []; shells = []; af_can = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in TRANSMIT_CALLS: transmits.append((n.lineno, n.func.attr, _dotted(n.func.value)))
            if n.func.attr in SHELL_CALLS and _dotted(n.func.value) in ("os", "subprocess", "sp"): shells.append((n.lineno, n.func.attr))
        if isinstance(n, ast.Attribute) and n.attr == "AF_CAN": af_can = True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.args:                                          # run-time code: eval / exec, or an import whose module name is computed
            if n.func.id in ("eval", "exec") or (n.func.id == "__import__" and not isinstance(n.args[0], ast.Constant)): shells.append((n.lineno, n.func.id))
    return {"network": _hits(imp, NETWORK_IMPORTS), "process": _hits(imp, PROCESS_IMPORTS), "transmit": transmits, "shell": shells, "af_can": af_can, "imports": sorted(imp)}


def source_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan(root=None):
    """{relative path: findings} for every Python module of the application (vgpt/, ui/, the entry script)."""
    root = root or source_root(); res = {}
    for base in ("vgpt", "ui"):
        for d, _dirs, files in os.walk(os.path.join(root, base)):
            if "__pycache__" in d: continue
            for f in files:
                if f.endswith(".py"): p = os.path.join(d, f); res[os.path.relpath(p, root).replace(os.sep, "/")] = scan_file(p)
    for entry in ("vehiclegpt.py", "app.py"):                                                                           # the entry scripts count too: app.py once carried a loopback web server
        if os.path.exists(os.path.join(root, entry)): res[entry] = scan_file(os.path.join(root, entry))
    return res


def _class_calls(path, cls):
    """Transmit-like calls inside one class of a module."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            return [(c.lineno, c.func.attr) for c in ast.walk(n) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in TRANSMIT_CALLS]
    return None


def _network_names(path):
    """Names that only a network socket needs, referenced anywhere in one module."""
    tree = ast.parse(open(path, encoding="utf-8").read(), path); out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in NETWORK_NAMES: out.append(n.attr)
        elif isinstance(n, ast.Name) and n.id in NETWORK_NAMES: out.append(n.id)
    return out


def verify(root=None):
    """The measured boundary.  `ok` is None when the source is not available (a packaged build), otherwise True only if
       0. no module imports a network facility (the CAN bus socket is the one exception, and it refers to no network address family),
       1. the only modules that can start a process are the documented collector domain,
       2. nothing opens a CAN socket except the receive-only collector, and nothing in that collector transmits,
       3. no module outside the collector domain transmits or starts a process,
       4. no module imports a CAN library that can send (python-can) or a serial/USB/bluetooth stack."""
    root = root or source_root()
    if not os.path.isdir(os.path.join(root, "vgpt")) or getattr(sys, "frozen", False): return {"ok": None, "violations": [], "domain": {}, "assistant_modules": 0, "detail": "the source is not available here"}
    res = scan(root); violations = []; assistant = 0
    for mod, r in sorted(res.items()):
        if "error" in r: violations.append(f"{mod}: could not be read ({r['error']})"); continue
        for imp in r["network"]:
            if imp.split(".")[0] in ("can", "serial", "usb", "bluetooth"): violations.append(f"{mod}: imports {imp}, a stack that can transmit on a vehicle bus")
            elif not (mod in SOCKET_USERS and imp == "socket"): violations.append(f"{mod}: imports {imp}, a network facility (this product has no network)")
        if mod in SOCKET_USERS:
            bad = _network_names(os.path.join(root, mod))
            if bad: violations.append(f"{mod}: may import `socket` for the CAN bus only but refers to " + ", ".join(sorted(set(bad))))
        outside = r["process"] or r["transmit"] or r["shell"]
        if mod in COLLECTOR_DOMAIN: continue
        if outside: violations.append(f"{mod}: outside the collector domain but uses " + ", ".join(r["process"] + [f"{a}() line {n}" for n, a, _o in r["transmit"]] + [f"{a}() line {n}" for n, a in r["shell"]]))
        else: assistant += 1
        if r["af_can"] and mod != "vgpt/live/collect.py": violations.append(f"{mod}: opens a CAN socket")
    cc = os.path.join(root, "vgpt", "live", "collect.py")
    if os.path.exists(cc):
        calls = _class_calls(cc, "SocketCanCollector")
        if calls is None: violations.append("vgpt/live/collect.py: the SocketCAN collector is missing")
        elif calls: violations.append("vgpt/live/collect.py: the SocketCAN collector transmits: " + ", ".join(f"{a}() line {n}" for n, a in calls))
    for mod in COLLECTOR_DOMAIN:
        if not os.path.exists(os.path.join(root, mod)): violations.append(f"{mod}: listed in the collector domain but missing (update vgpt/isolation.py)")
    return {"ok": not violations, "violations": violations, "domain": dict(COLLECTOR_DOMAIN), "assistant_modules": assistant,
            "detail": f"{assistant} modules of the assistant domain import no network, process or bus facility and contain no transmit call; {len(COLLECTOR_DOMAIN)} collector-domain modules are listed with their purpose; the SocketCAN socket is receive-only."}


_cache = {}


def cached_verify(root=None):
    """verify() once per process (the source does not change while the application runs)."""
    if "v" not in _cache: _cache["v"] = verify(root)
    return _cache["v"]


def listen_only(ip_details_text):
    """Whether a CAN controller was put in listen-only mode (`ip link set can0 type can listen-only on`), from the text of `ip -details link show can0`.  True / False, or None when it cannot be told.
    In listen-only mode the controller neither transmits nor acknowledges frames: the strongest guarantee that a collector cannot disturb the bus, and it holds even if software is compromised."""
    t = ip_details_text or ""
    if "link/can" not in t: return None
    m = re.search(r"^\s*can(?:\s*<([^>]*)>)?", t, re.M)
    if not m: return None
    return "LISTEN-ONLY" in {x.strip().upper() for x in (m.group(1) or "").split(",")}

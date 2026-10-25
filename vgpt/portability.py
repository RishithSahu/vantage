"""Does the product run on any operating system?  Read from the source, not promised.

The case study names the car operating systems the assistant lives among: Android (Cuttlefish, Vehicle HAL), QNX, and Linux boards (Raspberry Pi, Graviton, QEMU guests); the application itself is Python plus llama.cpp plus one local server, so what can
break it on another system is the use of a call, a module or a path that exists on one system only (a Windows call, a POSIX one that QNX or Android lack, a Linux-kernel one that QNX lacks).  `scan()` reads every module of the product and lists such uses; a use is accepted when it is
  - inside an `if` that tests the platform (`os.name`, `sys.platform`, `platform.system()`, `hasattr(...)`, `shutil.which(...)`, `os.path.exists(...)`), or
  - inside a `try` that catches ImportError / AttributeError / OSError / NotImplementedError / Exception, or
  - in a module that is documented as belonging to one system (`ONE_SYSTEM` below: the Windows test harness, the Linux-only SocketCAN collector).
Everything else is a finding, and `tests/test_portability.py` fails on a finding.  What this does NOT show is that the product was run on those systems: `docs/PLATFORMS.md` says which were run."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOWS_ONLY = {"windll", "WinDLL", "msvcrt", "winreg", "_winapi", "startfile", "STARTUPINFO", "CREATE_NEW_CONSOLE", "CREATE_NO_WINDOW", "DETACHED_PROCESS", "O_BINARY", "WinError", "wintypes", "HKEY_CURRENT_USER"}
POSIX_ONLY = {"fork", "forkpty", "getuid", "geteuid", "getgid", "setuid", "chown", "fchmod", "pwd", "grp", "fcntl", "termios", "resource", "SIGKILL", "SIGHUP", "SIGALRM", "SIGUSR1", "alarm", "AF_UNIX", "AF_CAN", "O_NONBLOCK",
              "mkfifo", "symlink", "getloadavg", "sched_setaffinity", "geteuid"}
LINUX_ONLY = {"epoll", "EPOLLIN", "AF_NETLINK", "AF_PACKET", "memfd_create", "SO_BINDTODEVICE", "pidfd_open", "eventfd", "inotify_init", "splice", "sched_getaffinity", "CLONE_NEWNS", "unshare", "setns"}       # in the Linux kernel, not in QNX or the Android NDK sysroot
HARD_PATHS = ("/dev/shm", "/proc/", "/sys/", "/etc/", "/tmp/", "/var/", "/home/", "C:\\", "c:\\", "%APPDATA%", "%LOCALAPPDATA%")
ONE_SYSTEM = {"vgpt/governor.py": "Windows job objects: the test harness that simulates the 2 GB / 2-core target on a Windows laptop; the product never imports it",
              "vgpt/live/collect.py": "SocketCAN is Linux only; its class checks `hasattr(socket, 'AF_CAN')` and refuses elsewhere"}
MODULES = {"msvcrt", "winreg", "_winapi", "wintypes", "pwd", "grp", "fcntl", "termios", "resource"}           # one system's MODULES count only when imported (a variable called `grp` or `resource` is not the module)
TABLES = {"WINDOWS", "MAC", "LINUX"}                                                                        # module-level lists of where each system keeps the browser: they are only read after the platform is tested
GUARDS = ("os.name", "sys.platform", "platform.system", "hasattr", "which(", "path.exists", "path.isdir", "path.isfile", "isatty", "WINDOWS", "MAC", "LINUX")
CATCH = ("ImportError", "AttributeError", "OSError", "NotImplementedError", "Exception", "ModuleNotFoundError", "BaseException")


def _src(node, text): return ast.get_source_segment(text, node) or ""


def _guarded(stack, text):
    for n in stack:
        if isinstance(n, ast.If) and any(g in _src(n.test, text) for g in GUARDS): return True
        if isinstance(n, ast.IfExp) and any(g in _src(n.test, text) for g in GUARDS): return True
        if isinstance(n, ast.Try):
            for h in n.handlers:
                names = [] if h.type is None else [_src(h.type, text)]
                if h.type is None or any(c in names[0] for c in CATCH): return True
        if isinstance(n, ast.BoolOp) and any(g in _src(n, text) for g in GUARDS): return True
    return False


def scan_file(path, rel=None):
    with open(path, encoding="utf-8") as fh: text = fh.read()
    tree = ast.parse(text, path); out = []

    def visit(node, stack):
        names = []
        if isinstance(node, ast.Attribute): names.append(node.attr)
        elif isinstance(node, ast.Name): names.append(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)): names += [a.name.split(".")[-1] for a in node.names] + ([node.module.split(".")[-1]] if isinstance(node, ast.ImportFrom) and node.module else []) + ([a.name.split(".")[0] for a in node.names])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and any(node.value.startswith(h) for h in HARD_PATHS) and "\n" not in node.value and len(node.value) < 120:
            table = any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in TABLES for t in n.targets) for n in stack)
            if not table and not _guarded(stack, text) and not isinstance(stack[-1] if stack else None, ast.Expr): out.append((node.lineno, "path", node.value))
        is_import = isinstance(node, (ast.Import, ast.ImportFrom))
        for nm in names:
            if nm in MODULES and not is_import: continue
            kind = "windows" if nm in WINDOWS_ONLY else "posix" if nm in POSIX_ONLY else "linux" if nm in LINUX_ONLY else None
            if kind and not _guarded(stack, text): out.append((node.lineno, kind, nm))
        for child in ast.iter_child_nodes(node): visit(child, stack + [node])
    visit(tree, []); return out


def scan(root=None):
    """{relative path: [(line, kind, what), ...]} for every product module with an unguarded system-specific use."""
    root = root or ROOT; res = {}
    for base in ("vgpt", "ui"):
        for d, _dirs, files in os.walk(os.path.join(root, base)):
            if "__pycache__" in d: continue
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(d, f); rel = os.path.relpath(p, root).replace(os.sep, "/")
                    if rel in ONE_SYSTEM or rel == "vgpt/portability.py": continue
                    hits = scan_file(p, rel)
                    if hits: res[rel] = hits
    if os.path.exists(os.path.join(root, "vehiclegpt.py")):
        h = scan_file(os.path.join(root, "vehiclegpt.py"))
        if h: res["vehiclegpt.py"] = h
    return res

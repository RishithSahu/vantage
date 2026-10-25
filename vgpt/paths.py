r"""Where VehicleGPT keeps its own files (logs, feedback). Never inside the install folder, which may be read-only (Program Files, a shared drive, a zip extracted by IT).
VEHICLEGPT_HOME overrides; otherwise %LOCALAPPDATA%\VehicleGPT on Windows, ~/Library/Application Support/VehicleGPT on macOS, $XDG_DATA_HOME or ~/.local/share/VehicleGPT elsewhere."""
import os, sys, tempfile


def _private_dir(path):
    """Create and probe a directory without touching a predictable user file."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    if os.name != "nt": os.chmod(path, 0o700)
    fd, probe = tempfile.mkstemp(prefix=".write-check-", dir=path)
    os.close(fd)
    os.unlink(probe)


def home():
    p = os.environ.get("VEHICLEGPT_HOME")
    if not p:
        if sys.platform == "win32": p = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local"), "VehicleGPT")
        elif sys.platform == "darwin": p = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "VehicleGPT")
        else: p = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share"), "VehicleGPT")
    try:
        _private_dir(p)
    except OSError:                                                                          # not writable: fall back to a temp folder rather than crash
        # A per-user, non-predictable fallback avoids exposing feedback/logs
        # through a shared /tmp/VehicleGPT directory.
        p = os.path.join(tempfile.gettempdir(), f"VehicleGPT-{os.getuid() if hasattr(os, 'getuid') else os.environ.get('USERNAME', 'user')}")
        _private_dir(p)
    return p


def log_dir():
    p = os.path.join(home(), "logs"); os.makedirs(p, exist_ok=True); return p


def feedback_file(): return os.path.join(home(), "feedback.jsonl")

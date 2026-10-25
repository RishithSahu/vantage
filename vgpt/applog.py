"""Logging for the shipped app: a rotating file in the user's VehicleGPT folder (1 MB x 3) plus the console. Questions are NOT logged unless VGPT_LOG_QUESTIONS=1 (privacy by default);
errors, timings and start-up facts are. Every unexpected server error gets a short id that is shown to the user and written next to the traceback, so a complaint can be traced."""
import logging, logging.handlers, os, sys, uuid
from . import paths

_DONE = False


def setup(console=True, level=logging.INFO):
    global _DONE
    log = logging.getLogger("vehiclegpt")
    if _DONE: return log
    log.setLevel(level); log.propagate = False
    try:
        fh = logging.handlers.RotatingFileHandler(os.path.join(paths.log_dir(), "vehiclegpt.log"), maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); log.addHandler(fh)
    except OSError: pass
    if console:
        ch = logging.StreamHandler(sys.stderr); ch.setLevel(logging.INFO if os.environ.get("VGPT_VERBOSE") else logging.WARNING); ch.setFormatter(logging.Formatter("%(levelname)s %(message)s")); log.addHandler(ch)
    _DONE = True; return log


def get(): return logging.getLogger("vehiclegpt")


def error_id(): return uuid.uuid4().hex[:8]


def log_questions(): return os.environ.get("VGPT_LOG_QUESTIONS") == "1"

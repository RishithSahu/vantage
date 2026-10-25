"""Records a person chose to leave on this computer - a thumbs up / down on an answer, a line of a study the person agreed to - kept small, private, encrypted where the platform can, and not for ever.

* minimised   the caller keeps only what the record is for (the question is cut short; no client details, no answer text beyond a short brief)
* encrypted   each record is sealed with AES-256-GCM under a key derived from the installation's own secret (`vgpt/securestore.py`), so the file is unreadable without the installation; where the `cryptography` package is not
              available the file is written in plain text inside the private VehicleGPT folder and `status()["encrypted"]` says so (set VGPT_RECORDS_PLAIN=1 to force that)
* retained    records older than the retention period (30 days; VGPT_RECORD_RETENTION_DAYS) are removed, on every append at most once an hour and whenever the file is read
* deletable   `delete_all()` removes every record and every rotated file

A record is one line: either the JSON object itself (plain) or {"v": 1, "n": nonce, "c": ciphertext} (sealed).  A plain line written by an earlier version is still read."""
import base64
import hashlib
import hmac
import json
import os
import threading
import time

from . import paths

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAVE_AES = True
except Exception:                                                                    # optional on an embedded target
    AESGCM = None; HAVE_AES = False

DAY = 86400


def _key():
    from . import securestore
    return hmac.new(securestore.secret(), b"vantage-records-v1", hashlib.sha256).digest()


class RecordStore:
    def __init__(self, path, retention_days=None, max_bytes=5_000_000, clock=time.time, plain=None):
        self.path = path; self.max_bytes = int(max_bytes); self.clock = clock; self.lock = threading.Lock(); self._last_purge = 0.0
        try: self.retention_days = float(retention_days if retention_days is not None else os.environ.get("VGPT_RECORD_RETENTION_DAYS", "30"))
        except ValueError: self.retention_days = 30.0
        self.plain = bool(plain) if plain is not None else (os.environ.get("VGPT_RECORDS_PLAIN") == "1" or not HAVE_AES)

    # ---- one line in, one record out
    def _seal(self, rec):
        raw = json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self.plain: return raw.decode("utf-8")
        n = os.urandom(12); ct = AESGCM(_key()).encrypt(n, raw, b"vantage-record")
        return json.dumps({"v": 1, "n": base64.b64encode(n).decode(), "c": base64.b64encode(ct).decode()}, separators=(",", ":"))

    def _open(self, line):
        try: j = json.loads(line)
        except ValueError: return None
        if not isinstance(j, dict): return None
        if j.get("v") == 1 and "c" in j and "n" in j:
            if not HAVE_AES: return None
            try: return json.loads(AESGCM(_key()).decrypt(base64.b64decode(j["n"]), base64.b64decode(j["c"]), b"vantage-record").decode("utf-8"))
            except Exception: return None
        return j

    def _files(self): return [f for f in (self.path + ".1", self.path) if os.path.exists(f)]

    # ---- the four operations
    def append(self, rec):
        """Add one record (a dict); the time is added when the record has none.  -> True when written."""
        rec = dict(rec); rec.setdefault("t", int(self.clock()))
        with self.lock:
            try:
                if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes: os.replace(self.path, self.path + ".1")
                with open(self.path, "a", encoding="utf-8") as f: f.write(self._seal(rec) + "\n")
            except OSError: return False
            if self.clock() - self._last_purge > 3600: self._purge_locked()
        return True

    def read_all(self):
        """Every record still within the retention period, oldest first (expired ones are removed first)."""
        with self.lock:
            self._purge_locked(); out = []
            for fn in self._files():
                try:
                    for line in open(fn, encoding="utf-8"):
                        r = self._open(line.strip())
                        if r is not None: out.append(r)
                except OSError: pass
        return out

    def purge(self):
        """Remove the records older than the retention period.  -> how many were removed."""
        with self.lock: return self._purge_locked()

    def _purge_locked(self):
        self._last_purge = self.clock(); cut = self.clock() - self.retention_days * DAY; removed = 0
        for fn in self._files():
            keep = []
            try:
                for line in open(fn, encoding="utf-8"):
                    line = line.strip()
                    if not line: continue
                    r = self._open(line)
                    if r is None: removed += 1; continue                                                 # a line nobody can read is no record
                    if float(r.get("t", 0) or 0) >= cut: keep.append(line)
                    else: removed += 1
                tmp = fn + ".tmp"
                if len(keep) == 0:
                    os.remove(fn)
                else:
                    with open(tmp, "w", encoding="utf-8") as f: f.write("\n".join(keep) + "\n")
                    os.replace(tmp, fn)
            except OSError: pass
        return removed

    def delete_all(self):
        """Remove every record (the person's right to be forgotten).  -> the number of files removed."""
        n = 0
        with self.lock:
            for fn in (self.path, self.path + ".1", self.path + ".tmp"):
                try: os.remove(fn); n += 1
                except OSError: pass
        return n

    def status(self):
        return {"encrypted": not self.plain, "retention_days": self.retention_days, "count": len(self.read_all()), "path": os.path.basename(self.path)}


def feedback_store(): return RecordStore(paths.feedback_file())
def study_store(): return RecordStore(os.path.join(paths.home(), "interaction_log.jsonl"), retention_days=float(os.environ.get("VGPT_STUDY_RETENTION_DAYS", "14")))


def read_file(path):
    """The records of a store at `path` (both files), for tools that only read: decrypts, ignores bad lines, applies no retention."""
    s = RecordStore(path); out = []
    for fn in s._files():
        try:
            for line in open(fn, encoding="utf-8"):
                r = s._open(line.strip())
                if r is not None: out.append(r)
        except OSError: pass
    return out

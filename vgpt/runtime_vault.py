"""The vehicle's database as the running product holds it: encrypted on the storage, in clear only in RAM, wiped when the process ends.

`vgpt/atrest.py` is the cipher (AES-256-GCM, chunked, authenticated).  This module is how the product uses it:

    python -m vgpt.runtime_vault seal   data/world_aster.db  --key <source>     plaintext -> data/world_aster.db.enc, then the plaintext is overwritten and removed
    python -m vgpt.runtime_vault unseal data/world_aster.db.enc out.db --key <source>

    open_database(path, key_spec) -> a database file the product can open
        path.enc exists      decrypted into a private RAM-backed folder (mode 0700; /dev/shm where there is one); the storage never sees the clear file
        only path exists     (no key configured, or a test build) the plain file is used as it is - nothing changes for a build without a key
    ProtectedStore           the same for a database that CHANGES (a linked car's history): the clear copy is in RAM, an encrypted snapshot goes to the storage every `interval` seconds when the data changed and
                             once more when the process stops; the clear copy is overwritten and removed at exit (`wipe`).

The key source is anything `atrest.load_key` understands (a file, env:NAME, credential:NAME for a systemd credential sealed to the TPM, tpm2:HANDLE, keyring:NAME, cmd:...), and is named by `VGPT_ATREST_KEY`.
A process that is already compromised while running can read its own memory; what this closes is everything else: a copied SD card or disk image, a stolen device, a backup, a crash dump (see hardening.py), a
temp folder left behind.  The clear copy is never written to the storage."""
import atexit
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
import threading
import time

from . import atrest


def _shm():
    """The RAM-backed folder, where the system has one (Linux), else None."""
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK): return "/dev/shm"
    return None


def _alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except SystemError: return False                                                                     # Windows: os.kill on a pid that is gone can raise SystemError (WinError 87) instead of OSError
    except OSError as e: return getattr(e, "winerror", None) != 87                                       # Windows: 87 = no such process
    return True


def sweep_stale(prefixes=("vehiclegpt_", "vantage-")):
    """Remove the working folders of processes that no longer exist (a crash, a SIGKILL, a power cut leaves them behind).  A folder is named <prefix><pid>_<random>."""
    gone = 0
    for base in {_shm() or tempfile.gettempdir(), tempfile.gettempdir()}:
        try: names = os.listdir(base)
        except OSError: continue
        for n in names:
            for pre in prefixes:
                if n.startswith(pre):
                    head = n[len(pre):].split("_", 1)[0]
                    if head.isdigit() and int(head) != os.getpid() and not _alive(int(head)): wipe_folder(os.path.join(base, n)); gone += 1
    return gone


def ram_folder(prefix="vantage-"):
    """A private folder whose contents live in RAM where the system has a RAM-backed folder (Linux /dev/shm), else the system temp folder; mode 0700.  Named <prefix><pid>_<random>."""
    base = _shm()
    d = tempfile.mkdtemp(prefix=f"{prefix}{os.getpid()}_", dir=base)
    try: os.chmod(d, 0o700)
    except OSError: pass
    return d


def wipe(path):
    """Overwrite a file with zeros, flush it, remove it.  (On flash storage an overwrite is not a guarantee - which is why the clear copy lives in RAM and not there.)"""
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            block = b"\0" * (1 << 20); left = size
            while left > 0: f.write(block[:min(left, len(block))]); left -= len(block)
            f.flush(); os.fsync(f.fileno())
    except OSError: pass
    try: os.remove(path)
    except OSError: pass


def wipe_folder(d):
    for root, _, files in os.walk(d, topdown=False):
        for n in files: wipe(os.path.join(root, n))
        try: os.rmdir(root)
        except OSError: pass
    shutil.rmtree(d, ignore_errors=True)


def key_spec(explicit=None):
    return explicit or os.environ.get("VGPT_ATREST_KEY") or None


def seal(path, spec, remove_plain=True):
    """data.db -> data.db.enc.  Returns the encrypted path."""
    key = atrest.load_key(spec); enc = path + ".enc"
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if c.execute("PRAGMA quick_check").fetchone()[0] != "ok": raise atrest.VaultError("the database is damaged: not sealing it")
    finally: c.close()
    work = ram_folder()
    try: atrest.Vault(key, work).snapshot(path, enc)
    finally: wipe_folder(work)
    check = os.path.join(ram_folder(), "check.db")
    try: atrest.decrypt_file(enc, check, key)                                                  # it must come back before the plain file goes
    finally: wipe_folder(os.path.dirname(check))
    if remove_plain:
        for suffix in ("", "-wal", "-shm", "-journal"):
            if os.path.exists(path + suffix): wipe(path + suffix)
    return enc


def open_database(path, spec=None, folder=None):
    """-> (path of a database file to open, a callable that removes the clear copy).  Without an encrypted file next to `path` the plain path is returned and the callable does nothing."""
    enc = path + ".enc"; spec = key_spec(spec)
    if not os.path.exists(enc): return path, (lambda: None)
    if not spec: raise atrest.VaultError("%s is encrypted and no key is configured (VGPT_ATREST_KEY)" % os.path.basename(enc))
    d = folder or ram_folder(); out = os.path.join(d, os.path.basename(path))
    atrest.decrypt_file(enc, out, atrest.load_key(spec)); os.chmod(out, 0o600)
    gone = []

    def cleanup():
        if not gone: gone.append(1); wipe_folder(d) if folder is None else wipe(out)
    atexit.register(cleanup)
    return out, cleanup


class ProtectedStore:
    """A database that changes while the product runs, kept as `<path>.enc` on the storage and as a clear file only in RAM."""

    def __init__(self, path, spec=None, interval=120.0):
        self.path = path; self.enc = path + ".enc"; self.spec = key_spec(spec); self.interval = float(interval)
        if not self.spec: raise atrest.VaultError("no key configured (VGPT_ATREST_KEY)")
        self.key = atrest.load_key(self.spec); self.dir = ram_folder(); self.work = os.path.join(self.dir, os.path.basename(path))
        self._stop = threading.Event(); self._thread = None; self._last = None; self.snapshots = 0
        if os.path.exists(self.enc): atrest.decrypt_file(self.enc, self.work, self.key); os.chmod(self.work, 0o600)
        elif os.path.exists(path): shutil.copyfile(path, self.work); os.chmod(self.work, 0o600)
        atexit.register(self.close)

    def _version(self):
        try:
            c = sqlite3.connect(f"file:{self.work}?mode=ro", uri=True, timeout=5)
            try: return c.execute("PRAGMA data_version").fetchone()[0], os.path.getmtime(self.work), os.path.getsize(self.work)
            finally: c.close()
        except (sqlite3.Error, OSError): return None

    def snapshot(self, force=False):
        v = self._version()
        if not force and v is not None and v == self._last: return False
        if not os.path.exists(self.work): return False
        atrest.Vault(self.key, self.dir).snapshot(self.work, self.enc); self._last = v; self.snapshots += 1
        if os.path.exists(self.path) and self.path != self.enc: wipe(self.path)                         # the plain original does not stay beside the encrypted file
        return True

    def start(self):
        def loop():
            while not self._stop.wait(self.interval):
                try: self.snapshot()
                except Exception: pass
        self._thread = threading.Thread(target=loop, daemon=True, name="vault-snapshot"); self._thread.start(); return self

    def close(self):
        if self._stop.is_set(): return
        self._stop.set()
        try: self.snapshot(force=True)
        except Exception: pass
        wipe_folder(self.dir)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m vgpt.runtime_vault"); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("seal"); a.add_argument("db"); a.add_argument("--key", required=True); a.add_argument("--keep-plain", action="store_true")
    b = sub.add_parser("unseal"); b.add_argument("enc"); b.add_argument("out"); b.add_argument("--key", required=True)
    c = sub.add_parser("make-key"); c.add_argument("path")
    n = ap.parse_args(argv)
    if n.cmd == "seal": print(seal(n.db, n.key, remove_plain=not n.keep_plain))
    elif n.cmd == "unseal": atrest.decrypt_file(n.enc, n.out, atrest.load_key(n.key)); print(n.out)
    else: print(atrest.make_key(n.path))


if __name__ == "__main__":
    main()

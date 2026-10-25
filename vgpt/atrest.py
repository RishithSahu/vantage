"""Encryption of the vehicle's data AT REST.

Vantage works on a database in a RAM-backed folder (`/dev/shm/...` on the Raspberry Pi, a temp folder elsewhere) and keeps on disk only an ENCRYPTED snapshot of it (AES-256-GCM, written every few minutes and
when the service stops).  Pulling the power, copying the SD card or reading the data folder therefore yields ciphertext; at most the last few minutes of data are lost.  What lives only in memory while
Vantage runs is what any running assistant must have.

    encrypt_file(src, dst, key) / decrypt_file(src, dst, key)    chunked AES-GCM; each chunk authenticates its position and whether it is the last, so truncation, reordering and tampering are detected
    Vault(key)                                                    .snapshot(db_path, enc_path)  an online SQLite backup, encrypted;  .restore(enc_path, db_path)

The key is 256 bits.  WHERE it lives is the integrator's decision, and a file next to the data defeats the purpose, so `--atrest-key` takes a SOURCE:

    <path>                a file holding 64 hex digits (mode 0600) - the test setup
    env:NAME              an environment variable holding the 64 hex digits (a service manager or a container secret puts it there)
    credential:NAME       a systemd credential: the file $CREDENTIALS_DIRECTORY/NAME.  With `LoadCredentialEncrypted=` in the unit and the credential made by `systemd-creds encrypt --with-key=tpm2`, the key is
                          unsealed by the TPM only on this machine in this boot state and never exists on the disk in the clear
    tpm2:HANDLE           the key sealed in a TPM 2.0 persistent object, read with `tpm2_unseal -c HANDLE`
    keyring:NAME          the Linux kernel keyring (a user key called NAME, read with `keyctl`)
    cmd:COMMAND ...       any command that prints the 64 hex digits (a secure element's tool, `secret-tool lookup ...`)

Only the file source is exercised with real storage in this repository's tests; the others are tested with a stand-in command and directory (the TPM, the keyring and systemd-creds themselves were not available).
This module protects the data from anyone who gets the storage but not the key; it does not protect a running system from root."""
import os
import secrets
import sqlite3
import struct
import tempfile

MAGIC = b"VGPTENC1"
CHUNK = 1 << 20
TAG = 16


class VaultError(Exception): pass


def make_key(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f: f.write(secrets.token_hex(32) + "\n")
    return path


def _parse(text, what):
    try: k = bytes.fromhex("".join(str(text).split()))
    except ValueError as e: raise VaultError(f"{what} does not hold a key written as hex digits") from e
    if len(k) != 32: raise VaultError(f"an at-rest key is 32 bytes (64 hex digits); {what} gave {len(k)}")
    return k


def _run(argv, what, timeout=15):
    import subprocess
    try: p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e: raise VaultError(f"could not run {argv[0]} for {what}: {type(e).__name__}") from e
    if p.returncode: raise VaultError(f"{argv[0]} failed for {what} (exit {p.returncode})")
    return p.stdout


def load_key(spec, create=False):
    """The 32-byte key from its source (see the top of this file).  Only a file source can be created on the spot (`create`)."""
    spec = str(spec)
    if spec.startswith("env:"):
        v = os.environ.get(spec[4:])
        if not v: raise VaultError(f"the environment variable {spec[4:]} is not set")
        return _parse(v, "the environment variable " + spec[4:])
    if spec.startswith("credential:"):
        d = os.environ.get("CREDENTIALS_DIRECTORY"); name = os.path.basename(spec[11:])
        if not d: raise VaultError("CREDENTIALS_DIRECTORY is not set: the service has no systemd credentials (LoadCredentialEncrypted= in the unit)")
        try: return _parse(open(os.path.join(d, name), encoding="ascii").read(), "the credential " + name)
        except OSError as e: raise VaultError(f"no credential {name}") from e
    if spec.startswith("tpm2:"): return _parse(_run(["tpm2_unseal", "-c", spec[5:]], "the TPM object " + spec[5:]), "the TPM object " + spec[5:])
    if spec.startswith("keyring:"):
        import subprocess
        kid = _run(["keyctl", "request", "user", spec[8:]], "the keyring entry " + spec[8:]).strip(); return _parse(_run(["keyctl", "pipe", kid], "the keyring entry " + spec[8:]), "the keyring entry " + spec[8:])
    if spec.startswith("cmd:"):
        import shlex
        argv = [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in shlex.split(spec[4:], posix=os.name != "nt")]                    # on Windows the splitter keeps the quotes around a token: take them off
        return _parse(_run(argv, "the key command"), "the key command")
    path = spec
    if not os.path.exists(path):
        if not create: raise VaultError(f"no at-rest key at {path}")
        make_key(path)
    try: k = bytes.fromhex(open(path, encoding="ascii").read().strip())
    except (OSError, ValueError) as e: raise VaultError(f"{path} is not a key file") from e
    if len(k) != 32: raise VaultError("an at-rest key is 32 bytes (64 hex digits)")
    return k


def _aes(key):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key)


def encrypt_file(src, dst, key, chunk=CHUNK):
    """src -> dst (written beside it and renamed, so a crash never leaves a half file).  Layout: MAGIC, file id (16), then per chunk: nonce (12), length (4), ciphertext + tag."""
    aes = _aes(key); fid = secrets.token_bytes(16); tmp = dst + ".tmp"; size = os.path.getsize(src)
    with open(src, "rb") as f, open(tmp, "wb") as o:
        o.write(MAGIC + fid); i = 0; done = 0
        while True:
            block = f.read(chunk); done += len(block); last = done >= size; nonce = secrets.token_bytes(12)
            ct = aes.encrypt(nonce, block, fid + struct.pack("<QB", i, 1 if last else 0)); o.write(nonce + struct.pack("<I", len(ct)) + ct); i += 1
            if last: break
        o.flush(); os.fsync(o.fileno())
    os.replace(tmp, dst)


def decrypt_file(src, dst, key):
    from cryptography.exceptions import InvalidTag
    aes = _aes(key); tmp = dst + ".tmp"
    with open(src, "rb") as f, open(tmp, "wb") as o:
        head = f.read(len(MAGIC) + 16)
        if head[:len(MAGIC)] != MAGIC or len(head) != len(MAGIC) + 16: raise VaultError("not an encrypted VehicleGPT file")
        fid = head[len(MAGIC):]; i = 0; last_seen = False
        while True:
            nonce = f.read(12)
            if not nonce: break
            ln = f.read(4)
            if len(nonce) != 12 or len(ln) != 4: raise VaultError("the encrypted file is truncated")
            n = struct.unpack("<I", ln)[0]; ct = f.read(n)
            if len(ct) != n: raise VaultError("the encrypted file is truncated")
            peek = f.peek(1)[:1] if hasattr(f, "peek") else b""
            last = not peek
            try: o.write(aes.decrypt(nonce, ct, fid + struct.pack("<QB", i, 1 if last else 0)))
            except InvalidTag as e: raise VaultError("the encrypted file does not authenticate: wrong key, or it was changed or cut") from e
            i += 1; last_seen = last
        if not last_seen: raise VaultError("the encrypted file ends early")
        o.flush(); os.fsync(o.fileno())
    os.replace(tmp, dst)


class Vault:
    def __init__(self, key, work_dir=None):
        self.key = key; self.work = work_dir

    def snapshot(self, db_path, enc_path):
        """An online backup of a live database (consistent, readers and the writer carry on), encrypted into `enc_path`.  The clear backup exists only in the work folder, for the time it takes."""
        d = self.work or os.path.dirname(db_path); os.makedirs(d, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix="snap-", suffix=".db", dir=d); os.close(fd)
        try:
            src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30); dst = sqlite3.connect(tmp)
            try: src.backup(dst, pages=2048, sleep=0.01)
            finally: dst.close(); src.close()
            encrypt_file(tmp, enc_path, self.key)
        finally:
            try: os.remove(tmp)
            except OSError: pass

    def restore(self, enc_path, db_path):
        decrypt_file(enc_path, db_path, self.key)
        c = sqlite3.connect(db_path)
        try:
            if c.execute("PRAGMA quick_check").fetchone()[0] != "ok": raise VaultError("the restored database is damaged")
        finally: c.close()

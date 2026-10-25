"""Signing and encrypting what leaves the head unit.

An evidence pack (a question, its answer, the data behind it, the versions that produced it) is the only thing the application ever hands to another person.  Two protections:

  * `sign` - an HMAC-SHA256 over the canonical JSON with a secret that is created once per installation and stays in the private VehicleGPT folder: anyone holding the same installation can prove the pack
    was not changed afterwards;
  * `seal` / `unseal` - AES-256-GCM with a key derived from a passphrase by scrypt, so a pack can be sent or stored without its contents being readable.  Needs the `cryptography` package; when it is missing
    `HAVE_AES` is False and the page says plainly that the pack is signed but not encrypted.

Not covered (and said so on the privacy page): the vehicle database itself is protected by the operating system's file permissions and, where available, disk encryption - not by this module."""
import hashlib
import hmac
import json
import os
import secrets

from . import paths

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAVE_AES = True
except Exception:                                                                    # the package is optional on an embedded target
    AESGCM = None; HAVE_AES = False

MAGIC = b"VGPT1"


def _secret_path(): return os.path.join(paths.home(), "secret.key")


def secret():
    p = _secret_path()
    try:
        with open(p, "rb") as f:
            k = f.read()
        if len(k) == 32: return k
    except OSError: pass
    k = secrets.token_bytes(32)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f: f.write(k)
    return k


def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
def sign(obj): return hmac.new(secret(), canonical(obj), hashlib.sha256).hexdigest()
def verify(obj, signature): return hmac.compare_digest(sign(obj), str(signature or ""))


def _key(passphrase, salt): return hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=2 ** 15, r=8, p=1, dklen=32, maxmem=2 ** 27)


def seal(data, passphrase):
    """bytes -> bytes: MAGIC + salt + nonce + AES-256-GCM(data).  Raises RuntimeError when the cryptography package is missing."""
    if not HAVE_AES: raise RuntimeError("encryption needs the 'cryptography' package")
    if len(str(passphrase or "")) < 8: raise ValueError("the passphrase needs at least 8 characters")
    salt, nonce = secrets.token_bytes(16), secrets.token_bytes(12)
    return MAGIC + salt + nonce + AESGCM(_key(passphrase, salt)).encrypt(nonce, data, MAGIC)


def unseal(blob, passphrase):
    if not HAVE_AES: raise RuntimeError("decryption needs the 'cryptography' package")
    if blob[:5] != MAGIC or len(blob) < 5 + 16 + 12 + 16: raise ValueError("not a sealed evidence pack")
    salt, nonce, ct = blob[5:21], blob[21:33], blob[33:]
    try: return AESGCM(_key(passphrase, salt)).decrypt(nonce, ct, MAGIC)
    except Exception: raise ValueError("wrong passphrase, or the pack was changed")

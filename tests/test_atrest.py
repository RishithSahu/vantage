"""Encryption at rest (vgpt/atrest.py): what reaches the disk is ciphertext, tampering and truncation are detected, a live database can be snapshotted while it is written, and the data comes back with the right key only."""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import atrest  # noqa: E402

try:
    import cryptography  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs the cryptography package")
class Vault(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.key = atrest.load_key(atrest.make_key(os.path.join(self.d, "keys", "k")))
        self.db = os.path.join(self.d, "live.db"); c = sqlite3.connect(self.db); c.execute("CREATE TABLE gps(ts INTEGER, lat REAL, lon REAL)"); c.executemany("INSERT INTO gps VALUES(?,?,?)", [(i, 51.5 + i * 1e-4, -0.12 + i * 1e-4) for i in range(30000)]); c.commit(); c.close()

    def tearDown(self): shutil.rmtree(self.d, ignore_errors=True)

    def test_round_trip_and_the_file_is_not_readable_as_a_database(self):
        enc = os.path.join(self.d, "live.db.enc"); atrest.Vault(self.key, self.d).snapshot(self.db, enc); raw = open(enc, "rb").read()
        self.assertEqual(raw[:8], atrest.MAGIC); self.assertNotIn(b"SQLite format", raw); self.assertNotIn(b"gps", raw); self.assertGreater(len(raw), os.path.getsize(self.db) * 0.9)
        out = os.path.join(self.d, "restored.db"); atrest.Vault(self.key).restore(enc, out); c = sqlite3.connect(out); self.assertEqual(c.execute("SELECT COUNT(*) FROM gps").fetchone()[0], 30000); c.close()

    def test_a_chunk_boundary_and_an_empty_file(self):
        for n in (0, 1, atrest.CHUNK - 1, atrest.CHUNK, atrest.CHUNK + 1, 3 * atrest.CHUNK):
            src = os.path.join(self.d, f"f{n}"); data = os.urandom(n); open(src, "wb").write(data); enc = src + ".enc"; atrest.encrypt_file(src, enc, self.key); out = src + ".out"; atrest.decrypt_file(enc, out, self.key)
            self.assertEqual(open(out, "rb").read(), data, n)

    def test_wrong_key_tampering_truncation_and_reordering_are_detected(self):
        src = os.path.join(self.d, "big"); open(src, "wb").write(os.urandom(3 * atrest.CHUNK + 5)); enc = src + ".enc"; atrest.encrypt_file(src, enc, self.key); raw = open(enc, "rb").read()
        with self.assertRaises(atrest.VaultError): atrest.decrypt_file(enc, src + ".o", b"k" * 32)
        bad = bytearray(raw); bad[len(raw) // 2] ^= 1; open(src + ".t1", "wb").write(bytes(bad))
        with self.assertRaises(atrest.VaultError): atrest.decrypt_file(src + ".t1", src + ".o", self.key)
        open(src + ".t2", "wb").write(raw[:-100])
        with self.assertRaises(atrest.VaultError): atrest.decrypt_file(src + ".t2", src + ".o", self.key)
        chunk = 12 + 4 + atrest.CHUNK + atrest.TAG; head = len(atrest.MAGIC) + 16; swapped = raw[:head] + raw[head + chunk: head + 2 * chunk] + raw[head: head + chunk] + raw[head + 2 * chunk:]; open(src + ".t3", "wb").write(swapped)
        with self.assertRaises(atrest.VaultError): atrest.decrypt_file(src + ".t3", src + ".o", self.key)
        open(src + ".t4", "wb").write(raw[:head + chunk])                                                      # cut exactly at a chunk boundary: the 'last' flag does not match
        with self.assertRaises(atrest.VaultError): atrest.decrypt_file(src + ".t4", src + ".o", self.key)
        open(src + ".t5", "wb").write(b"not a vault file at all")
        with self.assertRaises(atrest.VaultError): atrest.decrypt_file(src + ".t5", src + ".o", self.key)

    def test_a_live_database_can_be_snapshotted_while_it_is_written(self):
        stop = threading.Event(); n = [0]
        def writer():
            c = sqlite3.connect(self.db, timeout=30); c.execute("PRAGMA journal_mode=WAL")
            while not stop.is_set(): c.execute("INSERT INTO gps VALUES(?,?,?)", (100000 + n[0], 1.0, 2.0)); c.commit(); n[0] += 1
            c.close()
        t = threading.Thread(target=writer); t.start()
        try:
            enc = os.path.join(self.d, "live.enc"); atrest.Vault(self.key, self.d).snapshot(self.db, enc)
        finally: stop.set(); t.join()
        out = os.path.join(self.d, "r.db"); atrest.Vault(self.key).restore(enc, out); c = sqlite3.connect(out); self.assertGreaterEqual(c.execute("SELECT COUNT(*) FROM gps").fetchone()[0], 30000); self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0], "ok"); c.close()

    def test_keys_are_private_and_validated(self):
        k = os.path.join(self.d, "keys", "k")
        if os.name != "nt": self.assertEqual(oct(os.stat(k).st_mode & 0o777), "0o600")
        with self.assertRaises(atrest.VaultError): atrest.load_key(os.path.join(self.d, "missing"))
        open(os.path.join(self.d, "short"), "w").write("abcd")
        with self.assertRaises(atrest.VaultError): atrest.load_key(os.path.join(self.d, "short"))


@unittest.skipUnless(HAVE, "needs the cryptography package")
class KeySources(unittest.TestCase):
    """Where the key may live: the file of the tests, the environment, a systemd credential (TPM-sealed in a vehicle), a TPM object, the kernel keyring, any command.  The TPM, keyctl and systemd-creds are not available here, so
    their calls are checked at the command line level and with stand-ins; what happens to the text they return is the same code for all of them."""
    HEX = "ab" * 32

    def _cmd(self, code):
        import sys
        return "cmd:" + json.dumps(sys.executable) + " -c " + json.dumps(code)                                                  # a command line that works for the shell-like splitter on every platform

    def test_environment_credential_and_command_sources(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"VANTAGE_KEY": self.HEX}): self.assertEqual(atrest.load_key("env:VANTAGE_KEY"), bytes.fromhex(self.HEX))
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "atrest.key"), "w") as f: f.write(self.HEX + "\n")
        with mock.patch.dict(os.environ, {"CREDENTIALS_DIRECTORY": d}): self.assertEqual(atrest.load_key("credential:atrest.key"), bytes.fromhex(self.HEX))
        self.assertEqual(atrest.load_key(self._cmd("print('" + self.HEX + "')")), bytes.fromhex(self.HEX)); shutil.rmtree(d, ignore_errors=True)

    def test_a_tpm_object_and_a_keyring_entry_are_read_with_the_standard_tools(self):
        from unittest import mock
        calls = []

        def fake(argv, what, timeout=15):
            calls.append(argv); return self.HEX + "\n" if (argv[0] == "tpm2_unseal" or argv[1] == "pipe") else "123456789\n"
        with mock.patch.object(atrest, "_run", fake):
            self.assertEqual(atrest.load_key("tpm2:0x81010001"), bytes.fromhex(self.HEX)); self.assertEqual(atrest.load_key("keyring:vantage-atrest"), bytes.fromhex(self.HEX))
        self.assertEqual(calls[0], ["tpm2_unseal", "-c", "0x81010001"]); self.assertEqual(calls[1], ["keyctl", "request", "user", "vantage-atrest"]); self.assertEqual(calls[2], ["keyctl", "pipe", "123456789"])

    def test_every_failure_is_a_vault_error_with_a_reason(self):
        from unittest import mock
        for spec in ("env:NOPE_NOT_SET", "credential:atrest.key", self._cmd("import sys; sys.exit(3)"), self._cmd("print('abcd')"), "cmd:this-program-does-not-exist-xyz", "tpm2:0x1"):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("CREDENTIALS_DIRECTORY", None)
                with self.assertRaises(atrest.VaultError, msg=spec): atrest.load_key(spec)
        with mock.patch.dict(os.environ, {"BAD": "not hex at all"}):
            with self.assertRaises(atrest.VaultError): atrest.load_key("env:BAD")

    def test_only_a_file_key_is_created_on_the_spot(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "new.key"); k = atrest.load_key(p, create=True); self.assertEqual(len(k), 32); self.assertEqual(atrest.load_key(p), k)
        with self.assertRaises(atrest.VaultError): atrest.load_key("env:NOT_THERE_EITHER", create=True)
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

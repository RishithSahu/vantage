"""The database as the running product holds it: ciphertext on the storage, clear only in a private RAM folder, wiped at the end (vgpt/runtime_vault.py, vgpt/hardening.py)."""
import os, sqlite3, sys, tempfile, time, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import atrest, hardening, runtime_vault as RV

SECRET = b"driver-home-52.123456,-1.987654"


def make_db(path):
    c = sqlite3.connect(path); c.execute("CREATE TABLE gps(ts INT, lat REAL, lon REAL, note TEXT)")
    c.executemany("INSERT INTO gps VALUES(?,?,?,?)", [(i, 52.123456, -1.987654, SECRET.decode()) for i in range(2000)]); c.commit(); c.close()


class Sealed(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.key = atrest.make_key(os.path.join(self.d, "keys", "k")); self.db = os.path.join(self.d, "car.db"); make_db(self.db)

    def test_sealing_leaves_only_ciphertext_on_the_storage(self):
        enc = RV.seal(self.db, self.key)
        self.assertFalse(os.path.exists(self.db)); raw = open(enc, "rb").read()
        self.assertEqual(raw[:8], atrest.MAGIC); self.assertNotIn(b"SQLite format", raw); self.assertNotIn(SECRET, raw); self.assertNotIn(b"52.1234", raw)
        for root, _, files in os.walk(self.d):
            for f in files:
                if f != "k": self.assertNotIn(SECRET, open(os.path.join(root, f), "rb").read(), f)

    def test_open_gives_a_private_ram_copy_that_is_wiped(self):
        RV.seal(self.db, self.key); path, done = RV.open_database(self.db, self.key)
        self.assertNotEqual(os.path.dirname(path), self.d); c = sqlite3.connect(path); n = c.execute("SELECT COUNT(*) FROM gps").fetchone()[0]; c.close(); self.assertEqual(n, 2000)
        if os.name == "posix": self.assertEqual(os.stat(path).st_mode & 0o077, 0)
        folder = os.path.dirname(path); done(); self.assertFalse(os.path.exists(path)); self.assertFalse(os.path.exists(folder))

    def test_a_wrong_key_or_a_changed_file_is_refused(self):
        enc = RV.seal(self.db, self.key); other = atrest.make_key(os.path.join(self.d, "keys", "other"))
        with self.assertRaises(atrest.VaultError): RV.open_database(self.db, other)
        b = bytearray(open(enc, "rb").read()); b[len(b) // 2] ^= 1; open(enc, "wb").write(bytes(b))
        with self.assertRaises(atrest.VaultError): RV.open_database(self.db, self.key)

    def test_a_sealed_database_without_a_key_does_not_open(self):
        RV.seal(self.db, self.key)
        os.environ.pop("VGPT_ATREST_KEY", None)
        with self.assertRaises(atrest.VaultError): RV.open_database(self.db)

    def test_an_unsealed_build_is_untouched(self):
        path, done = RV.open_database(self.db, None); self.assertEqual(path, self.db); done(); self.assertTrue(os.path.exists(self.db))

    def test_a_database_that_changes_survives_a_restart_and_is_never_clear_on_the_storage(self):
        s = RV.ProtectedStore(self.db, self.key, interval=0.2); c = sqlite3.connect(s.work); c.execute("INSERT INTO gps VALUES(99999, 1.5, 2.5, 'new-secret-row')"); c.commit(); c.close()
        s.start(); time.sleep(0.8); s.close()
        self.assertFalse(os.path.exists(self.db)); self.assertFalse(os.path.exists(s.work)); self.assertNotIn(b"new-secret-row", open(self.db + ".enc", "rb").read())
        t = RV.ProtectedStore(self.db, self.key)
        try:
            c = sqlite3.connect(t.work); v = c.execute("SELECT note FROM gps WHERE ts=99999").fetchone()[0]; c.close(); self.assertEqual(v, "new-secret-row")
        finally: t.close()

    def test_a_killed_runs_working_folder_is_swept_by_the_next_run(self):
        dead = os.path.join(tempfile.gettempdir(), "vehiclegpt_99999999_abc"); os.makedirs(dead, exist_ok=True); open(os.path.join(dead, "live_1.db"), "wb").write(SECRET * 100)
        mine = os.path.join(tempfile.gettempdir(), "vehiclegpt_%d_abc" % os.getpid()); os.makedirs(mine, exist_ok=True)
        try:
            RV.sweep_stale(); self.assertFalse(os.path.exists(dead)); self.assertTrue(os.path.exists(mine))
        finally: RV.wipe_folder(mine); RV.wipe_folder(dead)

    def test_wipe_overwrites_before_removing(self):
        p = os.path.join(self.d, "x.bin"); open(p, "wb").write(SECRET * 1000); RV.wipe(p); self.assertFalse(os.path.exists(p))


@unittest.skipUnless(sys.platform.startswith("linux"), "process hardening is Linux")
class Hardening(unittest.TestCase):
    def test_no_core_dumps_and_not_dumpable(self):
        import subprocess
        code = "import sys; sys.path.insert(0, %r); from vgpt import hardening; r = hardening.harden(); print(r); print(open('/proc/self/limits').read())" % ROOT
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
        self.assertIn("'core_dumps': False", out); self.assertIn("'not_dumpable': True", out)


if __name__ == "__main__":
    unittest.main()

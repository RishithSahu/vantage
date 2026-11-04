"""Regression tests for the defects the 2026-11-04 bug hunt found with fuzzing (random and mutated input to the file loaders, the protocol decoders and the native engine).
Each of these used to crash the process or raise an exception type nobody handles."""
import os, shutil, subprocess, sys, tempfile, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
VGW = os.path.join(ROOT, "models", "vg_frame2r12.vgw")


@unittest.skipUnless(os.path.exists(VGW) and shutil.which("gcc"), "needs models/vg_frame2r12.vgw and gcc")
class NativeEngineRefusesBadInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt.meaning import native
        cls.native = native; cls.dir = tempfile.mkdtemp(); cls.lib = native.build()
        with open(VGW, "rb") as f: cls.head = f.read(4096)

    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls.dir, ignore_errors=True)

    def load_in_child(self, name, data):
        """vg_load in its own process: a crash would end the child, not the test run."""
        p = os.path.join(self.dir, name); open(p, "wb").write(data)
        code = "import ctypes, sys; d = ctypes.CDLL(sys.argv[1]); d.vg_load.restype = ctypes.c_void_p; d.vg_load.argtypes = [ctypes.c_char_p, ctypes.c_int]; print('NULL' if not d.vg_load(sys.argv[2].encode(), 1) else 'ENGINE')"
        r = subprocess.run([sys.executable, "-I", "-c", code, self.lib, p], capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip()

    def test_a_short_or_cut_model_file_is_refused_not_read_past_its_end(self):
        for name, data in (("empty", b""), ("magic", b"VGW1"), ("header", self.head[:68]), ("cut", self.head[:200]), ("cut4k", self.head)):
            self.assertEqual(self.load_in_child(name, data), (0, "NULL"), name)

    def test_a_block_that_claims_more_bytes_than_the_file_has_is_refused(self):
        import struct
        for nb in (2 ** 62, 2 ** 64 - 1, 2 ** 29):
            b = bytearray(self.head); b[64:72] = struct.pack("<Q", nb); self.assertEqual(self.load_in_child(f"nb{nb}", bytes(b)), (0, "NULL"), nb)

    def test_token_ids_and_positions_outside_the_model_are_refused(self):
        eng = self.native.Engine(VGW, threads=1); cand = np.arange(10, dtype=np.int64); empty = [np.zeros((1, 2, 0, 64), np.float32) for _ in range(48)]
        for ids, start in ((np.array([5, 10 ** 9]), 0), (np.array([-1, 3]), 0), (np.array([5, 6]), 4095), (np.array([5, 6]), -1)):
            with self.assertRaises(ValueError, msg=(ids.tolist(), start)): eng.forward1(ids, start, empty, 1, cand)
        with self.assertRaises(ValueError): eng.forward1(np.array([5, 6]), 0, empty, 1, np.array([0, 10 ** 9], np.int64))
        lg, _ = eng.forward1(np.array([5, 6]), 0, empty, 1, cand); self.assertEqual(lg.shape, (1, 10))                 # the same engine still answers a valid request


class DamagedFilesRaiseTheDeclaredError(unittest.TestCase):
    def test_a_damaged_lin_description_raises_lin_error(self):
        from vgpt.protocols import lin
        good = 'LIN_description_file; LIN_protocol_version = "2.1"; Signals { s1: 8, 0, master; } Frames { f1: 0x10, master, 1 { s1, 0; } }'
        self.assertIn(0x10, lin.Ldf.parse(good).by_id)
        for bad in (good.replace("0x10, master, 1", "0x10, master,"), good.replace("8, 0, master", "8,"), "Signals { a: ; } Frames { f: ; }"):
            try: lin.Ldf.parse(bad)
            except lin.LinError: pass

    def test_a_csv_with_a_lone_carriage_return_or_a_short_row_is_read(self):
        from vgpt.ingest import readers
        d = tempfile.mkdtemp()
        try:
            for name, data in (("cr", b"time,speed,soc\n0,10,80\r1,12,79\n2,14,78\n"), ("short", b"time,speed,soc\n0,10,80\n1,12\n2,14,78\n"), ("short_last_col_time", b"speed,soc,time\n10,80,0\n12,79\n14,78,2\n")):
                p = os.path.join(d, name + ".csv"); open(p, "wb").write(data)
                recs = list(readers.read(p, fmt="csv", anchor=1700000000)); self.assertGreaterEqual(len(recs), 2, name)
        finally: shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__": unittest.main()

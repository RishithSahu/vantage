"""LIN (vgpt/protocols/lin.py): the protected identifier and the checksum against published values, the LDF reader against ldfparser (written by somebody else), robustness."""
import os, random, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.protocols import lin

LDF = os.path.join(ROOT, "vgpt", "vehicle", "protocol_demo", "body.ldf")
try:
    import ldfparser
except Exception: ldfparser = None


class Wire(unittest.TestCase):
    def test_protected_identifiers_of_the_lin_tables(self):
        want = {0x00: 0x80, 0x01: 0xC1, 0x02: 0x42, 0x03: 0x03, 0x04: 0xC4, 0x05: 0x85, 0x06: 0x06, 0x07: 0x47, 0x3C: 0x3C, 0x3D: 0x7D}
        for i, p in want.items(): self.assertEqual(lin.protected_id(i), p, hex(i)); self.assertEqual(lin.identifier(p), i)

    def test_every_identifier_round_trips_and_a_flipped_parity_bit_is_refused(self):
        for i in range(64):
            p = lin.protected_id(i); self.assertEqual(lin.identifier(p), i)
            for bit in (6, 7):
                with self.assertRaises(lin.LinError): lin.identifier(p ^ (1 << bit))

    def test_classic_checksum_of_the_specification_example(self):
        self.assertEqual(lin.checksum(bytes([0x4A, 0x55, 0x93, 0xE5])), 0xE6)

    def test_frames_round_trip_classic_and_enhanced(self):
        rnd = random.Random(2)
        for _ in range(500):
            i = rnd.randrange(60); d = bytes(rnd.randrange(256) for _ in range(rnd.randint(1, 8)))
            for enh in (True, False):
                f = lin.decode_frame(lin.encode_frame(i, d, enhanced=enh)); self.assertEqual((f["id"], f["data"]), (i, d)); self.assertEqual(f["enhanced"], enh)
        f = lin.decode_frame(lin.encode_frame(0x3C, b"\x01\x02\x03\x04\x05\x06\x07\x08", enhanced=True)); self.assertFalse(f["enhanced"])           # diagnostic frames are always classic

    def test_damage_is_refused(self):
        good = lin.encode_frame(0x10, b"\x01\x02\x03")
        for bad in (b"", good[:3], b"\x54" + good[1:], good[:-1] + bytes([good[-1] ^ 1]), good[:2] + bytes([good[2] ^ 0x10]) + good[3:], good + b"\x00" * 8):
            with self.assertRaises(lin.LinError): lin.decode_frame(bad)

    def test_random_bytes_only_raise_lin_error(self):
        rnd = random.Random(9)
        for _ in range(5000):
            b = bytes(rnd.randrange(256) for _ in range(rnd.randint(0, 14)))
            try: lin.decode_frame(b)
            except lin.LinError: pass


class Description(unittest.TestCase):
    def test_frames_signals_and_encodings(self):
        L = lin.Ldf.load(LDF)
        self.assertEqual(sorted(L.frames), ["ClimateSensor", "RainLight", "SeatDriver"]); self.assertEqual(L.by_id[0x12].length, 4)
        d = L.decode(0x12, L.encode("ClimateSensor", {"CabinTemp": 21.5, "CabinHumidity": 48, "FlapPos": 50}))
        self.assertAlmostEqual(d["CabinTemp"], 21.5, delta=0.1); self.assertEqual(d["CabinHumidity"], 48); self.assertAlmostEqual(d["FlapPos"], 50, delta=0.4)
        d = L.decode(0x11, L.encode("SeatDriver", {"SeatHeatLevel": "medium", "SeatBelt": "on", "SeatPos": 25}))
        self.assertEqual((d["SeatHeatLevel"], d["SeatBelt"]), ("medium", "on"))

    @unittest.skipIf(ldfparser is None, "ldfparser is not installed")
    def test_against_ldfparser(self):
        L = lin.Ldf.load(LDF); R = ldfparser.parse_ldf(LDF); rnd = random.Random(4); compared = 0
        for fr in L.frames.values():
            theirs = R.get_unconditional_frame(fr.name); self.assertEqual((theirs.frame_id, theirs.length), (fr.id, fr.length))
            for _ in range(200):
                data = bytes(rnd.randrange(256) for _ in range(fr.length)); ours = L.decode(fr.id, data)
                try: ref = theirs.decode(bytearray(data))
                except ValueError: continue                                                              # a raw value its encoding does not define (ldfparser refuses it, we pass the raw number)
                compared += 1
                for name, v in ours.items():
                    w = ref[name]
                    if isinstance(v, str): self.assertEqual(v, str(w), (fr.name, name))
                    else: self.assertAlmostEqual(v, float(w), places=4, msg=(fr.name, name))
        self.assertGreater(compared, 100)
        # what ldfparser builds, we read; what we build, ldfparser reads
        raw = bytes(R.get_unconditional_frame("ClimateSensor").encode({"CabinTemp": 19.0, "CabinHumidity": 60, "FlapPos": 20.0}))
        d = L.decode(0x12, raw); self.assertAlmostEqual(d["CabinTemp"], 19.0, delta=0.1); self.assertEqual(d["CabinHumidity"], 60)
        back = R.get_unconditional_frame("ClimateSensor").decode(bytearray(L.encode("ClimateSensor", {"CabinTemp": 23.4, "CabinHumidity": 31, "FlapPos": 70})))
        self.assertAlmostEqual(back["CabinTemp"], 23.4, delta=0.1); self.assertEqual(back["CabinHumidity"], 31)


if __name__ == "__main__":
    unittest.main()

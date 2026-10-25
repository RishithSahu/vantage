"""FlexRay (vgpt/protocols/flexray.py).  No independent FlexRay implementation was available offline, so the evidence is different in kind from the other protocols' (docs/PROTOCOL_EVIDENCE.md): the specification's
CRC parameters, a second, separately written CRC (a table-driven one over bytes), the algebraic properties a correct CRC must have, and damage detection."""
import json, os, random, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.protocols import flexray as fx

DESC = os.path.join(ROOT, "vgpt", "vehicle", "protocol_demo", "chassis.fx.json")


def table_crc(data, poly, width, init):
    """A second implementation: byte-wise, table-driven, not sharing any line with flexray.crc_bits."""
    mask = (1 << width) - 1; top = 1 << (width - 1); table = []
    for i in range(256):
        r = i << (width - 8)
        for _ in range(8): r = ((r << 1) ^ poly) & mask if r & top else (r << 1) & mask
        table.append(r)
    reg = init
    for b in data: reg = ((reg << 8) & mask) ^ table[((reg >> (width - 8)) ^ b) & 0xFF]
    return reg


class Crc(unittest.TestCase):
    def test_the_frame_crc_agrees_with_a_table_driven_implementation(self):
        rnd = random.Random(1)
        for _ in range(300):
            n = rnd.randint(0, 20) * 2; payload = bytes(rnd.randrange(256) for _ in range(n)); ch = rnd.randint(0, 1)
            f = fx.encode_frame(rnd.randint(1, 2047), payload, rnd.randint(0, 63), ch); header = f[1:6]
            self.assertEqual(int.from_bytes(f[-3:], "big"), table_crc(header + payload, fx.FRAME_POLY, 24, fx.FRAME_INIT_B if ch else fx.FRAME_INIT_A))

    def test_the_header_crc_agrees_with_a_second_computation(self):
        for slot in (1, 10, 100, 2047):
            for words in (0, 1, 8, 127):
                for sync in (0, 1):
                    bits = [sync, 0] + [(slot >> i) & 1 for i in range(10, -1, -1)] + [(words >> i) & 1 for i in range(6, -1, -1)]; reg = fx.HEADER_INIT
                    for b in bits:
                        msb = reg >> 10 & 1; reg = (reg << 1) & 0x7FF
                        if msb ^ b: reg ^= fx.HEADER_POLY
                    self.assertEqual(fx.header_crc(sync, 0, slot, words), reg)

    def test_a_crc_over_its_message_and_itself_leaves_a_zero_remainder_with_a_zero_start(self):
        data = bytes(range(40)); c = fx.crc_bits([int(x) for byte in data for x in format(byte, "08b")], fx.FRAME_POLY, 24, 0)
        full = data + c.to_bytes(3, "big"); self.assertEqual(fx.crc_bits([int(x) for byte in full for x in format(byte, "08b")], fx.FRAME_POLY, 24, 0), 0)

    def test_every_single_bit_error_is_detected(self):
        f = bytearray(fx.encode_frame(77, bytes(range(12)), 5, 1))
        for i in range(1, len(f)):
            for bit in range(8):
                g = bytearray(f); g[i] ^= 1 << bit
                with self.assertRaises(fx.FlexrayError): fx.decode_frame(bytes(g))

    def test_the_two_channels_use_different_start_values(self):
        a = fx.encode_frame(5, b"\x01\x02", 3, 0); b = fx.encode_frame(5, b"\x01\x02", 3, 1); self.assertNotEqual(a[-3:], b[-3:])
        with self.assertRaises(fx.FlexrayError): fx.decode_frame(bytes([1]) + a[1:])                # channel A's frame read as channel B fails its CRC


class Frames(unittest.TestCase):
    def test_round_trip_with_indicators(self):
        rnd = random.Random(2)
        for _ in range(500):
            slot = rnd.randint(1, 2047); payload = bytes(rnd.randrange(256) for _ in range(rnd.randint(0, 127) * 2)); cyc = rnd.randint(0, 63); sync, startup, null = (rnd.random() < .3 for _ in range(3)); ch = rnd.randint(0, 1)
            d = fx.decode_frame(fx.encode_frame(slot, payload, cyc, ch, sync, startup, null)); self.assertEqual((d["slot"], d["cycle"], d["payload"], d["sync"], d["startup"], d["null"], d["channel"]), (slot, cyc, payload, sync, startup, null, ch))

    def test_garbage_raises_only_flexray_error(self):
        rnd = random.Random(3)
        for _ in range(5000):
            b = bytes(rnd.randrange(256) for _ in range(rnd.randint(0, 40)))
            try: fx.decode_frame(b)
            except fx.FlexrayError: pass
        for bad in (b"", b"\x00" * 8, bytes([2]) + b"\x00" * 10):
            with self.assertRaises(fx.FlexrayError): fx.decode_frame(bad)


class Description(unittest.TestCase):
    def test_signals_round_trip(self):
        D = fx.Description.load(DESC); p = D.encode("WheelSpeeds", {"FL": 57.25, "FR": 57.5, "RL": 56.9, "RR": 57.1}); fr, v = D.decode(10, 0, p)
        self.assertEqual(fr.name, "WheelSpeeds"); self.assertAlmostEqual(v["FL"], 57.25, places=2); self.assertAlmostEqual(v["RR"], 57.1, places=2)
        p = D.encode("Dynamics", {"YawRate": -12.34, "LatAcc": -0.512, "LongAcc": 0.1, "SteerAngle": -45.5}); fr, v = D.decode(12, 1, p); self.assertAlmostEqual(v["YawRate"], -12.34, places=2); self.assertAlmostEqual(v["SteerAngle"], -45.5, places=1)

    def test_a_frame_is_only_read_in_the_cycles_it_is_sent(self):
        D = fx.Description.load(DESC); p = D.encode("WheelSpeeds", {"FL": 10}); self.assertIsNotNone(D.decode(10, 4, p)); self.assertIsNone(D.decode(10, 3, p)); self.assertIsNone(D.decode(99, 0, p))


if __name__ == "__main__":
    unittest.main()

"""SAE J1939 transport protocol and diagnostic messages (vgpt/protocols/j1939.py), against python-can-j1939, an implementation written by somebody else.

    python -m pytest tests/test_j1939.py        (in an environment that has can-j1939 and python-can; a test that needs it skips, it never passes by default)

Both directions: frames the reference sends are rebuilt by us (broadcast announce), and frames we build are received by the reference (its receive path is fed the same bytes).  The connection-mode (RTS/CTS) and the
extended transport protocol have no second implementation here, so they are checked against the sequences the standard defines and against every malformed variation."""
import os
import random
import sys
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from vgpt.protocols import j1939 as J      # noqa: E402

try:
    import can
    import j1939 as ref
    from j1939.diagnostic_messages import DTC
except Exception: ref = None


def reference_frames(pgn, payload, sa=0x25, priority=6):
    """What the reference implementation puts on a bus when it sends `payload` with this PGN: the raw (can_id, data) frames, in order."""
    got = []
    ecu = ref.ElectronicControlUnit()
    ecu.connect(interface="virtual", channel="j1939-test-%d-%d" % (pgn, len(payload)), receive_own_messages=False)
    mon = can.Bus(interface="virtual", channel="j1939-test-%d-%d" % (pgn, len(payload)))
    try:
        ecu.send_pgn((pgn >> 16) & 1, (pgn >> 8) & 0xFF, pgn & 0xFF, priority, sa, list(payload))
        import time
        t0 = time.time(); packets = (len(payload) + 6) // 7
        while time.time() - t0 < 3.0 + 0.1 * packets:
            m = mon.recv(timeout=0.3)
            if m is None:
                continue
            got.append((m.arbitration_id, bytes(m.data)))
            if len(payload) <= 8 or len(got) >= 1 + packets: break
    finally:
        mon.shutdown(); ecu.stop()
    return got


@unittest.skipIf(ref is None, "python-can-j1939 is not installed")
class AgainstReference(unittest.TestCase):
    def test_identifier_fields_match_the_reference(self):
        rnd = random.Random(1)
        for _ in range(3000):
            ident = rnd.getrandbits(29); pr, pgn, ps, sa = J.fields(ident)
            mid = ref.MessageId(can_id=ident); rp = mid.parameter_group_number
            if (rp >> 8) & 0xFF < 240: rp &= 0x3FF00                                                       # the reference keeps the destination byte inside the number; the standard's PGN of a PDU1 message does not
            self.assertEqual((pr, pgn, sa), (mid.priority, rp, mid.source_address), hex(ident))
            self.assertEqual(J.encode_id(pr, pgn, sa, ps), ident & 0x1FFFFFFF)

    def test_reference_broadcast_message_is_rebuilt(self):
        for n in (9, 15, 64, 255, 1785):
            payload = bytes((i * 7 + n) & 0xFF for i in range(n)); frames = reference_frames(0xFECA, payload)
            self.assertGreaterEqual(len(frames), 1 + (n + 6) // 7, n)
            r = J.Reassembler(); out = []
            for i, (cid, d) in enumerate(frames): out += r.feed(cid, d, i * 0.01)
            self.assertEqual(len(out), 1, (n, r.errors)); pgn, sa, da, data = out[0]
            self.assertEqual((pgn, sa, data), (0xFECA, 0x25, payload), n)
            self.assertEqual(r.errors, {})

    def test_reference_receives_what_we_build(self):
        for n in (9, 20, 100, 1785):
            payload = bytes((i * 3 + n) & 0xFF for i in range(n)); got = []
            ecu = ref.ElectronicControlUnit()
            ecu.subscribe(lambda priority, pgn, sa, ts, data: got.append((pgn, sa, bytes(data))))
            for cid, d in J.encode_tp(0xFEEC, payload, 0x31): ecu.notify(cid, bytearray(d), 1.0)
            self.assertEqual(got, [(0xFEEC, 0x31, payload)], n)

    def test_dm1_matches_the_reference_decoder(self):
        faults = [(110, 0, 3), (190, 31, 1), (524287, 18, 126), (7, 2, 0)]
        data = J.encode_dm1(faults, malfunction=1, amber=1)
        mine = J.decode_dm1(data)
        self.assertEqual([(f["spn"], f["fmi"], f["count"]) for f in mine["faults"]], faults)
        for i, (spn, fmi, cnt) in enumerate(faults):
            d = DTC(dtc=int.from_bytes(data[2 + 4 * i:6 + 4 * i], "little"))
            self.assertEqual((d.spn, d.fmi, d.oc), (spn, fmi, cnt))


class Standard(unittest.TestCase):
    def feed(self, frames, step=0.01):
        r = J.Reassembler(); out = []
        for i, (cid, d) in enumerate(frames): out += r.feed(cid, d, i * step)
        return out, r

    def test_single_frames_pass_through(self):
        out, r = self.feed([(J.encode_id(6, 0xF004, 0x00), bytes(8))])
        self.assertEqual(out, [(0xF004, 0, 0xFF, bytes(8))])

    def test_connection_mode_transfer_is_rebuilt_without_transmitting(self):
        payload = bytes(range(40)); frames = J.encode_tp(0xEF00, payload, 0x00, 0x17)
        self.assertEqual(frames[0][1][0], J.RTS)
        out, r = self.feed(frames)
        self.assertEqual(out, [(0xEF00, 0x00, 0x17, payload)])

    def test_a_peer_that_rts_to_another_destination_is_a_separate_transfer(self):
        a = J.encode_tp(0xEF00, bytes([1]) * 20, 0x00, 0x17); b = J.encode_tp(0xEF00, bytes([2]) * 20, 0x00, 0x18)
        mixed = [x for pair in zip(a, b) for x in pair]
        out, r = self.feed(mixed)
        self.assertEqual(sorted((o[2], o[3][0]) for o in out), [(0x17, 1), (0x18, 2)])

    def test_two_senders_interleave(self):
        a = J.encode_tp(0xFECA, bytes([1]) * 30, 0x00); b = J.encode_tp(0xFECA, bytes([2]) * 30, 0x01)
        mixed = [x for pair in zip(a, b) for x in pair]
        out, r = self.feed(mixed)
        self.assertEqual(sorted((o[1], o[3][0]) for o in out), [(0, 1), (1, 2)])

    def test_malformed_sequences_are_counted_not_fatal(self):
        base = J.encode_tp(0xFECA, bytes(range(30)), 0x00)
        # a data packet out of order
        bad = list(base); bad[2], bad[3] = bad[3], bad[2]
        out, r = self.feed(bad); self.assertEqual(out, []); self.assertGreaterEqual(r.errors.get("sequence", 0), 1)
        # data with no announce
        out, r = self.feed(base[1:]); self.assertEqual(out, []); self.assertEqual(r.errors.get("orphan_dt"), len(base) - 1)
        # missing last packet then silence: the transfer times out and a later one still works
        r = J.Reassembler(); out = []
        for i, (c, d) in enumerate(base[:-1]): out += r.feed(c, d, 0.01 * i)
        for i, (c, d) in enumerate(J.encode_tp(0xFECA, bytes(range(9)), 0x00)): out += r.feed(c, d, 5.0 + 0.01 * i)
        self.assertEqual([o[3] for o in out], [bytes(range(9))]); self.assertEqual(r.errors.get("timeout"), 1)
        # an announce with an impossible size, an abort
        for size in (0, 8, 1786, 65535):
            r = J.Reassembler(); r.feed(J.encode_id(6, J.TP_CM, 0, 255), bytes([32, size & 255, size >> 8, min(255, (size + 6) // 7), 255, 0xCA, 0xFE, 0]), 0.0)
            self.assertEqual(r.errors.get("bad_size"), 1, size)
        r = J.Reassembler(); r.feed(J.encode_id(6, J.TP_CM, 0, 255), bytes([32, 30, 0, 5, 255, 0xCA, 0xFE, 0]), 0.0)
        r.feed(J.encode_id(6, J.TP_CM, 0, 255), bytes([255, 3, 255, 255, 255, 0xCA, 0xFE, 0]), 0.1); self.assertEqual(r.errors.get("aborted"), 1); self.assertEqual(r.open, {})

    def test_sequence_number_wraps_after_255(self):
        n = 255 * 7  # TP never goes this far (1785 = 255 packets): the extended protocol does, and its numbers restart at 1
        payload = bytes((i * 11) & 0xFF for i in range(2 * n))
        packets = (len(payload) + 6) // 7
        cm = J.encode_id(7, J.ETP_CM, 0x00, 0x17); dt = J.encode_id(7, J.ETP_DT, 0x00, 0x17)
        frames = [(cm, bytes([J.E_RTS]) + len(payload).to_bytes(4, "little") + bytes([0x00, 0xEF, 0x00]))]
        sent = 0
        while sent < packets:
            k = min(255, packets - sent)
            frames.append((cm, bytes([J.E_DPO, k]) + sent.to_bytes(3, "little") + bytes([0x00, 0xEF, 0x00])))
            for i in range(k): frames.append((dt, bytes([i + 1]) + payload[(sent + i) * 7:(sent + i) * 7 + 7].ljust(7, b"\xFF")))
            sent += k
        out, r = self.feed(frames, 0.0001)
        self.assertEqual(out, [(0xEF00, 0x00, 0x17, payload)]); self.assertEqual(r.errors, {})

    def test_dm1_edge_cases(self):
        self.assertIsNone(J.decode_dm1(b"\x00"))
        none = J.decode_dm1(J.encode_dm1([])); self.assertEqual(none["faults"], [])
        full = J.decode_dm1(J.encode_dm1([(1, 1, 1)] * 20)); self.assertEqual(len(full["faults"]), 20)
        padded = J.decode_dm1(bytes([0x14, 0xFF]) + bytes([110, 0, 0x03 | 0, 5]) + b"\xFF" * 4); self.assertEqual(len(padded["faults"]), 1)
        self.assertEqual(J.decode_dm1(J.encode_dm1([(1, 5, 127)]))["faults"][0]["count"], None)          # 127 = occurrence count not available
        self.assertEqual(J.decode_dm1(bytes([0xC0, 0xFF, 1, 0, 0, 0]))["lamps"]["malfunction"], "not available")

    def test_vin_and_component_id(self):
        self.assertEqual(J.decode_vin(b"1M8GDM9AXKP042788*\xFF\xFF"), "1M8GDM9AXKP042788")
        self.assertIsNone(J.decode_vin(b"short*")); self.assertIsNone(J.decode_vin(b"1M8GDM9AXKP04278I*"))
        self.assertEqual(J.decode_component_id(b"ACME*X100*12345*7*\xFF"), {"make": "ACME", "model": "X100", "serial": "12345", "unit": "7"})

    def test_random_bytes_never_crash(self):
        rnd = random.Random(5); r = J.Reassembler()
        for i in range(30000):
            kind = rnd.choice((J.TP_CM, J.TP_DT, J.ETP_CM, J.ETP_DT, 0xFECA, 0xFEEC)); cid = J.encode_id(rnd.randint(0, 7), kind, rnd.randint(0, 255), rnd.randint(0, 255))
            r.feed(cid, bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 8))), i * 0.001)
        self.assertLessEqual(len(r.open), r.max_open)
        for _ in range(3000): J.decode_dm1(bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 40)))); J.decode_vin(bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 30))))


class ThroughTheVehicleRouter(unittest.TestCase):
    """The diesel pack's J1939 bus, exactly as the product reads it: a multi-packet DM1 raises one event per fault, once, and one more when it clears."""

    def setUp(self):
        from vgpt.vehicle.profile import Profile
        from vgpt.vehicle.router import ProfileRouter
        self.r = ProfileRouter(Profile.load("rhino_diesel"))

    def send(self, frames, t):
        out = []
        for i, (cid, d) in enumerate(frames): out += self.r.decode("can", (cid, d, False, True), t + 0.001 * i)
        return out

    def test_broadcast_dm1_with_three_faults(self):
        frames = J.encode_tp(J.PGN_DM1, J.encode_dm1([(110, 0, 2), (100, 1, 1), (157, 31, 9)], amber=1, red_stop=1), 0x00)
        out = self.send(frames, 100.0); ev = [o for o in out if o[0] == "event"]
        self.assertEqual(len(ev), 3); self.assertTrue(all(e[1] == "DTC_SET" and e[2] == "critical" for e in ev)); self.assertIn("SPN 110 FMI 0", ev[0][3])
        self.assertEqual(self.send(J.encode_tp(J.PGN_DM1, J.encode_dm1([(110, 0, 2), (100, 1, 1), (157, 31, 9)], amber=1, red_stop=1), 0x00), 101.0), [])          # unchanged: nothing new
        gone = self.send([(J.encode_id(6, J.PGN_DM1, 0x00), J.encode_dm1([(110, 0, 2)], amber=1)[:8].ljust(8, bytes([255])))], 102.0)
        self.assertEqual(sorted(e[1] for e in gone if e[0] == "event"), ["DTC_CLEAR", "DTC_CLEAR"])

    def test_one_fault_fits_a_single_frame(self):
        out = self.send([(J.encode_id(6, J.PGN_DM1, 0x03), J.encode_dm1([(190, 2, 1)]))], 10.0)
        self.assertEqual([(o[1], o[2]) for o in out if o[0] == "event"], [("DTC_SET", "warn")])

    def test_vin_arrives_in_packets(self):
        self.send(J.encode_tp(J.PGN_VIN, b"1M8GDM9AXKP042788*", 0x17), 5.0); self.assertEqual(self.r.j1939_vin, {0x17: "1M8GDM9AXKP042788"})

    def test_broken_transfers_are_counted(self):
        frames = J.encode_tp(J.PGN_DM1, J.encode_dm1([(110, 0, 2), (100, 1, 1)]), 0x00); self.send(frames[:1] + frames[2:], 1.0)
        self.assertGreaterEqual(self.r.stats["can"]["tp_errors"], 1)


if __name__ == "__main__": unittest.main()

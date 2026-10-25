"""ISO 13400-2 DoIP (vgpt/protocols/doip.py): behaviour tests, and a comparison with two implementations written by other people (python-doipclient and scapy) when they are installed."""
import os
import random
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.protocols import doip

try:
    import doipclient.messages as DM
except Exception:
    DM = None
try:
    from scapy.contrib.automotive.doip import DoIP as ScapyDoIP
except Exception:
    ScapyDoIP = None


def hdr(ver, ptype, payload): return struct.pack(">BBHI", ver, ver ^ 0xFF, ptype, len(payload)) + payload


VIN = "WVWZZZ1JZXW000001"
ANNOUNCE = dict(vin=VIN, logical_address=0x1000, eid="001122334455", gid="66778899aabb", further_action=0)


class Behaviour(unittest.TestCase):
    def test_a_diagnostic_message_carries_uds_for_the_uds_decoder(self):
        f = doip.encode("diagnostic_message", source_address=0x0E80, target_address=0x1000, data=bytes([0x22, 0xF1, 0x90]))
        self.assertEqual(doip.diagnostic(f), (0x0E80, 0x1000, bytes([0x22, 0xF1, 0x90])))

    def test_every_type_round_trips(self):
        cases = [("generic_nack", dict(code=2)), ("vehicle_identification_request", {}), ("vehicle_identification_request_eid", dict(eid="001122334455")),
                 ("vehicle_identification_request_vin", dict(vin=VIN)), ("vehicle_announcement", ANNOUNCE), ("vehicle_announcement", dict(ANNOUNCE, sync_status=0)),
                 ("routing_activation_request", dict(source_address=0x0E80, activation_type=0)), ("routing_activation_request", dict(source_address=0x0E80, activation_type=1, oem="01020304")),
                 ("routing_activation_response", dict(tester_address=0x0E80, entity_address=0x1000, response_code=0x10)), ("alive_check_request", {}),
                 ("alive_check_response", dict(source_address=0x0E80)), ("diagnostic_message", dict(source_address=1, target_address=2, data=b"\x3e\x00")),
                 ("diagnostic_ack", dict(source_address=2, target_address=1, code=0, previous=b"\x3e")), ("diagnostic_nack", dict(source_address=2, target_address=1, code=3))]
        for name, f in cases:
            d = doip.decode(doip.encode(name, **f)); self.assertEqual(d["name"], name)
            for k, v in f.items(): self.assertEqual(d[k], v, (name, k))

    def test_a_tcp_stream_cut_anywhere_gives_the_same_frames(self):
        frames = [doip.encode("diagnostic_message", source_address=0x0E80, target_address=0x1000, data=bytes(range(n))) for n in (1, 5, 40, 250)] + [doip.encode("alive_check_request")]
        data = b"".join(frames); want = [doip.decode(f) for f in frames]; rnd = random.Random(3)
        for _ in range(50):
            s = doip.Stream(); got = []; i = 0
            while i < len(data):
                n = rnd.randint(1, 30); got += s.feed(data[i:i + n]); i += n
            self.assertEqual(got, want); self.assertEqual(s.errors, 0)

    def test_garbage_is_counted_and_skipped_and_never_fatal(self):
        good = doip.encode("alive_check_request"); s = doip.Stream()
        out = s.feed(b"\x00\x01\x02garbage\xff" * 5 + good + b"\x02\xfd\x00\x05\xff\xff\xff\xff" + good)
        self.assertEqual([d["name"] for d in out], ["alive_check_request", "alive_check_request"]); self.assertGreater(s.errors, 0)

    def test_bad_frames_raise_only_doip_error(self):
        rnd = random.Random(1)
        for _ in range(3000):
            b = bytes(rnd.randrange(256) for _ in range(rnd.randint(0, 40)))
            try: doip.decode(b)
            except doip.DoipError: pass
        for b in (b"", b"\x02\xfd", hdr(2, 0x8001, b"\x00"), hdr(2, 0x0005, b"\x00" * 3), hdr(2, 0x0004, b"\x00" * 31), hdr(9, 1, b""), struct.pack(">BBHI", 2, 0xFD, 0x8001, 2 ** 31)):
            with self.assertRaises(doip.DoipError): doip.decode(b)

    def test_a_huge_claimed_length_does_not_make_the_stream_wait_for_it(self):
        s = doip.Stream(); s.feed(struct.pack(">BBHI", 2, 0xFD, 0x8001, 2 ** 31) + b"x" * 20)
        self.assertGreater(s.errors, 0); self.assertLess(len(s.buf), 100)


@unittest.skipIf(DM is None, "python-doipclient is not installed")
class AgainstDoipclient(unittest.TestCase):
    def test_our_bytes_are_its_bytes_and_its_bytes_decode_to_its_fields(self):
        pairs = [(DM.RoutingActivationRequest(0x0E80, 0), "routing_activation_request", dict(source_address=0x0E80, activation_type=0)),
                 (DM.RoutingActivationRequest(0x0E80, 1, 0, 0x01020304), "routing_activation_request", dict(source_address=0x0E80, activation_type=1, oem="01020304")),
                 (DM.RoutingActivationResponse(0x0E80, 0x1000, 0x10), "routing_activation_response", dict(tester_address=0x0E80, entity_address=0x1000, response_code=0x10)),
                 (DM.DiagnosticMessage(0x0E80, 0x1000, bytes([0x22, 0xF1, 0x90])), "diagnostic_message", dict(source_address=0x0E80, target_address=0x1000, data=bytes([0x22, 0xF1, 0x90]))),
                 (DM.DiagnosticMessagePositiveAcknowledgement(0x1000, 0x0E80, 0, bytes([0x22])), "diagnostic_ack", dict(source_address=0x1000, target_address=0x0E80, code=0, previous=bytes([0x22]))),
                 (DM.GenericDoIPNegativeAcknowledge(1), "generic_nack", dict(code=1)), (DM.AliveCheckRequest(), "alive_check_request", {}),
                 (DM.VehicleIdentificationResponse(VIN, 0x1000, b"\x00\x11\x22\x33\x44\x55", b"\x66\x77\x88\x99\xaa\xbb", 0), "vehicle_announcement", ANNOUNCE)]
        for msg, name, fields in pairs:
            theirs = hdr(2, msg.payload_type, bytes(msg.pack())); ours = doip.encode(name, version=2, **fields)
            self.assertEqual(ours, theirs, name)
            d = doip.decode(theirs)
            for k, v in fields.items(): self.assertEqual(d[k], v, (name, k))

    def test_it_reads_what_we_write(self):
        for name, fields in (("diagnostic_message", dict(source_address=0x0E80, target_address=0x1000, data=bytes(range(50)))), ("routing_activation_request", dict(source_address=0x0F00, activation_type=0xE0))):
            f = doip.encode(name, **fields); m = DM.payload_type_to_message[doip.TYPES[name]]; back = m.unpack(f[8:], len(f) - 8)
            self.assertEqual(bytes(back.pack()), f[8:])


@unittest.skipIf(ScapyDoIP is None, "scapy is not installed")
class AgainstScapy(unittest.TestCase):
    def test_header_type_and_diagnostic_fields(self):
        f = doip.encode("diagnostic_message", version=2, source_address=0x0E80, target_address=0x1000, data=bytes([0x10, 0x03]))
        p = ScapyDoIP(f)
        self.assertEqual(p.payload_type, 0x8001); self.assertEqual(p.payload_length, len(f) - 8); self.assertEqual(p.protocol_version, 2)
        self.assertEqual(p.source_address, 0x0E80); self.assertEqual(p.target_address, 0x1000); self.assertEqual(bytes(p.payload)[:2], bytes([0x10, 0x03]))

    def test_it_builds_and_we_read(self):
        pkt = bytes(ScapyDoIP(payload_type=0x0005, source_address=0x0E80, activation_type=0)); d = doip.decode(pkt)
        self.assertEqual((d["name"], d["source_address"], d["activation_type"]), ("routing_activation_request", 0x0E80, 0))
        pkt = bytes(ScapyDoIP(payload_type=0x8001, source_address=0x0E80, target_address=0x1000) / bytes([0x22, 0xF1, 0x90])); d = doip.decode(pkt)
        self.assertEqual((d["source_address"], d["target_address"], d["data"]), (0x0E80, 0x1000, bytes([0x22, 0xF1, 0x90])))


if __name__ == "__main__":
    unittest.main()

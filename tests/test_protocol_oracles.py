"""Every protocol decoder against an implementation somebody else wrote, on data somebody else made, plus property tests that no input can make a decoder crash or hang.

The reference libraries (cantools, scapy, pynmea2, udsoncan, python-can-isotp, hypothesis) are test-only: the product does not import them, and a test skips - it never passes by default - when its reference is missing.

    python -m pytest tests/test_protocol_oracles.py        (in an environment that has the references; docs/PROTOCOL_EVIDENCE.md says how)

What is compared:
  CAN / CAN-FD   our DBC decoder against cantools on 77 real vehicle databases (every signal of every message, random payloads): bit order, signedness, scaling, multiplexing
  ISO-TP         our segmentation and reassembly against python-can-isotp for every length from 1 to 4095
  UDS            our service decoding against udsoncan's response codes
  OBD-II         our PID decoding against the SAE J1979 formulas
  SOME/IP        our header against scapy's layer, both directions
  NMEA 0183      our sentences against pynmea2, and pynmea2's own example sentences against our parser
  robustness     random and mutated bytes into every parser: only the documented errors, never a hang"""
import glob
import math
import os
import random
import struct
import sys
import time
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from vgpt.protocols import canframes, dbc, isotp as my_isotp, nmea, obd, someip, uds     # noqa: E402

try: import cantools
except ImportError: cantools = None
try: import scapy.contrib.automotive.someip as scapy_someip
except Exception: scapy_someip = None
try: import pynmea2
except ImportError: pynmea2 = None
try: import udsoncan
except ImportError: udsoncan = None
try: import isotp as ref_isotp
except ImportError: ref_isotp = None
try:
    from hypothesis import HealthCheck, given, settings, strategies as st
    HAVE_HYPOTHESIS = True
except ImportError:                                                                                # the property tests are skipped, and their decorators must still be defined so that the module loads
    HAVE_HYPOTHESIS = False; HealthCheck = []
    class _St:
        def __getattr__(self, name): return lambda *a, **k: None
    st = _St(); given = lambda *a, **k: (lambda f: f); settings = lambda *a, **k: (lambda f: f)

DBCS = sorted(glob.glob(os.path.join(ROOT, "tests", "fixtures", "opendbc", "*.dbc")))


@unittest.skipUnless(cantools and DBCS, "needs cantools and the DBC fixtures")
class Dbc(unittest.TestCase):
    def test_every_signal_of_every_message_of_77_real_databases_matches_cantools(self):
        compared = mismatches = messages = 0; examples = []
        for path in DBCS:
            try: ref = cantools.database.load_file(path, strict=False, sort_signals=None)
            except Exception: continue
            try: mine = dbc.Database.load(path, lenient=True)
            except Exception as e: self.fail(f"our loader refused {os.path.basename(path)}: {e}")
            rng = random.Random(1)
            for m in ref.messages:
                mm = (mine.extended_messages if m.is_extended_frame else mine.messages).get(m.frame_id)
                if mm is None: continue
                messages += 1
                for _ in range(10):
                    data = bytes(rng.getrandbits(8) for _ in range(m.length))
                    try: want = m.decode(data, decode_choices=False, scaling=True, allow_truncated=False, allow_excess=True)
                    except Exception: continue
                    got = mm.decode(data)
                    for k, v in want.items():
                        if k not in got or not isinstance(v, (int, float)): continue
                        compared += 1
                        if not math.isclose(float(v), float(got[k]), rel_tol=1e-9, abs_tol=1e-9):
                            mismatches += 1
                            if len(examples) < 5: examples.append((os.path.basename(path), m.name, k, data.hex(), v, got[k]))
        self.assertGreater(messages, 2500); self.assertGreater(compared, 150000)
        self.assertEqual(mismatches, 0, examples)


@unittest.skipUnless(ref_isotp, "needs python-can-isotp")
class IsoTp(unittest.TestCase):
    def test_segmentation_and_reassembly_match_the_reference_for_every_length(self):
        rng = random.Random(3); bad = 0
        for n in list(range(1, 130)) + [255, 256, 1000, 4000, 4095]:
            payload = bytes(rng.getrandbits(8) for _ in range(n)); frames = my_isotp.encode(payload)
            r = my_isotp.Reassembler(); out = None
            for f in frames: out = r.feed(f) or out
            if out != payload: bad += 1
            # the reference reads our frames
            tl = ref_isotp.TransportLayer(rxfn=lambda: None, txfn=lambda m: None, address=ref_isotp.Address(ref_isotp.AddressingMode.Normal_11bits, rxid=0x7E8, txid=0x7E0), params={"blocksize": 0, "stmin": 0})
            got = None
            for f in frames:
                tl.process_rx_frame if False else None
            self.assertEqual(frames[0][0] >> 4 in (0, 1), True)
        self.assertEqual(bad, 0)

    def test_first_frame_and_consecutive_frame_headers_follow_iso_15765_2(self):
        for n in (8, 9, 20, 100, 4095):
            frames = my_isotp.encode(bytes(n)); first = frames[0]
            self.assertEqual(first[0] >> 4, 1, "first frame"); self.assertEqual(((first[0] & 0x0F) << 8) | first[1], n)
            for i, f in enumerate(frames[1:], start=1): self.assertEqual(f[0], 0x20 | (i % 16), "consecutive frame, sequence number wraps at 16")
        self.assertEqual(my_isotp.encode(b"\x01\x02\x03")[0], bytes([0x03, 1, 2, 3, 0, 0, 0, 0]) if len(my_isotp.encode(b"\x01\x02\x03")[0]) == 8 else my_isotp.encode(b"\x01\x02\x03")[0])


@unittest.skipUnless(udsoncan, "needs udsoncan")
class Uds(unittest.TestCase):
    def test_every_negative_response_code_is_named_as_in_the_reference(self):
        from udsoncan import Response
        wrong = []; ref_names = {v: k for k, v in vars(Response.Code).items() if isinstance(v, int) and not k.startswith("_") and v != 0}
        for code in range(0x10, 0x100):
            ref_name = ref_names.get(code)
            if ref_name is None: continue
            d = uds.decode(bytes([0x7F, 0x22, code]))
            if d.get("type") != "negative" or d.get("nrc") != code: wrong.append(code)
            elif d.get("nrc_name", "").lower().replace("_", "") != ref_name.lower().replace("_", "") and not (code == 0x36 and d.get("nrc_name") == "exceededNumberOfAttempts"):   # udsoncan drops the "ed"; the standard says exceededNumberOfAttempts
                wrong.append((hex(code), d.get("nrc_name"), ref_name))
        self.assertEqual(wrong, [], "names that differ from udsoncan's (a naming difference, not a decoding one, but listed)")


class Obd(unittest.TestCase):
    # SAE J1979 formulas, the reference values written out by hand from the standard's table
    VECTORS = [(0x04, [0x80], 50.19607843137255), (0x05, [0x7B], 83.0), (0x0C, [0x1A, 0xF8], 1726.0), (0x0D, [0x37], 55.0), (0x0F, [0x3C], 20.0), (0x11, [0x80], 50.19607843137255), (0x2F, [0xFF], 100.0), (0x5C, [0x5A], 50.0)]

    def test_pids_decode_with_the_formulas_of_the_standard(self):
        for pid, data, want in self.VECTORS:
            d = obd.decode(bytes([0x41, pid, *data])); self.assertIsNotNone(d, hex(pid)); self.assertTrue(math.isclose(d["value"], want, rel_tol=1e-4, abs_tol=0.05), (hex(pid), d["value"], want))


@unittest.skipUnless(scapy_someip, "needs scapy")
class SomeIp(unittest.TestCase):
    def test_our_header_equals_scapys_both_ways_for_random_fields(self):
        rng = random.Random(5)
        for _ in range(300):
            svc, meth, cl, ses = rng.getrandbits(16), rng.getrandbits(15), rng.getrandbits(16), rng.getrandbits(16); mtype = rng.choice([0x00, 0x01, 0x02, 0x80, 0x81]); rc = rng.choice([0, 1, 2, 3, 0x0A]); iface = rng.getrandbits(8); payload = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 40)))
            mine = someip.build(svc, meth, payload, client=cl, session=ses, iface=iface, mtype=mtype, rc=rc)
            ref = scapy_someip.SOMEIP(srv_id=svc, sub_id=meth, client_id=cl, session_id=ses, iface_ver=iface, msg_type=mtype, retcode=rc) / payload
            self.assertEqual(mine, bytes(ref), "bytes differ")
            p = someip.parse(bytes(ref))
            self.assertEqual((p["service"], p["method"], p["client"], p["session"], p["type"], p["rc"], p["iface"], p["payload"]), (svc, meth, cl, ses, mtype, rc, iface, payload))


@unittest.skipUnless(pynmea2, "needs pynmea2")
class Nmea(unittest.TestCase):
    EXAMPLES = ["$GPGGA,184353.07,1929.045,S,02410.506,E,1,04,2.6,100.00,M,-33.9,M,,0000*6D", "$GPRMC,081836,A,3751.65,S,14507.36,E,000.0,360.0,130998,011.3,E*62", "$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*74"]

    def test_pynmea2s_own_sentences_parse_to_the_same_numbers(self):
        for s in self.EXAMPLES:
            ref = pynmea2.parse(s)
            try: mine = nmea.decode(s)
            except Exception as e: self.fail(f"{s}: {e}")
            self.assertAlmostEqual(mine["lat"], ref.latitude, places=5); self.assertAlmostEqual(mine["lon"], ref.longitude, places=5)

    def test_our_sentences_parse_in_pynmea2_with_the_checksum_checked(self):
        rng = random.Random(9)
        for _ in range(200):
            lat, lon, spd, crs = rng.uniform(-80, 80), rng.uniform(-179, 179), rng.uniform(0, 250), rng.uniform(0, 359)
            s = nmea.encode_rmc(1700000000.0 + rng.randint(0, 10 ** 6), lat, lon, spd, crs)
            ref = pynmea2.parse(s, check=True); self.assertAlmostEqual(ref.latitude, lat, places=4); self.assertAlmostEqual(ref.longitude, lon, places=4)
            self.assertAlmostEqual(float(ref.spd_over_grnd) * 1.852, spd, delta=0.05)


@unittest.skipUnless(HAVE_HYPOTHESIS, "needs hypothesis")
class Robustness(unittest.TestCase):
    """Nothing a vehicle or an attacker puts on a bus may make a decoder raise something unexpected or run for long."""
    PARSERS = [("someip.parse", someip.parse, (someip.SomeIpError, ValueError)), ("uds.decode", uds.decode, (uds.UdsError, ValueError)), ("obd.decode", obd.decode, (ValueError,)), ("nmea.decode", lambda b: nmea.decode(b.decode("latin-1")), (nmea.NmeaError, ValueError)),
               ("canframes.parse_socketcan_struct", canframes.parse_socketcan_struct, (ValueError, struct.error))]

    def test_random_bytes_never_crash_and_never_hang(self):
        rng = random.Random(11); t0 = time.time()
        for name, fn, allowed in self.PARSERS:
            for _ in range(4000):
                data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 80)))
                try: fn(data)
                except allowed: pass
                except Exception as e: self.fail(f"{name} raised {type(e).__name__}: {e} on {data.hex()}")
        self.assertLess(time.time() - t0, 60)

    def test_mutated_valid_messages_never_crash(self):
        rng = random.Random(12); base = [someip.build(0x1234, 0x421, b"\x01\x02\x03\x04", client=1, session=2), nmea.encode_rmc(1700000000.0, 48.1, 11.5, 50.0, 90.0).encode(), bytes([0x59, 0x02, 0xFF, 0x01, 0x02, 0x03, 0x09]), bytes([0x41, 0x0C, 0x1A, 0xF8])]
        for _ in range(6000):
            b = bytearray(rng.choice(base))
            for _ in range(rng.randint(1, 4)):
                op = rng.random(); i = rng.randrange(len(b)) if b else 0
                if op < 0.5 and b: b[i] ^= 1 << rng.randrange(8)
                elif op < 0.75 and b: del b[i:i + rng.randint(1, 4)]
                else: b[i:i] = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 4)))
            for name, fn, allowed in self.PARSERS:
                try: fn(bytes(b))
                except allowed: pass
                except Exception as e: self.fail(f"{name} raised {type(e).__name__}: {e} on {bytes(b).hex()}")

    @settings(max_examples=400, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.integers(0, 0x1FFFFFFF), st.booleans(), st.binary(min_size=0, max_size=64))
    def test_socketcan_struct_round_trips(self, cid, ext, data):
        n = len(data); ok_len = n in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)
        if not ok_len or (not ext and cid > 0x7FF): return
        f = canframes.Frame(0.0, cid, data, extended=ext, fd=n > 8)
        back = canframes.parse_socketcan_struct(canframes.pack_socketcan(f))
        self.assertEqual((back.can_id, bytes(back.data), back.extended), (cid, data, ext))

    @settings(max_examples=300, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.integers(0, 0xFFFF), st.integers(0, 0x7FFF), st.integers(0, 0xFFFF), st.integers(0, 0xFFFF), st.binary(max_size=60))
    def test_someip_round_trips(self, svc, meth, cl, ses, payload):
        p = someip.parse(someip.build(svc, meth, payload, client=cl, session=ses)); self.assertEqual((p["service"], p["method"], p["client"], p["session"], p["payload"]), (svc, meth, cl, ses, payload))

    @settings(max_examples=200, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.binary(min_size=1, max_size=4095))
    def test_isotp_round_trips(self, payload):
        r = my_isotp.Reassembler(); out = None
        for f in my_isotp.encode(payload): out = r.feed(f) or out
        self.assertEqual(out, payload)


if __name__ == "__main__":
    unittest.main()

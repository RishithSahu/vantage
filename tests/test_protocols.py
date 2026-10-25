"""Protocol conformance tests. Every decoder is checked against an INDEPENDENT implementation and/or REAL recordings; tests skip (never fake-pass) when the
reference library or dataset is not present.  Run with the dev virtualenv:  .devenv/Scripts/python -m unittest discover -s tests -v"""
import io, os, random, struct, sys, time, unittest, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.protocols import dbc, canframes, isotp, uds, someip, vhal, logs, obd
DL = os.path.join(os.path.expanduser("~"), "Downloads"); REF = os.path.join(ROOT, "data", "ref")


def have(*p): return all(os.path.exists(x) for x in p)


try:
    import cantools, can
except ImportError:
    cantools = can = None
try:
    import isotp as ref_isotp
except ImportError:
    ref_isotp = None
try:
    import udsoncan
except ImportError:
    udsoncan = None
try:
    import someipy
except ImportError:
    someipy = None


class CanFd(unittest.TestCase):
    def test_dlc_table_and_struct_roundtrip(self):
        self.assertEqual([canframes.DLC_TO_LEN[i] for i in range(9, 16)], [12, 16, 20, 24, 32, 48, 64])
        self.assertEqual(canframes.len_to_dlc(33), 14)
        for fd, n in ((False, 8), (True, 64), (True, 12)):
            f = canframes.Frame(0.0, 0x123, bytes(range(n)), fd=fd, brs=fd, esi=False)
            g = canframes.parse_socketcan_struct(canframes.pack_socketcan(f))
            self.assertEqual((g.can_id, g.data, g.fd, g.brs), (0x123, bytes(range(n)), fd, fd))

    def test_candump_text(self):
        f = canframes.parse_candump("(1436509052.249713) vcan0 044#2A366C2BBA3A")
        self.assertEqual((f.can_id, f.data.hex(), f.fd), (0x44, "2a366c2bba3a", False))
        f = canframes.parse_candump("(1.5) can0 123##1AABBCC")
        self.assertEqual((f.fd, f.brs, f.data.hex()), (True, True, "aabbcc"))
        self.assertTrue(canframes.parse_candump("(2.0) can0 1F334455#DEADBEEF").extended)
        self.assertIsNone(canframes.parse_candump("not a frame"))

    @unittest.skipUnless(can and have(os.path.join(REF, "test_CanFdMessage.asc")), "needs python-can + the real CAN-FD ASC files")
    def test_real_canfd_asc_matches_python_can(self):
        for name in ("test_CanFdMessage.asc", "test_CanFdMessage64.asc"):
            path = os.path.join(REF, name)
            mine = [m for m in (canframes.parse_asc(l) for l in open(path, encoding="utf-8")) if m]
            ref = list(can.ASCReader(path))
            self.assertEqual(len(mine), len(ref))
            for m, r in zip(mine, ref):
                self.assertEqual((m.can_id, m.data, m.fd, m.brs, m.esi), (r.arbitration_id, bytes(r.data), r.is_fd, bool(r.bitrate_switch), bool(r.error_state_indicator)))


class Dbc(unittest.TestCase):
    ZIP = os.path.join(DL, "road.zip")

    @unittest.skipUnless(cantools and have(ZIP), "needs cantools + the ROAD dataset zip")
    def test_decoder_matches_cantools_on_real_road_traffic(self):
        z = zipfile.ZipFile(self.ZIP)
        text = z.read("road/signal_extractions/DBC/anonymized.dbc").decode("utf-8", "replace")
        mine = dbc.Database.load(text)
        ref = cantools.database.load_string(text, "dbc")
        self.assertEqual(len(mine.messages), len(ref.messages))
        n = 0
        with z.open("road/ambient/ambient_dyno_drive_basic_long.log") as f:
            for i, raw in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                if i >= 60000:
                    break
                fr = canframes.parse_candump(raw)
                try:
                    r = ref.get_message_by_frame_id(fr.can_id).decode(fr.data, decode_choices=False, scaling=True, allow_truncated=True)
                except KeyError:
                    continue
                m = mine.decode(fr.can_id, fr.data)
                n += 1
                for k, v in r.items():
                    self.assertAlmostEqual(float(m[k]), float(v), delta=1e-6 * max(1, abs(float(v))), msg=f"{fr.can_id:X}.{k}")
        self.assertGreater(n, 40000)

    def test_motorola_signed_and_multiplex(self):
        text = ('BO_ 100 M: 8 X\n SG_ Mot : 7|16@0+ (0.1,0) [0|0] "" X\n SG_ Neg : 16|8@1- (1,0) [0|0] "" X\n'
                'BO_ 101 Mx: 8 X\n SG_ Sel M : 0|8@1+ (1,0) [0|0] "" X\n SG_ A m0 : 8|16@1+ (1,0) [0|0] "" X\n SG_ B m1 : 8|16@1+ (2,0) [0|0] "" X\n')
        d = dbc.Database.load(text)
        self.assertAlmostEqual(d.decode(100, bytes([0x01, 0x2C, 0xFE, 0, 0, 0, 0, 0]))["Mot"], 30.0)
        self.assertEqual(d.decode(100, bytes([0, 0, 0xFE, 0, 0, 0, 0, 0]))["Neg"], -2)
        self.assertEqual(d.decode(101, bytes([0, 5, 0, 0, 0, 0, 0, 0])), {"Sel": 0, "A": 5})
        self.assertEqual(d.decode(101, bytes([1, 5, 0, 0, 0, 0, 0, 0])), {"Sel": 1, "B": 10})


class IsoTpUds(unittest.TestCase):
    def test_roundtrip_sequence_wrap_and_loss(self):
        p = bytes(range(256)) * 8
        r = isotp.Reassembler()
        out = None
        for fr in isotp.encode(p):
            out = r.feed(fr) or out
        self.assertEqual(out, p)
        r = isotp.Reassembler()
        fr = isotp.encode(bytes(range(100)))
        r.feed(fr[0])
        r.feed(fr[2])
        self.assertEqual(r.errors, 1)

    @unittest.skipUnless(ref_isotp, "needs can-isotp")
    def test_matches_independent_segmentation(self):
        rng = random.Random(5)
        for n in (1, 7, 8, 9, 62, 63, 64, 255, 1000, 4095):
            payload = bytes(rng.randrange(256) for _ in range(n))
            sent, fc = [], []
            layer = ref_isotp.TransportLayer(rxfn=lambda t: fc.pop(0) if fc else None, txfn=lambda m: sent.append(bytes(m.data)),
                                             address=ref_isotp.Address(ref_isotp.AddressingMode.Normal_11bits, rxid=0x7E8, txid=0x7E0), params={"stmin": 0, "blocksize": 0})
            layer.send(payload)
            t0, fed = time.time(), False
            while time.time() - t0 < 5:
                layer.process()
                if sent and not fed and (sent[0][0] >> 4) == 1:
                    fc.append(ref_isotp.CanMessage(arbitration_id=0x7E8, data=bytes([0x30, 0, 0]), dlc=3))
                    fed = True
                if (fed or (sent and (sent[0][0] >> 4) == 0)) and not layer.transmitting():
                    break
            r = isotp.Reassembler()
            out = None
            for f in sent:
                out = r.feed(f) or out
            self.assertEqual(out, payload, n)

    @unittest.skipUnless(udsoncan, "needs udsoncan")
    def test_dtc_response_matches_udsoncan(self):
        from udsoncan import Response
        from udsoncan.services import ReadDTCInformation
        raw = uds.encode_dtc_response([(0x1A, 0x00, 0x01, 0x09), (0x01, 0x33, 0x00, 0x2F), (0xC1, 0x00, 0x87, 0xFF)])
        resp = Response.from_payload(raw)
        resp.service = ReadDTCInformation
        ReadDTCInformation.interpret_response(resp, subfunction=ReadDTCInformation.Subfunction.reportDTCByStatusMask, standard_version=2020)
        mine = uds.decode(raw)["dtcs"]
        self.assertEqual(len(mine), len(resp.service_data.dtcs))
        for a, b in zip(resp.service_data.dtcs, mine):
            self.assertEqual((a.id, a.status.get_byte_as_int()), (b["raw"], b["status"]))

    def test_dtc_text_negative_response_and_obd(self):
        self.assertEqual((uds.dtc_text(0x01, 0x33), uds.dtc_text(0xC1, 0x00), uds.dtc_text(0x1A, 0x00)), ("P0133", "U0100", "P1A00"))
        self.assertEqual(uds.parse_dtc("P1A00"), (0x1A, 0x00))
        self.assertEqual(uds.decode(bytes([0x7F, 0x19, 0x31]))["nrc_name"], "requestOutOfRange")
        self.assertEqual(obd.decode(bytes([0x41, 0x0C, 0x1A, 0xF8]))["value"], 1726.0)
        self.assertEqual(obd.decode(bytes([0x41, 0x05, 0x7B]))["value"], 83)
        self.assertEqual(obd.decode(bytes([0x7F, 0x01, 0x12])), {"error": 18})


class SomeIp(unittest.TestCase):
    @unittest.skipUnless(someipy, "needs someipy")
    def test_header_matches_independent_implementation_both_ways(self):
        from someipy._internal.someip_header import SomeIpHeader
        from someipy._internal.someip_message import SomeIpMessage
        rng = random.Random(9)
        for _ in range(500):
            pl = bytes(rng.randrange(256) for _ in range(rng.randint(0, 200)))
            a = [rng.randrange(0x10000) for _ in range(4)]
            iv, mt, rc = rng.randrange(256), rng.choice([0, 1, 2, 0x80, 0x81]), rng.choice([0, 1, 2, 3])
            ref = SomeIpMessage(SomeIpHeader(a[0], a[1], 8 + len(pl), a[2], a[3], 1, iv, mt, rc), pl).serialize()
            p = someip.parse(ref)
            self.assertEqual((p["service"], p["method"], p["client"], p["session"], p["iface"], p["type"], p["rc"], p["payload"]), (a[0], a[1], a[2], a[3], iv, mt, rc, pl))
            self.assertEqual(someip.build(a[0], a[1], pl, a[2], a[3], iv, mt, rc), ref)

    def test_sd_offer_endpoint_stream_split_and_rejection(self):
        m = someip.parse(someip.build_sd_offer([(0x4A11, 1, 1, 0)], ("192.168.0.10", 30490)))
        sd = someip.parse_sd(m["payload"])
        self.assertEqual((m["service"], m["method"]), (0xFFFF, 0x8100))
        self.assertEqual((sd["entries"][0]["type_name"], sd["entries"][0]["service"], sd["entries"][0]["ttl"]), ("OfferService", 0x4A11, 3))
        self.assertEqual(sd["options"][0], {"type": 4, "addr": "192.168.0.10", "proto": "UDP", "port": 30490})
        msgs, rest = someip.split(someip.build(1, 0x8001, b"\x01\x02") + someip.build(2, 0x8002, b"\x03") + b"\x00\x01\x80")
        self.assertEqual((len(msgs), len(rest)), (2, 3))
        for bad in (b"\x00" * 10, struct.pack(">HHIHHBBBB", 1, 2, 3, 0, 0, 1, 1, 2, 0), struct.pack(">HHIHHBBBB", 1, 2, 8, 0, 0, 9, 1, 2, 0)):
            with self.assertRaises(someip.SomeIpError):
                someip.parse(bad)


class VhalLogsObd(unittest.TestCase):
    def test_vhal_uses_real_aosp_ids_and_units(self):
        v = vhal.Vhal()
        self.assertGreater(len(v.props), 250)
        self.assertEqual(v.props["PERF_VEHICLE_SPEED"]["id"], 0x11600207)
        self.assertEqual(v.props["IGNITION_STATE"]["id"], 0x11400409)
        self.assertAlmostEqual(v.decode(v.encode("PERF_VEHICLE_SPEED", 27.78))[0][2], 100.008, places=2)
        self.assertEqual(v.decode(v.encode("EV_BATTERY_INSTANTANEOUS_CHARGE_RATE", 118e6))[0][2], 118.0)
        self.assertEqual(v.decode(v.encode("IGNITION_STATE", 4))[1], ("state", "ignition", "ON"))
        self.assertEqual(v.decode({"prop": 1}), [])

    @unittest.skipUnless(have(os.path.join(REF, "Android_2k.log"), os.path.join(REF, "Linux_2k.log")), "needs the Loghub real logs in data/ref")
    def test_real_android_and_linux_logs(self):
        for f, want in (("Android_2k.log", 1.0), ("Linux_2k.log", 0.995)):
            lines = open(os.path.join(REF, f), encoding="utf-8", errors="replace").read().splitlines()
            ok = sum(1 for l in lines if logs.parse(l))
            self.assertGreaterEqual(ok / len(lines), want, f)

    def test_log_formats_and_severity(self):
        self.assertEqual(logs.parse("<165>1 2026-10-01T22:14:15.003Z ecu1 qnx-thermal 231 ID47 - Pump flow low")["level"], "notice")
        self.assertEqual(logs.parse("<34>Oct 11 22:14:15 mymachine su[77]: su root failed")["level"], "crit")
        self.assertEqual(logs.parse("Oct 01 13:11:02.345 thermal_mgr 3 0 pump low")["level"], "warning")
        self.assertEqual(logs.parse("03-17 16:13:38.811  1702  2395 E WindowManager: boom")["severity"], 3)
        self.assertIsNone(logs.parse("???"))

    @unittest.skipUnless(have(os.path.join(DL, "Automotive-ResearchDataSet-VIF-AEGIS.zip")), "needs the VIF-AEGIS dataset")
    def test_real_obd_samples_roundtrip_through_raw_j1979_bytes(self):
        import csv
        z = zipfile.ZipFile(os.path.join(DL, "Automotive-ResearchDataSet-VIF-AEGIS.zip"))
        n = 0
        for i, r in enumerate(csv.DictReader(io.TextIOWrapper(z.open("obdData.csv"), encoding="utf-8"))):
            if i >= 40000:
                break
            pid = int(r["obdPid"], 16)
            if pid not in obd.PIDS or pid in (0x04, 0x11, 0x2F):
                continue
            try:
                val = float(r["data"])
                raw = obd.encode(pid, val)
            except (ValueError, KeyError):
                continue
            self.assertAlmostEqual(obd.decode(raw)["value"], val, delta=max(0.5, abs(val) * 0.01))
            n += 1
        self.assertGreater(n, 10000)


if __name__ == "__main__":
    unittest.main()

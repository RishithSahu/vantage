"""CANopen (vgpt/protocols/canopen.py): behaviour, and a comparison with python-canopen (written by other people): its nodes talk to each other over a virtual CAN bus, a third listener records every frame, and our
decoder must read exactly what its SDO server, emergency producer, heartbeat producer and NMT master put on the wire."""
import os
import random
import struct
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.protocols import canopen as co

try:
    import can
    import canopen
except Exception:
    canopen = None


class Behaviour(unittest.TestCase):
    def test_function_codes(self):
        want = {0x000: ("nmt", 0), 0x080: ("sync", 0), 0x081: ("emcy", 1), 0x0FF: ("emcy", 0x7F), 0x100: ("time", 0), 0x183: ("tpdo1", 3), 0x203: ("rpdo1", 3), 0x2A5: ("tpdo2", 0x25), 0x4F1: ("tpdo4", 0x71),
                0x4C2: ("tpdo4", 0x42), 0x583: ("sdo_tx", 3), 0x603: ("sdo_rx", 3), 0x703: ("heartbeat", 3), 0x7E5: ("other", 0x65)}
        for cid, (kind, node) in want.items():
            k, n = co.classify(cid)
            self.assertEqual((k, n), (kind, node), hex(cid))

    def test_the_frames_a_node_sends_are_read_back(self):
        d = co.decode(*co.heartbeat(0x23, 5)); self.assertEqual((d["kind"], d["node"], d["state"]), ("heartbeat", 0x23, "operational"))
        d = co.decode(*co.heartbeat(0x23, 0)); self.assertEqual(d["state"], "boot-up")
        d = co.decode(*co.emcy(0x23, 0x8130, 0x11, b"P0A80")); self.assertEqual((d["code"], d["register"], d["manufacturer"]), (0x8130, 0x11, b"P0A80")); self.assertIn("heartbeat", d["text"]); self.assertEqual(d["register_bits"], ["generic", "communication"])
        d = co.decode(*co.sdo_read_response(0x23, 0x2000, 1, struct.pack("<H", 1234))); self.assertEqual((d["type"], d["index"], d["sub"], d["data"]), ("upload", 0x2000, 1, struct.pack("<H", 1234)))
        d = co.decode(*co.sdo_read_request(0x23, 0x2000, 1)); self.assertEqual((d["type"], d["index"]), ("read_request", 0x2000))
        d = co.decode(*co.sdo_abort(0x23, 0x2001, 0, 0x06020000)); self.assertEqual((d["type"], d["text"]), ("abort", "object does not exist"))
        d = co.decode(*co.nmt("start", 0x23)); self.assertEqual((d["command"], d["target"]), ("start", 0x23))

    def test_damage_raises_only_canopen_error(self):
        rnd = random.Random(5)
        for _ in range(4000):
            cid = rnd.randrange(0, 0x800); data = bytes(rnd.randrange(256) for _ in range(rnd.randint(0, 8)))
            try: co.decode(cid, data)
            except co.CanopenError: pass
        for cid, data in ((0x000, b"\x01"), (0x723, b"\x05\x05"), (0x0A3, b"\x00" * 7), (0x5A3, b"\x43"), (0x800, b"")):
            with self.assertRaises(co.CanopenError): co.decode(cid, data)


@unittest.skipIf(canopen is None, "python-canopen / python-can are not installed")
class AgainstPythonCanopen(unittest.TestCase):
    """A server node and a client over python-can's in-process bus; a listener on the same channel records the frames."""

    def setUp(self):
        self.chan = "vgpt-test-%d" % random.randrange(1 << 30)
        self.frames = []; self.sniff = can.Bus(channel=self.chan, interface="virtual", receive_own_messages=False)
        self.notifier = can.Notifier(self.sniff, [lambda m: self.frames.append((m.arbitration_id, bytes(m.data)))])
        od = canopen.ObjectDictionary()
        for idx, name, dt in ((0x2000, "Speed", canopen.objectdictionary.UNSIGNED16), (0x2001, "Temp", canopen.objectdictionary.INTEGER8), (0x2002, "Odo", canopen.objectdictionary.UNSIGNED32)):
            v = canopen.objectdictionary.Variable(name, idx, 0); v.data_type = dt; v.access_type = "rw"; od.add_object(v)
        self.server_net = canopen.Network(); self.server_net.connect(channel=self.chan, interface="virtual"); self.node = canopen.LocalNode(0x23, od); self.server_net.add_node(self.node)
        self.client_net = canopen.Network(); self.client_net.connect(channel=self.chan, interface="virtual"); self.remote = self.client_net.add_node(0x23, od)
        self.node.sdo[0x2000].raw = 1234; self.node.sdo[0x2001].raw = -7; self.node.sdo[0x2002].raw = 1234567

    def tearDown(self):
        self.notifier.stop(); self.sniff.shutdown()
        for n in (self.client_net, self.server_net): n.disconnect()

    def decoded(self):
        out = []
        for cid, data in list(self.frames):
            try: out.append(co.decode(cid, data))
            except co.CanopenError: self.fail("our decoder refused a frame python-canopen sent: %x %s" % (cid, data.hex()))
        return out

    def test_sdo_reads_of_one_to_four_bytes(self):
        self.assertEqual(self.remote.sdo[0x2000].raw, 1234); self.assertEqual(self.remote.sdo[0x2001].raw, -7); self.assertEqual(self.remote.sdo[0x2002].raw, 1234567); time.sleep(0.2)
        d = self.decoded(); up = {(x["index"], x["sub"]): x["data"] for x in d if x.get("type") == "upload"}; req = [x for x in d if x.get("type") == "read_request"]
        self.assertEqual(len(req), 3); self.assertEqual(struct.unpack("<H", up[(0x2000, 0)])[0], 1234); self.assertEqual(struct.unpack("<b", up[(0x2001, 0)])[0], -7); self.assertEqual(struct.unpack("<I", up[(0x2002, 0)])[0], 1234567)

    def test_an_sdo_abort_names_the_reason(self):
        with self.assertRaises(Exception): self.remote.sdo.upload(0x3000, 0)
        time.sleep(0.2); ab = [x for x in self.decoded() if x.get("type") == "abort"]
        self.assertTrue(ab and ab[0]["text"] == "object does not exist" and ab[0]["index"] == 0x3000)

    def test_emergency_heartbeat_nmt_and_sync(self):
        self.node.emcy.send(0x4210, 0x09, b"\x01\x02\x03\x04\x05"); self.node.nmt.start_heartbeat(50); self.client_net.nmt.send_command(0x01); self.client_net.sync.transmit(); time.sleep(0.4); self.node.nmt.stop_heartbeat()
        d = self.decoded(); em = [x for x in d if x["kind"] == "emcy"]; hb = [x for x in d if x["kind"] == "heartbeat"]
        self.assertTrue(em and (em[0]["code"], em[0]["register"], em[0]["manufacturer"]) == (0x4210, 0x09, b"\x01\x02\x03\x04\x05")); self.assertEqual(em[0]["text"], "device temperature"); self.assertEqual(em[0]["node"], 0x23)
        self.assertTrue(len(hb) >= 3 and all(x["node"] == 0x23 for x in hb)); self.assertTrue(any(x["kind"] == "nmt" and x["command"] == "start" and x["target"] == 0 for x in d)); self.assertTrue(any(x["kind"] == "sync" for x in d))

    def test_what_we_build_the_library_reads(self):
        """Our frames, injected on the bus, are accepted by python-canopen's own client: an SDO read answered by our encoder."""
        self.server_net.disconnect()                                   # no server: we play it
        bus = can.Bus(channel=self.chan, interface="virtual")
        import threading

        def serve():
            end = time.time() + 3
            while time.time() < end:
                m = bus.recv(0.1)
                if m is not None and m.arbitration_id == 0x623 and m.data[0] == 0x40:
                    idx, sub = struct.unpack("<HB", bytes(m.data[1:4])); cid, data = co.sdo_read_response(0x23, idx, sub, struct.pack("<H", 4321)); bus.send(can.Message(arbitration_id=cid, data=data, is_extended_id=False)); return
        t = threading.Thread(target=serve, daemon=True); t.start(); time.sleep(0.1)
        try: self.assertEqual(self.remote.sdo[0x2000].raw, 4321)
        finally: t.join(4); bus.shutdown()


if __name__ == "__main__":
    unittest.main()

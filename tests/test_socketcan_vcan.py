"""The receive-only SocketCAN collector on a REAL Linux CAN interface (a virtual one, `vcan`), with the frames sent by another process through the kernel: nothing here is a stand-in socket.

    python -m pytest tests/test_socketcan_vcan.py        (Linux with the vcan module and passwordless sudo; otherwise every test is skipped, never passed)

What it shows: frames written by another process on the interface arrive in the collector bit-exact (classic and FD, standard and extended identifiers, remote and error frames are not mistaken for data); a J1939
multi-packet DM1 sent on the bus is rebuilt and decoded by the product's router into the right fault events; and the collector's socket cannot transmit (the wrapper has no send, and nothing it does puts a frame on the bus)."""
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

IFACE = "vcantst0"
LINUX = sys.platform.startswith("linux") and hasattr(socket, "AF_CAN")

SENDER = r'''
import socket, struct, sys, time
iface = sys.argv[1]; s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW); s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1); s.bind((iface,))
EFF = 0x80000000
def classic(cid, data, ext=False): s.send(struct.pack("=IB3x8s", cid | (EFF if ext else 0), len(data), bytes(data).ljust(8, b"\0")))
def fd(cid, data, ext=False): s.send(struct.pack("=IBB2x64s", cid | (EFF if ext else 0), len(data), 0, bytes(data).ljust(64, b"\0")))
for line in sys.stdin:
    kind, cid, ext, hexdata = line.split()
    (fd if kind == "fd" else classic)(int(cid, 16), bytes.fromhex(hexdata), ext == "1"); time.sleep(0.002)
'''


def sudo(*a): return subprocess.run(("sudo", "-n") + a, capture_output=True, text=True)


@unittest.skipUnless(LINUX and shutil.which("ip"), "SocketCAN needs Linux")
class RealInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sudo("modprobe", "vcan"); sudo("ip", "link", "del", IFACE)
        r = sudo("ip", "link", "add", "dev", IFACE, "type", "vcan")
        if r.returncode: raise unittest.SkipTest("cannot create a vcan interface: " + r.stderr.strip()[:100])
        sudo("ip", "link", "set", "up", IFACE)

    @classmethod
    def tearDownClass(cls): sudo("ip", "link", "del", IFACE)

    def send(self, rows):
        p = subprocess.run([sys.executable, "-c", SENDER, IFACE], input="\n".join(rows) + "\n", capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)

    def collect(self, rows, wait=1.5):
        from vgpt.live import collect
        got = []; c = collect.SocketCanCollector(lambda ts, bus, f: got.append((bus, f)), IFACE); c.start(); time.sleep(0.3)
        self.send(rows); t0 = time.time()
        while time.time() - t0 < wait and len(got) < len(rows): time.sleep(0.05)
        c.stop(); c.join(2)
        return got, c

    def test_frames_arrive_bit_exact(self):
        rows = ["classic 123 0 0102030405060708", "classic 7E8 0 03410C1A", "classic 18FEF100 1 FFFF000000000000", "fd 1F0 0 " + "".join("%02X" % i for i in range(64)), "fd 18DAF110 1 " + "AA" * 12]
        got, c = self.collect(rows); self.assertEqual(len(got), len(rows), c.stats())
        self.assertEqual([(f.can_id & 0x1FFFFFFF, f.extended, f.fd, f.data.hex().upper()) for _, f in got],
                         [(0x123, False, False, "0102030405060708"), (0x7E8, False, False, "03410C1A"), (0x18FEF100, True, False, "FFFF000000000000"), (0x1F0, False, True, "".join("%02X" % i for i in range(64))), (0x18DAF110, True, True, "AA" * 12)])
        self.assertEqual(c.errors, 0)

    def test_a_multi_packet_dm1_on_the_bus_becomes_fault_events(self):
        from vgpt.protocols import j1939 as J
        from vgpt.vehicle.profile import Profile
        from vgpt.vehicle.router import ProfileRouter
        r = ProfileRouter(Profile.load("rhino_diesel")); frames = J.encode_tp(J.PGN_DM1, J.encode_dm1([(110, 0, 2), (100, 1, 1)], amber=1), 0x00)
        rows = ["classic %X 1 %s" % (cid & 0x1FFFFFFF, d.hex()) for cid, d in frames]
        got, c = self.collect(rows); self.assertEqual(len(got), len(rows)); ev = []
        for t, (bus, f) in enumerate(got): ev += [o for o in r.decode(bus, f, 100.0 + 0.01 * t) if o[0] == "event"]
        self.assertEqual(sorted(e[1] for e in ev), ["DTC_SET", "DTC_SET"]); self.assertTrue(any("SPN 110 FMI 0" in e[3] for e in ev))

    def test_the_collector_cannot_put_a_frame_on_the_bus(self):
        from vgpt.live import collect
        c = collect.SocketCanCollector(lambda *a: None, IFACE)
        for name in ("send", "sendall", "sendto", "sendmsg", "setsockopt", "raw"): self.assertFalse(hasattr(c.sock, name), name)
        spy = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW); spy.bind((IFACE,)); spy.settimeout(0.5)
        try:
            c.start(); time.sleep(0.5)
            with self.assertRaises(socket.timeout): spy.recv(72)                                                          # while the collector runs, nothing it does is visible on the wire
        finally: c.stop(); c.join(2); spy.close()


if __name__ == "__main__": unittest.main()

"""Writes reference captures with python-can (an independent implementation of the file formats) so that vgpt/ingest/readers.py is tested against files it did not write itself.

    pip install python-can      (a throw-away environment; python-can is NOT a dependency of the application)
    python tests/fixtures/make_reference_captures.py

The drive is the one of tests/capture_factory.py (J1939 traffic of a heavy vehicle), 20 s, plus one CAN FD frame in the BLF."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import can  # noqa: E402
import capture_factory as F  # noqa: E402

OUT = os.path.join(HERE, "capture"); os.makedirs(OUT, exist_ok=True)
frames = sorted(F.can_all(20))
msgs = [can.Message(timestamp=ts, arbitration_id=cid & 0x1FFFFFFF, is_extended_id=bool(cid & 0x80000000), data=data, channel=1) for ts, cid, data in frames]
for name, cls in (("ref_j1939.blf", can.BLFWriter), ("ref_j1939.asc", can.ASCWriter), ("ref_j1939.trc", can.TRCWriter), ("ref_j1939_candump.log", can.CanutilsLogWriter)):
    path = os.path.join(OUT, name)
    with cls(path) as w:
        for m in msgs: w.on_message_received(m)
    print("wrote", name, os.path.getsize(path))
fd = can.Message(timestamp=F.T0 + 21.0, arbitration_id=0x18FEF100, is_extended_id=True, is_fd=True, bitrate_switch=True, data=bytes(range(12)), channel=1)
with can.BLFWriter(os.path.join(OUT, "ref_fd.blf")) as w:
    w.on_message_received(fd)
    w.on_message_received(can.Message(timestamp=F.T0 + 21.5, arbitration_id=0x123, is_extended_id=False, data=bytes(range(8)), channel=1))
print("wrote ref_fd.blf")

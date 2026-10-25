"""VLink: the one-way stream a car (or anything that holds a vehicle's data) sends to Vantage over a pipe.

The format knows nothing about vehicles, buses or protocols.  A record is a moment, a channel number and a block of bytes - whatever the car puts on a channel is for Vantage to recognise
(vgpt/pipe/sniff.py), not for this layer to describe.  Because nothing is promised about the content, the same link carries a Toyota's CAN frames, a Tesla's, an Android Automotive property
update, a SOME/IP notification, a log line or a camera picture, and a car that Vantage has never seen needs no change here.

    SYNC 2 bytes (A7 1C) | LENGTH u32 (of the body) | BODY: TIME u64 (microseconds, 0 = not stamped) | CHANNEL u16 | FLAGS u8 | PAYLOAD | CRC-32 u32 (over LENGTH and BODY)

    * the CRC and the sync marker let a reader that joins mid-stream, or meets damage, skip to the next good record (it counts what it skipped; it never guesses)
    * channel numbers are the sender's own: Vantage never assumes what channel 3 is; the sender MAY describe it with a control record (channel 0xFFFF, a JSON object), and the receiver treats that only as a hint
    * TIME is the sender's clock.  A value that looks like a calendar time (after 2001) is taken as one; a small value is taken as 'since the car woke up' and aligned by the receiver; 0 means 'stamp it when it arrives'
    * a link is a one-way pipe: this module (the codec and the reader) is all Vantage imports; the writer lives with the car (vgpt/car/writer.py), so Vantage's code cannot transmit

Plain stdlib; used by the car side (encode) and by Vantage (Parser, PipeReader)."""
import json
import os
import struct
import threading
import zlib

SYNC = b"\xa7\x1c"
CONTROL = 0xFFFF
MAX_PAYLOAD = 1 << 20
HEAD = struct.Struct("<2sI")
BODY = struct.Struct("<QHB")
CRC = struct.Struct("<I")
FLAG_FIRST = 0x01                                                                         # the first record of a connection (the receiver starts a fresh session on it)
EPOCH_FLOOR_US = 1_000_000_000 * 1_000_000                                                # 2001-09-09: below this a TIME is not a calendar time


class LinkClosed(OSError):
    """The other end of the pipe is gone."""


class Record:
    __slots__ = ("ts", "channel", "payload", "flags")

    def __init__(self, ts, channel, payload, flags=0): self.ts, self.channel, self.payload, self.flags = ts, channel, payload, flags
    def __repr__(self): return f"Record(ts={self.ts}, ch={self.channel}, {len(self.payload)} bytes)"

    def control(self):
        """The JSON object of a control record, or None."""
        if self.channel != CONTROL: return None
        try: o = json.loads(self.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError): return None
        return o if isinstance(o, dict) else None


def encode(ts, channel, payload, flags=0):
    """One record as bytes.  `ts` is seconds (float) or None."""
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD: raise ValueError("a record may carry at most 1 MiB")
    if not 0 <= int(channel) <= 0xFFFF: raise ValueError("channel out of range")
    us = 0 if ts is None else max(0, int(round(float(ts) * 1e6)))
    body = BODY.pack(us, int(channel), int(flags) & 0xFF) + payload
    head = HEAD.pack(SYNC, len(body))
    return head + body + CRC.pack(zlib.crc32(head[2:] + body) & 0xFFFFFFFF)


class Parser:
    """Incremental decoder: feed it whatever bytes arrived, get the records that are complete.  Garbage between records is skipped and counted."""
    def __init__(self): self.buf = bytearray(); self.records = 0; self.skipped = 0; self.bad_crc = 0; self.resyncs = 0

    def feed(self, data):
        self.buf += data; out = []; buf = self.buf; i = 0; n = len(buf); mark = HEAD.size
        while True:
            j = buf.find(SYNC, i)
            if j < 0:
                keep = 1 if n and buf[-1] == SYNC[0] else 0                                  # a SYNC may be split across two reads
                self.skipped += max(0, n - i - keep); i = n - keep; break
            if j > i: self.skipped += j - i; self.resyncs += 1
            i = j
            if n - i < HEAD.size: break
            ln = struct.unpack_from("<I", buf, i + 2)[0]
            if ln < BODY.size or ln > MAX_PAYLOAD + BODY.size: self.skipped += 1; i += 1; continue
            end = i + HEAD.size + ln + CRC.size
            if n < end: break
            if zlib.crc32(bytes(buf[i + 2:end - CRC.size])) & 0xFFFFFFFF != CRC.unpack_from(buf, end - CRC.size)[0]:
                self.bad_crc += 1; self.skipped += 1; i += 1; continue
            us, ch, fl = BODY.unpack_from(buf, i + HEAD.size)
            out.append(Record(None if us == 0 else us / 1e6, ch, bytes(buf[i + HEAD.size + BODY.size:end - CRC.size]), fl)); self.records += 1; i = end
        if i: del buf[:i]
        return out

    def stats(self): return {"records": self.records, "skipped_bytes": self.skipped, "bad_crc": self.bad_crc, "resyncs": self.resyncs}


class PipeReader(threading.Thread):
    """Vantage's end: reads a descriptor until the other end closes and hands every record to `on_record`.  It can only read."""
    def __init__(self, fd, on_record, on_close=None, name="vlink-reader"):
        super().__init__(daemon=True, name=name); self.fd, self.on_record, self.on_close = fd, on_record, on_close; self.parser = Parser(); self.stop_flag = threading.Event(); self.bytes = 0; self.error = ""

    def run(self):
        try:
            while not self.stop_flag.is_set():
                try: chunk = os.read(self.fd, 1 << 16)
                except OSError as e: self.error = str(e); break
                if not chunk: break
                self.bytes += len(chunk)
                for r in self.parser.feed(chunk): self.on_record(r)
        finally:
            try: os.close(self.fd)
            except OSError: pass
            if self.on_close: self.on_close(self)

    def stop(self): self.stop_flag.set()


def time_base(first_ts, now):
    """How to turn the sender's TIME into calendar seconds: returns (offset, kind).  A calendar-looking TIME needs no offset; a small one is aligned so that its first record lands at `now`."""
    if first_ts is None: return 0.0, "arrival"
    if first_ts * 1e6 >= EPOCH_FLOOR_US: return 0.0, "calendar"
    return now - first_ts, "relative"

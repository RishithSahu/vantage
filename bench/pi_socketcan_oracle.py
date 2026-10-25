"""Independent protocol oracle on Linux: the KERNEL's ISO-TP (can-isotp) and J1939 (can-j1939) stacks vs this project's own decoders/encoders, over a vcan interface.

   sudo python bench/pi_socketcan_oracle.py            (needs: modprobe vcan can_raw can_isotp can_j1939; root for `ip link add`)

Direction A (kernel -> ours): a kernel socket sends a message of N bytes; a raw socket records the exact frames the kernel put on the bus; our Reassembler must return the same N bytes.
Direction B (ours -> kernel): our encoder's frames are written to the bus with a raw socket; a kernel socket must deliver the same N bytes.
The kernel is not our code, so agreement is evidence, not self-consistency.  Writes results/protocol_oracle_pi.json."""
import json, os, random, select, socket, struct, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.protocols import isotp, j1939

IF = "vcan0"; FMT = "=IB3x8s"; random.seed(7)


def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)


def setup():
    for m in ("vcan", "can_raw", "can_isotp", "can_j1939"): sh(f"modprobe {m}")
    sh(f"ip link del {IF}"); r = sh(f"ip link add dev {IF} type vcan && ip link set up {IF}")
    if r.returncode: raise SystemExit("cannot create vcan: " + r.stderr)


def raw():
    s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW); s.bind((IF,))
    s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
    try: s.setsockopt(socket.SOL_SOCKET, 33, 8 << 20)      # SO_RCVBUFFORCE: a monitor that drops frames would fake a mismatch
    except OSError: s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    return s


def raw_send(s, cid, data, ext=False):
    s.send(struct.pack(FMT, cid | (socket.CAN_EFF_FLAG if ext else 0), len(data), bytes(data).ljust(8, b"\0")))


def drain(s, t=0.05):
    out = []
    while select.select([s], [], [], t)[0]:
        f = s.recv(16); cid, dlc, d = struct.unpack(FMT, f); out.append((cid & socket.CAN_EFF_MASK, d[:dlc]))
    return out


def payload(n): return bytes(random.randrange(256) for _ in range(n))


# ---------------- ISO-TP ----------------
def isotp_sock(rx, tx):
    s = socket.socket(socket.PF_CAN, socket.SOCK_DGRAM, socket.CAN_ISOTP); s.bind((IF, rx, tx)); return s


def isotp_kernel_to_ours(sizes):
    res = []
    for n in sizes:
        mon = raw(); rcv = isotp_sock(0x7E0, 0x7E8); snd = isotp_sock(0x7E8, 0x7E0)   # sender writes id 0x7E0; rcv answers flow control with 0x7E8
        p = payload(n); snd.send(p); time.sleep(0.02); frames = [d for cid, d in drain(mon) if cid == 0x7E0]
        got = kernel = None; r = isotp.Reassembler(); t = 0.0
        for d in frames:
            t += 0.001; out = r.feed(d, now=t)
            if out is not None: got = out
        rcv.settimeout(0.5)
        try: kernel = rcv.recv(5000)
        except Exception: kernel = None
        res.append({"n": n, "frames": len(frames), "ours_equals": got == p, "kernel_roundtrip": kernel == p, "errors": r.errors})
        for s in (mon, rcv, snd): s.close()
    return res


def isotp_ours_to_kernel(sizes):
    res = []
    for n in sizes:
        tx = raw(); rcv = isotp_sock(0x7E0, 0x7E8); rcv.settimeout(1.0); p = payload(n)
        for d in isotp.encode(p): raw_send(tx, 0x7E0, d); time.sleep(0.0005)
        try: got = rcv.recv(5000)
        except Exception: got = None
        res.append({"n": n, "kernel_equals": got == p}); tx.close(); rcv.close()
    return res


# ---------------- J1939 ----------------
def j_sock(addr, pgn=socket.J1939_NO_PGN):
    s = socket.socket(socket.PF_CAN, socket.SOCK_DGRAM, socket.CAN_J1939); s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1); s.bind((IF, socket.J1939_NO_NAME, pgn, addr)); return s


def j1939_kernel_to_ours(sizes, dest):
    res = []; PGN = 0xFEF1 if dest == 0xFF else 0xEF00        # PDU2 (broadcast) / PDU1 (destination specific)
    for n in sizes:
        mon = raw(); snd = j_sock(0x25); rcv = j_sock(0x30 if dest != 0xFF else 0x31)
        if dest != 0xFF: rcv.close(); rcv = j_sock(dest)
        p = payload(n)
        try: snd.sendto(p, (IF, socket.J1939_NO_NAME, PGN | (0 if dest == 0xFF else 0), dest))
        except OSError as e: res.append({"n": n, "error": str(e)}); mon.close(); snd.close(); rcv.close(); continue
        time.sleep(1.5 if n > 500 else 0.4 if n > 8 else 0.05); frames = [(c, d) for c, d in drain(mon, 0.1)]
        r = j1939.Reassembler(); msgs = []; t = 0.0
        for c, d in frames: t += 0.001; msgs += r.feed(c, d, now=t)
        mine = [m for m in msgs if m[3] == p] if msgs else []
        rcv.settimeout(0.3)
        try: k = rcv.recv(2000)
        except Exception: k = None
        res.append({"n": n, "frames": len(frames), "ours_equals": bool(mine), "pgn_ok": bool(mine) and (mine[0][0] & 0x3FFFF) == (PGN & 0x3FFFF) or bool(mine) and (mine[0][0] >> 8) == (PGN >> 8),
                    "kernel_delivered": k == p, "errors": r.errors}); [s.close() for s in (mon, snd, rcv)]
    return res


def j1939_ours_to_kernel(sizes):
    """Our BAM announce + data packets (global destination) -> the kernel's J1939 receiver."""
    res = []
    for n in sizes:
        tx = raw(); rcv = j_sock(0x31, 0xFEF1); rcv.settimeout(1.5); p = payload(n)
        if n <= 8:
            raw_send(tx, j1939.encode_id(6, 0xFEF1, 0x25), p, ext=True)
        else:
            for cid, d in j1939.encode_tp(0xFEF1, p, 0x25): raw_send(tx, cid, d, ext=True); time.sleep(0.02)
        try: got = rcv.recv(2000)
        except Exception: got = None
        res.append({"n": n, "kernel_equals": got == p}); tx.close(); rcv.close()
    return res


def main():
    setup(); out = {"kernel": os.uname().release, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    sizes = [1, 6, 7, 8, 9, 14, 15, 62, 63, 100, 255, 256, 1000, 4095, 4096, 5000]
    out["isotp_kernel_to_ours"] = isotp_kernel_to_ours(sizes)
    out["isotp_ours_to_kernel"] = isotp_ours_to_kernel([n for n in sizes if n <= 4095])
    js = [1, 8, 9, 10, 17, 100, 500, 1785]
    out["j1939_bam_kernel_to_ours"] = j1939_kernel_to_ours(js, 0xFF)
    out["j1939_rts_kernel_to_ours"] = j1939_kernel_to_ours([n for n in js if n > 8] + [8], 0x30)
    out["j1939_bam_ours_to_kernel"] = j1939_ours_to_kernel(js)
    def ok(rows, key): return sum(1 for r in rows if r.get(key)), len(rows)
    out["summary"] = {"isotp_kernel_to_ours": ok(out["isotp_kernel_to_ours"], "ours_equals"), "isotp_ours_to_kernel": ok(out["isotp_ours_to_kernel"], "kernel_equals"),
                      "j1939_bam_kernel_to_ours": ok(out["j1939_bam_kernel_to_ours"], "ours_equals"), "j1939_rts_kernel_to_ours": ok(out["j1939_rts_kernel_to_ours"], "ours_equals"),
                      "j1939_bam_ours_to_kernel": ok(out["j1939_bam_ours_to_kernel"], "kernel_equals")}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "protocol_oracle_pi.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    for k in out:
        if isinstance(out[k], list):
            for r in out[k]:
                if not (r.get("ours_equals") or r.get("kernel_equals")): print("MISMATCH", k, r)


if __name__ == "__main__":
    main()

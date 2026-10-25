# Protocol evidence: what was checked, against what, and how to repeat it

The case study asks for data from several in-vehicle protocols. Vantage reads thirteen families. For each, this page says which independent implementation (written by somebody else) it was compared with, and what is *not* covered.

| # | Family | Module | Compared with | What was compared |
|---|---|---|---|---|
| 1 | CAN, CAN-FD (DBC) | `vgpt/protocols/dbc.py`, `canframes.py` | **cantools** | every signal of every message of 77 real-vehicle DBC files (opendbc), random payloads: bit order (Intel/Motorola), sign, scale, multiplexing, value tables |
| 2 | SAE J1939 transport and diagnostics | `vgpt/protocols/j1939.py` | **can-j1939** (python-can-j1939) | 29-bit identifier fields on 3000 random ids; messages of 9, 15, 64, 255 and 1785 bytes sent by the reference as broadcast announce (BAM) and rebuilt by us; messages built by us received by the reference; DM1 fault records |
| 3 | OBD-II over ISO-TP | `isotp.py`, `obd.py` | **python-can-isotp**, SAE J1979 formulas | segmentation and reassembly of every length 1-4095; PID formulas |
| 4 | UDS (ISO 14229) | `uds.py` | **udsoncan** | service and negative-response codes (65-entry table) |
| 5 | SOME/IP and Service Discovery | `someip.py` | **scapy** | header, both directions |
| 6 | Android Vehicle HAL | `vhal.py`, `vhalparcel.py` | AOSP `VehicleProperty` table | property ids and units |
| 7 | NMEA 0183 GNSS | `nmea.py` | **pynmea2** | our parser on its example sentences and its parser on ours |
| 8 | Logs (logcat, syslog RFC 3164 / 5424, QNX slog2) | `logs.py` | the published line formats | parse tests with fixtures |
| 10 | LIN 2.x (body network) | `vgpt/protocols/lin.py` | **ldfparser**; the LIN tables of protected identifiers | the LIN Description File reader against ldfparser on up to 600 random payloads (200 for each of 3 frames of 3 signals each) with physical and logical encodings, both directions; PID parity of all 64 identifiers (10 published values), classic and enhanced checksum, damage and fuzz |
| 11 | DoIP (ISO 13400-2, UDS over automotive Ethernet) | `vgpt/protocols/doip.py` | **python-doipclient** and **scapy** | our bytes equal doipclient's for routing activation (with and without the OEM field), diagnostic message and acknowledgement, generic NACK, alive check, vehicle announcement with the VIN; scapy parses what we write and we parse what scapy builds; a TCP stream cut at 50 random places gives the same frames; 3,000 random blocks raise only `DoipError` |
| 12 | CANopen (CiA 301: NMT, SYNC, EMCY, heartbeat, SDO, PDO) | `vgpt/protocols/canopen.py` | **python-canopen** (its nodes talk over python-can's virtual bus; a third listener records every frame) | SDO reads of 1, 2 and 4 bytes and an abort ("object does not exist"), an emergency with error register and manufacturer bytes, a heartbeat producer, an NMT start and a SYNC are decoded exactly; python-canopen's own client accepts the SDO responses we build |
| 13 | FlexRay (specification 2.1: frame header, header CRC-11, frame CRC-24) | `vgpt/protocols/flexray.py` | **none available offline** - see below | specification parameters, a second table-driven CRC written separately, the algebraic properties of a CRC (zero remainder, every single-bit error of a 12-word frame detected at all its bit positions), round trip of 500 random frames with the sync / startup / null indicators |
| 9 | Camera (JPEG, ADAS object records) | `vgpt/camera.py` | decode and schema tests | not an independent-implementation check |

The reference libraries are test-only; the product does not import them, and a test that needs one is *skipped* when it is missing (it never passes by default).

## The Linux kernel as the oracle (Raspberry Pi 4, 2026-10-23)

The libraries above are written by other people but run in the same Python process. The Pi test `bench/pi_socketcan_oracle.py` uses something that shares no code with this project at all: **the Linux kernel's own can-isotp and can-j1939 stacks** (kernel 6.18, `vcan0`).

* kernel ISO-TP sends a message, a raw socket records the exact frames it put on the bus, our `isotp.Reassembler` must return the same bytes: 16 of 16 lengths (1 to 5000 bytes, including the 4095/4096 boundary where the 32-bit escape starts);
* our `isotp.encode` frames are written to the bus, the kernel receiver must deliver the same bytes: 14 of 14 (1 to 4095);
* kernel J1939 broadcast (BAM) of 1 to 1785 bytes, rebuilt by our `j1939.Reassembler`: 8 of 8;
* kernel J1939 connection mode (RTS/CTS, the kernel answers its own clear-to-send) of 8 to 1785 bytes, rebuilt by us from the frames on the bus: 7 of 7;
* our `j1939.encode_tp` BAM frames received by the kernel: 8 of 8.

**53 of 53** (`results/protocol_oracle_pi.json`). A first run had failures that were mistakes of the test (swapped ISO-TP addresses, no `SO_BROADCAST`, a monitor socket whose buffer overflowed on a 585-frame transfer); none was a decoder fault, and each was fixed in the test, not in the decoder.

## Four protocols added on 2026-10-23, and what each proves

Each is wired through the same path as the older ones: the profile (a source with its description file), the emitter (the car puts the bus on the wire in the form that bus has), the transmitter, the sniffer (the block's structure alone decides what it is: LIN needs the sync byte, valid parity and a checksum that agrees; DoIP the version, its inverse and a length that matches; FlexRay two CRCs), the fingerprint (a decoder file that expects a bus the link never carries scores lower: the Nova car, which shares its CAN with the Meridian car, was first recognised as Meridian's twin until this was added), the router and the ledger. `tests/test_protocol_cars.py` drives each demonstration car through the pipe and checks that the quantities that only travel on the new bus are in its database; `tests/test_pipe.py` recognises all 49 car models from their traffic alone (22 tests, 51 subtests).

**FlexRay is the weakest evidence on this page.** No FlexRay implementation written by somebody else could be installed or found offline, so the other protocols' "compared with an independent library" does not apply. What is shown is the specification's polynomials and start values, a second CRC computed by a different method, and the properties a correct CRC must have. A capture from a real FlexRay cluster, or a FIBEX file, has not been read. The cluster description (`chassis.fx.json`) is JSON written for the test bench, not FIBEX.

**On the Pi:** `bench/pi_socketcan_oracle.py` compares our ISO-TP and J1939 with the Linux kernel's own stacks on a `vcan` interface: 53 of 53 transfers agree in both directions (`results/protocol_oracle_pi.json`).

## J1939 in detail

Messages longer than 8 bytes travel in pieces. `vgpt/protocols/j1939.py` rebuilds them without ever transmitting (no clear-to-send, no acknowledgement):

* broadcast announce and data packets (BAM), up to 1785 bytes;
* connection-mode transfers (RTS/CTS), reconstructed from the data packets the two ECUs already exchange;
* the extended transport protocol (ETP) with data-packet offsets, bounded at 1 MB to protect the 2 GB target;
* one transfer per (source, destination) pair, so interleaved senders do not mix;
* anomalies counted and never fatal: out-of-order packets, orphan data packets, a transfer that stops (timeout), impossible sizes, aborts, restarts, duplicate packets.

On top of it: DM1 (active faults: lamp status, SPN, FMI with its text, occurrence count, conversion method) becomes one `DTC_SET` event per new fault (severity *critical* when the red stop lamp is on) and one `DTC_CLEAR` event when it disappears; DM2 is read but is not a present condition; vehicle identification (PGN 65260) and component identification (65259) are stored per source address.

Through the product router (`tests/test_j1939.py::ThroughTheVehicleRouter`): a three-fault DM1 sent as a broadcast transfer raises three events once, the same DM1 again raises none, and a DM1 with fewer faults raises the clears.

## Robustness

`hypothesis` property tests and 30 000 random frames per decoder: only the documented errors are raised, no hang, and the number of open transfers stays bounded.

## What is not covered

* DLT (the AUTOSAR log and trace protocol) is not implemented: no reference library could be found to check it against. DoIP, LIN, CANopen and FlexRay are (above).
* Real SocketCAN on a physical bus is not tested (it is tested on `vcan` against the kernel stacks, above); frames are read from files, a pipe or a simulator that writes exactly what a SocketCAN read returns (struct `can_frame` / `canfd_frame`).
* The camera path has no second implementation to compare with.
* Real captures from customer vehicles were not available; the 77 DBC files describe real vehicles but the traffic in the tests is generated.

## Repeat it

```bash
# an environment with the references (cantools, scapy, pynmea2, udsoncan, python-can-isotp, can-j1939, python-can, hypothesis; on Windows also pywin32 for can-j1939)
python -m pytest tests/test_protocol_oracles.py tests/test_j1939.py tests/test_protocols.py
```

Result on 2026-10-23: 43 passed, 2 skipped (skips are tests whose optional reference is absent).

"""Real automotive protocol decoders (stdlib only; they run in the 2 GB in-car footprint).
  dbc       CAN database (DBC) parser + signal extraction (Intel/Motorola, signed, scaled, float, multiplexed, value tables)
  canframes CAN / CAN-FD frames: Vector ASC, SocketCAN candump text, SocketCAN binary structs, CAN-FD DLC table
  isotp     ISO 15765-2 transport (single / first / consecutive / flow-control frames) reassembly
  uds       ISO 14229 diagnostic services (ReadDTCInformation, ReadDataByIdentifier, negative responses) + DTC text
  someip    AUTOSAR SOME/IP header, payload structs, SOME/IP-SD entries/options
  vhal      Android Vehicle HAL property model (real property IDs / units generated from the AOSP VehicleProperty.aidl)
  logs      Android logcat, Linux syslog (RFC 3164 / 5424) and QNX slog2info text lines
  obd       SAE J1979 / ISO 15031 OBD-II mode 01 PID decoding
Each module is validated against an independent open-source implementation and/or real recordings (see tests/ and bench/)."""

"""The live side of the platform: collectors that feed raw traffic into the single-writer runtime, and a virtual car that produces that traffic.

    collect.py    transports (UDP / TCP / file tail / candump / SocketCAN) -> (ts, bus, payload) -> Runtime.feed
    runtime.py    Runtime: profile router + ingestor + lake + ledger + context freezing, with per-source health and a bus monitor
    road.py       a drivable route between two places (heading, curvature, speed limits, junctions, traffic lights)
    car.py        VirtualCar: kinematics, powertrain (battery-electric or diesel), thermal, tyres, ADAS, camera, 12 V, charging, injectable faults
    sim.py        Simulation: car + the profile's Emitter + the runtime on a clock (real time or accelerated)"""

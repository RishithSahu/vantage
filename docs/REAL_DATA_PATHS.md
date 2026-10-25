# How vehicle data really reaches an in-car computer, and how Vantage simulates it with no network

Written 2026-10-21 for the owner's instruction: *"there should be no http or networking protocols at all ... fully offline ... but the app must still simulate how actual data would come to Vantage in a car"*.  This page is the research behind the offline design (`docs/OFFLINE_DESIGN.md`).  Everything marked **(AOSP docs)** was read in the Android Open Source Project documentation on 2026-10-21; **(QNX docs)**, **(KUKSA / COVESA)** likewise; what could not be verified is said so.

## 1. The one rule the research supports

In a real car the data an app sees does **not** arrive over the internet or over HTTP.  It arrives over the vehicle's own buses and then over **local inter-process calls** inside the computer.  So "no network protocol at all" costs nothing in realism: it is how the in-car path already works.

## 2. The physical chain, the same in every car

```
sensors / ECUs  ->  vehicle buses  ->  gateway or central computer  ->  the in-car computer (infotainment / HPC)
                    CAN / CAN-FD, LIN, FlexRay,                          where Vantage runs
                    100BASE-T1 Ethernet (SOME/IP)
```

The bus frames are raw bytes with an agreed layout (a DBC file for CAN, a service description for SOME/IP).  What differs between car operating systems is only **how the in-car computer's OS hands those values to an application**.

## 3. Android Automotive OS (AOSP docs)

| Layer | What it is | Notes |
|---|---|---|
| **Vendor VHAL implementation** (`IVehicleHardware`) | the car maker's code that talks to the vehicle's MCU / ECUs over whatever link the board has (CAN, SPI, UART, Ethernet) | "vendors must add vehicle bus-specific logic": this is the only layer that touches the bus |
| **DefaultVehicleHal** | the generic upper half of the Vehicle HAL; implements the AIDL interface `android.hardware.automotive.vehicle.IVehicle/default` | methods `getValues`, `setValues`, `subscribe`, `unsubscribe`; callback `IVehicleCallback.onPropertyEvent`; must not block binder threads |
| **CarService** | Java system service, the VHAL's client | does the (de)serialisation; keeps the permission model |
| **CarPropertyManager** | the API an app uses | `subscribePropertyEvents(propertyId, rateHz, callback)`; `onChangeEvent(CarPropertyValue)`; read `getStatus()` first and use `getValue()` only when it is `STATUS_AVAILABLE`; rates `ONCHANGE`, 1 Hz, 5 Hz, 10 Hz, 100 Hz; a first subscription receives the current value of every readable area |

All of this is **Binder IPC inside one device**: no socket is involved.  One value, as the VHAL delivers it (`VehiclePropValue`): `timestamp` (synchronised with `SystemClock.elapsedRealtimeNano()`), `prop` (property id), `areaId`, `value` (`int32Values` / `floatValues` / ...), `status` (`AVAILABLE`, `UNAVAILABLE`, `ERROR`; "if the status is not AVAILABLE, the value field must be ignored").  A property has a **change mode**: `ON_CHANGE` (an event when the value or status changes) or `CONTINUOUS` (events at a sample rate, bounded by the property's min and max rate).  Units are SI (speed in m/s, `EV_BATTERY_LEVEL` in Wh, fuel in ml, temperature in Celsius).

**The reference VHAL for emulators** (`FakeVehicleHardware`) keeps values in an in-memory map and "does not communicate with an actual vehicle bus".  From Android 14 its property list is read from JSON in `/vendor/etc/automotive/vhalconfig/`, and a test can override defaults through `/vendor/etc/automotive/vhaloverride/` (system property `persist.vendor.vhal_init_value_override`).  Injection at run time is done from a host with `adb` and `dumpsys` / `cmd car_service`: **`adb` is a debugging channel over USB or TCP and is not used by this product**.  Android 16 adds `GRPCVehicleHardware`, a proxy to a remote gRPC server: **a network protocol, not used**.

Not verified from the documentation I could read: the exact argument syntax of `inject-vhal-event` and `--genfakedata` (the "Debug VHAL" page returned 404), and how a given vendor links its MCU.

## 4. QNX (QNX docs)

* IPC is **synchronous message passing**; drivers and services run in user space and are reached through it.
* The CAN driver library "creates a resource manager for each receive or transmit device": an application **opens a device path and reads frames**.  No socket either.
* Logs go to the system logger (`slog2`); `slog2info` prints them.
* Not verified: a vendor-neutral "vehicle data service" API (it is the OEM's middleware, not part of the OS).

## 5. Linux, Automotive Grade Linux, Yocto (KUKSA / COVESA)

* **SocketCAN** is the kernel's CAN interface (`AF_CAN`: a socket family for the CAN bus, **not** the internet protocols).  A CAN provider reads frames from `can0`, decodes them with a DBC file and a mapping file to COVESA **VSS** signal names.
* Applications then get VSS values from the **KUKSA databroker** over **gRPC or VISS (WebSocket)**.  Those are network protocols, even when local: **not used here**.  Vantage reads SocketCAN and the DBC itself.
* The databroker holds only the "current" value, no history; the history Vantage keeps is its own.

## 6. AUTOSAR Adaptive

Services talk **SOME/IP over automotive Ethernet** inside the vehicle.  That is an in-vehicle network but still an IP network.  Vantage keeps its **decoder** for SOME/IP payloads (they appear in captures and in the simulation) but opens **no** Ethernet socket.

## 7. What Vantage accepts live, with no network

| Source | Where | How it reaches the app | Status |
|---|---|---|---|
| **Simulated bus** | every OS | the recorded drives are played as the *raw traffic* of the vehicle (CAN / CAN-FD frames, SOME/IP payloads, `VehiclePropValue` records, NMEA, UDS answers, log lines, camera metadata) straight into the decoders, in the order a gateway would see them; **in the same process, by a function call** | built and tested; the only live source on Windows |
| **SocketCAN** (listen-only) | Linux, Yocto, Pi OS | the one place a socket is created, and only `AF_CAN`; refuses unless the controller is listen-only | built; tested with a stand-in socket only; never a real controller |
| **CarPropertyManager** | Android Automotive | a Kotlin class subscribes to the properties a profile names and hands each `CarPropertyValue` to the Python data layer in the same process | **written with the Android project; not built or run** |
| **QNX resource manager** | QNX | `open()` / `read()` on the CAN device | **not written** |
| **Capture files** | every OS | candump, ASC, TRC, BLF, CSV, JSON, VSS, OpenXC, ELM327, NMEA, logs read from disk | built and tested |

## 8. What the simulator must get right to be believable

1. **Raw, not pre-digested.**  The decoders see bytes and parcelables, never the assistant's own signal names.  (Built: `vgpt/world/emit.py`, `vgpt/live/replay.py`.)
2. **Real timestamps.**  Bus frames carry the time they were seen; VHAL values carry `elapsedRealtimeNano`.  (Built.)
3. **Status.**  A value whose status is not `AVAILABLE` is ignored and never updates vehicle state.  (Built in `vgpt/protocols/vhal.py`.)
4. **Change mode and rate** (AOSP): `ON_CHANGE` properties deliver an event only when the value or status changes, `CONTINUOUS` ones at a rate between the property's minimum and maximum.  (Added with this change: `vgpt/live/vhal_delivery.py`, section 3 of `docs/OFFLINE_DESIGN.md`.)
5. **Units as the platform delivers them** (m/s, Wh, ml).  (Built: decoded from the generated property table.)

## 9. What was removed because it is a network protocol

The car-to-Vantage HTTP(S) link with its TLS pinning, the loopback web server, MQTT / VISS / HTTP subscribers, UDP / TCP collectors and forwarders, the installable-web-app layer, the browser launcher and the certificate handling.  They are in `vehiclegpt2_retired_networking_2026-10-21/`, outside the product tree, with the tests that went with them.

## 10. Sources

* Android Open Source Project: [VHAL interface](https://source.android.com/docs/automotive/vhal/vhal-interface), [VHAL reference implementation](https://source.android.com/docs/automotive/vhal/reference-implementation), [CarPropertyManager](https://developer.android.com/reference/android/car/hardware/property/CarPropertyManager).
* QNX: [Interprocess communication and resource managers](https://www.qnx.com/developers/docs/qnxeverywhere/com.qnx.doc.qpg/topic/sysarch_IPC_and_resmgr.html), [CAN DDK](https://www.qnx.com/developers/docs/qnxeverywhere/com.qnx.doc.ddk/topic/can/about.html).
* Eclipse KUKSA: [CAN provider](https://github.com/eclipse-kuksa/kuksa-can-provider/blob/main/README.md) and the COVESA VSS material.
* Raspberry Vanilla (AOSP for Raspberry Pi, including the car product): [android_local_manifest](https://github.com/raspberry-vanilla/android_local_manifest/).

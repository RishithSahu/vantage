# How a vehicle's data reaches the application, and how the application reads all of it by itself (2026-10-21)

**The promise**: point the application at a capture from any vehicle - a candump of a truck, an OBD-II log from a phone app, a CSV from a data logger, a JSON stream from a telematics box, a COVESA VSS feed, a Vector BLF from a test bench - and it works out what is in it, which quantity every signal is, in which unit, and stores it like the bus traffic of the vehicles it already knows.  Nothing is configured: no DBC for J1939 or OBD-II, no mapping table, no per-vehicle edit.  What it cannot be sure of it reports and does not store.

```
python vehiclegpt.py ingest capture.csv --db data/mycar.db          # read, recognise, store
python vehiclegpt.py ingest capture.csv --report-only               # only say what it carries and how sure it is
python vehiclegpt.py ingest body.log --dbc body.dbc --dbc pt.dbc    # CAN with a DBC: the signals are named by the DBC and recognised by name, unit and values
python vehiclegpt.py car --vantage <pi> --capture capture.log       # play it to Vantage as a vehicle (it also shows in the app's list when it sits in data/captures)
```

## 1. Research: the ways a car sends each kind of reading

A reading is the same physical thing whichever road it travels, but the road decides the bytes.  What the world actually uses, and what this repository does with each:

| transport (where it is found) | how a value is carried | what the application does |
|---|---|---|
| **SAE J1939** over CAN (heavy vehicles, buses, agricultural, marine, engines, FMS telematics; 250 / 500 kbit/s, 29-bit identifiers) | parameter groups (PGN) of eight bytes, each field an SPN with a scale and offset (speed = 1/256 km/h per bit, engine speed 0.125 rpm, coolant -40 degC offset ...); `0xFB..` reserved, `0xFE..` error, `0xFF..` not available | recognised by identifier structure; the catalogue's SPN table decodes it; reserved, error and not-available fields are dropped; the source address is ignored (or filtered with `"sa": [...]`) |
| **OBD-II / ISO 15031 / SAE J1979** (every car since 2001 (US) / 2004 (EU); 11-bit 0x7DF request, 0x7E8..0x7EF answer, ISO-TP framing; or K-line / adapters) | service 01 + a parameter id (PID): speed 0D, rpm 0C, coolant 05, load 04, fuel level 2F, fuel trims 06 / 07, MAP 0B, MAF 10, intake temp 0F, throttle 11, ambient 46, oil temp 5C, fuel rate 5E, module voltage 42, odometer A6 ... with the standard formulas | ISO-TP reassembly, the catalogue's PID table; ELM327 / OBDLink adapter traces are read too |
| **CAN with a DBC** (every OEM's own signals, 11-bit or 29-bit, classic and CAN FD) | start bit, length, byte order, factor, offset, unit, enum table, multiplexing - all in the DBC | the DBC is decoded (Intel / Motorola, signed, float, mux, FD); each signal is recognised from its name, unit and value range |
| **COVESA VSS** (Vehicle Signal Specification: KUKSA / AGL / cloud-connected fleets; JSON paths such as `Vehicle.Speed`) | a path and a value in the specification's own unit | the path is looked up in the catalogue's VSS table; leaf names are matched when a vendor extends the tree |
| **Android Automotive VHAL** (property ids from AOSP `VehicleProperty.aidl`: `PERF_VEHICLE_SPEED` = m/s, `ENGINE_COOLANT_TEMP`, `EV_BATTERY_LEVEL` = Wh ...) | `VehiclePropValue` parcelables | decoded with the AOSP property table, mapped by name, units converted exactly |
| **SOME/IP over Automotive Ethernet** (AUTOSAR Adaptive; services described in FIBEX / ARXML) | service / event / field, payload layout from the description | decoded when the service catalogue is given in the profile pack (`services.json`); a packet capture without the description cannot be decoded by anyone |
| **GNSS** (NMEA 0183 `$GPRMC` speed in knots, `$GPVTG` km/h, `$GPGGA`; u-blox UBX; gpsd JSON) | sentences | NMEA decoded (speed over ground, position, heading, fix quality); position stays on the device |
| **Phone-app OBD logs and data-logger exports** (Torque, Car Scanner, OBD Fusion, CSV from any logger) | a table: a header per quantity, often with the unit in the name (`Speed (OBD)(km/h)`, `Coolant [degF]`), a time column in any format | header split into name and unit; unit converted exactly; locale quirks handled (`;` separator, decimal comma, ISO times) |
| **JSON streams** (OpenXC `{"name":"vehicle_speed","value":..}`, telematics boxes, MQTT payloads) | objects with named values, flat or nested, a time key in any clock | flattened, the time key found, every named value matched like a column |
| **Diagnostics** (UDS 0x19 DTCs, 0x22 data identifiers; trouble-code texts) | ISO-TP messages | DTC set / clear events with the vehicle's own text; data identifiers when the pack declares them |
| **Logs** (QNX `slog2info`, Android `logcat`, Linux syslog, AUTOSAR **DLT**) | text lines / binary DLT messages | searched as text; profile rules turn lines into events |
| **Camera / ADAS** | feature streams (blockage, sharpness, lane confidence) or object lists | features only; pixels are never stored |

Container formats read: candump (`-L` and the default view), Vector **ASC**, PEAK **TRC** (1.x, 2.x), Vector **BLF** (CAN and CAN FD, zlib containers; verified against files written by python-can), CSV / TSV, JSON / JSON lines, NMEA, ELM327 traces, text and DLT logs; ASAM MDF4 (`.mf4`) through the optional `asammdf` package; a packet capture (pcap) is refused with an explanation (it needs the service description).

## 2. The catalogue: one place that knows every quantity (`vgpt/ingest/catalog.json`)

For each quantity: its label, unit, plausible range, the words people use for it (also the vocabulary the language model is trained on), and every standard that carries it.  Generated from the catalogue:

| quantity (concept) | unit | OBD-II PID | SAE J1939 | COVESA VSS | Android VHAL | names seen in DBC / CSV / JSON |
|---|---|---|---|---|---|---|
| **adaptive cruise state** (`acc_state`) | category |  |  |  |  | acc status, cruise state, adaptive cruise state, cruise control state, acc active |
| **driver-assistance state** (`adas_state`) | category |  |  |  |  | adas status, lka state, lane keeping state, driver assist state, autopilot state, l2 state |
| **emergency-braking state** (`aeb_state`) | category |  |  |  |  | aeb status, emergency braking state, collision mitigation, fcw state, aeb active |
| **blind-spot warning state** (`blindspot_state`) | category |  |  |  |  | bsm, blind spot, blindspot warning, side assist, bsd |
| **lane-marking confidence** (`lane_confidence`) | ratio |  |  |  |  | lane quality, lane marking confidence, lka confidence, lane detection quality |
| **distance to the vehicle ahead** (`lead_vehicle_distance`) | m |  |  |  |  | lead distance, headway, distance to lead, object distance, acc target distance, front vehicle distance |
| **battery current** (`battery_current`) | A |  |  | Vehicle.Powertrain.TractionBattery.CurrentCurrent |  | hv current, pack current, bms current, traction current, battery pack current, idc |
| **battery power** (`battery_power`) | kW |  |  | Vehicle.Powertrain.TractionBattery.CurrentPower |  | hv power, pack power, bms power, battery pack power, traction power, drive power |
| **state of charge** (`battery_soc`) | % | 5B |  | Vehicle.Powertrain.TractionBattery.StateOfCharge.Current<br>Vehicle.Powertrain.TractionBattery.StateOfCharge.Displayed | EV_BATTERY_LEVEL | soc, state of charge, hv battery level, bms soc, charge level, battery percent |
| **battery health** (`battery_soh`) | % |  |  | Vehicle.Powertrain.TractionBattery.StateOfHealth |  | soh, state of health, battery health, battery degradation, capacity retention |
| **battery temperature** (`battery_temperature`) | C |  |  | Vehicle.Powertrain.TractionBattery.Temperature.Average |  | hv battery temp, pack temp, bms temp, battery pack temp, traction battery temperature, avg cell temp |
| **battery voltage** (`battery_voltage`) | V |  |  | Vehicle.Powertrain.TractionBattery.CurrentVoltage |  | hv voltage, pack voltage, bms voltage, traction voltage, hv bus voltage, battery pack voltage |
| **hottest cell temperature** (`cell_temperature_max`) | C |  |  | Vehicle.Powertrain.TractionBattery.Temperature.Max |  | max cell temp, cell temp max, hottest cell, tmax |
| **coolest cell temperature** (`cell_temperature_min`) | C |  |  | Vehicle.Powertrain.TractionBattery.Temperature.Min |  | min cell temp, cell temp min, coolest cell, tmin |
| **highest cell voltage** (`cell_voltage_max`) | V |  |  |  |  | max cell voltage, cell volt max, highest cell, vmax |
| **lowest cell voltage** (`cell_voltage_min`) | V |  |  |  |  | min cell voltage, cell volt min, lowest cell, vmin |
| **charge limit** (`charge_limit`) | % |  |  | Vehicle.Powertrain.TractionBattery.Charging.ChargeLimit |  | charge limit, target soc, charge target, soc limit |
| **charge port state** (`charge_port_state`) | category |  |  | Vehicle.Powertrain.TractionBattery.Charging.IsChargingCableConnected | EV_CHARGE_PORT_CONNECTED, EV_CHARGE_PORT_OPEN | plug connected, cable connected, charge port open, charge flap, inlet connected |
| **charging current** (`charging_current`) | A |  |  | Vehicle.Powertrain.TractionBattery.Charging.ChargeCurrent.DC<br>Vehicle.Powertrain.TractionBattery.Charging.ChargeCurrent.Phase1 |  | charge current, chg current, evse current, charger current |
| **charging power** (`charging_power`) | kW |  |  | Vehicle.Powertrain.TractionBattery.Charging.ChargeRate<br>Vehicle.Powertrain.TractionBattery.Charging.MaxPower | EV_BATTERY_INSTANTANEOUS_CHARGE_RATE | charge power, charger power, ac charge power, dc charge power, chg power, charge rate |
| **charging state** (`charging_state`) | category |  |  | Vehicle.Powertrain.TractionBattery.Charging.IsCharging | EV_CHARGE_STATE | charge state, is charging, charging status, chg state, charge status |
| **charging voltage** (`charging_voltage`) | V |  |  | Vehicle.Powertrain.TractionBattery.Charging.ChargeVoltage.DC |  | charge voltage, chg voltage, evse voltage, charger voltage |
| **coolant flow** (`coolant_flow`) | L/min |  |  |  |  | coolant flow rate, coolant pump flow, cooling flow, water flow, pump flow, battery coolant flow |
| **regenerative braking power** (`regen_power`) | kW |  |  |  |  | regen, recuperation, recuperated power, regenerative power, brake energy recovery, energy recovery power |
| **dashboard range estimate** (`reported_range`) | km |  |  |  |  | dte, distance to empty, range display, dash range, range shown |
| **time to full charge** (`time_to_full`) | min |  |  | Vehicle.Powertrain.TractionBattery.Charging.TimeToComplete |  | time to full, charge time remaining, remaining charge time, minutes to full |
| **accelerator position** (`accelerator_position`) | % | 49 | SPN 91 (PGN 61443) | Vehicle.Chassis.Accelerator.PedalPosition |  | accel pedal, accelerator pedal, gas pedal, pedal position d, apps, acc pedal pos |
| **brake pedal position** (`brake_pedal`) | % |  |  | Vehicle.Chassis.Brake.PedalPosition |  | brake pedal pos, brake pedal position, bpp, brake switch, brk pedal, bremspedal |
| **brake pressure** (`brake_pressure`) | bar |  |  |  |  | brake press, master cylinder pressure, brake line pressure, mc pressure, bremsdruck |
| **lateral acceleration** (`lateral_acceleration`) | m/s2 |  |  | Vehicle.Acceleration.Lateral |  | lat accel, ay, acc y, accel y, lateral accel |
| **longitudinal acceleration** (`longitudinal_acceleration`) | m/s2 |  |  | Vehicle.Acceleration.Longitudinal |  | long accel, ax, acc x, accel x, longitudinal accel, lon accel |
| **parking brake** (`parking_brake`) | category |  |  | Vehicle.Chassis.ParkingBrake.IsEngaged |  | handbrake, epb, parking brake state, park brake |
| **vehicle speed** (`speed`) | km/h | 0D | SPN 84 (PGN 65265); SPN 1624 (PGN 65132) | Vehicle.Speed<br>Vehicle.OBD.Speed | PERF_VEHICLE_SPEED, PERF_VEHICLE_SPEED_DISPLAY | veh speed, vehicle velocity, vss, spd, speed kph, speed kmh |
| **steering angle** (`steering_angle`) | deg |  |  | Vehicle.Chassis.SteeringWheel.Angle | PERF_STEERING_ANGLE | steer angle, steering wheel angle, swa, steering wheel position, sas angle, lenkwinkel |
| **steering rate** (`steering_rate`) | deg/s |  |  |  |  | steer rate, steering speed, steering angle velocity, swa rate |
| **front-left tyre pressure** (`tyre_pressure_fl`) | kPa |  |  | Vehicle.Chassis.Axle.Row1.Wheel.Left.Tire.Pressure | TIRE_PRESSURE | tire pressure front left, tyre press fl, tpms fl, tp fl, fl tire pressure, reifendruck vl |
| **front-right tyre pressure** (`tyre_pressure_fr`) | kPa |  |  | Vehicle.Chassis.Axle.Row1.Wheel.Right.Tire.Pressure |  | tire pressure front right, tyre press fr, tpms fr, tp fr, fr tire pressure, reifendruck vr |
| **rear-left tyre pressure** (`tyre_pressure_rl`) | kPa |  |  | Vehicle.Chassis.Axle.Row2.Wheel.Left.Tire.Pressure |  | tire pressure rear left, tyre press rl, tpms rl, tp rl, rl tire pressure, reifendruck hl |
| **rear-right tyre pressure** (`tyre_pressure_rr`) | kPa |  |  | Vehicle.Chassis.Axle.Row2.Wheel.Right.Tire.Pressure |  | tire pressure rear right, tyre press rr, tpms rr, tp rr, rr tire pressure, reifendruck hr |
| **tyre temperature** (`tyre_temperature`) | C |  | SPN 242 (PGN 65268) | Vehicle.Chassis.Axle.Row1.Wheel.Left.Tire.Temperature |  | tire temp, tyre temp, tpms temp, tire temperature, tyr temp, tyre temp f |
| **front-left wheel speed** (`wheel_speed_fl`) | km/h |  |  | Vehicle.Chassis.Axle.Row1.Wheel.Left.Speed |  | wheel speed front left, whl spd fl, wss fl, fl wheel speed, v wheel fl, wheel fl |
| **front-right wheel speed** (`wheel_speed_fr`) | km/h |  |  | Vehicle.Chassis.Axle.Row1.Wheel.Right.Speed |  | wheel speed front right, whl spd fr, wss fr, fr wheel speed, v wheel fr, wheel fr |
| **rear-left wheel speed** (`wheel_speed_rl`) | km/h |  |  | Vehicle.Chassis.Axle.Row2.Wheel.Left.Speed |  | wheel speed rear left, whl spd rl, wss rl, rl wheel speed, v wheel rl, wheel rl |
| **rear-right wheel speed** (`wheel_speed_rr`) | km/h |  |  | Vehicle.Chassis.Axle.Row2.Wheel.Right.Speed |  | wheel speed rear right, whl spd rr, wss rr, rr wheel speed, v wheel rr, wheel rr |
| **yaw rate** (`yaw_rate`) | deg/s |  |  | Vehicle.AngularVelocity.Yaw |  | yaw, yaw velocity, gier rate, esp yaw, rotation rate z, yaw rate |
| **outside temperature** (`ambient_temperature`) | C | 46 | SPN 171 (PGN 65269) | Vehicle.Exterior.AirTemperature<br>Vehicle.OBD.AmbientAirTemperature | ENV_OUTSIDE_TEMPERATURE | outside temp, outdoor temp, ambient air, oat, external temp, exterior temp |
| **barometric pressure** (`barometric_pressure`) | kPa | 33 | SPN 108 (PGN 65269) | Vehicle.OBD.BarometricPressure |  | baro, baro press, atmospheric press, ambient pressure |
| **humidity** (`humidity`) | % |  |  |  |  | relative humidity, rh, air humidity, cabin humidity |
| **ambient light** (`light`) | lux |  |  | Vehicle.Exterior.LightIntensity |  | lux, light sensor, ambient light, brightness sensor, illuminance, daylight |
| **rain** (`rain`) | none |  |  | Vehicle.Exterior.Humidity<br>Vehicle.Body.Windshield.Front.WiperSystem.IsRainDetected |  | rain sensor, rain detected, wetness |
| **boost pressure** (`boost_pressure`) | kPa |  | SPN 102 (PGN 65270) |  |  | turbo boost, boost press, charge pressure, turbo pressure |
| **catalyst temperature** (`catalyst_temperature`) | C | 3C |  |  |  | cat temp, catalyst temp b1s1, scr temperature |
| **AdBlue level** (`def_level`) | % |  | SPN 1761 (PGN 65110) |  |  | adblue, def tank level, urea level, scr tank level, reductant level |
| **particulate filter soot load** (`dpf_soot_load`) | % |  |  |  |  | soot, dpf soot, particulate filter load, dpf load |
| **driving state** (`drive_mode`) | category |  |  |  | IGNITION_STATE | ignition, ignition state, key position, power state, vehicle state, drive state |
| **EGR valve position** (`egr_position`) | % | 2C |  |  |  | egr, commanded egr, egr valve |
| **total engine hours** (`engine_hours`) | h |  | SPN 247 (PGN 65253) |  |  | total engine hours, engine hrs, hour meter |
| **engine load** (`engine_load`) | % | 04 | SPN 92 (PGN 61443) | Vehicle.OBD.EngineLoad<br>Vehicle.Powertrain.CombustionEngine.EngineLoad |  | calculated load, engine load, eng load, load value, abs load |
| **engine oil pressure** (`engine_oil_pressure`) | kPa |  | SPN 100 (PGN 65263) | Vehicle.Powertrain.CombustionEngine.EOP |  | eop, oil press, oil pressure |
| **engine oil temperature** (`engine_oil_temperature`) | C | 5C | SPN 175 (PGN 65262) | Vehicle.Powertrain.CombustionEngine.EOT<br>Vehicle.OBD.OilTemperature | ENGINE_OIL_TEMP | eot, oil temp, eng oil temp |
| **engine rpm** (`engine_rpm`) | rpm | 0C | SPN 190 (PGN 61444) | Vehicle.Powertrain.CombustionEngine.Speed<br>Vehicle.OBD.EngineSpeed | ENGINE_RPM | eng speed, engspeed, n eng, rpm, revs, crank speed |
| **time since engine start** (`engine_run_time`) | s | 1F |  |  |  | run time, runtime, time since start |
| **engine torque** (`engine_torque`) | Nm |  |  | Vehicle.Powertrain.CombustionEngine.Torque |  | eng torque, engine torque nm, indicated torque, crankshaft torque |
| **engine torque (percent of reference)** (`engine_torque_percent`) | % | 62 | SPN 513 (PGN 61444) |  |  | actual engine percent torque, engine percent torque, trq pct, torque pct |
| **exhaust gas temperature** (`exhaust_gas_temperature`) | C | 78 | SPN 173 (PGN 65270) |  |  | egt, exhaust temp, exh gas temp, turbine inlet temperature |
| **fuel pressure** (`fuel_pressure`) | kPa | 0A | SPN 94 (PGN 65263) |  |  | fuel press, fuel supply pressure |
| **fuel rail pressure** (`fuel_rail_pressure`) | kPa | 23 |  |  |  | rail pressure, rail press, frp, common rail |
| **fuel rate** (`fuel_rate`) | L/h | 5E | SPN 183 (PGN 65266) | Vehicle.Powertrain.FuelSystem.InstantConsumption<br>Vehicle.OBD.FuelRate |  | fuel consumption rate, fuel flow, fuel burn, injection quantity, fuel use rate, engine fuel rate |
| **fuel temperature** (`fuel_temperature`) | C |  | SPN 174 (PGN 65262) |  |  | fuel temp |
| **current gear** (`gear`) | category |  | SPN 523 (PGN 61445) | Vehicle.Powertrain.Transmission.CurrentGear | CURRENT_GEAR | gear, gear position, current gear, selected gear, gear engaged |
| **ignition timing** (`ignition_timing`) | deg | 0E |  |  |  | timing advance, spark advance, ign timing, ignition advance |
| **intake air temperature** (`intake_air_temperature`) | C | 0F | SPN 172 (PGN 65269) | Vehicle.Powertrain.CombustionEngine.IntakeAirTemperature<br>Vehicle.OBD.IntakeTemp |  | iat, air intake temp, intake temp, inlet air temp |
| **intake manifold pressure** (`intake_manifold_pressure`) | kPa | 0B |  | Vehicle.Powertrain.CombustionEngine.MAP<br>Vehicle.OBD.IntakePressure |  | map, manifold press, manifold abs pressure, intake press |
| **inverter temperature** (`inverter_temperature`) | C |  |  |  |  | inverter temp, igbt temp, power electronics temp, pcu temp, inv temp |
| **air-fuel ratio (lambda)** (`lambda`) | none | 24 |  |  |  | lambda, equivalence ratio, air fuel equivalence, o2 lambda, afr |
| **long-term fuel trim** (`long_term_fuel_trim`) | % | 07 |  | Vehicle.OBD.LongTermFuelTrim1 |  | ltft, long term trim, fuel trim lt, ltft b1 |
| **mass air flow** (`mass_air_flow`) | g/s | 10 |  | Vehicle.Powertrain.CombustionEngine.MAF<br>Vehicle.OBD.MAF |  | maf, air mass flow |
| **motor (mechanical) power** (`motor_power`) | kW |  |  | Vehicle.Powertrain.ElectricMotor.Power |  | e motor power, em power, mechanical power, shaft power |
| **motor RPM** (`motor_rpm`) | rpm |  |  | Vehicle.Powertrain.ElectricMotor.Speed |  | e motor speed, em speed, motor speed, rotor speed, traction motor speed |
| **motor temperature** (`motor_temperature`) | C | 05 | SPN 110 (PGN 65262) | Vehicle.Powertrain.CombustionEngine.ECT<br>Vehicle.OBD.CoolantTemperature<br>Vehicle.Powertrain.ElectricMotor.Temperature | ENGINE_COOLANT_TEMP | coolant temp, ect, clt, engine coolant, water temp, coolant temperature |
| **motor torque** (`motor_torque`) | Nm |  |  | Vehicle.Powertrain.ElectricMotor.Torque |  | e motor torque, em torque, traction torque, motor trq, drive torque, torque request |
| **short-term fuel trim** (`short_term_fuel_trim`) | % | 06 |  | Vehicle.OBD.ShortTermFuelTrim1 |  | stft, short term trim, fuel trim st, sfT, stft b1 |
| **throttle position** (`throttle_position`) | % | 11 | SPN 51 (PGN 65266) | Vehicle.OBD.ThrottlePosition<br>Vehicle.Powertrain.CombustionEngine.TPS |  | tps, throttle pos, throttle opening, absolute throttle |
| **transmission temperature** (`transmission_temperature`) | C |  | SPN 177 (PGN 65272) |  |  | trans temp, atf temp, gearbox temp, transmission oil temp, gearbox oil temperature |
| **altitude** (`altitude`) | m |  |  | Vehicle.CurrentLocation.Altitude |  | alt, gps altitude, elevation, height |
| **12 V battery voltage** (`aux_battery_voltage`) | V | 42 | SPN 168 (PGN 65271) | Vehicle.LowVoltageBattery.CurrentVoltage<br>Vehicle.OBD.ControlModuleVoltage |  | 12v, 12 v, battery 12v, kl30, module voltage, control module voltage |
| **cabin temperature** (`cabin_temperature`) | C |  | SPN 170 (PGN 65269) | Vehicle.Cabin.HVAC.AmbientAirTemperature | HVAC_TEMPERATURE_CURRENT | interior temp, inside temp, cabin temp, cab temp, passenger compartment temperature |
| **door state** (`door_state`) | category |  |  | Vehicle.Cabin.Door.Row1.Driver.IsOpen |  | door open, door ajar, driver door, door status |
| **fuel level** (`fuel_level`) | % | 2F | SPN 96 (PGN 65276) | Vehicle.Powertrain.FuelSystem.RelativeLevel<br>Vehicle.OBD.FuelLevel | FUEL_LEVEL | fuel tank level, tank level, fuel gauge, fuel lvl, fuel remaining, fuel left |
| **GPS speed** (`gps_speed`) | km/h |  |  |  |  | gnss speed, gps spd, speed over ground, sog, gps velocity, nmea speed |
| **heading** (`heading`) | deg |  |  | Vehicle.CurrentLocation.Heading |  | course, bearing, course over ground, cog, gps heading, direction of travel |
| **headlight state** (`headlight_state`) | category |  |  | Vehicle.Body.Lights.Beam.Low.IsOn |  | headlamp, low beam, high beam, lights on, light switch |
| **climate-control power** (`hvac_power`) | W |  |  |  |  | ac power, heater power, climate power, hvac consumption, ptc power, compressor power |
| **odometer** (`odometer`) | km | A6 | SPN 245 (PGN 65248); SPN 917 (PGN 65217) | Vehicle.TraveledDistance | PERF_ODOMETER | odo, total distance, mileage, distance total, total vehicle distance, km counter |
| **seat belt state** (`seatbelt_state`) | category |  |  | Vehicle.Cabin.Seat.Row1.DriverSide.IsBelted |  | belt, seat belt, belt buckled, seatbelt status |
| **wiper state** (`wiper_state`) | category |  |  | Vehicle.Body.Windshield.Front.Wiping.Mode |  | wiper, wipers, wiper speed, wiper mode |

## 3. How a signal is recognised (never a guess)

1. **The name** is split (camelCase, underscores, digits, `Vehicle.Powertrain...` paths), abbreviations are expanded (`tmp`, `spd`, `trq`, `ect`, `eot`, `stft` ...), and compared with every phrase the catalogue knows for each quantity.
2. **The unit** stated in the name or the DBC must belong to the quantity's physical class: `kPa` can never make a temperature.
3. **The wheel position** (`FL`, `rear right`) must agree.
4. **The values** decide what the words leave open: 12 V against 400 V, where in the plausible range the readings sit.
5. A **tie** is broken by the words the signal shares with the quantity's own label (`EngSpeed` is the engine's, `MotorSpeed` the motor's); if it is still a tie the signal is **reported as unmapped**, not assigned.
6. **Units are converted exactly** (mph, knots, degF, psi, bar, gal/h, hp ...) into the registry's unit, so the answer is right whatever the source called it.

Measured (`tests/test_ingest.py`): at least 94 % of 150 real-world signal names recognised with at most 3 % wrong (the rest refused); of the 127 DBC signals of the four vehicle packs, 123 agree with the pack's own mapping, 2 are finer than the pack's (a throttle position is not an accelerator pedal; a percent torque is not a torque in Nm) and 2 are refused as ambiguous; an electric vehicle's three CAN buses read with only their DBC files give the same stored series as its own profile (median difference under 5 % of the range).

## 4. What is NOT covered, said plainly

* A **proprietary CAN capture with no DBC** (not J1939, not OBD-II): the identifiers cannot be named by anyone; the application says so and asks for the DBC.
* **SOME/IP / ARXML** traffic without its service description; **UDS data identifiers** an OEM keeps private.
* **Encrypted or signed** streams.
* **Live subscriptions**: MQTT 3.1.1, W3C VISS over WebSocket and HTTP polling have collectors (`vgpt/live/netcollect.py`, standard library only; protocols checked against the byte sequences of their specifications and each collector against a small server of its own, `tests/test_netcollect.py`).  MQTT takes a user name and password and can use TLS (certificate verified).  Use: `python tools_live.py ingest --db car.db --profile <pack> --mqtt broker:1883 --mqtt-topic 'vehicle/#'` (or `--viss host:8090 --viss-path Vehicle.Speed`, or `--http-poll http://dongle.local/data`); a bare number is named by its topic, a JSON object by its own keys (`prefix_objects=True` puts the topic in front).  MQTT is spoken in versions 3.1.1 and 5.0 at QoS 0, 1 and 2 (a QoS 2 message that the broker repeats is delivered once).  The VISS collector was also run against the independent `websockets` library's server (handshake, sub-protocol, ping, a 200 kB message in many frames).  Eclipse Kuksa's VISS (WebSocket) interface is read by the VISS collector; its **gRPC databroker is not** (no gRPC library is installed here and nothing is downloaded without asking).  **Not built**: MQTT sessions that survive a restart (a clean session each time), VISS access-token authentication.  The collectors are tested against servers written for the test, not against Mosquitto or a production VISS server.
* A capture that carries **no clock** (an adapter trace) is stored in order, one record after another, not at the moments it was recorded.
* The mapping uses English names (and the common German CAN names in the catalogue's lists, such as `Lenkwinkel`, `Bremsdruck`); a vendor's own abbreviation is refused rather than guessed - the report says which signals, so a person can add one line to the catalogue.
* Verified on generated drives rendered in each format and on files written by python-can; **not yet on a capture from a production vehicle** (none was available).

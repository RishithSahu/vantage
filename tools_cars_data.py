"""The car models of the test bench, as data for tools_make_cars.py.  Every `bind` row was read against the real database (see `tools_make_cars.py --describe`): (concept, "Message.Signal", options).
Options: unit (what the database signal is measured in), scale / offset (a fix on the decoded value), map (enum label or raw code -> number), snap (an enum that carries a continuous quantity as states),
when (a threshold that turns the value into 1 / 0), rate (Hz).  `ext` lists quantities the open database does not publish and that therefore ride a labelled SYNTHETIC extension database.

English signal names only for now (owner's rule, 2026-10-21): the VW MQB and PSA databases are left out until their names are mapped."""

OPENDBC = "opendbc (https://github.com/commaai/opendbc, MIT licence)"
LINUX = dict(log_format="rfc3164", log_name="Linux telematics / infotainment (syslog)")
ANDROID = dict(log_format="logcat", log_name="Android infotainment (logcat)")
QNX = dict(log_format="slog2info", log_name="QNX infotainment (slog2info)")
TYRES = ["tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr", "tyre_temperature"]
EV_ESSENTIALS = ["battery_soc", "battery_voltage", "battery_current", "battery_temperature", "motor_temperature", "reported_range", "charging_power", "coolant_flow", "_charging_active"]


def ice(**kw):
    return dict(powertrain="petrol", obd=True, **kw)


def wheels(prefix, names, unit, **opt):
    return [(f"wheel_speed_{w}", f"{prefix}.{n}", dict(unit=unit, rate=20, **opt)) for w, n in zip(("fl", "fr", "rl", "rr"), names)]


CARS = [
    dict(id="toyota_prius_hybrid", make="Toyota", name="Prius (XW30 hybrid)", year=2010, region="Japan / global", **ice(), cls="Hybrid hatchback (simulated with the petrol drive)", **LINUX, camera="Forward camera (driver assistance)",
         can=[("toyota_prius_2010_pt.dbc", "pt", "can0", False, 500000, "Powertrain and chassis CAN 500 kbit/s (OEM database toyota_prius_2010_pt)")],
         bind=[("speed", "SPEED.SPEED", dict(unit="mph", rate=20)), *wheels("WHEEL_SPEEDS", ["WHEEL_SPEED_FL", "WHEEL_SPEED_FR", "WHEEL_SPEED_RL", "WHEEL_SPEED_RR"], "mph"),
               ("steering_angle", "STEER_ANGLE_SENSOR.STEER_ANGLE", dict(unit="deg", rate=20)), ("engine_rpm", "POWERTRAIN.ENGINE_RPM", dict(unit="rpm", rate=20)),
               ("accelerator_position", "GAS_PEDAL.GAS_PEDAL", dict(unit="%", scale=100, rate=20)), ("brake_pedal", "BRAKE.BRAKE_PEDAL", dict(unit="%", scale=0.392157, rate=20)),
               ("longitudinal_acceleration", "ACCELEROMETER.ACCEL_X", dict(unit="m/s2", rate=50)), ("lead_vehicle_distance", "LEAD_INFO.LEAD_LONG_DIST", dict(unit="m", rate=10)),
               ("gear", "GEAR_PACKET.GEAR", dict(unit="none", map={"P": 0, "R": -1, "N": 0, "D": 1, "B": 1}, rate=5)), ("_ignition", "GEAR_PACKET.GEAR", dict(unit="none", when=">0.5")),
               ("seatbelt_state", "SEATS_DOORS.SEATBELT_DRIVER_UNLATCHED", dict(unit="none", map={"0": 1, "1": 0}, rate=2)), ("door_state", "SEATS_DOORS.DOOR_OPEN_FL", dict(unit="none", rate=2))],
         ext=TYRES + ["odometer"], vehicle=dict(specs={"veh.tyre_spec_front": 230, "veh.tyre_spec_rear": 230, "veh.service_km": 15000, "veh.sw": "ecu 3.1", "veh.wheelbase_m": 2.7, "veh.wheel_radius_m": 0.317, "veh.rated_max_speed": 180,
                                                      "veh.seats": 5, "veh.tank_l": 43, "veh.cvt": 1}),
         interfaces=["CAN 500 kbit/s, 11-bit identifiers: the real Prius powertrain / chassis database", "OBD-II service 01 over ISO 15765-4 CAN (the rest of the engine data)", "UDS diagnostic trouble codes", "syslog from the telematics unit", "NMEA 0183 GNSS",
                     "forward camera pictures (JPEG)", "tyre pressure and odometer on a synthetic extension CAN (the open database does not publish them)"], source=f"{OPENDBC}: toyota_prius_2010_pt.dbc"),

    dict(id="hyundai_elantra_ccan", make="Hyundai", name="Elantra 2.0 (C-CAN)", year=2016, region="Korea / global", **ice(), cls="Compact saloon, petrol", **ANDROID, camera="Forward camera (SmartSense)",
         can=[("hyundai_2015_ccan.dbc", "pt", "can0", False, 500000, "Chassis CAN 500 kbit/s (OEM database hyundai_2015_ccan)")],
         bind=[("speed", "EMS11.VS", dict(unit="km/h", rate=20)), ("engine_rpm", "EMS11.N", dict(unit="rpm", rate=20)), ("engine_load", "EMS11.TQI", dict(unit="%", rate=20)), ("_ignition", "EMS11.SWI_IGK", dict(unit="none")),
               *wheels("WHL_SPD11", ["WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR"], "km/h"), ("lateral_acceleration", "ESP12.LAT_ACCEL", dict(unit="m/s2", rate=50)),
               ("longitudinal_acceleration", "ESP12.LONG_ACCEL", dict(unit="m/s2", rate=50)), ("yaw_rate", "ESP12.YAW_RATE", dict(unit="deg/s", rate=50)), ("steering_angle", "SAS11.SAS_Angle", dict(unit="deg", rate=50)),
               ("motor_temperature", "EMS12.TEMP_ENG", dict(unit="C", rate=1)), ("engine_oil_temperature", "EMS19.CR_Ems_EngOilTemp", dict(unit="C", rate=1)), ("fuel_temperature", "EMS14.TEMP_FUEL", dict(unit="C", rate=1)),
               ("intake_air_temperature", "EMS15.IntAirTemp", dict(unit="C", rate=1)), ("ambient_temperature", "FATC11.CR_Fatc_OutTemp", dict(unit="C", rate=1)), ("cabin_temperature", "DATC11.CF_Datc_IncarTemp", dict(unit="C", rate=1)),
               ("odometer", "CLU12.CF_Clu_Odometer", dict(unit="km", rate=1)), ("transmission_temperature", "TCU11.TEMP_AT", dict(unit="C", rate=1)), ("lead_vehicle_distance", "SCC11.ACC_ObjDist", dict(unit="m", rate=10)),
               ("parking_brake", "TCS13.PBRAKE_ACT", dict(unit="none", rate=10)), ("brake_pedal", "TCS13.BrakeLight", dict(unit="%", map={"0": 0, "1": 100}, snap=True, rate=20)),
               ("headlight_state", "CGW1.CF_Gway_HeadLampLow", dict(unit="none", rate=2)), ("wiper_state", "CGW1.CF_Gway_WiperLowSw", dict(unit="none", map={"0": 0, "1": 1}, snap=True, rate=2))],
         ext=TYRES, vehicle=dict(specs={"veh.tyre_spec_front": 235, "veh.tyre_spec_rear": 235, "veh.service_km": 15000, "veh.sw": "ems 5.2", "veh.wheelbase_m": 2.7, "veh.wheel_radius_m": 0.32, "veh.rated_max_speed": 195, "veh.seats": 5, "veh.tank_l": 47}),
         interfaces=["CAN 500 kbit/s, 11-bit identifiers: the real Hyundai chassis database (engine, ESP, steering, climate, body)", "OBD-II service 01 over ISO 15765-4 CAN (fuel level, airflow, ...)", "UDS diagnostic trouble codes", "logcat from the Android infotainment unit",
                     "NMEA 0183 GNSS", "forward camera pictures (JPEG)", "tyre pressure on a synthetic extension CAN"], source=f"{OPENDBC}: hyundai_2015_ccan.dbc"),

    dict(id="bmw_3_series_e9x", make="BMW", name="3 Series (E90 320i)", year=2009, region="Germany / global", **ice(), cls="Compact saloon, petrol", **LINUX, camera=None,
         can=[("bmw_e9x_e8x.dbc", "pt", "can0", False, 500000, "PT-CAN 500 kbit/s (OEM database bmw_e9x_e8x)")],
         bind=[("engine_rpm", "AccPedal.EngineSpeed", dict(unit="rpm", rate=20)), ("accelerator_position", "AccPedal.AcceleratorPedalPercentage", dict(unit="%", rate=20)), ("speed", "Speed.VehicleSpeed", dict(unit="km/h", rate=20)),
               ("steering_angle", "SteeringWheelAngle.SteeringPosition", dict(unit="deg", rate=20)), ("gear", "TransmissionDataDisplay.GearAct", dict(unit="none", rate=10)),
               ("brake_pedal", "EngineAndBrake.BrakePressed", dict(unit="%", map={"0": 0, "1": 100}, snap=True, rate=20)), ("motor_temperature", "EngineData.TEMP_ENG", dict(unit="C", rate=1)),
               ("engine_oil_temperature", "EngineData.TEMP_EOI", dict(unit="C", rate=1)), ("aux_battery_voltage", "PowerBatteryVoltage.BatteryVoltage", dict(unit="V", rate=2)),
               ("parking_brake", "InstrumentClusterStatus_KOMBI.HandbrakeActive", dict(unit="none", rate=5)), ("_ignition", "AccPedal.EngineSpeed", dict(unit="rpm", when=">300"))],
         ext=TYRES + ["odometer"], vehicle=dict(specs={"veh.tyre_spec_front": 240, "veh.tyre_spec_rear": 260, "veh.service_km": 20000, "veh.sw": "dme 7.0", "veh.wheelbase_m": 2.76, "veh.wheel_radius_m": 0.32, "veh.rated_max_speed": 235,
                                                      "veh.seats": 5, "veh.tank_l": 63}),
         interfaces=["CAN 500 kbit/s, 11-bit identifiers: the real BMW E9x powertrain database", "OBD-II service 01 over ISO 15765-4 CAN", "UDS diagnostic trouble codes", "syslog from the head unit", "NMEA 0183 GNSS", "tyre pressure and odometer on a synthetic extension CAN"],
         source=f"{OPENDBC}: bmw_e9x_e8x.dbc"),

    dict(id="mercedes_e350_w212", make="Mercedes-Benz", name="E 350 (W212)", year=2010, region="Germany / global", **ice(), cls="Executive saloon, petrol V6", **QNX, camera=None,
         can=[("mercedes_benz_e350_2010.dbc", "pt", "can0", False, 500000, "Chassis CAN 500 kbit/s (OEM database mercedes_benz_e350_2010)")],
         bind=[("accelerator_position", "GAS_PEDAL.GAS_PEDAL", dict(unit="%", scale=0.392157, rate=20)), *wheels("WHEEL_SPEEDS", ["WHEEL_SPEED_FL", "WHEEL_SPEED_FR", "WHEEL_SPEED_RL", "WHEEL_SPEED_RR"], "mph"),
               ("steering_angle", "STEER_SENSOR.STEER_ANGLE", dict(unit="deg", rate=20)), ("brake_pedal", "BRAKE_MODULE.BRAKE_PRESSED", dict(unit="%", map={"0": 0, "1": 100}, snap=True, rate=20)),
               ("door_state", "DOOR_SENSORS.DOOR_OPEN_FL", dict(unit="none", rate=5)), ("seatbelt_state", "SEATBELT_SENSORS.SEATBELT_DRIVER_LATCHED", dict(unit="none", rate=2))],
         ext=TYRES + ["odometer", "_ignition"], vehicle=dict(specs={"veh.tyre_spec_front": 250, "veh.tyre_spec_rear": 270, "veh.service_km": 25000, "veh.sw": "me 9.7", "veh.wheelbase_m": 2.87, "veh.wheel_radius_m": 0.33, "veh.rated_max_speed": 250,
                                                                   "veh.seats": 5, "veh.tank_l": 80}),
         interfaces=["CAN 500 kbit/s, 11-bit identifiers: the real Mercedes E-Class chassis database (wheel speeds, steering, pedals, doors, belts)", "OBD-II service 01 over ISO 15765-4 CAN (vehicle speed, engine speed, temperatures, fuel)", "UDS diagnostic trouble codes",
                     "slog2info from the QNX head unit", "NMEA 0183 GNSS", "tyre pressure, odometer and ignition state on a synthetic extension CAN"], source=f"{OPENDBC}: mercedes_benz_e350_2010.dbc"),

    dict(id="volvo_v60_2015", make="Volvo", name="V60 T5", year=2015, region="Sweden / global", **ice(), cls="Estate, petrol", **LINUX, camera="Forward camera (City Safety)",
         can=[("volvo_v60_2015_pt.dbc", "pt", "can0", False, 500000, "Powertrain CAN 500 kbit/s (OEM database volvo_v60_2015_pt)")],
         bind=[("accelerator_position", "AccPedal.AccPedal", dict(unit="%", rate=20)), ("engine_rpm", "EngineData.EngineRpm", dict(unit="rpm", rate=20)), ("speed", "VehicleSpeed1.VehicleSpeed", dict(unit="km/h", rate=20)),
               ("steering_angle", "PSCM1.SteeringAngleServo", dict(unit="deg", rate=20)), ("wheel_speed_fr", "wheelspeed0.WhlSpdRF", dict(unit="km/h", rate=20)), ("wheel_speed_fl", "wheelspeed0.WhlSpdLF", dict(unit="km/h", rate=20)),
               ("wheel_speed_rr", "wheelspeed1.WhlSpdRR", dict(unit="km/h", rate=20)), ("wheel_speed_rl", "wheelspeed1.WhlSpdLR", dict(unit="km/h", rate=20)), ("wiper_state", "Accessories_01.Wipers", dict(unit="none", rate=2)),
               ("door_state", "Doors.DriverDoorClosed", dict(unit="none", map={"0": 1, "1": 0}, rate=5)), ("_ignition", "EngineData.EngineRpm", dict(unit="rpm", when=">300"))],
         ext=TYRES + ["odometer"], vehicle=dict(specs={"veh.tyre_spec_front": 240, "veh.tyre_spec_rear": 250, "veh.service_km": 30000, "veh.sw": "cem 4.1", "veh.wheelbase_m": 2.78, "veh.wheel_radius_m": 0.33, "veh.rated_max_speed": 230,
                                                      "veh.seats": 5, "veh.tank_l": 67}),
         interfaces=["CAN 500 kbit/s, 11-bit identifiers: the real Volvo V60 powertrain database", "OBD-II service 01 over ISO 15765-4 CAN", "UDS diagnostic trouble codes", "syslog from the infotainment unit", "NMEA 0183 GNSS", "forward camera pictures (JPEG)",
                     "tyre pressure and odometer on a synthetic extension CAN"], source=f"{OPENDBC}: volvo_v60_2015_pt.dbc"),

    dict(id="tesla_model_3", make="Tesla", name="Model 3 Long Range", year=2020, region="USA / global", powertrain="bev", obd=False, cls="Battery-electric saloon", **LINUX, camera="Forward camera (driver assistance)",
         can=[("tesla_model3_party.dbc", "party", "can0", False, 500000, "Party / powertrain CAN 500 kbit/s (OEM database tesla_model3_party)")],
         bind=[("speed", "DI_speed.DI_vehicleSpeed", dict(unit="km/h", rate=20)), ("motor_torque", "DI_torque.DI_torqueActual", dict(unit="Nm", rate=20)), ("accelerator_position", "DI_systemStatus.DI_accelPedalPos", dict(unit="%", rate=20)),
               ("gear", "DI_systemStatus.DI_gear", dict(unit="none", map={"DI_GEAR_P": 0, "DI_GEAR_R": -1, "DI_GEAR_N": 0, "DI_GEAR_D": 1}, rate=10)), ("_ignition", "DI_systemStatus.DI_gear", dict(unit="none", when=">1.5")),
               ("brake_pedal", "DI_systemStatus.DI_brakePedalState", dict(unit="%", map={"OFF": 0, "ON": 100}, snap=True, rate=20)),
               *wheels("ESP_wheelSpeeds", ["ESP_wheelSpeedFrL", "ESP_wheelSpeedFrR", "ESP_wheelSpeedReL", "ESP_wheelSpeedReR"], "km/h"), ("steering_angle", "SCCM_steeringAngleSensor.SCCM_steeringAngle", dict(unit="deg", rate=50)),
               ("steering_rate", "SCCM_steeringAngleSensor.SCCM_steeringAngleSpeed", dict(unit="deg/s", rate=50)), ("door_state", "UI_warning.anyDoorOpen", dict(unit="none", map={"OPEN": 1, "CLOSED": 0}, rate=2)),
               ("parking_brake", "DI_state.DI_parkBrakeState", dict(unit="none", map={"RELEASED": 0, "APPLIED": 1}, rate=5))],
         ext=TYRES + ["odometer", "aux_battery_voltage", "ambient_temperature"] + EV_ESSENTIALS, vehicle=dict(specs={"veh.cap_kwh": 75, "veh.soh": 97, "veh.nominal_voltage": 360, "veh.tyre_spec_front": 290, "veh.tyre_spec_rear": 290, "veh.service_km": 20000,
                                                                                                                   "veh.sw": "2020.48", "veh.wheelbase_m": 2.875, "veh.wheel_radius_m": 0.34, "veh.gear_ratio": 9.0, "veh.rated_max_speed": 233,
                                                                                                                   "veh.seats": 5, "veh.home_charger_kw": 11}),
         interfaces=["CAN 500 kbit/s, 11-bit identifiers: the real Model 3 drive-unit, ESP and steering database", "UDS diagnostic trouble codes", "syslog from the Linux computer", "NMEA 0183 GNSS", "forward camera pictures (JPEG)",
                     "battery, range, charging, tyres, odometer and 12 V on a synthetic extension CAN (the open database does not publish the battery bus)"], source=f"{OPENDBC}: tesla_model3_party.dbc"),

    dict(id="chevrolet_bolt_ev", make="Chevrolet", name="Bolt EV", year=2021, region="USA / global", powertrain="bev", obd=True, cls="Battery-electric hatchback", **LINUX, camera="Forward camera (Front Camera Module)",
         can=[("gm_global_a_high_voltage_management.dbc", "hv", "can0", False, 500000, "High-voltage management CAN 500 kbit/s (OEM database gm_global_a_high_voltage_management)")],
         bind=[("battery_voltage", "Pack_Stats.Pack_Voltage", dict(unit="V", rate=10)), ("battery_temperature", "Battery_temp.Temp_A", dict(unit="C", rate=1)), ("charging_voltage", "Charger_stats.HV_Voltage", dict(unit="V", rate=2)),
               ("charging_current", "Charger_stats.Charger_HV_Current", dict(unit="A", rate=2)), ("_charging_active", "AC_Stats.Mains_Voltage", dict(unit="none", map={"Unplugged": 0, "110V": 1, "220V": 1}, rate=1))],
         dids=[("battery_soc", "0xF1A0", ">H", 0.1, "%", 1), ("battery_current", "0xF1A1", ">h", 0.1, "A", 5), ("reported_range", "0xF1A2", ">H", 1.0, "km", 0.2), ("motor_temperature", "0xF1A3", ">H", 0.1, "C", 1)],
         ext=TYRES + ["odometer", "_ignition"], diag_name="UDS / OBD-II (ISO 15765-4) with vendor data identifiers", vehicle=dict(specs={"veh.cap_kwh": 65, "veh.soh": 98, "veh.nominal_voltage": 350, "veh.tyre_spec_front": 250, "veh.tyre_spec_rear": 250,
                                                                                                                                         "veh.service_km": 12000, "veh.sw": "hv 21.3", "veh.wheelbase_m": 2.67, "veh.wheel_radius_m": 0.33,
                                                                                                                                         "veh.gear_ratio": 7.05, "veh.rated_max_speed": 145, "veh.seats": 5, "veh.home_charger_kw": 7.2}),
         interfaces=["CAN 500 kbit/s, 11-bit identifiers: the real Bolt high-voltage management database (pack voltage, battery temperature, charger)", "OBD-II service 01 (speed, ambient temperature, control-module voltage)",
                     "UDS vendor data identifiers (state of charge, pack current, range, motor temperature) as a dongle reads them", "UDS diagnostic trouble codes", "syslog from the infotainment unit", "NMEA 0183 GNSS", "forward camera pictures (JPEG)",
                     "tyres, odometer and ignition on a synthetic extension CAN"], source=f"{OPENDBC}: gm_global_a_high_voltage_management.dbc"),
]


from tools_cars_data2 import MORE  # noqa: E402
CARS += MORE

try:
    from tools_cars_data3 import MORE3  # noqa: E402  (generated by tools_cars_auto.py)
    CARS += MORE3
except ImportError:
    pass

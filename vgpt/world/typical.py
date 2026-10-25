"""Typical canonical values (a car driving at ~57 km/h on a mild day) - the default state of the virtual vehicle and the reference point of the plug-and-play round-trip test."""
TYPICAL = {
    "speed": 57.3, "accelerator_position": 22.0, "brake_pedal": 0.0, "steering_angle": 4.0, "yaw_rate": 0.8, "lateral_acceleration": 0.25, "longitudinal_acceleration": 0.4, "brake_pressure": 0.0,
    "wheel_speed_fl": 57.3, "wheel_speed_fr": 57.3, "wheel_speed_rl": 57.3, "wheel_speed_rr": 57.3,
    "motor_torque": 92.0, "motor_rpm": 2290.0, "motor_temperature": 41.0, "motor_power": 22.0, "coolant_flow": 12.1, "ambient_temperature": 21.0, "hvac_power": 800.0, "cabin_temperature": 22.5,
    "battery_soc": 63.5, "battery_voltage": 381.0, "battery_current": 42.0, "battery_temperature": 27.5, "cell_temperature_max": 28.5, "cell_temperature_min": 26.5, "cell_voltage_min": 3.71,
    "charging_power": 0.0, "regen_power": 0.0, "aux_battery_voltage": 13.8, "brake_temperature": 95.0,
    "tyre_pressure_fl": 231.0, "tyre_pressure_fr": 233.0, "tyre_pressure_rl": 229.0, "tyre_pressure_rr": 230.0, "tyre_temperature": 28.0, "rain": 0, "light": 20000, "odometer": 12345.6,
    "adas_state": 1, "lead_vehicle_distance": 45.0, "lane_confidence": 0.9, "camera_blockage": 0.02, "camera_sharpness": 1.2, "camera_brightness": 120.0,
    "engine_rpm": 1750.0, "engine_torque": 310.0, "engine_load": 55.0, "fuel_rate": 14.5, "fuel_level": 62.0, "engine_oil_temperature": 96.0, "barometric_pressure": 100.0, "fuel_temperature": 38.0,
    "_charging_active": 0, "_ignition": 1,
}

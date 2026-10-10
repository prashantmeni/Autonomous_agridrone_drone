# Pixhawk Setup (PX4)

This page explains the expected Pixhawk/PX4 configuration for the project. The system depends on PX4 for flight-critical control.

## Purpose

Use this file to validate the airframe, telemetry, and failsafe setup before enabling missions or takeoff logic.

## Step-by-step

### 1. Flash PX4

Use QGroundControl to flash the appropriate PX4 firmware for the selected airframe.

### 2. Calibrate the flight controller

Perform the standard calibration sequence:

- accelerometer
- compass
- gyroscope
- level/ground alignment

### 3. Configure the airframe

Set the vehicle to a quadrotor or the matching airframe used in your test hardware. The project assumes a multicopter configuration.

### 4. Configure failsafes

The project treats PX4 failsafes as authoritative. Examples include:

- low battery behavior
- return-to-launch actions
- RC loss behavior

### 5. Connect the telemetry link

Typical serial connection:

```bash
dmesg | grep ttyACM
ls -l /dev/ttyACM*
```

If using a direct serial link, the project expects settings similar to:

```dotenv
MAVLINK_CONNECTION=/dev/ttyACM0
MAVLINK_BAUD=115200
```

### 6. Verify heartbeat

```bash
python3 scripts/check_mavlink.py
```

The app requires a healthy MAVLink connection before it should proceed with operational workflows.

## Safety rule

PX4 remains responsible for stabilization, actuator control, and emergency behavior. The Pi only orchestrates inputs and health policy.

## Related docs

- [mavlink.md](mavlink.md)
- [safety.md](safety.md)
- [flight-modes.md](flight-modes.md)
- [troubleshooting.md](troubleshooting.md)

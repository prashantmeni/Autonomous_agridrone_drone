# Recovery runbook: re-image the Pi with Raspberry Pi OS Bookworm

## Why

The Pixhawk streams MAVLink correctly when connected to a laptop (349 frames in
8 s, twice, on two cables) but delivers **0 bytes** to the Pi — on two different
USB ports, two different cables, and through both `pyserial` and a raw `dd`
read. The device enumerates cleanly every time (both CDC interfaces bound, no
kernel errors), so descriptors, power and the driver are fine.

The Pi currently runs a fresh **Debian 13 (trixie)** image on kernel
`6.18.50+rpt-rpi-v8`. Bookworm is the well-tested LTS image for the Pi 4, so
this is an image/kernel change, not a hardware change.

This is inference from the evidence above, not proof. If Bookworm behaves
identically, the next diagnostic is a USB-TTL console reading the flight
controller's own output.

## Before you flash

Backups are already saved on the laptop in `C:\Users\prash\pi-preflash-backup`:

| File | What it is |
| --- | --- |
| `.env` | `API_SECRET_KEY`, MAVLink device — **must be restored** |
| `drone.db` | mission/detection/event database |
| `logs/` | telemetry history (~34 MB) |

## Steps

1. Power the Pi off and remove the SD card.
2. Raspberry Pi Imager → Device **Raspberry Pi 4** → OS **Raspberry Pi OS (64-bit)**
   (the Bookworm release, not "other specific OS").
3. Storage: the 32 GB card.
4. Click the gear (OS customisation) and set:
   - hostname: `agridrone`
   - enable **SSH**, username + password
5. Write and verify, then eject.
6. Insert the card in the Pi, boot, and connect it to the **router**.
7. Find the new IP from your router's client list.

## Restore

From a machine with the repository:

```bash
git clone https://github.com/prashantmeni/Autonomous_agridrone_drone.git ~/agridrone
cd ~/agridrone
sudo apt-get update && sudo apt-get install -y python3-venv
./scripts/restore_pi.sh
```

`restore_pi.sh` installs the project into a venv, bridges `picamera2` from
dist-packages, writes `.env` from `.env.example`, installs and enables
`agridrone.service` and the watchdog timer, then waits for the API to answer.

Then restore the real secrets and the dashboard bundle:

```bash
# .env (API secret + MAVLink device) from the laptop backup
scp C:\Users\prash\pi-preflash-backup\.env <pi-user>@<pi-ip>:~/agridrone/.env

# dashboard: dist/ is git-ignored, so it must be copied across
scp -r dashboard/dist/* <pi-user>@<pi-ip>:~/agridrone/dashboard/dist/
```

Finally restart so the service picks up the real `.env`:

```bash
sudo systemctl restart agridrone.service
```

## Then verify the Pixhawk link

```bash
rpicam-hello --list-cameras            # CSI camera must enumerate
systemctl is-active agridrone.service
curl -s localhost:8000/api/drone/status | python3 -m json.tool
curl -s localhost:8000/api/drone/preflight
```

A connected flight controller should make preflight report a real battery
percentage and `ekf: PASS` (from `ESTIMATOR_STATUS`) instead of `UNKNOWN`.

## If Bookworm still delivers 0 bytes

Stop swapping cables and ports — they are already ruled out. Use a USB-TTL
console on the Pixhawk to read its own output (GND→pin 6, adapter RX→pin 8,
TX→pin 10, 115200 baud). That shows whether the FC is transmitting and why,
which a host-side read cannot.
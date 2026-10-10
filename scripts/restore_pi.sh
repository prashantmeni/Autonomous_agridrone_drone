#!/usr/bin/env bash
# Restore the AgriDrone companion computer on a fresh Raspberry Pi OS image.
#
# Rebuilds everything a working deployment needs, in one command:
#   - virtualenv + project install
#   - picamera2 bridge for CSI cameras (picamera2 is an apt package, not PyPI)
#   - .env from .env.example when absent
#   - systemd units (app + API watchdog) pointed at THIS checkout
#   - service and watchdog timer enabled and started
#
# Safe to re-run: every step is idempotent. Does not touch .env if it already
# exists (it holds the API secret and the MAVLink device).
#
# Usage:  ./scripts/restore_pi.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

log() { printf '==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "pyproject.toml" ] || die "run this from inside the repository ($REPO_DIR)"

# --------------------------------------------------------------- python env
log "creating virtualenv (.venv)"
if ! python3 -m venv .venv 2>/dev/null; then
    # Fresh Raspberry Pi OS images ship without venv support; without this the
    # failure surfaces later as a confusing "No such file or directory" when
    # the venv interpreter is missing.
    log "python3-venv is missing; install it with:"
    log "  sudo apt-get update && sudo apt-get install -y python3-venv"
    exit 1
fi
[ -x .venv/bin/python ] || { log "venv creation failed"; exit 1; }
./.venv/bin/python -m pip install --upgrade pip >/dev/null

log "installing project + dev dependencies"
./.venv/bin/python -m pip install -e ".[dev]"

# The crop-disease inference stack is declared in
# ai_disease_detection/requirements.txt but is NOT a dependency of the root
# package, so the install above leaves the Pi without an inference backend and
# every classification fails at runtime. Install the backend explicitly rather
# than the whole file: its opencv-python entry would replace the headless build
# installed above and needs libGL, which a headless Pi does not have.
log "installing onnxruntime (crop-disease inference backend)"
./.venv/bin/python -m pip install "onnxruntime>=1.17"

# ------------------------------------------------- CSI camera (picamera2)
# picamera2 ships as an apt package, so a venv cannot import it. Bridge
# dist-packages in with a .pth, but only when the venv and system interpreters
# share a major.minor version - otherwise the extension modules fail to load.
SYS_SITE=/usr/lib/python3/dist-packages
if [ -d "$SYS_SITE" ] && [ -x .venv/bin/python ]; then
  site=$(./.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  vpy=$(./.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  spy=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if [ -n "$site" ] && [ "$vpy" = "$spy" ]; then
    echo "$SYS_SITE" > "$site/pi_system_packages.pth"
    if ./.venv/bin/python -c "import picamera2" 2>/dev/null; then
      log "picamera2 available in venv (set camera.type: picamera2)"
    else
      log "warning: picamera2 not importable - CSI camera will report CAMERA_UNAVAILABLE"
    fi
  else
    log "picamera2 bridge skipped: venv python $vpy != system $spy"
  fi
else
  log "no $SYS_SITE (not a Raspberry Pi OS image?); skipping camera bridge"
fi

# -------------------------------------------------------------------- .env
if [ -f .env ]; then
  log ".env already present - leaving it untouched"
else
  [ -f .env.example ] && cp .env.example .env && log "created .env from .env.example (edit secrets)"
fi

# ---------------------------------------------------------------- dashboard
# dist/ is a build artifact and is git-ignored, so it is NOT restored by git.
# Build it on a machine with Node, then copy dashboard/dist over, or run
# `npm ci && npm run build` inside dashboard/ on the Pi.
if [ -f dashboard/dist/index.html ]; then
  log "dashboard bundle present"
else
  log "warning: dashboard/dist missing - the web UI will not load."
  log "         build it (npm ci && npm run build in dashboard/) and copy dist/ here."
fi

# ------------------------------------------------------------------ systemd
if ! command -v systemctl >/dev/null 2>&1; then
  log "systemctl not available - skipping service installation"
  exit 0
fi

# agridrone.service writes to logs/drone_prod.log via StandardOutput=append.
# A fresh clone has no logs/ directory and systemd then fails with
# status=209/STDOUT, refusing to start the process at all.
mkdir -p logs data

log "installing systemd units for $REPO_DIR"
sed -e "s#/home/agridrone123/agridrone#${REPO_DIR}#g" \
    systemd/agridrone.service > /tmp/agridrone.service.new
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

$SUDO cp /tmp/agridrone.service.new /etc/systemd/system/agridrone.service
$SUDO cp systemd/agridrone-watchdog.service /etc/systemd/system/
$SUDO cp systemd/agridrone-watchdog.timer /etc/systemd/system/
rm -f /tmp/agridrone.service.new
$SUDO chmod +x scripts/api_watchdog.sh 2>/dev/null || true

$SUDO systemctl daemon-reload
$SUDO systemctl enable --now agridrone.service
$SUDO systemctl enable --now agridrone-watchdog.timer

# -------------------------------------------------------------------- verify
log "waiting for the API to answer..."
for _ in $(seq 1 20); do
  if curl -sf -m 3 -o /dev/null http://127.0.0.1:8000/api/health; then
    log "API is up: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8000/"
    curl -s http://127.0.0.1:8000/api/camera/status || true
    echo
    systemctl is-active agridrone.service
    exit 0
  fi
  sleep 2
done

log "API did not answer yet. Inspect with:"
log "  systemctl status agridrone.service"
log "  journalctl -u agridrone.service -n 50 --no-pager"
exit 1

#!/usr/bin/env bash
set -e
python3 -m venv .venv || true
. .venv/bin/activate 2>/dev/null || true
pip install -e ".[dev]"

# Raspberry Pi CSI cameras: picamera2 is an apt (dist-packages) package and is
# not on PyPI, so a fresh venv cannot see it. Expose dist-packages to the venv
# with a .pth, but only when the venv and system interpreters share an ABI
# (same major.minor) — otherwise the extension modules would fail to load.
sys_site=/usr/lib/python3/dist-packages
if [ -d "$sys_site" ] && [ -x .venv/bin/python ]; then
  site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  vpy=$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  spy=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if [ -n "$site" ] && [ "$vpy" = "$spy" ]; then
    echo "$sys_site" > "$site/pi_system_packages.pth"
    if .venv/bin/python -c "import picamera2" 2>/dev/null; then
      echo "picamera2 available to venv (camera type: picamera2)"
    else
      echo "warning: picamera2 not importable; CSI cameras unavailable"
    fi
  else
    echo "picamera2 bridge skipped: venv python $vpy != system $spy"
  fi
fi
echo "env ready"

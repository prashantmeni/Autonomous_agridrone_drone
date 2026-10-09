#!/usr/bin/env bash
# Recover the companion computer when its API stops answering.
#
# systemd's Restart=always only reacts to the process *exiting*. A blocked
# event loop (e.g. the camera thread wedging the loop, or a hung inference
# call) leaves the process alive but the API unresponsive, which looks exactly
# like "down" to the dashboard and to the pilot. This probes the health
# endpoint and restarts the service after a few consecutive failures.
#
# Health probes that fail because the API is genuinely unhealthy (e.g. the
# flight controller is absent) still return HTTP 200 with a FAULT body, so a
# disconnected PX4 will not trigger a restart loop.
set -u

SERVICE=agridrone.service
URL="${AGRIDRONE_HEALTH_URL:-http://127.0.0.1:8000/api/health}"
THRESHOLD="${AGRIDRONE_FAIL_THRESHOLD:-3}"
STATE="${AGRIDRONE_WATCHDOG_STATE:-/tmp/agridrone-watchdog.failures}"

failures=0
[ -f "$STATE" ] && failures=$(cat "$STATE" 2>/dev/null || echo 0)
case "$failures" in ''|*[!0-9]*) failures=0 ;; esac

if curl -sf -m 10 -o /dev/null "$URL"; then
  if [ "$failures" -gt 0 ]; then
    logger -t agridrone-watchdog "API recovered after $failures failed probe(s)"
  fi
  echo 0 > "$STATE"
  exit 0
fi

failures=$((failures + 1))
echo "$failures" > "$STATE"
logger -t agridrone-watchdog "health probe failed ($failures/$THRESHOLD): $URL"

if [ "$failures" -ge "$THRESHOLD" ]; then
  logger -t agridrone-watchdog "restarting $SERVICE after $failures consecutive failures"
  systemctl restart "$SERVICE" || logger -t agridrone-watchdog "restart command failed"
  echo 0 > "$STATE"
fi

#!/bin/bash
# Log Pi power/throttle state to syslog AND to a durable file.
#
# The syslog copy is easy to lose (journal rotation, volatile storage, or an
# unclean reboot taking the history with it), which is exactly when the record
# matters: an unexpected reboot in flight. The file survives reboots so the
# throttle/undervoltage bit can be read after the fact.
set -u

LOG=/home/agridrone123/agridrone/logs/power.log
mkdir -p "$(dirname "$LOG")"

t=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
v=$(vcgencmd measure_volts core 2>/dev/null | cut -d= -f2)
c=$(vcgencmd measure_temp 2>/dev/null | cut -d= -f2)
mem=$(free -m 2>/dev/null | awk '/^Mem:/ {print $7"/"$2"MB"}')

# get_throttled is a bitmask since boot: 0x1 = under-voltage, 0x2 = throttled,
# 0x4 = soft temperature, 0x8 = hard temperature (Pi 4B/400).
state="ok"
case "$t" in
  0x0) ;;
  *0x1*|*0x2*|*0x4*|*0x8*) state="ALERT" ;;
esac

line="$(date -Is) $state throttled=${t:-unknown} volts=${v:-unknown} temp=${c:-unknown} avail_mem=${mem:-unknown} boot=$(uptime -s) load=$(cut -d' ' -f1-3 /proc/loadavg)"

logger -t power-monitor "$state throttled=$t volts=$v temp=$c boot=$(uptime -s) up=$(uptime -p)"
echo "$line" >> "$LOG"

# Keep the durable log bounded without rotating history away silently.
if [ "$(wc -l < "$LOG")" -gt 20000 ]; then
  tail -n 10000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

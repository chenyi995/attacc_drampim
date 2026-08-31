#!/usr/bin/env bash
# Keep exactly one governor alive.  Safe to re-run -- this is also the restart
# path after a login-node reboot.  Liveness is a pidfile, not pgrep: a pattern
# match on "governor.py" also hits any shell whose command line quotes it.
set -uo pipefail
ORCH=/home/cw636/chenyi/attacc_drampim/output/_orch2
R=$(cat "$ORCH/CURRENT_ROOT")
PIDF=$R/governor.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null \
   && grep -qa "governor.py" "/proc/$(cat "$PIDF")/cmdline" 2>/dev/null; then
  echo "governor already running: pid $(cat "$PIDF")"; exit 0
fi
setsid nohup python3 "$ORCH/governor.py" >> "$R/governor.out" 2>&1 &
echo $! > "$PIDF"
echo "governor started pid=$(cat "$PIDF") -> $R/governor.log"

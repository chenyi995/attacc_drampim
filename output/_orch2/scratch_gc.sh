#!/usr/bin/env bash
# Remove ramdirs that no live process references, then report:  <node> <freeGB> <reclaimedGB>
#
# NOT WIRED UP: governor.py's sweep_scratch() is dead code, deliberately.  On
# 2026-08-31 an earlier version of this script deleted ramdirs that had just
# been created -- make_ramdir builds the directory before the python process
# that references it exists, so "no live process uses it" is briefly true for
# every healthy slot -- and two tasks died in 18s with "Need to install
# ramulator".  MIN_AGE_MIN is that fix: a directory younger than a slot's
# startup window is never a GC candidate, however unreferenced it looks.
# Keep this guard in any future version before re-enabling the caller.
MIN_AGE_MIN=${MIN_AGE_MIN:-15}
H=$(hostname -s)
live=$(for p in $(pgrep -u "$USER" -x python3 2>/dev/null; pgrep -u "$USER" -x ramulator2 2>/dev/null); do
         tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^ATTACC_RAMULATOR_DIR=//p'; done | sort -u)
reclaimed=0
for d in /tmp/kvpim_${USER}_* /localdata/kvpim_${USER}/kvpim_${USER}_*; do
  [ -d "$d" ] || continue
  echo "$live" | grep -qxF "$d" && continue
  # younger than a slot's startup window -> its process may not exist yet
  [ -n "$(find "$d" -maxdepth 0 -mmin +$MIN_AGE_MIN 2>/dev/null)" ] || continue
  sz=$(du -sm "$d" 2>/dev/null | cut -f1)
  rm -rf "$d" 2>/dev/null && reclaimed=$(( reclaimed + ${sz:-0} ))
done
root=/localdata/kvpim_${USER}; [ -w "$root" ] || root=/tmp
echo "$H $(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc '0-9') $(( reclaimed/1024 ))"

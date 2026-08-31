#!/usr/bin/env bash
REPO=/home/cw636/chenyi/attacc_drampim
H=$(hostname -s); fail=0
scratch_root() {
  local base=/localdata/kvpim_${USER}
  if mkdir -p "$base" 2>/dev/null && [ -w "$base" ]; then echo "$base"; else echo /tmp; fi
}
root=$(scratch_root)
rd=$root/kvpim_${USER}_probefast_$$
mkdir -p "$rd" 2>/dev/null
ln -sf "$REPO/ramulator2/ramulator2" "$rd/ramulator2" 2>/dev/null
ln -sf "$REPO/ramulator2/trace_gen"  "$rd/trace_gen" 2>/dev/null
r1=$([ -d "$rd" ] && echo ok || { fail=1; echo BAD; })
r2=$([ -x "$rd/ramulator2" ] && echo ok || { fail=1; echo BAD; })
r3=$([ -e "$rd/trace_gen" ] && echo ok || { fail=1; echo BAD; })
r4=$(touch "$rd/w" 2>/dev/null && echo ok || { fail=1; echo BAD; })
av=$(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc '0-9')
rm -rf "$rd"
echo "[$H] root=$root dir=$r1 ram=$r2 tracegen=$r3 writable=$r4 avail=${av}G $([ $fail = 0 ] && echo PASS || echo FAIL)"

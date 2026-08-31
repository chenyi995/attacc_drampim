#!/usr/bin/env bash
# Log this node's own real usage every 2 min.  Cores come from a /proc
# utime+stime delta -- `ps -o pcpu` is a lifetime average and under-reports a
# process that just got busy, which is exactly the number we must not get wrong.
ROOT=$1
H=$(hostname -s); NCPU=$(nproc)
while true; do
  read -r cores rss < <(python3 - <<'PY'
import os, time
HZ = os.sysconf("SC_CLK_TCK"); me = os.getuid()
def snap():
    d = {}
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            if os.stat(f"/proc/{p}").st_uid != me: continue
            s = open(f"/proc/{p}/stat").read(); f = s[s.rindex(")")+2:].split()
            d[p] = (int(f[11]) + int(f[12]), int(f[21]))
        except (OSError, ValueError, IndexError): continue
    return d
a = snap(); time.sleep(3); b = snap()
cores = sum((v[0] - a.get(k, (v[0], 0))[0]) for k, v in b.items()) / HZ / 3
rss = 0
for p in b:
    try: rss += int(open(f"/proc/{p}/statm").read().split()[1])
    except (OSError, ValueError, IndexError): pass
print(f"{cores:.2f} {rss*os.sysconf('SC_PAGE_SIZE')/2**30:.1f}")
PY
)
  tot=$(free -g | awk '/^Mem:/{print $2}')
  printf "%s %s cores_used=%s/%s rss=%sG/%sG\n" \
    "$(date +%F' '%T)" "$H" "${cores:-0}" "$NCPU" "${rss:-0}" "$tot" >> "$ROOT/usage_${H}.log"
  sleep 120
done

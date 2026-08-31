#!/usr/bin/env bash
# Accurate per-node core usage: sample utime+stime for every process of ours
# from /proc twice, INTERVAL apart, and divide the delta by the interval.
# ps %CPU is a lifetime average and top's batch parse is fragile; this is not.
INT=${1:-4}
python3 - "$INT" <<'PY'
import os, sys, time
HZ = os.sysconf("SC_CLK_TCK")
def snap():
    d = {}
    me = os.getuid()
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            if os.stat(f"/proc/{p}").st_uid != me: continue
            f = open(f"/proc/{p}/stat").read()
            fields = f[f.rindex(")") + 2:].split()
            d[p] = (int(fields[11]) + int(fields[12]),        # utime+stime
                    open(f"/proc/{p}/comm").read().strip())
        except (OSError, ValueError, IndexError):
            continue
    return d
i = float(sys.argv[1])
a = snap(); time.sleep(i); b = snap()
per = {}
tot = 0.0
for p, (t1, c) in b.items():
    t0 = a.get(p, (t1, c))[0]
    cores = (t1 - t0) / HZ / i
    if cores > 0.02:
        per[c] = per.get(c, 0.0) + cores
        tot += cores
print(f"{os.uname().nodename.split('.')[0]} total={tot:.2f} cores  " +
      " ".join(f"{k}={v:.2f}" for k, v in sorted(per.items(), key=lambda x: -x[1])[:5]))
PY

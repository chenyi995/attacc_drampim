#!/usr/bin/env bash
# Per-Slurm-job RSS on this node: group our processes by the SLURM_JOB_ID in
# their own environ.  --mem is a cgroup HARD limit here (ConstrainRAMSpace=yes),
# so under-requesting means an OOM kill, not just slowness.
python3 - <<'PY'
import os, collections
me = os.getuid(); PS = os.sysconf("SC_PAGE_SIZE")
per = collections.Counter(); n = collections.Counter()
for p in os.listdir("/proc"):
    if not p.isdigit(): continue
    try:
        if os.stat(f"/proc/{p}").st_uid != me: continue
        env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
        jid = next((e.split(b"=", 1)[1].decode() for e in env
                    if e.startswith(b"SLURM_JOB_ID=")), None)
        if not jid: continue
        rss = int(open(f"/proc/{p}/statm").read().split()[1]) * PS / 2**30
        per[jid] += rss; n[jid] += 1
    except (OSError, ValueError, IndexError, StopIteration): continue
h = os.uname().nodename.split(".")[0]
for j, v in sorted(per.items(), key=lambda x: -x[1]):
    print(f"  {h} job {j}: {v:7.1f}G  ({n[j]} procs)")
PY

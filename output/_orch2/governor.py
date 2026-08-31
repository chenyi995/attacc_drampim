#!/usr/bin/env python3
"""Autonomous resource governor for the experiment1 756-run sweep (athena).

Policy (user instruction, 2026-08-30):
  * hard ceiling: never hold more than MAX_CORES cores / MAX_MEM_G GB;
  * take idle capacity when it exists, hand it back when it does not -- keep the
    cluster's unallocated core count at or above MIN_IDLE;
  * do not sit on cores we are not using.

Cores are the only consumable resource on this cluster
(SelectTypeParameters=CR_CORE, AllocMem=0 on every node), so the core count is
the real lever; --mem is only our own cgroup hard limit and does not block
anyone else's scheduling.

Granularity is one worker slot = SLOT_CORES cores = 9 rungs x (1 + RAMU_WORKERS).
Shrinking is graceful first: a drain flag makes the slot exit *after* its
current task, so no simulation work is thrown away.  Cancelling outright only
happens if a drain has not landed within DRAIN_DEADLINE while the cluster is
still tight, and then the claim is released so the task returns to the queue.
"""
import json, os, re, shutil, subprocess, sys, time

REPO = "/home/cw636/chenyi/attacc_drampim"
ORCH = f"{REPO}/output/_orch2"
USER = os.environ.get("USER", "cw636")

MAX_CORES   = int(os.environ.get("GOV_MAX_CORES", 300))    # user ceiling
# Memory rule (user, 2026-08-30): what must not be exceeded is EACH NODE's
# total memory -- ours plus every other user's -- at 95%.  There is no global
# ceiling on our own footprint.  Poll interval is minutes and a task's RSS can
# climb fast, so the guards trip below the line, not at it.
MAX_RSS_G      = int(os.environ.get("GOV_MAX_RSS_G", 4800))         # ours; raised to match what the nodes can supply at 95%
NODE_MEM_STOP  = float(os.environ.get("GOV_NODE_MEM_STOP", 0.62))   # place no new slot
NODE_MEM_DRAIN = float(os.environ.get("GOV_NODE_MEM_DRAIN", 0.72))  # drain one slot
NODE_MEM_KILL  = float(os.environ.get("GOV_NODE_MEM_KILL", 0.88))   # the line; 90-95% is unreachable safely at our sampling rate
MIN_IDLE    = int(os.environ.get("GOV_MIN_IDLE", 128))     # leave this many free
PERIOD      = int(os.environ.get("GOV_PERIOD", 90))
DRAIN_DEADLINE = int(os.environ.get("GOV_DRAIN_DEADLINE", 2700))  # 45 min
# A task is 9 rungs of single-threaded graph build + a small shared Ramulator
# pool; measured at ~10 cores, NOT the 27 its process count suggests.  This is
# re-derived from measured utilisation every cycle and kept in [8, 18].
SLOT_CORES_DEFAULT = int(os.environ.get("GOV_SLOT_CORES", 12))
SLOT_CORES_MIN, SLOT_CORES_MAX = 8, 18
MAX_BIG_TOTAL      = int(os.environ.get("GOV_MAX_BIG", 12))   # 2 per node
MAX_BIG_PER_NODE   = 2   # ~300G a big task with the collector on.  The sweep's
                         # makespan is set by how many big slots fit in RAM --
                         # big is 245 task-hours against small's 109 -- so this
                         # is maximised deliberately and the per-node memory
                         # guards, not this cap, are what hold the line.
# Measured on this sweep: resident memory tracks the decode work almost
# linearly at ~23 GB per work unit, so a slot's need is known BEFORE it is
# placed.  Counting slots was the wrong control variable -- a big task is
# 340-500G and a small one ~120G, and it was placing by count that kept
# pushing nodes past the line.
# 2026-08-31: stop modelling and bound it.  The projection is a per-TASK
# average, but peak memory is set by the A1 rung, which is no-reuse and so
# gets none of the shrink the other eight rungs get -- on LLAMA-65B k-lo the
# projection said 335G for a node that reached 956G.  Rather than refine the
# model again (four attempts tonight, each costing killed mid-flight tasks),
# cap concurrency where the worst OBSERVED slot fits: ~320G x 2 + ~150G of
# other users = 790G, under the 805G line.  Costs ~4 hours of makespan.
MAX_SLOTS_PER_NODE = 3   # a task's RSS keeps growing, so the fraction guards
                         # are reactive; capping slots per node is what keeps a
                         # node off the 80% line in the first place
NODES = ["node1", "node2", "node3", "node4", "node5", "node6"]
# node5 has only 49 GB of usable scratch (/tmp on /) and its /localdata is
# root-owned, while one trace-heavy rung can need 128 GB.  Keep big-model tasks
# off it (author's ruling 2026-08-31); small tasks still run there.
BIG_EXCLUDE = {"node5"}
# This sweep is CPU-only (no torch/cuda anywhere, no --gres=gpu on any job), so
# it has no business sitting on the GPU partition's cores when a plain CPU node
# would do.  Place on node1-4 first; node5/6 (athena-genai) only as a spillover.
NODE_PREF = {"node1": 0, "node2": 0, "node3": 0, "node4": 0, "node5": 1, "node6": 1}
GEOM = {"GPT-175B": (96, 96, 1), "LLAMA-65B": (80, 64, 1), "LLAMA-33B": (60, 52, 1),
        "GPT-13B": (40, 40, 1), "LLAMA-7B": (32, 32, 1), "LLAMA3-8B": (32, 32, 4)}
_TOKCACHE = {}


def _tok(wl):
    if wl not in _TOKCACHE:
        d = json.load(open(f"{REPO}/workload/sweep/{wl}"))
        _TOKCACHE[wl] = sum(sum(s["len"] for s in a["segs"]) for a in d["agents"])
    return _TOKCACHE[wl]


BIG_MEM_G   = int(os.environ.get("GOV_BIG_MEM_G", 450))
SMALL_MEM_G = int(os.environ.get("GOV_SMALL_MEM_G", 130))
# projected = SLOT_BASE_G + MEM_PER_W x W.  The linear term is the graph
# itself (measured 22.7 and 24.6 GB per work unit at two configurations); the
# constant is the nine rung processes' own footprint, which a pure per-W model
# left out and under-predicted the total by ~29%.
MEM_PER_W   = float(os.environ.get("GOV_MEM_PER_W", 25.0))
SLOT_BASE_G = float(os.environ.get("GOV_SLOT_BASE_G", 40.0))


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout
    except Exception as e:
        log(f"WARN command failed: {cmd}: {e}")
        return ""


def root():
    return open(f"{ORCH}/CURRENT_ROOT").read().strip()


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    try:
        with open(f"{root()}/governor.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def cluster_state():
    """-> {node: dict(alloc, idle, other, total)} from sinfo."""
    out = sh('sinfo -N -h -O "NodeList:16,CPUsState:24" | sort -u')
    st = {}
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) < 2:
            continue
        node, cpus = parts[0], parts[1]
        m = re.match(r"(\d+)/(\d+)/(\d+)/(\d+)", cpus)
        if not m or node not in NODES:
            continue
        a, i, o, t = (int(x) for x in m.groups())
        st[node] = dict(alloc=a, idle=i, other=o, total=t)
    return st


def my_jobs():
    out = sh(f'squeue -u {USER} -h -o "%i|%j|%t|%C|%m|%N"')
    jobs = []
    for ln in out.splitlines():
        f = ln.split("|")
        if len(f) < 6:
            continue
        jid, name, state, cores, mem, node = (x.strip() for x in f)
        node = node or (name.split("_")[-1] if name.split("_")[-1] in NODES else "")
        memg = 0
        mm = re.match(r"(\d+)([GMT]?)", mem)
        if mm:
            memg = int(mm.group(1)) * {"G": 1, "T": 1024, "M": 0, "": 0}.get(
                mm.group(2), 1)
        jobs.append(dict(jid=jid, name=name, state=state, cores=int(cores or 0),
                         memg=memg, node=node,
                         kind=("slot" if name.startswith("slot")
                               else "warm" if name.startswith("warm_") else "other"),
                         cls=("big" if name.startswith("slotb") else
                              "small" if name.startswith("slots") else "")))
    return jobs


def _task_of_slot(R, jid):
    """(model, config) the slot is currently on, from its own START/END log."""
    out = sh(f'grep -ah "job={jid} " {R}/slurm/slot_*.out 2>/dev/null | tail -1')
    m = re.search(r"START (\S+) (\S+) k=", out)
    return (m.group(1), m.group(2)) if m else None


_WCACHE = {}


def _wdec(model, wl):
    """Decode work of one task: output tokens x layers x context per agent."""
    key = (model, wl)
    if key not in _WCACHE:
        d = json.load(open(f"{REPO}/workload/sweep/{wl}"))
        a = d["agents"]
        tok = sum(sum(s["len"] for s in x["segs"]) for x in a)
        lout = sum(x["lout"] for x in a)
        _WCACHE[key] = lout * GEOM[model][0] * (tok / len(a)) / 1e9
    return _WCACHE[key]


def projected_gb(R, jid, cls):
    """What this slot will hold once its task stops growing.

    Checking CURRENT usage was the bug: a slot placed this cycle has not grown
    yet, so the next cycle saw its space as free and placed another on top of
    it.  Resident memory tracks decode work at ~23 GB per unit (measured on
    this sweep at three configurations), so the commitment is knowable up front.
    """
    t = _task_of_slot(R, jid)
    default = BIG_MEM_G if cls == "big" else SMALL_MEM_G
    if t is None:
        return default
    for f in ("tasks_big.txt", "tasks_small.txt"):
        try:
            for ln in open(f"{R}/{f}"):
                g = ln.split()
                if len(g) >= 4 and g[0] == t[0] and g[1] == t[1]:
                    return SLOT_BASE_G + MEM_PER_W * _wdec(t[0], g[2])
        except OSError:
            pass
    return default


def remaining_work(R):
    """Unclaimed-plus-in-flight work per class, in token x layers x KV-heads.

    Slots are split by WORK, not by queue length, and the count must include
    tasks already running: counting only the unclaimed queue made the share
    collapse to zero the moment the last task was picked up, which drained
    slots that were mid-task.
    """
    out = {"big": 0.0, "small": 0.0}
    for cls, f in (("big", "tasks_big.txt"), ("small", "tasks_small.txt")):
        path = f"{R}/{f}"
        if not os.path.exists(path):
            continue
        for ln in open(path):
            g = ln.split()
            if len(g) < 4:
                continue
            tid = f"{g[0]}__{g[1]}_k{g[3]}"
            c = f"{R}/claims/{tid}"
            if os.path.isdir(c) and os.path.exists(f"{c}/done"):
                continue
            lay, hd, gq = GEOM[g[0]]
            out[cls] += _tok(g[2]) * lay * (hd // gq) / 1e9
    return out


_HU = {"t": 0.0, "v": {}}


def heaviest_unclaimed(R, ttl=60.0):
    """Projected GB of the heaviest UNCLAIMED task in each class.

    This is what a slot placed right now will actually pick up: the queue is
    ordered longest-first and a worker takes the first task it can claim.  A
    reservation smaller than this admits a slot the node cannot hold.  Returns
    0 for a class whose queue is empty, so placement falls back to the class
    constant.
    """
    now = time.time()
    if now - _HU["t"] < ttl:
        return _HU["v"]
    out = {"big": 0.0, "small": 0.0}
    for cls, f in (("big", "tasks_big.txt"), ("small", "tasks_small.txt")):
        path = f"{R}/{f}"
        if not os.path.exists(path):
            continue
        for ln in open(path):
            g = ln.split()
            if len(g) < 4:
                continue
            c = f"{R}/claims/{g[0]}__{g[1]}_k{g[3]}"
            if os.path.isdir(c):        # claimed, parked, damaged or done
                continue
            out[cls] = max(out[cls], SLOT_BASE_G + MEM_PER_W * _wdec(g[0], g[2]))
    _HU.update(t=now, v=out)
    return out


def queue_stats(R):
    """(class -> [total, claimed]) plus the overall totals."""
    out = {}
    for cls, f in (("big", "tasks_big.txt"), ("small", "tasks_small.txt")):
        path = f"{R}/{f}"
        if not os.path.exists(path):
            out[cls] = [0, 0]
            continue
        tids = []
        for ln in open(path):
            g = ln.split()
            if len(g) >= 4:
                tids.append(f"{g[0]}__{g[1]}_k{g[3]}")
        claimed = sum(1 for t in tids if os.path.isdir(f"{R}/claims/{t}"))
        out[cls] = [len(tids), claimed]
    return out


def node_mem(jobs):
    """Per node: (used_fraction_all_users, our_GB, total_GB).

    MemTotal-MemAvailable is what actually fills a node -- MemAvailable already
    discounts reclaimable page cache, so this does not mistake cache for
    pressure.  Our own share comes from the Slurm job cgroups.
    """
    out_map = {}
    seen = set()
    for j in jobs:
        if j["state"] != "R" or not j["node"] or j["node"] in seen:
            continue
        seen.add(j["node"])
        out = sh(f'timeout 45 srun --jobid={j["jid"]} --overlap -w {j["node"]} '
                 f'bash -c \'H=$(hostname -s); '
                 f'MT=$(awk "/MemTotal/{{print \\$2}}" /proc/meminfo); '
                 f'MA=$(awk "/MemAvailable/{{print \\$2}}" /proc/meminfo); '
                 f'o=0; for d in /sys/fs/cgroup/memory/slurm_$H/uid_$(id -u)/job_*; do '
                 f'[ -d "$d" ] || continue; '
                 f'o=$(( o + $(cat $d/memory.usage_in_bytes 2>/dev/null || echo 0) )); done; '
                 f'echo "$H $MT $MA $(( o / 1048576 ))"\' 2>/dev/null', timeout=60)
        for ln in out.splitlines():
            f = ln.split()
            if len(f) == 4 and f[0] in NODES:
                try:
                    mt, ma, ours = int(f[1]), int(f[2]), int(f[3])
                except ValueError:
                    continue
                out_map[f[0]] = ((mt - ma) / mt, ours / 1024.0, mt / 1048576.0)
    return out_map


def cgroup_mem(jobs):
    """Per-node GB we actually hold, read from the Slurm job cgroups.

    Summing /proc RSS per process double-counts every copy-on-write page shared
    by a slot's ~23 forked processes and under-reported the true total by ~2x,
    which is exactly the number the RAM ceiling depends on.  memory.usage_in_bytes
    on the job cgroup is the number the kernel itself uses.
    """
    per = {}
    seen = set()
    for j in jobs:
        if j["state"] != "R" or not j["node"] or j["node"] in seen:
            continue
        seen.add(j["node"])
        out = sh(f'timeout 45 srun --jobid={j["jid"]} --overlap -w {j["node"]} '
                 f'bash -c \'H=$(hostname -s); t=0; '
                 f'for d in /sys/fs/cgroup/memory/slurm_$H/uid_$(id -u)/job_*; do '
                 f'[ -d "$d" ] || continue; '
                 f't=$(( t + $(cat $d/memory.usage_in_bytes 2>/dev/null || echo 0) )); done; '
                 f'echo "$H $(( t / 1073741824 ))"\' 2>/dev/null', timeout=60)
        for ln in out.splitlines():
            f = ln.split()
            if len(f) == 2 and f[0] in NODES and f[1].isdigit():
                per[f[0]] = int(f[1])
    return per


def measured_usage(R):
    """What we are ACTUALLY using right now, per node, from the in-job watchers
    (a /proc utime delta, not ps's lifetime average).  Stale lines are dropped:
    a node with no slot on it has nobody writing."""
    cores = rss = 0.0
    fresh = {}
    now = time.time()
    for n in NODES:
        f = f"{R}/usage_{n}.log"
        if not os.path.exists(f) or now - os.path.getmtime(f) > 400:
            continue
        try:
            last = open(f).read().strip().splitlines()[-1]
        except (OSError, IndexError):
            continue
        mc = re.search(r"cores_used=([\d.]+)", last)
        mr = re.search(r"rss=([\d.]+)G", last)
        if mc and mr:
            fresh[n] = (float(mc.group(1)), float(mr.group(1)))
            cores += float(mc.group(1))
            rss += float(mr.group(1))
    return cores, rss, fresh


def slot_cores_estimate(R, used_cores, slots_running):
    """Allocate what we use: re-derive the per-slot core size from measurement.

    A slot that measures far under its allocation is hoarding; one that pegs it
    is being throttled.  Kept inside [8, 18] because 9 single-threaded rung
    builds set the floor and the Ramulator burst sets the ceiling.
    """
    path = f"{R}/slot_cores"
    cur = SLOT_CORES_DEFAULT
    if os.path.exists(path):
        try:
            cur = int(open(path).read().strip())
        except ValueError:
            pass
    if slots_running >= 3 and used_cores > 0:
        per = used_cores / slots_running
        want = max(SLOT_CORES_MIN, min(SLOT_CORES_MAX, int(round(per * 1.25))))
        if abs(want - cur) < 2:          # hysteresis: ignore one-core wobble
            want = cur
        if want != cur:
            log(f"SLOT SIZE {cur} -> {want} cores "
                f"(measured {per:.1f} cores/slot over {slots_running} slots, +25% headroom)")
            with open(path, "w") as f:
                f.write(str(want))
            cur = want
    return cur


def sweep_scratch(jobs):
    """Delete ramdirs no live process is using, and report free scratch.

    A killed slot never runs worker.sh's cleanup, so its trace directory leaks.
    Enough of those filled node5/node6's 216 GB /tmp on 2026-08-31 and three
    rungs died with ENOSPC -- a failure that looks nothing like the memory ones
    and needs its own guard.
    """
    free = {}
    seen = set()
    for j in jobs:
        if j["state"] != "R" or not j["node"] or j["node"] in seen:
            continue
        seen.add(j["node"])
        out = sh(f'timeout 90 srun --jobid={j["jid"]} --overlap -w {j["node"]} '
                 f'bash {ORCH}/scratch_gc.sh 2>/dev/null', timeout=120)
        for ln in out.splitlines():
            f = ln.split()
            if len(f) == 3 and f[0] in NODES:
                try:
                    free[f[0]] = int(f[1])
                    if int(f[2]) > 0:
                        log(f"SCRATCH {f[0]}: reclaimed {f[2]}G of orphaned "
                            f"ramdirs, {f[1]}G free")
                except ValueError:
                    pass
    return free


# A missing rung whose log carries one of these is evidence the NODE ran out of
# something, not that the rung is broken.  Both HOLDs so far (05:51 GPT-175B
# "Killed", 14:55 LLAMA-7B "Errno 28") were of this kind, and A3b/A4b have since
# produced substantive output in every one of the 11 tasks that finished 9/9.
INFRA_CAUSE = re.compile(
    r"No space left|Errno 28|Errno 12|Killed|Need to install ramulator|"
    r"MemoryError|Cannot allocate memory|Out of memory",
    re.I)


def check_new_rungs(R):
    """A3b / A4b are new on this machine.  A finished task that produced no
    dag_A3b.json / dag_A4b.json is the real evidence, so check the outputs of
    tasks that actually completed rather than trusting a log grep.

    Returns (bad, infra).  Only ``bad`` -- an *unexplained* miss, which is what
    this guard was actually for -- may set HOLD; ``infra`` is logged so the rung
    still gets backfilled, without stopping phase 2 for the other 40+ tasks."""
    bad, infra = [], []
    claims = f"{R}/claims"
    if os.path.isdir(claims):
        for tid in sorted(os.listdir(claims)):
            if not os.path.exists(f"{claims}/{tid}/done"):
                continue
            if os.path.exists(f"{claims}/{tid}/excluded") or \
               os.path.exists(f"{claims}/{tid}/parked") or \
               os.path.exists(f"{claims}/{tid}/damaged"):
                continue        # dropped, parked, or already known damaged
            model, rest = tid.split("__", 1)
            out = f"{R}/{model}/{rest}"
            for rung in ("A3b", "A4b"):
                if not os.path.exists(f"{out}/dag_{rung}.json"):
                    tail = sh(f'tail -3 "{out}/dag_{rung}.log" 2>/dev/null')
                    msg = (f"{tid}: dag_{rung}.json missing after the task "
                           f"finished | {tail.strip()[:200]}")
                    (infra if INFRA_CAUSE.search(tail) else bad).append(msg)
    return bad, infra


def reap_claims(R, live_jids):
    """Release claims whose owning slot died mid-task, so the task can be redone.

    'Died mid-task' = owner job gone from squeue AND the worker never wrote its
    done marker.  A task that finished badly still has a done marker, so it is
    never silently retried in a loop.
    """
    claims = f"{R}/claims"
    if not os.path.isdir(claims):
        return 0
    n = 0
    for tid in sorted(os.listdir(claims)):
        d = f"{claims}/{tid}"
        if os.path.exists(f"{d}/done"):
            continue
        try:
            owner = open(f"{d}/owner").read()
        except OSError:
            continue
        m = re.search(r"jobid=(\S+)", owner)
        if not m or m.group(1) in live_jids or m.group(1) == "nojob":
            continue
        shutil.rmtree(d, ignore_errors=True)
        log(f"REAP released abandoned claim {tid} (owner job {m.group(1)} gone, no done marker)")
        n += 1
    return n


def drain_path(R, jid):
    return f"{R}/drain/{jid}"


_SCRATCH = {}


def plan(state, jobs, R):
    run = [j for j in jobs if j["state"] == "R"]
    pend = [j for j in jobs if j["state"] == "PD"]
    my_cores = sum(j["cores"] for j in run)
    idle = sum(v["idle"] for v in state.values())
    free_pool = idle + my_cores

    used_cores, _rss_unused, per_node = measured_usage(R)
    nm = node_mem(run)                       # node -> (used_frac_all_users, ours_G, total_G)
    mem_node = {n: v[1] for n, v in nm.items()}          # our GB per node
    node_frac = {n: v[0] for n, v in nm.items()}         # all-user fill per node
    used_rss = float(sum(mem_node.values()))             # our total, against MAX_RSS_G
    slots_run = [j for j in run if j["kind"] == "slot"]
    slots_pd = [j for j in pend if j["kind"] == "slot"]
    slot_cores = slot_cores_estimate(R, used_cores, len(slots_run))

    budget = max(0, min(MAX_CORES, free_pool - MIN_IDLE))
    other = sum(j["cores"] for j in run if j["kind"] != "slot")
    # Count REAL allocated cores, not slots x the current size estimate: slots
    # submitted before a size change still hold their original allocation, and
    # counting them at the new size is how the 300-core ceiling got breached.
    cores_slots = sum(j["cores"] for j in slots_run + slots_pd)
    cur_total = len(slots_run) + len(slots_pd)
    headroom = budget - other - cores_slots
    if headroom >= 0:
        n_total = cur_total + headroom // slot_cores
    else:
        avg = max(1, cores_slots // max(1, cur_total))
        n_total = max(0, cur_total - ((-headroom + avg - 1) // avg))

    q = queue_stats(R)
    want_big = min(MAX_BIG_TOTAL, max(0, q["big"][0] - q["big"][1]))
    want_small = max(0, q["small"][0] - q["small"][1])
    draining_ids = (set(os.listdir(f"{R}/drain"))
                    if os.path.isdir(f"{R}/drain") else set())
    cur_big = len([j for j in slots_run
                   if j["cls"] == "big" and j["jid"] not in draining_ids])
    cur_small = len([j for j in slots_run
                     if j["cls"] == "small" and j["jid"] not in draining_ids])
    # RAM, not cores, is what binds here: --mem is NOT enforced on this cluster
    # (CR_CORE, so every job cgroup gets the whole node as its limit), which
    # means nothing but this governor stops us eating a node's memory.
    mem_state = "ok"
    if used_rss > 0.86 * MAX_RSS_G:
        shed = max(1, (cur_big + cur_small) // 8)
        want_small = max(0, cur_small - shed)
        want_big = cur_big
        mem_state = f"OVER-{shed}"
    elif used_rss > 0.72 * MAX_RSS_G:
        want_big = min(want_big, cur_big)
        want_small = min(want_small, cur_small)
        mem_state = "frozen"
    rem = remaining_work(R)
    tot_rem = rem["big"] + rem["small"]
    share = (rem["big"] / tot_rem) if tot_rem > 0 else 0.0
    # Commit each slot's PROJECTED footprint against the node, plus whatever
    # the other users are actually holding.  Using our current usage here is
    # what let the same gigabytes be handed out cycle after cycle.
    committed = {n: 0.0 for n in NODES}
    for j in slots_run + slots_pd:
        if j["node"] in committed:
            committed[j["node"]] += projected_gb(R, j["jid"], j["cls"])
    node_head = {}
    for n in NODES:
        frac, ours, tot = nm.get(n, (0.0, 0.0, 1008.0))
        others = max(0.0, frac * tot - ours)
        node_head[n] = max(0.0, NODE_MEM_STOP * tot - others - committed[n])
    committed_total = sum(committed.values())

    run_big = len([j for j in slots_run if j["cls"] == "big"])
    run_small = len([j for j in slots_run if j["cls"] == "small"])
    # never plan below what is already running: a slot mid-task is doing useful
    # work, and draining it just to re-grow later is pure churn.  Shrinking
    # happens only through the explicit core/RAM guards above.
    want_big = min(MAX_BIG_TOTAL, want_big + run_big) if want_big else run_big
    want_small = want_small + run_small if want_small else run_small
    n_big = min(want_big, MAX_BIG_TOTAL,
                max(int(round(n_total * share)), run_big if mem_state == "ok" else 0))
    n_small = min(want_small, n_total - n_big)

    # Derive the targets from the RAM budget, not from a slot count.  A count
    # cap cannot know that twelve big tasks project ~4.2 TB on their own, so
    # the governor kept wanting to grow while the placement guard kept refusing
    # -- and every slot the per-node guard killed was immediately re-targeted.
    avg_big = avg_small = 0.0
    nb = ns = 0
    for cls, f in (("big", "tasks_big.txt"), ("small", "tasks_small.txt")):
        path = f"{R}/{f}"
        if not os.path.exists(path):
            continue
        for ln in open(path):
            g = ln.split()
            if len(g) < 4:
                continue
            tid = f"{g[0]}__{g[1]}_k{g[3]}"
            if os.path.exists(f"{R}/claims/{tid}/done"):
                continue
            gb = SLOT_BASE_G + MEM_PER_W * _wdec(g[0], g[2])
            if cls == "big":
                avg_big += gb; nb += 1
            else:
                avg_small += gb; ns += 1
    avg_big = (avg_big / nb) if nb else BIG_MEM_G
    avg_small = (avg_small / ns) if ns else SMALL_MEM_G
    # NOTE: these averages describe the QUEUE, not what is running.  The queue
    # is longest-first, so the running set is always the heaviest tasks and an
    # average over what is left always claims there is room.  They are reported
    # only; the binding decision below is made on real commitments.
    # what the nodes can actually hold, and what we allow ourselves
    usable = sum(max(0.0, NODE_MEM_KILL * t - max(0.0, f * t - o))
                 for f, o, t in nm.values()) if nm else float(MAX_RSS_G)
    mem_budget = min(float(MAX_RSS_G), usable)
    # Shed by real commitment: if what is already running projects past the
    # budget, the target must fall BELOW the running count, or the governor
    # keeps re-targeting every slot the per-node guard kills.
    over = committed_total - mem_budget
    if over > 0:
        by_size = sorted(((projected_gb(R, j["jid"], j["cls"]), j) for j in slots_run),
                         key=lambda x: -x[0])
        shed_big = shed_small = 0
        freed = 0.0
        for gb, j in by_size:
            if freed >= over:
                break
            freed += gb
            if j["cls"] == "big":
                shed_big += 1
            else:
                shed_small += 1
        n_big = max(0, min(n_big, run_big - shed_big))
        n_small = max(0, min(n_small, run_small - shed_small))

    room = {}
    for n in NODES:
        s = state.get(n)
        if not s:
            room[n] = 0
            continue
        mine = sum(j["cores"] for j in run if j["node"] == n)
        others = max(0, s["alloc"] - mine)
        room[n] = max(0, min(MAX_SLOTS_PER_NODE,
                             (s["total"] - others) // slot_cores))
        # do not add to a node that is already filling up -- the fraction counts
        # EVERY user's memory, because the kernel OOM killer does not care whose
        # pages they are
        if node_frac.get(n, 0.0) > NODE_MEM_STOP:
            room[n] = 0
        if _SCRATCH.get(n, 10**6) < 150:      # GB; one trace-heavy task needs ~128
            room[n] = 0
    return dict(node_head=node_head, committed=committed_total,
                mem_budget=mem_budget, avg_big=avg_big, avg_small=avg_small,
                idle=idle, free_pool=free_pool, budget=budget, my_cores=my_cores,
                used_cores=used_cores, used_rss=used_rss, per_node=per_node,
                mem_node=mem_node, node_frac=node_frac, mem_state=mem_state,
                cores_slots=cores_slots,
                rem=rem,
                slot_cores=slot_cores, n_big=n_big, n_small=n_small,
                q=q, room=room, other_cores=other,
                slots_run=slots_run, slots_pd=slots_pd)


def reconcile(R, p, hold):
    acts = []
    plan_by_cls = {"big": p["n_big"], "small": p["n_small"]}

    def state(cls):
        run = [j for j in p["slots_run"] if j["cls"] == cls]
        pd = [j for j in p["slots_pd"] if j["cls"] == cls]
        draining = [j for j in run if os.path.exists(drain_path(R, j["jid"]))]
        return run, pd, draining, len(run) + len(pd) - len(draining)

    # Shrink every class BEFORE growing any of them.  Reconciling the classes
    # independently, in order, is how the 300-core ceiling got overshot: the
    # grow for one class ran while the other still held the cores it was about
    # to give back.
    for cls in ("big", "small"):
        target = 0 if hold else plan_by_cls[cls]
        run, pd, draining, have = state(cls)
        if have <= target:
            continue
        over = have - target
        for j in pd[:over]:                      # pending slots cost nothing to drop
            sh(f"scancel {j['jid']}")
            acts.append(f"SHRINK cancelled pending {cls} slot {j['jid']}")
            over -= 1
        live = [j for j in run if j not in draining]
        for j in sorted(live, key=lambda x: (-NODE_PREF.get(x["node"], 9),
                                             -int(x["jid"])))[:over]:
            with open(drain_path(R, j["jid"]), "w") as f:
                f.write(str(int(time.time())))
            acts.append(f"SHRINK drain {cls} slot {j['jid']} on {j['node']} "
                        f"(stops after its current task)")

    # Grow only with cores that are free RIGHT NOW, counted from the real
    # allocation rather than from slot counts.
    cores_now = sum(j["cores"] for j in p["slots_run"] + p["slots_pd"])
    headroom = p["budget"] - p["other_cores"] - cores_now
    mem_head = dict(p["node_head"])                 # GB free under the stop line
    mem_left = MAX_RSS_G - p["committed"]          # our ceiling, on commitments
    room = dict(p["room"])
    bignodes = {}
    for j in p["slots_run"] + p["slots_pd"]:
        if j["node"] in room and room[j["node"]] > 0:
            room[j["node"]] -= 1
        if j["cls"] == "big":
            bignodes[j["node"]] = bignodes.get(j["node"], 0) + 1
    for cls in ("big", "small"):
        if hold:
            break
        target = plan_by_cls[cls]
        run, pd, draining, have = state(cls)
        need = target - have
        # Cheapest growth is cancelling a drain -- but only one the memory guard
        # did not set.  Undraining a slot whose node is still over the stop line
        # puts the two rules in a loop: on 2026-08-31 the guard re-drained
        # slot 190622 on node3 and this line undrained it, once per 90s cycle
        # for ten minutes, until the node reached 90% and the slot was killed
        # outright.  A drain that keeps being cancelled is not a drain.
        undrainable = [j for j in draining
                       if p["node_frac"].get(j["node"], 0.0) <= NODE_MEM_STOP]
        for j in undrainable[:max(0, need)]:
            os.remove(drain_path(R, j["jid"]))
            acts.append(f"UNDRAIN {cls} slot {j['jid']} on {j['node']}")
            need -= 1
        for j in draining:
            if j not in undrainable:
                acts.append(f"KEEP-DRAIN {cls} slot {j['jid']} on {j['node']}: "
                            f"node at {100 * p['node_frac'].get(j['node'], 0):.0f}%")
        while need > 0:
            if headroom < p["slot_cores"]:
                break
            # Reserve for the task this slot will ACTUALLY pick up, not for the
            # class average.  Workers pull longest-first, so a new slot takes
            # the heaviest unclaimed task in its class -- and the class constant
            # can be wildly under it.  LLAMA3-8B/N-hi projects 473G and was
            # admitted against SMALL_MEM_G=130, which is how the heaviest task
            # in the sweep came to share node3 with a big task: 717G of
            # predicted demand on a 1008G node, past a 62% stop line it should
            # never have cleared.  It was cancelled 8h in with 8 of 9 rungs
            # unfinished.
            need_g = max(BIG_MEM_G if cls == "big" else SMALL_MEM_G,
                         heaviest_unclaimed(R).get(cls, 0.0))
            if mem_left < need_g:
                acts.append(f"NOGROW {cls}: only {mem_left:.0f}G left under our "
                            f"{MAX_RSS_G}G ceiling")
                break
            cand = [n for n in room if room[n] > 0 and mem_head.get(n, 0) >= need_g and
                    n not in (BIG_EXCLUDE if cls == "big" else ()) and
                    (cls != "big" or bignodes.get(n, 0) < MAX_BIG_PER_NODE)]
            if not cand:
                acts.append(f"NOGROW no node has {need_g}G free for another {cls} slot")
                break
            n = min(cand, key=lambda k: (NODE_PREF.get(k, 9), -room[k]))
            out = sh(f'bash {ORCH}/slot_submit.sh {n} {cls}')
            room[n] -= 1
            headroom -= p["slot_cores"]
            mem_head[n] -= need_g
            mem_left -= need_g
            if cls == "big":
                bignodes[n] = bignodes.get(n, 0) + 1
            acts.append(f"GROW +1 {cls} slot on {n} ({headroom}c budget left): {out.strip()}")
            need -= 1

    # Per-node RAM emergency.  --mem is not enforced here, so the only thing
    # between us and the kernel OOM killer -- which would take other users'
    # processes down too -- is this check.  Drain at 0.72, cancel at 0.88.
    for n, frac in sorted(p["node_frac"].items(), key=lambda x: -x[1]):
        here = [j for j in p["slots_run"] if j["node"] == n]
        if not here:
            continue
        ours = p["mem_node"].get(n, 0)
        if frac > NODE_MEM_KILL:
            victim = sorted(here, key=lambda x: -int(x["jid"]))[0]
            sh(f"scancel {victim['jid']}")
            acts.append(f"RAM-EMERGENCY {n} is {100*frac:.0f}% full (all users; ours "
                        f"{ours:.0f}G) -- cancelled slot {victim['jid']}; claim requeues")
        elif frac > NODE_MEM_DRAIN and not any(
                os.path.exists(drain_path(R, j["jid"])) for j in here):
            for j in sorted(here, key=lambda x: -int(x["jid"]))[:1]:
                d = drain_path(R, j["jid"])
                if not os.path.exists(d):
                    with open(d, "w") as f:
                        f.write(str(int(time.time())))
                    acts.append(f"RAM-PRESSURE {n} is {100*frac:.0f}% full (all users; "
                                f"ours {ours:.0f}G) -- draining slot {j['jid']}")

    if p["idle"] < MIN_IDLE:
        for j in p["slots_run"]:
            d = drain_path(R, j["jid"])
            if not os.path.exists(d):
                continue
            try:
                t0 = int(open(d).read().strip())
            except (OSError, ValueError):
                continue
            if time.time() - t0 > DRAIN_DEADLINE:
                sh(f"scancel {j['jid']}")
                acts.append(f"ESCALATE cancelled {j['jid']} on {j['node']}: drain "
                            f"outstanding {int(time.time()-t0)}s, cluster still tight")
    return acts


def main():
    log(f"governor start: cap {MAX_CORES} cores, our RAM {MAX_RSS_G}G, "
        f"per-node all-user fill stop/drain/kill {NODE_MEM_STOP}/{NODE_MEM_DRAIN}/{NODE_MEM_KILL}, "
        f"keep >={MIN_IDLE} cluster cores idle, period {PERIOD}s")
    while True:
        try:
            R = root()
            state = cluster_state()
            jobs = my_jobs()
            reap_claims(R, {j["jid"] for j in jobs})
            # DISABLED 2026-08-31: this raced with slot startup.  A ramdir is
            # created before the python processes that reference it exist, so
            # "no live process uses it" is true for a second or two and the GC
            # deleted freshly-made ramdirs -- two tasks died in 18s with
            # "Need to install ramulator".  Any future version must require the
            # directory to be older than a slot's startup window.
            scratch_free = {}
            for n, gb in scratch_free.items():
                if gb < 150:
                    log(f"SCRATCH LOW {n}: {gb}G free -- a trace-heavy task can "
                        f"need 128G; not placing there")
            bad, infra_bad = check_new_rungs(R)
            if infra_bad:
                log("damaged rungs (node ran out of disk/memory -- backfill "
                    "later, NOT a rung defect): " + "; ".join(
                        m.split(" |")[0] for m in infra_bad[:6]))
            hold_file = f"{R}/HOLD"
            if bad and not os.path.exists(hold_file):
                with open(hold_file, "w") as f:
                    f.write("\n".join(bad) + "\n")
                log("ALERT a finished task is missing A3b/A4b output -- HOLD set, "
                    "phase 2 will not grow:\n  " + "\n  ".join(bad[:4]))
            hold = os.path.exists(hold_file)
            p = plan(state, jobs, R)
            acts = reconcile(R, p, hold)
            util = (100.0 * p["used_cores"] / p["my_cores"]) if p["my_cores"] else 0.0
            log(f"alloc={p['my_cores']}c used={p['used_cores']:.1f}c ({util:.0f}%) "
                f"rss={p['used_rss']:.0f}G(commit {p['committed']:.0f}G)/{MAX_RSS_G}G[{p['mem_state']}] "
                f"nodefill={{{' '.join(f'{n}:{100*f:.0f}%' for n, f in sorted(p['node_frac'].items()))}}} "
                f"idle={p['idle']} "
                f"budget={p['budget']}c slot={p['slot_cores']}c "
                f"(slotcores={p['cores_slots']}c) "
                f"memcap({p['mem_budget']:.0f}G avgB{p['avg_big']:.0f}/avgS{p['avg_small']:.0f}) "
                f"big={len([j for j in p['slots_run'] if j['cls']=='big'])}R"
                f"/{p['n_big']}T small={len([j for j in p['slots_run'] if j['cls']=='small'])}R"
                f"/{p['n_small']}T "
                f"work-left big {p['rem']['big']:.1f}u small {p['rem']['small']:.1f}u "
                f"tasks big {p['q']['big'][1]}/{p['q']['big'][0]} "
                f"small {p['q']['small'][1]}/{p['q']['small'][0]}"
                f"{' HOLD' if hold else ''}")
            for a in acts:
                log("  " + a)
            with open(f"{R}/GOVERNOR_STATUS", "w") as f:
                json.dump(dict(ts=time.strftime("%F %T"), hold=hold,
                               alloc_cores=p["my_cores"], used_cores=p["used_cores"],
                               util_pct=round(util, 1), used_rss_g=p["used_rss"],
                               node_frac={k: round(v, 3) for k, v in p["node_frac"].items()},
                               idle=p["idle"], slot_cores=p["slot_cores"],
                               n_big=p["n_big"], n_small=p["n_small"],
                               queues=p["q"], actions=acts), f, indent=1)
        except Exception as e:
            import traceback
            log(f"ERROR governor cycle: {type(e).__name__}: {e}\n{traceback.format_exc()[:800]}")
        time.sleep(PERIOD)


if __name__ == "__main__":
    main()

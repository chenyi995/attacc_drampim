#!/usr/bin/env python3
"""Data-driven ETA for the 84-task / 756-run sweep.

No guessing at constants: every number here is either read out of the repo
(workload token counts, model geometry) or measured from this run's own logs
(per-rung warm times, per-task slot times).  With no measurements yet it says
so instead of inventing a figure.

work unit for a task = workload tokens x (layers x KV heads)
  -- the DAG engine expands every request x layer, and the placement scan
     prices a run per KV head, so those two are what the cost tracks.
"""
import json, os, re, subprocess, sys, time

REPO = "/home/cw636/chenyi/attacc_drampim"
ORCH = f"{REPO}/output/_orch2"
GEOM = {  # layers, heads, gqa_size  (src/config.py model_table)
    "GPT-175B": (96, 96, 1), "LLAMA-65B": (80, 64, 1), "LLAMA-33B": (60, 52, 1),
    "GPT-13B": (40, 40, 1), "LLAMA-7B": (32, 32, 1), "LLAMA3-8B": (32, 32, 4),
}


def model_weight(m):
    layers, heads, gqa = GEOM[m]
    return layers * (heads // gqa)


def wl_tokens(wl):
    d = json.load(open(f"{REPO}/workload/sweep/{wl}"))
    return sum(sum(s["len"] for s in a["segs"]) for a in d["agents"])


def tasks(R):
    out = []
    for ln in open(f"{R}/tasks.txt"):
        f = ln.split()
        if len(f) < 4:
            continue
        model, cfg, wl, k = f[0], f[1], f[2], f[3]
        out.append(dict(model=model, cfg=cfg, wl=wl, k=k,
                        tid=f"{model}__{cfg}_k{k}",
                        work=wl_tokens(wl) * model_weight(model) / 1e9))
    return out


def measured(R):
    """(tid -> seconds) for tasks this run has actually finished."""
    done = {}
    out = subprocess.run(f'grep -ah "^\\[.*\\] END " {R}/slurm/slot_*.out 2>/dev/null',
                         shell=True, capture_output=True, text=True).stdout
    for ln in out.splitlines():
        m = re.search(r"END\s+(\S+)\s+(\S+)\s+k=(\S+)\s+rc=(\S+)\s+jsons=(\d+)/9\s+(\d+)s", ln)
        if m:
            done[f"{m.group(1)}__{m.group(2)}_k{m.group(3)}"] = dict(
                sec=int(m.group(6)), rc=m.group(4), jsons=int(m.group(5)))
    return done


def warm_rungs(R):
    """(model, workload, rung) -> seconds, from phase 1.  A warm rung is one
    rung run alone; a phase-2 task runs all 9 at once, so 9 warm rungs of the
    same workload bound a task from above (they share the node instead)."""
    out = subprocess.run(f'grep -ah "rc=" {R}/slurm/warm_*.out 2>/dev/null',
                         shell=True, capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        m = re.search(r"warm\s+(\S+)\s+(\S+)\s+(\S+)\s+rc=(\d+)\s+(\d+)s", ln)
        if m:
            rows.append(dict(model=m.group(1), wl=m.group(2), rung=m.group(3),
                             rc=int(m.group(4)), sec=int(m.group(5))))
    return rows


def main():
    R = open(f"{ORCH}/CURRENT_ROOT").read().strip()
    T = tasks(R)
    done = measured(R)
    warm = warm_rungs(R)
    total_work = sum(t["work"] for t in T)

    fin = [t for t in T if t["tid"] in done]
    rem = [t for t in T if t["tid"] not in done]
    done_work = sum(t["work"] for t in fin)

    print(f"tasks: {len(fin)}/{len(T)} finished   work: {done_work:.1f}/{total_work:.1f} units")
    print()

    # --- calibration -------------------------------------------------------
    rate = None                       # work units per slot-second
    src = None
    if len(fin) >= 2:
        secs = sum(done[t["tid"]]["sec"] for t in fin)
        rate = done_work / secs
        src = f"{len(fin)} finished tasks in this run"
    elif warm:
        # A phase-2 task is 9 rungs in parallel on one 28-core slot.  A warm
        # rung is 1 rung alone.  Treat a task as ~ the sum of its 9 rungs'
        # serial cost divided by the parallel speed-up the slot actually gets;
        # with 3 procs busy per rung and 28 cores, they do overlap, so the
        # honest bound is [slowest rung, sum of rungs].
        ok = [w for w in warm if w["rc"] == 0]
        if ok:
            per = {}
            for w in ok:
                per.setdefault((w["model"], w["wl"]), []).append(w["sec"])
            print("measured warm rungs (single rung, alone on the node):")
            for (m, wl), v in sorted(per.items()):
                u = wl_tokens(wl) * model_weight(m) / 1e9
                print(f"  {m:11s} {wl:38s} n={len(v):2d} "
                      f"median={sorted(v)[len(v)//2]:6d}s  work={u:7.1f}u "
                      f"-> {sorted(v)[len(v)//2]/max(u,1e-9):8.2f} s/u/rung")
            allrate = [w["sec"] / (wl_tokens(w["wl"]) * model_weight(w["model"]) / 1e9)
                       for w in ok]
            allrate.sort()
            s_per_u_rung = allrate[len(allrate) // 2]
            print(f"\n  median {s_per_u_rung:.2f} s per work-unit per rung "
                  f"({len(ok)} rungs measured)")
            src = f"{len(ok)} phase-1 rungs (no finished task yet)"
            rate = 1.0 / (s_per_u_rung * 9)      # optimistic: 9 rungs fully parallel
            rate_pess = 1.0 / s_per_u_rung       # pessimistic: fully serial
    if rate is None:
        print("NO MEASUREMENTS YET -- no honest ETA can be given.")
        print("The first phase-1 rung completions will produce one; check back.")
        return

    slots = int(subprocess.run(
        'squeue -u "$USER" -h -t R -o "%j" | grep -c "^slot_" || true',
        shell=True, capture_output=True, text=True).stdout.strip() or 0)
    slots_pd = int(subprocess.run(
        'squeue -u "$USER" -h -t PD -o "%j" | grep -c "^slot_" || true',
        shell=True, capture_output=True, text=True).stdout.strip() or 0)
    eff = max(slots, 1)
    rem_work = sum(t["work"] for t in rem)
    print(f"\ncalibration source: {src}")
    print(f"slots: {slots} running, {slots_pd} pending")
    print(f"remaining work: {rem_work:.1f} units over {len(rem)} tasks")
    for label, r in (("9 rungs overlap well (optimistic)", rate),
                     ("9 rungs barely overlap (pessimistic)", locals().get("rate_pess", rate))):
        h = rem_work / r / eff / 3600
        print(f"  {label:38s}: {h:8.1f} slot-hours/{eff} slots = {h:6.1f} h")


if __name__ == "__main__":
    main()

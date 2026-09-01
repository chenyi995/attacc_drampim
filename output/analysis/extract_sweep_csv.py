#!/usr/bin/env python3
"""Extract the sweep's measured results from the raw JSON into three CSVs.

Source of truth is the per-rung ``dag_<rung>.json`` that ``main.py`` writes --
not ladder.log, not the claim directories, not any intermediate.  Everything
here is copied straight out of those files; the only computed columns are
``duration_s`` in the tier table (end - start) and the counts in the
completeness table, and both are flagged where they are produced.

Run it through ``extract_sweep.sh``, which fixes the sweep root, reports row
counts and checksums, and re-reads a sample of values to check the CSV.

Writes into output/analysis/:

  sweep_rungs.csv         one row per (model, config, k, rung) -- the results
  sweep_tiers.csv         one row per (model, config, k, rung, tier)
  sweep_completeness.csv  one row per (model, config, k) -- what is present

A task missing rungs still contributes the rungs it has; nothing is silently
dropped, and sweep_completeness.csv records exactly what was absent at
extraction time so a partial task can never be mistaken for a whole one.

Why this does not simply call json.load
---------------------------------------
The reports total ~99 GB across 667 files, 152 MB on average and up to 944 MB,
and the wanted fields are about a thousandth of that: ``batches``, ``events``,
``tlb`` and ``ramulator_signature_cache`` are the bulk and none of them is
needed here.  main.py writes the reports pretty-printed with sorted keys, so
every top-level key sits on its own line at exactly two spaces of indent.  That
makes the file seekable: find the key lines, and a value spans from its colon to
the next key line.  Only the wanted slices are ever decoded.

That still reads every byte -- the fields are scattered through the file and NFS
delivers ~78 MB/s -- so the floor is I/O, not CPU.  Hence --jobs, and a cache
keyed by (path, mtime, size) that makes a re-run after the backfill nearly free.
"""
import argparse
import csv
import glob
import json
import mmap
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(HERE, ".extract_cache.pkl")

RUNGS = ["A1", "A2", "A3", "A3a", "A3b", "A4", "A4b", "A5", "A6"]

# Top-level keys to decode.  Everything else in the file is skipped unread.
WANT = {
    "makespan_s", "energy_nj", "link_bytes", "event_count",
    "policy", "kv_mapping", "decode_attn", "pim_prefill_mode",
    "energy_breakdown_nj", "gpu_time_s_unoverlapped", "pim_time_s_unoverlapped",
    "pim_pool_time_s_unoverlapped", "die_time_s_unoverlapped",
    "prefill_attention_rows", "prefill_attention_sides",
    "history_rows", "cacheblend_batch_size", "gemv_buffer_bytes",
    "pim_sweep_query_capacity", "pim_pe_freq_ghz", "pim_batch_command",
    "engine", "overlap_validation", "summary", "workload",
}
_KEYLINE = re.compile(rb'(?m)^(  "([A-Za-z_0-9]+)": )')
_DECODER = json.JSONDecoder()

RUNG_COLUMNS = [
    "model", "config", "k", "workload", "agents", "rung",
    "makespan_s", "energy_nj", "link_bytes", "event_count",
    "policy", "kv_mapping", "decode_attn", "pim_prefill_mode",
    "energy_gpu_nj", "energy_pim_nj", "energy_link_nj",
    "energy_die_nj", "energy_tlb_nj",
    "gpu_time_s_unoverlapped", "pim_time_s_unoverlapped",
    "pim_pool_time_s_unoverlapped", "die_time_s_unoverlapped",
    "prefill_rows_gpu", "prefill_rows_pim",
    "prefill_requests_gpu", "prefill_requests_pim",
    "requests", "tiers", "history_rows", "cacheblend_batch_size",
    "gemv_buffer_bytes", "pim_sweep_query_capacity", "pim_pe_freq_ghz",
    "pim_batch_command", "engine",
    # the simulator's own topology label -- NOT ground truth.  The config name
    # and the workload filename are; the sweep's workload JSONs carry no "kind"
    # at all, and a two-tier alltoall gets reported here as "supervisor".
    "workload_kind_reported",
    "overlap_passed", "overlap_events_checked",
]
TIER_COLUMNS = ["model", "config", "k", "rung", "tier",
                "start_s", "end_s", "duration_s", "requests"]
COMPLETE_COLUMNS = ["model", "config", "k", "workload", "agents",
                    "rungs_present", "rungs_missing", "n_present",
                    "claim_status"]


def read_fields(path):
    """Decode only the wanted top-level keys of one report."""
    out = {}
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            keys = [(m.start(1), m.group(2).decode(), m.end(1))
                    for m in _KEYLINE.finditer(mm)]
            for i, (_, name, after_colon) in enumerate(keys):
                if name not in WANT:
                    continue
                end = keys[i + 1][0] if i + 1 < len(keys) else mm.size()
                # raw_decode stops at the end of the value, so a trailing comma
                # or the file's closing brace costs nothing.
                out[name] = _DECODER.raw_decode(mm[after_colon:end].decode())[0]
        finally:
            mm.close()
    return out


def _job(args):
    path, stamp = args
    try:
        return path, stamp, read_fields(path), None
    except Exception as exc:                       # noqa: BLE001 - reported
        return path, stamp, None, f"{type(exc).__name__}: {exc}"


def num(x):
    return x if isinstance(x, (int, float)) else ""


def task_dirs(root):
    out = []
    for d in sorted(glob.glob(f"{root}/*/*_k*")):
        if not os.path.isdir(d):
            continue
        model = os.path.basename(os.path.dirname(d))
        cfg, _, k = os.path.basename(d).rpartition("_k")
        if cfg and k.isdigit():
            out.append((model, cfg, k, d))
    return out


def workload_of(root):
    out = {}
    for name in ("tasks_big.txt", "tasks_small.txt", "tasks.txt"):
        path = f"{root}/{name}"
        if not os.path.exists(path):
            continue
        for ln in open(path):
            g = ln.split()
            if len(g) >= 4:
                out.setdefault((g[0], g[1], g[3]), g[2])
    return out


def agents_of(wl, cache={}):
    if wl not in cache:
        try:
            cache[wl] = len(json.load(
                open(f"{REPO}/workload/sweep/{wl}")).get("agents", []))
        except (OSError, ValueError):
            cache[wl] = ""
    return cache[wl]


def claim_status(root, model, cfg, k):
    c = f"{root}/claims/{model}__{cfg}_k{k}"
    if not os.path.isdir(c):
        return "unclaimed"
    for mark in ("excluded", "parked", "damaged"):
        if os.path.exists(f"{c}/{mark}"):
            return mark
    return "done" if os.path.exists(f"{c}/done") else "in-flight"


def row_of(model, cfg, k, wl, agents, rung, j):
    by_class = (j.get("energy_breakdown_nj") or {}).get("by_class") or {}
    rows = j.get("prefill_attention_rows") or {}
    sides = j.get("prefill_attention_sides") or {}
    summary = j.get("summary") or {}
    ov = j.get("overlap_validation") or {}
    return {
        "model": model, "config": cfg, "k": k, "workload": wl,
        "agents": agents, "rung": rung,
        "makespan_s": num(j.get("makespan_s")),
        "energy_nj": num(j.get("energy_nj")),
        "link_bytes": num(j.get("link_bytes")),
        "event_count": num(j.get("event_count")),
        "policy": j.get("policy", ""),
        "kv_mapping": j.get("kv_mapping", ""),
        "decode_attn": j.get("decode_attn", ""),
        "pim_prefill_mode": j.get("pim_prefill_mode", ""),
        "energy_gpu_nj": num(by_class.get("GPU")),
        "energy_pim_nj": num(by_class.get("PIM")),
        "energy_link_nj": num(by_class.get("LINK")),
        "energy_die_nj": num(by_class.get("DIE")),
        "energy_tlb_nj": num(by_class.get("TLB")),
        "gpu_time_s_unoverlapped": num(j.get("gpu_time_s_unoverlapped")),
        "pim_time_s_unoverlapped": num(j.get("pim_time_s_unoverlapped")),
        "pim_pool_time_s_unoverlapped": num(j.get("pim_pool_time_s_unoverlapped")),
        "die_time_s_unoverlapped": num(j.get("die_time_s_unoverlapped")),
        "prefill_rows_gpu": num(rows.get("gpu")),
        "prefill_rows_pim": num(rows.get("pim")),
        "prefill_requests_gpu": sum(1 for v in sides.values() if v == "gpu"),
        "prefill_requests_pim": sum(1 for v in sides.values() if v == "pim"),
        "requests": len(summary.get("requests") or {}),
        "tiers": len(summary.get("tiers") or {}),
        "history_rows": num(j.get("history_rows")),
        "cacheblend_batch_size": num(j.get("cacheblend_batch_size")),
        "gemv_buffer_bytes": num(j.get("gemv_buffer_bytes")),
        "pim_sweep_query_capacity": num(j.get("pim_sweep_query_capacity")),
        "pim_pe_freq_ghz": num(j.get("pim_pe_freq_ghz")),
        "pim_batch_command": j.get("pim_batch_command", ""),
        "engine": j.get("engine", ""),
        "workload_kind_reported": (j.get("workload") or {}).get("kind", ""),
        "overlap_passed": ov.get("passed", ""),
        "overlap_events_checked": num(ov.get("events_checked")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--jobs", type=int, default=6,
                    help="parallel readers; the limit is NFS, not CPU")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    root = a.root or open(f"{REPO}/output/_orch2/CURRENT_ROOT").read().strip()
    if not os.path.isdir(root):
        print(f"sweep root not found: {root}", file=sys.stderr)
        return 1

    cache = {}
    if not a.no_cache and os.path.exists(CACHE):
        try:
            cache = pickle.load(open(CACHE, "rb"))
        except Exception:                          # noqa: BLE001
            cache = {}

    wl_of = workload_of(root)
    tasks = task_dirs(root)
    todo, hits, plan = [], 0, []
    for model, cfg, k, d in tasks:
        for rung in RUNGS:
            p = f"{d}/dag_{rung}.json"
            try:
                st = os.stat(p)
            except OSError:
                continue
            stamp = (int(st.st_mtime), st.st_size)
            plan.append((model, cfg, k, rung, p, stamp))
            if cache.get(p, (None,))[0] == stamp:
                hits += 1
            else:
                todo.append((p, stamp))

    total_bytes = sum(s[1] for *_, s in plan)
    print(f"sweep root: {root}")
    print(f"reports: {len(plan)}   cached: {hits}   to read: {len(todo)}"
          f"   ({total_bytes / 2**30:.1f} GB on disk)")
    if todo:
        est = sum(s[1] for _, s in todo) / 2**20 / 78 / max(1, a.jobs)
        print(f"reading with {a.jobs} workers -- expect roughly "
              f"{est / 60:.0f} min\n")
    t0 = time.time()
    done = 0
    if todo:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for path, stamp, fields, err in ex.map(_job, todo, chunksize=1):
                done += 1
                if err:
                    print(f"  !! {path}: {err}", file=sys.stderr)
                else:
                    cache[path] = (stamp, fields)
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)} files  {el / 60:.1f} min"
                          f"  (eta {el / done * (len(todo) - done) / 60:.1f} min)",
                          flush=True)
    if not a.no_cache:
        try:
            pickle.dump(cache, open(CACHE, "wb"), protocol=4)
        except Exception as exc:                   # noqa: BLE001
            print(f"  (cache not written: {exc})", file=sys.stderr)

    rung_rows, tier_rows, comp_rows = [], [], []
    present_by_task = {}
    for model, cfg, k, rung, p, _ in plan:
        entry = cache.get(p)
        if not entry or entry[1] is None:
            continue
        j = entry[1]
        wl = wl_of.get((model, cfg, k), "")
        agents = agents_of(wl) if wl else ""
        present_by_task.setdefault((model, cfg, k), []).append(rung)
        rung_rows.append(row_of(model, cfg, k, wl, agents, rung, j))
        for tid, t in sorted((j.get("summary") or {}).get("tiers", {}).items(),
                             key=lambda kv: int(kv[0])):
            s, e = t.get("start_s"), t.get("end_s")
            tier_rows.append({
                "model": model, "config": cfg, "k": k, "rung": rung,
                "tier": tid, "start_s": num(s), "end_s": num(e),
                # computed, not read: the only derived column in these files
                "duration_s": (e - s) if isinstance(s, (int, float))
                and isinstance(e, (int, float)) else "",
                "requests": num(t.get("requests")),
            })
    for model, cfg, k, _ in tasks:
        wl = wl_of.get((model, cfg, k), "")
        present = present_by_task.get((model, cfg, k), [])
        comp_rows.append({
            "model": model, "config": cfg, "k": k, "workload": wl,
            "agents": agents_of(wl) if wl else "",
            "rungs_present": " ".join(r for r in RUNGS if r in present),
            "rungs_missing": " ".join(r for r in RUNGS if r not in present),
            "n_present": len(present),
            "claim_status": claim_status(root, model, cfg, k),
        })

    order = {r: i for i, r in enumerate(RUNGS)}

    def write(name, cols, rows, key):
        path = os.path.join(a.outdir, name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(sorted(rows, key=key))
        print(f"  {name:26s} {len(rows):6d} rows")

    print()
    write("sweep_rungs.csv", RUNG_COLUMNS, rung_rows,
          lambda r: (r["model"], r["config"], r["k"], order[r["rung"]]))
    write("sweep_tiers.csv", TIER_COLUMNS, tier_rows,
          lambda r: (r["model"], r["config"], r["k"], order[r["rung"]],
                     int(r["tier"])))
    write("sweep_completeness.csv", COMPLETE_COLUMNS, comp_rows,
          lambda r: (r["model"], r["config"], r["k"]))
    full = sum(1 for r in comp_rows if r["n_present"] == 9)
    print(f"\n  tasks with all 9 rungs: {full} / {len(comp_rows)}")
    print(f"  elapsed: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

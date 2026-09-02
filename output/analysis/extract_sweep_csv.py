#!/usr/bin/env python3
"""Extract the sweep's measured results from the raw JSON into three CSVs.

Source of truth is the per-rung ``dag_<rung>.json`` that ``main.py`` writes --
not ladder.log, not the claim directories, not any intermediate.

Method: one regex per field, matched against the report.  That is all.

The reports are big -- ~99 GB across 667 files, 152 MB on average and 944 MB at
the worst -- because ``batches``, ``events``, ``tlb`` and
``ramulator_signature_cache`` dominate them, and none of those is wanted here.
Rather than being clever about skipping them, this reads each file once through
mmap and runs a handful of plain regexes over it.  Every byte is read; the work
is spread over processes and it takes as long as it takes.  A cache keyed by
(path, mtime, size) means the re-run after a backfill only re-reads the rungs
that actually changed.

The regexes can be this simple because main.py writes the reports
pretty-printed with SORTED keys, so a nested object's fields always appear in a
fixed order -- e.g. a tier is always ``end_s``, ``requests``, ``start_s``.
``check_against_json_load`` re-parses whole reports with ``json.load`` and
compares field by field; run it whenever the report format might have changed.

Writes into output/analysis/:

  sweep_rungs.csv         one row per (model, config, k, rung) -- the results
  sweep_tiers.csv         one row per (model, config, k, rung, tier)
  sweep_completeness.csv  one row per (model, config, k) -- what is present

A task missing rungs still contributes the rungs it has; nothing is silently
dropped, and sweep_completeness.csv records exactly what was absent at
extraction time so a partial task can never be mistaken for a whole one.
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
# Bump whenever read_fields starts producing a field the CSVs use.  A cache
# entry is (mtime, size) -> fields, so an entry written before a new field
# existed still LOOKS valid and would hand back a silently empty column.  The
# schema tag makes that impossible: a mismatch discards the cache and every
# report is read again.
CACHE_SCHEMA = 2

RUNGS = ["A1", "A2", "A3", "A3a", "A3b", "A4", "A4b", "A5", "A6"]

NUM = rb'(-?[0-9][0-9.eE+-]*)'


# Anchored to a TOP-LEVEL key: line start, exactly two spaces of indent.  The
# anchor is load-bearing, not decoration -- the report's "ablation" object
# repeats several of these names one level down (gemv_buffer_bytes,
# pim_pe_freq_ghz, pim_batch_command, decode_attn, kv_mapping ...), and an
# unanchored search happily returns the nested copy.  A2 has no top-level
# gemv_buffer_bytes at all, so the unanchored version invented one.
def _num(name):
    return re.compile(rb'(?m)^  "' + name.encode() + rb'": ' + NUM)


def _str(name):
    return re.compile(rb'(?m)^  "' + name.encode() + rb'": "([^"]*)"')


# One regex per scalar field, applied to the whole report.  Each of these keys
# occurs at most once at top level and its value is a plain scalar.
RE_NUM = {k: _num(k) for k in (
    "makespan_s", "energy_nj", "link_bytes", "event_count",
    "gpu_time_s_unoverlapped", "pim_time_s_unoverlapped",
    "pim_pool_time_s_unoverlapped", "die_time_s_unoverlapped",
    "history_rows", "cacheblend_batch_size", "gemv_buffer_bytes",
    "pim_sweep_query_capacity", "pim_pe_freq_ghz",
)}
RE_STR = {k: _str(k) for k in (
    "policy", "kv_mapping", "decode_attn", "pim_prefill_mode",
    "pim_batch_command", "engine",
)}
# Nested, but each is a flat object whose keys are sorted, so a single
# non-greedy match of its braces is unambiguous.
RE_BY_CLASS = re.compile(rb'(?m)^    "by_class": \{(.*?)\n    \}', re.S)
RE_CLASS_ITEM = re.compile(rb'"([A-Z]+)": ' + NUM)
RE_PREFILL_ROWS = re.compile(
    rb'(?m)^  "prefill_attention_rows": \{\s*"gpu": ' + NUM +
    rb',\s*"pim": ' + NUM)
RE_PREFILL_SIDES = re.compile(
    rb'(?m)^  "prefill_attention_sides": \{(.*?)\n  \}', re.S)
RE_SIDE_ITEM = re.compile(rb'"(gpu|pim)"')
# A tier's keys are sorted: end_s, requests, start_s.
RE_TIER = re.compile(
    rb'"([0-9]+)": \{\s*"end_s": ' + NUM +
    rb',\s*"requests": ' + NUM + rb',\s*"start_s": ' + NUM + rb'\s*\}')
RE_TIERS_BLOCK = re.compile(rb'(?m)^    "tiers": \{(.*?)\n    \}', re.S)
# summary.requests: one entry per request, keys sorted, so the order is fixed.
# Distinct from the tier pattern above -- a tier has end_s/requests/start_s, a
# request has end_s/first_token_s/prefill_end_s/tier -- so the two cannot be
# confused even before the enclosing block restricts where each is matched.
RE_REQUESTS_BLOCK = re.compile(rb'(?m)^    "requests": \{(.*?)\n    \}', re.S)
RE_REQ = re.compile(
    rb'"([A-Za-z0-9_]+)": \{\s*"end_s": ' + NUM +
    rb',\s*"first_token_s": ' + NUM +
    rb',\s*"prefill_end_s": ' + NUM +
    rb',\s*"tier": ' + NUM + rb'\s*\}')
# by_event, whose values are flat floats.  The decode phase is exactly the
# events carrying the `decode_` prefix: the report names every decode op as the
# prefixed twin of its prefill counterpart (decode_qkv/qkv,
# decode_ctx_pim_to_gpu/ctx_pim_to_gpu, decode_dram_store_master/
# dram_store_master, ...), so the split is a naming convention rather than a
# guess.  Everything without the prefix is prefill-side.
RE_BY_EVENT = re.compile(rb'(?m)^    "by_event": \{(.*?)\n    \}', re.S)
RE_EVENT_ITEM = re.compile(rb'"([A-Za-z0-9_]+)": ' + NUM)
RE_OVERLAP = re.compile(
    rb'(?m)^  "overlap_validation": \{(.*?)\n  \}', re.S)
RE_PASSED = re.compile(rb'"passed": (true|false)')
RE_EVENTS_CHECKED = re.compile(rb'"events_checked": ' + NUM)
RE_WORKLOAD_KIND = re.compile(
    rb'(?m)^  "workload": \{\s*"kind": "([^"]*)"')


def _f(m):
    if not m:
        return ""
    try:
        return float(m.group(1))
    except ValueError:
        return ""


def read_fields(path):
    """Pull every wanted field out of one report with plain regexes."""
    out = {}
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for k, rx in RE_NUM.items():
                out[k] = _f(rx.search(mm))
            for k, rx in RE_STR.items():
                m = rx.search(mm)
                out[k] = m.group(1).decode() if m else ""

            m = RE_BY_CLASS.search(mm)
            energy = {}
            if m:
                for item in RE_CLASS_ITEM.finditer(m.group(1)):
                    try:
                        energy[item.group(1).decode()] = float(item.group(2))
                    except ValueError:
                        pass
            out["energy_by_class"] = energy

            m = RE_PREFILL_ROWS.search(mm)
            out["prefill_rows_gpu"] = _f(m) if m else ""
            out["prefill_rows_pim"] = (float(m.group(2)) if m else "")

            m = RE_PREFILL_SIDES.search(mm)
            sides = RE_SIDE_ITEM.findall(m.group(1)) if m else []
            out["prefill_requests_gpu"] = sides.count(b"gpu")
            out["prefill_requests_pim"] = sides.count(b"pim")

            m = RE_TIERS_BLOCK.search(mm)
            tiers = []
            if m:
                for t in RE_TIER.finditer(m.group(1)):
                    tiers.append({
                        "tier": t.group(1).decode(),
                        "end_s": float(t.group(2)),
                        "requests": int(float(t.group(3))),
                        "start_s": float(t.group(4)),
                    })
            out["tiers"] = tiers
            out["requests"] = sum(t["requests"] for t in tiers)

            # Per-tier prefill/decode split.  A tier's prefill ends when its
            # LAST request's prefill ends, so the tier's decode span is whatever
            # remains of it.  Defined this way the two are exactly additive:
            # prefill_s + decode_s == the tier's makespan, with no third bucket
            # to explain away.
            m = RE_REQUESTS_BLOCK.search(mm)
            by_tier = {}
            if m:
                for q in RE_REQ.finditer(m.group(1)):
                    t = str(int(float(q.group(5))))
                    d = by_tier.setdefault(t, {"pe": [], "ft": [], "end": []})
                    d["end"].append(float(q.group(2)))
                    d["ft"].append(float(q.group(3)))
                    d["pe"].append(float(q.group(4)))
            for t in tiers:
                d = by_tier.get(t["tier"])
                if d and d["pe"]:
                    t["prefill_end_s"] = max(d["pe"])
                    t["first_token_s"] = max(d["ft"])
                    t["prefill_s"] = max(0.0, max(d["pe"]) - t["start_s"])
                    t["decode_s"] = max(0.0, t["end_s"] - max(d["pe"]))
                else:
                    for c in ("prefill_end_s", "first_token_s",
                              "prefill_s", "decode_s"):
                        t[c] = ""

            m = RE_BY_EVENT.search(mm)
            e_dec = e_pre = 0.0
            if m:
                for it in RE_EVENT_ITEM.finditer(m.group(1)):
                    try:
                        v = float(it.group(2))
                    except ValueError:
                        continue
                    if it.group(1).startswith(b"decode_"):
                        e_dec += v
                    else:
                        e_pre += v
            out["energy_decode_nj"] = e_dec
            out["energy_prefill_nj"] = e_pre

            m = RE_OVERLAP.search(mm)
            if m:
                p = RE_PASSED.search(m.group(1))
                e = RE_EVENTS_CHECKED.search(m.group(1))
                out["overlap_passed"] = (p.group(1) == b"true") if p else ""
                out["overlap_events_checked"] = _f(e) if e else ""
            else:
                out["overlap_passed"] = ""
                out["overlap_events_checked"] = ""

            m = RE_WORKLOAD_KIND.search(mm)
            out["workload_kind_reported"] = m.group(1).decode() if m else ""
        finally:
            mm.close()
    return out


def check_against_json_load(paths):
    """Re-parse whole reports and compare every field the regexes produced."""
    bad = checked = 0
    for p in paths:
        got = read_fields(p)
        j = json.load(open(p))
        exp = {}
        for k in RE_NUM:
            v = j.get(k)
            exp[k] = float(v) if isinstance(v, (int, float)) else ""
        for k in RE_STR:
            exp[k] = j.get(k, "") or ""
        exp["energy_by_class"] = {
            k: float(v) for k, v in
            ((j.get("energy_breakdown_nj") or {}).get("by_class") or {}).items()}
        rows = j.get("prefill_attention_rows") or {}
        exp["prefill_rows_gpu"] = float(rows["gpu"]) if "gpu" in rows else ""
        exp["prefill_rows_pim"] = float(rows["pim"]) if "pim" in rows else ""
        sides = j.get("prefill_attention_sides") or {}
        exp["prefill_requests_gpu"] = sum(1 for v in sides.values() if v == "gpu")
        exp["prefill_requests_pim"] = sum(1 for v in sides.values() if v == "pim")
        tiers = ((j.get("summary") or {}).get("tiers") or {})
        reqs = (j.get("summary") or {}).get("requests") or {}
        exp["tiers"] = []
        for t, v in sorted(tiers.items(), key=lambda kv: int(kv[0])):
            e = {"tier": t, "end_s": float(v["end_s"]),
                 "requests": int(v["requests"]), "start_s": float(v["start_s"])}
            pe = [float(r["prefill_end_s"]) for r in reqs.values()
                  if str(int(r["tier"])) == t]
            ft = [float(r["first_token_s"]) for r in reqs.values()
                  if str(int(r["tier"])) == t]
            if pe:
                e["prefill_end_s"] = max(pe)
                e["first_token_s"] = max(ft)
                e["prefill_s"] = max(0.0, max(pe) - e["start_s"])
                e["decode_s"] = max(0.0, e["end_s"] - max(pe))
            else:
                for c in ("prefill_end_s", "first_token_s",
                          "prefill_s", "decode_s"):
                    e[c] = ""
            exp["tiers"].append(e)
        bev = (j.get("energy_breakdown_nj") or {}).get("by_event") or {}
        exp["energy_decode_nj"] = sum(
            float(v) for k2, v in bev.items() if k2.startswith("decode_"))
        exp["energy_prefill_nj"] = sum(
            float(v) for k2, v in bev.items() if not k2.startswith("decode_"))
        exp["requests"] = len((j.get("summary") or {}).get("requests") or {})
        ov = j.get("overlap_validation") or {}
        exp["overlap_passed"] = ov.get("passed", "")
        exp["overlap_events_checked"] = (float(ov["events_checked"])
                                         if "events_checked" in ov else "")
        exp["workload_kind_reported"] = (j.get("workload") or {}).get("kind", "")
        for k, want in exp.items():
            checked += 1
            if got.get(k) != want:
                bad += 1
                print(f"  MISMATCH {os.path.basename(p)} {k}: "
                      f"regex={got.get(k)!r} json={want!r}")
    print(f"  {checked} fields from {len(paths)} reports, {bad} mismatched")
    return bad


def _job(args):
    path, stamp = args
    try:
        return path, stamp, read_fields(path), None
    except Exception as exc:                       # noqa: BLE001 - reported
        return path, stamp, None, f"{type(exc).__name__}: {exc}"


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
    # phase split: energy from by_event's decode_ prefix, time from the tiers.
    # prefill_time_s + decode_time_s == makespan_s by construction.
    "energy_prefill_nj", "energy_decode_nj",
    "prefill_time_s", "decode_time_s",
    "power_prefill_w", "power_decode_w", "power_overall_w",
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
                "start_s", "end_s", "duration_s", "requests",
                "prefill_end_s", "first_token_s", "prefill_s", "decode_s"]
COMPLETE_COLUMNS = ["model", "config", "k", "workload", "agents",
                    "rungs_present", "rungs_missing", "n_present",
                    "claim_status"]


def task_dirs(root, valid):
    """Output directories of real tasks, keyed by (model, config, k).

    ``valid`` is the (model, config, k) set the run's own queues declare, and it
    is the filter, not a nicety: the sweep root also holds bookkeeping
    directories whose entries are named <model>__<config>_k<n>, so a bare
    ``*/*_k*`` glob picks up claims/ and claims_bf/ as if "claims" were a model.
    That put 91 phantom rows into sweep_completeness.csv -- every claim, listed
    as an unclaimed task with no rungs.
    """
    out = []
    for d in sorted(glob.glob(f"{root}/*/*_k*")):
        if not os.path.isdir(d):
            continue
        model = os.path.basename(os.path.dirname(d))
        cfg, _, k = os.path.basename(d).rpartition("_k")
        if cfg and k.isdigit() and (model, cfg, k) in valid:
            out.append((model, cfg, k, d))
    return out


def workload_of(root):
    out = {}
    for name in ("tasks_big.txt", "tasks_small.txt", "tasks.txt"):
        p = f"{root}/{name}"
        if not os.path.exists(p):
            continue
        for ln in open(p):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--self-check", type=int, default=0, metavar="N",
                    help="re-parse N reports with json.load and compare, then exit")
    a = ap.parse_args()
    root = a.root or open(f"{REPO}/output/_orch2/CURRENT_ROOT").read().strip()
    if not os.path.isdir(root):
        print(f"sweep root not found: {root}", file=sys.stderr)
        return 1

    if a.self_check:
        import random
        files = sorted(glob.glob(f"{root}/*/*_k*/dag_A*.json"))
        random.seed(0)
        pick = random.sample(files, min(a.self_check, len(files)))
        print(f"self-check: {len(pick)} reports against json.load")
        return 1 if check_against_json_load(pick) else 0

    cache = {}
    if not a.no_cache and os.path.exists(CACHE):
        try:
            loaded = pickle.load(open(CACHE, "rb"))
            if loaded.get("__schema__") == CACHE_SCHEMA:
                cache = loaded
            else:
                print(f"  cache schema {loaded.get('__schema__')} != "
                      f"{CACHE_SCHEMA}; re-reading every report")
        except Exception:                          # noqa: BLE001
            cache = {}
    cache["__schema__"] = CACHE_SCHEMA

    wl_of = workload_of(root)
    tasks = task_dirs(root, set(wl_of))
    plan, todo, hits = [], [], 0
    for model, cfg, k, d in tasks:
        for rung in RUNGS:
            p = f"{d}/dag_{rung}.json"
            try:
                st = os.stat(p)
            except OSError:
                continue
            stamp = (int(st.st_mtime), st.st_size)
            plan.append((model, cfg, k, rung, p, stamp))
            entry = cache.get(p)
            if isinstance(entry, tuple) and entry[0] == stamp:
                hits += 1
            else:
                todo.append((p, stamp))

    gb = sum(s[1] for *_, s in plan) / 2**30
    print(f"sweep root: {root}")
    print(f"reports: {len(plan)}   cached: {hits}   to read: {len(todo)}"
          f"   ({gb:.1f} GB on disk)")
    t0 = time.time()
    if todo:
        print(f"reading with {a.jobs} processes; this is I/O bound, so it takes "
              f"as long as it takes.\n")
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            done = 0
            for path, stamp, fields, err in ex.map(_job, todo, chunksize=1):
                done += 1
                if err:
                    print(f"  !! {path}: {err}", file=sys.stderr)
                else:
                    cache[path] = (stamp, fields)
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)}  {el/60:.1f} min elapsed"
                          f"  (eta {el/done*(len(todo)-done)/60:.1f} min)",
                          flush=True)
    if not a.no_cache:
        try:
            pickle.dump(cache, open(CACHE, "wb"), protocol=4)
        except Exception as exc:                   # noqa: BLE001
            print(f"  (cache not written: {exc})", file=sys.stderr)

    rung_rows, tier_rows, comp_rows = [], [], []
    present = {}
    for model, cfg, k, rung, p, _ in plan:
        e = cache.get(p)
        if not e or e[1] is None:
            continue
        f = e[1]
        wl = wl_of.get((model, cfg, k), "")
        ec = f.get("energy_by_class") or {}
        present.setdefault((model, cfg, k), []).append(rung)
        row = {"model": model, "config": cfg, "k": k, "workload": wl,
               "agents": agents_of(wl) if wl else "", "rung": rung}
        for col in RUNG_COLUMNS:
            if col in row:
                continue
            if col in ("prefill_time_s", "decode_time_s",
                       "power_prefill_w", "power_decode_w", "power_overall_w"):
                continue                       # filled in below, needs the sum
            if col == "tiers":
                # read_fields returns the tiers themselves, which belong in
                # sweep_tiers.csv; this column is their COUNT.  Writing the
                # list here put a repr with embedded commas and quotes into a
                # CSV cell -- caught by extract_sweep.sh's spot check, which
                # compares this column against len(summary.tiers).
                row[col] = len(f.get("tiers") or [])
            elif col.startswith("energy_") and col.endswith("_nj") and \
                    col != "energy_nj":
                row[col] = ec.get(col[len("energy_"):-len("_nj")].upper(), "")
            else:
                row[col] = f.get(col, "")
        # Phase times are the sum over tiers, which are sequential, so they
        # add to the run's makespan.  Power is energy over the time that phase
        # actually occupied -- not over the whole run, which would understate
        # both.
        tiers_f = f.get("tiers") or []
        p_t = sum(t["prefill_s"] for t in tiers_f
                  if isinstance(t.get("prefill_s"), float))
        d_t = sum(t["decode_s"] for t in tiers_f
                  if isinstance(t.get("decode_s"), float))
        e_p, e_d = f.get("energy_prefill_nj"), f.get("energy_decode_nj")
        mk, e_all = f.get("makespan_s"), f.get("energy_nj")
        row["energy_prefill_nj"] = e_p if e_p is not None else ""
        row["energy_decode_nj"] = e_d if e_d is not None else ""
        row["prefill_time_s"] = p_t if tiers_f else ""
        row["decode_time_s"] = d_t if tiers_f else ""
        row["power_prefill_w"] = (e_p / 1e9 / p_t) if (e_p and p_t > 0) else ""
        row["power_decode_w"] = (e_d / 1e9 / d_t) if (e_d and d_t > 0) else ""
        row["power_overall_w"] = ((e_all / 1e9 / mk)
                                  if (isinstance(e_all, float) and
                                      isinstance(mk, float) and mk > 0) else "")
        rung_rows.append(row)
        for t in tiers_f:
            tier_rows.append({
                "model": model, "config": cfg, "k": k, "rung": rung,
                "tier": t["tier"], "start_s": t["start_s"], "end_s": t["end_s"],
                # computed, not read
                "duration_s": t["end_s"] - t["start_s"],
                "requests": t["requests"],
                "prefill_end_s": t.get("prefill_end_s", ""),
                "first_token_s": t.get("first_token_s", ""),
                "prefill_s": t.get("prefill_s", ""),
                "decode_s": t.get("decode_s", ""),
            })
    for model, cfg, k, _ in tasks:
        wl = wl_of.get((model, cfg, k), "")
        have = present.get((model, cfg, k), [])
        comp_rows.append({
            "model": model, "config": cfg, "k": k, "workload": wl,
            "agents": agents_of(wl) if wl else "",
            "rungs_present": " ".join(r for r in RUNGS if r in have),
            "rungs_missing": " ".join(r for r in RUNGS if r not in have),
            "n_present": len(have),
            "claim_status": claim_status(root, model, cfg, k),
        })

    order = {r: i for i, r in enumerate(RUNGS)}

    def write(name, cols, rows, key):
        with open(os.path.join(a.outdir, name), "w", newline="") as fh:
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

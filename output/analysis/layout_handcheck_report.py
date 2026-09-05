#!/usr/bin/env python3
"""Turn the layout-probe dumps into the tables a hand-check needs.

    python3 output/analysis/layout_handcheck_report.py <dump_dir> [--request ID]

``<dump_dir>`` holds ``layout_<rung>.jsonl`` (written by KVPIM_LAYOUT_DUMP) and
``dag_<rung>.json`` (the ordinary run report).  Prints, per rung:

* the placement inputs and the channel load vector of one representative scan;
* per channel the rows, the activations, the measured time and energy, so the
  two reductions can be re-added by hand -- time is the MAX over channels, the
  scan's energy is the SUM over channels times ``num_hbm_used``;
* the ladder summary (makespan, energy, link bytes) from the run reports.

Nothing here recomputes physics.  Every number is read from a file; the only
arithmetic is the max and the sum being checked, and it is printed beside the
terms it came from.
"""

import argparse
import json
import os
import sys

RUNGS = ("A1", "A2", "A3b", "A4", "A4b", "A5", "A6")


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pick_scan(scans, request=None, want_diff=False):
    """One representative scan: prefer a decode scan of ``request`` that
    carries corrections, since that is the only kind where the master/diff
    rungs can differ from the naive ones."""
    pool = [s for s in scans if s["kind"] == "scan"]
    if request:
        pool = [s for s in pool if s["request"] == request] or pool
    if want_diff:
        withdiff = [s for s in pool if s.get("reads_diff", 0) > 0]
        if withdiff:
            return max(withdiff, key=lambda s: s["reads_diff"])
    return pool[0] if pool else None


def show_scan(rung, scan):
    print("### {}  policy={}".format(rung, scan["policy"]))
    print("  event      : {}".format(scan["event_name"]))
    print("  request    : {} (tier {})   layer {}".format(
        scan["request"], scan["tier"], scan["layer"]))
    print("  geometry   : heads/HBM={}  master_channels={}  kv_heads={}  "
          "num_hbm_used={}".format(scan["heads_per_hbm"],
                                   scan["master_channels"], scan["kv_heads"],
                                   scan["num_hbm_used"]))
    print("  reads      : master={}  diff={}".format(scan["reads_master"],
                                                     scan["reads_diff"]))
    print("  loads[16]  : {}".format([int(x) for x in scan["loads"]]))
    print("  active     : {}".format(scan["active"]))
    print("  per channel:")
    print("    ch | tokens | ACT |  time (us) | energy 1 stack (nJ) | charged (nJ)")
    for term in scan["per_channel"]:
        print("    {:>2} | {:>6} | {:>3} | {:>10.4f} | {:>19.1f} | {:>12.1f}".format(
            term["channel"], term["rows"], term["acts"], term["time_s"] * 1e6,
            term["energy_nj_one_stack"], term["energy_nj_charged"]))
    times = [t["time_s"] for t in scan["per_channel"]]
    energies = [t["energy_nj_charged"] for t in scan["per_channel"]]
    print("  reduction  : time  = MAX = {:.4f} us  (channel {})".format(
        scan["scan_time_s"] * 1e6, scan["scan_time_channel"]))
    print("               energy= SUM = {:.1f} nJ  over {} channels".format(
        scan["scan_energy_nj"], len(energies)))
    ok_t = abs(max(times, default=0.0) - scan["scan_time_s"]) < 1e-18
    ok_e = abs(sum(energies) - scan["scan_energy_nj"]) < 1e-6
    print("  re-added   : max ok={}  sum ok={}   total ACT={}".format(
        ok_t, ok_e, scan["scan_acts"]))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir")
    parser.add_argument("--request", default=None,
                        help="prefer scans of this request id")
    parser.add_argument("--rungs", default=",".join(RUNGS))
    args = parser.parse_args()
    rungs = [r for r in args.rungs.split(",") if r]

    print("# layout hand-check:", args.dump_dir)
    print()
    print("## ladder summary (from dag_<rung>.json)")
    print()
    print("| rung | makespan (s) | energy (nJ) | link bytes | events | policy | kv_mapping | decode | prefill |")
    print("|---|---:|---:|---:|---:|---|---|---|---|")
    for rung in rungs:
        path = os.path.join(args.dump_dir, "dag_{}.json".format(rung))
        if not os.path.exists(path):
            print("| {} | (missing) | | | | | | | |".format(rung))
            continue
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        print("| {} | {:.6f} | {:.6g} | {} | {} | {} | {} | {} | {} |".format(
            rung, report.get("makespan_s", float("nan")),
            report.get("energy_nj", float("nan")),
            report.get("link_bytes", ""), report.get("event_count", ""),
            report.get("policy", ""), report.get("kv_mapping", ""),
            report.get("decode_attn", ""), report.get("pim_prefill_mode", "")))
    print()

    print("## one scan per rung, with both reductions spelled out")
    print()
    for rung in rungs:
        scans = load(os.path.join(args.dump_dir,
                                  "layout_{}.jsonl".format(rung)))
        scan = pick_scan(scans, request=args.request, want_diff=True)
        if scan is None:
            print("### {}  -- no PIM scan recorded".format(rung))
            print("  (A2 keeps decode attention on the GPU, so the engine "
                  "never builds a PIM scan: there is nothing to place.)")
            print()
            continue
        show_scan(rung, scan)

    print("## block table (per cached CHUNK, layer 0)")
    print()
    for rung in rungs:
        rows = load(os.path.join(args.dump_dir,
                                 "layout_{}.jsonl".format(rung)))
        blocks = [r for r in rows if r["kind"] == "blocks"]
        if not blocks:
            print("- {}: no block table".format(rung))
            continue
        record = blocks[-1]
        by_channel = {}
        for block in record["blocks"]:
            by_channel.setdefault(block["channel_base"], 0)
            by_channel[block["channel_base"]] += 1
        print("- {}: {} chunks kept of {} total; chunks per TLB channel: {}".format(
            rung, record["n_blocks_kept"], record["n_blocks_total"],
            dict(sorted(by_channel.items()))))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

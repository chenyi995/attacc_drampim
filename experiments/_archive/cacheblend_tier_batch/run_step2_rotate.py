#!/usr/bin/env python3
"""Validate three CacheBlend reused-Q rotation distribution policies.

RoPE rotation is a block-diagonal pairwise operation.  This experiment assigns
it *zero compute overhead* as requested and measures only the externally
visible GPU/PIM distribution timing and traffic.  The number of Q variants is
the number of distinct position deltas, not a dense-matmul cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results" / "step2"


def evaluate(deltas: tuple[int, ...], kinds: tuple[str, ...], query_bytes: int,
             link_bandwidth: float, die_rotate_cycle_s: float):
    """Evaluate one Q per distinct position delta.

    ``gpu_rotate`` sends every variant across the external link.  ``die`` and
    ``bank`` send the same single raw Q across that link; their distribution
    time is therefore identical.  A die has one rotate unit: non-zero variants
    become available one clock at a time, master targets first and diff targets
    afterwards.  Bank rotate is the requested timing-only idealisation, with
    rotation local at the receiving bank and no added die dispatch latency.
    """
    if len(deltas) != len(kinds):
        raise ValueError("--variant-kinds must contain one kind per delta")
    if set(kinds) - {"master", "diff"}:
        raise ValueError("variant kinds must be master or diff")
    targets = {}
    for delta, kind in zip(deltas, kinds):
        targets.setdefault(delta, set()).add(kind)
    # Master is served before diff on the one die rotate unit.  This makes a
    # shifted diff Q arrive one cycle after the preceding master Q; additional
    # distinct Q variants consume additional cycles in the same order.
    variants_by_target = tuple(sorted(
        targets, key=lambda delta: (0 if "master" in targets[delta] else 1, delta)))
    variants = len(variants_by_target)
    die_available = {}
    issued = 0
    for delta in variants_by_target:
        if delta == 0:
            die_available[delta] = 0.0
        else:
            issued += 1
            die_available[delta] = issued * die_rotate_cycle_s
    master_ready = max((die_available[delta] for delta in variants_by_target
                        if "master" in targets[delta]), default=0.0)
    diff_ready = max((die_available[delta] for delta in variants_by_target
                      if "diff" in targets[delta]), default=0.0)
    policies = {
        "gpu_rotate": {
            "gpu_to_pim_bytes": variants * query_bytes,
            "die_to_bank_bytes": 0,
            "bank_local_bytes": 0,
        },
        "die_rotate": {
            "gpu_to_pim_bytes": query_bytes,
            # Both die/bank distribute one incoming raw Q.  Internal bytes
            # are accounting only; the user requested no separate hardware
            # overhead model here.
            "die_to_bank_bytes": variants * query_bytes,
            "bank_local_bytes": 0,
            "die_variant_ready_s": die_available,
            "die_dispatch_time_s": max(die_available.values(), default=0.0),
            "master_q_ready_s": master_ready,
            "diff_q_ready_s": diff_ready,
        },
        "bank_rotate": {
            "gpu_to_pim_bytes": query_bytes,
            "die_to_bank_bytes": 0,
            "bank_local_bytes": query_bytes,
            "die_variant_ready_s": {delta: 0.0 for delta in variants_by_target},
            "die_dispatch_time_s": 0.0,
            "master_q_ready_s": 0.0,
            "diff_q_ready_s": 0.0,
        },
    }
    for result in policies.values():
        result["q_variants"] = variants
        result["external_link_time_s"] = result["gpu_to_pim_bytes"] / link_bandwidth
        result["rotate_compute_time_s"] = 0.0
        result.setdefault("die_variant_ready_s",
                          {delta: 0.0 for delta in variants_by_target})
        result.setdefault("die_dispatch_time_s", 0.0)
        result.setdefault("master_q_ready_s", 0.0)
        result.setdefault("diff_q_ready_s", 0.0)
        result["makespan_s"] = (result["external_link_time_s"] +
                                 result["die_dispatch_time_s"])
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deltas", default="0,-3,5",
                        help="comma-separated logical-position deltas")
    parser.add_argument("--query-bytes", type=int, default=8192,
                        help="one tensor-parallel Q shard; 4096 FP16 elements by default")
    parser.add_argument("--link-bandwidth", type=float, default=300e9,
                        help="bytes/s; AttAcc's 600 GB/s NVLink divided for one direction")
    parser.add_argument("--variant-kinds", default="master,diff,master",
                        help="master/diff target for every delta, used for die dispatch order")
    parser.add_argument("--die-rotate-cycle-ns", type=float, default=1.0,
                        help="one die rotate-unit issue cycle per shifted Q variant")
    parser.add_argument("--mode", choices=("all", "gpu_rotate", "die_rotate", "bank_rotate"),
                        default="all", help="policy to emit; all keeps the three-way comparison")
    args = parser.parse_args()
    deltas = tuple(int(value) for value in args.deltas.split(","))
    kinds = tuple(value.strip() for value in args.variant_kinds.split(","))
    if (not deltas or args.query_bytes <= 0 or args.link_bandwidth <= 0 or
            args.die_rotate_cycle_ns <= 0):
        raise ValueError("deltas, query bytes and link bandwidth must be positive/non-empty")
    RESULTS.mkdir(parents=True, exist_ok=True)

    shifted_all = evaluate(deltas, kinds, args.query_bytes, args.link_bandwidth,
                           args.die_rotate_cycle_ns * 1e-9)
    stable = evaluate(tuple(0 for _ in deltas), kinds, args.query_bytes,
                      args.link_bandwidth, args.die_rotate_cycle_ns * 1e-9)
    shifted = (shifted_all if args.mode == "all"
               else {args.mode: shifted_all[args.mode]})
    stable_values = {(entry["gpu_to_pim_bytes"], entry["makespan_s"])
                     for entry in stable.values()}
    if len(stable_values) != 1:
        raise AssertionError("zero-delta policies must degenerate to identical results")

    payload = {"assumptions": {
        "deltas": list(deltas), "query_bytes": args.query_bytes,
        "external_link_bandwidth_bytes_per_s": args.link_bandwidth,
        "variant_kinds": list(kinds), "die_rotate_cycle_s": args.die_rotate_cycle_ns * 1e-9,
        "selected_mode": args.mode,
        "rotate_compute_overhead": "zero (timing-only validation)",
    }, "shifted_positions": shifted, "all_positions_stable": stable,
       "stable_degeneracy_passed": True}
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2) + "\n",
                                             encoding="utf-8")
    lines = ["# Step 2: Q rotation distribution", "",
             "RoPE is represented as per-pair/block-diagonal rotation with zero compute",
             "charge; only distribution traffic affects elapsed time.", "",
             "## Shifted reused blocks", "",
             "| policy | Q variants | GPU→PIM bytes | die dispatch (ns) | diff Q ready (ns) | external link time (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, value in shifted.items():
        lines.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.12g} |".format(
            name, value["q_variants"], value["gpu_to_pim_bytes"],
            value["die_dispatch_time_s"] * 1e9, value["diff_q_ready_s"] * 1e9,
            value["external_link_time_s"]))
    lines.extend(["", "## Degeneracy check", "",
                  "All deltas set to zero: **passed**.  GPU rotate, die rotate and",
                  "bank rotate each send exactly one Q and have the same timing.", "",
                  "Die and bank modes have identical external distribution traffic; only",
                  "die mode adds the one-rotate-unit serial availability delay for shifted Q."])
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(RESULTS / "results.json")


if __name__ == "__main__":
    main()

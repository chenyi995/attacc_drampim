#!/usr/bin/env python3
"""Measure CacheBlend shared-KV batching directly with Ramulator.

The experiment holds one *master* reused KV segment fixed and gives each
query a private *diff* coverage segment.  ``incremental`` concatenates one
master+diff scan per arriving query; ``large`` emits one joint scan for the
shared master segment but keeps every private diff scan independent.  Both
traces therefore use the same physical layout, sequence length and total work.

The patched HBM3-PIM controller reports ``pim_activations`` for commands it
actually issues (ACT/ACTAB/ACTSB/ACTPB), not an estimate from source code.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAMULATOR = ROOT / "ramulator2"
GENERATOR = RAMULATOR / "trace_gen" / "gen_trace_attacc_bank.py"
RESULTS = Path(__file__).resolve().parent / "results" / "step1_current_layout"

# CacheBlend's current default layout: master owns channels 0--7 and diff
# owns channels 8--15.  Each class follows original AttAcc's 8-KiB head
# partitioning, with V fixed at K + 8 MiB.
MASTER_KEY = 0x000000000
MASTER_VALUE = MASTER_KEY + (1 << 23)
DIFF_KEY = 0x200000000
DIFF_VALUE = DIFF_KEY + (1 << 23)
CHANNELS_PER_CLASS = 8
K_WINDOW_BYTES = 1 << 23
KV_TILE_BYTES = 2 * K_WINDOW_BYTES
HEAD_PARTITION_BYTES = 1 << 13
HBM3_TCK_NS = 1.3


def _round_robin(streams: list[str]) -> str:
    """Preserve each request's command order while interleaving requests."""
    lines = [stream.splitlines(keepends=True) for stream in streams]
    return "".join(line for group in itertools.zip_longest(*lines, fillvalue="")
                   for line in group)


def _diff_block_base(query: int, *, rows: int, bytes_per_vector: int) -> int:
    """Mirror CacheBlendTLB's contiguous diff allocation for one layer.

    A block grows with its actual number of KV rows, is aligned to the
    original 8-KiB partition, and spills only after the 8-MiB K window is
    exhausted.  It does not manufacture a row number for a request.
    """
    span = math.ceil(rows * bytes_per_vector / HEAD_PARTITION_BYTES) * HEAD_PARTITION_BYTES
    if span > K_WINDOW_BYTES:
        raise ValueError("one diff block exceeds CacheBlendTLB's 8-MiB K window")
    linear_offset = query * span
    return DIFF_KEY + (linear_offset // K_WINDOW_BYTES) * KV_TILE_BYTES + \
        (linear_offset % K_WINDOW_BYTES)


def _arrival_threshold_metrics(records: list[dict[str, float]], *, total_queries: int,
                               q_bytes: int, link_gbps: float) -> list[dict[str, float]]:
    """Schedule fixed-size batches from GPU-to-PIM Q-link completions."""
    profiles = {int(row["batch_size"]): row for row in records}
    link_interval_ns = q_bytes / link_gbps
    arrivals = [(index + 1) * link_interval_ns for index in range(total_queries)]
    metrics = []
    for threshold in sorted(profiles):
        pim_ready = 0.0
        completions = [0.0] * total_queries
        for first in range(0, total_queries, threshold):
            size = min(threshold, total_queries - first)
            profile = profiles[size]
            start = max(pim_ready, arrivals[first + size - 1])
            done = start + profile["large_cycles"] * HBM3_TCK_NS
            for index in range(first, first + size):
                completions[index] = done
            pim_ready = done
        latencies = sorted(done - arrived for done, arrived in zip(completions, arrivals))
        p99_index = max(0, math.ceil(.99 * len(latencies)) - 1)
        metrics.append({"threshold": threshold,
                        "q_link_interval_ns": link_interval_ns,
                        "pim_makespan_ns": pim_ready,
                        "average_q_to_context_ns": sum(latencies) / len(latencies),
                        "p99_q_to_context_ns": latencies[p99_index],
                        "max_q_to_context_ns": latencies[-1]})
    return metrics


def _yaml(trace: Path, activation_path: Path) -> str:
    return """Frontend:
  impl: PIMLoadStoreTrace
  path: {trace}
  clock_ratio: 1

  Translation:
    impl: NoTranslation
    # diff starts in channel 8, whose byte base is 8 GiB.
    max_addr: 34359738368

MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: HBM3_5.2Gbps_NPC

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
    plugins:
      - ControllerPlugin:
          impl: CommandCounter
          commands_to_count: [ACT, ACTAB, ACTSB, ACTPB]
          path: {activation_path}

  AddrMapper:
    impl: HBM3-PIM
""".format(trace=trace, activation_path=activation_path)


def generate(trace: Path, *, seqlen: int, maxlen: int, dhead: int, nhead: int,
             key_addr: int, value_addr: int, shared_kv: bool) -> None:
    subprocess.run([sys.executable, str(GENERATOR), "--dhead", str(dhead),
                    "--nhead", str(nhead), "--seqlen", str(seqlen),
                    "--maxlen", str(maxlen), "--dbyte", "2", "--output",
                    str(trace), "--key-addr", hex(key_addr), "--value-addr",
                    hex(value_addr), "--channels", str(CHANNELS_PER_CLASS)] +
                   (["--shared-kv"] if shared_kv else []),
                   check=True, stdout=subprocess.DEVNULL)


def scan_trace(*, stem: str, seqlen: int, maxlen: int, dhead: int, nhead: int,
               key_addr: int, value_addr: int, shared_kv: bool) -> str:
    """Return one trace's commands without changing its physical placement."""
    trace = RESULTS / (stem + ".trace")
    generate(trace, seqlen=seqlen, maxlen=maxlen, dhead=dhead, nhead=nhead,
             key_addr=key_addr, value_addr=value_addr, shared_kv=shared_kv)
    try:
        return trace.read_text(encoding="utf-8")
    finally:
        trace.unlink(missing_ok=True)


def ramulate(name: str, trace: Path) -> tuple[int, int, int]:
    yaml = RESULTS / (name + ".yaml")
    activation_path = RESULTS / (name + ".activations")
    for prior in RESULTS.glob(activation_path.name + ".channel*"):
        prior.unlink()
    yaml.write_text(_yaml(trace, activation_path), encoding="utf-8")
    try:
        output = subprocess.run([str(RAMULATOR / "ramulator2"), "-f", str(yaml)],
                                check=True, text=True, stdout=subprocess.PIPE).stdout
    finally:
        yaml.unlink(missing_ok=True)
    cycles = sum(int(line.split()[-1]) for line in output.splitlines()
                 if "memory_system_cycles" in line)
    activation_files = sorted(RESULTS.glob(activation_path.name + ".channel*"))
    if len(activation_files) != 16:
        raise RuntimeError("expected one CommandCounter result for each HBM channel")
    activations = sum(int(line.rsplit(",", 1)[1])
                      for result in activation_files
                      for line in result.read_text(encoding="utf-8").splitlines()
                      if line.split(",", 1)[0] in {"ACT", "ACTAB", "ACTSB", "ACTPB"})
    # MACAB is the PIM bank's column-MAC command.  It is retained separately
    # from activation commands to distinguish an ACT bottleneck from a column
    # command bottleneck.
    macab = sum(int(line.rsplit(",", 1)[1])
                for result in activation_files
                for line in result.read_text(encoding="utf-8").splitlines()
                if line.split(",", 1)[0] == "MACAB")
    if not cycles:
        raise RuntimeError("Ramulator did not report memory_system_cycles")
    return cycles, activations, macab


def main() -> None:
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--maxlen", type=int, default=4096,
                        help="physical K/V reservation length; 4096 gives 8-KiB partitions")
    parser.add_argument("--dhead", type=int, default=128)
    parser.add_argument("--heads-per-query", type=int, default=16)
    parser.add_argument("--diff-rows", type=int, default=8,
                        help="private diff KV rows per query; increase for the stress case")
    parser.add_argument("--arrival-queries", type=int, default=32,
                        help="number of serialized Q-link arrivals for threshold sweep")
    parser.add_argument("--q-link-gbps", type=float, default=300.0,
                        help="one-way GPU-to-PIM Q-link bandwidth in GB/s")
    parser.add_argument("--act-stress", action="store_true",
                        help="interleave private diff streams across rows in the same bank")
    args = parser.parse_args()
    if args.act_stress:
        RESULTS = Path(__file__).resolve().parent / "results" / "step1_act_stress"
    batches = [int(value) for value in args.batch_sizes.split(",")]
    if not batches or any(value <= 0 for value in batches):
        raise ValueError("batch sizes must be positive")
    if args.heads_per_query <= 0:
        raise ValueError("heads per query must be positive")
    if args.diff_rows <= 0:
        raise ValueError("diff rows must be positive")
    if args.arrival_queries <= 0 or args.q_link_gbps <= 0:
        raise ValueError("arrival queries and Q-link bandwidth must be positive")
    head_groups = math.ceil(args.heads_per_query / CHANNELS_PER_CLASS)
    partition_bytes = math.ceil(args.maxlen * args.dhead / (2 * 2 * 4 * 4))
    if partition_bytes != 8192:
        raise ValueError("this current-layout experiment requires an 8-KiB head partition; "
                         "use --maxlen 4096 --dhead 128")
    RESULTS.mkdir(parents=True, exist_ok=True)
    bytes_per_vector = args.heads_per_query * args.dhead * 2

    records = []
    for batch in batches:
        # The joint part is exactly the common master KV segment.  Per-query
        # diff segments remain independent because their coverage vectors
        # cannot be reused by another query.
        large_trace = RESULTS / ("large_b{}.trace".format(batch))
        large_commands = []
        # Batch queries within one head partition.  Flattening all heads as
        # ``nhead = heads_per_query * batch`` would pair a query's own head
        # partitions first and would miss the intended cross-query row reuse.
        for group in range(head_groups):
            group_heads = min(CHANNELS_PER_CLASS,
                              args.heads_per_query - group * CHANNELS_PER_CLASS)
            group_offset = group * partition_bytes
            large_commands.append(scan_trace(
                stem="large_b{}_master_h{}".format(batch, group),
                seqlen=args.seqlen, maxlen=args.maxlen, dhead=args.dhead,
                nhead=group_heads * batch, key_addr=MASTER_KEY + group_offset,
                value_addr=MASTER_VALUE + group_offset, shared_kv=True))
        stress_diff_streams = []
        for query in range(batch):
            diff_key = _diff_block_base(query, rows=args.diff_rows,
                                        bytes_per_vector=bytes_per_vector)
            query_diff_commands = []
            for group in range(head_groups):
                group_heads = min(CHANNELS_PER_CLASS,
                                  args.heads_per_query - group * CHANNELS_PER_CLASS)
                group_offset = group * partition_bytes
                query_diff_commands.append(scan_trace(
                    stem="large_b{}_diff_q{}_h{}".format(batch, query, group),
                    seqlen=args.diff_rows, maxlen=args.maxlen, dhead=args.dhead,
                    nhead=group_heads, key_addr=diff_key + group_offset,
                    value_addr=diff_key + (1 << 23) + group_offset,
                    shared_kv=False))
            if args.act_stress:
                stress_diff_streams.append("".join(query_diff_commands))
            else:
                large_commands.extend(query_diff_commands)
        if args.act_stress:
            large_commands.append(_round_robin(stress_diff_streams))
        large_trace.write_text("".join(large_commands), encoding="utf-8")
        large_cycles, large_act, large_macab = ramulate("large_b{}".format(batch), large_trace)

        small_parts = []
        for query in range(batch):
            diff_key = _diff_block_base(query, rows=args.diff_rows,
                                        bytes_per_vector=bytes_per_vector)
            query_master_commands = []
            query_diff_commands = []
            for group in range(head_groups):
                group_heads = min(CHANNELS_PER_CLASS,
                                  args.heads_per_query - group * CHANNELS_PER_CLASS)
                group_offset = group * partition_bytes
                query_master_commands.append(scan_trace(
                    stem="incremental_b{}_master_q{}_h{}".format(batch, query, group),
                    seqlen=args.seqlen, maxlen=args.maxlen, dhead=args.dhead,
                    nhead=group_heads, key_addr=MASTER_KEY + group_offset,
                    value_addr=MASTER_VALUE + group_offset, shared_kv=True))
                query_diff_commands.append(scan_trace(
                    stem="incremental_b{}_diff_q{}_h{}".format(batch, query, group),
                    seqlen=args.diff_rows, maxlen=args.maxlen, dhead=args.dhead,
                    nhead=group_heads, key_addr=diff_key + group_offset,
                    value_addr=diff_key + (1 << 23) + group_offset,
                    shared_kv=False))
            # This must match the B=1 joint trace exactly: all head partitions
            # of master, then all head partitions of the same query's diff.
            small_parts.extend(query_master_commands + query_diff_commands)
        small_trace = RESULTS / ("incremental_b{}.trace".format(batch))
        small_trace.write_text("".join(small_parts), encoding="utf-8")
        small_cycles, small_act, small_macab = ramulate("incremental_b{}".format(batch), small_trace)
        records.append({"batch_size": batch,
                        "incremental_cycles": small_cycles,
                        "large_cycles": large_cycles,
                        "incremental_activations": small_act,
                        "large_activations": large_act,
                        "incremental_macab": small_macab,
                        "large_macab": large_macab,
                        "cycle_ratio_large_over_incremental": large_cycles / small_cycles,
                        "activation_ratio_large_over_incremental": (
                            large_act / small_act if small_act else 0.0),
                        "macab_ratio_large_over_incremental": (
                            large_macab / small_macab if small_macab else 0.0)})

    one = next((row for row in records if row["batch_size"] == 1), None)
    if one is not None and (one["incremental_cycles"] != one["large_cycles"] or
                            one["incremental_activations"] != one["large_activations"] or
                            one["incremental_macab"] != one["large_macab"]):
        raise RuntimeError("B=1 incremental and joint traces must be identical")

    with (RESULTS / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    arrival_metrics = _arrival_threshold_metrics(
        records, total_queries=args.arrival_queries, q_bytes=bytes_per_vector,
        link_gbps=args.q_link_gbps)
    with (RESULTS / "arrival_threshold.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=arrival_metrics[0].keys())
        writer.writeheader()
        writer.writerows(arrival_metrics)
    scenario = ("ACT-stress: diff blocks share banks and are command-interleaved"
                if args.act_stress else "current CacheBlend layout")
    lines = ["# Step 1: Ramulator batching — {}".format(scenario), "",
             "Master KV is shared in channels 0--7; each query has an independent",
             "diff coverage segment in channels 8--15.  `large` jointly scans only",
             "the shared master segment, while `incremental` rescans it per arrival.", "",
             "| batch | incremental cycles | large cycles | incremental ACT | large ACT | incremental MACAB | large MACAB | large/inc cycles | large/inc ACT |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append("| {batch_size} | {incremental_cycles} | {large_cycles} | "
                     "{incremental_activations} | {large_activations} | {incremental_macab} | {large_macab} | "
                     "{cycle_ratio_large_over_incremental:.3f} | "
                     "{activation_ratio_large_over_incremental:.3f} |".format(**row))
    lines.extend(["", "## Fixed Q-arrival threshold sweep", "",
                  "Q arrival is GPU→PIM Q-link completion.  The link serializes {}-B Q "
                  "vectors at {} GB/s; batch execution starts when its B-th Q is ready "
                  "and PIM is idle.  The sweep has {} arrivals and uses HBM3 tCK={} ns."
                  .format(bytes_per_vector, args.q_link_gbps,
                          args.arrival_queries, HBM3_TCK_NS),
                  "", "| threshold B | PIM makespan (ns) | avg Q→context (ns) | p99 Q→context (ns) | max Q→context (ns) |",
                  "|---:|---:|---:|---:|---:|"])
    for row in arrival_metrics:
        lines.append("| {threshold} | {pim_makespan_ns:.2f} | {average_q_to_context_ns:.2f} | "
                     "{p99_q_to_context_ns:.2f} | {max_q_to_context_ns:.2f} |".format(**row))
    lines.extend(["", "`pim_activations` counts ACT, ACTAB, ACTSB and ACTPB commands",
                  "actually issued by Ramulator's HBM3-PIM controller.",
                  "`MACAB` is the bank-level PIM column-MAC command count.",
                  "The trace generator's joint master schedule is pairwise interleaved;",
                  "the result measures that concrete trace rather than asserting an",
                  "arbitrary large-batch scheduler.",
                  "Each 8-channel head partition is batched across queries; the next",
                  "partition uses K/V addresses 8 KiB later, matching the TLB layout."])
    if args.act_stress:
        lines.extend(["", "Stress construction: private diff blocks are allocated by the "
                      "same contiguous CacheBlendTLB formula as the normal case.  Their "
                      "larger row count naturally spans many DRAM rows; the joint schedule "
                      "round-robins those command streams, while incremental keeps each "
                      "stream contiguous."])
    if len(records) > 1:
        batched = records[1:]
        worst_cycle_ratio = max(row["cycle_ratio_large_over_incremental"]
                                for row in batched)
        worst_macab_ratio = max(row["macab_ratio_large_over_incremental"]
                                for row in batched)
        lines.extend(["", "## Conclusion", ""])
        if worst_cycle_ratio <= 1.0:
            lines.append(
                "For the measured batches, the joint schedule does not stall behind "
                "ACT or column-MAC saturation: its total cycles remain no greater than "
                "{:.3f}× incremental.  ACT falls because only master is shared; the "
                "private diff scans remain.  MACAB stays within {:.3f}×, so the remaining "
                "time is dominated by essentially unchanged column work and the "
                "unbatchable diff portion.".format(worst_cycle_ratio, worst_macab_ratio))
        else:
            lines.append(
                "The joint schedule can lose despite issuing fewer ACTs: its worst case "
                "is {:.3f}× incremental while MACAB remains within {:.3f}×.  This is the "
                "row-conflict/ACT-PRE scheduling cost of interleaving long private diff "
                "streams; the shared-master saving must exceed it before batching wins."
                .format(worst_cycle_ratio, worst_macab_ratio))
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(RESULTS / "results.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does a workload give each rung something to separate it from the last?

    python3 output/analysis/b1_levers.py workload/probe/sweep/B1_turns.json [more.json ...]

No Ramulator, no run.  For every workload it builds the real reuse plan and
the real TLB / ledger (CACHEBLEND-TINY geometry: 8 KV heads on one HBM,
stripe 2) and reports three levers, each with the mechanism it feeds:

  A4e  -- co-read conflicts: for every turn's prefill read set, the rows on
          the busiest channel of one head under the naive write-order slot
          (A3b/A4c) and under the conflict-aware table (A4e).  Same busiest
          lane = the table has nothing to fix.
  A4c  -- repair rows: for every turn's read set, the DRAM rows the repairs
          (this agent's own and inherited corrections) occupy under A3b's
          naive stream and under A4c's per-head diff region, from the same
          ledger the scans use.  Equal = gathering buys no activations.
  A5/A6 -- decode-shaped turns: per turn m (rows computed), n (rows scanned)
          and R (rows the GPU would read back), and a CALIBRATED estimate of
          the two prefill prices.  Calibration (推导, from the 2026-09-05
          4-agent run, sides.jsonl): one bank sweep over the busiest lane =
          4.05 us x lane_rows / 1536; GPU attention = 100 us x (m*n)/(816*2864)
          (flash, TINY); readback = 6.06 us + bytes / 335 GB/s.  The real
          chooser prices through Ramulator and the device models; this
          estimate only says which side the workload SHAPES favour.
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.workload import load_workload, build_reuse_plan  # noqa: E402
from src.workload_runner import (  # noqa: E402
    NaiveKVLayout, LocalDiffKVLayout, TableLocalDiffKVLayout, _prepare_cacheblend_tlb,
    _parent_output_fingerprints, _cacheblend_tlb_rows, _pool_reads, _striped_append_channel_extents,
    _GEN_ROW_BYTES, _HBM_CHANNELS)

# KV heads on the busiest HBM: 8 = CACHEBLEND-TINY on 1 HBM (stripe 2);
# LEVERS_HEADS_PER_HBM=2 = LLAMA3-8B / TINY on 4 HBMs (stripe 8, the
# eight-channel geometry chenyi9 fixed on 2026-09-05).
HEADS_PER_HBM = int(os.environ.get("LEVERS_HEADS_PER_HBM", "8"))
STRIPE = max(1, 16 // HEADS_PER_HBM)
KV_ROW_BYTES = 2 * 1024 * 2  # K+V, hidden 1024, fp16 (MHA)


def lane_rows(groups):
    return {channel: sum(rows for _k, _v, rows in placed) for channel, _n, placed in groups}


ROW_ADDR_BYTES = 4    # the generator's address map: a 1 KiB DRAM row holds 256 KV rows


def dram_rows(key, value, count):
    """DRAM rows (1 KiB, `_GEN_ROW_BYTES`) the K stream of an extent of
    ``count`` KV rows touches; the V stream mirrors it at a fixed gap."""
    return set(range(key // _GEN_ROW_BYTES, (key + count * ROW_ADDR_BYTES - 1) // _GEN_ROW_BYTES + 1))


def repair_rows(groups, diff_keys):
    """Distinct DRAM rows that the diff extents occupy."""
    rows = set()
    for channel, _n, placed in groups:
        for key, value, count in placed:
            if key in diff_keys:
                rows.update((channel, r) for r in dram_rows(key, value, count))
    return len(rows)


def lane_headroom(groups):
    """Per lane: DRAM rows touched, and the packed minimum for the same bytes."""
    touched, nbytes = {}, {}
    for channel, _n, placed in groups:
        s = touched.setdefault(channel, set())
        for key, value, count in placed:
            s.update(dram_rows(key, value, count))
            nbytes[channel] = nbytes.get(channel, 0) + count * ROW_ADDR_BYTES
    return ({c: len(s) for c, s in touched.items()},
            {c: math.ceil(b / _GEN_ROW_BYTES) for c, b in nbytes.items()})


def analyse(path):
    wl = load_workload(path)
    plan = build_reuse_plan(wl, "recompute", epic_prefix_recompute_tokens=8)
    layer = 0
    outputs = _parent_output_fingerprints(wl)
    tlbs = {"A3b": NaiveKVLayout(256, "slice"),
            "A4c": LocalDiffKVLayout(256, "slice"),
            "A4e": TableLocalDiffKVLayout(256, "slice")}
    policies = {"A3b": "slice-append", "A4c": "master-diff-local-append",
                "A4e": "master-diff-table-local-append"}
    for tlb in tlbs.values():
        _prepare_cacheblend_tlb(wl, plan, 1, tlb, outputs)
    conflict = {"A3b": 0, "A4e": 0}
    mean_lane = {"A3b": 0.0, "A4e": 0.0}
    repairs = {"A3b": 0, "A4c": 0}
    headroom = {r: dict(max=0, mean=0.0, packed_max=0, ideal=0.0) for r in ("A3b", "A4c", "A4e")}
    turns = []
    for request in wl.requests:
        bindings = {rung: _cacheblend_tlb_rows(wl, plan, layer, request, tlb)
                    for rung, tlb in tlbs.items()}
        # busiest lane under naive slots vs the table (masters only matter here)
        for rung in ("A3b", "A4e"):
            reads, _m, _p = _pool_reads(tlbs[rung], [b[3] for b in bindings[rung] if b[1]])
            groups = _striped_append_channel_extents(reads, policy=policies[rung],
                                                     heads_per_hbm=HEADS_PER_HBM, tlb=tlbs[rung])
            lanes = lane_rows(groups)
            conflict[rung] += max(lanes.values(), default=0)
            mean_lane[rung] += sum(lanes.values()) / len(lanes) if lanes else 0.0
        # headroom: busiest lane's DRAM rows vs packed minimum vs balanced ideal
        for rung in ("A3b", "A4c", "A4e"):
            reads, _m, _p = _pool_reads(tlbs[rung], [b[3] for b in bindings[rung] if b[1]])
            groups = _striped_append_channel_extents(reads, policy=policies[rung],
                                                     heads_per_hbm=HEADS_PER_HBM, tlb=tlbs[rung])
            touched, packed = lane_headroom(groups)
            if touched:
                h = headroom[rung]
                h["max"] += max(touched.values()); h["mean"] += sum(touched.values()) / len(touched)
                h["packed_max"] += max(packed.values()); h["ideal"] += sum(packed.values()) / len(packed)
        # repair rows under the naive stream vs the diff region
        for rung in ("A3b", "A4c"):
            reads, _m, _p = _pool_reads(tlbs[rung], [b[3] for b in bindings[rung] if b[1]])
            groups = _striped_append_channel_extents(reads, policy=policies[rung],
                                                     heads_per_hbm=HEADS_PER_HBM, tlb=tlbs[rung])
            diff_locs = [b[3] for b in bindings[rung] if b[1] and b[2]]
            diff_only = _striped_append_channel_extents(diff_locs, policy=policies[rung],
                                                        heads_per_hbm=HEADS_PER_HBM, tlb=tlbs[rung])
            diff_keys = {key for _c, _n, placed in diff_only for key, _v, _r in placed}
            repairs[rung] += repair_rows(groups, diff_keys)
        # decode-shaped or not: the chooser's two prices, calibrated
        b = bindings["A4e"]
        m = sum(1 for pos, reused, corrected, loc in b
                if not reused or (corrected and loc.owner == request.request_id))
        resident = sum(1 for pos, reused, corrected, loc in b if reused and
                       (not corrected or loc.owner != request.request_id))
        n = m + resident
        lane = math.ceil(n / STRIPE)                   # one head's rows over its stripe
        sweeps = math.ceil(m / 8)
        t_bank = sweeps * 4.05e-6 * lane / 1536
        t_xpu = (6.06e-6 + resident * KV_ROW_BYTES / 335e9 if resident else 0.0) + \
            100e-6 * (m * n) / (816 * 2864)
        turns.append((request.request_id, m, n, resident, t_xpu, t_bank))
    pim = [t for t in turns if t[5] <= t[4]]
    gpu = [t for t in turns if t[5] > t[4]]
    owner = next(t for t in turns if t[0] == "a0_owner")
    print("%s  (heads/HBM %d, stripe %d)" % (os.path.basename(path), HEADS_PER_HBM, STRIPE))
    print("  A4e lever  busiest-lane rows over all turns: naive %d  table %d  (%.1f%% fewer); mean lane naive %.0f -> imbalance headroom naive %.1f%%, table %.1f%%" % (
        conflict["A3b"], conflict["A4e"], 100 * (1 - conflict["A4e"] / max(1, conflict["A3b"])), mean_lane["A3b"],
        100 * (1 - mean_lane["A3b"] / max(1, conflict["A3b"])), 100 * (1 - mean_lane["A4e"] / max(1, conflict["A4e"]))))
    print("  A4c lever  repair DRAM rows over all turns: A3b %d  A4c %d  (%.1f%% fewer)" % (
        repairs["A3b"], repairs["A4c"], 100 * (1 - repairs["A4c"] / max(1, repairs["A3b"]))))
    for rung, h in headroom.items():
        print("  headroom %s  busiest lane DRAM rows %d: packed min %d (fragmentation %.1f%%), mean lane %.0f (imbalance %.1f%%), balanced+packed %.0f (total %.1f%%)" % (
            rung, h["max"], h["packed_max"], 100 * (1 - h["packed_max"] / max(1, h["max"])), h["mean"],
            100 * (1 - h["mean"] / max(1, h["max"])), h["ideal"], 100 * (1 - h["ideal"] / max(1, h["max"]))))
    print("  A5/A6 lever  turns estimated PIM-side %d, GPU-side %d; owner ingest m=%d: GPU %.1f ms vs bank %.1f ms" % (
        len(pim), len(gpu), owner[1], owner[4] * 1e3, owner[5] * 1e3))
    chatty = [t for t in turns if t[1] <= 64 and t[0] != "a0_owner"]
    if chatty:
        t = chatty[-1]
        print("  example chatty turn %s: m=%d n=%d resident=%d  t_xpu %.0f us  t_bank %.0f us" % (
            t[0], t[1], t[2], t[3], t[4] * 1e6, t[5] * 1e6))
    writer = [t for t in turns if 64 < t[1] < 4000]
    if writer:
        t = writer[-1]
        print("  example writer turn %s: m=%d n=%d resident=%d  t_xpu %.0f us  t_bank %.0f us" % (
            t[0], t[1], t[2], t[3], t[4] * 1e6, t[5] * 1e6))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        analyse(path)

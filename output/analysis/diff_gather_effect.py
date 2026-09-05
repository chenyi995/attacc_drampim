#!/usr/bin/env python3
"""Does a dedicated diff pool beat leaving the repairs inline?  One variable.

    PYTHONPATH=$PWD python3 output/analysis/diff_gather_effect.py <workload.json> [-k 8]

THE ONE VARIABLE.  Both cases use the SAME allocator (``CacheBlendTLB``), the
SAME workload, the SAME reuse plan, the SAME append order and the SAME channel
striping.  The only thing that differs is where a correction is written:

  inline (A3b)   the repair is appended into the ordinary pool, right where the
                 software wrote it -- between whatever else that agent was
                 writing at the time.
  pool   (A4/A4b) the repair goes to the dedicated diff channel instead.

Nothing else is touched, and neither case is handed a penalty.  An earlier
version of this script compared ``NaiveKVLayout`` against ``CacheBlendTLB``,
which is NOT the same experiment: the naive layout pages EVERY block --
master included -- round-robin over the sixteen channels, so its number mixed
paged rotation together with the absence of a pool.  chenyi9 caught that; the
inline layout can perfectly well place two repairs adjacently when nothing was
written between them, and this version lets it.

Why the repairs end up apart at all, with no rotation to blame: a consumer
interleaves its repairs with its OWN fresh KV, segment by segment, exactly as
the software emits them.  Round one's repair, then a fresh chunk, then round
two's repair.  That is the multi-round structure, and it is what a dedicated
pool absorbs -- the pool sees only repairs, so consecutive repairs stay
consecutive however much other traffic separated them in time.

Counted, per consumer, over its own repairs:

  runs   maximal physically-contiguous groups (``tlb.scan_runs``) -- separate
         command streams, and what the TLB descriptor cost is charged per.
  rows   DRAM rows those runs occupy IN RAMULATOR'S ADDRESS SPACE, the only
         one that decides activations: MAC_AB broadcasts over 16 partitions,
         so a token costs 4 B and a 1024-B row holds 256 of them.  A run of n
         contiguous tokens is one row-aligned extent costing ceil(n/256) rows.
         (Not the TLB's own space, which is 256 B/token -- four tokens to a
         row, where gathering cannot save anything by construction.)

Reported as two separate claims:

  multi-round  one consumer's own repairs, spread over the rounds it patched.
  multi-head   the scan runs on heads_per_hbm heads.  Inline, each head keeps
               its own copy on its own channels; pooled, every head's repairs
               share the diff channel's rows.
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.workload import build_reuse_plan, load_workload      # noqa: E402
from src.workload_runner import (CacheBlendTLB,               # noqa: E402
                                 _prepare_cacheblend_tlb,
                                 _cacheblend_tlb_rows,
                                 _parent_output_fingerprints)

TOKENS_PER_ROW = 256            # 1024-B row at 4 B per token


class InlineDiffTLB(CacheBlendTLB):
    """Same allocator, no dedicated diff pool: a repair is written inline.

    ``reserve`` and ``locate`` relabel ``diff`` as ``master`` so the repair
    lands in the ordinary pool at the position the software appended it.  The
    KVLocation still reports ``kind='diff'`` so callers see the same read
    list; only the ADDRESS changes, which is the whole point.
    """

    _kv_mapping = "master-diff"          # same policy routing as the pooled case

    def reserve(self, layer, owner, fingerprint, owner_row, kind):
        return super().reserve(layer, owner, fingerprint, owner_row,
                               "master" if kind == "diff" else kind)

    def locate(self, layer, owner, fingerprint, owner_row, kind):
        location = super().locate(layer, owner, fingerprint, owner_row,
                                  "master" if kind == "diff" else kind)
        if kind == "diff":
            from dataclasses import replace
            location = replace(location, kind="diff")
        return location


def footprint(tlb, bindings, heads):
    """(#repair tokens, #runs, rows for one head, rows folded over heads)."""
    diff = [location for _, _, _, location in bindings
            if location.kind == "diff"]
    if not diff:
        return 0, 0, 0, 0
    runs = tlb.scan_runs(diff)
    rows_one_head = sum(-(-count // TOKENS_PER_ROW)
                        for _k, _v, count, _cb, _cc in runs)
    if len(runs) == 1:
        # one contiguous extent: every head's copy shares the same rows
        rows_all_heads = -(-(heads * len(diff)) // TOKENS_PER_ROW)
    else:
        # scattered: each head carries its own copy on its own channels
        rows_all_heads = heads * rows_one_head
    return len(diff), len(runs), rows_one_head, rows_all_heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--heads", type=int, default=8,
                        help="heads_per_hbm (LLAMA3-8B at --num-hbm 1 is 8)")
    parser.add_argument("--canonical", action="store_true",
                        help="canonical prefix corrections instead of scattered")
    args = parser.parse_args()

    workload = load_workload(args.workload)
    # ONE plan for both cases.  The correction positions are a property of the
    # reuse policy, not of the layout, so varying them here would smuggle in a
    # second variable.
    plan = build_reuse_plan(workload, "recompute",
                            epic_prefix_recompute_tokens=args.k,
                            recompute_canonical=args.canonical)

    cases = [("inline (A3b: repair written where the software put it)",
              InlineDiffTLB(256, "table")),
             ("pool   (A4/A4b: repair written to the diff channel)",
              CacheBlendTLB(256, "table"))]

    results = {}
    for label, tlb in cases:
        _prepare_cacheblend_tlb(workload, plan, 1, tlb,
                                _parent_output_fingerprints(workload))
        results[label] = {
            request.request_id: footprint(
                tlb, _cacheblend_tlb_rows(workload, plan, args.layer, request,
                                          tlb), args.heads)
            for request in workload.requests}

    labels = list(results)
    print("# diff placement, one variable.  layer {}, k={}, "
          "corrections {}".format(args.layer, args.k,
                                  "canonical prefix" if args.canonical
                                  else "scattered"))
    print("# workload:", os.path.basename(args.workload))
    print("# heads_per_hbm =", args.heads)
    print()
    print("| consumer | repair tokens | inline runs | inline rows x{h} | "
          "pool runs | pool rows x{h} |".format(h=args.heads))
    print("|---|---:|---:|---:|---:|---:|")
    totals = collections.defaultdict(lambda: [0, 0, 0, 0])
    shown = 0
    for request in workload.requests:
        rid = request.request_id
        a, b = results[labels[0]][rid], results[labels[1]][rid]
        for label, value in ((labels[0], a), (labels[1], b)):
            for index in range(4):
                totals[label][index] += value[index]
        if a[0] and shown < 6:
            print("| {} | {} | {} | {} | {} | {} |".format(
                rid, a[0], a[1], a[3], b[1], b[3]))
            shown += 1
    print()
    print("| total | repair tokens | runs | rows (1 head) | rows (x{}) |"
          .format(args.heads))
    print("|---|---:|---:|---:|---:|")
    for label in labels:
        t = totals[label]
        print("| {} | {} | {} | {} | {} |".format(label, t[0], t[1], t[2], t[3]))
    a, b = totals[labels[0]], totals[labels[1]]
    print()
    if b[1]:
        print("multi-round: runs {} -> {}  ({:.1f}x);  "
              "activations on one head {} -> {}  ({:.1f}x)"
              .format(a[1], b[1], a[1] / b[1], a[2], b[2],
                      a[2] / b[2] if b[2] else float("nan")))
    if b[3]:
        print("multi-head on top: {} -> {}  ({:.1f}x)  at heads={}"
              .format(a[3], b[3], a[3] / b[3], args.heads))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""LEGACY (2026-09-05): this rewrite models the per-scan synthetic placement of
A3b/A4/A4b as it stood on 2026-09-03.  Since commit b4f57ce the ladder rungs
place every object at write time in a persistent physical ledger
(PhysicalLedger), so this hand model no longer describes A3b/A4c/A4e and
must not be used as evidence for them.

Hand-calculation vs the engine, per scan (chenyi9 order 2026-09-04).

    python3 output/analysis/layout_handcheck_theory.py <dump_dir> [--rungs A3b,A4,...]

Reads the layout-probe dumps and, for every recorded scan, re-derives the
placement FROM THE WRITTEN RULE -- not by calling the engine.  That is the
point: importing ``_striped_append_channel_extents`` would check the engine
against itself.  The rules below are transcribed from the docstrings and the
2026-09-03 ruling, and a disagreement means either the code or this
transcription is wrong.

Geometry (settled 2026-09-04, see src/layout_probe.py):
  1 token   = 4 B of address space  (MAC_AB broadcasts over 16 partitions)
  1 column  = 32 B  = 8 tokens
  1 DRAM row= 1024 B = 256 tokens   = one stripe unit
Each slot placed in a channel starts on a fresh row, so a slot of n tokens
costs ceil(n/256) activations however few tokens it holds.

The rules
---------
``slice-append`` (A3b) -- no master/diff pool, 16 channels:
    stripe = max(1, 16 // heads);  head h starts at (h*stripe) % 16
    the master stream is cut into 256-token units; unit u of head h goes to
    (base + u % stripe) % 16.  A head's repairs are ONE contiguous append
    (ruling 2026-09-03: that head's prefill writes them together), landing on
    the next channel of its own rotation.  Repairs are NOT shared across heads.

``master-diff-slice-append`` (A4) -- master pool ch0..14, diff pool ch15:
    stripe_m = max(1, 15 // heads);  same rotation but over 15 channels.
    EVERY head's repairs are packed into ONE extent on ch15.

``master-diff-table-append`` (A4b, A5, A6) -- same pools, different master
    distribution: a single slot counter walks (head, unit) pairs and drops
    each on slot % 15, so co-read chunks land on different channels.
"""

import argparse
import json
import os
import sys

UNIT = 256                      # tokens per stripe unit = tokens per DRAM row
CHANNELS = 16
MASTER_CHANNELS = 15


def cut_units(tokens):
    """The append stream cut at unit boundaries: full units then a tail."""
    if tokens <= 0:
        return []
    full, tail = divmod(int(tokens), UNIT)
    units = [UNIT] * full
    if tail:
        units.append(tail)
    return units


def acts(tokens):
    return -(-int(tokens) * 4 // 1024)


def place(policy, heads, master_tokens, repair_runs):
    """Return {channel: [slot tokens, ...]} by the rule, in insertion order."""
    per = {}

    def add(channel, tokens):
        if tokens > 0:
            per.setdefault(channel, []).append(int(tokens))

    units = cut_units(master_tokens)
    repairs = sum(repair_runs)
    if policy == "slice-append":
        stripe = max(1, CHANNELS // heads)
        for head in range(heads):
            base = (head * stripe) % CHANNELS
            for index, rows in enumerate(units):
                add((base + (index % stripe)) % CHANNELS, rows)
            if repairs:
                add((base + (len(units) % stripe)) % CHANNELS, repairs)
    elif policy == "master-diff-slice-append":
        stripe_m = max(1, MASTER_CHANNELS // heads)
        for head in range(heads):
            base = (head * stripe_m) % MASTER_CHANNELS
            for index, rows in enumerate(units):
                add((base + (index % stripe_m)) % MASTER_CHANNELS, rows)
        if repairs:
            add(MASTER_CHANNELS, heads * repairs)
    elif policy == "master-diff-table-append":
        slot = 0
        for _head in range(heads):
            for rows in units:
                add(slot % MASTER_CHANNELS, rows)
                slot += 1
        if repairs:
            add(MASTER_CHANNELS, heads * repairs)
    else:
        raise SystemExit("unknown policy " + policy)
    return per


def measured_slots(scan):
    """{channel: [slot tokens, ...]} as the engine handed them to Ramulator."""
    return {int(channel): [entry["tokens"] for entry in entries]
            for channel, entries in scan.get("extents", {}).items()}


def check(scan):
    theory = place(scan["policy"], scan["heads_per_hbm"],
                   scan.get("read_extents_master_tokens",
                            scan["reads_master"]),
                   scan.get("read_extents_diff_runs", []))
    engine = measured_slots(scan)
    agree = theory == engine
    t_acts = sum(acts(n) for slots in theory.values() for n in slots)
    e_acts = sum(acts(n) for slots in engine.values() for n in slots)
    return agree, theory, engine, t_acts, e_acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir")
    parser.add_argument("--rungs", default="A3b,A4,A4b,A5,A6")
    parser.add_argument("--show", type=int, default=2,
                        help="print this many disagreeing scans in full")
    args = parser.parse_args()

    print("| rung | scans | agree | disagree | ACT theory | ACT engine |")
    print("|---|---:|---:|---:|---:|---:|")
    failures = []
    for rung in args.rungs.split(","):
        path = os.path.join(args.dump_dir, "layout_{}.jsonl".format(rung))
        if not os.path.exists(path):
            print("| {} | (no dump) | | | | |".format(rung))
            continue
        total = ok = 0
        t_sum = e_sum = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("kind") != "scan":
                    continue
                if not record.get("extents"):
                    # legacy chunk-count path (A1/A3/A3a): no extents are
                    # built, so there is nothing of this shape to check
                    continue
                if ("read_extents_master_tokens" not in record
                        and record.get("reads_diff", 0)):
                    # an older dump without the extent cut: only checkable
                    # when the scan carries no corrections, since the repair
                    # runs are then known to be empty
                    continue
                total += 1
                agree, theory, engine, t_acts, e_acts = check(record)
                t_sum += t_acts
                e_sum += e_acts
                if agree:
                    ok += 1
                elif len(failures) < args.show:
                    failures.append((rung, record, theory, engine))
        print("| {} | {} | {} | {} | {} | {} |".format(
            rung, total, ok, total - ok, t_sum, e_sum))
    for rung, record, theory, engine in failures:
        print()
        print("### disagreement in {} ({}, {} master + {} diff tokens)".format(
            rung, record["request"], record["reads_master"],
            record["reads_diff"]))
        print("  theory:", dict(sorted(theory.items())))
        print("  engine:", dict(sorted(engine.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

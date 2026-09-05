#!/usr/bin/env python3
"""Print the A3b / A4b layouts as a grid: one column per channel, one line per
DRAM row.

    PYTHONPATH=$PWD python3 output/analysis/layout_grid_csv.py [-o DIR]

The smallest example that shows the difference, and nothing else in it:

    heads_per_hbm = 4        four heads share this HBM's sixteen channels
    4 shared chunks          each 256 tokens = exactly one DRAM row
    k = 8                    eight tokens of each chunk are recomputed

Every head reads the same four chunks and carries its own four repairs.  So
per head: 4 x 256 master tokens and 4 x 8 repair tokens.

Geometry, which is all a reader needs to check the grid: MAC_AB broadcasts
over 16 partitions, so a token costs 4 B of address space and a 1024-B DRAM
row holds 256 tokens.  Every extent starts on a fresh row, so an 8-token
repair occupies a whole row and leaves 248 tokens of it empty.

The placement rules, transcribed (they reproduce the engine cell for cell --
checked against 2000 recorded scans by layout_handcheck_theory.py):

  A3b  slice-append.  stripe = 16 // heads = 4.  Head h starts at channel 4h
       and its units round-robin its own four channels.  There is no pool, so
       each repair is appended on the head's own channels too -- and since the
       rounds are separated in time, a repair cannot share a row with the next.

  A4b  master-diff-table-append.  The master pool is ch0..14 and a single slot
       counter walks (head, unit) pairs onto slot % 15.  Every head's repairs
       go to ch15 instead, where they are one packed append and share rows.

Writes two CSVs plus a readable text grid of each.
"""

import argparse
import csv
import os

CHANNELS = 16
MASTER_CHANNELS = 15
UNIT = 256                      # tokens per DRAM row
HEADS = 4
CHUNKS = 4                      # shared chunks the agent reads
ROUNDS = 4                      # separate repair appends (one per round)
K = 8


def a3b():
    """{channel: [(label, tokens), ...]} in placement order.

    A ROUND'S repairs are ONE contiguous append (ruling chenyi9 2026-09-03,
    corrected again 2026-09-04): that round's prefill produces them together
    and writes them back to back, so they PACK -- ceil(tokens/256) rows for
    the whole round, not one row per repaired chunk.  What A3b cannot do is
    pack ACROSS rounds (the agent wrote other KV in between) or ACROSS heads
    (different channels).  Those two are exactly what the diff pool buys.
    """
    per = {}
    stripe = CHANNELS // HEADS
    per_round = max(1, CHUNKS // ROUNDS)           # chunks repaired each round
    for head in range(HEADS):
        base = (head * stripe) % CHANNELS
        for unit in range(CHUNKS):                 # one 256-token unit per chunk
            channel = (base + unit % stripe) % CHANNELS
            per.setdefault(channel, []).append(
                ("h%d-chunk%d" % (head, unit), UNIT))
        for index in range(ROUNDS):
            channel = (base + (CHUNKS + index) % stripe) % CHANNELS
            per.setdefault(channel, []).append(
                ("h%d-round%d-repairs" % (head, index), per_round * K))
    return per


def a4b():
    per = {}
    slot = 0
    for head in range(HEADS):
        for unit in range(CHUNKS):
            per.setdefault(slot % MASTER_CHANNELS, []).append(
                ("h%d-chunk%d" % (head, unit), UNIT))
            slot += 1
    # every head's repairs, packed, on the diff channel
    per_round = max(1, CHUNKS // ROUNDS)
    per.setdefault(MASTER_CHANNELS, []).append(
        ("all-heads-all-rounds-repairs", HEADS * ROUNDS * per_round * K))
    return per


def a4c():
    """A4c: master exactly as A3b; the head's repairs gathered on ONE of its
    own channels as a single contiguous extent."""
    per = {}
    stripe = CHANNELS // HEADS
    per_round = max(1, CHUNKS // ROUNDS)
    for head in range(HEADS):
        base = (head * stripe) % CHANNELS
        for unit in range(CHUNKS):
            per.setdefault((base + unit % stripe) % CHANNELS, []).append(
                ("h%d-chunk%d" % (head, unit), UNIT))
        per.setdefault((base + stripe - 1) % CHANNELS, []).append(
            ("h%d-ALL-rounds-repairs" % head, ROUNDS * per_round * K))
    return per


def grid(per):
    """[[cell per channel] per DRAM row].  One slot may span several rows."""
    columns = []
    for channel in range(CHANNELS):
        cells = []
        for label, tokens in per.get(channel, []):
            rows = -(-tokens // UNIT)
            for index in range(rows):
                used = min(UNIT, tokens - index * UNIT)
                cells.append("%s (%d/%d tok)" % (label, used, UNIT))
        columns.append(cells)
    depth = max((len(column) for column in columns), default=0)
    return [[column[row] if row < len(column) else ""
             for column in columns] for row in range(depth)]


def write(name, per, out_dir):
    rows = grid(per)
    path = os.path.join(out_dir, name + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dram_row"] + ["ch%d" % c for c in range(CHANNELS)])
        for index, row in enumerate(rows):
            writer.writerow([index] + row)
    acts = sum(1 for row in rows for cell in row if cell)
    busiest = max((sum(1 for row in rows if row[c]) for c in range(CHANNELS)),
                  default=0)
    active = sum(1 for c in range(CHANNELS) if any(row[c] for row in rows))
    print("%s -> %s" % (name, path))
    print("   rows used (= activations): %d" % acts)
    print("   busiest channel: %d rows      active channels: %d"
          % (busiest, active))
    print()
    header = "row |" + "".join(" %-22s|" % ("ch%d" % c) for c in range(CHANNELS))
    print(header)
    for index, row in enumerate(rows):
        print("%3d |" % index
              + "".join(" %-22s|" % (cell or "-") for cell in row))
    print()
    return acts, busiest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default="output/analysis")
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    globals()["CHUNKS"] = args.chunks
    globals()["ROUNDS"] = args.rounds
    os.makedirs(args.out_dir, exist_ok=True)
    print("heads_per_hbm=%d  chunks=%d (256 tokens each)  rounds=%d  k=%d\n"
          % (HEADS, args.chunks, args.rounds, K))
    a_acts, a_busy = write("layout_A3b" + args.tag, a3b(), args.out_dir)
    b_acts, b_busy = write("layout_A4b" + args.tag, a4b(), args.out_dir)
    c_acts, c_busy = write("layout_A4c" + args.tag, a4c(), args.out_dir)
    print("A3b: %d activations, busiest channel %d rows" % (a_acts, a_busy))
    print("A4b: %d activations, busiest channel %d rows" % (b_acts, b_busy))
    print("A4c: %d activations, busiest channel %d rows" % (c_acts, c_busy))
    print("activations %.2fx fewer" % (a_acts / b_acts))
    if a_busy == b_busy:
        print("busiest channel UNCHANGED (%d rows): the pool saves activations "
              "and energy, not latency -- scan time is the max over channels."
              % a_busy)
    else:
        print("busiest channel %d -> %d rows (%.2fx): the pool saves LATENCY too."
              % (a_busy, b_busy, a_busy / b_busy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

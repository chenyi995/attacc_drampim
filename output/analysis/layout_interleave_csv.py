#!/usr/bin/env python3
"""One KV head, four channels: what agent 2 has to activate (chenyi9 example).

    PYTHONPATH=$PWD python3 output/analysis/layout_interleave_csv.py [-o DIR]

The append stream is SHARED by the agents, and they write into it as they go.
This is the sequence chenyi9 gave, in time order:

    chunk1  chunk2          the shared corpus, written first
    diff1   diff2           agent 1's repairs
    diff1b  diff2b          agent 2's repairs, right after
    chunk3                  agent 1's own new chunk
    chunk4  diff3           agent 2's own new chunk, and one more repair

Nothing is out of order and nothing is a penalty: an allocator cannot place a
repair for an agent that has not run yet, so agent 1's and agent 2's repairs
necessarily land between each other's.

Then agent 2 attends.  It reads chunk1, chunk2, diff1b, diff2b, chunk4, diff3
and skips everything belonging to agent 1.  The question is how many DRAM rows
it has to open.

Geometry: a token costs 4 B (MAC_AB broadcasts over 16 partitions), so a
1024-B row holds 256 tokens.  A chunk is 256 tokens = a full row.  A repair is
k=8 tokens, so 32 repairs fit in ONE row -- which is the whole point below.

  A3b   no pool.  One KV head owns four channels and EVERYTHING -- chunks and
        repairs alike -- goes into that one append stream, striped over the
        four.  A repair therefore sits in a row of its own, and the next
        agent's repair lands in the next row.
  A4b   master pool + diff channel.  Chunks stripe over the master channels;
        every repair goes to the diff channel instead.  Agent 1's and agent 2's
        repairs still interleave there -- the allocator still cannot see the
        future -- but at 8 tokens each they share a row, so the interleaving
        costs nothing.
"""

import argparse
import csv
import os

CHUNK = 256                     # tokens, = one DRAM row
K = 8                           # tokens per repair
ROW = 256                       # tokens per DRAM row

# APPEND ORDER.  Each entry is ONE contiguous append: a group of repairs
# produced by the same agent in the same round is written back to back, so it
# is one extent and shares a row -- it must NOT be spread one-per-channel
# (correction chenyi9 2026-09-04).  What cannot share is a different ROUND
# (the agent wrote other KV in between) or a different AGENT (they take turns).
STREAM = [
    (["chunk1"], "shared", CHUNK),          # the shared corpus
    (["chunk2"], "shared", CHUNK),
    (["diff1", "diff2"], "agent1", 2 * K),  # agent 1, round 1: ONE append
    (["diff1b", "diff2b"], "agent2", 2 * K),  # agent 2, round 1: ONE append
    (["chunk3"], "agent1", CHUNK),          # round 2
    (["chunk4"], "agent2", CHUNK),
    (["diff3"], "agent2", K),               # agent 2, round 2
]
AGENT2 = {"chunk1", "chunk2", "diff1b", "diff2b", "chunk4", "diff3"}


def place_a3b(channels=4):
    """One stream over the head's channels; every extent row-aligned."""
    per = {index: [] for index in range(channels)}
    slot = 0
    for labels, owner, tokens in STREAM:
        per[slot % channels].append((labels, owner, tokens))
        slot += 1
    return per, list(range(channels)), None


def place_a4b(master=3, diff_channel=3):
    """Chunks over the master channels, every repair packed on the diff channel."""
    per = {index: [] for index in range(master + 1)}
    slot = 0
    packed = []
    for labels, owner, tokens in STREAM:
        if tokens < CHUNK:                  # a repair append
            packed.append((labels, owner, tokens))
        else:
            per[slot % master].append((labels, owner, tokens))
            slot += 1
    # the diff channel is one contiguous append: consecutive repairs share rows
    per[diff_channel] = packed
    return per, list(range(master)), diff_channel


def rows_of(per, diff_channel):
    """{channel: [ [ (label, owner, tokens) ... ] per DRAM row ]}."""
    out = {}
    for channel, entries in per.items():
        rows = []
        if channel == diff_channel:
            # packed append: fill a row to 256 tokens before starting the next
            used = 0
            for entry in entries:
                if not rows or used + entry[2] > ROW:
                    rows.append([])
                    used = 0
                rows[-1].append(entry)
                used += entry[2]
        else:
            # every extent starts on a fresh row
            for entry in entries:
                span = -(-entry[2] // ROW)
                for _ in range(span):
                    rows.append([entry])
        out[channel] = rows
    return out


def render(name, per, diff_channel, out_dir):
    table = rows_of(per, diff_channel)
    channels = sorted(table)
    depth = max((len(table[c]) for c in channels), default=0)
    path = os.path.join(out_dir, name + ".csv")
    acts = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dram_row"]
                        + ["ch%d%s" % (c, " (diff)" if c == diff_channel else "")
                           for c in channels]
                        + ["agent2_must_activate"])
        for index in range(depth):
            cells = []
            hit = []
            for channel in channels:
                rows = table[channel]
                if index < len(rows):
                    entries = rows[index]
                    cells.append(" + ".join(
                        "%s[%s,%dtok]" % ("+".join(labels), owner, tokens)
                        for labels, owner, tokens in entries))
                    if any(l in AGENT2 for labels, _o, _t in entries
                           for l in labels):
                        hit.append("ch%d" % channel)
                else:
                    cells.append("")
            acts += len(hit)
            writer.writerow([index] + cells + [" ".join(hit)])
    per_channel = {c: sum(1 for index in range(len(table[c]))
                          if any(l in AGENT2
                                 for labels, _o, _t in table[c][index]
                                 for l in labels))
                   for c in channels}
    busiest = max(per_channel.values()) if per_channel else 0
    print("%s -> %s" % (name, path))
    header = "row |" + "".join(" %-34s|" % ("ch%d%s" % (c, "(diff)" if c == diff_channel else ""))
                               for c in channels) + " agent2 opens"
    print(header)
    for index in range(depth):
        line = "%3d |" % index
        hit = []
        for channel in channels:
            rows = table[channel]
            if index < len(rows):
                text = " + ".join("%s[%s]" % ("+".join(labels), owner)
                                  for labels, owner, _t in rows[index])
                line += " %-34s|" % text
                if any(l in AGENT2 for labels, _o, _t in rows[index]
                       for l in labels):
                    hit.append("ch%d" % channel)
            else:
                line += " %-34s|" % "-"
        print(line + " " + (" ".join(hit) if hit else "-"))
    print("   agent 2 activations: %d   busiest channel: %d rows\n"
          % (acts, busiest))
    return acts, busiest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default="output/analysis")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print("one HBM, 4 KV heads -> this is ONE head's 4 channels.  "
          "chunk=256 tok, repair k=8 tok, row=256 tok\n")
    per, _m, _d = place_a3b()
    a_acts, a_busy = render("layout_interleave_A3b", per, None, args.out_dir)
    per, _m, diff = place_a4b()
    b_acts, b_busy = render("layout_interleave_A4b", per, diff, args.out_dir)
    print("agent 2 activations: A3b %d -> A4b %d  (%.2fx)"
          % (a_acts, b_acts, a_acts / b_acts))
    print("busiest channel:     A3b %d -> A4b %d  (%.2fx)"
          % (a_busy, b_busy, a_busy / b_busy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

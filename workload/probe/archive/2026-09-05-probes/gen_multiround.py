#!/usr/bin/env python3
"""Build the smallest workload that shows the multi-round diff structure.

    python3 workload/handcheck/gen_multiround.py > workload/handcheck/wl_multiround.json

WHAT IT ADDS, and it is exactly one thing.  The swept workload gives a
consumer a context shaped like

    sys | own own own ... | reused reused reused ...

so its repairs are produced in ONE burst at the end and are contiguous
whether or not a diff pool exists -- there is nothing for a pool to gather.
A multi-round agent does not look like that.  It retrieves something, works,
retrieves the next thing, works again, so its context alternates:

    sys | reused_0 | own_0 | reused_1 | own_1 | reused_2 | own_2 | ...

Each reused block is position-shifted (the consumer's offsets differ from the
producer's), so each takes k recomputed rows -- and those repairs are now
separated by the agent's OWN fresh KV from the round in between.  That is the
multi-round structure, and it is the only difference from the swept workload.

Repairs stay consumer-private, as they are in the engine: agent 1's diff is
its own and agent 2's diff is its own; nothing here shares them.

Everything else is kept boring on purpose: 256-token blocks (one DRAM row at
4 B/token), one producer, two consumers, short outputs.  Small enough to check
the whole layout by hand.
"""

import hashlib
import json
import os
import sys

BLOCK = 256                     # tokens per block = one DRAM row
ROUNDS = int(os.environ.get("ROUNDS", "4"))          # retrieve/work rounds
CHUNKS_PER_ROUND = int(os.environ.get("CHUNKS", "1"))
CONSUMERS = int(os.environ.get("CONSUMERS", "2"))
LOUT = int(os.environ.get("LOUT", str(BLOCK)))   # generated tokens per agent

# CHUNKS_PER_ROUND > 1 is the reuse-heavy case chenyi9 asked for: a round
# reads several shared chunks (a chunk of a large codebase, say) and writes
# only one block of its own.  Reuse then dominates the context instead of
# alternating one-for-one with fresh KV, while the repairs of different
# rounds stay separated by that round's own output -- which is what keeps
# this a multi-round test.


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    shared = [{"role": "doc", "sha": sha("shared-%d" % index), "len": BLOCK}
              for index in range(ROUNDS * CHUNKS_PER_ROUND)]

    agents = []
    # The producer declares the shared corpus.  Ownership goes to the first
    # request in sorted (tier, id) order, so this one owns every block.
    agents.append({
        "id": "a0_owner", "tier": 0, "parent": None, "history_len": 0, "lout": LOUT,
        # a pad in front, so the consumer's copies sit at different offsets
        # and every reused block is position-shifted (-> k repaired rows)
        "segs": [{"role": "sys", "sha": sha("a0-sys"), "len": BLOCK}] + shared,
    })

    for index in range(CONSUMERS):
        rid = "b%d_reuser" % index
        segs = [{"role": "sys", "sha": sha(rid + "-sys"), "len": BLOCK}]
        for round_index in range(ROUNDS):
            # round r: read this round's retrieved chunks, then write the one
            # block of KV this round generated
            for offset in range(CHUNKS_PER_ROUND):
                segs.append(dict(shared[round_index * CHUNKS_PER_ROUND + offset]))
            segs.append({"role": "user",
                         "sha": sha("%s-own-%d" % (rid, round_index)),
                         "len": BLOCK})
        # Same tier, no declared parent: ownership follows sorted (tier, id),
        # so "a0_owner" declares the blocks and the "b*_reuser" agents reuse them.  A tier-1
        # agent would have to carry a parent_out segment, which is a second
        # mechanism this probe does not need.
        agents.append({"id": rid, "tier": 0, "parent": None,
                       "history_len": 0, "lout": LOUT, "segs": segs})

    workload = {
        "meta": {
            "format": "v2-dag",
            "kind": "multi-round diff placement probe -- NOT evidence-grade",
            "block_tokens": BLOCK,
            "rounds": ROUNDS,
            "lout": LOUT,
            "chunks_per_round": CHUNKS_PER_ROUND,
            "note": ("A consumer's context ALTERNATES reused block and own "
                     "fresh block, one pair per round, so its k repaired rows "
                     "are separated by the KV it wrote in between.  That is "
                     "the only difference from the swept workload, whose "
                     "consumers emit all their repairs in one burst.  Repairs "
                     "stay consumer-private."),
        },
        "agents": agents,
    }
    json.dump(workload, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

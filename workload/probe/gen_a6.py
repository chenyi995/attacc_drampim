#!/usr/bin/env python3
"""Multi-round workload whose agents differ ONLY in how much fresh KV they
write per round -- the axis the A6 prefill chooser is sensitive to.

Owner declares the shared corpus; every reuser reads the same ROUNDS x CHUNKS
shared chunks in the same order (same context, same repairs).  Even reusers
write OWN_SHORT fresh tokens per round ("ask a short question"), odd reusers
write OWN_LONG ("write a long answer").  The chooser prices, per request,
    t_xpu  = readback of the resident rows + one GPU block of m x n
    t_bank = ceil(m / 8) sweeps, each streaming the whole n-row context
so a short-fresh agent (m small against n) is where the GPU can win and a
long-fresh agent is where the banks win.  Nothing else differs.

    ROUNDS=8 CHUNKS=2 CONSUMERS=8 OWN_SHORT=16 OWN_LONG=256 LOUT=128 python3 gen_a6.py > wl.json
"""
import hashlib
import json
import os
import sys

BLOCK = 256
ROUNDS = int(os.environ.get("ROUNDS", "8"))
CHUNKS_PER_ROUND = int(os.environ.get("CHUNKS", "2"))
CONSUMERS = int(os.environ.get("CONSUMERS", "8"))
OWN_SHORT = int(os.environ.get("OWN_SHORT", "16"))
OWN_LONG = int(os.environ.get("OWN_LONG", "256"))
LOUT = int(os.environ.get("LOUT", "128"))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    shared = [{"role": "doc", "sha": sha("shared-%d" % index), "len": BLOCK}
              for index in range(ROUNDS * CHUNKS_PER_ROUND)]
    agents = [{
        "id": "a0_owner", "tier": 0, "parent": None, "history_len": 0, "lout": LOUT,
        "segs": [{"role": "sys", "sha": sha("a0-sys"), "len": BLOCK}] + shared,
    }]
    for index in range(CONSUMERS):
        own = OWN_SHORT if index % 2 == 0 else OWN_LONG
        rid = "b%d_%s" % (index, "shortfresh" if own == OWN_SHORT else "longfresh")
        segs = [{"role": "sys", "sha": sha(rid + "-sys"), "len": BLOCK}]
        for round_index in range(ROUNDS):
            for offset in range(CHUNKS_PER_ROUND):
                segs.append(dict(shared[round_index * CHUNKS_PER_ROUND + offset]))
            segs.append({"role": "user", "sha": sha("%s-own-%d" % (rid, round_index)),
                         "len": own})
        agents.append({"id": rid, "tier": 0, "parent": None, "history_len": 0,
                       "lout": LOUT, "segs": segs})
    json.dump({"meta": {"format": "v2-dag",
                        "kind": "A6 probe: agents differ only in fresh tokens per round",
                        "block_tokens": BLOCK, "rounds": ROUNDS,
                        "chunks_per_round": CHUNKS_PER_ROUND,
                        "own_short": OWN_SHORT, "own_long": OWN_LONG, "lout": LOUT},
               "agents": agents}, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

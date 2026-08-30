#!/usr/bin/env python3
"""gen_sweep.py -- one parametric generator for the whole workload sweep.

Builds a tiered DAG of nodes from (topology, N, C, D). Every node:
  segs = [sys(16)] + [ALL previous-tier outputs] + [C shared 256-token blocks]
  lout = one 256-token block
  history_len = t * 256  (its own prior outputs; small, append-scattered,
                          so depth D never fills the 32,768-token K cap)
The shared corpus is C blocks with STABLE sha -> reused across nodes at
position-shifted offsets (different preambles). `--private` gives each node
its own corpus (no-reuse control). k (recompute per block) is a RUN-time
policy (EPIC_K / --reuse recompute), not encoded here.

Topologies (tier width sequence; the fan on each edge follows from the widths
because every node reads ALL previous-tier outputs):
  broadcast  [1, N]        1 source, N consumers      (fan-out 1->N)
  reduce     [N, 1]        N producers, 1 reducer      (fan-in  N->1)
  alltoall   [N]*D         D tiers of N, all-to-all    (N->N)
  supervisor [1,N,1,N,...] hub<->workers, D tiers      (fan-out + fan-in)
  pipeline   [1]*D         chain                        (1->1)

    python3 gen_sweep.py --topology supervisor --N 16 --C 32 --D 2
"""
import argparse, hashlib, json
from pathlib import Path

BLOCK = 256
SYS = 16
CAP = 128  # blocks (32,768 token = 8-MiB K partition)


def sha16(s):
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def widths(topo, N, D):
    if topo == "broadcast":
        return [1, N]
    if topo == "reduce":
        return [N, 1]
    if topo == "alltoall":
        return [N] * D
    if topo == "pipeline":
        return [1] * D
    if topo == "supervisor":
        return [1 if t % 2 == 0 else N for t in range(D)]
    raise ValueError(topo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True,
                    choices=["broadcast", "reduce", "alltoall",
                             "supervisor", "pipeline"])
    ap.add_argument("--N", type=int, default=16, help="fan degree (wide-tier width)")
    ap.add_argument("--C", type=int, default=32, help="shared context blocks")
    ap.add_argument("--D", type=int, default=2, help="number of tiers")
    ap.add_argument("--private", action="store_true",
                    help="no-reuse control: each node gets a private corpus")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.C > CAP:
        raise SystemExit(f"C={a.C} blocks exceeds the {CAP}-block (8-MiB) cap")
    W = widths(a.topology, a.N, a.D)

    def corpus(nid):
        pref = f"priv-{nid}" if a.private else "corpus"
        return [{"role": "doc", "sha": sha16(f"{pref}-blk{i}"), "len": BLOCK}
                for i in range(a.C)]

    agents = []
    prev = []                         # (id) of the previous tier
    for t, w in enumerate(W):
        cur = []
        for i in range(w):
            nid = f"t{t}n{i}"
            parent = (prev[i] if i < len(prev) else prev[0]) if prev else None
            segs = [{"role": "sys", "sha": sha16(f"sys-{nid}"), "len": SYS}]
            if t == 0:
                segs.append({"role": "user", "sha": sha16("task"), "len": BLOCK})
            else:
                # Read ALL previous-tier outputs.  The v2-dag schema allows
                # exactly ONE parent_out -- the declared data dependency; the
                # remaining upstream outputs are ordinary fingerprinted
                # segments carrying the producer's sha, so cross-request reuse
                # still resolves through them.  Same convention as the retired
                # gen_debate.py / gen_mapreduce_sum.py peer reads.
                segs.append({"role": "parent_out", "sha": sha16(f"out-{parent}"),
                             "len": BLOCK, "delta": 0})
                for pid in prev:
                    if pid != parent:
                        segs.append({"role": "user", "sha": sha16(f"out-{pid}"),
                                     "len": BLOCK})
            segs.extend(corpus(nid))   # shared (or private) C-block corpus
            agents.append({
                "id": nid, "tier": t,
                "parent": parent,
                "history_len": t * BLOCK,
                "segs": segs, "lout": BLOCK})
            cur.append(nid)
        prev = cur

    n_agents = len(agents)
    max_seg_run = a.C * BLOCK          # largest single contiguous read = corpus
    tag = "private" if a.private else a.topology
    out = a.out or str(Path(__file__).resolve().parent /
                       f"workload_sweep_{tag}_N{a.N}_C{a.C}_D{a.D}.json")
    payload = {
        "meta": {
            "format": "v2-dag",
            "kind": "parametric sweep (mechanism illustration; NOT evidence-grade)",
            "topology": a.topology, "N": a.N, "C": a.C, "D": a.D,
            "private": a.private,
            "widths": W, "n_agents": n_agents,
            "block_tokens": BLOCK, "sys_tokens": SYS,
            "max_single_segment_tokens": max_seg_run,
            "cap_tokens": CAP * BLOCK,
            "generator": "gen_sweep.py",
            "note": "corpus = C blocks with stable sha (reused, position-shifted); "
                    "history_len = own prior outputs (small, append-scattered); "
                    "recompute k set at run time via EPIC_K.",
        },
        "agents": agents,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {out}: {n_agents} agents, widths={W}, "
          f"max single segment {max_seg_run} tok ({100*max_seg_run/(CAP*BLOCK):.0f}% cap)")


if __name__ == "__main__":
    main()

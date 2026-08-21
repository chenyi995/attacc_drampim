#!/usr/bin/env python3
"""Two-request "replay" workload for the cooperative-prefill micro-benchmark.

Request 0 (owner):   sys(n_sys) + doc(L, shared sha) + query(n_new)
Request 1 (replay):  sys(n_off, unique sha) + doc(L, same sha) + query(n_new)

Both sit in tier 0; run with ``--tier-batch-size 1`` so request 0 is a
plain GPU prefill that populates the AttAcc KV (batch 0) and request 1 is the
reuse prefill under test (batch 1).  Request 1's unique ``n_off``-token sys
prefix shifts the shared doc by ``n_off`` positions, i.e. the CacheBlend
position offset the GPU has to absorb by rotating Q.  The quantity of
interest is ``tiers[1]["prefill_s"]`` of A4 (pure GPU) vs A6/A5 (GPU+PIM).

Usage: gen_replay_pair.py --L 8192 --n-new 32 [--n-off 16] [--n-sys 32]
                          [--lout 2] --out workload/replay_L8192_q32.json
"""
import argparse, hashlib, json

def sha(tag): return hashlib.sha256(tag.encode()).hexdigest()[:16]

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, required=True, help="shared doc length (tokens)")
ap.add_argument("--n-new", type=int, required=True, help="new query tokens per request")
ap.add_argument("--n-off", type=int, default=16, help="replay's unique prefix = position offset")
ap.add_argument("--n-sys", type=int, default=32, help="owner's sys prompt length")
ap.add_argument("--lout", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
doc = sha(f"shared-doc-L{a.L}")
req0 = {"sample": 0, "seg_lens": [a.n_sys, a.L, a.n_new],
        "seg_sha": [sha("sys-owner"), doc, sha("query-owner")],
        "seg_role": ["sys", "doc", "query"], "L": a.n_sys + a.L + a.n_new, "lout": a.lout}
req1 = {"sample": 1, "seg_lens": [a.n_off, a.L, a.n_new],
        "seg_sha": [sha("sys-replay-offset"), doc, sha("query-replay")],
        "seg_role": ["sys", "doc", "query"], "L": a.n_off + a.L + a.n_new, "lout": a.lout}
json.dump([req0, req1], open(a.out, "w"), indent=1)
print(a.out, "L", a.L, "n_new", a.n_new, "n_off", a.n_off)

#!/usr/bin/env python3
"""Generate a shared-corpus RAG workload (legacy list format).

The shipped ``workload_2wikimqa_first8.json`` shares only its 41-token system
prompt across samples (each 2WikiMQA question carries its own passages), so
CacheBlend/EPIC have nothing to reuse.  This script keeps that workload's
shape (sys + 10 doc chunks + query, real chunk lengths, real ``lout``) but
draws the doc chunks from one shared pool with a Zipf-like popularity, i.e.
many questions over one knowledge base (the CacheBlend paper's setting).

Reuse is still cold-cache: the first request that carries a chunk computes
it, later requests reuse it.  Chunks keep their pool rank order inside a
request, so a reused chunk usually sits at a different position than in its
owner (position-shifted reuse).

Usage: gen_shared_rag.py [--pool 24] [--samples 8] [--docs 10] [--zipf 1.0]
                         [--seed 0] [--out workload/workload_rag_shared_p24_s8.json]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "workload_2wikimqa_first8.json"),
                    help="workload whose sys prompt, chunk lengths, query lengths and lout are reused")
    ap.add_argument("--pool", type=int, default=24, help="number of unique doc chunks in the corpus")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--docs", type=int, default=10, help="retrieved chunks per request")
    ap.add_argument("--zipf", type=float, default=1.0,
                    help="popularity exponent: P(rank r) ~ 1/r^zipf (0 = uniform)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = json.load(open(args.base))
    rng = random.Random(args.seed)
    sys_len, sys_sha = base[0]["seg_lens"][0], base[0]["seg_sha"][0]
    chunk_lens = [l for s in base for l, r in zip(s["seg_lens"], s["seg_role"]) if r == "doc"]
    query_lens = [s["seg_lens"][-1] for s in base]
    louts = [s["lout"] for s in base]

    pool = [{"sha": hashlib.sha256(f"shared-doc-{i}-seed{args.seed}".encode()).hexdigest()[:16],
             "len": rng.choice(chunk_lens)} for i in range(args.pool)]
    weights = [1.0 / (r + 1) ** args.zipf for r in range(args.pool)]

    samples = []
    for i in range(args.samples):
        chosen = set()
        while len(chosen) < args.docs:
            chosen.add(rng.choices(range(args.pool), weights)[0])
        docs = [pool[r] for r in sorted(chosen)]  # rank order -> position shifts
        qlen = rng.choice(query_lens)
        seg_lens = [sys_len] + [d["len"] for d in docs] + [qlen]
        seg_sha = ([sys_sha] + [d["sha"] for d in docs] +
                   [hashlib.sha256(f"query-{i}-seed{args.seed}".encode()).hexdigest()[:16]])
        seg_role = ["sys"] + ["doc"] * len(docs) + ["query"]
        samples.append({"sample": i, "seg_lens": seg_lens, "seg_sha": seg_sha,
                        "seg_role": seg_role, "L": sum(seg_lens), "lout": louts[i % len(louts)]})

    out = Path(args.out or HERE / f"workload_rag_shared_p{args.pool}_s{args.samples}.json")
    out.write_text(json.dumps(samples, indent=1) + "\n")

    seen, reused, total = set(), 0, 0
    for s in samples:
        for sha, ln in zip(s["seg_sha"], s["seg_lens"]):
            total += ln
            if sha in seen:
                reused += ln
            seen.add(sha)
    print(f"wrote {out}: {len(samples)} samples, {total} tokens, "
          f"{len(seen)} unique segments, reused {reused} ({reused / total:.1%})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Theoretical MULTI-SOURCE RAG workload -- the fifth group (chenyi9
2026-08-26): every request takes ONE 256-token chunk from EACH of many
DIFFERENT sources ("每处拿一个 chunk,来源尽量多").

Literature basis:
  * RAG (Lewis et al., NeurIPS'20, arXiv:2005.11401): a query's context is
    assembled from top-k retrieved passages of DISTINCT documents;
  * CacheBlend (EuroSys'25, arXiv:2405.16444): exactly this chunked
    multi-document context is what selective-recompute reuse serves;
  * MultiHop-RAG (COLM'24, arXiv:2401.15391): real corpus where each query
    cites several distinct articles and NEIGHBORING queries share articles
    -- the measured fan-out/shift pattern this generator's sliding-window
    source overlap imitates (our audited real workload of the same shape).

Structure (legacy RAG-list schema, single-turn fan-out): N independent
queries; query r reads sources r .. r+k-1 (sliding window, stride 1), one
256-token chunk per source.  Consecutive queries therefore share k-1
sources, and a shared chunk sits one slot earlier per step -- every reuse
is position-shifted, chunks are small and provenance is maximally spread:
the access pattern where the maskable naive store (A3a) recovers part of
the naive penalty while the maskless one (A3) still splits a run at every
corrected row.  First use walks the sources in query order, so pages
append staggered and channel aliasing stays workload-driven.

Theoretical workload (mechanism illustration): token lengths are stated
assumptions; NOT evidence-grade.  The admitted REAL workload of this shape
is MultiHop-RAG (see /data2/chenyi9/KV-PIM/workload/).
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha16(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--sources-per-query", type=int, default=96,
                        help="distinct sources per request, ONE chunk each "
                             "(k=96 x 256 tokens + sys + query = 76%% of the "
                             "32,768-row extent cap)")
    parser.add_argument("--chunk-tokens", type=int, default=256,
                        help="natural KV-block size; one block per source")
    parser.add_argument("--stride", type=int, default=1,
                        help="sliding-window step between consecutive "
                             "queries' source sets")
    parser.add_argument("--sys-tokens", type=int, default=300)
    parser.add_argument("--query-tokens", type=int, default=100)
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    samples = []
    for r in range(args.queries):
        first = r * args.stride
        sources = list(range(first, first + args.sources_per_query))
        seg_lens = [args.sys_tokens] + \
                   [args.chunk_tokens] * len(sources) + [args.query_tokens]
        seg_sha = [sha16("rag-system-prompt")] + \
                  [sha16(f"source-{m}-chunk0") for m in sources] + \
                  [sha16(f"query-{r}")]
        seg_role = ["sys"] + ["doc"] * len(sources) + ["query"]
        samples.append({"sample": f"q{r}", "seg_lens": seg_lens,
                        "seg_sha": seg_sha, "seg_role": seg_role,
                        "L": sum(seg_lens), "lout": args.answer_tokens,
                        "history_len": 0})

    total_sources = (args.queries - 1) * args.stride + args.sources_per_query
    out = args.out or str(Path(__file__).resolve().parent /
                          "workload_multisource_rag_n{}s{}.json".format(
                              args.queries, args.sources_per_query))
    with open(out, "w") as handle:
        json.dump(samples, handle, indent=1)
    total = sum(sample["L"] for sample in samples)
    print("wrote {} : {} queries x {} sources (pool {}), one {}-token chunk "
          "each; input tokens={}".format(
              out, args.queries, args.sources_per_query, total_sources,
              args.chunk_tokens, total))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Theoretical MAP-REDUCE summarization workload (thin-sharing contrast).

Pattern (literature basis):
  * Wu et al., recursively summarizing books (OpenAI, arXiv:2109.10862):
    split a long text, summarize the pieces, merge the summaries;
  * the LangChain/LlamaIndex map-reduce summarization pattern (the same
    scatter-gather shape in production orchestrators).

Structure: N mappers in one tier, each reading its OWN disjoint document
slice (single-consumer chunks -- deliberately almost no cross-request
sharing beyond the shared mapper system prompt), then one reducer whose
prompt carries every mapper's summary.  This is the LOW-REUSE CONTRAST
case among the four typical topologies: it checks that the shared-KV
machinery adds no overhead when there is little to share.  Theoretical
workload; token lengths are stated assumptions; NOT evidence-grade.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha16(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mappers", type=int, default=4)
    parser.add_argument("--slice-tokens", type=int, default=1024,
                        help="each mapper's private document slice")
    parser.add_argument("--sys-tokens", type=int, default=300)
    parser.add_argument("--summary-tokens", type=int, default=200,
                        help="each mapper's summary (lout)")
    parser.add_argument("--final-tokens", type=int, default=256,
                        help="reducer's merged summary (lout)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sys_mapper = {"role": "sys", "sha": sha16("mapper-system-prompt"),
                  "len": args.sys_tokens}
    agents = []
    for m in range(args.mappers):
        agents.append({"id": f"map{m}", "tier": 0, "parent": None,
                       "history_len": 0,
                       "segs": [dict(sys_mapper),
                                {"role": "doc",
                                 "sha": sha16(f"book-slice-{m}"),
                                 "len": args.slice_tokens}],
                       "lout": args.summary_tokens})
    reduce_segs = [{"role": "parent_out", "sha": sha16("map0-summary"),
                    "len": args.summary_tokens, "delta": 0},
                   {"role": "sys", "sha": sha16("reducer-system-prompt"),
                    "len": args.sys_tokens}]
    reduce_segs += [{"role": "user", "sha": sha16(f"map{m}-summary"),
                     "len": args.summary_tokens}
                    for m in range(1, args.mappers)]
    agents.append({"id": "reduce", "tier": 1, "parent": "map0",
                   "history_len": 0, "segs": reduce_segs,
                   "lout": args.final_tokens})

    out = args.out or str(Path(__file__).resolve().parent /
                          "workload_mapreduce_sum_m{}.json".format(args.mappers))
    payload = {"meta": {
        "format": "v2-dag",
        "kind": "theoretical (mechanism illustration; NOT evidence-grade; "
                "LOW-REUSE CONTRAST case)",
        "scenario": "{} mappers over disjoint slices + 1 reducer".format(
            args.mappers),
        "references": ["Wu et al. arXiv:2109.10862",
                       "LangChain map-reduce summarization pattern"],
        "assumed_token_lengths": vars(args),
        "generator": "gen_mapreduce_sum.py"}, "agents": agents}
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    total = sum(seg["len"] for agent in agents for seg in agent["segs"])
    print("wrote {} : {} agents, input tokens={}".format(
        out, len(agents), total))


if __name__ == "__main__":
    main()

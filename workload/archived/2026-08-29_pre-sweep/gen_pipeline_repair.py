#!/usr/bin/env python3
"""Theoretical PIPELINE/WATERFALL multi-agent repair workload.

Pattern (literature basis):
  * MetaGPT (Hong et al., ICLR'24, arXiv:2308.00352): waterfall SOP --
    roles hand artifacts down a fixed chain;
  * ChatDev (Qian et al., ACL'24, arXiv:2307.07924): phase chain with
    review loops (coding <-> review) before testing.

Structure: Architect -> [Engineer -> Reviewer] x R review cycles ->
Tester, one chain (each node's single ``parent_out`` = the previous
stage's artifact).  A shared SPEC document (k fingerprinted chunks) is
re-read by every stage at a different offset (position-shifted reuse);
each role's system prompt is one fingerprint shared across its own
recurrences.  Theoretical workload (mechanism illustration), token
lengths are stated assumptions; NOT evidence-grade.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha16(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults re-sized 2026-08-26 (chenyi9: longer contexts, more rounds).
    parser.add_argument("--cycles", type=int, default=5,
                        help="engineer<->reviewer review cycles")
    # 256-token natural KV-block granularity (ruling chenyi9 2026-08-26):
    # one block per channel in the naive rotation; conflicts only from wrap.
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--shared-tokens", type=int, default=12800,
                        help="shared SPEC document tokens")
    parser.add_argument("--sys-tokens", type=int, default=300)
    parser.add_argument("--task-tokens", type=int, default=200)
    parser.add_argument("--plan-tokens", type=int, default=256)
    parser.add_argument("--patch-tokens", type=int, default=256)
    parser.add_argument("--review-tokens", type=int, default=128)
    parser.add_argument("--report-tokens", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    chunks = max(1, -(-args.shared_tokens // args.chunk_tokens))
    base, rem = divmod(args.shared_tokens, chunks)
    spec = [{"role": "doc", "sha": sha16(f"spec-chunk-{i}"),
             "len": base + (1 if i < rem else 0)}
            for i in range(chunks)]
    spec = [seg for seg in spec if seg["len"] > 0]

    def sys_seg(role):
        return {"role": "sys", "sha": sha16(f"{role}-system-prompt"),
                "len": args.sys_tokens}

    agents = []
    history = {}
    depth = 2 * args.cycles + 2

    def node(node_id, tier, parent, parent_len, role, lout):
        # SPEC accretes stage by stage (ruling chenyi9 2026-08-26): each
        # stage pulls in new sections for the first time and re-reads all
        # sections seen so far -- first use fixes the pages' append
        # position, so later stages' reused pages scatter naturally.
        visible = spec[: -(-len(spec) * (tier + 1) // depth)]
        segs = [sys_seg(role)]
        if parent is None:
            segs.append({"role": "user", "sha": sha16("repair-task"),
                         "len": args.task_tokens})
        else:
            segs.insert(0, {"role": "parent_out",
                            "sha": sha16(parent + "-out"),
                            "len": parent_len, "delta": 0})
        segs.extend(dict(seg) for seg in visible)
        agents.append({"id": node_id, "tier": tier, "parent": parent,
                       "history_len": history.get(role, 0),
                       "segs": segs, "lout": lout})
        history[role] = history.get(role, 0) + sum(seg["len"] for seg in segs)

    node("architect", 0, None, 0, "architect", args.plan_tokens)
    prev, prev_len, tier = "architect", args.plan_tokens, 1
    for cycle in range(args.cycles):
        node(f"engineer.c{cycle}", tier, prev, prev_len, "engineer",
             args.patch_tokens)
        prev, prev_len, tier = f"engineer.c{cycle}", args.patch_tokens, tier + 1
        node(f"reviewer.c{cycle}", tier, prev, prev_len, "reviewer",
             args.review_tokens)
        prev, prev_len, tier = f"reviewer.c{cycle}", args.review_tokens, tier + 1
    node("tester", tier, prev, prev_len, "tester", args.report_tokens)

    out = args.out or str(Path(__file__).resolve().parent /
                          "workload_pipeline_repair_c{}k{}.json".format(
                              args.cycles, chunks))
    payload = {"meta": {
        "format": "v2-dag",
        "kind": "theoretical (mechanism illustration; NOT evidence-grade)",
        "scenario": "waterfall repair chain: architect -> (engineer <-> "
                    "reviewer) x {} -> tester".format(args.cycles),
        "references": ["MetaGPT arXiv:2308.00352 (ICLR'24)",
                       "ChatDev arXiv:2307.07924 (ACL'24)"],
        "assumed_token_lengths": vars(args),
        "generator": "gen_pipeline_repair.py"}, "agents": agents}
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    total = sum(seg["len"] for agent in agents for seg in agent["segs"])
    print("wrote {} : {} agents (chain depth {}), input tokens={}".format(
        out, len(agents), agents[-1]["tier"] + 1, total))


if __name__ == "__main__":
    main()

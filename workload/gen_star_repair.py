#!/usr/bin/env python3
"""Theoretical star-topology multi-agent CODING/REPAIR workload (fig:motiv).

Scenario (chenyi9 order 2026-08-26): ONE main/planner agent directs THREE
worker agents over several instruction rounds -- the canonical published
multi-agent coding pattern:

  * AutoGen (Wu et al., arXiv:2308.08155): an orchestrator/group-chat
    manager routes multi-round instructions to worker agents;
  * MetaGPT (Hong et al., ICLR'24, arXiv:2308.00352): role SOP -- a
    planner hands tasks to engineer roles that share the same codebase;
  * AgentCoder (Huang et al., arXiv:2312.13010): three cooperating coding
    roles iterating on one repair task.

This is a THEORETICAL workload (mechanism illustration for the paper's
motivation figure), NOT evidence-grade data: token lengths below are
stated assumptions, the sharing structure follows the referenced
frameworks.  Per the workload admission rules
(/data2/chenyi9/KV-PIM/workload/README.md) synthetic designs are机制对照
only -- fig:motiv is exactly that use.

Structure per round r (v2-dag supervisor schema):
  main_r   : [parent_out(main_{r-1} instruction) | round 0: task text]
             + 3 worker replies of round r-1 (fresh text)
             + the SHARED CODEBASE (k chunks, fingerprinted -> reused at a
               different offset than the owner = position-shifted reuse)
             -> lout = next instruction; history_len grows with own stream
  w{i}_r   : parent_out(main_r instruction) + shared worker system prompt
             (one fingerprint for all workers/rounds) + the same k codebase
             chunks (shifted vs main's copy) -> lout = worker reply;
             history_len = the worker's own earlier rounds

Knobs for the two fig:motiv sweeps:
  --chunks k          irregularity axis: SAME shared tokens split into k
                      chunks (left sweep of F1; naive layout scatters k
                      blocks over channels)
  --shared-tokens L_s transfer axis: total shared-codebase tokens
                      (right sweep of F1)
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha16(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunks", type=int, default=8,
                        help="shared codebase split into k fingerprinted chunks")
    parser.add_argument("--shared-tokens", type=int, default=2048,
                        help="total shared-codebase tokens (split over --chunks)")
    parser.add_argument("--sys-main", type=int, default=300)
    parser.add_argument("--sys-worker", type=int, default=300)
    parser.add_argument("--task-tokens", type=int, default=200)
    parser.add_argument("--instr-tokens", type=int, default=128,
                        help="main's per-round instruction length (lout)")
    parser.add_argument("--reply-tokens", type=int, default=256,
                        help="each worker's per-round reply length (lout)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base, rem = divmod(args.shared_tokens, args.chunks)
    chunk_lens = [base + (1 if i < rem else 0) for i in range(args.chunks)]
    code_segs = [{"role": "doc", "sha": sha16(f"codebase-chunk-{i}"),
                  "len": length}
                 for i, length in enumerate(chunk_lens) if length > 0]
    sys_worker = {"role": "sys", "sha": sha16("worker-system-prompt"),
                  "len": args.sys_worker}

    agents = []
    main_hist = 0
    worker_hist = [0] * args.workers
    for r in range(args.rounds):
        tier_main = 2 * r
        main_id = f"main.r{r}"
        segs = []
        if r == 0:
            segs.append({"role": "sys", "sha": sha16("main-system-prompt"),
                         "len": args.sys_main})
            segs.append({"role": "user", "sha": sha16("repair-task"),
                         "len": args.task_tokens})
        else:
            segs.append({"role": "parent_out",
                         "sha": sha16(f"main-instr-r{r-1}"),
                         "len": args.instr_tokens, "delta": 0})
            for w in range(args.workers):
                segs.append({"role": "user",
                             "sha": sha16(f"w{w}-reply-r{r-1}"),
                             "len": args.reply_tokens})
        segs.extend(code_segs)          # planner re-consults the codebase
        agents.append({"id": main_id, "tier": tier_main,
                       "parent": f"main.r{r-1}" if r else None,
                       "history_len": main_hist, "segs": segs,
                       "lout": args.instr_tokens})
        main_hist += sum(seg["len"] for seg in segs)

        for w in range(args.workers):
            segs = [{"role": "parent_out", "sha": sha16(f"main-instr-r{r}"),
                     "len": args.instr_tokens, "delta": 0},
                    dict(sys_worker)] + [dict(seg) for seg in code_segs]
            agents.append({"id": f"w{w}.r{r}", "tier": tier_main + 1,
                           "parent": main_id,
                           "history_len": worker_hist[w], "segs": segs,
                           "lout": args.reply_tokens})
            worker_hist[w] += sum(seg["len"] for seg in segs)

    out = args.out or str(Path(__file__).resolve().parent /
                          "workload_star_repair_r{}w{}k{}.json".format(
                              args.rounds, args.workers, args.chunks))
    payload = {
        "meta": {
            "format": "v2-dag",
            "kind": "theoretical (mechanism illustration for fig:motiv; "
                    "NOT evidence-grade)",
            "scenario": "one main agent directing {} workers over {} "
                        "instruction rounds on one repair task".format(
                            args.workers, args.rounds),
            "references": ["AutoGen arXiv:2308.08155",
                           "MetaGPT arXiv:2308.00352 (ICLR'24)",
                           "AgentCoder arXiv:2312.13010"],
            "assumed_token_lengths": {
                "sys_main": args.sys_main, "sys_worker": args.sys_worker,
                "task": args.task_tokens, "instruction": args.instr_tokens,
                "worker_reply": args.reply_tokens,
                "shared_codebase_total": args.shared_tokens,
                "chunks": args.chunks},
            "generator": "gen_star_repair.py",
        },
        "agents": agents,
    }
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    total_in = sum(seg["len"] for agent in agents for seg in agent["segs"])
    print("wrote {} : {} agents ({} rounds x (1 main + {} workers)), "
          "input tokens={}, shared codebase {}x{} tokens".format(
              out, len(agents), args.rounds, args.workers, total_in,
              args.chunks, chunk_lens[0]))


if __name__ == "__main__":
    main()

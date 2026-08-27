#!/usr/bin/env python3
"""Theoretical DEBATE multi-agent workload (grounded on a shared document).

Pattern (literature basis):
  * Du et al., multi-agent debate (ICML'24, arXiv:2305.14325): N agents
    answer in parallel; each round every agent re-reads ALL peers'
    previous answers and revises;
  * Mixture-of-Agents (Wang et al., arXiv:2406.04692): layered agents
    aggregate every previous-layer answer.

Structure: N debaters x R rounds + 1 judge.  Round 0: each debater reads
its shared system prompt + the shared reference document (k fingerprinted
chunks) + the question.  Round r>0: parent_out = the debater's OWN
previous answer (self-continuation), plus every PEER's previous answer as
fingerprinted segments -- the same answer text lands in every other
debater's prompt at a different offset, the densest cross-request
position-shifted sharing of the four typical topologies.  The judge reads
the question plus every final answer.  Theoretical workload; token
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
    parser.add_argument("--debaters", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    # 256-token natural KV-block granularity (ruling chenyi9 2026-08-26):
    # one block per channel in the naive rotation; conflicts only from wrap.
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--shared-tokens", type=int, default=12544,
                        help="shared reference-document tokens")
    parser.add_argument("--sys-tokens", type=int, default=300)
    parser.add_argument("--question-tokens", type=int, default=100)
    parser.add_argument("--answer-tokens", type=int, default=256)
    parser.add_argument("--verdict-tokens", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    chunks = max(1, -(-args.shared_tokens // args.chunk_tokens))
    base, rem = divmod(args.shared_tokens, chunks)
    doc = [{"role": "doc", "sha": sha16(f"refdoc-chunk-{i}"),
            "len": base + (1 if i < rem else 0)}
           for i in range(chunks)]
    doc = [seg for seg in doc if seg["len"] > 0]
    sys_debater = {"role": "sys", "sha": sha16("debater-system-prompt"),
                   "len": args.sys_tokens}
    question = {"role": "user", "sha": sha16("debate-question"),
                "len": args.question_tokens}

    def answer_seg(debater, round_index):
        return {"role": "user",
                "sha": sha16(f"answer-d{debater}-r{round_index}"),
                "len": args.answer_tokens}

    agents = []
    history = [0] * args.debaters
    for r in range(args.rounds):
        # Evidence ACCRETES round by round (ruling chenyi9 2026-08-26):
        # each round cites ~1/R new document sections for the first time
        # and re-reads everything cited so far -- first use fixes the
        # pages' append position, so reused pages scatter naturally.
        visible = doc[: -(-len(doc) * (r + 1) // args.rounds)]
        for d in range(args.debaters):
            segs = []
            if r == 0:
                segs.append(dict(sys_debater))
                segs.extend(dict(seg) for seg in visible)
                segs.append(dict(question))
                parent = None
            else:
                parent = f"d{d}.r{r-1}"
                segs.append({"role": "parent_out",
                             "sha": sha16(f"answer-d{d}-r{r-1}"),
                             "len": args.answer_tokens, "delta": 0})
                for peer in range(args.debaters):
                    if peer != d:
                        segs.append(answer_seg(peer, r - 1))
                segs.extend(dict(seg) for seg in visible)  # re-consult cited sections
            agents.append({"id": f"d{d}.r{r}", "tier": r, "parent": parent,
                           "history_len": history[d], "segs": segs,
                           "lout": args.answer_tokens})
            history[d] += sum(seg["len"] for seg in segs)
    judge_segs = [{"role": "parent_out",
                   "sha": sha16(f"answer-d0-r{args.rounds-1}"),
                   "len": args.answer_tokens, "delta": 0},
                  {"role": "sys", "sha": sha16("judge-system-prompt"),
                   "len": args.sys_tokens}, dict(question)]
    judge_segs += [answer_seg(d, args.rounds - 1)
                   for d in range(1, args.debaters)]
    agents.append({"id": "judge", "tier": args.rounds,
                   "parent": f"d0.r{args.rounds-1}", "history_len": 0,
                   "segs": judge_segs, "lout": args.verdict_tokens})

    out = args.out or str(Path(__file__).resolve().parent /
                          "workload_debate_d{}r{}k{}.json".format(
                              args.debaters, args.rounds, chunks))
    payload = {"meta": {
        "format": "v2-dag",
        "kind": "theoretical (mechanism illustration; NOT evidence-grade)",
        "scenario": "{} debaters x {} rounds over one shared document + "
                    "judge".format(args.debaters, args.rounds),
        "references": ["Du et al. arXiv:2305.14325 (ICML'24)",
                       "Mixture-of-Agents arXiv:2406.04692"],
        "assumed_token_lengths": vars(args),
        "generator": "gen_debate.py"}, "agents": agents}
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    total = sum(seg["len"] for agent in agents for seg in agent["segs"])
    print("wrote {} : {} agents, input tokens={}".format(
        out, len(agents), total))


if __name__ == "__main__":
    main()

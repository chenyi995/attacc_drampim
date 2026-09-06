#!/usr/bin/env python3
"""Owner + the long-fresh reusers of gen_a6.py + standalone fresh chat prompts.

    MODE=crossover  -> one standalone prompt each of 8, 16, ..., 1024 tokens
                       (find where the A6 chooser flips: <= 512 GPU, >= 1024 bank)
    MODE=split      -> CHAT_N prompts of CHAT_LEN tokens, CHAT_LOUT answers
                       (the A6 split workload: long reusers in the banks,
                       short chats on the GPU)

    python3 gen_a6_chat_mix.py <base_a6_workload.json> > wl.json
"""
import hashlib
import json
import os
import sys

MODE = os.environ.get("MODE", "split")
CHAT_N = int(os.environ.get("CHAT_N", "32"))
CHAT_LEN = int(os.environ.get("CHAT_LEN", "512"))
CHAT_LOUT = int(os.environ.get("CHAT_LOUT", "16"))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    base = json.load(open(sys.argv[1]))
    agents = [a for a in base["agents"] if a["id"] == "a0_owner" or "longfresh" in a["id"]]
    if MODE == "crossover":
        for length in (8, 16, 32, 64, 128, 256, 512, 1024):
            agents.append({"id": "s%d_fresh" % length, "tier": 0, "parent": None,
                           "history_len": 0, "lout": 128,
                           "segs": [{"role": "sys", "sha": sha("short-%d" % length), "len": length}]})
        kind = "A6 crossover probe: standalone fresh prompts of 8..1024 tokens beside long reusers"
    else:
        for index in range(CHAT_N):
            agents.append({"id": "c%02d_chat" % index, "tier": 0, "parent": None,
                           "history_len": 0, "lout": CHAT_LOUT,
                           "segs": [{"role": "sys", "sha": sha("chat-%d" % index), "len": CHAT_LEN}]})
        kind = ("A6 split: long multi-round reusers (bank side) + standalone chat prompts "
                "(GPU side), repairs per agent, no penalties")
    json.dump({"meta": {"format": "v2-dag", "kind": kind, "mode": MODE, "chat_n": CHAT_N,
                        "chat_len": CHAT_LEN, "chat_lout": CHAT_LOUT},
               "agents": agents}, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Baseline + one-axis sweeps for the seven-rung ladder (2026-09-05 plan).

The workload is a multi-agent RAG / codebase-agent session:

  * a supervisor (``a0_owner``) ingests the shared corpus -- CORPUS chunks of
    256 tokens (one DRAM row each) behind a 256-token system prompt.  This
    is a long FRESH prefill: under ``--gpu-model flash`` A6 sends it to the
    GPU while A5 must scan it in the banks -- the A5/A6 separation.
  * N worker agents, each R rounds; every round retrieves C shared chunks
    (agent i starts at chunk i*C, so agents overlap but are not identical,
    which gives A4e's co-read table something to decide) and writes OWN
    tokens of its own reasoning.  Retrieved chunks sit at different offsets
    than in the supervisor's context, so each takes k recomputed rows
    (k is the run-time ``--epic-prefix-recompute-tokens``, not a workload
    field).  Repairs are per-agent, as in the engine.  No penalties.

Two encodings of the same session:

  ``interleaved``  one request per agent whose context alternates
                   ``chunks | own`` per round (one prefill, one decode of
                   LOUT tokens).  Simple, hand-checkable.
  ``turns``        one request per (agent, round): tier = round, parent =
                   the agent's previous round, a ``parent_out`` segment for
                   its decoded output, ``history_len`` = the agent's earlier
                   prefill rows (resident, never recomputed), LOUT tokens
                   decoded per round.  The realistic multi-turn form; the
                   decode rows of round r are what round r+1 attends.

Standalone fresh chat prompts (S5) are extra tier-0 requests of 2k/4k/8k
tokens with no reuse; ``fresh_share`` is their fraction of all requests.

    python3 gen_sweep.py --all <outdir>        # the whole matrix + manifest.csv
    python3 gen_sweep.py --form turns --agents 8 --rounds 8 ... > wl.json
"""
import argparse
import csv
import hashlib
import json
import os
import sys

BLOCK = 256
SYS = 256
FRESH_LENGTHS = (2048, 4096, 8192)

BASELINE = dict(agents=8, rounds=8, chunks=2, own=128, lout=256, corpus=64,
                fresh_share=0.0)
# axis -> (parameter, values)  (the baseline value is not repeated)
SWEEPS = {
    "S1_agents": ("agents", (4, 16, 32)),
    "S2_rounds": ("rounds", (2, 4, 16)),
    "S3_lout": ("lout", (128, 512, 1024, 2048)),
    "S4_own": ("own", (16, 64, 256, 1024)),
    "S5_fresh": ("fresh_share", (0.25, 0.5, 0.75)),
    "S6_chunks": ("chunks", (1, 4, 8)),
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _chunk(index):
    return {"role": "doc", "sha": sha("corpus-%d" % index), "len": BLOCK}


def _retrieved(agent, round_index, p):
    """Chunk indices agent ``agent`` reads in round ``round_index``."""
    base = agent * p["chunks"] + round_index * p["chunks"]
    return [(base + offset) % p["corpus"] for offset in range(p["chunks"])]


def _fresh_prompts(p, n_session):
    share = p["fresh_share"]
    if share <= 0:
        return []
    count = int(round(share / (1 - share) * n_session))
    prompts = []
    for index in range(count):
        length = FRESH_LENGTHS[index % len(FRESH_LENGTHS)]
        prompts.append({"id": "f%02d_fresh%d" % (index, length), "tier": 0, "parent": None,
                        "history_len": 0, "lout": 128,
                        "segs": [{"role": "sys", "sha": sha("fresh-%d" % index), "len": length}]})
    return prompts


def build(p, form):
    owner = {"id": "a0_owner", "tier": 0, "parent": None, "history_len": 0, "lout": p["lout"],
             "segs": [{"role": "sys", "sha": sha("a0-sys"), "len": SYS}] +
                     [_chunk(index) for index in range(p["corpus"])]}
    agents = [owner]
    for agent in range(p["agents"]):
        wid = "w%02d" % agent
        if form == "interleaved":
            segs = [{"role": "sys", "sha": sha(wid + "-sys"), "len": SYS}]
            for round_index in range(p["rounds"]):
                segs += [_chunk(index) for index in _retrieved(agent, round_index, p)]
                segs.append({"role": "user", "sha": sha("%s-own-%d" % (wid, round_index)),
                             "len": p["own"]})
            agents.append({"id": wid, "tier": 0, "parent": None, "history_len": 0,
                           "lout": p["lout"], "segs": segs})
        else:
            history = 0
            for round_index in range(p["rounds"]):
                rid = "%s_t%02d" % (wid, round_index)
                segs = []
                if round_index == 0:
                    segs.append({"role": "sys", "sha": sha(wid + "-sys"), "len": SYS})
                    parent = None
                else:
                    parent = "%s_t%02d" % (wid, round_index - 1)
                    segs.append({"role": "parent_out", "sha": sha(parent + "-out"),
                                 "len": p["lout"]})
                segs += [_chunk(index) for index in _retrieved(agent, round_index, p)]
                segs.append({"role": "user", "sha": sha("%s-own-%d" % (wid, round_index)),
                             "len": p["own"]})
                agents.append({"id": rid, "tier": round_index, "parent": parent,
                               "history_len": history, "lout": p["lout"], "segs": segs})
                # next round holds this round's prefill rows as resident history
                # (its decoded output arrives as the parent_out segment)
                history += sum(seg["len"] for seg in segs)
    agents += _fresh_prompts(p, len(agents))
    meta = {"format": "v2-dag", "kind": "ladder sweep point (gen_sweep.py, 2026-09-05)",
            "form": form, "block_tokens": BLOCK, "sys_tokens": SYS}
    meta.update(p)
    return {"meta": meta, "agents": agents}


def stats(workload):
    agents = workload["agents"]
    prefill = sum(sum(seg["len"] for seg in a["segs"]) for a in agents)
    decode = sum(a["lout"] for a in agents)
    return len(agents), prefill, decode


def write_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []

    def emit(name, axis, value, p, form):
        workload = build(p, form)
        path = os.path.join(outdir, "%s_%s.json" % (name, form))
        with open(path, "w") as handle:
            json.dump(workload, handle, indent=1)
            handle.write("\n")
        n, prefill, decode = stats(workload)
        rows.append({"file": os.path.basename(path), "axis": axis, "value": value, "form": form,
                     "agents": p["agents"], "rounds": p["rounds"], "chunks": p["chunks"],
                     "own": p["own"], "lout": p["lout"], "fresh_share": p["fresh_share"],
                     "requests": n, "prefill_tokens": prefill, "decode_tokens": decode})

    for form in ("interleaved", "turns"):
        emit("B0", "baseline", "-", dict(BASELINE), form)
        for axis, (param, values) in SWEEPS.items():
            for value in values:
                p = dict(BASELINE)
                p[param] = value
                tag = ("%s_%s" % (axis, str(value).replace(".", "p")))
                emit(tag, axis, value, p, form)
    with open(os.path.join(outdir, "manifest.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", metavar="OUTDIR", help="write the whole matrix + manifest.csv")
    parser.add_argument("--form", choices=("interleaved", "turns"), default="interleaved")
    for key, value in BASELINE.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    if args.all:
        rows = write_all(args.all)
        print("%d workloads -> %s" % (len(rows), args.all))
        return 0
    p = {key: getattr(args, key) for key in BASELINE}
    json.dump(build(p, args.form), sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

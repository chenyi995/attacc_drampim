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
                   the agent's previous round.  Round r RE-LISTS the agent's
                   whole earlier context -- system prompt, every earlier
                   retrieved chunk, every earlier own block and every earlier
                   decoded output (the previous round's as ``parent_out``,
                   older ones as reused segments) -- at the same offsets, then
                   adds this round's chunks and own block.  So the planner
                   reuses the SAME physical objects: the earlier turns'
                   masters, the corrections they already wrote (inherited,
                   not recomputed -- C8, chenyi9 2026-09-05), the outputs.
                   No ``history_len`` placeholder.  LOUT tokens decoded per
                   round; the decode rows of round r are what round r+1
                   attends.

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

# B1 (2026-09-05, after the first small run): a session built so that every
# rung has something to separate it from the previous one.
#   * retrieval="scattered": a round's chunks are a seeded random subset of
#     the corpus, so chunks read together often sit an even number of
#     writes apart -- the same slot under the naive rotation (stripe 2), the
#     conflict A4e's table removes.  "consecutive" is the old pattern.
#   * two kinds of agent: CHATTY ones write own_chatty tokens per turn (the
#     decode-shaped prefill the banks win), WRITER ones write own_writer
#     (the GPU wins); chatty_share is the chatty fraction.  A6 splits them,
#     A5 must put every turn (and the corpus ingest) in the banks.
#   * turns form: every round is a request over the whole earlier context,
#     so each turn's prefill is m = own + repairs over n = the context.
#   * chunks=2 per round: a round's repairs (2 x k = 16 tokens = 64 B) are
#     far below one DRAM row, so A3b's one-row-per-round pages waste most of
#     each row and A4c's packed diff rows pay off; with 4 chunks a round
#     already fills a row and gathering changes nothing (lever probe,
#     output/analysis/b1_levers.py).
B1 = dict(agents=8, rounds=8, chunks=2, corpus=64, lout=128, retrieval="scattered",
          own_chatty=16, own_writer=256, chatty_share=0.5, fresh_share=0.0, lout_chatty=None,
          shared=0)
# C1 "classic" (chenyi9 2026-09-05: one workload that gives every rung its
# lever, small enough to run the seven-rung ladder on a real model):
#   * 6 worker agents x 6 turns, ONE retrieved chunk per turn -- 6 x 8 = 48
#     repair tokens per global round, so an agent's diffs of consecutive
#     turns share rows only under A4c's packed diff region (A3b pays one
#     row per turn; 68% fewer repair rows in the lever probe);
#   * scattered retrieval over a 64-chunk corpus -> co-read conflicts for
#     A4e's table (28% fewer busiest-lane rows at 8 channels per head);
#   * half chatty agents (16 own tokens, 32-token answers) and half writers
#     (256 own tokens, 128-token answers) -> the decode-shaped turns A5/A6
#     put in the banks and the writer turns A6 sends to the GPU;
#   * fresh_share 0.1 -> four standalone fresh chats (2k/4k/8k/2k tokens, no
#     reuse): the long fresh prefill that under flash belongs on the GPU,
#     so A6 separates from A5 even when every reuse turn favours the banks.
#   * shared=8 (added after the first C1 ladder, 2026-09-05): the workers
#     share ONE system prompt and re-read the same 8 corpus chunks (a 2k
#     "project brief") in every turn.  The first C1 had no KV row read by two
#     requests of the same decode batch, so the MQ command of A5/A6 had
#     nothing to share and their decode was identical to A4e's; a batch
#     that co-reads the brief is what one MQ sweep serves at once.
C1 = dict(B1, agents=6, rounds=6, chunks=1, lout_chatty=32, fresh_share=0.1, shared=8)
# C2, the second baseline (chenyi9 2026-09-05: "如果一个 baseline 不够就两个"):
# a chat-heavy session -- every worker is chatty (16 own tokens, 32-token
# answers), the corpus is small (16 chunks), the shared brief is the same 8
# chunks, no fresh chats.  Every turn is decode-shaped, so here A5's
# bank-side prefill wins outright (C1 makes it lose on the 16k ingest and
# the fresh chats) and A6 matches it; the layout levers stay (scattered
# retrieval, one chunk per turn); more agents sharing the brief give MQ more
# to share per sweep.
C2 = dict(B1, agents=8, rounds=8, chunks=1, corpus=16, lout_chatty=32,
          chatty_share=1.0, fresh_share=0.0, shared=8)
BASELINES = {"C1": C1, "C2": C2}
# One-axis sweeps around C1 (sweep points run A3b and A6 only).  Each axis
# moves one lever of the A3b -> A6 gain: how many agents share the brief and
# the banks (MQ, table conflicts), how long the session runs (repairs
# accumulate), how much of the session is decode-shaped (the chooser's PIM
# share), how large the resident context is (scan share of a step), how
# long the answers are (decode share of E2E), how many chunks a turn
# retrieves (repairs per turn), and how much fresh long-prompt work arrives
# (A6's GPU-side share).
C1_SWEEPS = {
    "S1_agents": ("agents", (4, 12, 16)),
    "S2_rounds": ("rounds", (3, 12)),
    "S3_chatty_share": ("chatty_share", (0.0, 0.25, 0.75, 1.0)),
    "S4_shared": ("shared", (0, 16, 32)),
    "S5_lout_chatty": ("lout_chatty", (8, 128)),
    "S6_lout": ("lout", (32, 512)),
    "S7_chunks": ("chunks", (2, 4)),
    "S8_fresh_share": ("fresh_share", (0.0, 0.25)),
    "S9_corpus": ("corpus", (32, 128)),
    "S10_retrieval": ("retrieval", ("consecutive",)),
}
B1_SWEEPS = {
    "T1_agents": ("agents", (4, 16)),
    "T2_rounds": ("rounds", (4, 16)),
    "T3_chunks": ("chunks", (1, 4)),
    "T4_own_chatty": ("own_chatty", (8, 32, 64)),
    "T5_chatty_share": ("chatty_share", (0.0, 0.25, 0.75, 1.0)),
    "T6_retrieval": ("retrieval", ("consecutive",)),
    "T7_corpus": ("corpus", (16, 32)),
    "T8_lout": ("lout", (32, 512)),
    # T9: agents that answer at different lengths -- chatty agents decode
    # lout_chatty tokens per turn, writers lout_writer -- so the decode
    # output stream interleaves unevenly and short-answer agents drop out of
    # later steps (the "a chunk / b others / c diff / d others / e output"
    # structure chenyi9 asked about, 2026-09-05).  Turn ORDER is still the
    # round index: the engine starts a tier when the whole previous tier is
    # done, so a fast agent cannot really run ahead in time; that is a
    # scheduler property, not a workload one.
    "T9_lout_mix": ("lout_chatty", (32, 8)),
}
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


def _shared_chunks(p):
    """The corpus chunks every worker re-reads in every turn (``shared``)."""
    return list(range(min(int(p.get("shared") or 0), p["corpus"])))


def _retrieved(agent, round_index, p):
    """Chunk indices agent ``agent`` reads in round ``round_index`` (the
    shared brief excluded: it is listed once, before the retrieved ones)."""
    pool = [index for index in range(p["corpus"]) if index not in set(_shared_chunks(p))]
    if p.get("retrieval", "consecutive") == "scattered":
        import random
        rng = random.Random(1000003 * agent + 7919 * round_index + 17)
        return sorted(rng.sample(pool, min(p["chunks"], len(pool))))
    base = agent * p["chunks"] + round_index * p["chunks"]
    return [pool[(base + offset) % len(pool)] for offset in range(p["chunks"])]


def _own_tokens(agent, p):
    """Own tokens an agent writes per round: chatty agents come first."""
    if "own_chatty" not in p:
        return p["own"]
    chatty = int(round(p["chatty_share"] * p["agents"]))
    return p["own_chatty"] if agent < chatty else p["own_writer"]


def _lout(agent, p):
    """Tokens an agent decodes per turn: ``lout_chatty`` for chatty agents
    when set (T9), else the workload's ``lout``."""
    if p.get("lout_chatty") is None or "own_chatty" not in p:
        return p["lout"]
    chatty = int(round(p["chatty_share"] * p["agents"]))
    return p["lout_chatty"] if agent < chatty else p["lout"]


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
        # shared > 0: one system prompt for every worker, then the brief
        sys_sha = sha("workers-sys") if _shared_chunks(p) else sha(wid + "-sys")
        brief = [_chunk(index) for index in _shared_chunks(p)]
        if form == "interleaved":
            segs = [{"role": "sys", "sha": sys_sha, "len": SYS}] + [dict(c) for c in brief]
            for round_index in range(p["rounds"]):
                segs += [_chunk(index) for index in _retrieved(agent, round_index, p)]
                segs.append({"role": "user", "sha": sha("%s-own-%d" % (wid, round_index)),
                             "len": _own_tokens(agent, p)})
            agents.append({"id": wid, "tier": 0, "parent": None, "history_len": 0,
                           "lout": _lout(agent, p), "segs": segs})
        else:
            context = [{"role": "sys", "sha": sys_sha, "len": SYS}] + [dict(c) for c in brief]
            for round_index in range(p["rounds"]):
                rid = "%s_t%02d" % (wid, round_index)
                parent = None if round_index == 0 else "%s_t%02d" % (wid, round_index - 1)
                segs = [dict(seg) for seg in context]
                if parent is not None:
                    # the previous round's decoded output, right after the
                    # context it was decoded from
                    segs.append({"role": "parent_out", "sha": sha(parent + "-out"),
                                 "len": _lout(agent, p)})
                segs += [_chunk(index) for index in _retrieved(agent, round_index, p)]
                segs.append({"role": "user", "sha": sha("%s-own-%d" % (wid, round_index)),
                             "len": _own_tokens(agent, p)})
                agents.append({"id": rid, "tier": round_index, "parent": parent,
                               "history_len": 0, "lout": _lout(agent, p), "segs": segs})
                # the next round re-lists everything above; the output this
                # round decodes becomes an ordinary reused segment two rounds on
                context = [dict(seg) for seg in segs]
                if parent is not None:
                    context[len(context) - len(segs) + segs.index(
                        next(seg for seg in segs if seg["role"] == "parent_out"))]["role"] = "user"
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
                     "own": p.get("own", "%s/%s" % (p.get("own_chatty"), p.get("own_writer"))),
                     "chatty_share": p.get("chatty_share", ""), "retrieval": p.get("retrieval", "consecutive"),
                     "lout_chatty": p.get("lout_chatty") or "",
                     "shared": p.get("shared") or 0,
                     "corpus": p["corpus"], "lout": p["lout"], "fresh_share": p["fresh_share"],
                     "requests": n, "prefill_tokens": prefill, "decode_tokens": decode})

    # The protocol (chenyi9 2026-09-05): baselines C1 / C2 run every combo
    # (A1 A2 A3b A4c A4e A5 A6); the C1_S* sweep points run A3b and A6.
    for name, preset in BASELINES.items():
        emit(name, "baseline", "-", dict(preset), "turns")
    for axis, (param, values) in C1_SWEEPS.items():
        for value in values:
            p = dict(C1)
            p[param] = value
            emit("C1_%s_%s" % (axis, str(value).replace(".", "p")), axis, value, p, "turns")
    if os.environ.get("LEGACY_MATRIX", "0") == "1":
        # the 2026-09-05 B0 / S1-S6 (both forms) and B1 / T1-T9 sets, kept
        # reproducible but no longer written by default
        for form in ("interleaved", "turns"):
            emit("B0", "baseline", "-", dict(BASELINE), form)
            for axis, (param, values) in SWEEPS.items():
                for value in values:
                    p = dict(BASELINE)
                    p[param] = value
                    emit("%s_%s" % (axis, str(value).replace(".", "p")), axis, value, p, form)
        emit("B1", "baseline", "-", dict(B1), "turns")
        for axis, (param, values) in B1_SWEEPS.items():
            for value in values:
                p = dict(B1)
                p[param] = value
                emit("%s_%s" % (axis, str(value).replace(".", "p")), axis, value, p, "turns")
    with open(os.path.join(outdir, "manifest.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", metavar="OUTDIR", help="write the whole matrix + manifest.csv")
    parser.add_argument("--form", choices=("interleaved", "turns"), default="interleaved")
    parser.add_argument("--preset", choices=("B0", "B1", "C1", "C2"), default="C1",
                        help="C1 / C2: the protocol baselines; B1: their parent "
                             "(chatty/writer agents, scattered retrieval); B0: the plain keys")
    seen = set()
    for preset in (BASELINE, B1):
        for key, value in preset.items():
            if key in seen:
                continue
            seen.add(key)
            kind = str if value is None else type(value)
            parser.add_argument("--" + key.replace("_", "-"), type=kind, default=None)
    args = parser.parse_args()
    if args.all:
        rows = write_all(args.all)
        print("%d workloads -> %s" % (len(rows), args.all))
        return 0
    p = dict({"B0": BASELINE, "B1": B1, "C1": C1, "C2": C2}[args.preset])
    for key in p:
        value = getattr(args, key)
        if value is not None:
            p[key] = value
    if p.get("lout_chatty") is not None:
        p["lout_chatty"] = int(p["lout_chatty"])
    json.dump(build(p, args.form), sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

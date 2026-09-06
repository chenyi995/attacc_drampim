#!/usr/bin/env python3
"""The run protocol's workloads (chenyi9 2026-09-05/06): a main agent that
summarizes, every round, what its workers read.

Topology of one session (``sessions`` of them run side by side, sharing
the device and the decode batches):

  * ``a0_corpus``: one owner ingests every document once (256 tokens each,
    one DRAM row) -- the shared masters.
  * ``workers`` worker chains: in round r worker w reads document
    ``r*workers + w`` -- NEW every round, reused from the owner at a
    different offset, so it takes k recomputed rows (k = the run-time
    --epic-prefix-recompute-tokens) -- writes a 16-token note and answers
    ``worker_lout`` tokens.  Its context re-lists everything it read and
    wrote before, so earlier rounds' corrections are inherited, not redone.
  * one main chain: in round r >= 2 it summarizes the workers' answers of
    round r-2 -- the ``workers`` answer blocks enter its context as reused
    segments at new offsets, ``workers x k`` fresh corrections per round,
    every earlier round's corrections still valid and inherited -- plus a
    16-token instruction, and answers ``main_lout`` tokens.

  Why r-2 and not r-1: the planner makes the FIRST request in (tier, id)
  order that lists a fingerprint its owner.  The worker's own next turn
  (tier r-1) names its answer as parent_out; the main's reference must come
  in a later tier or it would be taken for the writer and get no diff.

What each combo gets to show (structural probe, 8 channels per KV head):
  A3b -> A4c   the main's corrections of successive rounds land in
               different rows of the naive stream ("broken" diffs); the
               diff region gathers them (72% fewer repair rows, r16 w4 s1).
  A4c -> A4e   the workers' answers are co-read by the main; the table
               keeps them off one channel (8-26% fewer busiest-lane rows).
  A4e -> A5    every turn is decode-shaped (m = 32-48 over a 1-6k context):
               MQ shares the batch's common rows; the bank-side prefill wins.
  A5 -> A6     the corpus ingest is the one large fresh prefill; A6 keeps it
               on the GPU.

    python3 workload/probe/gen_main_workers.py --all workload/probe/sweep   # W1 + its sweeps + manifest.csv
    python3 workload/probe/gen_main_workers.py --rounds 24 --workers 2 --sessions 2 > wl.json
"""
import argparse
import csv
import hashlib
import json
import os
import sys

# W1, the protocol baseline (chenyi9 2026-09-06): TWO sessions -- both
# sessions' worker w read the same new document every round, so a decode
# batch holds co-read rows (MQ's material) and the table sees cross-session
# co-reads; 2 workers per main keep the main's corrections per round small
# (2 x k) so A3b's naive stream breaks them across rows without flooding the
# diff channel; 128-token worker answers are the blocks the main co-reads.
# Structural probe at 8 channels per KV head: A4c 72% fewer repair rows,
# A4e 35% fewer busiest-lane rows (1-session r16 w4: 72% / 8%).
BASELINE = dict(rounds=24, workers=2, sessions=2, worker_lout=128, main_lout=128,
                doc_tokens=256, note_tokens=16)
# One-axis sweeps around W1 (sweep points run A3b and A6 only).
SWEEPS = {
    "S1_rounds": ("rounds", (12, 48)),         # how many rounds of broken diffs accumulate
    "S2_workers": ("workers", (1, 4)),         # corrections per main round, co-read width
    "S3_sessions": ("sessions", (1, 4)),       # requests ready per tier (batch, MQ sharing)
    "S4_worker_lout": ("worker_lout", (32, 256)),   # size of the blocks the main co-reads
    "S5_main_lout": ("main_lout", (32, 512)),  # decode share of E2E
    "S6_doc_tokens": ("doc_tokens", (512, 1024)),   # resident context per round
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def seg(name, length, role="doc"):
    return {"role": role, "sha": sha(name), "len": length}


def out_sha(rid):
    """The fingerprint a child declares for its parent's answer."""
    return sha(rid + "-output")


def request(rid, tier, segs, lout, parent=None):
    return dict(id=rid, tier=tier, parent=parent, history_len=0, lout=lout, segs=segs)


def advance(previous, prefix, additions, rid, tier, lout):
    """Re-list the whole earlier context at the same offsets, then append."""
    segs = [dict(s, role="user" if s["role"] == "parent_out" else s["role"])
            for s in (previous["segs"] if previous else prefix)]
    if previous:
        segs.append({"role": "parent_out", "sha": out_sha(previous["id"]), "len": previous["lout"]})
    segs.extend(additions)
    return request(rid, tier, segs, lout, previous["id"] if previous else None)


def build(p):
    rounds, workers, sessions = p["rounds"], p["workers"], p["sessions"]
    docs = [seg("mw-document-%d" % j, p["doc_tokens"]) for j in range(rounds * workers)]
    agents = [request("a0_corpus", 0, [seg("corpus-prefix", 256, "sys")] + docs, 1)]
    prior = {}
    for r in range(rounds):
        for g in range(sessions):
            for w in range(workers):
                key = (g, "w", w)
                rid = "g%02d_w%d_t%03d" % (g, w, r)
                additions = [dict(docs[r * workers + w]), seg(rid + "-note", p["note_tokens"], "user")]
                a = advance(prior.get(key), [seg("g%d-w%d-prefix" % (g, w), 16, "sys")],
                            additions, rid, r, p["worker_lout"])
                prior[key] = a
                agents.append(a)
            key = (g, "m")
            rid = "g%02d_m_t%03d" % (g, r)
            if r < 2:
                additions = [seg(rid + "-instruction", 16, "user")]
            else:
                earlier = ["g%02d_w%d_t%03d" % (g, w, r - 2) for w in range(workers)]
                additions = [{"role": "doc", "sha": out_sha(wid), "len": p["worker_lout"]} for wid in earlier]
                additions.append(seg(rid + "-instruction", 16, "user"))
            a = advance(prior.get(key), [seg("g%d-m-prefix" % g, 16, "sys")], additions, rid, r, p["main_lout"])
            prior[key] = a
            agents.append(a)
    meta = {"format": "v2-dag", "kind": "main agent summarizing its workers (gen_main_workers.py, 2026-09-06)",
            "block_tokens": 256}
    meta.update(p)
    return {"meta": meta, "agents": agents}


def stats(workload):
    agents = workload["agents"]
    return (len(agents), sum(sum(s["len"] for s in a["segs"]) for a in agents),
            sum(a["lout"] for a in agents))


def write_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []

    def emit(name, axis, value, p):
        workload = build(p)
        path = os.path.join(outdir, name + ".json")
        with open(path, "w") as handle:
            json.dump(workload, handle, indent=1)
            handle.write("\n")
        n, prefill, decode = stats(workload)
        row = {"file": os.path.basename(path), "axis": axis, "value": value}
        row.update(p)
        row.update({"requests": n, "prefill_tokens": prefill, "decode_tokens": decode})
        rows.append(row)

    emit("W1_turns", "baseline", "-", dict(BASELINE))
    for axis, (param, values) in SWEEPS.items():
        for value in values:
            p = dict(BASELINE)
            p[param] = value
            emit("W1_%s_%s_turns" % (axis, value), axis, value, p)
    with open(os.path.join(outdir, "manifest.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", metavar="OUTDIR", help="write the baseline, the sweeps and manifest.csv")
    for key, value in BASELINE.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    args = parser.parse_args()
    if args.all:
        rows = write_all(args.all)
        print("%d workloads -> %s" % (len(rows), args.all))
        return 0
    p = {key: getattr(args, key) for key in BASELINE}
    json.dump(build(p), sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

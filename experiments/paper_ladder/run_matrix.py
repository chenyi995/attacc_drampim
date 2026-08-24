"""Paper ladder matrix: same workloads x A1-A6 x several models, in parallel.

Serves the paper's Question 1 (placement: where do prefill attention,
decode attention and the KV cache go under shared-KV reuse, and when is
the per-request dynamic rule worth it).  Question 2 (the MQ batching
microarchitecture) is measured by the C series on the experiment branch;
this matrix consumes its mechanisms (A5/A6 run the mq command) but does
not re-measure them.

Axes (ruling 2026-08-25):
- workloads: the three REAL multi-agent/multi-turn cases (Mooncake
  toolagent, ShareGPT multi-round, MultiHop-RAG) plus the synthetic
  relay case, IDENTICAL across every rung and model;
- models: LLAMA-7B / LLAMA-65B / GPT-175B (small/mid/large real dims);
- rungs: A1..A6 (A1 pairs with no-reuse by definition);
- software upstream: ONLY the guaranteed-recompute selection family --
  every policy recomputes some tokens and they differ ONLY in which
  tokens are selected.  EPIC counts as the family's "first k tokens per
  shifted segment" special case.  The zero-recompute endpoint
  (promptcache) is implemented in the simulator but EXCLUDED from the
  matrix by ruling.  The selection variants (cacheblend deviation-r /
  epic first-k / cachecraft overlap-scaled prefix / cachetune offline-r)
  sweep at the A6 point where the paper's method lives.

Budget: 64 cores = MAX_CONCURRENT runs x RAMULATOR_WORKERS each.
Restartable: a run whose result JSON already exists is skipped.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")

MAX_CONCURRENT = 16
RAMULATOR_WORKERS = 4

WORKLOADS = {
    "mooncake": "workloads/workload_mooncake_toolagent_n40_o0.json",
    "sharegpt": "workloads/workload_sharegpt_c10_r3-8_o0.json",
    "multihop": "workloads/workload_multihoprag_n32_o0.json",
    "relay": "workloads/workload_relay_s400w4t1.json",
}
MODELS = {"LLAMA-7B": 32, "LLAMA-65B": 80, "GPT-175B": 96}  # name -> ndec
RUNGS = ("A1", "A2", "A3", "A4", "A5", "A6")
# The ladder runs use one fixed selection variant (epic first-k) so rung
# deltas are attributable to placement alone; the variant sweep below
# then isolates the selection rule at fixed placement (A6).
LADDER_POLICY = ("epic", ["--reuse", "epic", "--epic-prefix-recompute-tokens", "8"])


def policy_args(name, ndec):
    partial_all = ",".join(str(i) for i in range(1, ndec))
    if name == "epic":
        return ["--reuse", "epic", "--epic-prefix-recompute-tokens", "8"]
    if name == "cacheblend":
        return ["--reuse", "cacheblend", "--cacheblend-recompute-ratio", "0.15",
                "--cacheblend-full-layers", "0",
                "--cacheblend-partial-layers", partial_all]
    if name == "cachecraft":
        return ["--reuse", "cachecraft", "--cachecraft-alpha", "0.1"]
    if name == "cachetune":
        return ["--reuse", "cachetune", "--cacheblend-recompute-ratio", "0.15",
                "--cacheblend-partial-layers",
                ",".join(str(i) for i in range(ndec))]
    raise ValueError(name)


def jobs():
    result = []
    # 1) The ladder: same workload, six rungs, three models.
    for wl, model, rung in itertools.product(WORKLOADS, MODELS, RUNGS):
        tag = "ladder_{}_{}_{}".format(wl, model, rung)
        if rung == "A1":
            reuse = ["--reuse", "no-reuse"]
        else:
            reuse = list(LADDER_POLICY[1])
        result.append((tag, wl, model, ["--ablation", rung] + reuse))
    # 2) Selection-variant sweep at the paper's point (A6).
    for wl, model, policy in itertools.product(
            WORKLOADS, MODELS, ("cacheblend", "cachecraft", "cachetune")):
        tag = "select_{}_{}_A6_{}".format(wl, model, policy)
        result.append((tag, wl, model,
                       ["--ablation", "A6"] + policy_args(policy, MODELS[model])))
    # 3) Physical-DAG dynamic-side fraction (event path; small models only,
    #    the event DAG on 80-96 layer models is out of smoke budget).
    for wl in WORKLOADS:
        for model in ("CACHEBLEND-TINY", "LLAMA-7B"):
            tag = "dag_{}_{}_dynamic".format(wl, model)
            result.append((tag, wl, model,
                           list(LADDER_POLICY[1]) + ["--pim-prefill-mode", "dynamic"]))
    return result


def run_one(job, timeout_s):
    tag, wl, model, extra = job
    out = os.path.join(RESULTS, tag + ".json")
    if os.path.exists(out):
        return tag, "cached"
    cmd = [sys.executable, "main.py", "--system", "dgx-attacc",
           "--model", model,
           "--workload", os.path.join(HERE, WORKLOADS[wl]),
           "--history-len", "3", "--pipeopt",
           "--ramulator-workers", str(RAMULATOR_WORKERS),
           "--workload-report", out] + extra
    log = os.path.join(RESULTS, tag + ".log")
    with open(log, "w") as handle:
        proc = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=handle,
                              timeout=timeout_s)
    if proc.returncode != 0 or not os.path.exists(out):
        return tag, "FAILED(rc={})".format(proc.returncode)
    return tag, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default="",
                        help="substring filter on job tags")
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    selected = [job for job in jobs() if args.only in job[0]]
    print("{} jobs, {} concurrent x {} ramulator workers".format(
        len(selected), args.concurrent, RAMULATOR_WORKERS))
    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = {pool.submit(run_one, job, args.timeout): job[0]
                   for job in selected}
        for future in futures:
            pass
        for future, tag in list(futures.items()):
            try:
                tag, status = future.result()
            except Exception as exc:  # timeout etc.
                status = "FAILED({})".format(type(exc).__name__)
            print(tag, status, flush=True)
            if "FAILED" in status:
                failures.append(tag)
    print("done; {} failures".format(len(failures)))
    for tag in failures:
        print("  " + tag)


if __name__ == "__main__":
    main()

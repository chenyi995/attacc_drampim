"""MQ-MAC batch-command study (PLAN_mq_command.md O5/O6).

Direct wrapper-level driver: every data point is one patched-Ramulator2 run of
a bank-level AttAcc attention sweep (score + softmax + context phases) over an
L-token K/V segment held by one head's channel, per-HBM view (nhead=1,
num_hbm=1).

Three schemes per (n queries, L):
  * dense      -- n independent single-query sweeps, each over its own private
                  copy of the same L tokens (the AttAcc reference shape: one
                  KV copy per agent, no sharing).  Time is the sum of the runs
                  (per-channel scans of one head serialize).
  * replicate  -- one shared sweep, legacy trace expansion: one MAC_AB per
                  (column, query), each re-reading the column.
  * mq         -- one shared sweep, MQ-MAC command: one MAC_AB per column
                  serves every resident Q; the command interval carries the
                  n-fold PE time and the IDD7 power stretch
                  (ramulator_wrapper.mq_interval_cycles).

Sweeps larger than the GEMV-buffer capacity (64 B per query slice) split into
consecutive passes, for replicate and mq alike.

Host-side derived metrics per data point:
  * act_per_pass  -- all-bank row activations of one sweep pass per bank:
                     ceil(L / 256) (one channel row holds 256 tokens).
  * mac_cmds      -- MAC_AB commands actually issued (from the run's DRAM
                     read traffic); DRAM read energy is proportional to it.
  * pe_util       -- useful PE ops / (measured cycles x PE ops per cycle),
                     with one op = one 16-lane MAC per column per query and
                     the PE clock in command-cycle units.
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model import Layer
from src.ramulator_wrapper import (Ramulator, mq_interval_cycles,
                                   mq_query_capacity)
from src.type import DataType, LayerType, PIMType

_TCK_NS = 0.769
_TOKENS_PER_CHANNEL_ROW = 256
_MEM_FANOUT_BA = 2 * 2 * 4 * 4  # pCH x rank x BG x bank (wrapper postprocess)


def _scan(ram, *, n, length, mode, power_constraint, pe_freq_ghz):
    """One sweep pass: n queries share one L-token resident K/V segment."""
    op = Layer(0, "score", LayerType.MATMUL, False, DataType.W16A16,
               1, length, 128, 1)
    op.pim_kv_runs = ((0x0, 0x800000, length, 0, 16),)
    op.pim_shared_kv = n > 1
    op.pim_shared_queries = n
    op.pim_batch_command = mode
    op.pim_pe_freq_ghz = pe_freq_ghz
    time_s, traffic = ram.run(PIMType.BA, op, power_constraint=power_constraint,
                              record_log=False)
    mac_cmds = int(round(traffic[-1] / (32 * _MEM_FANOUT_BA)))
    return time_s, mac_cmds


def _measure(ram, *, n, length, mode, power_constraint, pe_freq_ghz, cap):
    """Full service of n queries under one scheme, splitting sweeps at cap."""
    if mode == "dense":
        passes = [1] * n
        mode_used = "replicate"
    else:
        passes = [min(cap, n - start) for start in range(0, n, cap)]
        mode_used = mode
    total_s, total_macs = 0.0, 0
    for pass_n in passes:
        time_s, mac_cmds = _scan(ram, n=pass_n, length=length, mode=mode_used,
                                 power_constraint=power_constraint,
                                 pe_freq_ghz=pe_freq_ghz)
        total_s += time_s
        total_macs += mac_cmds
    act_per_bank = math.ceil(length / _TOKENS_PER_CHANNEL_ROW)
    acts = act_per_bank * len(passes)
    # Useful PE ops: every query multiplies every K and V column once.
    # Column count per channel = score (2L/16) + context (2L/16) commands of
    # the single-Q trace = the measured single-sweep mac_cmds at n = 1.
    cols = None
    return {
        "scheme": mode, "n": n, "L": length,
        "passes": len(passes), "time_ns": total_s * 1e9,
        "mac_cmds": total_macs, "act_allbank": acts,
    }


def _pe_util(row, single_cols, pe_freq_ghz):
    ops = row["n"] * single_cols          # one PE op per (column, query)
    cycles = row["time_ns"] / _TCK_NS
    ops_per_cycle = pe_freq_ghz * _TCK_NS  # PE ops per command cycle
    return ops / (cycles * ops_per_cycle) if cycles else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "results_mq_study.json"))
    parser.add_argument("--pe-freq-ghz", type=float, nargs="*",
                        default=[0.666, 1.3])
    parser.add_argument("--gemv-buffer-bytes", type=int, default=512)
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo)
    ram = Ramulator({"num_heads": 1, "dhead": 128}, "ramulator2",
                    output_log="", num_hbm=1, workers=1)
    cap = mq_query_capacity(args.gemv_buffer_bytes)

    jobs = []
    # Part A: interval model vs measurement (mq, one pass, n <= cap).
    for pc in (True, False):
        for pe in args.pe_freq_ghz:
            for n in (1, 2, 3, 4, 6, 8):
                jobs.append(("A", dict(n=n, length=4096, mode="mq",
                                       power_constraint=pc, pe_freq_ghz=pe)))
    # Part B: decode, N_ag agents share one chunk (PC, both PE clocks).
    for pe in args.pe_freq_ghz:
        for length in (1024, 4096):
            for n in (2, 4, 8, 16):
                for mode in ("dense", "replicate", "mq"):
                    jobs.append(("B", dict(n=n, length=length, mode=mode,
                                           power_constraint=True,
                                           pe_freq_ghz=pe)))
    # Part C: prefill, n_r queries over one reused segment (PC).
    for pe in args.pe_freq_ghz:
        for n in (4, 8, 16, 32):
            for mode in ("dense", "replicate", "mq"):
                jobs.append(("C", dict(n=n, length=4096, mode=mode,
                                       power_constraint=True,
                                       pe_freq_ghz=pe)))

    def run_job(job):
        part, kw = job
        row = _measure(ram, cap=cap, **kw)
        row.update(part=part, power_constraint=kw["power_constraint"],
                   pe_freq_ghz=kw["pe_freq_ghz"])
        if kw["mode"] == "mq":
            row["interval_model_cycles"] = mq_interval_cycles(
                min(kw["n"], cap), kw["power_constraint"], kw["pe_freq_ghz"])
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))

    # PE utilisation needs the single-query column count per L.
    single_cols = {}
    for row in rows:
        if row["scheme"] != "mq" or row["passes"] != 1 or row["n"] != 1:
            continue
        single_cols[row["L"]] = row["mac_cmds"]
    for length in {row["L"] for row in rows}:
        if length not in single_cols:
            _, cols = _scan(ram, n=1, length=length, mode="replicate",
                            power_constraint=True, pe_freq_ghz=0.666)
            single_cols[length] = cols
    for row in rows:
        row["pe_util"] = round(_pe_util(row, single_cols[row["L"]],
                                        row["pe_freq_ghz"]), 4)

    with open(args.out, "w") as handle:
        json.dump({"gemv_buffer_bytes": args.gemv_buffer_bytes,
                   "sweep_query_capacity": cap,
                   "single_query_columns": single_cols,
                   "rows": rows}, handle, indent=2, sort_keys=True)
    print("wrote {} rows to {}".format(len(rows), args.out))
    print("ramulator invocations:", ram.cache_report()["ramulator_invocations"])


if __name__ == "__main__":
    main()

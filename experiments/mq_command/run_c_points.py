"""C1/C2/C3 batch-command comparison (cycle-accurate, per-channel view).

The C numbering (user ruling 2026-08-21):

  C1  one shared compact copy in the bank, NO frequency boost: the original
      AttAcc behaviour -- N agents take turns, each running one full
      single-query sweep (score + softmax + context) over the L-token
      segment.  latency = N x t1, storage = 1x.
  C2  multi-channel replication, NO frequency boost: the chunk is copied
      into k channel groups and k agents scan concurrently.
      latency = ceil(N/k) x t1 (analytic from the measured t1),
      storage = k x.  Its concurrency also presupposes idle channels;
      with heads >= channels it degenerates to C1.
  C3  our MQ acceleration: the MQ-MAC command (one MAC_AB per column serves
      every resident Q) with a pipelined PE whose clock may be raised above
      AttAcc's synthesized 666 MHz.  The score phase runs once with n_q
      resident queries; the context phase ALSO runs once with the same n_q
      (streaming-P revision, ruling 2026-08-24) -- probability vectors are
      NOT resident: a P entry has (almost) no per-bank reuse (one scalar per
      V column per output pass), so each query's P streams through the
      double-buffered GEMV-buffer halves via MV_GB.  Its bound is the
      movement-bus bandwidth (32 B per nBL tCK per pCH, the stack-level
      1024-bit @ 5.2 Gbps TSV path) plus the MVSB<->MVGB direction
      turnaround (nRTW/nWTRL), all physically priced inside the Ramulator
      run.  storage = 1x; K and V are each read once for ALL n_q queries.

Buffer bookkeeping behind n_q: only the Q side is capacity-bound -- a Q
slice is 64 B/bank, so a GEMV buffer of S bytes holds n_q = S/64.  The
stock 512-B buffer already holds 8 queries at no hardware cost; n_q = 16
needs S = 1 KiB (2x AttAcc, ~12.2% die overhead), n_q = 32 needs S = 2 KiB
(4x, ~15.0%) -- the capacity axis costs SRAM + PE area and power.  The
trace still orders the MV_GB block before the context MACs with a barrier,
i.e. the context phase is priced load-then-compute (conservative: the real
double buffer overlaps P delivery with the V scan).

Every measured number is one patched-Ramulator2 run (nhead=1, num_hbm=1,
power-constrained HBM3_5.2Gbps).
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model import Layer
from src.ramulator_wrapper import Ramulator, mq_interval_cycles
from src.type import DataType, LayerType, PIMType

_TOKENS_PER_CHANNEL_ROW = 256
_MEM_FANOUT_BA = 2 * 2 * 4 * 4


def _scan(ram, *, n, length, phase, mode, pe_freq_ghz, power_constraint=True):
    op = Layer(0, "score", LayerType.MATMUL, False, DataType.W16A16,
               1, length, 128, 1)
    op.pim_kv_runs = ((0x0, 0x800000, length, 0, 16),)
    op.pim_shared_kv = n > 1
    op.pim_shared_queries = n
    op.pim_batch_command = mode
    op.pim_pe_freq_ghz = pe_freq_ghz
    op.pim_phase = phase
    time_s, traffic = ram.run(PIMType.BA, op, power_constraint=power_constraint,
                              record_log=False)
    return time_s * 1e9, int(round(traffic[-1] / (32 * _MEM_FANOUT_BA)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--pe-freq-ghz", type=float, nargs="*",
                        default=[0.666, 1.3, 2.08, 3.2],
                        help="C3 PE clocks; 0.666 GHz is AttAcc's synthesized "
                             "point, everything above is the pipelined-PE "
                             "assumption")
    parser.add_argument("--points", type=int, nargs="*", default=[8, 16, 32],
                        help="C3 resident-query counts n_q (P is streamed, "
                             "not resident; 8 fits the stock 512-B buffer)")
    parser.add_argument("--c2-copies", type=int, nargs="*", default=[2, 4, 8],
                        help="C2 replication factors k reported per point")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "results_c_points.json"))
    args = parser.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo)
    ram = Ramulator({"num_heads": 1, "dhead": 128}, "ramulator2",
                    output_log="", num_hbm=1, workers=1)
    length = args.length
    act_rows = math.ceil(length / _TOKENS_PER_CHANNEL_ROW)

    # C1's building block: one single-query full sweep at the original clock.
    t1_ns, t1_macs = _scan(ram, n=1, length=length, phase="full",
                           mode="replicate", pe_freq_ghz=0.666)

    rows = []
    for n_q in args.points:
        c1_ns = n_q * t1_ns
        c2 = {k: round(math.ceil(n_q / k) * t1_ns, 1) for k in args.c2_copies}
        for pe in args.pe_freq_ghz:
            score_ns, score_macs = _scan(ram, n=n_q, length=length,
                                         phase="score", mode="mq",
                                         pe_freq_ghz=pe)
            # Streaming-P: ONE context pass with all n_q queries.  The n_q-fold
            # MV_GB stream (P delivery over the movement bus, direction
            # turnaround included) is inside this run; the V columns are read
            # once for all queries.
            ctx_ns, ctx_macs = _scan(ram, n=n_q, length=length,
                                     phase="context", mode="mq",
                                     pe_freq_ghz=pe)
            total_ns = score_ns + ctx_ns
            rows.append({
                "n_agents": n_q, "n_q": n_q, "L": length,
                "p_residency": "streamed",
                "pe_freq_ghz": pe,
                "buffer_bytes": n_q * 64,
                "interval_cycles": mq_interval_cycles(n_q, True, pe),
                "c3_score_ns": round(score_ns, 1),
                "c3_context_ns": round(ctx_ns, 1),
                "c3_total_ns": round(total_ns, 1),
                "c3_per_agent_ns": round(total_ns / n_q, 1),
                "c3_mac_cmds": score_macs + ctx_macs,
                "c3_act_allbank": act_rows * 2,
                "c1_ns": round(c1_ns, 1),
                "c1_mac_cmds": n_q * t1_macs,
                "c1_act_allbank": n_q * 2 * act_rows,
                "c2_ns_by_copies": c2,
                "c3_speedup_vs_c1": round(c1_ns / total_ns, 2),
                "c2_equal_latency_copies": math.ceil(c1_ns / total_ns),
            })
    with open(args.out, "w") as handle:
        json.dump({"t1_ns": t1_ns, "t1_mac_cmds": t1_macs,
                   "c2_model": "ceil(N/k) x t1, storage k-fold",
                   "rows": rows}, handle, indent=2, sort_keys=True)
    print("C3 n_q  pe(GHz)  itv    score_ns  context_ns    total_ns  /agent  vs C1   C2 equal-k")
    for row in rows:
        print("{n_q:>4}    {pe_freq_ghz:>5}   {interval_cycles:>2}   {c3_score_ns:>9.0f}  "
              "{c3_context_ns:>10.0f}   {c3_total_ns:>9.0f}  "
              "{c3_per_agent_ns:>6.0f}  {c3_speedup_vs_c1:>5.2f}x   k={c2_equal_latency_copies}".format(**row))
    print("t1 = {:.0f} ns (C1 = N x t1; C2 = ceil(N/k) x t1); wrote {}".format(
        t1_ns, args.out))


if __name__ == "__main__":
    main()

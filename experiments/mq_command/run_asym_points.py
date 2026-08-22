"""Asymmetric MQ sweep measurement: the (n_q, n_c) = (16, 2) and (32, 4) points.

One asymmetric sweep serving n_q agents over an L-token shared segment is
composed of two kinds of cycle-accurate Ramulator runs (PLAN_mq_command.md,
extended scope):

  * one score-phase run   -- WRGB + score MACs + MVSB + SFM, n_q resident Qs,
                             MAC interval = mq_interval_cycles(n_q, ...);
  * ceil(n_q/n_c) context-phase runs -- MVGB + context MACs + MVSB, n_c
                             resident P vectors each, interval from n_c.

Buffer bookkeeping (the reason for the split): a Q slice is 64 B/bank, a P
slice is L/8 B/bank, so a buffer of S bytes holds n_q = S/64 score-side
queries but only n_c = 8S/L context-side probability vectors.
(16, 2) needs S = 1 KiB (2x AttAcc's GEMV buffer, ~12.2% die overhead);
(32, 4) needs S = 2 KiB (4x, ~15.0%).

Baselines per point: B1 compact = n_q serial single-query full sweeps of the
one shared copy; B2 multichannel(k) = ceil(n_q/k) x t1 with k replicated
copies (analytic from the measured t1).
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
                        default=[0.666, 1.3, 2.08, 3.2])
    parser.add_argument("--points", type=str, nargs="*", default=["16,2", "32,4"])
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "results_asym_points.json"))
    args = parser.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo)
    ram = Ramulator({"num_heads": 1, "dhead": 128}, "ramulator2",
                    output_log="", num_hbm=1, workers=1)
    length = args.length
    act_rows = math.ceil(length / _TOKENS_PER_CHANNEL_ROW)

    t1_ns, t1_macs = _scan(ram, n=1, length=length, phase="full",
                           mode="replicate", pe_freq_ghz=0.666)
    rows = []
    for point in args.points:
        n_q, n_c = (int(part) for part in point.split(","))
        passes = math.ceil(n_q / n_c)
        buffer_bytes = max(n_q * 64, n_c * length // 8)
        for pe in args.pe_freq_ghz:
            score_ns, score_macs = _scan(ram, n=n_q, length=length,
                                         phase="score", mode="mq",
                                         pe_freq_ghz=pe)
            ctx_ns, ctx_macs = _scan(ram, n=n_c, length=length,
                                     phase="context", mode="mq",
                                     pe_freq_ghz=pe)
            total_ns = score_ns + passes * ctx_ns
            b1_ns = n_q * t1_ns
            rows.append({
                "n_q": n_q, "n_c": n_c, "L": length, "pe_freq_ghz": pe,
                "buffer_bytes": buffer_bytes,
                "interval_score": mq_interval_cycles(n_q, True, pe),
                "interval_context": mq_interval_cycles(n_c, True, pe),
                "score_ns": round(score_ns, 1),
                "context_pass_ns": round(ctx_ns, 1), "passes": passes,
                "total_ns": round(total_ns, 1),
                "per_agent_ns": round(total_ns / n_q, 1),
                "mac_cmds": score_macs + passes * ctx_macs,
                "act_allbank": act_rows * (1 + passes),
                "b1_compact_ns": round(b1_ns, 1),
                "b1_mac_cmds": n_q * t1_macs,
                "b1_act_allbank": n_q * 2 * act_rows,
                "speedup_vs_b1": round(b1_ns / total_ns, 2),
                "b2_equal_latency_copies": math.ceil(b1_ns / total_ns),
            })
    with open(args.out, "w") as handle:
        json.dump({"t1_ns": t1_ns, "t1_mac_cmds": t1_macs, "rows": rows},
                  handle, indent=2, sort_keys=True)
    header = ("point      pe(GHz)  itv(s/c)  score_ns  ctx_ns x passes   "
              "total_ns  /agent   vs B1   MACcmd(B1)   ACT(B1)")
    print(header)
    for row in rows:
        print("({n_q:>2},{n_c:>2})   {pe_freq_ghz:>5}   {interval_score:>2}/{interval_context:<2}   "
              "{score_ns:>8.0f}  {context_pass_ns:>7.0f} x {passes:<2}   {total_ns:>9.0f}  "
              "{per_agent_ns:>6.0f}  {speedup_vs_b1:>5.2f}x  {mac_cmds:>5}({b1_mac_cmds:>6})  "
              "{act_allbank:>3}({b1_act_allbank:>4})".format(**row))
    print("t1 = {:.0f} ns; wrote {}".format(t1_ns, args.out))


if __name__ == "__main__":
    main()

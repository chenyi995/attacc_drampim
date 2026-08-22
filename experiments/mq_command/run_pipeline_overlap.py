"""Two-head pipeline overlap on the shared movement bus (turnaround study).

Question (2026-08-21): head i's context phase (MVGB down) and head i+1's score
phase (MVSB up) share one half-duplex TSV/GBUS path.  The model already
serializes every movement command at nBL granularity; this experiment adds the
just-introduced direction-turnaround constraints (MVSB->MVGB/WRGB = nRTW,
MVGB/WRGB->MVSB = nWTRL) and measures what two-head pipelining really costs.

Setup: one channel (--channels 1), two heads (--nhead 2) -> the generator's
original AttAcc two-head pipelined assembly runs both sweeps on the same
channel with score(head1) interleaved into softmax/context(head0).  MQ mode
with symmetric n queries per head.  Reference = 2x the single-head sweep
(same channel, same n), i.e. fully serial heads.

Swept: n in {1, 4, 8, 16}; turnaround (nRTW, nWTRL) in
  (0,0)   -- no-penalty, the pre-change model,
  (3,11)  -- HBM3_5.2Gbps preset values (JEDEC-grounded default),
  (12,44) -- 4x exaggerated, sensitivity bound.
Metrics per point: pipelined cycles, serial cycles, overlap benefit
(serial/pipelined), turnaround cost vs (0,0).
"""

import json
import math
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAM = os.path.join(REPO, "ramulator2", "ramulator2")
GEN = os.path.join(REPO, "ramulator2", "trace_gen", "gen_trace_attacc_bank.py")
OUT = os.path.join(os.path.dirname(__file__), "results_pipeline_overlap.json")
SCRATCH = os.path.join(os.path.dirname(__file__), ".overlap_tmp")
_TCK_NS = 0.769

sys.path.insert(0, REPO)
from src.ramulator_wrapper import mq_interval_cycles  # noqa: E402

YAML = """Frontend:
  impl: PIMLoadStoreTrace
  path: {trace}
  clock_ratio: 1

  Translation:
    impl: NoTranslation
    max_addr: 2147483648

MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: HBM3_5.2Gbps
      nCCDAB: {nccdab}
      nRTW: {nrtw}
      nWTRL: {nwtrl}

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
    plugins:

  AddrMapper:
    impl: HBM3-PIM
"""


def _run(nhead, n, length, nrtw, nwtrl, pe_freq_ghz=1.3):
    tag = "h{}_n{}_t{}_{}".format(nhead, n, nrtw, nwtrl)
    trace = os.path.join(SCRATCH, tag + ".trace")
    yaml = os.path.join(SCRATCH, tag + ".yaml")
    cmd = ["python3", GEN, "--dhead", "128", "--nhead", str(nhead),
           "--seqlen", str(length), "--dbyte", "2", "--channels", "1",
           "--output", trace]
    if n > 1:
        cmd += ["--shared-kv", "--shared-queries", str(n), "--mq"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    nccdab = mq_interval_cycles(n, True, pe_freq_ghz)
    with open(yaml, "w") as handle:
        handle.write(YAML.format(trace=trace, nccdab=nccdab,
                                 nrtw=nrtw, nwtrl=nwtrl))
    output = subprocess.run([RAM, "-f", yaml], check=True,
                            capture_output=True, text=True).stdout
    for line in output.splitlines():
        if "memory_system_cycles" in line:
            return int(line.split()[-1])
    raise RuntimeError("no cycle count for " + tag)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    length = 4096
    rows = []
    for n in (1, 4, 8, 16):
        for nrtw, nwtrl in ((0, 0), (3, 11), (12, 44)):
            single = _run(1, n, length, nrtw, nwtrl)
            pipe = _run(2, n, length, nrtw, nwtrl)
            rows.append({
                "n": n, "nrtw": nrtw, "nwtrl": nwtrl,
                "single_head_cycles": single,
                "serial_two_heads_cycles": 2 * single,
                "pipelined_two_heads_cycles": pipe,
                "overlap_benefit": round(2 * single / pipe, 3),
                "pipelined_us": round(pipe * _TCK_NS / 1000, 2),
            })
    base = {(row["n"]): row["pipelined_two_heads_cycles"]
            for row in rows if (row["nrtw"], row["nwtrl"]) == (0, 0)}
    for row in rows:
        row["turnaround_cost_pct"] = round(
            100.0 * (row["pipelined_two_heads_cycles"] / base[row["n"]] - 1), 2)
    with open(OUT, "w") as handle:
        json.dump({"length": length, "pe_freq_ghz": 1.3, "rows": rows},
                  handle, indent=2, sort_keys=True)
    print("n   nRTW/nWTRL   1-head cyc   serial cyc   pipelined cyc   overlap x   turnaround +%")
    for row in rows:
        print("{n:>2}   {nrtw:>2}/{nwtrl:<3}      {single_head_cycles:>8}   "
              "{serial_two_heads_cycles:>9}   {pipelined_two_heads_cycles:>10}   "
              "{overlap_benefit:>7}   {turnaround_cost_pct:>6}".format(**row))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

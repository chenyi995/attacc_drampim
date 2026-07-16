# xinyao_0715 CPU-parallel Ramulator run

Branch: `xinyao_0715`

This version treats `--sim-cores` / `--num-cores` as host CPU simulation workers, not simulated CPU cores. The wrapper generates one full trace, splits that trace into worker shards, runs multiple Ramulator processes in parallel, and sums the Ramulator statistics back into one workload result.

Commands:

```bash
python3 main.py --system dgx-attacc --pim bank --model LLAMA-7B --lin 17 --lout 2 --batch 1 --num-agent 8 --sim-cores 1 --rerun-ramulator
python3 main.py --system dgx-attacc --pim bank --model LLAMA-7B --lin 17 --lout 2 --batch 1 --num-agent 8 --sim-cores 2 --rerun-ramulator
```

Ramulator rows:

| sim workers | L | nhead | cycle | mac | softmax | mvgb | mvsb | wrgb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18 | 7 | 4215 | 192 | 7 | 56 | 233 | 56 |
| 2 | 18 | 7 | 4235 | 192 | 7 | 56 | 233 | 56 |

End-to-end output:

| sim workers | generation latency | throughput |
| ---: | ---: | ---: |
| 1 | 4.186487 ms | 238.87 tokens/s |
| 2 | 4.186871 ms | 238.84 tokens/s |

Conclusion: request totals are unchanged with host CPU parallelism (`mac`, `softmax`, `mvgb`, `mvsb`, and `wrgb` match exactly). Simulated latency is effectively unchanged; the 20-cycle difference is from splitting a tiny trace into independent Ramulator shards and summing shard cycles. On this very small run, wall-clock speedup is not meaningful, but the implementation no longer changes simulated request volume or capacity semantics.

Archived files:

- `sim_cores_1_output.csv`
- `sim_cores_2_output.csv`

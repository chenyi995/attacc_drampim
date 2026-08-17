# 2026-08-14 end-to-end no-reuse baseline

Both supplied JSON workloads were run without reuse through the unmodified
`System.simulate` baseline entry point, on `dgx-attacc`, eight A100a GPUs and
bank-level PIM, with `--pipeopt --ffopt`.

| Workload | Tier batch | Input / output | Throughput | Latency |
| --- | ---: | ---: | ---: | ---: |
| `workload_2wikimqa_first8.json` | 8 | 12,834 / 10 | 394.36 tokens/s | 20.29 ms |
| `workload_relay_s400w4t1.json`, supervisor | 1 | 550 / 150 | 60.24 tokens/s | 16.60 ms |
| `workload_relay_s400w4t1.json`, workers | 4 | 700 / 150 | 216.45 tokens/s | 18.48 ms |

The CSV records are `rag_no_reuse.csv` and `relay_no_reuse.csv`.  The RAG
workload is one dependency tier.  The relay workload ran tier 0 before tier 1;
the latter is the four-worker batch.  Ramulator generated the required
bank-level traces for every simulated sequence length during these runs.

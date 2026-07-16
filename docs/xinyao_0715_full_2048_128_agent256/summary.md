# Full run: inputlen=2048 outputlen=128 agent=256

Command run:

```bash
python3 main.py --system dgx-attacc --pim bank --lin 2048 --lout 128 --batch 1 --num-agent 256 --sim-cores 48 --rerun-ramulator
```

Assumptions: model was not specified, so this used `main.py` default `GPT-175B`, `A100a`, `ngpu=8`, `pim=bank`, `powerlimit=False`, RoPE enabled.

## Output summary

| metric | value |
| --- | ---: |
| host sim workers | 48 |
| model | GPT-175B |
| Lin / Lout / batch | 2048 / 128 / 1 |
| num_agent | 256 |
| cap field | 1280 GB |
| required_cap | 1,672,683,540,377.462 bytes |
| required_cap decimal | 1672.684 GB |
| required_cap binary | 1557.808 GiB |
| throughput | 16.19 tokens/s |
| generation latency | 61.774029 ms |
| generation matmul | 34.737645 ms |
| generation FC | 16.728382 ms |
| generation comm | 8.182127 ms |
| generation energy | 23,171,039,664.484 nJ |

## Capacity decomposition

| component | bytes |
| --- | ---: |
| weights | 347,892,350,976 |
| KV cache conventional part | 5,131,468,800 |
| temp | 1,044,000 |
| RoPE storage | 1,319,658,676,601.462 |
| total | 1,672,683,540,377.462 |

Conclusion: capacity calculation is internally consistent with the current code, but the requested configuration does not fit the configured aggregate capacity (`required_cap` is about 1.67 TB vs `cap=1280 GB`). Most of the extra capacity is RoPE storage for `num_agent=256`.

## Ramulator coverage

Matched Ramulator rows: 127, covering `L=2049..2175`, `nhead=615`, `dhead=128`, `dbyte=2`, `pim_type=BA`, `power_constraint=False`, `rope=True`, `num_agent=256`.

Sample rows:

| L | cycle | mac | softmax | mvgb | mvsb | wrgb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2049 | 467152 | 328684 | 615 | 44242 | 59525 | 4882 |
| 2112 | 469622 | 336016 | 615 | 44242 | 59509 | 4882 |
| 2175 | 474720 | 346336 | 615 | 44242 | 59669 | 4882 |

## Correctness note

Capacity, model/system parameters, request counts, traffic-derived energy, and the row coverage above are consistent with the current model and wrapper. The current host parallelism accelerates simulation by splitting a generated trace across Ramulator worker processes, so aggregate request counts are preserved because each trace line is processed once. Cycle/latency should be treated as a parallel-shard approximation rather than a bit-exact single full-trace Ramulator timing result; for exact timing comparison, run `--sim-cores 1` or move parallelism inside Ramulator's event loop/controllers.

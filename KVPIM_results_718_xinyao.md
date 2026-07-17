# 0718 KV-PIM Results

Units:

- Latency is `g_time (ms)`.
- Capacity values are decimal GB.
- This run uses the modified simulator where both K and V use the same Master+Diff storage model.
- PIM/AttAcc HBM capacity is set to 30 GiB per HBM stack.
- The CSV field names are still `k_row_cap` and `k_data_cap`, but in this run they represent K+V Master+Diff row-occupied capacity and raw data capacity.
- AttAcc dense baseline numbers are reused from `KVPIM_results_715_xinyao.md`; no dense `--no-rope` baseline was rerun.
- `AttAcc KV data` is dense per-agent K+V storage: `N x 10.263 GB` at `l = 2175`.
- `Slowdown vs AttAcc` is `(Latency - AttAcc latency) / AttAcc latency x 100`.
- `KV row compression` is `AttAcc KV data / KV row`; `KV data compression` is `AttAcc KV data / KV data`.

## lin=2048 lout=128 Sweep

Configuration: `model=GPT-175B`, `Lin=2048`, `Lout=128`, `batch=1`, `token_block=32`, `sim_cores=8`.

| Agent | Pb | Latency (ms) | Required (GB) | KV row (GB) | KV data (GB) | AttAcc latency (ms) | AttAcc KV data (GB) | Slowdown vs AttAcc (%) | KV row compression (x) | KV data compression (x) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.05 | 29.335135 | 431.203 | 83.309 | 43.104 | 29.332657 | 656.828 | +0.0084 | 7.88 | 15.24 |
| 64 | 0.10 | 29.341092 | 455.357 | 107.463 | 75.946 | 29.332657 | 656.828 | +0.0288 | 6.11 | 8.65 |
| 64 | 0.15 | 29.349186 | 494.737 | 146.843 | 108.787 | 29.332657 | 656.828 | +0.0563 | 4.47 | 6.04 |
| 64 | 0.20 | 29.352971 | 524.961 | 177.068 | 141.629 | 29.332657 | 656.828 | +0.0693 | 3.71 | 4.64 |
| 64 | 0.25 | 29.357631 | 558.309 | 210.415 | 174.470 | 29.332657 | 656.828 | +0.0851 | 3.12 | 3.76 |
| 64 | 0.30 | 29.361583 | 591.493 | 243.600 | 207.311 | 29.332657 | 656.828 | +0.0986 | 2.70 | 3.17 |
| 64 | 0.35 | 29.366733 | 623.833 | 275.939 | 240.153 | 29.332657 | 656.828 | +0.1162 | 2.38 | 2.74 |
| 64 | 0.40 | 29.371876 | 657.157 | 309.263 | 272.994 | 29.332657 | 656.828 | +0.1337 | 2.12 | 2.41 |
| 64 | 0.45 | 29.373941 | 689.668 | 341.775 | 305.836 | 29.332657 | 656.828 | +0.1407 | 1.92 | 2.15 |
| 64 | 0.50 | 29.379909 | 722.728 | 374.834 | 338.677 | 29.332657 | 656.828 | +0.1611 | 1.75 | 1.94 |
| 64 | 0.55 | 29.385897 | 755.537 | 407.644 | 371.518 | 29.332657 | 656.828 | +0.1815 | 1.61 | 1.77 |
| 64 | 0.60 | 29.388334 | 788.257 | 440.364 | 404.360 | 29.332657 | 656.828 | +0.1898 | 1.49 | 1.62 |
| 256 | 0.05 | 36.047806 | 525.416 | 177.523 | 141.629 | 36.037025 | 2627.312 | +0.0299 | 14.80 | 18.55 |
| 256 | 0.10 | 36.067720 | 656.954 | 309.061 | 272.994 | 36.037025 | 2627.312 | +0.0852 | 8.50 | 9.62 |
| 256 | 0.15 | 36.090404 | 788.375 | 440.482 | 404.360 | 36.037025 | 2627.312 | +0.1481 | 5.96 | 6.50 |
| 256 | 0.20 | 36.111763 | 919.802 | 571.909 | 535.725 | 36.037025 | 2627.312 | +0.2074 | 4.59 | 4.90 |
| 256 | 0.25 | 36.133685 | 1051.228 | 703.334 | 667.091 | 36.037025 | 2627.312 | +0.2682 | 3.74 | 3.94 |
| 256 | 0.30 | 36.158469 | 1182.654 | 834.760 | 798.457 | 36.037025 | 2627.312 | +0.3370 | 3.15 | 3.29 |
| 256 | 0.35 | 36.178624 | 1314.080 | 966.186 | 929.822 | 36.037025 | 2627.312 | +0.3929 | 2.72 | 2.83 |
| 256 | 0.40 | 36.197204 | 1445.506 | 1097.612 | 1061.188 | 36.037025 | 2627.312 | +0.4445 | 2.39 | 2.48 |
| 256 | 0.45 | 36.213000 | 1576.932 | 1229.038 | 1192.553 | 36.037025 | 2627.312 | +0.4883 | 2.14 | 2.20 |
| 256 | 0.50 | 36.232957 | 1708.358 | 1360.464 | 1323.919 | 36.037025 | 2627.312 | +0.5437 | 1.93 | 1.98 |
| 256 | 0.55 | 36.249636 | 1839.784 | 1491.890 | 1455.285 | 36.037025 | 2627.312 | +0.5900 | 1.76 | 1.81 |
| 256 | 0.60 | 36.271009 | 1971.210 | 1623.316 | 1586.650 | 36.037025 | 2627.312 | +0.6493 | 1.62 | 1.66 |

## Max Decoding Length

Configuration: `model=GPT-175B`, `Lin=2048`, `batch=1`, `agent=256`, `token_block=32`, `sim_cores=8`, PIM HBM capacity = `30 GiB / HBM`.

Total capacity: `1975.685 GB`.

Dense AttAcc baseline: `max lout = 0` at `Lin=2048`, `agent=256`. The required dense capacity at `lout=1` is `2821.793 GB`, already above the aggregate capacity.

| Pb | Max lout | Max context length | Required (GB) | KV row (GB) | KV data (GB) |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 17889 | 19936 | 1974.323 | 1626.421 | 1298.164 |
| 0.10 | 9409 | 11456 | 1975.011 | 1627.113 | 1437.895 |
| 0.15 | 5985 | 8032 | 1973.793 | 1625.897 | 1493.249 |
| 0.20 | 4129 | 6176 | 1971.106 | 1623.211 | 1521.214 |
| 0.25 | 2977 | 5024 | 1971.770 | 1623.875 | 1540.903 |
| 0.30 | 2177 | 4224 | 1968.312 | 1620.417 | 1550.658 |
| 0.35 | 1601 | 3648 | 1967.677 | 1619.783 | 1559.536 |
| 0.40 | 1153 | 3200 | 1962.030 | 1614.136 | 1561.288 |
| 0.45 | 833 | 2880 | 1974.562 | 1626.669 | 1579.105 |
| 0.50 | 545 | 2592 | 1968.447 | 1620.553 | 1577.746 |
| 0.55 | 321 | 2368 | 1971.421 | 1623.528 | 1584.420 |
| 0.60 | 129 | 2176 | 1971.210 | 1623.316 | 1587.380 |

## AttAcc Dense Baseline (`--no-rope`)

Configuration: dense baseline numbers are reused from `KVPIM_results_715_xinyao.md`: `model=GPT-175B`, `Lin=2048`, `Lout=128`, `batch=1`, `pim=bank`, no `--powerlimit/--pipeopt/--ffopt`.

| Agent | Dense latency (ms) | KV-PIM latency, Pb 0.05-0.60 (ms) | Dense required (GB) | KV-PIM required, Pb 0.05-0.60 (GB) |
|---:|---:|---:|---:|---:|
| 1 | 27.257086 | - | 358.156 | - |
| 64 | 29.332657 | 29.335135 - 29.388334 | 1004.720 | 431.203 - 788.257 |
| 256 | 36.037025 | 36.047806 - 36.271009 | 2975.204 | 525.416 - 1971.210 |

Notes:

- With V included in Master+Diff capacity, the lout=128 required capacity increases relative to 0715, but remains below dense AttAcc for both agent counts across Pb 0.05-0.60.
- Latency overhead vs dense remains small: agent=64 reaches +0.056 ms (+0.19%), and agent=256 reaches +0.234 ms (+0.65%) at `Pb=0.60`.
- With 30 GiB PIM HBM stacks, all agent=256 `Pb=0.05-0.60` lout=128 cases fit within the 1975.685 GB aggregate capacity; at `Pb=0.60`, the maximum is only `lout=129`.
- Max-lout capacity is substantially lower than the 0715 K-only-diff model because V now uses the same Master+Diff storage model as K.
- With 30 GiB PIM HBM stacks, max-lout remains positive through `Pb=0.60`; at `Pb=0.60`, the maximum context length is 2176 tokens.

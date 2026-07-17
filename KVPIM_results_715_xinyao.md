# Pb Sweep Results

Units:

- Latency is `g_time (ms)`.
- Capacity values are decimal GB.
- `Required` uses row-occupied K capacity for overflow checking.
- `K row` is the occupied-row capacity.
- `K data` is the raw MasterK + expected diffK data capacity.
- `AttAcc latency` is the dense `--no-rope` baseline at the same agent count (independent of Pb).
- `AttAcc K data` is the dense per-agent K storage: `N x 5.131 GB` (K at `l = 2175`), independent of Pb.
- `Slowdown vs AttAcc` is `(Latency - AttAcc latency) / AttAcc latency x 100`.
- `K row compression` is `AttAcc K data / K row`; `K data compression` is `AttAcc K data / K data`.

## lin=2048 lout=128 Sweep

Configuration: `model=GPT-175B`, `Lin=2048`, `Lout=128`, `batch=1`, `token_block=32`, `sim_cores=8`.

| Agent | Pb | Latency (ms) | Required (GB) | K row (GB) | K data (GB) | AttAcc latency (ms) | AttAcc K data (GB) | Slowdown vs AttAcc (%) | K row compression (x) | K data compression (x) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.05 | 29.333570 | 394.679 | 41.655 | 21.552 | 29.332657 | 328.414 | +0.0031 | 7.88 | 15.24 |
| 64 | 0.10 | 29.334865 | 406.757 | 53.732 | 37.973 | 29.332657 | 328.414 | +0.0075 | 6.11 | 8.65 |
| 64 | 0.15 | 29.339971 | 426.446 | 73.422 | 54.394 | 29.332657 | 328.414 | +0.0249 | 4.47 | 6.04 |
| 64 | 0.20 | 29.340834 | 441.559 | 88.534 | 70.814 | 29.332657 | 328.414 | +0.0279 | 3.71 | 4.64 |
| 64 | 0.25 | 29.346121 | 458.232 | 105.208 | 87.235 | 29.332657 | 328.414 | +0.0459 | 3.12 | 3.76 |
| 64 | 0.30 | 29.348725 | 474.825 | 121.800 | 103.656 | 29.332657 | 328.414 | +0.0548 | 2.70 | 3.17 |
| 64 | 0.35 | 29.351235 | 490.994 | 137.970 | 120.076 | 29.332657 | 328.414 | +0.0633 | 2.38 | 2.74 |
| 64 | 0.40 | 29.353334 | 507.657 | 154.632 | 136.497 | 29.332657 | 328.414 | +0.0705 | 2.12 | 2.41 |
| 64 | 0.45 | 29.354264 | 523.912 | 170.887 | 152.918 | 29.332657 | 328.414 | +0.0737 | 1.92 | 2.15 |
| 64 | 0.50 | 29.357435 | 540.442 | 187.417 | 169.338 | 29.332657 | 328.414 | +0.0845 | 1.75 | 1.94 |
| 64 | 0.55 | 29.358784 | 556.847 | 203.822 | 185.759 | 29.332657 | 328.414 | +0.0891 | 1.61 | 1.77 |
| 64 | 0.60 | 29.360160 | 573.207 | 220.182 | 202.180 | 29.332657 | 328.414 | +0.0938 | 1.49 | 1.62 |
| 256 | 0.05 | 36.040282 | 441.786 | 88.761 | 70.814 | 36.037025 | 1313.656 | +0.0090 | 14.80 | 18.55 |
| 256 | 0.10 | 36.047338 | 507.555 | 154.530 | 136.497 | 36.037025 | 1313.656 | +0.0286 | 8.50 | 9.62 |
| 256 | 0.15 | 36.057603 | 573.266 | 220.241 | 202.180 | 36.037025 | 1313.656 | +0.0571 | 5.96 | 6.50 |
| 256 | 0.20 | 36.067503 | 638.979 | 285.954 | 267.863 | 36.037025 | 1313.656 | +0.0846 | 4.59 | 4.90 |
| 256 | 0.25 | 36.080220 | 704.692 | 351.667 | 333.545 | 36.037025 | 1313.656 | +0.1199 | 3.74 | 3.94 |
| 256 | 0.30 | 36.089400 | 770.405 | 417.380 | 399.228 | 36.037025 | 1313.656 | +0.1453 | 3.15 | 3.29 |
| 256 | 0.35 | 36.097560 | 836.118 | 483.093 | 464.911 | 36.037025 | 1313.656 | +0.1680 | 2.72 | 2.83 |
| 256 | 0.40 | 36.107679 | 901.831 | 548.806 | 530.594 | 36.037025 | 1313.656 | +0.1961 | 2.39 | 2.48 |
| 256 | 0.45 | 36.115605 | 967.544 | 614.519 | 596.277 | 36.037025 | 1313.656 | +0.2181 | 2.14 | 2.20 |
| 256 | 0.50 | 36.130479 | 1033.257 | 680.232 | 661.959 | 36.037025 | 1313.656 | +0.2593 | 1.93 | 1.98 |
| 256 | 0.55 | 36.142822 | 1098.970 | 745.945 | 727.642 | 36.037025 | 1313.656 | +0.2936 | 1.76 | 1.81 |
| 256 | 0.60 | 36.149736 | 1164.683 | 811.658 | 793.325 | 36.037025 | 1313.656 | +0.3128 | 1.62 | 1.66 |

## Max Decoding Length

Configuration: `model=GPT-175B`, `Lin=2048`, `batch=1`, `agent=256`, `token_block=32`, `sim_cores=8`.

Total capacity: `1374.390 GB`.

| Pb | Max lout | Max context length | Required (GB) | K row (GB) | K data (GB) |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 21729 | 23776 | 1373.846 | 969.848 | 774.106 |
| 0.10 | 11937 | 13984 | 1373.976 | 993.084 | 877.598 |
| 0.15 | 7841 | 9888 | 1372.027 | 1000.801 | 919.152 |
| 0.20 | 5601 | 7648 | 1370.985 | 1005.045 | 941.891 |
| 0.25 | 4193 | 6240 | 1371.075 | 1008.457 | 956.930 |
| 0.30 | 3233 | 5280 | 1373.113 | 1012.761 | 969.161 |
| 0.35 | 2497 | 4544 | 1367.427 | 1008.812 | 971.290 |
| 0.40 | 1985 | 4032 | 1374.313 | 1016.906 | 983.611 |
| 0.45 | 1537 | 3584 | 1368.499 | 1012.149 | 982.554 |
| 0.50 | 1185 | 3232 | 1365.864 | 1010.345 | 983.657 |
| 0.55 | 897 | 2944 | 1364.060 | 1009.220 | 984.910 |
| 0.60 | 673 | 2720 | 1368.884 | 1014.573 | 992.112 |

## AttAcc Dense Baseline (`--no-rope`)

Configuration: `model=GPT-175B`, `Lin=2048`, `Lout=128`, `batch=1`, `pim=bank`, no
`--powerlimit/--pipeopt/--ffopt` (identical to the Pb sweeps above). Each agent stores its own
full dense K and V. Raw outputs: `attacc_drampim/outputs/attacc_dense_baseline/`.

| Agent | Dense latency (ms) | KV-PIM latency, Pb 0.05-0.60 (ms) | Dense required (GB) | KV-PIM required, Pb 0.05-0.60 (GB) |
|---:|---:|---:|---:|---:|
| 1 | 27.257086 | - | 358.156 | - |
| 64 | 29.332657 | 29.333570 - 29.360160 | 1004.720 | 394.679 - 573.207 |
| 256 | 36.037025 | 36.040282 - 36.149736 | 2975.204 | 441.786 - 1164.683 |

Notes:

- Dense required capacity is `weights (347.892 GB) + N x 10.263 GB` (per-agent K+V at
  `l = 2175`). The simulator's `--no-rope` CSV reports only one agent's KV
  (`required_cap = 358.156 GB` regardless of agent count), so the multi-agent dense capacity is
  computed analytically; the dense *latency* does scale with agents (gen-stage attention ops are
  multiplied by `num_agent` in both modes).
- Latency overhead of KV-PIM vs dense: agent=64 at most +0.028 ms (+0.09%), agent=256 at most
  +0.113 ms (+0.31%) at Pb=0.60. The attention command stream is identical; only the per-token
  diffK preamble is added.
- Capacity: dense agent=256 (2975.2 GB) exceeds the 1374.390 GB aggregate capacity by ~2.2x.
  Dense max context at 256 agents is 849 tokens (cannot even hold the 2048-token prefill);
  at 64 agents dense max context is 3399 (max lout 1351). KV-PIM at agent=256, Pb=0.05 reaches
  a 23776-token context (see Max Decoding Length above).
- KV-only comparison (excluding weights, KV-PIM = K row + shared V 5.131 GB): agent=64 dense
  656.8 GB vs 46.8-225.3 GB (14.0x-2.9x); agent=256 dense 2627.3 GB vs 93.9-816.8 GB
  (28.0x-3.2x) for Pb 0.05-0.60.
- Caveat: the KV-PIM capacity model counts one shared V and only K diffs; per-agent V diffs are
  not included, while dense counts per-agent V in full.

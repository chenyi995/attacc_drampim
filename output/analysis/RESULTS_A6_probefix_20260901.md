# A6 探针 head-folding 修复：验证结果

生成于运行目录 `output/a6probefix_20260901`，对照组是 2026-08-30 那轮 sweep。
**本页是快照：26 格中已完成 8 格**，补齐后重跑本脚本即可。

全部数字由 `output/analysis/a6_probefix_report.py` 生成，不要手工引用。

## 0. 改了什么，以及怎么读这几张表

**只有探针（probe）变了。** `_append_placement_pim_scan`（真正被仿真的那条路）是
把 11 行原地计算原样搬进新 helper `_placement_channel_runs`，表达式逐字相同、
函数内再无对 `loads`/`active` 的赋值，因此 committed 路径行为不变。探针的两处改动：

1. **喂给 Ramulator 的 runs**：`tlb.scan_runs(scan_locations)`（只覆盖 **1 个 head**
   的 TLB reuse run）→ `_placement_channel_runs()` 生成的**每通道一条、把
   `heads_per_hbm` 个 head 折进 row count** 的 run，且只提交最忙的那一条。
2. **聚合方式**：`sum(...)` → `max(...)`。16 个通道在 committed 路上是并发的
   `PIM:pool{c}-{c}` 事件，旧探针把它们串行相加了。

净效应是 `t_bank` 变大（旧探针把 PIM 的 bank 成本按单 head 定价、系统性低估），
于是原本被一律推去 PIM 的 request 里，该走 GPU 的被选了出来。

**读表要点：**

- **改善量 = 改判的 request 数。** 选边接近对半开的格子（`baseline`/`k-hi`/`C-lo`/
  `N-lo`/`reduce`）makespan 改善 30–63%；仍判全 PIM 的格子（`D-lo`/`broadcast`/
  `pipeline`）与旧值一致。没有一格变差。
- **KV link 流量是涨的，不是降的。** 改判的格子 link 相对旧 A6 上升（`baseline`
  25→123 GB，`k-hi` 41→127 GB），因为 GPU prefill 必须把 KV 读回主机；但仍显著
  低于全 GPU 的 A4。论文里 A6 的卖点需要按指标分开陈述：**时间与能量赢，KV 流量
  是相对全 PIM 的让步。**
- **`pipeline_k8` 是一个反例**：A6 22.39 s 反而慢于 A4 的 22.30 s（+0.4%）。探针判
  PIM 快 4.5 倍（`t_bank/t_xpu` = 0.22，三次一致），但该 workload 的 DAG 是串行链
  （`gpu 6.83 s + pool 15.54 s ≈ makespan 22.39 s`，几乎无重叠），PIM pool 才是瓶颈：
  把 2304 行（占 prefill 的 0.84%）挪到 PIM，GPU 省 0.023 s、pool 多 0.159 s。
  **探针孤立地给两边定价，没有"哪个设备是瓶颈"的概念。** 这一条不是本次修复引入的
  —— 旧 A6 同样判全 PIM、同样是 22.39 s。

**未决项（提交时仍未解决）：**

1. 选边未改变的格子并非逐字节一致：`D-lo` 的 `event_count` 与 `link_bytes` 逐字节
   相同，但 `makespan_s` 差 0.0085%（18.75999950678873 vs 18.76159292471786）。
   原因未确认（候选：探针的 Ramulator 调用改变了共享 signature cache 的填充状态；
   两轮的 `--ramulator-workers` 也不同）。按 `docs/sessions/2026-08-31.md` 的规矩，
   这需要一次受控复现才能定性。
2. `pipeline` 上探针估的 bank 成本（3 × 156 us）与实测 pool 增量（159 ms）差约 340 倍。
   即便计入 `pim_pool_time_s_unoverlapped` 是 16 通道求和而探针取 max（x16），仍差
   20 倍。另一嫌疑是 PIM prefill 强制该 batch 的 K/V 先落栈（Fugue 4.5.2 landing
   order）带来的额外写事件（A6 比 A4 多 9120 个 event）。


## 1. makespan（秒）

| cell | 新 A6 | 旧 A6 | 旧 A4 (全 GPU) | 旧 A5 (全 PIM) | Δ vs 旧 A6 | 新探针选边 |
|---|---:|---:|---:|---:|---:|---|
| `LLAMA-7B/C-lo_k8` | **43.76** | 64.95 | 88.91 | 64.95 | +32.6% | pim 16 / gpu 15 |
| `LLAMA-7B/D-lo_k8` | **18.76** | 18.76 | 55.56 | 18.76 | +0.0% | pim 15 / gpu 0 |
| `LLAMA-7B/N-lo_k8` | **17.56** | 24.91 | 33.14 | 24.91 | +29.5% | pim 3 / gpu 4 |
| `LLAMA-7B/baseline_k8` | **58.64** | 100.10 | 140.23 | 100.10 | +41.4% | pim 15 / gpu 16 |
| `LLAMA-7B/broadcast_k8` | **27.49** | 27.49 | 63.81 | 27.49 | +0.0% | pim 16 / gpu 0 |
| `LLAMA-7B/k-hi_k32` | **67.76** | 184.12 | 149.35 | 184.12 | +63.2% | pim 15 / gpu 16 |
| `LLAMA-7B/pipeline_k8` | **22.39** | 22.39 | 22.30 | 22.39 | +0.0% | pim 3 / gpu 0 |
| `LLAMA-7B/reduce_k8` | **26.60** | 44.88 | 63.35 | 44.88 | +40.7% | pim 15 / gpu 1 |

## 2. KV link 流量（GB）与能量（J）

| cell | link 新 A6 | link 旧 A6 | link 旧 A4 | energy 新 A6 | energy 旧 A6 |
|---|---:|---:|---:|---:|---:|
| `LLAMA-7B/C-lo_k8` | 81.93 | 21.04 | 120.41 | 960.3 | 1104.3 |
| `LLAMA-7B/D-lo_k8` | 10.07 | 10.07 | 76.45 | 442.9 | 442.9 |
| `LLAMA-7B/N-lo_k8` | 27.65 | 9.45 | 40.92 | 462.2 | 512.0 |
| `LLAMA-7B/baseline_k8` | 123.29 | 25.33 | 189.66 | 1151.7 | 1432.3 |
| `LLAMA-7B/broadcast_k8` | 10.56 | 10.56 | 83.41 | 639.0 | 639.0 |
| `LLAMA-7B/k-hi_k32` | 127.27 | 41.26 | 193.65 | 1393.7 | 2191.5 |
| `LLAMA-7B/pipeline_k8` | 5.86 | 5.86 | 19.92 | 696.4 | 696.4 |
| `LLAMA-7B/reduce_k8` | 18.08 | 14.73 | 84.45 | 664.5 | 788.8 |

## 3. 探针自身的开销：旧 vs 新

`old` 列按旧公式从同一批探针记录重算：旧探针把 `tlb.scan_runs()` 解出的每条 run
逐条交给 Ramulator 再求和；新探针只提交最忙的那一条通道 run。

| cell | 探针次数 | Ramulator 调用 旧→新 | 模拟行数 旧→新 |
|---|---:|---|---|
| `LLAMA-7B/C-lo_k8` | 31 | 208 → **31** (÷6.7) | 203,960 → 448,768 (×2.2) |
| `LLAMA-7B/D-lo_k8` | 15 | 45 → **15** (÷3.0) | 126,960 → 280,320 (×2.2) |
| `LLAMA-7B/N-lo_k8` | 7 | 39 → **7** (÷5.6) | 64,424 → 140,032 (×2.2) |
| `LLAMA-7B/baseline_k8` | 31 | 208 → **31** (÷6.7) | 332,984 → 718,592 (×2.2) |
| `LLAMA-7B/broadcast_k8` | 16 | 63 → **16** (÷3.9) | 139,648 → 307,200 (×2.2) |
| `LLAMA-7B/k-hi_k32` | 31 | 208 → **31** (÷6.7) | 348,176 → 993,024 (×2.9) |
| `LLAMA-7B/pipeline_k8` | 3 | 17 → **3** (÷5.7) | 26,952 → 59,136 (×2.2) |
| `LLAMA-7B/reduce_k8` | 16 | 63 → **16** (÷3.9) | 139,784 → 307,712 (×2.2) |

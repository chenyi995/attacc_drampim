# PIM channel 并行调度 —— LLAMA-7B 验证（2026-09-02）

## 1. 改了什么

之前 `--pipeopt` 关掉时，调度器把**所有**事件挂在一条 `SERIAL` 资源上，
于是一次 KV scan 的 16 条 per-channel lane 被**逐条串行相加**。但这 16 条
lane 是同一次逻辑 scan 在 16 个物理 channel 上的分身，物理上是并行的
——只有"这次 scan"这个宏事件才和别的宏事件串行。

改动（`src/workload_runner.py`、`src/cpp_eventcore.py`、`src/cppcore/eventcore.cpp`）：
`pipe=False` 时，连续发出的、依赖完全相同的 `PIM:pool*` + `pim_kv_scan`
事件被识别成**一个并行相位**：同时起跑，相位耗时取 max 而不是 sum，
之后的 DIE merge 仍逐条依赖每条 lane。Python 调度器、C++ `eventcore`
和 overlap 契约校验三处用同一条判据，保持三者语义一致。

同一次改动里还有 A6 选边探针的 head 折叠修正（见
`experiments/a6_probefix_20260901/RESULTS.md`）：抽出
`_placement_channel_runs()` 给分支和探针共用，探针只给**最忙 channel**
那条 run 计价（`numOp=1`、行数 = `max(loads) × 256`）。

## 2. 结果（LLAMA-7B，`--num-hbm 1`，无 `--pipeopt`，`--epic-prefix-recompute-tokens 8`）

old = 2026-08-30 sweep（串行 lane），new = 本次。两边工作量口径一致：
`event_count`、`link_bytes`、`energy_nj` 逐格相同，只有 makespan 变。

| 格子 | 档位 | old makespan | new makespan | 加速 |
|---|---|---:|---:|---:|
| baseline_k8 | A3 | 141.635 s | 19.539 s | 7.25× |
| baseline_k8 | A3a | 140.597 s | 18.456 s | 7.62× |
| baseline_k8 | A3b | 141.635 s | 19.539 s | 7.25× |
| baseline_k8 | A4 | 140.234 s | 24.009 s | 5.84× |
| baseline_k8 | A6 | 100.097 s | 15.309 s | 6.54× |
| broadcast_k8 | A3 | 64.004 s | 9.905 s | 6.46× |
| broadcast_k8 | A4 | 63.813 s | 12.887 s | 4.95× |
| broadcast_k8 | A5 | 27.490 s | 8.703 s | 3.16× |
| broadcast_k8 | A6 | 27.490 s | 8.703 s | 3.16× |
| reduce_k8 | A3 | 63.240 s | 10.617 s | 5.96× |
| reduce_k8 | A4 | 63.354 s | 12.557 s | 5.05× |

（`results/channel_parallel_llama7b.csv`，由 `collect_summaries.py` 从
slurm 日志的 `REPORT_SUMMARY` 行提取。按 `docs/RAW_DATA_MANIFEST.md`，
原始 `dag_*.json`（每个 50–115 MB）和 slurm `.log` 都不入库：前者留在
`output/channel_parallel_validation_20260902/`，后者留在本目录 `logs/`，
只把每个 run 的 `REPORT_SUMMARY` 与产物路径抄进 `results/report_summaries.txt`
作为证据。baseline_k8 的 A5、reduce_k8 的 A5/A6 这次没跑。）

## 3. BUG：A3b 在 `--num-hbm 1` 下退化成 A3

**现象**：上表里 baseline_k8 的 A3 和 A3b makespan、energy、event_count
**逐位相同**（19.538823025675715 s）。这不是巧合，也不是本次改动引入的：
old 那一列同样是 141.6345481195492 s 对 141.6345481195492 s，
broadcast_k8（64.0036402729786）、reduce_k8（63.239692437765925）也一样。
**之前所有 `--num-hbm 1` 的 A3/A3b 对比都是自己跟自己比**，
所以"A3b 相对 A3 没拉开差距"这个观察本身是无效的。

**根因**（`src/workload_runner.py` `_layout_channel_loads`）：
A3b 的 `slice` 策略把 head h 铺在 `stripe = max(1, 16 // heads_per_hbm)`
条 channel 上。LLAMA-7B 有 32 个 KV head，`--num-hbm 1` ⇒
`heads_per_hbm = ceil(32/1) = 32`，于是 `16 // 32 = 0`，被 `max(1, …)`
钳到 **stripe = 1**——每个 head 只剩一条 channel，正是 A3 的 `single`。
两个策略返回的 load 向量完全相同：

```
num_hbm=1  heads_per_hbm=32   c_master=16, c_diff=1
  single 34 | slice 34 | master-diff-slice 48 | master-diff-table 35   slice == single: True
num_hbm=4  heads_per_hbm=8    c_master=16, c_diff=1
  single 17 | slice  9 | master-diff-slice 16 | master-diff-table  9   slice == single: False
```

要让 A3b 真的是 A3b，`heads_per_hbm` 必须 < 16，即 HBM 堆栈数 > 2。

**在跑的修正实验**：`run_a3_l7b_hbm4.sbatch`（slurm array 193281，
2026-09-02 06:31 提交，A3 / A3a / A3b 各一个 task）改用 `--num-hbm 4`
⇒ 8 head/HBM ⇒ 每个 head 2 条 lane，A3b 的 busiest-channel load 应当从
17 降到 9（≈1.9×）。输出写到
`output/channel_parallel_validation_20260902/LLAMA-7B/baseline_k8_hbm4/`。
**注意：`--num-hbm 4` 也改了 A3 自己的基线，所以要跟同一批的 A3
比，不能跟上表的 `--num-hbm 1` A3 比。**

## 4. 顺带暴露的第二件事：A4 现在比 A3 慢

lane 并行之前 A4（140.234 s）略快于 A3（141.635 s）；并行之后 A4
（24.009 s）**慢于** A3（19.539 s），broadcast / reduce 同向。串行相加
时各 channel 的不均衡被求和抹平，取 max 之后才显出来。

根因同样是 `heads_per_hbm = 32` 太大：A4 的 `master-diff-slice` 里
`stripe_m = max(1, 15 // 32) = 1`，32 个 head 挤在 15 条 master channel
上（≈3 head 深，A3 是 16 条上 2 head 深），而 32 个 head 的 correction
chunk 全压在唯一的 diff channel（ch15）上 ⇒ busiest load 48 vs A3 的 34。
上面的数值表已给出这个 48/34；实测比值 24.009/19.539 = 1.23 低于 48/34 =
1.41，因为 scan 只占 makespan 的一部分。

这一条尚未单独跑实验验证：把 A4 也放到 `--num-hbm 4` 下，
`stripe_m = 15 // 8 = 1` 仍然是 1，所以 A4 的这个问题**不会**被
`--num-hbm 4` 修掉——需要更多 HBM，或者换成 A4b 的全局放置表
（`master-diff-table`，同参数下 35 vs 48）。

## 5. 复现

```bash
# 12 格（baseline / broadcast / reduce × A3–A6），--num-hbm 1
sbatch experiments/channel_parallel_validation/run_12_llama7b.sbatch
# baseline 补 A3a / A3b / A6，--num-hbm 1
sbatch experiments/channel_parallel_validation/run_a3a_a3b_a6.sbatch
# A3 / A3a / A3b，--num-hbm 4：让 A3b 不再退化成 A3
sbatch experiments/channel_parallel_validation/run_a3_l7b_hbm4.sbatch

python3 experiments/channel_parallel_validation/collect_summaries.py
```

# PIM channel 并行调度 + A3 阶梯 —— LLAMA-7B 验证（2026-09-02）

> ## ⚠️ 测 A3 / A3b 之前必读
>
> **A3b 会在 HBM 堆栈太少时静默退化成 A3，两者逐位相同。**
> A3b 的 `slice` 放置给每个 head `max(1, 16 // heads_per_hbm)` 条 channel；
> 一个 stack 上的 KV head 数一旦 ≥ channel 数，stripe 被钳到 1，
> `_layout_channel_loads('slice', …)` 返回的向量和 `'single'`（A3）**完全相等**
> ——makespan、energy 一位不差。这时候跑出来的"A3b"就是 A3，
> **拿它和 A3 比等于 A3 跟自己比**。
>
> **规则：跑 A3b 必须让「每个 stack 的 KV head 数 × 2 ≤ 16」，即 KV head 数
> ≤ 8 per stack。** LLAMA-7B（32 KV head）⇒ `--num-hbm >= 4`。
> A4 的 `master-diff-slice` 只有 15 条 master channel，塌得更早，需要 `>= 5`。
>
> 代码里已加护栏：`placement_degeneracy_warning()`
> （`src/workload_runner.py`），每次 run 在 `run_reuse_prefill` 入口检查，
> 一旦塌了就往 stderr 打一段带修复建议的警告。
> `tests/test_placement.py::SliceDegeneracyWarningTest` 逐 `heads_per_hbm`
> 对着真实 load 向量校验"警告当且仅当 slice == single"。
> `collect_summaries.py` 也会把 makespan 与同格 A3 相同的 A3b 行直接丢掉，
> 不让它进结果表。

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

## 2. 跑的是什么

| | |
|---|---|
| workload | `workload/sweep/wl_baseline_alltoall_N16_C32_D2.json`（第 3 节全部、第 4 节的 baseline 行）<br>`wl_broadcast.json` / `wl_reduce.json`（第 4 节另两行） |
| baseline 的 meta | `v2-dag`、topology `alltoall`、N=16 C=32 D=2、widths [16,16] ⇒ 32 agents、`block_tokens` 256、`private` false |
| 规模 | 32 requests、input 332 288 token、output 8 192 token、history 4 096 token |
| broadcast / reduce | 同 N/C/D，widths [1,16] / [16,1] ⇒ 17 agents |
| run 参数 | `--system dgx-attacc --model LLAMA-7B --ngpu 1 --engine dag --reuse recompute --epic-prefix-recompute-tokens 8 --cacheblend-batch-size 8`，**无** `--pipeopt` |

⚠️ 这三个 workload 的 `meta.kind` 自己写着
`"parametric sweep (mechanism illustration; NOT evidence-grade)"`——
是 `gen_sweep.py` 生成的合成 DAG，用于展示机制，不是论文级证据。

## 3. A3 阶梯（`--num-hbm 4`，A3b **没有**退化）

slurm array 193281，2026-09-02 06:31 提交、08:00 三个 task 全部 COMPLETED，
各 1.2–1.5 h。8 head/stack ⇒ 每个 head 2 条 lane。**这是本项目第一次
真正测到 A3b。**

| 档位 | placement | makespan | 相对 A3 | event_count |
|---|---|---:|---:|---:|
| A3 | single | 15.330 s | — | 5 480 896 |
| A3a | single + 读掩码 | 14.272 s | −6.90 % | 5 480 896 |
| A3b | slice（2 lane/head） | **13.356 s** | **−12.88 %（1.148×）** | 6 853 056 |

A3b 比 A3a 再快 6.42 %。event_count 变多是对的：一个 head 摊到 2 条
channel，lane 事件本来就该翻倍。

**1.148× 看起来远小于 load 模型的 1.94×（busiest 33→17），但两者其实一致。**
makespan 里有三块与 placement 无关：GPU prefill 9.53 s、DIE 0.02 s、链路
1.75 s，合计 11.31 s。扣掉之后 scan 关键路径 4.03 s → 2.04 s，正是 1.94×。
逐项归因见第 6 节。

## 4. channel 并行本身的效果（`--num-hbm 1`）

old = 2026-08-30 sweep（串行 lane），new = 本次。两边工作量口径一致：
`event_count`、`link_bytes`、`energy_nj` 逐格相同，只有 makespan 变。

| 格子 | 档位 | old makespan | new makespan | 加速 |
|---|---|---:|---:|---:|
| baseline_k8 | A3 | 141.635 s | 19.539 s | 7.25× |
| baseline_k8 | A3a | 140.597 s | 18.456 s | 7.62× |
| baseline_k8 | A4 | 140.234 s | 24.009 s | 5.84× |
| baseline_k8 | A6 | 100.097 s | 15.309 s | 6.54× |
| broadcast_k8 | A3 | 64.004 s | 9.905 s | 6.46× |
| broadcast_k8 | A4 | 63.813 s | 12.887 s | 4.95× |
| broadcast_k8 | A5 | 27.490 s | 8.703 s | 3.16× |
| broadcast_k8 | A6 | 27.490 s | 8.703 s | 3.16× |
| reduce_k8 | A3 | 63.240 s | 10.617 s | 5.96× |
| reduce_k8 | A4 | 63.354 s | 12.557 s | 5.05× |

**这张表里没有 A3b**：`--num-hbm 1` 下它退化成了 A3（见开头警告），
那个 19.539 s 不是 A3b 的测量值，已经从结果里删掉。A3 阶梯看第 3 节。

⚠️ 第 3 节和第 4 节**不能纵比**：`--num-hbm` 改的不只是调度，还改 placement、
`event_count` 和器件能耗（A3 的 `energy_nj` 从 2.37e12 涨到 7.34e12，
4 个 stack 的功耗）。`collect_summaries.py` 因此对 `_hbm4` 格子刻意留空
old / speedup 两列。baseline_k8 的 A5、reduce_k8 的 A5/A6 这次没跑。

（数据：`results/channel_parallel_llama7b.csv`。按 `docs/RAW_DATA_MANIFEST.md`，
原始 `dag_*.json`（每个 50–115 MB）和 slurm `.log` 都不入库：前者留在
`output/channel_parallel_validation_20260902/`，后者留在本目录 `logs/`，
只把每个 run 的 `REPORT_SUMMARY` 与产物路径抄进 `results/report_summaries.txt`
作为证据。）

## 5. 退化的根因，和它顺带暴露的 A4 回退

### 5.1 根因

`src/workload_runner.py` `_layout_channel_loads`：A3b 的 `slice` 把 head h
铺在 `stripe = max(1, 16 // heads_per_hbm)` 条 channel 上。LLAMA-7B 有 32 个
KV head，`--num-hbm 1` ⇒ `heads_per_hbm = ceil(32/1) = 32` ⇒ `16 // 32 = 0`
⇒ 被 `max(1, …)` 钳到 **stripe = 1**——每个 head 只剩一条 channel，
正是 A3 的 `single`。直接调函数的对照：

```
busiest-channel load, c_master=32, c_diff=1  (= 本实验的 scan 形状，见第 2 节)
num_hbm=1  heads_per_hbm=32
  single 66 | slice 66 | master-diff-slice 96 | master-diff-table 69   slice == single: True
num_hbm=4  heads_per_hbm=8
  single 33 | slice 17 | master-diff-slice 32 | master-diff-table 18   slice == single: False
```

**这个 bug 不是本次改动引入的，旧数据里同样存在**：2026-08-30 sweep 的
baseline_k8 A3 与 A3b 都是 141.6345481195492 s，broadcast_k8
（64.0036402729786）、reduce_k8（63.239692437765925）也一样。所以
**在 2026-09-02 之前，本项目所有 `--num-hbm 1` 的 A3/A3b 对比都是无效的**。

### 5.2 A4 现在比 A3 慢

lane 并行之前 A4（140.234 s）略快于 A3（141.635 s）；并行之后 A4
（24.009 s）**慢于** A3（19.539 s），broadcast / reduce 同向。串行相加
时各 channel 的不均衡被求和抹平，取 max 之后才显出来。

**总工作量几乎没变**：所有 PIM lane 的耗时之和 A3 130.247 s、A4 129.948 s，
差 0.2 %。A4 不是多干了活，是**干的活堆歪了**——busiest channel 从 66 涨到 96
（+45 %）。这次实测的 per-channel load 向量（`c_master=32`、`c_diff=1`）：

```
A3 single           : [66] × 16                                    ← 完美均衡
A4 master-diff-slice: [96, 96, 64 × 13, 32]
                       ^^^^^^          ^^  diff channel，不是瓶颈
```

**瓶颈是 master pool，不是 diff channel。** 两件事叠加：

1. master pool 从 16 条 channel 缩到 15 条（ch15 让给 diff）；
2. `stripe_m = max(1, 15 // 32) = 1`，head h 整块压在 ch(h mod 15)。

32 不能被 15 整除 ⇒ ch0 / ch1 各接到 **3** 个 head（96），其余 13 条接 2 个
（64）；A3 是 32 head 铺 16 条，**每条正好 2 个**（66），一点不歪。而 diff
channel 只有 32 个 correction chunk（每个复用块只重算 8 token），远够不上
瓶颈。**A4 把 correction 挪出去省的那点，不抵它为此丢掉一条 master channel
再撞上 32 ∤ 15 的代价。**

### 5.3 更正：`--num-hbm 4` 其实能修掉 A4 的回退

本文档 2026-09-02 早先的版本写着「`--num-hbm 4` 修不掉 A4（`15 // 8` 仍是
1）」。**这句是错的。** stripe 确实还是 1，但 8 个 head 铺 15 条 master
channel 时**每条最多 1 个**，根本没有堆叠：

| `--num-hbm` | heads/stack | A3 | A3b | A4 | A4b | A4/A3 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 66 | 66 | 96 | 69 | 1.455 |
| 2 | 16 | 33 | 33 | 64 | 35 | **1.939**（最差） |
| 4 | 8 | 33 | 17 | 32 | 18 | **0.970** |
| 8 | 4 | 33 | 9 | 11 | 9 | 0.333 |

即 A4 的回退只出现在 heads/stack > master channel 数的时候，`--num-hbm 4`
就没有了。**这是一条尚未跑实验的预测**：A4 / A4b 在 `--num-hbm 4` 下没跑过。
（护栏在 A4 + 4 stack 时仍会报警，那是对的：stripe 塌了意味着这一格测不到
slicing —— 同参数下 A4b 的全局放置表 18 vs A4 的 32，差 1.8×。）

## 6. makespan 逐项归因

无 `--pipeopt` 时所有宏事件串行，所以

```
makespan = GPU + DIE + 链路 + PIM scan 关键路径
```

前三项与 placement 无关。报告里的 `pim_pool_time_s_unoverlapped` 是**所有
lane 耗时之和**（= 总 scan 工作量，不是关键路径），placement 决定最忙那条
channel 占其中多少，正好是 load 模型的 `max(loads) / sum(loads)`：

```
PIM scan 关键路径 = pool_sum × max(loads) / sum(loads)
```

链路没有单独上报，所以**每个格子从它的 A3 行反解一次**（A3 的 `single`
在这里是完美均衡的，关键路径占比精确已知），然后**固定住去预测同格其余
档位**——所以除 A3 外每一行都是样本外预测，残差是对 load 模型的真检验。

`python3 attribute_makespan.py`（`c_master=32`、`c_diff=1`，见第 2 节 workload）：

| 格子 | 档位 | busiest | GPU | DIE | 链路 | PIM 关键路径 | 预测 | 实测 | 残差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_k8 | A3 | 66 | 9.53 | 0.11 | 1.75 | 8.14 | 19.54 | 19.54 | 反解 |
| baseline_k8 | A3a | 66 | 9.53 | 0.11 | 1.75 | 8.14 | 19.54 | 18.46 | +5.9 % |
| baseline_k8 | A4 | 96 | 9.53 | 0.11 | 1.75 | 11.81 | 23.21 | 24.01 | −3.3 % |
| baseline_k8_hbm4 | A3 | 33 | 9.53 | 0.02 | 1.75 | 4.03 | 15.33 | 15.33 | 反解 |
| baseline_k8_hbm4 | A3a | 33 | 9.53 | 0.02 | 1.75 | 4.06 | 15.36 | 14.27 | +7.6 % |
| baseline_k8_hbm4 | A3b | 17 | 9.53 | 0.03 | 1.75 | 2.04 | 13.35 | 13.36 | **−0.0 %** |
| broadcast_k8 | A4 | 96 | 5.92 | 0.06 | 0.32 | 5.23 | 11.53 | 12.89 | −10.5 % |
| reduce_k8 | A4 | 96 | 6.74 | 0.06 | 0.30 | 5.12 | 12.22 | 12.56 | −2.7 % |

两个交叉验证：

- 链路项在 `--num-hbm 1` 和 `--num-hbm 4` 两个**独立**反解里得到
  1.752113 s 和 1.750665 s，差 0.08 %——两套配置的 GPU / DIE / pool_sum
  完全不同，却反解出同一个链路时间。
- **A3b @ hbm4 预测 13.35 s、实测 13.356 s，残差 −0.0 %**，是纯样本外。

不吻合的地方（照实说）：

- **A3a 系统性高估 6–8 %**。读掩码改的是**工作量**（被掩掉的行不算），
  不只是分布，所以"busiest × pool_sum"这个模型对它本来就不成立。
- **A4 系统性低估 3–10 %**，即 A4 实际比 chunk 模型算的还要更差一点。
  chunk 模型按 256 行整块计价，忽略了每条 run 的固定开销与 run 长度差异；
  broadcast 那格 −10.5 % 最大，多半是它的 `c_master`/`c_diff` 与
  baseline 不同（本节对三个格子用了同一组假设值，没有逐格测量）。

## 7. 复现

```bash
# 12 格（baseline / broadcast / reduce × A3–A6），--num-hbm 1
sbatch experiments/channel_parallel_validation/run_12_llama7b.sbatch
# baseline 补 A3a / A6，--num-hbm 1（这个脚本里的 A3b 会退化，只当 A3 看）
sbatch experiments/channel_parallel_validation/run_a3a_a3b_a6.sbatch
# A3 / A3a / A3b，--num-hbm 4：唯一有效的 A3b 口径
sbatch experiments/channel_parallel_validation/run_a3_l7b_hbm4.sbatch

python3 experiments/channel_parallel_validation/collect_summaries.py
python3 experiments/channel_parallel_validation/make_report_summaries.py
python3 experiments/channel_parallel_validation/extract_device_times.py
python3 experiments/channel_parallel_validation/attribute_makespan.py
python3 -m unittest tests.test_placement       # 含退化护栏的用例
```

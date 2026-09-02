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

**但只有 1.148×，不是 load 模型预测的 1.89×。** busiest-channel load 17→9
说的是**一次 KV scan** 的时间，而 makespan 里还有 GPU prefill、链路、
DIE merge 等与 placement 无关的部分——lane 并行之后 scan 已经不再是
压倒性的大头（见第 4 节：并行化本身就吃掉了 7×）。要把这 1.148× 拆开
归因，得看 `dag_A3*.json` 的 per-device busy 时间，**本次没做**。

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
num_hbm=1  heads_per_hbm=32   c_master=16, c_diff=1
  single 34 | slice 34 | master-diff-slice 48 | master-diff-table 35   slice == single: True
num_hbm=4  heads_per_hbm=8    c_master=16, c_diff=1
  single 17 | slice  9 | master-diff-slice 16 | master-diff-table  9   slice == single: False
```

**这个 bug 不是本次改动引入的，旧数据里同样存在**：2026-08-30 sweep 的
baseline_k8 A3 与 A3b 都是 141.6345481195492 s，broadcast_k8
（64.0036402729786）、reduce_k8（63.239692437765925）也一样。所以
**在 2026-09-02 之前，本项目所有 `--num-hbm 1` 的 A3/A3b 对比都是无效的**。

### 5.2 A4 现在比 A3 慢

lane 并行之前 A4（140.234 s）略快于 A3（141.635 s）；并行之后 A4
（24.009 s）**慢于** A3（19.539 s），broadcast / reduce 同向。串行相加
时各 channel 的不均衡被求和抹平，取 max 之后才显出来。

根因同样是 `heads_per_hbm = 32` 太大：A4 的 `master-diff-slice` 里
`stripe_m = max(1, 15 // 32) = 1`，32 个 head 挤在 15 条 master channel
上（≈3 head 深，A3 是 16 条上 2 head 深），而 32 个 head 的 correction
chunk 全压在唯一的 diff channel（ch15）上 ⇒ busiest load 48 vs A3 的 34。
实测比值 24.009/19.539 = 1.23 低于 48/34 = 1.41，因为 scan 只占 makespan
的一部分。

这一条**尚未单独跑实验验证**。注意 `--num-hbm 4` 修不掉它
（`15 // 8` 仍然是 1，护栏在 A4 + 4 stack 时照样报警）——要么再加 stack
（≥ 5），要么换 A4b 的全局放置表（`master-diff-table`，同参数下 35 vs 48）。

## 6. 复现

```bash
# 12 格（baseline / broadcast / reduce × A3–A6），--num-hbm 1
sbatch experiments/channel_parallel_validation/run_12_llama7b.sbatch
# baseline 补 A3a / A6，--num-hbm 1（这个脚本里的 A3b 会退化，只当 A3 看）
sbatch experiments/channel_parallel_validation/run_a3a_a3b_a6.sbatch
# A3 / A3a / A3b，--num-hbm 4：唯一有效的 A3b 口径
sbatch experiments/channel_parallel_validation/run_a3_l7b_hbm4.sbatch

python3 experiments/channel_parallel_validation/collect_summaries.py
python3 -m unittest tests.test_placement       # 含退化护栏的用例
```

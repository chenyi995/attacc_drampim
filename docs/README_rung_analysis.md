# 九档之间到底分不分得开，A6 的选边选得对不对

本页记录 2026-09-01 对**已完成的 sweep 数据**做的三项检验，起因是一个直接的
问题：*会不会有两个档在所有指标上都差不多？* 会 —— 有三对。

**全部结论可重算**，不要手工引用本页数字：

```bash
python3 output/analysis/rung_discrimination.py            # 三项检验
python3 output/analysis/rung_discrimination.py --verify 5 # 顺带核对 JSON
```

数据源是每个任务的 `ladder.log` 里的 `REPORT_SUMMARY` 行。`main.py` 在同一次
调用里打印这一行并写出 `dag_<档>.json`，两者同源；`--verify` 会重新读 JSON
逐值断言，而不是假定它们一致（本页数字生成时：**108 个值、3 个任务，全部一致**）。

> **样本**：本页写作时 sweep 尚未跑完，九档齐全的任务有 **54–55 个**（总共 78 个
> 可跑任务）。**补跑结束后必须重跑一次** —— 缺的格子里包含大模型的 `D-hi` /
> `C-hi`，而最大差异多次恰好出现在 `GPT-13B/D-hi`、`GPT-13B/C-hi`，结论可能变。

---

## 1. 有三对档在全 sweep 上都分不开

判据是**全 sweep 的最大差异**，不是中位数：某一对档只要在**一个**配置上分得开，
那一档就证明了自己的机制存在；反过来，如果 54 个任务、四项指标里**处处**都差
不到 30%，这一步 ablation 就没有展示任何东西。

| 档对 | 全 sweep 最大差异 | makespan | energy | KV link | events |
|---|---|---|---|---|---|
| **A3 vs A3a** | **8.9%** | 8.9% | 4.6% | **0.0%** | **0.0%** |
| **A3b vs A4b** | **13.8%** | 9.1% | 4.6% | **0.0%** | 13.8% |
| **A4 vs A4b** | **28.8%** | 0.8% | 0.011% | **0.0%** | 28.8% |

其余 33 对都分得开（30.7% – 100%）。A1、A2 与后面各档的区分度极好（96% – 100%），
**A5 vs A6 也健康（82.5%）**。

值得单独指出：这三对里 **KV link 处处完全相同（0.0%）**。也就是说 A3a、A3b、A4b
相对各自的邻档**没有改变任何 KV 流量**，只改变了时序和事件计数。

**分不开的恰好就是最近两次新加的三档**：A3a（2026-08-26 加）、A3b 与 A4b
（2026-08-29 加）。这直接回答了 `README_sweep_design.md` 里那个待决问题
"A3a/A3b/A4b 是否进入论文" —— **按当前数据，它们没有可报告的效应**。

一个已知的口径冲突：`README_sweep_design.md` §7.1 引用归档结果称 A3→A3a 使
decode 能量从 1,480 mJ 降到 791 mJ（−47%）。那份佐证来自
`output/archived/2026-08-29_pre-unify/RESULTS_k2.md`（RAG、k2），**与本轮 sweep
矛盾**。需要确认是口径变了，还是那个结论不再成立。

---

## 2. A6 相对 A3 系列最优档的改善

逐任务取 A3/A3a/A3b 中最好的那个作基线：

| 指标 | 中位改善 | 范围 | A6 反而更差 |
|---|---|---|---|
| **KV link** | **+86.6%** | +28.4% ~ +89.7% | **0 / 54** |
| **energy** | **+46.5%** | −0.6% ~ +74.6% | 1 / 54 |
| makespan | +24.8% | **−24.3%** ~ +66.2% | 1 / 54 |

**KV link 是唯一没有反例的指标** —— 54 个任务全部改善 28% 以上。

makespan 有一个反例（`LLAMA-7B/k-hi`，−24.3%），energy 有一个几乎持平的
（`GPT-13B/pipeline`，−0.6%）。所以**"A6 全面更优"不成立**，论文里要按指标分开
陈述：KV 流量与能量是稳赢，时间不是。

按模型看，makespan 收益随模型增大**单调下降**（GPT-13B +26.1% → LLAMA-33B
+20.4% → LLAMA-65B +18.0% → GPT-175B +14.4%），而 energy 稳定在 48% 左右、
KV link 稳定在 86.8%。把卖点讲成"更快"会在大模型一端被审稿人盯住；讲成
"KV 流量降 87%、能量降 48%"则六个模型一致成立。

---

## 3. A6 的选边：应当成立的不变量破了两次

A6 逐 request 在 GPU 与 PIM 两条 prefill 路径之间选边，
`prefill_side = "pim" if t_bank <= t_xpu else "gpu"`
（`src/workload_runner.py` 的 `_run_cacheblend_prefill`）。

A4 强制全 GPU prefill、A5 强制全 PIM，两者都实跑了。A6 既然可以自由选择，
就**永远不应该输给其中任何一个**：

> **不变量：`A6 ≤ min(A4, A5)`**

60 个有这三档的任务里，**破了 2 个，都在 LLAMA-7B 上**：

| workload | 配置 | A4 | A5 | **A6** | 超出 min |
|---|---|---|---|---|---|
| `wl_baseline_alltoall_N16_C32_D2.json` (k=32) | **k-hi** | **149.4 s** | 184.1 s | **184.1 s** | **+23.3%** |
| `wl_pipeline_D4.json` (k=8) | pipeline | **22.3 s** | 22.4 s | 22.4 s | +0.4% |

第二个只超 0.4%，且该任务本身是 6/9 的受损任务（§3 末的静默崩溃）。**实质违反
只有 `LLAMA-7B / k-hi` 一个**，它在 node5 上 9/9 干净跑完 —— 不是节点问题，也
不是数据损坏。两次都是选边器把**全部请求判给 PIM**（A6 逐字节等于 A5）。

### 最优解一定是混合的，逐层数据说得很清楚

| `LLAMA-7B/k-hi` | tier 0 | tier 1 | 合计 |
|---|---|---|---|
| A4 全 GPU | 55.6 s | **93.8 s** | 149.4 |
| A5 全 PIM | **18.8 s** | 165.4 s | 184.1 |
| 逐层取优 | 18.8（PIM） | 93.8（GPU） | **≈112.6** |
| A6 实际 | 18.8 | 165.4 | **184.1** ❌ |

**tier 0 在 PIM 上快 3 倍，tier 1 在 PIM 上慢 1.76 倍。** LLAMA3-8B 在同一个
workload 上正是这么分的（tier 0→PIM、tier 1→GPU），拿到 64.9 s —— 比两个纯策略
都好。所以机制本身是对的，问题在阈值/估价。

A6 真正做了混合选边的 7 个任务（**全部是 LLAMA3-8B**），无一例外优于最优纯策略
25%–56%。分流依据是**上下文大小**，不是并发度：`reduce` 只把那一个读遍 16 路
上游的 reducer 送去 GPU，`supervisor` 只送 supervisor 那一个；按实际工作量算是
**138 : 1**，不是负载对半开。

### 逐步对照：探针与引擎用两套模型给同一件事定价

沿 `LLAMA-7B/k-hi` 的 tier-1 请求静态走一遍（不跑，只读代码 + 已有 JSON）。
选边是**比较**，两侧都要查。

**PIM 侧 `t_bank`：**

| 步骤 | 理论 / 引擎 | 探针（3520–3548 行） | 偏向 |
|---|---|---|---|
| 分解 | `_layout_channel_loads(policy, c_master, c_diff, heads_per_hbm, 15)` → 长度 16 的每通道负载向量 | `tlb.scan_runs()` 按**物理地址相邻**合并 | — |
| **policy** | `single`/`slice`/`master-diff-slice`/`master-diff-table` **就是 A3→A3b→A4→A4b 这条阶梯** | **完全不传** | ⚠️ |
| **heads_per_hbm** | 决定 head 在通道上叠几层 | **没有这个参数** | ⚠️ |
| 归约 | 每通道一个**并发事件** → DAG 取 **max**（docstring：*"the scan time is the busiest channel"*） | **`sum(item[0] for item in measured)`** | 高估 PIM |
| numOp | `scan_op.numOp = 1`，head 折进行数 | `numOp = kv_heads`，`n` 是单 head 行数 | — |
| sweeps | 每个 sweep 各自成事件，串不串行由 DAG 依赖决定 | `× sweeps` 硬乘（连 TLB plan 一起） | 高估 PIM |

**GPU 侧 `t_xpu`：**

| 步骤 | 引擎（3665–3697 行） | 探针（3503–3516 行） | 偏向 |
|---|---|---|---|
| **`dram_read_resident`** | `_append_channel_kv_stores(readback_rows)` —— 逐通道 PIM pool 事件，`time_s = bytes / (每HBM带宽 × ch/16)`；注释：*"a scattered layout pays its activations here too"* | **整笔缺失** | 低估 GPU |
| link `kv_pim_to_gpu` | ✓ | ✓ | |
| score / softmax / context | ✓ | ✓ | |

### 最根本的一条

**探针从不应用放置策略。** A6 骑在 A3→A4b 这条**专门研究通道放置**的阶梯顶上，
而它的选边估价里既没有 `policy` 也没有 `heads_per_hbm` —— 意味着它对 A3、A3b、
A4、A4b 给出**完全相同**的 PIM 估价。这与理论公式（`max(负载向量)`，而负载向量
由 policy 与 heads_per_hbm 生成）直接冲突。

LLAMA-7B 恰好是这个盲区最致命处。`head_mapping` 的落地规则是
*"head h → channel (offset+h) % channel_count；超过 channel_count 个 head 后推进
一个 8-KiB partition"*，而 `master-diff-slice` 的 `stripe_m = 15 // heads`：

| 模型 | kv_heads | num_hbm | heads_per_hbm | 15 个 master 通道上的分布 | 最忙通道 |
|---|---|---|---|---|---|
| GPT-13B | 40 | 10 | 4 | — | — |
| LLAMA3-8B | 8 | 1 | 8 | ch0–7 各 1 个，ch8–14 空 | **1 个 head** |
| GPT-175B | 96 | 10 | 10 | — | — |
| **LLAMA-7B** | 32 | **1** | **32** | ch0,ch1 各 3 个；ch2–14 各 2 个 | **3 个 head** |

理论上 LLAMA-7B 的最忙通道是 LLAMA3-8B 的 **3 倍**，探针完全看不见。

### 一个诚实的限制：静态读码不能定主因

**四处偏离的方向是相反的** —— PIM 侧的 `sum` 与 `× sweeps` 高估 PIM（偏 GPU），
GPU 侧漏掉 `dram_read_resident` 低估 GPU（也偏 GPU），而观察到的错误是**过度偏
PIM**。所以只能确定这四处都与理论公式不符，**不能断定哪一条占主导**。

要定论需要给探针加日志，打出每个请求的 `t_bank` / `t_xpu`，与事件流里实际发生的
时间对账。那需要跑，等 sweep 结束后再做。

**已作废的假设**（记在这里，免得被重复引用）：曾提出"`num_hbm=1` 的模型因此被选
错"，**被数据否定** —— 按全部配置统计，`num_hbm=1` 的两个模型反而是 PIM 收益最大
的（中位 −28.6% / −37.1%）。判别变量是 `heads_per_hbm`（32 对 8），不是 `num_hbm`。
另一个曾提出的"A6 没做负载均衡"也被否定：分流依据是上下文大小，工作量比 138:1。

---

## 待办

1. **补跑结束后重跑本页的脚本** —— 结论建立在 54/78 个任务上。
2. **A3a / A3b / A4b 是否进入论文** —— 现在有数据支撑决定了。
3. **先给探针加日志定主因，再动修复** —— §3 列的四处偏离方向相反，静态读码
   定不了主因。打出逐 request 的 `t_bank` / `t_xpu`，与事件流实际时间对账。
4. **修复方向：让探针和引擎共用同一个定价函数**，而不是再写一份平行逻辑 ——
   这个 bug 正是平行逻辑产生的。属引擎改动，会改变 A6 结果，**必须等 sweep
   跑完**，且改完要按 `docs/sessions/2026-08-31.md` 的办法做逐字节一致性验证。
5. **核对 §1 末尾那处与归档结果的口径冲突。**

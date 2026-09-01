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

## 3. A6 的选边：算法是对的，只在一个格子上错

A6 逐 request 比较 GPU 与 PIM 两条 prefill 路径的时间，
`prefill_side = "pim" if t_bank <= t_xpu else "gpu"`
（`src/workload_runner.py` 的 `_run_cacheblend_prefill`）。

sweep 里 A4（强制全 GPU prefill）和 A5（强制全 PIM prefill）都实跑了，所以
**选边器的判断可以直接检验，不需要相信它**：

| | |
|---|---|
| A6 与 A5 完全相同（判定"全走 PIM"） | **47 / 54** |
| A6 实际改选了 GPU | 7 / 54（**全部是 LLAMA3-8B**）|
| 在那 47 个里，实测 A5 反而慢于 A4 | **1** |

**PIM prefill 在 55 个格子里只输 2 个，两个都是 `k-hi`（k=32）**：

| 格子 | A5 − A4 | A6 |
|---|---|---|
| `LLAMA-7B / k-hi` | **+23.3%** | **未触发** ← 唯一的真错误 |
| `LLAMA3-8B / k-hi` | +21.7% | 救回（179.3 s → 64.9 s）|

`k-hi` 是 recompute token 最多、prefill 工作量最大的配置，而这两个是最小的两个
模型 —— GPU prefill 相对便宜。两个条件叠加时 PIM 才输，A6 抓到了其中一个。

### 未验证的线索：探针没有传 `heads_per_hbm`

同一份代码里给 PIM 定价有三处调用。DAG 真实事件的两处
（`workload_runner.py` 2251 行、2612 行）都传了：

```python
heads_per_hbm=_heads_per_hbm(_gqa_kv_heads_local(system, heads),
                             getattr(system.devices["Acc"], "num_hbm", 1))
```

而 A6 选边探针（3523 行附近）构造 `est` 时只设了 `m / n / k / numOp`、
`pim_kv_runs`、`pim_shared_*`，**没有 `heads_per_hbm`**；`get_time_and_energy_runs`
与 `output_runs` 里也搜不到这个参数，不会自行推导。`heads_per_hbm` 正是
1411 行 `num_hbm_used = ceil(kv_heads / heads_per_hbm)` 建模 HBM 堆栈并行度所用
的那一项。

**这是一处事实上的代码不对称，不是已证实的病因。** 曾有过一个"`num_hbm=1` 的
模型因此被选错"的假设，**已被数据否定**：按全部配置统计，`num_hbm=1` 的两个模型
反而是 PIM 收益最大的（中位 −28.6% / −37.1%）。那个假设是只看 `k-hi` 一列
得出的。真实的触发条件是 **`k-hi` × 小模型**的交互，与 `num_hbm` 无单调关系。

另外，LLAMA3-8B 之所以是唯一会触发 A6 的模型，**机制上是 GQA 而非上述问题**：
它是六个模型里唯一 `group_query=4` 的，探针里
`cap = mq_query_capacity(...) // _gqa_group(system)` 把容量除以 4，`sweeps` 涨
四倍，而 `_tlb_plan_cost(...) * sweeps` 按 sweep 计费、不随 KV head 变少而减少，
于是 PIM 路径被算贵。**它在 `k-hi` 上选对，是这个与堆栈并行度无关的原因造成的。**

---

## 待办

1. **补跑结束后重跑本页的脚本** —— 结论建立在 54/78 个任务上。
2. **A3a / A3b / A4b 是否进入论文** —— 现在有数据支撑决定了。
3. **探针补 `heads_per_hbm` 与另外两处对齐** —— 属引擎改动，会改变 A6 结果，
   必须等 sweep 跑完，且改完要按 `docs/sessions/2026-08-31.md` 的办法做
   逐字节一致性验证。
4. **核对 §1 末尾那处与归档结果的口径冲突。**

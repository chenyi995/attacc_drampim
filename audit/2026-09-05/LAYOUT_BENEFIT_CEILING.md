# 布局收益：理论上限、当前拿到多少、哪些 workload 能扩大收益

补核说明：真实 B1 有两个新检索 chunk 与 owner 偏移对齐，不需要修正；k8 新增 diff token 总数为 1008，不是 1024。下文每 agent 每轮 16 token、全局轮 128 token 是通常形状的解释，不是无例外的计数；表中唯一地址行数来自真实 ledger，仍成立。逐请求 k4/k8 见 [性能分析](../../docs/analysis/README.md)。

后续口径：[四项指标与每 KV head 八通道](METRICS_AND_EIGHT_CHANNELS.md)。本文原有数值属于原配置，八通道假设单独记录。

本轮针对 chenyi9 转来的 B1/T9、异步调度分析逐项核验。实现快照为 `da220d1ea5c57358fb2f2d26a897a5178d0b7f0c`；B1/T9 等输入和原 `b1_levers.py` 是工作区未提交内容，证据归档保留了各自 SHA256。**结论：布局仍有空间，但静态占行、通道扫描量和端到端收益是三种不同指标。当前没有 B1/T9 的真实阶梯结果，不能给出它们已兑现多少端到端性能上限。**

## 1. 先看结论

| 对象与口径 | 当前拿到的收益 | 相同条件下的乐观空间 |
|---|---|---|
| B1，A4c，累计 diff 占行 | 少 44.44%，拿到理想可省行数的 57.14% | 同 agent 连续紧排可少 77.78%；当前 allocator 未实现这个更强条件 |
| B1，A4e 相对 A4c，原回复的 reused 读集 | 最忙通道扫描 token 数少 10.94% | 允许每个请求单独选最优表时，乐观可省上界 20.69%；单一持久表未必能同时实现 |
| B1，A4e 相对 A4c，完整 prompt 快照 | 最忙通道扫描 token 数少 7.05% | 相同放宽条件下，可省上界 17.05% |
| 已有 small/out 小跑，A3b→A4e | E2E 少 0.517%，TBT 少 0.833% | 输入不是 B1，不能拿来除以上面 B1 的上限 |

这里的“完整 prompt 快照”含当轮 fresh 内容，每个请求只取一次，还没有加本次 decode 新生成的 token。“reused 子集”按 reused=True 的绑定取行，包含本轮和继承的 diff 及展开的 shadow，排除当轮 fresh own；它不是 A4c 的实际 GPU 回读量，也不是 A5 的完整 prefill 扫描量。两个结构指标都没有按真实 query 数、decode 步数或调度次数加权。

## 2. A4c：同 agent 的跨轮 diff，还没有完全聚在一起

背景：一个共享 chunk 有 256 token；每轮每个 agent 修正 2 个 chunk，每块 k=8，所以每轮产生 16 个 diff token。A3b 允许同一轮这些 diff 自然连续追加，合成一个 burst；不能错误地按每个 chunk 都算一行。不同轮之间插入其他对象后，旧 burst 和新 burst 分开。

当前 A4c 把 master 与 diff 分区，但只有一个全局 `diff_cursor`，所有 agent 依次向同一个 head 的 diff 区追加。B1 每个全局轮追加 8×16=128 token；同一 agent 相邻轮的 diff 大约每两轮才共享一行。**其他 agent 的 master/输出不会推进 diff cursor，其他 agent 的 diff 会。**

按一个 agent、一个 head 归一化，真实 ledger 的结果是：

| 读集 | A3b | 当前 A4c | 同 agent 连续紧排下界 |
|---|---:|---:|---:|
| 第 8 轮的一次扫描 | 8 行 | 4 行 | 1 行 |
| 第 1–8 轮每轮各扫描一次，累加 | 36 行 | 20 行 | 8 行 |

末轮理想可少 87.50%，即 diff 这一项的占行比可为 8×；累计理想可少 77.78%。当前累计拿到 `(A3b−当前)/(A3b−理想)=(36−20)/(36−8)=57.14%` 的可省行数。这不是获得了这么多“性能收益”。

一般式：设行容量 B，每轮已合法合并后的 burst 含 d_r 个 token，且所有这些历史 diff 仍在读集中；在各 burst 原本行对齐的条件下，

```text
A3b diff 占行       = Σ ceil(d_r / B)
连续紧排的占行下界  = ceil(Σ d_r / B)
```

当 d_r 固定且小于一行时，末轮 diff 占行比受轮数和 B/d_r 限制。B1 的轮数有限，不能用无限轮极限来报这份输入的收益。上述连续紧排是机制下界，不是当前全局 diff 区的已实现结果。

正文 [04-design.tex](../../../KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex) 第 79–90 行明确说一个 agent、一个 head 的修正连续紧排，并跨轮共享行；[贡献 README](../../docs/README_contributions.md) 也这样描述。实现 [PhysicalLedger](../../src/workload_runner.py) 第 942、997–1000 行只保证全局 diff 区紧排。因此“布局潜力已经用尽”不成立；更准确的表述是：**当前已经隔开 master 与 diff，但尚未保证同一 agent 的跨轮 diff 连续。** 这是 Fugue 新增布局的覆盖程度问题，AttAcc 原本没有这种 per-agent diff 布局。本轮仅记录差距，不把理想布局冒充当前实现，也不据此改变模拟器。

加入 master、own 和历史输出，收益会被稀释。对 B1 的 64 个 worker 请求，每轮取完整 prompt 的 K 行集合，累计总行数为 **11,320→10,296→9,528**；当前只少 9.05%，理想少 15.83%。不包含最初 owner 的全 corpus 导入；把它加入分母会继续稀释。这些是各请求唯一地址行数的和，V 有同样的占行几何。

## 3. A4e：通道分散的上限取决于原本有多不均衡

B1 的 TINY 几何是一个 HBM 中 8 个 KV head，16 个 channel，因此每 head 只有 2 个 channel。只看总工作量不变、等成本扫描的通道并行效应，最极端从全挤一个通道到平均分到两个，局部扫描上限为 **2×**。这不是全部注意力、更不是 E2E 的 2×；换成 p 个通道时，这个有条件的并行上限才是 p。

针对 B1 本身，我保持 A4c 的 diff 位置，比较 A4c→A4e，另算一个乐观下界：每个请求可以各自把 master 整块重新分到两个通道，diff 仍固定末通道；用 subset-sum 精确求这个请求的最小最大通道 token 数。当前是所有请求共用一个持久软件表，不能逐扫描搬块，所以此下界通常比真实可实现的最优表更乐观。

| 每个请求各取一次的读集 | A4c 最忙通道 token 数之和 | A4e | 乐观最小值 | 当前减少 | 乐观可省上界 | 当前占该乐观可省量 |
|---|---:|---:|---:|---:|---:|---:|
| 原回复的 reused 子集 | 139,728 | 124,448 | 110,824 | 10.94% | 20.69% | 52.86% |
| 完整 prompt，含当轮 fresh | 150,504 | 139,888 | 124,840 | 7.05% | 17.05% | 41.37% |

本表含 B1 全部 65 个请求；owner 在 reused 子集中没有读量，在完整 prompt 中包含初始导入的读量。[会话开始时的 b1_levers.py](archive/layout_ceiling/layout_ceiling_b1_levers_initial.txt) 第 73–84、112–115 行，实际用 A3b 比 A4e，得到少 9.46%。它同时改变了 diff 放置，不能归为 A4e 相对 A4c 的单独收益。“rows”在这个通道计数中是扫描的 token 数，另一个 repair 指标才是 DRAM 占行。

软件表没有达到逐请求乐观下界，也不是每个请求都改善：完整 prompt 快照有 12 个请求的最忙通道扫描量比 A4c 更高。原因包括全局共读关系不能对每个请求同时最优，以及当前表按共读块数量做贪心选择，未直接最小化每个请求的字节数、固定 diff 负载或 Ramulator 时间；见 `_block_slot_table` 第 839–876 行。这里是在解释当前收益，不是要求增加论文没有 claim 的额外优化。

## 4. 已有真实小跑：布局合计 E2E 改善约 0.52%

来源是 `scratch_0905/small/out/dag_A3b.json`、`dag_A4c.json`、`dag_A4e.json` 及同目录 CSV/log，本轮只读取现有结果。原始文件及 SHA256 已归档在 [existing_run](archive/layout_ceiling/existing_run)。输入 `wl_small_A4R4.json` 实际是 5 个请求、全部 tier=0：每个 worker 将四组 doc/doc/user 拼成一份 prompt，然后只执行一次 prefill 和一次 decode，没有 parent 输出或跨轮依赖。不能因文件名里的 A4R4 就视作 B1 那种多轮 DAG。模型标识 CACHEBLEND-TINY/A100a；三档均 recompute k=8、batch=8、history=0、pipeline=true，corrected-row hash、传输字节、prefill 行数及共同 GPU 工作一致。

| 档位 | E2E，ms | 加权 TBT，µs | 全部 PIM lane 服务时间之和，ms |
|---|---:|---:|---:|
| A3b | 96.991918 | 235.622266 | 403.582840 |
| A4c | 96.735997 | 234.625769 | 393.610891 |
| A4e | 96.490291 | 233.659977 | 393.500253 |

| 相邻档比较 | E2E 减少 | TBT 减少 | lane 服务时间之和减少 |
|---|---:|---:|---:|
| A3b→A4c | 0.264% | 0.423% | 2.471% |
| A4c→A4e | 0.254% | 0.412% | 0.028% |
| A3b→A4e | 0.517% | 0.833% | 2.498% |

这组结果还不能宣称 PIM 节能：A4e 的 PIM 能量比 A3b 高 0.813%。这里只报告原产物方向，没有由占行减少推导节能。

结果报告没有保留 events 和代码 revision 指纹。启动时序支持它来自新一轮去除 decode 固定传输延迟后的执行，但不足以严格绑定当前 commit。FlashAttention 有旧 session 的说明和 `run_dag_ladder.sh` 默认 flash 支持，但环境变量可覆盖，JSON/log/CSV 没有保存 GPU 模式，不能写成产物已经独立证明开启；HBM 数也须结合启动上下文解释。旧 session 第 16 节中的约 220 ms 属于先前执行，不和这里的新数据混算。chenyi9 已确定的 decode 小流量固定传输开销忽略口径，本轮沿用。

“PIM 只有 40% 忙”也要说清分母：PIM 总服务时间是 16 条 lane 相加，除以 `16×decode 窗口` 才得到约 41.8%；除以 `16×完整 E2E` 是 26.0%。这些都不是关键路径中 PIM 所占的比例，不能据此推断异步一定能填满空闲。

共同 GPU 服务量为 70.770256 ms。日志确认 A100a×1、TP=1；该统计只累加同一个 `GPU` 串行资源，不是多个 GPU 的时间求和。若只改布局且保持 GPU 工作及分批不变，即使将所有 PIM 耗时都设为零，E2E 也不能低于 GPU 服务量；得到极宽松上限 1.371×，最多少 27.03%。这只是排除不可能数值的资源下界，不是当前布局能够达到的上限。

## 5. 哪些情况能扩大布局收益

**A4c：跨轮反复读旧 diff，每轮修正量小，且全局 diff 区中夹入的其他修正少。** 保持模拟器不变，用现有输入做同一 ledger 几何计算，得到：

| workload | 每 agent/head，8 轮累计 diff 行：A3b→A4c→紧排下界 | 当前 diff 行减少 | 完整 prompt 总行减少 |
|---|---:|---:|---:|
| B1：8 agent、2 chunk | 36→20→8 | 44.44% | 9.05% |
| 4 agent、2 chunk | 36→12→8 | 66.67% | 13.71% |
| 8 agent、1 chunk | 36→12→8 | 66.67% | 16.77% |
| 8 agent、4 chunk | 36→36→8 | 0.00% | 0.00% |

所以，把 8 agent 减成 4，或每轮 2 chunk 减成 1，确实扩大了**当前实现的几何收益**；还不能据此报性能收益。更多轮数只有在旧 diff 仍被读取、且 master/own/输出成本没有更快吞掉其占比时，才更有帮助。

“4 个 chunk 每轮已经填满一行”的说法不准确：每 agent 是 4×8=32 token，八个 agent 合计才是 256 token，恰好使全局 diff 区每轮推进一整行。此时同 agent 跨轮紧排仍有潜力，但当前全局 cursor 无法获得；还可能因为修正集中末通道而损失平衡：本例最忙通道的唯一地址行数之和从 1084 增到 1159。

**A4e：少量热点共读块在朴素轮转中撞到同一个通道，软件表能把它们分开。** 例如每 head 两个通道时，经常一起读的两个块在写入序中相隔两块，会落到同一 slot。稳定、相互兼容的共读关系有利于表发挥作用；若读整个已均匀铺开的 corpus，或者很多互相冲突的共读组合都必须兼顾，空间就小。原输入的连续 chunk 控制点接近朴素轮转已平衡的情况：按 reused 子集，A4e 相对 A4c 只少 0.26%。这种输入可作为负对照；不能只保留有利的热点 case 就称普遍收益。

**E2E：被省下来的扫描时间必须暴露在完成时间上。** 如果 GPU 线性层或传输已经决定了每步完成时间，PIM 更快可能只增加等待。条件性的 Amdahl 估算为 `S_e2e = 1 / ((1−f)+f/S_scan)`，其中 f 必须是可被布局缩短的关键路径份额，不能用全部 lane 的平均忙碌率代替。减小共同固定成本或改成更偏注意力扫描的合理输入，可能提高曝光度；所有档必须使用同一输入和规则，不能为放大收益单独抬高 A3b 成本。

## 6. 对转来回复的判断：异步是另一种执行条件，不保证布局收益变大

写入侧确有跨轮 master/diff/输出分离，以及短输出 agent 提前退出预约循环；但地址在仿真开始前按预约顺序建立，并非按真实完成时刻重放写入。输出又按固定 256-token 对象进入 ledger，不能把 token 交错直接当成逐 token 物理碎片。T9 将部分输出从 128 缩到 32/8，在本次 diff/master 唯一占行统计中不改变行数；通道的有效扫描 token 数却会改变。

调度侧确有整 tier 屏障：`workload_runner.py:4762` 的 `request_ready=previous_tier_done`，及第 5171 行的整 tier 收尾。因此短回答可能等待同 tier 长回答。但仅移除屏障，不会自动重排已经预约的地址，也不保证实现完整异步：事件仍按加入顺序预约资源，decode GPU 组也按当前 tier/step/layer 建立。独立 agent 用四个固定时长的抽象事件验证了这种顺序限制；这不是性能实验，也不作为新的公平性整改项。

最关键的修正是：**当前 A6 比较每个请求自身的 GPU/PIM 服务时间，没有输入 GPU 排队或空闲状态。** 同一形状的请求不会仅因 GPU 正忙就改变选边；这符合 chenyi9 已确定的简单逐 request 口径。原脚本的 `28/37` 来自第 104–109 行的手工校准估式，包含固定扫描、GPU 和带宽系数，没有读取真实 A6 chooser 决策。它不能证明 B1 会选择多少 PIM 请求，更不能证明异步时会转移多少请求。

异步可能让本来就适合 PIM 的短 attention 与其他请求的 GPU decode 重叠，也可能增加竞争或改变 batch 效率；线性层仍需 GPU。因此正确说法是“异步提供了另一种可能暴露收益的运行条件，方向与幅度需要真实相同条件比较”，不能承诺“GPU 忙就交给 PIM，所以布局收益必然变大”。共同调度规则一致用于各档，本身不等于私加某一档贡献；这里不要求各档从共同规则中受益幅度相同。

## 7. 证据与计算口径

本轮没有修改实现、workload、论文或已有运行结果，没有启动 Ramulator 或新的性能任务。结构探针调用真实 `build_reuse_plan`、TLB、`PhysicalLedger` 和 extent helper；A4e 下界只是审计里的数学反事实。现有执行仍从 `get_time_and_energy_runs` 进入 Ramulator 获取扫描价格，不能因审计用了结构公式就把模型本身说成人工拟合。

结束检查发现 `output/analysis/b1_levers.py` 在会话中另有并行更新，新增了 headroom 诊断，原三项比较/选边公式仍保留；本审计没有写入该文件。本文关于转来回复的源码行号指向已归档初版，恢复文本的 SHA256 与初始快照完全一致；最终观察版本另行保存。新增诊断不是本文上界计算的来源，见 [并行变更记录](archive/layout_ceiling/layout_ceiling_concurrent_change.json)。模拟器和 workload 的初末摘要保持一致。

必须区分“不同地址行数”和 ACT 命令数：独立 trace 证据中，两个 V 行可按 row0→row1→row0→row1 重访，而同 row 的两个 extent 可以连续命中。ACT/PRE、重复 QK/PV 扫描、MQ、refresh 和最终时间仍由真实 trace/Ramulator 决定。结构百分比既不是 ACT 实测，也不是延迟的硬上限。

- [A4c ledger 结构证据](archive/layout_ceiling/layout_ceiling_a4c_evidence.json)、[计算脚本](archive/layout_ceiling/layout_ceiling_a4c_probe.txt)。
- [A4e 相邻档与乐观下界](archive/layout_ceiling/layout_ceiling_a4e_evidence.json)、[计算脚本](archive/layout_ceiling/layout_ceiling_a4e_probe.txt)、[独立复核](archive/layout_ceiling/layout_ceiling_a4e_review.json)。
- [已有小跑提取](archive/layout_ceiling/layout_ceiling_results_20260905.json)、[提取脚本](archive/layout_ceiling/layout_ceiling_results_20260905.txt)。
- [28/37 来源](archive/layout_ceiling/layout_ceiling_side_proxy.json)、[调度结构反例](archive/layout_ceiling/layout_ceiling_scheduler_probe.json)、[行访问顺序证据](archive/layout_ceiling/layout_ceiling_trace_order_evidence.json)。
- [原始输入/源码/产物摘要清单](archive/layout_ceiling/manifest.json)、[最终验证](archive/layout_ceiling/validation.json)、[本轮 session](../../docs/sessions/2026-09-05-layout-benefit-ceiling.md)。

主审负责 A4e 上限和报告；`ledger_trace_boundary_audit` 独立核 A4c、正文和 trace；`independent_fairness_audit` 独立核调度/选边并复核主审 A4e 算法；`attacc_model_provenance` 独立提取既有运行结果。所有数字由保留的 Python 脚本读取源文件后生成。

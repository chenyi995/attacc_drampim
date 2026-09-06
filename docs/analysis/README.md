# 性能分析：Fugue 为什么能节省时间

实际命令见 [实验指导](../experiments/README.md)，实现是否符合声明见 [审计入口](../audit/README.md)。占行、实际扫描时间、E2E 分层分析。

## 1. 每一级省什么

| 比较 | 能省的工作 | 成立条件 |
|---|---|---|
| A3b→A4c | 紧凑 diff 减少跨轮零散修正涉及的地址行/行切换 | diff 小且反复读取；普通内容没有淹没它；集中末通道的负担不能抵消收益 |
| A4c→A4e | 分散共读 master，缩短最慢通道 | 朴素轮转中共读集中，软件表确实能分开；总读取量并未因此减少 |
| A4e→A5 | prefill attention 留在 PIM，避免驻留 KV 回读；MQ 与已接受的 PE 配置也作用于 decode | 长 KV、少量 query、较少 sweeps；MAC 仍需完成，GPU 线性层仍存在 |
| A5→A6 | 将 PIM 明显更慢的 prefill 转到 GPU | 混合短复用与长/新 prefill；两侧存在各自擅长的请求 |

扫描更快只有影响完成时间，才改善 TBT/TTFT/E2E。若 GPU/链路仍决定每步结束，PIM 更快可能只增加等待。A6 比较请求服务价格，不按 GPU 队列忙闲选边。

## 2. 低 AI、k4 为什么可能适合 PIM

AI 是算术强度，即每搬运一字节完成多少计算。长 KV 上只有少量 query 时，同一批 KV 读取能摊到的计算少；GPU 还需回读驻留 KV，PIM 可以在数据所在处做 attention。

设当前计算 token 数 m、回读量 R、实际 PIM 扫描地址集合 S。k 只改变需要新修正的行，不减少 fresh own/system token；继承的旧 diff 不重新计算。减小 k 可能减少 m、diff 和 MQ sweeps，但 GPU 也少算，原来重算的 token 改成复用还可能增大 R。

```text
G_i = 驻留 KV 回读时间 + FlashAttention(m_i, R_i+m_i)
P_i = Σ必要 MQ sweeps max_channel(真实 Ramulator 扫描时间)
      + context 返回时间
选择 PIM 当 P_i <= G_i；否则 GPU。
```

R 的单位是 token rows，回读字节量按实际 KV head 宽度换算；R 为零时不产生回读费用。Q 在选边估价中按 chenyi9 已定规则忽略，QKV 等共同工作不单边计入上述价格。FlashAttention 有分块、MMA padding 和占用率，k 减半不意味着 GPU 时间减半。八通道的最慢 lane 也必须从真实布局定价，不能简单取上下文长度/8。A5 的 TBT 变化还包含 decode 的 MQ/PE 作用，不能全部归因于 prefill 放在 PIM。

## 3. B1 的真实 k8→k4 工作量

来自真实 plan/TLB，无设备定价或性能模拟。m 和 MQ 容量不依赖本 head 分给几个 channel；八通道下的时间仍需另行定价。

| 请求类别 | 数量 | m：k8→k4 | m 减少 |
|---|---:|---:|---:|
| 语料 owner | 1 | 16640→16640 | 0.00% |
| 话少，首轮 | 4 | 288→280 | 2.78% |
| 话少，后续 | 28 | 32→24 | 25.00% |
| 话多，首轮 | 4 | 528→520 | 1.52% |
| 话多，后续 | 26 | 272→264 | 2.94% |
| 话多，后续 | 2 | 264→260 | 1.52% |


后续话少含16个 fresh token，通常修正两个新 chunk，m=16+2k，从32降到24，只少25%。若没有 fresh token、只有两个需修正 chunk，才是 m=2k 的16→8；B1不是这种纯复用输入。

实际 w00_t01：prompt 保持 1440 token，m 32→24，R 1408→1416；PIM 原始 scan 输入 1472→1456 token rows，包含 shadow/继承修正，不等于 prompt 长度。TINY 的 MQ sweep 4→3，LLAMA3 GQA4 为 16→12。

两个 writer 请求的新 chunk 中有一个与 owner 绝对偏移恰好对齐，本来就不需修正，因此只少4个计算 token。B1 k8 新增 diff 总数实际 1008，k4 为 504；“每轮两个 chunk 全都产生 k 修正”是通常形状，不能当无例外计数。

## 4. k4 会不会扩大 A5/A6 差距

不一定。忽略排队、只看请求的局部服务价格，A5 用 P_i，A6 用 min(P_i,G_i)，A6 的选择优势为 max(P_i−G_i,0)。这不是 E2E 公式。

- 已适合 PIM 的短请求：两档都走 PIM，k4 共同减少 sweeps，没有额外选边收益。
- 原来 GPU 更快、k4 后 PIM 更快：A5 的亏损消失，该请求上的 A5/A6 差距收窄。
- 仍然很大的 fresh/长 prefill：k 几乎不改其工作，A5 仍可能慢，A6 可以送 GPU；B1 大语料 owner 的 m 不变。
- 混合 workload：短请求共同成本下降，可能让避开大 prefill 的优势占比更明显；也可能更多请求两档同侧，使差距缩小。绝对差、加速比、E2E 都不保证单调。

因此同时看真实 PIM/GPU 请求数、各类 t_xpu/t_bank、四项延迟。更多 PIM 请求不等价于更大的 A5/A6 收益；全短输入更适合展示 A4e→A5。当前没有八通道 B1 k4/k8 的真实选边结果，不能沿用手工校准的28/37。

## 5. 八通道与布局空间

八个等成本块集中在两个通道时，最慢 lane 做四块；铺到八个通道可从4t降到t。实际表必须真得到这个分布。完整共读集合可能掩盖热点，使当前贪心表退回朴素分布；详见 [八通道专项](../../audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md)。

当前 A4c 是整个 head 的全局 diff 追加区，其他 agent 的 diff 仍夹在同一 agent 跨轮修正之间，并非正文更强的每 agent 连续紧排。k4 减少全局 diff 占用可能帮助共享行，多通道本身不消除这一限制。

## 6. 短输出将后继 diff 推到半个 column 的机制

chenyi9 指的是连续追加的起点偏移：4-token 输出占半个 column，后继原本一列大小的 diff 从半列开始，读它就涉及两列。分离普通输出和 diff 的追加区，可以去掉普通输出造成的这个偏移；不是讨论4-token对象本身要不要整列读取。

当前 A3b 的短输出/master block 和新 diff burst 都按整行分配，短输出尾部不会继续给后继 diff 填充。因此当前布局未表达这一种跨对象连续追加，不能用已有结果宣称已经计入这部分收益。具体偏移/列命令与 AttAcc 来源核对见 [部分列追加审计](../../audit/2026-09-05/PARTIAL_COLUMN_APPEND_AUDIT.md)。diff 区自身累计仍可能跨列/跨行；这里消除的是普通输出带来的偏移。

## 7. 已有证据

- [布局上限](../../audit/2026-09-05/LAYOUT_BENEFIT_CEILING.md)：原两通道 B1 的真实结构；占行比例不是性能。
- [四项指标](../../audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md)：旧五请求小跑 TTFT/TBT/E2E，不能作为多轮八通道结果。
- [逐请求 k4 数据](../../audit/2026-09-05/archive/docs_roles_k4/k4_prefill_shape_evidence.json)。
- [最慢通道核验](../../audit/2026-09-05/SCAN_MAX_AUDIT.md)：执行/A6 均取 max(实际耗时)，lane-sum 仅诊断。

## 8. 链路按原 NVLink 配置

chenyi9 裁决：GPU 与 AttAcc/PIM 保持原 NVLink 连接，当前跑法沿用 `--pim-link nvlink3`。A5/A6 的分析使用这一固定硬件条件下的实际价格、工作量和选边结果。

此前讨论的 NVLink/PCIe 两级方案不采用，容量阈值示例不作为实验参数；原讨论保留在 [链路历史记录](../../audit/2026-09-05/LINK_TIER_ASSUMPTIONS.md)。

## 9. Scan 收益如何传到 TBT，GPU 与 pipeline 的作用

[Decode scan / TBT 专项](../../audit/2026-09-05/DECODE_SCAN_TBT_PIPELINE.md) 区分 Fig. 3b 的单 K 扫描、C1 的完整 decode scan 和 TBT。给出已有配对结果的降幅，解释“两个百分比的比值”与“省下的微秒传递率”的区别。

GPU 前后处理实际变快可提高 scan 在 TBT 中的占比；提高 FLOPS 不会消除 AttAcc 原有 Norm/激活固定项。已有运行确认 Flash/pipeline 开启，但新 DAG 的按追加顺序预约仍有已就绪工作错过 GPU 空窗的证据。具体候选、上游是否已有和独立检查均在专项中。

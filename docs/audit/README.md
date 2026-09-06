# 审计入口

这里检查实现、论文 claim 和相对公平性。怎么跑见 [实验指导](../experiments/README.md)，为什么节省时间见 [性能分析](../analysis/README.md)。

| 对象 | 入口 | 范围 |
|---|---|---|
| 已定 NVLink 配置 | [链路裁决](../../audit/2026-09-05/LINK_TIER_ASSUMPTIONS.md) | 沿用原 NVLink；两级方案不采用，已关闭讨论 |
| scan 是否误取平均 | [最慢 channel](../../audit/2026-09-05/SCAN_MAX_AUDIT.md) | 当前五档执行与 A6 取 max(实际耗时)，独立探针通过 |
| 短输出引起后继 diff 列偏移 | [部分列连续追加](../../audit/2026-09-05/PARTIAL_COLUMN_APPEND_AUDIT.md) | 当前 A3b 行对齐是否遗漏此效应；AttAcc 来源与收益方向分开 |
| 指标和八通道配置 | [指标专项](../../audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md) | scan/TBT/TTFT/E2E 定义、旧数据缺什么 |
| 布局与正文覆盖 | [布局上限](../../audit/2026-09-05/LAYOUT_BENEFIT_CEILING.md) | 全局 diff 区不保证每 agent 跨轮连续；结构不能当实测 |
| 既有 case 和用户裁决 | [CURRENT_ISSUES](../../audit/2026-09-05/CURRENT_ISSUES.md) | 按该文档标注 revision 解释，不表示后续所有修改都已完整重审 |
| 历史与证据 | [日期入口](../../audit/2026-09-05/README.md)、[archive](../../audit/2026-09-05/archive/README.md) | 按快照解释，已接受共同近似不自动重开 |

A1/A2 可作为独立 baseline；A3b 起按既定 claim 逐档比较。A3b 同轮连续 diff 可合并，跨轮按旧地址继续引用。共同 AttAcc 近似和用户已定链路计价沿用；新增候选问题先说明上游是否建模、影响哪档、证据边界，再由 chenyi9 判断，不因收益不够大自行改模型。

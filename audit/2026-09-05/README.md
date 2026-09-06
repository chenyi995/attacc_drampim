# 2026-09-05 Audit：从这里开始

最新补核：[短输出挤偏后继 diff 的部分列追加](PARTIAL_COLUMN_APPEND_AUDIT.md)——用户所指机制成立；当前行对齐与列命令取整尚未验证这一收益，分别说明 AttAcc 来源及公平性方向。

新增核验：[scan 取最慢 channel](SCAN_MAX_AUDIT.md)；文档用途分类与当前跑法见 [docs 入口](../../docs/README.md)。

最新口径：[四项指标与每 KV head 八通道](METRICS_AND_EIGHT_CHANNELS.md)——decode scan latency、TBT、TTFT、E2E 的定义及八通道受控比较。

本次新增：[布局收益上限与 B1/T9 分析](LAYOUT_BENEFIT_CEILING.md)——分清 diff 占行、A4e 相邻档通道收益和已有 E2E；说明何时收益可扩大。

**主体审计 `bb19f31`，已补核至 `ff5b91e`。既有实现问题见 [CURRENT_ISSUES.md](CURRENT_ISSUES.md)；本次布局收益见上方专项。**

上轮跨轮断链、prefill 漏 diff、A2 重算、collector 公式和能量诊断等具体反例已修复。本轮另查到条件性边界；当前默认 recompute、batch=8、history_len=0 及 turns 输出指纹规则不触发这些控制条件，不据此否定默认阶梯。

| 想看什么 | 入口 |
|---|---|
| 已关闭与剩余边界、是否影响默认配置 | [结论清单](CURRENT_ISSUES.md#decisions) |
| 实际旧 diff 读集、CacheBlend 取整、parent 来源 | [C8](CURRENT_ISSUES.md#c8) |
| batch=1 / A2 history>0 的真实时间戳 | [C6](CURRENT_ISSUES.md#c6) |
| AttAcc 能量、诊断与会话内新增链路规则 | [E1](CURRENT_ISSUES.md#e1)、[E2](CURRENT_ISSUES.md#e2) |
| 相邻档位、论文 claim 和 workload 范围 | [贡献核对](CURRENT_ISSUES.md#contributions-check) |
| 本轮为何更新、独立 agent 和验证方式 | [session](../../docs/sessions/2026-09-05-bb19f31-fix-verification.md) |
| 原始证据与历史 | [archive](archive/README.md) |

只做 audit 与文档，没有修改实现、论文或已有结果，没有运行性能模拟。接受用户已裁决的 AttAcc/共同近似；新增边界先说明条件和影响供过目，不自动升级为默认比较不公平。

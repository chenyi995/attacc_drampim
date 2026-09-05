# 2026-09-05 Audit：从这里开始

**这一轮的问题在 [CURRENT_ISSUES.md](CURRENT_ISSUES.md)。** 不需要按时间顺序读历次报告。

这份文档先解释系统与七档，再逐个 case 说明：论文声明什么、用什么小输入验证、应当怎样、代码实际怎样、AttAcc 是否已有对应建模、影响哪两档，以及哪些事项已确定、哪些实现仍待验收。

| 想看什么 | 打开哪里 |
|---|---|
| 已经确定、交给执行 agent 的事项 | [裁决清单](CURRENT_ISSUES.md#decisions) |
| 贡献 README 的四个例子与实现是否对应 | [四贡献核对](CURRENT_ISSUES.md#contributions-check) |
| 本轮需要过目的 case 与修改方向 | [CURRENT_ISSUES.md](CURRENT_ISSUES.md) |
| 原始代码定位、JSON、独立 agent 和历史裁决 | [archive/README.md](archive/README.md) |
| 本次为什么这样整理、验证了什么 | [session](../../docs/sessions/2026-09-05-audit-docs-cleanup.md) |

当前口径：接受 AttAcc 和各档共同的模型限制；不强求绝对精度。FlashAttention 必须共同启用。C2/C3/C6 已裁决修改；C5 按实际项估算并忽略 Q，C7 同轮/跨轮边界已定；C4 按共同近似记录，C8 按旧 master/diff 原址继续引用执行；本轮建模口径已定，实现仍待落实和验收。A1/A2 是独立 baseline，A5 的 prefill+MQ 是已接受机制包，A6 是简单逐 request 比价。

源码审计时点为 `8c51672`，上游对照 `c600051`。本次整理没有跑新性能实验，没有修改实现或已有结果。

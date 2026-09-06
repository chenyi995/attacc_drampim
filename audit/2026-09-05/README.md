# 2026-09-05 Audit：从这里开始

**主体审计 `bb19f31`，已补核至 `ff5b91e`。当前只需读 [CURRENT_ISSUES.md](CURRENT_ISSUES.md)。**

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

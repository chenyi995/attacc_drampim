# 2026-09-05 Audit：从这里开始

**最新复查版本 `fbe6756`，问题统一看 [CURRENT_ISSUES.md](CURRENT_ISSUES.md)。**

这轮还不能确认全部正确：跨轮旧 diff 的继承/实际 prefill 读取仍有缺口，tier 报表仍不同口径；能量诊断也需同步。GQA、diff 行地址隔离、默认 FlashAttention 和 A6 指定估价修改已经通过针对性复查。

| 想看什么 | 打开哪里 |
|---|---|
| 已修与未修清单 | [最新结论](CURRENT_ISSUES.md#decisions) |
| 多轮 a/c 原址引用、实际扫描和 A2 软件工作量 | [C8](CURRENT_ISSUES.md#c8) |
| 相同完成时间为何报出不同 tier 时间 | [C6](CURRENT_ISSUES.md#c6) |
| 能量来自哪里，新增外推是否共同适用 | [E1](CURRENT_ISSUES.md#e1) |
| 与论文四项贡献和 workload 的关系 | [贡献核对](CURRENT_ISSUES.md#contributions-check) |
| 本轮检查方式、修改文档的理由、独立 agent | [session](../../docs/sessions/2026-09-05-fbe6756-fix-verification.md) |
| 原始证据和历史裁决 | [archive 索引](archive/README.md) |

按用户口径接受 AttAcc/各档共同近似，FlashAttention 和 pipeline 共同开启；不重复请求已确定的建模裁决。此次只改 audit/session 等文档，没有改实现、workload、论文或已有结果，没有运行性能模拟。上游对照仍是 `c600051`。

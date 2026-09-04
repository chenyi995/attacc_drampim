# sessions/:每日调整记录

每天一个文件(`YYYY-MM-DD.md`),登记 **chenyi9 当天的裁决与口径调整**
(记名规则:文件中写决策账户名,不写真名)。与审计总台账的分工:
台账(`../README_manual_audit_findings.md`)按**问题条目**(R/U)组织,
本目录按**时间线**组织——某天改了什么口径、下了什么指令,来这里查。

已有:2026-08-24(阶梯重定义)、2026-08-25(审计与 workload 有效性)、
2026-08-26(引擎全覆盖与运行基建)、2026-08-27(head→HBM 重映射)、
2026-08-28(套件出数、debate 恢复、结果分析与原始数据入库)、
2026-08-29(magic-number 核查 → 参数化 sweep 重构)、
2026-08-30(sweep 多上游 workload 的 parent_out 阻塞修复 + 集群化运行记录;
由接手运行的 AI 登记,非 chenyi9 裁决)、
2026-08-31(引擎提速两轮、逐字节等价验证口径、**sweep 的内存可行性边界**
与 N-hi 停放裁决、编排的三个控制变量教训)。
2026-09-03(**一天两条线,已合并**。squire 侧:A3b..A6 换成 striped-append、
heads-per-HBM 按 per-GPU 重算、**真实 extent 进 Ramulator**(修正各占一行 vs
打包进 diff 通道,ACT 数才算得对)、手算校验、文档大清理。athena 侧:workload
重设 C=32→16 / sys=16→256(收益从 3 行对 4 行变成 2 行对 4 行)、
`collect_dag_ladder.py` 漏收 A3b/A4b 的 bug、按机器拆成 athena / squire 两份
跑法文档、78 任务的重跑编排)。
**跑批交接见 `2026-08-30-HANDOFF.md`**(athena Slurm 集群上 756-run sweep 的
运行状态、续跑方式、资源纪律与已知问题——接手跑批先看这页)。
更早的移植期(2026-08-22/23,
MQ 命令 C 模型与 822 分支启用)见 delta 文档 §1–§3 与 experiment 分支
存档。

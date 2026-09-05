# Session 文档索引

记录本分支每次会话中实际发生的代码、模型口径和文档变更，尤其记录**为什么修改**、依据来自哪里、如何验证，以及哪些事项尚未完成。

| 日期 | Session | 内容 |
|---|---|---|
| 2026-09-05（口径确定与交接） | [已定原则、待执行和验收](2026-09-05-audit-decisions-finalized.md) | C8 按 a/c 原址继续引用要求对齐；本轮无待重复裁决口径；明确不代表代码已修 |
| 2026-09-05（行列计时澄清） | [C4 与实际 decode 扫描输入](2026-09-05-row-column-audit-clarification.md) | 确认 ACTAB/PREA 由 Ramulator 处理；区分列地址边界与已证明性能问题；独立检查 reads/plan_reads |
| 2026-09-05（裁决与未定项复审） | [C5 实际项估价、round 边界及 C4/C8 复审](2026-09-05-rulings-and-pending-reaudit.md) | 记录 C2/C3/C6 待执行、Q 忽略裁决；关闭同轮合并指控，补查两轮旧 diff 继承；只改审计文档 |
| 2026-09-05（贡献例子对照） | [按贡献 README 核对实现与 audit](2026-09-05-contributions-alignment-audit.md) | 四项贡献配置差分、独立布局复核、真实构图字节和 MQ 命令检查；区分条件推导与性能结果 |
| 2026-09-05（审计目录整理） | [按论文 case 整理当前事项](2026-09-05-audit-docs-cleanup.md) | 当前问题统一到 CURRENT_ISSUES；历史证据归档、链接迁移、独立可读性核对 |
| 2026-09-05（相对公平裁决） | [沿用 AttAcc 共同限制、逐项先审后定](2026-09-05-attacc-relative-fairness-ruling.md) | 原始 AttAcc 来源复核；撤回共同模型限制的自动整改要求；FlashAttention 明确开启；各候选项先给上游建模与分档影响，由用户决定 |
| 2026-09-05（运行口径补审） | [pipeline、FlashAttention 与配置证据](2026-09-05-runtime-configuration-audit.md) | 检查入口与产物；独立复现 Python/native 空闲资源阻塞；区分历史 pipe=false、配置未知和 FA 解析模型；只改审计文档 |
| 2026-09-05（8c51672 复核） | [大量修复后的只读复核](2026-09-05-8c51672-fix-verification.md) | 关闭持久 row、STORE 计量、同计划等旧反例；复现 GQA 校验回归与 ledger→MAC 地址问题；独立 agent、定向测试及探针证据 |
| 2026-09-05（存储专项） | [存储与扫描对应性、A6 逐 request 口径](2026-09-05-storage-scan-and-request-choice.md) | 主审与独立 agent 复现五档 store/scan 通道错位与非持久 row；补查 A1/A2；撤回候选 DAG 强制要求；仅修改文档与审计证据 |
| 2026-09-05（cdd89db 复审） | [七档公平性只读复审与独立 agent audit](2026-09-05-cdd89db-fairness-reaudit.md) | 覆盖相对 AttAcc 的 110 个变更文件；小 workload/42 输入结构检查；两名 agent 复核；更新已修状态和剩余问题；未修改实现 |
| 2026-09-05 | [AttAcc 计量口径与 GPU query 旋转](2026-09-05-attacc-accounting-and-rotation.md) | 仓库/论文审计；确认七档设计口径；核查 PIM 时间来源；排除新增 DIE/TLB 成本与排队；删除 DIE 旋转；验证和剩余问题 |
| 2026-09-05（晚） | [审阅计量改动并修复 F01 / F02 / F04](2026-09-05-ladder-fixes-f01-f02-f04.md) | 判定上一会话改动对错并按小 commit 提交；A1 prefill 回到 GPU；A3b 按持久写入序 slot 放置；fresh prefill 按档选边；回归测试；未重跑阶梯 |

每份记录应包含以下信息：

1. 起始 revision、任务范围、用户确认的约束。
2. 修改前行为、问题原因、选择当前修法的理由。
3. 每个文件的具体修改及对执行、指标、接口兼容性的影响。
4. 实际执行的验证命令、结果和证据位置，区分真实仿真与设备桩。
5. 已完成与未完成事项、结果重跑要求、后续工作。

历史审计文件按其记录时点解释。后续修改在 session 中追加说明，不把旧反例覆盖成“从未发生”。

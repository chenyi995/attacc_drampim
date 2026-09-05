# 2026-09-05：本轮审计口径确定，交接执行与验收

chenyi9 询问是否还有待其审阅的事项。主审核对后，将最新说明落实到当前审计文档：本轮已提出问题的建模原则已经明确，当前没有待用户选择的模型口径。代码 revision 仍为 `8c51672a3ef8b936340354b3211963cde8945c49`；本次不执行修复、不修改论文或已有结果。

## 为什么更新 C8 状态

此前 C8 写作“是否需要保留旧 diff 原址，待过目”。chenyi9 随后明确举例：a 是共享 chunk，第一次复用生成 c，后面的内容正常追加，后续 f 仍 attention 最早的 a 和原来的 c。这已经说明期望的持久引用关系，继续询问是否允许用新 history master 占位替代它会重复请求已经给出的设计判断。

因此将 C8 改为执行目标已明确：保留原 master/diff 对象引用，下一轮扫描仍能追溯到旧地址；history_len 可以记录长度，不能替代已知对象的存储关系。这里记录用户明确的设计语义及主审据此整理的验收要求，不宣称用户另行批准了一种实现架构，也不宣称修复已完成。

## 本轮最终状态

- C1：各档共同开启 FlashAttention，pipeline 沿已定共同口径。
- C2/C3/C6：GQA、diff 地址和报表修复要求已定，待执行。
- C5：实际需要哪些项就估算哪些项；Q 输入传输接受忽略，真实历史回读按字节、空回读为零；论文与代码同步简单逐 request 比价。
- C7：同轮多组 diff 可正常追加紧排；跨轮已有输出/新 KV 后不能重新聚成同批。单 prefill 反例已撤回。
- C4：按共同 row/column 计时近似记录，ACTAB/PREA 和时序交给 Ramulator，不新增步长/跳转费用。
- C8：按原 master/diff 地址持久引用的语义对齐实现和跨轮验证。

## 文件变更与验证

CURRENT_ISSUES 更新标题、顶部交接表、C8 状态与验收要求，移除重复待裁决文字；两个审计入口同步状态。旧证据和历史 session 保留。新 session 说明为何从“待裁决”转为“设计语义已明确”，并区分口径、实现完成和验证通过。

执行 `python3 /tmp/finalize_audit_decisions.py`、`git diff --check`；保护文件摘要、论文摘要与文档链接检查记录在 [final_audit_decisions_manifest.json](../../audit/2026-09-05/archive/final_audit_decisions_manifest.json)。没有运行新模拟或改实现，没有 commit。执行 agent 应读取 [当前交接清单](../../audit/2026-09-05/CURRENT_ISSUES.md#decisions)，主审后续按同一口径验收。

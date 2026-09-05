# 历史报告和证据索引

这里保留追溯材料，不是当前整改清单。当前只读 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md) 即可；旧报告标题中的“阻断 / P0 / 必须修”按后续用户裁决重新解释。

| 用途 | 文档 / 证据 |
|---|---|
| C4 行列计时与默认路径 | [JSON](row_column_reachability_evidence.json)、[独立 probe](row_column_reachability_probe.txt)、[session](../../../docs/sessions/2026-09-05-row-column-audit-clarification.md) |
| round 边界裁决与多轮覆盖 | [两轮证据及 B0 摘要](independent_round_boundary_evidence.json)、[独立 probe](independent_round_boundary_probe.txt) |
| 最新裁决后的 C4/C7 复审 | [C4 JSON](pending_layout_reaudit_evidence.json)、[C4 probe](pending_layout_reaudit_probe.txt)、[C7 独立 JSON](independent_c7_reaudit_evidence.json)、[C7 probe](independent_c7_reaudit_probe.txt)、[验证记录](rulings_pending_reaudit_manifest.json) |
| 贡献 README 四例对照 | [主审 JSON](contributions_alignment_evidence.json)、[probe](contributions_alignment_probe.txt)、[独立 JSON](independent_contributions_evidence.json)、[独立 probe](independent_contributions_probe.txt)、[验证记录](contributions_alignment_manifest.json) |
| 8c51672 主审 | [技术报告](REAUDIT_8c51672.md)、[JSON](reaudit_8c51672_evidence.json)、[probe](reaudit_8c51672_probe.txt) |
| 独立存储复核 | [报告](INDEPENDENT_REAUDIT_8c51672.md)、[JSON](independent_8c51672_storage_evidence.json) |
| 存储到命令的地址边界 | [JSON](ledger_trace_boundary_8c51672_evidence.json)、[probe](ledger_trace_boundary_8c51672_probe.txt) |
| GQA / 能量 / 汇总专项 | [报告](MODEL_PROVENANCE_REAUDIT_8c51672.md)、[JSON](model_provenance_8c51672_probe.json) |
| pipeline / flash 入口复核 | [报告](RUNTIME_CONFIGURATION_AUDIT.md)、[JSON](runtime_configuration_8c51672_evidence.json) |
| 相对公平裁决与 AttAcc 来源 | [逐项记录](ATTACC_RELATIVE_FAIRNESS_REVIEW.md)、[原始源码摘录](ATTACC_UPSTREAM_REVIEW_EXCERPTS.md) |
| 更早的审计 | [初审](REPORT.md)、[cdd89db 复审](REAUDIT_cdd89db.md)、[旧存储专项](STORAGE_SCAN_CONSISTENCY.md) |
| 被替代的长篇修改建议 | [旧 README 快照](README_audit_fixes_history.md) |
| 本次简明案例的数据摘取 | [current_case_facts.json](current_case_facts.json)、[整理脚本](organize_audit_20260905.txt) |
| 本次文件移动和验证 | [cleanup_manifest.json](cleanup_manifest.json)、[验证脚本](validate_audit_cleanup.txt) |

JSON、日志、历史 probe 的计算内容保持原样；Markdown 只修复移动后的链接并标为历史。旧 manifest 和脚本中写死的原目录保留为执行时点记录，路径迁移见 cleanup_manifest；不要把历史生成/验证脚本当作当前自动执行入口。后续若复现历史调用，需按映射恢复它们所引用的当时路径。所有当前 case 都能直接查看原 JSON，不需要先运行脚本。

# 2026-09-05：按 row/column 命令口径收窄 C4

chenyi9 指出逐 channel 打开所需行、读取所需列、再提交下一段 KV 地址，其余交给 Ramulator。主审复核后确认这一职责划分正确；现有生成器一条 trace 包含同 channel 多段 extents，MACAB 前置条件由 Ramulator 处理 ACTAB/PREA。

为什么更新文档：此前用 V 子段起点随长度改变说明存储寻址边界，但这个事实不足以证明成本不同。需区分元素位置一致性和按行/列命令计时的公平性，不能凭列号变化追加人为开销或宣称 baseline 被罚。

独立 agent `ledger_trace_boundary_audit` 追踪真实 reuse plan、bindings、`_pool_reads`、ledger 和 Attention 生成器，确认 A3b 的 plan_reads 和实际 reads 不同：默认 decode 会补回 shadow master，因此简单前缀修正控制中，A3b/A4c 实际 master 扫描一致。手工子段能产生跨行命令，仅证明接口边界，不是这个默认路径的性能反例。

只给 CURRENT_ISSUES 的 C4 增补上述说明，保存新 probe/JSON 和索引。没有运行 Ramulator、没有性能结果、没有代码修改；C4 仍按既有共同近似规则记录。证据见 [当前 C4](../../audit/2026-09-05/CURRENT_ISSUES.md#c4) 和 [可达性 JSON](../../audit/2026-09-05/archive/row_column_reachability_evidence.json)。

执行：独立 agent 运行 `/tmp/ledger_prefix8_reachability_probe.py`；主审运行 `/tmp/clarify_row_column_audit.py`，并验证实现/已有保护文件摘要及 `git diff --check`。本轮为说明收窄，没有把尚未测量的 ACT/周期差异写成事实。

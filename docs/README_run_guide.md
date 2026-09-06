# 运行指南入口

当前跑法统一到 [运行协议](README_run_protocol.md)：model × workload × combo，每 GPU 5 个 HBM 栈、flash、pipeopt、k=8、batch 8，
baseline 跑七个 combo、sweep 只跑 A3b 与 A6。指标定义与 full events 见 [实验指导](experiments/README.md)。

chenyi9 裁决（2026-09-06）：PIM decode 路径的 attention 全部在 PIM 完成，包含本步 GPU 新生成的 K/V。
每步每请求只新增一个 token；其 Q 与历史 K 的 QK、历史 V 的 PV 扫描照常计时。
chenyi9 裁决：当前 token 对自身 K/V 的 QK、PV 忽略耗时，不另发一次 DRAM scan。
该贡献仍等待 Q 和本步 K/V 经原 NVLink 3 小量传输并落地，再合并 PIM 结果返回 GPU；从下一步起该 K/V 正常纳入历史扫描。
decode 不再执行 GPU local attention 或传输 GPU partial-LSE；小量传输保留字节数和带宽成本，不加固定启动延迟。
prefill 保留现有选边逻辑与消融档位。新报告以 `decode_new_kv_attn: pim` 和
`decode_new_kv_time_model: zero_current_token_qk_pv` 标识这一 decode 模型。

decode 关键路径（2026-09-06 修复）：`完整 QKV → Q 到达 → 历史 PIM scan → context 返回 → projection → FFN/Norm → 下一层 QKV`。
整请求 scan 与完整 Q/context 使用一致粒度；`decode_pipeline_granularity: whole_operation` 标明这一口径。
`pipeopt` 仍允许其他已就绪请求及独立 GPU/PIM/LINK 资源重叠；单次 GPU 算子和 PIM scan 的成本不因这次依赖修复而折减。

机制见 [性能分析](analysis/README.md)，实现判断见 [审计入口](audit/README.md)。原指南（2026-09-05 的 B0/S1–S8 矩阵）完整保存在
[分类前原文](archive/2026-09-05-before-categorization/README_run_guide.md.txt)，其 workload 与生成器归档在 `workload/probe/archive/2026-09-05-C-protocol/`。

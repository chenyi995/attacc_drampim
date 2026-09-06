# 运行指南入口

当前跑法统一到 [运行协议](README_run_protocol.md)：model × workload × combo，每 GPU 5 个 HBM 栈、flash、pipeopt、k=8、batch 8，
baseline 跑七个 combo、sweep 只跑 A3b 与 A6。指标定义与 full events 见 [实验指导](experiments/README.md)。

机制见 [性能分析](analysis/README.md)，实现判断见 [审计入口](audit/README.md)。原指南（2026-09-05 的 B0/S1–S8 矩阵）完整保存在
[分类前原文](archive/2026-09-05-before-categorization/README_run_guide.md.txt)，其 workload 可用 `LEGACY_MATRIX=1 gen_sweep.py --all` 复现。

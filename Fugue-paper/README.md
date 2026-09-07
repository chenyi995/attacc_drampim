# Fugue-paper — 最终实验材料

本目录仅保留各实验最后修改的图和最终数据。完整采样 CSV 和主图窗口 CSV 是同一最终数据的全量/子集，不是新旧两版。旧版图与历史脚本已移到本地 output 归档。

| 实验 | 最终结果与说明 |
| --- | --- |
| Experiment 1：Prefill attention 交点与 MQ | [README](Fugue-asplos-experiment-1-prefill-attention-crossover/README.md) |
| Experiment 2：MQ 的 link/cache/Q 敏感性 | [README](Fugue-asplos-experiment-2-mq-bandwidth-cache/README.md) |
| Experiment 3：CacheBlend 的 F1–F4 容量、延迟与能耗 | [README](Fugue-asplos-experiment-3-cacheblend-reuse/README.md) |
| Experiment 4：EPIC 长上下文 prefill | [README](Fugue-asplos-experiment-4-epic-long-prefill/README.md) |
| Experiment 5：EPIC 共享文档的并发 reader | [README](Fugue-asplos-experiment-5-epic-shared-readers/README.md) |

[完整复现指南](../docs/Fugue-asplos-reproduction.md) · [模型范围](../docs/Fugue-asplos-methodology.md) · [原始输入](../artifact/inputs/README.md)。

```bash
# 在仓库根目录运行；无需旧 output 或其它仓库。
python3 -m fugue all --jobs 8
# 仅从本目录最终数据重画图，不声称重跑仿真：
python3 -m fugue plot --from-paper --output output/Fugue-asplos-redraw
```

复现生成到新 output；本目录不会被绘图命令覆盖。共有 13 张最终图（PDF/PNG），实验 1–5 全部主结果已用自包含代码重跑并逐项比较；13 张 PNG 与原最终图逐字节相同。完整数值校验见 [复现检查](Fugue-asplos-reproduction-check.json)。

[独立源码包复现报告](../docs/Fugue-asplos-reproduction-checks.md)：50 个最终 CSV / 157,136 个数值字段核对通过，13 张 PNG 完全一致；无 .git 或旧 output 的源码包从头完成约 128 秒（本机、8 workers）。

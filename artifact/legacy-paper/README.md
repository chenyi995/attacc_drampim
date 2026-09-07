# Fugue：历史单模型实验归档

这里保留原 LLAMA-7B 单模型 Experiment 1–5 的最终图表、输入和来源记录。当前 KVChime 多模型结果见 [Fugue-paper](../../Fugue-paper/README.md)，历史 F1–F4 的定义不能用于解释新版 F0–F4。

| 实验 | 说明 |
| --- | --- |
| 1 | [Prefill attention 交点与 MQ](Fugue-asplos-experiment-1-prefill-attention-crossover/README.md) |
| 2 | [MQ 的 link/cache/Q 敏感性](Fugue-asplos-experiment-2-mq-bandwidth-cache/README.md) |
| 3 | [CacheBlend 的 F1–F4](Fugue-asplos-experiment-3-cacheblend-reuse/README.md) |
| 4 | [EPIC 长上下文 prefill](Fugue-asplos-experiment-4-epic-long-prefill/README.md) |
| 5 | [EPIC 共享文档 reader](Fugue-asplos-experiment-5-epic-shared-readers/README.md) |

在仓库根目录运行：

```bash
python3 -m fugue all --jobs 8 --output output/Fugue-legacy-fresh
# 从归档最终数据重画：
python3 -m fugue plot --from-paper --output output/Fugue-legacy-redraw
```

[历史复现检查](../../docs/Fugue-asplos-reproduction-checks.md)记录了 50 个 CSV、157,136 个数值字段通过校验，以及 13 张 PNG 逐字节一致。迁移保留原始图表、工作负载和来源文件的字节，仅更新 README 导航和目录校验清单。

[复现指南](../../docs/Fugue-asplos-reproduction.md) · [模型范围](../../docs/Fugue-asplos-methodology.md) · [固定输入](../inputs/README.md) · [原始检查记录](Fugue-asplos-reproduction-check.json)。

# KVChime：多模型最终实验

Experiment 1 已恢复原定的 2×3 六联布局，每模型一张，共四张；后续 performance/capacity 保持每张小图一个指标，共十张。正式数据与原始运行的 238,384 个数值字段一致。详见 [完成与校验记录](../docs/KVChime-reproduction-checks.md)。

- [Experiment 1：Prefill attention 的 GPU/PIM/MQ 边界](KVChime-experiment-1-prefill-boundary/README.md)
- [Experiment 2：MQ 对 link 与 cache 的敏感性](KVChime-experiment-2-link-cache-sensitivity/README.md)
- [Experiment 3：CacheBlend/EPIC 的 F0–F4](KVChime-experiment-3-software-reuse-F0-F4/README.md)
- [Experiment 4：共享 query 与跨 agent MQ](KVChime-experiment-4-shared-query-MQ/README.md)
- [Experiment 5：四种 prefill 选边策略](KVChime-experiment-5-device-selection/README.md)

LLAMA-7B、GPT-13B、LLAMA-65B 使用 AttAcc 原生模型；LLAMA3.1-8B 是显式 GQA 扩展。频率和面积属于 kvpim-rtl 的独立硬件实验，不按模型重复同一硬件数据。

[文章大纲](KVChime-paper-outline/README.md) · [Workload、指标与复现说明](../docs/KVChime-multi-model.md) · [代数正确性](../docs/KVChime-correctness.md)。

最终图只保留 performance 和 capacity；旧单模型图表归档在 `artifact/legacy-paper`。能耗原值仍在 CSV，简述见 Experiment 3 README；没有面积或能耗图。

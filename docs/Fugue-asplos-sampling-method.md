# 完整 workload 的代表请求仿真与外推

输入保存完整的大 workload；仿真器只为选中的代表请求建立事件图、调用 Ramulator，再用每个代表覆盖的请求数对完整 workload 估时。

这是 chenyi9 于 2026-09-06 指定的仿真方法。完整请求数来自输入中的 `agents`，不能由一个小 workload 加重复系数冒充完整输入。抽样属于仿真方法，workload 的文档、轮次、输出长度和读写关系属于 workload 构造。

在完整 JSON 的 `meta.simulation_sample.representative_for` 中列出每个请求对应的代表请求。例如完整输入确实包含一个 owner 和 64 个消费者时，owner 映射到自身，64 个消费者映射到其中一个消费者。输入仍有 65 个请求，事件图只建立 2 个请求；owner 的权重为 1，消费者的权重为 64。完整请求 ID、依赖和抽样映射都保留。这项抽样与回放方法同时用于全部七个 combo。

```json
"simulation_sample": {
  "selection": "按预先指定的相同形状分组，取首个请求作为代表",
  "representative_for": {
    "owner": "owner",
    "r00": "r00",
    "r01": "r00"
  }
}
```

上例仅展示映射格式；实际映射必须覆盖 `agents` 中的每一个请求。解析器核对层级、父依赖、输出长度、历史长度和每段的角色、长度、位置偏移，并检查共享指纹映射一致性。代表集合必须包含它自己的父依赖。没有抽样配置时，沿用完整仿真。

运行仍使用 `main.py --workload ... --engine dag --pipeopt --ablation ...`。选择发生在复用计划、物理分配和事件构建之前。`--validate-workload` 会给出完整请求数、代表数及权重，不调用硬件模拟器。

计价与重放规则：

- GPU 保存真实算子形状。FC、投影、归一化、激活及集合通信按扩大后的矩阵 M 重新计价；attention 保留每份请求自己的上下文，扩大独立 attention 矩阵数。GPU 合批事件按成员权重之和得到完整批量。
- PIM 每个实际 channel 的服务时间乘对应请求的权重。多个 channel 仍并行；同一 channel 仍竞争。共享 MQ sweep 的成员权重不同时，按最大权重补齐重复 sweep，并如实保留这项保守近似。
- NVLink 3 传输按权重放大原有时间和字节数，保留原模型已含的启动项。当前 token 自身 QK/PV 仍为零耗时，历史 attention 正常计时。
- 保留代表图的依赖、放置、修正计划、MQ 分组及 prefill 选边，用放大后的成本重新调度。原始事件和外推事件分开保存；先验证权重全部为 1 时逐事件重现原始起止时间。
- TTFT 按完整请求权重平均；TBT 按完整输出间隔数加权；E2E 是重放后依赖图的 makespan。不会把完整 E2E 简单乘请求数。

报告中的 `sampling` 保存完整映射；`workload`、顶层 `summary/events/makespan_s` 是实际仿真的代表集合；`full_workload` 描述完整输入；`extrapolation` 保存完整规模的估算、重放事件和假设。代表图的实际请求 batch 与外推后的逻辑 batch 同时保留。原有只外推稳态 decode 的方法也保留在 `src/decode_batch_estimate.py`，兼容入口仍为 `docs/experiments/estimate_decode_batch.py`，支持原 batch 8/64 用法和不同模型的张量并行配置。

外推假设被省略请求与代表在形状和阶段上相近。它不重新计算完整请求集合造成的布局变化、跨组 MQ 机会、A6 选边变化或队列到达变化，也不验证完整 KV 容量和答案质量。因此完整规模的结果必须标为外推，不能称作完整大图仿真或硬件实测。

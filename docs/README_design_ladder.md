# Fugue 设计阶梯与实现入口

A1/A2 是独立 baseline，不要求彼此只差一步；A3b 起按已声明机制比较。条件例子见 [贡献 README](README_contributions.md)，收益与限制见 [性能分析](analysis/README.md)，配置/命令见 [实验指导](experiments/README.md)。

| 档位 | 定位 | Prefill attention | 声明变化 |
|---|---|---|---|
| A1 | 无复用硬件基线 | GPU | 独立基线 |
| A2 | 软件复用、KV 回 GPU 计算 | GPU | 独立基线 |
| A3b | 朴素软件复用与 PIM | GPU | 持久写入序放置；同轮 diff 可合并 |
| A4c | diff 紧凑区域 | GPU | master 同 A3b；diff 按写入序紧凑追加进本 head 的 diff 行，diff 行像 master 块一样在该 head 的通道上轮转（第 j 行在通道 j mod 条带宽） |
| A4e | 软件放置表 | GPU | 表管 master 也管 diff：master 块分散共读；diff 按 agent 分组紧凑（一个 agent 各轮的修正共用它自己的 diff 行，别的 agent 不穿插），每个新 diff 行放到该 agent 所读行最少的通道 |
| A5 | PIM prefill 与 MQ | PIM | 继承布局，采用已接受的 MQ/PE/buffer 配置；线性层仍 GPU |
| A6 | 逐请求选边 | GPU/PIM | 两侧服务价格比较，首层决定后复用；平局 PIM |

源码：[preset](../src/ablation.py)、[复用计划](../src/workload.py)、[ledger/执行](../src/workload_runner.py)、[Ramulator](../src/ramulator_wrapper.py)。

256-token chunk 通常对应一行，但地址占行不等于 ACT 次数；后者由实际访问序列和行缓冲决定。并行扫描等待实际耗时最大的通道。

chenyi9 2026-09-06 裁决的 diff 布局：A4c 的 diff 行在 head 的通道上轮转（不再集中到末通道），A4e 由表决定哪些 diff 放在一起（按 agent）和放在哪个通道；
此前"全局 diff cursor、末通道"的实现见 [布局专项](../audit/2026-09-05/LAYOUT_BENEFIT_CEILING.md) 与 session §26。原长文含旧消融/旧比例，完整保存在 [分类前原文](archive/2026-09-05-before-categorization/README_design_ladder.md.txt)。

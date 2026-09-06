# Fugue 文档入口

用词（chenyi9 2026-09-05）：**model** 是被服务的 LLM（LLAMA3-8B 等，TINY 只证明流程能跑通），**workload** 是拓扑 JSON
（`workload/probe/sweep/*.json`），**combo** 是阶梯上的一档（A1 A2 A3b A4c A4e A5 A6）。固定开关：flash、pipeopt、k=8、batch 8，
每 GPU 5 个 HBM3-PIM 栈。主指标：E2E、TTFT、TBT、decode scan latency，加能量。

| 你要做什么 | 主文档 | 内容 |
|---|---|---|
| 跑实验：选 model 几何、baseline 与 sweep、命令、出数 | [运行协议](README_run_protocol.md) | 用词、每个 model 的 GPU/HBM、W1 baseline、六条 sweep 轴、squire 命令、`extract_protocol.py` |
| 指标怎么定义、要 full events 怎么跑 | [实验指导（指标与事件）](experiments/README.md) | 四项指标的定义与取数、`--workload-report-events full`、A6 side log |
| 理解为什么能节省时间 | [性能分析](analysis/README.md) | 布局、MQ、低 AI 与选边的收益、上限和限制 |
| 检查实现与论文、公平性 | [审计入口](audit/README.md) | 最慢 channel、既有问题与用户裁决、原始证据 |
| 理解各档声明 | [设计阶梯](README_design_ladder.md)、[贡献例子](README_contributions.md) | 机制映射；条件例子不是实测 |
| 查机器环境 | [squire](run/README_run_squire.md)、[athena](run/README_run_athena.md) | scratch、编译器、资源、作业提交 |
| 查修改原因 | [session 索引](sessions/README.md) | 用户决定、实际变更、验证与当时版本 |

运行协议是当前跑法的主入口；实验指导解释指标；性能分析解释结果；audit 判断实现与比较是否符合声明。
旧文档原文保存在 [分类前快照](archive/2026-09-05-before-categorization/README.md)，历史数字保持其原配置含义。

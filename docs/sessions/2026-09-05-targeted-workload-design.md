# 2026-09-05：持续汇总 workload、scan→TBT 手算与实验指南

## 用户要求和澄清

chenyi9 要求检查最新 commit，解释为什么 scan 每步有改善但 TBT 改善小，并构造能体现 A4c/A4e 布局、A5/A6 prefill 机制的 workload；输入必须先手算再写 README。用户进一步澄清：每一步都有同一个持续汇总 agent，它逐轮保留旧 diff，并不是最后新建一个独立 agent。

本轮按这一语义实现多轮 parent 链。首轮复用先前 owner 的文档，后续完整保留前缀及输出；每轮引入不同新文档及本轮 diff，旧 diff 指向首次 writer。这样既表达跨轮分散，也保留 A3b 同轮合法合并。

## 检查来源和主要判断

起始/审查 HEAD 为 `958dd24e6ae5e435d5b05a7d376877f161300f79`。该 commit 记录 S11 长上下文 TINY 结果；`ff6f225` 已加入回填与 head 流水。工作区未提交的 reader-load 表单独计算，未将其效果归到 HEAD。

旧结果从 scratch 的原始 protocol.csv 自动读取。S11 加长的是 A3b/A4c/A4e 间没有变化的共同简报，private、整步 scan 和 TBT 的降幅不同；不能把 private 降幅当成完整关键路径降幅。新 workload 因此分别增加真正可优化的跨轮 diff 与共读热点文档，而不只增加不变的扫描。

独立 agent 确认：新的文档要来自早先 owner 且产生位置变化才有 diff；同轮 burst 可以合并；跨轮继承要求旧前缀不变；单个汇总者不会凭配置 BATCH 变成很多就绪请求。HEAD partners 表还受整库共读集合影响，所以分组 owner 与 monolithic owner 分别保留。

## 文件及修改原因

- [实验 README](../../workload/probe/targeted/README.md)：解释持续汇总、周期共读、低query与混合选边，给出同配置相邻档的命令和统计分组。
- [手算结果](../../workload/probe/targeted/HANDCALC.md)：所有数据由脚本导出；分别标明旧运行实测、新输入几何、GPU解析价格和等待Ramulator验证的预算。
- `workload/probe/targeted/build.py` 与 `inputs/`：生成D、E、P三类输入及反例；未更改现有C1/C2生成器。
- `handcheck.py`、`summarize_checks.py` 与 `checks/`：真实reuse plan和物理账本；闭式公式与实际QK地址生成对照；源码/输入hash及逐请求extents；不启动Ramulator或DAG执行。
- `cohort_report.py`：从完成的compact报告导出完整/全部汇总/后四分之一/末轮TBT和TTFT，并保留完整E2E，避免短worker与准备期把累积轮次机制混在一起。不同cohort的TBT不与总体scan硬配对。
- [独立报告](../../workload/probe/targeted/independent_review.md)：独立核对继承、占行、取整、版本、反例和预算口径。
- docs实验、分析、运行协议和session索引：添加明确入口，并将旧预约空窗问题标成历史时点。

## 为什么保留反例、为什么分开几个输入

D的固定写流周期刻意暴露跨轮diff热通道，是受控机制实验，不能代表一般到达分布。增加一段后台内容会改变周期；增加到八个并行汇总链还会让普通master集中，可能遮住diff省时。七条链的普通master分布更均衡，但七条/八条都保留，且都没有按实测性能筛选。

E的周期共读针对大部分实际读取量；连续选择是平衡对照，整库owner是旧表失效对照。改变导入分组也改变prefill请求数，所以不能跨这两种输入计算“布局E2E收益”，必须在同一输入内比较档位。

P保留首次建立上下文与其较大的query数，后续才是q4。只保留该例实际需要的共享语料，并增加持续复用轮数，使低query阶段有机会摊薄准备成本；混合版本另加入新鲜长提示。局部PIM价格低不保证整个包含准备期的E2E低，README要求同时报告。

## 审查修正和验证边界

独立审查发现初稿解析预算直接创建context-return Layer，会多计小传输固定latency。已统一到HEAD真实 `_link_layer` 并显式采用 `min(batch_limit, hardware_capacity/GQA)`；预算重新计算。该修正只涉及新增手算工具，没有修改执行器或用户既定链路计价。

新输入的占行检查是layer0、最忙HBM的一条八通道head stripe、该轮decode开始前的读集。物理K占行与QK命令触及行分开，均不是实测ACT或完整scan时延。实际GPU预算也不是已验证关键路径，正式TBT/TTFT/E2E需要真实运行。

本轮完成输入生成、公式/真实地址一致性核验、现有报告的只读提取和独立审查；运行命令已写入README。没有启动新的性能模拟，没有改模拟器实现，没有commit或push。工作区中其他会话进行的runner、gen_sweep和C3修改保留原样。

验证命令：`build.py`、`handcheck.py`、`summarize_checks.py`、`validate_artifacts.py`，均位于上述targeted目录。`checks/formula_checks.json`保存闭式与真实QK地址核验，`checks/validation.json`保存输入hash、脚本/命令语法、旧S11加权TBT回算结果；旧报告完整配置和文件hash在`checks/existing_S11_provenance.json`。初稿静态中间目录保留在`/tmp/targeted_design_initial`，正式入口只指当前修正后的`checks/`。

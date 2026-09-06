# 2026-09-05：Fig. 3b 改为跨轮 diff 的物理行扫描

chenyi9 明确确认半列/半行错位 case 尚未实现，先要求计划，再授权修改论文、图内内容并保留原始数据和复现指南。当前任务完成的是这个已批准的小扫描对照。

起始论文 revision：44b777f53b9b2c7ce5ef2b4793f0dc41bf3afea9；模拟器 revision：4b74849f1910c2e8ac6077da1d7e8479ce993a9f。

## 为什么修改

旧 Fig. 3b 测同一小段修正因未对齐而跨列/跨行，主模拟器没有完整表达这一机制。新的对照采用当前真实 allocator 中跨轮 diff 独立行与紧凑 diff 区的差异，读集和 MAC 命令数量相同，由 Ramulator 产生 ACT/PRE。master 保持完整物理流读、替换位置 masked，避免以 master 尾部扫描触发共同粒度近似。

fixture 明确轮间写入来自其他请求，因而不在所选共享块扫描中；没有合并 A4c 的不同 diff extent 来减少 MAC，也没有给 A3b 增加手工跳转延迟。同轮连续 diff 可合并的规则保留。单 channel 控制用于隔离行局部性，结果是 K 扫描，不是完整 attention 或 E2E。

## 原始结果

| 布局 | K-scan cycles | 相对紧凑 | MAC_AB | ACTAB | PREA |
|---|---:|---:|---:|---:|---:|
| packed | 990 | 1.0000× | 136 | 5 | 4 |
| scattered | 1146 | 1.1576× | 136 | 8 | 7 |

由原始模拟结果自动生成；这是单 channel K 扫描，不是完整 attention 或 E2E。


正式 Fig. 3b 显示 packed/scattered 两柱、136 条 MAC_AB 的共同工作量和对应 ACTAB，正文引用自动生成的 scan_numbers.tex。模拟由 Codex 按 chenyi9 本轮授权执行并在新目录复现，未标为用户手工重跑。

## 文件变更与原因

- sections/03-motivation.tex：Problem 2、图注说明和 PDF Description 改为跨轮 diff 行分散；旧未对齐倍率由新结果替换，保留已有 citation key。
- fig/plots/motiv/plot_motiv.py、fig/ver1/fig1_motiv.pdf：右面板两柱和实际 ACTAB，标注 K-scan；数值从 CSV 读取。
- fig/plots/motiv/experiments/02_irregular_access：fixture、真实 allocator/trace 导出、Ramulator 调用、校验、数据导出和面板绘图；完整 raw 数据及来源。
- fig/plots/motiv/README.md、experiments/README.md、fig/plots/README.md：当前数据入口与复现步骤，旧案例归档。
- fig/plots/motiv/scan_numbers.tex、scan_summary.json、data_allbank.csv：从同一结果导出，禁止手工抄数。
- 旧脚本/数据/图原文归档到实验目录的 archive/2026-09-05-column-boundaries。

## 验证

原始 trace 与实际执行 MAC 地址 multiset 一致；同 master 地址、同读集、同 MAC 数。重新生成并在 /tmp/fugue-scan-reproduction 运行，trace 与结果逐字节/逐字段一致。独立 agent 同时核对日志、配置、数据宏和实际加载动态库。

使用论文 Makefile 和指定 TinyTeX，在 /tmp 本地构建；图及论文页面渲染检查。构建日志和最后校验记录随本次归档保存。模拟器实现未修改；论文目录写入通过文件系统授权完成，不提交或 push。

## 并行提交与版本复核

实验运行使用 4b74849f1910c2e8ac6077da1d7e8479ce993a9f。整理期间仓库推进至 6b1c286fe31152f14d0c2cfc25bc0661f151e4b2，变化位于链路定价与 prefill Q 传输，属于其他工作产生的提交。本次没有修改这些实现。用最新源码重新生成账本、extents 和全部扫描 trace，与保存的原始输入一致；原始模拟数据继续标注其实际运行 revision，未改写来源。详细摘要见实验 validation/checks.json。

论文编译通过，Fig. 3 与正文所在页已检查；保留了全文已有的 fig7reusetime.pdf page-group 警告，没有新增未定义引用或 Overfull。

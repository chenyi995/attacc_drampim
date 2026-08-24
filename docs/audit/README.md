# docs/audit/ — 整个项目的分块审计(2026-08-22 起)

目标读者:计算机专业学生或接手人——会编程、懂矩阵乘法,但不了解本项目、
LLM serving、DRAM/PIM。读完本目录应能回答:这个仓库由哪几层代码叠成、
每层是谁在什么时候加的、每层的机制结合代码在哪一行、以及当前有哪些
已知悬置问题。全部文档遵守 `docs/OUTPUT_SPEC.md`(输出格式规范)。

## 这个项目是什么(三十秒)

本仓库(kvpim-sim,目录名 `attacc_drampim_xinyao`)是 Fugue 论文
(GPU–PIM 共享 KV serving)的**性能/能耗模拟器**:在 AttAcc 官方模拟器
(ASPLOS'24,GPU+HBM-PIM 异构系统)之上,叠加了 KV 复用
(CacheBlend/EPIC)、物理地址级事件模拟、放置消融(A 系列)、
MQ 批命令微架构(C 系列)、多轮 agent 驻留 KV 等层。
配套仓库:RTL(`fugue-logic-die-rtl`,Genus 面积/时序)与论文
(`KVPIM-1Fugue-ASPLOS2027`)。

## 审计性质声明

本目录不只是导览:它按块 00 的**公平性准则(P1–P6)与完整性准则
(I1–I7)**执行审查——块 07 逐实验裁定对比是否同秤(通过/须声明/存疑
三档),块 08 记录实际执行过的防伪核查(引用抽查、改动面对账、数字
双通道交叉、复算、行为不回退证据),并提供**不依赖助手的核查命令集**
(块 08 §6)供宸逸抽查。审计由助手撰写,自我审计的局限与缓解见
块 00 §5。

## 分块地图(按代码叠层顺序 = 阅读顺序)

| 块 | 文件 | 内容 | 归属 / commit |
|---|---|---|---|
| 00 | `00_methodology.md` | **审计方法学**:公平性准则 P1–P6、完整性准则 I1–I7、引用规范、红线清单 | 审计基建,2026-08-22 |
| 01 | `01_upstream_attacc.md` | 上游 AttAcc 模拟器基线:Layer/Transformer 代价模型、xPU/PIM 设备、Ramulator2 封装、HBM3-PIM 命令集与 trace 生成 | AttAcc 官方(scale-snu),`c1540de` |
| 02 | `02_xinyao_reuse_stack.md` | KV 复用执行栈:workload JSON、ReusePlan(CacheBlend/EPIC)、物理 TLB + 事件 DAG(`workload_runner.py`) | xinyao,`0aced82`(2026-08-17) |
| 03 | `03_xinyao_a_series.md` | A 系列放置消融(A1–A6)、refined/flash GPU 模型、GPU+PIM vs GPU prefill 拐点研究 | xinyao,`47ae0c3`→`34d3cd7`(2026-08-21) |
| 04 | `04_mq_c_series.md` | C 系列:MQ-MAC 批命令(trace `--mq` + 时序模型 + DAG 集成)、非对称相位、D_i 位图、bank-whole prefill、总线转向 C++ | 宸逸+助手,`9d6fc7b`/`264d14a`/`3e338e6`/`711ae25`(2026-08-21) |
| 05 | `05_agentic_history.md` | agentic 多轮驻留 KV(`history_len`):解析式半 + 事件 DAG 半 | 宸逸+助手,未提交(2026-08-22) |
| 06 | `06_area_balance_0822.md` | 专题:MQ bank-PE 面积平衡点(容量×速率两轴、in-bank 预算、换算链)全链条审计 | 宸逸+助手,2026-08-22 会话 |
| 07 | `07_fairness_review.md` | **逐实验公平性审查**(A/C/面积/history 逐项过 P 准则;含"进正文前必须写明的声明清单") | 审计执行,2026-08-22 |
| 08 | `08_integrity_checks.md` | **完整性核查执行记录**(行号漂移的发现与修正、改动面对账、数字双通道交叉、复算、核查命令集) | 审计执行,2026-08-22 |

配套正文(非审计,面向设计理解):`docs/README_mq_design_space.md`
(两轴设计空间零基础版)、`docs/README_fugue_dataflow.md`(全数据流)、
`docs/README_design_check.md`(时序算术)。

## 与论文的对应(总映射,细节见各块"在论文中的意义"节)

以 `docs/EXPERIMENTS.md` 尾节为准:论文五级阶梯(GPU-only/PIM-append/
PIM-split/PIM-static/Fugue)≈ A2/A3/A4/A5/动态选边(未实装),AttAcc
参照=A1;C 系列支撑微架构与 die 面积章节(E4 方向);C-impl 对应正文
§4.3.2(D_i 写口过滤)与 §4.5.2(bank-whole 因果丢弃);块 05(history)
暂无论文实验编号。数字进论文正文前须宸逸手动复现(论文仓库 CLAUDE.md
的数据纪律)。

## 归属考证方法

以 `git log --stat` 与 `docs/LOG.md` 交叉:`c1540de..c600051` 为上游;
`0aced82`(分支 xy_0814)与 `47ae0c3..34d3cd7`(分支 xinyao_0821)为
xinyao;`9d6fc7b` 起(分支 chenyi-experiment-821)为宸逸+助手会话;
未提交工作区(2026-08-22)见块 05。每块文件头重复标注归属。

## 测试总表(38 个,`tests/test_workload.py`)

| 层 | 测试(节选) | 覆盖 |
|---|---|---|
| 上游回归 | `test_attacc_pipeopt_reference_matches_decoder_events`、`test_no_reuse_matches_real_attacc_for_a_small_request` 等 | JSON 入口 = 原版 AttAcc 逐位 |
| 复用栈 | `test_cacheblend_emits_trace_ordered_tlb_and_physical_addresses`、`test_cacheblend_decode_streams_master_and_diff_pools_sequentially` 等 | TLB 物理地址、master/diff 分池、事件序 |
| A 系列 | `test_ablation_a1_reproduces_the_original_attacc_legacy_report`、`test_split_prefill_overlaps_its_gpu_and_pim_branches` | A1=原版、A6 重叠语义 |
| C 系列 | `MQBatchCommandTests` 5 例 | MQ 间隔/容量、trace 列读一次、相位切分、sweep 拆分、bank-whole+D_i |
| history | `AgenticHistoryTests` 6 例 | 解析、扫描加宽、全重算层、物理 no-reuse、legacy 拒绝、ablation 入账 |

运行:`PYTHONPATH=$PWD python3 -m unittest tests.test_workload`(须 38/38)。

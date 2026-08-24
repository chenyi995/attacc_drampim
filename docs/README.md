# docs/ — 统筹索引(chenyi-experiment-821,严格论文模式)

本分支是 **Fugue 论文实验的主战场**。本目录统筹全部文档:本页(索引与各部分介绍)、
`EXPERIMENTS.md`(实验总纲:**有且仅有 A 系列与 C 系列**)、`HANDOFF.md`(交接)、
`LOG.md`(逐日日志)。

## 论文在哪里

- **论文仓库**:`/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027`(Fugue: A GPU–PIM
  Memory System for Efficient Shared-KV Serving;主文件 `main.tex`,章节
  `sections/01–09`,用户裁决与写作规则在其 `outline/` 与 `CLAUDE.md`——在那边
  工作以那边的规则为准,数字进正文前须用户手动复现)。
- **RTL 仓库**:`/data2/chenyi9/KV-PIM/fugue-logic-die-rtl`(logic die + in-bank PE
  的 N28/Genus 综合,C 系列 RTL sweep 所在)。

## 分支地图(本仓库 attacc_drampim_xinyao)

| 分支 | 内容 | 用途 |
|---|---|---|
| **chenyi-experiment-821(本分支)** | 全功能:MQ-MAC 全集成(CLI 默认 mq)、非对称相位、总线转向惩罚、D_i 位图、bank-whole prefill、A/C 全部实验与文档、32 个单测 | **论文实验在这里跑** |
| `xinyao_0821` | 上游 + 干净的 C 增量层(默认行为与上游逐位一致,27+4 单测) | 对照/合流基线 |
| `main` / `xinyao_07xx` / `xy_0814` / `v10-layout-schedule` | 上游历史 | 存档 |

## 仓库各部分介绍

| 路径 | 是什么 |
|---|---|
| `main.py` | 入口。关键旗标:`--ablation A1..A6`(A 系列)、`--reuse {no-reuse,cacheblend,epic}`、`--pim-batch-command {mq,replicate}`(默认 mq)、`--pim-prefill-mode {split,bank-whole}`、`--pe-freq-ghz`、`--gemv-buffer-bytes`、`--cacheblend-batch-size`、`--pim-link`、`--gpu-model` |
| `src/ablation.py` | **A 系列**的 legacy 代价模型:A1 原版 AttAcc(私有 KV)/A2 纯 GPU/A3 朴素映射/A4 master-diff/A5 全 PIM prefill/A6 split |
| `src/workload_runner.py` | 物理事件 DAG(逐地址、逐事件、Ramulator 计时):TLB、master/diff 分池、decode 批扫描(按 GEMV 容量拆 sweep)、D_i 位图事件、bank-whole prefill、逐 sweep 到达审计 |
| `src/ramulator_wrapper.py` | Ramulator2 封装 + **MQ 时序模型**(`mq_interval_cycles`:功耗拉伸/PE 吞吐/通路下限取大;YAML `nCCDAB` 覆盖;签名缓存) |
| `ramulator2/` | 补丁版 Ramulator2(HBM3-PIM 命令集 + 搬运总线方向转向约束);二进制需 gcc-toolset-11 编(见 HANDOFF) |
| `experiments/GPU_PIM_vs_GPU_prefill/` | **A 系列**研究:A4 vs A6(vs A5)的协同 prefill 拐点(EPIC p*、CacheBlend r 上限) |
| `experiments/mq_command/` | **C 系列**全部(C1/C2/C3 + 消融 + 实装):见其 README 与 `DATAFLOW.md` |
| `experiments/_archive/` | 非论文历史实验(cacheblend_tier_batch、end_to_end_20260814),仅存档 |
| `tests/test_workload.py` | 32 个单测(27 上游回归 + 5 个 MQ/C 层) |
| `docs/` | 本目录:索引 / 实验总纲 / 交接 / 日志 + 四份设计文档(下行) |
| `docs/OUTPUT_SPEC.md` | **给宸逸的输出格式规范**(读者假设/首现即释/多维度/数字纪律/实验须注论文落点/自检清单)——写任何文档前先读 |
| `docs/audit/` | **整个项目的分块审计**(01 上游 AttAcc / 02 复用栈 / 03 A 系列 / 04 C 系列 / 05 history / 06 面积平衡点),含归属考证与逐块 file:line 代码地图,入口 `docs/audit/README.md` |
| `docs/README_mq_design_space.md` | **MQ 两轴设计空间零基础版**:容量轴×速率轴、匹配频率 f\*、in-bank 面积预算线 |
| `docs/README_fugue_dataflow.md` | **零基础全数据流**:每步 GEMM 的 M/K/N、哪级切哪个维、五级累加、两处合并、三种"轮转" |
| `docs/README_design_check.md` | 时序算术(nCCDAB/nRC/免费槽/拐点)+ 论文 vs 仿真器 11 条差距(自足版) |
| `docs/SIM_VS_PAPER_AUDIT_0821.md` | 同上差距的 `file:line` 索引版 |
| `docs/PLAN_mq_command.md` | MQ 实现计划(O1–O7,含设计理由出处) |
| `experiments/mq_command/DATAFLOW.md` | MQ 硬件增量 + 17 步数据流 + 三路审计 + 三裁决项处置(①②已实装,③已量化关闭) |
| `CLAUDE.md`(仓库根) | 助手工作守则(范围纪律 + 三条输出规范),须留根目录以自动加载 |

## 实验总纲

见 `EXPERIMENTS.md`:**A 系列 = 放置消融(A1–A6)**,**C 系列 = 微架构选择与消融
(C1/C2/C3 + C-abl-1/2/3 + C-impl)**——论文实验有且仅有这两个系列。

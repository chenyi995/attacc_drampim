# kvpim-sim(xinyao_0902):GPU–PIM 共享 KV 服务仿真器 —— 项目总览

目标读者:有计算机背景、但不了解本项目/LLM serving/DRAM-PIM 细节的人。
概念首次出现即解释,关键术语标注英文;各专题另有独立 README(见文末索引)。

## 1. 这个项目是什么

大语言模型 (LLM) 生成回答分两个阶段:**prefill**(预填充:把整段输入
一次性算完,生成 KV 缓存)与 **decode**(逐词生成:每个新词都要把历史
KV 缓存从头读一遍做注意力 (attention))。KV 缓存 (KV cache) 指每个历史
token 存下的一条 K 向量和一条 V 向量。decode 阶段的瓶颈是**内存带宽**:
把 KV 从显存搬到计算单元的代价远超过计算本身。

AttAcc(ASPLOS'24)把 decode 注意力搬进 HBM 内存的每个 bank(DRAM 的
独立读写单元)旁边的小计算单元(bank PE, processing element)执行——
这类结构叫存内计算 (PIM, processing-in-memory)。本仓库在 AttAcc 的开源
仿真器上扩展,研究 **Fugue**:多智能体 (multi-agent)、多轮 (multi-round)
场景下,多个请求**共享**同一份 KV(相同的系统提示、共享的文档 chunk、
上一轮的输出),GPU 与 PIM 怎么分工、KV 怎么摆、一次 DRAM 列读怎么服务
多条查询。

## 2. 两条仿真路径(同一份编排,两种精度)

一个 workload(编排文件,JSON:多个请求、每个请求由若干段 (segment)
组成、段带指纹 (fingerprint) 表达跨请求共享、`history_len` 表达多轮
历史)可以从两条路径跑:

| 路径 | 入口 | 精度/用途 |
|---|---|---|
| **解析引擎** (analytic) | `main.py --ablation A1..A6`(`--engine analytic`,默认)→ `src/ablation.py` | 算子级封闭式代价模型 + Ramulator 形状缓存;快,用于**快速预估与两引擎交叉校验** |
| **物理事件引擎** (event DAG) | `main.py --ablation A1..A6 --engine dag` → `src/workload_runner.py` | 每请求每层展开成带真实时序、依赖与资源排队的事件图;**2026-08-26 起覆盖全部七档(含 A3a)**,重叠/排队由排程涌现而非公式假设 |

裁决(chenyi9 2026-08-26):**真实 workload 一律默认走物理事件引擎出数,
解析引擎不单独出数**(只作预估与校验)。放置语义两引擎一致:prefill
注意力 {gpu, pim, dynamic} 三选一;decode 注意力恒在 PIM,唯 A2 例外
——A2 的 decode 在 GPU,KV 放**远端哑存储**(经 NVLink/PCIe 的
GPU↔远端存储链路计入 `link_bytes`,R10 裁决;解析引擎的 A2 暂为
"KV 在 GPU 本地"旧口径,见总台账 U3)。

## 3. A1–A6 阶梯(放置消融,2026-08-24 定;布局档 2026-08-29 重切)

每一档只比上一档多一件事。**布局的物理模型(chenyi9 2026-08-29)**:一个
head 的一个 chunk(256 token)是**一个 channel 上的一个 row**;一次 decode
扫描的时间是**最忙的那条 channel**(一起读、又落在同一条 channel 上的 chunk
会串行)。head 之间是 HBM 间并行;`heads_per_hbm = ceil(局部 KV head / num_hbm)`
个 head 共享一个 HBM 的 16 条 channel。各档细节见 `intro/`:

| 档 | 一个 head 的 chunk 怎么落 channel | prefill attn | KV 布局(`kv_mapping`/`channel_placement`) | 批命令 |
|---|---|---|---|---|
| A1 | AttAcc 原样,无复用(参照点) | GPU | private | replicate |
| A2 | 纯软件复用,无 PIM 算力 | GPU(decode 也在 GPU) | none(KV 在**远端哑存储**,R10) | replicate |
| A3 | **不切片**:head→**1 条** channel,该 head 全部 chunk 压这一条(余闲) | GPU | naive / **single** | replicate |
| **A3a** | A3 布局但**可掩**(陈旧行随流读出被掩,run 不断) | GPU | naive-mask / single | replicate |
| **A3b** | **+ head 切片**:该 head 的 chunk 在自己 `16/heads_per_hbm` 条 channel 上轮转 | GPU | naive / **slice** | replicate |
| A4 | **+ master/diff 分离**:master 池 ch0–14 head-切片,修正行进 diff 池 ch15 | GPU | master-diff / **slice** | replicate |
| **A4b** | **+ placement table**:丢掉固定切片,全局把 co-read 的 chunk 摊到 15 条 master channel | GPU | master-diff / **table** | replicate |
| A5 | **A4b 布局** + 所有 prefill 注意力进 PIM(MQ n_cap=8:512 B/1.3004 GHz) | PIM | master-diff / table | **mq** |
| A6 | **Fugue(我们的方法)**:A5 + 逐请求动态选边 | dynamic | master-diff / table | **mq** |

阶梯 diff:A3b−A3 = head 切片;A4−A3b = master/diff 分离;A4b−A4 = 全局
co-read table 取代固定切片;A5−A4b = prefill 上 PIM + MQ;A6−A5 = 动态。
物理事件引擎(`workload_runner._append_placement_pim_scan`)是**唯一出数路径**;
解析引擎只作粗校验、不区分 single/slice/table。

> **2026-09-03 更新,本表的 `kv_mapping`/`channel_placement` 两列不变,但它们映射到的
> 布局模型换了。** A1 与 A3/A3a 仍走 `_layout_channel_loads` 的 chunk 计数
> (每段补齐到 256-token chunk);**A3b 及之后改走 striped-append 的真实行数**
> (policy 名相应变成 `slice-append` / `master-diff-slice-append` /
> `master-diff-table-append`),而且一条 channel 的**真实 extent 列表**作为一次
> Ramulator 仿真提交、**ACT 由行缓冲决定**。逐 token 的落点、逐档的 ACT 次数,
> 见 `README_data_layout_walkthrough.md`;手算与实测的逐格对照见
> `../workload/handcheck/README.md`;来龙去脉见 `sessions/2026-09-03.md` §3、§11。
>
> 同时,`--num-hbm` 的含义已修正为**整个系统**的堆栈数(`d3a3c4c`):堆栈和 KV head
> 都按 `--ngpu` 切,`heads_per_hbm = ceil(KV head 总数 / num_hbm)`。此前本段写的
> "真实配置 `num_hbm=16`、LLAMA3-8B → `heads_per_hbm=1`、A4≈A4b" **已作废** ——
> sweep 实际用的是 `--num-hbm 1/1/10/10/40/40`,LLAMA3-8B 的 heads_per_hbm 是 8。

关键约定:**attention batching(MQ 批命令)与"prefill 上 PIM"同步启用**
(A5 起);A1–A6 默认都在多轮 agentic 编排下运行(`history_len`);曾经的
"split 混合"档已废除。

**硬件轴(2026-08-27)**:
- **head→HBM 条带映射**(R18,恒开):一个 KV 头独占 HBM,run 覆盖的各
  channel 承载该头自己的 token 条带(废除上游"一 head 一 channel";
  全文 `README_head_hbm_remap.md`);
- **`--num-hbm`**(默认 5):PIM 侧 HBM 堆叠数(= 远端 KV 存储;GPU 的
  HBM 与 NVLink 不变)。KV 头数 < 堆叠数时每头独占
  `num_hbm // kv_heads` 个堆叠、K/V **头内序列切分**并发扫;
- **GQA 模型档**:`LLAMA3-8B`(32 Q 头 / 8 KV 头,组 4;与 LLAMA-7B
  同形对照)——组内 Q 头是共享 KV 头上的驻留查询,进 MQ/容量轴;
- **功耗约束 (PC) 默认开**(R19):MQ 间隔带每窗口能量钳位
  (n=8 → 8 tCK),`--no-powerlimit` 显式关。

## 4. 软件上游(复用策略,`--reuse`)

复用策略决定"哪些 KV 可以复用、复用时要重算多少来修正精度"。现有六个,
分两族(详见 `archived/README_software_upstream.md`,2026-09-03 归档):

- **cacheblend 族**(按比例采样重算行):`cacheblend`(EuroSys'25,在线
  选择,含全重算选择层)、`cachetune`(离线选择,无选择层);
- **epic 族**(逐段修正行):`epic`(每复用段重算**前** k 个 token)、
  `recompute`(2026-08-27 增,**通用计数策略**:每位移段内**随机**抽 k
  个 token 重算,阶梯实验默认用它)、
  `promptcache`(零重算基线,MLSys'24)、`cachecraft`(前缀长度按上下文
  重叠度逐 chunk 变化,SIGMOD'25 风格)。

## 5. C 系列(微架构)与 RTL

一次 DRAM 列读服务 n 条查询的 MQ-MAC 批命令、GEMV buffer 容量轴 × PE
频率速率轴、流式 P(概率向量不驻留、由 TSV 移动总线计价)等微架构内容
是 **C 系列**,主要文档在 experiment 分支
(`docs/README_c_series.md`、`docs/README_mq_design_space.md`);RTL 与
综合在 `fugue-logic-die-rtl` 仓库。本分支的 C3 机制实装(MQ 命令、D_i
位图、bank-whole prefill)见各 Ax README 的代码定位表。

## 6. 快速上手

```bash
# 一键七档(推荐,默认口径):指定一个 workload JSON,DAG 引擎跑
# A1/A2/A3/A3a/A4/A5/A6,产出 output/<时间戳>_<负载>_<模型>_k<比例>/
# dag_ladder.csv;A1 用 no-reuse,A2–A6 用 recompute(EPIC_K 定 k)
bash experiments/run_dag_ladder.sh workload/wl_tiny.json LLAMA-7B

# 全套件(每 workload 一条链,k=2 带暖先行、其余 k 免暖去 A1;
# NUM_HBM 指定 PIM 堆叠数;产出另含逐 tier 三指标 dag_ladder_tiers.csv)
NUM_HBM=16 N_PAR=2 RAMU_WORKERS=6 bash experiments/run_dag_suite.sh LLAMA3-8B

# 单档物理事件引擎(真实 workload 的默认出数方式)
python3 main.py --system dgx-attacc --model LLAMA-7B \
  --workload workload/wl_tiny.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --ablation A6 --engine dag --ramulator-workers 14 --cacheblend-batch-size 8 \
  --workload-report-events none --workload-report /tmp/a6_dag.json

# 解析引擎快扫(仅预估/交叉校验,不单独出数;--engine 默认 analytic)
python3 main.py --system dgx-attacc --model LLAMA-7B \
  --workload workload/wl_tiny.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --ablation A6 --workload-report /tmp/a6_analytic.json

# 回归
python3 -m unittest discover -s tests     # 41/41
```

目录约定(2026-08-26):自造 workload 放本仓库 `workload/`;运行输出放
`output/<时间戳>_<负载>_<模型>/`;Ramulator 签名缓存落盘在
`ramulator2/signature_cache_v2_headhbm.jsonl`(R18 轮换版;首跑建缓存
≤64 核,复跑秒级起步;旧 `signature_cache.jsonl` 为重映射前归档)。

真实 workload 的准入标准与现存源见
`/data2/chenyi9/KV-PIM/workload/README.md` 与其 `SOURCES.md`
(旧版五负载文档已归档:`archived/README_workloads.md`)。

## 7. 文档索引(2026-08-26 重组:intro/ 收 A 档,archived/ 收弃用件)

> **2026-09-03 大清理。** 今天之前写的 md **除运行方法外全部归档**到
> `archived/`(每份带归档说明,注明何时归档、为什么、被什么取代)。理由是当天的
> 三处引擎修正 —— heads-per-HBM `d3a3c4c`、striped-append 布局 `84f87f5`、
> 真实 extent 进 Ramulator `897c294` —— 动了放置与计价的主路径,那些页都没有在
> 新引擎上复核过。**下面列的就是 `docs/` 现存的全部活跃文档。**

### 怎么跑(保留)

- `README_run_athena_slurm.md` / `README_run_squire_local.md`:**怎么跑**(两台机器
  各一份;athena 有 Slurm、没有 `/opt/rh`、scratch 在 `/localdata`,squire 没有
  Slurm、scratch 必须 `/data2`、cppcore 要 gcc-toolset-11)(旧的合并版
  与本机无-Slurm 跑法;含 cppcore 的 gcc-toolset-11 构建坑、scratch 必须放
  /data2、退出码 2 的真因、排错表)
- `README_run_sweep_guide.md`:**sweep 运行指南**(另一台 ~300 核机器 clone 后
  照着跑:setup→跑→提取→独立复核→commit;2026-08-29 写,流程仍适用,
  但它引用的设计规范与结果页已归档)
- `RAW_DATA_MANIFEST.md`:**原始数据本地清单**(>50 MB 的事件轨迹 `dag_A*.json`
  不入库、记本机路径;≤50 MB 报告与 .log 已入库)
- `../experiments/run_local_a3b_a6.sh`:**squire**(无 Slurm)重跑 A3b..A6 的入口
- `../output/_orch2/colpack_submit.sh` + `colpack_tasks.sh`:**athena** 上 2026-09-03
  那轮的槽位提交与 78 任务队列(baseline 七档、其余点 A3b+A6)
- `../workload/gen_sweep.py` + `../workload/sweep/` + `../experiments/run_sweep.sh`:
  参数化 sweep 的 generator / workload / 批跑脚本
- `../output/analysis/`:结果分析工具(`extract_sweep_csv.py` 等)

### 当前口径(今天写的)

- `../workload/gen_sweep.py` + `../workload/sweep/` + `../experiments/run_sweep.sh`:
  参数化 sweep 的 generator / workload / 批跑脚本
### 已归档

- `../workload/archived/2026-08-29_pre-sweep/`、`../output/archived/2026-08-29_pre-unify/`:
  归档的旧手调 generator 与旧结果(各带 README)
- **结果页三份已于 2026-09-03 全部归档**(`archived/README_sweep_results.md`、
  `archived/README_baseline_postfix_results.md`、`archived/README_rung_analysis.md`):
  它们的数字跑在当天三处引擎修正之前(heads-per-HBM `d3a3c4c`、striped-append
  `84f87f5`、真实 extent 进 Ramulator `897c294`),**不要引用**。
  逐格哪些还站得住,见论文仓库 `fig/plots/exp1/README.md` 的注意事项第 4 条。
  **当前没有有效的结果页 —— 要有,得用修正后的引擎重跑。**
- **其余今天之前的页也已归档**(2026-09-03):`archived/README_sweep_design.md`
  (sweep 设计规范)、`archived/README_software_upstream.md`(复用策略族)、
  `archived/README_manual_audit_findings.md`(审计总台账)、
  `archived/README_experiments.md`(证据矩阵怎么跑)、
  `archived/README_engine_future_work.md`(引擎重构提案,未实施)、
  `archived/README_simulator_assessment.md`(仿真器评估),以及
  `archived/intro/` 下的**全部九档逐档说明**(A1–A6、A3a、A3b、A4b)。
  每份的归档说明都写明了哪部分仍然成立、哪部分已被取代。
- `../workload/handcheck/`:**手算校验**(一个小到能笔算全部布局的 workload,
  跑完 A3b–A6,把手算的逐 channel 行数/extent/ACT 与实测事件并排;
  公共扫描与 private 扫描逐格一致。含 `compare_theory_vs_measured.py`)
- `README_data_layout_walkthrough.md`:**A3b–A6 的数据布局走查**(GPT-13B /
  baseline / 一个 tier-1 请求 / 一个 head,逐 token 说清 KV 落哪条 channel、
  哪个 stripe unit、行内第几个槽;并推出**一次 ACT 恰好覆盖 256 个 token**,
  以及每档一次 decode 扫描各条 channel 要开几次 ACT;2026-09-03)
- `README_run_athena_slurm.md` / `README_run_squire_local.md`:**怎么跑**(两台机器
  各一份)(旧的合并版
  `output/_orch2/` 编排,与 2026-09-03 新加的本机无-Slurm 跑法;含
  cppcore 的 gcc-toolset-11 构建坑、scratch 必须放 /data2、退出码 2 的
  真因,以及排错表)
- `../experiments/run_local_a3b_a6.sh`:**squire**(无 Slurm)重跑 A3b..A6 的入口
- `../output/_orch2/colpack_submit.sh` + `colpack_tasks.sh`:**athena** 上 2026-09-03
  那轮的槽位提交与 78 任务队列(baseline 七档、其余点 A3b+A6)
- `sessions/`:每日调整记录(chenyi9 裁决时间线)。**最新:`sessions/2026-09-04.md`**
  —— A3b 的重算曾经是免费的(`shadow_reads` 兼管了 DRAM 激活),已修;
  当前 sweep 值 k=8 上 A3b 最忙通道载荷 +12.5%,A3b 那一档的旧数偏快
- `archived/`:已归档,**每份都带归档说明,注明何时归档、为什么、被什么取代**
  —— 旧 workload 文档、走查稿、三份被合并的 audit、旧跑法
  `README_run_experiments.md`、`README_delta_vs_xinyao0821.md`、
  `README_cppcore_branch.md`(分支已并入主线)、
  `README_head_hbm_remap.md`(放置模型已两次改版)、
  `README_sweep_progress.md`(那轮 sweep 已结束);
  **2026-09-03 新归档**:三份结果页(`README_sweep_results.md`、
  `README_baseline_postfix_results.md`、`README_rung_analysis.md`——数字跑在当天
  三处引擎修正之前)与 `archived/intro/` 下的 `README_A3b.md`、`README_A4b.md`
  (所写的 `_layout_channel_loads` 分支已不再是这两档实际走的 policy)

分支说明:当前主线是 **`xinyao_0902`**,它是历史各分支的**代码超集**
(`xinyao_0821`/`0901`、`chenyi-822`/`-dirty`/`-cppcore-exp` 相对它均无
未合入的 `src/` 改动)。分支演进:`chenyi-822-dirty` → `chenyi-822-cppcore-exp`
(核心 C++ 化)→ `xinyao_0902`(PIM 通道并行调度修复)。
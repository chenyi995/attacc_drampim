# kvpim-sim(chenyi-822-dirty):GPU–PIM 共享 KV 服务仿真器 —— 项目总览

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

## 3. A1–A6 阶梯(放置消融,2026-08-24 定)

每一档只比上一档多一件事(2026-08-26 起共**七档**,A3a 见下表);各档细节见 `intro/`:

| 档 | 含义 | prefill attn | KV 布局 | 批命令 |
|---|---|---|---|---|
| A1 | AttAcc 原样,无复用(参照点) | GPU | private | replicate |
| A2 | 纯软件复用,无 PIM 算力 | GPU(decode 也在 GPU) | none(KV 在**远端哑存储**,R10;解析引擎旧口径见 U3) | replicate |
| A3 | 软件复用 + AttAcc,乱序布局(不分 channel) | GPU | naive | replicate |
| **A3a** | A3 布局但**可掩**(陈旧行随流读出被掩,run 不断;2026-08-26 增) | GPU | naive-mask | replicate |
| A4 | + 分裂 channel(master/diff 分池) | GPU | master-diff | replicate |
| A5 | + 所有 prefill 注意力进 PIM(MQ n_cap=8:512 B/1.3004 GHz,能量钳位 8 tCK) | PIM | master-diff | **mq** |
| A6 | **Fugue(我们的方法)**:A5 + 逐请求动态选边 | dynamic | master-diff | **mq** |

关键约定:**attention batching(MQ 批命令,一次列读服务多条驻留查询)
与"prefill 上 PIM"同步启用**(A5 起);A1–A6 默认都在多轮 agentic 编排
下运行(`history_len`,由 workload 决定);曾经的"split 混合"档(GPU 算
新行、PIM 扫旧行、LSE 缝合)已废除。

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
分两族(详见 `README_software_upstream.md`):

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

- `intro/README_A1.md` … `README_A6.md` + `README_A3a.md`:每档定位
- `README_software_upstream.md`:复用策略族与文献来源
- `README_delta_vs_xinyao0821.md`:相对 xinyao_0821 基线的全部改动
- `README_head_hbm_remap.md`:head→HBM 重映射全记录(R18:错误、
  归因、量化、新映射规范)
- `README_manual_audit_findings.md`:**唯一审计总台账**(R/U 条目、
  流程裁决、阶梯诊断与 workload 有效性附录)
- `README_run_experiments.md`:**当前主线实验怎么跑**(五组拓扑负载 ×
  DAG 七档 × 五个重算比例,一键脚本与图)
- `README_experiments.md`:论文证据矩阵怎么跑
- `README_cppcore_branch.md`:`chenyi-822-cppcore-exp` 分支说明(核心 C++ 化)
- `RAW_DATA_MANIFEST.md`:**原始数据本地清单**(>50 MB 的事件轨迹
  `dag_A*.json` 不入库、记本机路径手动下载;≤50 MB 报告与 .log 已入库)
- `../output/analysis/`:结果分析工具与表格(`make_results_tables.py`
  → `RESULTS_k2.md`:七档的延迟/能量/prefill 放置/额外指标四表)
- `README_sweep_design.md`:**参数化 sweep 设计规范**(一个 gen_sweep.py +
  (topology, N, C, D, k),取代五组手调 workload,零 magic number;2026-08-29)
- `README_run_sweep_guide.md`:**运行指南**(另一台机器 clone 后照着跑:
  setup→跑 98 run→提取→独立复核→写 RESULTS_sweep.md→commit;2026-08-29)
- `../workload/gen_sweep.py` + `../workload/sweep/` + `../experiments/run_sweep.sh`:
  参数化 sweep 的 generator / workload / 批跑脚本
- `../workload/archived/2026-08-29_pre-sweep/`、`../output/archived/2026-08-29_pre-unify/`:
  归档的旧手调 generator 与旧结果(各带 README)
- `sessions/`:每日调整记录(chenyi9 裁决时间线)
- `archived/`:已归档——旧 workload 文档、走查稿、三份被合并的 audit
- `PORTING_PLAN.md`(不入库):干净分支 chenyi-822 的逐步移植计划

分支说明:本分支 (`chenyi-822-dirty`) 是**快速集成分支**;同容将按
逐步人工审阅规范挪入干净分支 `chenyi-822`。

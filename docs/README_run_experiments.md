# 怎么跑实验(DAG 六档阶梯;2026-08-26 版)

目标读者:接手人。本页讲**当前主线实验**——四组理论拓扑负载在物理事件
引擎上跑 A1–A6 六档、扫三个重算比例、出四组三柱动机图。论文证据矩阵
(真实 workload × A1–A6 × 三模型)另见 `README_experiments.md`。

## 1. 目前的四组实验(fig:motiv 负载套件)

四个**理论负载**(theoretical:结构照已发表框架、token 长度为声明的
假设值,机制展示级,不作证据;生成器与文献引用都在 `workload/gen_*.py`
的 docstring 与 JSON meta 里):

| 组 | 拓扑 | 文献依据 | 规模 | 位移决策 | 角色 |
|---|---|---|---|---|---|
| **star** | 1 main 指挥 3 worker × 5 轮 | AutoGen 2308.08155 / MetaGPT 2308.00352 / AgentCoder 2312.13010 | 20 请求,157k token,上下文至 40.7k 行 | 558 | 主场景(修 bug 协作) |
| **pipeline** | 架构→(工程师⇄审查)×5→测试,深度 12 链 | MetaGPT / ChatDev 2307.07924 | 12 请求,90k token | 299 | 接力链 |
| **debate** | 3 辩手 × 5 轮互看答案 + 裁判 | Du et al. 2305.14325 / MoA 2406.04692 | 16 请求,126k token | 428 | 全对全输出共享最密 |
| **mapreduce** | 8 mapper 私有片段 + reducer | Wu et al. 2109.10862 / LangChain 范式 | 9 请求,201k token | 8 | **薄共享对照**(验证低复用不添乱) |
| **multisource**(第五组,2026-08-26 增) | 12 个单轮查询,每个从 **96 个不同来源各取一个 256-token 块**(滑窗重叠,邻查询共享 95 源) | Lewis RAG 2005.11401 / CacheBlend 2405.16444 / MultiHop-RAG 2401.15391(真实同形负载=已过审的 multihop) | 12 请求,300k token,单请求 L=24,976(76% cap) | 1,056 | **多来源单块**:小块+来源最散——A3a(可掩)与 A3(断流)分离的判别场景 |

共同设计约定(均为 chenyi9 2026-08-26 裁决):

- **256-token 块**为自然 KV 块粒度;naive 布局按 append 序页粒度轮换,
  row conflict 只由轮换自身产生,**不得调参构造**;
- **共享内容逐轮引入**(每轮首拉 ~1/R、后续轮全量重读)——首次使用
  决定页的 append 位置,复用页的 channel 混叠自然发生;
- **单段上界**:一个 (请求,层) 连续 KV 段 ≤ 8 MiB = 32,768 行(AttAcc
  K 分区),负载顶到上界 76–86%("合理范围内尽量大");
- **重算比例轴**:EPIC k ∈ {2, 8, 32}(主图口径 k=2,低重算);
- **三柱口径**:问题① = A2(纯软件复用,KV 在远端哑存储,链路字节 =
  GPU↔远端存储经 NVLink/PCIe);问题② = A3(naive 乱序布局,无读掩:
  重算行处 master run 断开);我们 = A6(Fugue)。

## 2. 一键跑法

```bash
# 单负载 × 六档(并行发射)× 一个重算比例:
EPIC_K=2 bash experiments/run_dag_ladder.sh workload/workload_star_repair_r5w3k47.json LLAMA-7B
# 全套 4 负载 × 3 比例:
for K in 2 8 32; do for W in star_repair_r5w3k47 pipeline_repair_c5k50 \
    debate_d3r5k49 mapreduce_sum_m8; do
  EPIC_K=$K bash experiments/run_dag_ladder.sh workload/workload_${W}.json LLAMA-7B
done; done
```

- 输出:`output/<时间戳>_<负载>_<模型>_k<EPIC_K>/`——六份 `dag_Ax.json`
  + 日志 + `dag_ladder.csv`(每档:makespan、链路字节、**每部件能耗**
  GPU/LINK/PIM/DIE/TLB、**prefill 归边普查** pim/gpu/mixed/none 与行加权
  share);
- 画图:单负载三柱 `python3 experiments/plot_motiv_bars.py <dag_ladder.csv>`;
  **四组主图** `python3 experiments/plot_motiv_groups.py <输出stem>
  star=<csv> pipeline=<csv> debate=<csv> mapreduce=<csv>`;
- 旋钮:`EPIC_K`(重算比例,默认 8)、`RAMU_WORKERS`(每档 Ramulator
  worker 数,默认 15 = **96 核预算**:6 档×15 仿真 + 6 构图)。

## 3. 引擎与缓存行为(为什么第一遍慢、之后快)

- 裁决口径:**真实负载一律 `--engine dag` 出数**(物理事件引擎,六档
  全覆盖);解析引擎(`--engine analytic`,默认值,保历史命令兼容)只作
  快速预估与交叉校验;
- 每档运行分三段:**空跑构图**(单核 Python,收集全部扫描算子)→
  **预热**(≤worker 数并行把新形状仿真进落盘缓存
  `ramulator2/signature_cache.jsonl`)→ **正式构图**(全程查表);
  同负载复跑或换比例大量命中缓存,分钟级;全新负载首跑小时级;
- A2 不经 Ramulator,总是几分钟内先出。

## 4. 重新生成/改负载

`workload/gen_star_repair.py | gen_pipeline_repair.py | gen_debate.py |
gen_mapreduce_sum.py`,参数见 `--help`;改尺寸时守住单段 ≤32,768 行
(上界内尽量大),块粒度固定 256 token,不要人为挑块数/混叠。生成后用
`src/workload.py::load_workload + build_reuse_plan` 校验(各生成器输出
行会打印规模),体检口径同 `/data2/chenyi9/KV-PIM/workload/README.md`。

# 参数化 workload sweep — 设计规范 v2（CS 术语，待审阅，未写代码）

> **已归档 2026-09-03。** 参数化 sweep 的设计规范,页首自陈是"**先于代码写的规划**",
> `gen_sweep.py` 只实现了它的一个子集(§2.1 已逐处标注未实现的 SL/LL large out、
> scatter、funnel)。workload 族的设计((N,C,D,k)、零 magic number)仍然有效,
> 但页内 §7"哪个轴凸显哪对档"的判断建立在旧的 chunk 计价模型上,
> 已被 2026-09-03 三处引擎修正(heads-per-HBM `d3a3c4c`、striped-append 布局 `84f87f5`、真实 extent 进 Ramulator `897c294`) 取代。
> **workload 定义看 `../../workload/sweep/`;布局口径看
> `../README_data_layout_walkthrough.md`。**


> ⚠️ **2026-08-29 更新**：阶梯已重切为 **9 档**（A1 A2 A3 A3a **A3b** A4 **A4b**
> A5 A6，见 `README.md` §3），且 sweep 扩成 **6 模型 × 14 config × 9 档 = 756 run**
> 的 experiment1。本文下面写的"7 档 / 98 run"是旧口径,workload 族设计（(N,C,D,k)、
> 零 magic number）不变、仍适用。**怎么跑以 `README_run_sweep_guide.md` 为准**（含先
> 并行预热 Ramulator 缓存再跑）；§7 的"哪个轴凸显哪对档"里,decode 布局那几对现在还多了
> **A3(head→1channel) vs A3b(head 切片)** 和 **A4(固定切片) vs A4b(全局 co-read
> table)**,且需 `heads_per_hbm>1`(head 挤 HBM)才明显。

> ⚠️ **实现状态(2026-08-31 核对 `gen_sweep.py` 与全部 12 个
> `workload/sweep/*.json` 后如实标注)**:本文是**先于代码写的规划**(标题里的
> "未写代码"),`gen_sweep.py` 实现了它的一个**子集**。**未实现**的部分已在下面
> 逐处标出:**SL / LL 节点(large out)**、**scatter**、**funnel 拓扑**。
> 已实现的部分见 **§2.1 实现清单**。写方法一节时以 §2.1 为准,不要照本文的
> 规划部分写——那会和实际跑的东西对不上。
>
> 用**一个**参数化 generator 取代 5 个手调场景。把每个 workload 建成一个
> **分层 DAG（tiered DAG）**：node = agent = request，按 **tier（拓扑层）**
> 排布；tier 之间的连边是标准 dataflow primitive（**fan-out / fan-in /
> all-to-all / pipeline**）。扫 **(N, C, D, k)** 覆盖整个负载族。**全整数、
> 零 magic number、cap 天然不越**，并刻意设计成能凸显 A1–A6 各档区别（§7）。

## 1. 基本单位（固定）

| 量 | 值 | 说明 |
|---|---|---|
| **block** | 256 token | 长度单位（= 自然 KV-block）|
| **sys** | 16 token | 每 node 私有 system call |
| **cap** | 128 block = 32,768 token | 8-MiB K 分区单段硬上限（越界引擎拒绝）|
| **small** | 1 block（256）| node 只读/写一个对象 |
| **large** | W × block | node 读/写 W 个对象（W = 相连 tier 的 degree）。**只有"读"这一侧被实现**:tier>0 的 node 读全部上游输出 = W×block。**"写"这一侧未实现** —— 生成器里 `lout` 恒为 1 block |

## 2. Node 类型（在 I/O 尺寸上做 2×2）

| type | in | out | CS 含义（这个 node 在 dataflow 里扮演什么）|
|---|---|---|---|
| type | in | out | CS 含义 | 实现状态 |
|---|---|---|---|---|
| **SS** | small | small | map / 独立 worker（读一个、写一个）| ✅ 已实现(tier 0 的 node、pipeline 的每一节) |
| **SL** | small | **large** | **producer / source**：写一个大对象供下游 fan-out | ❌ **未实现** —— `lout` 恒为 1 block |
| **LS** | **large** | small | **reducer / sink**：fan-in 读全部上游、归约成一个 | ✅ 已实现(reduce 的末层、任何 tier>0 的 node) |
| **LL** | **large** | **large** | **all-to-all node**：读全部上游、也写给全部下游 | ⚠️ **半实现** —— 读是 large,写仍是 1 block |

`small = 1 block；large = W block`（= W 个 small 拼接）。

**"large out" 整体未实现的后果**:无论扇出度 N 多大,生产者永远只写 256 token。
即建模里假设"派活的内容量与下游人数无关"。真实 fan-out 场景中输出常随扇出度增长,
所以本 sweep **可能低估 fan-out 边上的 prefill 成本**。要补就是让生产者
`lout = W × BLOCK`,但那会改变 workload 定义、需要重跑。

### 2.1 实现清单(以 `gen_sweep.py` 与 `workload/sweep/*.json` 实测为准)

**已实现**:

| 项 | 实际形态 |
|---|---|
| 分层 DAG | node = agent = request,按 tier 排布 |
| **broadcast**(fan-out 1→N) | N 个消费者**各读同一份完整输出**(同一 sha,各 256 token) |
| **fan-in**(N→1) | reducer 读全部 N 个上游,各 256 token |
| **all-to-all**(N→N) | 每个 node 读全部 N 个上游,各 256 token |
| **pipeline**(1→1) | 链式,读上一个 |
| **supervisor** | `[1,N,1,N,…]`,fan-out 与 fan-in 交替 |
| 四条轴 | N(4/16/64)、C(16/32/64)、D(1/2/4)、k(2/8/32,运行时经 `EPIC_K`) |
| 共享语料 | C 个 block,sha 稳定 → 跨 node 复用,各自位移偏移 |
| **private 对照** | `--private`:每个 node 独占语料(零共享对照) |
| history | `history_len = t × 256`(自己前几轮的输出,append 散落) |

**未实现**(本文规划过、代码没有):

| 项 | 说明 |
|---|---|
| **SL / LL 的 large out** | `lout` 恒为 1 block,见上 |
| **scatter** | fan-out 只有 broadcast 一种。scatter(各读 1/N)意味着**零共享**,是共享的反面;零共享对照已由 `--private` 覆盖,故未单独建模 |
| **funnel 拓扑** | `--topology` 的 choices 只有 broadcast/reduce/alltoall/supervisor/pipeline |

## 3. Dataflow primitive（tier 间连边）——两个方向对称

设 tier `t` 有 `W_t` 个 node、tier `t+1` 有 `W_{t+1}` 个。连边类型由
`(W_t → W_{t+1})` 和下游读"一个还是全部"决定：

| primitive | degree 变化 | 下游每个 node 读什么 | CS 名 |
|---|---|---|---|
| **fan-out** | 1 → N | **已实现:broadcast** —— 都读那 1 个的完整输出。~~或各读大输出的 1/N（scatter）~~ ❌ **scatter 未实现**(它意味着零共享,已由 `--private` 对照覆盖)| broadcast |
| **fan-in** | N → 1 | 那 1 个 reducer 读**全部** N 个 | gather / reduce |
| **all-to-all** | N → N | 每个都读**全部** N 个上游 | all-reduce / gossip |
| **pipeline** | 1 → 1 | 读上一个 | chain |

**fan-out（1→N）与 fan-in（N→1）是一对对偶 primitive，各自独立。**

## 4. Topology = tier 的 width 序列（primitive 的组合）

一个 workload = 一串 tier 宽度 `[W0, W1, …]`，连边类型逐段由 §3 推出。
六个常见 topology（都是标准分布式 dataflow 模式）：

| topology | width 序列 | 由哪些 primitive 组成 | ≈ 旧场景 |
|---|---|---|---|
| **broadcast** | `[1, N]` | 纯 fan-out | 一发多 |
| **reduce**（map-reduce）| `[N, 1]` | 纯 fan-in | mapreduce |
| **all-to-all** | `[N]×D` | 每层 all-to-all | debate |
| **supervisor** | `[1, N]×R`（即 `[1,N,1,N,…]`）| fan-out 与 fan-in **交替** | star（旧名，已弃）|
| **pipeline** | `[1]×D` | 纯 chain | pipeline |
| **funnel** ❌ **未实现** | `[N]×(D−1)+[1]` | all-to-all 若干层 + 末尾 fan-in | judge 收敛 |

> **"star" 就是 supervisor**：一个 hub（degree 1）fan-out 给 N 个 worker，
> worker 再 fan-in 回 hub，多轮 → `[1,N,1,N,…]`。它**同时含** fan-out 和
> fan-in；纯 fan-in 单独由 reduce / funnel 覆盖。

每个 node 结构统一：`[sys 16] + [上游输入(按 primitive:一个或全部)] + [context: C block(共享，自己 offset 滑一格)] → 输出(small/large)`；对 context 每 block 重算 **k** 个 token（= diff）。**history 按 append 序散落存**（多段、不连续 = irregular access 的来源），不吃单段 cap。

## 5. 四个 sweep 轴 + baseline

| 轴 | 含义（CS）| 取值 | baseline |
|---|---|---|---|
| **N** | fan degree（扇出/扇入度 = 一个 tier 的宽度）| **4 / 16 / 64** | 16 |
| **C** | shared context 大小（block 数）| **16 / 32 / 64** | 32 |
| **D** | tier 数（DAG 深度）| **1 / 2 / 4** | 2 |
| **k** | 每 block 重算 token（diff 密度）| **2 / 8 / 32** | 8 |

- **N 为什么是一个数**：N 是**当前 config 的 fan degree**；宽 tier 都用这个
  N（窄 tier = 1）。要每 tier 不同宽度 = 任意 `[W0,W1,…]` 序列，sweep 时固定
  topology、只扫 degree N，保持"一次动一个变量"。
- **cap 检查**：最大单段 = context C ≤ 64 block ≤ 128（fan-in 输入与 history
  都散落存、不算单段）。最大只用 **50% cap**，无 magic number。

## 6. 完整 sweep 表

baseline = **alltoall**、N16、C32、D2、k8（supervisor 在 D=2 会退化成
`[1,N]`=broadcast，故 baseline 用 all-to-all；supervisor 作 D4 变体第 12 组）。
**OFAT（每次动一个标量轴）+ topology 变体 + 1 个 private 对照**：

| # | 组名 | topology | N | C | D | k | workload 文件 |
|---|---|---|---|---|---|---|---|
| 1 | **baseline** | alltoall `[16,16]` | 16 | 32 | 2 | 8 | wl_baseline_alltoall_N16_C32_D2 |
| 2 | N-lo | alltoall | **4** | 32 | 2 | 8 | wl_N4 |
| ~~3~~ | ~~N-hi~~ | ~~alltoall~~ | ~~**64**~~ | ~~32~~ | ~~2~~ | ~~8~~ | ~~wl_N64（128 agents，最重）~~ **已放弃，见 §6.1** |
| 4 | C-lo | alltoall | 16 | **16** | 2 | 8 | wl_C16 |
| 5 | C-hi | alltoall | 16 | **64** | 2 | 8 | wl_C64（50% cap）|
| 6 | D-lo | alltoall | 16 | 32 | **1** | 8 | wl_D1 |
| 7 | D-hi | alltoall | 16 | 32 | **4** | 8 | wl_D4（64 agents）|
| 8 | k-lo | alltoall | 16 | 32 | 2 | **2** | wl_baseline（k=2）|
| 9 | k-hi | alltoall | 16 | 32 | 2 | **32** | wl_baseline（k=32）|
| 10 | broadcast | **broadcast** `[1,N]` | 16 | 32 | — | 8 | wl_broadcast |
| 11 | reduce | **reduce** `[N,1]` | 16 | 32 | — | 8 | wl_reduce |
| 12 | supervisor | **supervisor** `[1,N,1,N]` | 16 | 32 | 4 | 8 | wl_supervisor_D4 |
| 13 | pipeline | **pipeline** `[1]×D` | 16 | 32 | 4 | 8 | wl_pipeline_D4 |
| 14 | **private**（对照）| alltoall + 私有 corpus | 16 | 32 | 2 | 8 | wl_private |

**~~每组跑 7 档 A1–A6 → 14 × 7 = 98 run~~**(旧口径)。

### 6.1 实际跑的口径(2026-08-31 更新,与 §6 表的差异)

| 项 | §6 表(规划) | **实际** |
|---|---|---|
| 档 | 7(A1–A6) | **9**(A1 A2 A3 **A3a A3b** A4 **A4b** A5 A6) |
| 模型 | 未指定 | **6**(LLAMA3-8B / LLAMA-7B / GPT-13B / LLAMA-33B / LLAMA-65B / GPT-175B)|
| run 数 | 98 | 6 × 14 × 9 = 756,**减去放弃的 N-hi 整行 6 个 → 702** |

**N-hi 已整行放弃**(2026-08-31 裁决,取代同日早先"只砍四个大的"的决定):
`wl_N64` 在**六个模型上一个都不跑**。

**先是硬容量,后是实测代价**。本轮实测出两个可用的规模模型:

> `W = 输出 token 数 × 层数 × (总 token / agent 数) / 1e9`(decode 工作量)
> **建图秒数 ≈ 122 + 2135 × W**(在 W=7.17 上外推 6.5 倍,实测差 15%)
> **常驻内存 ≈ 40 + 25 × W**(两个配置的线性系数 22.7 / 24.6 GB/W 一致,总量误差 2%)

代入:

| 任务 | W | 预测内存 | 单节点 1008 GB |
|---|---|---|---|
| GPT-175B × N-hi | 52.0 | **~1340 GB** | ❌ 放不下 |
| LLAMA-65B × N-hi | 43.3 | ~1120 GB | ❌ 放不下 |
| LLAMA-33B × N-hi | 32.5 | ~850 GB | ⚠️ 需近乎独占节点 |
| GPT-13B × N-hi | 21.7 | ~580 GB | ✓ 装得下,仍放弃 |
| **LLAMA-7B / LLAMA3-8B × N-hi** | 17.3 | ~470 GB | ✓ 装得下,**但两次都跑挂了** |

一个任务**不能跨节点拆分**(9 档已并行,单档建图是单线程),所以这不是调度能
解决的。不停放的后果是活锁:被领取 → 撑爆节点 → 被杀 → 因队列"最长优先"又被
第一个领走(实测 3.5 小时零完成)。

**为什么最后连两个小的也放弃了**。早先的判断是"保留那两个小模型的 N-hi 是零
成本的"——调度模拟显示砍四个大的与全砍 N-hi 的 makespan 相同。**那个判断是错
的,错在把成本只算作 makespan**。实际代价是两次长时间失败:

| 任务 | 结局 |
|---|---|
| `LLAMA-7B / N-hi` | 6 档死于 ENOSPC(node5 的 `/tmp` 只有 49 G),存活 3/9 |
| `LLAMA3-8B / N-hi` | 跑满 8 小时后被内存守卫取消,存活 **1/9** |

即使内存"装得下"(~470 GB 预测),W=17.33 仍是全 sweep 最重的任务——次重的只有
9.13。它需要**独占一个节点约 18 小时**,而准入按类别常量只给它预留 130 GB,
于是它被放到已有 big 槽的节点上,两者预测合计 717 GB 压在 1008 GB 机器上
(修复见 commit `a4456fe`)。

**科研代价:N 轴从三点降为两点。** 只剩 N=4(`N-lo`)与 N=16(`baseline`),
4 倍跨度、**没有第三点显示曲率**。任何"agent 数量如何 scaling"的结论都只能是
两点连线。这是本次放弃的真实代价,§7.1 的缺口 1 已按此重写。

补法(若日后要补):给**一个**模型做降规模 N-hi(64 agent 而非 128,W 减半
→ 约 600 GB),独占节点跑,只为把 N 轴补成三点。

**已落地**：`gen_sweep.py`（一个脚本出全部）、`workload/sweep/*.json`（12 个
distinct workload，已生成、全 ≤50% cap）、`experiments/run_sweep.sh`（批跑，
config 顺序、每 config 内 9 档并行；集群版编排见 `output/_orch2/`）。老脚本/负载归档在
`workload/archived/2026-08-29_pre-sweep/`。

## 7. 怎么 sweep 才凸显 A1–A6 各档区别（结合已跑结果）

每个轴/topology 专门放大某一对相邻档；佐证取自归档
`output/archived/2026-08-29_pre-unify/RESULTS_k2.md`（RAG，k2）。

| 要区分 | 档 | 靠哪个 sweep 点 | 机制 | 旧结果佐证 |
|---|---|---|---|---|
| **共享 vs 不共享** | A1 vs A2–A6 | ~~N-hi~~ → **N-lo↔baseline（只剩 4↔16）** | A1 dense 每 request 一份 KV，degree↑ 爆炸 | A1 87 s / **282 kJ**（PIM 279 kJ）vs A6 8.8 s / 1.5 kJ（**9.8× / 180×**）|
| **软件复用 vs PIM decode** | A2 vs A3+ | **D-hi + C-hi** | A2 每 tier 把整份 KV 过 link | A2 **KV over link 297.65 GiB**、LINK 26.6 J vs A5 **2.38 GiB** / 0.2 J（**125×**）|
| **naive 布局 irregular access** | A3 vs A3a | **k-hi** | A3 陈旧行断流；A3a 掩掉 | A3 decode E **1,480 mJ** → A3a **791 mJ** |
| **split-channel 布局** | A3a vs A4 | ~~N-hi +~~ **C-hi 单独扛**（row conflict 多）| A4 master/diff 分池、跨 channel 并行 | RAG 13.1→10.0 s；star 27.9→14.0 更明显 |
| **prefill 上 PIM** | A4 vs A5 | **C-hi + k-lo**（prefill memory-bound）| A5 prefill 进 bank、MQ 一次列读摊 N 条 | A5 prefill E 554k→**1,292k**、LINK 1661→**212**、KV-link→2.38 |
| **动态选边** | A5 vs A6 | **k-hi / cold start**（部分 prefill 更宜 GPU）| A6 把 GPU 上更便宜的 prefill 拉回 | pipeline A6 **58% PIM** vs A5 92%、能量 −8% |

**topology 各自打什么**：
- **all-to-all**（`[N]×D`）：输出全对全复用最密 → 放大 die 合并 / A3a 掩码；
- **reduce / funnel**（fan-in N→1）：聚合传输最重 → 放大 A2 的 link 成本；
- **broadcast**（fan-out 1→N）：一份 KV 服务 N 个 → 放大 MQ batching（A5/A6）；
- **pipeline**（chain）：无共享 fan → 布局档（A3/A4）差异最小的对照；
- **private**：no-reuse 地板，验证低复用时各档不添乱。

一条 OFAT sweep 就把"每上一档多解决一件事"逐段量出来。


## 7.1 每条轴对应论文的哪个贡献(2026-08-31 加)

稿子 §1 的三条贡献(`sections/01-intro.tex` 的 contributions 列表)与本 sweep
轴的对应。**判据**:一条贡献要成立,必须有至少一条轴能让它的机制**从不起作用
变到起作用**,否则那条贡献在评测里没有自变量。

### 轴的全集(五条,不是四条)

前面 §5 的四条是 **workload 轴**;还有一条 **system 轴**同样在扫,而且对贡献 2
最关键,一并列出:

| 轴 | 取值 | 属性 |
|---|---|---|
| **N** fan degree | 4 / 16 / 64 | workload |
| **C** 共享 context | 16 / 32 / 64 block | workload |
| **D** DAG 深度 | 1 / 2 / 4 | workload |
| **k** 每 block 重算 token | 2 / 8 / 32 | workload(运行时 `EPIC_K`)|
| **topology** | broadcast / reduce / alltoall / supervisor / pipeline | workload |
| **模型** → `heads_per_hbm` | LLAMA3-8B 8 / LLAMA-7B 32 / GPT-13B 4 / LLAMA-33B ~5 / GPT-175B 3 / LLAMA-65B 2 | **system** |

### 贡献 ↔ 轴

| 贡献(稿子 §1) | 它的机制在什么条件下才起作用 | 对应的轴 | 对应档 |
|---|---|---|---|
| **C1 异构 GPU–PIM 存储系统**:通道分成 shared 侧与 per-agent 侧,die 合并两侧的分数与上下文 | 两侧都非空,且**比例要变** —— 全共享则 diff 通道空转,全私有则 shared 侧无意义 | **k**(直接决定落到 diff 侧的量)+ **C**(决定 shared 侧的量);两者张成 shared:per-agent 比例平面 | A3a→A4 |
| **C2 sharing-aware 数据映射**:共享块保持 dense、重算 token 紧凑打包、逻辑位置由 driver 装进 die,**一份存储服务所有 agent** | "一份服务所有" —— 必须有**多个 agent 读同一块**,且数量要变 | **N**(同读同一指纹的 agent 数)+ **k**(重算 token 的紧凑打包量);**模型轴**决定 `heads_per_hbm`,而 A4 与 A4b **只有在 head 挤 HBM 时才分开** | A3→A3a→A3b→A4→A4b |
| **C3 多 agent 工作流步的执行模型**:(a) 按 computed-token 数逐 prefill 选边;(b) 多 agent 共乘一次行激活;(c) 依赖允许时各阶段流水 | (a) 阈值的自变量是 computed token 数 q;(b) 需要**足够多**的同读 agent 才填得满批;(c) 需要**依赖结构**有可重叠与不可重叠两端 | (a) **k**(q = k × 位移块数)+ **C**;(b) **N** —— 注意 **N-lo=4 低于 MQ 批容量 8**,是"批填不满"的有效低点;(c) **D** + **topology**(pipeline 纯链 = 无重叠端,all-to-all = 最大重叠端)| A4b→A5→A6 |

**结论:三条贡献各自都有至少一条轴,而且多数有两条。**

### 但有三处覆盖缺口,必须记下来

1. **N-hi 整行放弃,N 轴只剩两点**(2026-08-31 裁决,六个模型一个都不跑,
   见 §6.1 与 `sessions/2026-08-31.md`)。**这是当前最严重的缺口**,而且比早先
   "只砍四个大的"那版更严重:

   - N 是 **C2 与 C3(b) 的主轴**,现在只有 N=4(`N-lo`)与 N=16(`baseline`)。
     4 倍跨度、**两点连线,没有第三点显示曲率** —— 无法区分"随 N 线性增长"与
     "随 N 饱和/爆炸",而 A1 的机制恰恰是**随 degree 爆炸**,两点看不出爆炸。
   - `heads_per_hbm` 最小、A4↔A4b 区分度最高的**大模型一端本来就靠 N-hi**,
     现在彻底没有了。
   - **A1 vs A2–A6 这一对(共享 vs 不共享)失去了它的指定轴**,只能退回
     `N-lo↔baseline`,而这两点的 degree 差异小得多。

   **可用的替代**:`k-hi`(k=32)与 `C-hi`(C=64)仍在,二者都能加压复用结构;
   `supervisor`/`broadcast` 的 `[1,N]` 也提供 fan-out degree 变化。**但没有一条
   是 N 本身。**若要补,给一个模型做降规模 N-hi(64 agent 而非 128,W 减半
   → 约 600 GB),独占节点跑,只为把 N 轴补成三点。
2. **`large out` 未实现**(见 §2):生产者输出不随扇出度增长,所以 fan-out 边上
   被产出的 token 数被固定住,轻微低估 C3(a) 阈值输入的一端。影响小于第 1 条。
3. **MQ 批容量固定为 8**,不是自变量。它是硬件能力(n_cap=8,512 B / 1.3004 GHz),
   属 C 系列微架构实验(见 `README.md` §5),不在本 sweep 的范围。C3(b) 在本
   sweep 里由 **N** 提供自变量(有多少 agent 可供批),而不是由批容量提供。

## 8. cap / history 建模 / 与旧结果

- **cap**：所有点 C ≤ 64 block（≤ 50% 的 128-block cap）；fan-in 输入与
  history 散落存、不占单段。**无需反推 85%、无 magic number。**
- **history 存储口径（需你确认的建模点）**：新模型把多轮 node 的 history 按
  **append 序散落**（多段），与 irregular-access 故事一致；旧模型当**一整段
  连续**（才要把 47/49/50 反推到 85% cap）。散落后 D 不再吃 cap。
- **旧结果**：归档 `output/archived/2026-08-29_pre-unify/`（21 run + 旧
  workload + RESULTS 快照），仍有效、作对照。

---

## 待你审阅
1. **CS 术语的 topology**（fan-out / fan-in / all-to-all / pipeline / supervisor / reduce）对吗？"star = supervisor" 这样澄清可以吗？
2. **取值点** N{4,16,64}、C{16,32,64}、D{1,2,4}、k{2,8,32}、baseline (supervisor,16,32,2,8) 行不行？
3. **规模** 14 组 / 98 run，还是砍成 10 组 / 70 run？
4. **history 建模**改成 append 序散落（多段、D 不吃 cap）同意吗？
5. 确认后我写 `gen_sweep.py`（一个脚本、`--topology --N --C --D --k` 出全部组）+ 跑批脚本。

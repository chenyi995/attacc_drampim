# 参数化 workload sweep — 设计规范 v2（CS 术语，待审阅，未写代码）

> ⚠️ **2026-08-29 更新**：阶梯已重切为 **9 档**（A1 A2 A3 A3a **A3b** A4 **A4b**
> A5 A6，见 `README.md` §3），且 sweep 扩成 **6 模型 × 14 config × 9 档 = 756 run**
> 的 experiment1。本文下面写的"7 档 / 98 run"是旧口径,workload 族设计（(N,C,D,k)、
> 零 magic number）不变、仍适用。**怎么跑以 `README_run_sweep_guide.md` 为准**（含先
> 并行预热 Ramulator 缓存再跑）；§7 的"哪个轴凸显哪对档"里,decode 布局那几对现在还多了
> **A3(head→1channel) vs A3b(head 切片)** 和 **A4(固定切片) vs A4b(全局 co-read
> table)**,且需 `heads_per_hbm>1`(head 挤 HBM)才明显。

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
| **large** | W × block | node 读/写 W 个对象（W = 相连 tier 的 degree）|

## 2. Node 类型（在 I/O 尺寸上做 2×2）

| type | in | out | CS 含义（这个 node 在 dataflow 里扮演什么）|
|---|---|---|---|
| **SS** | small | small | map / 独立 worker（读一个、写一个）|
| **SL** | small | large | **producer / source**：写一个大对象供下游 fan-out |
| **LS** | large | small | **reducer / sink**：fan-in 读全部上游、归约成一个 |
| **LL** | large | large | **all-to-all node**：读全部上游、也写给全部下游 |

`small = 1 block；large = W block`（= W 个 small 拼接）。

## 3. Dataflow primitive（tier 间连边）——两个方向对称

设 tier `t` 有 `W_t` 个 node、tier `t+1` 有 `W_{t+1}` 个。连边类型由
`(W_t → W_{t+1})` 和下游读"一个还是全部"决定：

| primitive | degree 变化 | 下游每个 node 读什么 | CS 名 |
|---|---|---|---|
| **fan-out** | 1 → N | 都读那 1 个的输出（broadcast），或各读大输出的 1/N（scatter）| broadcast / scatter |
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
| **funnel** | `[N]×(D−1)+[1]` | all-to-all 若干层 + 末尾 fan-in | judge 收敛 |

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
| 3 | N-hi | alltoall | **64** | 32 | 2 | 8 | wl_N64（128 agents，最重）|
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

**每组跑 7 档 A1–A6 → 14 × 7 = 98 run。** 想省可只留 9 组 OFAT + private = 70 run。

**已落地**：`gen_sweep.py`（一个脚本出全部）、`workload/sweep/*.json`（12 个
distinct workload，已生成、全 ≤50% cap）、`experiments/run_sweep.sh`（批跑，
config 顺序、每 config 内 7 档并行）。老脚本/负载归档在
`workload/archived/2026-08-29_pre-sweep/`。

## 7. 怎么 sweep 才凸显 A1–A6 各档区别（结合已跑结果）

每个轴/topology 专门放大某一对相邻档；佐证取自归档
`output/archived/2026-08-29_pre-unify/RESULTS_k2.md`（RAG，k2）。

| 要区分 | 档 | 靠哪个 sweep 点 | 机制 | 旧结果佐证 |
|---|---|---|---|---|
| **共享 vs 不共享** | A1 vs A2–A6 | **N-hi** | A1 dense 每 request 一份 KV，degree↑ 爆炸 | A1 87 s / **282 kJ**（PIM 279 kJ）vs A6 8.8 s / 1.5 kJ（**9.8× / 180×**）|
| **软件复用 vs PIM decode** | A2 vs A3+ | **D-hi + C-hi** | A2 每 tier 把整份 KV 过 link | A2 **KV over link 297.65 GiB**、LINK 26.6 J vs A5 **2.38 GiB** / 0.2 J（**125×**）|
| **naive 布局 irregular access** | A3 vs A3a | **k-hi** | A3 陈旧行断流；A3a 掩掉 | A3 decode E **1,480 mJ** → A3a **791 mJ** |
| **split-channel 布局** | A3a vs A4 | **N-hi + C-hi**（row conflict 多）| A4 master/diff 分池、跨 channel 并行 | RAG 13.1→10.0 s；star 27.9→14.0 更明显 |
| **prefill 上 PIM** | A4 vs A5 | **C-hi + k-lo**（prefill memory-bound）| A5 prefill 进 bank、MQ 一次列读摊 N 条 | A5 prefill E 554k→**1,292k**、LINK 1661→**212**、KV-link→2.38 |
| **动态选边** | A5 vs A6 | **k-hi / cold start**（部分 prefill 更宜 GPU）| A6 把 GPU 上更便宜的 prefill 拉回 | pipeline A6 **58% PIM** vs A5 92%、能量 −8% |

**topology 各自打什么**：
- **all-to-all**（`[N]×D`）：输出全对全复用最密 → 放大 die 合并 / A3a 掩码；
- **reduce / funnel**（fan-in N→1）：聚合传输最重 → 放大 A2 的 link 成本；
- **broadcast**（fan-out 1→N）：一份 KV 服务 N 个 → 放大 MQ batching（A5/A6）；
- **pipeline**（chain）：无共享 fan → 布局档（A3/A4）差异最小的对照；
- **private**：no-reuse 地板，验证低复用时各档不添乱。

一条 OFAT sweep 就把"每上一档多解决一件事"逐段量出来。

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

# `output/analysis/` 交接说明：RESULTS_k2/k8/k32 三份结果 + A1–A6 逐档差异 + A4 row-conflict 核查

> **这份文档写给不了解本代码仓库的人或 AI。** 目标：读完就能看懂
> `output/analysis/` 里那三份已经跑好的结果 MD（`RESULTS_k2.md` /
> `RESULTS_k8.md` / `RESULTS_k32.md`）、知道 A1–A6 阶梯里**每一档比上一档多做了
> 什么**、并且能回答一个具体核查问题：**A4 到底有没有实现"避免 row conflict"**。
> 所有结论都给了代码出处（`文件:行号`），可自行复核。

---

## 0. 30 秒背景（不懂这个项目也能往下读）

- 大语言模型（LLM）生成分两段：**prefill**（把整段输入一次算完、写出 KV cache）
  与 **decode**（逐字生成，每个新字都要把历史 KV cache 从头读一遍做 attention）。
  **KV cache** = 每个历史 token 存下的一条 K 向量 + 一条 V 向量。decode 的瓶颈是
  **内存带宽**（把 KV 从显存搬到算力单元），不是算力。
- **AttAcc**（ASPLOS'24）把 decode attention 搬进 HBM 每个 **bank** 旁边的小算力
  单元里算，这类结构叫 **PIM**（processing-in-memory，存内计算）。本仓库在 AttAcc
  的开源仿真器上扩展，研究 **Fugue**：多智能体（multi-agent）、多轮场景下多个请求
  **共享同一份 KV**（相同 system prompt、共享文档 chunk、上一轮输出）时，GPU 与 PIM
  怎么分工、KV 怎么摆。
- **A1–A6 是一个"放置消融（placement ablation）"阶梯**：每一档只在上一档基础上
  **多改一件事**，所以"相邻两档的差"就精确对应论文里的一个设计决策。这份文档第 3、4
  节把每一档的"那一件事"讲透。
- **两条仿真路径**（同一个 workload JSON 可从两条跑）：
  - **解析引擎（analytic）**：`src/ablation.py`，封闭式代价公式，快，用于预估/交叉校验。
  - **物理事件引擎（DAG / event）**：`src/workload_runner.py`，把每请求每层展开成带
    真实时序、依赖、资源排队的事件图。
  - **裁决（2026-08-26）：真实 workload 一律走物理事件引擎出数。** ⚠️ **`RESULTS_k*.md`
    里的数全部来自 DAG 引擎的 `dag_A*.json`**——所以第 4 节的 row-conflict 核查
    主要查 `src/workload_runner.py`（不是 `ablation.py`）。

---

## 1. `RESULTS_k2.md` / `RESULTS_k8.md` / `RESULTS_k32.md` 是什么

三份结构完全相同的结果表，区别只在 **k**：

| 文件 | k | k 的含义 |
|---|---|---|
| `RESULTS_k2.md` | 2 | reuse policy 对每个"位移过的 chunk（shifted chunk）"重算的 token 数 |
| `RESULTS_k8.md` | 8 | 同上，重算 8 个 |
| `RESULTS_k32.md` | 32 | 同上，重算 32 个 |

- **k 是什么**：复用 KV 会有精度损失，reuse policy（这里用 `recompute`）会对每个被
  位移的共享 chunk **重算前若干个 token 的 KV** 来修正精度。k 越大 → 重算越多 →
  精度越好但省得越少。k 是 **run 时参数**（脚本里的 `EPIC_K` / CLI
  `--epic-prefix-recompute-tokens`），不改 workload 本身。这三份就是同一批 workload、
  同一个 A1–A6 阶梯，在三个 k 下各跑一遍。

- **怎么生成的**：`output/analysis/make_results_tables.py`，从每个 run 目录下的
  `dag_A1.json … dag_A6.json`（外加 `dag_A3a.json`）读**实测字段**汇总。**所有数值
  都是仿真实测，不是估算**；每张表都标了来源 run 目录名与 JSON 字段。要复现：
  ```bash
  python3 output/analysis/make_results_tables.py      # 重新生成三份 RESULTS_k*.md
  ```

- **覆盖哪些 workload**：**5 个手调的"真实拓扑"负载**（`make_results_tables.py`
  顶部 `WL` 列表）：
  1. **star-repair** — 星型（1 main 指挥 3 worker，5 轮），仿 AutoGen/MetaGPT。
  2. **pipeline-repair** — 瀑布链（architect→engineer/reviewer×5→tester），仿 ChatDev。
  3. **debate** — 3 个对称 debater 多轮辩论 + 1 judge，仿 Mixture-of-Agents。
  4. **map-reduce** — 8 mapper（各读私有切片）+ 1 reducer，**低复用对照**。
  5. **multi-source RAG** — 12 个单轮 RAG，滑窗共享 96 中的 95 个 source chunk。

- **每份 MD 的结构**（读的时候按这个顺序理解）：
  1. **各 workload 的编排（分 tier 讲）**：每个 workload 拆成 tier（见术语表），逐 tier
     说"谁在干什么"。tier map 取自该 run 的 `dag_A2.json` 的 `workload.tiers` 字段。
  2. **总表**：per-workload 的端到端延迟、能量、A6-vs-A1/A2 的加速比。
  3. **逐 workload 明细**：**per-layer 延迟**（layer = 编排轮次/链位，见术语表）、
     **per-rung 功率**、**能量拆分**（prefill vs decode、GPU/PIM/LINK/DIE）、以及每个数
     对应的源 `dag_A*.json` 文件名。

- **和新 sweep 的关系**（重要，别混淆）：这 5 个是**旧的手调 workload**。因为它们的
  共享 chunk 数是非整数（47/49/50，是"把最深 history 填到 32,768-token cap 的
  85–86%"反推出来的），后来另立了一套**参数化 sweep**（`workload/gen_sweep.py` +
  `docs/README_sweep_design.md`，结果在 `RESULTS_sweep.md`）用干净整数
  `(topology, N, C, D, k)` 消除 magic number。**这 5 个手调负载仍然有效**，作为
  "有名字的真实拓扑对照"保留；sweep 是它们的结构化超集。想了解 sweep 见
  `docs/README_run_sweep_guide.md`。

---

## 2. A1–A6 阶梯总览：一句话一档

| 档 | 一句话 | prefill attn | decode attn | KV 布局（`kv_mapping`） | 批命令 |
|---|---|---|---|---|---|
| **A1** | AttAcc 原样，**无复用**（参照点） | GPU | PIM | `private`（每请求各存一份） | replicate |
| **A2** | **纯软件复用**，无 PIM 算力 | GPU | **GPU** | `none`（KV 在远端哑存储） | replicate |
| **A3** | 软件复用 + PIM decode，**乱序布局** | GPU | PIM | `naive`（append 序散落分页） | replicate |
| **A3a** | A3 布局但**陈旧行可掩**（run 不碎） | GPU | PIM | `naive-mask` | replicate |
| **A4** | **+ 分裂 channel**（master/diff 分池） | GPU | PIM | `master-diff` | replicate |
| **A5** | + **所有 prefill attn 进 PIM** + MQ 批 | **PIM** | PIM | `master-diff` | **mq** |
| **A6** | **Fugue（本方法）**：A5 + 逐请求动态选边 | **dynamic** | PIM | `master-diff` | **mq** |

- 权威定义：`src/ablation.py` 的 `PRESETS`（约 87–118 行）与 `PRESET_LABELS`。
- 一条铁律：**"prefill 上 PIM" 与 "MQ 批命令" 绑定同时启用**（A5 起）；A1–A4 都用
  legacy 的 `replicate` 命令。原因见 `ablation.py:112-118` 注释。
- 每一档都默认跑在多轮 agentic 编排下（每请求带 `history_len`，由 workload 决定）。

---

## 3. 逐档差异：每一档比上一档**多做了什么**（详细）

> 读法：每小节 = "A(x) 相比 A(x-1) 改了哪个 config 字段 → 物理上意味着什么 →
> 代码在哪 → 结果里预期看到什么"。字段名对应 `AblationConfig`（`ablation.py`）。

### A1（参照点，无上一档）
- **是什么**：原封不动的 AttAcc。decode attention 在 PIM（AttAcc 的本职），prefill
  attention 在 GPU，**KV 完全不共享**（`kv_mapping="private"`：哪怕两个请求 system
  prompt 一样，也各存各的、各扫各的）。批命令 `replicate`（一列读服务一条查询）。
- **意义**：这是"没有任何复用优化"的地板线。后面每一档省下来的东西都相对它衡量。

### A2 vs A1 —— 改的是"**是否软件复用**"
- **唯一改动**：`kv_mapping: private → none`，且 `decode_attn: pim → gpu`。
- **物理意义**：A2 打开**软件层复用**——共享的 KV 只算/存一份，多个请求逻辑上共用。
  但 A2 **不给 PIM 算力**：decode attention 退回 **GPU**，KV 放在**远端哑存储**
  （dumb storage），每步 decode 都要把 KV 经 NVLink/PCIe 链路搬到 GPU（计入
  `link_bytes`，R10 裁决）。也就是"只有软件复用、没有存内计算"。
- **A2-A1 隔离的变量**：**软件复用本身**的收益/代价（`ablation.py:76` 注释：
  "A2-A1 = software reuse alone"）。
- **结果预期**：共享度高的 workload（RAG、star）A2 比 A1 省很多重复计算；但因为 KV
  过链路，`link_bytes` 会很大（历史归档：RAG 的 A2 KV-over-link 达 297.65 GiB，而
  A5 只有 2.38 GiB）。**注意 A2 会把 prefill/decode 两相位合成一个时间戳**
  （GPU-only 路径），所以它的"分相位延迟"在表里显示 n/a，只有 makespan/能量可比。

### A3 vs A2 —— 改的是"**decode 搬进 PIM bank + KV 常驻**"
- **唯一改动**：`decode_attn: gpu → pim`，`kv_mapping: none → naive`。
- **物理意义**：把 decode attention 从 GPU 搬回 **PIM bank**（AttAcc 的本行），KV
  **常驻在 PIM 的 HBM 里**、不再每步过链路。但此时 KV 是**乱序摆的**：`naive` 布局
  按**软件 append 顺序**把每份 reservation 切成 256-token 的页（page），**轮流丢到
  16 个 channel**上（round-robin）。同一段逻辑 context 的页因此散落在不同地址、被
  别的请求的页夹在中间。
- **A3-A2 隔离的变量**：**PIM decode + KV 常驻**（`ablation.py:76`："A3-A2 = PIM
  decode + KV residency"）。
- **代价（关键）**：这种"PIM-oblivious"的乱序分页会产生 **row conflict**——见第 4 节。
- **代码**：`NaiveKVLayout`（`workload_runner.py:1025`）。`shadow_reads=False`：被修正
  的行，其 master 副本被**跳过（skip）**，把 master run 切断，修正行单独一次 row
  activation。
- **结果预期**：相对 A2，decode 能量大降（历史归档：某负载 decode 能量 1480→791 mJ），
  链路字节大降；但因乱序布局，decode 时间没到理想值（留给 A4 修）。

### A3a vs A3 —— 改的是"**陈旧行可掩，run 不碎**"（更温和的 naive）
- **唯一改动**：`kv_mapping: naive → naive-mask`（`master_shadow` 语义从 `skip`→
  `read-mask`）。
- **物理意义**：还是 A3 那套"round-robin 分页乱序布局"，**channel 仍不分池**；唯一区别
  是——被修正过的**陈旧 master 行不再从地址流里剔除**，而是**照样读出来、在 score 阶段
  用掩码丢掉**。这样一段 chunk 的 master run **保持连续、不在每个修正点断裂**。
- **为什么单列一档**：`skip`（A3）会因为"每遇一个修正就断 run"制造大量冷启动 run，惩罚
  偏重；`read-mask` 把这部分惩罚摘掉，**只留下"channel 乱序"这一种惩罚**。于是 A3→A3a
  隔离出"run 碎裂"的代价，A3a→A4 才干净地隔离出"channel 冲突"的代价。
- **代码**：`NaiveMaskKVLayout`（`workload_runner.py:1128`），继承 `NaiveKVLayout`，
  只把 `shadow_reads` 翻成 `True`。
- **结果预期**：A3a 通常介于 A3 与 A4 之间——比 A3 好（run 连续了），但仍差于 A4
  （channel 还乱）。

### A4 vs A3a —— 改的是"**分裂 channel：master/diff 分池并发**"（← row-conflict 核查在此）
- **唯一改动**：`kv_mapping: naive-mask → master-diff`。
- **物理意义**：换掉"round-robin 单 channel 分页"，改成 **PIM-aware 布局**——把 16 条
  channel 分成两个**互不相交**的池：
  - **master 池 = channel 0–14**（15 条）：放所有**不变的共享行**，每段一条**连续**的
    AttAcc 条带流（不再挖洞、不再冷启动碎片）。
  - **diff 池 = channel 15**（1 条）：放**重算出来的修正行**，一条连续 extent。
  - 两池 channel 集不相交 → decode 时**并发扫描**（各自事件在 DAG 里落在不同设备上，
    时间取 **max 而非 sum**）。被修正的 master 行仍留在 master 流里读、被掩码丢掉
    （`read-mask` 继承自 A3a），所以 master run 依然连续。
- **A4-A3a 隔离的变量**：**PIM-aware layout / 避免 channel 冲突**（`ablation.py:77`：
  "A4-A3 = PIM-aware layout"）。**这就是"避免 row conflict"那一档**——详见第 4 节。
- **默认 15/1 划分**：`master_pool_channels=15`（`ablation.py:149`；DAG 引擎里是常量
  `_KV_CHANNELS`，`workload_runner.py:522-524`）。为什么 15/1 而非 8/8：diff 行很少，
  给它 8 条会白白腰斩 master 流（audit 3a / R5 裁决 2026-08-25）。
- **结果预期**：相对 A3a，decode 布局惩罚消失，decode 时间/能量进一步降到接近理想。

### A5 vs A4 —— 改的是"**所有 prefill attn 进 PIM + MQ 批命令**"
- **改动（一个 package，三件事同时开）**：`prefill_attn: gpu → pim`，
  `pim_batch_command: replicate → mq`，`pim_pe_freq_ghz=1.3004`，`gemv_buffer_bytes=512`。
- **物理意义**：到 A4 为止 prefill attention 一直在 GPU；A5 把它**也搬进 PIM bank**。
  搬进去之后就能用 **MQ 批命令（multi-query）**：**一次 DRAM 列读同时服务多条驻留查询**
  （n_cap=8：buffer = 8×64 B = 512 B），而不是 replicate 那样一列一查询。PE 频率钉在
  **1.3004 GHz**（= 1/tCK，RTL 核对的平衡点：功耗钳位把 MQ 间隔锁在 8 tCK / n=8，见
  `ablation.py:112-129` 注释与 `kvpim-rtl/docs/Fugue-asplos2027`）。
- **A5-A4 隔离的变量**：**prefill attention 上 PIM（连带它才能开的批处理）**
  （`ablation.py:77-78`："A5-A4 = prefill attention on the PIM (with the batching it
  enables)"）。
- **为什么 MQ 只在 A5/A6**：批命令与"prefill 上 PIM"是**同一个 C3 微架构设计点**，绑定
  出现；A1–A4 即便理论上能批也一律 replicate（`ablation.py:107-111`）。
- **结果预期**：prefill attention 不再占 GPU、也不再把 reused KV 经链路读回；共享度高时
  一次列读摊到多条查询，PIM 侧吞吐大增。

### A6 vs A5 —— 改的是"**逐请求动态选边**"（Fugue 本方法）
- **唯一改动**：`prefill_attn: pim → dynamic`。
- **物理意义**：A5 把**所有** prefill attention 一刀切放 PIM；A6 改成**逐请求（逐 class）
  动态判**该请求的 prefill attention 走 GPU 还是走 PIM——共享多、驻留查询多的请求上
  PIM 吃批处理红利，共享少的走 GPU 更划算。这条**动态规则就是 Fugue 的方法本体**。
- **A6-A5 隔离的变量**：**动态 per-request 规则**（`ablation.py:78-79`："A6-A5 = the
  dynamic per-request rule (Fugue)"）。
- **哪些请求上了 PIM**：看 `dag_A6.json` 的 `prefill_attention_sides` 字段（qID→
  `gpu`/`pim`），`make_*_tables.py` 的 "prefill-PIM%" 列就是这里 `=='pim'` 的占比。
- **结果预期**：A6 ≤ A5（动态永不比全上 PIM 差），差距大小取决于 workload 里"低共享
  请求"的比例——纯高共享负载 A6≈A5，混合负载 A6 明显更优。

---

## 4. 核查：A4 到底有没有实现"避免 row conflict"？

### 结论

> **✅ 有。A4 就是引入 row-conflict 避免的那一档**（A3/A3a 没有，A5/A6 继承 A4 的布局）。
> 机制 = **master/diff 分到不相交的 channel 池、并发扫描**（时间取 max 不是 sum）
> **＋ 每池一条连续流、修正行用 read-mask 掩掉而不打断 run**。
> 证据在 DAG 引擎 `src/workload_runner.py`（就是产出 `RESULTS_k*.md` 的那条路径），
> 解析引擎 `src/ablation.py` 同构。

### 先说清楚"row conflict"在这里指什么

一次 decode 要把一段 context 的很多行一起读。若这些**一起读的行落在同一条 channel
上**，它们只能**排队串行**（一条 channel 一次只服务一个 row activation）——这个串行化
就是这里说的 row conflict。避免它 = 让"一起读的行"落到**不同 channel**、从而**并行**。
naive 布局（A3/A3a）恰恰相反：round-robin 分页 + 别的请求的页夹在中间，导致同一
context 的页**挤在同 channel、且各自成为独立 row activation**——冲突拉满。

### 代码证据（DAG 引擎，权威）

1. **两个不相交的 channel 池**——`workload_runner.py:522-524`：
   ```python
   _KV_CHANNELS = {
       "master": tuple(range(0, 15)),   # ch0–14：不变的共享行
       "diff":   tuple(range(15, 16)),  # ch15：重算的修正行
   }
   ```
   master 与 diff 的 channel 集**完全不相交**。

2. **A4 布局把两池摆进不相交 channel、每池一条连续流**——`CacheBlendTLB.finalize()`
   （`workload_runner.py:731+`）：docstring 明说"prevent K/V or master/diff overlap"，
   "Every master extent spans channels 0–14 and every diff extent spans channel 15"，
   且"Individual ownership blocks do **not** reserve a fresh head partition: doing so
   creates artificial holes and turns one physical scan into many cold-start runs"——
   即**连续、不挖洞**（这是 read-mask 连续性的物理落地）。

3. **两池并发计时（取 max 不是 sum）**——消费端按 `(channel_base, channel_count)`
   把该次读**分组成独立 pool 事件**（`workload_runner.py:1225-1228`），每个 pool 事件
   落在不同设备名 `PIM:pool0-14` / `PIM:pool15-15` 上，DAG 调度器让不同设备**并发**，
   每池拿**per-HBM 带宽份额**。解析引擎同构：`_decode_block_time` 对 `profile.pools`
   逐池算时间后 `exec_time = max(pool_times)`（`ablation.py:995-1003`，注释：
   "Disjoint channel pools stream concurrently; extents inside a pool are serial"）。

4. **修正行 read-mask、不打断 master run**——`_physical_reads`
   （`workload_runner.py:655+`）：被修正的行**两次读**（diff 行被打分 + 被遮蔽的 master
   行照读但从 score 掩掉），`CacheBlendTLB.shadow_reads=True`。docstring:
   "reading the shadowed row instead would break the master stream into one cold-start
   ..."——即用掩码换取 run 连续。

### 对照：A3/A3a 为什么**没有**避免（这才反衬出 A4 有）

- **A3 `NaiveKVLayout`**（`workload_runner.py:1025+`）：`shadow_reads=False`，且明说
  "**No master/diff channel split and no PIM-aware remap**：每份 reservation 切成
  256-token 页，页按软件 append 序 **round-robin 丢 16 个单 channel 池**……共享 channel
  的页地址不相邻，各自一次独立 row activation——**the row-conflict penalty of a
  PIM-oblivious paged store**"。冲突是从"轮转"里**涌现**的，不是人为构造的
  （代码注释 `workload_runner.py:966`："the row conflicts must come from the rotation
  itself"）。
- **A3a `NaiveMaskKVLayout`**（`workload_runner.py:1128+`）：**只**把 `shadow_reads`
  翻成 `True`（run 不再因修正而碎），**channel 仍不分池、仍 round-robin**——所以
  **row conflict 依旧存在**，A3a 只修了"run 碎裂"那一半，没修"channel 冲突"那一半。

### 诚实边界（别把结论说过头）

- A4 避免的 row conflict 特指两件事：**(a) co-read 的 master vs diff 落到不相交
  channel 并发**、**(b) 每池一条连续流（修正行掩码不打断 run）**。它**不是**声称"逐行
  在 bank 间做花式轮转"——单条 run 在池内本就按 head→HBM 条带（R18，恒开）铺满该池的
  channel；A4 改的是"池的划分方式"，把 naive 的"单 channel 挤压"换成"两池并发 + 连续流"。
- **一处已知的过期注释**：`workload_runner.py:1220-1224` 的 docstring still说
  "a master/diff pool has **eight** channels"——那是旧的 8/8 划分遗留文字；**以常量
  `_KV_CHANNELS`（15/1）为准**（R5/audit-3a 裁决 2026-08-25）。核查时相信代码常量，
  别信这条 prose。

### 你可以自己在 JSON 里验证

- 打开任一 `dag_A4.json`，看每次 decode 的读事件设备名：应出现 **`PIM:pool0-14`**
  与 **`PIM:pool15-15`** 两组并存（master + diff 分池的直接证据）。
- 对比 `dag_A3.json`：应是 16 个**单 channel** 池、无 master/diff 分组。
- 布局自述字段：TLB 的 `report()` 里 A4 的 `channel_sets` 是
  `{"master": [0..14], "diff": [15]}`，A3 是 `{"naive": [0..15]}`。

---

## 5. 术语表（给不熟本项目的读者）

| 术语 | 含义 |
|---|---|
| **rung / 档** | A1…A6（含 A3a）这个阶梯上的一级；相邻两级只差一个设计决策。 |
| **makespan** | 整个 workload 的调度长度（schedule length），报告字段 `makespan_s`。 |
| **tier** | workload DAG 里的一层并发单元（同 tier 的 agent 可并发）；`RESULTS` 的"分 tier"表就是逐 tier 讲谁在干什么。报告字段 `summary.tiers` / `workload.tiers`。 |
| **layer** | 一个 workload 内部的"编排轮次/链位"（如 debate 的每一辩论轮、pipeline 的每个链位）；`RESULTS` 的 per-layer 延迟表按它拆。 |
| **k** | reuse policy 对每个 shifted chunk 重算的 token 数（run 时 `EPIC_K`）；k2/k8/k32 三份就是 k=2/8/32。 |
| **KV cache** | 每个历史 token 存下的一条 K + 一条 V 向量。 |
| **PIM / bank PE** | processing-in-memory；HBM bank 旁的小算力单元，decode attention 在这里算。 |
| **row conflict** | 一起读的行落在同一 channel 上被迫串行；第 4 节的核查对象。 |
| **master / diff 池** | master = 不变的共享行（ch0–14）；diff = 重算的修正行（ch15）；A4 起分池并发。 |
| **MQ 命令** | multi-query batch command：一次列读服务多条驻留查询；A5/A6 才开。 |
| **`prefill_attention_sides`** | `dag_A*.json` 字段，qID→`gpu`/`pim`/`dynamic`，A6 的选边结果看这里。 |
| **link_bytes** | 经 NVLink/PCIe 的 GPU↔远端存储字节；A2 因 KV 过链路而很大。 |

---

## 6. 一页速查：这个文件夹里有什么

| 文件 | 是什么 |
|---|---|
| `RESULTS_k2.md` / `k8.md` / `k32.md` | **已跑好的结果**（5 手调负载 × 7 档，k=2/8/32），本文第 1 节 |
| `make_results_tables.py` | 生成上面三份的脚本（读 `output/<run>/dag_A*.json`） |
| `RESULTS_sweep.md` | 参数化 sweep 的结果（98 run），见 `docs/README_run_sweep_guide.md` |
| `make_sweep_tables.py` | 生成 `RESULTS_sweep.md` 的脚本 |
| `md2pdf.py` | 把 MD 转 PDF 的小工具 |
| `README.md` | **本文** |

**想深入**：阶梯权威定义 `src/ablation.py`（`PRESETS`）；DAG 引擎布局
`src/workload_runner.py`（`CacheBlendTLB` / `NaiveKVLayout` / `NaiveMaskKVLayout`）；
每档单独文档 `docs/intro/README_A1.md … README_A6.md` + `README_A3a.md`；原始数据本地
清单 `docs/RAW_DATA_MANIFEST.md`（`dag_A*.json` 不入 git、只在本机）。

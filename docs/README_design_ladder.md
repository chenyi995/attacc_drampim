# 设计阶梯：A1 → A2 → A4c → A4d → A5 → A6（2026-09-04 定）

目标读者：有计算机背景、不熟悉本仓库的人。每一档只比上一档多**一件事**，本页把那件事说到
**哪个函数、哪个参数、归约（reduction）里哪一项变了**。

术语（首次出现即解释）：
- **prefill / decode**：LLM 推理的两个阶段。prefill 把整段输入一次算完、写出 KV 缓存；
  decode 逐词生成，每个新词都要把历史 KV 从头读一遍做注意力（attention）。
- **KV 缓存（KV cache）**：每个历史 token 存下的一条 K 向量、一条 V 向量。
- **PIM**（processing-in-memory）：把 decode 注意力搬进 HBM 每个 bank 旁的小计算单元。
  本仓库基于 AttAcc（ASPLOS'24）的开源仿真器。
- **chunk**：256 个 token 的 KV 块。在 Ramulator 的地址空间里 1 token = 4 B
  （`MAC_AB` 是 all-bank 广播，一个地址点名 16 个分区），所以 **1 个 chunk = 1 个 1024 B 的
  DRAM row**。这条几何贯穿全页，已对照 `ramulator2/trace_gen/gen_trace_attacc_bank.py`
  验证（`HBM_GS['col']=32 B`，`HBM_GS['row']=1024 B`，`score_mac` 每 16 token 走 2 列）。
- **ACT**（activation）：打开一个 DRAM row 的命令，tRC 量级；扫描代价主要由 ACT 数决定。
- **复用（reuse）与修正（repair / diff）**：多个 agent 共享同一段 KV（相同 system prompt、
  共享文档 chunk、上一轮输出）。被复用的 chunk 叫 **master**；因为前置上下文变了，
  消费者要重算其中 k 个 token（本仓库 k=8），重算出来的行叫 **diff**。
  **修正是 per-agent 的**：agent 1 的 diff 是它自己的，agent 2 的是它自己的，不共享。
- **channel**：一个 HBM 堆栈 16 条独立通道。`heads_per_hbm` 个 KV head 共用这 16 条。
  一次扫描的时间 = **最忙那条通道**（各通道并行，见 §7）。
- **档（rung）**：阶梯上的一级。preset 在 `src/ablation.py` 的 `PRESETS`。

---

## 0. 一览

| 档 | 角色 | 相对上一档**唯一**的变化 | `kv_mapping` / `channel_placement` | prefill attn | 批命令 |
|---|---|---|---|---|---|
| **A1** | 硬件 baseline（AttAcc 原样） | — | `private` / slice | GPU | replicate |
| **A2** | 软件 baseline（复用，但没有 PIM） | decode 从 PIM 搬到 GPU；KV 放远端哑存储，每步经链路流回 | `none` | GPU | replicate |
| **A4c** | 布局设计 1 | decode 回到 PIM；master 按 head 切片铺满 16 条；**每个 head 的 diff 聚到它自己通道上的几行** | `master-diff-local` / slice | GPU | replicate |
| **A4d** | 布局设计 2 | **各 head 的 diff 合并成一段**，放 ch15（溢出到 ch14…）；master 不变 | `master-diff-merged` / slice | GPU | replicate |
| **A4e** | 布局设计 3（论文的 placement table） | diff 回到 A4c 的 per-head 行；**master chunk 的通道由写入时的冲突感知表决定**，不再是 naive 的写入序轮转 | `master-diff-table-local` / slice | GPU | replicate |
| **A5** | prefill 加速 | prefill 注意力搬到 PIM；批命令 replicate → **MQ** | 同 A4e | **PIM** | **mq** |
| **A6** | 最终设计（Fugue） | prefill 逐请求**动态选边** GPU / PIM | 同 A4e | **dynamic** | mq |

被淘汰的中间档 A3 / A3a / A3b / A4 / A4b 仍在代码里（消融要用），为什么淘汰见 §9。

---

## 1. A1 — 硬件 baseline：AttAcc 原样，无复用

| 项 | 值 |
|---|---|
| 复用 | 无：`--reuse no-reuse`，每个请求的 KV 私有、完整重算 |
| KV 布局 | `NoReuseKVLayout`（`src/workload_runner.py`）：一个请求一层一段连续 extent，横跨 16 条通道 |
| 放置 policy | `slice`，走旧的 chunk 计数模型 `_layout_channel_loads` |
| prefill attention | GPU |
| decode attention | PIM |

这是 AttAcc 论文的参照点：PIM 做 decode，但一个字节都不复用。

## 2. A2 — 软件 baseline：复用有了，PIM 没了

**相对 A1 的变化**：`decode_attn: pim → gpu`，`kv_mapping: private → none`。

| 项 | 值 |
|---|---|
| 复用 | 有：`--reuse recompute --epic-prefix-recompute-tokens 8`，每个位移的复用段重算 k=8 个 token |
| KV 存哪 | **远端哑存储**（PIM-HBM 或普通 DRAM，无算力）。KV 缓存整体在远端 |
| 每一步做什么 | decode 每步把整层上下文 KV 经链路流回 GPU（`kv_remote_to_gpu`），GPU 算注意力 |
| 代码路径 | `run_cacheblend_dag` 在 `decode_attn == "gpu"` 时分派到 `_run_gpu_software_only` —— **完全另一条事件构建路径，没有任何 PIM 事件** |

实测 `energy_breakdown_nj.by_class` 只有 `GPU` 和 `LINK` 两项（`link_bytes` 是 A1 的 32 倍）。
这一档没有布局可言，它是"软件复用能省多少计算、但被链路吃掉多少"的对照。

## 3. A4c — 布局设计 1：decode 回 PIM，diff 按 head 局部聚集

**相对 A2 的变化**：decode 回到 PIM（`decode_attn: gpu → pim`），KV 回到 PIM-HBM，
`kv_mapping = master-diff-local`。这一步跨过了三个被淘汰的中间档（A3b → A4 → A4b，见 §9），
A4c 是它们的教训收敛出来的布局。

### 3.1 布局规则（`_striped_append_channel_extents` 的 `master-diff-local-append` 分支）

```
stripe = 16 // heads_per_hbm                 # 每个 head 独占的通道数
for head h:
    base = h × stripe
    master：每个 chunk 在**写入时**定一个 slot = 该 head 条带里第几条；naive 存储的规则是
            写入序轮转 slot = i % stripe（第 i 个写进这条带的 chunk）。持久，之后每次扫描都落同一条
    diff  ：该 head 全部修正合成一段，落在 (base + stripe − 1)        # 自己通道中的最后一条，多几行
```

> 2026-09-04 深夜之前 master 是"每次扫描按读取顺序轮转"（`u % stripe`）—— 同一个 chunk 在不同扫描里
> 会换通道，不物理，且天然完美均衡。现已改为持久的写入序轮转（`_chunk_slot_table(mode="append")`），
> 这才是 naive 存储的真实行为：两个一起读的 chunk 可能碰巧错开，也可能碰巧撞上（写入序差正好 = stripe）。

- **master 一条通道都不让出去**。这是 A4c 与 A4/A4b 的本质区别（A4/A4b 把 ch15 从
  master 池里拿走给全局 diff，所有扫描都为此付 1/16 的带宽，哪怕它一个修正都不读）。
- **一个 head 的 diff 是一段**。分配器 `LocalDiffKVLayout`（继承 `CacheBlendTLB`）给修正
  单独的游标，所以一个 head **跨轮**产生的修正在物理上连续 —— 这正是 A3b 买不到的：A3b 把
  修正就地写在 master 流里，两轮之间隔着这一轮自己写的 KV，每轮的修正各占一行。

### 3.2 手算样例（`heads_per_hbm=4`，8 个 chunk，8 轮各修 1 个）

`output/analysis/layout_A4c_R8.csv`（`layout_grid_csv.py` 生成，第一列 = 物理行号，之后 16 列 = ch0…15）：

| dram_row | ch0 | ch1 | ch2 | **ch3** |
|---|---|---|---|---|
| 0 | h0-chunk0 | h0-chunk1 | h0-chunk2 | h0-chunk3 |
| 1 | h0-chunk4 | h0-chunk5 | h0-chunk6 | h0-chunk7 |
| 2 | — | — | — | **h0 全部 8 轮修正 64/256** |

| | A3b | A4b | **A4c** |
|---|---:|---:|---:|
| 用掉的行 = ACT | 64 | 33 | 36 |
| 最忙通道 | 4 行 | 3 行 | **3 行** |
| 让出的通道 | 0 | **1（全局）** | **0** |

A4c 拿到 A4b 的最忙通道数，不付那条全局通道。

## 4. A4d — 布局设计 2：各 head 的 diff 合并

**相对 A4c 的唯一变化**：diff 的落点。`kv_mapping = master-diff-merged`。

```
master：完全同 A4c（= A3b 的切片，铺满 16 条）
diff  ：heads_per_hbm 个 head 的修正合成一段 = heads × Σrepairs 个 token
        按 256 token 切成 unit，第 i 个 unit 落 ch(15 − i)         # 一条放得下就一条，放不下溢到 ch14、ch13…
```

分配器 `MergedDiffKVLayout`，**只有 `_kv_mapping` 不同**（决定 `layout_policy`）。

**买到的东西**：A4c 里 4 个 head 各一段修正，各占一行、各在自己通道上 —— 4 次 ACT；A4d 合成
一段，4 × 64 = 256 token **正好一行** —— 1 次 ACT。修正越大（k 越大）差距越明显：

| k（`heads=4`，16 chunk）| A3b 最忙/总 ACT | A4b | A4c | **A4d** |
|---:|---|---|---|---|
| 8 | 8 / 128 | 16 / 80 | 5 / 68 | **5 / 66** |
| 64 | 8 / 128 | 16 / 80 | 8 / 80 | **5 / 80** |
| 128 | 8 / 128 | 32 / 96 | 12 / 96 | **6 / 96** |

A4b 同样做了 head 合并，却最差（16–32）——因为它把合并后的 diff 压在一条**被移出 master 池**的通道上，
两头亏；A4d 把它摊在**仍在 master 池里**的几条通道上。

## 4b. A4e — 布局设计 3：论文的 Conflict-Aware Channel Placement

**相对 A4c 的唯一变化**：master chunk 的 slot 不再由写入序决定，而由**冲突感知表**决定。diff 与 A4c 相同
（per-head 一行，不跨 head gather，不留专用 diff 通道 —— 这两处是对论文 §4 的有意删减）。
`kv_mapping = master-diff-table-local`，分配器 `TableLocalDiffKVLayout`。

规则（论文 §4 "Conflict-Aware Channel Placement"，`_chunk_slot_table(mode="table")`）：

```
以 shared chunk 为放置单位；写入 chunk c 时驱动已知哪些 sweep 会读 c、以及那些 sweep 还读哪些 chunk
（每次 prefill / decode 都点名自己的 chunk）。
把 c 放到"那些一起被读的、已放置的 chunk"没占的 slot 上；都占了就放它们占得最少的那个。
软件表记录 (channel, row)；每次 sweep 按表发扫描。表跨扫描持久。
```

手算（stripe = 4）：

| 写入序 doc1…doc6，某 sweep 读 {doc1, doc5} | doc1 | doc5 | |
|---|---:|---:|---|
| naive 写入序轮转（A4c） | slot 0 | slot 0 | **撞**（写入序差 4 = stripe）|
| **表（A4e）** | slot 0 | slot 1 | 错开 |

五个 sweep 的 co-read 集合 `{1,5} {2,6} {1,2,3,4} {3,7} {4,8}`：naive 下每个 sweep 内同 slot 最多 2 个 chunk，表下最多 1 个。
钉在 `ConflictAwareSlotTableTest`。A4e 相对 A4c 的收益 = 消掉 naive 轮转"碰巧撞上"的那部分。

## 5. A5 — prefill 加速：prefill 注意力进 PIM + MQ 批命令

**相对 A4e 的变化**（布局逐字节相同）：

| 参数 | A4e | A5 | 在哪生效 |
|---|---|---|---|
| `prefill_attn` | gpu | **pim** | `pim_prefill_mode="pim"` → `prefill_side = "pim"`（`_run_cacheblend_prefill`）：prefill 注意力变成 PIM 扫描事件 `pim_kv_scan_score_softmax_pv`，不再是 GPU 的 `gpu_prefill_score/softmax/context` |
| `pim_batch_command` | replicate | **mq** | `_apply_pim_batch`：`op.pim_batch_command`、`op.pim_pe_freq_ghz`；Ramulator 对**同一批 extent** 用不同命令定价。MQ = 一条 `MAC_AB` 服务全部驻留 query |
| `pim_pe_freq_ghz` | — | 1.3004 | 同上（PC 能量钳位下 n=8 的平衡点频率） |
| `gemv_buffer_bytes` | — | 512 | 驻留 8 个 query；决定 prefill 的 sweep 次数 |

**2026-09-04 之前 A5/A6 建在 A4b 的布局上**（`master-diff / table`）。A4b 在所有实测 workload 上都输给
A3b（§8），让 prefill 加速继承一个布局退步没有意义，所以 A5/A6 改建在最终布局 **A4e** 上。
这是本页定阶梯时做的改动，A5/A6 的数字随之全变。

## 6. A6 — 最终设计（Fugue）：prefill 逐请求动态选边

**相对 A5 的唯一变化**：`prefill_attn: pim → dynamic`。

选边器（`src/workload_runner.py`，`_run_cacheblend_prefill` 里 `pim_prefill_mode == "dynamic"` 分支）
对**每个请求**判一次、跨层稳定：

```
t_xpu  = 驻留行读回链路 + GPU 全上下文注意力（score / softmax / context）
t_bank = 最忙通道的 PIM 扫描（用与提交扫描完全相同的 extent 定价）× sweep 次数
         + TLB plan + Q/ctx 链路
prefill_side = "pim" if t_bank <= t_xpu else "gpu"
```

被判给 GPU 的请求，其 prefill 的 PIM lane 组不存在；判给 PIM 的与 A5 相同。
`A6 ≤ min(A4e, A5)` 应恒成立；严格小于时说明选边器真的做了混合。

---

## 7. 两个归约在哪：latency 取 max，energy 取 sum

| 归约 | 位置 | 做法 |
|---|---|---|
| **latency 取 max** | `_schedule_cacheblend`（`src/workload_runner.py`），`pipe=True`（`--pipeopt`，默认）| 一次扫描的每条活跃通道是一个独立事件 `PIM:pool{c}-{c}`，各自占自己的资源时间轴（`resource = event.device`）；**紧随其后的 DIE 合并事件依赖每一条 lane**，所以扫描对下游生效的时刻 = 最慢 lane 的结束 = **最忙通道**。与 serial 模式的区别：空下来的通道可以立刻接**独立**的事件（同一步的公共段/私有段、其他 batch 组），而不是等最慢 lane。serial 模式（`--no-pipeopt`）把所有事件放一条 `SERIAL` 轴、`availability["SERIAL"] = max(lane.end)`，2026-09-04 起不再是出数口径 |
| **energy 取 sum** | `_finalize_cacheblend_report`：`sum(event.energy_nj for event in scheduled)` | 每条 lane 的 `energy_nj` = `sum(energy_vec)/1000 × num_hbm_used`（`_cacheblend_event`）；报告把所有事件相加 |

各档在这两个归约里动的是哪一项：

| 转换 | max 里变的 | sum 里变的 |
|---|---|---|
| A1 → A2 | PIM lane 组整个消失；时间轴变 GPU + LINK 串行 | PIM 项消失，`kv_remote_to_gpu` 链路项暴涨 |
| A2 → A4c | 候选集变成 16 条通道的 lane 时间 | 项数 = 活跃通道数；每项能量随载荷 |
| A4c → A4d | **只有 diff 所在的 1–2 条通道的 lane 时间变**（其余 master 通道逐字节相同） | diff 那几条的能量 |
| A4c → A4e | master 各 chunk 的 slot 换了：naive 轮转碰巧撞上的 co-read chunk 被错开，**最忙通道的 unit 数变** | 总量不变，只是搬家 |
| A4e → A5 | 放置不变；**每条 lane 的 `time_s`** 变（同一批 extent、MQ 定价）；prefill 从 GPU 事件变 PIM 事件 | 每项能量降（MQ）；`by_class` 归属从 GPU 搬到 PIM |
| A5 → A6 | 按请求：判给 GPU 的没有 PIM lane 组 | 按请求在 GPU 项与 PIM 项之间搬 |

---

## 8. 实测证据（`CACHEBLEND-TINY`，`--num-hbm 2 --ngpu 1`，`heads_per_hbm=4`，k=8，**`--pipeopt`**）

三个人造的多轮 workload（`workload/handcheck/gen_multiround.py`，消费者上下文
`sys | shared | own | shared | own | …`，每轮取一个块修一次、写一块自己的 KV）。
**全部过 Ramulator，`--engine dag --pipeopt`（2026-09-04 起的默认）。** 括号 = A3b / 该档，>1 为更快。
A4c / A4d 的 master 已是持久的写入序轮转（§3.1）；A4e 是冲突感知表（§4b）。

### 8.1 端到端 makespan

| workload | A3b | A4 | A4b | **A4c** | A4d | **A4e** |
|---|---:|---:|---:|---:|---:|---:|
| 2 轮 × 每轮修 8 块，3 agent | 0.082330 | 0.9719× | 0.9838× | **1.0080×** | 1.0036× | 1.0066× |
| 16 轮 × 每轮修 1 块，3 agent | 0.190656 | 0.9867× | 0.9962× | **1.0058×** | 1.0036× | 1.0035× |
| 16 轮 × 每轮修 1 块，8 agent | 0.505671 | 0.9867× | 0.9974× | 1.0041× | 1.0020× | **1.0040×** |

### 8.2 第 0 层 PIM 扫描时间（布局真正作用的那一层）

| workload | A3b | A4 | A4b | A4c | A4d | **A4e** |
|---|---:|---:|---:|---:|---:|---:|
| 16 轮，3 agent | 0.002313 | 0.7767× | 0.9203× | **1.0672×** | 1.0233× | **1.0672×** |
| 16 轮，8 agent | 0.006941 | 0.8034× | 0.9555× | **0.9788×** | 0.9434× | **1.0793×** |

**8 agent 那行是 A4e 的存在理由**：3 agent 时写入序轮转碰巧不撞，A4c = A4e；8 agent 时
轮转开始撞（A4c 扫描层跌到 0.9788×，比 A3b 还慢），表把它救回到 **1.0793×**。这就是 §4b
"naive 可能碰巧赚到、也可能碰巧撞上；表消掉运气"的实测。

### 8.3 为什么 A4b 输、A4c/A4e 赢 —— 按扫描类型拆（16 轮，3 agent，serial 模式实测，秒）

批解码的扫描被引擎拆成两段：**公共段**（几个 agent 共读的 shared chunk，5120 token，
按定义不带修正——修正是 per-agent 的进不去）和**私有段**（各自的行 + 自己的修正）。

| 档 | 公共段（无修正）| 私有段（带修正）| 合计 |
|---|---:|---:|---:|
| A3b | 0.001063 | 0.001158 | 0.002313 |
| A4b | **0.001409（0.75×）** | 0.001012 | 0.002513 |
| **A4c** | **0.001063（=A3b）** | **0.001012** | **0.002167** |

A4b 在私有段赢了（gather 兑现），却在公共段输更多——那些扫描一个修正都不读，
ch15 全空，它白白少一条 master 通道。A4c 公共段与 A3b 逐字节相同，私有段拿到同样的收益。

### 8.4 端到端幅度为什么只有 0.4–0.8%

GPU 占 makespan 92–94%，其中 **prefill attention（$L^2$，A1–A4e 都在 GPU）占 GPU 时间 88.7%**
（8 agent 实测）。布局只能动 PIM 那 6–8%。另外 decode 每步每层有 22.03 us 的固定 kernel 开销
（`devices.py` 的 `+8.29`/`+6.87` 常数，**AttAcc 原版 `c1540de` 就有**，A100 单 kernel 实测拟合），
小 batch 下不被摊薄；加大并发到 8 agent 把它从 GPU 的 12.5% 摊到 5.6%，但 prefill 跟着涨。
`--pipeopt` 相对 serial 只再省 0.7–1.0%：decode 连续步骤链式依赖，空通道只能接**独立**的扫描。
**布局消融应看 §8.2 的扫描层面数**；makespan 被 GPU prefill 主导是配置的性质，不是布局的性质。

### 8.5 证据等级

- **实测**：§8.1–8.3 全部（Ramulator、`--engine dag`，8.1–8.2 为 pipeopt，8.3 为 serial 模式的逐段拆解）；§3.2、§4 的 ACT 表由放置规则算出，
  该规则已在 2000 条真实扫描上与引擎逐通道核对（`output/analysis/layout_handcheck_theory.py`）。
- **未测**：A5 / A6 在 A4d 布局上的数（今天刚改建，还没跑）；真实模型（LLAMA3-8B / GPT-13B）
  与 baseline sweep workload 上的 A4c / A4d；`heads_per_hbm ≥ 16`（多个 head 挤一条通道）。
- **人造 workload**：§8 的三个 workload 是为了暴露多轮结构造的，`kind` 标 NOT evidence-grade。
  baseline sweep 的 workload 里消费者的修正是一次 prefill 连着写完的（`.R……RRRRRRRR`），
  没有跨轮隔离，所以在它上面 A4c/A4d 相对 A3b 的收益会小得多（同分配器对照：1.1×）。

---

## 9. 被淘汰的中间档，以及为什么

| 档 | 是什么 | 为什么不在设计阶梯里 |
|---|---|---|
| A3 | head → 1 条通道，其余闲 | 朴素下界，只作消融 |
| A3a | A3 + 掩码门（陈旧行随流读出被掩，run 不切断） | 只作消融 |
| A3b | A3 + head 切片（铺满自己的 16/heads 条） | **master 布局被 A4c 原样继承**；输在修正跨轮各占一行 |
| A4 | A3b + 全局 diff 池 ch15，master 池 15 条按 head 切片 | `stripe_m = 15 // heads` 在 `num_hbm=1` 上钳到 1，退化；且全局让出一条通道 |
| A4b | A4 + 全局放置表 | 修回了 A4 的退化，但仍全局让出 ch15，**在所有实测 workload 上输给 A3b**（§8） |

A4/A4b 的错不是"分池"这个想法，是分池的**粒度**：把 diff 提到全局一条通道，代价（少一条 master）
落在所有扫描上、收益只落在带修正的扫描上。A4c/A4d 把它做到 head 局部，代价只是那条通道多几行。

## 10. 今天同时修掉的引擎问题（不修它们，§8 的数出不来）

| 问题 | 位置 | 修法 |
|---|---|---|
| `sorted()` 分配 = 全知排序，抹掉跨 agent 交错 | `CacheBlendTLB.finalize`、`NoReuseKVLayout.finalize` | 改插入序（append 序），与 `NaiveKVLayout` 一致 |
| master / diff 段各自 `sum` 后重新打包，分池收益被抹平 | `_striped_append_channel_extents` | extent 由分配器的真实邻接决定：`tlb.scan_runs` 驱动，逐段送进 `_channel_extent_addresses` |
| `_pool_reads` 一个列表供三处用 | `_pool_reads` | 拆成 `reads`（DRAM 流过的）/ `masked`（die 丢掉的）/ `plan_reads`（TLB 描述符按它切）|
| A4d 曾静默别名成 A4c（TLB 类的 `_kv_mapping` 决定 policy） | `MergedDiffKVLayout` | 单独的类；`PresetRoutesToItsOwnPolicyTest` 钉住 preset → TLB 类 → policy 三跳一致 |

来龙去脉见 `sessions/2026-09-04.md`。**这些改动让所有档的绝对值全变**，2026-09-03 及之前的
结果页已全部归档，不要引用。

## 11. 复现

`--pipeopt` 默认 ON（2026-09-04 起），`--no-pipeopt` 才是 serial 模式。

```bash
# 五档对照（人造多轮 workload，小模型，每档约 10 分钟）
ROUNDS=16 CHUNKS=1 CONSUMERS=2 LOUT=256 python3 workload/handcheck/gen_multiround.py > workload/handcheck/wl_mr_R16.json
export PYTHONPATH=$PWD KVPIM_CPPCORE=1 ATTACC_RAMULATOR_DIR=/data2/chenyi9/kvpim_run_scratch/x
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY --workload workload/handcheck/wl_mr_R16.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 --ablation A4c --engine dag \
  --workload-report out.json --workload-report-events none --cacheblend-batch-size 8 --num-hbm 2 --ngpu 1
# 布局手算 CSV
PYTHONPATH=$PWD python3 output/analysis/layout_grid_csv.py --chunks 8 --rounds 8 --tag _R8
# 逐通道探针（KVPIM_LAYOUT_DUMP=<file>）与手算对账
PYTHONPATH=$PWD python3 output/analysis/layout_handcheck_theory.py <dump_dir>
```

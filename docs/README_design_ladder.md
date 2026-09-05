# Fugue 设计阶梯：A1 → A2 → A3b → A4c → A4e → A5 → A6

目标读者：有计算机背景、不熟悉本仓库的人。每一档只比上一档多**一件事**，每件事都是论文
（`KVPIM-1Fugue-ASPLOS2027`）的一条 claim；本页把那件事说到**哪个函数、哪个参数、归约里哪一项变了**。

本分支（`chenyi-0905`）的根基是 **AttAcc 原版**（上游 `c600051`，jwchoi，2024-06-24）；
之上只有 Fugue 的改动。不属于论文 claim 的消融档（A3、A3a、A4、A4b、A4d）不在本分支。

术语（首次出现即解释）：
- **prefill / decode**：LLM 推理的两个阶段。prefill 把整段输入一次算完、写出 KV 缓存；
  decode 逐词生成，每个新词都要把历史 KV 从头读一遍做注意力（attention）。
- **KV 缓存（KV cache）**：每个历史 token 存下的一条 K 向量、一条 V 向量。
- **PIM**（processing-in-memory）：把 decode 注意力搬进 HBM 每个 bank 旁的小计算单元（AttAcc，ASPLOS'24）。
- **chunk**：256 个 token 的 KV 块。Ramulator 的地址空间里 1 token = 4 B（`MAC_AB` 是 all-bank 广播，
  一个地址点名 16 个分区），所以 **1 个 chunk = 1 个 1024 B 的 DRAM row**。已对照
  `pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py` 验证（`HBM_GS['col']=32 B`，`HBM_GS['row']=1024 B`）。
- **ACT**（activation）：打开一个 DRAM row 的命令，tRC 量级；扫描代价主要由 ACT 数决定。
- **复用（reuse）与修正（repair / diff）**：多个 agent 共享同一段 KV。被复用的 chunk 叫 **master**；
  因为前置上下文变了，消费者要重算其中 k 个 token（k=8），重算出来的行叫 **diff**。
  **修正是 per-agent 的**，agent 之间不共享。
- **channel**：一个 HBM 堆栈 16 条独立通道；`heads_per_hbm` 个 KV head 共用这 16 条，每个 head 独占
  `stripe = 16 // heads_per_hbm` 条。一次扫描的时间 = **最忙那条通道**（§7）。
- **档（rung）**：阶梯上的一级。preset 在 `src/ablation.py` 的 `PRESETS`。

---

## 0. 一览

| 档 | 论文里的角色 | 相对上一档**唯一**的变化 | `kv_mapping` / `channel_placement` | prefill attn | 批命令 |
|---|---|---|---|---|---|
| **A1** | 硬件 baseline（AttAcc 原样） | — | `private` / slice | GPU | replicate |
| **A2** | 软件 baseline（复用，但没有 PIM） | decode 搬到 GPU；KV 放远端哑存储，每步经链路流回 | `none` | GPU | replicate |
| **A3b** | 朴素 PIM 存储 | decode 回到 PIM；一个 head 的 chunk 与修正**混在一条 append 流里**，按写入序轮转到该 head 的通道上 | `naive` / slice | GPU | replicate |
| **A4c** | claim 1：per-agent diff | **该 head 的修正聚到它自己通道上的几行**，与 master 分开；master 一条通道都不让 | `master-diff-local` / slice | GPU | replicate |
| **A4e** | claim 2：placement table | **master chunk 的通道由写入时的冲突感知表决定**，不再是写入序轮转 | `master-diff-table-local` / slice | GPU | replicate |
| **A5** | claim 3：prefill 进 PIM | prefill 注意力搬到 PIM；批命令 replicate → **MQ** | 同 A4e | **PIM** | **mq** |
| **A6** | Fugue | prefill 逐请求**动态选边** GPU / PIM | 同 A4e | **dynamic** | mq |

---

## 1. A1 — 硬件 baseline：AttAcc 原样，无复用

| 项 | 值 |
|---|---|
| 复用 | 无：`--reuse no-reuse`，每个请求的 KV 私有、完整重算 |
| KV 布局 | `NoReuseKVLayout`（`src/workload_runner.py`）：一个请求一层一段连续 extent，横跨 16 条通道 |
| 放置 policy | `slice`，AttAcc 的 chunk 计数模型 `_layout_channel_loads` |
| prefill / decode attention | GPU / PIM。prefill 走 `_append_gpu_prefill_layer`：GPU 算 QKV 与整段 attention，K/V 经链路落到 PIM 供 decode；多轮时驻留历史先经 `kv_pim_to_gpu` 回读（2026-09-05 晚修复 F01；此前 A1 的 prefill 实际在 PIM 里做连续扫描，只是记账记成了 GPU） |

## 2. A2 — 软件 baseline：复用有了，PIM 没了

**相对 A1**：`decode_attn: pim → gpu`，`kv_mapping: private → none`。

| 项 | 值 |
|---|---|
| 复用 | `--reuse recompute --epic-prefix-recompute-tokens 8`：每个位移的复用段重算 k=8 个 token |
| KV 存哪 | 远端哑存储（无算力）。decode 每步把整层上下文 KV 经链路流回 GPU（`kv_remote_to_gpu`） |
| 代码路径 | `run_cacheblend_dag` 在 `decode_attn == "gpu"` 时分派到 `_run_gpu_software_only` —— **没有任何 PIM 事件** |

实测 `energy_breakdown_nj.by_class` 只有 `GPU` 和 `LINK`；`link_bytes` 是 A1 的 32 倍。

## 3. A3b — 朴素 PIM 存储：decode 回 PIM，一条 append 流

**相对 A2**：decode 回到 PIM，KV 回到 PIM-HBM，`kv_mapping = naive`（`NaiveKVLayout`），policy `slice-append`。

```
stripe = 16 // heads_per_hbm
for head h:  base = h × stripe
    master 与 diff 在同一条流里；每个对象的 slot 由写入序表决定并持久：
        chunk  -> _chunk_slot_table(mode="append")，与 A4c 同一张表（第 i 个写入的 chunk 落 slot i % stripe）
        修正   -> 首次被扫描到时追加到同一轮转的下一个 slot
    落 (base + slot) % 16；同一 chunk 在任何 scan 里都在同一条通道（2026-09-05 晚修复 F02；此前按本次 scan 的 unit 序号重新轮转）
    一轮里一次产生的修正是一段连续 append（共享行）；不同轮的修正被这一轮自己写的 KV 隔开 → 各占一行
```

这就是论文 §3 "Problem 2" 的场景：一个 agent 的上下文由散落各处的部分拼成，扫描行行跳，
每跳一次 ACT；一起读的 chunk 按写入序落到同一条通道时串行。

## 4. A4c — claim 1：per-agent diff 行（论文 §4 "Master and Per-Agent Diff"）

**相对 A3b 的唯一变化**：diff 从 append 流里拿出来，聚到该 head 自己通道上的几行。
`kv_mapping = master-diff-local`，分配器 `LocalDiffKVLayout`（修正有独立的分配游标）。

```
master：同 A3b（写入序轮转，持久，_chunk_slot_table(mode="append")）
diff  ：该 head 全部修正合成一段，落在 (base + stripe − 1)   # 自己通道中的一条，多几行
```

- **master 一条通道都不让出去**。论文早先的设计给 diff 划专用通道（ρ_b × 16 条），那让**所有**扫描
  为一条空转的 diff 通道付 1/16 的带宽 —— 而共读 shared chunk 的公共段扫描（流量大头）按定义不带修正。
  实测这一项让全局 diff 通道的方案在每个 workload 上都输给 A3b（§8.3），A4c 把它消掉。
- **一个 head 的 diff 是一段**：跨轮产生的修正在物理上连续。A3b 买不到的正是这一点。
- **不跨 head 合并**：不同 head 的 K 不同、Q 也不同，MAC 不能共享；合并只会把各 head 的列读串行到一条通道上。

手算（`heads_per_hbm=4`，8 chunk，8 轮各修 1 个；`output/analysis/layout_grid_csv.py --chunks 8 --rounds 8`）：

| | A3b | **A4c** |
|---|---:|---:|
| 用掉的行 = ACT | 64 | 36 |
| 最忙通道 | 4 行 | **3 行** |

## 5. A4e — claim 2：Conflict-Aware Channel Placement（论文 §4 同名小节）

**相对 A4c 的唯一变化**：master chunk 在 head 条带里的 slot 由**写入时的冲突感知表**决定。
`kv_mapping = master-diff-table-local`，分配器 `TableLocalDiffKVLayout`，规则 `_chunk_slot_table(mode="table")`。

```
以 shared chunk 为放置单位；写入 chunk c 时驱动已知哪些 sweep 会读 c、以及那些 sweep 还读哪些 chunk。
把 c 放到"那些一起被读的、已放置的 chunk"没占的 slot 上；都占了就放它们占得最少的那个。
软件表记录 (channel, row)；每次 sweep 按表发扫描。表跨扫描持久。
```

naive（A4c）是写入序轮转：可能碰巧错开，也可能碰巧撞上 —— 写入序差正好等于 stripe 的两个 chunk 一定撞。

| 写入序 doc1…doc6，某 sweep 读 {doc1, doc5}，stripe=4 | doc1 | doc5 | |
|---|---:|---:|---|
| 写入序轮转（A4c） | slot 0 | slot 0 | **撞** |
| **表（A4e）** | slot 0 | slot 1 | 错开 |

钉在 `tests/test_placement.py::ConflictAwareSlotTableTest`。

## 6. A5 — claim 3：prefill 注意力进 PIM + MQ 批命令（论文 §4 "MQ"，§5）

**相对 A4e**（布局逐字节相同）：

| 参数 | A4e | A5 | 在哪生效 |
|---|---|---|---|
| `prefill_attn` | gpu | **pim** | `pim_prefill_mode="pim"` → prefill 注意力变成 PIM 扫描事件 `pim_kv_scan_score_softmax_pv` |
| `pim_batch_command` | replicate | **mq** | `_apply_pim_batch`：一条 `MAC_AB` 服务全部驻留 query；Ramulator 对同一批 extent 用不同命令定价 |
| `pim_pe_freq_ghz` | — | 1.3004 | PC 能量钳位下 n=8 的平衡点 |
| `gemv_buffer_bytes` | — | 512 | 驻留 8 个 query |

## 7. A6 — Fugue：prefill 逐请求动态选边（论文 §5）

**相对 A5 的唯一变化**：`prefill_attn: pim → dynamic`。选边器对每个请求判一次、跨层稳定：

```
t_xpu  = 驻留行读回链路 + GPU 全上下文注意力
t_bank = 最忙通道的 PIM 扫描（与提交扫描相同的 extent 定价）× sweep 次数 + TLB plan + Q/ctx 链路
prefill_side = "pim" if t_bank <= t_xpu else "gpu"
```

`A6 ≤ min(A4e, A5)` 应恒成立；严格小于时说明选边器真的做了混合。

---

## 8. 两个归约在哪：latency 取 max，energy 取 sum

| 归约 | 位置 | 做法 |
|---|---|---|
| **latency 取 max** | `_schedule_cacheblend`，`--pipeopt`（默认） | 一次扫描的每条活跃通道是独立事件 `PIM:pool{c}-{c}`，各占自己的资源时间轴；紧随其后的 DIE 合并依赖每一条 lane，所以扫描对下游生效的时刻 = **最忙通道**。空下来的通道可以立刻接独立的事件 |
| **energy 取 sum** | `_finalize_cacheblend_report`：`sum(event.energy_nj)` | 每条 lane 的能量 = `sum(energy_vec)/1000 × num_hbm_used` |

| 转换 | max 里变的 | sum 里变的 |
|---|---|---|
| A1 → A2 | PIM lane 组消失；时间轴变 GPU + LINK | PIM 项消失，`kv_remote_to_gpu` 暴涨 |
| A2 → A3b | 候选集变成 16 条通道的 lane 时间 | 项数 = 活跃通道数 |
| A3b → A4c | 只有 diff 所在的那条通道变；修正跨轮合并成一行 | diff 那条的能量 |
| A4c → A4e | master 各 chunk 的 slot 换了：碰巧撞上的 co-read chunk 被错开，最忙通道的 unit 数变 | 总量不变，只是搬家 |
| A4e → A5 | 放置不变；每条 lane 的 `time_s` 变（MQ 定价）；prefill 从 GPU 事件变 PIM 事件 | 每项能量降；`by_class` 从 GPU 搬到 PIM |
| A5 → A6 | 按请求：判给 GPU 的没有 PIM lane 组 | 按请求在 GPU / PIM 之间搬 |

---

## 9. 实测证据（`CACHEBLEND-TINY`，`--num-hbm 2 --ngpu 1`，`heads_per_hbm=4`，k=8，`--pipeopt`）

三个人造多轮 workload（消费者上下文 `sys | shared | own | shared | own | …`，每轮取一块修一次、写一块自己的 KV）。
**全部过 Ramulator。** 括号 = A3b / 该档，>1 为更快。表里同时列出不在本分支的消融档 A4 / A4b / A4d 作对照
（它们的数来自 `chenyi-0904-test` 分支）。

### 9.1 端到端 makespan

| workload | A3b | A4 | A4b | **A4c** | A4d | **A4e** |
|---|---:|---:|---:|---:|---:|---:|
| 2 轮 × 每轮修 8 块，3 agent | 0.082330 | 0.9719× | 0.9838× | **1.0080×** | 1.0036× | 1.0066× |
| 16 轮 × 每轮修 1 块，3 agent | 0.190656 | 0.9867× | 0.9962× | **1.0058×** | 1.0036× | 1.0035× |
| 16 轮 × 每轮修 1 块，8 agent | 0.505671 | 0.9867× | 0.9974× | 1.0041× | 1.0020× | **1.0040×** |

### 9.2 第 0 层 PIM 扫描时间（布局真正作用的一层）

| workload | A3b | A4 | A4b | A4c | A4d | **A4e** |
|---|---:|---:|---:|---:|---:|---:|
| 16 轮，3 agent | 0.002313 | 0.7767× | 0.9203× | **1.0672×** | 1.0233× | **1.0672×** |
| 16 轮，8 agent | 0.006941 | 0.8034× | 0.9555× | 0.9788× | 0.9434× | **1.0793×** |

**8 agent 那行是 A4e 的存在理由**：3 agent 时写入序轮转碰巧不撞，A4c = A4e；8 agent 时轮转开始撞
（A4c 跌到 0.9788×，比 A3b 还慢），表把它救回到 1.0793×。

### 9.3 全局 diff 通道为什么输（serial 模式的逐段拆解，16 轮 / 3 agent，秒）

| 档 | 公共段（共读 chunk，无修正）| 私有段（带修正）| 合计 |
|---|---:|---:|---:|
| A3b | 0.001063 | 0.001158 | 0.002313 |
| 全局 diff 通道（A4b） | **0.001409（0.75×）** | 0.001012 | 0.002513 |
| **A4c** | **0.001063（=A3b）** | **0.001012** | **0.002167** |

### 9.4 端到端幅度为什么只有 0.4–0.8%

GPU 占 makespan 92–94%，其中 prefill attention（$L^2$，A1–A4e 都在 GPU）占 GPU 时间 88.7%。布局只能动
PIM 那 6–8%。另有 decode 每步每层 22.03 us 的固定 kernel 开销（`src/devices.py` 的 `+8.29` / `+6.87`，
**AttAcc 原版就有**）。**布局消融应看 §9.2**；A5 把 prefill 搬进 PIM 后 PIM 占比才上去。

### 9.5 证据等级

- **实测**：§9.1–9.3；§4–§5 的手算表由放置规则算出，该规则已在 2000 条真实扫描上与引擎逐通道核对。
- **未测**：A5 / A6 在 A4e 布局上；真实模型 + 论文 sweep workload 上的 A4c / A4e；`heads_per_hbm ≥ 16`。
- 三个 workload 是为暴露多轮结构造的（`NOT evidence-grade`），不在本分支（workload 另行迁移）。

## 10. 复现

```bash
bash set_pim_ramulator.sh && (cd ramulator2 && mkdir -p build && cd build && cmake .. && make -j)
(cd src/cppcore && make)                      # gcc-toolset-11
export PYTHONPATH=$PWD KVPIM_CPPCORE=1 ATTACC_RAMULATOR_DIR=<scratch on /data2>
python3 main.py --system dgx-attacc --model LLAMA3-8B --workload <wl.json> \
  --reuse recompute --epic-prefix-recompute-tokens 8 --ablation A4e --engine dag \
  --workload-report out.json --workload-report-events none --cacheblend-batch-size 8 --num-hbm 1 --ngpu 1
python3 -m unittest discover -s tests        # 102/102
```

`--pipeopt` 默认 ON；`--no-pipeopt` 是 AttAcc 的 serial 约定，会抹掉布局收益。

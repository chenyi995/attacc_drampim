# Fugue 七档实验：修改建议与验收标准

本文按照 2026-09-05 用户确认的口径，说明仓库应当怎样修改。除下列已落地项外，正文仍是**待实施建议**，不能视为全部修复完成。

**已落地的计量修正：** 按 Fugue 正文删除 DIE query 旋转/position-transform，默认旋转归 GPU；TLB 是 A3b 起共同需要的寻址机制，各档统一不额外计 latency/energy。其余新增 DIE bookkeeping 也不另收取原始 AttAcc 没有的开销，只保留依赖且不占用调度资源。原始 AttAcc 的 PIM 命令、传输与 energy table 保留。详见 [PIM 计量来源核查](../audit/2026-09-05/PIM_TIMING_PROVENANCE.md)。

这次修改的用户依据、技术原因、逐文件变化和验证边界，完整记录在 [session 文档](sessions/2026-09-05-attacc-accounting-and-rotation.md)。

审计对象为 `8750b5b`，原始反例、源码定位和统计见 [审计报告](../audit/2026-09-05/REPORT.md)、[复现脚本](../audit/2026-09-05/reproduce.py) 与 [证据](../audit/2026-09-05/evidence.json)。本文件对“什么改动可以接受”的定义优先于原报告中的初始要求。

## 1. 已确认的实验口径

**A1、A2 是独立 baseline；从 A3b 开始，按论文 claim 逐级增加机制。** 不要求 A1→A2 或 A2→A3b 只有一个差异。可以自行构造 baseline 和 workload，以合理展示机制收益为目的，不要求先证明它们代表生产流量。

| 档位 | 接受的定义 | 这一档相对上一档允许改变什么 |
|---|---|---|
| A1 | 硬件侧 baseline：无跨请求软件 KV 复用的 AttAcc；按当前论文，GPU prefill、PIM decode | 独立 baseline，不做单变量要求。“硬件 baseline”不意味着整模型或所有 prefill 必须在 PIM。 |
| A2 | 软件侧 baseline：软件 KV 复用，attention 在 GPU，远端内存可以只用于存储 | 独立 baseline，不做单变量要求，也不强制实现最优 GPU 缓存系统。 |
| A3b | 最 naive 的软硬结合：软件复用 + GPU prefill + PIM decode，实际写入时 append-order 布局 | 作为后续机制的起点；允许简单，但读写必须对应真实且持久的地址。 |
| A4c | A3b + 每个 agent、每个 KV head 的 diff 集中存放 | diff 分配、相关地址/描述符/读写流量及由此产生的布局变化；不增加 co-read placement table。 |
| A4e | A4c + 软件表，把可能共同读取的 chunks 分散放置 | master chunk 的 channel/row 选择及 table 的合理维护成本；diff 机制、重算集合保持不变。 |
| A5 | A4e + PIM prefill，配套 MQ | 接受作为一组机制。配套的 PE 频点可以保留，需在配置中明示；MQ 若也改善 decode，归入这一档的整体收益。 |
| A6 | A5 + 自动选择 GPU/PIM prefill | 只增加选边逻辑；GPU/PIM 候选执行器、布局、MQ、频点和输入与 A5 使用相同实现。 |

因此，不再将“A1/A2 同时改变多个因素”“A5 同时引入 prefill offload 与 MQ”“workload 是人为构造”本身列为问题。A5 无需强制拆成更多正式档位；需要解释细节时，再做可选 microbenchmark。

需要修的，是**超出这些允许变化的差异，或让结果不再对应所定义系统的错误**。例如：A3b/A4c 重算不同 token、同一 chunk 每次读取就换地址、A1 报 GPU 实际跑 PIM、共享数据尚未写入就被读取。

## 2. 推荐实施顺序

| 优先级 | 修改包 | 主要文件 | 完成标志 |
|---|---|---|---|
| P0-1 | 固定逻辑 workload 与 ReusePlan | `main.py`, `src/workload.py`, `experiments/run_dag_ladder.sh` | A3b–A6 的逻辑输入和修正索引 hash 相同 |
| P0-2 | 修正算子形状与 baseline 路由 | `src/model.py`, `src/workload_runner.py`, `src/config.py`, `src/devices.py` | 设备归属、GQA 维度和传输量符合各档定义 |
| P0-3 | 统一持久地址与读写依赖 | `src/workload_runner.py`, `src/cpp_eventcore.py`, `src/cppcore/eventcore.cpp` | 每次读都能追溯到有效写入，物理资源冲突不能重叠 |
| P1-1 | 做实 A3b / A4c / A4e 布局 | `src/workload_runner.py`, `src/layout_probe.py` | naive append、per-agent/head diff、co-read table 三者可独立解释 |
| P1-2 | 做实 A5 / A6 执行与选边 | `src/workload_runner.py`, `src/ramulator_wrapper.py` | fresh/reuse prefill 都按模式执行；A6 记录可核查的候选完成时间 |
| P1-3 | 构造能展示每级机制的 workload | 建议新增 `workload/gen_claim_suite.py` 及生成的 JSON | 真正产生跨轮 diff、共同读取冲突、MQ 批次和 GPU/PIM 两类优势场景 |
| P1-4 | 锁定构建、输出指标、重跑 | `set_pim_ramulator.sh`, `experiments/collect_dag_ladder.py`, `output/analysis/extract_sweep_csv.py` | 同版本、同配置、同输入，缺档或混版本时报错 |
| P2 | 扩大 policy / 模型 / workload 覆盖 | `src/workload.py`, `src/workload_runner.py` 及实验脚本 | 基本 ladder 正确后再扩大矩阵，不用旧路径混出新标签 |

先修 P0，再用小 workload 贯通七档。不要先跑大矩阵，再根据曲线去改输入或计量模型。

## 3. 具体怎样改

### 3.1 先生成一次逻辑计划，再套物理布局

当前 `main.py` 根据 `kv_mapping != naive` 决定是否将随机重算位置改成前 k 个；默认 ladder 脚本又使用 `--reuse recompute`。这使 A3b 与 A4c 输入不同。

建议：

1. 按论文当前定义，将正式 ladder 的共同 policy 设置为 `epic`、k=8；先确认实际 planner 产生期望的修正集合。若需要随机修正做敏感性实验，所有档位共用同一 seed 和同一索引集合。
2. 删除“按档位改变逻辑修正位置”的分支。布局可以重排物理地址，但不能改变被修正 token 的身份。
3. 建立与 layout 无关的逻辑记录：`agent_id, request_id, turn, layer, chunk_id, token_index, KV_version, corrected`。`agent_id` 在多轮间保持稳定，不能只用每轮不同的 request ID 代替持久 agent。
4. 为 A3b–A6 保存 `workload_hash` 和 `reuse_plan_hash`。A1 的无复用计划当然可以不同；A2 是否复用同一个软件计划则在 baseline 定义中固定。

**验收：** 当前反例中 A3b 的 `[20,132,155,197,207,215,244,248]` 与 A4c 的 `[0…7]` 差异消失。仅改变 rung，逻辑 KV、修正索引、请求长度、输出长度、到达/依赖保持一致。

### 3.2 A1/A2 可以独立设计，但必须执行自己声明的工作

A1 保持当前论文中的 GPU prefill / PIM decode。替换 `_append_physical_no_reuse_prefill_layer()` 当前全量 PIM prefill；从实际事件统计设备归属，避免先写死 `side_rows["gpu"]` 再调用另一设备。

A2 可继续使用“远端存 KV、GPU 算 attention”的简单 baseline，不要求补一个复杂的最优缓存系统。但应明示 KV 驻留位置、回读规则与是否批量执行，并使用正确算子尺寸、带宽和完整 prefill/decode。其与 A1/A3b 的合法系统差异不再视为消融违规。

全档统一 attention 算子工厂，不能把 score / softmax / context 一起改成 `m=n=L`。每个 head 下，Q 数为 M、可见 KV 数为 N 时：

| 算子 | 正确逻辑形状 |
|---|---|
| score | `(m=M, n=N, k=dhead)` |
| softmax | 对每个 query 的 N 个 scores 归一化 |
| context | `(m=M, n=dhead, k=N)` |

GQA 的 Q projection 宽度为 `Hq*dhead`，K/V 分别为 `Hkv*dhead`；KV 流量应按 KV heads 计，不能在部分路径仍按 Q heads 计。GPU 数与 stack 数的口径只定义一次，容量和能量不再额外套默认 `NUM_ATTACC=8`。

**状态（2026-09-05 晚，commit `dfac28b`）：** A1 的 prefill 已走 `_append_gpu_prefill_layer`（GPU score / softmax / context，历史 KV 经 `kv_pim_to_gpu` 回读），设备归属按实际分支计入；算子形状、GQA 宽度与 GPU 数口径**未在本轮处理**。

**验收：** A1 的 prefill 有 GPU score / softmax / context；L=256、dhead=128 的 context 为 `(256,128,256)`；LLAMA3-8B QKV projection 宽度为 6144；1/2/8 GPU 的容量和能量复制因子可逐项核对。这些修正是恢复定义，不是给 baseline 新增优化。

### 3.3 建立一份真正持久的物理地址账本

优先整理 `_prepare_cacheblend_tlb()`、`_striped_append_channel_extents()`、`_channel_extent_addresses()` 和 channel store/scan helpers。

- 物理单位为 256-token block。长 segment 的 key 必须包含 `block_index`，如 `(layer, kv_head, fingerprint, block_index)`；1024-token segment 应是四个可独立放置的块。
- 地址在**实际写入时**分配一次。之后 store、readback、decode scan、PIM prefill scan 和报表全部读取同一条 `(stack, channel, row, offset)` 记录。
- `scan` 只做地址查询、连续 extent 合并和命令生成，不能根据当前读集合重新从 0 编号、重新分配 channel。
- 物理重排必须由明确的写入或搬移产生；若模型暂不支持搬移，就不允许读时隐式压缩。

**A3b 的 naive 要体现在写入策略简单。** 使用持久 append cursor，不做 co-read 优化；相邻、同批的真实写入应允许正常打包，不应为了增加差距而人为规定“每个修正 token/每个 chunk 必须独占一行”。若有行对齐限制，应统一写进存储模型。

**状态（2026-09-05 晚，commit `5e3c564`）：** A3b 的 scan 放置改为持久写入序 slot（`_place_master_by_slot("append")`，与 A4c 同一张表；修正按首次出现顺序追加到同一轮转）。block_index 级账本、trace 地址回查**未在本轮处理**。

**验收：** 同一个 c4 单独读、与 c0 一起读时物理地址相同；四个 block 不再因同 fingerprint 被强制塞到一个 slot。trace 地址必须能逐个回查写入账本。

### 3.4 A4c：只集中每个 agent、每个 KV head 的 diff

建议 diff arena 按 `(agent_id, layer, kv_head)` 建立；每条有效 diff 再带 chunk、token index 和版本信息。多个轮次新产生的修正追加到同一 arena 的尾部，能放进现有尾行的就继续放；不能在不同 agent 或不同 KV head 之间合并。

master 继续按 append-order 放置，A4c 不使用 co-read table。**零 diff 时 A3b/A4c 的 master 分布必须一致。** 有 diff 时，分离写流造成的自然占用/地址差异可以属于布局 claim；需要把规则写清楚，而不能偷偷额外引入读集合驱动的 master 重排。若论文要求 master channel 始终完全不变，就让两档引用相同的 master 分配结果，仅分流 diff。

对多轮实验，使用真实的“写入其他 KV → 后续产生新修正 → 再读取”序列。区分新增修正与覆盖旧版本；旧 diff 被覆盖不能仍当成有效数据计入收益。最小实现可在请求生命周期结束时整段释放，暂不实现压缩，但要计入未回收占用，并相应限制论文的 compaction 描述。

**验收：** 除 diff 布局及其必要元数据/传输外，A3b/A4c 逻辑工作相同。报告实际 ACT 数、有效/分配字节、每通道扫描时间。diff 占用变少不必然降低最忙通道时间，不应把容量收益直接折算成 latency speedup。

### 3.5 A4e：保留 A4c diff，只替换 master placement 策略

先把“可能一起用”的信息来源写明确。对于这里允许的合成编排 workload，可以声明**编排 DAG、chunk ID 与共同读取关系在首次写入前已知**；在这个实验定义下，提前构建 table 是可接受的，不必先实现通用在线预测器。

建议将 allocator 的 channel 选择分成两个明确策略：A4c `append_order`；A4e `coread_table`。两者共用 block 粒度、容量约束、diff arena、row 分配器与 trace generator。table 的结果在写入时固化，读取时查询。有限通道无法消除全部冲突时按明示的 tie-break 处理，不声称所有 co-read 都可完全分离。

不要用运行结束后的真实 latency 反过来选地址。若输入是未来内容未知的在线 trace，则另行限制可见 hints，或标明采用的是预知编排实验；这不阻碍先完成合成机制验证。

**验收：** A4c/A4e 的 correction list 和 diff 机制完全相同；只改变软件表导致的 master 放置。在无 co-read 冲突的输入上允许收益很小，在有冲突的输入上给出通道负载如何被改善。

### 3.6 先修共享数据依赖和物理资源冲突

地址已预留不等于数据已生成。建议保存 `materialized_event[logical_KV_version]`；每次 scan/readback 必须依赖所读版本的 store 完成事件。外部已经 resident 的 KV 则明确记录初始状态，其初始化成本是否计入测量窗口要在各档一致定义。

将 `PIM`、`PIM:pool0-14`、`PIM:pool0-0` 从互不相关的字符串改成实际资源集合。scan/store 至少在重叠 channel 上互斥；更细的 bank 并发只有在 timing 模型支持时再开放。Python 与 C++ event core 使用相同规则。

**验收：** 反转同 tier 请求的输入顺序，不再允许 consumer 先于 owner 写入完成；覆盖同一 channel 的读写不会因名字不同而重叠；不相交 channel 仍可并行。

### 3.7 A5：所有 prefill attention 真正走 PIM，并使用 MQ

重构 `if not reusable` 分支，使 fresh 与 reuse prefill 都先进入统一的模式分派。推荐复用两个构造器：`build_gpu_prefill()` 和 `build_pim_prefill()`，而不是为每个 rung 复制一份 attention 逻辑。

- A3b/A4c/A4e：选 GPU prefill。
- A5：选 PIM prefill，包括无复用请求。QKV 和 FFN 仍按模型在 GPU，不与“attention 在 PIM”混淆。
- PIM 构造器按合法顺序建立 KV landing、Q distribution、scan、全局 score 处理、PV 和输出回传；fresh KV 未落地不能扫描。
- MQ 使用实际 512-B 容量和当前配套频点；A5/A6 参数相同。允许这一档的 MQ 同时改善 decode。
- GPU 构造器不生成没有必要的 `q_gpu_to_pim`；两条路径均按实际 chunk 位置计算 Q variant 数及相关代价。

**验收：** A5 的实际 prefill attention 事件全部归属 PIM；fresh-only 输入也满足。报表分别列 prefill 与 decode 收益，便于解释该机制包的作用，但不要求再拆正式 rung。

### 3.8 A6：在相同候选执行器之上自动选边

按论文现有 event-based 描述，建议比较候选 prefill 的预计**完成时间**：

```text
state = 当前已提交事件的资源可用时间与依赖状态
gpu_candidate = build_gpu_prefill(同一逻辑计划, state 的副本)
pim_candidate = build_pim_prefill(同一逻辑计划, state 的副本)
gpu_finish = 试排 gpu_candidate 后该 prefill 的完成时间
pim_finish = 试排 pim_candidate 后该 prefill 的完成时间
提交完成更早的候选；相同时按固定规则选 PIM
```

不必每次重新模拟整个系统：复用已缓存的算子/trace 时长，复制调度 frontier，对当前候选子图试排即可。试排不能真的提交 KV 分配、推进真实时钟或污染另一候选。

若决策单位是一整个 prefill，就估计其所有层，包含层间修正量变化；不能只看第一层后缓存结果，却声称比较完整 prefill。把 readback、DRAM store/read、Q variants 的实际传输、尾批、链路和硬件资源等待纳入相同成本模型。DIE/TLB 只保留依赖，各档均不额外计价。

输出每个决策的 `gpu_finish_estimate`、`pim_finish_estimate`、chosen side、资源等待项及实际完成时间。这里的“更早”是给定当前已提交状态下的候选比较，不保证 greedy 选择使整个 workload 全局最优。

如果暂时保留静态成本选边，它仍可称自动选择，但论文和结果必须明确为静态 cost model，不能继续写已实现事件完成时间选择。推荐实现前者，以保持现有论文 claim。

## 4. 怎样构造合理、能够体现收益的 workload

不要求合成输入代表生产分布。建议把它们直接命名为 **claim demonstration / 机制展示**，设计上制造对应机制解决的情形；所有档位使用同一份输入。每项保留一个自然的低收益对照，主要用途是确认收益来自对应机制。

下面是建议新增的 workload 定义，文件名和脚本尚未实现。参数是起点，不是对未经测量 speedup 的承诺。

| 场景 | 建议生成方式 | 展示什么 / 观察什么 |
|---|---|---|
| `diff_rounds` | 稳定 agent 身份；32 个共享 chunks；8 轮，每轮首次修正其中 4 个 chunk、每块 k=8；轮间插入其他 agent/其他 chunk 的真实 KV 写入 | A4c 将每 head 最终 256 个有效修正集中；A3b 的散布必须由真实 append 顺序产生。观察 diff 行数、ACT、最忙通道、decode 时间。 |
| `diff_one_round` | 相同有效修正集合一次产生并连续写入 | A3b 也能正常打包，A4c 收益可能小；验证没有人为禁止 naive 的自然合并。 |
| `coread_conflicts` | 对每 head 有 S 个通道的配置，写入 `c0…c(2S−1)`，部分 agent 共同读取写序相隔 S 的 chunks；首次写入前提供这组 co-read hints | A4e 消除 append-order 的冲突。不要只使用每 head 一个通道的模型来演示该机制。 |
| `coread_balanced` | 同样 chunk 数量，co-read 本来已跨不同通道 | table 收益应很小；验证没有额外改工作量。 |
| `mq_shared` | 多个同时 ready 的查询读取同一 KV；query group 从小批到满 8 slices，context 取 4K/8K/16K，输出先用 128/256 token | A5 的 PIM prefill + MQ 整体收益；分别报告 prefill 和 decode，不假设每个长度均胜出。GQA 下按 slices 而非请求数计算容量。 |
| `placement_mix` | 同一 workload 混合 fresh-heavy、reuse-heavy、不同 Q 数/上下文和到达间隔，并产生 GPU 忙/链路忙/PIM 忙的时段 | A6 自动选边相对 A5 always-PIM 的收益；先单独测各类的 GPU/PIM 候选成本，确认确有交叉。 |
| `ladder_integrated` | N=16、C=32、D=4、k=8、lout=256，shared corpus 子集读取、不同前缀偏移、明确多轮写入和共同读取关系 | 在一个连贯的 agent 工作流中同时出现上述机制。它是机制集成输入，不声称代表真实平均流量。 |

生成时尤其注意：

- **位置差异要真实存在。** 不同 agent 使用不同前缀长度或 corpus 子集/顺序，实际计算 chunk base 和 position delta；不能只在注释写“位置不同”。如果各 tier 的长度恰好相同，planner 仍可能产生零 corrections。
- **多轮要改变存储状态。** 将 Request/turn 与 agent 身份分开，逐轮产生 store/attach/overwrite；不能把整段对话拼成一次 prefill 后称为跨轮 diff accumulation。上表 8 轮的修正应是 32 个不同 chunk 上仍有效的修正，而不是重复计算同一批后虚增数量。
- **散布要来自有效写入。** A3b 可以因其他 agent 的真实写入而使目标 agent 的 diff 不连续；不要插入没有语义的 padding 来让 baseline 变慢。计入被插入工作的时间和空间，或明确统一的 warm-state 测量边界。
- **展示 latency 要检查关键通道。** A4c 少几个 diff rows，只有缩短瓶颈通道或减少关键路径成本时才会改善 latency；否则诚实报告容量/ACT/能耗收益。A4e 负责另一类通道平衡问题。
- **synthetic 输出长度可设定。** 可以用较长输出展示 decode 收益，但要标明生成规则。真实短答案数据保留原始长度；另做合成长度实验即可，不需要用真实数据的名字承载人为拉长的答案。
- **保持简单的参数邻点。** integrated 输入先用 k=2/8/32、轮数 1/4/8、每 head 通道数 1/2/4 做少量检查；不要求先重现完整生产分布，也不必把所有组合跑成巨型矩阵。

如果 `placement_mix` 在正确计量下全选 GPU 或全选 PIM，先确认所覆盖的 Q 数、reuse、context 和资源等待是否确实跨越两侧的成本交叉点。扩大合理输入范围可以接受；不要通过漏算某一侧传输或换档位时改 token 集合来制造交叉。

## 5. 报表与可复现性最低要求

正式重跑前修复 Ramulator 子模块身份校验，清洁构建一次并保存 generator / source / binary hash。不要在未确认独立子模块的目录中执行可能作用于父仓库的 `git reset --hard`。缓存签名绑定模型配置和构建版本，避免沿用旧时序结果。

每个 run 至少保存：

- 实际生效的 rung、GPU/stack 数、GQA 几何、频点、MQ 容量、policy、workload 与 ReusePlan hash。
- makespan、逐请求 TTFT、逐 token decode 时间；PIM prefill **请求比例**与**处理行比例**分列。
- master/diff 有效字节、实际分配/峰值驻留字节；每通道 ACT、扫描时长、link bytes；energy 若仍是代理公式需明示来源，不能把布局估计 ACT 当成 controller 实测计数。
- A6 的候选成本与选边记录；A3b/A4c/A4e 的写入账本和 scan 地址抽样。

collector 检查配置/hash 一致性，缺档或旧 A1 混入时失败。CacheCraft/CacheTune 当前没有完整 decode 路径：扩展政策实验前先统一执行器，未支持时显式报错，不必让这一扩展阻塞基本七档修复。

## 6. 建议的验收清单

| 检查 | 应满足的结果 |
|---|---|
| 同一 workload、同 seed 切换 A3b–A6 | 逻辑计划和修正 token 索引相同 |
| 零 corrections：A3b vs A4c | master 放置规则和结果一致 |
| 一个 chunk 在不同 scan 子集中出现 | 每个档位内部的持久地址不变 |
| 1024-token segment | 四个 256-token block 独立寻址 |
| 同 tier 消费者排在 owner 前 | 仍等待 owner 对应版本 store 完成 |
| `PIM:pool0-14` 与 `PIM:pool0-0` | 共享物理 channel 上冲突不能并行 |
| fresh-only prefill | A1/A3b/A4c/A4e 在 GPU；A5 在 PIM；A6 经选边后执行（2026-09-05 晚 commit `67ce7c4` 起成立，`test_fresh_prefill_follows_the_rung_prefill_side`） |
| context / GQA / GPU 数量 | 算子 FLOPs、KV bytes、能量与容量按真实几何计算 |
| A4e vs A4c | diff 机制相同，只由 placement table 改变 master 放置 |
| A5 vs A6 的 PIM 候选 | 布局、MQ、频点、算子代价相同；A6 只负责选择 |
| 合成多轮 diff 输入 | 有真实跨轮存储状态，修正版本不重复计数 |
| 七档汇总 | 不混 workload、逻辑计划、硬件配置或模拟器版本 |

现有 102 个单元测试通过不代表以上约束成立。优先将审计脚本中的结构反例变成有明确 expected behavior 的回归检查，再跑小规模真实 trace。上述项目完成后，才把 integrated workload 扩展成论文矩阵。

**建议的第一批实际改动是：固定 ReusePlan → 修复 attention 形状和 A1/A5 路由 → 持久地址与读写依赖 → 三种布局 → A6 → 机制 workload 与重跑。** 在这个口径下，目标是让你认可的每一级设计变化确实产生可解释的收益，而不是要求重新设计整套实验哲学。

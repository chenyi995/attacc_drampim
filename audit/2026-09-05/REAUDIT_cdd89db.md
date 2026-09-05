# 当前版本复审：尚不能确认七档差异全部公平

审查日期：2026-09-05。对象：`cdd89db04a85edae029fd3151165f1a488d6139c`，AttAcc 基点：`c600051`。本报告取代旧报告对**当前代码状态**的判断；旧反例和旧测试日志保留原样，不能视作本版本测试结果。

**结论：不能签署“每档只改变声明机制、相对 AttAcc 的每处改动均有充分依据、A3b 没有被额外弱化”的保证。** 已复现不属于 diff gather 的 master 读写差率和资源差异，确认默认实验的修正集合随档变化。另有低估 Fugue 收益的实现缺口。因此不能把现有收益统一解释为保守估计，也不能把这些错误推断成作者主观故意。

**后续专项与口径修订：** [存储与扫描专项](STORAGE_SCAN_CONSISTENCY.md) 已确认 A3b–A6 的 store/scan 通道规则分离、trace row 随读集重建，并补查 A1/A2。用户进一步明确 A6 就是逐 request 估计哪边快选哪边，正文公式可能不准确；据此撤回“必须构建两套候选 DAG”的要求，R07 仅保留估价与实际工作量的对应性问题。当前自动选边机制本身可以接受。

本轮**只做审计**：未修改实现、参数、测试、workload 或现有实验结果；未启动性能矩阵、真实 Ramulator、GPU benchmark、RTL 或综合。主审使用简单 workload 和固定设备桩验证结构，另解析全部 42 个新 sweep 输入；两名 agent 分别独立复审和核查计量来源。

## 1. 审稿口径与逐档裁定

沿用用户确认的范围：A1/A2 可以是合理的独立自建 baseline；从 A3b 起才要求按 claim 逐级变化。A4c 是每 agent/head 的 diff 集中；A4e 是把可能共同读取的 chunks 分散放置的软件表；A5 允许 PIM prefill、MQ 与配套频点作为机制包；A6 加自动选边。合成 workload 本身不构成问题。额外 DIE/TLB latency/energy 不计，旋转在 GPU，不重新要求加回无依据费用。

| 比较 | 本次确认的部分 | 尚不能确认的部分 | 裁定 |
|---|---|---|---|
| A1 / A2 | 独立 baseline 合法；A1 prefill 已在 GPU，A2 无 PIM attention | A1 不是逐事件完全原版：有新 DAG、padding、GPU local decode 等共同模型；平台实际默认 A100a | 接受设计角色，不能直接称数值与正文/原版全等 |
| A3b → A4c | master 的 append **channel slot** 已统一；都用 replicate、同设备单价 | 零 diff master 读写有 15 倍差率；pool 资源互斥不同；修正集合不同；一次生成的修正仍按 fingerprint 分页；部分 master extent 也变 | **不通过纯 diff 归因** |
| A4c → A4e | preset 与布局分支主要只改变 append/table slot；未发现这一档专属费率或额外 GPU 优化 | row 地址不持久、chunk 粒度不完整、table 使用整个已知 workload；实际存储读写未消费该 table | 配置差异符合 claim，物理公平性证据不足 |
| A4e → A5 | fresh/reused prefill 均服从 PIM；MQ 与频点差异属于接受的机制包 | GPU 分支多发无用途的 Q；query 旋转流量、GQA、PIM/普通读写计量不完整；部分 decode 漏用 MQ | 主要机制已接入，收益尚不能全部归因 |
| A5 → A6 | preset 只把 `pim` 改为 `dynamic`，逐 request 选较小估价，符合用户澄清；未发现 A6 专用 scan 缩时系数 | 估计与实际分支的 lane、尾批、操作成本覆盖不一致 | **机制符合接受口径**；估价对应性仍需校正，不要求两候选 DAG |

依据：[preset](../../src/ablation.py)、[论文方法](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:68>)、[论文选边](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/05-execution.tex:39>)。本报告中的 P0 表示阻断当前公平性结论，P1 表示影响执行/归因的实质问题，P2 表示来源、范围或报告需要收紧；不是线上系统事故等级。

## 2. 哪些旧问题已经修好

| 旧问题 | 当前代码与轻量复核 | 状态 |
|---|---|---|
| F01：A1 名义 GPU、实际 PIM prefill | `workload_runner.py:4355` 调 GPU helper；256-token 单请求有 GPU score，无 PIM prefill scan | **已修** |
| F04：fresh prefill 强制 GPU | `4434` 起先按档决定；A5 小例子 prefill 全 PIM，A6 记录所选侧 | **已修** |
| F04：fresh GPU context 形状错误 | `3619` 起 context 设置 k；记录为 `(m,n,k)=(256,128,256)` | **已修**；decode local shape 是另一处，仍存在 |
| F02：同一个 master 随 scan 改 channel | `937` 与 `912` 使用同一 append table；c0+c4 的零 diff 通道分布相同 | **channel 部分已修**；row、读写费用与部分 extent 未修 |
| CLI 的固定 8 台 AttAcc 倍数 | `main.py:545` 显式传 `num_attacc=num_gpu` | **已修**；手建读写与不满 stack 余数另见 R08 |
| 新增 DIE/TLB 收费、零时长 metadata 占队列、DIE 旋转 | Python/C++ metadata 跳过队列，DIE/TLB 费用为 0，`die` 模式拒绝 | **已修**；GPU RoPE 计算仍是零成本假设 |

以前的 102/105/108 项测试记录属于各自历史时点。本轮没有重跑全套测试，不把旧测试通过次数写成本轮验证结果。

## 3. 阻断公平性确认的代码证据

### R01 · P0：零 diff master 读写仍有 15 倍差率，而且资源互斥不同

[store/read helper](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:2192>) 使用旧 TLB 的 `channel_count`，公式为 `bytes / (per_HBM_BW × channel_count / 16)`。Naive allocator 给位置记 1 条通道；A4c 继承的旧 master pool 给位置记 15 条。scan 却用新的 head-local placement，两份模型没有统一。

真实 DAG、同一个 256-token fresh master、零 diff、1 HBM、设备桩带宽 `10^12 B/s`：

| 档位 | store 资源 | 该事件时间 | 该事件能量 |
|---|---|---:|---:|
| A3b | `PIM:pool0-0` | 2.097152 µs | 131.072 nJ |
| A4c/A4e/A5/A6 | `PIM:pool0-14` | 0.1398101333 µs | 131.072 nJ |

**15 倍是此单个事件的公式差率，不是 E2E speedup，也不是真实硬件测量。** 在这个控制组中没有任何 diff 可聚合，不能把差率解释为 claim 1。`dram_read_resident` 也复用同一函数，因此 GPU prefill 回读同样受影响。

还存在另一项混杂：调度按资源字符串排队（`2291`），把 `PIM:pool0-14`、`PIM:pool0-0` 和 `PIM` 当三种独立设备。A3b 单通道 store 会与同名 scan 互斥，A4c 的 broad-pool store 却不会与所覆盖的单通道 scan 互斥。轻量构造三个各占 1 秒的事件，三者都排在 `[0,1]`。后档可能额外获得声明之外的带宽和重叠。

建议验收：同一份最终物理账本同时供 scan、store、readback 使用；资源按实际 channel 集合互斥；零 diff 控制组先证明 master 地址、字节、费用和争用完全一致。仅改一个 `15` 常数不足以解决问题。

### R02 · P0：默认七档没有共用同一份修正计划

[实际脚本](</data2/chenyi9/KV-PIM/attacc_drampim_822/experiments/run_dag_ladder.sh:60>) 使用 `recompute`；[入口](</data2/chenyi9/KV-PIM/attacc_drampim_822/main.py:465>) 对 `naive` 禁用 canonical，对其他布局启用；[planner](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload.py:479>) 分别随机选 k 行和选前 k 行。

相同 256-token shared 段、seed 0、k=8：A3b 为 `[20,132,155,197,207,215,244,248]`，A4c/A4e/A5/A6 为 `[0,1,2,3,4,5,6,7]`。新 sweep **42/42** 个文件都触发集合差异；B0 interleaved 有 124 个复用段的集合不同，B0 turns 为 182 个。

确定的问题是软件工作的 token 身份随物理档位变化，不能称“只改布局”。当前 TLB 费用已经清零，scan 也会补读 shadow master，**不能直接沿用旧的描述符收费推断这项差异现在造成多少性能优势**。真实影响应通过同一冻结计划验证。若使用 `epic`，该 canonical 分支不触发；但当前正式脚本用的是 `recompute`，论文却写 EPIC k=8。

建议先保存不可变 `ReusePlan` 及每请求/层/segment 的 corrected-index hash，各档只转换其物理地址；不能只检查 k 一样。

### R03 · P1：A3b 额外的分页行为与已声明的朴素 append 不一致

[Naive 分配](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:1819>) 每个 fingerprint 单独分页；[修正扫描](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:948>) 又按 `(owner,fingerprint)` 分组；[extent 地址构造](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:793>) 每组另起 DRAM row。

独立 agent 的控制组：同一 consumer 连续产生两个 chunk 的各 8 行修正，二者之间没有 own KV。A3b 放为两行，A4c 放为一行。不同通道上的两行可以并行，所以这不等于 A3b latency 必然更差；但不能将其全部描述成“不同轮次的 own KV 隔开修正”。`output/analysis/layout_interleave_csv.py` 和 `layout_grid_csv.py` 的手算恰好允许同轮两组修正一起 append，和生产分支不是同一规则。

更窄的零 diff 反例：同一个 512-token master 仅读 `[128,384)`，A3b 的旧分页先切成 128+128，再成为两条 row-aligned extent；A4c 得到一条 256-token extent。相同 master slot 并不保证 master geometry 相同。此子区间反例不代表每个对齐 256-token 全读 workload 都发生同样变化，但足以否定接口范围内的普遍保证。

合理的 page-per-object baseline 可以接受，前提是按真实规则定义并公平比较；当前需要先对齐已确认的连续 append 定义，避免把非 claim 的 master 重整和同轮打包差异算作跨轮 diff gather 收益。

### R04 · P1：channel 持久了，trace row 仍按每次 scan 重新编排

`chunk_slot` 保存 `fingerprint→slot`，并没有保存正文要求的 `(channel,row)`。`_channel_extent_addresses:802` 每次从 channel base、cursor 0 开始拼接读集：同一 c4 与 c0 同读时在本通道 row 1；单读 c4 时变 row 0。TLB 报告的地址、读写的旧 pool 地址和送给 Ramulator 的 synthetic extent 不是一份贯穿生命周期的物理地址账本。

此外，整个 fingerprint 的所有 256-token units 都固定在一个 slot（`890`）；1024-token history/segment 不会按正文的四个独立 chunk 分配。所有源对象还在模拟前预留，decode reservation 写成 step-major 不等于 allocator 真的逐 token 写入，因为字典按整块聚合。per-agent diff 在实际 `turns` 中以每次 request id 分配，也没有跨 turn 的物理继承。

因此可以确认“Ramulator 给构造出来的 extent 计时”，不能确认“Ramulator 逐次访问始终写在该位置的 KV”。当前最主要的差异在上层给了什么输入，而非 cycles 乘数。

### R05 · P1：共享 KV 生产依赖和正文的事件执行仍不完整

`build_reuse_plan:398` 将 same-tier owner 声明为候选、不是依赖；[runner](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:4334>) 只继承上一 tier 的完成事件，并未按复用 owner 的具体 KV store 加依赖。将合法 workload 的 consumer 放在列表前、owner 放后，轻量设备桩得到：consumer 首次 scan 在 **5.066 µs 完成**，owner 首次 store 在 **18.066 µs 才开始**；scan 没有 owner store 的 ancestor，现有 validator 仍返回报告。

这是依赖缺失的反例，不是硬件时间结果；也不能宣称所有按 owner 排序的现有运行都实际发生此时间倒置。默认输入顺序可能恰好掩盖它。

另有公共调度差异：每个 tier 的所有请求先构造 prefill，再构造 batch decode；下一 tier 等整个上一 tier，而非只等声明 parent（`4718`）；QKV batch 按输入切块而不是一个在线请求就绪队列（`3241`）。这些行为影响 prefill/decode overlap 和 MQ 共读机会，不能仅用“默认 pipeopt 是 AttAcc 的惯例”证明正文的异步执行已实现。

### R06 · P1：GPU prefill 分支多付一次 Q-to-PIM

[4476 行](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:4476>) 在决定 GPU/PIM 侧之前创建 `q_gpu_to_pim`。进入 GPU attention 分支后仍保留它。shared 小例子的 A3b/A4c/A4e 均多发 32768 B；PIM 侧确实需要这份 Q，GPU 侧不需要。

这会增加 GPU prefill baseline 的链路费用，倾向抬高 A5 相对 A4e 的收益；A6 选 GPU 时也受损，故并非 A6 无条件获益。flash 对每次 X2G 增加的固定 latency 会放大这种事件粒度差异。应以各候选实际需要的输入构造 DAG。

### R07 · P1：逐 request 选边符合澄清后的机制，估价仍须对齐执行

[估计器](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:3640>) 比较 GPU 回读链路加三算子，与某条 PIM lane 的时间乘 sweep 次数加 Q/context 链路，按 request 选择较小值、跨层沿用。**按用户后续澄清，这个简单选择规则可以接受，不要求输入队列状态或构造两套候选 DAG。** 本条撤回原先依据正文 DAG wording 的机制违规结论。

仍需核对的估价缺口：用最多 token 的 lane 代替所有 lane 实际定价后的最慢者；满 sweep 的价格也用于尾批；未覆盖实际 GPU 分支的 DRAM readback、额外 Q 等操作；忽略旋转 variants。这些是两侧工作量与分支执行的对应性问题，可能双向影响选择。第一层估价跨层使用可作为简化披露，不单独判 claim 违规。专项 [SS06](STORAGE_SCAN_CONSISTENCY.md) 用人为可检查的设备价格复现仅询价一个 lane 的覆盖缺口；不是实际硬件误判率测量。

原设计指南的 **`A6 ≤ min(A4e,A5)` 恒成立**不是局部选择可保证的全局性质，因为不同选择会改变后续队列和 batching；本轮已修正文档。论文的公式和 DAG/completion 表述建议按用户确认的简单逐 request 定义同步，不要求反过来升级实现。

### R08 · P1：GQA、head/stack 工作量和 MQ 覆盖尚不完整

这部分包含相反方向的偏差，不能合并成“都保守”：

- `src/model.py` 仍建 `3*hdim/tp` QKV。LLAMA3-8B 为 12288 列，而按仓库自身 32 Q/8 KV、dhead 128 的几何应为 6144。KV 链路还用 `2*local_hidden`，该 GQA 配置 KV 字节多算 4 倍；GPU 回读比重高的 A3b/A4e 可能更吃亏。
- PIM 的 GQA query 数放在 `pim_shared_queries`，ALU energy 却只依赖 `layer.get_flops()`，少了新增 GQA 组的操作倍数。见专项 MP-06。
- [private decode](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:3466>) 和 batch=1 decode 未传 `_apply_pim_batch`。独立 probe 看到 common GQA scan 为 MQ/1.3004 GHz，private 的四 query scan 却无 command/frequency，wrapper 默认 replicate。**这会低估 A5/A6 的部分 GQA MQ 收益**；MHA 单 query private 不受这项影响。
- 最忙 stack 的时间可以用向上取整 head 数，但 energy 不应把所有 stack 都当满载。LLAMA-33B、2 GPU/10 HBM 的公式按每 GPU 30 heads 计，实际 26，多 15.38%，倾向低估 PIM 节能。CLI `num_attacc` 修复不覆盖此余数。
- 多 head 被折成长 sequence，再令 `numOp=1`；每 channel 多个独立 head 的 Q/softmax 边界未保留完整 trace 证明。论文的 LLAMA-7B/1HBM 正是需要核查的配置。不能以 MAC 总量等价替代 Q 装载、softmax、输出等价。

这些是共同算子模型或路径遗漏，不是发现 A4e/A6 有隐藏专用时长系数。推荐先做每个 Q/KV head 的维度与 byte/MAC 守恒表。

### R09 · P1：新 workload 的位置变化没有进入 query variants

论文 [04-design.tex:175](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex:175>) 要求 GPU 按 chunk base 产生 query variant。parser 的 `delta` 默认是 0（`workload.py:301`）；`_prefill_location_deltas:2647` 只读该字段，不从 consumer/owner 实际位置推导。

42 个新 sweep 文件的 **nonzero delta 总数为 0**。B0 interleaved 中 124 个复用段实际前缀偏移不同，却都仍为 delta 0；B0 turns 对应 126 个（不含无法直接按 owner input 推导的 parent_out）。软件重算能检测 offset 变化，旋转流量路径却检测不到同一变化。

这低估 PIM 侧 query variants 的流量、等待和潜在 buffer 使用；A5 的 prefill 也在 PIM，暴露通常比 A3b/A4e 的 GPU prefill 大。GPU RoPE 的计算费用为零是另外一个已披露的简化，两者不可混淆：即使不计旋转计算，也必须忠实反映声明需要的 query 传输。

### R10 · P1/P2：多轮 workload 与 placement 信息范围需要重新标清

合成机制 workload 合法，但目前两种写法并非相同逻辑工作量：

| B0 输入 | requests | tiers | input tokens | output tokens | 单独预置 history tokens | reused tokens | k=8 correction tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| interleaved | 9 | 1 | 59648 | 2304 | 0 | 32768 | 992 |
| turns | 65 | 8 | 73984 | 16640 | 200704 | 47104 | 1456 |

`interleaved` 将各“轮”的内容一次 prefill，只有一次 decode；`turns` 每轮解码并把 parent_out 带入下一轮。两者可分别作机制与多轮压力测试，不能因注释“同一 session 两种写法”就当作等价执行对照。

`history_len` 在 [2866 行](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:2866>) 为每个新 request 分配新的已驻留 extent，不从前一轮的真实地址继承，也不在本轮生成/搬运。小例子 parent input 首地址 0、child 的“同一历史”首地址 3584；没有迁移事件。这可以作为预置历史抽象，但不能支撑“跨轮真实物理布局保持且由此受益”的测量结论；它还可能把多轮 diff/fragmentation 变成规整 master history。

table 的 `chunk_coread` 使用全 workload（`2896`），与当前正文“写入前已知共读集合”的假设一致，**不把已知 DAG 本身判不公平**。需要披露它不是未知在线请求预测，未来若采用动态到达，应明确可见信息边界。

特别纠正现有 session §10 和 run guide 的归因：当前 B0 的 **9/9、65/65 个输出 fingerprint 都已在 table 的 `chunk_order` 中**，`coread` 也显式加 output。因此“A4e 比 A4c 慢是因为它没有 decode 输出信息”的解释不能由当前代码支持。它缺少更细粒度真实写入/共读建模是另一件事；已有速度差原因须重新做可复核分解。

### R11 · P1/P2：不是每一项相对 AttAcc 的计量扩展都已被校准

[计量来源专项报告](MODEL_PROVENANCE_REAUDIT.md) 逐项区分如下：

| 项目 | 可以确认 | 不能扩大为 |
|---|---|---|
| AttAcc `ENERGY_TABLE` | 对 `c600051` 保留；用户同意据此算能量 | 新增操作数、GQA、所有复制维度必然正确 |
| bank scan 时间 | wrapper 使用 Ramulator cycles × 0.769 ns，无按 rung 后乘手工 speedup | 所有 PIM latency 都来自 Ramulator，或输入就是完整真实物理执行 |
| KV store/read | bytes/BW 与 bytes×AttAcc 单价 | 由 Ramulator 得到；BW 用内部 all-bank rate 也未证明适用于普通读写 |
| MQ interval / n=8 / 512 B / 1.3004 GHz | 公式与正文及用户 energy 裁决一致 | 整个 workload 平均功耗就是每个命令的功率上限，或 RTL 已在本轮重验 |
| GPU flash | 所有档共享开关，有 FA-2 算法依据 | 所有 shape/H100 都实测校准，或只改变了 attention kernel |
| `apply_attacc_pipeline` 抽取、C++ 加速、路径隔离 | host 重构有正当目的，未发现按档偷调费用 | 新 DAG 的物理并发已由原版 AttAcc 背书 |

cuBLAS 表的[作者仓库](https://github.com/harshithkantamneni/triton-vs-cublas-llm-benchmarks)确有 A100、76 GEMM 与 p50 的来源说明；本轮没有成功取得具名原始 CSV 逐项核验。FA-2 的[原论文](https://arxiv.org/abs/2307.08691)存在，但本仓库明确是近似读图，另套 occupancy、短 Q padding、L2 常驻等假设。来源存在不等于外推已校准。

flash/refined 还把 all-reduce 的 6.06 µs 截距用于每次 GPU↔PIM transfer，加 far-HBM 限速；它不是原 AttAcc X2G 公式，也不是已验证的同种物理事务。其影响随传输拆分粒度变化。强化 GPU attention 本身是合理的 baseline 改善，不能反过来要求只保留较慢的 legacy；但不能不披露这个模式同时改了 GEMM、通信和内存假设。

此外，脚本不传 `--gpu`，实际默认 **A100a**，而正文平台段写 H100。A100a+HBM3 是原 AttAcc 已有合成平台，不是本轮偷偷加带宽；只是现有入口结果不能直接称正文 H100 数据。

### R12 · P1/P2：Ramulator 来源可追踪，但缓存和命令可执行性未闭环

`_address_mapping_signature` 删除 row index，多 extent 的键又对每段独立删除（`ramulator_wrapper.py:602`）。两段地址 `(0,64)` 与 `(0,1088)` 可得到同一键，却分别是同 row 与不同 row。这里已证明不同物理关系落同键，未声称未运行的具体 trace 必然产生多少周期差。当前每段行对齐会减少部分触发机会，不能证明通用键正确。持久缓存也未绑定 generator/controller/binary hash。

MQ 的 column 共享与 interval 是真实 trace/YAML 输入修改；但 context 主路径先搬完所有 P、再 barrier、再 MAC，没有按有界双缓冲组织论文的 streaming P。它计了总 movement 字节，不能称 P 流量为零；过度串行与无限预装假设的净偏差方向不定。

原 AttAcc 的 full-scan ALU energy 已存在 QK/PV 未分别完整计数的局限；这与新增 GQA 少乘数要分开。ACT counter 虽加在 controller，energy wrapper 未按实测 ACT 数消费；不能把旧摊销能量称成新增逐 ACT 精确能量。按用户要求，本轮不更改单价、不补无依据 DIE/TLB。

默认安装还缺对应 generator/source，存在旧二进制；外部 scratch 可以正确安装，但没有 binary/source/cache hash 就不能核验具体历史结果。本轮没有执行初始化或 rebuild，也没有断言既有结果都是假的。

### R13 · P1：软件 policy sweep 和正文 decode 分工没有统一

`run_reuse_prefill:4889` 只将 cacheblend/epic/recompute/no-reuse 送进当前 PIM DAG；CacheCraft、CacheTune 回落旧 prefill runner。小例子这两类报告都无 `makespan_s`、无 decode event，不能用来支持正文“在同一 Fugue 上换软件策略”的完整结果。

PIM decode 仍对最新 token 运行 GPU local score/softmax/context 并传 LSE tuple（`3360`），不完全等于正文 always-in-bank；local context 仍有错误形状 `(1,1,1)`，应与 head-width 的输出区分。它是各 PIM 档共同的执行模型问题，不是某个后档偷优化。

### R14 · P2：实验入口、旧手算和汇总尚不能提供版本一致的证据

- 正式新矩阵是 42 个多轮探针，B0 全七档、其他点 A3b/A6；正文还是 14 个 tiered-DAG × 6 模型 × 7 档。用户已允许新矩阵，不要求改回 588 点，但论文输入/范围应同步，sweep 只有两端不能证明每个中间档都公平。
- 新 `collect_dag_ladder.py` / `summarize_ladder.py` 含 A4c/A4e；旧 `extract_sweep_csv.py:56`、`layout_handcheck_report.py:28` 等仍列 A4/A4b，可能漏掉当前中间档。旧手算规则不证明当前 persistent-slot 实现正确。
- tier collector 使用最后 attention **start** 而非最后 token 完成推 `cum_end_s`；不能把它当真正 tier completion/端到端完成。TBT 的请求均值与正文按 step 加权的指标也不同，必须明确。
- `SKIP_A1` 可复制旧 JSON；collector 只看文件是否存在，不核验 workload/plan/GPU mode/source hash；`run_sweep.sh` 会日志记录失败但最终仍可显示 done。结果目录混用会破坏可比性。
- source 缓存加速、并行脚本、监视器本身不是模拟硬件新机制；但只跑选定机制探针、只保存有利设置、引用旧结果，不能代替每一对的归因检查。最新 session 已把被停止的 flash sweep 半成品标作废，本轮未将其用作收益证据。

## 4. Workload 偏差方向：无法统一称低估或高估

| 因素 | 可预期方向与边界 |
|---|---|
| A3b 旧 pool 单通道费率、后档 broad-pool 与 lane 不互斥 | 有利 A4c 及后档；确定存在非 claim 混杂 |
| GPU prefill 多发 Q，GQA KV 字节多算，delta=0 漏旋转流量 | 倾向抬高 PIM prefill 相对 GPU 的收益；实际幅度未知 |
| 规则化 256-token 语料、重复访问、同 tier 就绪、多轮修正 | 合法的机制压力测试，较易暴露 gather/placement/MQ；不是生产分布保证 |
| 多个旧独立 head 拼成一份 trace、预置 history 重整 | 可能理想化 PIM 执行/跨轮布局；需要数值与地址验证，不能给定净方向 |
| owner 的长 fresh prefill、加入大量 fresh chats | A5 所有 prefill 上 PIM 会受损；A6 能避开时正是合法 claim，不属于刻意削弱 A5 |
| GPU/FFN 固定共同开销、private-heavy decode、MQ 漏传 | 可掩盖布局和 MQ 收益；GQA private 漏 MQ 是明确反向缺口 |
| 不满 stack 按满载复制 PIM energy | 高估部分 PIM energy，可能低估节能 |
| table 完整已知 DAG、资源/缓存错误、GPU 近似模型 | 在各自范围内披露；未做敏感性/修复对照前净方向未知 |

建议用少量合法控制组验证规则，而不是追求每个 workload 都得到单调变好：零修正、同轮连续修正、跨轮有真实 own 写入、零冲突/有冲突、private-heavy GQA、已知/未知未来共读、GPU/PIM 各自拥塞。A4e 或 A6 在部分场景变慢并不自动表示不公平；需要看是否忠实执行声明机制和同一工作量。

## 5. 证据、覆盖和后续验收顺序

- [主审结构证据 JSON](reaudit_cdd89db_evidence.json)：真实 planner/layout/DAG，固定 1 µs 的设备桩；七档 fresh 路由、store 差率、修正集合、owner 依赖、资源别名、history、policy 分派及 42 个 workload 统计。只用于结构判断。
- [探针源码文本](reaudit_cdd89db_probe.txt)：归档为 `.txt`，没有改动仓库的实现或测试代码。可在仓库根目录用 `PYTHONDONTWRITEBYTECODE=1 KVPIM_CPPCORE=0 python3 audit/2026-09-05/reaudit_cdd89db_probe.txt` 重新产生 JSON；不会运行真实 Ramulator。
- [独立 agent 报告](INDEPENDENT_REAUDIT.md)：另行读取当前源码并用小探针复核，另发现同轮 diff、master 子区间、MQ 漏传等问题。
- [计量来源 agent 报告](MODEL_PROVENANCE_REAUDIT.md)：逐项对照 AttAcc、MQ、GPU、缓存和能量乘数。
- [110 个变更文件覆盖表](reaudit_cdd89db_file_coverage.csv)：逐文件记录检查方式、源码 hash、相关问题。生成数据按 parser/结构核查，历史结果按来源核查；不假称 12 万行全部有独立数值证明。
- [快照 manifest](reaudit_cdd89db_manifest.json)：HEAD、AttAcc 基点、论文/代码 hash、验证方式、agent 与“不修改实现”检查。

优先顺序：先统一 corrected rows 与完整物理读写/资源账本（R01–R05，存储专项 SS01–SS05）；再清理多余操作并对齐 A6 逐 request 的两侧估价（R06–R07）；随后核对 GQA、MQ 覆盖、RoPE/history 与能量维度（R08–R13）；最后冻结来源、缓存、workload 和指标再做获授权的硬件结果验证（R14）。本轮仅给出审计和建议，**没有实施这些修复，也没有重新测量论文收益**。

# 本轮审计结论与执行交接：已定口径、证据和验收要求

**本轮问题统一看这一份。** 对象是代码 `8c51672`、原始 AttAcc `c600051` 和当前论文正文。这里整理已有审计证据，并按贡献 README 增补轻量构图与命令检查，没有新增性能实验。最新裁决已落在下表：C2/C3/C6 待执行，C5 按实际需要的成本估算并忽略 Q 传输；C7 的同轮/跨轮边界已确定，同轮合并不再列为问题；C4 经复审按既有共同近似规则记录；C8 按用户明确的 a/c 原址继续引用语义作为执行要求。本轮没有继续等待用户选择的建模原则。本文记录裁决，不代表代码已修。

审计只问：**不同档位的性能变化，能不能归因于论文允许增加的机制？** 原始 AttAcc 已有或所有档共同使用的模型近似，有依据即可接受；不要求绝对精确，也不要求它对各档造成完全相等的误差。本文说“复现”，指真实代码的构图、地址或汇总行为，不等于测到了硬件加速比。

<a id="decisions"></a>

## 已裁决的交接清单

以下是 chenyi9 本轮裁决。执行 agent 可按此修改；本次主审和独立 agent 只做 audit、证据及文档，没有实施代码或论文修复。

| 项目 | 已确定的处理 | 执行时要核对什么 |
|---|---|---|
| [C1](#c1) FlashAttention | 继续执行此前“所有档共同开启”的决定 | 各入口实际配置一致；pipeline 按共同口径开启 |
| [C3](#c3) diff 地址 | 修正新增 diff 区分配 | 不只要求地址整数不同，还应对应不重叠的实际 ALL-BANK 行；保持 head 的 master 通道范围和已声明布局 |
| [C5](#c5) A6 估价 | 按实际路径需要哪些项，就估算哪些项；Q 传输可忽略 | 没有历史时回读为零，有真实回读时按真实字节估算；论文与 codebase 同步到简单公式，详见 C5 |
| [C2](#c2) GQA | 修复 | 校验器、prefill/batch decode KV 字节统一按 KV heads；不能只放宽校验 |
| [C6](#c6) 报表 | 更新 | A2 纳入 tier 汇总，相关时间列使用对应指标的起止范围 |
| [C7](#c7) A3b 轮次边界 | 同一 round 的多组 diff 可正常追加紧排；跨轮有输出/新写入后不能重新并成同批 | 撤回单 prefill 合并的违规判断；真实跨轮存储覆盖另见 C8 |
| [C8](#c8) 跨轮持久引用 | 后续 request 继续引用原 master 和该 agent 已产生的 diff；输出/新内容正常追加 | 保留旧对象地址，不能仅用新 history master 占位代替；沿用 C7 的同轮/跨轮规则 |

**本轮口径已明确，当前没有需要用户再次裁决的事项。** C8 以 chenyi9 对 a/b/c/e/f 的说明为依据：a 是原 shared chunk，c 是该 agent 已产生的 diff，后续 f 仍 attention 原来的 a/c；后续输出或其他 KV 在后面追加。该说明已经明确目标，无需再问是否接受用新 history 占位替代它。C4 按共同计时近似记录，C7 的同轮合并指控撤回。**口径确定不等于代码修复完成；下列已定项仍需执行 agent 落实并由 audit 验收。** 记录见 [最终交接 session](../../docs/sessions/2026-09-05-audit-decisions-finalized.md)。

## 先认识系统和七个比较点

LLM 先处理整段输入，这叫 **prefill**；之后逐 token 生成，这叫 **decode**。为了不重复计算历史内容，系统保存 **KV 缓存**，即每个 token 的 K、V 向量。GPU 负责大部分计算；**PIM** 是内存旁的计算单元，可以直接扫描内存里的 KV 做 attention。

多个 agent 复用同一段内容时，共享的原始 KV 块叫 **master**；为某个 agent 重算的少量 token 叫 **diff**。这里的 diff 是重新计算的 KV，**不是两个数值相减的差**。一个 **head** 是一组 attention 计算；正常情况下，各 KV head 有自己的 KV 数据。**Q/query** 是当前 token 用来查询历史 KV 的向量；一次 **request** 是一个请求，**prompt** 是它的输入。

| 档位 | 系统做什么 | 允许增加的机制 |
|---|---|---|
| A1 | 不做跨请求软件复用；GPU prefill，PIM decode | 独立硬件 baseline |
| A2 | 软件复用；attention 全在 GPU，远端内存存 KV | 独立软件 baseline |
| A3b | 软件复用 + PIM decode；修正跟着普通写入存放 | 后续消融的朴素起点 |
| A4c | 把同一 agent、同一 KV head 的修正集中存放 | diff 布局，允许跨轮修正共享行 |
| A4e | 将以后可能一起读取的 master 块分散到不同通道 | 软件放置表；diff 规则沿用 A4c |
| A5 | prefill attention 也在 PIM，多个 query 共用 KV 读取 | PIM prefill + MQ 及已接受的配套频点，是一组机制 |
| A6 | 每个 request 估算 GPU/PIM 哪边快，选择较快的一边 | 简单选边；其他硬件与 A5 相同 |

MQ 指多个 query 利用同一次 KV 列读取。A1/A2 不要求逐步只改一项；从 A3b 开始才按允许机制比较。表格对应论文 [Comparison points](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:65>)；A6 以用户后来明确的简单逐 request 定义为准，**不要求实现正文旧公式里的两套候选 DAG**。

<a id="contributions-check"></a>

## 按 README_contributions 的四个例子核对

**结论：四项机制与当前档位配置对应，例子的主要布局/操作可以构造；还不能据此确认每档的完整性能比较都正确。** [贡献 README](../../docs/README_contributions.md) 已明确这些是条件下的推导、选边时间是假设值。其内嵌脚本读取常量并生成示意表，没有调用实际 allocator、命令生成器或 Ramulator；复算通过不等于实现通过。下面另用真实 planner/构图和小规模命令生成检查，独立 agent 复核了前两个例子。

先用相同 reuse policy、没有 CLI 覆盖的 `resolve_config` 比较相邻档；除档位名称外，实际变化是：

| 相邻档 | 展开后的配置变化 |
|---|---|
| A3b → A4c | `kv_mapping`：`naive` → `master-diff-local` |
| A4c → A4e | `kv_mapping`：`master-diff-local` → `master-diff-table-local` |
| A4e → A5 | `pim_batch_command`：`replicate` → `mq`；`pim_pe_freq_ghz`：`0.666` → `1.3004`；`pim_prefill_query_batch`：`4` → `8`；`prefill_attn`：`gpu` → `pim` |
| A5 → A6 | `prefill_attn`：`pim` → `dynamic` |

这些变化在用户接受的范围内。A5 的 query 容量和频点属于 prefill + MQ 配套；当前各档缓冲默认已经同为 512 B，不能再写成 A5 独自扩大缓冲。配置差异表不能替代对映射分支和执行路径的检查。来源：[档位与配置展开](../../src/ablation.py:98)、[本次配置证据](archive/contributions_alignment_evidence.json)。

| README 的例子 | 用实现核对到什么 | 本轮 audit 如何解释 |
|---|---|---|
| ① A3b → A4c：跨轮 diff 聚合 | 同一 consumer 的单次 prefill 中静态展开八段修正，真实 planner/ledger 中两档 master 通道相同，64 个 diff 的**名义占行**从 8 变为 1；A4c 都在该 head 的 ch3 | 支持给定分配规则的高层几何；单次 prefill 不能证明跨轮行为（C8），也未测到 ACT 从 16 降为 9。名义行到真实命令仍受 [C3](#c3)/[C4](#c4) 影响，不能用例子关闭它们 |
| ② A4c → A4e：M0/M4 分散 | 五个 producer 分别生产一块，再由 consumer 共读 M0/M4，实际放置表能让 M0 留 ch0、M4 去 ch1 | 支持给定共读关系下的放置；其他已知共读关系可能改变选择，不保证每个 pair 都变快 |
| ③ A4e → A5：MQ + PIM prefill | 用 README 的历史长度和新增 query 数构图，A4e 每 head 回读 4 MiB 历史 KV，A5 为 0；真实命令生成也保留 MQ 读列复用 | 两部分机制都接入了。只是字节/命令验证，不是注意力加速测量；MHA 控制例不能关闭 [C2](#c2) 的 GQA 分支问题 |
| ④ A5 → A6：逐 request 选较小估价 | 实现逐 request 比较 `t_bank` 和 `t_xpu`，相等选 PIM；README 假设值复算为 920 → 220 µs，即 4.18× | 规则符合用户定义，不要求双候选 DAG；估价按最新 [C5](#c5) 裁决列实际需要的项，Q 传输接受忽略；假设数没有验证实际估价 |

**例子①还依赖什么输入。** A3b 的轮转计数会同时计入普通 KV 块和 diff burst。独立 probe 中，两档 master 通道均为 `[0,1,2,3,0,1,2,3]`；每段修正之间插四块自写 KV 时，A3b 的 diff 通道为 `[0,1,2,3,0,1,2,3]`，与 README 表格一致；只插一块则为 `[0,2,0,2,0,2,0,2]`。所以“轮间穿插自写 KV”本身不保证逐轮轮转。README 已显式写了轮转假设，可以作为示意；若要作为可执行验证例，应补上自写长度。按最新 round 裁决，此次是一个 request 的静态展开，只能核对分配几何，不能当成八个真实服务回合的验证；同轮紧排本身允许，跨轮证据另见 C8。来源：[普通块/diff 计数](../../src/workload_runner.py:934)、[独立证据](archive/independent_contributions_evidence.json)。

**例子②还依赖什么输入。** 如果一个 producer 自己先读取 M0–M4 全部五块，planner 会同时记录这个全体共读集合；随后即使 consumer 只读 M0/M4，真实表也可能让它们都留在 ch0。算法考虑所有已知共读伙伴，贪心选冲突较少的通道；它没有承诺每一对块都能分开。建议例子补上“没有其他共读关系改变这一选择”。放置表依据整份 workload 预先收集关系，沿用已说明的静态预分配抽象，不把本例说成在线预测验证。来源：[放置算法](../../src/workload_runner.py:823)、[共读关系收集](../../src/workload_runner.py:3212)。

**例子③具体保留了哪些费用。** 在一层、四个 MHA KV heads 的构图中，按每 head 归一：两档都传新增 KV 4,096 B；A5 另外传 Q 2,048 B、返回 context 2,048 B。单通道、单个 256-token 行、八个 query 的实际命令生成器输出 trace 中，`PIM_MAC_AB` 命令为 512 → 64，`PIM_SFM` 均为 8，其他 query 私有搬运命令数不变。MQ 的每 query 计算仍由命令间隔/频点计价，不能把少发命令直接理解为少做 MAC。上述构图使用固定价格设备桩且 `pipe=True`，不涉及实际 GPU kernel；正式比较仍须统一开启 FlashAttention（[C1](#c1)）。来源：[实际 prefill 分支](../../src/workload_runner.py:4827)、[MQ 命令展开](../../pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py:586)、[间隔计价](../../src/ramulator_wrapper.py:72)。

**PIM 时间是否由示例比例拟合。** 被检查的 placement 扫描路径调用 Ramulator（或复用其缓存结果），将返回周期乘 `tCK` 换算成时间；没有把 README 的 ACT/读列减少比例直接乘到延迟上。MQ 在送入模拟前按驻留 query 数和频点设置命令间隔，不是在输出后简单除以 query 数。这支持计时来源与声明一致，但地址输入是否代表实际存储仍是 C3/C4，不能仅凭“来自 Ramulator”保证公平。此次没有重跑正式结果或验证每份历史缓存来源。依据：[模拟调用和周期换算](../../src/ramulator_wrapper.py:667)、[MQ 间隔设置](../../src/ramulator_wrapper.py:564)。

**这些例子是否高估 workload 收益。** 它们有意选中机制有用的场景：多轮稀疏修正、轮转恰好冲突的一对块、长驻留历史配少量新 query，以及分别适合两侧的请求。它们适合解释机制；若把局部 43.75% ACT 推导、2× 通道并行或 4.18× 假设选边收益当成整体 workload 的预期收益，会把有利场景当作一般情况，存在偏乐观风险，不能据此推算整体收益。反过来，只有一批 diff、原来已均匀分散的共读块、很少历史或所有请求都适合 PIM 时，对应步骤收益可能很小。本次没有统计正式 workload 的场景占比，也没有运行性能实验，因此不能判定正式结果整体高估或低估。

**相对 AttAcc 怎样判。** 原始 AttAcc 没有这四项软件复用/布局/MQ/选边组合；各例是新增机制的说明，应核对新增逻辑。共同链路、能量和调度近似仍按用户口径接受。上述例子条件不自动成为新的整改项；当前重点仍是新增地址选择和除已接受的 Q 传输近似外，候选计价是否符合实际需要。没有发现本次例子能证明“刻意削弱 A3b”。完整验证范围和未修改实现的记录见 [本次 session](../../docs/sessions/2026-09-05-contributions-alignment-audit.md)。

## 本轮阅读地图

| Case | 一句话说明 | 对应的论文机制 / 比较 | 当前状态 |
|---|---|---|---|
| [C1](#c1) | 运行命令可能没开启 FlashAttention | 所有档共同 GPU baseline | 已确定必须开；没有证据说所有历史运行都没开 |
| [C2](#c2) | LLaMA3-8B 一类的 GQA 输入在 PIM 路径被拒绝，batch 字节也未统一 | 论文模型覆盖；A2 与 PIM 档 | 已裁决修复，尚未实施 |
| [C3](#c3) | 新 diff 区的地址看似不同，ALL-BANK 实际选到同一组行 | A3b→A4c，后续档继承 | 已裁决修正分配，尚未实施 |
| [C4](#c4) | 同一块 V，读整块和读子段会用不同的物理步长 | A3b–A6 共同的固定存储/子段扫描 | 已重审；按既有共同近似规则记录，无现有整改要求 |
| [C5](#c5) | A6 按实际需要的项估价，Q 传输接受忽略 | A5→A6 的简单选边 | 已裁决估算原则；公式和实现待同步 |
| [C6](#c6) | 有 A2 输入，tier 汇总表却不输出 A2 | 七档结果呈现 | 已裁决更新报表，尚未实施 |
| [C7](#c7) | 同轮可合并，跨轮输出/写入后须保留分隔 | A3b baseline 与 A4c 跨轮聚合的界线 | 定义已裁决；撤回原同轮反例 |
| [C8](#c8) | 当前多轮输入未沿用前一轮 diff 的物理布局 | A3b→A4c 的跨轮收益覆盖 | 原址继续引用要求已明确；待执行/验收 |

**关于 A4e 和 A5：** 当前没有发现 A4e 私有的缩时系数；也没有新证据表明 A5 的 MQ 接入又丢失。C3/C4 是这些档共同依赖的存储问题，不能改写成“A4e 表独有作弊”或“MQ 已测出虚假加速”。

<a id="c1"></a>

## C1：同样跑七档，命令入口可能选择了不同 GPU 模型

**论文/用户允许什么。** 论文 [平台说明](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:21>) 把 GPU kernel 和设备计价作为比较背景；用户进一步明确：我们新增的 FlashAttention 必须给所有档共同开启。不是只有 A2 要开，A1 和 A3b–A6 的 GPU 部分也一样。

**具体 case。** 若直接执行一键 ladder，没有设置 `GPU_MODEL`。脚本没有传 `--gpu-model`，于是 main.py 使用默认 `legacy`；改用 sweep 脚本则默认传 flash。

| 入口 | 实际默认行为 |
|---|---|
| `python3 main.py …` | legacy，需加 `--gpu-model flash` |
| `bash experiments/run_dag_ladder.sh …` | 未设环境变量时是 legacy，需加 `GPU_MODEL=flash` |
| `bash experiments/run_sweep.sh …` | 默认 flash，环境变量仍可覆盖 |

**AttAcc 本身有没有。** 原始 AttAcc 没有我们新增的 flash 模型。它的 legacy 不能用作“不必开启 flash”的依据。flash 参数来自共同解析模型、有论文图来源即可；非本库硬件实测本身不算公平性问题。

**影响什么比较。** 如果不同档实际混用模型，GPU 的快慢就不只由消融机制决定；即使全用 legacy，也没有执行用户指定的共同配置。当前证据只确认入口默认，**没有证明每份历史结果都使用 legacy**。

**当前决定。** 必须开启已经确定。README 示例已补齐；代码默认未改。建议只核对共同实际配置，不因缺少某个日志字段就直接否定全部结果。证据：[main 参数](../../main.py:140)、[ladder 透传](../../experiments/run_dag_ladder.sh:73)、[sweep 默认](../../experiments/run_sweep.sh:22)；旧编号 RC01。

<a id="c2"></a>

## C2：论文中的 GQA 模型，有的档能够构图，有的档直接报错

**论文声明什么。** [Models](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:122>) 包含 LLaMA3-8B。普通 **MHA** 是每个 Q head 对应自己的 KV head；它的 **GQA** 则是多个 Q head 共用较少的 KV head：要存、要传的是 KV head 的数据，不能把每个 Q head 都当成一份独立 KV。

**具体 case。** 同一短 workload 分别采用 CACHEBLEND-TINY（MHA）和 LLAMA3-8B（GQA）配置，均缩为一层。两者模型尺寸也不同，这张表不是“只改 GQA 一个参数”的性能对照。测试用固定价格函数代替真实设备（设备桩），实际执行的是列出操作及其依赖的构图逻辑和校验逻辑，不衡量 GPU/PIM 真正速度。

| 档位 | MHA 控制 | GQA 输入 |
|---|---|---|
| A1 | 完成 | KV 字节校验失败 |
| A2 | 完成 | 完成 |
| A3b | 完成 | KV 字节校验失败 |
| A4c | 完成 | KV 字节校验失败 |
| A4e | 完成 | KV 字节校验失败 |
| A5 | 完成 | KV 字节校验失败 |
| A6 | 完成 | KV 字节校验失败 |

**代码实际做什么。** 发送 KV 的一部分代码已按较少的 KV head 算字节；校验器却仍按 Q head 要求字节数，于是拒绝正确的 KV link。另一个 batch decode（合并处理多个请求的一步）分支还使用旧字节口径：同一控制例的一步 KV，按 KV head 应为 **4,096 B**，被捕获的 batch decode 事件却记录 **16,384 B**。这些事件在最终校验前被捕获，整份报告随后报错，不能说实际传输了这些字节。正确的 prefill 字节也会先被旧校验器拒绝，不能把该 decode 事件当成报错的唯一来源。只放宽校验器不能自动消除这处分支不一致。

**AttAcc 本身有没有。** 原始模型没有当前完整 GQA 数据通路，也没有这个新多请求 validator。这是新增路径之间没有同步，不能当成大家共同使用同一常数的近似。GPU attention 内部是否完整优化 GQA 复用是另一项共同模型假设，不在这里要求改进。

**影响什么比较、需要决定什么。** 报错意味着该路径不能产出结果，**不是测到了 PIM 较慢**。batch 多传字节则影响走这条路径的链路成本。**chenyi9 已裁决修复 GQA。** 执行时同时统一校验器和 batch KV 字节口径；本次只写交接，没有修改实现。该问题不等于 MQ 本身不公平。

证据：[实际构图 JSON](archive/model_provenance_8c51672_probe.json) 的 `rung_gqa`/`rung_mha_control`；[validator](../../src/workload_runner.py:2756)、[batch 字节](../../src/workload_runner.py:3474)。旧编号 V01a/V01b、MR-01/MR-02。

<a id="c3"></a>

## C3：A4c 说“另存 diff 行”，地址却没有隔开实际访问的行

**论文声明什么。** [Master and Per-Agent Diff](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex:69>) 与 [A4c 定义](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:81>) 允许 master 和 diff 使用同一 head 的通道，但要求修正集中在独立 diff 行。它们是两份不同的数据；集中后的收益应来自修正布局。

**理解这个 case 只需三个概念。** 通道 channel 是可以并行工作的内存资源；每个通道内有多个 bank，bank 里有 row（行）。AttAcc 的 **ALL-BANK** 命令用一个地址驱动该通道内的多个 bank。因此“两个地址整数不相等”不一定表示两次命令访问不同的一组行。

**具体 case。** 同一通道放一个 master 块和一个 diff 块。新 allocator（分配记录，代码名 PhysicalLedger）给 diff 加了一段很大的地址偏移。按照实际 mapper 解码，两者首条命令是：

| 数据 | channel | pseudochannel 子通道 | row | column |
|---|---:|---:|---:|---:|
| master 原块 | 15 | 0 | 0 | 0 |
| diff 修正块 | 15 | 1 | 0 | 0 |

**应发生与实际发生的区别。** 这个偏移只变了子通道位；ALL-BANK 操作会覆盖两边的 bank，行和列却相同。因此不能靠该偏移证明 master/diff 已存到独立的 ALL-BANK 行。这个检查直接运行命令生成函数并按源代码解码，没有运行 Ramulator 性能实验。

**AttAcc 本身有没有。** 原始 AttAcc 有上述 ALL-BANK 语义；mapper 和相关 action/preq 文件与当前逐字节相同。它没有新增的 master/diff 区分配。**来源不是“原始 mapper 被我们改坏”，而是新地址选择是否符合未改的原语义。**

**影响什么比较、需要决定什么。** A3b 没有这个专门 diff 区，A4c 起使用它。行地址重合可能改变行命中/激活，因而使收益混入布局声明以外的因素；目前没有周期数据能量化或断言一定高估。**chenyi9 已裁决修正 diff 地址分配。** 要验证的是两份数据不占用同一实际 ALL-BANK 行；仅检查地址整数不相等不够。后续档继承该地址规则，本次未改实现。

证据：[命令/解码 JSON](archive/ledger_trace_boundary_8c51672_evidence.json) 的 `diff_master_region`；[新 diff 区](../../src/workload_runner.py:820)、[AttAcc 来源核对](archive/ATTACC_UPSTREAM_REVIEW_EXCERPTS.md)。旧编号 V02。

<a id="c4"></a>

## C4：同一块 V 不应因为这次少读一些 token，就换一套存储步长

**论文声明什么。** [固定 placement](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex:116>) 说写入时决定的 channel/row 在以后扫描中保持；[PIM attention](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex:159>) 说共享 master 的字节保持不动。扫描应从已存的数据中取需要的部分。

**具体 case。** 先放好同一份 256-token V。一次读完整块，另一次只读前 16 个 token，期间没有搬移或重排。V 是一个矩阵，计算不同输出维度时，要按它存好时的步长找到对应数据。

| 读取请求 | 第二组输出维度对应的起点，相对 V base（存储起始地址） |
|---|---:|
| 完整块 | +512 B |
| 仅前一段 | +32 B |

**代码实际做什么。** 新扫描接口用 extent 表示连续读取子段，但 generator 把“本次读多少 token”当作矩阵存储步长。相同的存储对象因此得到不同的维度起点。这里展示的是命令地址，不是测量延迟。

**AttAcc 本身有没有。** 原始 generator 建模整块 dense attention，没有当前“持久对象中任意子段”的接口。这个接口是新增的，同时被多个布局使用；**共同使用接口并不自动证明某一档获得了额外优惠**。

**影响什么比较、需要决定什么。** A3b 与 A4c/A4e 的分段不同，地址/取整行为可能影响扫描成本归因。但本轮没有测出档间净偏置。经过下述重审，按用户既有共同近似规则继续记录；当前不单独提出整改，也不宣称 A4e 的软件表已经不公平。

**本轮重新 audit。** 使用实际的 A3b/A4c/A4e/A5/A6 layout 类各存同一份 master，然后分别读整块、前段、第二段。五档的同输入结果一致：完整块第二组 V 输出起点相对本次 V base 为 +512 B，前段为 +32 B，第二段也为 +32 B。原始 AttAcc 的 `context_mac(addr_offset, L)` 也用 L 算矩阵跨度，但没有当前固定对象的 extent 子段接口；新增接口将本次 extent 长度代入该 dense 公式，却未携带对象原始跨度。这是沿用原公式时的接口边界，不能声称原始 AttAcc 已验证持久对象的子段地址。

**复审结论。** 地址随读取范围变化的事实仍成立；相同输入下它是各档共同的行为，没有证据证明 A4c/A4e 独享优惠。本轮没有数值 KV 校验或周期实验，不能确定对不同分段 workload 的净偏置。**主审按用户既有共同近似口径归类：保留这一边界说明，不单独要求整改或重复裁决。** 这不等于证明物理地址绝对正确；当前没有新增档间独享优惠的证据。新证据：[各档与原始 AttAcc 公式对照](archive/pending_layout_reaudit_evidence.json)。

**按逐 channel、逐行/列计时口径进一步澄清。** 用户所述流程正确：生成器给出需要访问的 row/column，Ramulator 根据当前开行状态决定 ACTAB、PREA 和命令时序，不需要人为增加“步长/跳转费用”。当前 [MACAB 前置条件](../../pim_ramulator_src/HBM3-PIM.cpp:552) 正是这个机制。同 channel 多段 KV 放在同一 trace 中，行缓冲状态由模拟器维护。

前述 V 子段的列起点差异本身不能证明延迟或公平性错误；如果行集合、列命令数量和依赖没有改变，仅数值布局解释不同不足以要求修改性能模型。需要审查的是实际送进模拟器的行、列命令是否漏读、多读或越过存储区间。

独立 agent 进一步跟踪真实 reuse plan → bindings → `_pool_reads` → ledger → generator。A3b 的逻辑 `plan_reads` 虽跳过被修正的 master token，实际 decode 使用的 `reads` 会补回 shadow master；与 A4c 对照时，该完整块仍生成相同的 master 行扫描。不能把绕过 `_pool_reads` 后手工截出的子段反例说成默认 A3b 多付 ACT。通用子段接口可能生成跨行地址是另一个有条件的输入边界，当前仍按共同近似记录，没有证明正式档间净偏差。证据：[实际路径可达性](archive/row_column_reachability_evidence.json)、[物理读取选择](../../src/workload_runner.py:1513)、[decode 调用](../../src/workload_runner.py:3372)。

证据：[同一对象两次读取](archive/ledger_trace_boundary_8c51672_evidence.json) 的 `subsets.full`/`subsets.first16`；[子段起点](../../src/workload_runner.py:1030)、[V 命令生成](../../pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py:218)。旧编号 V03。

<a id="c5"></a>

## C5：A6 按实际需要的操作，使用简单公式估价

**最新裁决。** chenyi9 明确：“C5 就实际找一个公式估算一下啊？需要哪些部分就估算哪些项。”结合此前“Q 传输很容易，可以不考虑”，本轮采用以下交接口径：**Q 传输在选边估算中忽略；没有历史 KV 时删除空回读费用；真正需要的历史回读仍按真实字节估算。** 此前对删除范围的澄清由这条最新指令收敛，不解释成一律删除有用的历史回读。

**审计建议的公式。** 当前实现对每个 request 在首次选边的那一层比较以下 attention 服务时间，后续层沿用该选择，相等选 PIM。它是简单局部估价，不是两套候选 DAG，也不是整个 workload 的完成时间保证。

```text
T_GPU = Link(B_history) + FlashAttention(q, N)
T_PIM = Σ_s max_c RamulatorTime(extents_c, queries_s, MQ配置)
        + Link(B_context)

选择 PIM 当且仅当 T_PIM ≤ T_GPU；否则选择 GPU。

Link(0) = 0
Link(B > 0) = τ_link + max(B / BW_link, B / BW_far_HBM)
             （当前 flash X2G 模型；参数直接使用共同设备配置）
B_history = 2 × R × H_KV × d_head × bytes_per_element
B_context = q × H_Q × d_head × bytes_per_element
```

| 量 / 项 | 从实际路径取什么 | 为什么保留或省略 |
|---|---|---|
| q | `len(compute_positions)`，当前 request 真正新算/修正的 token 数 | GPU/PIM 都要为这些 query 做 attention |
| R、N | R 为 GPU 分支真正回读的未重算驻留 KV 行数，即 `len(readback_rows)`；GPU 逻辑上下文 N = R + q | GPU 已经重算的 KV 不再回读；全新 prompt 的 R=0 |
| H_KV、H_Q | 一个 GPU 的 TP 分片负责的全部本地 KV/Q heads，覆盖其全部 HBM；d_head 是每 head 维度，不能再除 stack 数 | GQA 下 KV 回读按 KV heads，context 返回按 Q heads；与 C2 修复采用同一口径 |
| `FlashAttention(q,N)` | 当前 GPU 分支在 `--gpu-model flash` 下的 score/softmax/context 同一计价接口 | 沿用共同 FA 模型，融合 softmax 按现有模型处理，不另造单价或乘收益系数 |
| `RamulatorTime` | 与真实 PIM 分支相同的通道完整 extents、MQ 参数及实际驻留 query 数；每通道含完整 QK+softmax+PV，通道间取 max，sweep 间求和 | 必须读全需要的物理区间；不能只抽一条 lane 或给尾批套满批价格 |
| s、queries_s | 每 sweep 计算 q_s 个 token；`op.m=q_s`，驻留 query 数 `queries_s=q_s×gqa_size`；按缓冲/batch 限制分组，尾批用实际数量 | 复用现有扫描计价，不套 README 中的假设耗时或读列加速比 |
| Q 输入传输 | 普通 Q 和额外旋转 Q 的传输时间在这套选边估算中按零近似 | 用户已接受忽略，不再要求通过补 extra-Q 费用修复 C5 |
| context 返回 | PIM attention 结果需要返回 GPU，按实际返回字节计价 | 当前 GPU 侧后续投影需要结果；用户未将它裁定为零 |
| GPU 线性计算、新 KV 写入链路 | 两侧相同的操作作为共同项，保持原执行与 pipeline | 本式比较随选边改变的 attention 服务成本；不另做全局调度优化 |
| DIE/TLB、普通 STORE | 按已裁决口径为零额外成本 | 不添加缺乏 AttAcc 依据的费用 |

链路式直接对应 [flash X2G 计价](../../src/devices.py:379)：τ_link 为现有启动时间，BW_link 为当前 `pim_link_bandwidth/2`，BW_far_HBM 为 `far_hbm_bandwidth`。不重新拟合参数；没有传输时不创建空操作。独立 agent 已复核 TP/GQA 宽度、query 数和完整 attention 命令范围。

实际 PIM 扫描区间可能还含被遮罩 master 行等数据，应使用执行分支给出的 extents，不能仅用 GPU 的 N 重建一份更短的扫描。上述 Q 忽略是估价近似，不代表删除 query 数据、GPU 旋转操作或执行依赖。

**现代码要改哪里。** [_resolve_prefill_side](../../src/workload_runner.py:3970) 目前无条件创建 `kv_pim_to_gpu`，flash 链路给空操作带来启动成本；应先判断实际回读字节。其 PIM 候选目前计普通 Q/ctx 链路，extra-Q 不计；按上述统一口径，选边公式中的 Q 输入项应明确忽略，ctx 保留。复用真实分支的形状、GQA 宽度、lane 和尾批；保留逐 request 比较和 PIM tie-break。当前没有执行这些修改。

**论文与 codebase 的交接位置。** 论文 [05-execution 的选边段](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/05-execution.tex:28>) 当前主要是文字，仍描述构造双候选 DAG；执行 agent 应同步为上述简单估价原则，若添加公式，零字节项必须为零、Q 输入项标明忽略、有历史时保留真实回读。代码对应上述选择器；仓库 [设计阶梯的现状公式](../../docs/README_design_ladder.md:134) 也需同步。本次未修改论文或这些实现公式。

**AttAcc 本身有没有。** 原始没有 A6 选择器或 query 旋转复用；这里的选择式是新增逻辑。链路/GPU 采用共同设备模型，PIM 采用 Ramulator 返回周期，不引入新拟合单价。Q 传输差异已按用户裁决接受，不再列成待修的公平性指控。

**怎样审查执行结果。** 全新 prompt：`B_history=0`、回读费用为零；有驻留历史：只按实际需搬的 KV heads 字节计回读；位置偏移产生 extra-Q：按接受的 Q 忽略近似处理；MQ 尾批/多通道：实际数量计价。核对这些组成项即可，不要求构建复杂调度器或用硬件加速比拟合选边。

旧反例仍留在 [操作覆盖证据](archive/reaudit_8c51672_evidence.json) 的 `A6_estimate_vs_executed_operation_coverage`，用于追溯“空回读为何要删”和“extra-Q 漏项为何撤回”，不把设备桩时间当成 GPU/PIM 性能。旧编号 V06a/V06b。

<a id="c6"></a>

## C6：A2 已提供结果，按 tier 汇总时却被漏掉

**论文声明什么。** [七档 comparison](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:65>) 包含 A2；[Metrics](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:172>) 要按统一指标比较各档。tier 是一组同阶段的 request，tier 表是展示结果的一种汇总。

**具体 case。** 给 collector 两份小 JSON，分别标为 A2、A3b。A2 只有 GPU attention，没有 PIM batch；另一档有 PIM batch 记录。输出的 tier 表只出现 **A3b**。

**代码实际做什么。** collector 用“有没有 PIM batch”找应该输出的 tier，因此跳过了 A2；部分时间列又从 attention 开始时刻推导，不是完整 request 的结束时刻。这里的 JSON 是受控汇总输入，不是新跑出的实验结果。

**AttAcc 本身有没有。** 原始没有这个七档、多 tier 汇总器。问题在新增报表逻辑，不属于共同硬件近似。

**影响什么比较、需要决定什么。** 缺行会使读者看不到 A2 的同口径对照；它不等于原始模拟没跑，也不否定已经正确的 `makespan_s` 或 step-weighted TBT。**chenyi9 已裁决更新报表。** 执行时补齐 A2，并核对相关时间列的指标范围；本次未改 collector 或已有结果。

证据：[汇总输入和输出](archive/model_provenance_8c51672_probe.json) 的 `summary`；[collector](../../experiments/collect_dag_ladder.py:119)。旧编号 V08/MR-04。

<a id="c7"></a>

## C7：同一 round 可以紧排；跨轮须保留输出/新写入的分隔

**chenyi9 已明确裁决。** 同一 round 有多组修正 diff，A3b 可以按自己的排列方法正常追加、紧凑存放。多轮之间已有输出或其他新 KV 写入时，A3b 不得把两轮修正重新当成同一批聚合；这正是 A4c 跨轮 diff 区要解决的不足。不能为了让 A4c 显得更快，强行把同轮每组 diff 都拆成独占行。

**撤回旧反例。** `前缀 → D → own → D` 的旧控制只有一个 request、一次 prefill。预约表按 fingerprint 聚合、ledger 将两组修正紧排，确实发生；但该轮内的 QKV/写入是整批提交，`own` 在上下文中出现，不等于发生了“前一轮已输出、下一轮再写”的时间边界。因此它不能证明 A3b 违反跨轮规则，原“是否允许同批分组”的待定项撤回。

**重新核对公开的两轮输入。** 独立 agent 经 `load_workload` 载入两个不同 request，以 parent/parent_out 和 history_len 表达前后关系；两轮复用同一 D，结果如下：

| 轮次 request | D 的修正 token 数 | A3b diff 对象的 owner |
|---|---:|---|
| `A_t0` | 8 | `A_t0` |
| `A_t1` | 8 | `A_t1` |

两个 owner 对应两个独立 burst/extent，A3b 没有将其折成同一个修正对象。公开 loader 拒绝重复 request_id，因此手工把相同 owner 的两轮预约塞进一个字典，再观察合并，不能当成这个公开输入的反例。证据：[两轮公开输入与 helper 边界](archive/independent_round_boundary_evidence.json)。

**AttAcc 来源与当前判断。** 原始没有这类多轮共享 KV baseline；按用户定义审核新增逻辑即可。本轮没有发现上述公开两 request 路径跨轮合并 diff，C7 的同轮指控关闭。这里证明的是对象未合并，尚未证明后续扫描继承所有旧物理位置；后者是下面 C8 的实际覆盖问题。

<a id="c8"></a>

## C8：多轮 workload 没有完整保留“旧 diff 散落、以后继续扫描”的布局

**与论文声明的关系。** [A3b/A4c 定义](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:75>)、[per-agent 跨轮修正](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex:85>) 和用户最新裁决都依赖一件事：旧轮的 diff 仍以原布局存在，以后的 request 会继续扫描它。这样才能呈现 A3b 的跨轮分散与 A4c 的集中存放之差。

**Case A：interleaved 是布局代理，不是逐轮执行。** [gen_sweep](../../workload/probe/gen_sweep.py:95) 把多个 round 的 doc/own 段放进一个 request；[gen_multiround](../../workload/probe/gen_multiround.py:68) 也采用这种展开，且标为 `NOT evidence-grade`。它能作为预设多轮末态的静态布局控制，但不能仅凭段落交错就证明执行发生了多轮输出和再次 prefill。两个现有 B0 JSON 的只读摘要是：

| 编码 | meta 声明 round 数 | worker w00 的 request 数 | w00 总 decode token 数 |
|---|---:|---:|---:|
| `interleaved` | 8 | 1 | 256 |
| `turns` | 8 | 8 | 2,048 |

两种编码的执行工作量也不同，不能把它们的 E2E 差异直接解释为只改了 diff 布局。此表没有运行这些 workload，也没有声称七档在同一编码内用了不同输入。

**Case B：turns 有逐 request 边界，但 history 把旧布局抽象掉了。** [turns 生成器](../../workload/probe/gen_sweep.py:104) 将每轮的旧 prefill 行数累计进 history_len。当前 [_history_tlb_rows](../../src/workload_runner.py:3250) 为新 request 找的是 `新request::history` 的 master 占位，并不回指旧 request 的 master/diff 对象。真实两轮小输入中，第二轮 history 的 528 条 binding 全部是新 owner `A_t1` 的 master；前一轮 diff 出现在第二轮 bindings 中的数量是 0。A3b 和 A4c 得到相同的这种抽象。

这不是发现第二轮没有计算历史 attention，而是历史长度被计入、历史原有的碎片位置没有保留。旧 diff 的额外散落行因此没有作为旧 diff 再次进入后续扫描。该路径不能证明论文所说的完整跨轮布局收益；也不能靠同轮的聚合例子补上这份证据。

**AttAcc 本身有没有。** 原始没有当前跨 request 的软件 KV 复用和旧 diff 持久布局；history_len 是新增共同抽象。共同近似按用户口径可接受，所以这不是自动要求重做调度/执行器。C8 单独列入执行要求，是因为该抽象没有保留用户已明确要求继续引用的旧 master/diff 对象，也是 A3b→A4c 所声称的一个收益来源。

**已明确的执行要求。** 按 chenyi9 的 a/b/c/e/f 例子：第一次复用产生 c 后，a/c 保持各自地址；中间的 b/e 或后续输出不会使其消失；之后 f 扫描时仍引用原 a/c。用于跨轮 claim 的输入与绑定应保留这一对象关系。history_len 可以继续记录长度，但不能将已知旧 master/diff 的物理身份替换成新的普通 master 占位。执行时对齐这条语义，保持 A3b 同轮正常追加、跨轮不重新聚合，A4c 才有独立 diff 区的跨轮紧排。

**验收要看什么。** 下一轮绑定和 trace 能追溯到上一轮实际 master/diff；中间输出按真实追加规则保留；扫描地址对应这些持久对象，ACT/换行成本由 Ramulator 处理。主审本轮只记录目标与证据，没有修代码，没有测出净加速变化；不要求用人为惩罚同轮 A3b 来体现收益。

证据：[两轮 binding 与 B0 JSON 摘要](archive/independent_round_boundary_evidence.json)、[独立 probe](archive/independent_round_boundary_probe.txt)。只调用公开 loader、reuse plan、allocator/bindings 并读取 JSON，没有运行 Ramulator 或性能实验。

## 其他边界：已说明，不当作新的必须修复事项

| 场景 | 与论文的关系 | 按当前口径怎样看 |
|---|---|---|
| 同一 stack 的 KV heads 多于 channel，head 映射绕回同一地址 | 关系到论文 LLaMA-7B 等配置及每 head 独立 KV | 新 ledger 的共同边界；未证明某档单独获益。原 V04 按共同边界记录 |
| JSON 预约次序与执行次序不同；STORE 元数据未与 scan ledger 统一 | 论文称写入时确定 placement | 按共同静态预分配近似记录；涉及跨轮旧对象引用时遵守已明确的 C8。STORE 已零成本，不因字段未统一就认定额外性能惩罚。原 V05a/b |
| 不同 agent 的修正交错预约，同一 agent 的 diff 被隔开 | 对应 per-agent 跨轮集中 | 条件性 case，当前默认静态构图未证明必现；原 V05d |
| pipeline 开启后，后来已就绪的任务仍错过空闲窗口 | 对应跨设备并行；当前所有档共用调度器 | 有结构反例，但按共同近似不要求直接换调度器。AttAcc 是公式 overlap，没有同型 DAG（带依赖关系的事件图），不能声称代码完全一样。原 RC02 |
| 旧报告缺 GPU/pipe 字段、缓存缺完整库指纹 | 关系到比较是否用同一配置 | 未证明实际分档混配置/版本，不仅因元数据缺失判不公平。原 RC03、V07 |

AttAcc energy 单价、QK/PV ALU 记账、ACT 摊销、整 stack 复制、共同 FA 近似、history/人工 workload，以及已统一的 STORE/DIE/TLB 零成本继续接受。原始依据见 [共同口径记录](archive/ATTACC_RELATIVE_FAIRNESS_REVIEW.md)。这些不再挤进当前整改清单。

## 证据入口与已关闭事项

本页数表由脚本读取审计 JSON 自动生成；贡献对照另见 [主审证据](archive/contributions_alignment_evidence.json) 和 [独立证据](archive/independent_contributions_evidence.json)。原 C1–C7 的提取结果在 [current_case_facts.json](archive/current_case_facts.json)，每个 case 均链接原始证据。主审与独立 agent 的历史全文在 [archive 索引](archive/README.md)。没有用旧严重级别替代用户裁决，没有证明有人故意削弱 baseline。

高层 scan 地址持久、共同随机修正计划、STORE 零收费、A1 GPU prefill/精确长度、owner 先写依赖、MQ 接入、A6 全 lane/尾批估价等已确认的修复，不再重复列为当前问题。当前以顶部裁决清单为准：已接受的 Q 近似不重列问题；已确定的修复与 C8 持久引用要求一并交接；C4 按共同近似记录，C7 同轮合并不再算问题。整理原因见 [session](../../docs/sessions/2026-09-05-audit-docs-cleanup.md)。

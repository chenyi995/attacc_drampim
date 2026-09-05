# Fugue / AttAcc 独立审计：实现、消融公平性与 workload 偏差

审计日期：2026-09-05。对象：`attacc_drampim_822`，HEAD `8750b5b`；以 `c600051` 为代码变更参照。论文：`/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027` 当前工作区文本。

后续代码修正：按用户确认的 AttAcc 计量口径，已删除 DIE 旋转/position-transform，并排除各档新增 DIE/TLB latency、energy 与资源排队。105 个测试通过。具体来源核查与仍未修复的 PIM 读写计价差异见 [补充报告](PIM_TIMING_PROVENANCE.md)。下文其余反例与 `evidence.json` 为初始审计快照。

**口径更新（用户确认）：A1/A2 是独立 baseline，不要求它们之间或到 A3b 为单变量；从 A3b 起按 claim 增量。A5 的 PIM prefill + MQ 组合可以接受，自行构造合理的 baseline/workload 展示收益也可以接受。具体修改按 [修改建议 README](../../docs/README_audit_fixes.md) 执行。** 下文保留代码证据；不能把上述已接受的实验设计本身当作违规。

**修订后的审稿结论：当前 A3b 起的部分实现仍有超出 claim 的差异，以及不符合各档定义的执行错误，需要修正模型后重跑。** 包括基线执行设备错误、输入重算集合随档位变化、非持久物理布局、缺失数据依赖等可复现问题。此判断针对实现与证据有效性，不否定已确认的七档设计，也不等于证明 Fugue 的设计没有收益。

**关于高估还是低估：不能给所有指标一个统一方向。** 有明显抬高 A6 相对 A1/A2 收益的风险；共享率、同步到达、未来 co-read 信息也使部分实验偏乐观。但一些 workload 不产生位置修正、缺少真实跨轮 diff 积累，且实现缺少论文所说的流水重叠，会压低特定机制的边际收益。净影响必须通过修正后的配对实验计算，不能用“保守估计”概括。

## 1. 审计范围与证据边界

- 审查变更集全部 36 个文件的作用和关联调用链，重点从 CLI / preset → workload / reuse plan → 地址分配 → GPU/PIM 算子 → trace / 调度 → 汇总指标追踪；变更规模为 +14,739 / −195 行。文件级覆盖见文末。
- 对照论文设计、执行、方法和评估章节及图表数据出处。读取了论文指向的另一个仓库中的 workload 和旧 runner；下文始终将其标为**外部输入**，不冒充本仓库随附材料。
- 运行 `python3 -m unittest discover -s tests`：**102 个测试通过，83.278 秒**。日志见 [unittest.log](unittest.log)。这些测试通过不能抵消下列反例。
- 新增独立 [reproduce.py](reproduce.py)，调用真实 planner、layout 和 DAG 构造器，用记录算子形状的设备桩隔离结构问题；结果为 [evidence.json](evidence.json)。桩上的秒数只用于验证先后关系，**不是硬件性能测量**。
- [manifest.json](manifest.json) 记录 revision、被审查变更文件、论文文本及外部输入的 SHA-256，便于区分后续版本。`evidence.json` 的 `paper_contract_checks` 汇总 8 项独立约束检查；本次均为 false。
- 没有修改生产代码，没有运行有破坏性的初始化脚本，没有将现存旧二进制当成已验证的新模拟器；没有完成 Ramulator 干净重建或论文 588 点矩阵重跑。未对外部 RTL 做逐行审计、综合复现或数值精度验证。不能宣称仓库和论文的“每一处”都已被实验性证明正确。

复现结构证据：

```bash
PYTHONPATH=. python3 audit/2026-09-05/reproduce.py > /tmp/fugue-audit-evidence.json
python3 -m unittest discover -s tests
```

论文约束主要来自 [方法章节](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:68>)、[运行时章节](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/05-execution.tex:41>)、[物理布局与查询变换](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex:75>)。

## 2. 相邻两档是否只改一处？

| 比较 | 论文定义 / 应隔离的因素 | 实际审计结果 |
|---|---|---|
| A1 / A2 | 独立硬件侧、软件侧 baseline | 接受多因素差异，不要求单步消融。仍需修复 A1 实际 PIM prefill 却报告 GPU 的不一致。 |
| A3b | 最 naive 的软件复用 + PIM decode 组合 | 接受作为后续增量的起点，不要求相对 A2 单变量；仍需保证算子形状、物理存储与时序正确。 |
| A3b → A4c | 保留 master 放置，只将每个 agent/head 的 diff 紧凑存放 | **不成立。** 无 diff 时 master 分布仍变；默认 runner 下重算 token 集合也变。 |
| A4c → A4e | append-order → co-read placement table | preset 层面基本只改映射策略，但表使用完整未来 workload，且大 segment 未按 256-token chunk 分配。 |
| A4e → A5 | prefill 放入 PIM 并使用 MQ | 接受机制包，无需强制拆档；配套频点与 decode MQ 收益应明示。仍需修复 fresh prefill 强制 GPU 的执行偏差。 |
| A5 → A6 | event-based per-prefill placement | preset 只改模式，但执行的是静态时长求和比较，不是候选 DAG 的完成时间比较。 |

依据：[PRESETS](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ablation.py:107>)、[runner](</data2/chenyi9/KV-PIM/attacc_drampim_822/experiments/run_dag_ladder.sh:58>)、[入口 canonicalization](</data2/chenyi9/KV-PIM/attacc_drampim_822/main.py:465>)。表中采用用户确认的 claim 粒度；已披露且被接受的机制组合本身不构成不公平。

## 3. 阻断性能归因的发现

### F01 · P0：A1 的 physical DAG 把 prefill attention 放到了 PIM，报表却记成 GPU

[workload_runner.py:4179](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:4179>) 先增加 `side_rows["gpu"]`，随后调用 [3565 行的 helper](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:3565>)。该 helper 明确生成 Q 链路传输、PIM 全范围扫描、context 回传；没有 GPU prefill score / softmax / context。

256-token 单请求反例：报表 `{gpu: 256, pim: 0}`，事件却有 `PIM:pim_kv_scan_score_softmax_pv`，没有 GPU prefill attention。这直接违反论文 A1 “every prefill attention on the GPU”。分析引擎中的 A1 或 upstream-equivalence 测试正确，不代表 physical DAG 的 A1 正确。

此外，scan 只依赖 Q 的 address plan，不依赖新 KV store；store 还在 scan 之后构造。这让 A1 同时有不应存在的全量 PIM prefill 开销和不合法的读写顺序。前者对长 prefill 有抬高基线、夸大 Fugue speedup 的明显风险，后者方向相反；没有重跑不能给出净倍数。

### F02 · P1：A3b → A4c 在零 diff 时仍改变 master；A3b 同一数据随读集合变通道

[通道构造器](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:811>) 中，A3b 的 `slice-append` 根据**本次 scan 的 unit 序号**轮转；A4c 的 `master-diff-local-append` 则查询持久 `chunk_slot`。

反例：按序写入 `c0…c4`，每块 256 token，每个 head 有 4 个通道，然后仅读 `c0+c4`，没有任何 diff。

- A3b：每个 head 的 slot 0、1 各 256 token。
- A4c：每个 head 的 slot 0 为 512 token。
- A3b 单读 `c4` 又将它放到 slot 0；和 `c0` 一起读时变成 slot 1。

因此当前 A3b 不代表论文的“写入时 append-order 轮转且之后不搬动”的物理基线。这个反例甚至会使 A4c 比 A3b 差，说明不能把任何方向的差值归因于 diff packing。

### F03 · P1：默认 ladder 改变了修正 token 的身份，且不是论文声称的 EPIC 配置

[run_dag_ladder.sh:59](</data2/chenyi9/KV-PIM/attacc_drampim_822/experiments/run_dag_ladder.sh:59>) 对非 A1 使用 `--reuse recompute`。而 [main.py:472](</data2/chenyi9/KV-PIM/attacc_drampim_822/main.py:472>) 按 `kv_mapping != naive` 开启 canonicalization；[planner](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload.py:473>) 将随机 k 个位置改为前 k 个位置。

相同 workload、seed=0、k=8：A3b 修正 `[20,132,155,197,207,215,244,248]`；A4c 修正 `[0,1,2,3,4,5,6,7]`。这不仅改变地址，也改变了被修正的数据语义、碎片形状和 mask 分布。随机位置更易碎片化，而连续前缀更规整，存在有利于后者的偏差。保持相同 k 不足以保证公平。

必须在选档位之前生成一次不可变 ReusePlan，各档只转换其物理表示，并比较实际修正索引的 hash。论文 [方法章节:166](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:166>) 写的是 EPIC k=8，不能用当前分档变化的 `recompute` 默认为等价替代。

### F04 · P1：fresh prefill 绕过 A5/A6 放置；GPU context 形状错误

[4237 行](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:4237>) 的 `if not reusable` 对所有档位直接执行 GPU prefill，不看 `pim_prefill_mode`。所以 A5 不是“every prefill in banks”，A6 也不会为这部分请求作选择。

同一路径 [4263 行](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:4263>) 给 score、softmax、context 统一赋 `m=n=L`，没有更新 context 的 `k`。实测 L=256、dhead=128 时 context 是 `(256,256,1)`，应为 `(256,128,256)`；该 context 算子的 FLOPs 少算 128 倍，**不是整个请求快 128 倍**。A3b–A6 的 fresh 分支均受影响，A2 的独立构造路径不受这个具体错误影响。

decode 还为最新 token 生成本地 GPU attention 和 LSE merge；本地 context 也有 `(1,1,1)` 的形状问题。故“decode 全在 banks”也不是严格执行的算子分工。需要统一算子工厂和形状不变量，不能靠事件名称推断设备和计算量。

### F05 · P1：A6 用孤立成本求和，没有比较事件完成时间

[4354–4426 行的决策](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:4354>) 计算 `t_xpu = readback + GPU attention`、`t_bank = 最多行通道的 scan × sweeps + TLB + Q/context transfer`，然后比较二者。它不查询当前 GPU、LINK、DIE、PIM 的可用时间，没有构造并试排两份候选依赖图，也没有以预计 completion 为目标。

估计还忽略部分 DRAM read/store、query variants、score assembly 和资源等待；按行数最大不一定选到碎片最多、耗时最大的通道；尾批用满批乘法也可能多算。选择结果按 request 缓存跨层复用，未按层内重算集合变化重估。因此不能声称实现了论文 [event-based placement](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/05-execution.tex:41>) 的较早完成选项。误选方向不固定。

另外，复用分支在选择 GPU/PIM **之前**已经生成 `q_gpu_to_pim`，GPU 路径也承担不需要的 Q 传输；这会偏向让 PIM 看起来划算。PIM prefill 使用整请求 QKV 和整块 KV store 依赖，未实现论文所述“前段扫描与后段生成/传输重叠”。

### F06 · P1：共享 KV 可以先被消费、后被生产；调度器还把重叠物理资源当成独立资源

[预分配](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:2798>) 为整个 workload 先建立所有位置；共享数据的 owner 按 planner 顺序选出，执行却按 workload 请求顺序走。同 tier 消费者缺少到 owner KV materialization 的依赖边。

把两个共享文档请求的输入顺序反转，owner 为 a、先构图的是 b：设备桩下，b 的首个共享 scan 在 **5.09 µs 结束**，a 的 master store 到 **17.65 µs 才开始**。当前校验没有阻止它。这里的时间只证明不合法先后顺序。

[调度器:2213](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:2213>) 用字符串作资源身份：`PIM`、`PIM:pool0-14`、`PIM:pool0-0` 三者独立。三条各 1 秒事件，无依赖时都从 0 秒开始、1 秒结束，即使代表的 channels 重叠。生产代码确实混用这些命名，store 与 scan 因而可能不合法并发。需要按实际 channel/bank 集合做冲突检测，并独立校验读前写入。

与此同时，`previous_tier_done` 是整层 barrier，decode 构造有固定 step/layer 循环，原始 arrival timestamp 不参与就绪判断。这又压制部分真实跨 tier prefill/decode overlap。不能把错误并发与过强 barrier 当成互相抵消。

### F07 · P1：placement table 使用未来读集合；存储粒度、行地址和生命周期不完整

[预处理:2870](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:2870>) 在执行前把所有请求的 co-read 集合交给 [chunk table:1533](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:1533>)。写入序列 `[a,b]` 不变：无未来 reader 时 a、b 都在 slot 0；加入未来同时读 a、b 的请求后 b 改为 slot 1。这是完整 workload 知识。

如果论文明确限定整张 DAG 和未来 chunk 身份在写入时已知，该假设可以成立；对在线 agent/tool 请求则不能默认成立。应把结果标为已知未来 co-read 的上界或增加只使用当前已知 hints 的实验。

论文规定 **256-token chunk**，实现却按整个 fingerprint 决定 slot：一个 1024-token segment 的四个 256-token units 全在同一 slot，外部 Mooncake 的 512-token block 也会遇到这一问题。这削弱实际可用并行度，使结果依赖输入分段格式。

[物理地址构造:789](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:789>) 每次从 cursor=0 按扫描长度重建行地址；不是统一的持久 `(chunk, block_index) → (channel,row,offset)`。TLB 的 master/diff pool 分配与 scan 的 per-head local 布局还属于两套地址模型。diff 长度被汇总后再连续摆放，缺少真实洞、逐轮分配、释放、压缩搬移与空闲容量约束。不能仅凭 trace 中有地址就认定实现了论文的持久物理存储。

### F08 · P1：CacheCraft / CacheTune 的 A6 实验没有走同一 physical DAG

[run_reuse_prefill:4745](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:4745>) 只把 `cacheblend / epic / recompute / no-reuse` 分派到完整路径；`cachecraft / cachetune` 落入 legacy split-prefill。

同一两请求例子：EPIC A6 有 47 个 decode 事件并返回 makespan、energy、placement 等；CacheCraft / CacheTune 均为 **0 个 decode 事件**，返回仅 5 个字段，没有相同端到端指标。CLI 的 `--engine dag --ablation A6` 标签不能弥补这个 dispatch 差异。论文声称的跨 policy A6 比较目前不成立。

此外，CacheBlend 以随机抽样代理重算，CacheCraft / CacheTune 也只有计数/前缀启发式，没有相应模型状态、质量测量和完整上游实现。可以研究给定修正集合的系统成本，但不能据此认定论文算法的质量—性能权衡被复现。

## 4. 成本、模型与硬件一致性

### F09 · P1：GQA 只接入了部分模型；GPU、KV 传输和容量口径不一致

[model.py:104](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/model.py:104>) 仍以 `3 × hidden` 构造 QKV projection。LLAMA3-8B 当前模型几何需要 `4096 + 2×8×128 = 6144`，实际宽度 **12288**。PIM 复用路径的 KV bytes 多处仍是 `2 × local_hidden`，没有除 GQA group；A2 某些路径已除 group。对于 group=4，这部分 PIM 路径 KV 流量高估 4 倍，可能低估其通信优势。共同的 QKV 多算还会稀释 layout 的端到端收益。

[analytic memory report](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ablation.py:1119>) 也按 dense heads 计 KV bytes；physical DAG 没有相应完整峰值 KV 驻留统计。论文若以 DAG 为唯一指标来源，不能拿这份分析引擎数字补充而不解释口径。

### F10 · P1：GPU/stack 数量的能耗与容量倍数需要纠正

[make_pim_config:265](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/config.py:265>) 默认 `num_attacc=8`，main 随 `--ngpu` 改系统配置时没有相应传入；[PIM device](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/devices.py:410>) 又按它乘能耗和容量。1-GPU、2-GPU 点会带着 8 的复制因子，不能作为已匹配平台的能耗证据；`num_hbm` 与该因子还须明确是全系统还是每 GPU 口径。

每 stack 的 head 数按最忙 stack 向上取整，然后复制到所有使用的 stacks，残余 head 的能耗可能多算。例如 26 个 local heads / 6 stacks 按 5×6 而非 26 计，差 15.4%。这不是说 latency 也应直接乘同一个比例，而是指出能耗不能复制最忙 stack 的负载来代表所有 stacks。

### F11 · P1：RoPE / query variants、全局 softmax 与真实数值语义之间有缺口

论文 [178 行](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex:178>) 规定依据 query position 与 chunk base 产生每个 chunk scan 的 Q variant。代码 [_prefill_location_deltas](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/workload_runner.py:2612>) 主要读取显式 `position_delta`，未从实际拼接位置与 chunk base 完整导出；默认合成输入没有这些非零标记。不同 chunk 基址不应仅因元数据为 0 就免去变换/传输。GPU rotate 还没有独立计算成本。

扫描事件以 `score_softmax_pv` 为整体定价，prefill 的 `die_score_assembly` 却依赖完整 scan 完成；这没有显式表达论文“master/diff 合并 score → 一次全局 softmax → P 路由 → PV”的次序与代价。decode 的部分路径则使用 LSE 合并。当前是时序代理模型，不能用事件命名证明所述数据通路或数学等价性。

最低限度需要两个不同验证：一是固定 master/diff 数据下，对 dense attention 的数值等价；二是所选 reuse/recompute policy 相对完整模型的输出质量。前者通过不能证明后者。

### F12 · P1/P2：activation、softmax 和 MQ 能量缺乏完整证据链

控制器新增 activation 计数，但 [wrapper](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ramulator_wrapper.py:459>) 解析的是 MAC、SFM、数据移动和 cycles，并未接入新增 `pim_activations` 来计算 energy；`layout_probe` 的 ACT 数是布局公式估计。不能把报告中这些量称为实际命令计数驱动的 ACT 能耗。

MQ 窗口预算使用的 `MQ_EOP` 来自既有 energy table，尚无本仓库证据证明它与论文 RTL 不同频点、softmax 和外围能耗闭合。shared GQA query 数与 Layer FLOPs 的关系也未贯穿 energy 路径。论文的面积、时钟和功耗曲线需要外部 RTL 报告与参数映射；本次未重做其综合，因此将结论标为**未验证**，不擅自判定面积数字错误。

同时有几项不能作为“不公平改硬件”的指控：512-B GEMV buffer 没有在 A5/A6 偷增到更大；MQ 上限 8 slices 和时钟/窗口约束在代码中有明确模型。新 turnaround 修正、全档统一 power limit 本身是共同修正。应审计参数校准和 A4e→A5 的成组变化，而不是只看新增行数。

### F13 · P1/P2：A2 是受限的远端 GPU baseline；A1 还有额外 padding

A2 每 token / layer 回读全段 resident KV，GPU 执行 attention；这与论文明确描述的 remote dumb memory baseline 大体相符。按用户确认的口径，它可以作为独立的简单 baseline，不要求实现最优软件系统。GPU-local reuse、分层缓存和更细流水属于可选补充。当前粗聚合回读/计算会影响 baseline 的成本，应如实说明，但不再仅凭它与 A3b 系统策略不同就判定违规。

A1 的连续扫描还把短长度按 256-token 粒度向上取整，而部分共享路径使用实际 extents。必须说明这是物理最小扫描还是仅由 generator 导致的额外工作，并在各档统一。称这种基线额外开销“conservative”不能说明对 Fugue 的 speedup 保守。

## 5. Workload 是否夸大收益？

### 5.1 输入可追溯性存在版本断层

本仓库跟踪的 workload 只有两个 test fixtures，README 也说明实验 workload 不随分支提供。论文图表 README 指向外部 `attacc_drampim_xinyao`；其中旧图表实验是早期九档/54 点版本，并明确标过后续修复尚未反映。不能作为当前七档、6 models × 14 workloads × 7 rungs 的 588 点证据。论文评估性能段落目前多为描述，不能把旧数值当成已完成的当前论文结果。

外部 `workload/gen_sweep.py` 及现存 JSON 默认 **N16/C16/D2**、C 轴为 8/16/40；论文 [109 行](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:109>) 是 **N16/C32/D2**、C 轴 16/32/64。外部 generator 注释还说明为机制展示调整过 C、避开特殊平衡点，metadata 写有 `mechanism illustration; NOT evidence-grade`。机制 microbenchmark 可以这样设计，但不能未经说明变成代表性总体收益。

外部旧 `experiments/paper_ladder/run_matrix.py` 对若干 native-history workload 强加 `--history-len 3`，且 ladder 未显式选 DAG；某些 dynamic 配置还用过 2.6 GHz / 768 B。它们与当前论文配置不同。本仓库当前 runner 显式选择 DAG，这一点是正确的；问题是旧来源与新实现尚未形成同一可复现链条。

### 5.2 用当前 parser / EPIC k=8 重新统计外部合成输入

下表 reuse 比例是 `plan.reused_tokens / sum(request.total_length)`；history 单独统计，比例不等于实际峰值物理 KV 节省，也不等于全部真实计算可免除。

| 外部输入 | 请求 / tier | 输入 token | planner reused | 修正 token | 特别发现 |
|---|---:|---:|---:|---:|---|
| N16/C16/D2 all-to-all | 32 / 2 | 208,896 | 192,256，92.0% | 3,016 | tier 0 修正数为 0 |
| N64 | 128 / 2 | 1,622,016 | 1,568,512，96.7% | 24,328 | N 增大同时增强共享率与同步可批量性 |
| N4 | 8 / 2 | 39,936 | 32,512，81.4% | 568 | 不是只改变并发数量而保持共享结构强度 |
| D1 | 16 / 1 | 73,728 | 65,280，88.5% | **0** | 根本不测试 diff 修正收益 |
| private corpus | 32 / 2 | 208,896 | 65,280，31.25% | 968 | 仍共享 task / 上轮输出，不能称零复用 control |

所有合成输入的同一 tier 内，同 fingerprint 的文档偏移只有 **1 种**：agent 的前缀长度与 corpus 顺序一致。因此“每个 agent 的 corpus 都因位置不同而修正”不符合这些实际输入；baseline 的第一 tier 没有 diff。broadcast 和 pipeline 中少量 corrections 主要来自父节点输出，而非 corpus 的位置重排。

还有四个必须区分的偏差：

1. **偏向高估共享/MQ 收益：** 所有 agent 读取相同 corpus、无 arrival 时间、同 tier 集中到达，固定 256-token 输出/文档，提供大量相同 KV 和整齐的批次。N 轴因此混入共享率、上下文长度和 batching 的联动，不能单独解释为并发扩展性。
2. **压低或未测试 diff 边际收益：** corpus 在尾部连续出现，同 tier 位置相同，很多修正为 0；每个 request 的修正集中一次产生。D 增大不等于同一持久 agent diff 在真实执行中逐轮碎片化。多轮 adapter 把对话拼成一个请求的情况也不等于真实多次 append / detach。
3. **偏向有利的整齐几何：** 256-token 段正好匹配宣称的 chunk/row；缺少非整除长度、稀疏 corrections、多次 detach、channel 容量不平衡、热冷文档、低共享/不同到达过程。另一方面当前大段同 slot 的 bug 又会压低真实长段并行度。
4. **需要保留不利模型点：** 某些 head/HBM 组合每 head 只有一个 channel，placement table 根本无选择空间；例如当前 7B 配置。它是合法边界，不应为追求图中收益而删除，也不适合证明 table 对所有模型都有收益。

private **corpus** 的设置本身可以合理；不合理的是把它解释成完全 private / zero-reuse 系统。保留共享 task 和通信 DAG 后，31.25% planner reuse 是预期可出现的。

### 5.3 外部真实输入也不能统一视为“保守”

| 外部输入 | 请求数 | 输入 / reused | 输出总数 | history 总数 | 审计解释 |
|---|---:|---:|---:|---:|---|
| Mooncake toolagent n40 | 40 | 479,301 / 85,504 | 10,202 | 0 | 只有 1 tier、0 corrections、未保留 timestamp；测试共享前缀而非跨位置修正 |
| Mooncake multi-turn | 43 | 119,100 / 17,510 | 19,055 | 450,148 | timestamp 留在 raw 中但 engine 不使用；旧 runner 的 history=3 会将其缩到 129 |
| MultiHop-RAG n32 | 32 | 212,785 / 43,096 | **75** | 0 | 平均每请求仅 2.34 输出 token，decode 收益在端到端中被显著弱化 |
| ShareGPT 10 conversations | 52 | 12,923 / 11,308 | 14,414 | 22,172 | 无 doc 类型 corpus，主要是对话历史/父回复；旧 history override 会缩到 156 |

这些是**当前 parser 对外部文件的静态统计**，不是原始服务流量分布，更不是本分支已经完成的完整评估。

Mooncake 官方说明其 hash 对应 **512-token prefix blocks**，真实 trace 包含相对毫秒 timestamp，并建议按时间重放；相同 prefix hash 可支持前缀复用判断，不能仅凭它证明任意位置的同内容 chunk。当前 toolagent 子集去掉时间、同步构图，显著改变 batching / queueing 条件。[Mooncake FAST25 原始说明](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)

MultiHop-RAG 的极短答案确实可能是任务特性，不能为放大 PIM decode 收益随意拉长后声称是真实数据。应同时报告原生输出长度和明确标注的长度敏感性实验。跨模型使用同一 tokenizer 的 token 计数也须披露，不能视为每种模型的原生 token 工作量。

**workload 判定：共享与同步条件整体偏有利；diff / position-independent / 多轮存储机制的覆盖不足，部分条件压低其边际收益。两者可以同时成立。现有实验不足以判定真实部署平均 speedup 是高估多少或低估多少。**

## 6. 可复现性、报表与测试

### F14 · P1：当前 checkout 不能证明 trace generator、补丁与二进制匹配

`ramulator2` 是 gitlink，但当前目录没有自己的 `.git`，也缺少应安装的 `src/dram/impl/HBM3-PIM.cpp`、`trace_gen/gen_trace_attacc_bank.py`，只存在无法绑定到当前补丁版本的旧二进制。未完成干净重建前，不能用现存缓存认定新 timing / activation 修改已生效。

既有 `set_pim_ramulator.sh` 没有充分校验子模块身份，直接 `cd ramulator2; git reset --hard ...`。在当前缺少子模块 Git 元数据的目录结构下，Git 可能向上找到父仓库；**本次没有执行该脚本**。需先修复子模块初始化与身份检查，再建立锁定 revision 的构建步骤。

持久 PIM run cache 的签名没有把 simulator build hash / generator source hash 全部纳入，修改时序或 trace 代码后可能命中旧结果。至少应绑定配置、源码、二进制和 workload / ReusePlan hash，不能依靠结果文件的名字区分版本。

### F15 · P1/P2：报告字段不等于论文指标，汇总可能混用旧档位

`collect_dag_ladder.py` 会跳过缺失档位、读取已有 `dag_A*.json`，没有强制检查相同 workload / revision / config hash。`SKIP_A1` 还能复制外部 A1。复用缓存结果本身合理，但必须验证同平台、同修正方案与同引擎。

部分 tier CSV 取的是 Q 到达或 attention **开始**时间，而非逐请求首 token 完成；`pim_prefill_share` 用处理行数加权，而论文写的是 prefill 请求比例。两种统计都可有价值，但不能同名混用。physical DAG 没有完整峰值 KV bytes / 生命周期报告，按 tier 的首尾时间也不能替代严格定义的 TTFT、weighted TPOT。

若 CLI 中 `pim_prefill_query_batch`、readback / shadow / pool split 等开关只进入 analytic 配置或 metadata，没有贯穿 DAG 调用，则不能依据命令行和结果标签断言生效。`main.py` 的两条引擎路径需明确列出支持项，对未实现的组合报错。

`layout_handcheck_theory.py` 等手算脚本仍有旧布局/旧档位假设，部分遇到新 local policy 不支持，部分 mismatch 不导致失败；`diff_gather_effect.py` 的替代布局也不是当前 A3b/A4c 配对。它们只能作为其具体简化模型的 microcheck，不能证明最新七档公平性。

现有测试大多检查 helper 自洽、mock API、事件名称/数量、analytic A1 等；缺少“论文定义 → 实际算子/物理读写”的端到端不变量。这解释了 102 个测试通过而 F01–F08 仍然成立。

## 7. 文件级覆盖记录

下面覆盖相对 `c600051` 的全部 36 个变更文件；“未发现特定问题”不代表已经形式化证明正确。继承代码中的 GQA、计量与 setup 问题也一并报告，因为会影响论文结论，不能仅以“不是本次改动”免责。

| 文件 / 同类文件 | 审查侧重与结果 |
|---|---|
| `.gitignore` | 产物/缓存排除；本身无公平性结论，但实验 provenance 需另行保存 |
| `README.md`, `docs/README.md`, `docs/README_design_ladder.md` | 对照实际七档、输入缺失、旧手算与旧设计叙述，见 F01–F08、F15 |
| `experiments/run_dag_ladder.sh`, `collect_dag_ladder.py`, `run_layout_handcheck.sh` | 真实 CLI、重算 policy、复用旧 A1、缺档与汇总口径，见 F03、F15 |
| `main.py` | preset 解析、引擎分派、canonicalization、history override、配置落地，见 F03、F08–F10、F15 |
| `src/ablation.py` | preset 因果变化、analytic 与 DAG 分离、容量口径，见比较表、F09 |
| `src/workload.py` | owner、位置偏移、policy 代理、k 集合、history / token 输入语义，见 F03、F06、F08、workload 节 |
| `src/workload_runner.py` | 主要调用链、布局/算子/依赖/路由/指标，见 F01–F11、F13、F15 |
| `src/config.py`, `src/model.py`, `src/devices.py`, `src/system.py` | 模型与 stack/GQA、设备时间能量、共同开关，见 F09–F13 |
| `src/ramulator_wrapper.py` | extents / 单通道缩放、MQ interval、计数解析、缓存签名，见 F12、F14 |
| `pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py` | 地址 extents、replicate/MQ、命令顺序与缓冲约束；有实质模型工作，但上游物理映射不变量失效不能由 generator 自动修复 |
| `pim_ramulator_src/HBM3-PIM.cpp`, `hbm3_pim_controller.cpp` | turnaround 与 ACT 计数改动；全档共同修正不自动构成不公平，计数未贯穿能耗，见 F12、F14 |
| `src/cpp_eventcore.py`, `src/cppcore/eventcore.cpp`, `src/cppcore/Makefile` | host 调度加速与资源语义；加速模拟程序不等于给被模拟硬件加资源，资源集合冲突仍见 F06 |
| `src/gemm_table.py` | 可选 GPU GEMM 时间表 / 模式；没有证据表明默认启用了更差的新增 kernel。需实测校准基线，不作无依据指控 |
| `src/layout_probe.py` | 手算布局、activation / run 统计，不能充当真实 controller ACT 能耗，见 F12 |
| `output/analysis/extract_sweep_csv.py` | 结果字段/阶段时间/版本解释，见 F15 |
| `output/analysis/diff_gather_effect.py`, `layout_grid_csv.py`, `layout_handcheck_report.py`, `layout_handcheck_theory.py`, `layout_interleave_csv.py` | 旧布局 microchecks、简化物理假设与当前 ladder 不完全一致，见 F15 |
| `output/analysis/layout_A3b_R8.csv`, `layout_A4c_R8.csv` | 旧手算产物；不能替代当前真实 DAG 的共享 master 不变量 |
| `tests/test_workload.py`, `tests/test_placement.py` | 102 tests 通过，关键论文约束缺少独立断言，见各反例 |
| `tests/fixtures/workload_relay_s400w4t1.json`, `workload_2wikimqa_first8.json` | 仅 fixtures，不是完整论文输入矩阵 |

## 8. 重新达到可接受证据标准的顺序

1. **先修正确性，再看性能。** 统一 GPU attention 形状；修复 A1 / A5 执行设备；所有 policy 走同一完整引擎或显式拒绝不支持组合；固定 ReusePlan。增加零-diff master 相同、同 chunk 跨 scan 地址不变、每次读依赖有效生产者、资源集合互斥等独立不变量。
2. **实现一种真实持久物理存储。** 以 256-token block 为单位统一读写地址；per-agent/head diff 的逐轮分配、detach、洞与 compaction 成本可追踪；规定峰值容量和 eviction 行为。在线 table 与全未来 table 分开报告。
3. **把 placement 做成论文写的算法。** 候选 GPU/PIM 图使用相同算子成本和真实 resource availability，比较 completion；补齐 Q rotation、全局 softmax/P 路由、DRAM read/write 代价与分段流水。用真实 timestamp 测试 runtime batching / phase overlap。
4. **按已确认的 claim 粒度重做对照。** A1/A2 可作为独立 baseline；A5 的 prefill offload、MQ 与配套频点无需强制拆档。A3b 起保持相同逻辑修正集合与成本模型，仅增加对应机制；A2 的 GPU-local 优化与 MQ/频点拆分属于可选补充。
5. **冻结输入与平台。** 当前论文 C32 基线、C16/C64 邻点都要提供原文件；保留不利的单通道/head 点。合成 workload 明确标为机制实验，覆盖错位/不整齐长度/低共享/多轮 diff 与 arrivals；真实 trace 报原始分布、采样方法、tokenizer、history、输出长度。原生结果与合成敏感性结果分开。
6. **重新生成结果。** 清洁构建并绑定 source/build/cache hash，所有运行验证相同平台参数，缺档/混版本即失败；输出逐请求 arrival/TTFT、逐 token decode latency、定义明确的加权汇总、PIM 请求比例与行比例、峰值实际 KV bytes、命令计数及闭合 energy breakdown，再重跑论文矩阵。

在修复结构错误和超出 claim 的差异之前，当前性能评估仍需要修改和重跑。合成机制展示可以作为本阶段正式目标；按 [修改建议 README](../../docs/README_audit_fixes.md) 落地，不要求先证明生产流量代表性，也不要求重新拆分已确认的档位。

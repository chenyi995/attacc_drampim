# 最新复查：上轮修复已通过，仍有条件性边界

新设计与 W1 的当前事项见 [2026-09-06 复审第 6 节](../2026-09-06/README.md#6-需要过目的事项先说-attacc再说本项目影响)。本文件继续记录下述历史版本的修复和边界。

**主体复查版本 `bb19f31`，收尾补核至 `ff5b91e`，对照上轮 `fbe6756` 和原始 AttAcc `c600051`。** 上轮的跨轮断链、prefill 漏旧 diff、A2 重算旧 diff、已测 CacheBlend 继承位置、collector 混用 batch 时间、能量诊断不同步等具体反例已修复。本轮没有发现这些问题继续影响当前默认阶梯的证据。

这里的默认阶梯是 [ladder 脚本](../../experiments/run_dag_ladder.sh:63) 的 `recompute + batch 8 + FlashAttention + pipeline`，配合当前重列上下文、`history_len=0` 的 turns 输入。**这是针对修复和小输入的验收，不是全部 workload 的性能认证，也不是任意配置都正确的保证。** 继续检查发现了两个报表边界、两个复用输入边界，触发条件如下。它们不直接推翻默认比较，先供用户过目。

<a id="decisions"></a>

## 当前结论与阅读入口

| 项目 | 本轮结论 | 默认阶梯是否触发剩余问题 |
|---|---|---|
| [C1 Flash / pipeline](#c1) | 脚本保持共同开启 | 未发现变化；直接 main 仍需显式 flash |
| [C2 GQA](#c2) | 保持上轮 KV 字节修复 | 本轮未改变相应字节公式 |
| [C3 地址与容量](#c3) | diff/master 实际行隔离保持；diff 上界也已补齐 | 上轮容量缺口关闭 |
| [C4 行列扫描](#c4) | 保持用户接受的共同近似，由 Ramulator 决定 ACT/PRE | 不重开列步长指控 |
| [C5 A6 估价](#c5) | 简单逐 request 规则保持，旧 diff 已补进估价读集 | 未发现本轮另加费用/缩时系数 |
| [C6 tier 报表](#c6) | collector 同公式通过；底层 summary 有 batch=1、A2 history>0 两个边界 | 当前 batch=8、history=0 不触发这两个控制反例 |
| [C7 A3b 轮次](#c7) | 同轮正常紧排，跨轮沿用旧对象；不人为削弱 baseline | 已接受规则保持 |
| [C8 持久复用](#c8) | 上轮四个具体反例关闭；另有 CacheBlend 取整校验、重复输出指纹来源边界 | 当前 recompute 和 21 个 turns JSON 不触发这两个新控制条件 |
| [E1 能量诊断](#e1) | 实际事件和诊断已同用 H/h，数值一致 | 上轮诊断问题关闭 |
| [E2 会话内新增链路规则](#e2) | decode/bitmap 不收固定延迟，prefill 仍收；共同生效 | 改变绝对性能，旧结果须按模型版本解释 |

**怎么理解“还有问题”。** C6 的两个条件会使 A2/PIM 指标不同口径，使用对应配置时值得审阅。CacheBlend 校验是所有软件复用档共同拒绝执行的输入边界，没有形成分档加速优惠；重复输出指纹是有条件的来源错误，尚未量化性能影响。按用户要求，不把共同入口限制自动升级为默认阶梯不公平，也不重复要求已接受的共同近似证明各档误差相等。本次没有修改实现或替用户裁决这些新边界。

## 先认识比较对象

prefill 是处理输入，decode 是逐 token 输出；master 是共享的原始 KV，diff 是某个 agent 重算的 KV。diff 不是数值相减。后续请求要读取原 master 和已产生的 diff，其他写入继续追加。

| 档位变化 | 用户允许的机制 |
|---|---|
| A1 / A2 | 独立硬件 / 软件 baseline，不要求彼此只改一个变量 |
| A3b | 朴素软件复用与 PIM 结合，作为后续起点 |
| A3b → A4c | 同一 KV head 的 diff 集中布局，保留 master 全部通道 |
| A4c → A4e | 软件表分散可能共读的 master，继承 diff 布局 |
| A4e → A5 | PIM prefill + MQ，以及已接受的频点/缓冲配套 |
| A5 → A6 | 每 request 简单估算哪边快选哪边，首层选择后沿用，相等选 PIM |

主体实现变更为 workload 计划、runner、collector、layout probe 和相应测试；收尾新增的共同链路固定延迟规则另见 E2。[preset](../../src/ablation.py) 未改，GPU/PIM 能量单价、Ramulator wrapper、trace generator 未改。没有发现新增的档位私有拟合系数。配置差分仍在上述允许范围；下面另核对真正执行了哪些操作。

<a id="c8"></a>

## C8：旧反例关闭，新边界只在特定复用输入下出现

**原要求及本轮通过的证据。** 用户 a/b/c/e/f 的要求是：a/c 留在原址，后轮继续读取它们。主审复跑上轮输入，独立 agent 另行检查，结果如下：

| 上轮问题 | 当前实际行为 |
|---|---|
| 第三轮找中间轮不存在的 diff | `inherits_from` 追溯真正写入者；三轮仍读取第一轮同一对象 |
| 第二轮 prefill 漏旧 diff | GPU 回读旧 diff；PIM 的真实 scan 输入包含旧 diff，并保留对应 master 遮罩 |
| A2 继续重算旧 diff | 两轮控制中 A2/A3b/A4c/A4e 都只计算新 2 行、读取驻留 16 行，attention 上下文为 18 行 |
| CacheBlend 两轮位置不同 | 原 ratio=0.25 控制继承 writer 的逐层位置；full layer 0、partial layers 1/2 的三层绑定也通过 |
| 换同长度前缀仍继承 | 前缀 fingerprint 序列变化会阻止旧修正继承 |

存储专项进一步检查“只有继承 diff”和“继承 diff + 本轮新 diff”两组输入。A3b–A6 均保留旧地址、不重写旧 diff；A5/A6 实际 scan 中旧 diff、新 diff 各一次，没有漏掉可见 KV。额外扫描被遮罩的 master 属于已声明机制。A3b/A4c 的计划、写入对象集合和 decode 读取对象序列一致，没有为了体现收益拆散同轮 A3b。

上述都是实际构图/helper 的结构结果，未测性能。来源：[主审重放](archive/bb19f31_c8_replay_bb19f31_evidence.json)、[独立 C8 复验](archive/independent_c8_bb19f31_evidence.json)、[实际存储/scan 专项](archive/ledger_scan_bb19f31_evidence.json)。

### C8.5：CacheBlend 的旧数量校验与继承后的取整不一致

**触发条件。** 使用 CacheBlend，既继承旧修正，又第一次复用其他段，且比例取整不能直接相加。小输入有旧 D 的 8 行，下一轮还有另外 8 行首次复用；ratio=0.3 时旧修正 3 行、新抽样 3 行，共 6 行，但校验仍要求 `ceil(16×0.3)=5`。六个软件复用档 A2–A6 均报错。ratio=0.15 的控制也有 4 对 3 的冲突；原 ratio=0.25 控制能通过。

**原因与建议。** [采样](../../src/workload.py:580) 已按“继承旧结果 + 新采样”处理，[校验](../../src/workload.py:642) 仍按所有复用行一次取整。若需要支持这类 CacheBlend 输入，数量规则应与持久继承定义一致；不能为了满足旧总数而静默丢掉已经存好的旧 diff。新抽样预算和继承数量应分别说明、校验。

**AttAcc 有没有、是否影响默认比较。** 原始 AttAcc 没有 CacheBlend 或这套新校验。它是共同入口失败，不是某档被加罚时，也没有性能偏差数据。当前 ladder 使用 recompute，不触发。因此按用户口径先记支持范围，**不将它列为默认阶梯公平性的阻断项**。

### C8.6：重复输出指纹时，继承可能覆盖明确的 parent 来源

**具体 case。** a 输出指纹 O；w0 读取 a/O 后，在自己的上下文又输出相同指纹 O；w1 明确声明 parent=w0，应该读取 w0/O。若旧段指纹、前缀和偏移碰巧一致，[继承分支](../../src/workload.py:488) 会覆盖刚指定的 parent，将 w1/O 的 owner 改回 a，且不作本应需要的位移修正。

主审通过公开 loader、plan 和 TLB 重现 owner=a；独立 agent 捕获的 A5 prefill 实际 scan 也读 a/O。两个相同 token 指纹不保证在不同上下文下具有相同 KV。decode 沿用该绑定是源码推导，本次没有直接捕获这个反例的 batched decode 扫描。

**AttAcc 有没有、是否影响默认比较。** 原始没有这套 parent/继承机制。这是来源语义错误，目前没有测出分档净偏差；21 个当前 turns JSON 没有不同 parent 共享输出指纹，不触发控制条件。建议保留 `parent_out` 的明确 producer，只有物理来源也一致时才继承旧修正；不依据这个边界笼统判全部 workload 不公平。

两项新边界的完整输入、计划及错误：[独立证据](archive/independent_c8_bb19f31_boundaries_evidence.json)；主审独立复现及当前 21 个输入条件检查：[JSON](archive/bb19f31_main_verification.json)。

<a id="c6"></a>

## C6：collector 已同口径，底层 summary 仍有两个条件性错误

**已修部分。** [collector](../../experiments/collect_dag_ladder.py:147) 已对七档统一从 request summary 取值，PIM batch 记录只作诊断。相同 summary、不同 batch 字段的七档两 tier 控制通过，不再出现上轮同样 end 却报不同 tier_total 的问题。

当前 `ttft_s=max(first_token_s)`、`prefill_end_s=max(prefill_end_s)`、`tier_total_s=cum_end_s=max(end_s)` 均是从运行起点计的 tier 完成时刻；`decode_s` 是最后一个首 token 完成后到 tier 结束的尾段。它不是每请求平均 TTFT/decode 时长，也不应将各 tier 的累计时刻相加。公式共同适用本身不另列问题。

但 summary 的事件分类仍有边界。下表来自一层、单请求、lout=1 的真实构图，GPU 操作固定 0.001 s、PIM lane 固定 0.002 s；**用于查时间戳，不是性能测量或两档速度比较**：

| 控制 | prefill_end_s | first_token_s | 最后 GPU 算子完成 | end_s |
|---|---:|---:|---:|---:|
| A3b，history=0，batch=1 | 0.026 | 0.018 | 0.026 | 0.026 |
| A2，history=0，batch=8 | 0.012 | 0.025 | 0.025 | 0.026 |
| A3b，history=0，batch=8 | 0.012 | 0.026 | 0.026 | 0.026 |
| A2，history=3，batch=8 | 0.012 | 0.000 | 0.025 | 0.026 |

**C6.1：batch size=1 的 PIM decode 后处理被算进 prefill。** [单路 decode](../../src/workload_runner.py:3481) 调共同后处理 helper，生成的名字是 `gpu_*`；[summary](../../src/workload_runner.py:4315) 只把 `decode_*` 认作 decode。结果首 token 时间停在 attention 附近，后面的投影/FFN 却归入 prefill。batch>1 用 `decode_batch_gpu_*`，不触发该错误；**不是“只有一个 request 就一定错”，选择条件是 batch 参数**。建议统一事件阶段标记，让单路/批处理都覆盖完整生成计算。

**C6.2：A2 history_len>0 时可能记录首 token 为零。** A2 的 decode query 位置从 `total_length+history_len` 起，而 summary 匹配 `total_length`。history=3、lout=1 的控制就返回 0；与它对照的 PIM 路径可匹配。建议统一首个 decode query 的位置规则，避免把“未匹配”误报为“零时间”。当前重列旧段、history_len=0 的 turns 输入不触发。

**AttAcc 是否已有、影响哪里。** 原始 AttAcc 没有这些 request/tier summary；这是新增报告接口，不是硬件共同近似。这两个条件会影响 A2 与 PIM 档的 TTFT、prefill、decode/TBT 解读，但不改变该控制中实际调度和 end/makespan。当前默认 batch=8、history=0 不触发，不能因此否定其全部报表；使用对应配置时应先处理或如实限定相关指标。

还有一项诊断命名：`batch_first_attention_s` 仍可能取更早的 Q arrival，因为旧提取逻辑对两种时间共同取 min。它不参与新的主指标；可改名或只取 attention_start。证据：[summary 与七档 collector](archive/tier_summary_bb19f31_evidence.json)、[逐事件控制输出](archive/bb19f31_raw/README.md)。

<a id="e1"></a>

## E1：能量诊断与实际事件已对齐

runner 把实际 `energy_scale=H/h` 传给 [layout probe](../../src/layout_probe.py:219)，两者使用相同单位和通道求和。在 8 组 PIM 路径与 4 组控制中，诊断/事件能量比均为 1，旧 E1 关闭。

H 是本地真实 KV heads，h 是最忙 stack 已折入 trace 的 heads。能量仍按 AttAcc 单价和共同 head 数外推；原始 AttAcc 按满 stack 复制，新方法按实际 head 数线性外推。这一共同近似已说明，不把它改写为每个部分占用 stack 都独立模拟过。此轮修的是诊断接线，没有改 PIM 时间、MQ 命令或分档单价。[证据](archive/head_energy_bb19f31_evidence.json)。

<a id="e2"></a>

## E2：用户已确定链路固定延迟规则，共同实现已补查

**chenyi9 在本轮再次明确裁决：链路计价由用户确定，decode 小流量传输的固定开销可以忽略。该规则不再作为待审问题，只核对实现是否共同生效。**

收尾时另一会话提交了 `9c40891` 和记录它的 `ff5b91e`。[实现 session §16](../../docs/sessions/2026-09-05-ladder-fixes-f01-f02-f04.md:350) 将“prefill 收固定链路延迟，decode 不收”记录为 chenyi9 的裁决。本次只补核实现及共同适用范围，没有修改这项规则，也没有复核该 session 中另一轮运行的性能数字。

[统一链路 helper](../../src/workload_runner.py:319) 按操作名打标：`decode_*` 和 bitmap 不收固定延迟，其他操作仍收；[GPU 设备模型](../../src/devices.py:389) 根据该标记决定是否加 `nvlink_latency`。这是**按阶段/名字分类，不是字节大小阈值**；A2 decode 整段 KV 回读也适用，不能描述成只有 PIM 小包受益。带宽传输时间和远端 HBM 约束仍保留。

原始 AttAcc legacy 链路没有当前 refined/flash 的这项固定启动延迟；这个共同取舍有上游口径可对照，但不能称相同公式完整照抄上游。它会改变绝对性能和瓶颈，不能与此前收全部固定延迟的结果混为同一模型。没有改能量公式、Ramulator 模型或某个消融档的私有系数。链路到达时间变化可能间接改变批次组合，不能据此保证总能量或实际送入 Ramulator 的任务组合不变。

新增定向测试通过；另用 4 KiB 和 16 MiB 的共同 helper 控制核对，同字节数的 prefill/decode 价格差正好是一次固定延迟，bitmap 与 decode 一致。后一个大包控制仅验证代码按阶段分类，不将其称作小流量。独立补核见 [链路证据](archive/link_latency_ff5b91e_evidence.json)；新增提交和代码快照见 [收尾记录](archive/ff5b91e_late_snapshot.json)。此前固定价格设备桩的 C6/C8 反例不依赖真实链路单价，其结构结论不变；不能把那批假价格数字当成新模型性能。

<a id="c3"></a>

## C3：实际行隔离保持，新增容量上界正确

上轮已证明 4 MiB diff 偏移改变实际 ALL-BANK row，master K/V 与 diff K/V 行分离。此次 [容量检查](../../src/workload_runner.py:992) 在 diff 末端超过 8 MiB 时拒绝；源码条件推导最大为 1,048,576 个 diff tokens，K diff 保持在 4–8 MiB、V diff 在 12–16 MiB。等于上界允许，再多一个 token 拒绝。

边界是对当前源码条件的计算，未做百万行分配或性能模拟。原始 AttAcc 有 ALL-BANK 语义，没有新增 diff allocator；此前缺的容量保护已补齐。[本轮存储证据](archive/ledger_scan_bb19f31_evidence.json)，实际行解码沿用 [上轮控制](archive/c3_fbe6756_evidence.json)。

<a id="c1"></a>
<a id="c2"></a>
<a id="c4"></a>
<a id="c5"></a>
<a id="c7"></a>

## 保持有效的共同配置和裁决

- **C1 / C2：** ladder、sweep 默认 flash，ladder 显式 pipeline；直接 main 仍需传 flash。GQA 的 KV-head 字节公式本轮未改。历史结果不能仅凭现在的默认值认证。
- **C4：** 按 channel 输入需要的行/列，ACTAB/PREA 交 Ramulator；不加人为“步长/跳转费用”。共同 V 子段边界不重开，不能用绕过实际 `_pool_reads` 的输入指控 A3b。
- **C5：** A6 仍比较 `Link(真实驻留KV)+共同GPU attention` 与 `各sweep最慢channel完整PIM扫描+context返回`；估价忽略 Q、零回读为零、首层选择后沿用。C8 的旧 diff 已补进同一输入集合。论文简单公式保持。
- **C7：** A3b 同轮 diff 正常追加紧排，跨轮保留旧物理对象，不人为拆散以制造 A4c 收益。全局 diff_cursor 可能让不同 agent 交错的已知边界保持；没有新的独享优惠证据。
- **计量来源：** 本轮没有修改 Ramulator wrapper/generator、GPU/PIM 能量单价或 preset；共同链路固定延迟的新增规则见 E2。PIM scan 时间仍来自模拟/缓存周期乘 tCK，能量按 AttAcc 单价计；DIE/TLB、普通 STORE 仍不另加费用，旋转仍按已定 GPU 路径。

<a id="contributions-check"></a>

## 对论文四项贡献和 workload 能确认到哪里

本轮没有发现 A3b→A4c、A4c→A4e、A4e→A5、A5→A6 增加用户未声明的档位私有机制。旧 master/diff 能继续进入后续实际扫描，消除了上一轮不能验证跨轮布局的具体漏读；这比只核对相同计划 hash 更进一步。

仍不能把 [贡献 README](../../docs/README_contributions.md) 的条件性 ACT、通道并行和假设选边收益当成整体性能结果。`interleaved` 的单次展开与 `turns` 的逐轮输出不是同一执行工作量，不能跨编码归因布局收益。当前默认输入不触发本页新增边界，不代表整个 workload 的收益已实测或能判定总体高估/低估。

## 验证与记录

主审加三个独立 agent 完成复核；主体定向测试 **7 个通过**，会话内新增链路测试 **1 个通过**。本轮只用小 helper、实际构图的固定价格设备、地址/计划检查和受控 collector，没有跑 Ramulator 性能实验。没有修改实现、测试、workload、论文或已有结果。

详细修改理由、各 agent 分工和执行记录见 [本轮 session](../../docs/sessions/2026-09-05-bb19f31-fix-verification.md)，文件 hash 与链接核验见 [manifest](archive/bb19f31_audit_manifest.json)。上轮全文原样保存在 [fbe6756 审计快照](archive/CURRENT_ISSUES_before_bb19f31.txt)；历史“未修”应按时点理解，当前状态以本页为准。

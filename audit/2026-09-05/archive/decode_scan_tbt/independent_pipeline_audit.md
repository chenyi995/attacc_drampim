# Decode scan 收益传到 TBT：独立 pipeline 代码审计

审计对象：`167fe08e608b89e32f57402953314ced0b1194c6`，2026-09-05。上游对照为原始 AttAcc `c600051`。本轮只读实现，未运行新的 Ramulator 或性能仿真；以下是代码结构结论，不是测得的延迟损失。本文由独立 agent 提供，实际已有运行的收益百分比由主审另行分析。

**结论：当前开启 pipeline 时确有 GPU、链路与 PIM channel 的并行，但不是“凡已就绪且资源空闲就立即执行”的调度器。** Decode scan 的实际 channel 时长取最大值，随后仍需等待 GPU 局部 attention、返回链路和 GPU 后处理。因此 scan 的百分比收益不必等于 TBT 的百分比收益。另有可由事件构造顺序引入的等待，不能仅以 `overlap_validation.passed=true` 排除。

chenyi9 已接受各档共同采用的 AttAcc 近似。下面区分必要依赖、共同调度限制和条件性多余 fence，不将共同限制重新认定为 A3b 被特意削弱，也不要求各档近似误差相互抵消。

## 1. 已有并行，以及必须保留的等待

- **PIM channel 真正分开计时。** `src/workload_runner.py:2451–2459` 取得逐 channel 的实际设备价格；`2477–2488` 发出各自的 `PIM:poolC-C` 事件。`pipe=True` 时资源按设备名分开，见 `2580–2586`。不同 channel 可并行，同一个 channel 的扫描串行是资源约束。单请求 decode 的合并依赖全部扫描加 GPU tuple，见 `3491–3502`；batched decode 同理，见 `3904–3923`。这里没有用 channel 平均值替代延迟。
- **GPU 局部 attention 可与 PIM 旧 KV 扫描重叠。** 单请求的局部 attention 依赖 QKV（`3415–3431`）；PIM 扫描依赖 Q 传输/地址准备（`3449–3488`）。合并时才等待两边。当前 token 的 K/V 传输也只依赖 QKV（`3401–3413`），并非等 attention 后才发起。
- **元数据和写入记账没有占用执行资源。** `src/cpp_eventcore.py:18` 的 `DIE/TLB/STORE` 是 dependency-only；`workload_runner.py:2580` 不为它们预约资源。因此不得把它们当成当前有价串行操作。
- **同一请求的层间 hidden state、下一 token 的自回归输入、一次 MQ sweep 所需的全部 Q，以及 attention 后的 projection/FFN，存在真实数据依赖。** 见 `3380–3383、3505–3517、3626–3629、3821–3834、3925–3937`。MQ sweep 等待其成员 Q 到齐不是平白增加的屏障。

与上游的关系：原始 `src/system.py:110–167` 是聚合 pipeline 公式，`240–255` 遍历 decode 操作取价；没有逐请求、逐 channel 的事件 DAG。当前分资源并行是该仓库新增的显式调度模型，不能仅因函数注释使用 “AttAcc convention” 就称与上游的 overlap 数值等价。

## 2. 可疑点 P1：提前预约未来资源，不回填空闲窗口

**代码确定，发生多少次和损失多大需从本次完整 events 判断。** Python 调度器按事件列表一次向前遍历，开始时刻取 `max(resource_available, dependency_finish)`，然后直接把资源末尾移到该事件结束：`workload_runner.py:2542–2545、2580–2588`。增量版本 `2606–2615` 与 native `src/cppcore/eventcore.cpp:103、144–152` 相同。

最小结构是：先排一项 GPU 工作，但它正在等待 PIM；之后追加另一项无此依赖的 GPU 工作。前者会把未来 GPU 时段预约下来，后者无法使用预约前的空闲窗口，即使其输入已齐。已有独立结构探针保存在相邻归档的 [脚本](../layout_ceiling/layout_ceiling_scheduler_probe.txt) 与 [结果](../layout_ceiling/layout_ceiling_scheduler_probe.json)；那是固定时长逻辑例子，不是本轮性能测量。当前源码仍使用同一规则。

Overlap checker 在 `2743–2757` 复核的是同样的资源末尾递推。通过检查说明没有违反这套预约规则，不说明不存在可填充的空闲窗口。

**上游是否已有：** 原始 AttAcc 没有这样的请求事件预约器，也没有用 ready-first 策略重排请求；它的串行操作累计与聚合 overlap 公式不能证明这个新增调度器的空闲窗口不可避免。

**对布局收益的影响：** 会改变等待和关键路径，但方向不是由代码即可确定。它可能使某些 scan 改善被其他预约遮住，也可能把本可隐藏的 scan 等待暴露出来。不能直接宣称“修好后 A4c/A4e 收益一定更大”。

## 3. 可疑点 P2：batched decode 的分阶段构造与跨 microbatch 耦合

这里的 microbatch 指 GPU 或 PIM 一次处理的一小组请求，不等于单个请求。

**输入已就绪的顺序没有用于 GPU QKV 分组。** `3640–3642` 按 `active` 的原顺序切组；`3653–3654` 要等组内全部请求的上一层输入。函数文档 `3533–3538` 说 QKV 按 input-ready order，但这段代码没有对应的 readiness 排序。等待组内所有输入可以是批处理取舍；“按就绪次序分组”则不是当前实现事实。

**所有组的 QKV 与 Q/KV 发送先被预约。** `3640–3683` 构造全部 Stage-A 操作，之后才建立 GPU local attention 和 tuple（`3759–3787`）以及 context 返回（`3917–3923`）。后者使用同一 `LINK` 资源，因此早已完成 scan 的请求也可能排在其后构造前已预约的其他请求 Q/KV 传输之后。这不是其计算结果必须依赖那些其他请求的 Q/KV。

**“global Q-ready queue” 的范围只是当前 tier、输出步、层。** `3626–3629` 固定这三层循环；`3735` 排序的只是当层 `active`。排序发生在按源顺序预约 Q 链路之后（`3706–3719`），并不意味着一个跨所有层、所有步的异步就绪队列。

**同一层内也可能留下可填空窗。** Stage B 为一个组依次构造 local attention、扫描、context、GPU post，然后才进入下一组（`3740–3952`）。第一组的 post 等待 PIM 时，第二组的 QKV 可能早已完成，但第二组的 local attention 已被追加在第一组 post 后面。这是只含一层时也可触发的 P1 实例；不必等到跨层执行才能观察。

**快组不能在慢组的 PIM 等待窗口先做下一层 QKV。** 当层各组的 local attention、扫描、context、FFN 都先构造完成（`3740–3952`），下一层 QKV 才追加。即使某个快组的 FFN 已完成，另一慢组的 FFN 尚在等待 PIM，下一层 QKV 仍排在所有这些已预约 GPU 事件后面。实际数据只要求这个快组自己的上一层输出；共同 GPU 不要求把空闲窗口空着。

**上游是否已有：** 原始 AttAcc 的 batch 维度是聚合形状，不模拟不同请求不同 ready time，也没有这种 Stage-A/Stage-B 请求队列。因此不能把这些具体排队后果作为上游已验证的行为。组内等待本身是常见批处理取舍；是否值得拆组不是本轮审计能够根据代码直接判定的性能结论。

**适用范围：** `cacheblend_batch_size > 1` 的物理 decode 路径，A3b–A6 共用。若只有一个组，跨组部分不触发；跨请求延迟异质性越明显，越值得在已有 events 中核实是否确有空闲窗口。没有测得损失大小。

## 4. 可疑点 P3：batch-size 1 与 tier 的构造边界

`4862` 按请求 ID 处理请求；当 batch-size 为 1 时，`5262–5271` 为该请求构造完所有 decode token，再进入下一请求。由于 P1 的预约规则，后续独立请求不能自动填入前一请求 decode 中的 GPU 空闲窗口。不能把“batch-size 1”理解成按所有请求 ready time 交错运行。

`4863` 还让新 tier 的请求依赖 `previous_tier_done`，`5283` 则用整个旧 tier 的结束事件更新它。若 workload 的 tier 本来就是所有成员必须同步完成的业务屏障，这是规定的工作负载语义；若只依赖各自 parent，则这个屏障比实际数据依赖更宽。两者要根据输入口径区分，不能凭这一行就判全部 tier 同步不公平。此边界主要影响启动/跨 tier 延迟，不必然改变同一请求稳定段的 TBT。

上游 AttAcc 不表达多代理 tier/parent DAG，因而没有相同的业务屏障定义。

## 5. 已排除的候选：当前层 KV 写回的下一层 fence

单请求 `3516`、batched `3952` 都将 `(post_last, store)` 设为下一层依赖。`STORE` 自己虽然零时间，但依赖这一层新 K/V 的 LINK（`3509–3514、3944–3949`）。下一层 QKV 消费上一层 hidden state，不读取刚写到 PIM 的上一层 K/V；后续 token 重新扫描上一层 KV 时才需要这项写入完成。

但独立复核资源顺序后，这条依赖**不能作为当前常规 decode 的额外延迟证据**：单请求 KV link 在 `3408` 追加，context return 在 `3499` 追加，两者使用同一 LINK；post 又依赖 context return。Batched 路径也有相同顺序（`3678`、`3919`、`3936`）。因此非空上下文的正常路径中，post 完成已经晚于 KV link/store，附加 store 依赖被现有返回链路覆盖。长 LINK 排队不会打破这个顺序。

这条依赖从数据语义上可能冗余，但本轮不将它列为需要修复的性能瓶颈。单请求空上下文、不生成 PIM context return 的特殊分支不能由上述证明覆盖；它不是本次长上下文 scan 收益的解释。

上游没有逐 token 的 STORE 节点；其生成期 `comm_x2g` 是聚合操作（原始 `src/model.py:174–177` 等）。没有上游逐层 store fence 可用来直接证明当前这条依赖必要。

## 6. 共同粒度限制：整块 GPU、整次扫描与 TP 通信

当前 QKV 是整个组的一个 GPU 事件，然后才能发送 Q；projection 在完整 context 返回后执行，见 `3649–3675、3925–3937`。它没有把同一请求的 QKV/projection 按 head 切开，与逐 head 的扫描流水衔接。原始 AttAcc 的 `minimum_ratio = 1/(num_heads/num_xpu)` 会折减 QKV、projection 和通信的计入时间（原始 `src/system.py:126–165`；当前保留在 `src/system.py:40–73`）。这段公式只由旧 `System.simulate` 使用（`306–308`），当前物理 DAG 不调用它。

因此，测试 `tests/test_workload.py:27–56` 证明的是保留的旧 `System.simulate` 与抽出的公式一致，不能证明新物理 DAG 获得了同样的 head pipeline。用户已允许共同建模近似；这里应明确“当前是整操作 DAG 粒度”，不自动要求追加一个新优化或给某档补贴时间。

另一个较粗的资源分类是 TP all-reduce：`src/model.py:173–175` 的 `comm_g2g` 进入 post 序列后，单请求 helper `2919–2931` 和 batch 路径 `3925–3936` 都将其标为 `GPU`；设备侧实际通过通信模型取价，见 `src/devices.py:393–404`。这使 TP 通信不能和另一独立请求的 GPU 操作重叠。它是条件性资源粒度限制，是否有可用的独立通信/计算资源仍依赖具体系统；上游也把 G2G 作为序列中的操作累计，没有给出请求间异步通信模型。TP 为单卡且通信价格为零时不构成有价瓶颈。

## 7. GPU 变快为何可能使布局更显眼，但不能保证收益比例

对于一个没有其他排队干扰的层，可用结构式理解：

`层时间 = GPU 前后处理 + 链路开销 + max(PIM 旧 KV 扫描, GPU 局部 attention 分支)`。

这是解释依赖的简化式，不是对已有结果的拟合。若两档 scan 都被 GPU 分支遮住，scan 改善未必传到该层完成；GPU 分支加快后，PIM 可能成为较慢的一侧。即使原先 PIM 已是较慢的一侧，减少共同的 GPU 前后处理也可能增加相同 scan 省时在 TBT 中的百分比。真实 batch 路径还有 P1–P3 的等待，不能据此断言提高 GPU 峰值、GPU 带宽或切换 GPU 模型必定产生某个收益比例。

“GPU 变快”还应区分 compute 与 memory：设备模型同时计计算和访存；只提高一个峰值不保证 QKV/FFN 或短 attention 按比例缩短。Flash 分支存在于 `src/devices.py:57–59、129–146`，但当前 `main.py:175–177` 的默认 GPU model 是 `legacy`；pipeline 默认开启见 `214–216`。实际结果是否使用 Flash 要看该次命令/manifest，不能由默认 pipeline 倒推。Flash 吞吐表是论文图的近似读数，见 `src/gemm_table.py:15–21`，不是本仓库 GPU 实测。

还有一个上游原有的限制：ACT/NORM 的访存时间公式包含不随 GPU 峰值带宽缩小的固定截距，当前见 `src/devices.py:295–307`，原始 AttAcc 见 `src/devices.py:164–176`。当前 Flash 只将 MATMUL/SOFTMAX 分流到 Flash 定价（`284–286`），不会移除 ACT/NORM 截距。因此只提高 GPU 算力或带宽，并不等于缩短所有 GPU 事件。它是解释 TINY 配置为何仍可能由 GPU 公共成本主导的依据，不是本轮要求修复的差异。

当前 `_link_layer` 对 decode 发送关闭固定启动 latency，见 `workload_runner.py:330–347`。因此旧报告里“每 token 多笔固定 NVLink latency 淹没收益”的解释不能直接应用到当前实现。

## 8. 解释已有 TBT/scan 数字时的限定

`summarize_decode_scans` 将 service 定义为单次扫描最慢 channel 的自身时长，将 elapsed 定义为最早 channel 开始至最晚 channel 结束，见 `4327–4375`。`per_step_elapsed` 是某请求某层的扫描区间，不含该 token 所有层的 GPU/链路工作，也不是 TBT。跨 channel 排队造成的 elapsed 与单次 Ramulator service 不应混用。

若比较“保留比例”，必须给出定义：扫描百分比改善与 TBT 百分比改善之比，是两个归一化指标的比值；它不等于“省下的扫描秒数有多少进入 TBT”。后者需要在同一请求/输出步/层范围内匹配绝对时间，且共享 MQ 扫描不可按成员重复当作互相独立的可省时间。不要混合均值、分位数和不同 token/请求的统计集合。

本轮确认的是：真实 channel 最大值没有被平均掉；pipeline 有效但有上述调度与粒度限制。每项限制是否实际触发、GPU 更快后的净效果，以及 scan 收益保留到 TBT 的具体比例，需以相应运行的完整事件时间线和配置证据为准。

> 历史记录：保留当时的技术证据和评价，当前待审事项统一看 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md)。

# 8c51672 修复复核：多项已修，仍有阻断问题

> **后续裁决更新：** chenyi9 要求沿用 AttAcc 的共同限制，以相对公平为准，逐项先展示来源再由用户判断。当前事项状态以 [相对 AttAcc 审阅清单](ATTACC_RELATIVE_FAIRNESS_REVIEW.md) 为准；下文 P0/P1、“阻断”“需修”是此前审计时的评价，不再自动代表整改要求。反例保留，未修改代码。

日期：2026-09-05。源码 `8c51672a3ef8b936340354b3211963cde8945c49`，本轮重点比较 `cdd89db..HEAD` 的 34 个变更文件，并检查改动调用到的 mapper、命令生成器及论文当前工作区。覆盖清单见 [CSV](reaudit_8c51672_file_coverage.csv)。前轮结论以各自 revision 为准；本报告是当前状态。

**chenyi9，不能确认“都正确了”。** 修复并非无效：多项旧反例已关闭。但 GQA 的字节校验出现确定的运行回归，PhysicalLedger 到 ALL-BANK 命令的地址转换仍不成立，A6 的两侧估价也还有操作覆盖遗漏。现有结果不能据此签署逐档公平性保证。

本轮主审、独立公平性 agent、计量 agent、ledger→trace 专项 agent 分别复核。执行的是轻量 helper、真实 DAG+设备桩、命令生成函数和定向单测；时间桩只用于检查结构，解析模型返回值也不冒充硬件测量。所有实现文件保持本轮开始时的内容。

**后续运行口径补审：** 用户要求 pipeline 开启、GPU 使用 FlashAttention。新增 [RC01–RC04 报告](RUNTIME_CONFIGURATION_AUDIT.md) 已核对入口和实际历史产物，并由独立 agent 复现开 pipeline 仍有资源空闲阻塞。CLI 默认 legacy；部分旧结果明确 pipe=false；FA 参数为论文图近似读数。此补审不关闭下列 V01–V08；前轮 manifest 保留历史完成时点，新文档 hash 见补审 manifest。

## 1. 使用的最新口径

保留 chenyi9 在前文及 [session §12–13](../../../docs/sessions/2026-09-05-ladder-fixes-f01-f02-f04.md) 确认的规则：A1/A2 是独立 baseline；A3b 起逐级加入 diff 布局、co-read 表、PIM prefill+MQ、自动选边。所有复用档共用随机修正计划；A1 使用精确 L。普通 DRAM store/read 不额外计时计能，KV 移动只计 AttAcc 链路，STORE/DIE/TLB 保留依赖。平台使用 A100，decode 当前 token 在 GPU 本地算是允许的。

**A6 只要求逐 request 比较 GPU/PIM 的估计耗时并选较小者；不要求双候选 DAG、排队预测或全局最优。** 本轮指出估价操作缺项，并未恢复旧的复杂选择要求。

## 2. 已确认修复，不再重复旧指控

| 旧问题 | 当前核查 | 状态 |
|---|---|---|
| c16 单扫/共扫改变 row | 五档相同对象的 ledger extent 均持久；独立 probe 重做旧反例 | 高层 ledger 的 row 漂移已修；到 MAC 的转换另见 V02/V03 |
| A3b/A4c 零 diff 子区间几何不同 | 实际两种 allocator 对相同 master 子区间返回相同 extent 几何 | 已修 |
| 同一连续 burst 的不同 fingerprint 修正不能合并 | A3b 能将它们放为一个连续对象 | 已修；重复同一 fingerprint 的跨段情况另见 V05 |
| master 普通读写旧的 15 倍差率、pool 争用差异 | store/readback 改为全零 `STORE`，也不占用 Python/native 调度资源 | 按新裁决已修，不要求恢复 DRAM 收费 |
| A1 256-token 桶、A2 多回读当前 token | A1 返回精确 token/head 数；A2 只读此前驻留行 | 已修；A1 仍是 count/dense 抽象 |
| 修正集合随 rung 改变 | CLI 固定 `recompute_canonical=False`；主审复用例子五档 hash 相同 | 核心执行计划已修；A2 报告未导出同一 hash，不等于计划不同 |
| consumer 先扫 owner 后写 | 倒序 JSON 控制组的 consumer scan 均有 owner store ancestor | 已修该反例；预约顺序另见 V05 |
| GPU prefill 多发 Q-to-PIM | A3b/A4c/A4e 的 GPU prefill 不再生成该 link | 已修 |
| private / 单请求 decode 遗漏 MQ | 两条路径均调用 `_apply_pim_batch` | 已接入，不再报告旧参数遗漏 |
| 当前 token 的 GPU local context 形状 | 两条 decode 分支均恢复 dhead 输出宽度 | 已修 |
| A6 只估一条 lane、尾批按满批收费 | 估价覆盖全部 lane，按 divmod 给尾批实际 query 数定价 | 已修这两项；V06 是其他缺项 |
| 位移只看 JSON 的 delta=0 | `_apply_plan_deltas` 推导 consumer/owner 偏移，真实 DAG 出现额外 Q variants | 已修位移输入缺失 |
| GQA QKV/部分 KV bytes/MAC 能量 | QKV 宽与 prefill KV 字节已修，MAC 能量两接口纳入 GQA query 数 | 部分修复；V01 指出未贯通路径 |
| 缓存丢失 extent 相对 K row | 改 K/V row 关系可使模拟桩重新调用 | 已修这项；实际共享库指纹仍遗漏 |
| TBT 只有请求均值、tier 完成取 attention start | 加权 TBT 公式正确；`cum_end_s` 改用 request end | 部分修复，V08 说明剩余报表问题 |

本轮实际跑了 **11 项定向测试，全部通过**，日志见 [测试记录](reaudit_8c51672_targeted_tests.log)。此前 session 的全套通过记录属于实施时点，不作为本轮执行次数。通过这些测试不覆盖下面的命令映射及 GQA 入口反例。

## 3. V01 · P0：GQA 修正未贯通 validator，默认路径直接失败

[prefill](../../../src/workload_runner.py:4837) 已用 `_kv_hidden` 除 GQA group，然而 [validator](../../../src/workload_runner.py:2756) 仍要求 `KV bytes = rows × 2 × local_hidden × dbyte`；[finalize](../../../src/workload_runner.py:5138) 传入的仍为 Q hidden width。合法的 GQA KV link 因而被拒绝。

主审使用同一个小模型/同一 prompt，仅改变 GQA group；所有设备均为桩，错误发生于真实报告校验：

| 档位 | MHA 控制组 | GQA group=4 |
|---|---|---|
| A1 | ok | error：`CacheBlend KV link byte count is invalid` |
| A2 | ok | ok |
| A3b | ok | error：`CacheBlend KV link byte count is invalid` |
| A4c | ok | error：`CacheBlend KV link byte count is invalid` |
| A4e | ok | error：`CacheBlend KV link byte count is invalid` |
| A5 | ok | error：`CacheBlend KV link byte count is invalid` |
| A6 | ok | error：`CacheBlend KV link byte count is invalid` |

计量 agent 另用仓库 LLAMA3-8B 配置独立复现同样失败，见 [专项](MODEL_PROVENANCE_REAUDIT_8c51672.md)。这不是“数值不够准”，而是该路径不能正常返回结果。

还有独立的遗漏：[batched decode](../../../src/workload_runner.py:3474) 仍为 `kv_bytes = 2*q_bytes`，没有像 single decode 那样除 group。仅修改 validator 会放行这个错误流量；应一起核对构造和校验使用的 KV head 数。非默认 private PIM-prefill helper 也保留旧式，计量专项单列，不与正式 A1 GPU-prefill 混淆。

## 4. V02 · P0：diff 区的偏移没有隔离 ALL-BANK 的存储行

新 [PhysicalLedger](../../../src/workload_runner.py:820) 用 `_DIFF_REGION_BYTES = 1 << 29` 区分 master/diff。但在本仓库 HBM3_8Gb_2R 地址布局中，这个偏移改变 **pseudochannel**，没有改变 row/column。[mapper](../../../pim_ramulator_src/hbm3_pim_linear_mappers.cpp:66) 负责解码地址；[ACTAB](../../../pim_ramulator_src/patches/src_dram_lambdas_action.patch:8) 与 [MACAB 行就绪检查](../../../pim_ramulator_src/patches/src_dram_lambdas_preq.patch:8) 对 ALL-BANK 命令遍历整个 channel 的所有 pCH/rank/BG/bank。

纯命令生成/地址解码反例的两个首 MAC 为：

| 对象 | channel | pseudochannel | row | column |
|---|---:|---:|---:|---:|
| master 首 MAC | 15 | 0 | 0 | 0 |
| diff 首 MAC | 15 | 1 | 0 | 0 |

两个数值地址看似相隔很远，却访问 ALL-BANK 同一组物理行/列。不能用这个偏移实现声称独立的 master/diff 区；某些读集还会出现不该有的行命中。因此新的持久地址整数本身不足以证明容量和 ACT 归因正确。

**证据边界：** 这是受版本控制的 generator、mapper 和 HBM action/preq 源码的确定矛盾。未验证运行中二进制的实际周期，也没有将其换算成 speedup。应按实际 ALL-BANK row 地址空间表达独立存放，而不能仅换一个看起来很大的 byte offset。

## 5. V03 · P1：token 子段到 K/V MAC 的转译仍不对应固定存储

[ledger](../../../src/workload_runner.py:1030) 以 `start + first*4` 给任意子段地址；mapper 以 transaction 粒度丢弃低位，而 [generator](../../../pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py:162) 沿本次 extent 重新执行 MAC tile。平均 bytes/token 不能直接等同于任意 token 在现有 MAC 布局中的独立起点。

更直接的 V 反例不依赖 TLB 与 trace 单位比较：同一已存的 256-token 对象、相同 V base，完整扫描的第二个输出维块从 **V+512** 开始，仅读 first16 则从 **V+32** 开始。因为 [context_mac](../../../pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py:218) 把本次 `e_len` 用作矩阵 stride。期间没有重排或迁移；两者不能都代表同一物理 V 矩阵。

另外，受控 packed diff 例子包含 256 tokens：ledger 的 row 集合为 `[0]`，生成的 score MAC row 集合却为 `[0, 1]`。读取已存满块的最后少量 tokens 也可跨入下一 row。token0/token1 的命令则可能只差会被 mapper 丢弃的低位，接口未携带区分对应 lane 的元数据。

应保留已存 K/V 的 tile/stride、有效 lane 和访问边界，按实际被打开的 row/column 计时；不能按每次读取长度隐式重排。细碎 extent 的逐段向上取整还会增加重复 MAC。这些问题可能同时影响 A3b 和后续档，净偏差不能统一称保守，也不能给出未经计时的倍率。

## 6. V04 · P1（超过 16 heads/HBM）：多个 head 共用 channel 时地址重复

`PhysicalLedger._channel` 对 channel 取模，`extent_groups` 没有为绕回同一 channel 的 head 增加独立 row 区。控制组 heads=32 时，ch0 出现完全相同的 extent `[0, 8388608, 256]`，分别代表不同 head。

这个边界与当前正文 LLaMA-7B / one GPU / one stack 配置有关，见 [模型段](</data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/sections/06-methodology.tex:122>)。已有 head-slicing 告警只能解释通道并行退化，不能授权不同 head 使用同一份 KV 地址。该情况需具备独立分配或明确拒绝；小于该边界的控制组不因本条被判错误。

V02–V04 的脚本与完整输出见 [命令探针](ledger_trace_boundary_8c51672_probe.txt)、[JSON](ledger_trace_boundary_8c51672_evidence.json)。它只调用当前受版本控制的命令生成器并按源码解码，不能当作实际 binary 仿真结果。

## 7. V05 · P1/P2：持久扫描账本已建立，但“实际写入驱动”仍不完整

[store helper](../../../src/workload_runner.py:2447) 仍只读旧 `KVLocation`；独立 probe 发现创建 store 事件后 `_ledgers` 仍为空，首次 scan 才建立 ledger。报告中的 store 地址、scan 的 `dram_addresses` 与计时 extent 也没有统一。**当前 store 已零收费，所以旧的带宽/争用惩罚已经消失；这一条不能继续写成 15 倍性能差。** 要关闭的是物理身份、容量、报告与写入依赖的验证缺口，不是恢复成本。

新 ledger 使用 `_reserved_rows` 字典的首次插入顺序，而 [_prepare_cacheblend_tlb](../../../src/workload_runner.py:3129) 仍按 JSON 顺序预约；实际 DAG 现在按 request_id 排序。两种 JSON 顺序可以对应相同实际执行顺序，却得到不同 append 放置。称其为“预约时静态分配”可以理解；称其为已经重放真实写入流则证据不足。

重复 fingerprint 的 diff reservation 还会合入同一个字典 key，丢失中间 own KV 的写入边界。独立 agent 用实际 reuse plan 构造 `prefix | doc | own | doc`，两组修正被当成一个 burst。这个问题反向给予 A3b 更紧凑的布局，不是弱化 A3b 的证据。是否连续应由明确的写入流/事件定义决定，不能由 fingerprint 是否重复决定。

`diff_cursor` 仍跨 owner/layer 共用；真正交错的同一 agent 后续修正可能被其他 agent 隔开。当前 runner 按 request 整块预约、history 又是预置抽象，因此不能把这个边界直接外推为默认 workload 已发生的性能惩罚。完整反例和已修控制组见 [独立复审](INDEPENDENT_REAUDIT_8c51672.md)。

## 8. V06 · P1：A6 的选择规则接受，但估价仍漏/多计操作

全部 lane 和真实尾批已经修好。剩下两项在真实执行分支与 estimator 之间不一致：

- [_resolve_prefill_side](../../../src/workload_runner.py:4000) 在 `readback_rows=0` 时仍给 GPU 侧创建回读算子；fresh GPU 执行路径不会执行它。调用实际 A100a flash 通信解析模型，零字节回读仍返回 **6.06 µs**；legacy 返回零。这会人为提高 fresh 请求的 GPU 估价。
- 位移推导修复后，PIM 分支实际产生 `gpu_rotate_q_extra_to_pim`，但 estimator 的 Q 链路仍只覆盖普通 Q/ctx。轮转放在 GPU 是正确的；漏掉已有额外传输才是此处问题。

同一设备桩对所有算子调用给固定价格，比较 estimator 与真实分支已有操作的服务时间之和（并行 scan 每 sweep 取 lane max），得到：

| 控制组 | 本次计算 tokens | GPU 估价 µs | PIM 估价 µs | GPU 分支操作和 µs | PIM 分支操作和 µs | 选择 |
|---|---:|---:|---:|---:|---:|---|
| fresh_16 | 16 | 4.000 | 4.000 | 3.000 | 4.000 | pim |
| shifted_reuse | 5 | 4.000 | 3.000 | 4.000 | 8.000 | pim |

这些数值**不是 DAG 完成时间或硬件性能**；只是用相同价格证明估价对象与提交操作不同。简单逐 request 选择本身符合 chenyi9 的定义。修正操作覆盖即可，不需要升级为双 DAG 选择器，也不能由局部选择推导全局 makespan 恒优。

## 9. V07/V08：缓存与结果汇总仍有未闭合项

**V07 · P1：实际 Ramulator 共享库不在工具链指纹中。** 当前 executable 的 ELF `NEEDED` 包含 `libramulator.so`，而 [_toolchain_fingerprint](../../../src/ramulator_wrapper.py:284) 只覆盖 executable 与 bank generator。只改共享库的纯文件桩验证不会改变 cache key，可能沿用旧核心实现的结果。K 相对 row 关系修复有效；只改 V row 仍碰撞是更窄的通用 API 边界，当前固定 K/V 间距 ledger 不触发，不能混称默认必现。

**V08 · P1/P2：tier CSV 修复仅覆盖部分字段。** `cum_end_s` 已从 request end 计算，但 [collect](../../../experiments/collect_dag_ladder.py:119) 仍只遍历存在 batches 的 tier。A2 的报告没有 PIM batches，整档仍会缺行；`prefill_s/decode_s/tier_total_s` 仍由 attention-start 差构造。计量 agent 用小 JSON 输入复现缺档与终点不一致。新的 step-weighted TBT 公式则正确，不应重报为未修。

计量实现、自动生成数字表和复现方法见 [计量专项](MODEL_PROVENANCE_REAUDIT_8c51672.md) 与 [证据表](model_provenance_8c51672_evidence.md)。多 stack 不满载的能量复制、history 抽象和旧 policy 分派覆盖等前轮局限，当前改动未宣称全部解决，仍按其适用范围解释。

## 10. 对逐档公平性的当前裁定

| 档位/比较 | 当前裁定 |
|---|---|
| A1/A2 | 独立 baseline 定义接受；精确 L/远端驻留数量已修。A1 GQA 被 validator 阻断，A2 的 tier 汇总缺行需纠正 |
| A3b→A4c | master 高层几何、共同修正集合、STORE 计量已明显对齐；diff/master 的 ALL-BANK 地址重叠和子段转译阻断当前物理归因 |
| A4c→A4e | preset 仍主要只改变 table，未发现新的专属时长系数；公共 ledger→trace 问题及写入账本缺口使真实布局归因仍不能确认 |
| A4e→A5 | PIM prefill+MQ 为允许机制包，GPU 多余 Q 和 MQ 漏接已修；GQA 入口/批量字节、公共存储命令问题仍影响比较 |
| A5→A6 | 简单逐 request 规则符合定义，所有 lane/尾批估价已修；零字节回读与 variants 流量覆盖仍不同 |

本轮没有证据证明有人故意弱化 baseline。已有改动中，STORE 清零移除了原先额外惩罚，重复 fingerprint 合并甚至可能增强 A3b；另一方面错误的 diff row 映射和估价缺项仍可能造成不真实的收益或损失。不能据此将全套结果统一解释成保守估计。

论文工作区已经写 A100，并补上 decode 最新 token 在 GPU 的描述；这两条不再判正文不一致。旧的 event-based/DAG 选边文字仍在，以 chenyi9 已确认的简单逐 request 定义审查实现。

## 11. 可复核记录

- [主审脚本归档](reaudit_8c51672_probe.txt)、[主审 JSON](reaudit_8c51672_evidence.json)、[定向单测日志](reaudit_8c51672_targeted_tests.log)。
- [独立审计](INDEPENDENT_REAUDIT_8c51672.md)、[独立存储 JSON](independent_8c51672_storage_evidence.json)、[命令边界 JSON](ledger_trace_boundary_8c51672_evidence.json)。
- [覆盖清单](reaudit_8c51672_file_coverage.csv)、[版本/文件核验](reaudit_8c51672_manifest.json)、[session](../../../docs/sessions/2026-09-05-8c51672-fix-verification.md)。

以上结论是对当前实现和模型一致性的审计。已修项与现存问题分别列出，没有修改实现或已有性能结果，也没有用设备桩的速度证明论文收益。

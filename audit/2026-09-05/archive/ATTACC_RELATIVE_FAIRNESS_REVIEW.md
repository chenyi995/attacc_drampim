> 历史记录：保留当时的技术证据和评价，当前待审事项统一看 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md)。

# 相对 AttAcc 的逐项审阅清单：先查来源，再由用户裁决

日期：2026-09-05。当前实现 `8c51672`；原始 AttAcc 对照 `c600051`。本页按 chenyi9 最新口径重分类此前 V01–V08、RC01–RC04 和计量专项的剩余事项。**本页的状态取代旧报告的“阻断 / 必须修复”标签；旧反例和原始数据保留，不代表这些事项已获准修改。**

## 1. 用户已确定的审计标准

- 以相对公平为目标。原始 AttAcc 已有且当前共同沿用的近似、遗漏或错误，有对应依据即可接受，不为提高绝对精度要求重建模型。
- 对方法和 baseline 共同适用的模型限制，不单独列为问题。**不要求同一个近似在每档造成完全相等的数值误差，也不要求证明误差精确抵消。** 重点检查额外的分档规则、错误输入、独有的计数/分支遗漏，以及超出已声明机制的改动。
- 有事实差异时，先展示 AttAcc 是否建模、当前怎么变、影响哪些档、证据到哪一步，再由 chenyi9 决定是否算问题。新增机制并不自动等于不公平，未量化的风险也不自动成为整改任务。
- **FlashAttention 是我们增加的共同 GPU 模型，必须启用。** 它不能以原始 AttAcc 没有为由退回 legacy；各档 GPU 部分共用同一模型。其已有来源和解析近似如实说明，不额外要求以硬件实测取代共同模型。
- A1/A2 的独立 baseline、A3b 起的逐级机制、A5 的 PIM prefill+MQ 配套机制、A6 的逐 request 简单比价，均保持此前已接受的定义。

以下“建议保留 / 暂不列问题”是主审依据新规则的分类建议；没有把用户尚未逐项作出的裁决写成既成决定。代码、实验脚本、workload 和已有结果均未改。

## 2. AttAcc 已有或共同使用的部分：撤回自动整改要求

原始源码直接由 git 对象提取，见 [来源摘录](ATTACC_UPSTREAM_REVIEW_EXCERPTS.md) 和 [hash 记录](attacc_relative_fairness_review_evidence.json)。下表的原始文件行号均属于 `c600051`。

| 事项 / 原编号 | 原始 AttAcc 是否建模 | 当前共同口径与比较影响 | 本次分类 |
|---|---|---|---|
| PIM energy 不由 Ramulator 原生输出 | 有。`src/devices.py:326` 用命令派生流量、FLOPs 和 AttAcc energy table 算能量 | 各 PIM 档共同单价；不能因为是公式就认定拟合了某档收益 | 沿用，不列问题 |
| QK/PV 的 ALU 合并记账，MP-06 | 有。原始 `devices.py:330–350` 在 score 收一份 ALU，context 返回零 | 当前保留同一口径；这是上游限制，没有证据是 A5 私有优惠 | 撤回补收 PV ALU 的要求 |
| ACT energy 摊入列访问，而非逐次 ACT 单计，MP-06 | 有。原始 wrapper 的 `mem_acc = mac*32` 与 bank 因子配合单位能量 | 共同 AttAcc energy 模型；新增 ACT 统计不代表必须另加 energy | 沿用，不要求精细化 |
| 不满 stack 仍按代表 stack 复制能量，MP-05 | 有。原始 `ramulator_wrapper.py:163` 向上取 head 数，`:212` 将 traffic 乘 stack 数 | 当前头数分配和复制代码有所扩展，但“代表负载复制”有明确上游依据，所有 PIM 档使用 | 共同近似，不因绝对高估本身要求改成逐 stack 求和；发现某档额外复制才单列 |
| A100a 合成带宽、无限 L2 等平台假设，MP-04 | 原始 config 已有相关平台常数 | 共同平台假设；正文已改 A100，不能再报旧 H100 不一致 | 不列本轮公平性问题 |
| 普通 KV store/read、DIE/TLB 的独立时间/能量 | 原始 AttAcc 没有这些新增 bookkeeping 成本；已有 KV 链路项 | 已按用户裁决统一零成本、保依赖；不要求补普通 RD/WR 性能模型 | 已接受，不恢复旧收费建议 |
| shape/count 仿真、数据值和完整多请求缓存生命周期未建模 | 原始 wrapper 以长度/head/精度等生成 dense attention trace，并不执行 KV 数值正确性或多 agent 写入系统 | A1 的 count/dense 表示与共同 history 抽象不能仅因不完整就判不公平 | 共同抽象本身不列问题；新布局是否另给某档不同工作量另看下表 |
| shape CSV 没有版本指纹，V07/MR-03 的旧入口部分 | 有。原始 `ramulator_wrapper.py:220–241` 按 shape/配置查 CSV，没有构建 hash | 旧接口共同继承；新缓存还增加了部分指纹，不能因未做到完美就判更差 | 没有混用版本证据时不列问题；不同档实际混版本才提交裁决 |
| FA 读图效率、短 Q split-k 默认关闭、融合/L2/occupancy 近似，RC04/MP-04 | FA 是新增；原始有解析 GPU 模型及部分共同平台假设，但没有当前 FA 实现 | 当前各档使用同一 FA 公式与参数时是共同近似；不能仅因不是本库实测就判不公平 | 启用 flash 是硬要求；近似模型本身不列整改，仅如实注明来源 |
| GPU 内部按 Q head 计 K/V 流量，MR-02 的共同 GPU 部分 | 原始 MATMUL 按 numOp 计流量，未实现当前 GQA KV 共享优化 | 当前各档 GPU attention 共用同一公式；不是某个 rung 单独禁用优化 | 共同近似先不列问题；与下面新 validator 执行失败、batch/single 字节不一致分开 |
| X2G 启动延迟、远端 HBM 限速与辅助 stream 参数，MP-08 | 原始 X2G 只按带宽；当前借用原始 G2G 拟合截距并新增 HBM streaming 近似，不是原式照搬 | 属于 refined/flash 的共同扩展，有来源说明；此前未证明某档单独使用不同参数 | 共同使用时不要求重新校准。A6 给不存在的传输加价是另一个分支问题，见 V06a |
| P 先搬后算、有界 buffer 未完整模拟，MP-07 | 原始 bank generator 单 head 尾路径 `:342–351` 也先输出 context movement，再输出 MAC | 继承部分不要求补完整 buffer/streaming 仿真；MQ 是允许的新增机制，不能将上游共同粗粒度调度算作它的私有优惠 | 先按共同限制保留；若指出 MQ 独有遗漏，需另给差异证据 |
| workload 人工构造、history 预置、多轮不是完整缓存回放 | 原始没有当前多 agent workload；属于本仓库新增输入抽象 | 用户已接受合理自造 workload；同一输入/计划适用各档即可，不能强加生产代表性要求 | 不因简化本身列问题；只检查是否分档换输入或机制未按声明执行 |

## 3. 最近运行补审的逐项重分类

| 编号与事实 | 原始 AttAcc 是否有对应建模 | 当前是否是分档差异 | 交用户审阅的分类 |
|---|---|---|---|
| RC01：CLI/裸 ladder 默认 legacy，sweep 默认 flash | 原始没有我们新增的 flash | 入口不同可能切换模型；实际是否混用须看命令，不能猜测 | **必须启用 flash，已由用户明确确定。** 文档命令已补齐，代码默认未改 |
| RC02：pipe=True 下 append-order 会让 ready event 错过空闲窗口 | 原始只有顺序计价、`wrt_io_busy` 和 `_pipeline` 公式，没有同型多请求 DAG 资源日历 | 当前各档共用 `_schedule_cacheblend`；已有探针证明空洞，但未证明某档用了不同调度规则，也未测净收益偏置 | **撤回“必须换 ready queue”的要求。** 建议作为共同调度近似保留；原始 AttAcc 不是同一个实现，是否仍列问题由用户定 |
| RC03：旧 JSON 有 pipe=false，其他配置字段缺失 | 原始默认 pipe=False，但 `_pipeline(level=False)` 仍估计 attention↔通信重叠，并非设备全串行 | 当前 DAG 的 pipe=False 含义与原始公式不完全相同；历史字段只证明当时开关值，不证明刻意削弱某档 | 继续如实标注开/关/未知；撤回仅凭旧 false 或缺 metadata 就排除全部结果的结论。相同配置可作为共同口径；分档混用再交裁决 |
| RC04：flash 参数来自论文图近似读取 | 原始无 flash，属于新增且有来源说明的解析模型 | 统一启用时各档相同；不是只有 Fugue 用更快 GPU | 记录来源即可，不把“非硬件实测”本身列为公平性问题 |

**更正原始 pipeline 的描述：** `c600051:src/system.py:281–282` 在 PIM 路径无条件调用 `_pipeline(decoder_block, pipe)`；其 `:127–133` 在 pipe=False 时仍减少通信时间，`:156` 仍令 softmax 的独立时间为零。因此前文“AttAcc 原始 no-pipe 就是全串行”的说法过强。当前 DAG 的 SERIAL 时间轴是新增约定，不能仅靠同名开关说二者完全相同。这个更正提供来源事实，不自动要求重写调度器。

## 4. 既有实现反例：逐项给来源与相对影响，等待裁决

| 编号 | 原始 AttAcc 建模了吗 | 当前事实与影响边界 | 主审建议，尚待用户裁决 |
|---|---|---|---|
| V01a / MR-01：GQA validator 拒绝合法 KV link | 没有当前多请求 KV validator；原 model 也没有当前完整 GQA 数据通路 | 新 producer 已用 KV width，validator 仍用 Q width；PIM 档在该输入下失败，A2 不走这条验证 | 建议保留为新增执行回归；这不是共同参数近似 |
| V01b / MR-02：batched KV bytes 与 single 不同 | 原始有 MHA 通信尺寸，没有当前 batch/single 两套 GQA 扩展 | 当前 batched 保留 Q width，single/prefill 已用 KV width；会增加走 batched PIM 路径的流量 | 建议保留“新增分支不一致”；若用户接受所有 PIM 档共同沿用旧字节口径，需明确该口径，不自行改 |
| V02：A4c 起 diff 区偏移落在 pseudochannel | 原始有 ALL-BANK mapper/action/preq；这几份源码与当前逐字节一致。原始没有 master/diff region | 新 diff 区只用于 A4c/A4e 等，A3b 用普通 append；偏移不能隔离 ALL-BANK row，存在专属于新布局的地址语义差异 | 建议保留为布局 claim 的候选问题。修改原 mapper 不是本项建议；关键是新地址是否使用了原语义允许的独立区域 |
| V03：同一对象子段使用本次 extent 长度作为 V stride | 原始有 dense K/V tiling，没有持久对象任意子段 extent 接口 | 新接口在所有布局复用，但不同布局的 extent 分段不同；已有命令反例，尚无实际周期/收益偏差量化 | 待裁决：不是原始同型场景；也不能仅因共同接口有反例就宣布逐档不公平 |
| V04：多 head 绕回 channel 时地址重复 | 原始以 head count 生成 trace，没有当前 PhysicalLedger 的 head→channel modulo 分配 | 新 ledger 的共同边界，A3b–A6 都可能遇到；尚未证明某档单独改变规则或获取额外优惠 | 建议先按共同边界说明，撤回自动要求更强分配器；用户决定是否需要管 |
| V05a / N01：STORE 元数据未使用 scan ledger 地址 | 原始没有当前 STORE/PhysicalLedger 写入回放 | STORE 已统一零成本，地址字段本身不计价；其不一致不再证明新增时间/能量惩罚 | 建议不以元数据不完整单独判性能问题；只有造成不同扫描/依赖才另列 |
| V05b / N03：JSON 预约序与 DAG 执行序不同 | 原始没有多 request 写入顺序与预分配规则 | 当前可视作共同静态预分配；与“实际写入驱动”的文档措辞不同。未证明只是 JSON 重排就获得分档专属优化 | 待裁决是否接受预分配定义；不自动要求完整写入回放 |
| V05c / N02：重复 fingerprint 的 diff 跨 own KV 合并 | 原始没有软件 diff 与 burst packing | 新 A3b 可能得到更紧凑 packing；这会增强 baseline，而不是弱化 A3b | 给用户选择接受为合理 naive baseline，或要求严格连续 burst；不代替用户选 |
| V05d / N04：diff cursor 跨 owner 共用 | 原始没有 per-agent diff gather | 涉及 A4c 起的 claim；交错预约可破坏 per-agent 连续性，但当前静态构图未证明默认输入必现 | 条件性待裁决，不作为默认性能错误 |
| N05：A3b master/diff 使用不同轮转计数 | 原始没有这个软件 baseline | 保持各档 master 放置相同是此前已接受的约束；没有必要强求最字面的单流轮转 | 沿用接受的 baseline 定义，不列问题，文档如实描述 |
| V06a：A6 给 fresh GPU 候选加不存在的回读 | 原始没有自动选边；原始 X2G 无当前新增 flash 链路截距 | 当前只有 A6 chooser 把不存在的传输放入 GPU 候选；实际 fresh GPU 分支不执行该操作 | 建议保留为 A6 独有候选成本差异，不要求更复杂选择器 |
| V06b：A6 PIM 候选漏已有 extra-Q 链路 | 原始没有 query rotation variants 或 A6 chooser | 新执行路径有 extra-Q，选择器少计；只影响动态选择决策，固定 A5 不做该决策 | 建议保留为新增选择器操作覆盖问题；仍按逐 request 比价 |
| V07a / MR-03：新工具链指纹漏共享库 | 原始缓存完全没有该指纹；新实现部分增强 | 当前所有档共用该缓存，先前文件桩只证明可碰撞，没有证明正式结果确实混版本 | 撤回缺指纹即不公平的判断；共同固定版本下不管，实际混版再审 |
| V07b / MR-05：只改 V row 的通用 extent API 缓存边界 | 原始无任意 extent API；属于新增通用接口 | 当前 ledger 固定 K/V 间距，不触发旧探针的任意 V 条件 | 不作为当前 ladder 问题，仅保留边界记录 |
| V08 / MR-04：tier 汇总漏 A2、部分列用错终点 | 原始没有当前多 tier CSV 汇总器 | 新 collector 依赖 PIM batch 记录，A2 因无 batch 缺行；这是报表分档差异，不是公共硬件近似 | 建议保留为结果呈现问题；E2E/TBT 正确字段不因该项一并否定 |

此前已修的共同计划、STORE 收费、A1 路由/精确长度、MQ 接入、GQA ALU query 数等不重新开项。旧专项中 GPU 内部 GQA 流量、FA occupancy、辅助 HBM streaming 参数等共同模型近似，在没有分档额外优惠证据时按本页共同限制处理，不扩展为新的绝对精度改造任务。

## 5. 证据和当前决策边界

来源判断以原始 git 对象和当前调用链为依据；独立计量 agent `attacc_model_provenance` 另行读取原始 pipeline、energy、head/stack 复制、P 搬运顺序、GQA 和缓存代码，复核了上述继承与新增边界。执行反例引用 [主审](REAUDIT_8c51672.md)、[独立存储审计](INDEPENDENT_REAUDIT_8c51672.md)、[计量专项](MODEL_PROVENANCE_REAUDIT_8c51672.md) 和 [运行补审](RUNTIME_CONFIGURATION_AUDIT.md) 的已有证据。本轮没有新跑性能实验，没有把 helper 的任意时长解释为实际收益。

本页不是修复批准单。FlashAttention 必须开启是明确决定；其他新增/分档事项交 chenyi9 逐项决定。共同 AttAcc 限制不再妨碍接受同口径比较，也不要求为了绝对精度添加 AttAcc 没有的成本。此前关于“存在建模不完整所以整套结果不能用”的笼统表述，由这里的来源和相对影响分类替代。

变更原因和逐文件记录见 [session](../../../docs/sessions/2026-09-05-attacc-relative-fairness-ruling.md)；文件核验见 [manifest](attacc_relative_fairness_review_manifest.json)。历史 manifest 保持原样，对应其生成时点。

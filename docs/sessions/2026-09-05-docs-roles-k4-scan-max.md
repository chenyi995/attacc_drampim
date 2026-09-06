# 2026-09-05：文档分类、k4 与最慢通道、部分列追加

chenyi9 要求：分析低 AI 复用 prefill、k4 是否更适合PIM及能否拉开A5/A6；docs 至少分为实验指导、性能分析、audit，写清怎么跑；scan取最长channel；补查4-token输出把后继diff推到半列处的连续追加效应。继续沿用八通道与四项延迟指标，只audit、不改模拟器。

## 为什么整理

原README/run guide/design ladder重复混放命令、旧数字和audit状态，旧指南仍将k8/NUM_HBM1视为固定主配置，也没有完整scan/TTFT取数说明。因此建立三类主文档，旧入口跳转，机器页增加参数优先级说明。分类前原文字节（含用户未提交的run-guide增补）保存在docs/archive/2026-09-05-before-categorization，未用git checkout/reset覆盖。

## 文件变更

| 文件 | 原因与变更 |
|---|---|
| docs/README.md | 改为用途导航，去掉重复旧跑法 |
| docs/experiments/README.md | 新指导：八通道/k4/k8、GQA的MQ容量、完整事件命令、B1/RUNGS默认、指标数据要求 |
| docs/analysis/README.md | 新分析：逐级节省、真实m/R/sweeps、A5/A6条件、输出挤偏后续diff |
| docs/audit/README.md | 按核验对象链接证据，不重复维护第二套状态 |
| README_run_guide、README_audit_fixes | 兼容入口 |
| README_design_ladder | 精简当前机制、A1/A2独立基线，旧比例归档 |
| docs/run两个机器页 | 保留机器环境和历史例子，注明新参数以指导为准 |
| audit/.../SCAN_MAX_AUDIT.md | 记录执行与A6均取max(actual time)，无需改代码 |
| audit/.../PARTIAL_COLUMN_APPEND_AUDIT.md | 正确表述用户输出造成后继起点偏移的concern，记录当前行对齐覆盖边界 |
| audit/.../LAYOUT_BENEFIT_CEILING.md | 补两个offset对齐例外；真实名义行数不变，简化token公式加限定 |
| docs/sessions及audit索引 | 本轮原因、验证、证据 |

## 分析与独立核验

ledger_trace_boundary_audit使用真实plan/TLB比较k8/k4：后续话少m32→24，LLAMA3 GQA4的sweeps16→12；owner工作不变。两个writer请求只修正一个新chunk。k8新增diff总数1008，k4为504，没有device/Ramulator定价。

independent_fairness_audit核实际per-lane返回、join、A6与AttAcc。固定价格让token少通道更慢，helper和A6取7µs，不取均值或最多token通道。warm dump用真实价格但缺唯一sweep身份，full events用于完整归因。

用户纠正后，审计聚焦“短普通输出推进后继diff起点”，不再用“小diff自身整组取整”替代该问题。当前A3b短master与新burst整行对齐，消除了这类偏移；该机制的潜在收益尚未由当前几何表达。上游AttAcc没有此碎片append对照，按模型覆盖范围报告，不擅自改baseline。

主审核入口/collector：ladder固定events=none，sweep对B1默认A3b/A6，因此指导提供直接main.py保留full和显式RUNGS的跑法，不声称旧collector已能输出缺失指标。k4是否扩大A5/A6要看实际价格与E2E；更多两档同走PIM不意味着更大选择收益。

文档与数字由/tmp/docs_roles_k4_20260905.py及保留的Python探针生成/复制。验证包括本地链接、bash代码块语法、原文归档摘要、实现/workload初末hash。没有执行文档里的模拟命令，没有修改模型、collector、workload或论文。

## 最新澄清与独立文档复核

部分列专项记录两层覆盖限制：A3b 整行分配消除输出尾部偏移，trace 列命令数量也未按起点偏移调整。独立两轮 plan 复核成立；收益方向分开报告，因为行分散可能反向影响 A3b。证据保存在 audit/2026-09-05/archive/partial_column_append。

attacc_model_provenance 复核新指导和分析后，去掉 PIM 价格单边“共同项”，补充 A5 的 MQ/PE 也影响 decode；子集只含 A4e/A5/A6 时将 summary 参考档改为 A4e，避免相对结果表缺失。

chenyi9 随后提议按 context 分 NVLink/PCIe。新增 LINK_TIER_ASSUMPTIONS.md，并在实验指导/分析增加对应一节：区分现有固定链路选项与尚未实现的容量溢出；按模型 KV 字节量推导假设快层容量，说明大量复用与大量 fresh 请求的 A5/A6 方向不同。用户提议记录为实验假设，没有当作已经确定的容量或已完成的实现。独立 agent 核上游与当前链路，证据在 archive/link_tier。

## 最终链路裁决：沿用原 NVLink

chenyi9 裁决：“那就不要做两级了，就是NV Link 链接GPU 和之前一样”。这是对上一节实验提议的最终决定，前述分层假设不再是当前方案。

因此将实验指导第7节与性能分析第8节改为原 NVLink 配置；移除当前指导中的容量阈值和改用 PCIe 的跑法建议。LINK_TIER_ASSUMPTIONS 标为已关闭、两级方案不采用，audit 与 session 索引同步状态。旧文字以原始字节保留在 audit/2026-09-05/archive/link_tier 中，便于追溯为什么撤回。

仅同步文档；运行命令仍为 --pim-link nvlink3，模拟器实现和已有计价无需修改。没有运行性能实验。修改脚本和验证结果保存在 archive/link_tier/nvlink_ruling_docs.txt、nvlink_ruling_validation.json。

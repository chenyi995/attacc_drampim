# W1 与旋转 diff 设计复审

日期：2026-09-06。chenyi9 要求重新 audit 论文、实现与构造 workload 的合理性，并计算是否能出现各档差异。沿用此前“只 audit、不改代码”及独立 agent 复核的授权。

## 为什么需要这轮复核

本轮开始时，新 diff 布局和 W1 协议正在提交：A4c 改为 diff 行在 head 的通道间轮换；A4e 新增按 agent 聚集并用表放置 diff。此前按末通道集中 diff 得出的地址与收益结论不能直接沿用。W1 也替代了旧 C 协议，必须按新输入的 parent 链与实际存储重新计算。

计算快照为 `c78dc76`，设计实现来自 `753044b`；收尾时为 `074cbeb`。期间新增提交只更新文档和 collector 标题，已比较涉及的模拟器源码与输入哈希，计算仍适用。论文版本为 `8fe2022`。完整指纹及源码副本在 [本轮证据](../../audit/2026-09-06/evidence/provenance.json)。

## 审计发现及为什么这样记录

- 新旋转与按 agent 分组已进入论文设计/评估声明，本轮没有发现修改 A3b 分支以削弱 baseline。把它记录为已有依据的机制变更，避免沿用旧阶梯误报“偷偷加入分组”。
- W1 的 main 确实持续继承不同轮次回答的旧 diff。修正占行下降与最忙通道列请求下降是不同指标，报告同时记录二者，防止把局部容量/占行收益当作 scan 或 TBT。
- A3b 同轮 burst 可以合并；A4c/e 对象各自取整会增加列请求。保留这一不利现象和独立地址证据，没有改输入或 generator 来强求单调加速。
- W1 的两对 worker 有共读，但实际 batch 的全组交集为空，不能用它证明跨会话 decode MQ。报告将这点与 prefill MQ、GQA 内部 MQ 分开，避免误判 A5 所有收益都不存在。
- 新 A4e 放置分数不是当前扫描的唯一物理行数；独立探针核出了具体落点差异。按启发式定义与文字是否一致处理，未把已接受的全 DAG 信息重新判为不公平。
- 论文方法章节、旧实验矩阵和图的 TODO 尚未同步；当前执行模型的 Ramulator 计价与分析脚本的手工标定也必须区分。报告列出具体章节、源码依据和 AttAcc 是否有相同机制。
- 首次审计公式假设 worker 每轮必然产生修正，在 worker 回答更长的变体触发了断言：某轮复用文档恰好与 owner 的 offset 相同。修正的是审计公式，使其按偏移扣除该轮修正，并保留发现过程；workload 没有改动。

## 实际写入的内容

| 文件 | 内容与理由 |
|---|---|
| [audit/2026-09-06/README.md](../../audit/2026-09-06/README.md) | 当前结论、可读手算、合理性边界、各事项的 AttAcc 来源与相邻档影响；本轮主入口 |
| [CALCULATIONS.md](../../audit/2026-09-06/CALCULATIONS.md) | 每个 W1 变体的实际地址/行/列数及 GPU 侧收支平衡预算 |
| [w1_handcheck.py](../../audit/2026-09-06/w1_handcheck.py) | 从真实输入构建复用计划和 PhysicalLedger，核相等工作量及生成器地址；留下可复算过程 |
| [finalize_report.py](../../audit/2026-09-06/finalize_report.py) | 从结构 JSON 计算报告数字，逐字复制独立证据，记录收尾源码一致性 |
| `audit/2026-09-06/evidence/` | 输入、源码/hash、原始静态输出、独立审查脚本/报告；区分静态结果和历史实跑 |
| 旧 audit 入口、性能分析入口、session 索引 | 加入当前复核链接，避免读者把旧版本通过项当成新设计的完整认证 |

只写入审计文档、计算脚本与其证据；模拟器、论文、实验输入、运行结果均未修改。没有提交或推送。

## 验证与独立检查

执行 `python3 audit/2026-09-06/w1_handcheck.py` 和 `python3 audit/2026-09-06/finalize_report.py`。它们只做静态计划、地址生成与解析 GPU 价格计算，没有运行 Ramulator、完整 DAG 或性能 sweep。验证记录见 [checks.json](../../audit/2026-09-06/evidence/checks.json) 和 [handcheck.log](../../audit/2026-09-06/handcheck.log)。

独立 agent 分工：Rawls 审论文/代码声明及 A4e 分数；Dirac 审真实地址、跨行边界、层间分组及 decode 后段；Harvey 审当前产物、MQ 成立条件和 AttAcc/实际计价来源。原始脚本、报告及 JSON 已复制到 [independent](../../audit/2026-09-06/evidence/independent)，复制清单在 [manifest](../../audit/2026-09-06/evidence/independent_manifest.json)。

当前 W1 没有完成的新全阶梯性能报告可供本轮证明 TBT/TTFT/E2E。收支平衡门槛不冒充选边测量；旧 M 结果也不冒充当前 W1。需过目的事项集中在主报告第 6 节；本轮没有替 chenyi9 决定实现修法。

## 追加：连续 diff 能合并，修正审计定性

chenyi9 指出：A4c 已把 diff 单独连续存放，同轮能合并，跨越很多轮只要实际连续也能合并。这个判断成立。此前审计列出的额外列请求虽然忠实于当前生成器，但没有明确把它定性为实现漏算收益，容易让读者误以为是布局本身的代价；本次更正这一表述。

复核确认 `PhysicalLedger.extent_groups` 按对象输出描述符，明确不跨相邻对象合并；QK 和 PV 随后分别对每个描述符取整。Ramulator 的行缓冲不能删除已经生成的冗余列命令。应保留对象的逻辑元数据，同时按实际连续地址合并兼容的物理扫描段；同一规则对 A3b 也适用。这不是新增布局 claim。

新增 [coalescing_check.py](../../audit/2026-09-06/coalescing_check.py)，从上一轮保存的 W1 地址提取 diff，仅合并 K/V 均连续且同通道的描述符；断言 token 地址集合与数量不变，再调用真实生成器统计 QK/PV。结果、差别及合并条件见 [更正说明](../../audit/2026-09-06/COALESCING_CLARIFICATION.md)。A4c 的全局流仍可能混有其他 agent 的未读修正，不能跨过这些空洞硬连；A4e 的按 agent 分组则能形成更长的跨轮连续段。

已更新主报告及其生成脚本：本项改列为“低估已声明布局收益的实现遗漏，应交执行 agent 修复并重算”。修复前原始数据保留；模拟器、workload 和论文未改，没有运行性能仿真。

## 追加：审阅执行agent修复及“不改”回复

chenyi9 表示已修复，并要求判断md里“不改”的回复有无道理。本轮读取主报告第8节、工作区 `extent_groups` 合并改动、相关测试以及论文/指南diff。

新增 [review_execution_reply.py](../../audit/2026-09-06/review_execution_reply.py)：按真实W1计划检查当前输出恰好等于旧地址按共同物理连续规则合并，token读集和重算量不变；同时由真实生成器检查修复后残留的额外尾行。新证据保存在 `execution_reply_review/`，未覆盖修复前原始数据。执行已有 `PhysicalLedgerTest`，12个测试通过；没有再次运行全量测试或性能仿真。

审阅结论见 [EXECUTION_REPLY_REVIEW.md](../../audit/2026-09-06/EXECUTION_REPLY_REVIEW.md)。A4e保留有明确定义的累计评分是合理启发式；MQ与r−2语义文案已收窄，不要求另加机制。尾列取整的上游来源成立，可以作为保留近似，但当前默认W1仍触发布局相关的额外整行，不能声称各档误差相同。这里区分“保留简化的范围决定”和“实现已严格对应存储”的正确性结论，没有替chenyi9追加修法或推翻此前省略半列case的决定。

主报告追加第9节链接，保留执行agent第8节原回复。论文尚余的图TODO与少数方法措辞仅记录，没有修改论文。模拟器与测试文件均保持执行agent的原改动。

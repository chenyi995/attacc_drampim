# 2026-09-05：布局收益上限与 B1/T9 回复核验

起始 revision：`da220d1ea5c57358fb2f2d26a897a5178d0b7f0c`。chenyi9 要求看转来 AI 对 B1/T9、写入交错与异步的解释，回答布局理论最大多少、已经得到多少、什么情况下能更大。沿用此前只 audit、记录 session、独立 agent 复核的授权；沿用 decode 小流量固定传输开销忽略和 A6 简单逐请求比较服务时间的既定口径。

## 为什么新增这份报告

转来回复把静态 diff 占行、混合档位的最忙通道计数、手工估式选边数量和旧小跑利用率放在一起，读者容易误解为 B1 的真实性能数据。原 audit 主入口主要记录实现问题，不适合把本次理论边界混入旧事项状态，因此新增 [布局收益专项](../../audit/2026-09-05/LAYOUT_BENEFIT_CEILING.md)，保留旧事项的历史结论。

本轮明确了四个差别：

1. 当前全局 diff 区紧排与正文同 agent 跨轮连续紧排不是同一个保证；B1 当前拿到理想可省 diff 行数的 57.14%。
2. A4e 必须和 A4c 比。原 reused 子集独立收益是 10.94%，不是原 A3b→A4e 的 9.46%。加入当轮 fresh 后为 7.05%。
3. 找到的新 small/out 结果实际为 5 个 tier=0 请求，与 B1 多轮 DAG 不同，E2E 合计改善 0.517%；不能用它计算 B1 达成理论上限的比例，也不能混用旧 session 的 220 ms。
4. A6 没有 queue/busy 输入；解除 tier 屏障不自动修改预分配地址，不自动构成完整异步。因此纠正确定性的收益承诺，保留条件性的重叠机会。

## 文件变更及原因

| 文件 | 变更与原因 |
|---|---|
| `audit/2026-09-05/LAYOUT_BENEFIT_CEILING.md` | 新增面向不了解项目的读者的解释，逐一标注理论、结构和已有真实运行；给相邻档收益与扩大条件 |
| `audit/2026-09-05/archive/layout_ceiling/` | 保存 Python 审计计算、JSON、独立复核、原始输入/关键源码/小跑产物与 SHA256，防止未提交 workload 或 scratch 输出后续被替换后无法复核 |
| `audit/2026-09-05/README.md` | 增加本次专项入口，旧问题文档的范围不变 |
| 本 session 及 `docs/sessions/README.md` | 记录为什么改文档、具体分析边界和验证方式 |

未写实现、测试、论文、workload 或性能输出。已有 dirty 文件包括 generator、run guide、旧修复 session 及 B1/T 系列输入，本审计保持原样；不以本次只读分析补做用户尚未下令的运行。结束摘要检查另发现 `output/analysis/b1_levers.py` 有非本审计写入的并行更新，新增 headroom 诊断。初始版本已从会话读取内容恢复并严格验证等于初始 SHA256，最终观察版本另存；主模拟器、workload 等初始文件均未变化。此差异单列记录，不能把所有工作区变化都归为本轮 audit 的写入。

## 实际验证与独立复核

全部计算保存为 Python 源文件，在 `/tmp` 执行并归档为 `.txt`；这些文件仍可由 Python 执行。几何探针没有 device/模型定价调用，trace 探针只生成小段命令，调度反例使用任意固定时长事件，没有启动 Ramulator。

- `python3 /tmp/layout_ceiling_a4c_probe.py`：真实 plan/ledger，B1、T9 32/8、4 chunk、4 agent、1 chunk；总行、diff 行、最忙 lane 几何，与正文逐 agent 条件对照。
- `python3 /tmp/layout_ceiling_a4e_probe.py`：真实三档 ledger，逐请求两种读集，master 整块 subset-sum 下界，assert 两档读量相等、当前工作量不低于乐观下界。
- `python3 /tmp/layout_ceiling_a4e_review.py`：独立检查 A4e 数值、identity、固定 diff 和每 head 两通道条件。
- `python3 /tmp/layout_ceiling_side_proxy.py`、`layout_ceiling_scheduler_probe.py`、`layout_ceiling_trace_order_probe.py`：分别验证估式来源、加入顺序调度限制、唯一行数不等于 ACT 次数。
- `python3 /tmp/layout_ceiling_results_20260905.py`：只读既有三档 JSON/CSV/log，检查相同条件、算 TBT/E2E/lane 时间和利用率分母。
- `python3 /tmp/layout_ceiling_report.py`：从证据 JSON 生成报告和本 session，使用 Python 复制证据并计算 SHA256。

`ledger_trace_boundary_audit` 独立审 A4c 与正文；`independent_fairness_audit` 独立审 A6/调度和主审 A4e 算法；`attacc_model_provenance` 独立提取真实小跑。主审核了源码和关键证据，再合并报告。已完成的是理论/结构上界和已有结果核验；B1/T9 没有真实性能结果，因此不声称获得其 E2E 上限兑现率。

最终检查：归档摘要、相同运行的 14 项共同配置检查和原始结果抽查通过；本轮文档本地链接无缺失；实现/workload 初末摘要一致。`git diff --check` 的全仓输出仅报既有 manifest.csv 的 CRLF 行尾，未修改该输入；文档范围检查单列通过。详见 [验证记录](../../audit/2026-09-05/archive/layout_ceiling/validation.json) 和 [验证脚本](../../audit/2026-09-05/archive/layout_ceiling/layout_ceiling_validate.txt)。

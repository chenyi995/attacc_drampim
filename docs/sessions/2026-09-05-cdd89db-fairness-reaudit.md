# Session：cdd89db 七档公平性只读复审

> 后续用户澄清 A6 只需逐 request 选择估价较快一侧，因此本轮记录中原先依据正文要求候选 DAG 的判断已撤回；不是代码修复。最新存储专项、修订原因与证据见 [后续 session](2026-09-05-storage-scan-and-request-choice.md)。本文保留前轮实际审计过程，当前结论以已更新主报告和存储专项为准。

日期：2026-09-05。开始/结束审计对象均为 `cdd89db04a85edae029fd3151165f1a488d6139c`；AttAcc 对照基点 `c600051`。起始 tracked 工作树干净，已有 untracked `experiments/paper_ladder/`。本次没有创建 commit、push 或修改实现。

## 1. 用户要求与边界

用户要求重新审查仓库大量改动，更新 audit，确认 A3b 到各档是否只改变声明机制、是否存在未声明优化或额外弱化 baseline，并明确要求 agent 单独审核。随后补充：

> “你可以用一些简单的workload 验证，但是你不要修改代码，只能audit”

因此本次只改审计/说明文档，归档 JSON、覆盖表、manifest 和审计探针的文本。不改变实现、测试、workload、实验脚本或结果；不执行完整矩阵、真实 Ramulator、GPU benchmark、RTL、综合或安装脚本。诊断程序最初写在 `/tmp`，归档为 audit 中的 `.txt`，没有作为生产功能接入。

沿用已确认的比较口径：A1/A2 独立 baseline；A3b 是朴素软件复用+PIM decode；A4c 每 agent/head 的 diff 集中；A4e 共读 placement table；A5 PIM prefill+MQ/配套频点；A6 自动选边。合成输入可以用来展示机制。DIE/TLB 额外成本继续排除，旋转归 GPU。

## 2. 为什么重新修改 audit 文档

旧主报告检查 `8750b5b`。当前已有 A1 GPU prefill、A3b master slot、fresh prefill、fresh GPU context 和 CLI `num_attacc` 的修复，以及新 workload/运行脚本。直接复用旧“全部未修”状态会误导。

另一方面，新设计/运行指南说“同一计划只换 preset”、A6 恒不差于两侧，仍不能由代码支持。复审需要同时撤销已经修好的旧指控、记录现存的额外差异，并解释证据的范围。发现某些配置不公平不能证明作者故意；发现某些遗漏低估 PIM 也不能证明整个结果保守。

## 3. 怎样审、由谁独立复核

主审对照当前 HEAD 与 `c600051` 的 diff、当前论文设计/执行/方法章节、实际 CLI、planner、布局、DAG、batch、报表、脚本和生成器。覆盖清单逐个登记相对基点的 **110 个文件**；数据文件按结构检查，历史产物按出处和时点检查，不宣称每行都有独立数值证明。

按用户明确要求启用两个 agent：

| Agent | 独立范围 | 产出 |
|---|---|---|
| `independent_fairness_audit` | 独立读当前源码/论文，另做小型布局、设备桩、MQ 参数探针 | [独立复审](../../audit/2026-09-05/INDEPENDENT_REAUDIT.md) |
| `attacc_model_provenance` | AttAcc diff、GPU/PIM 单价与工作量、MQ interval/trace、缓存/二进制来源 | [计量专项](../../audit/2026-09-05/MODEL_PROVENANCE_REAUDIT.md) |

主审读完两份报告后再次校核关键结论。例如，要求独立报告明确：修正集合不一致已经证实，但本版本 TLB 零计价、scan 补 full master，不能直接把旧的描述符收费惩罚解释成当前测得的性能劣势。

## 4. 实际验证和结果

主审命令：`PYTHONDONTWRITEBYTECODE=1 KVPIM_CPPCORE=0 python3 /tmp/fugue_reaudit_cdd89db.py`，exit 0。设备沿用旧审计的固定 1 µs 桩，仅使用真实 planner、layout 和 DAG 构造器；所有秒数只验证公式/依赖，不表示真实硬件表现。没有修改 monkey-patched 生产文件。

- 七档 256-token fresh 小例子确认 A1 GPU 路由、A5 PIM 路由、GPU context 形状、DIE/TLB 零成本等修复。
- 零 diff master store：A3b 2.097152 µs，A4c 0.1398101333 µs，15 倍来自旧 pool 带宽份额；此比值是单事件，非 E2E。
- shared consumer 列表排在 owner 前的合法小例子：consumer scan 5.066 µs 已结束，owner store 18.066 µs 才开始；无数据依赖，validator 放行。
- 资源别名 `PIM`、`PIM:pool0-14`、`PIM:pool0-0` 可同时占用同一物理 channel；master slot 虽持久，送给 trace 的 row 地址仍随读集变化。
- 42 个新 sweep 输入全部成功解析，并用实际 planner/单层 layout 元数据核查；42/42 的 A3b 与 A4c corrected-row 集合不同，全部 segment delta=0。这里没有展开 42 个性能 DAG。
- B0 interleaved 与 turns 的 requests/output/history 工作量不同；turns 的 history 是新预置 extent。table 预先知道 9/9 和 65/65 个 output fingerprint，所以旧文档“没有 output 信息”的性能原因推断缺证据。
- CacheCraft/CacheTune 的 PIM policy 路径仍回旧 runner，缺 decode 和 makespan。
- agent 另行确认同轮修正分页、master 子区间几何改变、私有 GQA decode 漏传 MQ，及不满 stack 的能量余数。这些既有可能抬高收益的路径，也有低估收益的路径。

外部来源检查只打开了 cuBLAS benchmark 作者仓库和 FA-2 论文页面：确认出处存在。具名 CSV 抓取未成功，未逐项验证 throughput 表；不宣称 H100/所有形状已校准。源码中已有完整本地论文内容，未修改论文目录。

## 5. 逐文件归档及修改理由

| 文件 | 修改 | 为什么 |
|---|---|---|
| [REAUDIT_cdd89db.md](../../audit/2026-09-05/REAUDIT_cdd89db.md) | 新主报告，逐档裁定、R01–R14、偏差方向、已修状态、建议顺序 | 当前版本不能沿用旧状态，也不能因 preset 名称就确认公平 |
| 两份 agent 报告 | 保留各自独立检查结果、来源与边界 | 让主审结论可以被单独复核，而不只依赖同一套旧报告 |
| [reaudit_cdd89db_evidence.json](../../audit/2026-09-05/reaudit_cdd89db_evidence.json) | 归档本轮结构反例、42 输入统计 | 不覆盖旧 evidence，也不把设备桩结果充作性能结果 |
| [reaudit_cdd89db_probe.txt](../../audit/2026-09-05/reaudit_cdd89db_probe.txt) | 归档 `/tmp` 中的诊断源码文本 | 提供可复现构造；不改生产代码或测试 |
| [file_coverage.csv](../../audit/2026-09-05/reaudit_cdd89db_file_coverage.csv) / [manifest](../../audit/2026-09-05/reaudit_cdd89db_manifest.json) | 110 个变更文件的范围、hash、论文/代码快照、验证方式 | 明确看了什么、怎样看以及版本界限 |
| 旧 `REPORT.md` / `PIM_TIMING_PROVENANCE.md` | 顶部添加新状态入口，保留旧正文和证据 | 旧 A1/fresh 路由等结论已过时；需要防止被当成当前事实 |
| `docs/README_audit_fixes.md` | 增加最新已修/未修状态入口 | 保持修复建议与当前状态一致，本轮不实施 |
| `docs/README_design_ladder.md` / `README_run_guide.md` | 增加审计状态提示 | 机制目标、同一计划要求、历史手算和 A6 全局下界不等于已证明性质 |
| `docs/README.md` / `docs/sessions/README.md` / 本文 | 新审计/session 入口，更新已有 workload 的事实 | 按用户要求持续记录本次动作及理由 |

## 6. 未完成事项与结论边界

本次没有修 R01–R14，没有重新获得可信的论文 speedup/energy 曲线，也没有完成 RTL/数值 attention 等价证明。主结论是**仍不能确认全档公平和所有相对 AttAcc 扩展都充分有据**，不是断言设计没有收益或有人蓄意操纵。

下一步建议先修同一逻辑计划、实际物理账本/资源及 owner 依赖，再修候选执行/选边、GQA/MQ/RoPE/history，最后冻结来源与指标后做获授权的验证。这些都是报告里的建议，未在本 session 实施。

最终文档检查及“实现/测试/workload 未改”快照比对结果记在本次 manifest；本 session 的历史 105/108 测试记录没有作为当前通过的测试重新引用。

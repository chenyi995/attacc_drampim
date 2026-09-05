# 2026-09-05：fbe6756 修改后的审计验收

本 session 记录主审与三个独立 agent 在 chenyi9 要求“又改过一轮，再检查”后进行的复查。**只做 audit，没有实施本轮发现的修复。**

## 对象、授权与方法

- 当前代码 `fbe6756eb062e44f09a0d100538563c5bd93ff08`，对照上次 `8c51672a3ef8b936340354b3211963cde8945c49` 和原始 AttAcc `c600051`。
- 阅读当前贡献 README、论文正文、两次版本间的实现/测试/脚本变更。21 个 regenerated turns JSON 用内存重生成逐份比较，不逐行人工阅读重复 JSON。
- 保持此前全部裁决：A1/A2 独立 baseline；A3b 同轮可紧排、跨轮须保留旧物理对象；A4c 布局、A4e 表、A5 prefill+MQ 配套、A6 简单逐 request 选边；共用 flash/pipeline；忽略估价 Q、保留真实回读；DIE/TLB/普通 STORE 不加无依据费用。
- 轻量 helper/设备桩测试在用户此前授权内；未启动 Ramulator、正式 sweep 或性能运行。原有 `experiments/paper_ladder/` 是开始时已存在的未跟踪结果，未改。

## 审查到的实现变更及原因

这里说明执行 agent 已经提交的改动及其依据，不冒称为本次主审实施。

| 已提交修改 | 原因/依据 | 本轮验收 |
|---|---|---|
| `7938a76`、`6092d3c`：diff base 512 MiB→4 MiB 与容量保护 | 原偏移只变 pseudochannel，未隔开 AttAcc ALL-BANK row | 原别名控制关闭，实际 K/V 行已分离，master 通道保留 |
| `b7363f2`：GQA validator/batch/A1 helper | 要按 KV heads 计存储与链路字节 | 七档 GQA/MHA 构图及 KV 宽度通过 |
| `17872f4`：A6 空回读与 Q 估价 | 用户指定只算实际项、Q 可忽略 | 零历史无回读估价；有历史计回读；context 返回保留；C8 输入仍需补全 |
| `3cdda44`：A2 tier 行 | 纯 GPU 无 PIM batch 不能被漏表 | 缺行修复，时间口径仍未统一 |
| `76fca34`：ladder 默认 flash | 新增 FlashAttention 必须各档共用 | 当前默认与 sweep 一致；直接 main 仍须显式 flash |
| `5618da2`、`c2cccf7`、`26495c2`、`1f1b5db`：旧 context 重列和 inherits_from | 用户要求后轮读原 master/diff，不能用新 history master 占位 | JSON/两轮绑定局部修复；第三轮来源、实际 prefill、A2 同策略、CacheBlend 分支仍缺口 |
| `a4669f4`：PIM 能量按实际 head 数 H/h 外推 | 避免 ceil(H/h) 把未满 stack 的 head 数补满；仍用 AttAcc 单价 | 共同路径缩放一致，不改时间；layout_probe 旧公式未同步 |

论文 `sections/05-execution.tex` 的选边段已同步为简单式，首层决定后沿用，Q 估价忽略，零历史不计回读。主审只读论文、未修改。此前实现 session §14 末“MP-05 未做、论文仍双 DAG”的文字是更早时点，**现在已不适用**；首层决定跨层沿用本来就是用户接受的规则，不是遗留问题。旧 session 的 C1 commit 字符串也以本次 git 查得的 `76fca34` 为准。

## 为什么改审计文档

1. 旧 CURRENT_ISSUES 仍把 C2/C3/C5/C6/C8 全写成待执行，已经不能反映当前代码。先完整保存为 archive 的 `CURRENT_ISSUES_before_fbe6756.txt`，再更新状态；保留 C1–C8 的锚点方便历史追溯。
2. 不用“计划哈希相同、两轮测试通过”替代实际扫描验收。把 C8 集中为一节，用 18-token 控制说明旧 diff 在绑定中存在、在 prefill 消费中消失，避免不熟悉项目的读者把问题误解为人为跳转罚时。
3. C6 保留“缺行已修”的事实，再用相同 summary 的 collector 反例解释剩余时间口径问题，不否定已有正确 makespan/cum_end。
4. 新能量缩放有共同 head 数依据，写清 AttAcc 原方法与本方法区别；只将诊断不一致列为新交接项，不说成实际性能造假。补充修改原因和线性外推限制，满足 session 记录要求。
5. C4 共同近似、C7 同轮合并撤回保持；diff 容量、per-agent 间隙、前缀变更作为边界说明，未凭假设扩大整改范围。
6. 更新 audit/README、archive/README、session 索引；修正运行指南开头仍声称裸 ladder 默认 legacy 的过期一句话，使其与本页现有正确表格一致。

## 主审与独立复核

| 审查方 | 有界范围 | 产物与结论 |
|---|---|---|
| 主审 | 当前 preset、GQA/MHA 七档、A6 估价调用、collector、21 个 turns JSON、论文与 session | `fbe6756_main_audit_evidence.json`、`fbe6756_workload_evidence.json`；C2/C5 通过，C6 部分修复 |
| independent_fairness_audit | 公开 loader→plan→bindings→实际 prefill/decode，三轮与 CacheBlend；真实两 agent 间隙 | `independent_c8_fbe6756_evidence.json`、`independent_c8_agent_gap_evidence.json`；独立确认 C8 未贯通 |
| ledger_trace_boundary_audit | diff/master 地址、真实 generator 与 mapper、heads/master 几何、容量边界 | `c3_fbe6756_evidence.json`；C3 原别名关闭，未重开 C4 |
| attacc_model_provenance | 上游与现有 head/stack 能量链、GQA/MQ、layout probe | `head_energy_a4669f4_evidence.json`；共同外推有依据，诊断仍旧公式 |

主审另外独立用三类 layout 复跑手工/当前生成器的三轮绑定，均在后轮找不到实际并未创建的中间 owner diff，见 `fbe6756_c8_binding_evidence.json`。C8.2 的漏读也逐行核对 compute/readback/scan/full_rows 源码。两个独立检查均与 agent 结论一致。

## 验证与结果边界

运行已有定向单测，共 **15 tests，45.601 s，OK**：

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest
  tests.test_placement.PhysicalLedgerTest
  tests.test_placement.StackEnergyScaleTest
  tests.test_workload.AgenticHistoryTests.test_gqa_model_builds_every_rung_with_kv_head_wide_links
  tests.test_workload.AgenticHistoryTests.test_a_later_turn_attends_the_diff_its_earlier_turn_wrote
  tests.test_workload.AgenticHistoryTests.test_every_layout_executes_the_same_correction_plan
  tests.test_workload.AgenticHistoryTests.test_consumer_waits_for_the_owner_store_even_when_listed_first
  tests.test_workload.AgenticHistoryTests.test_fresh_prefill_follows_the_rung_prefill_side
```

上面为换行展示的一条命令，测试结果由本次工具输出记录；未为补日志而重跑。已有两轮单测只验 owner/计算行数，未验实际 prefill 读集或第三轮，因此测试通过与 C8 反例并不冲突。

probe 的 GPU/PIM 时间是固定假价格；能量专项的 traffic 也是受控替代值。验证的是真实执行构图、传入地址、乘数及汇总器行为，不是 GPU/PIM 性能或数值 attention 正确性。小例子中的每个数均可追溯到归档 JSON。没有运行完整 sweep、没有重算旧实验结果，也没有声称全面证明公平。

审计开始保存源码/测试/脚本/workload 的跟踪文件 hash、已有结果 hash、论文 tex hash。结束核验和文档链接核验见 [manifest](../../audit/2026-09-05/archive/fbe6756_audit_manifest.json)。本 session 与 [当前问题页](../../audit/2026-09-05/CURRENT_ISSUES.md) 是本轮交付；下一步是执行 agent 修复 C6/C8 与诊断输出，再按实际读集复验，无需再次讨论已定原则。

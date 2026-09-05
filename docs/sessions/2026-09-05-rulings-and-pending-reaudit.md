# 2026-09-05：记录裁决，重新审计 C4/C7/C8，并给出 C5 估算式

代码 revision：`8c51672a3ef8b936340354b3211963cde8945c49`。chenyi9 明确当前 agent 的任务仍是 audit；已裁决内容写入文档，供执行 agent 修改。主审只更新审计/索引/session，独立 agent 只生成 `/tmp` 证据。

## 用户裁决与最新解释

- C3：diff 地址需要修正。主审将验收目标写为“在原始 ALL-BANK 语义下不重叠”，避免仅比较地址整数。
- C5：Q 传输容易，可忽略；历史回读需要删除无依据项，并同步论文和 codebase。用户随后明确“实际找一个公式估算，需要哪些部分就估算哪些项”。因此当前交接保留实际需要的历史回读、空回读为零；忽略 Q 输入传输，按真实模型估 attention/扫描和结果返回。先前删除范围澄清由最新指令收敛，不再等待重复批准。
- C2：修复 GQA；C6：更新报表。已从待裁决移到已确定待执行，没有写成修复已完成。
- C7：用户进一步明确同一 round 多组 diff 可以正常追加紧排，跨轮已有输出/新写入后不能聚成同批。因此定义已定，撤回旧单 prefill 反例；重新核对公开两轮输入及现有 turns/interleaved 编码，形成 C8 多轮覆盖边界。C4 重新审计后，主审按用户既有共同近似规则归类，不再提出重复裁决；不是记录一条用户未作出的专项批准。

## 为什么调整原来的审计结论

原 C5 将漏算 extra-Q 当作待修问题，现已被用户接受的近似取代；继续要求补费会违背裁决。新 C5 给执行 agent 一个可落地的局部服务时间估算式：GPU 的真实回读 + 共同 FA 计价，PIM 的每 sweep 通道 max 求和 + 结果返回，零字节 Link=0，Q 输入传输忽略。它沿用设备和 Ramulator 模型，不加入手拟合系数，不要求双候选 DAG。正文当前还是双 DAG 的文字描述，需由执行 agent 同步，主审只定位。

原 C7 的上下文 `D → own → D` 不能直接证明物理上曾按三个时刻写入。真实 prefill 整批 QKV/链路提交，合并又在软件预约前置发生。新报告按用户最新 round 定义关闭同轮合并的指控；公开两轮的 owner 不同，没有折成同一个 diff 对象。token 并未去重或丢失，合扫/分扫对应同一高层存储。

## 独立复核与主审验证

独立 agent `independent_fairness_audit` 用同一个 producer 预存 D/E，比较同文档被 own 隔开、不同文档被 own 隔开、无 own 的连续修正。实际记录预约调用顺序、字典键顺序、diff 对象、ordinal、固定地址和 scan extents。结果确认合并来源于 `(layer, owner, fingerprint, kind)` 聚合；不是 PIM 必须提供的自动合并。数据自动带入当前 C7。

主审使用五个实际 layout 类存同一份 master，分别生成整块、前子段、后子段的 V 命令。五档同输入结果一致，原子段步长现象仍在。读取 `git show c600051:pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py` 对照：原公式使用 dense L，没有 extents 接口。保留为新增接口的共同边界；没有以“共同代码”证明任何分段 workload 的误差会完全抵消，也没有自动扩大整改清单。

独立两轮复审还确认：下一轮 history_len 使用新 owner 的 master 占位，没有沿用旧 diff 对象；interleaved 是单 prefill 展开，turns 虽有 parent 关系仍抽象掉旧布局。该限制记为 C8，先问是否需要为跨轮 claim 补齐引用，未将整体净收益方向写成定论。

另一独立 agent `attacc_model_provenance` 复核 C5 公式，确认现有共同项处理可接受，补清 GPU TP 分片 heads 宽度、GQA 驻留 query 数、完整 QK/softmax/PV 命令流，以及当前每 request 首次层选择、后层沿用的范围。链路式直接使用现有 flash X2G 的启动时间和带宽。

这些是 planner/地址/命令构造检查，未运行 Ramulator、GPU 性能、完整 workload 或数值 attention。原始 JSON 和旧 session 不改写，新增证据另存。

## 文件变更与验证

| 文件 | 本次变更原因 |
|---|---|
| CURRENT_ISSUES.md | 顶部新增已裁决交接表；更新各 case 状态；C5 改为实际项估算式；C7 按最新裁决关闭同轮指控；C4 按共同近似记录，C8 写入仍需过目的跨轮覆盖证据 |
| audit README、archive README | 链接最新裁决和新证据，维持单一当前问题文档 |
| docs/README_audit_fixes.md、README_design_ladder.md | 更新裁决入口；把设计阶梯已有公式标作代码现状并指向 C5 的待执行新口径 |
| docs/sessions/README.md、本 session | 记录用户原话、范围变化、复审原因与交接原则 |
| archive 新 probe/JSON/manifest | 保存数据来源、主审/独立验证以及未改实现的摘要 |

执行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /tmp/reaudit_pending_layout_cases.py
PYTHONDONTWRITEBYTECODE=1 python3 /tmp/independent_c7_8c51672_recheck.py
python3 /tmp/record_audit_rulings.py
PYTHONDONTWRITEBYTECODE=1 python3 /tmp/independent_c7_round_boundary_probe.py
python3 /tmp/apply_round_boundary_ruling.py
python3 /tmp/finalize_remaining_audit_items.py
python3 /tmp/validate_audit_rulings.py
git diff --check
```

独立 probe 由 agent 执行。脚本及 JSON 见 [archive 索引](../../audit/2026-09-05/archive/README.md)。最终验证见 [rulings_pending_reaudit_manifest.json](../../audit/2026-09-05/archive/rulings_pending_reaudit_manifest.json)。实现、论文、贡献 README 和既有性能结果保持；没有 commit。本轮交接入口为 [CURRENT_ISSUES](../../audit/2026-09-05/CURRENT_ISSUES.md#decisions)，C7 定义已定，C4 按既有共同近似规则记录；目前需用户重点审阅 C8 的跨轮 claim 覆盖。

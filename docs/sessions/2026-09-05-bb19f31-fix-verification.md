# 2026-09-05：bb19f31 修复后的独立复查

chenyi9 要求“再检查一遍”，中途“继续”是恢复同一轮审计。本 session 记录复查证据和**为什么更新审计文档**；本次没有实施所发现问题的修复。

## 时点与方法

代码 `bb19f31134362548405a3bc225c43139b4a768af`，对照 `fbe6756` 和原始 AttAcc `c600051`。开始时仅已有未跟踪的 `experiments/paper_ladder/`，中途恢复后 HEAD 和状态相同。

与上轮相比，非文档改动只有 `src/workload.py`、`src/workload_runner.py`、`src/layout_probe.py`、`experiments/collect_dag_ladder.py`、`tests/test_workload.py`、`tests/test_collect_ladder.py`。主审阅读全部这些差分，并复跑上轮小反例；三个独立 agent 分别从计划执行、物理读集、计量/summary 检查。沿用先前授权的小输入验证，没有启动性能模拟或正式 sweep。

用户已定口径不变：A1/A2 独立 baseline；A3b 同轮可正常紧排，跨轮保留旧对象；A4c 布局、A4e 表、A5 prefill+MQ 配套、A6 简单选边；共同 flash/pipeline；DIE/TLB/普通 STORE 不加无依据费用；共同 AttAcc 近似不要求误差相等。新边界先展示条件、上游建模和分档影响，本次不替用户追加建模裁决。

## 执行 agent 已提交的修改、原因和验收

| commit | 为什么修改 | 本轮实际检查 |
|---|---|---|
| `00007de` | 继承必须指向真正写入者，前缀变化不能复用旧修正；CacheBlend 保留逐层采样位置 | 三轮 writer、换前缀、原 .25 CacheBlend 与三层绑定通过；额外查到取整校验和重复 parent 输出来源边界 |
| `3c5ba3a` | 旧 diff 应作为驻留行读，A2 不应继续重算；A2 decode 链路不应归入 prefill | 实际 GPU 回读/PIM scan 已补全，A2 工作量对齐；单路 PIM 后处理和 A2 history 仍有另一层 summary 问题 |
| `46dfffd` | 七档 collector 不应混用 PIM batch starts 与 request ends | 七档两 tier 相同 summary 的五个主指标一致；说明累计时刻和尾段语义，未把它冒称每请求平均指标 |
| `dcf5e28` | layout probe 应与实际事件同用 head 能量倍数 | 8 路径+4 控制的诊断/事件比为 1；原 E1 关闭 |
| `25ec3b5` | 防止 diff K 超出保留区域进入 master V | 源码条件在容量内允许、超界拒绝；未分配百万行 |
| `a0fc30f` | 补三轮、实际驻留读、A2 分类、CacheBlend 和前缀的定向测试 | 本轮执行 7 个相关测试通过，工具记录 24.356 s |

上表是对已有提交的审计，不是本次主审写入代码。此前实现 session §15 对这些改动的原因基本准确；通过已覆盖控制不等于所有输入均可执行、所有 summary 已统一。

## 新发现怎样分类，为什么不否定默认比较

- C6.1：batch=1 的 PIM decode 调通用后处理 helper，名字没有 decode 前缀，summary 漏掉首 token 的投影/FFN并误算入 prefill。A2 不走这个归类错误，属于有条件的指标公平性问题。batch>1 不触发，不能以 request 数量判断分支。
- C6.2：A2 history_len>0 的 query 位置与 summary 首位置不一致，控制中 first_token 为 0。当前 turns 的 history_len=0 不触发；不要将这个旧占位入口问题泛化到重列历史段的默认输入。
- C8.5：CacheBlend 旧修正继承与新抽样分别向上取整，但 validator 仍要求全体一次向上取整。主审独立复现 .3 下 6 对 5；六个软件复用档都在入口失败，没有测到档间速度差。依用户共同影响规则，只记支持边界，不据此否定默认 recompute 阶梯。
- C8.6：相同输出指纹可使继承覆盖显式 parent 的 producer。主审核对真实 loader/plan/TLB，独立 agent 捕获 A5 prefill 读错来源；decode 是共享绑定的源码推导，未冒称直接捕获。当前 21 个 turns 输入没有不同 parent 共享输出指纹，因此不声称它们已经受影响。
- collector 的 batch_first_attention_s 可实际取到更早的 Q arrival，属于诊断命名。主指标已经不依赖它，不当成 collector 同公式修复失败。

这些接口均是相对 AttAcc 新增，原始没有对应 request/tier summary、CacheBlend validator 或 parent 继承。共同 head 能量外推、C4 行列近似等已接受项不重开；没有发现新的私有档位系数或刻意削弱 A3b 的证据。

## 主审与独立 agent 的证据

| 审查方 | 检查 | 保存文件 |
|---|---|---|
| 主审 | 原 C8 probe 重放、六个变更文件、定向测试；独立复现新 validator/parent 来源；21 个输入条件 | `bb19f31_c8_replay_bb19f31_evidence.json`、`bb19f31_main_verification.json`、`bb19f31_tests.json` |
| independent_fairness_audit | 三轮/多层、A2 工作量、实际 prefill 读集、CacheBlend 取整与重复输出来源 | `independent_c8_bb19f31_evidence.json`、`independent_c8_bb19f31_boundaries_evidence.json` |
| ledger_trace_boundary_audit | 旧 diff + 新 diff 的真实物理读写，A3b/A4c 对齐，diff 上界 | `ledger_scan_bb19f31_evidence.json` |
| attacc_model_provenance | head 能量、七档 collector、真实 summary 在 batch/history 分支的含义 | `head_energy_bb19f31_evidence.json`、`tier_summary_bb19f31_evidence.json` 和 `bb19f31_raw/` |

所有文件在 [archive](../../audit/2026-09-05/archive/README.md)，对应脚本作为 .txt 一起保留。脚本中的 /tmp 路径是执行时点路径，归档只复制没有改写其计算；重放需按 manifest 的文件映射恢复依赖。设备价格/traffic 为受控桩，数值仅用于检查形状、计量公式或事件时刻归类，不能用于论文性能表。

## 实际修改哪些文档、为何修改

1. 原 CURRENT_ISSUES 原样存为 `CURRENT_ISSUES_before_bb19f31.txt`。当前页关闭上轮具体反例，保留 C1–C8/E1 锚点；将新的有条件边界写出触发配置与 AttAcc 来源，不沿用“全部未修”的旧状态。
2. 当前页的 summary 数表从新 JSON 由脚本提取，避免手填实验数。表前明确价格为固定桩；扫描章节区分物理 shadow 行与可见 KV，避免误把合理多读说成重复错误。
3. 更新 audit README、archive 索引和 session 索引。只保留一份当前问题入口，旧证据逐字节保留。
4. 本 session 记录已有代码为何修、哪些控制通过、新边界怎样定性，以及这次文档更新的理由。没有改实现、测试、workload、论文、既有结果，没有提交或推送。

## 验证记录

本轮定向测试 7 个通过（24.356 s），命令及完整 stdout/stderr 见 [测试 JSON](../../audit/2026-09-05/archive/bb19f31_tests.json)。collector 测试有未关闭临时文件的 ResourceWarning，测试通过；它不是性能或公平性证据，本次未因此修改测试。

开始/结束核对文件 hash 时检测到外部新增提交，详细说明见下节；已有结果和论文 tex 单独检查，当前文档链接也检查。结果见 [manifest](../../audit/2026-09-05/archive/bb19f31_audit_manifest.json)。当前结论仅限本次代码阅读、结构和受控报表验证，不认证所有历史缓存或给出净加速比。

## 收尾期间外部新增提交：补核到 ff5b91e

第一次最终 hash 检查发现 `src/devices.py`、`src/workload_runner.py`、`tests/test_workload.py` 与主体审计起点不同。随后 git 显示另一会话已提交 `9c40891`（共同链路固定延迟规则）和 `ff5b91e`（实现 session §16）。**这些源码修改/提交不是本次 audit agent 实施的。** 不将“审计期间文件完全未变”写成通过；manifest 同时保留主体起点与新增提交后的快照。

主审只读补查三个文件差分，并执行新增的链路定向测试，1 个通过。独立计量 agent 核对各档标记是否共用。规则按 decode/bitmap 操作名省略固定启动延迟，prefill 其余传输仍收；不是大小阈值，A2 decode 的历史 KV 回读也享用相同规则。能量公式、带宽项、PIM trace 生成器和周期计价公式未修改；链路到达时间变化可能间接改变批次和总能量，不保证全程序输入组合/能耗完全不变。

§16 将该规则记录为 chenyi9 的裁决；本次据此说明来由与共同适用范围，没有重新裁决规则、修改实现或确认另会话的性能数字。由于模型口径有变化，旧结果与新结果不可混成同一配置。

当前问题页增加 E2，版本头更新为“主体 bb19f31，补核至 ff5b91e”。C6/C8 的旧和新控制依赖事件/输入结构，固定桩不使用真实链路价格，所以其结论不因这笔提交消失。收尾新增测试和快照在 `ff5b91e_late_tests.json`、`ff5b91e_late_snapshot.json`；仍使用本轮唯一的 CURRENT_ISSUES 和 session，不另造第二份当前报告。

## 用户本轮再次确认链路口径

chenyi9 明确：“链路计价是我定的，decode 小流量传输固定开销可以忽略不计。”本次将它作为已确定规则记录，不再列为待用户审阅或待修问题。审计仅核对共同 helper 是否给各档一致打标，以及是否仍保留传输带宽成本；实现按 decode 名字分类的事实如实说明。

收尾另用同一真实解析设备模型检查 4 KiB 和 16 MiB 的阶段标签，20 个控制通过，证明现代码不按字节阈值切换；大包控制不解释为用户声称它是小流量。新增脚本和 JSON 以 `link_latency_ff5b91e` 保存。未运行性能模拟。

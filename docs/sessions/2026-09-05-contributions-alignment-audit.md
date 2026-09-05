# 2026-09-05：按 README_contributions 核对实现与当前 audit

本次只修改 audit 文档、索引、session 和新增审计证据。实现时点 `8c51672a3ef8b936340354b3211963cde8945c49`；用户指定 `docs/README_contributions.md` 作为四项贡献及具体例子的参照。没有修改该 README、论文、实现、测试、运行脚本、workload 或已有结果，没有运行 Ramulator 性能实验。

## 为什么补这一轮

前一轮把问题统一到 CURRENT_ISSUES，但还没有逐个回答贡献 README 的例子能否由代码构造。README 的复算脚本主要读常量、算理想行数和假设时间，不能证明实际分配和执行。此次将“说明机制的例子”“真实代码的结构验证”“模拟性能结果”明确分开，避免把例子的成立当作关闭 C3/C4/C5 的依据，也避免把特定例子的额外条件误判成新增违规。

## 如何检查、检查到了什么

1. 阅读贡献 README、论文 Evaluation 的 placement ladder 段落，以及对应的配置、分配、共读关系收集、PIM prefill、MQ generator/计价和 A6 选边函数。没有执行 README 的自我生成脚本，以保留用户原文。
2. 主审调用真实 `resolve_config` 比较 A3b 起相邻档。变化在当前允许的四项机制内；A5 配套 query 批量、频点一并列出。缓冲默认各档已经都是 512 B，未误报为新的独享容量。
3. 独立 agent `independent_fairness_audit` 只读复核前两个例子，用真实 reuse plan + `_prepare_cacheblend_tlb` + `PhysicalLedger` 构造布局。记录自写一块/四块，以及分别生产/同一 producer 共读五块的控制。结果支持存在满足示意表的输入，同时显示表格依赖哪些输入。名义占行没有被当成实际 ACT，八段静态展开没有被当成八轮生命周期验证。
4. 主审使用一层、四个 MHA KV heads、8192-token 驻留历史、8 个新增 token 的小输入，真实构图但用固定价格设备桩。核对 A4e/A5 回读和新 KV/Q/context 字节，确认 MQ query 数。`pipe=True`；设备桩不调用真实 GPU，因此没有“本次已测 FlashAttention”的结论。正式运行仍必须共同启用 flash。
5. 直接运行现有 trace generator，比较单通道、256-token 行、8-query 的 replicate/MQ 命令数，检查 MQ 降低 MAC 读列命令但没有删掉 softmax 和私有搬运。没有运行 Ramulator，也没有用命令数推导延迟或 energy。
6. 检查 A6 仍采用逐 request 两侧估价比较，符合用户定义；README 的时间只作为假设算术。原 C5 两个候选操作覆盖差异继续保留，不要求全局最优或正文旧双 DAG 算法。

独立 agent 随后复核新增对照节，确认例子条件、名义行数和静态展开范围写法准确；按其建议，把 workload 净偏差方向改为“存在偏乐观风险”，避免未测整体就断定一定高估。另核对 placement 扫描仍按 Ramulator 周期换算，MQ 修改模拟输入间隔，没有使用 README 示例比例做输出缩时。

具体数值由探针 JSON 自动带入 [CURRENT_ISSUES 的四贡献对照](../../audit/2026-09-05/CURRENT_ISSUES.md#contributions-check)，避免手抄。正式 workload 场景分布/加速未重新测量；示例偏向有收益的条件，不能外推其平均收益。

## 文档改动及理由

| 文件 | 改了什么、为什么 |
|---|---|
| CURRENT_ISSUES.md | 在 case 地图前增加四贡献对照、配置实际差分、可复现的例子条件和 workload 解读；C1–C7 保持为同一份当前事项，未另建重复问题报告 |
| audit 当日 README、archive README | 增加对照入口与新证据索引，维持当日根目录只有两个入口文档和 archive |
| docs/sessions/README.md、本 session | 记录本轮授权范围、为什么调整 audit、独立 agent 分工和验证边界 |
| archive 新 probe/JSON/manifest | 保存真实代码检查和独立复核证据；旧 JSON/日志和 cleanup_manifest 不覆盖，旧清理 manifest 仍代表上一次整理时点 |

对贡献 README 只提出两项措辞建议，未擅自改原文：例子①若要作为代码复现输入，明确自写 KV 大小；例子②明确没有其他共读关系改变选择。其“条件下推导/假设耗时”声明是正确的，不将示意图判成伪造测量。

## 验证与复现

在仓库根目录执行的轻量命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /tmp/contributions_alignment_probe.py
PYTHONDONTWRITEBYTECODE=1 python3 /tmp/independent_contributions_8c51672_probe.py
python3 /tmp/update_contributions_audit_docs.py
python3 /tmp/finalize_contributions_audit.py
python3 /tmp/validate_contributions_audit.py
git diff --check
```

独立 probe 由 agent 执行。主审 probe 内部只启动 trace generator，不启动 Ramulator。当前归档是 `.txt` 源码副本；需复现时先复制至上述 `/tmp` 路径并核对输出路径，文档生成脚本不属于实现或自动测试。探针 JSON 是阅读结论的直接证据，不必为阅读审计重新运行脚本。

最终验证记录在 [contributions_alignment_manifest.json](../../audit/2026-09-05/archive/contributions_alignment_manifest.json)：实现及现有结果摘要保持、贡献 README 保持、历史证据保持、论文保持、Markdown 链接/源码行号和 `git diff --check` 检查。没有 commit、没有性能结论或新增代码整改；后续候选项仍先由 chenyi9 裁决。

# 2026-09-05：沿用 AttAcc 共同限制，候选问题先交用户裁决

## 用户裁决及为什么调整审计

chenyi9 裁决：如果 AttAcc 本来就有该问题，且当前对方法和 baseline 共同适用，有既有依据即可沿用，不为绝对精度要求重建模型。每个候选问题先说明原始 AttAcc 是否建模、当前怎么改、对比较有什么影响，再由用户决定是否算问题。FlashAttention 是我们新增的共同 GPU 模型，必须启用。

上一轮把调度不够充分、模型非硬件实测、缓存指纹不完整等直接列成整改条件，超出了现在明确的相对公平标准。因此本轮调整的是判断规则和文档状态，不改实现来追求更理想的性能模型。共同近似不需要造成完全相等的数值误差才允许使用；但新增分支、数据量或布局差异需要如实先呈现。

## 起始代码与来源验证

当前 HEAD `8c51672`，原始 AttAcc 对照 `c600051`。主审直接读取原始 `system.py`、`devices.py`、`ramulator_wrapper.py`、`model.py`、`config.py` 与 bank trace generator；原文摘录和 SHA 由 Python 脚本归档。独立 agent `attacc_model_provenance` 另行核对原始 pipeline、计量、stack 复制、P movement 顺序、GQA 和缓存。

确认共同继承的例子包括 QK/PV ALU 记账、ACT 单价摊销、代表 stack 向上复制、shape 缓存未版本化、单 head 先搬全部 P 再做 context MAC；不再因这些近似本身要求修改。

同时更正：原始 `pipe=False` 仍会进入 `_pipeline` 的基础 attention↔通信重叠公式，不能称作设备全串行；当前 append-order DAG 不是原始同一实现。当前各档共用该 DAG 调度，但是否要追究空闲窗口由用户判断，不直接要求 ready queue。ALL-BANK mapper/action/preq 的原始与当前源码逐字节相同，V02 的候选差异来自新增 diff 区地址，而非原 mapper 被改坏。

## 实际文档变化

| 文件 | 修改与原因 |
|---|---|
| `audit/2026-09-05/ATTACC_RELATIVE_FAIRNESS_REVIEW.md` | 新建当前审阅入口；每项列 AttAcc 是否建模、当前共同/分档影响、证据边界与待裁决状态；区分已明确的 flash 要求 |
| `audit/2026-09-05/ATTACC_UPSTREAM_REVIEW_EXCERPTS.md` | 原始 git 对象的带原行号摘录，作为来源判断证据 |
| `audit/2026-09-05/attacc_relative_fairness_review_*` | 归档来源提取脚本、hash JSON 和验证清单；不运行原始代码 |
| `REAUDIT_8c51672.md`、`MODEL_PROVENANCE_REAUDIT_8c51672.md`、`INDEPENDENT_REAUDIT_8c51672.md`、`RUNTIME_CONFIGURATION_AUDIT.md` | 顶部说明旧优先级不再自动等于整改决定，链接新清单；原技术反例保留 |
| `docs/README.md`、`docs/README_audit_fixes.md`、`docs/README_run_guide.md` | 以新裁决为准；撤回必须升级调度/实测/metadata 的要求，保留 flash 必须启用与共同配置核对 |
| `docs/sessions/README.md` 与本文件 | 记录本轮用户决定、主审解释、来源验证和实际范围 |

上表省略目录的四份审计文件均位于 `audit/2026-09-05/`。旧 manifest 不覆盖改写，仍证明其原生成时点；本轮生成新的 manifest。

## 验证与决策边界

执行 `python3 /tmp/attacc_relative_fairness_review.py` 读取 git 对象、提取源码并保存 hash；执行 `python3 /tmp/validate_attacc_relative_fairness_review.py` 检查新文档链接、源码/历史结果未变及差异空白。未运行性能实验，沿用此前小探针的技术事实，不把其任意时长当实测收益。

代码、测试、实验脚本、workload 和已有结果未修改。最新 [逐项审阅清单](../../audit/2026-09-05/archive/ATTACC_RELATIVE_FAIRNESS_REVIEW.md) 中“建议保留”仍待 chenyi9 裁决；没有把主审建议写成用户已同意。FlashAttention 必须开启已经确定。核验结果见 [manifest](../../audit/2026-09-05/archive/attacc_relative_fairness_review_manifest.json)。

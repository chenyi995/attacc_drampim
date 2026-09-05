# 2026-09-05：整理 audit 目录，按论文 case 写清当前事项

## 用户要求与修改原因

chenyi9 要求整理 `audit/2026-09-05`，去掉不必要的重复入口，明确这一轮的问题在哪个文档，写到不了解项目的人也能看懂，尤其说明具体 case 与论文 claim 的关系。

之前同一日期目录混有多轮报告、后续纠正、模型专项、脚本和 JSON；读者可能把旧“必须修复”当成当前裁决。因此将当前说明合并成 `CURRENT_ISSUES.md`，其他报告和原始证据归档，不再并列多个“最新报告”。接受共同 AttAcc 限制和用户逐项裁决的口径未变。

## 改动内容

- audit 顶层保留 `README.md` 和 `CURRENT_ISSUES.md`；历史报告、JSON、日志和 probe 移到 `archive/`，删除可重新生成的 Python 字节码缓存。
- 当前文档先解释 KV/master/diff、prefill/decode 和七档；每个 case 按“论文/用户定义→具体输入→预期和实际→AttAcc 是否已有→相对影响与待裁决内容”组织。GQA 失败、diff 地址、V 子段读取、A6 选边、A2 汇总和 A3b packing 均有具体例子；flash 启用单列为已明确要求。
- 已修和共同限制从当前问题段移走，只留简短状态说明；没有用旧 P0/P1 标签替代用户裁决。
- 数表由 `organize_audit_20260905.txt` 读取已有 JSON 生成 `current_case_facts.json`，没有新跑性能实验；设备桩/命令生成证据与硬件性能明确区分。
- 修复移动后的 Markdown 链接；历史正文里的当时命令、manifest 路径及 probe 计算内容保留。路径变化及前后 hash 记录在 cleanup_manifest。
- `docs/README.md` 统一指向当前文档；`docs/README_audit_fixes.md` 改为简短入口，旧长篇先保存到 archive；运行/设计指南更新审计入口。其余 docs/session 只更新受影响的链接。

## 独立可读性检查与验证

独立 agent `independent_fairness_audit` 只读核对论文与旧报告，建议按具体 claim 和 case 组织、解释首次出现的术语、避免把桩时间展示成硬件收益；其编辑建议已用于新稿。新稿复核后，已按建议明确 GQA 表使用两种模型配置、错误字节来自校验前捕获、C4 涉及共同接口，并保留静态预分配待裁决。措辞修订记录在 `polish_audit_current.txt`，未改变证据数字。

运行 `python3 /tmp/organize_audit_20260905.py` 执行可追溯移动、链接修复和证据提取；`python3 /tmp/validate_audit_cleanup.py` 验证原 JSON/日志/probe 字节不变、实现/测试/workload/已有运行结果未改、文档链接和表格数据一致。这里只整理文档，不新增代码修复或实验。

当前实现仍是 `8c51672`，整理不改变之前的技术事实或用户裁决。最终入口：[CURRENT_ISSUES.md](../../audit/2026-09-05/CURRENT_ISSUES.md)；详细验证：[cleanup_manifest.json](../../audit/2026-09-05/archive/cleanup_manifest.json)。

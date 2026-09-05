# Session：8c51672 大量修复后的只读复核

日期：2026-09-05；对象：`8c51672a3ef8b936340354b3211963cde8945c49`，比较前审 `cdd89db`。chenyi9 本次要求：“刚才进行了大量修改 再检查一下，是不是都正确了？”

**结论：多项旧问题已修，但不能确认全部正确。** 新报告分别关闭已经修好的反例，记录 GQA 校验回归、ledger→MAC 地址转换、A6 操作估价和汇总缺口。完整当前判断见 [主报告](../../audit/2026-09-05/archive/REAUDIT_8c51672.md)。

## 1. 依据与范围

延续 chenyi9 的 audit-only、小 workload 验证、独立 agent、更新 audit/README/session 要求。本轮先读 `agent.md`，用 `whoami` 确认账号名称。读取实施 session §12–13，采纳其记录的新裁决：随机修正共用、A100、A1 精确 L、普通 DRAM 成本清零、decode 当前 token 在 GPU。A6 简单逐 request 选择仍接受，不要求候选 DAG。

修改开始前，tracked 工作树干净；已有 untracked `experiments/paper_ladder/` 保留。主审将全部 tracked 文件的初始 SHA 写入 `/tmp/reaudit_8c51672_start.json`，用于区分本轮文档变更与此前实现修改。

## 2. 为什么再次更新文档

仓库新增了 PhysicalLedger、STORE 零成本、GQA/QKV、owner 依赖、MQ、A6 全 lane/尾批和汇总修复。旧报告不能继续将 c16 row 漂移、旧读写差率、缺失 owner 依赖等标为当前未修。

另一方面，源码注释和新增单测主要验证高层几何，不能替代命令生成器/mapper 的一致性。GQA 改了生产侧字节量但校验仍沿用旧公式，也说明“某处已改正确”不等于完整路径正确。因此文档需要同时记录已关闭项、新回归和未贯通部分，避免一概肯定或一概否定。

## 3. 主审与独立检查

| 审查者 | 具体检查 | 证据 |
|---|---|---|
| 主审 | 七档 MHA/GQA 小 DAG、共同修正 hash、owner 依赖、Q 流量、A1 L、A6 估价与执行操作、flash 空回读公式 | `reaudit_8c51672_probe.txt/.json` |
| `independent_fairness_audit` | 真实 allocator 的持久 row、burst、store/scan 映射、预约顺序和 per-agent diff 边界 | `INDEPENDENT_REAUDIT_8c51672.md`、独立存储探针 |
| `attacc_model_provenance` | GQA validator/批量 KV bytes、QKV/MAC energy、缓存指纹、STORE、TBT/tier CSV | `MODEL_PROVENANCE_REAUDIT_8c51672.md`、计量探针与数据表 |
| `ledger_trace_boundary_audit` | PhysicalLedger 到受版本控制 Attention generator、HBM mapper/ACTAB/MACAB 的转译 | `ledger_trace_boundary_8c51672_probe.txt/.json` |

主审与计量 agent 各自复现 GQA 入口失败；主审核对命令 agent 给出的 mapper/action/preq 代码。命令探针只生成命令并解码，没有测量 binary 时间或 KV 数值结果。

主报告写完后，命令 agent 再次只读核对 V02–V04 与 JSON/源码，确认关键事实、数字及证据边界一致。主审采纳其文字建议：区分 mapper 解码与 ACTAB/preq 遍历，明确 V 子段控制组的已存长度，以及 head 混叠的每 HBM 配置边界。

## 4. 实际验证

定向现有测试：

```bash
PYTHONDONTWRITEBYTECODE=1 KVPIM_CPPCORE=0 PYTHONPATH=.:tests python3 -m unittest \
  test_placement.PhysicalLedgerTest \
  test_workload.AgenticHistoryTests.test_every_layout_executes_the_same_correction_plan \
  test_workload.AgenticHistoryTests.test_consumer_waits_for_the_owner_store_even_when_listed_first \
  test_workload.AgenticHistoryTests.test_shifted_reuse_gets_query_variants_without_a_delta_field
```

结果：**11 项通过**；日志保留。没有把前次全套测试记录作为本轮结果。

主审探针：

```bash
PYTHONDONTWRITEBYTECODE=1 KVPIM_CPPCORE=0 KVPIM_PREFILL_SIDE_LOG= python3 /tmp/reaudit_8c51672_probe.py
```

设备桩调用真实布局/DAG/校验与选边函数。第一次输出后，为确认 fresh 空回读的真实模型行为，补充只调用 A100a `xPU` 通信解析公式再执行探针；代码仍只在 `/tmp`。最终 JSON、归档文本一致。定向测试在 Python event core 模式执行；本轮 native 的 STORE 映射通过源码核查，不声称重跑了 native 全套。

所有数字复制与计算由 `/tmp/write_reaudit_8c51672_docs.py` 从主审及 agent JSON/日志生成，脚本也归档为 `.txt`。GQA 成功/失败表、A6 操作价格表、地址解码/子段偏移及测试数量均可回查源数据。probe µs 是结构证据；flash 空回读耗时是当前解析公式的返回值，两者都不是硬件测量。

## 5. 文件修改及原因

| 文件 | 本轮内容与原因 |
|---|---|
| `audit/2026-09-05/REAUDIT_8c51672.md` | 新当前报告：已修对照、V01–V08、逐档结论、证据范围 |
| `INDEPENDENT_REAUDIT_8c51672.md` / `MODEL_PROVENANCE_REAUDIT_8c51672.md` | 两名 agent 分别撰写，保留独立证据 |
| 本轮 probe `.txt`、JSON、日志、计量 evidence.md | 按实际执行输出归档，避免只靠叙述和手抄数字 |
| `reaudit_8c51672_file_coverage.csv` / `reaudit_8c51672_manifest.json` | 保存本轮改动覆盖、版本、源码初始/最终指纹和文档核验结果 |
| `docs/README.md` / `docs/README_audit_fixes.md` / design/run guide | 增加当前版本入口和状态，避免旧报告的已修项继续被当作当前问题 |
| 前次 `REAUDIT_cdd89db.md` / `STORAGE_SCAN_CONSISTENCY.md` | 仅加历史状态提示，旧反例正文与证据原样保留 |
| `docs/sessions/README.md` / 本文件 | 索引本轮记录，解释为何修改及如何验证 |

前轮 manifest 保存当时文档指纹；当前更新入口后的文件指纹以本轮 manifest 为准。源码、测试、workload、已有实验结果和 HEAD 保持本轮开始时的状态。

## 6. 当前裁定

已确认的修复按主报告表格关闭；GQA 正常执行、ALL-BANK 存储隔离、固定 V 子段转译、多 head 地址身份、真实写入账本与估价覆盖仍有明确问题或证据缺口。普通 DRAM 成本清零是 chenyi9 允许的计量选择，不重新要求加回；A6 的简单规则也是接受的，不升级审稿标准。

本轮完成的是对这批修复的审查和记录，没有实施代码修复。最终文件核验和文档链接结果见 [manifest](../../audit/2026-09-05/archive/reaudit_8c51672_manifest.json)。

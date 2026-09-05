# 2026-09-05：pipeline 与 FlashAttention 运行口径补审

## 起点与授权

源码 `8c51672a3ef8b936340354b3211963cde8945c49`。接续 `8c51672` 修复复核；用户要求检查实际运行是否合理：pipeline 应开启、无依赖工作可并行、GPU 应采用 FlashAttention，未开启必须说明。延续此前“只 audit，不改代码”的限制以及独立 agent 复核要求。

## 为什么修改文档

原有运行指南要求 flash，但快捷命令省略了该参数；CLI 默认 legacy，ladder 未设置 GPU_MODEL 也会落到 legacy。只看 --pipeopt 或 overlap checker 会遗漏资源空闲却阻塞 ready event 的调度问题。旧报告中的管线记录与新默认值也必须区分；FlashAttention 的解析计价不能写成硬件实测。基于实际读取的源码和小探针，新增运行配置作为共同验收条件，不改变既定消融定义或 A6 的简单逐 request 选边。

## 文件变化及原因

| 文件 | 改动与理由 |
|---|---|
| `audit/2026-09-05/RUNTIME_CONFIGURATION_AUDIT.md` | 新增入口对照、调度反例、历史产物配置状态、FlashAttention 数据来源及修改建议 |
| `audit/2026-09-05/REAUDIT_8c51672.md` | 增加补审入口，避免读者将旧报告已修项等同运行配置已合格 |
| `docs/README.md` | 快捷命令明确 flash/pipeline，并链接补审；仅改文档命令，不改变运行代码 |
| `docs/README_run_guide.md` | 增加开/关/无法核实的验收要求，说明 pipeline checker 局限及 flash 解析模型性质；移除从单点外推所有形状的表述 |
| `docs/README_audit_fixes.md` | 增加共同配置、结果 provenance、合理调度与 FA 数据来源的整改标准 |
| `docs/sessions/README.md` 与本文件 | 记录本轮依据、实际变化与验证边界 |
| `audit/2026-09-05/runtime_configuration_8c51672_*` | 归档主探针、JSON、日志、生成与验证脚本和 hash 清单，数表由脚本从证据生成 |
| `audit/2026-09-05/independent_8c51672_pipeline_*` | 归档独立 agent 的反向设备调度反例，区分独立证据与主审复制 |

## 实际验证

`PYTHONDONTWRITEBYTECODE=1 python3 /tmp/audit_runtime_config_8c51672.py`：AST 读取默认参数，调用真实 Python 全量/增量及已加载 native 调度函数；任意时长的极小 DAG 能复现 ready event 错过空闲资源，checker 仍通过。独立 agent 运行 `/tmp/independent_8c51672_pipeline_probe.py`，将等待资源换成 PIM channel，Python/native 复现同一类问题。均未调用性能设备接口，没有硬件性能测量。

只读盘点仓库内两处结果目录：363 个文件、27 份 JSON；5 份明确 pipe=false，22 份 pipe 未记录，27 份 GPU 模型未记录。盘点包括 gitignored 结果，不覆盖外部 scratch。探针记录路径和 hash，未改结果。

`python3 /tmp/write_runtime_configuration_audit.py` 生成本轮文档、归档脚本和证据；`python3 /tmp/validate_runtime_configuration_audit.py` 验证文档链接、证据关系、源码/结果 hash、改动范围及 git diff 空白。详见 [验证清单](../../audit/2026-09-05/archive/runtime_configuration_8c51672_manifest.json)。历史主审 manifest 保留为历史快照，本次有自己的清单。

## 结论与剩余工作

当前不能确认全部运行符合新要求。文档快捷命令已补齐；实现中的 legacy 默认、DAG 配置元数据缺失和调度空转仍未修改。旧结果保留，但按关闭/无法核实标明；flash 参数仍属论文图近似校准。此前 GQA、存储→扫描等 V01–V08 也不会因本次补审而自动关闭。后续代码修复和正式运行不在本次实际改动中。

完整结论见 [运行配置补审](../../audit/2026-09-05/archive/RUNTIME_CONFIGURATION_AUDIT.md)。

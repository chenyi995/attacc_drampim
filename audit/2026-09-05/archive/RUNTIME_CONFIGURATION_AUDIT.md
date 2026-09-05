> 历史记录：保留当时的技术证据和评价，当前待审事项统一看 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md)。

# 运行配置与并行合理性补审：8c51672

> **后续裁决更新：** 按 chenyi9 新要求，共同近似不因绝对精度不足成为问题。FlashAttention 必须开启；RC02 的调度空洞暂按共同近似待审，不要求直接换调度器；RC03 缺元数据/旧 pipe=false 不再自动排除结果；RC04 非本库实测只作来源说明。各项 AttAcc 原有建模与当前相对影响见 [逐项审阅清单](ATTACC_RELATIVE_FAIRNESS_REVIEW.md)，其状态优先于下文此前评价。原始 AttAcc 的 pipe=False 仍含公式 overlap，与当前 DAG SERIAL 不是同一实现。

日期：2026-09-05。源码 `8c51672a3ef8b936340354b3211963cde8945c49`。本次补充 [主审报告](REAUDIT_8c51672.md)，只修改文档和审计证据；未修改实现、测试、实验脚本、workload 或已有结果，未启动性能实验。用户新增要求：pipeline 必须开，GPU attention 必须采用 FlashAttention；没有开启必须说明。独立 agent `independent_fairness_audit` 另行复核入口与 Python/native 调度。

**不能确认当前所有运行符合该要求。** pipeline 的 CLI 和 ladder 入口已开启，但仍有事件追加顺序导致的空闲资源阻塞；GPU 的默认模型仍是 legacy，直接 ladder 缺少环境变量时不会启用 flash。部分历史结果明确关闭 pipeline；其余缺少配置记录的结果不能推定已开启。当前 `flash` 是解析模型，不能称为本仓库 FlashAttention 硬件实测。

## 1. 共同验收条件

所有档的 GPU 部分共用同一 FlashAttention 计价来源、硬件配置和相关参数；所有档启用 pipeline。A1/A2 的独立 baseline 身份不豁免这两个运行条件。无数据依赖且资源允许的任务应能并行；同一 GPU、channel 或链路仍需遵守资源冲突，不能一律假定同时执行。

分别核对入口默认、实际解析参数、设备实际采用的模型、最终事件时间轴和结果元数据。报告逐项标注 **已开启 / 已关闭 / 无法核实**；开关存在或 checker 通过，不等于实际并行合理。设备桩只支持结构验证，不支持 FlashAttention 性能结论。

## 2. RC01 · P1：GPU 默认未启用 flash

| 调用方式 | pipeline | GPU 模型 | 当前结论 |
|---|---|---|---|
| `main.py`，省略相关参数 | 默认开启 | 默认 `legacy` | 不满足 flash 要求 |
| `run_dag_ladder.sh`，未设或设空 `GPU_MODEL` | 显式 `--pipeopt` | 未传 `--gpu-model`，继承 `legacy` | 不满足 flash 要求 |
| `GPU_MODEL=flash bash experiments/run_dag_ladder.sh …` | 显式开启 | 显式 `flash` | 入口配置满足，实际产物仍需核实 |
| `run_sweep.sh`，未覆盖环境变量 | 经 ladder 显式开启 | export 默认 `flash` | 入口配置满足；环境可覆盖为其他模型 |
| 直接调用 `run_reuse_prefill(...)` | Python API 默认关闭 | 由传入的 GPU 对象决定 | 不能套用 CLI 默认值 |

来源：[CLI GPU 默认](../../../main.py:140)、[pipeline 默认](../../../main.py:180)、[实际传入 DAG](../../../main.py:581)、[ladder 参数](../../../experiments/run_dag_ladder.sh:66)、[可选 GPU 参数](../../../experiments/run_dag_ladder.sh:73)、[sweep export](../../../experiments/run_sweep.sh:22)、[Python API](../../../src/workload_runner.py:5215)。本轮已在 `docs/README.md` 的两个快捷命令补齐 flash，并显式写出 pipeline；**没有修改脚本或默认行为**。只设置环境变量 `GPU_MODEL=flash` 后直接运行 `main.py` 不够，CLI 仍须传 `--gpu-model flash`。

GPU 模型在各档间不一致属于额外消融差异；全部档一致使用 legacy 也不满足用户指定的 GPU baseline。其收益偏差不能一概定向：GPU attention 被低估性能时可能夸大 PIM offload 收益，但 flash 同时改变 GEMM/链路等计价，不能从某个形状推导“所有请求 PIM 都赢”或“A6 必然等于 A5”。

## 3. RC02 · P1：开 pipeline 后仍存在就绪任务错过空闲窗口

[Python 全量调度](../../../src/workload_runner.py:2493)、[增量调度](../../../src/workload_runner.py:2558) 和 [C++ 调度](../../../src/cppcore/eventcore.cpp:103) 都按事件追加顺序维护每个资源的单一 availability。一个排在前面的任务即使在等别的设备，也会预约本资源将来的结束时间；之后已就绪的独立任务无法填入前面的空闲时段。

主审直接调用真实调度函数，手工指定以下任意单位的事件时长（**不是 GPU/Ramulator 性能数据**）：

| 事件 | 资源 | 数据依赖 | 实际开始 | 实际结束 |
|---|---|---|---:|---:|
| cb-0 | PIM:pool0-1 | 无 | 0 | 10 |
| cb-1 | GPU | cb-0 | 10 | 11 |
| cb-2 | GPU | 无 | 11 | 12 |
| cb-3 | LINK | 无 | 0 | 1 |

`cb-2` 无依赖，GPU 开头空闲，本可先执行。只对同一个 DAG 使用合法的 ready-first 次序，结束时间由 12 变为 11，没有让同一 GPU 上的任务重叠。独立 LINK 能在开头执行，说明当前实现确实有跨资源 overlap；问题是不能保证有活可做时不空转。Python 全量、增量及当前加载的 native 库均复现。

独立 agent 反过来用 GPU producer → 等待它的 PIM event → 同 channel 无依赖 PIM event，也在 Python/native 复现该空洞；其已就绪 PIM event 开始于 11。见 [独立脚本](independent_8c51672_pipeline_probe.txt) 与 [证据](independent_8c51672_pipeline_evidence.json)。

两组反例的 `overlap_validation.passed` 仍为 true，因为 [checker](../../../src/workload_runner.py:2695) 重放了同一种追加顺序规则。它证明所选调度约定内部一致，不能证明合理使用空闲资源。建议检查 ready queue 或合法空闲窗口插入，并使增量/native 与最终调度一致；增加“有已就绪任务且资源可用时，不因列表顺序无理由空转”的独立验收。**此建议不改变 A6 逐 request 简单比价的定义，不要求全局最优。**

此外，[owner 依赖](../../../src/workload_runner.py:4698) 被并入整个 `request_ready`，以及 tier 的统一 barrier，需按实际数据依赖逐条检查；“所有消费者都必须等 owner”应落实到消费该 KV 的操作，不能自然推广到每个可先做的算子。本轮没有用这些潜在的过宽依赖推算七档收益，也未证明所有 workload 都会受到同样影响。

## 4. RC03 · P1：已有结果的运行配置不能统一验收

本轮只读盘点 `output/` 和 `experiments/paper_ladder/results/`，包含 git 忽略文件，共 363 份 log/JSON，其中 JSON 27 份。没有扫描外部 scratch 目录，不能把本盘点扩展为所有实验结果。路径、大小、SHA256、配置标记及 JSON 字段全部在 [证据](runtime_configuration_8c51672_evidence.json)。

其中 5 份 JSON 明确记录 `overlap_validation.pipe=false`，均位于历史目录 `output/20260826-174745_workload_star_repair_r3w3k8_LLAMA-7B/`：

| 产物 | 报告档位 | pipeline | GPU 模型 |
|---|---|---|---|
| [dag_A1.json](../../../output/20260826-174745_workload_star_repair_r3w3k8_LLAMA-7B/dag_A1.json) | A1（历史标签） | 关闭 | 无法核实 |
| [dag_A3.json](../../../output/20260826-174745_workload_star_repair_r3w3k8_LLAMA-7B/dag_A3.json) | A3（历史标签） | 关闭 | 无法核实 |
| [dag_A4.json](../../../output/20260826-174745_workload_star_repair_r3w3k8_LLAMA-7B/dag_A4.json) | A4（历史标签） | 关闭 | 无法核实 |
| [dag_A5.json](../../../output/20260826-174745_workload_star_repair_r3w3k8_LLAMA-7B/dag_A5.json) | A5（历史标签） | 关闭 | 无法核实 |
| [dag_A6.json](../../../output/20260826-174745_workload_star_repair_r3w3k8_LLAMA-7B/dag_A6.json) | A6（历史标签） | 关闭 | 无法核实 |

余下 22 份 JSON 无可用 pipe 字段；这不证明关闭，应标记“无法核实”。本范围内明确记录 pipe=true 的 JSON 为 0 份。全部 27 份 JSON 未记录 gpu_model，日志标记搜索也未找到 GPU 模型配置，因此不能认证使用了 FlashAttention。以上 A3/A4 是旧档位，不能重贴当前 A3b/A4c/A4e 标签，更不能按今天的默认值倒推历史命令。

当前 PIM DAG 在 [report](../../../src/workload_runner.py:5179) 保存 `overlap_validation.pipe`，但 [A2 report](../../../src/workload_runner.py:4547) 没有该字段；两类 DAG 均未保存 `gpu_model`，[main](../../../main.py:605) 也没有补齐。另一个 analytic engine 有 gpu_model 字段，不能用它来证明 DAG 产物配置。

建议统一记录实际 `GPU.gpu_model`、解析后的 pipe、模型/硬件/精度、FlashAttention 参数来源、split-k、argv、代码/输入 hash 和实际调度 backend；所有档相同字段。汇总检查混用配置、混 revision、缺档和来源不明的旧结果。尤其 [SKIP_A1 复制](../../../experiments/run_dag_ladder.sh:89) 仅复制已有 A1 文件，当前没有检查其 GPU 模型与 pipeline 是否匹配。历史关闭或无法核实的结果可保留为历史材料，但不作为新口径下已通过验收的正式 ladder。

## 5. RC04 · 证据限制：flash 模型不等于 FlashAttention 实测

[GPU 分支](../../../src/devices.py:57) 确实按配置选择 FlashAttention 模型；[流量](../../../src/devices.py:106) 使用融合 attention 假设，softmax 没有独立 off-chip traffic，[时间](../../../src/devices.py:129) 用效率曲线及 occupancy 等公式计价。不存在“只改了标签、完全没切模型”的证据。

但 [gemm_table.py](../../../src/gemm_table.py:15) 明确注明 FlashAttention 效率是论文图近似读数，不是本库测量。因而这里能确认的是 **GPU 按 FlashAttention 解析模型计价**，不能认证“这些 GPU 数值是实跑 FlashAttention kernel 的结果”。若论文使用“实测”，需给出对应硬件、版本、形状、精度/掩码/GQA 条件和原始 benchmark；若接受解析模型，必须明示来源与近似边界。当前 `--attn-splitk` 默认关闭且单独可选，也应披露；不能以缺少小 Q 的并行优化为隐藏 baseline 代价，更不能仅凭本轮静态审计认定所有形状都应开启该选项。

## 6. 已做和未做

本轮已完成入口/模型分支/报告字段核查、历史产物只读盘点、Python/native 极小调度验证和独立 agent 复核；更新 audit、运行说明、整改建议和 session。上一轮设备桩测试不变成 FlashAttention 性能证据，本轮也没有运行 FlashAttention 或 Ramulator 性能矩阵。

实施建议：先统一且记录运行配置，再解决调度中的无依据空转并核对依赖范围；与主报告 V01–V08 的实现阻断一起处理后，才有条件验证正式数据。不能仅补上命令行开关，就宣布七档已公平。不开 pipeline 可能低估布局带来的并行收益；GPU baseline 过慢可能高估 offload 收益；两者叠加及不同 DAG 的影响须用合规配置验证，本轮不编造收益修正比例。

证据：[主探针](runtime_configuration_8c51672_probe.txt)、[输出](runtime_configuration_8c51672_probe.log)、[JSON](runtime_configuration_8c51672_evidence.json)、[独立探针](independent_8c51672_pipeline_probe.txt)、[独立 JSON](independent_8c51672_pipeline_evidence.json)、[验证清单](runtime_configuration_8c51672_manifest.json)、[session](../../../docs/sessions/2026-09-05-runtime-configuration-audit.md)。历史 `reaudit_8c51672_manifest.json` 的文档 hash 对应上一轮完成时点；本次补审的 hash 由新清单保存。

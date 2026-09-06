# Decode scan 到 TBT：GPU 速度和 pipeline 审查

> 图版本说明：下文 Fig. 3b 数值指历史 `03_multiagent_reuse` 曲线。正式图已改为固定八列扫描，见 [修订记录](../../docs/sessions/2026-09-05-fig3b-fixed-columns-revision.md)。本文 C1v2 的 scan/TBT 数据与 pipeline 结论不受换图影响。

本轮审核源码 `167fe08e608b89e32f57402953314ced0b1194c6`，对照原始 AttAcc `c600051`。只分析已有结果和事件，没有改模拟器、论文或跑新的性能仿真。独立审核见 [独立报告](archive/decode_scan_tbt/independent_pipeline_audit.md)。

**结论：GPU 前后处理实际变快，通常会让 scan 收益在 TBT 中更显眼；单纯提高 GPU 峰值 FLOPS 不保证这一点。当前 Flash/pipeline 的实跑开关已开，但不能确认 decode 流水已经充分：确实找到已就绪 GPU 工作错过空窗的事件。** 该限制由 A3b–A6 共用，尚不能判断修正后哪档多受益，不据此认定 A3b 被刻意削弱。

## 1. 单 scan 到底节省多少

必须区分两个 workload、两种 scan 范围。

**Fig. 3b 是受控 K 扫描。** 八轮 A3b 为 1402 cycles、A4e 为 342 cycles，延迟减少 **75.61%**；十六轮减少 **75.02%**。它不含完整 attention 或 GPU 前后处理，没有对应 TBT；不能用这组约 75% 与另一个 workload 的 TBT 拼出“保留率”。原始数据在论文 `fig/plots/motiv/experiments/03_multiagent_reuse/result.csv`。

**已有 C1v2、CACHEBLEND-TINY 的完整工作流。** 固定同一 workload、同一修正计划，Flash、pipeline、NVLink 开启，batch 8、4 HBM、每 head 八通道。各档使用相同 run_config。以下是相邻档，以及布局合计：

| 比较 | 私有 scan 服务时间减少 | 共读 scan 服务时间减少 | 每步扫描跨度减少 | 加权 TBT 减少 |
|---|---:|---:|---:|---:|
| A3b->A4c | 3.055% | 0.000% | 4.068% | 0.386% |
| A4c->A4e | 27.027% | 0.000% | 9.984% | 1.372% |
| A3b->A4e | 29.256% | 0.000% | 13.646% | 1.752% |

布局合计 A3b→A4e 的绝对值：私有 scan **1.245095→0.880826 µs**；每请求、每层、每步所有 scan 的首尾跨度 **3.541422→3.058165 µs**；加权 TBT **174.880317→171.815860 µs**。

私有/共读 service 都取一次扫描实际最慢 channel。每步跨度含同 channel 排队和共有/私有 scan 的时间关系，不能称为单次 Ramulator service。TBT 则含一个 token 的所有层、GPU 运算和调度等待。

## 2. 有多少“保留”到 TBT

若按**两个相对降幅的比值**作粗略读数：

- TBT 降幅 / 私有 scan 降幅 = **5.99%**。
- TBT 降幅 / 每步扫描跨度降幅 = **12.84%**。

这里约 6% / 13% 是“百分比改善的比值”，**不是说只有这些比例的节省微秒传到了 TBT**。即使每一微秒扫描省时都完整传递，TBT 的分母更大，其百分比仍会更小。当前 summary 的 scan 均值包含首 token，而 TBT 统计排除首 token 间隔；shared scan 还服务多个成员。这些汇总不能恢复严格的逐 token 绝对省时传递率。

这组布局主要改善私有 scan；共读 scan 时间没有下降，因而整体扫描改善先从约 29% 稀释到约 14%。GPU 前后处理、必要依赖和排队再使 TBT 百分比更小。不能仅凭这两个百分比的差距判断存在 bug。

**证据范围：这是 TINY 流程诊断，不是正式 LLM 的性能结论。** 当前 `out_C1v2_LLAMA3-8B_hbm4_k8` 只留有 A2 完成结果，没有可配对的 A3b/A4e scan 和 TBT。C1v2 原始报告路径、SHA256、各档运行配置、数值提取脚本均在 [metrics.json](archive/decode_scan_tbt/metrics.json)、[extract_metrics.py](archive/decode_scan_tbt/extract_metrics.py)。运行记录的 revision 为 `4b74849` 且 dirty；不能将今天源码 revision 当成该次运行的来源。

## 3. GPU 变快，在什么条件下有帮助

把有因果依赖、不能隐藏的扫描时间记为 S，GPU 的前后处理记为 G，其余共同时间记为 C。一个解释性的串行近似是：

`TBT ≈ G + S + C`，所以 `TBT 的相对降幅 ≈ scan 的相对降幅 × S/(G+S+C)`。

这是条件推导，未用于拟合已有数据。若实际 G 下降而 S、C 不变，相同 scan 省时在 TBT 中所占比例会上升。若是两条可并行分支，则对应结构为 `C + max(G_branch, S_branch)`；当 GPU 分支始终较慢时，scan 省时可能完全被遮住，GPU 加快后才可能显现。

**“实际 GPU 工作更快”和“提高配置 FLOPS”不同。** QKV/FFN 还受带宽与形状效率限制；ACT/NORM 采用原 AttAcc 的固定截距：activation 8.29 µs、norm 6.87 µs。Flash 分支不取消这些共同项，提高 FLOPS 也不取消。依据：当前 `src/devices.py:250–307`；原始 `c600051:src/devices.py:164–176`。按 chenyi9 的既定口径保留这些上游共同近似，不列为本轮待修问题。

历史 C1v1 A6 的完整事件诊断中，Norm + GELU 约占 decode GPU 服务时间的 **64.85%**，显示这些共同项在 TINY 上很重。这是另一份旧 workload 的分项解释，不能与 C1v2 的 TBT 配对计算。数据见 [timeline_analysis.json](archive/decode_scan_tbt/timeline_analysis.json)。

流水更好也不保证布局的相对收益更大：总体可以更快，同时把一部分扫描等待隐藏在其他 GPU 工作后面。合理目标是各档采用同样有效的流水与依赖，而不是保证改完后倍率增加。

## 4. pipeline 开了吗，哪里确有额外串行

**已开且有作用。** 五档 C1v2 的 `run_config.gpu_model=flash`、`pipeopt=true`；逐 channel 资源独立，scan 合并等最慢 channel，GPU 局部 attention 和 PIM 旧 KV 扫描仅在结果合并处汇合。DIE/TLB/STORE 没有新加计时。当前 decode 小流量固定启动费已关闭，不用“每次 NVLink 有 6 µs”解释本次差距。

但 `pipeopt=true` 和 `overlap_validation.passed=true` 只证明遵守当前资源排程，不证明空闲资源被充分利用。

| 项目 | 当前行为与影响 | AttAcc 本身是否建模 | 审查判断 |
|---|---|---|---|
| P1：GPU 空窗不回填 | 先预约一个等待 PIM 的 GPU post，再追加的已就绪 GPU 工作无法填入其前方空窗 | 原版无这套请求 DAG 预约器，不能以原版公式证明该空窗必要 | 有实际事件反例，值得过目；净 TBT 损失未量化 |
| P2：跨组/跨层构造顺序 | 所有 QKV/发送先排；每组 local→scan→FFN 排完再建下一组；快组下一层 QKV 也在后面 | 原版不表达不同请求 ready time；它按 head 折减 QKV/proj/link 的流水成本 | 现实现有额外耦合；只有一个组时跨组部分不触发 |
| 整操作与逐 head 流水 | 当前整个 QKV 完成才送 Q，完整 context 回来才 projection；没有逐 head 切分这些 GPU 操作 | **原 AttAcc 有 head 数相关的 overlap 公式**；当前只在旧 System.simulate 路径调用 | 不能把新 DAG 称为已完整继承原版的 head pipeline；共同粒度是否保留由用户决定 |
| tier 同步 | 下一轮统一等前一 tier 的全部请求结束 | 原版无 multi-agent tier/parent 调度模型 | 若 tier 是业务屏障则合理，否则偏保守；不能把它直接当成本轮稳定 TBT 主因 |

这些行为由 A3b–A6 共用，没有看到专门削弱 A3b 的分支。由于各档扫描长短不同，共同的粗排程仍可能改变相对收益；影响方向需匹配完整事件和同等调度对照才能确定。

### 实际空窗证据（已有 C1v1 A6 时间线）

第一组 GPU 在 **53076.019226–53080.337633 µs** 空闲，窗口长 **4.318407 µs**。
第二组 local-score 事件 `cb-894` 的全部依赖已在 **53075.844558 µs** 完成，自身只需 **0.038815 µs**，可以放入这个窗口。
实际却在 **53111.881655 µs** 才开始，因它被追加在第一组等待后执行的 FFN 之后。

这证明“有 ready 工作时 GPU 仍可能空闲”，**不代表每 token 可以节省这个事件被推迟的全部时间**。小事件的启动提前量不是 TBT 降幅。脚本直接分析保存的依赖和时间戳，没有重新定价或重排执行后冒充实跑。

当前源码对应 `src/workload_runner.py:2580–2588` 的尾部预约、`3626–3683` 的 Stage A、`3740–3953` 的 Stage B；native `src/cppcore/eventcore.cpp:144–152` 相同。更细证据、上游行号和适用条件见 [独立审查](archive/decode_scan_tbt/independent_pipeline_audit.md)。

## 5. 已排除的解释

- 没有用平均 channel 时间充当 scan latency。
- 已有 C1v2 不是 Flash/pipeline 漏开；CLI 默认 GPU 模型仍是 legacy，所以其他运行仍要看各自 manifest。
- 当前常规 decode 的 KV store fence 被更晚的 context-return LINK 和 post 完成覆盖，不能当成已证实的额外等待。
- ACT/NORM 固定成本、共同 GPU/通信近似按用户接受的 AttAcc 口径解释，不因其压低收益而单独修改某档。

## 6. 证据与本轮产物

[metrics.json](archive/decode_scan_tbt/metrics.json) / [提取脚本](archive/decode_scan_tbt/extract_metrics.py)：相同输入和修正计划的五档结果、Fig. 3b 单 scan 减幅。

[timeline_analysis.json](archive/decode_scan_tbt/timeline_analysis.json) / [分析脚本](archive/decode_scan_tbt/analyze_timeline.py)：旧完整事件的原时间戳、依赖、可填空窗和 GPU 分项；[提取器](archive/decode_scan_tbt/extract_events.py) 从原始 4.3 GB 报告去掉地址数组，临时中间文件不作为新的性能结果。

[独立审查](archive/decode_scan_tbt/independent_pipeline_audit.md)：独立 agent 只读检查；[session](../../docs/sessions/2026-09-05-decode-scan-tbt-pipeline-audit.md)：本轮范围、判断与为何更新文档。

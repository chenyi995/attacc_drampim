# 指标口径与每 KV head 八通道的布局实验

chenyi9 确定：报告 decode scan latency 的减少、TBT、TTFT 和 E2E；布局分析采用每 KV head 8 个 channel 的场景，考察一次共读集中在其中少数通道时的空间。本轮据此补充报表定义与配置依据。原 B1/小跑的 2-channel/head 结果保留为原配置的结果，不改标签。

## 1. 四项指标分开报告

| 指标 | 定义与粒度 | 作用 |
|---|---|---|
| Decode PIM scan latency | 一次逻辑 scan 的并行 lane 服务时间取最大值；必要的顺序阶段按依赖累加。按相同 request、decode step、layer 匹配，分别标明 shared/private 与 batch 成员，再汇总均值及分布 | 直接衡量布局是否缩短扫描 |
| TBT | 主表按生成间隔加权：`Σ(end−first_token)/Σ(lout−1)`；lout≤1 没有 TBT 样本 | 衡量扫描加速是否改善逐 token 完成速度 |
| TTFT | 每请求 `first_token_completed−request_release`；包含排队、prefill 和首 token 全部计算。不能减实际 GPU 开始时刻来隐藏等待 | 衡量首 token 响应时间 |
| E2E | 整份 workload 的最终完成时间减共同起点，即当前 makespan；如另报单请求完成延迟，另列 `request_end−request_release` | 衡量整个多请求/多轮任务是否变快 |

每项同时给绝对值和相对上一档的降幅 `1−new/old`；另给 A3b→A4e 的布局合计。A1/A2 可按原基线定位比较，A3b 往上按已确定的相邻 claim 比较。A2 没有 PIM scan，这一列为 N/A，不能填 0 造成虚假比较。TBT 可补请求均值，TTFT 可补请求分布；必须标明样本粒度，不能把同一共享 batch 的成员当成独立重复实验。

当前事件名称是 `decode_*pim_kv_scan_score_softmax_pv`，该指标包含银行侧 QK、softmax、PV；不是只测 ACT。单次通道并行阶段的服务时间是 `max_c(t_c)`，**不是 `Σ_c(t_c)`，也不是把总 lane 时间除以通道数。** shared/private 若是不同阶段，应保留其依赖和资源关系，不能无条件把两个 max 相加或取一次 max；MQ 共享事件也不能除以 query 数后冒充一个请求的延迟。

实际调度中的 `最后完成−最早开始` 可能包含通道排队，可作为另列的 scan elapsed；主表的 scan service latency 用真实 Ramulator 服务时间与扫描内部依赖统计，明确排除 GPU/Q 传输等待。对 A3b/A4c/A4e，使用相同读集与批量口径比较；对 A5/A6，保留其已声明的 MQ/选边差异，不能混用不同粒度的数据。

## 2. 现有结果可补 TTFT，但缺真实 scan latency

独立 agent 从上一轮归档的 small/out JSON 提取如下。**这仍是旧小跑，不是新八通道实验，也不是 B1 多轮 DAG。**

| 档位 | Decode scan latency | TBT，µs | TTFT 均值，ms | E2E，ms |
|---|---|---:|---:|---:|
| A3b | 缺事件，无法恢复 | 235.622266 | 36.908240 | 96.991918 |
| A4c | 缺事件，无法恢复 | 234.625769 | 36.906426 | 96.735997 |
| A4e | 缺事件，无法恢复 | 233.659977 | 36.906997 | 96.490291 |

该输入所有请求在共同模型起点 0 提交，故 TTFT 由 `first_token_s` 直接得到；包含 owner/资源等待及首 token 后续 GPU 算子。各档 5 个请求的 TTFT 相同，所以经验 p95 等于均值，不意味着测到了真实系统的稳定尾延迟。A3b→A4e：TTFT 少 0.003367%，TBT 少 0.833%，E2E 少 0.517%。

已有 `2.498%` 是全部 lane 服务时间之和的减少，不能填入 scan latency 降幅。JSON 的 `events=null`，batch/sweep 仅保留开始和 admission，没有结束或时长。当前 `experiments/run_dag_ladder.sh` 固定传入 `--workload-report-events none`；执行 agent 若要产出这四列，需要保留 `full` 事件或实现等价、可审计的扫描汇总。`main.py` 已支持 `--workload-report-events full`；这是一项产物要求，本轮没有修改运行入口。

多轮时还需保留逐 request 的 release 时间。现有 `summary.requests.first_token_s` 是全局完成时刻，collector 的 tier `ttft_s` 也是全局时刻；不能直接当作后续每轮的 TTFT。统一 release 定义为请求提交/业务依赖满足而进入调度器的时刻，额外调度等待计入 TTFT；不能用“实际开始执行”代替。若继续按全局起点统计，则列名应是首 token 累计完成时间，不能与逐请求 TTFT 混用。这是报表定义说明，不自动改动已有调度规则。

## 3. 八通道可以用现有配置表达

依据 `src/config.py` 与 `workload_runner.py` 的 `_heads_per_hbm`，每个 HBM 有 16 个 channel，按其中的 KV heads 划分。GQA 要用 KV head 数，不用 query head 数。

| 仓库模型 | Query heads | KV heads | GPU 数 | HBM 总数 | 每 HBM KV heads | 每 KV head channels |
|---|---:|---:|---:|---:|---:|---:|
| LLAMA3-8B | 32 | 8 | 1 | 1 | 8 | 2 |
| LLAMA3-8B | 32 | 8 | 1 | 2 | 4 | 4 |
| LLAMA3-8B | 32 | 8 | 1 | 4 | 2 | 8 |

因此可采用仓库中的 `LLAMA3-8B、NGPU=1、NUM_HBM=4`，对应每 head 8 个 channel；TINY 同样有 8 个 KV head，4 个 HBM 下也是这个几何。运行时 `GPU_MODEL=flash`、pipeline 开启，并在产物中记录实际解析后的值。

同一八通道实验里，所有档使用相同模型、HBM/GPU 数、容量、带宽、频率、输入、修正计划与共同调度口径。A3b/A4c 都拥有相同的八个 master 通道，A4e 只改变其声明的软件放置表。不能只给 Fugue 增加通道，也不能把八通道 Fugue 除以旧两通道 A3b 来报布局收益。

HBM 从 1 改成 4 同时改变物理资源总量，因此跨 HBM 配置的对比属于硬件敏感性分析；布局收益应在每个配置内部做相邻档比较。本文记录的是 chenyi9 确定的八通道场景，不宣称默认脚本或旧结果已经采用它。

## 4. 一次访问集中在少数通道，潜在收益确实更大

条件推导：一次需要读 8 个等成本 master 块，每个块扫描耗时 t，没有其他固定开销或资源竞争；可用 8 个通道。真实 `_block_slot_table` 在孤立共读集合上能得到以下放置：

| 朴素布局中八个块的分布 | 朴素扫描 | 分散后理想扫描 | 局部加速上限 | 局部延迟减少上限 |
|---|---:|---:|---:|---:|
| 全在 1 个通道 | 8t | t | 8× | 87.5% |
| 均分在 2 个通道 | 4t | t | 4× | 75.0% |
| 均分在 4 个通道 | 2t | t | 2× | 50.0% |
| 已经均分在 8 个通道 | 1t | t | 1× | 0.0% |

例如共读写入序号为 `0,1,8,9,16,17,24,25` 的八个块，朴素轮转自然落在 ch0/ch1，各四块；合适的软件表可以铺到八个通道，局部由 4t 变成 t。这里改变的是请求选择哪些已写入的块，A3b 仍按原有写入序轮转，没有人为改坏其 allocator。这可以作为展示通道冲突的受控 case，配合同输入中的均衡访问说明收益边界。

这个倍数主要说明 A4e 的并行空间，不说明 A4c 一定受益更多：A4c 将 diff 集中到末通道，master 已经很均匀时仍可能由该通道决定完成时间。真实增益仍需按实际地址交给 Ramulator，diff、GPU、链路和排队也会稀释 E2E。

**当前软件表还有一个需要读者知道的条件。** 我用同一放置 helper 加入“整个 corpus 曾被一次共读”的关系后，上述集中案例的软件表退回了与朴素相同的分布。原因是当前表按“是否曾共读”统计已放置伙伴，整库集合把大量块都连在一起，热点子集的信息被淹没。因此八通道只提供空间，具体 workload 的共读表是否能把热点分开必须检查；不能把理想分布当作当前实现的输出。

这来自 Fugue 新增的软件表，AttAcc 没有同类表；本轮作为收益成立条件记录，不自行裁定修改算法，也不把它解释成削弱 A3b。若 corpus 由多个独立请求分块产生、消费请求稳定共读少量热点，上述孤立共读例子更容易成立；实际 workload 应保留其真实生成与读取关系。

## 5. 证据与交接

主审核对模型/硬件几何，调用当前软件表 helper 做有界等成本推导；`attacc_model_provenance` 独立核查 TTFT、TBT、E2E 和 scan 指标缺失。没有修改模拟器、collector、workload 或论文，没有运行新的性能任务。

- [配置与通道放置证据](archive/metrics_eight_channels/metrics_eight_channels_geometry.json)、[脚本](archive/metrics_eight_channels/metrics_eight_channels_geometry.txt)。
- [已有指标与定义](archive/metrics_eight_channels/metrics_eight_channels_existing_run.json)、[提取脚本](archive/metrics_eight_channels/metrics_eight_channels_existing_run.txt)。
- [本轮 session](../../docs/sessions/2026-09-05-metrics-eight-channels.md)、[上轮布局上限](LAYOUT_BENEFIT_CEILING.md)。

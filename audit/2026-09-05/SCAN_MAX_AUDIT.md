# 最慢 channel 审计

chenyi9 提醒 scan 应取耗时最长 channel。当前 A3b/A4c/A4e/A5/A6 符合这一点，没有发现以平均通道时间替代 scan latency 的错误。

| 路径 | 证据 |
|---|---|
| Ramulator | ramulator_wrapper.py:801–811 使用 per_run=True，第719–725行逐lane返回 cycles×tCK；旁边 aggregate 的 sum 不用于该分lane路径 |
| 扫描事件 | workload_runner.py:2440–2477 每lane保留独立时长；单路decode3484、batch decode3901、PIM prefill5066收尾依赖全部lane |
| 完成时刻 | scheduler2571–2575取资源与全部依赖的最大完成时刻；makespan5275取max(end) |
| A6 | 4121取所有lane实际返回时长max；4093的最多token通道只是形状字段，不决定价格 |
| 诊断 | 5281的pim_pool_time_s_unoverlapped是lane服务时间总和，不替代scan latency |

原始 AttAcc c600051 对单次 trace 返回 memory_system_cycles×tCK，没有后来新增的 per-channel DAG 分解，也没有通道平均的依据。当前新增路径正确等待最慢lane，不需因为平均利用率诊断而改共同模型。

独立固定价格探针故意设256-token通道2µs、8-token通道7µs、GPU5µs：五档共同helper都7µs收尾，lane-sum9µs；A6选GPU5µs。排除了平均4.5µs或最多token通道2µs。这是逻辑探针，不是性能结果。

当前placement helper的warm路径通过_WARM_BYPASS使用原计价入口后才dump，未写零占位。独立控制也验证了这一点。但dump没有event ID/query positions/sweep ID，不同MQ sweep可以写出完全相同记录，且默认layer0/前400条、append输出。dump适合核对地址和max(per-channel服务时间)，完整归因用最终full events。

max(duration)与带排队elapsed分别命名。旧small/out events=null，不能从lane-sum恢复真实scan latency；保存方法见 [实验指导](../../docs/experiments/README.md)。

证据：[执行/选边探针](archive/docs_roles_k4/scan_max_20260905_probe.txt)、[结果](archive/docs_roles_k4/scan_max_20260905_evidence.json)、[warm/dump探针](archive/docs_roles_k4/scan_max_20260905_dump_probe.txt)、[结果](archive/docs_roles_k4/scan_max_20260905_dump_evidence.json)。独立审阅者为 independent_fairness_audit，未运行Ramulator或修改实现。

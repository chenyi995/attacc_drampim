# Targeted workload 独立静态复核

审计对象是 HEAD `958dd24` 和 `checks` 记录的未提交 reader-load 版本。只读检查 `build.py`、`handcheck.py`、输入、已有静态 JSON 和实现；未运行新的 Ramulator、DAG 或性能实验，本轮只更新本报告。实现快照及 SHA 在 [checks/all.json](checks/all.json)，两个版本不能合并标注为 HEAD。

**结论：D/E 输入符合用户授权的受控布局实验，反例保留得有意义；它们证明布局和扫描地址条件，尚不证明加速或 TBT 收益。P 的后续轮确实是短 prefill，但首轮不是；初稿的小链路定价和 batch 容量口径均已修正，当前预算仍是解析条件，不是性能预测。**

## D：跨轮累积 diff，而非单轮人为拆碎

`build.py:21–28` 每轮保留旧 segments，追加上轮输出，再追加新内容；parent 形成同一汇总者的链。旧 fingerprint 和绝对位置保留，符合 planner 的继承条件。`build.py:33–46` 让文档已有 corpus owner，而汇总者的新引用发生位置变化，故产生新 diff；背景 worker 的内容是新 master。

A3b 对同一轮连续修正允许正常合并，下一轮使用新的 request owner。这里每轮新增一份文档修正，不是把同一轮故意拆成多行来惩罚 baseline。各档同一个输入和同一个 plan；`handcheck.py` 的逐档断言检查逻辑/新增/继承修正、计算及扫描 token 数一致。

周期写流使当前 layer-0、一个八通道 head stripe 内的 A3b diff 集中在 ch7。它是刻意选定的受控相位；不能声称代表一般 agent 流量。背景多一段与多会话输入都保留，能显示这个相位对输入的敏感性。

以下来自 `checks/D_rolling_r*.json` 最后一轮汇总者的 `diff` 字段；均是所示 head stripe 的 **K 侧**：物理行是 K payload 占用，QK 地址行是按当前 generator MAC 地址公式访问到的行。它们不是 K+V 两侧合计：

| 轮数 | A3b 物理 diff 行 | A4c 物理 diff 行 | A4c QK 地址行 |
|---|---|---|---|
| 16 | 16 | 1 | 1 |
| 32 | 32 | 1 | 2 |
| 64 | 64 | 2 | 3 |

这两种行数不能混用。`handcheck.py:42–49` 正确分开了 payload 范围与 `ceil(extent_tokens/16) × 2` 列命令地址；短 extent 的末端取整会碰到下一行。这里没有 ACT 计数，没有 V 阶段完整命令时序，更没有 PIM 时长。A4c 仍按 object 分 extent，不等于整条紧凑行只发一次 MAC。

`sessions7` 的各汇总者 master token 分布较均匀，而 diff 地址仍受其他会话插入影响；不得把单会话的紧凑行数直接乘用到它。`sessions8` 出现 master 热点，构成遮蔽 diff 省时的可能条件，但“已经遮蔽时延收益”需要真实扫描结果，静态 JSON 尚不能证明。两组的最忙 channel QK MAC 数没有因 A3b→A4c 而下降；这是行局部性实验，不是减少 MAC 数的实验。

## E：周期共读与软件表

每个 owner 实际输入一组文档；所有 summary 选择固定步长的文档。它不是每档给不同输入，也没有额外给 A4e 移动 diff。限制是：固定步长选择本身就是受控冲突形状，不能当随机检索平均收益。

以下仅统计 `s00_t001` 的 `document_master`，是最忙通道触及的文档 QK 地址行数，排除了 own、输出、sys 和 diff：

| 输入 | HEAD A4c | HEAD A4e partners | 未提交 A4e reader-load |
|---|---|---|---|
| E_coread_stride1 | 8 | 8 | 8 |
| E_coread_stride4 | 32 | 8 | 8 |
| E_coread_stride8 | 64 | 8 | 8 |
| E_coread_stride8_monolithic_owner | 64 | 64 | 8 |

这支持“HEAD 表在分组 owner 的周期共读 case 中能消除文档通道冲突”，不支持同倍率的 ACT、扫描时长或 TBT 加速。`build.py:62–63` 的 owner 分组是该结论的必要条件之一；monolithic owner 反例表明，HEAD partners 会被整库共读关系影响，不能把未提交 reader-load 的结果归到 HEAD。stride1 的无明显冲突对照也保留了。

这些是固定输入上的算法对照，不应把 source owner 分组包装成硬件参数。若实际业务就是一次 ingest 整库，monolithic 结果才对应那种业务；这组受控 case 的用途是解释机制边界。

## P：m、容量与预算

`P_reuse_q4` 中，被检查的 `s00` 首轮 `m=532`，其中新增 diff 为 512；后续轮 `m=4`，末轮继承 diff 为 512，新增 diff 为 0。所以准确名称应解释为“后续轮 q4”，不能写每一轮都只有四个 query。


默认 GEMV buffer 容量来自 `src/ramulator_wrapper.py:40–69`，请求 query 数还需除以 GQA 组大小。LLAMA3-8B 的硬件容量为两个 prefill query，后续 m4 因而需要两个 sweep；这个结论还要求运行 batch 上限不小于二。真正实现使用 `min(batch_size, query_capacity/GQA)`，不是只看硬件容量。TINY 的 GQA 不同，不能套用同一个 sweep 数。

**初稿预算问题已解决。** `handcheck.py:118、124` 对回读和 context 返回都使用传入的真实 `_link_layer`；`summarize_checks.py:20–29` 明确传入 HEAD helper 并重新计价。小 context 返回不再额外计固定启动 latency。`handcheck.py:126–127` 显式固定运行 batch 上限并与硬件容量取最小值，修正了初稿只按硬件容量数 sweep 的缺口。GPU readback 加 FlashAttention、扣除 context 后除 sweep 数的代数，在已声明配置下可作为服务成本的盈亏平衡条件；它仍不能替代真实 PIM 价格、排队 TTFT 或 E2E。
最终 P 输入已改为 64 份实际需要的文档、32 轮，见 `build.py:92–93` 和匹配输入 hash 的 `checks/P_reuse_q4.json`。对 `s00_t001`，LLAMA3-8B 的解析值为 m=4、驻留 token=16412、batch 上限=8、sweeps=2；GPU 侧价格 548.811 µs/层，context 返回 0.109 µs/层，PIM 平均每 sweep 的盈亏平衡门槛 274.351 µs。上述数值直接读取修正后的 JSON，与 `HANDCALC.md` 一致。它们不是测得的 PIM 时长。


`gpu_budgets()` 给的是 GPU 线性层/Norm/ACT 未重叠服务量。其“达到某百分比改善需要多少 exposed scan”的字段只有在这些 GPU 成本与 scan 可加的简化式下成立。当前 pipeline 有重叠，不应把它当已核实的 TBT 门槛或实际关键路径。

## 证据范围

- `handcheck.py:56–57` 固定一层、两个 KV heads/HBM，并仅展示一个八通道 stripe。它验证 layer-0 的持久 ledger；没有证明完整多层模型每一层的预约相位完全相同。
- 读集是所选轮次 prefill 完成后的完整上下文，未加入该轮后续每一个 decode token；没有运行 batch admission、common/private 分组或 scheduler。
- E/P 多个 summary 存在共同 master，但是否同批取其交集、是否混入 owner/fresh 请求，需要实际 batch 记录。D 的背景 worker 通常不共享汇总文档；不能因为配置 BATCH 较大就假设 diff 能在 MQ 中共享。
- 确认旧 diff 被保存并继续纳入静态读集；没有用新的 history placeholder 替换它，也没有把每轮历史重算成新 diff。

另已只读核对 `summarize_checks.py:60–78` 与 `checks/formula_checks.json`：各 case、各 HEAD rung 的最后一个 target，共 285 份逐 channel QK 地址列表与真实 generator 输出一致；每次比较前会清空命令列表。这增加了公式与生成器一致性的证据，但仍只覆盖所选 layer-0/K 侧读集，不是所有层、PV 或 Ramulator 时序验证。本轮未重新执行该检查。

本报告表格与修正后的预算仅从已有 `checks/all.json` 读取；原始计算代码为同目录 `handcheck.py` 和 `summarize_checks.py`。新输入没有实测性能值。

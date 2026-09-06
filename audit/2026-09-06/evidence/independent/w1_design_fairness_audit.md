# W1 与新 diff 布局：独立只读审查

2026-09-06。实现快照 `c78dc76be28bc9ef307cb1aacf683759320b2c89`，论文快照 `8fe2022db0fc9d90909f52661841dc414f5ae2d3`。本轮读到的 `src/workload_runner.py` 已无未提交修改；旋转、按 agent 分组及平局回归轮转已在 `753044b` 提交。因此本报告审的是当前已提交实现，不能继续把它称为 dirty reader-load 版本。

范围：比较实现 `958dd24..c78dc76`，核对论文设计/评估/方法章节和真实 W1 输入；只执行一次 layer0、每 head 8 通道的 plan/PhysicalLedger 静态探针，没有设备价格、DAG 计时、Ramulator、性能或能耗实验，也没有修改仓库实现。探针为 `/tmp/w1_independent_static_c78dc76.py`，结果为同名 `.json`。

本轮裁决：新的 A4c 轮转和 A4e 按 agent 分组已经进入设计声明，不能判作偷加优化；没有发现这次修改削弱 A3b。仍有必须澄清的论文版本矛盾和 workload 收益解释，以及一个能用 W1 复现的“最少读行”口径差别。本报告不从这些静态结果推断实测加速方向。

## 1. 已通过：新机制有声明；A3b 本轮没有被改弱

- 论文 `sections/07-evaluation.tex:119–124` 已明确：A4c 是打包并轮转 diff 行；A4e 是 master 放置表加按 agent 分组 diff；A5 才把 prefill 放到 PIM。因此 A4e 的分组不再是未声明的优化。
- 设计 `sections/04-design.tex:79–102` 声明 diff 不独占通道、按行轮转，跨轮紧排；`:133–139` 明确每个 agent 独占自己的 diff 行，并按负载放置、平局遵循轮转。对应代码 `src/workload_runner.py:971–1002,1056–1087`。
- A3b 的 master append 分配仍为 `:864–867`，burst 分支仍为 `:1004–1055`；同次 prefill 同 owner 的连续预约继续可以紧排，跨请求的 owner 改变会结束 burst。这些分支在本次比较中没有变动。修改集中在 A4c/A4e diff 布局和 A4e table 的平局策略；`src/workload.py`、`src/ablation.py`、`src/system.py`、`src/devices.py` 本次比较没有变化。
- A4e 新平局规则 `:889–892` 沿用朴素 slot，属于明示的 placement 规则。本轮没有证据表明它修改了 A3b 本身。

**AttAcc 来源：**原始 AttAcc `c600051` 没有这套多 agent master/diff 账本或 A3b–A6 阶梯。这些属于 Fugue 的新增机制，应由论文声明支持；本轮已经有对应声明。原来双方共用的 DRAM/attention 简化不在本轮重新升级为必修问题。

## 2. 待修正文一致性：方法章节还是旧阶梯和旧实验

这是确定的版本矛盾，不能用“代码符合新设计章节”代替全文一致。

- `sections/06-methodology.tex:81–84` 仍把 A4c 写成 per-agent diff、集中在 head 的一个通道；当前 A4c 是各 agent 共用打包流、完整 diff 行轮转。按 agent 独占行从 A4e 才开始。
- `:85–87` 仍写 A4e 只改 master table，遗漏已声明并实现的 diff 分组/放置。
- `:97–118` 仍是 14 个 all-to-all 等旧 workload 配置；`:122–133` 仍给旧 HBM 配置和 588 次实验；`sections/07-evaluation.tex:102–104` 也仍引用那 14 个配置。当前仓库协议 `docs/README_run_protocol.md:39–48` 已改为 W1 baseline 七档、12 个单轴变体只跑 A3b/A6。
- `sections/04-design.tex:84–87` 无条件写第 j 个 diff 行去 j mod s 通道，但 `:133–139` 又给 A4e 最少负载覆盖规则。应把前者明确限定为 A4c/default rotation，后者是 A4e table 的覆盖与平局规则。
- 设计源文件 `:65–67` 自己保留了示意图仍画旧 dedicated diff channel 的 TODO；这里仅核实源文件承认图未同步，没有把未看过的 PDF 视觉内容当独立证据。

**建议：**以当前实际实现和 W1 运行协议统一方法/设计/评估措辞，保留每份实验结果对应的版本。此项是新增 Fugue 论文/协议一致性问题，不是 AttAcc 上游模型问题。

## 3. 已复现、需限定措辞：A4e 分数不是当前扫描的最少物理读行数

代码的确定行为：

1. `_prepare_cacheblend_tlb` 为整个 workload 建 `chunk_coread`，每个 request 都包含它的段指纹及自身 history/output 指纹，见 `src/workload_runner.py:3505–3511`。
2. `:979–992` 对同一 parent-chain root 的每个 request 累加 master block。一个历史 master 被 24 个 request 重复读取，就计 24 次；这不是去重后的当前读行数。
3. `:1072–1073` 对新 diff 行只加一次 1，没有按它之后被多少轮读取再加权。
4. `:994–1002` 用这个分数选新 diff 行通道。所以该分数既不是当前 request 的行数，也不是 master/diff 使用同一重复读取次数权重的总扫描量，更不是 Ramulator 通道耗时。

**真实 W1 静态反例**（k8、一个 layer、每 head 8 通道；当前行数是 `_pool_reads` 后 master 行加已经分配且本请求引用的旧 diff 行，不计尚未分配的新 diff 行）：

| 新 diff 行所属请求 | 实现用于放置的分数 ch0…ch7 | 当前读集唯一物理行数 ch0…ch7 | 实际落点 | 当前最少行通道 |
|---|---|---|---|---|
| `g00_m_t018` | 153,134,144,143,135,136,144,142 | 10,9,9,9,10,8,9,9 | ch1 | ch5 |
| `g01_m_t018` | 144,143,134,136,144,142,153,135 | 9,9,10,8,9,9,10,9 | ch2 | ch3 |

本探针重建分数并断言它与真实 PhysicalLedger 每次分配相符；8 次新 diff 行分配中出现以上 2 次差别。因此 README 的“放到 main 所读行最少的通道”（`workload/probe/README.md:25`）若让读者理解成当前扫描物理读行数，不能成立。

**状态与范围：**这是可交付给用户的具体启发式定义问题；还不能判定它刻意偏袒 A4e，也没有证明这两个落点会导致更高或更低的实际延迟。已接受的 known-DAG 信息不重新判作不公平。可以保留当前启发式并明确“各轮 master 读取次数累计 + 已分配 diff 行数”的分数；若论文坚持当前扫描最少行，则应另行对齐实现。不要称它为精确负载最优或实际最快通道。

**AttAcc 来源：**这是新 A4e 放置表自身的评分定义，原始 AttAcc 没有该规则。

## 4. 已证伪默认 W1 的一条收益解释：两两共读不等于整批 MQ 共读

`workload/probe/README.md:15,26` 用“两会话同号 worker 读相同文档历史”解释跨会话 MQ。两两共读事实成立，但默认 batch8 的实际共享条件更强：

- W1 的常规 tier 是 2 个 main + 4 个 worker，共 6 个请求；tier0 还多 corpus owner，第一 token 有 7 个请求。相同输出长度下，它们始终装进一个 batch8 组。
- `src/workload_runner.py:3880–3882` 形成 batch；`:4034–4046` 对 **组内所有请求** 的物理 master 读地址取交集；`:4052` 交集非空才建共享扫描。
- main 读 worker 输出，w0/w1 分别读不同的新文档，彼此的 prefix 也不同。真实 W1 的 tier0/1/2/23、输出位置0，以及 tier0输出位置1 的整组段指纹交集均为空。新增 decode output 也是每请求独有，不能补出整组共同 master。

因此默认 W1、batch8 的这些常规 batch 不能仅凭同号 worker 两两共读，就声称已兑现跨请求 decode MQ。对 MHA 模型没有 GQA 组内共享可替代该理由；对 GQA 模型，private 读集仍可能存在同 KV head 的多 query 共享，那是另一来源。本结论不否定 A5 prefill 单请求多 query 的 MQ，也不否定变体、不同 batch 成员/长度条件下出现公共读集。

**建议：**将 W1 的 A5 解释先写为 PIM prefill 与其 query 数带来的 MQ；跨请求 decode MQ 必须展示真实 `common` 地址数和 batch 成员，不能由两两相同文档直接推出。不得据此虚构加速，亦不要求为漂亮结果新增 scheduler 机制。

**AttAcc 来源：**原版没有这种带指纹共读的 agent batch 求交集；这是本仓库多 agent 共享执行路径和 workload 解释之间的条件问题。

## 5. 条件性支持边界：agent 身份以 parent 链根代替

`src/workload_runner.py:3515–3524` 把 parent 链根当 agent 身份，`:975–976` 按这个 root 分组。当前 W1 `gen_main_workers.py:86–93,101–120` 每条 parent 链严格对应同一个 main 或 worker，因而按 agent 分组的声明在 W1 上成立。

通用 `Request` 模式并没有独立 agent_id（`src/workload.py:74–85`），parent 校验也不排除多个不同任务继承同一个 parent（`:315–340`）。如果其他输入把 parent 当“生产者依赖”而非“同 agent 的上一轮”，不同 agent 会共用一个 diff 组。此项仅是其他输入的身份约定边界，不是已发现 W1 串组，也不阻断当前 W1。

## 6. 手算和范围应如何解释

- 958dd24 的旧 handcheck 假设 A4c diff 在最后一个 channel，当前已经改成整行轮转；那些旧槽位、间隙和上界数字不能直接引用为当前结果。
- `A4c 每个 head 的全局 diff 行数变少` 不推出 `某个 main 的最忙通道 scan 时间变少`。A4c 行中会混入其他 agent 的 diff；当前请求只读自己引用的 token，因而仍需保留地址间隙。A4e 才按 agent 分行。代码 `src/workload_runner.py:975–976,1062–1087,1099–1143` 明确保留该差别。
- 同轮 A3b 多 fingerprint 修正仍可在一个 burst 中紧排（`:1035–1044`）；A4c/A4e 则保留每个 fingerprint/row piece 的对象身份（`:1083–1086`），extent 不会自动跨对象合并（`:1140–1143`）。因此不能只用 ceil(全部 diff tokens/256) 替代真实 generator 的列命令计数。主审正在专查该边界；本独立报告不把占行数当 ACT，也不把列命令数当实际时间。
- 本轮没有发现为 A4c 增益而降低 A3b 工作量或改变它的代码。当前材料不足以保证每个 W1 变体都将显著受益，也不足以把集中构造的布局杠杆描述为一般应用的平均收益。

## 复核方法

从仓库目录执行 `python3 /tmp/w1_independent_static_c78dc76.py`。它只载入已有 W1 JSON，构建共同 k8 复用 plan、一个 layer 的 TLB 和真实 A4e PhysicalLedger；从源码同规则重建评分、核实每次 diff 行实际 slot，再计算当前请求的 nominal row 集合；最后直接检查几个默认 batch 的输入指纹交集。输出 JSON 明确保存静态范围。未启动 generator 时序模型、Ramulator、设备或完整 DAG；没有新性能测量。

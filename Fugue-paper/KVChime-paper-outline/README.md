# KVChime: Enabling Shared-KV Attention in PIM for Multi-Agent LLM Inference

中文题目：**KVChime：面向多智能体推理的 PIM 共享 KV 注意力执行**。

命名含义：KV + Chime，表示多个 agent 围绕共享 KV 协同执行。[arXiv 命名检索记录](naming-check.md)（2026-09-07）：同名及连字符、空格变体的官方 API 查询共返回 0 条结果。

**一句话概括：KVChime 将查询侧 RoPE 对齐、共享 KV 与私有替换 token 的逻辑拼接，以及 Multi-Query 执行与动态选边结合起来，使多智能体既能保留 KV 复用的存储收益，又能直接在 PIM 中执行注意力。**

本文档是新论文的写作大纲，包含论证顺序、三项贡献、三张 motivation 图的设计以及新的 evaluation 口径。文中的预期收益是待验证假设；现有实验的支持范围单独列出。论文暂定名为 **KVChime**，现有实验文件名保留，便于追溯。

## 1. 全文主线

多智能体反复使用相同的文档、工具输出和历史上下文。软件 KV reuse 减少了每次请求需要重新计算的 token，但每次 attention 仍需消费较长的历史 KV；随着共享工作集、私有更新和并发请求增加，近端 GPU HBM 的容量压力上升。一旦部分 KV 必须从远端读取，attention 的有效供数速度受到互连限制，小 Q 的 selective prefill 也可能变成 memory bound。

PIM 能就近消费这些 KV，但普通单查询 PIM 的计算上限较低：当一个共享 KV 需要服务多个 prefill query 或多个 agent 的 decode query 时，额外的乘法可能让 PIM 进入 compute bound。于是出现一个值得解决的区间：**GPU 受数据供给限制，普通 PIM 受计算能力限制。**

把软件 reuse 接到 PIM 上，还必须处理两个执行问题：共享 chunk 在不同请求中的位置不同，如何避免为了 RoPE 把大块 K 读回 GPU；部分 token 重算后，如何让每个请求读到自己的更新版本，同时继续共享其他 KV，而不重新拼出并保存一份完整副本。

KVChime 先解决位置和版本语义，使共享 KV 能够留在 PIM 中被正确消费；再通过 MQ 提高共享扫描的有效计算吞吐，并按完整服务成本选择 prefill attention 的执行设备。评价沿着这条路径回答：软件复用获得了什么、直接嫁接 PIM 丢失了什么、KVChime 恢复了什么、MQ 又扩大了哪些受益场景。

### 写作时统一的三个因果关系

- **HBM 容量不足不会降低 HBM 的标称带宽。** 应写成“远端访问比例增加，使 attention 的有效带宽受 link 限制”，并明确缓存驻留位置。
- **cache 变长不等于计算强度必然下降。** 在固定 Q 下，历史长度增加会同时增加计算量和 KV 读取量；主要是复用降低 Q，以及容量溢出改变数据来源，造成上述瓶颈。
- **MQ 缓解的是 PIM compute bound。** 共享列读减少重复读取；相应的多次 MAC、PE 频率和查询状态共同提高有效吞吐。只合并请求而不提供足够的计算速率，并不会自动抬高 PIM 的计算上限。

## 2. 三项贡献

1. **面向 PIM 的位置对齐：将已有的查询侧 RoPE 对齐方法用于驻留 PIM 的共享 KV，让不同位置的请求通过各自的 Q 操作数访问同一份 K，避免仅为位置修正而回读、旋转和重写整块 K。**
2. **支持共享与替换的注意力数据流：用共享 chunk 引用、私有替换 token 和新增 token 构成请求的逻辑 KV 视图，在 QK、softmax 和 PV 全流程中选择正确版本，避免完整 KV 的重复物化。**
3. **面向共享操作数的 MQ 执行与动态选边：让一个 bank 列读服务多个查询，在时序、功耗和状态容量约束下提高计算吞吐，并按完整 attention 服务成本在 GPU 与 PIM 之间选择 prefill 路径。**

三项贡献分别回答“位置怎么对齐”“更新怎么拼接”“多个查询怎么高效执行”。论文不包含 row/channel placement 优化；物理存储与读取沿用 AttAcc 的基础映射。

### 贡献一：把位置对齐应用到 PIM 驻留 KV

**PIM 中保存什么。** 对 chunk c，保存一份参考坐标系中的 K，以及未施加 RoPE 的 V；记录该 chunk 生成时的参考起点 r_c。采用固定频率 RoPE 时，chunk 内第 t 个 key 为：

\[
\bar k_{c,t}=R(r_c+t)k_{c,t}.
\]

请求 i 将它放在逻辑起点 b_{i,c}，定义位移 Δ_{i,c}=b_{i,c}−r_c。其逻辑位置 p 的查询由 GPU 对齐成：

\[
q'_{i,c}=R(p-\Delta_{i,c})q_i,\qquad
(q'_{i,c})^T\bar k_{c,t}
=(R(p)q_i)^T R(b_{i,c}+t)k_{c,t}.
\]

因此，PIM 接收对齐后的 Q 并执行普通 MAC，共享 K 保持参考坐标系。相同位移的 chunk 可以复用同一 Q 变体；不同位移需要不同变体，其生成与传输成本必须计入。不能同时写成“缓存的是完全不带局部位置的 raw K”以及“每个 chunk 只旋转一次 Q”：本方案保留 chunk 内相对位置，Q 补偿的是 chunk 整体位移。

例如，一个 chunk 按位置 0–3 缓存。Agent A 把它放在位置 32–35，查询位于 40，则使用 R(8)q_A；Agent B 把它放在 96–99，查询位于 107，则使用 R(11)q_B。两者读相同的缓存 K 字节，但得到各自正确的相对位置。Q 的内容仍然各不相同。

**已有方法与本文的关系。** CacheBlend 已显式处理 K 的位置修正，并通过选择性重算修复复用后的上下文偏差；LazyAttention 已讨论通过 Q 的相对旋转对齐共享 chunk。本文把这种位置处理接入 PIM 的驻留操作数和执行接口，不声称发现新的 RoPE 恒等式或首先实现位置无关复用。[CacheBlend 正文与附录](https://arxiv.org/html/2405.16444v3)、[LazyAttention §3](https://arxiv.org/html/2606.04302v1)。

RoPE 对齐只保证**选定 KV 视图的位置语义**；它不会恢复旧 hidden state 中缺失的跨文档依赖，也不替代 CacheBlend/EPIC 的 token 重算策略。对于 GPU attention，KV 本来就要送到 GPU，位置处理可以与读取或 kernel 融合；对于驻留 PIM 的 attention，强制把 K 送回 GPU 才会破坏就地执行的收益。因此不能把 GPU 路径的一次必要 KV 读取再重复记成一次独立的 RoPE 回读。

**本项预期体现的收益：** PIM 路径减少因位置对齐产生的 link 流量、位置相关 K 副本及相应延迟。Q 变体数量较多时的额外开销也要报告。

### 贡献二：共享 KV 加私有替换的逻辑拼接

软件重算产生的新 K/V 是对旧逻辑位置的**替换**。一个请求的 attention 应消费：未被替换的共享 token、本请求的替换 token，以及新增 token；同一位置只能有一个有效版本。

数据流依次说明五件事：

1. 软件提供 chunk 引用、请求中的逻辑位置和重算集合 D_i。
2. GPU 只为需要更新或新生成的 token 产生新 K/V；每份 K 的参考坐标系必须与描述符一致。原地修改公共 KV 会污染其他请求，因此更新保留为私有版本。
3. QK 按请求视图选择 shared 或 private K；被替换的旧 K 不贡献有效 score。若物理读取粒度使旧数据仍被读出，只能屏蔽其贡献，不能把实际读取成本凭空删除。
4. causal mask 和 softmax 按完整逻辑序列执行，所有 chunk 与 private token 共享该查询的正确归一化域。不能分别 softmax 后直接相加；分段归约必须携带可正确合并的归一化状态。
5. PV 使用与 score 相同的版本选择和逻辑位置，结果回到 GPU 执行 projection/FFN。后续 decode 继续使用同一逻辑视图，输出 token 进入本请求的私有尾部。

这里的“拼接”是 attention 操作数和结果的逻辑组装，避免在 GPU HBM 或远端 HBM 中构造一份新的完整物理 KV。不同坐标系的 private K 可以使用各自的描述符与 Q 变体；不能未经对齐就混用。

**最小例子。** 只看一个 head、一个 transformer layer：两个 agent 共享四个 KV token S=[s0,s1,s2,s3]。A 重算位置 1 得到 a1，B 重算位置 2 得到 b2，两者分别新增 a4、b4。

| 请求 | 逻辑 attention 视图 | 私有存储 |
| --- | --- | --- |
| A | [s0, a1, s2, s3, a4] | a1、a4 |
| B | [s0, s1, b2, s3, b4] | b2、b4 |

两个请求的逻辑长度都是 5，替换不会把长度变成 6。保留公共 cache pool 时，完整物化路径占 4+5+5=14 个 KV token 槽；共享加替换路径占 4+2+2=8 个槽。这个 **14→8** 是用于解释机制的存储计数，不是实验容量结果，也没有计入描述符和临时缓冲。

**本项预期体现的收益：** 保留软件复用，减少完整副本和写回流量，并降低物化造成的 TTFT 开销；基础 PIM decode 可以继续执行。是否两项延迟都改善，应由完整关键路径验证。

### 贡献三：共享扫描的 MQ 与动态执行

一个 bank 读到 K/V 列后，让多个驻留查询依次完成各自的 MAC，并保留独立累加、mask、softmax 和输出状态。MQ 同时服务两种来源：一个请求 selective prefill 中的多个 query，以及多个就绪 agent 对同一共享 KV 的 decode query。本文的 MQ 是多查询执行机制，不是模型结构中的 MQA。

多个 agent 消费同一 KV 时，多个 GEMV 在数学上可组织成以共同 K/V 为操作数的小 GEMM，但硬件仍可通过复用列读、轮转多个向量来实现。只有相同模型、层、兼容 head 和相同 KV 版本的共享部分可以合并；私有替换、私有尾部仍要分别处理。各 agent 的 Q、概率、mask 和输出不能合并成一份。

设计点围绕现有 **8 个驻留 query、约 1.3 GHz PE、满组间隔 8 tCK** 展开。频率平衡实验负责解释这个点，不能仅因仿真当前设置了该值就宣称它是最优点。更高频率的收益需要同时考虑 DRAM 供数间隔与功耗约束。

prefill 选边的首要工作量变量是实际参与计算的 Q，而不是仅看 recompute 比例。决策还包含 KV 驻留状态、历史长度、chunk 位移数量、link 流量及能否 overlap。比较的是两条完整可行路径：

- GPU：所需旧 KV 回读、位置处理与拼接、GPU attention，以及必须完成的存储更新。
- PIM：Q 对齐与传输、MQ scan、softmax/归约、未被合法 overlap 隐藏的新 KV 传输，以及结果返回。

本文提出的动态选边作用于 **prefill attention**；decode 以 PIM 执行为基础，F4 再对可共享部分使用跨 agent MQ。合并窗口的等待时间、非共享扫描和不满组开销都属于成本。

## 3. 章节大纲

| 章节 | 段落组织 | 要回答的问题 |
| --- | --- | --- |
| 1. Introduction | 多 agent 的重复上下文 → 复用后 Q 变少与容量压力 → GPU/PIM 两难 → 位置与替换问题 → KVChime 与三项贡献 | 为什么已有软件 reuse 和普通 PIM 还不足以直接组合？ |
| 2. Background | selective recomputation、近远端 KV、RoPE、AttAcc bank MAC、prefill 与 decode 的 Q 来源 | 软件决定复用什么，硬件需要执行什么？ |
| 3. Motivation | 小 Q 的 roofline；跨位置 KV 的表示差异；共享 token 的替换与物化 | 性能、位置和数据流三个障碍分别在哪里？ |
| 4. Shared-KV Execution in PIM | 参考坐标系与 Q 对齐；shared/private 视图；QK→softmax→PV；最小双 agent 例子 | 如何直接消费共享 KV，又保持每个请求的正确版本？ |
| 5. MQ Architecture and Scheduling | 共享列读与多次 MAC；频率/状态/功耗约束；prefill 与并发 decode；完整服务成本选边 | 如何扩大 PIM 受益范围，并处理共享 cache 上的并发查询？ |
| 6. Methodology | 原生 AttAcc 配置、输入来源、F0–F4、时序/面积方法、精度与指标口径 | 结果如何获得，各对照究竟改变了什么？ |
| 7. Evaluation | 频率平衡 → 面积 → MQ 与选边边界 → F0–F4 完整效果及消融 | 硬件代价是否合理，每项机制对哪些指标有效？ |
| 8. Related Work | 软件非前缀 reuse；query-side RoPE；PIM attention；共享操作数批处理 | 本文贡献落在 PIM 的执行实现与协同设计 |
| 9. Conclusion | 驻留共享 KV 的正确执行，以及计算与存储收益如何共同保留 | 仅总结经过验证的结论 |

## 4. Motivation 的三张图

本节定义新图的内容和取数口径；这次仅整理大纲，不生成或替换图文件。

### M1：The bandwidth–compute dilemma after KV reuse

**图型：roofline。** 横轴 arithmetic intensity，纵轴 attention throughput。展示 GPU 的近端/远端供数限制与普通 PIM 的低计算平台，标出 selective prefill 的工作点；核心是有一段区域同时受到 GPU memory bound 与 PIM compute bound 的限制。MQ 的提高放在 evaluation 展开，motivation 保持简洁。

单 head 的 QK+PV 在忽略 softmax、每份 KV 理想读取一次时，有 FLOPs≈4QNd、KV bytes≈2Nds，因此 I_KV≈2Q/s；FP16 下约为 Q FLOP/byte。这里 N 是实际完整逻辑 KV 长度，s 是元素字节数。重算 token 替换已有位置，不再增加 N。

应固定每条线的计算精度、带宽边界和数据移动口径。若统一使用逻辑 KV 字节定义横轴，远端流量、重复读取和额外写回要反映在有效供数上限里；不能把 PIM 内部 bank 带宽直接当成 GPU 经 link 访问 KV 的带宽。

**图要支持的结论：** 软件 reuse 使小 Q attention 暴露数据供给瓶颈，而普通 PIM 不一定有足够的多查询计算吞吐。HBM 溢出的论断还需要工作集与容量预算的证据；当前预设 KV 已在远端的 sweep 只能说明这一驻留条件下的代价，不能证明真实 workload 已经发生溢出。

### M2：Position shifts change the representation of reusable keys

**图型：一张图内并列 pre-RoPE 与 post-RoPE 的对应 token 相似度。** 使用同一文档在不同请求/位置中的真实中间张量，按相同 token、层和 head 配对，比较 K 的 cosine similarity；两幅子图使用相同样本、排序和色阶。V 单独作为控制量，不混入“RoPE 导致的差异”。

标准 LLaMA RoPE 直接旋转 Q/K，不旋转 V。图题宜写“KV 复用中的 K 位置差异”，避免把 V 也描述成经过 RoPE 的张量。

取数要区分两种证据：真实不同前文下的 pre/post K 相似度，反映上下文差异与显式旋转共同存在；固定同一 raw K、只改变位置的控制实验，用来隔离 RoPE 本身的作用。控制实验中 pre-RoPE 相同是构造条件，不能冒充真实跨请求测量。附加的对齐后 score/output 误差用于验证 Q 对齐正确性，不能仅靠相似度高低证明 attention 正确。

**图要支持的结论：** 同一 chunk 不能无视消费者位置直接复用其位置编码 K；在 PIM 场景中，应避免为此反复搬运并重编码大块共享 K。相似度没有预设必须达到的差距，也不能保证随位移单调变化。

### M3：Selective recomputation requires replacement, not append

**图型：共享 chunk 与两个 agent 的流程示意。** 沿用上面的四个 shared token、两个不同替换位置与两个私有新增 token。图中只保留公共块、A/B 的替换位置以及各自的逻辑序列。

问题路径展示“读回 shared KV → GPU 位置对齐并替换 → 拼出 A/B 两份完整 KV → 写入 PIM 供 decode”；旁边标出三种约束：原地覆盖会污染另一请求，直接追加会重复计算旧位置，完整复制会增加容量与流量。设计章节再解释 KVChime 如何直接执行这些逻辑视图。

**图要支持的结论：** 替换语义同时影响共享安全、softmax 的 token 集合、PV 版本选择与存储成本。该图是机制示意，不是实测热点分布，也不引入 placement。

## 5. Evaluation 的四个部分

新论文用 E1–E4 表示评价部分；它们与系统实验目录 Experiment 1–5 不是同一套编号。最终实测图只保留 performance 和 capacity；面积保留 RTL 仓库表格与核验，能耗仅在系统实验 README 中简述并留原始 CSV，不安排面积或能耗图。

### E1：PE frequency under a DRAM power budget

**问题：** 为共享列读服务多个查询，需要多快的 PE？继续提频何时不再改善有效命令吞吐？

沿用 `kvpim-rtl` 仓库内既有频率证据与约 1.3 GHz / 8 tCK 设计点，不重新 sweep；频率性能图及原始数据、独立绘图脚本只在该仓库维护。

### E2：Hardware area and state overhead

保留最终 RTL 组件汇总和审查口径，只在 `kvpim-rtl` 整理；不安排面积图，也不为本文重做综合或三档面积消融。正文若需要，只用简短文字引用该仓库核验过的最终组件结果，不能把组件面积说成整个 HBM 的面积。

### E3：MQ raises useful PIM throughput and changes device selection

**问题：** MQ 把哪些原本受 PIM 算力限制的 attention 变得更划算？完整成本下，何时选 PIM，何时选 GPU？

这部分接入当前目录的 **Experiment 1 交点曲线与 Experiment 2 敏感性长条形图**。比较普通 PIM、MQ PIM 和 GPU，分别 sweep 实际 Q、历史 cache 大小和 link 带宽；同时保留 scan 与完整 attention service 两个口径。

主图使用每张一个响应指标的交点曲线；多个 Q/link/cache/model 组合按 AttAcc 样式展开为分组长条形图。交点是范围或存在反复切换时，如实报告；不能预设固定的“200 token 以下选 PIM”，也不为了得到交点修改开销。频率来自 E1，面积代价来自 E2。

**分解项：** GPU kernel、旧 KV 回读、位置处理、Q 输入、PIM scan、softmax/归约、结果返回、新 KV 写入及其中暴露在关键路径上的部分。RoPE/Q 变体与新数据流若新增了成本，应重新定价后再作为 KVChime 的边界。

### E4：F0–F4 end-to-end benefits and attribution

**问题：** 软件 reuse、原生 PIM、共享执行支持以及 MQ/选边，分别改变了什么？

F0 采用 CacheBlend/EPIC 对照体系中的 **full-KV-recompute GPU 路径**，作为无非前缀 chunk reuse 的整体参照；F1 才是相应软件的 reuse 路径。两种软件分别做一套 F0–F4，不把两种重算策略混为一个 baseline。各档使用相同模型、请求、输出长度和硬件资源预算。

| 档位 | 软件与位置处理 | prefill attention | decode attention | KV 保存/执行方式 | 主要比较目的 |
| --- | --- | --- | --- | --- | --- |
| F0：GPU full recompute | 不启用非前缀 chunk reuse；按请求重新计算完整 KV | GPU | GPU | 每请求原生 KV；容量不足时使用明确的共同容量策略 | 整体参照，量化软件 reuse 的价值 |
| F1：Software reuse on GPU | CacheBlend/EPIC 复用；所需旧 KV 读到 GPU 并完成位置处理 | GPU | GPU | 远端仅作共享 cache pool，GPU 保存执行所需的当前视图 | 展示 reuse 的 TTFT 收益、缓存共享机会及回读/位置处理代价 |
| F2：Software reuse + native PIM | 相同重算集合；GPU 对齐、替换并拼出完整 KV | GPU | 原生单查询 PIM | 保留公共 pool，并将各请求完整私有 KV 写入 PIM | 量化接入原生 PIM 的 decode 收益与物化代价 |
| F3：KVChime shared execution | 相同重算集合；PIM 路径使用 Q 对齐与 shared/private 逻辑视图 | 固定 GPU | 支持逻辑视图的单查询 PIM | 公共 KV 一份，私有替换/新增各自保存；不完整物化 | 验证位置对齐与拼接能保留复用/容量收益，同时执行 PIM decode |
| F4：KVChime MQ + selection | F3 的存储与语义，增加 MQ 和执行成本选择 | GPU 或 MQ PIM，动态选择 | 可共享部分跨 agent MQ；私有部分独立执行 | 与 F3 相同的逻辑 KV；额外计 MQ 状态和等待 | 验证 prefill 的 TTFT 收益及共享 decode 的 TBT 收益 |

F3 的 prefill 固定 GPU，因此该阶段需要的旧 KV 读取仍要计入；避免完整副本不等于 GPU attention 可以不读 KV。F3 在 decode 中利用位置对齐与逻辑视图，F4 才进一步尝试让 prefill 也直接消费 PIM 中的 KV。

**比较顺序与应报告的结论：**

- F0→F1：软件 reuse 对 TTFT 的作用，以及公共缓存和活动请求副本分别占多少容量。F1 仍可能保留完整 GPU 活动 KV，不能预先承诺其“近端+远端峰值总容量”一定小于 F0。
- F1→F2：PIM decode 是否降低 TBT；为它付出的完整拼接、写回、私有副本是否侵蚀 TTFT、容量和 E2E。可能侵蚀部分收益，不预设会全部丢失。
- F2→F3：保持相同单查询 PIM 与软件重算策略，观察物化流量、TTFT 和副本容量的变化；F3 相对 F1 是否兼顾共享存储与 PIM decode。
- F3→F4：相同 shared/private 视图下，分别报告 MQ prefill 与跨 agent MQ decode 的效果，以及动态选边避免了哪些不利情况。没有共享或只有一个就绪 query 时，decode 收益可以为零。

F1 与 F3 同时改变了执行设备和数据流，不能直接把全部差值解释为 RoPE。应补一项固定设备/视图的对照：PIM 执行前 GPU 读回 K 做位置处理，再写回；对比 Q 对齐后直接执行。另用固定对齐方法的“完整物化 vs 逻辑视图”隔离拼接收益。MQ 与选边也用 F3 加“仅 MQ”和“仅选边”的小消融分开解释，主图仍保留 F0–F4。

**必须报告的指标：**

| 指标 | 统一口径与分解 |
| --- | --- |
| 容量 | GPU HBM 与远端 HBM 分开；共同 cache pool、每请求 KV、替换/新增、描述符和临时峰值分开；总峰值在同一时间点取值，不能简单相加各自不同时刻的峰值 |
| Scan | prefill/decode 分开；必要时再拆 shared/private，说明是服务延迟还是累计设备工作量 |
| TTFT | 请求释放到首 token，包含回读、位置处理、拼接/描述符准备、attention、GPU 其余层算子和必要等待 |
| TBT | 首 token 后相邻 token 间隔；并发时包含排队与 MQ 组批等待，不用总 batch 时间除以 batch 伪造单请求间隔 |
| E2E latency | 每请求从释放到最后 token；工作负载 makespan 另列，统一输出 token 数 |


除 speedup/降低比例外保留绝对值。按原有用户要求始终报告 scan，主图突出受影响的 TTFT/TBT/E2E，完整 CSV 保留全部指标；能耗只在实验 README 中用一句话概括，不画图。F0 的完整重算、F1–F4 的缓存预热应在冷启动与预热后两种口径中交代，不能只向某一个方案收取共享 pool 的准备成本。

**workload 组织：** 先用当前两个 CacheBlend 小输入和 EPIC 长上下文输入验证复用与 prefill；再使用同一文档被 1/2/4 个 agent 同时读取的简单 case 验证 MQ decode。控制共享比例、私有替换量、位置偏移和就绪时间；使用真实 batch 形状，不做请求外推。HBM 溢出机制另列容量预算/驻留敏感性，不能把“小输入、预设远端”称为已经测到真实容量溢出。

## 6. 现有证据与新大纲的对应关系

当前系统实验统一为 LLAMA-7B、GPT-13B、LLAMA-65B、LLAMA3.1-8B（GQA），在本仓库代码中运行。详细口径见 [多模型 methodology](../../docs/KVChime-multi-model.md)，代数正确性见 [数学说明](../../docs/KVChime-correctness.md)。

| 系统实验目录 | 说明 | 证据边界 |
|---|---|---|
| [Experiment 1](../KVChime-experiment-1-prefill-boundary/README.md) | 普通/MQ PIM 与 GPU 服务曲线 | 原生矩形 attention 与暴露传输；不是独立 RoPE kernel 时间 |
| [Experiment 2](../KVChime-experiment-2-link-cache-sensitivity/README.md) | Link/cache 敏感性 | 相同形状的 profile，link 成本解析重计 |
| [Experiment 3](../KVChime-experiment-3-software-reuse-F0-F4/README.md) | CacheBlend/EPIC F0–F4、容量、TTFT/TBT/E2E | 冻结 token/重算输入的硬件回放；非数值 LLM 运行 |
| [Experiment 4](../KVChime-experiment-4-shared-query-MQ/README.md) | 相同共享对象的跨 query/agent MQ | 已就绪消费者的机制实验；不含线上到达排队 |
| [Experiment 5](../KVChime-experiment-5-device-selection/README.md) | 固定 GPU/PIM、估计器、oracle | 两点校准的朴素估计器；如实报告 regret |
| RTL 仓库 | 既有频率与最终面积表 | 只在 kvpim-rtl 内整理，无新综合或频率扫描 |

目前数学证明保证给定重算集合的逻辑视图等价；命令仿真检查唯一版本和完整 query 的 QK→softmax→PV 数据流，不执行数值张量。Q 旋转算术仍沿用原生算子未单独计时的范围；额外 Q 版本和 descriptor link 已计入。因而不要声称独立量到了 RoPE kernel 收益或完整数值精度。

旧单模型结果保留在 `artifact/legacy-paper`，不能用其 F4 定义解释新版 MQ decode。前面的 motivation 示意图和未完成的数值相似度图是论文规划，不是本次已产出的实测 performance/capacity 图。

## 7. 正文措辞与相关工作定位

Introduction 可用的收束段：

> We present KVChime, a PIM execution architecture for shared-KV multi-agent inference. KVChime applies query-side positional alignment to stationary keys, executes request-specific replacements through logical KV views, and combines multi-query bank execution with cost-based prefill scheduling. These mechanisms are designed to preserve software reuse while reducing materialization and improving attention execution over shared contexts.

Related Work 中分别承认 [CacheBlend](https://arxiv.org/abs/2405.16444) 与 [EPIC](https://arxiv.org/abs/2410.15332) 的软件复用/上下文处理路线，以及 [LazyAttention](https://arxiv.org/abs/2606.04302) 的位置处理方法。本文强调如何把选定的 KV 视图留在 PIM 中执行，以及对应的数据流、硬件吞吐和调度代价。

尚未补齐证据之前，正文使用“we design / we enable / we evaluate”描述机制和评价计划，不填入预期 speedup 充当结果。正式论文的结论应同时保留不受益场景：例如 cache 已驻留 GPU、Q 较大、共享比例低或组批等待抵消扫描收益。

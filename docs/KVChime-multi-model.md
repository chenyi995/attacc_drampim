# KVChime 多模型实验与复现口径

最新最终图只包含 performance 和 capacity；能耗原值继续保留，在实验 README 中简述。面积只在 `kvpim-rtl` 仓库整理表格与证据，不生成面积图，也不复制到这里。当前论文图包见 [Fugue-paper](../Fugue-paper/README.md)。

本轮完成状态、失败尝试的去向和最终校验见 [完成记录](KVChime-reproduction-checks.md)。

## 模型与硬件

| 模型 | 来源 | 层数 | Q / KV heads | hidden / FFN | TP |
|---|---|---:|---:|---:|---:|
| LLAMA-7B | AttAcc 原生 | 32 | 32 / 32 | 4096 / (4096 × 8/3) | 1 |
| GPT-13B | AttAcc 原生 | 40 | 40 / 40 | 5120 / 20480 | 1 |
| LLAMA-65B | AttAcc 原生 | 80 | 64 / 64 | 8192 / (8192 × 8/3) | 4 |
| LLAMA3.1-8B | 显式 GQA 扩展 | 32 | 32 / 8 | 4096 / 14336 | 1 |

原生模型参数以 `src/config.py` 和每图 `models.json` 为准。LLAMA-65B 承接 AttAcc 的多 GPU 模型假设，使用 TP4；每个模型内部各方案的 GPU 数相同。LLAMA-65B 的并发 8K workload 在 TP2 下容量不足，因此全部实验统一用 TP4；旧失败运行和容量断言保留在 output。全模型层数、每 GPU head 数、FC 和 TP 通信均按相应模型计算。

GQA 的 32 Q heads / 8 KV heads、32 层、4096 hidden、14336 FFN 参考 [Meta Llama 3 论文 Table 3](https://arxiv.org/html/2407.21783v2#S3.T3)。原生配置已有 `gqa_size` 字段，但现有模型均为 1；本次在 `fugue/kvchime_model.py` 显式补充 GQA 操作数形状。把同一 KV head 的 Q heads 折到矩阵行维度，保持全部 QK/PV 算术，K/V 存储和读取只保留 8 个 heads；QKV projection 的 K/V 输出宽度和模型权重容量也相应缩减。原生 `src/` 不修改，不把 MHA 的 latency 直接除以 4。

默认 A100a、FP16、每 GPU 对应 5 个 PIM HBM、NVLink 3 单向 300 GB/s、native bank PIM 与 power constraint。MQ 最多驻留 8 个 query，设计点约 1.3 GHz；满组列计算间隔为 8 tCK。单 query、MQ 尾组以及 power-constrained 间隔都按 `fugue/attention.py` 的配置计入。带宽敏感性只改变 link 传输成本，其他硬件固定。

## Workload 的实际行为

以下八个形状在四个模型上各跑一次，每个形状均比较 F0–F4，总计 32 个模型/workload 组合、160 行方案结果。

| Workload | 同时请求数 | 总 prompt N | 部分重算 Q | 输出 token |
|---|---:|---:|---:|---:|
| KVChime-CB-3 | 1 | 3420 | 595 | 10 |
| KVChime-CB-2 | 1 | 3458 | 604 | 10 |
| KVChime-EPIC-long-4000 | 1 | 4028 | 106 | 16 |
| KVChime-EPIC-long-8000 | 1 | 8033 | 186 | 16 |
| KVChime-EPIC-long-16000 | 1 | 16045 | 346 | 16 |
| KVChime-EPIC-shared-readers-B1 | 1 | 8035 | 188 | 16 |
| KVChime-EPIC-shared-readers-B2 | 2 | 8035 | 188 | 16 |
| KVChime-EPIC-shared-readers-B4 | 4 | 8035 | 188 | 16 |

CacheBlend 使用仓库内冻结的 demo 2/3 文本、chunk 和 token IDs。第一层完全重算；第二层完整 QKV projection 后做部分 attention；后续层使用冻结的重算数量。旧 token 的重算比例为源设定 16%，再加新 suffix。实际 top-k 位置未通过 LLM 数值运行测量，本实验使用保持数量的等距旧位置，不能称为软件准确率结果。

EPIC 使用冻结的 LongContext 4K/8K/16K 文本构造，每个请求按次序读取若干文档 chunk，然后处理查询；kvlink-16 重算集合包含非首文档边界的前 16 个 token 及完整 query。具体 chunk 长度、顺序与索引见 `data/workload.json` 和 `artifact/inputs/epic.json`。8K readers case 把相同文档给 1/2/4 个同时就绪请求，描述多 agent 共读，不引入额外拓扑。

这些是同一组软件源文本的**硬件形状回放**。四个模型保留相同冻结 token IDs 和长度，不逐模型重新分词，也不运行语言模型生成答案。输出长度是明确的仿真截断。Experiment 4 另取同一 EPIC 4K 文档，构造 1/2/4/8 个已就绪消费者及 4-agent 的共享比例控制；共享比例按 chunk 个数指定，实际共享 token 数同时报告，不能称为真实服务到达 trace。

## F0–F4 的含义

| 方案 | Prefill attention | Decode attention | KV 读写与容量 |
|---|---|---|---|
| F0 | GPU，全 prompt 重算 | GPU | 无远端共享池；GPU 留完整请求 KV，作为全重算参照 |
| F1 | 软件部分重算，GPU attention | GPU | 远端 HBM 只作不可变 cache；复用部分读回 GPU，GPU 留完整请求 KV |
| F2 | 同样软件重算，GPU attention | 原生 PIM，逐 query | 读回复用部分并拼成完整私有上下文，写回远端；共享池之外每请求保留完整副本 |
| F3 | GPU 固定 | 共享视图 PIM，逐 query | 共享块保持驻留，远端仅新增私有重算/生成 KV；GPU prefill 所需复用 KV 仍须读取 |
| F4 | 简单成本估计选 GPU / MQ PIM | 共享视图 MQ PIM | 与 F3 相同版本/容量语义；共享列可服务多个 Q 或多个已就绪 agent |

F0 是关闭 reuse 的全重算参照，F1 才是本文的 GPU 软件 reuse 对照；不能将 F0 标成 EPIC/CacheBlend 已启用优化后的方案。F1 读取已含在 GPU attention 服务成本内；RoPE 旋转算术没有单独 native kernel 成本，因此 F1 对比支持**读回和执行位置开销**，不是测得的独立 RoPE kernel 时间。

位置对齐、旧项替换和跨块 softmax 的等价性见 [数学说明](KVChime-correctness.md)。F3/F4 按不同 chunk 位移发送不同 Q 版本，额外 Q 和视图 descriptor 传输有计费。物理 channel/bank/row 的原生分布保持不变，不做 placement 优化。

## Latency 与容量包括什么

单层 GPU attention 服务由 QK、softmax、PV、复用 KV 读回及需要暴露的新 KV 导出组成；不加虚构的固定 launch 开销。单层 PIM 服务包含 Q/descriptor 输入、原生命令 scan（QK + PV）、softmax、输出返回和未被覆盖的新 KV 传输。新 KV 的 overlap 上限来自命令日志中首次使用私有 K 前的窗口；只扣除此窗口，不直接删掉全部传输。KV 导入/导出按原生 link 成本计价；窗口给出可隐藏传输的上限，没有另外引入 DMA 与 bank 命令争用模型。单层 sweep 使用 Q 输入及输出，只有 workload 的多 chunk 视图增加额外 Q 版本/descriptor。

F2 主结果保留读回、attention、完整导出的串行物化路径；`F2-overlap-control.csv` 同时给出导出与 attention 重叠的对照。F3/F4 固定 GPU 路径允许新 diff 写入与 GPU attention 重叠，不能把 F2 主结果差距全部归因于少存储；请同时检查这一对照。

- **TTFT**：所有正常 Transformer 层的 prefill QKV projection、软件选择读取、attention 服务、输出 projection、FFN、归一化与原生 TP 通信之和。
- **TBT**：生成首 token 之后各 decode 步的平均延迟；每步包含全部层的 QKV → attention → projection/FFN。多个同步请求按一个批次的服务时间报告，不按请求数除 latency。
- **E2E**：`TTFT + (输出 token 数 − 1) × TBT`；不包含排队、采样器或网络前端，沿用 AttAcc decoder-block 范围，不另计 LM head、host 调度或缺少原生成本的 Q 旋转/top-k 算术。
- **scan**：PIM 的 QK/PV 命令服务，不含 GPU projection、link 或 softmax；报告 decode 每生成 token 跨所有层的 scan。F0/F1 的 PIM scan 为零，不定义其相对 scan speedup。
- **容量**：同一时间快照的 GPU KV + 远端共享池 + 远端私有 KV 峰值；共享池只计一次，F2 的完整私有副本逐请求计入。另保留 GPU/远端分项和模型权重容量，不把两个不同时刻的峰值相加。

reuse 方案的共享池预热在 `warmup.csv` 独立报告，不混入在线 TTFT；一次性运行或复用很少时需要加回预热成本。较大池包含未被该单请求消费的对象，因此 F1 总容量相对 F0 不保证下降，必须按真实峰值报告。

## 五个系统实验各回答什么

1. **Prefill 边界**：每模型一张原定的 2×3 六联图，上排依次为 a 普通 PIM/C=0/300 GB/s、b 普通 PIM/C=1024/300 GB/s、c MQ/C=1024/450 GB/s；下排为 d MQ/C=0/300 GB/s、e MQ/C=1024/300 GB/s、f MQ/C=1024/32 GB/s。每格比较 GPU 与指定 PIM 的单层服务时间，保留原图留白、配色和聚焦窗口；交点超出默认窗口时扩展边界。四个模型共四张六联图。交点用相邻实际采样区间，未扫描整数不补造精确值。完整 8320 行 sweep 保留 Q=1–2048、8 种 C、13 种 link。
2. **Link/cache 敏感性**：分别固定 C=1024 或 link=300，按 Q、link/cache、model 展开长条形图；纵轴统一为 GPU/MQ 服务时间，>1 表示 MQ 更快。
3. **F0–F4**：分别画 TTFT、TBT、E2E、decode scan 和 KV 容量；每张图只有一个响应指标，模型外层分组，workload 内层分组。性能图为基准 latency / 本方案 latency（加速比），容量图为本方案 / F0。归一化只在同一模型、同一 workload 内进行。
4. **共享 MQ**：分别画 scan 与 TBT，展示就绪 query/agent 对共享列读取的影响。MHA 单 agent 的 decode 无跨 query 复用，因此 F3/F4 的 decode 必须相同；GQA 单 agent 已有同 KV head 的多个 Q，允许收益。
5. **四种选边**：固定 GPU、固定 MQ PIM、估计选边、逐点 oracle。每模型扫 C=0/1024/4096、Q=8/32/64/128/256/512/768/1024，报告实际 regret，保留错误选择。

频率设计点和面积证据属于 `kvpim-rtl`；硬件配置不随模型改变，不重复画四份相同硬件结果。绘图排版参照 AttAcc ASPLOS 2024 Fig. 13 的按 model/workload 分组、单一指标长条形图。

## 选边算法

每种 head 几何仅用 N=1024/4096、8-query 的两条 native profile 校准 `t8(N)=max(0,aN+b)`。按对象的 query 消费次数（包括 GQA group）估计 scan，结合已知传输大小决定设备，不读取当前形状的实际 PIM 时序结果作为算法答案。

```text
scan_hat = 0
for object in shared_and_private_objects:
    q = sum(queries of ready consumers) * GQA_group_size
    full, tail = divmod(q, 8)
    scan_hat += max(0, a * object.tokens + b) * (full + I(tail)/8)
pim_hat = scan_hat + softmax + Q_and_descriptor_input + output + full_new_KV_write
choose PIM if pim_hat < gpu_service else GPU
```

`I(0)=0`；其他尾组间隔沿用 simulator。估计器保守计全量新 KV 写入，不假装能提前准确预测 overlap。Oracle 用独立运行得到的两种实际服务成本的最小值，只作质量上界。对象开销与 bank padding 使两点拟合不精确；F4 不保证每点优于 F3。

## 仿真与独立重画

```bash
CXX=/path/to/c++20/g++ python3 -m fugue all-models --jobs 8 --output output/KVChime-fresh
python3 -m fugue all-models-plot --output output/KVChime-fresh
python3 -m fugue all-models-verify --output output/KVChime-fresh
# 不运行仿真，独立重画单图：
cd Fugue-paper/KVChime-experiment-3-software-reuse-F0-F4/paper/TTFT_ms
python3 plot.py --output-dir redraw
```

GPU 使用 native AttAcc 算子；PIM profile 运行本仓库打包的 Ramulator。每次全新运行重新生成命令并运行 profile，不读取论文 latency 表。相同操作数形状的 query tile 复用本次 profile，最多 8 个 resident query；`profile_repeats` 保留重复次数。这是原生矩形 attention 的算子成本组合，每个请求、每一层、每个生成步骤均计入，不把少量 request 外推到大 batch。沿用原生非融合 QK/softmax/PV 成本，不宣称 FlashAttention 或完整逐值 causal 数值执行。

默认 8 workers，并限制 CPU affinity、每进程地址空间和 BLAS 线程；无需真实 GPU、下载模型权重或兄弟仓库。新/失败运行保留在不同 output；阶段日志记录命令、代码/输入哈希和返回码。每张最终图的 `paper/<figure>/` 包含原始绝对值、命令 profile 计数/哈希、配置、独立绘图脚本和 PDF/PNG；完整 trace/YAML/命令日志位于本次 output；论文图包中的完整事件表用无损 gzip 保存，解压后与原始 CSV 的字节哈希一致。

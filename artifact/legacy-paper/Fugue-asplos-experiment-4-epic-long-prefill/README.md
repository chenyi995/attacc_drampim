# Experiment 4：EPIC 长上下文中的 Prefill Attention 成本

一句话：使用 EPIC 的真实长文本与 `kvlink-16` 规则，在本仓库 AttAcc 中直接回放较长上下文，分开核算 attention 计算、传输和整模型成本。

## Workload 与执行位置

| case | 原始文档目标长度 | 文档块数 | 实际 N | 每个请求 Q | batch | 每 HBM 仿真 head 数 |
| --- | --- | --- | --- | --- | --- | --- |
| long-4000 | 4000 | 6 | 4028 | 106 | 1 | 7 |
| long-8000 | 8000 | 11 | 8033 | 186 | 1 | 7 |
| long-16000 | 16000 | 21 | 16045 | 346 | 1 | 7 |

EPIC 源码提交 `3204410f1723ed0f39575b4eba8c17578a1ee1a1`。执行其 `LongContext.generate_context`、`encode_and_trim`、`Data_set._split_docs` 和两个 prompt 构造方法：从本地 Paul Graham essays 按原仓库遍历顺序读入文本，按预先选定的 4000/8000/16000 token 档位截取，以默认 512 token 块向句号/换行边界延伸分块；不为制造收益填充 token 或构造地址冲突。保留每段 BOS，所以实际 N 略大于目标文档长度。选样档位在仿真之前记录。

这三个是 EPIC LongContext 原始问题“Generate at most one word”的 workload 形状。

`kvlink-16` 的实际执行代码选每个文档开头 16 token，加上整个问题 token，去重得到 Q；所有 32 层均只为这些 token 产生 QKV、执行 Q×N attention、projection/FFN。系统块保留缓存。N 为原来完整 prompt 的长度，重计算是替换 N 中的部分位置，不是把 Q 再追加为 N+Q。上游 EPIC 在运行前分别缓存 system、文档和问题全部 chunk，这里保留该约定。每个请求对应的 token ID、重算位置、chunk hash 均在 [workload](workload/Fugue-asplos-workload.json)；原始文本与原始 token 分块也保留。

GPU/模型使用 **原版 A100a、LLAMA-7B、32 层、32 MHA head、FP16、5 HBM、NVLink 3 单向 300 GB/s**。为与实验 1–3 一致，使用本地 LLAMA tokenizer，未冒充 EPIC 默认 Llama-3.1-8B GQA 的模型/分词配置。长上下文只评估该形状的硬件成本，不验证原 LLAMA 模型在长位置上的数值可用性或回答质量。每请求固定 16 个输出 token，来自 EPIC `SamplingParams` 默认上限；真实 one-word/EOS 请求可能更早结束。

当前运行入口为仓库中的 `python3 -m fugue run --experiments 4,5`，实际导入同一仓库的 `src/`，Ramulator 从同一仓库捆绑源码构建。[来源与检查](provenance/Fugue-asplos-current.json) 保存模型和复现信息；原生 GPU、trace mapping 与 DRAM 模型保持不变。

## F1–F4 与 latency 包含什么

| 方案 | prefill | decode | KV 存储 |
| --- | --- | --- | --- |
| F1 | GPU | GPU | 远端 HBM 为不可变 chunk cache；GPU 保留本批所有层完整 active KV |
| F2 | GPU | 原版 PIM | 取回未重算部分，在原版 qkv→完整 KV 写出→attention 顺序中物化每请求完整 KV |
| F3 | 固定 GPU | 原版 PIM | 旧 chunk 引用共享，仅写入本请求重算/新生成版本；GPU attention 仍需读回未重算 KV |
| F4 | 比较完整 GPU 与 MQ PIM service，逐层选最短 | 原版 PIM | 与 F3 相同的远端共享与 diff 存储 |

设 G=GPU QK+softmax+PV，P=MQ native scan+原版独立 softmax 成本，R=未重算 KV 读回，D=新/重算 KV 写出，H=完整 N KV 写出，I/O=Q 输入和 attention 输出。每项都按实际 batch 的字节或算子维度计算：

- F1 service = G + R。
- F2 service = R + H + G，保留原版串行写出顺序。
- F3 service = max(G + R, D)，不同方向的 diff 导出可与 GPU 路径重叠。
- F4 service = min(F3 service, I + P + D + O)。本 workload 的更新散布于文档边界，保守设 KV overlap window W=0，没有套用连续旧前缀的扫描隐藏时间。

TTFT 对所有 32 层累加 **QKV→attention service→projection/FFN/norm**；TBT 为后续 15 个 decode step 的整模型平均；E2E=TTFT+15×TBT。第一输出 token 计入 prefill。所有 TTFT/TBT/E2E 都是整个批次的完成延迟，**不除以 batch**；能量表为整个批次，另有每请求能量和 throughput 列。生成步数 16 是统一 timing cap，E2E 不能解释为软件实际生成的答案长度。

MQ 为约 1.3 GHz、8 个 resident Q 对应 8 tCK，尾组按实际 r 设置间隔并计入独立 trace；普通 PIM 使用原版 r=1。MQ 只复用一个请求内部的 KV 扫描，不假设跨 reader 合并扫描。原生头数为 ceil(32×batch/5)，`fast_mode=False`；5 HBM 的对称建模沿用原 wrapper。F2/F3/F4 的 decode 使用真实单 Q trace，经原版 Ramulator wrapper 和原版 `System.simulate` 计算，TBT/energy 完全一致。

## 结果与 attention 是否暴露

| case | F4 TTFT 降低 | F4 E2E 降低 | F4 E2E 能量降低 | F3 远端容量降低（相对 F2） |
| --- | --- | --- | --- | --- |
| long-4000 | 37.85% | 7.57% | 1.41% | 48.59% |
| long-8000 | 31.67% | 10.72% | 2.94% | 48.80% |
| long-16000 | 20.43% | 11.24% | 5.91% | 48.90% |

上表延迟/能量降幅均为 F4 相对 F3；容量降幅单独比较 F3 与 F2。16K case 中纯 attention 已占 F3 TTFT 的约 58%，连同旧 KV 读回占约 82%。但 MQ 的 scan+softmax 比 GPU kernel 略慢；F4 仍选择 PIM，是因为省去 857.375 µs/层的旧 KV 读回。不能把全部收益归因于 MQ 算术更快。

| case | F3 纯 attention / TTFT | F3 attention service / TTFT | GPU kernel µs/层 | MQ kernel µs/层 | GPU 完整 service µs/层 | MQ 完整 service µs/层 | F4 选择 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long-4000 | 31.83% | 64.33% | 209.706 | 162.943 | 423.899 | 174.521 | PIM |
| long-8000 | 43.84% | 74.52% | 612.598 | 578.350 | 1041.148 | 598.666 | PIM |
| long-16000 | 57.96% | 81.95% | 2071.624 | 2160.899 | 2928.998 | 2198.692 | PIM |

“纯 attention”是 QK/softmax/PV 核心执行时间，“attention service”还包含关键路径上暴露的读回/写出/Q/O 传输。两者不能混称。全部六个 case 中 F4 的 32 层都选择 PIM；这说明这些预声明 workload 位于 PIM 优势区，**本实验不声称展示 GPU↔PIM 来回切换**。GPU 优势区和交点仍由实验 1/2 提供。

![prefill breakdown](figures/Fugue-asplos-experiment4-prefill-breakdown.png)

| case | 方案 | TTFT ms | TBT ms | E2E ms |
| --- | --- | --- | --- | --- |
| long-4000 | F1 | 21.085 | 7.970 | 140.636 |
| long-4000 | F2 | 28.124 | 5.621 | 112.443 |
| long-4000 | F3 | 21.085 | 5.621 | 105.403 |
| long-4000 | F4 | 13.105 | 5.621 | 97.423 |
| long-8000 | F1 | 44.710 | 10.496 | 202.154 |
| long-8000 | F2 | 58.749 | 5.828 | 146.171 |
| long-8000 | F3 | 44.710 | 5.828 | 132.133 |
| long-8000 | F4 | 30.551 | 5.828 | 117.973 |
| long-16000 | F1 | 114.371 | 15.550 | 347.617 |
| long-16000 | F2 | 142.412 | 6.232 | 235.898 |
| long-16000 | F3 | 114.371 | 6.232 | 207.858 |
| long-16000 | F4 | 91.001 | 6.232 | 184.488 |

![latency](figures/Fugue-asplos-experiment4-latency.png)

| case | 方案 | TTFT J | 每步 TBT J | E2E J |
| --- | --- | --- | --- | --- |
| long-4000 | F1 | 1.3374 | 0.4962 | 8.7798 |
| long-4000 | F2 | 1.3594 | 0.4364 | 7.9056 |
| long-4000 | F3 | 1.3380 | 0.4364 | 7.8842 |
| long-4000 | F4 | 1.2268 | 0.4364 | 7.7730 |
| long-8000 | F1 | 2.3741 | 0.5658 | 10.8605 |
| long-8000 | F2 | 2.4179 | 0.4467 | 9.1186 |
| long-8000 | F3 | 2.3751 | 0.4467 | 9.0758 |
| long-8000 | F4 | 2.1084 | 0.4467 | 8.8090 |
| long-16000 | F1 | 5.2626 | 0.7050 | 15.8377 |
| long-16000 | F2 | 5.3501 | 0.4673 | 12.3599 |
| long-16000 | F3 | 5.2645 | 0.4673 | 12.2743 |
| long-16000 | F4 | 4.5396 | 0.4673 | 11.5494 |

![energy](figures/Fugue-asplos-experiment4-energy.png)

能量沿用原版动态 pJ 模型：pJ/10^12=J；事务重叠只减时间，不删除能量。原版 PIM score 的能量口径包含合并扫描的访存与原版 QK 算术项，没有额外修补 PV 算术能量；X2G 保留原版 link 能量，不添加端点 DMA 或静态板级功耗。结果是模型估计，非板卡实测。

## 容量、预热和公平性

| case | 方案 | 远端峰值 GiB | GPU KV 峰值 GiB | 同一时刻总 KV 峰值 GiB |
| --- | --- | --- | --- | --- |
| long-4000 | F1 | 1.967 | 1.974 | 3.941 |
| long-4000 | F2 | 3.941 | 0.061 | 3.995 |
| long-4000 | F3 | 2.026 | 0.061 | 2.080 |
| long-4000 | F4 | 2.026 | 0.002 | 2.026 |
| long-8000 | F1 | 3.922 | 3.930 | 7.852 |
| long-8000 | F2 | 7.852 | 0.123 | 7.967 |
| long-8000 | F3 | 4.021 | 0.123 | 4.136 |
| long-8000 | F4 | 4.021 | 0.003 | 4.021 |
| long-16000 | F1 | 7.834 | 7.842 | 15.676 |
| long-16000 | F2 | 15.676 | 0.245 | 15.914 |
| long-16000 | F3 | 8.011 | 0.245 | 8.248 |
| long-16000 | F4 | 8.011 | 0.005 | 8.011 |

![capacity](figures/Fugue-asplos-experiment4-capacity.png)

不可变共享池按内容 hash 去重；F2 每请求额外存完整 N×32 层，F3/F4 每请求额外存 Q×32 层及 15 个 decode token。F1 的 active KV 留在 GPU。图中堆叠值来自同一个峰值时刻；独立 GPU 峰值和远端峰值另列，不把不同时刻的峰值相加。容量图不包括共同 GPU 权重；[summary](tables/Fugue-asplos-summary.csv) 同时给出原版权重、临时空间和 GPU 总需求，所有 case 均小于原版 80 GiB。引用表、allocator 碎片和数值 RoPE 转换开销未建模。

| case | 预热 ms | F4 在线 E2E ms | 预热 + F4 E2E ms | 预热 + F4 E2E J |
| --- | --- | --- | --- | --- |
| long-4000 | 290.747 | 97.423 | 388.170 | 36.631 |
| long-8000 | 569.197 | 117.973 | 687.170 | 65.124 |
| long-16000 | 1123.819 | 184.488 | 1308.307 | 122.819 |

在线表假设 cache 预热完成，上表显式计入每 case 独立的公共 warmup；这几个 case 是独立实验，不能把不同 case 的缓存池累计。预热逐 chunk 的 token、时间、能量在 [warmup](tables/Fugue-asplos-warmup.csv)。

F2→F3 的主表延迟收益同时包含完整写出字节减少和写出调度改变。为避免混淆，[F2 export-overlap sensitivity](tables/Fugue-asplos-F2-export-overlap-sensitivity.csv) 保留 F2 完整写出字节、容量和能量，仅允许其写出在取回后与 GPU attention 重叠：R+max(H,G)。当 G 足以隐藏完整写出时，该敏感性 TTFT 与 F3 一致；当 H 更长时仍留下写出差额。逐 case 的精确值见该表，不能把 F2→F3 主表的 TTFT 差全部称为纯存储节省。F3→F4 在同一服务模型下比较，独立于这个差异。

共享存储、引用和流量是实验账本；底层扫描继续采用原生每请求逻辑视图，未实现把不同请求的地址物理别名到同一 bank 的 allocator。因此本实验覆盖实际 batched head 数带来的通道/命令压力，**不声称测量了共享物理地址的热点冲突或跨请求合并读收益**。没有引入新 placement。

## 可核查的最终数据

[scan 表](tables/Fugue-asplos-scan.csv)、[完整 summary](tables/Fugue-asplos-summary.csv)、[选边成本](tables/Fugue-asplos-decisions.csv)、[事件](tables/Fugue-asplos-events.csv)、[算子](tables/Fugue-asplos-operators.csv) 和 [传输](tables/Fugue-asplos-transfers.csv) 保留最终版本。

## 独立复现本实验

本目录只保留最终版图和数据。旧图/旧脚本已归档到本地 output，不是复现依赖；完整最终 CSV 与主图聚焦 CSV 是同一份最终数据的不同视图。原始实验说明及不利结果保留在上文。

在仓库根目录安装依赖后运行（如尚未构建，先执行 build）：

```bash
python3 -m fugue build --jobs 8
python3 -m fugue run --experiments 4,5 --jobs 8
python3 -m fugue plot --experiments 4,5
python3 -m fugue verify --experiments 4,5
```

也可直接使用 `python3 -m fugue all --jobs 8` 从头复现所有实验。已成功的相同阶段可用 `--resume`；失败重试使用新的 `--output`。新 trace、YAML、逐 channel 命令、时间/能量事件和日志生成在 `output/Fugue-asplos-reproduce/`，不依赖历史 output。只重画本目录图表使用 `python3 -m fugue plot --from-paper --experiments 4,5`。

[完整复现指南](../../../docs/Fugue-asplos-reproduction.md) · [模型范围](../../../docs/Fugue-asplos-methodology.md) · [固定输入与来源](../../inputs/README.md) · [当前来源记录](provenance/Fugue-asplos-current.json)。

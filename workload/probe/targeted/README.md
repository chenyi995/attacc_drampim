# 让布局的扫描省时进入 TBT：持续汇总与周期共读 workload

chenyi9 要表达的是**同一个汇总 agent 每一步继续汇总**。它保留旧上下文及旧 diff，再追加本步内容；多步之后，一次 decode 会读到此前各轮仍有效的修正。本目录已经生成对应 JSON，并用真实 reuse plan、物理账本及独立公式核对地址和工作量。

先读 [手算结果](HANDCALC.md)：包含已有 S11 实测为何被稀释、新 workload 的逐通道数据、GPU 时间预算和 PIM/GPU 分界条件。新输入只完成了静态验证，尚无这些输入的 Ramulator、TBT 或 E2E 实测，不能把占行降幅直接写成性能降幅。

## 1. 先区分每一档要改善什么

| 对比 | 应重点报告 | 输入必须满足什么 |
|---|---|---|
| A3b→A4c | 相同请求的 decode scan、TBT；输出较长时也看 E2E | 多轮小 diff 被同一汇总链持续读取，减少行切换发生在忙通道上 |
| A4c→A4e | 共读 scan、TBT；同时报告 A3b→A4e | **大量实际读取的文档**在朴素布局中撞通道，真实软件表确实将它们分开 |
| A4e→A5 | 后续复用轮的 TTFT、整段 E2E；decode MQ 单列 | 驻留上下文长，本轮实际 query 少；多轮低 query 请求摊薄初次建立上下文的代价 |
| A5→A6 | 各类请求 TTFT、E2E、逐请求选边 | 同一输入中同时有适合 PIM 的短复用与适合 GPU 的大 prefill |

A5 同时引入 MQ/已接受的 PE 配置，因此它也可能改善 TBT。A6 与 A5 的 decode 机制相同，新增的直接收益来自 prefill 选边；全局 TBT 仍可能因资源竞争变化而变化。

最新 commit `958dd24` 是 S11 结果记录；调度回填和 head 流水已在 `ff6f225`。本目录按 `958dd24` 源码核验，并把工作区尚未提交的 reader-load 表作为单独一列。不能把新表的效果算作旧 commit 的结果，也不能只给 A4e 换一个代码版本运行。

## 2. D：每一步都有汇总的持续链

主输入为 `inputs/D_rolling_r16.json`、`r32`、`r64`、`r128`。每个输入有一个共享文档 owner、一个持续汇总者、两个独立的后台 worker。

每轮汇总者的内容是：

```text
原来的完整前缀 | 上一轮输出 | 本轮复用的文档 | 本轮指令
                   128            256             16 token
```

首轮前缀是 16 token。每轮文档来自较早的 owner，在汇总上下文中的位置不同，所以 k8 产生八个新的 diff。后续 request 的 parent 指向前轮，旧段的 fingerprint、顺序和绝对偏移保持原样。因此，第 R 轮读到 R 份有效 diff，其中只有本轮那一份新算；没有把同一文档已失效的多个版本累计进去。

**闭式手算：** 第 R 轮 prompt 为 `16 + R×(256+16) + (R−1)×128 = 400R−112`；累计 diff 为 `8R`，继承 diff 为 `8(R−1)`。首轮算 `16+8+16` 个 token，后续算 `8+16`，不是重新计算整个 prompt。

两个后台 worker 每轮分别写两段/一段各 16 token 的独立内容，均输出八个 token。它们表达穿插的其他请求流量；汇总者不读取它们的私有内容。三条链在同一轮的预约流中，每次新增对象数为：

```text
worker A：上一轮输出 + 两段新内容 = 3 个 master 对象
worker B：上一轮输出 + 一段新内容 = 2 个 master 对象
汇总者：上一轮输出 + 一段新指令 + 一个 diff burst = 3 个对象
合计 8，恰好是每 head 的通道周期。
```

所以这组受控输入的跨轮 diff 在 A3b 中持续落到同一通道的不同行。A4c 的独立 diff 区将它们集中。普通 master 的写入内容和通道在 A3b/A4c 两档一致，没有为 A4c 减掉普通扫描。这个固定周期是为了暴露机制，不能宣称代表任意到达顺序的平均收益。

同轮连续产生的 diff 仍按 A3b 原规则合并；本例依靠真正的 parent 多轮，不能改成一个 request 中罗列很多文档。当前分配器没有实现短输出把后继对象推到半列的机制，本例不依赖它。

物理紧凑也不等于列命令减少：当前 dhead128 生成器对每个八-token extent 发两次 QK 列请求，起点在末列时还会触及下一行。手算表分别报告物理占行、QK 地址触及行和命令数，未把唯一行数称为实测激活数。

### 把它放到能体现 TBT 的区间

`D_rolling_r64_sessions7.json` 是七份上述独立工作流共享一块设备，每份每步都有自己的汇总者。`BATCH=8` 是上限；后台 worker 的短回答结束后，实际七个汇总请求仍同时 decode。它们的 diff 都压在朴素布局的忙通道上，A4c 能减少这个通道的跨轮行切换，同时普通 master 分布较均衡。更长的 R 增加真实累计扫描量，较长输出让有累积 diff 的轮次在 TBT/E2E 中有权重。

仍须保留两个反例，不能只报最有利输入：

- `D_rolling_r64_background_plus1.json`：只给后台 A 每轮增加一小段，写入周期改变，A3b 的 diff 自然散到多个通道。A4c 不保证同样获益。
- `D_rolling_r64_sessions8.json`：八条链会产生普通历史 master 的周期热点；即使 diff 行数下降，另一条通道也可能决定 scan 完成。这是几何上的遮蔽条件，是否真的遮住时延需实测。

这些 probe 不是预测 A3b 被削弱；所有档读同一输入、相同逻辑 KV，背景写入也是共同的实际工作。七份链的主候选与八份链的反例均在测性能之前由地址手算发现并保留。

## 3. E：让每一步的大部分扫描都命中共读冲突

`E_coread_stride1/4/8.json` 固定全库 512 个 256-token chunk、每次读取其中 64 个，八个汇总请求共同读取同一组文档，连续两轮，每轮输出 128 token。导入按每个 owner 八个 chunk 分组，owner 没有额外前置 block；选择下标从零计数。

| 输入 | 共读文档下标 | 八通道下朴素分布 |
|---|---|---|
| stride1 | `0,1,2,...,63` | 八个通道均衡，负对照 |
| stride4 | `0,4,8,...,252` | ch0/ch4，各承担一半 |
| stride8 | `0,8,16,...,504` | 全在 ch0 |

这就是“每隔通道数篇文档取一篇一起读”。若实际是四个通道，周期四才回到同一通道；本协议 LLAMA3-8B 是每 head 八个通道，所以周期四会落到两个通道。这里冲突的是**同一通道上的不同行**，不是两个对象共用同一个物理 column 地址。

在该分组导入方式下，HEAD 的实际表已经核到：stride8 的共读文档由最忙通道承担全部 64 行，变成每通道八行。总数据和总列请求保持一致；逐 channel 的实际表在手算结果中。共读的是大部分驻留内容，因此该例直接改变大头扫描，避免 S11 只给无法优化的简报加长度的问题。

`E_coread_stride8_monolithic_owner.json` 保留整库一次导入的反例。HEAD 的 partners 表把整库都视为彼此共读，对初始 corpus 退化成轮转；不能打散热点。工作区的 reader-load 表会得到另一结果，已分版本列出。两种导入方式各自做档内同输入比较，不能将它们之间的 E2E 差异归因于布局，因为导入请求数和 query 形状也不同。

## 4. P：用低 query 的持续复用展示 A5，用混合请求展示 A6

`P_reuse_q4.json` 使用 64 个共享文档、八个请求链、32 轮；每个 owner 导入八个文档，每轮只输出八个 token。首轮建立上下文，此后原样保留它并追加上一轮输出和四-token 新指令。**k 仍然是八**；q4 指后续本轮实际只计算四个新 token，不是把 k 偷换成四。

后续旧文档及原有 diff 都继承，新增 diff 为零。LLAMA3 的 GQA=4、512 B buffer 能放八个物理 query slices，因此一个 prefill sweep 最多两个 request queries，m4 要两次 sweep；TINY 的 MHA 是一次。首轮的 query 数大得多，已另列，不能把它也称作 q4。

每层 GPU 价格按实际代码计算：`驻留KV回读 + FlashAttention(m, R+m)`。PIM 必须满足 `所有真实sweep的max(channel time)之和 + context返回 < GPU价格`；手算表直接给每次 sweep 的平均允许时间。Q 及小 context 传输遵守 chenyi9 原有计价，不加未校准固定费用。PIM 价格留给 Ramulator，未用拟合价格决定选边。

32 轮是为了让后续大量低-query 请求有机会摊薄初次导入与首轮修正的开销；整库只保留此例实际需要的文档。仍须完整报告含导入的 E2E，再单列首轮/后续 TTFT。局部低-query prefill 赢，不足以证明包含冷启动的 A5 总 E2E 一定赢。

`P_mixed_q4_fresh.json` 在同样复用链中加入新的 2k、4k、8k 提示，各输出八个 token。A5 的 prefill attention 固定选 PIM，A6 按每个 request 的两侧价格选较快者；记录每个请求的 m、R、sweeps、两侧价格和 side。所有短轮若本来就走 PIM，A5/A6 不应凭空分开；拉开差距的是避免把不适合 PIM 的 prefill attention 送进去。

## 5. 为什么 scan 降 15%，TBT 可能只降一点

只对**同一批请求、同一组步、同一统计边界**讨论传递。示意模型 `T=G+S` 中，G 为其余不可隐藏时间，S 为可被布局优化且暴露在关键路径上的扫描：

```text
扫描降幅 g；扫描占比 f = S/(G+S)
TBT 降幅约 g×f
g=15% 时：f=10%/30%/50% → TBT 约降 1.5%/4.5%/7.5%
想达到5% TBT，要求 S≥0.5G；想达到10%，要求 S≥2G。
```

这是串行预算，不是调度器的精确公式。已有 scan_step 是所有 scan 的首尾跨度，含排队，也可能与 GPU 重叠；不能无条件把它乘层数当作不可隐藏的 S，更不能拿 scan_private 的降幅与另一批请求的总体 TBT 相除。

GPU 原有线性层和 norm/activation 共同成本继续沿用 AttAcc。Flash 和 pipeline 均开启。手算文件给出当前 GPU 模型的每层工作量作为量级参考，既不删除这些共同项，也不靠修改 GPU 速度制造收益。增加 BATCH 只有实际有更多就绪请求时才可能摊薄 GPU 工作；S11 只有几个 worker，改上限不会变成大 batch。

## 6. 复现静态检查

从仓库根目录执行，生成输入与手算，不启动性能模拟：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 workload/probe/targeted/build.py
PYTHONDONTWRITEBYTECODE=1 python3 workload/probe/targeted/handcheck.py
PYTHONDONTWRITEBYTECODE=1 python3 workload/probe/targeted/summarize_checks.py
```

`checks/` 保留逐请求三档 extents、diff 原始 writer、m/R、逐通道行/列计数、源码快照和 hash。`summarize_checks.py` 再用闭式公式核对 D、E，用真实 trace generator 的 QK 地址核对手算列地址；这些都是静态检查，不是 Ramulator 时间。

几何结果明确为 layer0、最忙 HBM 的一个八通道 head stripe，第二个 head 在另八条通道重复。完整运行的每层服务时间仍由 Ramulator 决定；其他层的追加偏移、行开关状态、PV、Q/结果传输、排队均不能由此处行数表替代。

## 7. 性能运行命令与出数

输入、生成器和手算已就绪。运行时固定同一代码版本、同一输入、相同 k/硬件；默认已有 NVLink、Flash、pipeline。三类分别比较 `A3b A4c`、`A3b A4c A4e`、`A4e A5 A6`。以下示例按档串行运行，输出到新目录：

```bash
export PYTHONDONTWRITEBYTECODE=1
export ATTACC_RAMULATOR_DIR=/data2/chenyi9/KV-PIM/scratch_0905
export PYTHONPATH="$PWD" KVPIM_CPPCORE=1
FUGUE_WL=workload/probe/targeted/inputs/D_rolling_r64_sessions7.json
FUGUE_OUT=/data2/chenyi9/KV-PIM/scratch_0905/targeted_D64_sessions7_LLAMA3
test ! -e "$FUGUE_OUT" || exit 1
mkdir -p "$FUGUE_OUT"
for FUGUE_RUNG in A3b A4c; do
  ATTACC_RAMULATOR_LOG="$FUGUE_OUT/ramulator_${FUGUE_RUNG}.out" \
  KVPIM_PREFILL_SIDE_LOG="$FUGUE_OUT/sides_${FUGUE_RUNG}.jsonl" \
  python3 main.py --system dgx-attacc --gpu A100a --model LLAMA3-8B \
    --ngpu 1 --num-hbm 5 --pim-link nvlink3 --word 2 --powerlimit \
    --engine dag --pipeopt --gpu-model flash --reuse recompute \
    --epic-prefix-recompute-tokens 8 --cacheblend-batch-size 8 \
    --ramulator-workers 7 --ablation "$FUGUE_RUNG" --workload "$FUGUE_WL" \
    --workload-report "$FUGUE_OUT/dag_${FUGUE_RUNG}.json" \
    --workload-report-events none > "$FUGUE_OUT/dag_${FUGUE_RUNG}.log" 2>&1 || exit 1
done
python3 experiments/summarize_ladder.py "$FUGUE_OUT" "$FUGUE_WL" A3b
python3 workload/probe/targeted/cohort_report.py "$FUGUE_OUT" "$FUGUE_WL" \
  --output "$FUGUE_OUT/cohorts.csv"
```

E 例替换输入为 `E_coread_stride8.json`，档位为 `A3b A4c A4e`；P 例替换为两个 `P_*.json`，档位为 `A4e A5 A6`、参考档 A4e。每个输入另用输出目录；不同档的源码、workload hash、corrected_rows_sha 必须核对。

`cohort_report.py` 输出完整 workload、全部汇总轮、最后四分之一汇总轮和最后一轮的加权 TBT/TTFT，并在所有行保留完整 E2E。这样既能看累积轮次的机制，也不隐去准备期和短 worker。总体 scan 仍从标准 summary 读取；它不是每个 cohort 的 scan，不能混成同一条降幅。

解释关键路径时，用较小的 D16 与 `--workload-report-events full` 留事件，按同一 request、layer、decode position 比较各 lane、GPU readiness 和 token completion。长例默认 compact，避免不必要的巨量地址事件。所有正式图同时保留绝对时间、相邻档降幅、实际 batch 人数与选边结果。

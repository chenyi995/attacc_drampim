# 实验指导：指标定义、full events 与八通道

**跑什么、怎么跑以 [运行协议](../README_run_protocol.md) 为准**（model 几何表、W1 baseline 与 W1_S* sweep、`run_sweep.sh`、`extract_protocol.py`）。
本页保留四项指标的定义与取数、`--workload-report-events full` 的用法和 A6 side log 的读法。机制解释见 [性能分析](../analysis/README.md)，
实现核查见 [审计入口](../audit/README.md)。本页 2026-09-05 整理时没有启动性能任务。

针对最新 S11 中“private scan 改善、TBT 改善较小”的问题，新增 [持续汇总、周期共读与低-query复用实验](../../workload/probe/targeted/README.md)。输入 JSON、闭式手算、真实账本核验、PIM/GPU 分界预算及运行/分组出表命令均已提供；其新输入尚未进行性能模拟。

## 1. 固定条件

| 项目 | 当前口径 |
|---|---|
| 模型/硬件 | 按协议表：每 GPU 5 个 HBM 栈（`--num-hbm = 5 × --ngpu`）；LLAMA3-8B `--ngpu 1 --num-hbm 5`，8 个 KV head，最忙栈 2 个，每 head 8 个 channel（`--num-hbm 4` 同几何）；TINY 只证明流程能跑通 |
| GPU attention | 显式 `--gpu-model flash`；直接 main.py 不读取 GPU_MODEL 环境变量作为该选项 |
| 执行/精度 | `--engine dag --pipeopt --powerlimit --word 2 --pim-link nvlink3`，各档相同 |
| 复用 | A2–A6 同一点统一 `--reuse recompute`、协议用 k8（k4 作敏感性）；A1 用 no-reuse，不传 k |
| 批量 | batch=8；512 B buffer 放 8 个物理 Q slice，LLAMA3 GQA=4，所以每个 prefill sweep 最多 2 个 request queries，不能说是 8 个 |
| 档位差异 | 只用 ablation preset；A5/A6 的 MQ、PE 频率与 buffer 按已接受的配置，不改成其他档的设置 |
| 链路/能量 | 沿用 chenyi9 已定 AttAcc 模型、Q 估价忽略与 decode 固定开销口径；无附加 DIE/TLB 费用 |
| 产物 | 完整指标保留 `--workload-report-events full`；模型/workload/k/硬件各用独立目录 |

八通道是所有档共用的硬件配置，A3b/A4c/A4e 拥有相同数量的 master 通道；不能用八通道 Fugue 对旧两通道 baseline。每个 k 点内部核对 A2–A6 的 corrected-row hash；k4 与 k8 之间 hash 变化是预期。

## 2. 选择输入与档位

真正多轮使用 `*_turns.json`，每轮有自己的 prefill/decode 和 parent 输出。`*_interleaved.json` 是多组内容拼成单次 prompt，不能代替多轮依赖。

| 要回答什么 | 已有输入/控制 | 比较档位 |
|---|---|---|
| 完整阶梯 | `W1_turns.json`（协议 baseline） | 七档 |
| diff 聚合 | W1、`W1_S1_rounds_*`（轮数）、`W1_S2_workers_*`（每 main 的 worker 数） | A3b、A4c |
| 软件表 | W1、`W1_S4_worker_lout_*`（共读块大小）、`W1_S3_sessions_1`（单会话，无跨会话共读的负对照） | A4c、A4e，可带 A3b |
| MQ 与选边 | `W1_S3_sessions_*`（batch 内共读）、`W1_S6_doc_tokens_*`（导入与驻留上下文长度） | A4e、A5、A6 |
| 输出长度 | `W1_S4_worker_lout_*`、`W1_S5_main_lout_*` | 归因布局需 A3b/A4c/A4e，归因选边需 A4e/A5/A6 |

本页最初引用的 B1/T1–T9 与 C1/C2 文件已归档到 `workload/probe/archive/2026-09-05-C-protocol/`。

文件均位于 `workload/probe/sweep/`。T5 所有 worker 话少仍包含大语料 owner，不等于所有请求都短。按 owner、首轮、后续短/长请求分别解释选边。

八通道集中访问的受控定义：写入序轮转保持原样，共读块取 `0,1,8,9,16,17,24,25`，自然集中 ch0/ch1。当前 JSON 没有专门命名的这一轴，B1 随机散取不能直接贴这个标签。各档使用同一输入和生成/消费依赖，先看实际软件表是否分散，再报收益；整库导入的完整共读关系可能让当前贪心表无法分散热点。

短输出的半列偏移控制：可用 lout=3、4、5、7、8、9 等长度，固定同 channel 后继 diff 的大小和读集。但当前 A3b 按行分配，改变 lout 仍不会把输出尾部接给下一个 diff。须先核真实地址与列命令；详见 [部分列连续追加审计](../../audit/2026-09-05/PARTIAL_COLUMN_APPEND_AUDIT.md)。这些是受控输入的定义，不能把已有 T9 当成已经覆盖该机制。

## 3. 环境

Ramulator 二进制与 trace generator 的准备见 [squire](../run/README_run_squire.md) 或 [athena](../run/README_run_athena.md)。机器页旧实验参数不覆盖本页。

```bash
cd /data2/chenyi9/KV-PIM/attacc_drampim_822
export KVPIM_SCRATCH=/data2/chenyi9/KV-PIM/scratch_0905
export ATTACC_RAMULATOR_DIR="$KVPIM_SCRATCH"
export ATTACC_RAMULATOR_LOG="$KVPIM_SCRATCH/ramulator.out"
export PYTHONPATH="$PWD" KVPIM_CPPCORE=1
test -x "$ATTACC_RAMULATOR_DIR/ramulator2"
test -d "$ATTACC_RAMULATOR_DIR/trace_gen"
```

## 4. 保存完整指标：带 full events 的七档（例：C1，k4 与 k8）

协议默认 `EVENTS=none`：`summary.decode_scans` 已给每次 scan 的 service/elapsed 与每步跨度，`extract_protocol.py` 直接出表。
只有要离线核对单次 scan 的地址和依赖时才用 full events；TINY 一档的 full 报告 4.6 GB，LLAMA3-8B 会大一个量级，一次只跑一档。

以下按档顺序执行，避免多档完整地址事件同时驻留；7 workers 加一个构图进程的并行度是针对当前这一档。机器资源管理仍按机器页。

```bash
FUGUE_MODEL=LLAMA3-8B
FUGUE_WL=workload/probe/sweep/W1_turns.json
FUGUE_STAMP=$(date +%Y%m%d-%H%M%S)
FUGUE_OUTROOT="$KVPIM_SCRATCH/B1_8ch_${FUGUE_MODEL}_${FUGUE_STAMP}"
FUGUE_RUNGS="A1 A2 A3b A4c A4e A5 A6"
FUGUE_REF=A3b

for FUGUE_K in 4 8; do
    FUGUE_OUT="$FUGUE_OUTROOT/k${FUGUE_K}"
    mkdir -p "$FUGUE_OUT"
    for FUGUE_RUNG in $FUGUE_RUNGS; do
        FUGUE_REUSE=(--reuse recompute --epic-prefix-recompute-tokens "$FUGUE_K")
        if [ "$FUGUE_RUNG" = A1 ]; then
            FUGUE_REUSE=(--reuse no-reuse)
        fi
        KVPIM_PREFILL_SIDE_LOG="$FUGUE_OUT/sides_${FUGUE_RUNG}.jsonl" \
        python3 main.py \
            --system dgx-attacc --gpu A100a --model "$FUGUE_MODEL" \
            --workload "$FUGUE_WL" "${FUGUE_REUSE[@]}" \
            --ablation "$FUGUE_RUNG" --engine dag \
            --gpu-model flash --pipeopt --powerlimit --word 2 \
            --pim-link nvlink3 --num-hbm 5 --ngpu 1 \
            --cacheblend-batch-size 8 --ramulator-workers 7 \
            --workload-report "$FUGUE_OUT/dag_${FUGUE_RUNG}.json" \
            --workload-report-events full \
            > "$FUGUE_OUT/dag_${FUGUE_RUNG}.log" 2>&1 || exit 1
    done
    python3 experiments/collect_dag_ladder.py "$FUGUE_OUT" "$FUGUE_WL" "$FUGUE_MODEL"
    python3 experiments/summarize_ladder.py "$FUGUE_OUT" "$FUGUE_WL" "$FUGUE_REF"
done
```

只研究选边，FUGUE_RUNGS 设为 `A4e A5 A6` 且 FUGUE_REF=A4e；只看布局设为 `A3b A4c A4e`、FUGUE_REF=A3b。参考档必须在输出中，否则 summarizer 会省略相对结果表。换输入同时换输出目录；k4/k8 使用同一 workload。A1 与 k 无关，复用其产物时仍须同输入和硬件，不能复制别的配置。

A6 的 side log 记录真实 m、R、scan rows、sweeps、t_xpu_s、t_bank_s 与 side。它才回答 k4 是否跨过阈值；旧手工校准的 28/37 不能替代实际决策。

## 5. 现有 ladder/sweep 入口

这两个入口默认 `--workload-report-events none`（`EVENTS=full` 可改）；summary 里已有 scan latency 统计。参数可设成八通道/k4：

```bash
GPU_MODEL=flash EPIC_K=4 NUM_HBM=5 NGPU=1 RAMU_WORKERS=7 \
RUNGS="A4e A5 A6" \
KVPIM_PREFILL_SIDE_LOG="$KVPIM_SCRATCH/W1_8ch_k4.sides.jsonl" \
bash experiments/run_dag_ladder.sh workload/probe/sweep/W1_turns.json \
    LLAMA3-8B "$KVPIM_SCRATCH/W1_8ch_k4_summary"

GPU_MODEL=flash EPIC_K=4 NUM_HBM=5 NGPU=1 \
RUNGS="A4e A5 A6" PARALLEL=1 RAMU_WORKERS=7 \
bash experiments/run_sweep.sh "$KVPIM_SCRATCH/short_prefill_8ch_k4" \
    '^(W1_turns|W1_S3_sessions_.*_turns)[.]json$' LLAMA3-8B
```

`run_sweep.sh` 把 `C[0-9]+_turns.json` 视为 baseline（七档），其余点只跑 A3b/A6；要归因 A5/A6 显式给 RUNGS。这些 summary 入口与上一节 full 入口的产物粒度不同，不能把 lane-sum 改名为 scan latency。

## 6. 指标取数

| 指标 | 主表定义 | 当前可得信息 |
|---|---|---|
| Decode scan latency | 同一次实际 scan 的 max(channel time_s)；按匹配的 request、step、layer 汇总，必要阶段保留依赖 | full 事件有时长、query positions、依赖和 batch 身份；现有 collector 尚无完整该列 |
| TBT | Σ(end−first_token)/Σ(lout−1)，排除 lout≤1 | summarize 已有加权与请求均值 |
| TTFT | first_token_completed−request_release，包含排队 | first_token_s 已有；多轮 release 需统一提交/业务依赖定义，不能用全局时间戳或实际开始执行时间替代 |
| E2E | 整份 workload makespan | report、collector 已有 |

各项给绝对值和相邻档 `1−new/old` 降幅；A2 无 PIM scan 标 N/A。数据无法恢复时留缺失，不能填零。MQ shared/private 的多个 scan 不能无条件相加或取一次 max，也不能将共享服务时间除 query 数冒充一个请求的延迟；按 full 事件的身份与依赖处理。

最长通道按真实耗时选，不按 token 数选。pim_pool_time_s_unoverlapped 是 lane 总服务工作量；最后完成减最早开始还可能包含排队，应另列 elapsed。当前 scan event 包含 QK、softmax、PV，列名和解释保持一致。

KVPIM_LAYOUT_DUMP 可核地址、per-channel 时间和 scan_time_s=max；默认 layer0/前400条。当前 warm 会取真实价格，但 dump 缺唯一 sweep 身份，完整归因用最终 full events。保留输入、完整运行参数和源码版本，区分旧两通道与新八通道结果。详细定义与旧 TTFT 见 [指标专项](../../audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md)。

## 7. 已确定的 NVLink 配置

chenyi9 裁决：保持之前的 NVLink 连接 GPU，沿用本指南命令中的 `--pim-link nvlink3`。各档共用相同链路配置和已有计价规则。

两级内存、context 阈值切换 PCIe 及为拉开 A5/A6 而降低链路带宽的提议不采用。A5/A6 按现有 NVLink 下的逐请求价格和实际指标比较。裁决与原审查来源见 [链路记录](../../audit/2026-09-05/LINK_TIER_ASSUMPTIONS.md)。

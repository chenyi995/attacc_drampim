# 运行协议：模型 × workload × combo（chenyi9 裁决 2026-09-05）

本页是跑实验的唯一入口。机器细节（编译器、scratch、监视器）见 `run/README_run_squire.md` 与 `run/README_run_athena.md`；
指标定义见 `../audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md`；改动理由见 `sessions/`。

## 0. 用词

| 词 | 指什么 | 例子 |
|---|---|---|
| **model** | 被服务的 LLM，决定层数、头数、GQA 与 GPU/HBM 几何 | `LLAMA3-8B`、`LLAMA-65B`；`CACHEBLEND-TINY` 只用来证明流程能跑通 |
| **workload** | 拓扑 JSON：多 agent 多轮的请求依赖图，谁读谁写、每轮多长 | `workload/probe/sweep/C1_turns.json` |
| **combo** | 阶梯上的一档，机制的组合 | `A1 A2 A3b A4c A4e A5 A6`（`--ablation`） |

固定开关：`--engine dag --pipeopt --gpu-model flash --powerlimit --word 2 --pim-link nvlink3`，k=8（`--epic-prefix-recompute-tokens 8`），
batch 8。所有 combo 同一 model、同一 workload、同一几何、同一修正计划（报告里 `corrected_rows_sha` 必须一致）。

## 1. 每个 model 的 GPU / HBM 几何

硬件按 AttAcc 发表的 DGX-A100：一个 GPU 后面 5 个 HBM3-PIM 栈，`--num-hbm = 5 × --ngpu`。`--ngpu` 是张量并行度，
按 model 大小取；最忙的栈放 ceil(本地 KV 头 / 5) 个头，一个头条带化到 16 // 该数 个通道，这就是布局能用的通道数。

| model | 层 | KV 头（GQA） | `--ngpu` | `--num-hbm` | 最忙栈的头数 | 每头通道数 |
|---|---:|---:|---:|---:|---:|---:|
| CACHEBLEND-TINY | 4 | 8 | 1 | 5 | 2 | 8 |
| LLAMA3-8B | 32 | 8（32/4） | 1 | 5 | 2 | 8 |
| LLAMA-7B | 32 | 32 | 1 | 5 | 7 | 2 |
| GPT-13B | 40 | 40 | 2 | 10 | 4 | 4 |
| LLAMA-33B | 60 | 52 | 4 | 20 | 3 | 5 |
| LLAMA-65B | 80 | 64 | 8 | 40 | 2 | 8 |
| GPT-175B | 96 | 96 | 8 | 40 | 3 | 5 |

`experiments/run_sweep.sh` 按这张表自动设 `NGPU`/`NUM_HBM`；显式给环境变量可覆盖。只模拟最忙的栈，所以 LLAMA3-8B 用
`--num-hbm 4` 与 `5` 结果相同（都是 2 头一栈）。

## 2. 协议

- **baseline workload** 跑全部七个 combo：`C1_turns.json`、`C2_turns.json`。
- **sweep workload** 只跑 `A3b` 与 `A6`：`C1_S*_turns.json`，每个文件只动 C1 的一个变量。
- 汇总以 `A3b` 为参考档（`summarize_ladder.py <outdir> <wl> A3b`）；baseline 另看相邻档。

### 2.1 两个 baseline

`workload/probe/gen_sweep.py` 的 `C1` / `C2` 预设，生成：`python3 workload/probe/gen_sweep.py --all workload/probe/sweep`。

| | C1（混合型 agent 会话） | C2（纯对话型） |
|---|---|---|
| 语料 owner | 256 系统提示 + 64 chunk（16640 token）的一次导入 | 16 chunk（4352 token） |
| worker | 6 个 × 6 轮；一半话少（每轮写 16 token、答 32）、一半写手（写 256、答 128） | 8 个 × 8 轮，全部话少 |
| 每轮读什么 | 共享系统提示 + 共享 8 chunk 简报 + 散取 1 个 chunk + 自己的历史（含上一轮输出） | 同左 |
| 额外 | 4 个独立新鲜提示（2k/4k/8k/2k，无复用） | 无 |
| 请求数 | 41 | 65 |

C1 让每一档都有自己的杠杆（TINY 上七档逐档分开，见 §4）：修正小且多轮反复读（A4c），散取共读撞通道（A4e），
batch 共读简报（A5/A6 的 MQ），16k 导入和新鲜提示该留 GPU、话少轮该进 PIM（A6 对 A5）。
C2 把 A5 也放到赢的一侧：全是 decode 形状的轮次，PIM 侧 prefill 直接赢，A6 与之持平，用来说明 A6 不是靠"永远选 GPU"。

### 2.2 sweep 轴（围绕 C1，各跑 A3b + A6）

| 轴 | 文件 | 值（C1 的值不重复） | 动的是哪根杠杆 |
|---|---|---|---|
| S1 agent 数 | `C1_S1_agents_*` | 4, 12, 16 | 共读简报的 batch 大小（MQ）、表的冲突 |
| S2 轮数 | `C1_S2_rounds_*` | 3, 12 | 修正累积、驻留上下文 |
| S3 话少占比 | `C1_S3_chatty_share_*` | 0, 0.25, 0.75, 1.0 | 选边器送 PIM 的份额 |
| S4 共享简报 | `C1_S4_shared_*` | 0, 16, 32 chunk | 一步里扫描的占比、MQ 可共享的行 |
| S5 话少的回答长度 | `C1_S5_lout_chatty_*` | 8, 128 | 输出交错、decode 占比 |
| S6 写手的回答长度 | `C1_S6_lout_*` | 32, 512 | decode 在 E2E 里的份额 |
| S7 每轮 chunk 数 | `C1_S7_chunks_*` | 2, 4 | 每轮修正量、共读冲突 |
| S8 新鲜提示占比 | `C1_S8_fresh_share_*` | 0, 0.25 | A6 留在 GPU 的份额 |
| S9 语料大小 | `C1_S9_corpus_*` | 32, 128 chunk | 导入长度、散取的分散度 |
| S10 检索方式 | `C1_S10_retrieval_consecutive` | 连续 | 表的负对照（连续块本来就均衡） |

结构探针（不跑 Ramulator）：`LEVERS_HEADS_PER_HBM=2 python3 output/analysis/b1_levers.py workload/probe/sweep/C1_turns.json`，
给出 A4c 修正行、A4e 最忙 lane、最忙 lane 相对平均 lane 的余量。

旧的 B0 / S1–S6 / B1 / T1–T9 集合不再默认生成，`LEGACY_MATRIX=1 python3 workload/probe/gen_sweep.py --all <dir>` 可复现。

## 3. 怎么跑（squire）

```bash
cd /data2/chenyi9/KV-PIM/attacc_drampim_822
export KVPIM_SCRATCH=/data2/chenyi9/KV-PIM/scratch_0905        # Ramulator 二进制、trace_gen 软链、签名缓存都在这里
setsid nohup experiments/mem_guard.sh $KVPIM_SCRATCH/guard.log > /dev/null 2>&1 < /dev/null &   # 先起监视器

# 一个 baseline，七个 combo 并行（6 个 PIM combo × 8 worker + 7 = 55 核）
setsid nohup bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_LLAMA3-8B '^C1_turns' LLAMA3-8B \
    > $KVPIM_SCRATCH/proto_LLAMA3-8B.out 2>&1 < /dev/null &

# 一条 sweep 轴，每点 A3b + A6，三点并行（3 × (2 × 8 + 2) = 54 核）
setsid nohup bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_LLAMA3-8B 'C1_S3_' LLAMA3-8B \
    > $KVPIM_SCRATCH/proto_LLAMA3-8B_S3.out 2>&1 < /dev/null &

# 整个协议（两个 baseline + 23 个 sweep 点）按 manifest 顺序串行分批
setsid nohup bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_LLAMA3-8B '.' LLAMA3-8B \
    > $KVPIM_SCRATCH/proto_LLAMA3-8B_all.out 2>&1 < /dev/null &
```

规则：没有明确指令不跑（`agent.md` §1.7）；≤ 64 核、≤ 500 GB；同一时刻只跑一条 `run_sweep.sh`。
大 model（LLAMA-65B、GPT-175B）一次只跑一档：`RUNGS=A5 PARALLEL=1 RAMU_WORKERS=16`，内存按 `run/README_run_squire.md` 的表。
单档手跑与 athena 的 sbatch 写法见机器页。

看进度：`tail $KVPIM_SCRATCH/proto_LLAMA3-8B/sweep.log`、`grep -h "done\|FAILED" $KVPIM_SCRATCH/proto_LLAMA3-8B/C1_turns.log`、
`tail -1 $KVPIM_SCRATCH/guard.log`。停：按 PID 杀（机器页 §3），别用会匹配到自己 shell 的 `pkill -f`。

## 4. 出数

每个点的目录里：`dag_<combo>.json`（报告，含 `run_config`：git 版本、gpu_model、ngpu、num_hbm、k、batch、workload sha256）、
`dag_ladder.csv`（collector）、`summary.md`（`summarize_ladder.py`，参考档 A3b）、`<点>.sides.jsonl`（A6 每个请求的 t_xpu / t_bank / side）。

整个 outroot 一次出表：`python3 experiments/extract_protocol.py <outroot> --ref A3b` 写 `protocol.csv`（每 (workload, combo) 一行）和
`protocol.md`（baseline 全表与相对表、每条 sweep 轴的 A6 对 A3b 比值、A3b–A6 的修正计划 sha 是否一致）。要 full events 时见
`experiments/README.md` §4。

`summary.md` 的列：E2E（makespan）、TTFT（首 token 完成 − 请求 release，含排队）、TBT 均值 / 按 step 加权（论文用加权）/ 最大、
scan_private / scan_shared（一次 decode scan 最慢 lane 的服务时长，共读与私有分开）、scan_step（一个请求一层一步内所有 scan 的
首尾跨度）、能量与平均功率、prefill 行 PIM/GPU、代码版本。相对表是 参考档 / 本档，大于 1 为变好。

**TINY 上的流程验证（C1，2026-09-05，4 HBM）**，只证明七档能跑通且逐档分开，不是性能结论：
A3b→A4c E2E 1.003 / 扫描每步 1.042；A4c→A4e 1.012 / 1.111；A4e→A5 0.901 / 1.382（导入进 PIM 输 tier 0）；
A5→A6 1.210 / 1.000（选边）；A6 对 A3b E2E 1.106、TTFT 1.239、TBT 1.032。
数据在 `scratch_0905/out_C1v2_CACHEBLEND-TINY_hbm4_k8/summary_ref_A3b.md`，不进仓库。

数字只能由脚本复制和计算（`agent.md` §3）；结果目录不进仓库，汇总表进 `output/analysis/`。

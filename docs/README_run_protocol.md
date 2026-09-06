# 运行协议：模型 × workload × combo（chenyi9 裁决 2026-09-05）

本页是跑实验的唯一入口。机器细节（编译器、scratch、监视器）见 `run/README_run_squire.md` 与 `run/README_run_athena.md`；
指标定义见 `../audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md`；改动理由见 `sessions/`。

另一会话 9-05 的设计研究 [持续汇总与周期共读 workload](../workload/probe/targeted/README.md)（D/E/P 输入与手算）不在 W1 协议里，作机制参考。

## 0. 用词

| 词 | 指什么 | 例子 |
|---|---|---|
| **model** | 被服务的 LLM，决定层数、头数、GQA 与 GPU/HBM 几何 | `LLAMA3-8B`、`LLAMA-65B`；`CACHEBLEND-TINY` 只用来证明流程能跑通 |
| **workload** | 拓扑 JSON：多 agent 多轮的请求依赖图，谁读谁写、每轮多长 | `workload/probe/sweep/W1_turns.json` |
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

- **baseline workload `W1`**（`workload/probe/sweep/W1_turns.json`）跑全部七个 combo。
- **sweep** `W1_S*_turns.json` 各动 W1 的一个变量，只跑 `A3b` 与 `A6`。
- 汇总以 `A3b` 为参考档；baseline 另看相邻档。旧的 C1/C2 协议已退役，见 `workload/probe/archive/`。

### 2.1 W1：main agent 每轮总结它的 worker（chenyi9 2026-09-06）

一个会话 = 一个 main agent + 2 个 worker，两个会话并排。语料 owner 一次导入 48 篇 256-token 文档；每轮每个 worker 读一篇**新**文档
（从 owner 复用、偏移不同，k=8 个修正行）、写 16 token 笔记、答 128 token；main 每轮总结 worker 两轮前的回答（2 个回答块作为复用段
进入 main 的上下文，每轮新增 2×8 个修正，旧修正逐轮继承）、答 128 token；两个会话的同号 worker 每轮读同一篇文档。24 轮，145 个请求，
main 末轮上下文 9k。每一档的杠杆和 r−2 的原因见 `workload/probe/README.md`。

### 2.2 sweep 轴（围绕 W1，各跑 A3b + A6）

| 轴 | 文件 | 值 | 动的是哪根杠杆 |
|---|---|---|---|
| S1 轮数 | `W1_S1_rounds_*` | 12, 48 | 断开的 diff 积累多少 |
| S2 每 main 的 worker 数 | `W1_S2_workers_*` | 1, 4 | main 每轮修正量、共读宽度 |
| S3 会话数 | `W1_S3_sessions_*` | 1, 4 | 每 tier 就绪请求数（batch、MQ 共享） |
| S4 worker 回答长度 | `W1_S4_worker_lout_*` | 32, 256 | main 共读的块大小 |
| S5 main 回答长度 | `W1_S5_main_lout_*` | 32, 512 | decode 在 E2E 里的份额 |
| S6 文档长度 | `W1_S6_doc_tokens_*` | 512, 1024 | 每轮驻留上下文 |

结构探针：`LEVERS_HEADS_PER_HBM=2 python3 output/analysis/b1_levers.py workload/probe/sweep/W1_turns.json`。
重新生成整套：`python3 workload/probe/gen_main_workers.py --all workload/probe/sweep`。

## 3. 怎么跑（squire）

```bash
cd /data2/chenyi9/KV-PIM/attacc_drampim_822
export KVPIM_SCRATCH=/data2/chenyi9/KV-PIM/scratch_0905        # Ramulator 二进制、trace_gen 软链、签名缓存都在这里
setsid nohup experiments/mem_guard.sh $KVPIM_SCRATCH/guard.log > /dev/null 2>&1 < /dev/null &   # 先起监视器

# W1 baseline，七个 combo 并行（6 个 PIM combo × 8 worker + 7 = 55 核）
setsid nohup bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_w1_LLAMA3-8B '^W1_turns' LLAMA3-8B \
    > $KVPIM_SCRATCH/proto_w1_LLAMA3-8B.out 2>&1 < /dev/null &

# 一条 sweep 轴（每点 A3b + A6，三点并行 54 核），或整套（W1 + 12 个 sweep 点，按 manifest 顺序分批）
bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_w1_LLAMA3-8B 'W1_S3_' LLAMA3-8B
bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_w1_LLAMA3-8B '.' LLAMA3-8B

# 所有 model 的 W1 七档，串行（大 model 一次一档）：experiments/run_w1_all_models.sh <outroot>
```

规则：没有明确指令不跑（`agent.md` §1.7）；≤ 64 核、≤ 500 GB；同一时刻只跑一条 `run_sweep.sh`。
大 model（LLAMA-65B、GPT-175B）一次只跑一档：`RUNGS=A5 PARALLEL=1 RAMU_WORKERS=16`，内存按 `run/README_run_squire.md` 的表。
单档手跑与 athena 的 sbatch 写法见机器页。

看进度：`tail $KVPIM_SCRATCH/proto_w1_LLAMA3-8B/sweep.log`、`grep -h "done\|FAILED" $KVPIM_SCRATCH/proto_w1_LLAMA3-8B/W1_turns.log`、
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

**TINY 上的流程验证**：W1 的七档结果记在 session 记录（`sessions/2026-09-05-ladder-fixes-f01-f02-f04.md` §26），只证明能跑通与逐档方向，不是性能结论；
数据在 `scratch_0905/proto_w1_CACHEBLEND-TINY/`，不进仓库。

数字只能由脚本复制和计算（`agent.md` §3）；结果目录不进仓库，汇总表进 `output/analysis/`。

## 5. 扫描收益在哪个区间才看得见（审计 DECODE_SCAN_TBT_PIPELINE 的结论）

布局与 MQ 改的是 decode 的 PIM 扫描；一步 decode 的时间是 GPU 的线性层（权重读取、AttAcc 原有的 norm/激活固定项）加扫描加链路。
扫描占一步的份额 ≈ batch × 驻留上下文 × KV 字节 对 权重字节的比：batch 8、上下文 2–5k 时，即使在真实 model 上扫描也只占一步的几个百分点，
布局把扫描降 30% 只能在 TBT 上体现 1–2%（9-05 的 C1 上实测 A3b→A4e 扫描 −29%、TBT −1.75%）。这不是调度 bug，是工作点。

用设备模型加 TINY 的扫描标定推导（`LLAMA3-8B`、flash、每头 8 通道；推导值，不是实测）：

| batch | 上下文 4k | 上下文 16k | 上下文 64k |
|---|---|---|---|
| 8 | 扫描占一步 7% | 24% | 56% |
| 32 | 20% | 50% | 80% |
| 64 | 29% | 62% | 87% |

所以要让扫描收益在 TBT 上"正确体现"，要把 decode 放进 KV 主导的区间：W1 的 S6（文档长度）和 S1（轮数）拉长驻留上下文、S3（会话数）加大 batch 内的请求数，是 workload 侧的杠杆，
batch 是运行侧的杠杆（`BATCH=32 bash experiments/run_sweep.sh <outroot> '^W1_turns' LLAMA3-8B`，同一 workload，只改 batch）。
两者都是 AttAcc 论文自己的评测区间（长上下文、大 batch）。调度器已改为回填空窗（审计 P1），不再让已就绪的 GPU 工作等在
一个仍在等 PIM 的事件后面；decode 的 QKV 与投影按 KV head 切片（AttAcc 原版 `minimum_ratio` 的 head 流水），扫描只等第一个 head 的 Q，
投影只有最后一个 head 留在扫描之后。两项对所有 combo 一致，见 session §23–24。

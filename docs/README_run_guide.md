# 运行指南：七档阶梯的 baseline + sweep（chenyi9 裁决 2026-09-05）

这一页定死跑论文数据时**必须开的选项**、公平规则、baseline 与 sweep 矩阵、跑法与汇总。
所有档同一份 workload、同一份复用计划，只换 `--ablation`。

## 1. 必须开的选项

| 选项 | 值 | 为什么 |
|---|---|---|
| `--engine dag` | 固定 | 物理事件 DAG 引擎（真实 extent 进 Ramulator） |
| `--pipeopt` | 开（默认 ON） | 各设备各自的资源时间轴，通道 lane 取 max；`--no-pipeopt` 是 AttAcc 的串行约定，会抹掉布局收益 |
| `--gpu-model flash` | **必须显式传** | GPU attention 按 FlashAttention-2 融合核定价（每 (head, request, 128 行 Q 块) 一个 CTA，效率随 key 长度取 FA-2 A100 曲线，S 不落 HBM，decode 用 flash-decoding）。默认 `legacy` 是 AttAcc 原版 xPU 公式，attention 只有 ~11 TFLOPS，会让 bank sweep 在任何形状上都赢 GPU，A6 ≡ A5（session 2026-09-05 §9） |
| `--powerlimit` | 开（默认 ON） | Ramulator 功率受限预设：nCCDAB 6 tCK，MQ 命令按能量钳位（n=8 时 8 tCK）。`--no-powerlimit` 是 NPC 4 tCK |
| `--pim-link nvlink3` | 默认 | 600 GB/s，AttAcc 原版假设；决定 GPU 侧回读驻留 KV 的代价 |
| `--word 2` | 默认 | FP16 |
| `--reuse recompute --epic-prefix-recompute-tokens 8` | A2–A6 | 每个位移段重算 k=8 行；A1 用 `--reuse no-reuse` 且不带 k |
| `--cacheblend-batch-size 8` | 固定 | decode batch；也是 MQ sweep 的驻留 Q 上限（min(batch, 512 B / 64 B)） |
| `--num-hbm 1 --ngpu 1` | TINY / LLAMA3-8B | 8 头落一个 HBM，stripe = 2 通道；GPT-13B / LLAMA-33B 用 2/10，65B / 175B 用 8/40 |
| `--ramulator-workers` | 按核数预算 | 每档一个进程 + W 个 Ramulator worker；不改变模拟结果 |

不传、保持默认的：`--attn-splitk`（关，保持与 8-21 flash 矩阵可比）、`--ffopt`（DAG 引擎不用）、
`--cacheblend-rotate-mode gpu`、`--pe-freq-ghz` / `--gemv-buffer-bytes`（跟 preset：A5/A6 1.3004 GHz / 512 B）、
`--pim-batch-command`（跟 preset：A1–A4e replicate，A5/A6 mq）。

已修的记账口径（都在本分支）：`num_attacc = --ngpu`（`fc0216d`）；DIE/TLB 零成本只保依赖；A1 prefill 在 GPU；
A3b 持久写入序放置；fresh prefill 按档选边。

## 2. 公平规则

1. 一个 workload JSON → 一份复用计划（`build_reuse_plan`，`recompute`，k=8，seed 0）→ 七档共用。
2. 修正是 per-agent 的、只由位移决定；共享 chunk 由 `(tier, id)` 排序最先的请求声明，其余按写入序复用；没有人为的"每个修正独占一行"之类惩罚。
3. 每档只换 preset（`src/ablation.py`）：A1 硬件 baseline、A2 软件 baseline、A3b 朴素 PIM 存储、A4c per-head diff 行、A4e placement table、A5 prefill 进 PIM + MQ、A6 动态选边。
4. 报三个数：**E2E = 整个 workload 的总时间**（`makespan_s`）、**TBT** = 每请求 `(end_s − first_token_s)/(lout−1)` 的均值、**能量与平均功率**（`energy_nj`，按类拆 GPU / LINK / PIM）。

## 3. flash 下两侧的已知交叉点（选边器的价格，CACHEBLEND-TINY）

| 形状 | GPU（flash） | bank sweep | 赢家 |
|---|---|---|---|
| 独立新 prompt 4352 token | 0.48 ms | 3.39 ms | GPU 7× |
| 复用 agent，新算 2416 行 / 上下文 6512 | 0.46 ms | 2.70 ms | GPU 6× |
| 独立新 prompt 512 | 28 µs | 102 µs | GPU |
| 长上下文 4352 上只加 8 个 token | 85 µs（memory-bound + 回读） | ≈ 10 µs | PIM 8× |

所以 A5 只在"复用重、每轮新 token 少"的请求上赢，A6 靠把长的新 prompt 送回 GPU 与 A5 分开。

## 4. Baseline：多 agent 多轮 RAG / 代码库 agent（`workload/probe/gen_sweep.py`）

| 项 | 值 | 目的 |
|---|---|---|
| 共享语料 | 64 × 256-token chunk（16k），supervisor `a0_owner` 首轮 prefill 声明 | 长新 prompt：A6 送 GPU、A5 进 PIM，这是 A5/A6 分开处 |
| worker | 8 个 | 布局冲突需要多 agent 同时读 |
| 每轮 | 检索 2 个 chunk（agent i 从 chunk 2i 起，彼此重叠但不相同，有位移 → 各 k 行修正）+ 自己写 128 token | 每轮 prefill 是 decode 形状（m≈144，n 数千），PIM 侧有利；co-read 集合不同，A4e 的表才有事做 |
| 轮数 | 8 | 大于 stripe，修正跨轮分散 |
| system prompt / lout | 256 / 256 | |

两种写法（同一 session）：

- `*_interleaved.json`：一个 agent 一个请求，上下文按轮交替 `chunks | own`，一次 prefill、一次 decode。可手算。
- `*_turns.json`：一个 (agent, 轮) 一个请求，tier = 轮，`parent` = 上一轮，`parent_out` 段 = 上一轮 decode 输出，
  `history_len` = 该 agent 之前各轮的 prefill 行（驻留、不重算），每轮 decode lout。真实多轮；上一轮 decode 的行
  是下一轮要读的——A4e 的表对 decode 输出行没有信息的短板在这里暴露。引擎里 history 是抽象驻留段，
  不是上一轮真实落的行，论文里要说明。

## 5. Sweep（每次只动一个轴；`workload/probe/sweep/manifest.csv`，42 个文件）

**档数规则（chenyi9 裁决 2026-09-05）：只有 baseline B0 跑七档；每个 sweep 点只跑 A3b 和 A6**（朴素 PIM 存储 vs Fugue）。
`run_sweep.sh` 按文件名自动选档，`RUNGS=...` 可覆盖。

| # | 轴 | 取值（粗体 = baseline） | 隔离的差别 | legacy 下已有证据 |
|---|---|---|---|---|
| S1 | agent 数 | 4 / **8** / 16 / 32 | A3b→A4c→A4e：co-read 冲突随 N 涨 | N8→N16 TBT 收益 5–6% → 15–16% |
| S2 | 轮数 | 2 / 4 / **8** / 16 | A3b→A4c：修正跨轮分散 | R > stripe 后出现 |
| S3 | lout | 128 / **256** / 512 / 1024 / 2048 | TBT 收益换成 E2E 收益；A4e 对 decode 行的短板 | lout 1024：A4c E2E 1% → 3.6%，A4e 反输 A4c |
| S4 | 每轮新写 token | 16 / 64 / **128** / 256 / 1024 | A4e↔A5：m/n 小 PIM 赢、大 GPU 赢；A6 贴下界 | flash 交叉点在数百 |
| S5 | 新 prompt 占比 | **0** / 25 / 50 / 75%（2k/4k/8k 独立 prompt） | A5↔A6：A5 全进 PIM 在这些请求上输给 A4e，A6 分流应同时胜过两者 | 分裂探针 A6 选 GPU 32 / PIM 5 |
| S6 | 每轮检索 chunk 数 | 1 / **2** / 4 / 8 | 上下文长度 n：sweep、回读随 n，算力随 m·n | |
| S7 | k | 2 / **8** / 32 | 复用策略：A3b 每修正多占行，A5 的 m 含修正行 | 运行时 `EPIC_K`，无需新 JSON |
| S8 | 模型 | **TINY**（8 头，stripe 2）→ LLAMA3-8B（8 KV 头 GQA 4）→ LLAMA-7B（32 头，每通道 2 头） | stripe 与通道拥挤度 | 运行时 MODEL，无需新 JSON |

`turns` 写法下 S5 的新 prompt 数按每轮请求数折算，文件更大（S5_0p75_turns 有 260 个请求、1M prefill token），先跑 interleaved。

## 6. 跑法

```bash
# 一次性环境（scratch 放 /data2；Ramulator 二进制与 trace_gen 软链；签名缓存会落在这里）
export KVPIM_SCRATCH=/data2/<you>/scratch_0905
mkdir -p $KVPIM_SCRATCH && ln -sfn <ramulator2 binary> $KVPIM_SCRATCH/ramulator2 \
  && ln -sfn $PWD/pim_ramulator_src/trace_gen $KVPIM_SCRATCH/trace_gen

# 内存监视（整机实际占用 10 s 均值 > 700 GB 才 kill 本批最大进程；核数超 64 只告警）
setsid nohup experiments/mem_guard.sh $KVPIM_SCRATCH/guard.log >/dev/null 2>&1 &

# 单点：七档，flash，pipeopt，61 核
GPU_MODEL=flash RUNGS="A1 A2 A3b A4c A4e A5 A6" NUM_HBM=1 NGPU=1 RAMU_WORKERS=9 \
KVPIM_PREFILL_SIDE_LOG=$KVPIM_SCRATCH/sides.jsonl \
bash experiments/run_dag_ladder.sh workload/probe/sweep/B0_interleaved.json CACHEBLEND-TINY $KVPIM_SCRATCH/out_B0

# 整个矩阵：B0 七档两两并行（62 核）；sweep 点只跑 A3b + A6，四个并行（56 核）。顺序：B0，再 S5、S4（A5/A6），再 S1、S3（布局），其余
bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep '^B0_'
bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep '^S5_.*interleaved'
bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep '^S4_.*interleaved'
bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep '^S1_.*interleaved'
bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep '^S3_'
EPIC_K=2  bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep_k2  '^B0_interleaved'   # S7
EPIC_K=32 bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep_k32 '^B0_interleaved'
bash experiments/run_sweep.sh $KVPIM_SCRATCH/sweep_llama3 '^(B0|S1|S4|S5)_' LLAMA3-8B   # S8，论文数

# 汇总一个点：E2E / TBT / 能量 / 功率，相对某档的比值
python3 experiments/summarize_ladder.py $KVPIM_SCRATCH/sweep/B0_interleaved workload/probe/sweep/B0_interleaved.json A3b
```

每个点的目录里：`dag_A*.json`（报表）、`dag_ladder.csv` / `dag_ladder_tiers.csv`（collect 脚本）、
`<点>.sides.jsonl`（A6 选边的两侧价格）。原始结果不进仓库；汇总表进 `output/analysis/`。

## 7. 运行量（CACHEBLEND-TINY，61–62 核）

一个七档点 5–15 分钟；一个 A3b + A6 点约 2–5 分钟（`turns` 写法与 S1_32 / S5_0p75 更慢）。interleaved 21 点约 1–2 小时，turns 21 点约 2–4 小时。
LLAMA3-8B 上只跑 B0 + S1 + S4 + S5。

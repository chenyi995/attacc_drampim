# 怎么跑：athena 集群（Slurm）—— `chenyi-0905` 分支

athena（`athena.egr.duke.edu`）是带 Slurm 的集群登录节点，活干在 `node1`–`node6` 上。
跑法只有一条：写 sbatch 提交，**不在登录节点上跑真实负载**。squire（本机直跑）见同目录的
`README_run_squire.md`。选项与公平规则在 `../README_run_guide.md`。

> 本页的机器数据来自 xinyao 分支 2026-09-03 的 athena 记录；`chenyi-0905` 分支**还没有在 athena 上跑过**，
> 路径、`.so`、缓存都要按 §1 重新验证一遍再用。

## 0. 硬约束

| 项 | 值 |
|---|---|
| 登录节点 | 48 核 / 251 GB，只用来提交和看进度 |
| 计算节点 | `node1`–`node6`，各 96–128 核 / 1 TB |
| `/`（`/tmp`） | 214 GB，接近满，**不能放 scratch** |
| `/localdata` | 计算节点上 42 TB，**登录节点上不存在**；scratch 放这里 |
| `/data2` | **不存在**（那是 squire 的盘） |
| 编译器 | 系统 g++ 11.4，**没有 `/opt/rh/gcc-toolset-*`** |
| 仓库路径 | `/home/cw636/chenyi/attacc_drampim`（= `/zpool-00/home/...`） |

分区：node1 `athena-mini`；node2/3 `athena`；node4 `athena-small`；node5/6 `athena-genai`。
`athena` 和 `athena-small` 前面常压着别人的高优先级作业，`athena-mini` / `athena-genai` 更快。

## 1. 先决条件

**Ramulator2。** 集群上没有现成的二进制，按 `set_pim_ramulator.sh` 的 cp/patch 列表覆盖到一份干净的
ramulator2 `b7c7027` 源码，`cd build && cmake .. && make -j`（系统 g++ 11.4 可以）。
不要在本分支的 `ramulator2/` 目录里跑 `set_pim_ramulator.sh`：它不是 git checkout。

**事件核。** `src/cppcore/Makefile` 写死的是 squire 的 gcc-toolset-11 路径，本机没有；
显式给编译器：`cd src/cppcore && make clean && make CXX=g++`。树里若带着别的机器编的 `libeventcore.so`，
先重编再跑测试。

**scratch。** 计算节点上：

```bash
export KVPIM_SCRATCH=/localdata/kvpim_$USER
mkdir -p $KVPIM_SCRATCH
ln -sfn $REPO/ramulator2/ramulator2 $KVPIM_SCRATCH/ramulator2      # 你编好的二进制
ln -sfn $REPO/pim_ramulator_src/trace_gen $KVPIM_SCRATCH/trace_gen
```

**签名缓存不要发布到共享卷。** 2026-08-31 的教训：每次 publish 把整个缓存重写回去，两小时从 1 MB 膨胀到 193 GB，
打爆 98% 满的共享卷。本分支的缓存只在 `$KVPIM_SCRATCH/signature_cache.jsonl`（节点本地），不要 rsync 到 home。

**验证环境**（登录节点上跑就行）：

```bash
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
python3 -m unittest discover -s tests          # 应与 squire 一致：121/121
```

## 2. 资源纪律

- **一个 rung 一个核。** `档数 × (W+1)` 算的是暖机阶段（每档 fork W 个 Ramulator worker），
  真正耗时的是 DAG 构建，每档单线程。9-02 实测申请 222 核只忙 49.9 核。
- **内存贴着实际要。** 9-02（九档、上下文 8464）峰值：GPT-175B 461 GB、LLAMA-65B 386、LLAMA-33B 297、
  GPT-13B 194、LLAMA3-8B 169、LLAMA-7B 170（`sacct --format=MaxRSS`）。要价过大排不上：Slurm 的 backfill 只塞装得下的作业。
- **walltime 按实际工作量给**（大模型 baseline 24 h、其余 16 h、两档的点 6 h）。写 3 天几乎永远 backfill 不进去。
- 大模型的作业设 `KVPIM_NOGC=0`（不回收引用环多占约 100 GB）。
- 本分支没有 xinyao 分支的 `output/_orch2` 认领队列脚本；用 `run_dag_ladder.sh` 按点、按档提交即可（下）。

## 3. 提交

一个 sweep 点一个作业（B0 七档；其余点 `RUNGS="A3b A6"`），`run_dag_ladder.sh` 默认 flash、pipeopt、k=8、batch 8：

```bash
#!/usr/bin/env bash
#SBATCH --job-name=kvpim_B0
#SBATCH --partition=athena-genai --nodelist=node6
#SBATCH --cpus-per-task=9 --mem=60G --time=6:00:00
#SBATCH --output=%x_%j.out
REPO=/home/cw636/chenyi/attacc_drampim
export KVPIM_SCRATCH=/localdata/kvpim_$USER
export ATTACC_RAMULATOR_DIR=$KVPIM_SCRATCH ATTACC_RAMULATOR_LOG=$KVPIM_SCRATCH/ramulator.out
export PYTHONPATH=$REPO KVPIM_CPPCORE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $REPO
RUNGS="A1 A2 A3b A4c A4e A5 A6" NUM_HBM=1 NGPU=1 RAMU_WORKERS=4 \
KVPIM_PREFILL_SIDE_LOG=$KVPIM_SCRATCH/B0.sides.jsonl \
bash experiments/run_dag_ladder.sh workload/probe/sweep/B0_interleaved.json CACHEBLEND-TINY $KVPIM_SCRATCH/out_B0
```

`--cpus-per-task` 按"一档一核 + 暖机余量"给，不按 `档数 × (W+1)`。大模型（GPT-175B / LLAMA-65B）一个作业只跑一档：
`RUNGS=A5`，`--mem` 按上表。几十个单档作业共用一个 `OUT` 时给 `SKIP_COLLECT=1`，最后手动收一次
`collect_dag_ladder.py`。

看进度 / 停：

```bash
squeue -u $USER
grep -h "done\|FAILED" kvpim_B0_*.out
scancel <jobid>            # 或 scancel -u $USER -n kvpim_B0
```

续跑：`run_dag_ladder.sh` 不跳过已有的档；只补缺档时用 `RUNGS="A5 A6"` 再提一次，`SKIP_COLLECT=1`，收数时再合。

## 4. 单档手跑（调试）

`/localdata` 只有计算节点上有，先 `srun` 上去：

```bash
srun --partition=athena --nodelist=node2 --cpus-per-task=8 --mem=60G --time=4:00:00 --pty bash
export PYTHONPATH=$PWD KVPIM_CPPCORE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export KVPIM_SCRATCH=/localdata/kvpim_$USER ATTACC_RAMULATOR_DIR=/localdata/kvpim_$USER
export ATTACC_RAMULATOR_LOG=$ATTACC_RAMULATOR_DIR/ramulator.out
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/probe/sweep/B0_interleaved.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --ablation A6 --engine dag --pipeopt --gpu-model flash \
  --workload-report out.json --workload-report-events none \
  --cacheblend-batch-size 8 --num-hbm 1 --ngpu 1 --ramulator-workers 4
```

直接调 `main.py` 时 `--gpu-model flash` 必须显式传。A1 用 `--reuse no-reuse` 且不带 `--epic-prefix-recompute-tokens`。

## 5. 出数

与 squire 相同：`experiments/summarize_ladder.py <outdir> <wl.json> [ref]`、`<outdir>/dag_ladder.csv`、
`<outdir>/…sides.jsonl`。把结果目录 rsync 回 squire 的 `/data2` 再汇总也行，原始结果不进仓库。

## 6. 排错

| 现象 | 处理 |
|---|---|
| `make` 报 `/opt/rh/gcc-toolset-11/...: No such file` | 那是 squire 的路径；`make CXX=g++` |
| `std::length_error: vector::_M_range_insert` | `.so` 与源码不符；`make clean && make CXX=g++`；临时 `KVPIM_CPPCORE=0` |
| 退出码 2、日志只有 banner | `WorkloadValidationError` 被 argparse 吞掉；用 `runpy` 包一层看 traceback |
| `(Priority)` 排不上 | 要价过大或 walltime 过长；按 §2 缩小 |
| ENOSPC | scratch 落到 `/` 了；确认 `ATTACC_RAMULATOR_DIR` 指向 `/localdata` |
| 登录节点卡死 | 有人在登录节点跑了真实负载；所有活都 `sbatch` / `srun` 到 node 上 |

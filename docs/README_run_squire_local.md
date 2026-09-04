# 怎么跑:squire(本机直跑,无 Slurm)

**squire** 是那台 128 核、带 `/data2`、**没有 Slurm** 的机器。它上面
`which sbatch` 是空的,`output/_orch2/*.sh`(sbatch、athena 分区、集群 REPO 路径)
**全部用不了**,所以有一套单独的本机跑法。

> athena 集群(有 Slurm,活干在 `node1`–`node6` 上)的跑法是**另一份文档**:
> `README_run_athena_slurm.md`。两台机器的 scratch 路径、编译器、并行度都不一样,
> 不要交叉套用。

---

## 0. squire 的硬约束

| 项 | 值 |
|---|---|
| 核数 / 内存 | 128 核 / 754 GB(约 690 GB 可用) |
| `/`(`/tmp` 在上面) | 50 GB,**95% 满,只剩约 3 GB** |
| `/localdata` | **不可写** |
| `/data2` | 7 TB,剩约 1.9 TB |
| 编译器 | 系统 gcc **8.5.0**;另有 `/opt/rh/gcc-toolset-{9,11,14}` |
| Python | 3.10.13 |

> **scratch 必须放 `/data2`。** 放 `/tmp` 会在跑到一半时 ENOSPC —— 单档 trace
> 可能要上百 GB。仓库脚本用 `/data2/chenyi9/kvpim_run_scratch`。

---

## 1. 先决条件

### 1.1 Ramulator2

```bash
bash set_pim_ramulator.sh
cd ramulator2 && mkdir -p build && cd build && cmake .. && make -j
cp ramulator2 ../ramulator2 && cd ../../
```

### 1.2 C++ event core —— **必须用 gcc-toolset-11**

```bash
cd src/cppcore && make clean && make && cd ../..     # 应打印 gcc-toolset-11 的绝对路径
```

`Makefile` 里已经写死成

```make
CXX = /opt/rh/gcc-toolset-11/root/usr/bin/g++
```

> ⚠️ **2026-09-03 修过的坑**:这一行原本是 `CXX ?= ...`。GNU make **内置**就定义了
> `CXX = g++`,所以 `?=` 永远不生效,`make` 一直在用系统编译器(squire 的
> gcc 8.5.0)。编出来的 `.so` 在运行时抛
> `terminate called after throwing an instance of 'std::length_error' /
> vector::_M_range_insert`,直接崩掉整个 run,而且**在未改动的代码上同样复现**。
> 仓库里若带着一个来历不明的 `libeventcore.so`,**先 `make clean && make` 重编**。
> 命令行 `make CXX=...` 仍可覆盖 —— athena 上就是靠这个用系统 g++ 11.4 编的。

验证过:A1 / `wl_tiny` / `CACHEBLEND-TINY` 在 `KVPIM_CPPCORE=1` 下 rc=0,
makespan `0.059978271549260116`,与 `KVPIM_CPPCORE=0` **逐位相同** —— C++ 核与
Python 核互相印证。

### 1.3 验证环境

```bash
export PYTHONPATH=$PWD
python3 tests/test_placement.py                    # 46 个,全过
KVPIM_CPPCORE=1 python3 tests/test_workload.py     # 41 个,全过
```

`test_workload.py` 在 `KVPIM_CPPCORE=1` 下崩,基本就是 1.2 那个坑。

---

## 2. 跑

### 2.1 一键(A3b–A6,一次一个模型)

```bash
cd <repo>
setsid nohup bash experiments/run_local_a3b_a6.sh \
  > /data2/chenyi9/kvpim_run_scratch/driver.log 2>&1 < /dev/null &
```

- 一次一个模型,**模型内 5 档并行**;核预算 `5 × RAMU_WORKERS(20) + 5 = 105 ≤ 128`。
- 输出 `output/sweep_a3b_a6_20260903_append/<MODEL>/baseline_k8/dag_<档>.json`。
- **可续跑**:某个模型目录里已经有 5 个 `dag_A*.json` 就跳过。
- 环境:`KVPIM_CPPCORE=1`、`KVPIM_NOGC=0`、各种 `*_NUM_THREADS=1`。
- 换模型:`MODELS="GPT-13B LLAMA-7B" bash experiments/run_local_a3b_a6.sh`
- 换并行宽度:`RAMU_WORKERS=14 bash experiments/run_local_a3b_a6.sh`
- 换档位:脚本里的 `RUNGS="A3b A4 A4b A5 A6"`。

> 这个脚本跑的是**固定五档 + baseline 一个配置**。2026-09-03 的新跑法
>(baseline 七档 + 其余点两档、78 个任务的认领队列)目前只在 athena 上编排,
> 见 `README_run_athena_slurm.md` §3;squire 上要跑同样的东西,直接照 §2.2
> 手跑对应的档即可。

看进度:

```bash
tail -f /data2/chenyi9/kvpim_run_scratch/driver.log
pgrep -c -f "main.py --system dgx-attacc"          # 还有几个 run 在跑
ls output/sweep_a3b_a6_20260903_append/*/baseline_k8/dag_A*.json | wc -l
grep -h REPORT_SUMMARY output/sweep_a3b_a6_20260903_append/*/baseline_k8/dag_*.log
```

停:

```bash
pkill -f run_local_a3b_a6.sh; pkill -f "main.py --system dgx-attacc"
```

### 2.2 单档手跑

```bash
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RD=/data2/chenyi9/kvpim_run_scratch/manual
mkdir -p $RD
ln -sf $PWD/ramulator2/ramulator2 $RD/ ; ln -sf $PWD/ramulator2/trace_gen $RD/
cp ramulator.out $RD/
export ATTACC_RAMULATOR_DIR=$RD ATTACC_RAMULATOR_LOG=$RD/ramulator.out

python3 main.py --system dgx-attacc --model GPT-13B \
  --workload workload/sweep/wl_baseline_alltoall_N16_C16_D2.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --ablation A3b --engine dag \
  --workload-report out.json --workload-report-events none \
  --cacheblend-batch-size 8 --num-hbm 10 --ngpu 2 --ramulator-workers 20
```

A1 是唯一用 `--reuse no-reuse` 且**不带** `--epic-prefix-recompute-tokens` 的档。
**不设 `ATTACC_RAMULATOR_DIR` 的话**,trace 和 yaml 会直接落进 `ramulator2/`,
把工作区弄脏。**始终设它。**

### 模型表(两台机器通用,来自 `output/_orch2/common.sh`)

| 模型 | `--ngpu` | `--num-hbm` |
|---|---:|---:|
| LLAMA-7B | 1 | 1 |
| LLAMA3-8B | 1 | 1 |
| GPT-13B | 2 | 10 |
| LLAMA-33B | 2 | 10 |
| LLAMA-65B | 8 | 40 |
| GPT-175B | 8 | 40 |

---

## 3. 出数

```bash
python3 experiments/collect_dag_ladder.py <OUT_DIR> <workload.json> <MODEL>
python3 output/analysis/extract_sweep_csv.py \
        --root output/<SWEEP_ROOT> --outdir output/analysis/<TAG> --jobs 8
python3 output/analysis/extract_sweep_csv.py \
        --root output/<SWEEP_ROOT> --self-check 6
```

---

## 4. 排错

| 现象 | 原因 / 处理 |
|---|---|
| `std::length_error: vector::_M_range_insert`,进程 abort/segv | `libeventcore.so` 陈旧或用系统 gcc 8.5 编的。`cd src/cppcore && make clean && make`(见 §1.2)。临时绕开:`KVPIM_CPPCORE=0` |
| **退出码 2,日志里只有 banner** | `main.py:588` 把 `WorkloadValidationError` 交给 `parser.error()`,argparse `sys.exit(2)`,真实异常被吞。想看真因用 `runpy` 包一层并 `except SystemExit` 打 traceback |
| `CacheBlend event has an invalid shape or cost` | `validate_cacheblend_events` 里 `event.rows <= 0`。tiny 模型 / 短上下文下 `_append_placement_pim_scan` 的 report 轮转会给尾部 channel 分到 0 行。**已知未修**,见 `sessions/2026-09-03.md` |
| ENOSPC / 跑一半死掉 | scratch 落到 `/`(只剩 3 GB)。设 `ATTACC_RAMULATOR_DIR` 到 `/data2` |
| `ramulator2/` 里冒出一堆 `.trace` / `.yaml` | 没设 `ATTACC_RAMULATOR_DIR` |
| 内存吃爆 | `KVPIM_NOGC=0`(本机脚本已经这么设) |
